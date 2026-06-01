"""Unit tests for dynamic per-station ensemble weight computation.

Covers the pure math in _compute_weights_from_stats — the DB-backed helpers
are integration-tested separately (they need an event loop + schema).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.storage.db as storage_db
from backend.modeling.station_weights import (
    DEFAULT_WEIGHTS,
    GLOBAL_MSE_PRIOR,
    MAX_LIVE_STATION_BIAS_ABS_F,
    MIN_LIVE_STATION_SOURCE_SAMPLES,
    WEIGHT_CAP,
    WEIGHT_FLOOR,
    _clamp_station_bias,
    _compute_weights_from_stats,
    _use_live_station_bias,
    _use_live_station_weight,
    backfill_forecast_daily_errors,
)
from backend.storage.models import (
    Base,
    City,
    Event,
    ForecastDailyError,
    ForecastObs,
    MetarObs,
    StationSourceWeight,
)


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "station_weights_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


def _past_date(days_back: int = 1) -> str:
    return (
        datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days_back)
    ).strftime("%Y-%m-%d")


def _event_end_utc(date_et: str) -> datetime:
    return datetime.strptime(date_et, "%Y-%m-%d").replace(
        tzinfo=ZoneInfo("America/New_York")
    ).astimezone(timezone.utc) + timedelta(days=1)


def test_biased_ecmwf_auto_demotes():
    """A source with high MSE gets a small weight; unbiased sources dominate."""
    stats = {
        "nws":       {"mse_fast": 1.0, "mse_slow": 1.0, "n_samples": 30},
        "wu_hourly": {"mse_fast": 1.5, "mse_slow": 1.5, "n_samples": 30},
        "hrrr":      {"mse_fast": 2.0, "mse_slow": 2.0, "n_samples": 30},
        "nbm":       {"mse_fast": 1.5, "mse_slow": 1.5, "n_samples": 30},
        "ecmwf_ifs": {"mse_fast": 25.0, "mse_slow": 25.0, "n_samples": 30},
    }
    w = _compute_weights_from_stats(stats)
    assert w["nws"] > w["ecmwf_ifs"]
    assert w["ecmwf_ifs"] <= WEIGHT_FLOOR + 1e-6  # clamped to floor
    # Weights still sum to 1.
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_cold_start_shrinks_toward_uniform():
    """With n=0 for everyone, shrinkage dominates → near-uniform weights."""
    stats = {src: {"mse_fast": 1.0, "mse_slow": 1.0, "n_samples": 0}
             for src in ("nws", "wu_hourly", "hrrr", "nbm", "ecmwf_ifs")}
    # Even with wildly different MSEs, n=0 forces the prior to dominate.
    stats["ecmwf_ifs"]["mse_fast"] = 0.01
    stats["ecmwf_ifs"]["mse_slow"] = 0.01
    w = _compute_weights_from_stats(stats)
    # Spread should be small (all ≈ 0.2)
    assert max(w.values()) - min(w.values()) < 0.05


def test_empty_returns_empty():
    assert _compute_weights_from_stats({}) == {}


def test_weights_floor_holds_and_sum_is_one():
    """WEIGHT_FLOOR holds after clamping even for a terrible source; weights sum to 1.

    Note: WEIGHT_CAP binds pre-renormalization only — in a 2-source case where
    the loser hits the floor, the winner can exceed CAP after renormalization.
    This is acceptable: the floor guarantees the loser still contributes and the
    cap matters most when we have 4-5 sources.
    """
    stats = {
        "nws":       {"mse_fast": 0.001, "mse_slow": 0.001, "n_samples": 30},
        "ecmwf_ifs": {"mse_fast": 100.0, "mse_slow": 100.0, "n_samples": 30},
    }
    w = _compute_weights_from_stats(stats)
    assert w["ecmwf_ifs"] >= WEIGHT_FLOOR - 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_defaults_cover_all_ensemble_sources():
    """Sanity: fallback defaults defined for every source the model uses."""
    for src in (
        "nws", "open_meteo", "wu_hourly", "hrrr", "hrrr_15min", "nbm", "ecmwf_ifs",
        "ecmwf_aifs", "gfs_graphcast", "pangu_weather", "fourcastnet_v2", "aurora",
    ):
        assert src in DEFAULT_WEIGHTS


def test_live_station_weight_gates_require_mature_samples():
    thin = StationSourceWeight(
        station_id="KATL",
        source="wu_hourly",
        weight=0.9,
        bias_f=6.0,
        mae_7d=1.0,
        n_samples=MIN_LIVE_STATION_SOURCE_SAMPLES - 1,
    )
    mature = StationSourceWeight(
        station_id="KATL",
        source="hrrr",
        weight=0.2,
        bias_f=6.0,
        mae_7d=2.0,
        n_samples=MIN_LIVE_STATION_SOURCE_SAMPLES,
    )
    noisy = StationSourceWeight(
        station_id="KATL",
        source="nws",
        weight=0.2,
        bias_f=1.5,
        mae_7d=4.5,
        n_samples=MIN_LIVE_STATION_SOURCE_SAMPLES,
    )

    assert _use_live_station_weight(thin) is False
    assert _use_live_station_bias(thin) is False
    assert _use_live_station_weight(mature) is True
    assert _use_live_station_bias(mature) is True
    assert _clamp_station_bias(mature.bias_f) == pytest.approx(MAX_LIVE_STATION_BIAS_ABS_F)
    assert _use_live_station_weight(noisy) is True
    assert _use_live_station_bias(noisy) is False


def test_backfill_forecast_daily_errors_upserts_latest_checkpoint_forecast(tmp_path, monkeypatch):
    _engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    date_et = _past_date()
    event_end = _event_end_utc(date_et)

    async def _seed():
        async with session_factory() as session:
            city = City(
                city_slug="atlanta",
                display_name="Atlanta",
                metar_station="KATL",
                enabled=True,
                is_us=True,
                unit="F",
                tz="America/New_York",
            )
            session.add(city)
            await session.flush()
            session.add(Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                resolution_station_id="KATL",
            ))
            obs_time = event_end - timedelta(hours=8)
            session.add(MetarObs(
                city_id=city.id,
                metar_station="KATL",
                observed_at=obs_time,
                temp_f=80.0,
                temp_c=(80.0 - 32.0) * 5.0 / 9.0,
            ))
            session.add(ForecastDailyError(
                station_id="KATL",
                date_et=date_et,
                source="nws",
                forecast_high_f=70.0,
                observed_high_f=70.0,
                err_f=0.0,
            ))
            for high, fetched_at in (
                (79.0, event_end - timedelta(hours=20)),
                (81.0, event_end - timedelta(hours=17)),
                (90.0, event_end + timedelta(hours=1)),
            ):
                session.add(ForecastObs(
                    city_id=city.id,
                    source="nws",
                    date_et=date_et,
                    fetched_at=fetched_at,
                    high_f=high,
                ))
            await session.commit()
            return city.id

    city_id = _run(_seed())
    written = _run(backfill_forecast_daily_errors(
        "KATL",
        city_id,
        "America/New_York",
        max_days=3,
    ))

    async def _load_row():
        async with session_factory() as session:
            return (
                await session.execute(
                    select(ForecastDailyError).where(
                        ForecastDailyError.station_id == "KATL",
                        ForecastDailyError.date_et == date_et,
                        ForecastDailyError.source == "nws",
                    )
                )
            ).scalar_one()

    row = _run(_load_row())
    assert written >= 1
    assert row.forecast_high_f == pytest.approx(79.0)
    assert row.observed_high_f == pytest.approx(80.0)
    assert row.err_f == pytest.approx(-1.0)


@pytest.mark.parametrize("source", ["nws", "hrrr", "open_meteo"])
def test_backfill_forecast_daily_errors_ignores_late_night_next_day_like_forecast_for_all_sources(
    tmp_path, monkeypatch, source
):
    _engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    date_et = _past_date()
    event_end = _event_end_utc(date_et)

    async def _seed():
        async with session_factory() as session:
            city = City(
                city_slug="atlanta",
                display_name="Atlanta",
                metar_station="KATL",
                enabled=True,
                is_us=True,
                unit="F",
                tz="America/New_York",
            )
            session.add(city)
            await session.flush()
            session.add(Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                resolution_station_id="KATL",
            ))
            session.add(MetarObs(
                city_id=city.id,
                metar_station="KATL",
                observed_at=event_end - timedelta(hours=9),
                temp_f=78.0,
                temp_c=(78.0 - 32.0) * 5.0 / 9.0,
            ))
            for high, fetched_at in (
                (79.0, event_end - timedelta(hours=20)),
                (88.0, event_end - timedelta(hours=1)),
            ):
                session.add(ForecastObs(
                    city_id=city.id,
                    source=source,
                    date_et=date_et,
                    fetched_at=fetched_at,
                    high_f=high,
                ))
            await session.commit()
            return city.id

    city_id = _run(_seed())
    written = _run(backfill_forecast_daily_errors(
        "KATL",
        city_id,
        "America/New_York",
        max_days=3,
    ))

    async def _load_row():
        async with session_factory() as session:
            return (
                await session.execute(
                    select(ForecastDailyError).where(
                        ForecastDailyError.station_id == "KATL",
                        ForecastDailyError.date_et == date_et,
                        ForecastDailyError.source == source,
                    )
                )
            ).scalar_one()

    row = _run(_load_row())
    assert written >= 1
    assert row.forecast_high_f == pytest.approx(79.0)
    assert row.observed_high_f == pytest.approx(78.0)
    assert row.err_f == pytest.approx(1.0)
