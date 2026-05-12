from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.storage.db as storage_db
from backend.modeling import calibration_engine as ce
from backend.modeling.bma_weights_repo import load_sigma_by_source_for_city
from backend.storage.models import (
    Base,
    Bucket,
    City,
    Event,
    ForecastObs,
    MetarObs,
    SourceLeadTimeSkill,
)


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "lead_time_skill_test.db"
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
    return (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days_back)).strftime("%Y-%m-%d")


def _event_end_utc(date_et: str) -> datetime:
    return datetime.strptime(date_et, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(timezone.utc)


async def _seed_city(session_factory) -> City:
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
        await session.commit()
        await session.refresh(city)
        return city


async def _seed_event_with_high(
    session_factory,
    city: City,
    *,
    date_et: str,
    high_f: float = 80.0,
    winning_bucket_idx: int | None = None,
    buckets: list[tuple[float | None, float | None]] | None = None,
) -> Event:
    async with session_factory() as session:
        event = Event(
            city_id=city.id,
            date_et=date_et,
            status="ok",
            trading_enabled=True,
            winning_bucket_idx=winning_bucket_idx,
        )
        session.add(event)
        await session.flush()
        for idx, (lo, hi) in enumerate(buckets or []):
            session.add(
                Bucket(
                    event_id=event.id,
                    bucket_idx=idx,
                    label=f"{lo}-{hi}",
                    low_f=lo,
                    high_f=hi,
                )
            )
        obs_local = datetime.strptime(date_et, "%Y-%m-%d").replace(
            hour=16, minute=52, tzinfo=ZoneInfo("America/New_York"),
        )
        session.add(
            MetarObs(
                city_id=city.id,
                metar_station="KATL",
                observed_at=obs_local.astimezone(timezone.utc),
                temp_f=high_f,
                temp_c=(high_f - 32.0) * 5.0 / 9.0,
            )
        )
        await session.commit()
        await session.refresh(event)
        return event


async def _add_forecast(
    session_factory,
    city: City,
    *,
    date_et: str,
    source: str,
    high_f: float,
    fetched_at: datetime,
    model_run_at: datetime | None = None,
) -> None:
    async with session_factory() as session:
        session.add(
            ForecastObs(
                city_id=city.id,
                source=source,
                date_et=date_et,
                fetched_at=fetched_at,
                model_run_at=model_run_at,
                high_f=high_f,
            )
        )
        await session.commit()


async def _skill_row(session_factory, city_id: int, source: str, lead: int):
    async with session_factory() as session:
        return (
            await session.execute(
                select(SourceLeadTimeSkill).where(
                    SourceLeadTimeSkill.city_id == city_id,
                    SourceLeadTimeSkill.source == source,
                    SourceLeadTimeSkill.lead_time_bucket_hours == lead,
                )
            )
        ).scalar_one_or_none()


def test_past_event_with_canonical_high_populates_without_winner(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_seed_city(session_factory))
    date_et = _past_date()
    event_end = _event_end_utc(date_et)
    _run(_seed_event_with_high(session_factory, city, date_et=date_et, high_f=80.0))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=81.0,
        fetched_at=event_end - timedelta(hours=25),
        model_run_at=event_end - timedelta(hours=26),
    ))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="ecmwf_aifs", high_f=79.0,
        fetched_at=event_end - timedelta(hours=25),
        model_run_at=event_end - timedelta(hours=30),
    ))

    diag = _run(ce.compute_source_lead_time_skills(
        city.id, days_back=10, return_diagnostics=True,
    ))

    assert diag["events_resolved"] == 0
    assert diag["events_with_settlement"] == 1
    assert diag["source_bucket_combos_written"] > 0
    assert diag["skills"]["hrrr:24h"]["n_obs"] == 1
    assert diag["skills"]["ecmwf_aifs:24h"]["n_obs"] == 1
    _run(engine.dispose())


def test_multiple_hrrr_rows_same_checkpoint_count_as_one_sample(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_seed_city(session_factory))
    date_et = _past_date()
    event_end = _event_end_utc(date_et)
    _run(_seed_event_with_high(session_factory, city, date_et=date_et, high_f=80.0))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=85.0,
        fetched_at=event_end - timedelta(hours=30),
        model_run_at=event_end - timedelta(hours=31),
    ))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=81.0,
        fetched_at=event_end - timedelta(hours=25),
        model_run_at=event_end - timedelta(hours=26),
    ))

    _run(ce.compute_source_lead_time_skills(city.id, days_back=10))
    row = _run(_skill_row(session_factory, city.id, "hrrr", 24))

    assert row is not None
    assert row.n_obs == 1
    assert row.mae_f == pytest.approx(1.0)
    _run(engine.dispose())


def test_checkpoint_selection_mirrors_latest_forecast_by_fetched_at(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_seed_city(session_factory))
    date_et = _past_date()
    event_end = _event_end_utc(date_et)
    _run(_seed_event_with_high(session_factory, city, date_et=date_et, high_f=80.0))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=81.0,
        fetched_at=event_end - timedelta(hours=26),
        model_run_at=event_end - timedelta(hours=25),
    ))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=82.0,
        fetched_at=event_end - timedelta(hours=24, minutes=5),
        model_run_at=event_end - timedelta(hours=36),
    ))

    _run(ce.compute_source_lead_time_skills(city.id, days_back=10))
    row = _run(_skill_row(session_factory, city.id, "hrrr", 24))

    assert row is not None
    assert row.n_obs == 1
    assert row.mae_f == pytest.approx(2.0)
    _run(engine.dispose())


def test_forecast_fetched_after_checkpoint_is_not_used_for_that_checkpoint(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_seed_city(session_factory))
    date_et = _past_date()
    event_end = _event_end_utc(date_et)
    _run(_seed_event_with_high(session_factory, city, date_et=date_et, high_f=80.0))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=82.0,
        fetched_at=event_end - timedelta(hours=26),
        model_run_at=event_end - timedelta(hours=30),
    ))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=80.0,
        fetched_at=event_end - timedelta(hours=23),
        model_run_at=event_end - timedelta(hours=24),
    ))

    _run(ce.compute_source_lead_time_skills(city.id, days_back=10))
    row_24 = _run(_skill_row(session_factory, city.id, "hrrr", 24))
    row_18 = _run(_skill_row(session_factory, city.id, "hrrr", 18))

    assert row_24.mae_f == pytest.approx(2.0)
    assert row_18.mae_f == pytest.approx(0.0)
    _run(engine.dispose())


def test_polymarket_winner_mismatch_is_reported_and_excluded(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_seed_city(session_factory))
    date_et = _past_date()
    event_end = _event_end_utc(date_et)
    _run(_seed_event_with_high(
        session_factory,
        city,
        date_et=date_et,
        high_f=80.0,
        winning_bucket_idx=0,
        buckets=[(None, 79.0), (79.0, None)],
    ))
    _run(_add_forecast(
        session_factory, city, date_et=date_et, source="hrrr", high_f=80.0,
        fetched_at=event_end - timedelta(hours=25),
        model_run_at=event_end - timedelta(hours=26),
    ))

    diag = _run(ce.compute_source_lead_time_skills(
        city.id, days_back=10, return_diagnostics=True,
    ))

    assert diag["settlement_mismatches"] == 1
    assert diag["events_with_settlement"] == 1
    assert diag["source_bucket_combos_written"] == 0
    assert diag["settlement_mismatch_details"][0]["derived_bucket_idx"] == 1
    _run(engine.dispose())


def test_bma_sigma_loader_ignores_provisional_skill_rows(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_seed_city(session_factory))

    async def seed():
        async with session_factory() as session:
            session.add_all([
                SourceLeadTimeSkill(
                    city_id=city.id,
                    source="hrrr",
                    lead_time_bucket_hours=24,
                    mae_f=1.0,
                    bias_f=0.0,
                    n_obs=29,
                ),
                SourceLeadTimeSkill(
                    city_id=city.id,
                    source="nws",
                    lead_time_bucket_hours=24,
                    mae_f=2.0,
                    bias_f=0.0,
                    n_obs=30,
                ),
            ])
            await session.commit()
            return session

    _run(seed())

    async def load():
        async with session_factory() as session:
            return await load_sigma_by_source_for_city(session, city.id, 24)

    sigma = _run(load())
    assert sigma == {"nws": 2.0}
    _run(engine.dispose())


def test_refresh_lead_time_scheduler_logs_city_slug():
    source = Path("backend/worker/scheduler.py").read_text()
    assert "city.city_slug" in source
    assert "city.slug" not in source
