import asyncio
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.storage.db as storage_db
from backend.config import Config
from backend.market_context.adapter import MarketContextLLMError
from backend.market_context.service import (
    _generate_market_context_output,
    build_market_context_input,
    refresh_market_context_snapshot,
)
from backend.market_context.types import MarketContextOutput, SECTION_KEYS
from backend.storage.models import (
    Base,
    Bucket,
    CalibrationParams,
    City,
    Event,
    ForecastObs,
    MarketContextSnapshot,
    MarketSnapshot,
    MetarObs,
    MetarObsExtended,
    ModelSnapshot,
    Signal,
    StationProfile,
)
from backend.storage.repos import get_market_context_snapshot
from backend.tz_utils import city_local_date
from web.routes import dashboard_router


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "market_context_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _seed_market_context_fixture(session_factory):
    et = ZoneInfo("America/New_York")
    today_local = datetime.now(et).date()
    today_et = today_local.isoformat()
    prev1_et = (today_local - timedelta(days=1)).isoformat()
    prev2_et = (today_local - timedelta(days=2)).isoformat()

    async with session_factory() as session:
        city = City(
            city_slug="atlanta",
            display_name="Atlanta",
            metar_station="KATL",
            enabled=True,
            is_us=True,
            unit="F",
            tz="America/New_York",
            nws_office="FFC",
            nws_grid_x=52,
            nws_grid_y=88,
        )
        session.add(city)
        await session.flush()

        session.add(
            StationProfile(
                metar_station="KATL",
                observation_minutes=json.dumps([52]),
                observation_frequency="hourly",
                samples_analyzed=24,
                confidence=0.94,
            )
        )
        session.add(
            CalibrationParams(
                city_id=city.id,
                bias_nws=-0.4,
                bias_wu_daily=0.8,
                bias_wu_hourly=0.2,
                n_samples=12,
                last_realized_high=75.0,
            )
        )

        current_event = Event(
            city_id=city.id,
            date_et=today_et,
            status="ok",
            forecast_quality="ok",
            gamma_slug=f"atlanta-{today_et}",
        )
        prev_event_1 = Event(city_id=city.id, date_et=prev1_et, status="ok", forecast_quality="ok")
        prev_event_2 = Event(city_id=city.id, date_et=prev2_et, status="ok", forecast_quality="ok")
        session.add_all([current_event, prev_event_1, prev_event_2])
        await session.flush()

        def add_buckets(event_id: int):
            return [
                Bucket(event_id=event_id, bucket_idx=0, label="72–74°", low_f=72.0, high_f=74.0),
                Bucket(event_id=event_id, bucket_idx=1, label="74–76°", low_f=74.0, high_f=76.0),
                Bucket(event_id=event_id, bucket_idx=2, label="76–78°", low_f=76.0, high_f=78.0),
            ]

        current_buckets = add_buckets(current_event.id)
        prev1_buckets = add_buckets(prev_event_1.id)
        prev2_buckets = add_buckets(prev_event_2.id)
        session.add_all(current_buckets + prev1_buckets + prev2_buckets)
        await session.flush()

        now_local = datetime.now(et).replace(second=0, microsecond=0)
        obs_specs = [
            (now_local.replace(hour=10, minute=52), 70.8, 55.0, 7.0, 11.0, "SCT", 4200, 30.02, 210),
            (now_local.replace(hour=11, minute=52), 72.4, 55.5, 9.0, 14.0, "SCT", 5000, 29.99, 225),
            (now_local.replace(hour=12, minute=52), 73.9, 56.0, 11.0, 16.0, "FEW", 7000, 29.97, 235),
            (now_local.replace(hour=13, minute=52), 75.1, 56.5, 13.0, 18.0, "FEW", 9000, 29.95, 245),
        ]
        running_high = 0.0
        for observed_local, temp_f, dewpoint_f, wind_kt, gust_kt, cover, base_ft, altimeter, wind_dir in obs_specs:
            running_high = max(running_high, temp_f)
            observed_at = observed_local.astimezone(timezone.utc)
            temp_c = round((temp_f - 32.0) * 5.0 / 9.0, 1)
            dewpoint_c = round((dewpoint_f - 32.0) * 5.0 / 9.0, 1)
            obs = MetarObs(
                city_id=city.id,
                metar_station=city.metar_station,
                observed_at=observed_at,
                fetched_at=observed_at,
                report_at=observed_at,
                temp_c=temp_c,
                temp_f=temp_f,
                daily_high_f=running_high,
                raw_text="KATL synthetic",
                raw_json=json.dumps({"source": "nws_obs"}),
            )
            session.add(obs)
            await session.flush()
            session.add(
                MetarObsExtended(
                    metar_obs_id=obs.id,
                    dewpoint_c=dewpoint_c,
                    dewpoint_f=dewpoint_f,
                    humidity_pct=52.0,
                    wind_dir_deg=wind_dir,
                    wind_speed_kt=wind_kt,
                    wind_gust_kt=gust_kt,
                    altimeter_inhg=altimeter,
                    cloud_cover=cover,
                    cloud_base_ft=base_ft,
                    condition="Partly Cloudy",
                )
            )

        session.add_all(
            [
                ForecastObs(
                    city_id=city.id,
                    source="nws",
                    date_et=today_et,
                    high_f=75.0,
                    raw_payload_hash="nws-today",
                    raw_json=json.dumps({"high_f": 75.0}),
                    fetched_at=datetime.now(timezone.utc) - timedelta(minutes=20),
                ),
                ForecastObs(
                    city_id=city.id,
                    source="wu_daily",
                    date_et=today_et,
                    high_f=76.0,
                    raw_payload_hash="wu-d-today",
                    raw_json=json.dumps({"high_f": 76.0}),
                    fetched_at=datetime.now(timezone.utc) - timedelta(minutes=15),
                ),
                ForecastObs(
                    city_id=city.id,
                    source="wu_hourly",
                    date_et=today_et,
                    high_f=75.6,
                    raw_payload_hash="wu-h-today",
                    raw_json=json.dumps({"high_f": 75.6, "peak_hour": "4:52 PM ET"}),
                    fetched_at=datetime.now(timezone.utc) - timedelta(minutes=10),
                ),
                ForecastObs(
                    city_id=city.id,
                    source="wu_history",
                    date_et=today_et,
                    high_f=75.1,
                    raw_payload_hash="wu-hist-today",
                    raw_json=json.dumps({"high_f": 75.1, "obs_time": "1:52 PM ET"}),
                    fetched_at=datetime.now(timezone.utc) - timedelta(minutes=5),
                ),
            ]
        )

        previous_days = [
            (prev1_et, 74.6, 74.0, 75.2, 74.5, [0.25, 0.50, 0.25]),
            (prev2_et, 73.4, 72.8, 73.9, 73.1, [0.40, 0.45, 0.15]),
        ]
        for day_data, event, buckets in [
            (previous_days[0], prev_event_1, prev1_buckets),
            (previous_days[1], prev_event_2, prev2_buckets),
        ]:
            date_key, nws_high, wu_daily_high, wu_hourly_high, realized_high, probs = day_data
            session.add_all(
                [
                    ForecastObs(
                        city_id=city.id,
                        source="nws",
                        date_et=date_key,
                        high_f=nws_high,
                        raw_payload_hash=f"nws-{date_key}",
                        raw_json=json.dumps({"high_f": nws_high}),
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_daily",
                        date_et=date_key,
                        high_f=wu_daily_high,
                        raw_payload_hash=f"wu-d-{date_key}",
                        raw_json=json.dumps({"high_f": wu_daily_high}),
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_hourly",
                        date_et=date_key,
                        high_f=wu_hourly_high,
                        raw_payload_hash=f"wu-h-{date_key}",
                        raw_json=json.dumps({"high_f": wu_hourly_high, "peak_hour": "4:12 PM ET"}),
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_history",
                        date_et=date_key,
                        high_f=realized_high,
                        raw_payload_hash=f"wu-hist-{date_key}",
                        raw_json=json.dumps({"high_f": realized_high, "obs_time": "4:12 PM ET"}),
                    ),
                    ModelSnapshot(
                        event_id=event.id,
                        mu=realized_high,
                        sigma=1.3,
                        probs_json=json.dumps(probs),
                        inputs_json=json.dumps({"projected_high": realized_high, "spread": 1.5, "w_metar": 0.5}),
                    ),
                ]
            )

        current_inputs = {
            "projected_high": 75.4,
            "mu_forecast": 75.2,
            "spread": 1.0,
            "remaining_rise": 1.3,
            "w_metar": 0.62,
            "prob_new_high": 0.41,
            "adaptive": {
                "predicted_daily_high": 75.5,
                "peak_already_passed": False,
                "composite_peak_timing": "4:52 PM ET",
                "peak_timing_source": "wu_hourly",
                "kalman_trend_per_hr": 1.7,
                "regression_r2": 0.64,
            },
        }
        session.add(
            ModelSnapshot(
                event_id=current_event.id,
                mu=75.0,
                sigma=1.2,
                probs_json=json.dumps([0.18, 0.61, 0.21]),
                inputs_json=json.dumps(current_inputs),
            )
        )

        market_specs = [
            (current_buckets[0], 0.16, 0.20, 0.18, -3.0),
            (current_buckets[1], 0.32, 0.36, 0.34, 6.0),
            (current_buckets[2], 0.45, 0.49, 0.47, -2.0),
        ]
        for bucket, bid, ask, mid, change_pts in market_specs:
            base_time = datetime.now(timezone.utc) - timedelta(hours=3)
            session.add_all(
                [
                    MarketSnapshot(
                        bucket_id=bucket.id,
                        fetched_at=base_time,
                        yes_bid=bid - 0.04,
                        yes_ask=ask - 0.04,
                        yes_mid=max(0.01, mid - 0.04),
                        yes_bid_depth=100,
                        yes_ask_depth=120,
                        spread=0.04,
                    ),
                    MarketSnapshot(
                        bucket_id=bucket.id,
                        fetched_at=base_time + timedelta(hours=1),
                        yes_bid=bid + 0.02,
                        yes_ask=ask + 0.02,
                        yes_mid=min(0.99, mid + 0.02),
                        yes_bid_depth=110,
                        yes_ask_depth=130,
                        spread=0.04,
                    ),
                    MarketSnapshot(
                        bucket_id=bucket.id,
                        fetched_at=datetime.now(timezone.utc) - timedelta(minutes=5),
                        yes_bid=bid,
                        yes_ask=ask,
                        yes_mid=mid,
                        yes_bid_depth=140,
                        yes_ask_depth=160,
                        spread=0.04,
                    ),
                    Signal(
                        bucket_id=bucket.id,
                        model_prob=mid + (0.08 if bucket.bucket_idx == 1 else -0.02),
                        mkt_prob=mid,
                        raw_edge=0.06 if bucket.bucket_idx == 1 else -0.02,
                        exec_cost=0.01,
                        true_edge=0.05 if bucket.bucket_idx == 1 else -0.03,
                        reason_json=json.dumps({"bucket_idx": bucket.bucket_idx}),
                        gate_failures_json=json.dumps([]),
                    ),
                ]
            )

        await session.commit()
        return {
            "city_id": city.id,
            "city_slug": city.city_slug,
            "today_et": today_et,
        }


def _valid_output_for_context(context):
    return {
        "sections": {
            key: f"{context.city_display} {key} cites 75.1F, 34.0%, and 1.0F spread without inventing missing feeds."
            for key in SECTION_KEYS
        },
        "final_selection": context.final_selection.model_dump(),
    }


def test_build_market_context_input_selects_bucket_and_flip_signals(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    seeded = _run(_seed_market_context_fixture(session_factory))

    async def run_test():
        async with session_factory() as session:
            city = await session.get(City, seeded["city_id"])
        context = await build_market_context_input(city, seeded["today_et"])
        assert context.final_selection.label == "74–76°"
        assert context.final_selection.confidence_pct >= 40
        assert context.current_observations["current_temp_f"] == 75.1
        assert context.short_range_models["missing_external_models"] == ["HRRR", "NAM", "RAP", "ECMWF"]
        assert context.final_selection.flip_signals
        assert context.market_pricing["underpriced_buckets"]

    _run(run_test())
    _run(engine.dispose())


def test_market_context_output_validator_rejects_mismatched_selection(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    seeded = _run(_seed_market_context_fixture(session_factory))

    async def run_test():
        async with session_factory() as session:
            city = await session.get(City, seeded["city_id"])
        context = await build_market_context_input(city, seeded["today_et"])

        async def fake_generate_json(self, **kwargs):
            payload = _valid_output_for_context(context)
            payload["final_selection"]["bucket_idx"] = 2
            payload["final_selection"]["label"] = "76–78°"
            return payload

        monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_PROVIDER", "openai", raising=False)
        monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_MODEL", "gpt-test", raising=False)
        monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_API_KEY", "test-key", raising=False)
        monkeypatch.setattr(
            "backend.market_context.adapter.MarketContextLLMAdapter.generate_json",
            fake_generate_json,
        )

        try:
            await _generate_market_context_output(context)
            assert False, "Expected MarketContextLLMError"
        except MarketContextLLMError:
            pass

    _run(run_test())
    _run(engine.dispose())


def test_refresh_market_context_snapshot_persists_and_preserves_last_good_on_failure(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    seeded = _run(_seed_market_context_fixture(session_factory))

    async def run_test():
        async with session_factory() as session:
            city = await session.get(City, seeded["city_id"])
        context = await build_market_context_input(city, seeded["today_et"])

        monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_PROVIDER", "openai", raising=False)
        monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_MODEL", "gpt-test", raising=False)
        monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_API_KEY", "test-key", raising=False)

        async def good_generate_json(self, **kwargs):
            return _valid_output_for_context(context)

        monkeypatch.setattr(
            "backend.market_context.adapter.MarketContextLLMAdapter.generate_json",
            good_generate_json,
        )

        first = await refresh_market_context_snapshot("atlanta", seeded["today_et"])
        assert first["generation_status"] == "success"
        assert first["sections"]["market_pricing_analysis"]

        async def bad_generate_json(self, **kwargs):
            raise MarketContextLLMError("provider down")

        monkeypatch.setattr(
            "backend.market_context.adapter.MarketContextLLMAdapter.generate_json",
            bad_generate_json,
        )

        second = await refresh_market_context_snapshot("atlanta", seeded["today_et"])
        assert second["generation_status"] == "failed"
        assert "provider down" in second["last_error"]
        assert second["sections"]["market_pricing_analysis"] == first["sections"]["market_pricing_analysis"]

        async with session_factory() as session:
            stored = await get_market_context_snapshot(session, city.id, seeded["today_et"])
            assert stored is not None
            assert stored.generation_status == "failed"
            assert stored.sections_json is not None

    _run(run_test())
    _run(engine.dispose())


def test_city_page_renders_market_context_states(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    seeded = _run(_seed_market_context_fixture(session_factory))

    app = FastAPI()
    app.include_router(dashboard_router)
    client = TestClient(app)

    monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_PROVIDER", "", raising=False)
    monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_MODEL", "", raising=False)
    monkeypatch.setattr(Config, "MARKET_CONTEXT_LLM_API_KEY", "", raising=False)

    response = client.get(f"/city/atlanta?date={seeded['today_et']}")
    assert response.status_code == 200
    assert "Market Context has not been generated" in response.text

    async def insert_snapshot(status: str, last_error: str | None = None):
        async with session_factory() as session:
            snapshot = MarketContextSnapshot(
                city_id=seeded["city_id"],
                date_et=seeded["today_et"],
                generation_status=status,
                sections_json=json.dumps({key: f"section for {key}" for key in SECTION_KEYS}),
                selection_json=json.dumps(
                    {
                        "bucket_id": 2,
                        "bucket_idx": 1,
                        "label": "74–76°",
                        "low_f": 74.0,
                        "high_f": 76.0,
                        "calibrated_prob": 0.61,
                        "raw_model_prob": 0.61,
                        "market_prob": 0.34,
                        "true_edge": 0.05,
                        "confidence_pct": 63,
                        "rationale": "Synthetic rationale",
                        "flip_signals": ["Synthetic trigger"],
                        "life_or_death_call": "If life depended on being correct, I would select 74–76° because synthetic data says so.",
                        "most_likely_peak_time": "4:52 PM ET",
                        "confidence_components": {"base_prob": 30.0},
                    }
                ),
                source_context_json=json.dumps({"city_slug": "atlanta"}),
                provider="openai",
                model_name="gpt-test",
                generated_at=datetime.now(timezone.utc),
                freshness_at=datetime.now(timezone.utc),
                last_error=last_error,
            )
            session.add(snapshot)
            await session.commit()

    _run(insert_snapshot("success"))
    response = client.get(f"/city/atlanta?date={seeded['today_et']}")
    assert response.status_code == 200
    assert "section for market_pricing_analysis" in response.text
    assert "74–76° · 63%" in response.text

    async def update_snapshot_to_failed():
        async with session_factory() as session:
            snapshot = await get_market_context_snapshot(session, seeded["city_id"], seeded["today_et"])
            snapshot.generation_status = "failed"
            snapshot.last_error = "provider down"
            await session.commit()

    _run(update_snapshot_to_failed())
    response = client.get(f"/city/atlanta?date={seeded['today_et']}")
    assert response.status_code == 200
    assert "Last refresh failed. Showing the last stored snapshot." in response.text
    assert "provider down" in response.text

    client.close()
    _run(engine.dispose())
