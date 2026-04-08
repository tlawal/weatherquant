import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.api.routes as api_routes
import backend.storage.db as storage_db
from backend.storage.models import Base, Bucket, City, Event, ForecastObs, MetarObs, ModelSnapshot, Signal, WorkerHeartbeat
from backend.tz_utils import city_local_date


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "api_routes_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _create_city(session_factory, slug: str = "atlanta") -> City:
    async with session_factory() as session:
        city = City(
            city_slug=slug,
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


def test_health_handles_naive_heartbeat_timestamps(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))

    async def seed():
        async with session_factory() as session:
            session.add(
                WorkerHeartbeat(
                    job_name="scheduler_alive",
                    last_run_at=datetime.utcnow().replace(microsecond=0),
                    run_count=1,
                )
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.health())

    assert response["status"] == "ok"
    assert response["worker_heartbeat_age_s"] is not None

    _run(engine.dispose())


def test_get_city_state_handles_naive_forecast_timestamps(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        fetched_at = datetime.utcnow().replace(microsecond=0)
        date_et = city_local_date(city)

        async with session_factory() as session:
            session.add(
                MetarObs(
                    city_id=city.id,
                    metar_station=city.metar_station,
                    observed_at=fetched_at,
                    fetched_at=fetched_at,
                    temp_c=20.0,
                    temp_f=68.0,
                    daily_high_f=70.0,
                )
            )
            session.add_all(
                [
                    ForecastObs(
                        city_id=city.id,
                        source="nws",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=71.0,
                        raw_payload_hash="nws",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_daily",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=72.0,
                        raw_payload_hash="wu_daily",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_hourly",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=73.0,
                        raw_payload_hash="wu_hourly",
                        raw_json="{}",
                    ),
                ]
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.get_city_state(city.city_slug))

    assert response["city_slug"] == city.city_slug
    assert response["metar_age_s"] is not None
    assert response["forecasts"]["nws"]["age_s"] is not None
    assert response["forecasts"]["wu_daily"]["age_s"] is not None
    assert response["forecasts"]["wu_hourly"]["age_s"] is not None
    assert response["daily_high_f"] == 70.0

    _run(engine.dispose())


def test_get_city_state_exposes_hotter_bucket_fields(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        fetched_at = datetime.utcnow().replace(microsecond=0)
        date_et = city_local_date(city)

        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                forecast_quality="ok",
                gamma_slug=f"atlanta-{date_et}",
            )
            session.add(event)
            await session.flush()

            session.add(
                MetarObs(
                    city_id=city.id,
                    metar_station=city.metar_station,
                    observed_at=fetched_at,
                    fetched_at=fetched_at,
                    temp_c=19.0,
                    temp_f=66.2,
                    daily_high_f=69.0,
                )
            )
            session.add_all(
                [
                    ForecastObs(
                        city_id=city.id,
                        source="nws",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=72.0,
                        raw_payload_hash="nws",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_daily",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=69.8,
                        raw_payload_hash="wu_daily",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_hourly",
                        date_et=date_et,
                        fetched_at=fetched_at,
                        high_f=66.0,
                        raw_payload_hash="wu_hourly",
                        raw_json="{}",
                    ),
                ]
            )
            session.add(
                ModelSnapshot(
                    event_id=event.id,
                    mu=69.0,
                    sigma=0.15,
                    probs_json="[0.999, 0.001]",
                    inputs_json='{"prob_new_high":0.001,"prob_hotter_bucket":0.001,"prob_new_high_raw":0.49,"lock_regime":true,"observed_bucket_idx":0,"observed_bucket_upper_f":70.0,"city_state":"resolved"}',
                    forecast_quality="ok",
                )
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.get_city_state(city.city_slug))

    assert response["prob_new_high"] == 0.001
    assert response["prob_hotter_bucket"] == 0.001
    assert response["prob_new_high_raw"] == 0.49
    assert response["lock_regime"] is True
    assert response["observed_bucket_idx"] == 0
    assert response["observed_bucket_upper_f"] == 70.0
    assert response["city_state"] == "resolved"

    _run(engine.dispose())


def test_get_city_signals_exposes_hotter_bucket_fields(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        fetched_at = datetime.utcnow().replace(microsecond=0)
        date_et = city_local_date(city)

        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                forecast_quality="ok",
                gamma_slug=f"atlanta-{date_et}",
            )
            session.add(event)
            await session.flush()

            bucket = Bucket(
                event_id=event.id,
                bucket_idx=0,
                label="68-69",
                low_f=68.0,
                high_f=69.0,
            )
            session.add(bucket)
            await session.flush()

            session.add(
                Signal(
                    bucket_id=bucket.id,
                    model_prob=0.999,
                    mkt_prob=0.995,
                    raw_edge=0.004,
                    exec_cost=0.006,
                    true_edge=-0.002,
                    reason_json='{"prob_new_high":0.001,"prob_hotter_bucket":0.001,"prob_new_high_raw":0.49,"lock_regime":true,"observed_bucket_idx":0,"observed_bucket_upper_f":70.0,"city_state":"resolved"}',
                    gate_failures_json="[]",
                    computed_at=fetched_at,
                )
            )
            await session.commit()

    _run(seed())
    response = _run(api_routes.get_city_signals(city.city_slug))

    assert len(response) == 1
    assert response[0]["prob_new_high"] == 0.001
    assert response[0]["prob_hotter_bucket"] == 0.001
    assert response[0]["prob_new_high_raw"] == 0.49
    assert response[0]["lock_regime"] is True
    assert response[0]["observed_bucket_idx"] == 0
    assert response[0]["observed_bucket_upper_f"] == 70.0
    assert response[0]["city_state"] == "resolved"

    _run(engine.dispose())
