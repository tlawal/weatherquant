import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.api.routes as api_routes
import backend.storage.db as storage_db
from backend.storage.models import Base, City, ForecastObs, MetarObs, WorkerHeartbeat
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
