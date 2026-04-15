import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.api.routes as api_routes
import backend.storage.db as storage_db
from backend.storage.models import (
    Base,
    Bucket,
    City,
    Event,
    ForecastObs,
    MetarObs,
    ModelSnapshot,
    Position,
    Signal,
    WorkerHeartbeat,
)
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


def test_unredeemed_wins_skips_events_without_positions(tmp_path, monkeypatch):
    """Regression for the phantom $0 unredeemed-wins panel: a resolved event
    with condition_ids but no Position rows must not surface as a claimable
    winning."""
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et="2026-04-10",
                status="ok",
                trading_enabled=True,
                resolved_at=datetime.utcnow(),
                winning_bucket_idx=1,
            )
            session.add(event)
            await session.flush()

            # Two buckets, both with condition_ids (resolvable), no positions.
            for idx, (lo, hi) in enumerate([(76.0, 77.0), (78.0, 79.0)]):
                session.add(
                    Bucket(
                        event_id=event.id,
                        bucket_idx=idx,
                        label=f"{int(lo)}-{int(hi)}°F",
                        low_f=lo,
                        high_f=hi,
                        yes_token_id=f"yes-{idx}",
                        no_token_id=f"no-{idx}",
                        condition_id=f"cond-{idx}",
                    )
                )
            await session.commit()

    _run(seed())
    result = _run(api_routes.unredeemed_wins())
    assert result == {"unredeemed": []}

    _run(engine.dispose())


def test_unredeemed_wins_includes_events_with_winning_position(tmp_path, monkeypatch):
    """Counterpart: when a real winning Position exists, the event surfaces."""
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et="2026-04-10",
                status="ok",
                trading_enabled=True,
                resolved_at=datetime.utcnow(),
                winning_bucket_idx=1,
            )
            session.add(event)
            await session.flush()

            winning_bucket = None
            for idx, (lo, hi) in enumerate([(76.0, 77.0), (78.0, 79.0)]):
                b = Bucket(
                    event_id=event.id,
                    bucket_idx=idx,
                    label=f"{int(lo)}-{int(hi)}°F",
                    low_f=lo,
                    high_f=hi,
                    yes_token_id=f"yes-{idx}",
                    no_token_id=f"no-{idx}",
                    condition_id=f"cond-{idx}",
                )
                session.add(b)
                if idx == 1:
                    winning_bucket = b
            await session.flush()

            assert winning_bucket is not None
            session.add(
                Position(
                    bucket_id=winning_bucket.id,
                    side="YES",
                    net_qty=10.0,
                    avg_cost=0.42,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    last_mkt_price=0.99,
                )
            )
            await session.commit()

    _run(seed())
    result = _run(api_routes.unredeemed_wins())
    assert len(result["unredeemed"]) == 1
    entry = result["unredeemed"][0]
    assert entry["winning_bucket_idx"] == 1
    assert entry["total_expected_payout"] == 10.0

    _run(engine.dispose())


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
