import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.engine.signal_engine as signal_engine
import backend.storage.db as storage_db
from backend.storage.models import (
    Base,
    Bucket,
    City,
    Event,
    ForecastObs,
    MarketSnapshot,
    MetarObs,
    ModelSnapshot,
)


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "signal_engine_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _create_city(session_factory, slug: str = "la") -> City:
    async with session_factory() as session:
        city = City(
            city_slug=slug,
            display_name="Los Angeles",
            metar_station="KCQT",
            enabled=True,
            is_us=True,
            unit="F",
            tz="America/Los_Angeles",
            lat=34.05,
            lon=-118.24,
        )
        session.add(city)
        await session.commit()
        await session.refresh(city)
        return city


def test_verified_settlement_uses_official_floor_and_preserves_raw_high(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def _empty_bins(city_id: int, days_back: int = 30):
        return []

    monkeypatch.setattr(signal_engine, "get_reliability_metrics", _empty_bins)

    date_et = "2026-04-16"
    city_tz = ZoneInfo(city.tz)
    midday = datetime(2026, 4, 16, 12, 53, tzinfo=city_tz)
    evening = datetime(2026, 4, 16, 18, 5, tzinfo=city_tz)

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                forecast_quality="ok",
                trading_enabled=True,
                settlement_source_verified=True,
                gamma_slug=f"{city.city_slug}-{date_et}",
            )
            session.add(event)
            await session.flush()

            buckets = [
                Bucket(event_id=event.id, bucket_idx=0, label="68-69", low_f=68.0, high_f=69.0),
                Bucket(event_id=event.id, bucket_idx=1, label="70-71", low_f=70.0, high_f=71.0),
                Bucket(event_id=event.id, bucket_idx=2, label="72-73", low_f=72.0, high_f=73.0),
            ]
            session.add_all(buckets)
            await session.flush()

            session.add_all(
                [
                    MetarObs(
                        city_id=city.id,
                        metar_station=city.metar_station,
                        observed_at=midday,
                        fetched_at=midday,
                        temp_c=21.0,
                        temp_f=69.8,
                        daily_high_f=69.8,
                    ),
                    MetarObs(
                        city_id=city.id,
                        metar_station=city.metar_station,
                        observed_at=evening,
                        fetched_at=evening,
                        temp_c=18.0,
                        temp_f=64.4,
                        daily_high_f=69.8,
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="nws",
                        date_et=date_et,
                        fetched_at=evening,
                        high_f=70.0,
                        raw_payload_hash="nws",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_hourly",
                        date_et=date_et,
                        fetched_at=evening,
                        high_f=69.8,
                        raw_payload_hash="wu_hourly",
                        raw_json=json.dumps(
                            {
                                "peak_hour": "7:00 PM PDT",
                                "peak_hour_local": "7:00 PM PDT",
                                "peak_hour_local_mins": 19 * 60,
                            }
                        ),
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_history",
                        date_et=date_et,
                        fetched_at=evening,
                        high_f=69.1,
                        raw_payload_hash="wu_history",
                        raw_json=json.dumps({"obs_time": "12:53 PM PDT"}),
                    ),
                ]
            )

            for idx, bucket in enumerate(buckets):
                session.add(
                    MarketSnapshot(
                        bucket_id=bucket.id,
                        fetched_at=evening,
                        yes_bid=0.45 - idx * 0.1,
                        yes_ask=0.55 - idx * 0.1,
                        yes_mid=0.50 - idx * 0.1,
                        yes_bid_depth=200.0,
                        yes_ask_depth=200.0,
                        spread=0.02,
                    )
                )

            await session.commit()

    async def exercise():
        await seed()
        signals = await signal_engine._compute_city_signals(city, date_et)
        async with session_factory() as session:
            snapshot = (
                await session.execute(
                    select(ModelSnapshot).order_by(ModelSnapshot.id.desc()).limit(1)
                )
            ).scalar_one()
        return signals, json.loads(snapshot.inputs_json)

    signals, inputs = _run(exercise())

    assert len(signals) == 3
    assert inputs["ground_truth_high"] == 69.1
    assert inputs["ground_truth_source"] == "wu_history"
    assert inputs["observed_high_floor"] == 69.1
    assert inputs["observed_high_floor_source"] == "wu_history"
    assert inputs["official_observed_high"] == 69.1
    assert inputs["official_observed_high_source"] == "wu_history"
    assert inputs["raw_observed_high"] == 69.8
    assert inputs["raw_observed_high_source"] == "raw_metar"

    first_reason = signals[0].reason
    assert first_reason["ground_truth_high"] == 69.1
    assert first_reason["ground_truth_source"] == "wu_history"
    assert first_reason["official_observed_high"] == 69.1
    assert first_reason["raw_observed_high"] == 69.8

    _run(engine.dispose())


def test_verified_settlement_prefers_tgftp_over_wu_history(tmp_path, monkeypatch):
    """When a tgftp MetarObs row exists, it must be the preferred settlement
    floor source for verified events — even if wu_history reports a higher value."""
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    async def _empty_bins(city_id: int, days_back: int = 30):
        return []

    monkeypatch.setattr(signal_engine, "get_reliability_metrics", _empty_bins)

    date_et = "2026-04-16"
    city_tz = ZoneInfo(city.tz)
    midday = datetime(2026, 4, 16, 12, 53, tzinfo=city_tz)
    evening = datetime(2026, 4, 16, 18, 5, tzinfo=city_tz)

    async def seed():
        async with session_factory() as session:
            event = Event(
                city_id=city.id,
                date_et=date_et,
                status="ok",
                forecast_quality="ok",
                trading_enabled=True,
                settlement_source_verified=True,
                gamma_slug=f"{city.city_slug}-{date_et}",
            )
            session.add(event)
            await session.flush()

            buckets = [
                Bucket(event_id=event.id, bucket_idx=0, label="68-69", low_f=68.0, high_f=69.0),
                Bucket(event_id=event.id, bucket_idx=1, label="70-71", low_f=70.0, high_f=71.0),
                Bucket(event_id=event.id, bucket_idx=2, label="72-73", low_f=72.0, high_f=73.0),
            ]
            session.add_all(buckets)
            await session.flush()

            session.add_all(
                [
                    # tgftp official high = 69.1 (lower but more authoritative)
                    MetarObs(
                        city_id=city.id,
                        metar_station=city.metar_station,
                        observed_at=midday,
                        fetched_at=midday,
                        temp_c=20.6,
                        temp_f=69.1,
                        daily_high_f=69.1,
                        source="tgftp",
                    ),
                    # Raw METAR high = 69.8 (hotter off-cycle reading)
                    MetarObs(
                        city_id=city.id,
                        metar_station=city.metar_station,
                        observed_at=midday,
                        fetched_at=midday,
                        temp_c=21.0,
                        temp_f=69.8,
                        daily_high_f=69.8,
                    ),
                    MetarObs(
                        city_id=city.id,
                        metar_station=city.metar_station,
                        observed_at=evening,
                        fetched_at=evening,
                        temp_c=18.0,
                        temp_f=64.4,
                        daily_high_f=69.8,
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="nws",
                        date_et=date_et,
                        fetched_at=evening,
                        high_f=70.0,
                        raw_payload_hash="nws",
                        raw_json="{}",
                    ),
                    ForecastObs(
                        city_id=city.id,
                        source="wu_hourly",
                        date_et=date_et,
                        fetched_at=evening,
                        high_f=69.8,
                        raw_payload_hash="wu_hourly",
                        raw_json=json.dumps(
                            {
                                "peak_hour": "7:00 PM PDT",
                                "peak_hour_local": "7:00 PM PDT",
                                "peak_hour_local_mins": 19 * 60,
                            }
                        ),
                    ),
                    # wu_history reports 69.8 (matches raw METAR, not tgftp)
                    ForecastObs(
                        city_id=city.id,
                        source="wu_history",
                        date_et=date_et,
                        fetched_at=evening,
                        high_f=69.8,
                        raw_payload_hash="wu_history",
                        raw_json=json.dumps({"obs_time": "12:53 PM PDT"}),
                    ),
                ]
            )

            for idx, bucket in enumerate(buckets):
                session.add(
                    MarketSnapshot(
                        bucket_id=bucket.id,
                        fetched_at=evening,
                        yes_bid=0.45 - idx * 0.1,
                        yes_ask=0.55 - idx * 0.1,
                        yes_mid=0.50 - idx * 0.1,
                        yes_bid_depth=200.0,
                        yes_ask_depth=200.0,
                        spread=0.02,
                    )
                )

            await session.commit()

    async def exercise():
        await seed()
        signals = await signal_engine._compute_city_signals(city, date_et)
        async with session_factory() as session:
            snapshot = (
                await session.execute(
                    select(ModelSnapshot).order_by(ModelSnapshot.id.desc()).limit(1)
                )
            ).scalar_one()
        return signals, json.loads(snapshot.inputs_json)

    signals, inputs = _run(exercise())

    assert len(signals) == 3
    # tgftp (69.1) must be preferred over wu_history (69.8) for verified events
    assert inputs["ground_truth_high"] == 69.1
    assert inputs["ground_truth_source"] == "tgftp"
    assert inputs["observed_high_floor"] == 69.1
    assert inputs["observed_high_floor_source"] == "tgftp"
    assert inputs["official_observed_high"] == 69.1
    assert inputs["official_observed_high_source"] == "tgftp"
    # Raw high is still 69.8 from the unfiltered METAR
    assert inputs["raw_observed_high"] == 69.8

    _run(engine.dispose())
