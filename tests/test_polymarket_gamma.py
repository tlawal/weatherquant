import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.ingestion.polymarket_gamma as gamma_ingestion
import backend.storage.db as storage_db
from backend.storage.models import Base, Bucket, City, Event


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "gamma_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _create_city(session_factory, slug: str = "atlanta", name: str = "Atlanta") -> City:
    async with session_factory() as session:
        city = City(
            city_slug=slug,
            display_name=name,
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


async def _get_event_and_buckets(session_factory, city_id: int, date_et: str):
    async with session_factory() as session:
        event = (
            await session.execute(
                select(Event).where(Event.city_id == city_id, Event.date_et == date_et)
            )
        ).scalar_one()
        buckets = (
            await session.execute(
                select(Bucket)
                .where(Bucket.event_id == event.id)
                .order_by(Bucket.bucket_idx)
            )
        ).scalars().all()
        return event, list(buckets)


def _market(question: str, yes_token: str, no_token: str) -> dict:
    return {
        "question": question,
        "clobTokenIds": [yes_token, no_token],
    }


def test_process_event_data_normalizes_unsorted_ranges(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory))

    event_data = {
        "id": "evt-atlanta",
        "resolutionSource": "This market resolves using Wunderground observations.",
        "markets": [
            _market("Will the highest temperature in Atlanta be between 60-61°F on March 30?", "yes-60", "no-60"),
            _market("Will the highest temperature in Atlanta be between 68-69°F on March 30?", "yes-68", "no-68"),
            _market("Will the highest temperature in Atlanta be between 72-73°F on March 30?", "yes-72", "no-72"),
            _market("Will the highest temperature in Atlanta be between 58-59°F on March 30?", "yes-58", "no-58"),
            _market("Will the highest temperature in Atlanta be between 62-63°F on March 30?", "yes-62", "no-62"),
        ],
    }

    _run(
        gamma_ingestion._process_event_data(
            city, "2026-03-30", "highest-temperature-in-atlanta", event_data
        )
    )
    event, buckets = _run(_get_event_and_buckets(session_factory, city.id, "2026-03-30"))

    assert event.status == "ok"
    assert [bucket.bucket_idx for bucket in buckets] == [0, 1, 2, 3, 4]
    assert [(bucket.low_f, bucket.high_f) for bucket in buckets] == [
        (58.0, 59.0),
        (60.0, 61.0),
        (62.0, 63.0),
        (68.0, 69.0),
        (72.0, 73.0),
    ]
    assert {
        (bucket.low_f, bucket.high_f): bucket.yes_token_id for bucket in buckets
    } == {
        (58.0, 59.0): "yes-58",
        (60.0, 61.0): "yes-60",
        (62.0, 63.0): "yes-62",
        (68.0, 69.0): "yes-68",
        (72.0, 73.0): "yes-72",
    }

    _run(engine.dispose())


def test_process_event_data_orders_open_ended_buckets(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory, slug="testville", name="Testville"))

    event_data = {
        "id": "evt-open-ended",
        "resolutionSource": "Weather Underground is the resolution source.",
        "markets": [
            _market("Will the highest temperature in Testville be above 80°F on March 30?", "yes-above", "no-above"),
            _market("Will the highest temperature in Testville be between 70-71°F on March 30?", "yes-70", "no-70"),
            _market("Will the highest temperature in Testville be below 60°F on March 30?", "yes-below", "no-below"),
            _market("Will the highest temperature in Testville be between 60-61°F on March 30?", "yes-60", "no-60"),
        ],
    }

    _run(
        gamma_ingestion._process_event_data(
            city, "2026-03-30", "highest-temperature-in-testville", event_data
        )
    )
    event, buckets = _run(_get_event_and_buckets(session_factory, city.id, "2026-03-30"))

    assert event.status == "ok"
    assert [bucket.bucket_idx for bucket in buckets] == [0, 1, 2, 3]
    assert [(bucket.low_f, bucket.high_f) for bucket in buckets] == [
        (None, 60.0),
        (60.0, 61.0),
        (70.0, 71.0),
        (80.0, None),
    ]
    assert [bucket.yes_token_id for bucket in buckets] == [
        "yes-below",
        "yes-60",
        "yes-70",
        "yes-above",
    ]

    _run(engine.dispose())


def test_extract_resolution_url_nws_site():
    event_data = {
        "resolutionSource": (
            "Resolves per https://www.weather.gov/wrh/timeseries?site=KATL&hours=48"
        )
    }
    url, station = gamma_ingestion._extract_resolution_url(event_data)
    assert station == "KATL"
    assert url and "site=KATL" in url


def test_extract_resolution_url_wu_history():
    event_data = {
        "description": (
            "This market resolves using "
            "https://www.wunderground.com/history/daily/us/tx/houston/KHOU/date/2026-4-10"
        )
    }
    url, station = gamma_ingestion._extract_resolution_url(event_data)
    assert station == "KHOU"
    assert url and "wunderground.com" in url


def test_extract_resolution_url_wu_hourly():
    event_data = {
        "description": (
            "Hourly: https://www.wunderground.com/hourly/us/tx/houston/KHOU/date/2026-4-10"
        )
    }
    _, station = gamma_ingestion._extract_resolution_url(event_data)
    assert station == "KHOU"


def test_extract_resolution_url_wu_airport_history():
    event_data = {
        "description": (
            "https://www.wunderground.com/history/airport/KHOU/2026/4/10/DailyHistory.html"
        )
    }
    _, station = gamma_ingestion._extract_resolution_url(event_data)
    assert station == "KHOU"


def test_extract_resolution_url_alias_fallback():
    event_data = {
        "resolutionSource": (
            "Resolves to the daily high reported at William P. Hobby Airport."
        )
    }
    _, station = gamma_ingestion._extract_resolution_url(event_data)
    assert station == "KHOU"


def test_extract_resolution_url_no_match_returns_none_station():
    event_data = {
        "resolutionSource": "Resolves at the official high temperature for the day.",
    }
    _, station = gamma_ingestion._extract_resolution_url(event_data)
    assert station is None
