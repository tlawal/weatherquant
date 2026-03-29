import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.ingestion.metar as metar_ingestion
import backend.storage.db as storage_db
from backend.storage.models import Base, City, MetarObs, MetarObsExtended
from backend.storage.repos import insert_metar_obs, upsert_metar_obs_extended


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "metar_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _create_city(session_factory, slug: str = "atlanta", station: str = "KATL") -> City:
    async with session_factory() as session:
        city = City(
            city_slug=slug,
            display_name="Atlanta",
            metar_station=station,
            enabled=True,
            is_us=True,
            unit="F",
            tz="America/New_York",
        )
        session.add(city)
        await session.commit()
        await session.refresh(city)
        return city


async def _count_metar_rows(session_factory) -> int:
    async with session_factory() as session:
        result = await session.execute(select(func.count()).select_from(MetarObs))
        return int(result.scalar_one())


async def _get_obs_and_ext(session_factory, metar_obs_id: int):
    async with session_factory() as session:
        obs = await session.get(MetarObs, metar_obs_id)
        ext = await session.scalar(
            select(MetarObsExtended).where(MetarObsExtended.metar_obs_id == metar_obs_id)
        )
        return obs, ext


def _sample_nws_props(**overrides):
    props = {
        "source": "nws_obs",
        "timestamp": "2026-03-29T14:05:00+00:00",
        "temperature": {"value": 20.0, "unitCode": "wmoUnit:degC"},
        "dewpoint": {"value": 10.0, "unitCode": "wmoUnit:degC"},
        "relativeHumidity": {"value": 65.4, "unitCode": "wmoUnit:percent"},
        "windDirection": {"value": 180, "unitCode": "wmoUnit:degree_(angle)"},
        "windSpeed": {"value": 18.52, "unitCode": "wmoUnit:km_h-1"},
        "windGust": {"value": 8.0, "unitCode": "wmoUnit:m_s-1"},
        "barometricPressure": {"value": 101325, "unitCode": "wmoUnit:Pa"},
        "seaLevelPressure": {"value": 1016.7, "unitCode": "wmoUnit:hPa"},
        "visibility": {"value": 16093.44, "unitCode": "wmoUnit:m"},
        "precipitationLastHour": {"value": 5.08, "unitCode": "wmoUnit:mm"},
        "cloudLayers": [
            {"amount": "BKN", "base": {"value": 1524, "unitCode": "wmoUnit:m"}}
        ],
        "presentWeather": [{"intensity": "light", "weather": "rain"}],
        "textDescription": "Light Rain",
        "rawMessage": "KATL 291405Z 18010KT 10SM -RA BKN050 20/10 A2992",
    }
    props.update(overrides)
    return props


def _sample_aviationweather_obs(**overrides):
    obs = {
        "stationId": "KATL",
        "obsTime": "2026-03-29T14:05:00Z",
        "reportTime": "2026-03-29T14:05:00Z",
        "temp": 20.0,
        "dewp": 10.0,
        "wdir": 180,
        "wspd": 10.0,
        "wgst": 18.0,
        "altim": 29.92,
        "slp": 1016.7,
        "visib": 10.0,
        "clouds": [{"cover": "BKN", "base": 5000}],
        "wxString": "-RA",
        "rawOb": "KATL 291405Z 18010G18KT 10SM -RA BKN050 20/10 A2992",
    }
    obs.update(overrides)
    return obs


def test_parse_nws_extended_full_payload():
    ext = metar_ingestion.parse_nws_extended(_sample_nws_props())

    assert ext["dewpoint_c"] == 10.0
    assert ext["dewpoint_f"] == 50.0
    assert ext["humidity_pct"] == 65.4
    assert ext["wind_dir_deg"] == 180
    assert ext["wind_speed_kt"] == 10.0
    assert ext["wind_gust_kt"] == pytest.approx(15.6, abs=0.1)
    assert ext["altimeter_inhg"] == pytest.approx(29.92, abs=0.01)
    assert ext["sea_level_pressure_mb"] == 1016.7
    assert ext["precip_in"] == 0.2
    assert ext["cloud_cover"] == "BKN"
    assert ext["cloud_base_ft"] == 5000
    assert ext["condition"] == "Rain"
    assert "rain" in ext["wx_string"].lower()


def test_parse_nws_extended_missing_values_and_no_signal():
    ext = metar_ingestion.parse_nws_extended(
        _sample_nws_props(
            dewpoint={"value": None, "unitCode": "wmoUnit:degC"},
            relativeHumidity={"value": None, "unitCode": "wmoUnit:percent"},
            windSpeed={"value": None, "unitCode": "wmoUnit:km_h-1"},
            windGust={"value": None, "unitCode": "wmoUnit:km_h-1"},
            barometricPressure={"value": None, "unitCode": "wmoUnit:Pa"},
            seaLevelPressure={"value": None, "unitCode": "wmoUnit:hPa"},
            precipitationLastHour={"value": None, "unitCode": "wmoUnit:mm"},
            cloudLayers=[],
            presentWeather=[],
            textDescription=None,
        )
    )

    assert "dewpoint_c" not in ext
    assert "humidity_pct" not in ext
    assert "wind_speed_kt" not in ext
    assert "altimeter_inhg" not in ext
    assert "precip_in" not in ext
    assert "condition" not in ext


def test_parse_nws_extended_unit_conversions():
    ext = metar_ingestion.parse_nws_extended(
        _sample_nws_props(
            windSpeed={"value": 11.5078, "unitCode": "wmoUnit:m_s-1"},
            windGust={"value": 23.0156, "unitCode": "wmoUnit:km_h-1"},
            barometricPressure={"value": 1013.25, "unitCode": "wmoUnit:hPa"},
            seaLevelPressure={"value": 29.92, "unitCode": "unit:inHg"},
            precipitationLastHour={"value": 0.25, "unitCode": "unit:in"},
        )
    )

    assert ext["wind_speed_kt"] == pytest.approx(22.4, abs=0.1)
    assert ext["wind_gust_kt"] == pytest.approx(12.4, abs=0.1)
    assert ext["altimeter_inhg"] == pytest.approx(29.92, abs=0.01)
    assert ext["sea_level_pressure_mb"] == pytest.approx(1013.2, abs=0.1)
    assert ext["precip_in"] == 0.25


def test_parse_nws_condition_is_short_and_normalized():
    ext = metar_ingestion.parse_nws_extended(
        _sample_nws_props(
            presentWeather=[],
            textDescription="Thunderstorms with heavy rain, fog, and gusty winds across the metro area",
        )
    )

    assert ext["condition"] == "Thunderstorm"
    assert len(ext["condition"]) <= 32


def test_upsert_metar_obs_extended_only_fills_missing_values(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory, slug="upsert-city"))
    observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    async def scenario():
        async with session_factory() as session:
            obs = await insert_metar_obs(
                session,
                city_id=city.id,
                metar_station=city.metar_station,
                observed_at=observed_at,
                report_at=observed_at,
                temp_c=20.0,
                temp_f=68.0,
                daily_high_f=68.0,
                raw_text=None,
                raw_json=json.dumps({"source": "seed"}),
            )
            await upsert_metar_obs_extended(
                session,
                metar_obs_id=obs.id,
                dewpoint_f=50.0,
            )
            await upsert_metar_obs_extended(
                session,
                metar_obs_id=obs.id,
                dewpoint_f=51.0,
                humidity_pct=60.0,
            )
            return obs.id

    obs_id = _run(scenario())
    _, ext = _run(_get_obs_and_ext(session_factory, obs_id))

    assert ext.dewpoint_f == 50.0
    assert ext.humidity_pct == 60.0
    _run(engine.dispose())


def test_merge_when_nws_arrives_first(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory, slug="nws-first"))
    observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    nws_props = _sample_nws_props(
        timestamp=observed_at.isoformat(),
        windGust={"value": None, "unitCode": "wmoUnit:m_s-1"},
    )
    aviation_obs = _sample_aviationweather_obs(
        obsTime=observed_at.isoformat().replace("+00:00", "Z"),
        reportTime=observed_at.isoformat().replace("+00:00", "Z"),
        wgst=18.0,
    )

    async def scenario():
        await metar_ingestion._insert_or_merge_metar_observation(
            city=city,
            station_id=city.metar_station,
            observed_at=observed_at,
            report_at=observed_at,
            temp_c=20.0,
            temp_f=68.0,
            raw_text=nws_props["rawMessage"],
            raw_json=json.dumps(nws_props),
            ext_data=metar_ingestion.parse_nws_extended(nws_props),
        )
        await metar_ingestion._insert_or_merge_metar_observation(
            city=city,
            station_id=city.metar_station,
            observed_at=observed_at,
            report_at=observed_at,
            temp_c=20.0,
            temp_f=68.0,
            raw_text=aviation_obs["rawOb"],
            raw_json=json.dumps(aviation_obs),
            ext_data=metar_ingestion.parse_aviationweather_extended(aviation_obs, temp_c=20.0),
        )

        async with session_factory() as session:
            obs = await session.scalar(select(MetarObs).where(MetarObs.city_id == city.id))
            ext = await session.scalar(
                select(MetarObsExtended).where(MetarObsExtended.metar_obs_id == obs.id)
            )
            return obs, ext

    obs, ext = _run(scenario())

    assert _run(_count_metar_rows(session_factory)) == 1
    assert ext.wind_gust_kt == 18.0
    assert ext.wind_speed_kt == 10.0
    assert '"source": "nws_obs"' in obs.raw_json
    _run(engine.dispose())


def test_merge_when_aviationweather_arrives_first(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory, slug="aviation-first"))
    observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    aviation_obs = _sample_aviationweather_obs(
        obsTime=observed_at.isoformat().replace("+00:00", "Z"),
        reportTime=observed_at.isoformat().replace("+00:00", "Z"),
    )
    nws_props = _sample_nws_props(
        timestamp=observed_at.isoformat(),
        dewpoint={"value": None, "unitCode": "wmoUnit:degC"},
        relativeHumidity={"value": None, "unitCode": "wmoUnit:percent"},
        precipitationLastHour={"value": 7.62, "unitCode": "wmoUnit:mm"},
    )

    async def scenario():
        await metar_ingestion._insert_or_merge_metar_observation(
            city=city,
            station_id=city.metar_station,
            observed_at=observed_at,
            report_at=observed_at,
            temp_c=20.0,
            temp_f=68.0,
            raw_text=aviation_obs["rawOb"],
            raw_json=json.dumps(aviation_obs),
            ext_data=metar_ingestion.parse_aviationweather_extended(aviation_obs, temp_c=20.0),
        )
        await metar_ingestion._insert_or_merge_metar_observation(
            city=city,
            station_id=city.metar_station,
            observed_at=observed_at,
            report_at=observed_at,
            temp_c=20.0,
            temp_f=68.0,
            raw_text=nws_props["rawMessage"],
            raw_json=json.dumps(nws_props),
            ext_data=metar_ingestion.parse_nws_extended(nws_props),
        )

        async with session_factory() as session:
            obs = await session.scalar(select(MetarObs).where(MetarObs.city_id == city.id))
            ext = await session.scalar(
                select(MetarObsExtended).where(MetarObsExtended.metar_obs_id == obs.id)
            )
            return obs, ext

    obs, ext = _run(scenario())

    assert _run(_count_metar_rows(session_factory)) == 1
    assert ext.precip_in == 0.3
    assert ext.dewpoint_f == 50.0
    assert '"source": "nws_obs"' not in obs.raw_json
    _run(engine.dispose())


def test_backfill_recent_nws_extended_is_idempotent(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    city = _run(_create_city(session_factory, slug="backfill-city"))
    observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(hours=1)
    nws_props = _sample_nws_props(timestamp=observed_at.isoformat())

    async def seed_row():
        async with session_factory() as session:
            obs = await insert_metar_obs(
                session,
                city_id=city.id,
                metar_station=city.metar_station,
                observed_at=observed_at,
                report_at=observed_at,
                temp_c=20.0,
                temp_f=68.0,
                daily_high_f=68.0,
                raw_text=nws_props["rawMessage"],
                raw_json=json.dumps(nws_props),
            )
            return obs.id

    obs_id = _run(seed_row())
    repaired_first = _run(metar_ingestion.backfill_recent_nws_extended())
    repaired_second = _run(metar_ingestion.backfill_recent_nws_extended())
    _, ext = _run(_get_obs_and_ext(session_factory, obs_id))

    assert repaired_first == 1
    assert repaired_second == 0
    assert ext is not None
    assert ext.condition == "Rain"
    _run(engine.dispose())
