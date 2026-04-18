"""Tests for backend/ingestion/madis_hfmetar.py.

These tests avoid requiring the real netCDF4 C library by monkeypatching
sys.modules["netCDF4"] with a fake Dataset factory. They also avoid real
HTTPS by monkeypatching _fetch_netcdf (for data-path tests) or by hand-
rolling a fake aiohttp session (for the TLS-error test).
"""
from __future__ import annotations

import asyncio
import gzip
import logging
import ssl
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.ingestion.madis_hfmetar as madis_mod
import backend.storage.db as storage_db
from backend.storage.models import Base, City, MadisObs, MetarObs, MetarObsExtended


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "madis_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


async def _create_atlanta(session_factory):
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


async def _count_madis(session_factory):
    async with session_factory() as session:
        result = await session.execute(select(func.count()).select_from(MadisObs))
        return int(result.scalar_one())


async def _count_metar(session_factory, source=None):
    async with session_factory() as session:
        q = select(func.count()).select_from(MetarObs)
        if source is not None:
            q = q.where(MetarObs.source == source)
        result = await session.execute(q)
        return int(result.scalar_one())


async def _fetch_metar_rows(session_factory, source=None):
    async with session_factory() as session:
        q = select(MetarObs)
        if source is not None:
            q = q.where(MetarObs.source == source)
        q = q.order_by(MetarObs.observed_at)
        rows = (await session.execute(q)).scalars().all()
        return list(rows)


async def _fetch_extended(session_factory, metar_obs_id):
    async with session_factory() as session:
        result = await session.execute(
            select(MetarObsExtended).where(
                MetarObsExtended.metar_obs_id == metar_obs_id
            )
        )
        return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Fake netCDF4.Dataset — minimal surface to let the parser run without the
# real HDF5/netCDF C stack.
# ---------------------------------------------------------------------------
class _StationName:
    """Mimic a numpy char-array entry: has `.tobytes()` returning ASCII bytes."""

    def __init__(self, name: str) -> None:
        self._name = name.encode("ascii").ljust(8, b" ")

    def tobytes(self) -> bytes:
        return self._name


class _Var:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, key):
        # netCDF4 variables support `[:]` to copy the whole array; return the
        # underlying list unchanged.
        return self._values


class _FakeDataset:
    """Records: list of dicts with keys: station, temp_k, epoch,
    and optionally dewpoint_k, rh, wind_dir, wind_speed_ms, wind_gust_ms,
    altimeter_pa, precip_m."""

    def __init__(self, records):
        def _col(key, default=None):
            return [r.get(key, default) for r in records]

        self.variables = {
            # Required.
            "stationId": _Var([_StationName(r["station"]) for r in records]),
            "stationName": _Var([_StationName("LONG NAME") for _ in records]),
            "temperature": _Var(_col("temp_k")),
            "timeObs": _Var(_col("epoch")),
            # Optional — only include the key when at least one record has it.
            **({"dewpoint": _Var(_col("dewpoint_k", 1e36))}
               if any("dewpoint_k" in r for r in records) else {}),
            **({"relHumidity": _Var(_col("rh", 1e36))}
               if any("rh" in r for r in records) else {}),
            **({"windDir": _Var(_col("wind_dir", -9999))}
               if any("wind_dir" in r for r in records) else {}),
            **({"windSpeed": _Var(_col("wind_speed_ms", -9999))}
               if any("wind_speed_ms" in r for r in records) else {}),
            **({"windGust": _Var(_col("wind_gust_ms", -9999))}
               if any("wind_gust_ms" in r for r in records) else {}),
            **({"altimeter": _Var(_col("altimeter_pa", -9999))}
               if any("altimeter_pa" in r for r in records) else {}),
            **({"precip1hr": _Var(_col("precip_m", -9999))}
               if any("precip_m" in r for r in records) else {}),
        }

    def close(self):
        pass


def _install_fake_netcdf(monkeypatch, records):
    """Patch sys.modules so `import netCDF4` inside fetch_madis_latest
    resolves to our fake. `records` is a list of dicts (see _FakeDataset).
    For backward compat, also accepts tuples (station, temp_k, epoch)."""

    def _normalize(r):
        if isinstance(r, tuple):
            return {"station": r[0], "temp_k": r[1], "epoch": r[2]}
        return dict(r)

    normed = [_normalize(r) for r in records]

    def _ctor(path, mode):
        return _FakeDataset(normed)

    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=_ctor))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_fetch_madis_inserts_new_observations(tmp_path, monkeypatch, caplog):
    """Happy path: two fresh KATL observations get inserted."""

    async def go():
        _, session_factory = await _setup_test_db(tmp_path, monkeypatch)
        await _create_atlanta(session_factory)

        async def fake_fetch(filename, http):
            return gzip.compress(b"fake-netcdf-bytes")

        monkeypatch.setattr(madis_mod, "_fetch_netcdf", fake_fetch)
        # Also reset the module global so a prior test's success doesn't
        # leak into this one.
        monkeypatch.setattr(madis_mod, "_last_success_file", None)

        t0 = int(datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc).timestamp())
        _install_fake_netcdf(
            monkeypatch,
            [("KATL", 293.15, t0), ("KATL", 294.15, t0 + 300)],
        )

        with caplog.at_level(logging.INFO, logger="backend.ingestion.madis_hfmetar"):
            await madis_mod.fetch_madis_latest()

        # Legacy MadisObs rows still get written.
        assert await _count_madis(session_factory) == 2
        # Unified MetarObs rows (source='madis') — this is the new primary path.
        assert await _count_metar(session_factory, source="madis") == 2
        assert any(
            "fetched 2 observations" in r.getMessage() for r in caplog.records
        ), f"expected success log; got: {[r.getMessage() for r in caplog.records]}"

    _run(go())


def test_fetch_madis_writes_extended_fields(tmp_path, monkeypatch):
    """Dew point, humidity, wind, gust, altimeter, precip get parsed with
    proper unit conversions and stored in MetarObsExtended."""

    async def go():
        _, session_factory = await _setup_test_db(tmp_path, monkeypatch)
        await _create_atlanta(session_factory)

        async def fake_fetch(filename, http):
            return gzip.compress(b"fake")

        monkeypatch.setattr(madis_mod, "_fetch_netcdf", fake_fetch)
        monkeypatch.setattr(madis_mod, "_last_success_file", None)

        t0 = int(datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc).timestamp())
        _install_fake_netcdf(
            monkeypatch,
            [{
                "station": "KATL",
                "temp_k": 297.15,      # 24°C / 75.2°F
                "epoch": t0,
                "dewpoint_k": 288.15,   # 15°C / 59°F
                "rh": 55.0,             # 55%
                "wind_dir": 270,
                "wind_speed_ms": 5.0,   # 5 m/s ≈ 9.7 kt
                "wind_gust_ms": 8.0,    # 8 m/s ≈ 15.5 kt
                "altimeter_pa": 101325, # 1 atm ≈ 29.92 inHg
                "precip_m": 0.0025,     # 2.5 mm ≈ 0.098 in
            }],
        )

        await madis_mod.fetch_madis_latest()

        rows = await _fetch_metar_rows(session_factory, source="madis")
        assert len(rows) == 1
        row = rows[0]
        assert row.source == "madis"
        assert row.metar_station == "KATL"
        assert row.temp_f is not None
        assert 74.5 < row.temp_f < 76.0  # ~75.2

        ext = await _fetch_extended(session_factory, row.id)
        assert ext is not None, "MetarObsExtended row must exist for extended data"
        assert ext.dewpoint_f is not None and 58.0 < ext.dewpoint_f < 60.0
        assert ext.humidity_pct == 55.0
        assert ext.wind_dir_deg == 270
        assert ext.wind_speed_kt is not None and 9.0 < ext.wind_speed_kt < 10.5
        assert ext.wind_gust_kt is not None and 15.0 < ext.wind_gust_kt < 16.0
        assert ext.altimeter_inhg is not None and 29.8 < ext.altimeter_inhg < 30.0
        assert ext.precip_in is not None and 0.09 < ext.precip_in < 0.11

    _run(go())


def test_fetch_madis_skips_extended_when_no_optional_vars(tmp_path, monkeypatch):
    """When the netCDF file lacks dewpoint/wind/etc., we still write the
    MetarObs row but omit the MetarObsExtended row."""

    async def go():
        _, session_factory = await _setup_test_db(tmp_path, monkeypatch)
        await _create_atlanta(session_factory)

        async def fake_fetch(filename, http):
            return gzip.compress(b"fake")

        monkeypatch.setattr(madis_mod, "_fetch_netcdf", fake_fetch)
        monkeypatch.setattr(madis_mod, "_last_success_file", None)

        t0 = int(datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc).timestamp())
        _install_fake_netcdf(monkeypatch, [("KATL", 293.15, t0)])  # no extended keys

        await madis_mod.fetch_madis_latest()

        rows = await _fetch_metar_rows(session_factory, source="madis")
        assert len(rows) == 1
        ext = await _fetch_extended(session_factory, rows[0].id)
        assert ext is None

    _run(go())


def test_fetch_madis_dedupes_on_repoll(tmp_path, monkeypatch):
    """Re-polling the same file does not re-insert existing observations."""

    async def go():
        _, session_factory = await _setup_test_db(tmp_path, monkeypatch)
        await _create_atlanta(session_factory)

        async def fake_fetch(filename, http):
            return gzip.compress(b"fake")

        monkeypatch.setattr(madis_mod, "_fetch_netcdf", fake_fetch)
        monkeypatch.setattr(madis_mod, "_last_success_file", None)

        t0 = int(datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc).timestamp())
        _install_fake_netcdf(monkeypatch, [("KATL", 293.15, t0)])

        await madis_mod.fetch_madis_latest()
        first = await _count_madis(session_factory)
        # Second poll of an identical payload — dedupe by (station, observed_at)
        # must prevent duplicate inserts, even though _last_success_file no
        # longer short-circuits the fetch.
        await madis_mod.fetch_madis_latest()
        second = await _count_madis(session_factory)

        assert first == 1
        assert second == 1

    _run(go())


def test_fetch_madis_all_404_logs_last_url(tmp_path, monkeypatch, caplog):
    """When every hourly probe 404s, the terminal log includes the last URL
    attempted so an operator can reproduce with `curl -k`."""

    async def go():
        _, session_factory = await _setup_test_db(tmp_path, monkeypatch)
        await _create_atlanta(session_factory)

        async def fake_fetch(filename, http):
            return None  # every step 404s

        monkeypatch.setattr(madis_mod, "_fetch_netcdf", fake_fetch)
        monkeypatch.setattr(madis_mod, "_last_success_file", None)

        with caplog.at_level(logging.WARNING, logger="backend.ingestion.madis_hfmetar"):
            await madis_mod.fetch_madis_latest()

        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "no file found" in m
            and "last url=https://madis-data.ncep.noaa.gov" in m
            for m in msgs
        ), f"expected terminal log with last url; got: {msgs}"

    _run(go())


def test_fetch_netcdf_tls_error_logs_distinct_message(caplog):
    """_fetch_netcdf catches SSL/TLS errors separately, logs at ERROR with a
    distinct tag, and does NOT retry (retrying a handshake failure is
    wasted budget)."""

    call_count = {"n": 0}

    class _TLSFailingGet:
        def __init__(self, url):
            call_count["n"] += 1

        async def __aenter__(self):
            raise ssl.SSLError("test: cert chain build failure")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeSession:
        def get(self, url):
            return _TLSFailingGet(url)

    async def go():
        with caplog.at_level(logging.ERROR, logger="backend.ingestion.madis_hfmetar"):
            result = await madis_mod._fetch_netcdf(
                "20260417_2200.gz", _FakeSession()
            )
        assert result is None
        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "TLS verification error" in m
            and "url=https://madis-data.ncep.noaa.gov" in m
            and "20260417_2200.gz" in m
            for m in msgs
        ), f"expected TLS error log; got: {msgs}"
        # No retries on TLS failure.
        assert call_count["n"] == 1, (
            f"expected 1 TLS attempt (no retries); got {call_count['n']}"
        )

    _run(go())
