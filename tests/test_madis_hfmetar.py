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
from backend.storage.models import Base, City, MadisObs


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
    """Records: list of (station_str, temp_k, epoch_s)."""

    def __init__(self, records):
        self.variables = {
            # Real MADIS HFMETAR files expose stationId (4-char ICAO) as the
            # match field; stationName is the long-form location name.
            "stationId": _Var([_StationName(s) for s, _, _ in records]),
            "stationName": _Var([_StationName("LONG NAME") for _ in records]),
            "temperature": _Var([t for _, t, _ in records]),
            "timeObs": _Var([e for _, _, e in records]),
        }

    def close(self):
        pass


def _install_fake_netcdf(monkeypatch, records):
    """Patch sys.modules so `import netCDF4` inside fetch_madis_latest
    resolves to our fake."""

    def _ctor(path, mode):
        return _FakeDataset(records)

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

        count = await _count_madis(session_factory)
        assert count == 2
        assert any(
            "fetched 2 observations" in r.getMessage() for r in caplog.records
        ), f"expected success log; got: {[r.getMessage() for r in caplog.records]}"

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
