"""Phase C4 — Herbie side-channel harness.

Out-of-band fetcher for HRRR / NBM / IFS / AIFS via the Herbie library directly
against NOAA NOMADS + ECMWF Open Data + the AWS open-data registry. Writes one
row per (city, source, model_run, lead) to ``herbie_forecast_timing`` for
offline join against the production Open-Meteo path. **This module does not
feed the production ensemble.** It is observability + evidence for the day-90
promote/kill decision documented in the plan §C4.

If ``herbie-data`` (and its grib/cfgrib stack) isn't installed, every fetch is
a logged no-op — the rest of the system runs unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select

from backend.storage.db import get_session
from backend.storage.models import City, ForecastObs, HerbieForecastTiming
from backend.storage.repos import get_all_cities

log = logging.getLogger(__name__)


# Module-level import probe — checked once on first call.
_HERBIE_AVAILABLE: Optional[bool] = None
_HERBIE_IMPORT_ERROR: Optional[str] = None


def _have_herbie() -> bool:
    global _HERBIE_AVAILABLE, _HERBIE_IMPORT_ERROR
    if _HERBIE_AVAILABLE is not None:
        return _HERBIE_AVAILABLE
    try:
        import herbie  # noqa: F401
        import xarray  # noqa: F401
        import cfgrib  # noqa: F401
        _HERBIE_AVAILABLE = True
    except ImportError as e:
        _HERBIE_AVAILABLE = False
        _HERBIE_IMPORT_ERROR = str(e)
        log.warning("herbie not installed — side-channel disabled (%s)", e)
    return _HERBIE_AVAILABLE


# Source → (Herbie model name, run cadence hours, expected availability minutes after run)
_HERBIE_SOURCES = {
    "herbie_hrrr": {"model": "hrrr",   "runs_per_day": 24, "avail_minutes": 45,  "om_source": "hrrr"},
    "herbie_nbm":  {"model": "nbm",    "runs_per_day": 24, "avail_minutes": 90,  "om_source": "nbm"},
    "herbie_ifs":  {"model": "ifs",    "runs_per_day": 4,  "avail_minutes": 360, "om_source": "ecmwf_ifs"},
    "herbie_aifs": {"model": "aifs",   "runs_per_day": 4,  "avail_minutes": 360, "om_source": "ecmwf_aifs"},
}


@dataclass
class _HerbieResult:
    high_f: Optional[float]
    model_run_at: datetime
    fetched_at: datetime
    available_at: Optional[datetime]


def _most_recent_run(model: str, now_utc: datetime) -> datetime:
    """Snap now_utc back to the most-recent model initialization time."""
    if model in ("hrrr", "nbm"):
        return now_utc.replace(minute=0, second=0, microsecond=0)
    if model in ("ifs", "aifs"):
        # 00/06/12/18 UTC
        run_hour = (now_utc.hour // 6) * 6
        return now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    return now_utc.replace(minute=0, second=0, microsecond=0)


def _fetch_one_sync(
    *, model: str, lat: float, lon: float, run: datetime, lead_hours: int,
) -> Optional[_HerbieResult]:
    """Blocking Herbie call — extracts daily-max 2 m T over a ~24h window.

    Run on an executor so the asyncio loop isn't blocked by GRIB decode.
    Returns None on any decode/network failure (graceful degrade).
    """
    if not _have_herbie():
        return None
    try:
        from herbie import Herbie  # type: ignore
        import numpy as np  # noqa: F401
    except ImportError:
        return None

    fetched_at = datetime.now(timezone.utc)
    try:
        H = Herbie(run, model=model, fxx=lead_hours, save_dir="/tmp/herbie")
        ds = H.xarray(":TMP:2 m above ground:")
        # Subset to nearest grid cell.
        cell = ds.sel(latitude=lat, longitude=(lon % 360), method="nearest")
        t_kelvin = float(cell["t2m"].values) if "t2m" in cell else float(cell.to_array().values[0])
        high_f = (t_kelvin - 273.15) * 9 / 5 + 32
        return _HerbieResult(
            high_f=round(high_f, 2),
            model_run_at=run,
            fetched_at=fetched_at,
            available_at=getattr(H, "available_at", None),
        )
    except Exception as e:
        log.debug("herbie fetch failed model=%s run=%s lead=%d: %s", model, run, lead_hours, e)
        return None


async def _record(
    *, city: City, source: str, result: _HerbieResult, om_source: str,
) -> None:
    """Persist one HerbieForecastTiming row + join against latest Open-Meteo obs."""
    async with get_session() as sess:
        row_om = (
            await sess.execute(
                select(ForecastObs)
                .where(ForecastObs.city_id == city.id, ForecastObs.source == om_source)
                .order_by(ForecastObs.fetched_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        high_f_om = row_om.high_f if row_om else None
        om_fetched_at = row_om.fetched_at if row_om else None
        latency_delta = None
        if om_fetched_at is not None:
            latency_delta = (result.fetched_at - om_fetched_at).total_seconds()
        abs_diff = (
            abs(result.high_f - high_f_om)
            if (result.high_f is not None and high_f_om is not None)
            else None
        )

        sess.add(
            HerbieForecastTiming(
                city_id=city.id,
                source=source,
                model_run_at=result.model_run_at,
                lead_hours=float((result.model_run_at - result.fetched_at).total_seconds() / -3600.0),
                herbie_fetched_at=result.fetched_at,
                herbie_available_at=result.available_at,
                open_meteo_fetched_at=om_fetched_at,
                latency_delta_seconds=latency_delta,
                high_f_herbie=result.high_f,
                high_f_open_meteo=high_f_om,
                abs_diff_f=abs_diff,
            )
        )
        await sess.commit()


async def _run_source(source: str) -> None:
    """Fetch one (most recent) run × all enabled cities for a single source."""
    if not _have_herbie():
        return
    cfg = _HERBIE_SOURCES[source]
    model = cfg["model"]
    om_source = cfg["om_source"]

    now_utc = datetime.now(timezone.utc)
    run = _most_recent_run(model, now_utc)
    # Within the avail window? If not, skip — the model run may not have landed yet.
    minutes_since_run = (now_utc - run).total_seconds() / 60.0
    if minutes_since_run < cfg["avail_minutes"]:
        log.debug("herbie %s: run %s only %.0f min old, skipping (need %d)",
                  source, run, minutes_since_run, cfg["avail_minutes"])
        return

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    loop = asyncio.get_running_loop()
    for city in cities:
        if city.lat is None or city.lon is None:
            continue
        # HRRR is CONUS-only.
        if model == "hrrr" and not getattr(city, "is_us", True):
            continue
        # Lead-time aimed at end-of-day daily-high; settle for an integer hour ≥ avail.
        lead_hours = max(1, int(minutes_since_run // 60))
        from functools import partial
        result = await loop.run_in_executor(
            None,
            partial(_fetch_one_sync,
                    model=model, lat=city.lat, lon=city.lon,
                    run=run, lead_hours=lead_hours),
        )
        if result is None:
            continue
        try:
            await _record(city=city, source=source, result=result, om_source=om_source)
        except Exception as e:
            log.warning("herbie %s: record failed for city=%s: %s", source, city.slug, e)


async def fetch_herbie_hrrr() -> None:
    await _run_source("herbie_hrrr")


async def fetch_herbie_nbm() -> None:
    await _run_source("herbie_nbm")


async def fetch_herbie_ifs() -> None:
    await _run_source("herbie_ifs")


async def fetch_herbie_aifs() -> None:
    await _run_source("herbie_aifs")
