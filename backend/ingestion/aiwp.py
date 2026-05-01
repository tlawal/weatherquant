"""
NOAA AIWP (AI Weather Prediction) archive ingestion — §13 of the Bayesian-upgrade plan.

Fetches near-real-time AI-NWP forecasts from the NOAA-hosted AWS Open Data registry:

  s3://noaa-oar-mlwp-data/  (anonymous public read)
  https://noaa-oar-mlwp-data.s3.amazonaws.com/
  https://registry.opendata.aws/aiwp/

Hosted models (filename prefix → meaning):
  PANG_v100_{IFS,GFS}  Pangu-Weather (Bi et al. 2023, Nature 619:533)
  FOUR_v200_{IFS,GFS}  FourCastNet v2-small (Pathak et al. 2022 / 2024 update)
  FOUR_v100_{IFS,GFS}  FourCastNet v1 (older)
  GRAP_v100_{IFS,GFS}  GraphCast Operational (Lam et al. 2023)

Cadence: 00z + 12z daily, near-real-time. Latency ~5h (GFS-init) or ~8h (IFS-init).
Forecast horizon: f000 → f240 in 6-hour steps (10 days, 41 timesteps).
File format: NetCDF4 (HDF5), ~3 GB each.

We download once per (model, run) to /tmp, extract t2 (2-m temperature) at each
city's lat/lon for every forecast timestep, compute the daily-high in city-local
time per active date, and insert ForecastObs rows. Idempotent — skips re-download
if the run is already in the DB.

Reference: Radford, Ebert-Uphoff, Stewart, et al. 2025, BAMS 106:E68–E76.
"""
from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np

from backend.tz_utils import active_dates_for_city
from backend.storage.db import get_session
from backend.storage.repos import (
    get_all_cities,
    get_latest_forecast,
    insert_forecast_obs,
)
from sqlalchemy import select
from backend.storage.models import ForecastObs

log = logging.getLogger(__name__)

AIWP_S3_BASE = "https://noaa-oar-mlwp-data.s3.amazonaws.com"

# Model registry — flip on a new model by adding a row.
# Tuple shape: (filename_model_code, version, init_condition_source)
AIWP_MODELS: dict[str, tuple[str, str, str]] = {
    # (Pangu-Weather, IFS-initialized — our preferred IC for consistency with
    # the existing IFS/AIFS members. README confirms IFS-init is the newer,
    # higher-fidelity option since 01/15/25.)
    "pangu_weather":  ("PANG", "v100", "IFS"),
    # (FourCastNet v2-small, IFS-initialized.)
    "fourcastnet_v2": ("FOUR", "v200", "IFS"),
    # ── Future flips (adding a row here is the only code change required):
    # "fourcastnet_v1":  ("FOUR", "v100", "IFS"),  # superseded by v2-small
    # "graphcast_aiwp":  ("GRAP", "v100", "IFS"),  # we already have GraphCast
    #                                              #   via Open-Meteo
}

# Confirmed `t2` in both PANG_v100 and FOUR_v200 IFS files (2026-04 probe).
# Listed in fallback order in case future model updates rename the variable.
T2_VARIABLE_NAMES: tuple[str, ...] = ("t2", "t2m", "2t", "T2M", "2T")

# How long to wait for a 3 GB streaming download (worst-case Railway egress
# + Internet). 10 minutes is generous; typical is 2–4 minutes.
_AIWP_TIMEOUT = aiohttp.ClientTimeout(total=600)

# Look back this many days when searching for the latest available run.
# Today's 12z run uploads ~20:00 UTC (IFS) or ~17:00 UTC (GFS). Looking back
# 2 days handles weekend / holiday delays without spinning over older data.
_AIWP_LOOKBACK_DAYS = 3


async def _list_dir_keys(http: aiohttp.ClientSession, prefix: str) -> list[str]:
    """List S3 keys with the given prefix using anonymous public access.

    Returns the full S3 keys (e.g.
    "PANG_v100_IFS/2026/0429/PANG_v100_IFS_2026042912_f000_f240_06.nc").
    """
    url = f"{AIWP_S3_BASE}/?prefix={prefix}&list-type=2"
    async with http.get(url) as resp:
        resp.raise_for_status()
        body = await resp.text()
    return re.findall(r"<Key>([^<]+)</Key>", body)


def _parse_init_utc(filename: str) -> Optional[datetime]:
    """Parse the init datetime from a NOAA AIWP filename.

    Pattern: ``MMMM_vNNN_III_YYYYMMDDHH_fXXX_fYYY_ZZ.nc``
    Returns a tz-aware UTC datetime on the run's init hour, or None if the
    filename doesn't match.
    """
    m = re.search(r"_(\d{10})_f\d{3}_f\d{3}_\d{2}\.nc$", filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)


async def _find_latest_run_url(
    http: aiohttp.ClientSession,
    source_key: str,
) -> tuple[Optional[str], Optional[datetime]]:
    """Locate the most recently uploaded AIWP run for the given model.

    Walks back from today over `_AIWP_LOOKBACK_DAYS` days. Returns
    (full_https_url, init_utc) or (None, None) if no recent file exists.
    """
    model_code, version, ic = AIWP_MODELS[source_key]
    now_utc = datetime.now(timezone.utc)
    for back in range(_AIWP_LOOKBACK_DAYS):
        d = now_utc - timedelta(days=back)
        prefix = f"{model_code}_{version}_{ic}/{d:%Y/%m%d}/"
        try:
            keys = await _list_dir_keys(http, prefix)
        except aiohttp.ClientError as e:
            log.warning("aiwp %s: S3 list failed for %s: %s",
                        source_key, prefix, e)
            continue
        if not keys:
            continue
        # Filenames embed the init datetime, so descending sort gives newest.
        keys.sort(reverse=True)
        for key in keys:
            init_utc = _parse_init_utc(key)
            if init_utc is not None:
                return f"{AIWP_S3_BASE}/{key}", init_utc
    return None, None


async def _is_run_already_ingested(
    source_key: str,
    init_utc: datetime,
) -> bool:
    """Idempotency check: have we already inserted ForecastObs rows for
    this exact (source, model_run_at) pair? If so, skip the 3 GB download.
    """
    async with get_session() as sess:
        result = await sess.execute(
            select(ForecastObs.id)
            .where(
                ForecastObs.source == source_key,
                ForecastObs.model_run_at == init_utc,
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


async def _stream_download(
    http: aiohttp.ClientSession,
    url: str,
    dst: Path,
) -> int:
    """Stream-download to disk. Returns bytes written."""
    bytes_total = 0
    async with http.get(url) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as f:
            async for chunk in resp.content.iter_chunked(1 << 20):  # 1 MiB
                f.write(chunk)
                bytes_total += len(chunk)
    return bytes_total


def _humansize(n: int) -> str:
    nf = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if nf < 1024.0:
            return f"{nf:.1f}{unit}"
        nf /= 1024.0
    return f"{nf:.1f}TB"


def _normalize_lon(target_lon: float, ds_lon_max: float) -> float:
    """AIWP files use 0..360 lon convention. City.lon is signed (-180..180).
    Convert as needed.
    """
    if ds_lon_max > 180 and target_lon < 0:
        return target_lon + 360.0
    if ds_lon_max <= 180 and target_lon > 180:
        return target_lon - 360.0
    return target_lon


def _daily_high_kelvins(
    times,             # numpy datetime64 array, UTC
    kelvins,           # parallel numpy float array, K
    target_date_et: str,
    city_tz,           # zoneinfo.ZoneInfo
) -> Optional[float]:
    """Max temperature over forecast steps that fall within `target_date_et`
    in the city's local timezone. Returns None if no forecast step lands
    on that date.

    Note: 6-hour timestep granularity may underestimate the true peak by
    ~0.5–1°F (typical peak occurs between forecast steps). Acceptable as
    an ensemble member; lead-skill weighting will adjust automatically once
    we have residuals.
    """
    target_date = datetime.strptime(target_date_et, "%Y-%m-%d").date()
    vals: list[float] = []
    for t, k in zip(times, kelvins):
        # numpy datetime64 → seconds → tz-aware UTC datetime
        ts = (t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        dt_utc = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        if dt_utc.astimezone(city_tz).date() == target_date:
            vals.append(float(k))
    if not vals:
        return None
    return max(vals)


async def fetch_aiwp_model(source_key: str) -> int:
    """Download the latest NOAA AIWP run for one model and write
    ForecastObs rows for every (city, active_date) combination.

    Steps:
        1. List S3 prefix for last 3 days, find newest filename.
        2. Skip if (source_key, model_run_at) already exists in DB.
        3. Stream-download to a TemporaryDirectory.
        4. Open with xarray engine='h5netcdf' (memory-mapped).
        5. For each enabled city × active_dates_for_city(city):
             extract t2 at the nearest grid point, compute daily-high in
             local tz, insert ForecastObs.
        6. Cleanup tempdir.

    Returns: number of ForecastObs rows inserted.
    On error, logs and returns 0 — never crashes the worker.
    """
    if source_key not in AIWP_MODELS:
        raise ValueError(f"unknown AIWP source_key: {source_key}")

    fetched_at = datetime.now(timezone.utc)

    try:
        async with aiohttp.ClientSession(timeout=_AIWP_TIMEOUT) as http:
            url, init_utc = await _find_latest_run_url(http, source_key)
            if url is None or init_utc is None:
                log.warning(
                    "aiwp %s: no run available in last %d days",
                    source_key, _AIWP_LOOKBACK_DAYS,
                )
                return 0

            log.info("aiwp %s: latest run init=%s url=%s",
                     source_key, init_utc.isoformat(), url)

            if await _is_run_already_ingested(source_key, init_utc):
                log.info(
                    "aiwp %s: run init=%s already ingested — skipping download",
                    source_key, init_utc.isoformat(),
                )
                return 0

            with tempfile.TemporaryDirectory(prefix="aiwp_") as tmpdir:
                dst = Path(tmpdir) / "forecast.nc"
                t0 = datetime.now(timezone.utc)
                size = await _stream_download(http, url, dst)
                dl_secs = (datetime.now(timezone.utc) - t0).total_seconds()
                log.info(
                    "aiwp %s: downloaded %s in %.1fs",
                    source_key, _humansize(size), dl_secs,
                )

                rows = await _extract_and_insert(
                    source_key=source_key,
                    nc_path=dst,
                    init_utc=init_utc,
                    fetched_at=fetched_at,
                )

        log.info(
            "aiwp %s: inserted %d ForecastObs rows for run init=%s",
            source_key, rows, init_utc.isoformat(),
        )
        return rows
    except Exception as e:
        # Graceful-degrade: log + return 0; signal engine treats a missing
        # source as one fewer ensemble member, same as any other source
        # being temporarily unavailable.
        log.warning("aiwp %s: fetch failed (%s) — skipping this cycle",
                    source_key, e, exc_info=True)
        return 0


async def _extract_and_insert(
    source_key: str,
    nc_path: Path,
    init_utc: datetime,
    fetched_at: datetime,
) -> int:
    """Open the downloaded NetCDF, extract per-city per-date daily-highs,
    insert ForecastObs rows. Returns rows inserted.
    """
    # xarray import is lazy: keeps the module loadable if the dependency
    # is somehow missing in production (the cron job would log + skip).
    import xarray as xr
    from zoneinfo import ZoneInfo

    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    try:
        # Find 2m-temperature variable
        t2 = None
        for name in T2_VARIABLE_NAMES:
            if name in ds.data_vars:
                t2 = ds[name]
                break
        if t2 is None:
            log.error(
                "aiwp %s: no 2m-temp variable in %s; data_vars=%s",
                source_key, nc_path.name, list(ds.data_vars),
            )
            return 0

        # Coordinate name detection (NOAA AIWP uses 'latitude'/'longitude'
        # but some derived products may use 'lat'/'lon').
        lat_coord = "latitude" if "latitude" in ds.coords else "lat"
        lon_coord = "longitude" if "longitude" in ds.coords else "lon"
        lon_max = float(ds[lon_coord].max().values)

        if "time" not in ds.coords:
            log.error("aiwp %s: dataset has no 'time' coordinate", source_key)
            return 0
        times = ds["time"].values  # numpy datetime64 array

        rows_inserted = 0
        async with get_session() as sess:
            cities = await get_all_cities(sess, enabled_only=True)
            for city in cities:
                if city.lat is None or city.lon is None:
                    continue
                active_dates = active_dates_for_city(city)
                if not active_dates:
                    continue

                target_lon = _normalize_lon(float(city.lon), lon_max)
                sel = t2.sel(
                    **{lat_coord: float(city.lat), lon_coord: target_lon},
                    method="nearest",
                )
                kelvins = sel.values  # 1-D over time

                tz = ZoneInfo(getattr(city, "tz", None) or "America/New_York")
                unit = getattr(city, "unit", "F") or "F"

                for date_et in active_dates:
                    high_k = _daily_high_kelvins(times, kelvins, date_et, tz)
                    if high_k is None:
                        continue
                    # AIWP `t2` is in Kelvin (verified). Convert to city unit.
                    if unit == "C":
                        high_val = float(high_k) - 273.15
                    else:
                        high_val = (float(high_k) - 273.15) * 9.0 / 5.0 + 32.0
                    await insert_forecast_obs(
                        sess,
                        city_id=city.id,
                        source=source_key,
                        date_et=date_et,
                        high_f=round(high_val, 2),
                        fetched_at=fetched_at,
                        model_run_at=init_utc,
                    )
                    rows_inserted += 1
        return rows_inserted
    finally:
        ds.close()


# ─── Convenience wrappers for the scheduler ─────────────────────────────────

async def fetch_pangu() -> int:
    return await fetch_aiwp_model("pangu_weather")


async def fetch_fourcastnet_v2() -> int:
    return await fetch_aiwp_model("fourcastnet_v2")


__all__ = [
    "AIWP_MODELS",
    "fetch_aiwp_model",
    "fetch_pangu",
    "fetch_fourcastnet_v2",
]
