"""
MADIS HFMETAR ingestion — benchmarking only, does NOT feed trading logic.

Fetches high-frequency METAR observations from NOAA MADIS via netCDF over
HTTPS, parses them, and stores in the `madis_obs` table for side-by-side
comparison with TGFTP on the city dashboard.
"""
from __future__ import annotations

import gzip
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
import asyncio

log = logging.getLogger(__name__)

MADIS_BASE_URL = (
    "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF"
)
_USER_AGENT = "WeatherQuant/1.0 (contact@weatherquant.local)"
_TIMEOUT = aiohttp.ClientTimeout(total=30)
_MAX_RETRIES = 4  # step back 1 hour each → 4 hours back

# Cache the last successful filename so repeated failures don't re-probe
_last_success_file: Optional[str] = None


def _compute_filename() -> str:
    """Compute the MADIS HFMETAR filename for the current hour.

    MADIS HFMETAR files are named YYYYMMDD_HHMM.gz where HHMM is
    the top of the hour (e.g. 2000, 2100, ...). They are published
    hourly, not every 5 minutes.
    """
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d')}_{now.hour:02d}00.gz"


def _k_to_c(k: float) -> float:
    """Convert Kelvin to Celsius."""
    return round(k - 273.15, 1)


def _c_to_f(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round(c * 9 / 5 + 32, 1)


async def _fetch_netcdf(filename: str, http: aiohttp.ClientSession) -> Optional[bytes]:
    """Fetch a gzipped netCDF file from MADIS. Returns raw bytes or None."""
    url = f"{MADIS_BASE_URL}/{filename}"
    for attempt in range(3):
        try:
            async with http.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                elif resp.status == 404:
                    log.debug("madis: file %s not found (404)", filename)
                    return None
                else:
                    log.warning("madis: HTTP %d for %s (attempt %d)", resp.status, filename, attempt + 1)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("madis: fetch error for %s (attempt %d): %s", filename, attempt + 1, e)
    return None


async def fetch_madis_latest() -> None:
    """Fetch the latest MADIS HFMETAR netCDF file and insert observations.

    Tries the current hour, stepping back up to 4 hours
    if the file isn't available yet.
    """
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_all_cities,
        get_madis_obs_by_key,
        insert_madis_obs,
        update_heartbeat,
    )

    global _last_success_file

    # Get eligible cities (US with metar_station)
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)
    eligible = [c for c in cities if c.is_us and c.metar_station]
    if not eligible:
        log.debug("madis: no eligible US cities with metar_station")
        return

    station_to_city = {c.metar_station.upper(): c for c in eligible}

    # Try filenames stepping back in 1-hour intervals
    now = datetime.now(timezone.utc)
    filename = None
    raw_gz = None

    async with aiohttp.ClientSession(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as http:
        for step in range(_MAX_RETRIES):
            # Compute filename for this step (top of each hour)
            offset_hours = step
            target = now - timedelta(hours=offset_hours)
            candidate = f"{target.strftime('%Y%m%d')}_{target.hour:02d}00.gz"
            log.debug("madis: trying file %s (step=%d)", candidate, step)

            # Skip if same as last success (already processed)
            if candidate == _last_success_file:
                log.debug("madis: skipping already-processed file %s", candidate)
                return

            raw = await _fetch_netcdf(candidate, http)
            if raw is not None:
                filename = candidate
                raw_gz = raw
                break

    if raw_gz is None:
        log.warning("madis: no file found in last %d intervals", _MAX_RETRIES)
        return

    # Gunzip and parse netCDF
    try:
        import netCDF4
    except ImportError:
        log.error("madis: netCDF4 package not installed — cannot parse MADIS data")
        return

    inserted = 0
    skipped = 0

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        # Decompress gzip → temp file
        tmp.write(gzip.decompress(raw_gz))
        tmp.flush()

        try:
            ds = netCDF4.Dataset(tmp.name, "r")
        except Exception as e:
            log.error("madis: failed to open netCDF file %s: %s", filename, e)
            return

        try:
            # Debug: log available variables
            var_names = list(ds.variables.keys())
            log.debug("madis: netCDF variables in %s: %s", filename, var_names[:20])

            # Extract arrays — MADIS METAR uses stationName, temperature, timeObs
            if "stationName" not in ds.variables:
                log.error("madis: 'stationName' variable not found in %s (vars: %s)", filename, var_names[:10])
                return
            if "temperature" not in ds.variables:
                log.error("madis: 'temperature' variable not found in %s (vars: %s)", filename, var_names[:10])
                return

            station_names = ds.variables["stationName"][:]  # char array
            temperatures = ds.variables["temperature"][:]   # Kelvin

            # MADIS uses 'timeObs' for observation time (not 'observationTime')
            if "timeObs" in ds.variables:
                obs_times = ds.variables["timeObs"][:]
            elif "observationTime" in ds.variables:
                obs_times = ds.variables["observationTime"][:]
            else:
                log.error("madis: no time variable found in %s (tried timeObs, observationTime; vars: %s)", filename, var_names[:10])
                return

            n_stations = len(temperatures)

            async with get_session() as sess:
                for i in range(n_stations):
                    # Station name: char array → string
                    raw_name = station_names[i]
                    if hasattr(raw_name, "tobytes"):
                        station = raw_name.tobytes().decode("ascii", errors="ignore").strip()
                    else:
                        station = str(raw_name).strip()

                    station = station.upper()
                    if station not in station_to_city:
                        continue

                    city = station_to_city[station]

                    # Temperature: Kelvin → °C → °F
                    temp_k = float(temperatures[i])
                    if temp_k < 200 or temp_k > 350:
                        # Skip unreasonable values (MADIS uses fill values)
                        continue
                    temp_c = _k_to_c(temp_k)
                    temp_f = _c_to_f(temp_c)

                    # Observation time: epoch seconds → UTC datetime
                    epoch = float(obs_times[i])
                    if epoch < 1e9:
                        continue
                    obs_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)

                    # Dedupe
                    existing = await get_madis_obs_by_key(sess, station, obs_dt)
                    if existing:
                        skipped += 1
                        continue

                    await insert_madis_obs(
                        sess,
                        city_id=city.id,
                        metar_station=station,
                        observed_at=obs_dt,
                        temp_c=temp_c,
                        temp_f=temp_f,
                        source_file=filename,
                    )
                    inserted += 1

                await update_heartbeat(sess, "fetch_madis")
        except Exception as e:
            log.error("madis: error parsing netCDF file %s: %s", filename, e)
        finally:
            ds.close()

    _last_success_file = filename
    log.info(
        "madis: fetched %d observations for %d stations (file=%s, skipped=%d)",
        inserted, len(station_to_city), filename, skipped,
    )
