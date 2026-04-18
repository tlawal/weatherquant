"""
MADIS HFMETAR ingestion — primary current-observation source for US cities.

Fetches high-frequency METAR observations from NOAA MADIS via netCDF over
HTTPS (5-min ASOS cadence, typically 2–10 min ahead of api.weather.gov),
parses them, and writes each record to:

  1. `metar_obs` with source="madis" + a matching `metar_obs_extended` row,
     so every downstream consumer of get_latest_metar / get_todays_extended_obs
     (signal engine, gating, market context, Today's Observations table,
     Current Temp card, API) automatically picks the freshest source.
  2. `madis_obs` (legacy benchmarking table, still populated for the
     NWS-vs-MADIS speed benchmark card).

All downstream reads select the newest row across sources — NWS API / TGFTP
automatically become the fallback whenever MADIS hasn't caught up.
"""
from __future__ import annotations

import gzip
import logging
import ssl
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

# MADIS public server's cert chain fails to verify under aiohttp's default
# SSL context (see aio-libs/aiohttp#7287 — `requests` works, aiohttp does not).
# NOAA's own support email documents `curl -k` / `wget` for this host, so
# skipping verification matches the documented usage. Scoped to MADIS only —
# every other ingestor keeps full TLS verification. MADIS is a public,
# read-only gov data endpoint: no auth, no secrets in flight.
_SSL_CTX: object = False  # value passed to aiohttp.TCPConnector(ssl=...)

# Last successfully processed filename. Kept for informational logging only —
# we intentionally do NOT use it to short-circuit future fetches, because
# MADIS rewrites the current-hour file in place every ~5 min as new ASOS
# observations append. Per-observation dedupe via get_madis_obs_by_key
# handles repeated inserts cheaply.
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


# ---------------------------------------------------------------------------
# Unit conversions for extended fields (Pa → inHg, m/s → kt, m → in).
# ---------------------------------------------------------------------------
def _pa_to_inhg(pa: float) -> float:
    return round(pa * 0.00029529983071445, 2)


def _ms_to_kt(ms: float) -> float:
    return round(ms * 1.94384449, 1)


def _m_to_in(m: float) -> float:
    return round(m * 39.37007874, 3)


# ---------------------------------------------------------------------------
# Defensive variable lookup. MADIS HFMETAR schema names vary slightly across
# MADIS software generations; try each plausible name in order and return the
# first match plus the name that hit. `None` if nothing matches.
# ---------------------------------------------------------------------------
_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "station": ("stationId", "stationID", "staId", "stationName"),
    "obs_time": ("timeObs", "observationTime"),
    "temperature_k": ("temperature",),
    "dewpoint_k": ("dewpoint", "dewPoint", "dewpointTemperature"),
    "rel_humidity": ("relHumidity", "relativeHumidity"),
    "wind_dir": ("windDir", "windDirection"),
    "wind_speed_ms": ("windSpeed",),
    "wind_gust_ms": ("windGust",),
    "altimeter_pa": ("altimeter", "altimeterSetting"),
    "slp_pa": ("seaLevelPressure",),
    "precip_1h_m": (
        "precip1hr",
        "precip1Hour",
        "precipAccum",
        "precipAccum1hr",
        "precip1Hr",
    ),
    "visibility_m": ("visibility",),
}


def _pick_var(ds, logical_name: str) -> Optional[str]:
    """Return the first schema variable name in ds.variables that matches one
    of the candidates for `logical_name`, or None."""
    for cand in _FIELD_CANDIDATES[logical_name]:
        if cand in ds.variables:
            return cand
    return None


def _scalar(arr, idx: int) -> Optional[float]:
    """Safely pull a float at index idx from a netCDF variable array. Returns
    None on mask, NaN, or out-of-sanity-range values. MADIS encodes 'missing'
    with fill values like 3.4e38 or -9999."""
    try:
        v = arr[idx]
    except (IndexError, TypeError):
        return None
    # numpy masked arrays expose `.mask`; masked element → skip
    if hasattr(v, "mask") and bool(getattr(v, "mask", False)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    # Reject obvious MADIS fill sentinels.
    if abs(f) > 1e30 or f <= -9000:
        return None
    return f


async def _fetch_netcdf(filename: str, http: aiohttp.ClientSession) -> Optional[bytes]:
    """Fetch a gzipped netCDF file from MADIS. Returns raw bytes or None.

    Logs include the full URL on every failure so a production operator can
    reproduce with `curl -k URL`. TLS/cert failures are logged at ERROR with
    a distinct tag and are not retried (retrying a handshake failure wastes
    the retry budget).
    """
    url = f"{MADIS_BASE_URL}/{filename}"
    for attempt in range(3):
        try:
            async with http.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                if resp.status == 404:
                    log.debug("madis: 404 url=%s", url)
                    return None
                log.warning(
                    "madis: HTTP %d url=%s (attempt %d)",
                    resp.status, url, attempt + 1,
                )
        except (
            aiohttp.ClientConnectorCertificateError,
            aiohttp.ClientSSLError,
            ssl.SSLError,
        ) as e:
            log.error(
                "madis: TLS verification error url=%s err=%s (%s)",
                url, e, type(e).__name__,
            )
            return None  # retrying a TLS handshake failure is pointless
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning(
                "madis: fetch error url=%s attempt=%d err=%s (%s)",
                url, attempt + 1, e, type(e).__name__,
            )
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
        get_metar_obs_by_key,
        insert_madis_obs,
        insert_metar_obs,
        update_heartbeat,
        upsert_metar_obs_extended,
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
    last_url: Optional[str] = None

    connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=_TIMEOUT,
        headers={"User-Agent": _USER_AGENT},
    ) as http:
        for step in range(_MAX_RETRIES):
            # Compute filename for this step (top of each hour)
            offset_hours = step
            target = now - timedelta(hours=offset_hours)
            candidate = f"{target.strftime('%Y%m%d')}_{target.hour:02d}00.gz"
            last_url = f"{MADIS_BASE_URL}/{candidate}"
            log.debug("madis: trying file %s (step=%d)", candidate, step)

            # NOTE: we intentionally do NOT short-circuit on
            # candidate == _last_success_file. The current-hour file is
            # rewritten in place every ~5 min as new ASOS obs append; we
            # must re-fetch to pick those up. Per-observation dedupe via
            # get_madis_obs_by_key handles repeated inserts cheaply.

            raw = await _fetch_netcdf(candidate, http)
            if raw is not None:
                filename = candidate
                raw_gz = raw
                break

    if raw_gz is None:
        log.warning(
            "madis: no file found in last %d intervals (last url=%s)",
            _MAX_RETRIES, last_url,
        )
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
            # Log full variable list at INFO so a future schema change
            # (e.g. field rename) is obvious from production logs.
            var_names = list(ds.variables.keys())
            log.info("madis: netCDF variables in %s: %s", filename, var_names)

            # Required variables.
            id_var = _pick_var(ds, "station")
            if id_var is None:
                log.error(
                    "madis: no station-id variable found in %s (tried %s; vars: %s)",
                    filename, _FIELD_CANDIDATES["station"], var_names,
                )
                return
            if id_var == "stationName":
                log.warning(
                    "madis: falling back to stationName (long-form location) as station id — matches will likely fail. vars=%s",
                    var_names,
                )
            temp_var = _pick_var(ds, "temperature_k")
            time_var = _pick_var(ds, "obs_time")
            if not temp_var or not time_var:
                log.error(
                    "madis: missing required variable in %s (temp=%s, time=%s; vars=%s)",
                    filename, temp_var, time_var, var_names,
                )
                return

            # Optional / extended variables. Any that don't exist in this file
            # simply resolve to None and are omitted from the extended row.
            dewpoint_var = _pick_var(ds, "dewpoint_k")
            rh_var = _pick_var(ds, "rel_humidity")
            wdir_var = _pick_var(ds, "wind_dir")
            wspd_var = _pick_var(ds, "wind_speed_ms")
            wgust_var = _pick_var(ds, "wind_gust_ms")
            alt_var = _pick_var(ds, "altimeter_pa")
            slp_var = _pick_var(ds, "slp_pa")
            precip_var = _pick_var(ds, "precip_1h_m")
            vis_var = _pick_var(ds, "visibility_m")

            log.info(
                "madis: resolved vars file=%s id=%s time=%s temp=%s dew=%s rh=%s "
                "wdir=%s wspd=%s wgust=%s alt=%s slp=%s precip=%s vis=%s",
                filename, id_var, time_var, temp_var, dewpoint_var, rh_var,
                wdir_var, wspd_var, wgust_var, alt_var, slp_var, precip_var, vis_var,
            )

            station_names = ds.variables[id_var][:]
            temperatures = ds.variables[temp_var][:]
            obs_times = ds.variables[time_var][:]

            def _arr(name):
                return ds.variables[name][:] if name else None

            dewpoints = _arr(dewpoint_var)
            humidities = _arr(rh_var)
            wind_dirs = _arr(wdir_var)
            wind_speeds = _arr(wspd_var)
            wind_gusts = _arr(wgust_var)
            altimeters = _arr(alt_var)
            slps = _arr(slp_var)
            precips = _arr(precip_var)
            visibilities = _arr(vis_var)

            n_stations = len(temperatures)

            def _decode_station(raw):
                """Decode a MADIS stationName entry to a stripped uppercase
                ICAO code. Handles whitespace AND null-byte padding."""
                if hasattr(raw, "tobytes"):
                    b = raw.tobytes()
                else:
                    b = str(raw).encode("ascii", errors="ignore")
                return b.decode("ascii", errors="ignore").strip(" \t\n\r\x00").upper()

            sample_decoded = [_decode_station(station_names[i]) for i in range(min(10, n_stations))]
            log.info(
                "madis: file=%s id_var=%s n_records=%d configured_stations=%s sample_decoded=%s",
                filename, id_var, n_stations, sorted(station_to_city.keys()), sample_decoded,
            )

            async with get_session() as sess:
                for i in range(n_stations):
                    station = _decode_station(station_names[i])
                    if station not in station_to_city:
                        continue

                    city = station_to_city[station]

                    # Temperature: Kelvin → °C → °F (required).
                    temp_k = _scalar(temperatures, i)
                    if temp_k is None or temp_k < 200 or temp_k > 350:
                        continue
                    temp_c = _k_to_c(temp_k)
                    temp_f = _c_to_f(temp_c)

                    # Observation time: epoch seconds → UTC datetime.
                    epoch = _scalar(obs_times, i)
                    if epoch is None or epoch < 1e9:
                        continue
                    obs_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)

                    # ---- Extended fields (all optional) --------------------
                    dewpoint_c = dewpoint_f = None
                    dp_k = _scalar(dewpoints, i) if dewpoints is not None else None
                    if dp_k is not None and 200 <= dp_k <= 350:
                        dewpoint_c = _k_to_c(dp_k)
                        dewpoint_f = _c_to_f(dewpoint_c)

                    humidity_pct = _scalar(humidities, i) if humidities is not None else None
                    if humidity_pct is not None:
                        # Some files store fraction 0–1 rather than 0–100.
                        if 0.0 <= humidity_pct <= 1.0001:
                            humidity_pct = humidity_pct * 100.0
                        if not (0.0 <= humidity_pct <= 100.5):
                            humidity_pct = None
                        else:
                            humidity_pct = round(humidity_pct, 1)

                    wd = _scalar(wind_dirs, i) if wind_dirs is not None else None
                    wind_dir_deg = int(round(wd)) if wd is not None and 0 <= wd <= 360 else None

                    ws_ms = _scalar(wind_speeds, i) if wind_speeds is not None else None
                    wind_speed_kt = _ms_to_kt(ws_ms) if ws_ms is not None and 0 <= ws_ms <= 120 else None

                    wg_ms = _scalar(wind_gusts, i) if wind_gusts is not None else None
                    wind_gust_kt = _ms_to_kt(wg_ms) if wg_ms is not None and 0 <= wg_ms <= 150 else None

                    # Altimeter: Pa → inHg. Fall back to seaLevelPressure.
                    alt_pa = _scalar(altimeters, i) if altimeters is not None else None
                    if alt_pa is None and slps is not None:
                        alt_pa = _scalar(slps, i)
                    altimeter_inhg = _pa_to_inhg(alt_pa) if alt_pa is not None and 80000 <= alt_pa <= 110000 else None

                    pr_m = _scalar(precips, i) if precips is not None else None
                    precip_in = _m_to_in(pr_m) if pr_m is not None and 0 <= pr_m <= 1.0 else None

                    # Dedupe: same (city, station, observed_at) already present
                    # in metar_obs — skip both MetarObs and MadisObs writes.
                    existing_metar = await get_metar_obs_by_key(
                        sess, city.id, station, obs_dt
                    )
                    existing_madis = await get_madis_obs_by_key(sess, station, obs_dt)

                    if existing_metar and existing_madis:
                        skipped += 1
                        continue

                    # ---- Unified write: MetarObs + MetarObsExtended -------
                    if not existing_metar:
                        metar_row = await insert_metar_obs(
                            sess,
                            city_id=city.id,
                            metar_station=station,
                            observed_at=obs_dt,
                            temp_c=temp_c,
                            temp_f=temp_f,
                            source="madis",
                        )
                        # 1:1 extended row when any extended field has data.
                        has_extended = any(
                            v is not None for v in (
                                dewpoint_c, humidity_pct, wind_dir_deg,
                                wind_speed_kt, wind_gust_kt, altimeter_inhg,
                                precip_in,
                            )
                        )
                        if has_extended:
                            await upsert_metar_obs_extended(
                                sess,
                                metar_obs_id=metar_row.id,
                                dewpoint_c=dewpoint_c,
                                dewpoint_f=dewpoint_f,
                                humidity_pct=humidity_pct,
                                wind_dir_deg=wind_dir_deg,
                                wind_speed_kt=wind_speed_kt,
                                wind_gust_kt=wind_gust_kt,
                                altimeter_inhg=altimeter_inhg,
                                precip_in=precip_in,
                            )

                    # ---- Legacy: keep writing MadisObs for backward-compat
                    # with any benchmark queries still reading that table. --
                    if not existing_madis:
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
            log.exception("madis: error parsing netCDF file %s: %s", filename, e)
        finally:
            ds.close()

    _last_success_file = filename
    log.info(
        "madis: fetched %d observations for %d stations (file=%s, skipped=%d)",
        inserted, len(station_to_city), filename, skipped,
    )
