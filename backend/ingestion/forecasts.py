"""
Forecast ingestion — NWS API (15 min) + Weather Underground (60 min).

Sources per city:
  1. NWS API daily high (reliable, rate-limited)
  2. WU hourly forecast via weather.com v1 API (max of hourly temps for the day)
  3. WU history via weather.com v1 API (settlement source for past obs)

WU history is the SETTLEMENT SOURCE fallback for Polymarket resolution.
If WU fails entirely, forecast_quality → degraded and auto-trading stops.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


def _compute_model_run_at(source_key: str, fetch_time: datetime) -> Optional[datetime]:
    """Compute the most recent model initialization time for scheduled NWP models.

    Uses known model schedules to infer which model run produced the forecast
    we are fetching. Critical for lead-time skill analysis.

    Schedules:
      - hrrr:   hourly runs (every hour at :00)
      - nbm:    01z, 07z, 13z, 19z  (4x/day)
      - ecmwf_ifs: 00z, 12z  (2x/day, 6-8h dissemination)
      - gfs:    00z, 06z, 12z, 18z  (4x/day)
      - open_meteo: use fetch_time (no explicit model run time exposed)
    """
    if source_key == "open_meteo":
        return None  # Open-Meteo general doesn't expose model init time

    # All schedules are in UTC
    utc = fetch_time.astimezone(timezone.utc)
    hour = utc.hour
    minute = utc.minute

    if source_key in ("hrrr", "hrrr_15min"):
        # HRRR runs every hour; use the start of the current hour
        return utc.replace(minute=0, second=0, microsecond=0)

    elif source_key == "nbm":
        # NBM runs at 01z, 07z, 13z, 19z
        run_hours = [1, 7, 13, 19]
        # Find most recent run hour
        recent_run = max((h for h in run_hours if h <= hour), default=19)
        return utc.replace(hour=recent_run, minute=0, second=0, microsecond=0)

    elif source_key == "ecmwf_ifs":
        # ECMWF IFS runs at 00z, 12z
        run_hours = [0, 12]
        recent_run = max((h for h in run_hours if h <= hour), default=12)
        # If we're before the first run of the day, it's yesterday's 12z run
        if hour < 0:  # Shouldn't happen with max() logic, but explicit
            recent_run = 12
            return (utc - timedelta(days=1)).replace(hour=recent_run, minute=0, second=0, microsecond=0)
        return utc.replace(hour=recent_run, minute=0, second=0, microsecond=0)

    elif source_key == "gfs":
        # GFS runs at 00z, 06z, 12z, 18z
        run_hours = [0, 6, 12, 18]
        recent_run = max((h for h in run_hours if h <= hour), default=18)
        if hour < 0:
            recent_run = 18
            return (utc - timedelta(days=1)).replace(hour=recent_run, minute=0, second=0, microsecond=0)
        return utc.replace(hour=recent_run, minute=0, second=0, microsecond=0)

    return None

import aiohttp

from backend.config import Config
from backend.tz_utils import active_dates_for_city, city_local_date, city_local_now, city_local_tomorrow
from backend.storage.db import get_session
from backend.storage.models import City
from backend.storage.repos import (
    get_all_cities,
    get_latest_forecast,
    get_latest_successful_forecast,
    insert_forecast_obs,
    update_heartbeat,
    upsert_event,
    get_event,
)
from backend.ingestion.model_metadata import (
    fetch_openmeteo_metadata,
    parse_nws_update_time,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

NWS_BASE = "https://api.weather.gov"
WU_BASE = "https://www.wunderground.com"
_USER_AGENT = "WeatherQuant/1.0 (contact@weatherquant.local)"
_TIMEOUT = aiohttp.ClientTimeout(total=20)

# WU scraping headers — mimic browser to reduce blocking
_WU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ─── NWS API ──────────────────────────────────────────────────────────────────

async def fetch_nws_all() -> None:
    """Fetch NWS API daily high for all enabled US cities."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        if not city.is_us:
            continue
        if not (city.nws_office and city.nws_grid_x and city.nws_grid_y):
            continue

        for active_date in active_dates_for_city(city):
            try:
                high_f, nws_data = await _fetch_nws_high(city, active_date)
            except Exception as e:
                log.error("nws: %s date=%s failed: %s", city.city_slug, active_date, e)
                high_f = None
                nws_data = None

            # Use NWS updateTime as authoritative model_run_at; fallback to hour-start proxy
            model_run_at = None
            if high_f is not None:
                model_run_at = parse_nws_update_time(nws_data) if nws_data else None
                if model_run_at is None:
                    now_utc = datetime.now(timezone.utc)
                    model_run_at = now_utc.replace(minute=0, second=0, microsecond=0)

            async with get_session() as sess:
                raw = json.dumps({
                    "source": "nws",
                    "high_f": high_f,
                    "nws_update_time": (nws_data.get("properties") or {}).get("updateTime") if nws_data else None,
                    "nws_generated_at": (nws_data.get("properties") or {}).get("generatedAt") if nws_data else None,
                })
                await insert_forecast_obs(
                    sess,
                    city_id=city.id,
                    source="nws",
                    date_et=active_date,
                    model_run_at=model_run_at,
                    high_f=high_f,
                    raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                    raw_json=raw,
                    parse_error=None if high_f is not None else "parse_failed",
                )
            # Phase A5 — log ingestion latency: how stale is this forecast cycle?
            age_s = None
            if model_run_at is not None and high_f is not None:
                age_s = int((datetime.now(timezone.utc) - model_run_at).total_seconds())
            log.info(
                "ingest: src=nws city=%s date=%s high_f=%s model_run_age_s=%s",
                city.city_slug, active_date, high_f, age_s,
            )

        async with get_session() as sess:
            await update_heartbeat(sess, "fetch_nws", success=True)


async def fetch_open_meteo_all() -> None:
    """Fetch Open-Meteo forecast for all enabled international cities."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        if city.is_us:
            continue

        active_dates = active_dates_for_city(city)
        if not active_dates:
            continue

        start_date = active_dates[0]
        end_date = active_dates[-1]

        try:
            highs_by_date = await _fetch_open_meteo_high(city, start_date, end_date)
        except Exception as e:
            log.error("open-meteo: %s failed: %s", city.city_slug, e)
            highs_by_date = {}

        for active_date in active_dates:
            high_f = highs_by_date.get(active_date)

            async with get_session() as sess:
                raw = json.dumps({"source": "open_meteo", "high_f": high_f})
                await insert_forecast_obs(
                    sess,
                    city_id=city.id,
                    source="open_meteo",
                    date_et=active_date,
                    model_run_at=None,  # Open-Meteo general doesn't expose model init time
                    high_f=high_f,
                    raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                    raw_json=raw,
                    parse_error=None if high_f is not None else "parse_failed",
                )
            log.info("open-meteo: %s date=%s high_f=%s", city.city_slug, active_date, high_f)

        async with get_session() as sess:
            await update_heartbeat(sess, "fetch_open_meteo", success=True)


async def _fetch_open_meteo_high(city: City, start_date: str, end_date: str) -> dict[str, Optional[float]]:
    """Fetch Open-Meteo hourly forecast and return a {date_et: max_temp} map."""
    if city.lat is None or city.lon is None:
        log.warning("open-meteo: missing coords for %s", city.city_slug)
        return {}

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "hourly": "temperature_2m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }

    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
            async with http.get(url, params=params) as resp:
                if resp.status != 200:
                    log.error("open-meteo: HTTP %d for %s", resp.status, url)
                    return {}
                data = await resp.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        if not temps or not times:
            return {}

        # Group by local date (time strings are YYYY-MM-DDTHH:00 with timezone=auto)
        date_temps: dict[str, list[float]] = {}
        for ts, t in zip(times, temps):
            if t is None:
                continue
            day = ts[:10]
            date_temps.setdefault(day, []).append(t)

        highs: dict[str, Optional[float]] = {}
        for day, day_temps in date_temps.items():
            high_c = max(day_temps)
            if city.unit == "F":
                highs[day] = round(high_c * 9 / 5 + 32, 1)
            else:
                highs[day] = round(high_c, 1)

        return highs

    except Exception as e:
        log.exception("open-meteo: failed for %s %s-%s", city.city_slug, start_date, end_date)
        return {}


# ─── Open-Meteo Multi-Model (HRRR + GFS) ────────────────────────────────────

_OM_MODELS = {
    "hrrr": "gfs_hrrr",          # GFS+HRRR blend, NA high-resolution
    "hrrr_15min": "ncep_hrrr_conus_15min",  # HRRR CONUS 15-minute output
    "nbm": "ncep_nbm_conus",    # NCEP National Blend of Models, US CONUS
    "ecmwf_ifs": "ecmwf_ifs",   # ECMWF Integrated Forecast System, global 9–25 km
    # Phase C4 — ECMWF AIFS (AI Integrated Forecasting System), 0.25° single-level.
    # Released real-time via the open ECMWF catalog, no embargo. Marked as
    # experimental in the UI until the 30-day MAE comparison vs ecmwf_ifs lands.
    "ecmwf_aifs": "ecmwf_aifs025_single",
}

# Sources marked experimental in the UI (asterisk badge on city pages).
EXPERIMENTAL_FORECAST_SOURCES: frozenset[str] = frozenset({"ecmwf_aifs"})


async def fetch_open_meteo_models_all(source_filter: Optional[set[str]] = None) -> None:
    """Fetch Open-Meteo NWP forecasts for all enabled cities.

    source_filter: optional iterable of `_OM_MODELS` keys (e.g. {"hrrr", "hrrr_15min"})
    to restrict the fetch. None = fetch all configured models.
    """
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        if city.lat is None or city.lon is None:
            continue

        active_dates = active_dates_for_city(city)
        if not active_dates:
            continue

        start_date = active_dates[0]
        end_date = active_dates[-1]

        for source_key, om_model in _OM_MODELS.items():
            if source_filter is not None and source_key not in source_filter:
                continue
            # Skip HRRR for non-US cities (HRRR is North America only)
            if source_key == "hrrr" and not city.is_us:
                continue

            try:
                data_by_date = await _fetch_open_meteo_model_high(city, start_date, end_date, om_model)
            except Exception as e:
                log.error("om-%s: %s failed: %s", source_key, city.city_slug, e)
                data_by_date = {}

            # Authoritative model initialization time from Open-Meteo metadata
            meta = await fetch_openmeteo_metadata(source_key)
            model_run_at = None
            if data_by_date:
                any_high = next((h for h, _ in data_by_date.values() if h is not None), None)
                if any_high is not None:
                    if meta and meta.get("last_run_initialisation_time"):
                        model_run_at = datetime.fromtimestamp(
                            meta["last_run_initialisation_time"], tz=timezone.utc
                        )
                    else:
                        model_run_at = _compute_model_run_at(source_key, datetime.now(timezone.utc))

            for active_date in active_dates:
                high_f, hourly_data = data_by_date.get(active_date, (None, None))

                async with get_session() as sess:
                    raw = json.dumps({
                        "source": source_key, "model": om_model,
                        "high_f": high_f, "hourly": hourly_data,
                        "model_run_at": model_run_at.isoformat() if model_run_at else None,
                        "openmeteo_metadata": meta,
                    })
                    await insert_forecast_obs(
                        sess,
                        city_id=city.id,
                        source=source_key,
                        date_et=active_date,
                        model_run_at=model_run_at,
                        high_f=high_f,
                        raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                        raw_json=raw,
                        parse_error=None if high_f is not None else "parse_failed",
                    )
                if high_f is not None:
                    # Phase A5 — log ingestion latency per Open-Meteo model run
                    age_s = None
                    if model_run_at is not None:
                        age_s = int((datetime.now(timezone.utc) - model_run_at).total_seconds())
                    log.info(
                        "ingest: src=%s city=%s date=%s high_f=%s model_run_age_s=%s",
                        source_key, city.city_slug, active_date, high_f, age_s,
                    )

    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_om_models", success=True)


async def _fetch_open_meteo_model_high(
    city: City, start_date: str, end_date: str, om_model: str
) -> dict[str, tuple[Optional[float], Optional[dict]]]:
    """Fetch a specific Open-Meteo model and return {date_et: (high_f, hourly_data)}.

    hourly_data contains time/temp arrays for each date (used for chart overlays).
    """
    unit = getattr(city, "unit", "F") or "F"
    temp_unit = "fahrenheit" if unit == "F" else "celsius"

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "hourly": "temperature_2m",
        "temperature_unit": temp_unit,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
        "models": om_model,
    }

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
        async with http.get(url, params=params) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.error("om-%s: HTTP %d for %s — %s", om_model, resp.status, city.city_slug, body[:200])
                return {}
            data = await resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    # Open-Meteo returns model-suffixed keys when `models` param is used
    temps = hourly.get(f"temperature_2m_{om_model}", hourly.get("temperature_2m", []))
    if not temps or not times:
        log.warning("om-%s: no hourly temps for %s (keys: %s)", om_model, city.city_slug, list(hourly.keys()))
        return {}

    # Group by local date and build hourly pairs
    date_pairs: dict[str, list[tuple[str, float]]] = {}
    for ts, t in zip(times, temps):
        if t is None:
            continue
        day = ts[:10]
        date_pairs.setdefault(day, []).append((ts, t))

    out: dict[str, tuple[Optional[float], Optional[dict]]] = {}
    for day, pairs in date_pairs.items():
        high_f = round(max(t for _, t in pairs), 1)
        hourly_data = {"times": [ts for ts, _ in pairs], "temps": [round(t, 1) for _, t in pairs]}
        out[day] = (high_f, hourly_data)

    return out


async def _fetch_nws_high(city: City, active_date_str: str) -> tuple[Optional[float], Optional[dict]]:
    """Fetch NWS gridpoint forecast and return (today's daytime high °F, raw response dict)."""
    if not city.is_us or not city.nws_office:
        return None, None
    url = (
        f"{NWS_BASE}/gridpoints/{city.nws_office}"
        f"/{city.nws_grid_x},{city.nws_grid_y}/forecast"
    )
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(
                timeout=_TIMEOUT, headers=headers
            ) as http:
                async with http.get(url) as resp:
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        log.warning("nws: rate limited, retrying in %ds", wait)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        log.error("nws: HTTP %d for %s", resp.status, url)
                        return None, None
                    data = await resp.json(content_type=None)

            periods = (data.get("properties") or {}).get("periods") or []
            # Find active_date's daytime period
            for period in periods:
                if not period.get("isDaytime", True):
                    continue
                start = period.get("startTime", "")
                if active_date_str in start:
                    temp = period.get("temperature")
                    unit = period.get("temperatureUnit", "F")
                    if temp is None:
                        continue
                    temp_f = float(temp)
                    if unit == "C":
                        temp_f = temp_f * 9 / 5 + 32
                    return round(temp_f, 1), data

            # Fallback — first daytime period
            for period in periods:
                if period.get("isDaytime", True):
                    temp = period.get("temperature")
                    if temp is not None:
                        return round(float(temp), 1), data

        except asyncio.TimeoutError:
            log.warning("nws: timeout for %s (attempt %d/3)", city.city_slug, attempt + 1)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            log.error("nws: error for %s: %s", city.city_slug, e)
            if attempt < 2:
                await asyncio.sleep(1)

    return None, None


# ─── WU Scraping ─────────────────────────────────────────────────────────────

def wu_history_url(city: City, date_et: str) -> str:
    dt_str = date_et.replace("-", "")
    units = "m" if getattr(city, "unit", "F") == "C" else "e"
    
    country = "US"
    if not getattr(city, "is_us", True):
        country = (getattr(city, "wu_state", "") or "GB").upper()
        
    return f"https://api.weather.com/v1/location/{city.metar_station}:9:{country}/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units={units}&startDate={dt_str}"


async def fetch_wu_all() -> None:
    """Fetch WU hourly forecast + history for all enabled cities where WU is the settlement source."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        if not (city.wu_state and city.wu_city and city.metar_station):
            log.debug("wu: city %s missing WU config, skipping", city.city_slug)
            continue

        for date_et in active_dates_for_city(city):
            # Skip WU scraping for cities where WU is NOT the settlement source
            async with get_session() as sess:
                event = await get_event(sess, city.id, date_et)
            if event and not event.settlement_source_verified:
                log.debug("wu: %s date=%s settlement_source_verified=False, fetching forecast anyway", city.city_slug, date_et)

            # Rate limit: check wu_hourly + wu_history cooldown.
            async with get_session() as sess:
                last_hourly = await get_latest_successful_forecast(sess, city.id, "wu_hourly", date_et)
                last_history = await get_latest_successful_forecast(sess, city.id, "wu_history", date_et)

            check_sources = [last_hourly, last_history]
            all_succeeded = all(f and f.fetched_at for f in check_sources)

            if all_succeeded:
                oldest_age = max(
                    (datetime.now(timezone.utc) - (f.fetched_at.replace(tzinfo=timezone.utc) if f.fetched_at.tzinfo is None else f.fetched_at)).total_seconds()
                    for f in check_sources
                )
                if oldest_age < Config.WU_MIN_SCRAPE_INTERVAL_SECONDS:
                    log.debug("wu: %s date=%s all sources fresh, rate limited (oldest=%.0fs < %ds)",
                              city.city_slug, date_et, oldest_age, Config.WU_MIN_SCRAPE_INTERVAL_SECONDS)
                    continue
            else:
                from backend.storage.repos import get_latest_forecast as _get_latest_any
                source_names = ["wu_hourly", "wu_history"]
                async with get_session() as sess:
                    recent_attempts = []
                    for src in source_names:
                        rec = await _get_latest_any(sess, city.id, src, date_et)
                        if rec and rec.fetched_at:
                            recent_attempts.append(rec.fetched_at)

                if recent_attempts:
                    newest_attempt_age = min(
                        (datetime.now(timezone.utc) - (ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts)).total_seconds()
                        for ts in recent_attempts
                    )
                    if newest_attempt_age < Config.WU_FAILED_RETRY_INTERVAL_SECONDS:
                        log.debug("wu: %s date=%s has failed source(s) but attempted %.0fs ago, waiting",
                                  city.city_slug, date_et, newest_attempt_age)
                        continue

                log.info("wu: %s date=%s has failed source(s) [hourly=%s history=%s], retrying now",
                         city.city_slug, date_et,
                         "ok" if last_hourly else "FAIL",
                         "ok" if last_history else "FAIL")

            try:
                await _scrape_wu_city(city, date_et)
            except Exception as e:
                log.exception("wu: Unhandled exception scraping %s date=%s: %s", city.city_slug, date_et, e)

        # Stagger per city to be polite to WU
        await asyncio.sleep(5)


async def _scrape_wu_city(city: City, date_et: str) -> None:
    # Fetch station profile for resolution-aware filtering
    from backend.storage.repos import get_station_profile, get_latest_forecast
    valid_minutes = None
    if city.metar_station:
        async with get_session() as sess:
            profile = await get_station_profile(sess, city.metar_station)
        if profile and profile.observation_minutes:
            valid_minutes = json.loads(profile.observation_minutes)

    # Isolate each scraper so one failure doesn't block the others
    hourly_peak, peak_hour = None, None
    try:
        hourly_peak, peak_hour = await _fetch_wu_hourly_api(city, date_et)
    except Exception as e:
        log.exception("wu_hourly_api: %s date=%s scrape exception: %s", city.city_slug, date_et, e)

    log.info("wu: %s date=%s fetched — hourly=%.1f", city.city_slug, date_et, hourly_peak or 0)

    # Smart scheduling for WU History (~35 mins after station observation)
    now_utc = datetime.now(timezone.utc)
    fetch_history = True
    history_high, obs_time = None, None

    async with get_session() as sess:
        last_hist = await get_latest_successful_forecast(sess, city.id, "wu_history", date_et)

    if last_hist and last_hist.fetched_at:
        _fetched = last_hist.fetched_at
        if _fetched.tzinfo is None:
            _fetched = _fetched.replace(tzinfo=timezone.utc)
        age_hist = (now_utc - _fetched).total_seconds()

        if age_hist < 3600:
            fetch_history = False  # Default to waiting an hour if we already fetched recently

            # Override: If we haven't fetched since the optimal data-drop minute, allow it
            if valid_minutes and len(valid_minutes) > 0:
                target_min = (valid_minutes[-1] + 35) % 60
                last_m = last_hist.fetched_at.minute
                now_m = now_utc.minute

                # Check if we crossed the target minute since the last fetch
                # Requires age > 900 (15m) to avoid rapid spam if target minute just hit
                if age_hist > 900:
                    if last_hist.fetched_at.hour != now_utc.hour:
                        if now_m >= target_min:
                            fetch_history = True
                    elif last_m < target_min <= now_m:
                        fetch_history = True

    if fetch_history:
        try:
            history_high, obs_time = await _fetch_wu_history_api(city, date_et, valid_minutes=valid_minutes)
        except Exception as e:
            log.exception("wu_history: %s scrape exception: %s", city.city_slug, e)
    elif last_hist:
        history_high = last_hist.high_f
        try:
            raw_data = json.loads(last_hist.raw_json)
            obs_time = raw_data.get("obs_time")
        except Exception:
            obs_time = None

    wu_ok = hourly_peak is not None or history_high is not None
    parse_err = None if wu_ok else "all_wu_sources_failed"

    # DB inserts — isolate per source so one failure doesn't block others
    async with get_session() as sess:
        try:
            raw = json.dumps({"high_f": hourly_peak, "peak_hour": peak_hour, "source": "wu_hourly"})
            await insert_forecast_obs(
                sess,
                city_id=city.id,
                source="wu_hourly",
                date_et=date_et,
                model_run_at=None,  # WU doesn't expose model initialization time
                high_f=hourly_peak,
                raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                raw_json=raw,
                parse_error=None if hourly_peak is not None else "parse_failed",
            )
        except Exception as e:
            log.exception("wu: %s failed to insert wu_hourly: %s", city.city_slug, e)

        if fetch_history:
            try:
                raw = json.dumps({"high_f": history_high, "obs_time": obs_time, "source": "wu_history"})
                await insert_forecast_obs(
                    sess,
                    city_id=city.id,
                    source="wu_history",
                    date_et=date_et,
                    model_run_at=None,  # WU history is actual observations, not model forecasts
                    high_f=history_high,
                    raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                    raw_json=raw,
                    parse_error=None if history_high is not None else "parse_failed",
                )
            except Exception as e:
                log.exception("wu: %s failed to insert wu_history: %s", city.city_slug, e)

        # Update forecast_quality on the event
        try:
            event = await get_event(sess, city.id, date_et)
            if event:
                quality = "ok" if wu_ok else "degraded"
                event.forecast_quality = quality
                event.wu_scrape_error = parse_err
                await sess.commit()
        except Exception as e:
            log.exception("wu: %s failed to update event quality: %s", city.city_slug, e)

        try:
            await update_heartbeat(sess, "fetch_wu", success=wu_ok, error=parse_err)
        except Exception as e:
            log.exception("wu: %s failed to update heartbeat: %s", city.city_slug, e)

    log.info(
        "wu: %s hourly=%.1f history=%.1f quality=%s",
        city.city_slug,
        hourly_peak or 0,
        history_high or 0,
        "ok" if wu_ok else "degraded",
    )


async def _fetch_wu_hourly_api(city: City, date_et: str | None = None) -> tuple[Optional[float], Optional[str]]:
    """Fetch WU hourly forecast via weather.com v1 API, returning (max_temp_f, peak_hour_et_str).

    Uses the same v1 API pattern as wu_history to avoid HTML-scraping accuracy issues
    (the HTML page mixes actual temp and feels-like temp columns, causing off-by-1 errors).
    Filters to *date_et* hours only so other days don't inflate the result.
    """
    country = "US"
    if not getattr(city, "is_us", True):
        country = (getattr(city, "wu_state", "") or "GB").upper()
    units = "m" if getattr(city, "unit", "F") == "C" else "e"

    url = (
        f"https://api.weather.com/v1/location/{city.metar_station}:9:{country}"
        f"/forecast/hourly/48hour.json"
        f"?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units={units}"
    )

    target_date = date_et or city_local_date(city)

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_WU_HEADERS) as http:
                async with http.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        forecasts = data.get("forecasts", [])

                        # Filter to the requested date's hours only
                        date_forecasts = [
                            f for f in forecasts
                            if f.get("fcst_valid_local", "")[:10] == target_date
                            and f.get("temp") is not None
                        ]

                        if date_forecasts:
                            best = max(date_forecasts, key=lambda f: f["temp"])
                            high_f = float(best["temp"])
                            peak_hour_str = None
                            local_dt_str = best.get("fcst_valid_local", "")
                            if local_dt_str:
                                try:
                                    dt_et = datetime.fromisoformat(local_dt_str).astimezone(ET)
                                    peak_hour_str = dt_et.strftime("%-I:%M %p ET")
                                except Exception:
                                    pass
                            return high_f, peak_hour_str
                        log.warning("wu_hourly_api: %s date=%s — no forecasts in response", city.city_slug, target_date)
                        return None, None
                    elif resp.status == 404:
                        log.warning("wu_hourly_api: %s date=%s — 404", city.city_slug, target_date)
                        return None, None
                    else:
                        body = await resp.text()
                        log.warning("wu_hourly_api: HTTP %d for %s date=%s — body: %.200s",
                                    resp.status, city.city_slug, target_date, body)
        except Exception as e:
            log.warning("wu_hourly_api: failed for %s date=%s (attempt %d/3): %s", city.city_slug, target_date, attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(2)

    return None, None


def _at_valid_minute(obs: dict, valid_minutes: list[int], tolerance: int = 1) -> bool:
    """Check if an observation's timestamp falls on a valid station minute."""
    gmt = obs.get("valid_time_gmt")
    if not gmt:
        return True  # Can't filter, include it
    dt = datetime.fromtimestamp(int(gmt), tz=timezone.utc)
    minute = dt.minute
    return any(
        min(abs(minute - m) % 60, abs(m - minute) % 60) <= tolerance
        for m in valid_minutes
    )


async def _fetch_wu_history_api(
    city: City, date_et: str, valid_minutes: list[int] | None = None,
) -> tuple[Optional[float], Optional[str]]:
    """Fetch WU actual historical observations, returning (max_temp_f, obs_time_et_str).

    obs_time_et_str is the ET local time when the max temperature was observed,
    e.g. "1:52 PM ET", derived from valid_time_gmt on the peak observation.

    If valid_minutes is provided, only observations at those station minutes
    (±1 tolerance) are considered for the daily high.
    """
    url = wu_history_url(city, date_et)
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_WU_HEADERS) as http:
                async with http.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        obs = data.get("observations", [])
                        valid_obs = [o for o in obs if o.get("temp") is not None]
                        if valid_obs:
                            best = max(valid_obs, key=lambda o: o["temp"])
                            high_f = float(best["temp"])
                            obs_time_str = None
                            gmt = best.get("valid_time_gmt")
                            if gmt:
                                try:
                                    dt_et = datetime.fromtimestamp(int(gmt), tz=ET)
                                    obs_time_str = dt_et.strftime("%-I:%M %p ET")
                                except Exception:
                                    pass
                            return high_f, obs_time_str
                        log.info("wu_history: %s — 200 OK but 0 valid observations (total=%d)",
                                 city.city_slug, len(obs))
                        return None, None
                    elif resp.status == 404:
                        log.info("wu_history: %s — 404 (no data for %s)", city.city_slug, date_et)
                        return None, None
                    else:
                        body = await resp.text()
                        if resp.status == 400 and "NDF-0001" in body:
                            log.info("wu_history: %s — 400 NDF-0001 (no data yet for %s)", city.city_slug, date_et)
                            return None, None
                        log.warning("wu_history: HTTP %d for %s — body: %.200s",
                                    resp.status, city.city_slug, body)
        except Exception as e:
            log.warning("wu_history: fetch failed for %s: %s", city.city_slug, e)
            await asyncio.sleep(2)
    return None, None
