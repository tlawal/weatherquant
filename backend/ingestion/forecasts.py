"""
Forecast ingestion — NWS API (15 min) + Weather Underground scraping (60 min).

Three sources per city:
  1. NWS API daily high (reliable, rate-limited)
  2. WU daily high scrape (settlement source — must work!)
  3. WU hourly peak scrape (max of hourly temps for the day)

WU scraping is the SETTLEMENT SOURCE for Polymarket resolution.
If WU scraping fails, forecast_quality → degraded and auto-trading stops.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
from bs4 import BeautifulSoup

from backend.config import Config
from backend.tz_utils import city_local_date
from backend.storage.db import get_session
from backend.storage.models import City
from backend.storage.repos import (
    get_all_cities,
    get_daily_high_metar,
    get_latest_forecast,
    get_latest_successful_forecast,
    insert_forecast_obs,
    update_heartbeat,
    upsert_event,
    get_event,
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

        today_et = city_local_date(city)

        try:
            high_f = await _fetch_nws_high(city)
        except Exception as e:
            log.error("nws: %s failed: %s", city.city_slug, e)
            high_f = None

        async with get_session() as sess:
            raw = json.dumps({"source": "nws", "high_f": high_f})
            await insert_forecast_obs(
                sess,
                city_id=city.id,
                source="nws",
                date_et=today_et,
                high_f=high_f,
                raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                raw_json=raw,
                parse_error=None if high_f is not None else "parse_failed",
            )
            await update_heartbeat(
                sess, "fetch_nws", success=(high_f is not None)
            )
        log.info("nws: %s high_f=%s", city.city_slug, high_f)


async def fetch_open_meteo_all() -> None:
    """Fetch Open-Meteo forecast for all enabled international cities."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        if city.is_us:
            continue

        city_date = city_local_date(city)

        try:
            high_f = await _fetch_open_meteo_high(city)
        except Exception as e:
            log.error("open-meteo: %s failed: %s", city.city_slug, e)
            high_f = None

        async with get_session() as sess:
            raw = json.dumps({"source": "open_meteo", "high_f": high_f})
            await insert_forecast_obs(
                sess,
                city_id=city.id,
                source="open_meteo",  # always "open_meteo" so web route query works
                date_et=city_date,
                high_f=high_f,
                raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                raw_json=raw,
                parse_error=None if high_f is not None else "parse_failed",
            )
            await update_heartbeat(
                sess, "fetch_open_meteo", success=(high_f is not None)
            )
        log.info("open-meteo: %s high_f=%s (date=%s)", city.city_slug, high_f, city_date)


async def _fetch_open_meteo_high(city: City) -> Optional[float]:
    """Fetch Open-Meteo hourly forecast and return today's max temperature (°C or °F)."""
    if city.lat is None or city.lon is None:
        log.warning("open-meteo: missing coords for %s", city.city_slug)
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }

    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
            async with http.get(url, params=params) as resp:
                if resp.status != 200:
                    log.error("open-meteo: HTTP %d for %s", resp.status, url)
                    return None
                data = await resp.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        if not temps:
            return None

        # Filter for "today" based on the city's ET context or UTC? 
        # Open-Meteo returns 24 values for 1 day usually starting from 00:00 of the requested day.
        # User said: "Math needs to be done so that it's still from the current day at which its called and doesn't roll over".
        # Since we are fetching 'forecast_days=1', it returns 24 hours starting from 00:00 of the current day (local time of the lat/lon).
        
        high_c = max(temps)
        if city.unit == "F":
            return round(high_c * 9 / 5 + 32, 1)
        return round(high_c, 1)

    except Exception as e:
        log.exception("open-meteo: failed for %s", city.city_slug)
        return None



async def _fetch_nws_high(city: City) -> Optional[float]:
    """Fetch NWS gridpoint forecast and return today's daytime high (°F)."""
    if not city.is_us or not city.nws_office:
        return None
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
                        return None
                    data = await resp.json(content_type=None)

            periods = (data.get("properties") or {}).get("periods") or []
            # Find today's daytime period
            today_et_str = city_local_date(city)
            for period in periods:
                if not period.get("isDaytime", True):
                    continue
                start = period.get("startTime", "")
                if today_et_str in start:
                    temp = period.get("temperature")
                    unit = period.get("temperatureUnit", "F")
                    if temp is None:
                        continue
                    temp_f = float(temp)
                    if unit == "C":
                        temp_f = temp_f * 9 / 5 + 32
                    return round(temp_f, 1)

            # Fallback — first daytime period
            for period in periods:
                if period.get("isDaytime", True):
                    temp = period.get("temperature")
                    if temp is not None:
                        return round(float(temp), 1)

        except asyncio.TimeoutError:
            log.warning("nws: timeout for %s (attempt %d/3)", city.city_slug, attempt + 1)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            log.error("nws: error for %s: %s", city.city_slug, e)
            if attempt < 2:
                await asyncio.sleep(1)

    return None


# ─── WU Scraping ─────────────────────────────────────────────────────────────

def _wu_daily_url(city: City) -> str:
    return f"{WU_BASE}/weather/{city.metar_station}"


def _wu_hourly_url(city: City) -> str:
    return f"{WU_BASE}/hourly/{city.metar_station}"


def wu_history_url(city: City, date_et: str) -> str:
    dt_str = date_et.replace("-", "")
    units = "m" if getattr(city, "unit", "F") == "C" else "e"
    
    country = "US"
    if not getattr(city, "is_us", True):
        country = (getattr(city, "wu_state", "") or "GB").upper()
        
    return f"https://api.weather.com/v1/location/{city.metar_station}:9:{country}/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units={units}&startDate={dt_str}"


async def fetch_wu_all() -> None:
    """Scrape WU daily + hourly for all enabled cities where WU is the settlement source."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        if not (city.wu_state and city.wu_city and city.metar_station):
            log.debug("wu: city %s missing WU config, skipping", city.city_slug)
            continue

        today_et = city_local_date(city)

        # Skip WU scraping for cities where WU is NOT the settlement source
        async with get_session() as sess:
            event = await get_event(sess, city.id, today_et)
        if event and not event.settlement_source_verified:
            log.debug("wu: %s settlement_source_verified=False (temporarily unverified on Polymarket), fetching forecast anyway", city.city_slug)

        # Rate limit: source-aware.  If ALL three WU sources have succeeded
        # recently, use the normal 30-min cooldown.  If ANY source has never
        # succeeded today, use a shorter retry interval so broken sources
        # recover quickly without hammering WU.
        async with get_session() as sess:
            last_daily = await get_latest_successful_forecast(sess, city.id, "wu_daily", today_et)
            last_hourly = await get_latest_successful_forecast(sess, city.id, "wu_hourly", today_et)
            last_history = await get_latest_successful_forecast(sess, city.id, "wu_history", today_et)

        all_sources = [last_daily, last_hourly, last_history]
        all_succeeded = all(f and f.fetched_at for f in all_sources)

        if all_succeeded:
            oldest_age = max(
                (datetime.now(timezone.utc) - (f.fetched_at.replace(tzinfo=timezone.utc) if f.fetched_at.tzinfo is None else f.fetched_at)).total_seconds()
                for f in all_sources
            )
            if oldest_age < Config.WU_MIN_SCRAPE_INTERVAL_SECONDS:
                log.debug("wu: %s all sources fresh, rate limited (oldest=%.0fs < %ds)",
                          city.city_slug, oldest_age, Config.WU_MIN_SCRAPE_INTERVAL_SECONDS)
                continue
        else:
            # At least one source missing — use shorter retry interval
            # Gate on the most recent *attempt* (any row, including failures)
            # to avoid hammering WU servers
            from backend.storage.repos import get_latest_forecast as _get_latest_any
            async with get_session() as sess:
                recent_attempts = []
                for src in ["wu_daily", "wu_hourly", "wu_history"]:
                    rec = await _get_latest_any(sess, city.id, src, today_et)
                    if rec and rec.fetched_at:
                        recent_attempts.append(rec.fetched_at)

            if recent_attempts:
                newest_attempt_age = min(
                    (datetime.now(timezone.utc) - (ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts)).total_seconds()
                    for ts in recent_attempts
                )
                if newest_attempt_age < Config.WU_FAILED_RETRY_INTERVAL_SECONDS:
                    log.debug("wu: %s has failed source(s) but attempted %.0fs ago, waiting",
                              city.city_slug, newest_attempt_age)
                    continue

            log.info("wu: %s has failed source(s) [daily=%s hourly=%s history=%s], retrying now",
                     city.city_slug,
                     "ok" if last_daily else "FAIL",
                     "ok" if last_hourly else "FAIL",
                     "ok" if last_history else "FAIL")

        try:
            await _scrape_wu_city(city, today_et)
        except Exception as e:
            log.exception("wu: Unhandled exception scraping %s: %s", city.city_slug, e)
            
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
    daily_high = None
    try:
        daily_high = await _scrape_wu_daily(city)
    except Exception as e:
        log.exception("wu_daily: %s scrape exception: %s", city.city_slug, e)

    hourly_peak, peak_hour = None, None
    try:
        hourly_peak, peak_hour = await _fetch_wu_hourly_api(city)
    except Exception as e:
        log.exception("wu_hourly_api: %s scrape exception: %s", city.city_slug, e)

    log.info("wu: %s fetched — daily=%.1f hourly=%.1f", city.city_slug, daily_high or 0, hourly_peak or 0)

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

    wu_ok = daily_high is not None or hourly_peak is not None or history_high is not None
    parse_err = None if wu_ok else "all_wu_sources_failed"

    # DB inserts — isolate per source so one failure doesn't block others
    async with get_session() as sess:
        # Floor: wu_daily can't be below any already-known high for today.
        # The WU daily page transitions to night forecast in late afternoon, causing
        # the scraper to pick up the overnight low instead of the day's high.
        # Use history_high + hourly_peak (already fetched above) as primary floors
        # since they come from APIs rather than HTML scraping. METAR provides
        # an additional floor if available.
        try:
            metar_high = await get_daily_high_metar(sess, city.id, date_et, city_tz=getattr(city, "tz", "America/New_York"))
        except Exception as e:
            log.exception("wu: %s failed get_daily_high_metar: %s", city.city_slug, e)
            metar_high = None

        floor = max(
            (v for v in [metar_high, history_high, hourly_peak] if v is not None),
            default=None,
        )
        if daily_high is not None and floor is not None and daily_high < floor:
            log.info(
                "wu_daily: %s scraped %.1f < floor %.1f "
                "(metar=%.1f, history=%.1f, hourly=%.1f) — using floor",
                city.city_slug, daily_high, floor,
                metar_high or 0, history_high or 0, hourly_peak or 0,
            )
            daily_high = floor

        try:
            raw = json.dumps({"high_f": daily_high, "source": "wu_daily"})
            await insert_forecast_obs(
                sess,
                city_id=city.id,
                source="wu_daily",
                date_et=date_et,
                high_f=daily_high,
                raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                raw_json=raw,
                parse_error=None if daily_high is not None else "parse_failed",
            )
        except Exception as e:
            log.exception("wu: %s failed to insert wu_daily: %s", city.city_slug, e)

        try:
            raw = json.dumps({"high_f": hourly_peak, "peak_hour": peak_hour, "source": "wu_hourly"})
            await insert_forecast_obs(
                sess,
                city_id=city.id,
                source="wu_hourly",
                date_et=date_et,
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
        "wu: %s daily=%.1f hourly=%.1f history=%.1f quality=%s",
        city.city_slug,
        daily_high or 0,
        hourly_peak or 0,
        history_high or 0,
        "ok" if wu_ok else "degraded",
    )


async def _scrape_wu_daily(city: City) -> Optional[float]:
    """Scrape WU station page for daily high forecast."""
    url = _wu_daily_url(city)
    html = await _fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")

    # Selector priority — WU redesigns frequently, use multiple fallbacks
    selectors = [
        # Pattern 1: structured data-ng-model or similar
        ("span", {"data-test": "daily-temperature-high"}),
        # Pattern 2: Today's high in forecast today block  
        ("div", {"class": re.compile(r"todaySummaryTemp|today-summary", re.I)}),
    ]

    for tag, attrs in selectors:
        el = soup.find(tag, attrs)
        if el:
            temp = _extract_temp_from_element(el)
            if temp is not None:
                return temp

    # Fallback: regex scan for temperature patterns in relevant sections
    # Look for "High: 87°F" or similar in body text
    text = soup.get_text(" ")
    match = re.search(r"(?:high|today).*?(\d{2,3})\s*°?\s*F", text, re.I)
    if match:
        try:
            val = float(match.group(1))
            if 0 <= val <= 130:
                return val
        except ValueError:
            pass

    # Last resort: find all temperature spans and take the max plausible value
    temps = []
    for el in soup.find_all(["span", "div", "td"], string=re.compile(r"^\s*\d{2,3}\s*°?\s*$")):
        t = _extract_temp_from_element(el)
        if t and 50 <= t <= 130:
            temps.append(t)
    if temps:
        return max(temps)

    log.warning("wu_daily: %s — could not parse high temp from %s", city.city_slug, url)
    return None


async def _fetch_wu_hourly_api(city: City) -> tuple[Optional[float], Optional[str]]:
    """Fetch WU hourly forecast via weather.com v1 API, returning (max_temp_f, peak_hour_et_str).

    Uses the same v1 API pattern as wu_history to avoid HTML-scraping accuracy issues
    (the HTML page mixes actual temp and feels-like temp columns, causing off-by-1 errors).
    Filters to today's ET hours only so tomorrow's highs don't inflate the result.
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

    today_et = city_local_date(city)  # YYYY-MM-DD

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_WU_HEADERS) as http:
                async with http.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        forecasts = data.get("forecasts", [])

                        # Filter to today's ET hours only
                        today_forecasts = [
                            f for f in forecasts
                            if f.get("fcst_valid_local", "")[:10] == today_et
                            and f.get("temp") is not None
                        ]

                        if today_forecasts:
                            best = max(today_forecasts, key=lambda f: f["temp"])
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
                        log.warning("wu_hourly_api: %s — no today forecasts in response", city.city_slug)
                        return None, None
                    elif resp.status == 404:
                        log.warning("wu_hourly_api: %s — 404", city.city_slug)
                        return None, None
                    else:
                        body = await resp.text()
                        log.warning("wu_hourly_api: HTTP %d for %s — body: %.200s",
                                    resp.status, city.city_slug, body)
        except Exception as e:
            log.warning("wu_hourly_api: failed for %s (attempt %d/3): %s", city.city_slug, attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(2)

    return None, None


def _extract_temp_from_element(el) -> Optional[float]:
    """Extract a temperature float from a BS4 element's text."""
    if el is None:
        return None
    text = el.get_text(strip=True)
    # Match "87", "87°", "87°F", "87 °F"
    m = re.match(r"^(M?-?\d{1,3})(?:\s*°?\s*[FC]?)?\s*$", text)
    if m:
        try:
            val = float(m.group(1))
            if -60 <= val <= 140:
                return val
        except ValueError:
            pass
    return None


async def _fetch_html(url: str, retries: int = 3) -> Optional[str]:
    """Fetch HTML page with browser-like headers and retry logic."""
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(
                timeout=_TIMEOUT, headers=_WU_HEADERS
            ) as http:
                async with http.get(url, allow_redirects=True) as resp:
                    if resp.status == 429:
                        wait = 10 * (attempt + 1)
                        log.warning("wu: rate limited (429), waiting %ds", wait)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        log.error("wu: HTTP %d from %s", resp.status, url)
                        return None
                    return await resp.text()
        except asyncio.TimeoutError:
            log.warning("wu: timeout for %s (attempt %d/%d)", url[:60], attempt + 1, retries)
            if attempt < retries - 1:
                await asyncio.sleep(3 * (attempt + 1))
        except Exception as e:
            log.error("wu: error for %s: %s", url[:60], e)
    return None


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
                        log.warning("wu_history: HTTP %d for %s — body: %.200s",
                                    resp.status, city.city_slug, body)
        except Exception as e:
            log.warning("wu_history: fetch failed for %s: %s", city.city_slug, e)
            await asyncio.sleep(2)
    return None, None
