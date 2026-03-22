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
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
from bs4 import BeautifulSoup

from backend.config import Config
from backend.storage.db import get_session
from backend.storage.models import City
from backend.storage.repos import (
    get_all_cities,
    get_latest_forecast,
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

    today_et = date.today().isoformat()

    for city in cities:
        if not city.is_us:
            continue
        if not (city.nws_office and city.nws_grid_x and city.nws_grid_y):
            continue

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

    today_et = date.today().isoformat()

    for city in cities:
        if city.is_us:
            continue
        
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
                source="open_meteo",
                date_et=today_et,
                high_f=high_f,
                raw_payload_hash=hashlib.md5(raw.encode()).hexdigest(),
                raw_json=raw,
                parse_error=None if high_f is not None else "parse_failed",
            )
            await update_heartbeat(
                sess, "fetch_open_meteo", success=(high_f is not None)
            )
        log.info("open-meteo: %s high_f=%s", city.city_slug, high_f)


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
            today_et_str = date.today().isoformat()
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


def _wu_history_url(city: City, date_et: str) -> str:
    dt_str = date_et.replace("-", "")
    units = "m" if getattr(city, "unit", "F") == "C" else "e"
    
    country = "US"
    if not getattr(city, "is_us", True):
        country = (getattr(city, "wu_state", "") or "GB").upper()
        
    return f"https://api.weather.com/v1/location/{city.metar_station}:9:{country}/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units={units}&startDate={dt_str}"


async def fetch_wu_all() -> None:
    """Scrape WU daily + hourly for all enabled cities."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    today_et = date.today().isoformat()

    for city in cities:
        if not (city.wu_state and city.wu_city and city.metar_station):
            log.debug("wu: city %s missing WU config, skipping", city.city_slug)
            continue

        # Rate limit: check last WU scrape time
        async with get_session() as sess:
            last_daily = await get_latest_forecast(sess, city.id, "wu_daily", today_et)

        if last_daily and last_daily.fetched_at:
            age = (
                datetime.now(timezone.utc) - last_daily.fetched_at
            ).total_seconds()
            if age < Config.WU_MIN_SCRAPE_INTERVAL_SECONDS:
                log.debug("wu: %s rate limited (age=%.0fs < %ds)", city.city_slug, age, Config.WU_MIN_SCRAPE_INTERVAL_SECONDS)
                continue

        await _scrape_wu_city(city, today_et)
        # Stagger per city to be polite to WU
        await asyncio.sleep(5)


async def _scrape_wu_city(city: City, date_et: str) -> None:
    daily_high = await _scrape_wu_daily(city)
    hourly_peak = await _scrape_wu_hourly(city)
    history_high = await _fetch_wu_history_api(city, date_et)

    wu_ok = daily_high is not None or hourly_peak is not None or history_high is not None
    parse_err = None if wu_ok else "all_wu_sources_failed"

    async with get_session() as sess:
        if daily_high is not None or True:  # always write, even None
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

        if hourly_peak is not None or True:
            raw = json.dumps({"high_f": hourly_peak, "source": "wu_hourly"})
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

        if history_high is not None or True:
            raw = json.dumps({"high_f": history_high, "source": "wu_history"})
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

        # Update forecast_quality on the event
        event = await get_event(sess, city.id, date_et)
        if event:
            quality = "ok" if wu_ok else "degraded"
            event.forecast_quality = quality
            event.wu_scrape_error = parse_err
            await sess.commit()

        await update_heartbeat(sess, "fetch_wu", success=wu_ok, error=parse_err)

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


async def _scrape_wu_hourly(city: City) -> Optional[float]:
    """Scrape WU hourly page and return max hourly temp for the day."""
    url = _wu_hourly_url(city)
    html = await _fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")

    temps: list[float] = []

    # Primary: table rows with hourly temps
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        for cell in cells:
            t = _extract_temp_from_element(cell)
            if t and 20 <= t <= 130:
                temps.append(t)

    if not temps:
        # Fallback: all temperature-looking spans
        for el in soup.find_all(["span", "div"], string=re.compile(r"^\s*\d{2,3}\s*°?\s*$")):
            t = _extract_temp_from_element(el)
            if t and 20 <= t <= 130:
                temps.append(t)

    if not temps:
        log.warning("wu_hourly: %s — no temps found", city.city_slug)
        return None

    return max(temps)


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


async def _fetch_wu_history_api(city: City, date_et: str) -> Optional[float]:
    """Fetch WU actual historical observations using the internal API to resolve settlement ground truth."""
    url = _wu_history_url(city, date_et)
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_WU_HEADERS) as http:
                async with http.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        obs = data.get("observations", [])
                        temps = [o.get("temp") for o in obs if o.get("temp") is not None]
                        if temps:
                            return float(max(temps))
                    elif resp.status == 404:
                        return None
        except Exception as e:
            log.warning("wu_history: fetch failed for %s: %s", city.city_slug, e)
            await asyncio.sleep(2)
    return None
