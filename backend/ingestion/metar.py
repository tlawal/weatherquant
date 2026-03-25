"""
METAR ingestion — polls aviationweather.gov every 60 seconds.

Ground truth for real-time temperature observations.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp

from backend.config import Config
from backend.tz_utils import city_local_date
from backend.storage.db import get_session
from backend.storage.repos import (
    get_all_cities,
    get_daily_high_metar,
    insert_metar_obs,
    insert_metar_obs_extended,
    update_heartbeat,
)

log = logging.getLogger(__name__)

METAR_URL = "https://aviationweather.gov/api/data/metar"
ET = ZoneInfo("America/New_York")

_USER_AGENT = "WeatherQuant/1.0 (contact@weatherquant.local)"
_TIMEOUT = aiohttp.ClientTimeout(total=15)


def _c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _parse_temp(obs: dict) -> Optional[tuple[float, float]]:
    """Return (temp_c, temp_f) or None if not parseable."""
    # Primary: temp field (degrees C)
    temp_c = obs.get("temp")
    if temp_c is not None:
        try:
            tc = float(temp_c)
            return tc, _c_to_f(tc)
        except (ValueError, TypeError):
            pass

    # Fallback: temperature string in raw METAR
    raw = obs.get("rawOb", "") or ""
    import re
    m = re.search(r"\b(M?)(\d{2})/(M?\d{2})\b", raw)
    if m:
        try:
            sign = -1 if m.group(1) == "M" else 1
            tc = sign * float(m.group(2))
            return tc, _c_to_f(tc)
        except (ValueError, TypeError):
            pass

    return None


def _parse_obs_time(obs: dict) -> Optional[datetime]:
    """Parse observation time from METAR response."""
    raw_ts = obs.get("obsTime") or obs.get("reportTime")
    if raw_ts:
        try:
            # Can be epoch int or ISO string
            if isinstance(raw_ts, (int, float)):
                return datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
            return datetime.fromisoformat(str(raw_ts).rstrip("Z")).replace(
                tzinfo=timezone.utc
            )
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _humidity_from_magnus(temp_c: float, dewp_c: float) -> float:
    """Compute relative humidity (%) via the Magnus formula."""
    import math
    a, b = 17.625, 243.04
    rh = 100.0 * math.exp((a * dewp_c) / (b + dewp_c)) / math.exp((a * temp_c) / (b + temp_c))
    return round(max(0.0, min(100.0, rh)), 1)


# Map wx_string + cloud cover to human-readable condition
_WX_CONDITION_MAP = [
    ("TS", "Thunderstorm"),
    ("GR", "Hail"),
    ("SN", "Snow"),
    ("FZRA", "Freezing Rain"),
    ("RA", "Rain"),
    ("DZ", "Drizzle"),
    ("SH", "Showers"),
    ("FG", "Fog"),
    ("HZ", "Haze"),
    ("BR", "Mist"),
]

_COVER_CONDITION = {
    "CLR": "Fair", "SKC": "Fair", "FEW": "Fair",
    "SCT": "Partly Cloudy", "BKN": "Mostly Cloudy", "OVC": "Cloudy",
}


def _derive_condition(wx_string: Optional[str], cloud_cover: Optional[str]) -> str:
    """Derive a human-readable condition string from wx_string and cloud cover."""
    if wx_string:
        wx_upper = wx_string.upper()
        for token, label in _WX_CONDITION_MAP:
            if token in wx_upper:
                return label
    if cloud_cover:
        return _COVER_CONDITION.get(cloud_cover.upper(), "Fair")
    return "Fair"


def _parse_extended(obs: dict, temp_c: Optional[float] = None) -> dict:
    """Parse extended METAR fields from aviationweather.gov JSON response.

    Returns a dict suitable for insert_metar_obs_extended().
    """
    ext: dict = {}

    # Dewpoint
    dewp_c = obs.get("dewp")
    if dewp_c is not None:
        try:
            dc = float(dewp_c)
            ext["dewpoint_c"] = round(dc, 1)
            ext["dewpoint_f"] = round(dc * 9 / 5 + 32, 1)
            if temp_c is not None:
                ext["humidity_pct"] = _humidity_from_magnus(float(temp_c), dc)
        except (ValueError, TypeError):
            pass

    # Wind
    wdir = obs.get("wdir")
    if wdir is not None:
        try:
            ext["wind_dir_deg"] = int(wdir)
        except (ValueError, TypeError):
            pass

    wspd = obs.get("wspd")
    if wspd is not None:
        try:
            ext["wind_speed_kt"] = float(wspd)
        except (ValueError, TypeError):
            pass

    wgst = obs.get("wgst")
    if wgst is not None:
        try:
            ext["wind_gust_kt"] = float(wgst)
        except (ValueError, TypeError):
            pass

    # Pressure
    altim = obs.get("altim")
    if altim is not None:
        try:
            ext["altimeter_inhg"] = round(float(altim), 2)
        except (ValueError, TypeError):
            pass

    slp = obs.get("slp")
    if slp is not None:
        try:
            ext["sea_level_pressure_mb"] = round(float(slp), 1)
        except (ValueError, TypeError):
            pass

    # Visibility
    visib = obs.get("visib")
    if visib is not None:
        try:
            ext["visibility_sm"] = round(float(visib), 1)
        except (ValueError, TypeError):
            pass

    # Cloud cover — use the dominant (highest) cover level
    clouds = obs.get("clouds")
    cover = obs.get("cover")
    if isinstance(clouds, list) and clouds:
        # clouds is array of {cover: "BKN", base: 5000}
        # Use the first layer's cover as the dominant one
        first = clouds[0]
        ext["cloud_cover"] = first.get("cover")
        base = first.get("base")
        if base is not None:
            try:
                ext["cloud_base_ft"] = int(base)
            except (ValueError, TypeError):
                pass
    elif cover:
        ext["cloud_cover"] = str(cover)

    # Weather string (present weather)
    wx = obs.get("wxString")
    if wx:
        ext["wx_string"] = str(wx)[:64]

    # Derive human-readable condition
    ext["condition"] = _derive_condition(ext.get("wx_string"), ext.get("cloud_cover"))

    return ext


async def fetch_metar_all() -> None:
    """Fetch Current Weather for all enabled cities. US uses METAR; Intl uses Open-Meteo."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    if not cities:
        log.warning("metar: no enabled cities configured")
        return

    us_cities = [c for c in cities if c.is_us and c.metar_station]
    # International cities with ICAO metar_station also use the global aviationweather.gov feed
    intl_metar_cities = [c for c in cities if not c.is_us and c.metar_station]
    # International cities without a station code use Open-Meteo
    intl_openmeteo_cities = [c for c in cities if not c.is_us and not c.metar_station and c.lat and c.lon]

    if us_cities:
        await _fetch_us_metars(us_cities)

    if intl_metar_cities:
        # ICAO stations are global — aviationweather.gov covers all of them
        await _fetch_us_metars(intl_metar_cities)

    # Supplementary: weather.gov observations as fallback/cross-validation
    nws_obs_cities = us_cities + intl_metar_cities
    if nws_obs_cities:
        try:
            await _fetch_nws_observations(nws_obs_cities)
        except Exception as e:
            log.warning("nws_obs: supplementary fetch failed: %s", e)

    if intl_openmeteo_cities:
        await _fetch_intl_open_meteo(intl_openmeteo_cities)


async def _fetch_us_metars(cities: list[City]) -> None:
    station_map = {c.metar_station: c for c in cities if c.metar_station}
    stations = ",".join(station_map.keys())
    url = f"{METAR_URL}?ids={stations}&format=json&latest=1"

    try:
        async with aiohttp.ClientSession(
            timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
        ) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.error("metar: HTTP %d from %s", resp.status, url)
                    return
                data = await resp.json(content_type=None)
    except Exception as e:
        log.error("metar: fetch failed: %s", e)
        return

    if not isinstance(data, list):
        return

    for obs in data:
        station_id = (obs.get("stationId") or obs.get("station") or "").upper()
        city = station_map.get(station_id)
        if not city:
            continue

        today_local = city_local_date(city)

        temp = _parse_temp(obs)
        if temp is None:
            continue
        temp_c, temp_f = temp

        obs_time = _parse_obs_time(obs)
        raw_str = json.dumps(obs, default=str)

        async with get_session() as sess:
            prev_high = await get_daily_high_metar(sess, city.id, today_local, city_tz=getattr(city, "tz", "America/New_York"))
            daily_high = max(
                (v for v in [temp_f, prev_high] if v is not None), default=temp_f
            )

            # Extract reportTime if available separately
            report_at = _parse_obs_time({"obsTime": obs.get("reportTime")}) if obs.get("reportTime") else obs_time
            raw_text = obs.get("rawOb")

            metar_row = await insert_metar_obs(
                sess,
                city_id=city.id,
                metar_station=station_id,
                observed_at=obs_time,
                report_at=report_at,
                temp_c=temp_c,
                temp_f=temp_f,
                daily_high_f=daily_high,
                raw_text=raw_text,
                raw_json=raw_str,
            )

            # Parse and store extended fields (dewpoint, wind, pressure, etc.)
            ext_data = _parse_extended(obs, temp_c=temp_c)
            if ext_data:
                try:
                    await insert_metar_obs_extended(sess, metar_obs_id=metar_row.id, **ext_data)
                except Exception as e:
                    log.debug("metar_ext: failed for %s: %s", station_id, e)

    await _mark_heartbeat_success()


async def _fetch_nws_observations(cities: list[City]) -> None:
    """Supplementary: fetch latest observation from weather.gov for US cities."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
    async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=headers) as http:
        for city in cities:
            try:
                url = f"https://api.weather.gov/stations/{city.metar_station}/observations/latest"
                async with http.get(url) as resp:
                    if resp.status != 200:
                        log.debug("nws_obs: HTTP %d for %s", resp.status, city.metar_station)
                        continue
                    data = await resp.json(content_type=None)

                props = data.get("properties", {})
                temp_c_val = (props.get("temperature") or {}).get("value")
                if temp_c_val is None:
                    continue
                temp_c = float(temp_c_val)
                temp_f = _c_to_f(temp_c)

                ts_str = props.get("timestamp")
                obs_time = (
                    datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
                    if ts_str else datetime.now(timezone.utc)
                )

                today_local = city_local_date(city)
                async with get_session() as sess:
                    prev_high = await get_daily_high_metar(sess, city.id, today_local, city_tz=getattr(city, "tz", "America/New_York"))
                    daily_high = max(
                        (v for v in [temp_f, prev_high] if v is not None), default=temp_f
                    )
                    await insert_metar_obs(
                        sess,
                        city_id=city.id,
                        metar_station=city.metar_station,
                        observed_at=obs_time,
                        report_at=obs_time,
                        temp_c=temp_c,
                        temp_f=temp_f,
                        daily_high_f=daily_high,
                        raw_text=props.get("rawMessage"),
                        raw_json=json.dumps({"source": "nws_obs", **props}, default=str),
                    )
                await asyncio.sleep(0.3)  # rate limit courtesy
            except Exception as e:
                log.error("nws_obs: %s failed: %s", city.metar_station, e)


async def _fetch_intl_open_meteo(cities: list[City]) -> None:
    """Fetch current weather for international cities via Open-Meteo."""
    # Group by lat/lon to use Open-Meteo's bulk API
    lats = ",".join(str(c.lat) for c in cities)
    lons = ",".join(str(c.lon) for c in cities)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&current_weather=true"

    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.error("open-meteo: HTTP %d from %s", resp.status, url)
                    return
                data = await resp.json()
    except Exception as e:
        log.error("open-meteo: fetch failed: %s", e)
        return

    # Open-Meteo returns a list if multiple locations are requested
    if not isinstance(data, list):
        data = [data]

    for i, obs_data in enumerate(data):
        if i >= len(cities):
            break
        city = cities[i]
        today_local = city_local_date(city)
        curr = obs_data.get("current_weather")
        if not curr:
            continue

        temp_c = curr.get("temperature")
        if temp_c is None:
            continue

        # User said for intl cities use C but my internal daily_high_f column handles whatever is passed.
        # Actually for intl cities I should store C in the F column as well if that's what's used.
        temp_internal = float(temp_c)
        if city.unit == "F":
             temp_internal = _c_to_f(temp_internal)

        obs_time_raw = curr.get("time") # ISO string
        try:
            obs_time = datetime.fromisoformat(obs_time_raw).replace(tzinfo=timezone.utc)
        except Exception:
            obs_time = datetime.now(timezone.utc)

        raw_str = json.dumps(obs_data)

        async with get_session() as sess:
            prev_high = await get_daily_high_metar(sess, city.id, today_local, city_tz=getattr(city, "tz", "America/New_York"))
            daily_high = max(
                (v for v in [temp_internal, prev_high] if v is not None), default=temp_internal
            )

            await insert_metar_obs(
                sess,
                city_id=city.id,
                metar_station=city.metar_station or "OM",
                observed_at=obs_time,
                report_at=obs_time, # Open-Meteo usually just has one time
                temp_c=temp_c,
                temp_f=temp_internal,
                daily_high_f=daily_high,
                raw_text=None,
                raw_json=raw_str,
            )


def should_poll_station(
    observation_minutes: list[int], now_minute: int,
    pre_window: int = 2, post_window: int = 10,
    window: int | None = None,
) -> bool:
    """Return True if now_minute is within [-pre_window, +post_window] of any observation minute.

    Asymmetric window: poll from 2 min before to 10 min after each observation
    minute, covering delayed appearances on aviationweather.gov.
    Legacy `window` param is accepted but ignored.
    """
    for m in observation_minutes:
        diff = (now_minute - m) % 60
        if diff <= post_window or diff >= (60 - pre_window):
            return True
    return False


async def fetch_metar_smart() -> None:
    """Fetch METAR only for stations near their observation window."""
    from backend.storage.repos import get_station_profile

    now = datetime.now(timezone.utc)
    now_minute = now.minute

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    # All cities with METAR station (US and intl ICAO) use smart polling
    metar_cities = [c for c in cities if c.metar_station]
    in_window = []
    for city in metar_cities:
        async with get_session() as sess:
            profile = await get_station_profile(sess, city.metar_station)
        if not profile or not profile.observation_minutes or (profile.confidence or 0) < 0.7:
            in_window.append(city)  # no confident profile → always poll
            continue
        valid_minutes = json.loads(profile.observation_minutes)
        if should_poll_station(valid_minutes, now_minute):
            in_window.append(city)

    if in_window:
        await _fetch_us_metars(in_window)
        log.info("smart_poll: fetched %d/%d METAR stations at minute :%02d",
                 len(in_window), len(metar_cities), now_minute)

    # International cities without METAR station use Open-Meteo (always polled)
    intl_openmeteo_cities = [c for c in cities if not c.is_us and not c.metar_station and c.lat and c.lon]
    if intl_openmeteo_cities:
        await _fetch_intl_open_meteo(intl_openmeteo_cities)


async def _mark_heartbeat_success() -> None:
    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_metar", success=True)


async def _mark_heartbeat_error(error: str) -> None:
    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_metar", success=False, error=error)
