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
from backend.storage.db import get_session
from backend.storage.repos import (
    get_all_cities,
    get_daily_high_metar,
    insert_metar_obs,
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


async def fetch_metar_all(session=None) -> None:
    """Fetch METAR for all enabled cities and persist to DB."""
    close_session = session is None
    if session is None:
        pass  # handled via context manager

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    if not cities:
        log.warning("metar: no enabled cities configured")
        return

    station_map = {c.metar_station: c for c in cities if c.metar_station}
    if not station_map:
        log.warning("metar: no cities have metar_station configured")
        return

    stations = ",".join(station_map.keys())
    url = f"{METAR_URL}?ids={stations}&format=json&latest=1"

    try:
        async with aiohttp.ClientSession(
            timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
        ) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.error("metar: HTTP %d from %s", resp.status, url)
                    await _mark_heartbeat_error(f"HTTP {resp.status}")
                    return
                data = await resp.json(content_type=None)

    except asyncio.TimeoutError:
        log.error("metar: request timed out")
        await _mark_heartbeat_error("timeout")
        return
    except Exception as e:
        log.error("metar: fetch failed: %s", e)
        await _mark_heartbeat_error(str(e))
        return

    if not isinstance(data, list):
        log.error("metar: unexpected response format: %r", type(data))
        return

    today_et = date.today().isoformat()  # system TZ is ET per config

    for obs in data:
        station_id = (obs.get("stationId") or obs.get("station") or "").upper()
        city = station_map.get(station_id)
        if not city:
            continue

        temp = _parse_temp(obs)
        if temp is None:
            log.warning("metar: could not parse temp for station %s", station_id)
            temp_c, temp_f = None, None
        else:
            temp_c, temp_f = temp
            
        # Unify units: if city uses Celsius, map the native C observation into the F column 
        # so downstream processing stays completely seamless without having to branch.
        if getattr(city, "unit", "F") == "C" and temp_c is not None:
            temp_f = temp_c

        obs_time = _parse_obs_time(obs)
        raw_str = json.dumps(obs, default=str)

        async with get_session() as sess:
            # Compute daily high (query + compare)
            prev_high = await get_daily_high_metar(sess, city.id, today_et)
            daily_high = max(
                (v for v in [temp_f, prev_high] if v is not None), default=None
            )

            await insert_metar_obs(
                sess,
                city_id=city.id,
                metar_station=station_id,
                observed_at=obs_time,
                temp_c=temp_c,
                temp_f=temp_f,
                daily_high_f=daily_high,
                raw_json=raw_str,
            )

        log.debug(
            "metar: %s temp=%.1f°F daily_high=%.1f°F",
            station_id,
            temp_f or 0,
            daily_high or 0,
        )

    await _mark_heartbeat_success()


async def _mark_heartbeat_success() -> None:
    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_metar", success=True)


async def _mark_heartbeat_error(error: str) -> None:
    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_metar", success=False, error=error)
