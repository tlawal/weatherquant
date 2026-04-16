"""
TGFTP METAR ingestion — polls tgftp.nws.noaa.gov every 60 seconds.

Primary source for the settlement-high card. Parses raw METAR text
from NOAA's TGFTP server, which is typically faster than the
weather.com (wu_history) API used previously.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

from backend.tz_utils import city_local_date
from backend.storage.db import get_session
from backend.storage.models import City
from backend.storage.repos import (
    get_all_cities,
    get_daily_high_metar,
    get_metar_obs_by_key,
    insert_metar_obs,
    update_heartbeat,
)

log = logging.getLogger(__name__)

TGFTP_BASE_URL = "https://tgftp.nws.noaa.gov/data/observations/metar/stations"
_USER_AGENT = "WeatherQuant/1.0 (contact@weatherquant.local)"
_TIMEOUT = aiohttp.ClientTimeout(total=15)
_MAX_RETRIES = 3

# METAR temperature regex: M##/M## (e.g. "18/06", "M01/M03")
_TEMP_RE = re.compile(r"\b(M?)(\d{2})/(M?\d{2})\b")

# METAR observation time regex: DDHHMMZ (e.g. "141853Z")
_OBS_TIME_RE = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")


def _c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _parse_tgftp_temp(raw_metar: str) -> Optional[tuple[float, float]]:
    """Parse temperature from raw METAR string. Returns (temp_c, temp_f) or None."""
    m = _TEMP_RE.search(raw_metar)
    if not m:
        return None
    try:
        sign = -1 if m.group(1) == "M" else 1
        tc = sign * float(m.group(2))
        return tc, _c_to_f(tc)
    except (ValueError, TypeError):
        return None


def _parse_tgftp_obs_time(raw_metar: str) -> Optional[datetime]:
    """Parse observation time from raw METAR DDHHMMZ token.

    Handles month rollover: if the parsed day > today's UTC day,
    assume it belongs to the previous month.
    """
    m = _OBS_TIME_RE.search(raw_metar)
    if not m:
        return None
    try:
        day = int(m.group(1))
        hour = int(m.group(2))
        minute = int(m.group(3))

        now = datetime.now(timezone.utc)
        year = now.year
        month = now.month

        # Handle month rollover (e.g. on the 1st of the month, the METAR
        # day might be 30 or 31 from the previous month)
        if day > now.day:
            # Must be from the previous month
            month -= 1
            if month == 0:
                month = 12
                year -= 1

        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _parse_tgftp_response(text: str) -> Optional[tuple[str, datetime, float, float]]:
    """Parse a TGFTP station response.

    Returns (raw_metar_line, observed_at, temp_c, temp_f) or None.
    The first line is a date header (e.g. "2026/04/14 18:53") — skip it.
    The last METAR line is the freshest (TGFTP sometimes appends SPECI reports).
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None

    # Last line is the freshest observation
    raw_metar = lines[-1]

    temp = _parse_tgftp_temp(raw_metar)
    if temp is None:
        return None
    temp_c, temp_f = temp

    obs_time = _parse_tgftp_obs_time(raw_metar)
    if obs_time is None:
        return None

    return raw_metar, obs_time, temp_c, temp_f


async def _fetch_station(station: str, http: aiohttp.ClientSession) -> Optional[str]:
    """Fetch raw text from TGFTP for a single station with retries."""
    url = f"{TGFTP_BASE_URL}/{station}.TXT"

    for attempt in range(_MAX_RETRIES):
        try:
            async with http.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
                elif resp.status == 404:
                    log.debug("tgftp: station %s not found (404)", station)
                    return None
                else:
                    log.warning("tgftp: HTTP %d for station %s (attempt %d)", resp.status, station, attempt + 1)
        except asyncio.TimeoutError:
            log.warning("tgftp: timeout for station %s (attempt %d)", station, attempt + 1)
        except Exception as e:
            log.warning("tgftp: error for station %s (attempt %d): %s", station, attempt + 1, e)

        if attempt < _MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)

    return None


async def fetch_tgftp_all() -> None:
    """Fetch TGFTP METAR for all enabled cities with a metar_station.

    Staggers requests with 1s delay to be polite to NOAA.
    """
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    if not cities:
        log.warning("tgftp: no enabled cities configured")
        return

    # Only US cities with a metar_station are relevant for TGFTP
    eligible = [c for c in cities if c.is_us and c.metar_station]
    if not eligible:
        log.debug("tgftp: no eligible US cities with metar_station")
        return

    success_count = 0
    fail_count = 0

    async with aiohttp.ClientSession(
        timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
    ) as http:
        for city in eligible:
            station = city.metar_station.upper()
            try:
                raw_text = await _fetch_station(station, http)
                if raw_text is None:
                    fail_count += 1
                    continue

                parsed = _parse_tgftp_response(raw_text)
                if parsed is None:
                    log.debug("tgftp: failed to parse response for %s", station)
                    fail_count += 1
                    continue

                raw_metar, obs_time, temp_c, temp_f = parsed

                today_local = city_local_date(city)
                async with get_session() as sess:
                    # Dedupe: skip if we already have this (station, observed_at)
                    existing = await get_metar_obs_by_key(sess, city.id, station, obs_time)
                    if existing is not None:
                        continue

                    prev_high = await get_daily_high_metar(
                        sess,
                        city.id,
                        today_local,
                        city_tz=getattr(city, "tz", "America/New_York"),
                        source="tgftp",
                    )
                    daily_high = max(
                        (v for v in [temp_f, prev_high] if v is not None),
                        default=temp_f,
                    )
                    await insert_metar_obs(
                        sess,
                        city_id=city.id,
                        metar_station=station,
                        observed_at=obs_time,
                        report_at=obs_time,
                        temp_c=temp_c,
                        temp_f=temp_f,
                        daily_high_f=daily_high,
                        raw_text=raw_metar,
                        source="tgftp",
                    )
                    success_count += 1

            except Exception as e:
                log.error("tgftp: %s failed: %s", station, e)
                fail_count += 1

            # Stagger requests — be polite to NOAA
            await asyncio.sleep(1)

    log.info(
        "tgftp: fetched %d stations (%d success, %d fail)",
        len(eligible), success_count, fail_count,
    )

    if success_count > 0:
        await _mark_heartbeat_success()


async def _mark_heartbeat_success() -> None:
    try:
        async with get_session() as sess:
            await update_heartbeat(sess, "fetch_tgftp_metar", success=True)
    except Exception:
        pass
