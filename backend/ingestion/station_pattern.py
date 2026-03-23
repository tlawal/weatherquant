"""
Station pattern detection — auto-detect METAR reporting cadence.

METAR stations report at specific minutes past each hour (e.g., KATL at :52).
Polymarket/WU only counts observations at these official timestamps.
This module analyzes recent observations to detect each station's pattern.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.storage.db import get_session
from backend.storage.repos import (
    get_all_cities,
    get_metar_obs_for_station,
    get_station_profile,
    upsert_station_profile,
)
from backend.storage.models import City
from backend.ingestion.forecasts import wu_history_url
from backend.tz_utils import city_local_date
import aiohttp

log = logging.getLogger(__name__)


async def detect_observation_pattern(city: City) -> Optional[dict]:
    """Analyze recent observations to detect the station's reporting cadence.
    
    If the city is US or uses Wunderground, we fetch from WU History.
    Otherwise, we analyze recent MetarObs.

    Algorithm:
    1. Check if WU API should be used.
    2. If so, fetch the history for the last 2 days to get observation minutes.
    3. Find peaks, merge, and classify frequency.
    4. Compute confidence.

    Returns: {"minutes": [52], "frequency": "hourly", "samples": 23, "confidence": 0.94}
             or None if insufficient data
    """
    
    # Use WU if US city or uses WU as resolution source
    if city.is_us or (city.wu_state and city.wu_city):
        return await _detect_wu_pattern(city)

    metar_station = city.metar_station
    if not metar_station:
        return None
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    async with get_session() as sess:
        obs_list = await get_metar_obs_for_station(sess, metar_station, since)

    if len(obs_list) < 3:
        log.debug("station_pattern: %s — only %d obs (<3), skipping", metar_station, len(obs_list))
        return None

    # Extract minute of each observation
    minutes = [ob.observed_at.minute for ob in obs_list if ob.observed_at]

    if not minutes:
        return None

    # Count distinct hours with data
    hours_with_data = len(set(
        (ob.observed_at.year, ob.observed_at.month, ob.observed_at.day, ob.observed_at.hour)
        for ob in obs_list if ob.observed_at
    ))

    if hours_with_data < 2:
        log.debug("station_pattern: %s — only %d hours with data (<2)", metar_station, hours_with_data)
        return None

    # Build histogram
    minute_counts = Counter(minutes)
    threshold = hours_with_data * 0.4  # >40% of hours

    # Find raw peaks
    raw_peaks = [m for m, count in minute_counts.items() if count >= threshold]

    if not raw_peaks:
        # Lower threshold and try again
        threshold = hours_with_data * 0.2
        raw_peaks = [m for m, count in minute_counts.items() if count >= threshold]

    if not raw_peaks:
        log.debug("station_pattern: %s — no peaks found", metar_station)
        return None

    # Merge adjacent peaks (±1 tolerance) — keep the one with highest count
    merged = _merge_adjacent_peaks(raw_peaks, minute_counts)

    # Classify frequency
    if len(merged) == 1:
        frequency = "hourly"
    elif len(merged) == 2:
        gap = abs(merged[0] - merged[1])
        gap = min(gap, 60 - gap)  # handle wraparound
        frequency = "half_hourly" if 25 <= gap <= 35 else "irregular"
    else:
        frequency = "irregular"

    # Compute confidence: what fraction of expected observations hit a peak minute?
    total_peak_hits = sum(
        1 for m in minutes
        if any(abs((m - peak) % 60) <= 1 or abs((peak - m) % 60) <= 1 for peak in merged)
    )
    confidence = round(total_peak_hits / len(minutes), 3) if minutes else 0.0

    result = {
        "minutes": sorted(merged),
        "frequency": frequency,
        "samples": len(obs_list),
        "confidence": confidence,
    }
    log.info("station_pattern: %s → %s", metar_station, result)
    return result


async def _detect_wu_pattern(city: City) -> Optional[dict]:
    """Detect observation pattern by querying the WU historical JSON directly.
    Retrieves yesterday and today's local dates so we have ~48h of samples.
    """
    import asyncio
    from collections import Counter
    
    today_et = city_local_date(city)
    dt_today = datetime.fromisoformat(today_et)
    yesterday_et = (dt_today - timedelta(days=1)).strftime("%Y-%m-%d")

    urls = [
        wu_history_url(city, yesterday_et),
        wu_history_url(city, today_et)
    ]

    all_minutes = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10), headers=headers) as http:
            for url in urls:
                async with http.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        obs = data.get("observations", [])
                        for o in obs:
                            gmt = o.get("valid_time_gmt")
                            if gmt:
                                dt = datetime.fromtimestamp(int(gmt), tz=timezone.utc)
                                all_minutes.append(dt.minute)
    except Exception as e:
        log.warning("station_pattern_wu: failed to fetch for %s: %s", city.city_slug, e)
        return None

    if len(all_minutes) < 3:
        log.debug("station_pattern_wu: %s — only %d obs (<3), skipping", city.city_slug, len(all_minutes))
        return None

    minute_counts = Counter(all_minutes)
    
    # We expect roughly 48 samples if hourly
    # Just take the most common minute if it has strong presence
    best_minute, count = minute_counts.most_common(1)[0]
    
    samples = len(all_minutes)
    confidence = round(count / samples, 3)

    # Some stations are half-hourly (e.g. they report at :15 and :45)
    # Check if a second peak has substantial representation
    merged = _merge_adjacent_peaks(list(minute_counts.keys()), minute_counts)
    
    if len(merged) == 1:
        frequency = "hourly"
    elif len(merged) == 2:
        gap = abs(merged[0] - merged[1])
        gap = min(gap, 60 - gap)
        frequency = "half_hourly" if 25 <= gap <= 35 else "irregular"
    else:
        frequency = "irregular"

    # Merge again if there are multiple peaks, but use the same logic as the metar_obs one
    # Count distinct hours with data? We don't have hour tracking easily without parsing again,
    # but since this is WU, it's very reliable. If we get a single strong peak, it's hourly.
    
    # For WU, it's usually very clean. Let's just use the merged peaks that have at least 20% of samples.
    threshold = samples * 0.2
    raw_peaks = [m for m, count in minute_counts.items() if count >= threshold]
    if not raw_peaks:
        return None
        
    merged = _merge_adjacent_peaks(raw_peaks, minute_counts)
    
    if len(merged) == 1:
        frequency = "hourly"
    elif len(merged) == 2:
        gap = abs(merged[0] - merged[1])
        gap = min(gap, 60 - gap)
        frequency = "half_hourly" if 25 <= gap <= 35 else "irregular"
    else:
        frequency = "irregular"

    total_peak_hits = sum(
        1 for m in all_minutes
        if any(abs((m - peak) % 60) <= 1 or abs((peak - m) % 60) <= 1 for peak in merged)
    )
    confidence = round(total_peak_hits / samples, 3) if samples else 0.0

    result = {
        "minutes": sorted(merged),
        "frequency": frequency,
        "samples": samples,
        "confidence": confidence,
    }
    log.info("station_pattern_wu: %s → %s", city.city_slug, result)
    return result


def _merge_adjacent_peaks(peaks: list[int], counts: Counter) -> list[int]:
    """Merge peaks within ±1 of each other, keeping the highest-count minute."""
    if not peaks:
        return []

    sorted_peaks = sorted(peaks)
    merged = []
    used = set()

    for p in sorted_peaks:
        if p in used:
            continue

        # Find cluster: all peaks within ±1
        cluster = [p]
        for q in sorted_peaks:
            if q != p and q not in used:
                gap = min(abs(p - q), 60 - abs(p - q))
                if gap <= 1:
                    cluster.append(q)

        # Pick the one with highest count
        best = max(cluster, key=lambda m: counts[m])
        merged.append(best)
        used.update(cluster)

    return merged


async def refresh_all_station_profiles() -> None:
    """Run pattern detection for all active stations. Called daily or on startup."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    stations_processed = 0
    for city in cities:
        if not city.metar_station:
            continue

        result = await detect_observation_pattern(city)
        if result is None:
            log.debug("station_pattern: %s — no pattern detected", city.metar_station)
            continue

        async with get_session() as sess:
            await upsert_station_profile(
                sess,
                metar_station=city.metar_station,
                observation_minutes=json.dumps(result["minutes"]),
                observation_frequency=result["frequency"],
                samples_analyzed=result["samples"],
                confidence=result["confidence"],
            )
        stations_processed += 1

    log.info("station_pattern: refreshed %d station profiles", stations_processed)


async def refresh_missing_station_profiles() -> None:
    """Refresh profiles only for stations that don't have one yet."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        if not city.metar_station:
            continue

        async with get_session() as sess:
            existing = await get_station_profile(sess, city.metar_station)

        if existing is not None:
            continue

        result = await detect_observation_pattern(city)
        if result is None:
            continue

        async with get_session() as sess:
            await upsert_station_profile(
                sess,
                metar_station=city.metar_station,
                observation_minutes=json.dumps(result["minutes"]),
                observation_frequency=result["frequency"],
                samples_analyzed=result["samples"],
                confidence=result["confidence"],
            )
        log.info("station_pattern: created profile for %s", city.metar_station)
