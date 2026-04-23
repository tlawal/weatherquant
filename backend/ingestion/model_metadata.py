"""Fetch authoritative model initialization metadata from NWP APIs.

Open-Meteo metadata endpoints return Unix timestamps for:
  - last_run_initialisation_time: model initialization / reference time
  - last_run_availability_time: when data is accessible on API servers
  - last_run_modification_time: when download/conversion finished
  - temporal_resolution_seconds: native model output interval
  - update_interval_seconds: typical time between model updates

Note: Open-Meteo operates redundant servers with eventual consistency.
The docs recommend waiting an additional 10 minutes after availability_time
before relying on the newest data.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)
_OM_TIMEOUT = aiohttp.ClientTimeout(total=10)

# Mapping from WeatherQuant source_key -> Open-Meteo metadata endpoint slug
_OM_META_ENDPOINTS = {
    "ecmwf_ifs":  "https://api.open-meteo.com/data/ecmwf_ifs025/static/meta.json",
    "nbm":        "https://api.open-meteo.com/data/ncep_nbm_conus/static/meta.json",
    "hrrr":       "https://api.open-meteo.com/data/ncep_hrrr_conus/static/meta.json",
    "hrrr_15min": "https://api.open-meteo.com/data/ncep_hrrr_conus_15min/static/meta.json",
}

# Simple TTL cache: key -> (cached_at_utc, value)
_meta_cache: dict[str, tuple[datetime, dict]] = {}
_META_TTL_SECONDS = 300  # 5 minutes


async def fetch_openmeteo_metadata(source_key: str) -> dict | None:
    """Fetch Open-Meteo metadata for a model. Returns dict or None on failure.

    Cached for 5 minutes to avoid hammering the free metadata endpoints.
    """
    url = _OM_META_ENDPOINTS.get(source_key)
    if not url:
        return None

    now = datetime.now(timezone.utc)
    cached = _meta_cache.get(source_key)
    if cached and (now - cached[0]).total_seconds() < _META_TTL_SECONDS:
        return cached[1]

    try:
        async with aiohttp.ClientSession(timeout=_OM_TIMEOUT) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.warning("om-meta %s: HTTP %d", source_key, resp.status)
                    return None
                data = await resp.json()
    except Exception as e:
        log.warning("om-meta %s: failed %s", source_key, e)
        return None

    # Validate expected fields
    if not isinstance(data, dict) or "last_run_initialisation_time" not in data:
        log.warning("om-meta %s: missing init time in response", source_key)
        return None

    result = {
        "last_run_initialisation_time": data.get("last_run_initialisation_time"),
        "last_run_availability_time": data.get("last_run_availability_time"),
        "last_run_modification_time": data.get("last_run_modification_time"),
        "temporal_resolution_seconds": data.get("temporal_resolution_seconds"),
        "update_interval_seconds": data.get("update_interval_seconds"),
        "data_end_time": data.get("data_end_time"),
    }
    _meta_cache[source_key] = (now, result)
    return result


def clear_meta_cache() -> None:
    """Clear the metadata cache (useful for testing)."""
    _meta_cache.clear()


# ── NWS metadata helpers ────────────────────────────────────────────────────

def parse_nws_update_time(data: dict) -> datetime | None:
    """Parse the NWS gridpoint forecast updateTime as authoritative model_run_at.

    NWS response properties contains:
      - generatedAt: when this API response was generated
      - updateTime:  when the forecast content was last updated (closer to model/human run)
    We use updateTime as the model_run_at proxy.
    """
    props = data.get("properties") or {}
    update_time_str = props.get("updateTime")
    if not update_time_str:
        return None
    try:
        # NWS returns ISO 8601 with timezone, e.g. "2026-04-23T09:06:15+00:00"
        dt = datetime.fromisoformat(update_time_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        log.warning("nws: failed to parse updateTime %r: %s", update_time_str, e)
        return None
