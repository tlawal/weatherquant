"""
Polymarket Gamma event + bucket ingestion.

Discovers today's temperature events by slug, parses 8 YES/NO bucket markets,
verifies settlement source is Weather Underground, and persists to DB.

Also runs a weekly city discovery scan.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp

from backend.storage.db import get_session
from backend.storage.repos import (
    get_all_cities,
    get_city_by_slug,
    update_heartbeat,
    upsert_bucket,
    upsert_city,
    upsert_event,
)

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
ET = ZoneInfo("America/New_York")
_TIMEOUT = aiohttp.ClientTimeout(total=30)
_HEADERS = {"User-Agent": "WeatherQuant/1.0 (contact@weatherquant.local)"}

# Patterns to verify settlement source
_WU_SOURCE_PATTERNS = [
    re.compile(r"weather\s*underground", re.I),
    re.compile(r"wunderground", re.I),
    re.compile(r"wunder\.com", re.I),
]

# Regex to extract resolution URL and station ID
_URL_PATTERN = re.compile(r'https?://[^\s<>"\']+')
_NWS_SITE_PATTERN = re.compile(r'[?&]site=([A-Z0-9]{3,8})', re.I)

# Regex to parse bucket temperature ranges from market labels
_RANGE_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)\s*°?\s*[FCfc]?"),
    re.compile(r"(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)"),
]
_ABOVE_PATTERN = re.compile(r"(?:above|higher than|over|≥|>=)\s*(\d+\.?\d*)", re.I)
_BELOW_PATTERN = re.compile(r"(?:below|lower than|under|<)\s*(\d+\.?\d*)", re.I)


def _build_slugs(city_slug: str, target_date: date) -> list[str]:
    """Build possible Polymarket event slugs for a city/date."""
    month = target_date.strftime("%B").lower()
    day = target_date.day
    year = target_date.year
    
    slugs = [f"highest-temperature-in-{city_slug}-on-{month}-{day}-{year}"]
    
    # NYC fallbacks
    if city_slug == "new-york-city":
        slugs.append(f"highest-temperature-in-new-york-on-{month}-{day}-{year}")
        slugs.append(f"highest-temperature-in-nyc-on-{month}-{day}-{year}")
    elif city_slug == "la":
        slugs.insert(0, f"highest-temperature-in-los-angeles-on-{month}-{day}-{year}")
    elif city_slug == "sf":
        slugs.insert(0, f"highest-temperature-in-san-francisco-on-{month}-{day}-{year}")
    
    return slugs


def _extract_resolution_url(event_data: dict) -> tuple[str | None, str | None]:
    """Extract (resolution_source_url, resolution_station_id) from Gamma event data."""
    # Fields to search for URLs (in priority order)
    fields_to_check = [
        event_data.get("resolutionSource") or "",
        event_data.get("resolvedBy") or "",
        event_data.get("description") or "",
        event_data.get("overview") or "",
        event_data.get("question") or "",
    ]
    for text in fields_to_check:
        urls = _URL_PATTERN.findall(str(text))
        for url in urls:
            # Prefer NWS timeseries or similar station-based URLs
            m = _NWS_SITE_PATTERN.search(url)
            if m:
                return url, m.group(1).upper()
            # Also catch weather.gov station observation URLs
            station_m = re.search(r'weather\.gov/stations/([A-Z0-9]{3,8})', url, re.I)
            if station_m:
                return url, station_m.group(1).upper()
    # No structured URL found — return the plain text resolutionSource as URL if present
    plain = event_data.get("resolutionSource") or event_data.get("resolvedBy") or ""
    return (plain[:500] if plain else None), None


def _is_wu_source(source_text: str) -> bool:
    if not source_text:
        return False
    for patt in _WU_SOURCE_PATTERNS:
        if patt.search(source_text):
            return True
    return False


def _parse_bucket_range(label: str, description: str = "") -> tuple[Optional[float], Optional[float]]:
    """Parse (low, high) from a bucket market label. None = open-ended."""
    text = f"{label} {description}"

    # "above X"
    m = _ABOVE_PATTERN.search(text)
    if m:
        return float(m.group(1)), None

    # "below X"
    m = _BELOW_PATTERN.search(text)
    if m:
        return None, float(m.group(1))

    # "X - Y"
    for patt in _RANGE_PATTERNS:
        m = patt.search(text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi

    return None, None


def _validate_buckets(buckets: list[dict]) -> list[str]:
    """Return list of parse errors — empty means valid."""
    errors = []
    if len(buckets) < 3:
        errors.append(f"expected at least 3 buckets, got {len(buckets)}")
        return errors

    # Check monotonic ordering of boundaries
    last_hi = None
    for i, b in enumerate(buckets):
        lo, hi = b.get("low_f"), b.get("high_f")
        if i == 0 and lo is not None:
            if lo < 0 or lo > 150:
                errors.append(f"bucket 0 low_f={lo} out of plausible range")
        if last_hi is not None and lo is not None:
            if lo < last_hi - 0.1:
                errors.append(f"non-monotonic boundary at bucket {i}: lo={lo} < prev_hi={last_hi}")
        last_hi = hi

    return errors


async def fetch_gamma_all() -> None:
    """Discover/refresh today's events for all enabled cities."""
    from backend.tz_utils import city_local_date, et_today

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    et_date = et_today()
    city_dates = {}
    for city in cities:
        city_date = city_local_date(city)
        city_dates[city.city_slug] = city_date
        if city_date != et_date:
            log.info("gamma: %s local_date=%s differs from et=%s (tz=%s)",
                     city.city_slug, city_date, et_date, getattr(city, "tz", "?"))

    results = await asyncio.gather(
        *[_fetch_city_event(city, city_dates[city.city_slug]) for city in cities],
        return_exceptions=True,
    )

    for city, result in zip(cities, results):
        if isinstance(result, Exception):
            log.error("gamma: %s error: %s", city.city_slug, result)

    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_gamma", success=True)


async def _fetch_city_event(city, date_str: str) -> None:
    from datetime import date as date_type
    target_date = date_type.fromisoformat(date_str)
    slugs = _build_slugs(city.city_slug, target_date)
    
    event_data = None
    final_slug = slugs[0]

    async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_HEADERS) as http:
        for slug in slugs:
            log.debug("gamma: fetching slug=%s", slug)
            url = f"{GAMMA_API}/events/slug/{slug}"
            try:
                async with http.get(url) as resp:
                    if resp.status == 200:
                        event_data = await resp.json(content_type=None)
                        final_slug = slug
                        break
                    elif resp.status == 404:
                        continue
                    else:
                        log.error("gamma: HTTP %d for %s", resp.status, url)
            except Exception as e:
                log.error("gamma: fetch failed for %s: %s", slug, e)
                continue

    if not event_data:
        log.info("gamma: no event found for city=%s after trying %d slugs", city.city_slug, len(slugs))
        async with get_session() as sess:
            await upsert_event(
                sess,
                city_id=city.id,
                date_et=date_str,
                gamma_slug=slugs[0],
                status="no_event",
                trading_enabled=False,
            )
        return

    await _process_event_data(city, date_str, final_slug, event_data)


async def _process_event_data(city, date_et: str, slug: str, event_data: dict) -> None:
    """Parse event data, validate, and persist buckets."""
    gamma_event_id = str(event_data.get("id") or event_data.get("conditionId") or "")

    # Verify settlement source
    settlement_source = (
        event_data.get("resolutionSource")
        or event_data.get("resolvedBy")
        or event_data.get("description")
        or ""
    )
    # Also check nested fields
    for key in ["description", "resolutionSource", "question", "overview"]:
        text = str(event_data.get(key) or "")
        if _is_wu_source(text):
            settlement_source = text
            break

    wu_verified = _is_wu_source(settlement_source)
    resolution_source_url, resolution_station_id = _extract_resolution_url(event_data)

    markets = event_data.get("markets") or []
    log.info(
        "gamma: %s event=%s markets=%d wu_verified=%s",
        city.city_slug,
        gamma_event_id,
        len(markets),
        wu_verified,
    )

    # Parse buckets from markets
    raw_buckets: list[dict] = []
    for i, market in enumerate(markets):
        label = (
            market.get("outcomePrices")
            or market.get("question")
            or market.get("groupItemTitle")
            or market.get("title")
            or ""
        )
        if isinstance(label, list):
            label = " ".join(str(x) for x in label)
        description = str(market.get("description") or "")
        question = str(market.get("question") or market.get("title") or "")

        lo, hi = _parse_bucket_range(question, description)
        if lo is None and hi is None:
            # Try the label field
            lo, hi = _parse_bucket_range(str(label), "")

        import json
        
        # Extract YES/NO token IDs
        yes_token = None
        no_token = None
        tokens = market.get("clobTokenIds") or []
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except Exception:
                tokens = []
                
        if isinstance(tokens, list) and len(tokens) >= 2:
            yes_token, no_token = str(tokens[0]), str(tokens[1])
        elif isinstance(tokens, dict):
            yes_token = str(tokens.get("1") or "")
            no_token = str(tokens.get("0") or "")

        # Fallback to outcomes array
        outcomes = market.get("outcomes") or []
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = []
                
        if isinstance(outcomes, list):
            for outcome in outcomes:
                if isinstance(outcome, dict):
                    name = str(outcome.get("name") or "").upper()
                    tid = str(outcome.get("clobTokenId") or "")
                    if name == "YES" and tid:
                        yes_token = tid
                    elif name == "NO" and tid:
                        no_token = tid

        if lo is None and hi is None:
            log.debug("gamma: skipping non-bucket market '%s'", label)
            continue

        raw_buckets.append(
            {
                "bucket_idx": len(raw_buckets),  # Use length so indices are sequential
                "label": str(question or label)[:256],
                "low_f": lo,
                "high_f": hi,
                "yes_token_id": yes_token,
                "no_token_id": no_token,
                "condition_id": str(market.get("conditionId") or ""),
            }
        )

    parse_errors = _validate_buckets(raw_buckets)
    if parse_errors:
        log.warning("gamma: %s bucket parse errors: %s", city.city_slug, parse_errors)
        status = "bad_buckets"
        trading_enabled = False
    else:
        status = "ok"
        trading_enabled = wu_verified

    async with get_session() as sess:
        event = await upsert_event(
            sess,
            city_id=city.id,
            date_et=date_et,
            gamma_event_id=gamma_event_id,
            gamma_slug=slug,
            settlement_source=settlement_source[:256] if settlement_source else None,
            settlement_source_verified=wu_verified,
            resolution_source_url=resolution_source_url,
            resolution_station_id=resolution_station_id,
            status=status,
            trading_enabled=trading_enabled,
        )

        for b in raw_buckets:
            await upsert_bucket(sess, event_id=event.id, **b)

    log.info(
        "gamma: %s status=%s buckets=%d trading_enabled=%s",
        city.city_slug,
        status,
        len(raw_buckets),
        trading_enabled,
    )


async def discover_cities() -> None:
    """Search Gamma for all 'highest temperature in' events to build city registry."""
    log.info("gamma: running city discovery scan")
    url = f"{GAMMA_API}/events?limit=100&active=true"
    # Also try the public search endpoint
    search_url = f"{GAMMA_API}/events?limit=200&slug_search=highest-temperature-in"

    found_cities: dict[str, tuple[bool, str]] = {}

    for endpoint in [search_url, url]:
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_HEADERS) as http:
                async with http.get(endpoint) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json(content_type=None)

            events = data if isinstance(data, list) else (data.get("events") or [])
            for ev in events:
                slug = str(ev.get("slug") or "")
                if "highest-temperature-in-" in slug:
                    # Detect unit
                    title = str(ev.get("title") or ev.get("question") or "")
                    desc = str(ev.get("description") or "")
                    
                    is_us = True
                    unit = "F"
                    if "°C" in title or "°C" in desc or "Celsius" in desc:
                        is_us = False
                        unit = "C"
                    
                    found_cities[slug] = (is_us, unit)
        except Exception as e:
            log.warning("gamma discover: %s error: %s", endpoint, e)

    # Parse city slugs from event slugs
    slug_pattern = re.compile(r"highest-temperature-in-([a-z0-9-]+)-on-")
    for slug, (is_us, unit) in found_cities.items():
        m = slug_pattern.search(slug)
        if not m:
            continue
        city_slug = m.group(1)

        async with get_session() as sess:
            existing = await get_city_by_slug(sess, city_slug)
            if existing is None:
                # Add as discovered but disabled
                display = city_slug.replace("-", " ").title()
                await upsert_city(
                    sess,
                    {
                        "city_slug": city_slug,
                        "display_name": display,
                        "enabled": False,
                        "is_us": is_us,
                        "unit": unit,
                    },
                )
                log.info("gamma: discovered new city: %s (is_us=%s unit=%s disabled)", city_slug, is_us, unit)

    log.info("gamma: city discovery found %d temp-market slugs", len(found_cities))
