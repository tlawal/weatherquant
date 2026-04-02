"""
Canonical city registry — single source of truth for all city metadata.

Ordered by trading flow: US East → West → South America → Europe →
Middle East → South Asia → East Asia → Northeast Asia → Southeast Asia → Oceania.
Within each cluster, ordered by population/liquidity importance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# ── Registry ────────────────────────────────────────────────────────────────

CITY_REGISTRY: list[dict] = [
    # ── US East (UTC-5) ─────────────────────────────────────────────────────
    {"priority_order": 1, "cluster": "us_east", "utc_offset": -5,
     "city_slug": "atlanta", "display_name": "Atlanta",
     "nws_office": "FFC", "nws_grid_x": 49, "nws_grid_y": 81,
     "lat": 33.6367, "lon": -84.4279, "metar_station": "KATL",
     "wu_state": "ga", "wu_city": "atlanta",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/New_York"},

    {"priority_order": 2, "cluster": "us_east", "utc_offset": -5,
     "city_slug": "miami", "display_name": "Miami",
     "nws_office": "MFL", "nws_grid_x": 105, "nws_grid_y": 51,
     "lat": 25.7933, "lon": -80.2906, "metar_station": "KMIA",
     "wu_state": "fl", "wu_city": "miami",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/New_York"},

    {"priority_order": 3, "cluster": "us_east", "utc_offset": -5,
     "city_slug": "nyc", "display_name": "NYC",
     "nws_office": "OKX", "nws_grid_x": 37, "nws_grid_y": 39,
     "lat": 40.7772, "lon": -73.8726, "metar_station": "KLGA",
     "wu_state": "ny", "wu_city": "new-york-city",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/New_York"},

    {"priority_order": 4, "cluster": "us_east", "utc_offset": -5,
     "city_slug": "toronto", "display_name": "Toronto",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 43.6777, "lon": -79.6248, "metar_station": "CYYZ",
     "wu_state": "ca", "wu_city": "mississauga",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "America/Toronto"},

    # ── US Central (UTC-6) ──────────────────────────────────────────────────
    {"priority_order": 5, "cluster": "us_central", "utc_offset": -6,
     "city_slug": "chicago", "display_name": "Chicago",
     "nws_office": "LOT", "nws_grid_x": 65, "nws_grid_y": 76,
     "lat": 41.9786, "lon": -87.9048, "metar_station": "KORD",
     "wu_state": "il", "wu_city": "chicago",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Chicago"},

    {"priority_order": 6, "cluster": "us_central", "utc_offset": -6,
     "city_slug": "houston", "display_name": "Houston",
     "nws_office": "HGX", "nws_grid_x": 63, "nws_grid_y": 104,
     "lat": 29.9902, "lon": -95.3368, "metar_station": "KIAH",
     "wu_state": "tx", "wu_city": "houston",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Chicago"},

    {"priority_order": 7, "cluster": "us_central", "utc_offset": -6,
     "city_slug": "dallas", "display_name": "Dallas",
     "nws_office": "FWD", "nws_grid_x": 88, "nws_grid_y": 107,
     "lat": 32.8471, "lon": -96.8517, "metar_station": "KDAL",
     "wu_state": "tx", "wu_city": "dallas",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Chicago"},

    {"priority_order": 8, "cluster": "us_central", "utc_offset": -6,
     "city_slug": "austin", "display_name": "Austin",
     "nws_office": "EWX", "nws_grid_x": 158, "nws_grid_y": 87,
     "lat": 30.1975, "lon": -97.6664, "metar_station": "KAUS",
     "wu_state": "tx", "wu_city": "austin",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Chicago"},

    # ── US Mountain (UTC-7) ─────────────────────────────────────────────────
    {"priority_order": 9, "cluster": "us_mountain", "utc_offset": -7,
     "city_slug": "denver", "display_name": "Denver",
     "nws_office": "BOU", "nws_grid_x": 75, "nws_grid_y": 66,
     "lat": 39.8561, "lon": -104.6737, "metar_station": "KDEN",
     "wu_state": "co", "wu_city": "denver",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Denver"},

    # ── US West (UTC-8) ────────────────────────────────────────────────────
    {"priority_order": 10, "cluster": "us_west", "utc_offset": -8,
     "city_slug": "la", "display_name": "LA",
     "nws_office": "LOX", "nws_grid_x": 149, "nws_grid_y": 41,
     "lat": 33.9416, "lon": -118.4085, "metar_station": "KLAX",
     "wu_state": "ca", "wu_city": "los-angeles",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Los_Angeles"},

    {"priority_order": 11, "cluster": "us_west", "utc_offset": -8,
     "city_slug": "sf", "display_name": "SF",
     "nws_office": "MTR", "nws_grid_x": 85, "nws_grid_y": 98,
     "lat": 37.6213, "lon": -122.3790, "metar_station": "KSFO",
     "wu_state": "ca", "wu_city": "san-francisco",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Los_Angeles"},

    {"priority_order": 12, "cluster": "us_west", "utc_offset": -8,
     "city_slug": "seattle", "display_name": "Seattle",
     "nws_office": "SEW", "nws_grid_x": 124, "nws_grid_y": 60,
     "lat": 47.4502, "lon": -122.3088, "metar_station": "KSEA",
     "wu_state": "wa", "wu_city": "seatac",
     "enabled": 1, "is_us": 1, "unit": "F", "tz": "America/Los_Angeles"},

    # ── South America (UTC-3) ──────────────────────────────────────────────
    {"priority_order": 13, "cluster": "south_america", "utc_offset": -3,
     "city_slug": "sao-paulo", "display_name": "Sao Paulo",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": -23.4356, "lon": -46.4731, "metar_station": "SBGR",
     "wu_state": "br", "wu_city": "guarulhos",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "America/Sao_Paulo"},

    {"priority_order": 14, "cluster": "south_america", "utc_offset": -3,
     "city_slug": "buenos-aires", "display_name": "Buenos Aires",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": -34.8222, "lon": -58.5358, "metar_station": "SAEZ",
     "wu_state": "ar", "wu_city": "ezeiza",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "America/Argentina/Buenos_Aires"},

    # ── Europe (UTC+0 to UTC+1) ───────────────────────────────────────────
    {"priority_order": 15, "cluster": "europe", "utc_offset": 0,
     "city_slug": "london", "display_name": "London",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 51.5053, "lon": -0.0553, "metar_station": "EGLC",
     "wu_state": "gb", "wu_city": "london",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Europe/London"},

    {"priority_order": 16, "cluster": "europe", "utc_offset": 1,
     "city_slug": "paris", "display_name": "Paris",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 49.0097, "lon": 2.5479, "metar_station": "LFPG",
     "wu_state": "fr", "wu_city": "paris",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Europe/Paris"},

    {"priority_order": 17, "cluster": "europe", "utc_offset": 1,
     "city_slug": "madrid", "display_name": "Madrid",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 40.4719, "lon": -3.5626, "metar_station": "LEMD",
     "wu_state": "es", "wu_city": "madrid",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Europe/Madrid"},

    {"priority_order": 18, "cluster": "europe", "utc_offset": 1,
     "city_slug": "milan", "display_name": "Milan",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 45.6306, "lon": 8.7281, "metar_station": "LIMC",
     "wu_state": "it", "wu_city": "milan",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Europe/Rome"},

    {"priority_order": 19, "cluster": "europe", "utc_offset": 1,
     "city_slug": "munich", "display_name": "Munich",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 48.3538, "lon": 11.7861, "metar_station": "EDDM",
     "wu_state": "de", "wu_city": "munich",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Europe/Berlin"},

    {"priority_order": 20, "cluster": "europe", "utc_offset": 1,
     "city_slug": "warsaw", "display_name": "Warsaw",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 52.1657, "lon": 20.9671, "metar_station": "EPWA",
     "wu_state": "pl", "wu_city": "warsaw",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Europe/Warsaw"},

    # ── Middle East (UTC+2) ────────────────────────────────────────────────
    {"priority_order": 21, "cluster": "middle_east", "utc_offset": 2,
     "city_slug": "tel-aviv", "display_name": "Tel Aviv",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 32.0114, "lon": 34.8867, "metar_station": "LLBG",
     "wu_state": "il", "wu_city": "tel-aviv",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Jerusalem"},

    # ── South Asia (UTC+5:30) ──────────────────────────────────────────────
    {"priority_order": 22, "cluster": "south_asia", "utc_offset": 5,
     "city_slug": "lucknow", "display_name": "Lucknow",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 26.7606, "lon": 80.8893, "metar_station": "VILK",
     "wu_state": "in", "wu_city": "lucknow",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Kolkata"},

    # ── East Asia (UTC+8) ─────────────────────────────────────────────────
    {"priority_order": 23, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "shanghai", "display_name": "Shanghai",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 31.1443, "lon": 121.8083, "metar_station": "ZSPD",
     "wu_state": "cn", "wu_city": "shanghai",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Shanghai"},

    {"priority_order": 24, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "beijing", "display_name": "Beijing",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 40.0799, "lon": 116.6031, "metar_station": "ZBAA",
     "wu_state": "cn", "wu_city": "beijing",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Shanghai"},

    {"priority_order": 25, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "shenzhen", "display_name": "Shenzhen",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 22.6393, "lon": 113.8107, "metar_station": "ZGSZ",
     "wu_state": "cn", "wu_city": "shenzhen",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Shanghai"},

    {"priority_order": 26, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "chongqing", "display_name": "Chongqing",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 29.7192, "lon": 106.6416, "metar_station": "ZUCK",
     "wu_state": "cn", "wu_city": "chongqing",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Shanghai"},

    {"priority_order": 27, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "wuhan", "display_name": "Wuhan",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 30.7838, "lon": 114.2081, "metar_station": "ZHHH",
     "wu_state": "cn", "wu_city": "wuhan",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Shanghai"},

    {"priority_order": 28, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "chengdu", "display_name": "Chengdu",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 30.5785, "lon": 103.9471, "metar_station": "ZUUU",
     "wu_state": "cn", "wu_city": "chengdu",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Shanghai"},

    {"priority_order": 29, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "hong-kong", "display_name": "Hong Kong",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 22.3080, "lon": 113.9185, "metar_station": "VHHH",
     "wu_state": "hk", "wu_city": "hong-kong",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Hong_Kong"},

    {"priority_order": 30, "cluster": "east_asia", "utc_offset": 8,
     "city_slug": "taipei", "display_name": "Taipei",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 25.0797, "lon": 121.2342, "metar_station": "RCTP",
     "wu_state": "tw", "wu_city": "taipei",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Taipei"},

    # ── Northeast Asia (UTC+9) ─────────────────────────────────────────────
    {"priority_order": 31, "cluster": "northeast_asia", "utc_offset": 9,
     "city_slug": "seoul", "display_name": "Seoul",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 37.4602, "lon": 126.4407, "metar_station": "RKSI",
     "wu_state": "kr", "wu_city": "incheon",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Seoul"},

    {"priority_order": 32, "cluster": "northeast_asia", "utc_offset": 9,
     "city_slug": "tokyo", "display_name": "Tokyo",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 35.5494, "lon": 139.7798, "metar_station": "RJTT",
     "wu_state": "jp", "wu_city": "tokyo",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Tokyo"},

    # ── Southeast Asia (UTC+8) ─────────────────────────────────────────────
    {"priority_order": 33, "cluster": "southeast_asia", "utc_offset": 8,
     "city_slug": "singapore", "display_name": "Singapore",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": 1.3644, "lon": 103.9915, "metar_station": "WSSS",
     "wu_state": "sg", "wu_city": "singapore",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Asia/Singapore"},

    # ── Oceania (UTC+12) ──────────────────────────────────────────────────
    {"priority_order": 34, "cluster": "oceania", "utc_offset": 12,
     "city_slug": "wellington", "display_name": "Wellington",
     "nws_office": None, "nws_grid_x": None, "nws_grid_y": None,
     "lat": -41.3272, "lon": 174.8053, "metar_station": "NZWN",
     "wu_state": "nz", "wu_city": "wellington",
     "enabled": 0, "is_us": 0, "unit": "C", "tz": "Pacific/Auckland"},
]

# ── Derived lookups ─────────────────────────────────────────────────────────

CITY_REGISTRY_BY_SLUG: dict[str, dict] = {c["city_slug"]: c for c in CITY_REGISTRY}

_REGISTRY_ONLY_FIELDS = {"cluster", "priority_order", "utc_offset"}


def get_db_city_dicts() -> list[dict]:
    """Return city dicts with registry-only fields stripped, ready for DB upsert."""
    return [
        {k: v for k, v in c.items() if k not in _REGISTRY_ONLY_FIELDS}
        for c in CITY_REGISTRY
    ]


# ── Accessors ───────────────────────────────────────────────────────────────

def get_city_cluster(city_slug: str) -> Optional[str]:
    """Return the cluster for a city, or None if not in registry."""
    entry = CITY_REGISTRY_BY_SLUG.get(city_slug)
    return entry["cluster"] if entry else None


def get_city_priority(city_slug: str) -> int:
    """Return the priority order for a city (999 for unknown/discovered cities)."""
    entry = CITY_REGISTRY_BY_SLUG.get(city_slug)
    return entry["priority_order"] if entry else 999


def get_active_cities(current_utc_hour: int) -> list[dict]:
    """Return cities currently in their active trading window (6am-10pm local).

    Uses zoneinfo for DST-correct local hour computation.
    The current_utc_hour parameter is accepted for interface clarity
    but the actual computation uses the real current UTC time.
    """
    now_utc = datetime.now(timezone.utc)
    active = []
    for city in CITY_REGISTRY:
        local_hour = now_utc.astimezone(ZoneInfo(city["tz"])).hour
        if 6 <= local_hour < 22:
            active.append(city)
    return active
