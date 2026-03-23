"""
Timezone utilities — single source of truth for per-city date computation.

Every date operation in the system flows through this module:
  - Per-city operations → city_local_date(city)
  - Global operations (PnL, arming) → et_today()
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def city_local_date(city) -> str:
    """Return YYYY-MM-DD in the city's local timezone.

    This is THE canonical function for computing what 'today' means for a city.
    Used for: event keying, forecast storage, METAR daily high, slug generation.
    """
    tz = ZoneInfo(city.tz) if hasattr(city, "tz") and city.tz else ET
    return datetime.now(tz).strftime("%Y-%m-%d")


def city_local_now(city) -> datetime:
    """Return current tz-aware datetime in the city's local timezone."""
    tz = ZoneInfo(city.tz) if hasattr(city, "tz") and city.tz else ET
    return datetime.now(tz)


def city_local_tomorrow(city) -> str:
    """Return tomorrow's YYYY-MM-DD in the city's local timezone."""
    tz = ZoneInfo(city.tz) if hasattr(city, "tz") and city.tz else ET
    return (datetime.now(tz) + timedelta(days=1)).strftime("%Y-%m-%d")


def et_today() -> str:
    """Return YYYY-MM-DD in ET — for global operations (daily PnL, arming)."""
    return datetime.now(ET).strftime("%Y-%m-%d")


def et_now() -> datetime:
    """Return current tz-aware datetime in ET."""
    return datetime.now(ET)
