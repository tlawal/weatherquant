"""
WeatherQuant — centralized configuration from environment variables.

All trading safety parameters are here. Changing them requires restart.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


class Config:
    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "sqlite+aiosqlite:////tmp/weatherquant.db",
    )

    # ── Service mode ─────────────────────────────────────────────────────────
    SERVICE_TYPE: str = os.environ.get("SERVICE_TYPE", "api").lower()

    # ── Polymarket CLOB ───────────────────────────────────────────────────────
    POLYMARKET_PRIVATE_KEY: str = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    FUNDER_ADDRESS: str = os.environ.get("FUNDER_ADDRESS", "")
    POLYMARKET_HOST: str = os.environ.get(
        "POLYMARKET_HOST", "https://clob.polymarket.com"
    )
    CHAIN_ID: int = _int("CHAIN_ID", 137)

    # ── Security ──────────────────────────────────────────────────────────────
    ADMIN_TOKEN: str = os.environ.get("ADMIN_TOKEN", "")
    ARMING_SECRET: str = os.environ.get("ARMING_SECRET", "")
    ARMING_TOKEN_TTL_SECONDS: int = _int("ARMING_TOKEN_TTL_SECONDS", 60)

    # ── Trading Safety ────────────────────────────────────────────────────────
    AUTO_TRADE_DEFAULT: bool = _bool("AUTO_TRADE_DEFAULT", default=False)
    BANKROLL_CAP: float = _float("BANKROLL_CAP", 10.0)
    MAX_POSITION_PCT: float = _float("MAX_POSITION_PCT", 0.10)
    MAX_DAILY_LOSS: float = _float("MAX_DAILY_LOSS", 3.0)  # positive number, max loss
    MIN_TRUE_EDGE: float = _float("MIN_TRUE_EDGE", 0.10)
    MIN_LIQUIDITY_SHARES: float = _float("MIN_LIQUIDITY_SHARES", 10.0)
    MAX_POSITIONS_PER_EVENT: int = _int("MAX_POSITIONS_PER_EVENT", 2)
    MAX_LIQUIDITY_PCT: float = _float("MAX_LIQUIDITY_PCT", 0.20)

    # ── Trading Window (ET hours) ─────────────────────────────────────────────
    TRADING_WINDOW_CLOSE_ET: int = _int("TRADING_WINDOW_CLOSE_ET", 19)
    TRADING_FORCE_DISABLE_ET: int = _int("TRADING_FORCE_DISABLE_ET", 21)

    # ── Data Freshness TTLs (seconds) ─────────────────────────────────────────
    WU_STALE_TTL_SECONDS: int = _int("WU_STALE_TTL_SECONDS", 7200)  # 2 hours
    METAR_STALE_TTL_SECONDS: int = _int("METAR_STALE_TTL_SECONDS", 300)  # 5 min

    # ── API Rate Limiting ─────────────────────────────────────────────────────
    WU_MIN_SCRAPE_INTERVAL_SECONDS: int = _int("WU_MIN_SCRAPE_INTERVAL_SECONDS", 300)

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_JSON: bool = _bool("LOG_JSON", default=True)

    # ── Cities (initial set; more discovered via Gamma) ───────────────────────
    INITIAL_CITIES: list[dict] = [
        {'city_slug': 'atlanta', 'display_name': 'Atlanta', 'metar_station': 'KATL', 'wu_state': 'ga', 'wu_city': 'atlanta', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'miami', 'display_name': 'Miami', 'metar_station': 'KMIA', 'wu_state': 'fl', 'wu_city': 'miami', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'nyc', 'display_name': 'NYC', 'metar_station': 'KLGA', 'wu_state': 'ny', 'wu_city': 'new-york-city', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'chicago', 'display_name': 'Chicago', 'metar_station': 'KORD', 'wu_state': 'il', 'wu_city': 'chicago', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'dallas', 'display_name': 'Dallas', 'metar_station': 'KDAL', 'wu_state': 'tx', 'wu_city': 'dallas', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'seattle', 'display_name': 'Seattle', 'metar_station': 'KSEA', 'wu_state': 'wa', 'wu_city': 'seatac', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'sf', 'display_name': 'SF', 'metar_station': 'KSFO', 'wu_state': 'ca', 'wu_city': 'san-francisco', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'la', 'display_name': 'LA', 'metar_station': 'KLAX', 'wu_state': 'ca', 'wu_city': 'los-angeles', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'austin', 'display_name': 'Austin', 'metar_station': 'KAUS', 'wu_state': 'tx', 'wu_city': 'austin', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'denver', 'display_name': 'Denver', 'metar_station': 'KDEN', 'wu_state': 'co', 'wu_city': 'denver', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'houston', 'display_name': 'Houston', 'metar_station': 'KIAH', 'wu_state': 'tx', 'wu_city': 'houston', 'enabled': True, 'is_us': True, 'unit': 'F'},
        {'city_slug': 'london', 'display_name': 'London', 'metar_station': 'EGLC', 'wu_state': 'gb', 'wu_city': 'london', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'paris', 'display_name': 'Paris', 'metar_station': 'LFPG', 'wu_state': 'fr', 'wu_city': 'paris', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'madrid', 'display_name': 'Madrid', 'metar_station': 'LEMD', 'wu_state': 'es', 'wu_city': 'madrid', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'munich', 'display_name': 'Munich', 'metar_station': 'EDDM', 'wu_state': 'de', 'wu_city': 'munich', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'warsaw', 'display_name': 'Warsaw', 'metar_station': 'EPWA', 'wu_state': 'pl', 'wu_city': 'warsaw', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'milan', 'display_name': 'Milan', 'metar_station': 'LIMC', 'wu_state': 'it', 'wu_city': 'milan', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'shanghai', 'display_name': 'Shanghai', 'metar_station': 'ZSPD', 'wu_state': 'cn', 'wu_city': 'shanghai', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'beijing', 'display_name': 'Beijing', 'metar_station': 'ZBAA', 'wu_state': 'cn', 'wu_city': 'beijing', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'shenzhen', 'display_name': 'Shenzhen', 'metar_station': 'ZGSZ', 'wu_state': 'cn', 'wu_city': 'shenzhen', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'chongqing', 'display_name': 'Chongqing', 'metar_station': 'ZUCK', 'wu_state': 'cn', 'wu_city': 'chongqing', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'wuhan', 'display_name': 'Wuhan', 'metar_station': 'ZHHH', 'wu_state': 'cn', 'wu_city': 'wuhan', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'chengdu', 'display_name': 'Chengdu', 'metar_station': 'ZUUU', 'wu_state': 'cn', 'wu_city': 'chengdu', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'seoul', 'display_name': 'Seoul', 'metar_station': 'RKSI', 'wu_state': 'kr', 'wu_city': 'incheon', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'tokyo', 'display_name': 'Tokyo', 'metar_station': 'RJTT', 'wu_state': 'jp', 'wu_city': 'tokyo', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'taipei', 'display_name': 'Taipei', 'metar_station': 'RCTP', 'wu_state': 'tw', 'wu_city': 'taipei', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'hong-kong', 'display_name': 'Hong Kong', 'metar_station': 'VHHH', 'wu_state': 'hk', 'wu_city': 'hong-kong', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'singapore', 'display_name': 'Singapore', 'metar_station': 'WSSS', 'wu_state': 'sg', 'wu_city': 'singapore', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'toronto', 'display_name': 'Toronto', 'metar_station': 'CYYZ', 'wu_state': 'ca', 'wu_city': 'mississauga', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'sao-paulo', 'display_name': 'Sao Paulo', 'metar_station': 'SBGR', 'wu_state': 'br', 'wu_city': 'guarulhos', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'buenos-aires', 'display_name': 'Buenos Aires', 'metar_station': 'SAEZ', 'wu_state': 'ar', 'wu_city': 'ezeiza', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'wellington', 'display_name': 'Wellington', 'metar_station': 'NZWN', 'wu_state': 'nz', 'wu_city': 'wellington', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'tel-aviv', 'display_name': 'Tel Aviv', 'metar_station': 'LLBG', 'wu_state': 'il', 'wu_city': 'tel-aviv', 'enabled': True, 'is_us': False, 'unit': 'C'},
        {'city_slug': 'lucknow', 'display_name': 'Lucknow', 'metar_station': 'VILK', 'wu_state': 'in', 'wu_city': 'lucknow', 'enabled': True, 'is_us': False, 'unit': 'C'}
    ]

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of critical missing config warnings."""
        warnings = []
        if not cls.POLYMARKET_PRIVATE_KEY:
            warnings.append("POLYMARKET_PRIVATE_KEY not set — trading disabled")
        if not cls.ADMIN_TOKEN:
            warnings.append("ADMIN_TOKEN not set — all write endpoints unauthenticated!")
        if not cls.ARMING_SECRET:
            warnings.append("ARMING_SECRET not set — arming disabled")
        if cls.BANKROLL_CAP > 100:
            warnings.append(f"BANKROLL_CAP={cls.BANKROLL_CAP} exceeds $100 safety limit!")
        return warnings
