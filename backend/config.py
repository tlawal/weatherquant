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
    KELLY_FRACTION: float = _float("KELLY_FRACTION", 0.10)
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

    # ── Cities (canonical registry; more discovered via Gamma) ────────────────
    from backend.city_registry import get_db_city_dicts as _get_db_city_dicts
    INITIAL_CITIES: list[dict] = _get_db_city_dicts()
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
