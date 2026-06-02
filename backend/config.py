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
    PROXY_ADDRESS: str = os.environ.get("PROXY_ADDRESS", "")  # Preferred over FUNDER_ADDRESS for on-chain queries
    FUNDER_ADDRESS: str = os.environ.get("FUNDER_ADDRESS", "")
    POLYMARKET_HOST: str = os.environ.get(
        "POLYMARKET_HOST", "https://clob.polymarket.com"
    )
    CHAIN_ID: int = _int("CHAIN_ID", 137)
    POLYGON_RPC_URL: str = os.environ.get(
        "POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com"
    )

    # ── Security ──────────────────────────────────────────────────────────────
    ADMIN_TOKEN: str = os.environ.get("ADMIN_TOKEN", "")
    ARMING_SECRET: str = os.environ.get("ARMING_SECRET", "")
    ARMING_TOKEN_TTL_SECONDS: int = _int("ARMING_TOKEN_TTL_SECONDS", 60)

    # ── Trading Safety ────────────────────────────────────────────────────────
    AUTO_TRADE_DEFAULT: bool = _bool("AUTO_TRADE_DEFAULT", default=False)
    BANKROLL_CAP: float = _float("BANKROLL_CAP", 10.0)
    MAX_POSITION_PCT: float = _float("MAX_POSITION_PCT", 0.10)
    KELLY_FRACTION: float = _float("KELLY_FRACTION", 0.10)
    MAX_DAILY_LOSS: float = _float("MAX_DAILY_LOSS", 5.0)  # positive number, max loss
    MIN_TRUE_EDGE: float = _float("MIN_TRUE_EDGE", 0.10)
    MIN_ORDER_NOTIONAL_DOLLARS: float = _float("MIN_ORDER_NOTIONAL_DOLLARS", 1.0)
    MIN_NOTIONAL_BUMP_MAX_KELLY_MULTIPLE: float = _float("MIN_NOTIONAL_BUMP_MAX_KELLY_MULTIPLE", 3.0)
    MIN_LIQUIDITY_SHARES: float = _float("MIN_LIQUIDITY_SHARES", 10.0)
    MAX_POSITIONS_PER_EVENT: int = _int("MAX_POSITIONS_PER_EVENT", 2)
    MAX_LIQUIDITY_PCT: float = _float("MAX_LIQUIDITY_PCT", 0.20)
    MAX_ENTRY_PRICE: float = _float("MAX_ENTRY_PRICE", 0.36)
    MAX_SPREAD: float = _float("MAX_SPREAD", 0.04)
    MIN_ORDERBOOK_DEPTH_DOLLARS: float = _float("MIN_ORDERBOOK_DEPTH_DOLLARS", 2000.0)

    # ── Exit Engine ──────────────────────────────────────────────────────────
    QUICK_FLIP_TARGET: float = _float("QUICK_FLIP_TARGET", 0.08)
    URGENT_EXIT_MAX_SPREAD: float = _float("URGENT_EXIT_MAX_SPREAD", 0.06)
    CONSENSUS_DEBOUNCE_RUNS: int = _int("CONSENSUS_DEBOUNCE_RUNS", 2)
    EXPIRY_DISCOUNT: float = _float("EXPIRY_DISCOUNT", 0.10)
    EXPIRY_WINNER_HOLD_MIN_PROB: float = _float("EXPIRY_WINNER_HOLD_MIN_PROB", 0.85)
    EXPIRY_PASSIVE_SELL_MIN_BID: float = _float("EXPIRY_PASSIVE_SELL_MIN_BID", 0.99)
    EXPIRY_MARKET_WIN_MIN_BID: float = _float("EXPIRY_MARKET_WIN_MIN_BID", 0.98)
    EXPIRY_RISK_EXIT_MAX_DISCOUNT: float = _float("EXPIRY_RISK_EXIT_MAX_DISCOUNT", 0.02)
    # URGENT exit anti-whipsaw gates (added April 2026 after Atlanta shakeout)
    URGENT_MIN_POSITION_AGE_SECONDS: int = _int("URGENT_MIN_POSITION_AGE_SECONDS", 3600)
    URGENT_MIN_EXIT_MODEL_PROB: float = _float("URGENT_MIN_EXIT_MODEL_PROB", 0.15)
    URGENT_MIN_BID_DEPTH: float = _float("URGENT_MIN_BID_DEPTH", 5.0)
    URGENT_ADJACENT_DEBOUNCE_MULTIPLIER: int = _int("URGENT_ADJACENT_DEBOUNCE_MULTIPLIER", 2)
    EXIT_MARKET_SELL_MAX_SPREAD: float = _float("EXIT_MARKET_SELL_MAX_SPREAD", 0.06)
    # EDGE_DECAY exit (Phase A2 — EV-based exit gate, fires before URGENT).
    # Exit when ev_at_bid stays at or below threshold for N consecutive runs.
    EDGE_DECAY_THRESHOLD: float = _float("EDGE_DECAY_THRESHOLD", -0.005)
    EDGE_DECAY_DEBOUNCE_RUNS: int = _int("EDGE_DECAY_DEBOUNCE_RUNS", 3)
    EDGE_DECAY_MIN_POSITION_AGE_SECONDS: int = _int("EDGE_DECAY_MIN_POSITION_AGE_SECONDS", 1800)
    EDGE_DECAY_ENTRY_MIN_EV: float = _float("EDGE_DECAY_ENTRY_MIN_EV", 0.01)
    EDGE_DECAY_MIN_EV_DROP: float = _float("EDGE_DECAY_MIN_EV_DROP", 0.03)
    EDGE_DECAY_REQUIRE_MODEL_DETERIORATION: bool = _bool("EDGE_DECAY_REQUIRE_MODEL_DETERIORATION", default=True)
    EDGE_DECAY_MIN_MODEL_PROB_DROP: float = _float("EDGE_DECAY_MIN_MODEL_PROB_DROP", 0.03)
    EDGE_DECAY_MIN_SOURCE_TEMP_DETERIORATION_F: float = _float("EDGE_DECAY_MIN_SOURCE_TEMP_DETERIORATION_F", 0.75)
    MANUAL_EDGE_DECAY_MIN_AGE_SECONDS: int = _int("MANUAL_EDGE_DECAY_MIN_AGE_SECONDS", 3600)
    EDGE_DECAY_MIN_BID: float = _float("EDGE_DECAY_MIN_BID", 0.03)
    EDGE_DECAY_HISTORY_KEEP: int = _int("EDGE_DECAY_HISTORY_KEEP", 10)
    # OBS_PROXIMITY exit: pre-observation profit protection for fragile
    # daily-high buckets near scheduled station observation minutes.
    OBS_EXIT_ENABLED: bool = _bool("OBS_EXIT_ENABLED", default=True)
    OBS_EXIT_WINDOW_MINUTES: int = _int("OBS_EXIT_WINDOW_MINUTES", 20)
    TEMP_SENSITIVITY_THRESHOLD_F: float = _float("TEMP_SENSITIVITY_THRESHOLD_F", 1.0)
    OBS_MIN_PROFIT_CENTS: float = _float("OBS_MIN_PROFIT_CENTS", 5.0)
    OBS_REENTRY_COOLDOWN_MINUTES: int = _int("OBS_REENTRY_COOLDOWN_MINUTES", 5)
    OBS_MIN_DEPTH_USD: float = _float("OBS_MIN_DEPTH_USD", 100.0)
    OBS_MAX_ORDERBOOK_IMBALANCE: float = _float("OBS_MAX_ORDERBOOK_IMBALANCE", 0.72)

    # ── Portfolio Risk ───────────────────────────────────────────────────────
    MAX_DRAWDOWN_PCT: float = _float("MAX_DRAWDOWN_PCT", 0.25)
    MAX_CLUSTER_EXPOSURE_PCT: float = _float("MAX_CLUSTER_EXPOSURE_PCT", 0.60)
    MAX_STRATEGY_LOSS: float = _float("MAX_STRATEGY_LOSS", 3.0)

    # ── Telegram Notifications ───────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ENABLED: bool = _bool("TELEGRAM_ENABLED", default=False)

    # ── Trading Window (ET hours) ─────────────────────────────────────────────
    TRADING_WINDOW_CLOSE_ET: int = _int("TRADING_WINDOW_CLOSE_ET", 19)
    TRADING_FORCE_DISABLE_ET: int = _int("TRADING_FORCE_DISABLE_ET", 21)

    # ── Data Freshness TTLs (seconds) ─────────────────────────────────────────
    WU_STALE_TTL_SECONDS: int = _int("WU_STALE_TTL_SECONDS", 7200)  # 2 hours
    METAR_STALE_TTL_SECONDS: int = _int("METAR_STALE_TTL_SECONDS", 300)  # 5 min

    # ── Settlement High Source ────────────────────────────────────────────────
    # "tgftp" (default) uses TGFTP METAR as primary; "wu_history" falls back
    # to the old WU history API. Flip via env var for safe rollback.
    SETTLEMENT_HIGH_PRIMARY: str = os.environ.get("SETTLEMENT_HIGH_PRIMARY", "tgftp").strip().lower()

    # ── API Rate Limiting ─────────────────────────────────────────────────────
    WU_MIN_SCRAPE_INTERVAL_SECONDS: int = _int("WU_MIN_SCRAPE_INTERVAL_SECONDS", 1800)
    WU_FAILED_RETRY_INTERVAL_SECONDS: int = _int("WU_FAILED_RETRY_INTERVAL_SECONDS", 300)
    GAMMA_FETCH_CONCURRENCY: int = _int("GAMMA_FETCH_CONCURRENCY", 6)

    # ── Market Context LLM ────────────────────────────────────────────────────
    MARKET_CONTEXT_LLM_PROVIDER: str = os.environ.get(
        "MARKET_CONTEXT_LLM_PROVIDER", ""
    ).strip().lower()
    MARKET_CONTEXT_LLM_MODEL: str = os.environ.get(
        "MARKET_CONTEXT_LLM_MODEL", ""
    ).strip()
    MARKET_CONTEXT_LLM_API_KEY: str = os.environ.get(
        "MARKET_CONTEXT_LLM_API_KEY", ""
    ).strip()
    MARKET_CONTEXT_LLM_BASE_URL: str = os.environ.get(
        "MARKET_CONTEXT_LLM_BASE_URL", ""
    ).strip()
    MARKET_CONTEXT_LLM_TIMEOUT_SECONDS: int = _int(
        "MARKET_CONTEXT_LLM_TIMEOUT_SECONDS", 45
    )

    # ── Wallet Tracker (read-only public-market analytics) ───────────────────
    WALLET_TRACKER_ENABLED: bool = _bool("WALLET_TRACKER_ENABLED", default=False)
    WALLET_TRACKER_START_CITY: str = os.environ.get("WALLET_TRACKER_START_CITY", "all").strip().lower()
    WALLET_TRACKER_UPDATE_INTERVAL_MINUTES: int = _int("WALLET_TRACKER_UPDATE_INTERVAL_MINUTES", 60)
    WALLET_TRACKER_MIN_VOLUME_USD: float = _float("WALLET_TRACKER_MIN_VOLUME_USD", 100.0)
    WALLET_TRACKER_MIN_TRADES: int = _int("WALLET_TRACKER_MIN_TRADES", 3)
    WALLET_TRACKER_MIN_ACTIVE_DAYS: int = _int("WALLET_TRACKER_MIN_ACTIVE_DAYS", 2)
    WALLET_TRACKER_MAX_WALLETS_PER_CITY: int = _int("WALLET_TRACKER_MAX_WALLETS_PER_CITY", 10)
    WALLET_TRACKER_DISPLAY_LIMIT: int = _int("WALLET_TRACKER_DISPLAY_LIMIT", 50)
    WALLET_TRACKER_SKILL_WINDOW_DAYS: int = _int("WALLET_TRACKER_SKILL_WINDOW_DAYS", 90)
    WALLET_TRACKER_MIN_RESOLVED_MARKETS: int = _int("WALLET_TRACKER_MIN_RESOLVED_MARKETS", 3)
    WALLET_TRACKER_MIN_ADJUSTED_SCORE: float = _float("WALLET_TRACKER_MIN_ADJUSTED_SCORE", 0.20)
    WALLET_TRACKER_TRUNCATE_ADDRESSES: bool = _bool("WALLET_TRACKER_TRUNCATE_ADDRESSES", default=True)
    WALLET_TRACKER_READ_ONLY: bool = _bool("WALLET_TRACKER_READ_ONLY", default=True)
    WALLET_TRACKER_LOOKBACK_DAYS: int = _int("WALLET_TRACKER_LOOKBACK_DAYS", 14)
    WALLET_TRACKER_FETCH_LIMIT: int = _int("WALLET_TRACKER_FETCH_LIMIT", 10000)
    WALLET_TRACKER_CONDITION_CHUNK_SIZE: int = _int("WALLET_TRACKER_CONDITION_CHUNK_SIZE", 1)
    WALLET_TRACKER_FETCH_PAUSE_SECONDS: float = _float("WALLET_TRACKER_FETCH_PAUSE_SECONDS", 0.25)
    WALLET_TRACKER_TAKER_ONLY: bool = _bool("WALLET_TRACKER_TAKER_ONLY", default=False)

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_JSON: bool = _bool("LOG_JSON", default=True)

    # ── Cities (canonical registry; more discovered via Gamma) ────────────────
    from backend.city_registry import get_db_city_dicts as _get_db_city_dicts
    INITIAL_CITIES: list[dict] = _get_db_city_dicts()

    @classmethod
    def market_context_llm_ready(cls) -> bool:
        provider = cls.MARKET_CONTEXT_LLM_PROVIDER
        model = cls.MARKET_CONTEXT_LLM_MODEL
        if not provider or not model:
            return False
        if provider == "anthropic":
            return bool(cls.MARKET_CONTEXT_LLM_API_KEY or os.environ.get("ANTHROPIC_API_KEY"))
        if provider == "gemini":
            return bool(cls.MARKET_CONTEXT_LLM_API_KEY or os.environ.get("GEMINI_API_KEY"))
        if provider == "openai":
            return bool(cls.MARKET_CONTEXT_LLM_API_KEY or os.environ.get("OPENAI_API_KEY"))
        if provider == "openrouter":
            return bool(cls.MARKET_CONTEXT_LLM_API_KEY or os.environ.get("OPENROUTER_API_KEY"))
        return bool(cls.MARKET_CONTEXT_LLM_API_KEY)

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
        if cls.BANKROLL_CAP * cls.MAX_POSITION_PCT < cls.MIN_ORDER_NOTIONAL_DOLLARS:
            warnings.append(
                "BANKROLL_CAP × MAX_POSITION_PCT is below Polymarket minimum notional — "
                "auto sizing can only trade via the min-notional bump"
            )
        if cls.MARKET_CONTEXT_LLM_PROVIDER and not cls.market_context_llm_ready():
            warnings.append(
                "MARKET_CONTEXT_LLM_PROVIDER configured but model/API key incomplete — Market Context refresh disabled"
            )
        return warnings
