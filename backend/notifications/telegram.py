"""
Telegram notifications for WeatherQuant trading events.

All functions are fire-and-forget: exceptions are logged as warnings
and never propagated to callers.
"""
from __future__ import annotations

import logging

import aiohttp

from backend.config import Config

logger = logging.getLogger(__name__)

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT = aiohttp.ClientTimeout(total=10)


async def send_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """Send a message via the Telegram Bot API.

    Returns True on success, False otherwise.  Never raises.
    """
    try:
        if not Config.TELEGRAM_ENABLED:
            logger.debug("Telegram notifications disabled — skipping")
            return False

        token = Config.TELEGRAM_BOT_TOKEN
        chat_id = Config.TELEGRAM_CHAT_ID

        if not token or not chat_id:
            logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
            return False

        url = _API_URL.format(token=token)
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                logger.warning(
                    "Telegram API returned %s: %s", resp.status, body
                )
                return False
    except Exception:
        logger.warning("Failed to send Telegram message", exc_info=True)
        return False


# ── Trade lifecycle notifications ────────────────────────────────


async def notify_trade_filled(
    city_slug: str,
    bucket_label: str,
    side: str,
    shares: float,
    price: float,
    edge: float,
) -> bool:
    """Notification for a successful fill."""
    try:
        arrow = "\u2b06" if side.upper() == "BUY" else "\u2b07"
        text = (
            f"{arrow} <b>Fill</b> | {city_slug}\n"
            f"Bucket: {bucket_label}\n"
            f"Side: <b>{side.upper()}</b>  Shares: {shares:.2f}\n"
            f"Price: {price:.4f}  Edge: {edge:+.2%}"
        )
        ok = await send_telegram(text)
        if not ok:
            logger.warning("notify_trade_filled: Telegram send skipped or failed")
        return ok
    except Exception:
        logger.warning("notify_trade_filled failed", exc_info=True)
        return False


async def notify_exit_triggered(
    city_slug: str,
    level: str,
    reason: str,
    price: float,
    shares: float,
    details: dict | None = None,
) -> None:
    """Notification for exit engine actions."""
    try:
        detail_lines = ""
        if details:
            entry_ev = details.get("entry_ev_at_bid")
            current_ev = details.get("current_ev_at_bid")
            ev_delta = details.get("ev_delta")
            true_edge_delta = details.get("true_edge_delta")
            model_delta = details.get("model_prob_delta")
            market_delta = details.get("market_prob_delta")
            source_deltas = details.get("source_high_deltas") or {}
            price_adjustment = details.get("price_adjustment") or {}
            if details.get("entry_model_snapshot_unavailable"):
                detail_lines += "\nEntry model snapshot unavailable"
            elif entry_ev is not None or current_ev is not None:
                detail_lines += (
                    f"\nEV: {entry_ev if entry_ev is not None else '—'}"
                    f" → {current_ev if current_ev is not None else '—'}"
                )
                if ev_delta is not None:
                    detail_lines += f" (Δ {ev_delta:+.3f})"
            if model_delta is not None or market_delta is not None:
                detail_lines += (
                    f"\nModel Δ: {model_delta:+.3f}" if model_delta is not None else "\nModel Δ: —"
                )
                detail_lines += (
                    f"  Market Δ: {market_delta:+.3f}" if market_delta is not None else "  Market Δ: —"
                )
            if true_edge_delta is not None:
                detail_lines += f"\nTrue edge Δ: {true_edge_delta:+.3f}"
            if source_deltas:
                parts = [
                    f"{k.replace('_high', '').replace('_peak', '')}:{v:+.1f}°"
                    for k, v in list(source_deltas.items())[:5]
                ]
                detail_lines += "\nSource Δ: " + ", ".join(parts)
            if price_adjustment:
                detail_lines += (
                    "\nOrder price: "
                    f"{price_adjustment.get('reference_price', '—')} → "
                    f"{price_adjustment.get('order_price', '—')}"
                )
        text = (
            f"\u26a0\ufe0f <b>Exit</b> | {city_slug}\n"
            f"Level: {level}  Reason: {reason}\n"
            f"Price: {price:.4f}  Shares: {shares:.2f}"
            f"{detail_lines}"
        )
        await send_telegram(text)
    except Exception:
        logger.warning("notify_exit_triggered failed", exc_info=True)


async def notify_exit_failed(
    city_slug: str,
    level: str,
    reason: str,
    price: float,
    shares: float,
    error: str,
    details: dict | None = None,
) -> None:
    """Notification when an exit order fails to execute."""
    try:
        detail_lines = ""
        price_adjustment = (details or {}).get("price_adjustment") if details else None
        if price_adjustment:
            detail_lines = (
                "\nOrder price: "
                f"{price_adjustment.get('reference_price', '—')} → "
                f"{price_adjustment.get('order_price', '—')}"
            )
        text = (
            f"\U0001f6a8 <b>Exit FAILED</b> | {city_slug}\n"
            f"Level: {level}  Reason: {reason}\n"
            f"Price: {price:.4f}  Shares: {shares:.2f}\n"
            f"Error: {error[:120]}"
            f"{detail_lines}"
        )
        await send_telegram(text)
    except Exception:
        logger.warning("notify_exit_failed failed", exc_info=True)


async def notify_gate_blocked(
    city_slug: str,
    bucket_label: str,
    failures: list[str],
) -> None:
    """Notification when gates block a trade (only for signals with edge > 0)."""
    try:
        items = "\n".join(f"  - {f}" for f in failures)
        text = (
            f"\u26d4 <b>Gate blocked</b> | {city_slug}\n"
            f"Bucket: {bucket_label}\n"
            f"{items}"
        )
        await send_telegram(text)
    except Exception:
        logger.warning("notify_gate_blocked failed", exc_info=True)


async def notify_daily_pnl(
    realized: float,
    unrealized: float,
    positions: int,
) -> None:
    """Daily P&L summary."""
    try:
        total = realized + unrealized
        text = (
            f"\U0001f4ca <b>Daily P&amp;L</b>\n"
            f"Realized: {realized:+,.2f}\n"
            f"Unrealized: {unrealized:+,.2f}\n"
            f"Total: <b>{total:+,.2f}</b>\n"
            f"Open positions: {positions}"
        )
        await send_telegram(text)
    except Exception:
        logger.warning("notify_daily_pnl failed", exc_info=True)
