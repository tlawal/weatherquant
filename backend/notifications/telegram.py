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
) -> None:
    """Notification for a successful fill."""
    try:
        arrow = "\u2b06" if side.upper() == "BUY" else "\u2b07"
        text = (
            f"{arrow} <b>Fill</b> | {city_slug}\n"
            f"Bucket: {bucket_label}\n"
            f"Side: <b>{side.upper()}</b>  Shares: {shares:.2f}\n"
            f"Price: {price:.4f}  Edge: {edge:+.2%}"
        )
        await send_telegram(text)
    except Exception:
        logger.warning("notify_trade_filled failed", exc_info=True)


async def notify_exit_triggered(
    city_slug: str,
    level: str,
    reason: str,
    price: float,
    shares: float,
) -> None:
    """Notification for exit engine actions."""
    try:
        text = (
            f"\u26a0\ufe0f <b>Exit</b> | {city_slug}\n"
            f"Level: {level}  Reason: {reason}\n"
            f"Price: {price:.4f}  Shares: {shares:.2f}"
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
) -> None:
    """Notification when an exit order fails to execute."""
    try:
        text = (
            f"\U0001f6a8 <b>Exit FAILED</b> | {city_slug}\n"
            f"Level: {level}  Reason: {reason}\n"
            f"Price: {price:.4f}  Shares: {shares:.2f}\n"
            f"Error: {error[:120]}"
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
