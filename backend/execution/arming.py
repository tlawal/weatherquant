"""
2-step arming state machine with kill switch.

States:
  DISARMED (default, safe)
  ARMING_PENDING (issued token, TTL=60s)
  ARMED (auto-trading allowed)

Transitions:
  DISARMED → ARMING_PENDING: POST /arming/request (admin token required)
  ARMING_PENDING → ARMED: POST /arming/confirm (arming token + ARMING_SECRET)
  ARMED → DISARMED: POST /arming/disable (kill switch, always available)
  AUTO: transitions to DISARMED if:
    - token expires
    - ET hour >= TRADING_FORCE_DISABLE_ET
    - daily_loss > MAX_DAILY_LOSS

All transitions are audit-logged.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.config import Config
from backend.storage.db import get_session
from backend.storage.repos import (
    append_audit,
    get_arming_state,
    update_arming_state,
)

log = logging.getLogger(__name__)


async def get_state() -> dict:
    """Return current arming state as dict."""
    async with get_session() as sess:
        arming = await get_arming_state(sess)
    return {
        "state": arming.state,
        "armed_at": arming.armed_at.isoformat() if arming.armed_at else None,
        "armed_by": arming.armed_by,
        "token_expires_at": arming.token_expires_at.isoformat() if arming.token_expires_at else None,
        "disarmed_at": arming.disarmed_at.isoformat() if arming.disarmed_at else None,
        "disarmed_reason": arming.disarmed_reason,
    }


async def is_armed() -> bool:
    """Quick check: is arming state currently ARMED?"""
    async with get_session() as sess:
        arming = await get_arming_state(sess)

    # Auto-expire pending token
    if arming.state == "ARMING_PENDING" and arming.token_expires_at:
        if datetime.now(timezone.utc) > arming.token_expires_at:
            async with get_session() as sess:
                await update_arming_state(
                    sess,
                    state="DISARMED",
                    arming_token=None,
                    token_expires_at=None,
                    disarmed_reason="arming_token_expired",
                    disarmed_at=datetime.now(timezone.utc),
                )
                await append_audit(
                    sess,
                    actor="system",
                    action="arming_token_expired",
                    payload={"previous_state": "ARMING_PENDING"},
                )
            return False

    return arming.state == "ARMED"


async def request_arming(actor: str) -> tuple[bool, str, Optional[str]]:
    """
    Step 1: Request arming — issues a short-lived token.

    Returns (success, message, token_if_success)
    """
    async with get_session() as sess:
        arming = await get_arming_state(sess)

    if arming.state == "ARMED":
        return False, "Already ARMED — use /arming/disable first", None

    if not Config.ARMING_SECRET:
        return False, "ARMING_SECRET not configured on server — cannot arm", None

    token = secrets.token_urlsafe(16)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=Config.ARMING_TOKEN_TTL_SECONDS)

    async with get_session() as sess:
        await update_arming_state(
            sess,
            state="ARMING_PENDING",
            arming_token=token,
            token_expires_at=expires_at,
        )
        await append_audit(
            sess,
            actor=actor,
            action="arming_requested",
            payload={
                "expires_at": expires_at.isoformat(),
                "ttl_seconds": Config.ARMING_TOKEN_TTL_SECONDS,
            },
        )

    log.warning("arming: PENDING — actor=%s expires_at=%s", actor, expires_at.isoformat())
    return True, "Arming token issued. Confirm within 60 seconds.", token


async def confirm_arming(
    token: str, secret: str, actor: str
) -> tuple[bool, str]:
    """
    Step 2: Confirm arming with token + ARMING_SECRET.

    Returns (success, message)
    """
    async with get_session() as sess:
        arming = await get_arming_state(sess)

    if arming.state != "ARMING_PENDING":
        return False, f"Cannot confirm: state is {arming.state} (expected ARMING_PENDING)"

    # Check token expiry
    if arming.token_expires_at and datetime.now(timezone.utc) > arming.token_expires_at:
        async with get_session() as sess:
            await update_arming_state(
                sess,
                state="DISARMED",
                arming_token=None,
                token_expires_at=None,
                disarmed_reason="token_expired_at_confirm",
            )
            await append_audit(
                sess,
                actor=actor,
                action="arming_confirm_failed",
                payload={"reason": "token_expired"},
                ok=False,
            )
        return False, "Arming token has expired — start over"

    # Verify token
    if not secrets.compare_digest(str(arming.arming_token or ""), token):
        async with get_session() as sess:
            await append_audit(
                sess,
                actor=actor,
                action="arming_confirm_failed",
                payload={"reason": "invalid_token"},
                ok=False,
            )
        log.warning("arming: CONFIRM FAILED — bad token from actor=%s", actor)
        return False, "Invalid arming token"

    # Verify secret
    if not Config.ARMING_SECRET or not secrets.compare_digest(Config.ARMING_SECRET, secret):
        async with get_session() as sess:
            await append_audit(
                sess,
                actor=actor,
                action="arming_confirm_failed",
                payload={"reason": "invalid_secret"},
                ok=False,
            )
        log.warning("arming: CONFIRM FAILED — bad secret from actor=%s", actor)
        return False, "Invalid arming secret"

    now = datetime.now(timezone.utc)
    async with get_session() as sess:
        await update_arming_state(
            sess,
            state="ARMED",
            arming_token=None,
            token_expires_at=None,
            armed_at=now,
            armed_by=actor,
        )
        await append_audit(
            sess,
            actor=actor,
            action="arming_confirmed",
            payload={"armed_at": now.isoformat()},
        )

    log.warning("🔴 TRADING ARMED — actor=%s", actor)
    return True, "Trading is now ARMED. All safety gates still apply per trade."


async def disarm(actor: str, reason: str = "manual_kill_switch") -> tuple[bool, str]:
    """
    Kill switch — immediately DISARMS regardless of current state.

    Returns (success, message)
    """
    now = datetime.now(timezone.utc)
    async with get_session() as sess:
        await update_arming_state(
            sess,
            state="DISARMED",
            arming_token=None,
            token_expires_at=None,
            disarmed_at=now,
            disarmed_reason=reason,
        )
        await append_audit(
            sess,
            actor=actor,
            action="arming_disabled",
            payload={"reason": reason, "disarmed_at": now.isoformat()},
        )

    log.warning("🟢 TRADING DISARMED — actor=%s reason=%s", actor, reason)
    return True, f"Trading DISARMED. Reason: {reason}"


async def auto_check_disarm() -> None:
    """
    Called by scheduler — auto-disarms on time or loss conditions.
    """
    from zoneinfo import ZoneInfo
    from backend.tz_utils import et_today

    ET = ZoneInfo("America/New_York")
    now_et = datetime.now(ET)
    today_et = et_today()

    if not await is_armed():
        return

    # Auto-disarm after TRADING_FORCE_DISABLE_ET
    if now_et.hour >= Config.TRADING_FORCE_DISABLE_ET:
        await disarm(
            "system",
            reason=f"auto_disarm_after_{Config.TRADING_FORCE_DISABLE_ET}:00_ET",
        )
        return

    # Auto-disarm on daily loss
    from backend.storage.repos import get_daily_realized_pnl
    async with get_session() as sess:
        daily_pnl = await get_daily_realized_pnl(sess, today_et)

    if daily_pnl < -Config.MAX_DAILY_LOSS:
        await disarm(
            "system",
            reason=f"auto_disarm_daily_loss_limit: pnl=${daily_pnl:.2f}",
        )
