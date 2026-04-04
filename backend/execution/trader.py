"""
Trader — places limit orders via CLOB after all gates pass.

Flow per trade:
  1. Run safety gates
  2. Compute Kelly position size
  3. Place limit order at yes_ask (aggressive limit, not market)
  4. Persist to orders table
  5. Poll for fill (up to 30s)
  6. Update fills, positions tables
  7. Audit log everything

Manual trades from the dashboard go through this same path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import date, datetime, timezone
from typing import Optional

from backend.config import Config
from backend.tz_utils import city_local_date
from backend.engine.gating import run_all_gates, GateResult
from backend.engine.signal_engine import BucketSignal
from backend.execution.arming import is_armed
from backend.execution.risk_manager import compute_size
from backend.ingestion.polymarket_clob import get_clob
from backend.storage.db import get_session
from backend.storage.repos import (
    append_audit,
    get_all_positions,
    get_event,
    get_position,
    insert_fill,
    insert_order,
    update_order_status,
    upsert_position,
)

log = logging.getLogger(__name__)

import re as _re


def _rewrite_balance_error(raw: str) -> str | None:
    """If raw is a Polymarket insufficient-balance error, return a friendly message."""
    m = _re.search(
        r"balance:\s*(\d+).*?sum of active orders:\s*(\d+).*?order amount:\s*(\d+)",
        raw, _re.DOTALL
    )
    if not m:
        return None
    bal    = int(m.group(1)) / 1_000_000
    locked = int(m.group(2)) / 1_000_000
    need   = int(m.group(3)) / 1_000_000
    avail  = bal - locked
    return (
        f"Insufficient balance: need ${need:.2f} but only ${avail:.2f} available "
        f"(${bal:.2f} total, ${locked:.2f} locked in active orders). "
        f"Cancel active orders to free funds."
    )


async def execute_signal(
    signal: BucketSignal,
    bankroll: float,
    actor: str = "auto_trader",
    dry_run: bool = False,
    manual: bool = False,
    qty_override: float | None = None,
    order_type: str = "limit",
    side: str = "BUY",
    limit_price_override: float | None = None,
) -> dict:
    """
    Attempt to execute a trade for the given signal.

    Returns status dict with outcome and audit payload.
    """
    result = {
        "city": signal.city_slug,
        "bucket_idx": signal.bucket_idx,
        "label": signal.label,
        "true_edge": signal.true_edge,
        "status": "unknown",
        "gate_failures": [],
        "order_id": None,
        "error": None,
    }

    # ── Fetch event ───────────────────────────────────────────────────────────
    # We need the city_id for gate checks — derive from signal's event_id
    # (We store city_id indirectly; use city_slug to look it up)
    from backend.storage.repos import get_city_by_slug, get_event_with_buckets
    async with get_session() as sess:
        city = await get_city_by_slug(sess, signal.city_slug)
        if not city:
            result["status"] = "error"
            result["error"] = f"city not found: {signal.city_slug}"
            return result

        from backend.tz_utils import city_local_now, city_local_tomorrow
        now_local = city_local_now(city)
        if now_local.hour >= 20:
            active_date = city_local_tomorrow(city)
        else:
            active_date = city_local_date(city)
        event = await get_event(sess, city.id, active_date)
        if not event:
            result["status"] = "no_event"
            return result

    # ── Run all safety gates ──────────────────────────────────────────────────
    gate_result = await run_all_gates(signal, event, city.id)

    # For manual trades, filter out bot-only gates that don't apply to
    # human-initiated trades. Keep critical safety gates (daily loss, max positions).
    BOT_ONLY_GATES = {
        "GATE_ARMED", "GATE_EDGE", "GATE_TRADING_WINDOW",
        "GATE_SETTLEMENT_SOURCE", "GATE_BRACKET_SURPASSED",
        "GATE_MKT_EXTREME", "GATE_DATE_ALIGNMENT", "GATE_WU_FRESHNESS",
        "GATE_LIQUIDITY", "GATE_METAR",
    }
    if manual:
        filtered = [f for f in gate_result.failures
                    if not any(f.startswith(g) for g in BOT_ONLY_GATES)]
        gate_result = GateResult(passed=len(filtered) == 0, failures=filtered)

    result["gate_failures"] = gate_result.failures

    if not gate_result.passed:
        result["status"] = "gate_blocked"
        async with get_session() as sess:
            await append_audit(
                sess,
                actor=actor,
                action="trade_gate_blocked",
                payload={**result, "signal": signal.reason},
                ok=False,
                error_msg="; ".join(gate_result.failures),
            )
        return result

    # ── Compute position size ─────────────────────────────────────────────────
    if limit_price_override is not None:
        limit_price = limit_price_override
    elif side == "SELL":
        limit_price = signal.yes_bid or signal.yes_mid or 0.0
    else:
        limit_price = signal.yes_ask or signal.yes_mid or 0.0
    if limit_price <= 0:
        result["status"] = "error"
        result["error"] = "no valid price"
        return result

    if side == "SELL":
        # SELL: must specify qty, validate against held position
        if not qty_override or qty_override <= 0:
            result["status"] = "error"
            result["error"] = "qty required for sell orders"
            return result
        async with get_session() as sess:
            pos = await get_position(sess, signal.bucket_id)
        if not pos or pos.net_qty < qty_override:
            result["status"] = "error"
            result["error"] = f"insufficient shares (held={pos.net_qty if pos else 0}, requested={qty_override})"
            return result
        shares = qty_override
        cost = round(shares * limit_price, 2)
    elif qty_override and qty_override > 0:
        # Manual trade — user specified their own quantity, skip auto-sizing
        shares = qty_override
        cost = round(shares * limit_price, 2)
    else:
        # Auto-size via Kelly criterion
        async with get_session() as sess:
            all_positions = await get_all_positions(sess)

        open_exposure = sum(
            (p.net_qty * p.avg_cost) for p in all_positions if p.net_qty > 0
        )

        sizing = compute_size(
            model_prob=signal.model_prob,
            limit_price=limit_price,
            bankroll=bankroll,
            open_exposure=open_exposure,
            ask_depth=signal.yes_ask_depth,
        )

        if sizing.rejected:
            result["status"] = "sizing_rejected"
            result["error"] = sizing.reject_reason
            async with get_session() as sess:
                await append_audit(
                    sess,
                    actor=actor,
                    action="trade_sizing_rejected",
                    payload={**result, "sizing": {
                        "kelly_f": sizing.kelly_f,
                        "kelly_size": sizing.kelly_size,
                        "reject_reason": sizing.reject_reason,
                    }},
                    ok=False,
                )
            return result

        shares = sizing.size
        cost = round(shares * limit_price, 2)

    result["shares"] = shares
    result["limit_price"] = limit_price
    result["estimated_cost"] = cost

    log.info(
        "trade: %s bucket=%d %.2f shares @ $%.4f = $%.2f (edge=%.3f %s)",
        signal.city_slug,
        signal.bucket_idx,
        shares,
        limit_price,
        cost,
        signal.true_edge,
        "[DRY-RUN]" if dry_run else "",
    )

    # ── Look up YES token ID for this bucket ──────────────────────────────────
    from backend.storage.repos import get_buckets_for_event
    async with get_session() as sess:
        buckets = await get_buckets_for_event(sess, event.id)

    bucket = next((b for b in buckets if b.id == signal.bucket_id), None)
    if not bucket or not bucket.yes_token_id:
        result["status"] = "error"
        result["error"] = "bucket or yes_token_id not found"
        return result

    # ── For SELL: verify on-chain conditional token balance ────────────────────
    if side == "SELL":
        try:
            import aiohttp
            from backend.execution.chain_utils import (
                erc1155_balance, get_wallet_address, CTF_ADDRESS,
            )
            wallet = get_wallet_address()
            async with aiohttp.ClientSession() as _http:
                onchain_raw = await erc1155_balance(
                    _http, CTF_ADDRESS, wallet, bucket.yes_token_id,
                )
            onchain_shares = onchain_raw / 1_000_000

            if onchain_shares <= 0:
                result["status"] = "error"
                result["error"] = (
                    f"on-chain token balance is 0 "
                    f"(DB shows {pos.net_qty if pos else 0} shares); "
                    "position may have already been sold or transferred"
                )
                # Correct DB to match on-chain reality
                async with get_session() as sess:
                    pos_obj = await get_position(sess, signal.bucket_id)
                    if pos_obj and pos_obj.net_qty > 0:
                        pos_obj.net_qty = 0
                        await sess.commit()
                        log.warning(
                            "sell: corrected DB position to 0 for bucket %d "
                            "(was %.2f, on-chain=0)",
                            signal.bucket_id, pos.net_qty,
                        )
                return result

            if onchain_shares < shares:
                log.warning(
                    "sell: on-chain balance %.2f < requested %.2f for bucket %d; "
                    "capping to on-chain",
                    onchain_shares, shares, signal.bucket_id,
                )
                result["warning"] = (
                    f"Capped sell from {shares:.0f} to {onchain_shares:.0f} shares "
                    f"(on-chain balance < DB position)"
                )
                shares = onchain_shares
                cost = round(shares * limit_price, 2)
                result["shares"] = shares
                result["estimated_cost"] = cost
                # Correct DB position to match on-chain
                async with get_session() as sess:
                    pos_obj = await get_position(sess, signal.bucket_id)
                    if pos_obj and pos_obj.net_qty != onchain_shares:
                        pos_obj.net_qty = onchain_shares
                        await sess.commit()
                        log.info(
                            "sell: corrected DB position to %.2f for bucket %d",
                            onchain_shares, signal.bucket_id,
                        )
        except Exception as e:
            log.warning(
                "sell: on-chain balance check failed (proceeding anyway): %s", e
            )

    # ── Persist order (pending) ───────────────────────────────────────────────
    async with get_session() as sess:
        order = await insert_order(
            sess,
            bucket_id=signal.bucket_id,
            side="sell_yes" if side == "SELL" else "buy_yes",
            qty=shares,
            limit_price=limit_price,
            status="pending" if not dry_run else "dry_run",
            gates_json=json.dumps({"passed": True, "failures": []}),
        )

    result["order_id"] = order.id

    if dry_run:
        result["status"] = "dry_run"
        return result

    # ── Place CLOB order ──────────────────────────────────────────────────────
    clob = get_clob()
    if not clob or not clob.can_trade:
        result["status"] = "error"
        result["error"] = "CLOB client not available or no credentials"
        async with get_session() as sess:
            await update_order_status(sess, order.id, "rejected", cancel_reason="no_clob_client")
        return result

    if order_type == "market":
        if side == "SELL":
            amount = shares  # SELL: amount is in shares
        else:
            amount = max(shares * limit_price, 1.0)  # BUY: amount is in dollars, min $1
        clob_result = await clob.place_market_order(
            token_id=bucket.yes_token_id,
            side=side,
            amount=amount,
        )
    else:
        clob_result = await clob.place_limit_order(
            token_id=bucket.yes_token_id,
            side=side,
            size=shares,
            price=limit_price,
        )

    if not clob_result or "error" in clob_result:
        result["status"] = "order_failed"
        if not clob_result:
            result["error"] = "CLOB returned no result"
        else:
            raw_err = clob_result.get("error", "Unknown CLOB error")
            result["error"] = _rewrite_balance_error(raw_err) or raw_err
            
        async with get_session() as sess:
            await update_order_status(sess, order.id, "rejected", cancel_reason=result["error"][:100])
            await append_audit(
                sess,
                actor=actor,
                action="trade_order_failed",
                payload=result,
                ok=False,
                error_msg=result["error"][:200],
            )
        return result

    # Extract CLOB order ID
    clob_order_id = (
        clob_result.get("orderID")
        or clob_result.get("order_id")
        or clob_result.get("id")
        or ""
    )

    async with get_session() as sess:
        await update_order_status(sess, order.id, "open", clob_order_id=clob_order_id)

    # ── Poll for fill (up to 30s) ─────────────────────────────────────────────
    fill_result = await _poll_for_fill(
        order_id=order.id,
        clob_order_id=clob_order_id,
        token_id=bucket.yes_token_id,
        expected_size=shares,
        expected_price=limit_price,
    )

    if fill_result:
        # Update position
        async with get_session() as sess:
            await upsert_position(
                sess,
                bucket_id=signal.bucket_id,
                side="yes",
                fill_qty=fill_result["qty"] if side == "BUY" else -fill_result["qty"],
                fill_price=fill_result["price"],
                last_mkt_price=fill_result["price"],
            )
            await append_audit(
                sess,
                actor=actor,
                action="trade_filled",
                payload={
                    **result,
                    "fill": fill_result,
                    "signal_reason": signal.reason,
                },
            )
        result["status"] = "filled"
        result["fill"] = fill_result
    else:
        result["status"] = "timeout"
        result["error"] = "fill not confirmed within 30s"
        async with get_session() as sess:
            await update_order_status(
                sess, order.id, "timeout", cancel_reason="fill_poll_timeout"
            )
            # Attempt cancel
            if clob and clob_order_id:
                await clob.cancel_order(clob_order_id)
                await update_order_status(sess, order.id, "cancelled", cancel_reason="timeout_cancel")
            await append_audit(
                sess,
                actor=actor,
                action="trade_timeout",
                payload=result,
                ok=False,
                error_msg="fill poll timeout",
            )

    return result


async def _poll_for_fill(
    order_id: int,
    clob_order_id: str,
    token_id: str,
    expected_size: float,
    expected_price: float,
    timeout_s: float = 30.0,
    poll_interval_s: float = 2.0,
) -> Optional[dict]:
    """Poll CLOB for fill confirmation."""
    clob = get_clob()
    if not clob:
        return None

    start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start) < timeout_s:
        await asyncio.sleep(poll_interval_s)

        # Simple heuristic: check if the order book changed (ask depth reduced)
        # In production this should poll GET /order/{order_id} from CLOB
        # For now we use a time-based optimistic fill after first check
        try:
            if not clob._client:
                break
            loop = asyncio.get_event_loop()
            from py_clob_client.clob_types import OpenOrderParams
            open_orders = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: clob._client.get_orders(
                        OpenOrderParams(market=token_id)
                    ),
                ),
                timeout=8.0,
            )
            # If our order is no longer in open orders, it was filled or cancelled
            order_ids = [o.get("id") or o.get("orderID") for o in (open_orders or [])]
            if clob_order_id and clob_order_id not in order_ids:
                fill_price = expected_price
                fill_qty = expected_size

                async with get_session() as sess:
                    await insert_fill(
                        sess,
                        order_id=order_id,
                        qty=fill_qty,
                        price=fill_price,
                    )
                    await update_order_status(
                        sess, order_id, "filled",
                        fill_price=fill_price,
                        fill_qty=fill_qty,
                    )
                return {"qty": fill_qty, "price": fill_price}

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            log.warning("fill_poll: error: %s", e)
            break

    return None


async def execute_top_signals(
    signals: list[BucketSignal],
    bankroll: float,
    max_trades: int = 2,
    actor: str = "auto_trader",
) -> list[dict]:
    """
    Execute the top N actionable signals.

    Used by the auto-trader scheduler loop.
    """
    if not await is_armed():
        log.debug("trader: not armed, skipping auto-execution")
        return []

    actionable = [s for s in signals if s.actionable]
    if not actionable:
        log.debug("trader: no actionable signals")
        return []

    results = []
    for signal in actionable[:max_trades]:
        result = await execute_signal(signal, bankroll, actor=actor)
        results.append(result)
        log.info("trader: %s → status=%s", signal.city_slug, result["status"])

        # Brief pause between trades
        await asyncio.sleep(1)

    return results
