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
    manual: bool = False,
    qty_override: float | None = None,
    order_type: str = "limit",
    side: str = "BUY",
    limit_price_override: float | None = None,
    strategy: str = "default",
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
    gate_result = await run_all_gates(signal, event, city.id, strategy=strategy)

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

    # ── Enforce Polymarket $1 minimum notional for market BUY orders ──────────
    # Polymarket's CLOB rejects market BUY orders with notional < $1. Rather
    # than silently bumping the dollar amount and leaving `shares` stale (which
    # makes the DB / UI lie about the real fill), bump `shares` up to the
    # smallest whole count that clears the $1 floor. Limit orders and SELLs are
    # unaffected — limits size by shares and SELLs pass shares directly.
    if order_type == "market" and side == "BUY" and shares * limit_price < 1.0 and limit_price > 0:
        min_shares = math.ceil(1.0 / limit_price)
        log.info(
            "trade: bumping shares %.2f → %d for Polymarket $1 min (price=$%.4f)",
            shares, min_shares, limit_price,
        )
        result["warning"] = (
            f"Adjusted from {shares:g} to {min_shares} shares "
            f"(Polymarket $1 minimum at {limit_price*100:.1f}¢)"
        )
        shares = float(min_shares)
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

    # ── Capture pre-trade position baseline (used by fill poller to compute delta)
    async with get_session() as sess:
        _baseline_pos = await get_position(sess, signal.bucket_id)
    baseline_qty = _baseline_pos.net_qty if _baseline_pos else 0.0
    baseline_avg_cost = _baseline_pos.avg_cost if _baseline_pos else 0.0

    # ── Persist order (pending) ───────────────────────────────────────────────
    async with get_session() as sess:
        order = await insert_order(
            sess,
            bucket_id=signal.bucket_id,
            side="sell_yes" if side == "SELL" else "buy_yes",
            qty=shares,
            limit_price=limit_price,
            status="pending",
            gates_json=json.dumps({"passed": True, "failures": []}),
        )

    result["order_id"] = order.id

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
            amount = shares * limit_price  # BUY: amount is in dollars (shares already ≥ $1 notional)
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
        side=side,
        baseline_qty=baseline_qty,
        baseline_avg_cost=baseline_avg_cost,
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


async def _fetch_real_fill(
    token_id: str,
    side: str,
    baseline_qty: float,
    baseline_avg_cost: float,
) -> Optional[tuple[float, float]]:
    """Query Polymarket data API for the real filled size/avg_price.

    Returns (fill_qty_abs, fill_price) or None on failure. fill_qty_abs is
    always positive; caller handles BUY vs SELL sign. fill_price is the
    effective price of *this fill only* (not the running avg_cost), derived
    from the change in weighted cost basis between the baseline position and
    the post-fill position.
    """
    try:
        import aiohttp
        from backend.execution.chain_utils import get_wallet_address

        wallet = get_wallet_address()
        url = f"https://data-api.polymarket.com/positions?user={wallet}"
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.warning("fill_poll: data-api returned %d", resp.status)
                    return None
                positions = await resp.json()

        # Find the position for this token
        new_size = 0.0
        new_avg_price = 0.0
        for p in positions or []:
            if p.get("asset") == token_id:
                new_size = float(p.get("size", 0) or 0)
                new_avg_price = float(p.get("avgPrice", 0) or 0)
                break
        # If not found and we had a baseline, position was fully closed → size=0.

        if side == "BUY":
            delta = new_size - baseline_qty
            if delta <= 0:
                return None  # API lag — let caller retry or fallback
            # Effective price of this fill alone, derived from weighted cost basis:
            #   new_size * new_avg = baseline_qty * baseline_avg + delta * fill_price
            new_cost_basis = new_size * new_avg_price
            old_cost_basis = baseline_qty * baseline_avg_cost
            fill_price = (new_cost_basis - old_cost_basis) / delta
            if fill_price <= 0 or fill_price > 1:
                # Sanity fallback — use the post-fill avg price
                fill_price = new_avg_price if new_avg_price > 0 else 0.0
            return (delta, fill_price)
        else:  # SELL
            delta = baseline_qty - new_size
            if delta <= 0:
                return None  # API lag
            # For a SELL, the "fill price" is what we received per share.
            # We don't have that directly from the positions API, so use the
            # post-fill avg_cost as a proxy — it's unchanged by sells (FIFO
            # realization doesn't move avg_cost). The caller passes this
            # through to realized_pnl via upsert_position; the mark price is
            # a better proxy, but expected_price (bid at submit) is fine.
            return (delta, 0.0)  # caller will substitute expected_price
    except Exception as e:
        log.warning("fill_poll: _fetch_real_fill error: %s", e)
        return None


async def _poll_for_fill(
    order_id: int,
    clob_order_id: str,
    token_id: str,
    expected_size: float,
    expected_price: float,
    side: str = "BUY",
    baseline_qty: float = 0.0,
    baseline_avg_cost: float = 0.0,
    timeout_s: float = 30.0,
    poll_interval_s: float = 2.0,
) -> Optional[dict]:
    """Poll CLOB for fill confirmation, then read the actual fill from the
    Polymarket data API so the DB reflects on-chain reality (not the user's
    expected values, which may differ when Polymarket's $1 minimum kicks in
    or when a FOK market order walks the book across levels).
    """
    clob = get_clob()
    if not clob:
        return None

    start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start) < timeout_s:
        await asyncio.sleep(poll_interval_s)

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
                # Try to read the real fill from the data API, with short
                # retries to absorb indexer lag (usually < 4s).
                real = None
                for _ in range(3):
                    real = await _fetch_real_fill(
                        token_id, side, baseline_qty, baseline_avg_cost,
                    )
                    if real is not None:
                        break
                    await asyncio.sleep(1.5)

                if real is not None:
                    fill_qty, derived_price = real
                    # For SELLs the derived_price is 0 (not recoverable from
                    # positions API alone); use the limit/expected price.
                    fill_price = derived_price if derived_price > 0 else expected_price
                    log.info(
                        "fill_poll: real fill qty=%.4f price=$%.4f (expected %.4f @ $%.4f)",
                        fill_qty, fill_price, expected_size, expected_price,
                    )
                else:
                    log.warning(
                        "fill_poll: data-api did not report fill; falling back to expected "
                        "(qty=%.4f price=$%.4f)",
                        expected_size, expected_price,
                    )
                    fill_qty = expected_size
                    fill_price = expected_price

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
    strategy: str = "default",
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
        result = await execute_signal(signal, bankroll, actor=actor, strategy=strategy)
        results.append(result)
        log.info("trader: %s → status=%s", signal.city_slug, result["status"])

        # Brief pause between trades
        await asyncio.sleep(1)

    return results
