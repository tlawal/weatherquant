"""
Exit Engine — manages risk and takes profit for open positions.

Interval: 300 seconds (5 min)
4-level cascade:
    1. EMERGENCY: METAR obs contradicts bucket by >= 3°F (Market sell)
    2. URGENT: Model consensus shifted to different bucket (Limit sell bid - 1c)
       — debounced: requires CONSENSUS_DEBOUNCE_RUNS consecutive shifts
       — spread-guarded: suppressed when spread > URGENT_EXIT_MAX_SPREAD
    3. PROFIT: Quick Flip! Position is at entry + QUICK_FLIP_TARGET (Limit sell at bid)
    4. EXPIRY: 30 min before market close (Sell at bid - EXPIRY_DISCOUNT)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, delete, desc

from backend.config import Config
from backend.storage.db import get_session
from backend.storage.repos import get_all_positions, get_open_orders, get_city_by_slug
from backend.engine.signal_engine import run_signal_engine, BucketSignal
from backend.execution.trader import execute_signal
from backend.tz_utils import city_local_now

log = logging.getLogger(__name__)

# Polymarket taker fee roughly 2%
FEE_RATE = 0.02

# ── DB-backed consensus history (survives deploys) ──────────────────────────
# In-memory cache is populated from DB on first call and kept in sync.
_consensus_cache: dict[int, list[int]] = {}
_consensus_cache_loaded = False


async def _load_consensus_cache() -> None:
    """Warm the in-memory cache from ConsensusHistory rows."""
    global _consensus_cache, _consensus_cache_loaded
    if _consensus_cache_loaded:
        return
    from backend.storage.models import ConsensusHistory
    async with get_session() as sess:
        rows = (await sess.execute(
            select(ConsensusHistory)
            .order_by(ConsensusHistory.recorded_at.asc())
            .limit(500)  # cap memory
        )).scalars().all()
    cache: dict[int, list[int]] = {}
    for r in rows:
        cache.setdefault(r.event_id, []).append(r.bucket_id)
    _consensus_cache = cache
    _consensus_cache_loaded = True
    log.info("consensus_cache: loaded %d events, %d total rows from DB",
             len(cache), sum(len(v) for v in cache.values()))


async def _record_consensus(event_id: int, bucket_id: int) -> None:
    """Write consensus to DB and update in-memory cache."""
    from backend.storage.models import ConsensusHistory
    hist = _consensus_cache.setdefault(event_id, [])
    hist.append(bucket_id)
    if len(hist) > 10:
        hist.pop(0)
    async with get_session() as sess:
        sess.add(ConsensusHistory(
            event_id=event_id,
            bucket_id=bucket_id,
            recorded_at=datetime.now(timezone.utc),
        ))
        # Prune old rows for this event (keep last 10)
        all_rows = (await sess.execute(
            select(ConsensusHistory.id)
            .where(ConsensusHistory.event_id == event_id)
            .order_by(desc(ConsensusHistory.recorded_at))
        )).scalars().all()
        if len(all_rows) > 10:
            stale_ids = all_rows[10:]
            await sess.execute(
                delete(ConsensusHistory)
                .where(ConsensusHistory.id.in_(stale_ids))
            )
        await sess.commit()


def _stable_consensus(event_id: int, held_bucket_id: int, multiplier: int = 1) -> int | None:
    """Return the consensus bucket_id only if the last N runs consistently
    agree on a DIFFERENT bucket than the held one. Returns None if still noisy."""
    hist = _consensus_cache.get(event_id, [])
    n = Config.CONSENSUS_DEBOUNCE_RUNS * multiplier
    if len(hist) < n:
        return None  # not enough data to be confident
    recent = hist[-n:]
    # All recent runs must agree on the same bucket, AND it must differ from held
    if all(b == recent[0] for b in recent) and recent[0] != held_bucket_id:
        return recent[0]  # stable shift confirmed
    return None  # noisy or still matches held — suppress URGENT


async def _run_exit_cascade_for_position(
    pos, signal: BucketSignal, consensus_bucket_id: Optional[int], consensus_sig: Optional[BucketSignal] = None
) -> dict | None:
    """Evaluate the 4-level cascade for a single position."""
    
    # ── Price sanity checks ──
    bid = signal.yes_bid or 0.0
    if bid <= 0:
        return None  # No phantom exits on stale/zero books
    
    if pos.avg_cost <= 0:
        return None
        
    async with get_session() as sess:
        city = await get_city_by_slug(sess, signal.city_slug)
        if not city:
            return None
            
    now_local = city_local_now(city)
    
    # Calculate net profit (accounting for 2% fee)
    gross_pnl_per_share = bid - pos.avg_cost
    fee_per_share = bid * FEE_RATE
    net_pnl_per_share = gross_pnl_per_share - fee_per_share
    
    # Extract METAR observations
    current_temp = signal.reason.get("current_temp_f")
    raw_high = signal.reason.get("raw_high")
    obs_high = raw_high if raw_high is not None else current_temp
    
    # ── 1. EMERGENCY ──
    # METAR obs contradicts bucket by >= 3°F causing impossible situation or deep loss
    if obs_high is not None and signal.high_f is not None:
        if obs_high > signal.high_f + 1.0:
            # We already busted this bucket. It's essentially worth 0. Try to salvage anything.
            log.warning("exit: EMERGENCY %s - busted high (obs=%.1f > bucket=%.1f)", signal.city_slug, obs_high, signal.high_f)
            return {"level": "EMERGENCY", "price": bid, "reason": "busted_high"}
            
    if obs_high is not None and signal.low_f is not None and signal.model_prob < 0.05 and signal.city_state in ("resolved", "volatile"):
        if obs_high < signal.low_f - 3.0 and now_local.hour >= 18:
            log.warning("exit: EMERGENCY %s - deep miss (obs=%.1f < bucket=%.1f)", signal.city_slug, obs_high, signal.low_f)
            return {"level": "EMERGENCY", "price": bid, "reason": "deep_miss"}
            
    # ── 2. URGENT ── (debounced + spread-guarded + confidence-gated)
    # Model consensus shifted to different bucket, we are holding a non-consensus bucket.
    if consensus_bucket_id and pos.bucket_id != consensus_bucket_id and consensus_sig:
        age_s = (now_local.astimezone(timezone.utc) - pos.entry_time.astimezone(timezone.utc)).total_seconds() if pos.entry_time else float('inf')
        spread = signal.spread or 0.0
        bid_depth = signal.yes_bid_depth or 0.0
        held_prob = signal.model_prob
        cons_prob = consensus_sig.model_prob

        if age_s < Config.URGENT_MIN_POSITION_AGE_SECONDS:
            log.info("exit: URGENT suppressed %s — position age %ds < %ds", signal.city_slug, int(age_s), Config.URGENT_MIN_POSITION_AGE_SECONDS)
        elif held_prob >= Config.URGENT_MIN_EXIT_MODEL_PROB and held_prob >= cons_prob * 0.40:
            log.info("exit: URGENT suppressed %s — probability gate (held=%.2f, cons=%.2f)", signal.city_slug, held_prob, cons_prob)
        elif bid_depth < Config.URGENT_MIN_BID_DEPTH:
            log.warning("exit: URGENT suppressed %s — thin bid depth %.1f < %.1f", signal.city_slug, bid_depth, Config.URGENT_MIN_BID_DEPTH)
        elif spread > Config.URGENT_EXIT_MAX_SPREAD:
            log.warning("exit: URGENT suppressed %s — spread %.3f > %.3f (consensus→%s)", signal.city_slug, spread, Config.URGENT_EXIT_MAX_SPREAD, consensus_bucket_id)
        else:
            # URGENT exit: only sell non-moon-bag portion
            sell_qty = pos.net_qty - (pos.moon_bag_qty or 0.0)
            if sell_qty <= 0:
                log.info("exit: URGENT suppressed %s — only moon-bag remaining (%.1f shares)", signal.city_slug, pos.net_qty)
            else:
                # Check depth for passive vs aggressive sell
                if bid_depth < sell_qty * 2:
                    sell_price = bid  # Passive limit sell at the bid exactly to not sweep thin books
                else:
                    sell_price = max(0.01, bid - 0.01)  # Aggressive: bid - 1c

                log.info("exit: URGENT %s - consensus shifted to bucket_id=%s. Exiting %.1f shares (keeping %.1f moon-bag).",
                         signal.city_slug, consensus_bucket_id, sell_qty, pos.moon_bag_qty or 0.0)
                return {"level": "URGENT", "price": sell_price, "reason": "consensus_shifted",
                        "qty_override": sell_qty}

    # ── Update trailing stop high-water mark ──
    # Ratchet up max_bid_seen and trailing stop on every cycle
    if bid > (pos.max_bid_seen or 0.0):
        async with get_session() as sess:
            from sqlalchemy import update
            import backend.storage.models as m
            await sess.execute(
                update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                .values(
                    max_bid_seen=bid,
                    trailing_stop_price=max(bid - 0.05, pos.trailing_stop_price or 0.0) if pos.tier_1_exited else None,
                )
            )
            await sess.commit()

    # ── 3. PROFIT — Tiered partial exits + trailing stop ──
    tier_1_target = pos.avg_cost + 0.08  # +8¢ → sell 50%
    tier_2_target = pos.avg_cost + 0.15  # +15¢ → sell 25% more
    original_qty = pos.original_qty if pos.original_qty > 0 else pos.net_qty

    # Tier 1: Sell 50% at +8¢
    if not pos.tier_1_exited and bid >= tier_1_target:
        tier_1_qty = round(original_qty * 0.50, 2)
        tier_1_qty = min(tier_1_qty, pos.net_qty)  # can't sell more than we have
        if tier_1_qty > 0:
            log.info("exit: PROFIT Tier-1 %s — selling 50%% (%.1f shares) at +8¢ (bid=%.3f, entry=%.3f)",
                     signal.city_slug, tier_1_qty, bid, pos.avg_cost)
            return {"level": "PROFIT", "price": bid, "reason": "tier_1_50pct",
                    "qty_override": tier_1_qty,
                    "post_exit_update": {"tier_1_exited": True,
                                          "moon_bag_qty": round(original_qty * 0.25, 2),
                                          "trailing_stop_price": bid - 0.05,
                                          "max_bid_seen": bid}}

    # Tier 2: Sell 25% at +15¢
    if pos.tier_1_exited and not pos.tier_2_exited and bid >= tier_2_target:
        tier_2_qty = round(original_qty * 0.25, 2)
        tier_2_qty = min(tier_2_qty, pos.net_qty - (pos.moon_bag_qty or 0.0))
        if tier_2_qty > 0:
            log.info("exit: PROFIT Tier-2 %s — selling 25%% (%.1f shares) at +15¢ (bid=%.3f)",
                     signal.city_slug, tier_2_qty, bid)
            return {"level": "PROFIT", "price": bid, "reason": "tier_2_25pct",
                    "qty_override": tier_2_qty,
                    "post_exit_update": {"tier_2_exited": True}}

    # Trailing stop: After Tier 1, if bid drops below trailing stop, exit non-moon portion
    if pos.tier_1_exited and pos.trailing_stop_price and bid < pos.trailing_stop_price:
        trailing_qty = pos.net_qty - (pos.moon_bag_qty or 0.0)
        if trailing_qty > 0:
            log.info("exit: PROFIT Trailing-Stop %s — bid %.3f < stop %.3f. Selling %.1f shares.",
                     signal.city_slug, bid, pos.trailing_stop_price, trailing_qty)
            return {"level": "PROFIT", "price": bid, "reason": "trailing_stop",
                    "qty_override": trailing_qty}

    # Legacy quick-flip for positions that haven't been initialized with tiers
    if not pos.tier_1_exited and not pos.original_qty:
        target_price = pos.avg_cost + Config.QUICK_FLIP_TARGET
        if bid >= target_price:
            log.info("exit: PROFIT QuickFlip %s! (entry=%.3f, bid=%.3f, net_pnl_share=%.3f)", signal.city_slug, pos.avg_cost, bid, net_pnl_per_share)
            return {"level": "PROFIT", "price": bid, "reason": "quick_flip"}

    # ── 4. EXPIRY ──
    # 30 min before market close (7:30 PM local for daily high markets)
    # Moon-bag exits here too — no position survives past market close
    if now_local.hour == 19 and now_local.minute >= 30:
        log.info("exit: EXPIRY %s - 30 min to close. Dumping remaining (including moon-bag).", signal.city_slug)
        return {"level": "EXPIRY", "price": max(0.01, bid - Config.EXPIRY_DISCOUNT), "reason": "market_close"}

    return None


async def run_exit_engine() -> None:
    """Run the 4-level exit engine cascade on all open positions."""
    log.info("exit_engine: evaluating open positions")
    
    async with get_session() as sess:
        positions = await get_all_positions(sess)
        open_orders = await get_open_orders(sess)
        
    active_positions = [p for p in positions if p.net_qty > 0]
    if not active_positions:
        log.debug("exit_engine: no active positions")
        return
        
    # Find buckets with pending SELL orders so we don't double-exit
    pending_sell_buckets = {o.bucket_id for o in open_orders if o.side == "sell_yes"}
    
    # Get latest signals to inform exits
    signals = await run_signal_engine()
    sig_map = {s.bucket_id: s for s in signals}
    
    # Warm consensus cache from DB on first run (survives deploys)
    await _load_consensus_cache()

    # Find consensus bucket for each event (highest model prob)
    event_consensus = {}
    for s in signals:
        if s.event_id not in event_consensus or s.model_prob > event_consensus[s.event_id].model_prob:
            event_consensus[s.event_id] = s

    # Persist consensus to DB + update in-memory cache
    for event_id, sig in event_consensus.items():
        await _record_consensus(event_id, sig.bucket_id)

    exits_triggered = 0

    for pos in active_positions:
        if pos.bucket_id in pending_sell_buckets:
            continue

        signal = sig_map.get(pos.bucket_id)
        if not signal:
            continue

        # Use debounced consensus — only fires if last N runs consistently agree
        # on a different bucket than the held position.
        consensus_sig = event_consensus.get(signal.event_id)
        multiplier = 1
        if consensus_sig and abs(consensus_sig.bucket_idx - signal.bucket_idx) == 1:
            multiplier = Config.URGENT_ADJACENT_DEBOUNCE_MULTIPLIER

        stable_consensus_id = _stable_consensus(signal.event_id, pos.bucket_id, multiplier)

        cascade = await _run_exit_cascade_for_position(pos, signal, stable_consensus_id, consensus_sig)
        if cascade:
            exits_triggered += 1
            sell_price = round(cascade["price"], 3)
            exit_qty = cascade.get("qty_override", pos.net_qty)
            log.info(
                "exit_engine: triggering %s exit for bucket %d (shares=%.1f of %.1f). Limit=%.3f.", 
                cascade["level"], pos.bucket_id, exit_qty, pos.net_qty, sell_price
            )
            
            # Save exit triggered status
            async with get_session() as sess:
                from sqlalchemy import update
                import backend.storage.models as m
                await sess.execute(
                    update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                    .values(current_exit_status=f"{cascade['level']} exit triggered: {cascade['reason']} ({exit_qty:.1f} shares)")
                )
                await sess.commit()

            result = await execute_signal(
                signal=signal,
                bankroll=0.0, # Not used for sells
                actor="exit_engine",
                manual=True, # Bypasses sizing & most gates
                order_type="limit",
                side="SELL",
                limit_price_override=sell_price,
                qty_override=exit_qty,
            )
            
            # ── Handle result ──
            if result.get("status") in ("filled", "timeout", "open"):
                # Apply post-exit tier state updates (tier flags, trailing stop, etc.)
                post_update = cascade.get("post_exit_update")
                if post_update:
                    async with get_session() as sess:
                        from sqlalchemy import update
                        import backend.storage.models as m
                        await sess.execute(
                            update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                            .values(**post_update)
                        )
                        await sess.commit()
                        log.info("exit_engine: applied post-exit update for bucket %d: %s", pos.bucket_id, post_update)

                try:
                    from backend.notifications.telegram import notify_exit_triggered
                    await notify_exit_triggered(
                        city_slug=signal.city_slug,
                        level=cascade["level"],
                        reason=cascade["reason"],
                        price=sell_price,
                        shares=exit_qty,
                    )
                except Exception:
                    log.debug("Telegram exit notification failed (non-critical)", exc_info=True)
            else:
                # Exit order failed — update position status and notify
                err = result.get("error", "unknown")
                log.warning(
                    "exit_engine: %s exit for bucket %d FAILED — %s",
                    cascade["level"], pos.bucket_id, err,
                )
                async with get_session() as sess:
                    from sqlalchemy import update
                    import backend.storage.models as m
                    await sess.execute(
                        update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                        .values(current_exit_status=f"{cascade['level']} exit FAILED: {err[:80]}")
                    )
                    await sess.commit()
                try:
                    from backend.notifications.telegram import notify_exit_failed
                    await notify_exit_failed(
                        city_slug=signal.city_slug,
                        level=cascade["level"],
                        reason=cascade["reason"],
                        price=sell_price,
                        shares=exit_qty,
                        error=err,
                    )
                except Exception:
                    log.debug("Telegram exit-failed notification failed (non-critical)", exc_info=True)
        else:
            # Save active monitoring status with tier awareness
            bid = signal.yes_bid or 0.0
            if pos.tier_1_exited:
                moon_info = f" | Moon-bag: {pos.moon_bag_qty:.1f} shares" if pos.moon_bag_qty else ""
                trail_info = f" | Trail: ${pos.trailing_stop_price:.3f}" if pos.trailing_stop_price else ""
                status = f"Monitoring: Tier-1 locked{' + Tier-2 locked' if pos.tier_2_exited else ''}{moon_info}{trail_info} (bid ${bid:.2f})"
            else:
                target_price = pos.avg_cost + 0.08
                status = f"Monitoring: await Tier-1 +8¢ (target ${target_price:.2f}, bid ${bid:.2f}) or Shift"
            async with get_session() as sess:
                from sqlalchemy import update
                import backend.storage.models as m
                await sess.execute(
                    update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                    .values(current_exit_status=status)
                )
                await sess.commit()
            
    log.info("exit_engine: complete (%d exits triggered)", exits_triggered)

