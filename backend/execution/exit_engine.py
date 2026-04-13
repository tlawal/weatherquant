"""
Exit Engine — manages risk and takes profit for open positions.

Interval: 300 seconds (5 min)
4-level cascade:
    1. EMERGENCY: METAR obs contradicts bucket by >= 3°F (Market sell)
    2. URGENT: Model consensus shifted to different bucket (Limit sell bid - 1c)
    3. PROFIT: Quick Flip! Position is at entry + 5c (Limit sell at bid)
    4. EXPIRY: 30 min before market close (Sell at bid - 10c)
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.storage.db import get_session
from backend.storage.repos import get_all_positions, get_open_orders, get_city_by_slug
from backend.engine.signal_engine import run_signal_engine, BucketSignal
from backend.execution.trader import execute_signal
from backend.tz_utils import city_local_now

log = logging.getLogger(__name__)

# Polymarket taker fee roughly 2%
FEE_RATE = 0.02


async def _run_exit_cascade_for_position(
    pos, signal: BucketSignal, consensus_bucket_id: Optional[int]
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
            
    # ── 2. URGENT ──
    # Model consensus shifted to different bucket, we are holding a non-consensus bucket
    if consensus_bucket_id and pos.bucket_id != consensus_bucket_id:
        log.info("exit: URGENT %s - consensus shifted to bucket_id=%s. Exiting holding.", signal.city_slug, consensus_bucket_id)
        return {"level": "URGENT", "price": max(0.01, bid - 0.01), "reason": "consensus_shifted"}
        
    # ── 3. PROFIT (Quick Flip) ──
    # Buy at entry, full exit at entry + 0.05
    target_price = pos.avg_cost + 0.05
    if bid >= target_price:
        log.info("exit: PROFIT QuickFlip %s! (entry=%.3f, bid=%.3f, net_pnl_share=%.3f)", signal.city_slug, pos.avg_cost, bid, net_pnl_per_share)
        return {"level": "PROFIT", "price": bid, "reason": "quick_flip"}
        
    # ── 4. EXPIRY ──
    # 30 min before market close (7:30 PM local for daily high markets)
    if now_local.hour == 19 and now_local.minute >= 30:
        log.info("exit: EXPIRY %s - 30 min to close. Dumping remaining.", signal.city_slug)
        return {"level": "EXPIRY", "price": max(0.01, bid - 0.10), "reason": "market_close"}
        
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
    
    # Find consensus bucket for each event (highest model prob)
    event_consensus = {}
    for s in signals:
        if s.event_id not in event_consensus or s.model_prob > event_consensus[s.event_id].model_prob:
            event_consensus[s.event_id] = s

    exits_triggered = 0
    
    for pos in active_positions:
        if pos.bucket_id in pending_sell_buckets:
            continue
            
        signal = sig_map.get(pos.bucket_id)
        if not signal:
            continue
            
        consensus_sig = event_consensus.get(signal.event_id)
        consensus_bucket_id = consensus_sig.bucket_id if consensus_sig else None
        
        cascade = await _run_exit_cascade_for_position(pos, signal, consensus_bucket_id)
        if cascade:
            exits_triggered += 1
            sell_price = round(cascade["price"], 3)
            log.info(
                "exit_engine: triggering %s exit for bucket %d (shares=%.1f). Limit=%.3f.", 
                cascade["level"], pos.bucket_id, pos.net_qty, sell_price
            )
            
            # Save exit triggered status
            async with get_session() as sess:
                from sqlalchemy import update
                import backend.storage.models as m
                await sess.execute(
                    update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                    .values(current_exit_status=f"{cascade['level']} exit triggered: {cascade['reason']}")
                )
                await sess.commit()

            await execute_signal(
                signal=signal,
                bankroll=0.0, # Not used for sells
                actor="exit_engine",
                manual=True, # Bypasses sizing & most gates
                order_type="limit",
                side="SELL",
                limit_price_override=sell_price,
                qty_override=pos.net_qty # Dump the whole position
            )
        else:
            # Save active monitoring status
            target_price = pos.avg_cost + 0.05
            bid = signal.yes_bid or 0.0
            async with get_session() as sess:
                from sqlalchemy import update
                import backend.storage.models as m
                await sess.execute(
                    update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                    .values(current_exit_status=f"Monitoring: await +5¢ Quick-Flip (target ${target_price:.2f}, bid ${bid:.2f}) or Shift")
                )
                await sess.commit()
            
    log.info("exit_engine: complete (%d exits triggered)", exits_triggered)
