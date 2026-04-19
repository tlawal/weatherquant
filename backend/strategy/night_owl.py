"""
Night Owl Strategy Orchestrator

Runs between 23:00 and 06:00 ET to exploit stale orderbooks after NWP models 
process the 00z and 06z cycles. During this window, Polymarket orderbooks 
sleep but deterministic models (HRRR, GFS, NBM) refresh with new insight for 
tomorrow's peak temperature.

Key alpha thesis: Overnight model updates (00z/06z) shift the true probability
distribution, but market makers aren't updating orderbooks. The delta between
the fresh model and the stale market price is the edge.

CRITICAL: Unlike the daytime auto-trader, Night Owl requires a FORECAST DELTA
of >= 1.5°F from the previous snapshot to trade. Pure edge without a delta
is not sufficient — the edge must come from *new information*, not stale
model agreement.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import desc, select

from backend.engine.signal_engine import run_signal_engine, BucketSignal
from backend.execution.arming import is_armed
from backend.execution.trader import execute_top_signals
from backend.ingestion.polymarket_clob import get_clob
from backend.config import Config
from backend.storage.db import get_session
from backend.storage.models import ModelSnapshot, Event

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# Minimum forecast shift (°F) required to trigger a Night Owl trade.
# Below this threshold, the overnight model update didn't provide new alpha.
FORECAST_DELTA_THRESHOLD = 1.5

# Tiered confidence based on delta magnitude
DELTA_TIERS = [
    # (min_delta, kelly_multiplier, label)
    (4.0, 1.5, "high_conviction"),
    (2.5, 1.2, "medium_conviction"),
    (1.5, 1.0, "standard"),
]


async def _get_previous_snapshot(event_id: int, hours_back: int = 8) -> Optional[ModelSnapshot]:
    """Fetch the most recent model snapshot from before the overnight window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    async with get_session() as sess:
        snap = (await sess.execute(
            select(ModelSnapshot)
            .where(ModelSnapshot.event_id == event_id)
            .where(ModelSnapshot.computed_at < cutoff)
            .order_by(desc(ModelSnapshot.computed_at))
            .limit(1)
        )).scalar_one_or_none()
    return snap


async def _get_latest_snapshot(event_id: int) -> Optional[ModelSnapshot]:
    """Fetch the most recent model snapshot (post-00z/06z update)."""
    async with get_session() as sess:
        snap = (await sess.execute(
            select(ModelSnapshot)
            .where(ModelSnapshot.event_id == event_id)
            .order_by(desc(ModelSnapshot.computed_at))
            .limit(1)
        )).scalar_one_or_none()
    return snap


async def _compute_forecast_deltas() -> dict[int, dict]:
    """Compare latest vs previous snapshots for all active events.

    Returns dict mapping event_id → {delta_mu, delta_sigma, prev_mu, curr_mu,
    conviction_tier, kelly_mult}.
    """
    deltas: dict[int, dict] = {}
    now_utc = datetime.now(timezone.utc)

    async with get_session() as sess:
        # Get tomorrow's events (the ones Night Owl targets)
        tomorrow_et = (datetime.now(ET) + timedelta(days=1)).strftime("%Y-%m-%d")
        today_et = datetime.now(ET).strftime("%Y-%m-%d")

        events = (await sess.execute(
            select(Event)
            .where(Event.date_et.in_([today_et, tomorrow_et]))
            .where(Event.status == "ok")
        )).scalars().all()

    for event in events:
        prev = await _get_previous_snapshot(event.id)
        curr = await _get_latest_snapshot(event.id)

        if prev is None or curr is None:
            continue
        if prev.mu is None or curr.mu is None:
            continue

        delta_mu = abs(curr.mu - prev.mu)

        if delta_mu < FORECAST_DELTA_THRESHOLD:
            log.debug(
                "night_owl: event %d delta_mu=%.2f°F < threshold %.1f — skipping",
                event.id, delta_mu, FORECAST_DELTA_THRESHOLD,
            )
            continue

        # Determine conviction tier
        kelly_mult = 1.0
        tier_label = "below_threshold"
        for min_delta, mult, label in DELTA_TIERS:
            if delta_mu >= min_delta:
                kelly_mult = mult
                tier_label = label
                break

        deltas[event.id] = {
            "delta_mu": round(delta_mu, 2),
            "delta_sigma": round(abs(curr.sigma - prev.sigma), 2) if prev.sigma and curr.sigma else 0.0,
            "prev_mu": round(prev.mu, 2),
            "curr_mu": round(curr.mu, 2),
            "direction": "warmer" if curr.mu > prev.mu else "cooler",
            "conviction_tier": tier_label,
            "kelly_mult": kelly_mult,
            "prev_snapshot_age_hrs": round((now_utc - prev.computed_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600, 1) if prev.computed_at else None,
        }

        log.info(
            "night_owl: event %d (%s) delta=%.1f°F (%s) tier=%s "
            "prev_mu=%.1f curr_mu=%.1f",
            event.id, event.date_et, delta_mu,
            deltas[event.id]["direction"], tier_label,
            prev.mu, curr.mu,
        )

    return deltas


async def run_night_owl() -> None:
    """Evaluate and execute the Night Owl strategy if within time window."""
    if not await is_armed():
        return

    now_et = datetime.now(ET)

    # Only run in the 23:00 to 06:00 window (00z runs ~1-3am, 06z runs ~7-9am but we cut off at 6am)
    if not (now_et.hour >= 23 or now_et.hour < 6):
        return

    clob = get_clob()
    bankroll = Config.BANKROLL_CAP
    if clob and clob.can_trade:
        balance = await clob.get_balance()
        if balance is not None:
            bankroll = min(balance, Config.BANKROLL_CAP)

    log.info("night_owl: evaluating 00z/06z models for overnight alpha")

    # Step 1: Compute forecast deltas — only events with significant overnight
    # model shifts qualify for Night Owl trading.
    deltas = await _compute_forecast_deltas()

    if not deltas:
        log.info("night_owl: no events with delta >= %.1f°F — standing down",
                 FORECAST_DELTA_THRESHOLD)
        return

    log.info("night_owl: %d events with significant overnight deltas", len(deltas))

    # Step 2: Generate signals across all cities
    signals = await run_signal_engine()

    # Step 3: Filter signals to only those with qualifying deltas
    delta_event_ids = set(deltas.keys())
    qualifying_signals = [
        s for s in signals
        if s.event_id in delta_event_ids
    ]

    if not qualifying_signals:
        log.info("night_owl: no actionable signals in delta-qualified events")
        return

    log.info(
        "night_owl: %d qualifying signals from %d delta-qualified events",
        len(qualifying_signals), len(delta_event_ids),
    )

    # Step 4: Execute — pass 'night_owl' strategy context to bypass
    # the standard 19:00 TRADING_WINDOW_CLOSE_ET gate in gating.py.
    results = await execute_top_signals(
        qualifying_signals,
        bankroll=bankroll,
        max_trades=Config.MAX_POSITIONS_PER_EVENT,
        strategy="night_owl",
    )

    if results:
        success = [r for r in results if r.get("status") == "filled"]
        if success:
            log.info(
                "night_owl: executed %d successful trades overnight "
                "(deltas: %s)",
                len(success),
                {eid: d["delta_mu"] for eid, d in deltas.items()},
            )
