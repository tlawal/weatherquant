"""
Safety gates — every gate must pass before any trade executes.

Gates return a list of failures. Empty list = all gates passed.
Design: fail explicitly and verbosely so every rejection is auditable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from backend.config import Config
from backend.tz_utils import city_local_date, et_today
from backend.engine.signal_engine import BucketSignal
from backend.storage.db import get_session
from backend.storage.models import City, Event
from backend.storage.repos import (
    get_all_positions,
    get_arming_state,
    get_daily_high_metar,
    get_daily_realized_pnl,
    get_event,
    get_latest_forecast,
    get_latest_metar,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class GateResult:
    passed: bool
    failures: list[str]

    def __bool__(self) -> bool:
        return self.passed


async def run_all_gates(
    signal: BucketSignal,
    event: Event,
    city_id: int,
    strategy: str = "default",
    side: str = "BUY",
) -> GateResult:
    """
    Run all safety gates for a candidate trade signal.

    Returns GateResult with list of failure reasons.
    All failures are logged for audit trail.
    """
    failures: list[str] = []
    today_et = et_today()
    now_et = datetime.now(ET)

    # ── Gate: Armed ──────────────────────────────────────────────────────────
    async with get_session() as sess:
        arming = await get_arming_state(sess)

    if arming.state != "ARMED":
        failures.append(f"GATE_ARMED: arming_state={arming.state} (required=ARMED)")

    # ── Gate: Kill switch handled by arming state ─────────────────────────────
    # (disarming transitions to DISARMED automatically)

    # ── Gate: Trading window ─────────────────────────────────────────────────
    if now_et.hour >= Config.TRADING_WINDOW_CLOSE_ET:
        failures.append(
            f"GATE_TRADING_WINDOW: hour_et={now_et.hour} >= close_hour={Config.TRADING_WINDOW_CLOSE_ET}"
        )

    # ── Gate: Event exists and has OK buckets ───────────────────────────────
    if not event:
        failures.append("GATE_EVENT_EXISTS: no event found for today")
    else:
        if event.status != "ok":
            failures.append(f"GATE_EVENT_STATUS: event.status={event.status} (required=ok)")

        # ── Gate: Settlement source verified ─────────────────────────────────
        if not event.settlement_source_verified:
            failures.append(
                f"GATE_SETTLEMENT_SOURCE: settlement source not verified as WU "
                f"(source={event.settlement_source!r})"
            )

        # ── Gate: Forecast quality ────────────────────────────────────────────
        if event.forecast_quality != "ok":
            failures.append(
                f"GATE_WU_FRESHNESS: forecast_quality={event.forecast_quality} "
                f"(WU data likely stale or missing)"
            )

    # ── Gate: METAR freshness ─────────────────────────────────────────────────
    async with get_session() as sess:
        metar = await get_latest_metar(sess, city_id)

    if metar is None:
        failures.append("GATE_METAR: no METAR observation found")
    else:
        fetched = metar.fetched_at
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - fetched).total_seconds()
        if age_s > Config.METAR_STALE_TTL_SECONDS:
            failures.append(
                f"GATE_METAR_FRESHNESS: last METAR age={age_s:.0f}s > {Config.METAR_STALE_TTL_SECONDS}s"
            )

    # ── Gate: Liquidity ───────────────────────────────────────────────────────
    if signal.yes_ask_depth < Config.MIN_LIQUIDITY_SHARES:
        failures.append(
            f"GATE_LIQUIDITY: ask_depth={signal.yes_ask_depth:.1f} < "
            f"min={Config.MIN_LIQUIDITY_SHARES}"
        )

    # ── Gate: True edge threshold ─────────────────────────────────────────────
    if signal.true_edge < Config.MIN_TRUE_EDGE:
        failures.append(
            f"GATE_EDGE: true_edge={signal.true_edge:.4f} < min={Config.MIN_TRUE_EDGE}"
        )

    # ── Gate: Market price thresholds ─────────────────────────────────────────
    if signal.mkt_prob >= Config.MAX_ENTRY_PRICE:
        failures.append(
            f"GATE_MAX_PRICE: mkt_prob={signal.mkt_prob:.4f} >= max "
            f"entry threshold of {Config.MAX_ENTRY_PRICE}"
        )
    elif signal.mkt_prob < 0.02:
        failures.append(f"GATE_MIN_PRICE: mkt_prob={signal.mkt_prob:.4f} < 0.02")

    # ── Gate: Maximum Spread ──────────────────────────────────────────────────
    if signal.spread is None or signal.spread > Config.MAX_SPREAD:
        failures.append(
            f"GATE_SPREAD: spread={signal.spread} > max={Config.MAX_SPREAD}"
        )

    # ── Gate: Daily loss limit ────────────────────────────────────────────────
    async with get_session() as sess:
        daily_pnl = await get_daily_realized_pnl(sess, today_et)

    if daily_pnl < -Config.MAX_DAILY_LOSS:
        failures.append(
            f"GATE_DAILY_LOSS: daily_realized_pnl=${daily_pnl:.2f} < "
            f"-${Config.MAX_DAILY_LOSS:.2f} limit"
        )

    # ── Gate: Portfolio-level risk (drawdown, cluster, strategy) ──────────────
    # Skipped for SELL orders: selling reduces exposure, drawdown, and strategy
    # risk rather than increasing it.
    if side != "SELL":
        from backend.execution.portfolio_risk import check_portfolio_risk
        portfolio_failures = await check_portfolio_risk(
            city_slug=signal.city_slug,
            bankroll=Config.BANKROLL_CAP,
            strategy=strategy,
        )
        failures.extend(portfolio_failures)

    # ── Gate: Max open positions per event ───────────────────────────────────
    from backend.storage.models import Bucket as BucketModel
    async with get_session() as sess:
        all_positions = await get_all_positions(sess)

    # Position-count gates are entry-only; selling does not open new positions
    if side != "SELL":
        # Count positions for THIS specific event by joining through bucket→event
        event_id = event.id if event else None
        if event_id is not None:
            event_bucket_ids = set()
            async with get_session() as sess:
                from sqlalchemy import select
                rows = (await sess.execute(
                    select(BucketModel.id).where(BucketModel.event_id == event_id)
                )).scalars().all()
                event_bucket_ids = set(rows)
            event_position_count = sum(
                1 for p in all_positions
                if p.net_qty > 0 and p.bucket_id in event_bucket_ids
            )
            if event_position_count >= Config.MAX_POSITIONS_PER_EVENT:
                failures.append(
                    f"GATE_MAX_POSITIONS: event {event_id} has {event_position_count} "
                    f"positions >= max {Config.MAX_POSITIONS_PER_EVENT}"
                )

        # Global safety cap (across all events)
        total_open = sum(1 for p in all_positions if p.net_qty > 0)
        if total_open >= Config.MAX_POSITIONS_PER_EVENT * 6:
            failures.append(
                f"GATE_MAX_POSITIONS_GLOBAL: total open positions={total_open} "
                f"excessive (>= {Config.MAX_POSITIONS_PER_EVENT * 6}), review required"
            )

    # Fetch city object for timezone and date alignment gates
    async with get_session() as sess:
        city_obj = await sess.get(City, city_id)
    city_tz = getattr(city_obj, "tz", "America/New_York") if city_obj else "America/New_York"

    # ── Gate: Already-surpassed bracket ─────────────────────────────────────
    # If METAR daily high already exceeds this bucket's ceiling, the bracket
    # is impossible — the final high will be at least as high as what's observed.
    if signal.high_f is not None:
        async with get_session() as sess:
            obs_high = await get_daily_high_metar(sess, city_id, today_et, city_tz=city_tz)
        if obs_high is not None and obs_high >= signal.high_f:
            failures.append(
                f"GATE_BRACKET_SURPASSED: observed high {obs_high:.1f} "
                f"already exceeds bucket ceiling {signal.high_f:.1f}"
            )

    # Note: GATE_DATE_ALIGNMENT removed - with 3-day horizon and explicit date
    # selection, users intentionally trade any available date. The gate was
    # protecting against stale events from 8PM rollover, which no longer exists.

    if failures:
        log.warning(
            "gates: FAILED on %s bucket=%s: %s",
            signal.city_slug,
            signal.bucket_idx,
            "; ".join(failures),
        )
    else:
        log.info(
            "gates: ALL PASSED for %s bucket=%s edge=%.3f",
            signal.city_slug,
            signal.bucket_idx,
            signal.true_edge,
        )

    return GateResult(passed=len(failures) == 0, failures=failures)
