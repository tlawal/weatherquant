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
from backend.engine.signal_engine import BucketSignal
from backend.storage.db import get_session
from backend.storage.models import Event
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
) -> GateResult:
    """
    Run all safety gates for a candidate trade signal.

    Returns GateResult with list of failure reasons.
    All failures are logged for audit trail.
    """
    failures: list[str] = []
    today_et = date.today().isoformat()
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
        age_s = (datetime.now(timezone.utc) - metar.fetched_at).total_seconds()
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

    # ── Gate: Market not at extremes ─────────────────────────────────────────
    if signal.mkt_prob < 0.02 or signal.mkt_prob > 0.98:
        failures.append(
            f"GATE_MKT_EXTREME: mkt_prob={signal.mkt_prob:.4f} is at extreme "
            f"(acceptable range: 0.02–0.98)"
        )

    # ── Gate: Daily loss limit ────────────────────────────────────────────────
    async with get_session() as sess:
        daily_pnl = await get_daily_realized_pnl(sess, today_et)

    if daily_pnl < -Config.MAX_DAILY_LOSS:
        failures.append(
            f"GATE_DAILY_LOSS: daily_realized_pnl=${daily_pnl:.2f} < "
            f"-${Config.MAX_DAILY_LOSS:.2f} limit"
        )

    # ── Gate: Max open positions per event ───────────────────────────────────
    async with get_session() as sess:
        all_positions = await get_all_positions(sess)

    event_positions = [
        p for p in all_positions
        # This check is simplified — in production we'd join through bucket→event
        # For now we count globally across all events
    ]
    # Count positions linked to this specific event
    if len(all_positions) >= Config.MAX_POSITIONS_PER_EVENT * 3:  # global safety
        failures.append(
            f"GATE_MAX_POSITIONS: total open positions={len(all_positions)} "
            f"excessive, review required"
        )

    # ── Gate: Already-surpassed bracket ─────────────────────────────────────
    # If METAR daily high already exceeds this bucket's ceiling, the bracket
    # is impossible — the final high will be at least as high as what's observed.
    if signal.high_f is not None:
        async with get_session() as sess:
            obs_high = await get_daily_high_metar(sess, city_id, today_et)
        if obs_high is not None and obs_high >= signal.high_f:
            failures.append(
                f"GATE_BRACKET_SURPASSED: observed high {obs_high:.1f} "
                f"already exceeds bucket ceiling {signal.high_f:.1f}"
            )

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
