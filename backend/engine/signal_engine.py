"""
Signal engine — computes per-bucket edge and selects candidate trades.

Edge = model_prob - market_prob - execution_cost

Execution cost model:
  half_spread + slippage_estimate based on depth
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from backend.config import Config
from backend.modeling.distribution import edge as compute_edge
from backend.modeling.temperature_model import compute_model, ModelResult
from backend.modeling.calibration import get_calibration_async
from backend.modeling.calibration_engine import get_reliability_metrics, remap_probability
from backend.storage.db import get_session
from backend.storage.models import Bucket, Event, City
from backend.storage.repos import (
    get_all_cities,
    get_buckets_for_event,
    get_calibration,
    get_daily_high_metar,
    get_event,
    get_latest_forecast,
    get_latest_market_snapshot,
    get_latest_metar,
    get_latest_model_snapshot,
    insert_model_snapshot,
    insert_signal,
    update_heartbeat,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class BucketSignal:
    city_slug: str
    city_display: str
    unit: str
    event_id: int
    bucket_id: int
    bucket_idx: int
    label: str
    low_f: Optional[float]
    high_f: Optional[float]
    model_prob: float
    mkt_prob: float
    raw_edge: float
    exec_cost: float
    true_edge: float
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    yes_mid: Optional[float]
    spread: Optional[float]
    yes_ask_depth: float
    reason: dict = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    actionable: bool = False


def _execution_cost(spread: Optional[float], ask_depth: float) -> float:
    """
    Estimate total execution cost = half_spread + slippage.

    Slippage is depth-dependent: thin markets have more impact.
    """
    half_spread = (spread or 0.04) / 2  # default 2% each side if unknown
    if ask_depth > 200:
        slippage = 0.005
    elif ask_depth > 100:
        slippage = 0.010
    elif ask_depth > 50:
        slippage = 0.015
    else:
        slippage = 0.025  # thin market, high impact
    return float(round(half_spread + slippage, 4))


async def run_signal_engine() -> list[BucketSignal]:
    """
    Run the full signal engine for all enabled cities.

    Returns list of BucketSignal ordered by true_edge descending.
    """
    today_et = date.today().isoformat()
    signals: list[BucketSignal] = []

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        city_signals = await _compute_city_signals(city, today_et)
        signals.extend(city_signals)

    # Sort by true_edge descending
    signals.sort(key=lambda s: s.true_edge, reverse=True)

    async with get_session() as sess:
        await update_heartbeat(sess, "run_model", success=True)

    return signals


async def _compute_city_signals(city: City, today_et: str) -> list[BucketSignal]:
    """Compute signals for all buckets in a city's today event."""
    async with get_session() as sess:
        event = await get_event(sess, city.id, today_et)
        if not event:
            return []

        if event.status not in ("ok",):
            return []

        buckets = await get_buckets_for_event(sess, event.id)
        metar = await get_latest_metar(sess, city.id)
        daily_high = await get_daily_high_metar(sess, city.id, today_et)

        nws_obs = await get_latest_forecast(sess, city.id, "nws", today_et)
        wu_daily_obs = await get_latest_forecast(sess, city.id, "wu_daily", today_et)
        wu_hourly_obs = await get_latest_forecast(sess, city.id, "wu_hourly", today_et)
        wu_history_obs = await get_latest_forecast(sess, city.id, "wu_history", today_et)

        cal = await get_calibration(sess, city.id)
        # NEW: Reliability metrics for probability remapping
        reliability_bins = await get_reliability_metrics(city.id)

    if not buckets:
        log.debug("signal: %s — no buckets", city.city_slug)
        return []

    # Build bucket boundary list
    bucket_ranges = [(b.low_f, b.high_f) for b in buckets]

    # Resolve ground truth: prefer WU history since that is Polymarket's settlement source
    ground_truth_high = wu_history_obs.high_f if (wu_history_obs and wu_history_obs.high_f is not None) else daily_high

    # Build calibration dict
    cal_dict = None
    if cal:
        cal_dict = {
            "bias_nws": cal.bias_nws,
            "bias_wu_daily": cal.bias_wu_daily,
            "bias_wu_hourly": cal.bias_wu_hourly,
            "weight_nws": cal.weight_nws,
            "weight_wu_daily": cal.weight_wu_daily,
            "weight_wu_hourly": cal.weight_wu_hourly,
        }

    # Run temperature model
    model = compute_model(
        nws_high=nws_obs.high_f if nws_obs else None,
        wu_daily_high=wu_daily_obs.high_f if wu_daily_obs else None,
        wu_hourly_peak=wu_hourly_obs.high_f if wu_hourly_obs else None,
        daily_high_metar=ground_truth_high,
        current_temp_f=metar.temp_f if metar else None,
        calibration=cal_dict,
        buckets=bucket_ranges,
        forecast_quality=event.forecast_quality or "ok",
        unit=getattr(city, "unit", "F"),
    )

    if model is None:
        log.warning("signal: %s — model returned None (insufficient data)", city.city_slug)
        return []

    # Persist model snapshot
    async with get_session() as sess:
        await insert_model_snapshot(
            sess,
            event_id=event.id,
            mu=model.mu,
            sigma=model.sigma,
            probs_json=json.dumps(model.probs),
            inputs_json=json.dumps(model.inputs),
            forecast_quality=model.forecast_quality,
        )

    # Compute signal per bucket
    signals: list[BucketSignal] = []
    for i, bucket in enumerate(buckets):
        if i >= len(model.probs):
            continue

        model_prob = model.probs[i]

        # If METAR high already exceeds this bucket's ceiling, probability is 0
        # (the final daily high can only go up, never down)
        if ground_truth_high is not None and bucket.high_f is not None:
            if ground_truth_high >= bucket.high_f:
                model_prob = 0.0

        # Get latest market snapshot
        async with get_session() as sess:
            mkt_snap = await get_latest_market_snapshot(sess, bucket.id)

        if not mkt_snap or mkt_snap.yes_mid is None:
            # No market data — count as signal with no actionable edge
            sig = BucketSignal(
                city_slug=city.city_slug,
                city_display=city.display_name,
                unit=getattr(city, "unit", "F"),
                event_id=event.id,
                bucket_id=bucket.id,
                bucket_idx=i,
                label=bucket.label or f"Bucket {i}",
                low_f=bucket.low_f,
                high_f=bucket.high_f,
                model_prob=float(round(model_prob, 4)),
                mkt_prob=0.0,
                raw_edge=0.0,
                exec_cost=0.0,
                true_edge=0.0,
                yes_bid=None,
                yes_ask=None,
                yes_mid=None,
                spread=None,
                yes_ask_depth=0.0,
                gate_failures=["no_market_data"],
            )
            signals.append(sig)
            continue

        mkt_prob = mkt_snap.yes_mid
        ask_depth = mkt_snap.yes_ask_depth or 0.0
        spread = mkt_snap.spread
        exec_cost = _execution_cost(spread, ask_depth)

        # Apply probability calibration (remap based on historical reliability)
        calibrated_prob = remap_probability(model_prob, reliability_bins)

        # Edge calculation based on calibrated probability
        raw_edge_buy = calibrated_prob - mkt_prob
        true_edge = raw_edge_buy - exec_cost

        reason = {
            **model.inputs,
            "bucket_idx": i,
            "label": bucket.label,
            "model_prob_raw": float(round(model_prob, 4)),
            "model_prob_cal": float(round(calibrated_prob, 4)),
            "mkt_prob": float(round(mkt_prob, 4)),
            "raw_edge": float(round(raw_edge_buy, 4)),
            "exec_cost": float(round(exec_cost, 4)),
            "true_edge": float(round(true_edge, 4)),
            "spread": spread,
            "ask_depth": ask_depth,
        }

        actionable = (
            true_edge >= Config.MIN_TRUE_EDGE
            and 0.02 <= mkt_prob <= 0.98  # avoid extreme markets
            and ask_depth >= Config.MIN_LIQUIDITY_SHARES
            and event.forecast_quality == "ok"
        )

        sig = BucketSignal(
            city_slug=city.city_slug,
            city_display=city.display_name,
            unit=getattr(city, "unit", "F"),
            event_id=event.id,
            bucket_id=bucket.id,
            bucket_idx=i,
            label=bucket.label or f"Bucket {i}",
            low_f=bucket.low_f,
            high_f=bucket.high_f,
            model_prob=float(round(model_prob, 4)),
            mkt_prob=float(round(mkt_prob, 4)),
            raw_edge=round(raw_edge_buy, 4),
            exec_cost=round(exec_cost, 4),
            true_edge=round(true_edge, 4),
            yes_bid=mkt_snap.yes_bid,
            yes_ask=mkt_snap.yes_ask,
            yes_mid=float(round(mkt_prob, 4)),
            spread=spread,
            yes_ask_depth=ask_depth,
            reason=reason,
            actionable=actionable,
        )
        signals.append(sig)

        # Persist signal to DB
        async with get_session() as sess:
            await insert_signal(
                sess,
                bucket_id=bucket.id,
                model_prob=sig.model_prob,
                mkt_prob=sig.mkt_prob,
                raw_edge=sig.raw_edge,
                exec_cost=sig.exec_cost,
                true_edge=sig.true_edge,
                reason_json=json.dumps(reason, default=str),
                gate_failures_json=json.dumps(sig.gate_failures),
            )

    return signals
