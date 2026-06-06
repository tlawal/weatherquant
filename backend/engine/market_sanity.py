"""Market-implied posterior gate for auto-entry decisions.

This layer treats market prices as noisy evidence, not ground truth.  It does
not rewrite displayed model probabilities; it only blocks auto-entry when a
liquid, fresh market materially contradicts the model edge.
"""
from __future__ import annotations

import math
from typing import Any, Optional


def _clamp_prob(prob: float) -> float:
    try:
        return max(0.001, min(0.999, float(prob)))
    except (TypeError, ValueError):
        return 0.5


def logit(prob: float) -> float:
    p = _clamp_prob(prob)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def market_quality_weight(
    *,
    market_snapshot_age_s: Optional[float],
    spread: Optional[float],
    bid_depth: float,
    ask_depth: float,
) -> float:
    """Return market evidence weight in [0, 0.45]."""
    try:
        age_s = float(market_snapshot_age_s) if market_snapshot_age_s is not None else None
        spread_f = float(spread) if spread is not None else None
        bid = float(bid_depth or 0.0)
        ask = float(ask_depth or 0.0)
    except (TypeError, ValueError):
        return 0.0

    min_depth = min(bid, ask)
    if age_s is None or spread_f is None:
        return 0.0
    if age_s > 300 or spread_f > 0.08 or min_depth < 5:
        return 0.0
    if spread_f <= 0.03 and min_depth >= 20 and age_s <= 90:
        return 0.35
    if spread_f <= 0.06 and min_depth >= 10:
        return 0.15
    return 0.0


def evaluate_market_sanity(
    *,
    model_prob: float,
    market_prob: float,
    exec_cost: float,
    model_true_edge: float,
    market_snapshot_age_s: Optional[float],
    spread: Optional[float],
    bid_depth: float,
    ask_depth: float,
    min_true_edge: float,
    threshold_calibration_n: int = 0,
) -> dict[str, Any]:
    """Return gate diagnostics and an optional auto-entry block reason."""
    mprob = _clamp_prob(model_prob)
    market = _clamp_prob(market_prob)
    cost = max(0.0, float(exec_cost or 0.0))
    true_edge = float(model_true_edge or 0.0)
    w = market_quality_weight(
        market_snapshot_age_s=market_snapshot_age_s,
        spread=spread,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
    )
    posterior = sigmoid((1.0 - w) * logit(mprob) + w * logit(market))
    posterior_edge = posterior - market - cost
    gap = abs(mprob - market)
    model_edge_ok = true_edge >= min_true_edge
    posterior_edge_ok = posterior_edge >= 0.02
    strong_exact_calibration = int(threshold_calibration_n or 0) >= 100
    gap_ok = True if w <= 0.0 else (gap <= 0.20 or strong_exact_calibration)

    blocked = False
    reason = None
    if model_edge_ok and not posterior_edge_ok:
        blocked = True
        reason = "posterior_edge"
    elif model_edge_ok and not gap_ok:
        blocked = True
        reason = "probability_gap"

    failure = None
    if blocked:
        failure = (
            "GATE_MARKET_SANITY: "
            f"posterior_edge={posterior_edge:.3f} "
            f"gap={gap:.2f} weight={w:.2f}"
        )

    return {
        "blocked": blocked,
        "failure": failure,
        "block_reason": reason,
        "weight": round(w, 4),
        "posterior_prob": round(posterior, 6),
        "posterior_edge": round(posterior_edge, 6),
        "model_prob": round(mprob, 6),
        "market_prob": round(market, 6),
        "model_true_edge": round(true_edge, 6),
        "exec_cost": round(cost, 6),
        "gap": round(gap, 6),
        "model_edge_ok": model_edge_ok,
        "posterior_edge_ok": posterior_edge_ok,
        "gap_ok": gap_ok,
        "threshold_calibration_sample_count": int(threshold_calibration_n or 0),
        "market_snapshot_age_s": (
            round(float(market_snapshot_age_s), 1)
            if market_snapshot_age_s is not None
            else None
        ),
        "spread": round(float(spread), 6) if spread is not None else None,
        "bid_depth": round(float(bid_depth or 0.0), 3),
        "ask_depth": round(float(ask_depth or 0.0), 3),
        "gate_only": True,
    }
