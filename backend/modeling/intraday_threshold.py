"""Intraday threshold-crossing probabilities for same-day high markets.

This module keeps the first production version deliberately conservative:
it produces a shadow probability distribution from monotone exceedance
probabilities, but does not replace the legacy trading probability path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from scipy.stats import norm


BucketRange = tuple[Optional[float], Optional[float]]


@dataclass
class IntradayThresholdResult:
    """Shadow intraday threshold model output."""

    probs: list[float]
    survival: dict[float, float]
    alpha: float
    mode: str
    features: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "alpha": round(self.alpha, 3),
            "probs": [round(p, 6) for p in self.probs],
            "survival": {
                _fmt_threshold(k): round(v, 6)
                for k, v in sorted(self.survival.items())
            },
            "features": self.features,
            "notes": self.notes,
        }


def _fmt_threshold(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _pava_increasing(values: list[float]) -> list[float]:
    """Pool-adjacent-violators algorithm for isotonic increasing fit."""
    if not values:
        return []

    levels: list[float] = []
    weights: list[float] = []
    lengths: list[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1.0)
        lengths.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_w = weights[-2] + weights[-1]
            merged_level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / merged_w
            merged_len = lengths[-2] + lengths[-1]
            levels[-2:] = [merged_level]
            weights[-2:] = [merged_w]
            lengths[-2:] = [merged_len]

    out: list[float] = []
    for level, length in zip(levels, lengths):
        out.extend([level] * length)
    return out


def enforce_monotone_survival(
    survival: dict[float, float],
) -> dict[float, float]:
    """Return threshold survival probabilities monotone in threshold.

    For ascending thresholds, P(H >= threshold) must be non-increasing. PAVA
    makes the smallest squared adjustment needed to restore that invariant.
    """
    if not survival:
        return {}
    thresholds = sorted(float(t) for t in survival)
    clipped = [_clamp01(survival[t]) for t in thresholds]
    decreasing = [-v for v in clipped]
    adjusted = [-v for v in _pava_increasing(decreasing)]
    return {
        t: _clamp01(v)
        for t, v in zip(thresholds, adjusted)
    }


def bucket_probs_from_survival(
    buckets: list[BucketRange],
    survival: dict[float, float],
    *,
    observed_high: Optional[float] = None,
) -> list[float]:
    """Convert exceedance probabilities S(t)=P(H>=t) into bucket masses."""
    if not buckets:
        return []

    monotone = enforce_monotone_survival(survival)

    def s_at(threshold: Optional[float]) -> float:
        if threshold is None:
            return 1.0
        threshold_f = float(threshold)
        if observed_high is not None and threshold_f <= observed_high:
            return 1.0
        return _clamp01(monotone.get(threshold_f, 0.0))

    probs: list[float] = []
    for lo, hi in buckets:
        if observed_high is not None and hi is not None and observed_high >= hi:
            probs.append(0.0)
            continue

        if lo is None and hi is None:
            prob = 1.0
        elif lo is None:
            prob = 1.0 - s_at(hi)
        elif hi is None:
            prob = s_at(lo)
        else:
            prob = s_at(lo) - s_at(hi)
        probs.append(max(0.0, prob))

    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    return probs


def _finite_thresholds(buckets: list[BucketRange]) -> list[float]:
    thresholds = {
        float(bound)
        for lo, hi in buckets
        for bound in (lo, hi)
        if bound is not None and math.isfinite(float(bound))
    }
    return sorted(thresholds)


def predict_intraday_threshold_probabilities(
    *,
    buckets: list[BucketRange],
    observed_high: Optional[float],
    current_temp_f: Optional[float],
    projected_high: float,
    consensus_high: float,
    sigma: float,
    remaining_rise: float,
    hour_local: float,
    peak_hour_local: Optional[float] = None,
    trend_per_hr: Optional[float] = None,
    trusted_spread: Optional[float] = None,
    forecast_quality: str = "ok",
    lock_regime: bool = False,
) -> Optional[IntradayThresholdResult]:
    """Estimate same-day bucket probabilities from threshold crossings.

    This is a physics-informed shadow prior, not a promoted learned model. It
    treats already-observed temperatures as hard lower bounds and makes
    near-consensus low thresholds easier to cross than a symmetric daily-high
    Normal would imply. A trained classifier can later replace the per-threshold
    survival estimate while keeping the same bucket conversion contract.
    """
    if not buckets or observed_high is None:
        return None
    if forecast_quality != "ok":
        return None

    thresholds = _finite_thresholds(buckets)
    if not thresholds:
        return None

    peak_h = peak_hour_local
    if peak_h is None:
        peak_h = 15.5
    hours_to_peak = max(0.0, peak_h - hour_local)
    trend = float(trend_per_hr or 0.0)
    current = float(current_temp_f) if current_temp_f is not None else float(observed_high)
    base_sigma = max(0.65, float(sigma))
    spread = max(0.0, float(trusted_spread or 0.0))

    # Before peak with a non-negative trend, use the stronger of forecast
    # consensus and observation projection for threshold-crossing center. This
    # avoids assigning excessive probability to buckets well below a tightly
    # clustered forecast panel on clear warming days.
    center = max(float(projected_high), float(consensus_high))
    if hours_to_peak > 0.25 and trend > 0:
        trend_projection = current + min(float(remaining_rise), trend * hours_to_peak)
        center = max(center, trend_projection)
        center += min(0.5, trend * 0.25)
    center = max(center, float(observed_high))

    survival: dict[float, float] = {}
    notes: list[str] = ["shadow_only", "threshold_survival_prior"]
    for threshold in thresholds:
        if threshold <= observed_high:
            survival[threshold] = 1.0
            continue

        margin = center - threshold
        # For thresholds below expected high, crossing uncertainty is narrower
        # than the symmetric daily-high bucket PDF. Wider forecast spread keeps
        # the prior from becoming overconfident on unsettled synoptic days.
        shrink = min(0.65, max(0.0, margin) * 0.18)
        spread_inflation = min(0.55, spread * 0.06)
        threshold_sigma = base_sigma * (1.0 - shrink + spread_inflation)
        threshold_sigma = max(0.65, min(base_sigma, threshold_sigma))
        if margin >= 0.0:
            threshold_sigma = min(threshold_sigma, max(0.65, 0.85 + spread * 0.02))
        if lock_regime:
            threshold_sigma = min(threshold_sigma, 0.35)

        survival[threshold] = _clamp01(1.0 - float(norm.cdf(threshold, center, threshold_sigma)))

    survival = enforce_monotone_survival(survival)
    probs = bucket_probs_from_survival(buckets, survival, observed_high=observed_high)

    # Alpha is the currently earned promotion weight for trading. Keep at zero
    # until offline validation proves improvement over market/legacy.
    alpha = 0.0
    features = {
        "observed_high": round(float(observed_high), 3),
        "current_temp_f": round(current, 3),
        "projected_high": round(float(projected_high), 3),
        "consensus_high": round(float(consensus_high), 3),
        "threshold_center": round(center, 3),
        "sigma": round(base_sigma, 3),
        "remaining_rise": round(float(remaining_rise), 3),
        "hour_local": round(float(hour_local), 3),
        "peak_hour_local": round(float(peak_h), 3),
        "hours_to_peak": round(hours_to_peak, 3),
        "trend_per_hr": round(trend, 3),
        "trusted_spread": round(spread, 3),
    }

    return IntradayThresholdResult(
        probs=probs,
        survival=survival,
        alpha=alpha,
        mode="physics_threshold_shadow",
        features=features,
        notes=notes,
    )
