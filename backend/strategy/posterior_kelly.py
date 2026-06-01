"""Posterior-aware Kelly sizing helpers.

The normal Kelly path sizes from one bucket probability. For BMA mixtures that
single number can hide source disagreement: one source may put nearly all mass
in the bucket while most sources do not. This module computes component-level
Kelly fractions and returns a conservative weighted-median fraction for sizing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

from backend.strategy.kelly import calculate_kelly_fraction


@dataclass(frozen=True)
class PosteriorKellyResult:
    aggregate_prob: float
    aggregate_kelly_f: float
    conservative_kelly_f: float
    weighted_median_component_kelly_f: float
    p10_component_kelly_f: float
    min_component_prob: float
    median_component_prob: float
    max_component_prob: float
    component_count: int
    haircut_applied: bool

    def to_dict(self) -> dict:
        return {
            "aggregate_prob": round(self.aggregate_prob, 6),
            "aggregate_kelly_f": round(self.aggregate_kelly_f, 6),
            "conservative_kelly_f": round(self.conservative_kelly_f, 6),
            "weighted_median_component_kelly_f": round(
                self.weighted_median_component_kelly_f, 6
            ),
            "p10_component_kelly_f": round(self.p10_component_kelly_f, 6),
            "min_component_prob": round(self.min_component_prob, 6),
            "median_component_prob": round(self.median_component_prob, 6),
            "max_component_prob": round(self.max_component_prob, 6),
            "component_count": self.component_count,
            "haircut_applied": self.haircut_applied,
        }


def _component_bucket_probability(
    *,
    mu: float,
    sigma: float,
    low_f: Optional[float],
    high_f: Optional[float],
) -> float:
    lo_cdf = 0.0 if low_f is None else float(norm.cdf(float(low_f), mu, sigma))
    hi_cdf = 1.0 if high_f is None else float(norm.cdf(float(high_f), mu, sigma))
    return max(0.0, min(1.0, hi_cdf - lo_cdf))


def _weighted_quantile(values_and_weights: list[tuple[float, float]], q: float) -> float:
    rows = sorted(
        (float(value), max(0.0, float(weight)))
        for value, weight in values_and_weights
        if weight > 0
    )
    if not rows:
        return 0.0
    total = sum(weight for _, weight in rows)
    if total <= 0:
        return 0.0
    threshold = max(0.0, min(1.0, q)) * total
    acc = 0.0
    for value, weight in rows:
        acc += weight
        if acc >= threshold:
            return value
    return rows[-1][0]


def posterior_aware_kelly(
    *,
    bma_shadow: dict | None,
    low_f: Optional[float],
    high_f: Optional[float],
    yes_price: float,
    fractional_kelly: float,
    max_position_size: float,
) -> PosteriorKellyResult | None:
    """Compute a conservative Kelly fraction from BMA component disagreement.

    Returns None when BMA components are unavailable or the price is invalid.
    The returned Kelly fractions already include `fractional_kelly` but do not
    include external regime multipliers; the risk manager applies those later.
    """
    if not isinstance(bma_shadow, dict):
        return None
    if not (0.0 < float(yes_price) < 1.0):
        return None
    components = bma_shadow.get("components")
    if not isinstance(components, list) or not components:
        return None

    parsed: list[tuple[float, float, float]] = []
    for comp in components:
        try:
            mu = float(comp.get("mu"))
            sigma = float(comp.get("sigma"))
            weight = float(comp.get("weight"))
        except (TypeError, ValueError, AttributeError):
            continue
        if sigma <= 0 or weight <= 0:
            continue
        parsed.append((mu, sigma, weight))
    total_w = sum(weight for _, _, weight in parsed)
    if total_w <= 0:
        return None
    parsed = [(mu, sigma, weight / total_w) for mu, sigma, weight in parsed]

    component_probs: list[tuple[float, float]] = []
    component_kelly: list[tuple[float, float]] = []
    for mu, sigma, weight in parsed:
        p = _component_bucket_probability(
            mu=mu,
            sigma=sigma,
            low_f=low_f,
            high_f=high_f,
        )
        component_probs.append((p, weight))
        component_kelly.append((
            calculate_kelly_fraction(
                p,
                yes_price,
                fractional_kelly=fractional_kelly,
                max_position_size=max_position_size,
            ),
            weight,
        ))

    aggregate_prob = sum(p * weight for p, weight in component_probs)
    aggregate_kelly = calculate_kelly_fraction(
        aggregate_prob,
        yes_price,
        fractional_kelly=fractional_kelly,
        max_position_size=max_position_size,
    )
    median_kelly = _weighted_quantile(component_kelly, 0.50)
    p10_kelly = _weighted_quantile(component_kelly, 0.10)
    conservative_kelly = min(aggregate_kelly, median_kelly)

    return PosteriorKellyResult(
        aggregate_prob=aggregate_prob,
        aggregate_kelly_f=aggregate_kelly,
        conservative_kelly_f=conservative_kelly,
        weighted_median_component_kelly_f=median_kelly,
        p10_component_kelly_f=p10_kelly,
        min_component_prob=_weighted_quantile(component_probs, 0.0),
        median_component_prob=_weighted_quantile(component_probs, 0.50),
        max_component_prob=_weighted_quantile(component_probs, 1.0),
        component_count=len(component_probs),
        haircut_applied=conservative_kelly < aggregate_kelly,
    )
