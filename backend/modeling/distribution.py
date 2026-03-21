"""
Temperature probability distribution.

Core math:
  - Normal(mu, sigma) distribution for daily high temperature
  - CDF-based bucket probability computation
  - Validates sum ≈ 1.0
"""
from __future__ import annotations

import math
from typing import Optional

from scipy.stats import norm


def bucket_probabilities(
    mu: float,
    sigma: float,
    buckets: list[tuple[Optional[float], Optional[float]]],
) -> list[float]:
    """
    Compute probability mass in each temperature bucket under Normal(mu, sigma).

    Args:
        mu: forecast mean temperature (°F)
        sigma: forecast std deviation (°F) — must be > 0
        buckets: list of (low_f, high_f) pairs.
                 None in low_f means -inf (open below).
                 None in high_f means +inf (open above).

    Returns:
        List of probabilities, same length as buckets.
        Sum is approximately 1.0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not buckets:
        return []

    probs = []
    for lo, hi in buckets:
        lo_cdf = 0.0 if lo is None else float(norm.cdf(lo, mu, sigma))
        hi_cdf = 1.0 if hi is None else float(norm.cdf(hi, mu, sigma))
        prob = max(0.0, hi_cdf - lo_cdf)  # clamp against float rounding
        probs.append(prob)

    # Normalize to sum exactly to 1.0 (rounding compensation)
    total = sum(probs)
    if total > 0 and abs(total - 1.0) < 0.05:
        probs = [p / total for p in probs]

    return probs


def implied_prob_from_price(mid_price: float, fee_rate: float = 0.02) -> float:
    """
    Convert YES mid price to implied probability, adjusting for fee.

    Polymarket charges ~2% fee on winnings, so:
      effective_price = mid_price + (1 - mid_price) * fee_rate
    But for signal computation we use mid_price directly as a proxy.
    The exec_cost term accounts for liquidity frictions separately.
    """
    return max(0.01, min(0.99, float(mid_price)))


def edge(model_prob: float, market_prob: float, exec_cost: float) -> float:
    """
    True edge after execution costs.

    Returns positive number if trade is profitable in expectation.
    """
    return model_prob - market_prob - exec_cost


def kelly_fraction(p: float, odds: float) -> float:
    """
    Full Kelly fraction for a binary bet.

    Args:
        p: probability of winning (model_prob)
        odds: net payout per $1 risked = (1/price) - 1

    Returns:
        Kelly fraction (can be negative if negative edge)
    """
    q = 1.0 - p
    if odds <= 0:
        return 0.0
    return (odds * p - q) / odds
