"""Phase B5 — weighted-variance ensemble σ."""
from __future__ import annotations

import math

import pytest

from backend.modeling.temperature_model import _ensemble_sigma


def test_falls_back_when_under_three_sources():
    """Under-3 sources → caller's fallback wins."""
    out = _ensemble_sigma([(80.0, 1.0), (82.0, 1.0)], fallback_sigma=1.5, sigma_floor=1.0)
    assert out == 1.5


def test_zero_weights_treated_as_no_evidence():
    out = _ensemble_sigma([(80.0, 0.0), (81.0, 0.0), (82.0, 0.0)], fallback_sigma=2.0, sigma_floor=1.0)
    assert out == 2.0


def test_floor_holds_when_forecasts_agree():
    """Three forecasts that all match → variance 0 but σ floored."""
    out = _ensemble_sigma(
        [(80.0, 1.0), (80.0, 1.0), (80.0, 1.0)],
        fallback_sigma=99.0,  # ignored when len>=3
        sigma_floor=1.0,
    )
    assert out == 1.0


def test_unweighted_three_sources_matches_population_stdev():
    """With equal weights, σ matches the population stdev (divisor n, not n-1)."""
    vals = [(78.0, 1.0), (80.0, 1.0), (82.0, 1.0)]
    out = _ensemble_sigma(vals, fallback_sigma=0.0, sigma_floor=0.0)
    # mu=80, var = ((-2)^2 + 0 + 2^2) / 3 = 8/3 → σ ≈ 1.633
    assert out == pytest.approx(math.sqrt(8.0 / 3.0), abs=1e-6)


def test_weights_pull_mu_and_widen_or_tighten_appropriately():
    """High-weight outlier inflates σ more than low-weight outlier of same delta."""
    base = [(80.0, 1.0), (80.0, 1.0)]
    low_weight_outlier = _ensemble_sigma(base + [(90.0, 0.1)], fallback_sigma=0.0, sigma_floor=0.0)
    high_weight_outlier = _ensemble_sigma(base + [(90.0, 1.0)], fallback_sigma=0.0, sigma_floor=0.0)
    assert high_weight_outlier > low_weight_outlier


def test_floor_overrides_tiny_disagreement():
    """A trivially small disagreement still respects the floor."""
    vals = [(80.0, 1.0), (80.05, 1.0), (80.1, 1.0)]
    out = _ensemble_sigma(vals, fallback_sigma=0.0, sigma_floor=1.0)
    assert out == 1.0
