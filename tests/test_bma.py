"""Phase M1 — Bayesian Model Averaging predictive distribution."""
from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from backend.modeling.bma import (
    BMA_MIN_N_FOR_SIGMA,
    BMA_PRIOR_SIGMA_F,
    BMAComponent,
    BMAPredictive,
    bma_bucket_probabilities,
    build_bma_predictive,
    predictive_to_dict,
)


# ─────────────────────── BMAPredictive moments ───────────────────────────────

def test_single_component_mean_and_variance_match_underlying_gaussian():
    """A 1-component mixture is just N(μ, σ²) — moments must match exactly."""
    p = BMAPredictive(components=[
        BMAComponent(source="hrrr", mu=82.0, sigma=2.5, weight=1.0),
    ])
    assert p.mean == pytest.approx(82.0)
    assert p.variance == pytest.approx(2.5 ** 2)
    assert p.sigma == pytest.approx(2.5)
    # No disagreement between sources → between-source variance share is 0.
    assert p.is_bimodal_indicator == pytest.approx(0.0)


def test_two_component_equal_weight_means_and_variance_closed_form():
    """Two equal-weight kernels at μ=80 σ=2 and μ=84 σ=2.

    Mean: 0.5·80 + 0.5·84 = 82
    Within-source var: 0.5·4 + 0.5·4 = 4
    Between-source var: 0.5·(80-82)² + 0.5·(84-82)² = 0.5·4 + 0.5·4 = 4
    Total: 8 → σ = √8 ≈ 2.828
    Bimodal indicator: between/total = 4/8 = 0.5
    """
    p = BMAPredictive(components=[
        BMAComponent(source="hrrr", mu=80.0, sigma=2.0, weight=1.0),
        BMAComponent(source="ifs", mu=84.0, sigma=2.0, weight=1.0),
    ])
    assert p.mean == pytest.approx(82.0)
    assert p.variance == pytest.approx(8.0)
    assert p.sigma == pytest.approx(math.sqrt(8.0))
    assert p.is_bimodal_indicator == pytest.approx(0.5)


def test_weights_normalized_in_post_init():
    """Caller may pass un-normalized weights; mixture must self-normalize."""
    p = BMAPredictive(components=[
        BMAComponent(source="a", mu=80.0, sigma=2.0, weight=2.0),
        BMAComponent(source="b", mu=80.0, sigma=2.0, weight=2.0),
    ])
    assert p.components[0].weight == pytest.approx(0.5)
    assert p.components[1].weight == pytest.approx(0.5)


def test_pdf_sums_to_one_via_numeric_integration():
    """Mixture PDF must integrate to 1 across the support."""
    p = BMAPredictive(components=[
        BMAComponent(source="a", mu=78.0, sigma=2.0, weight=1.0),
        BMAComponent(source="b", mu=86.0, sigma=2.0, weight=1.0),
    ])
    # Riemann sum over [50, 110] with dx=0.1 — well outside both kernels' support.
    dx = 0.1
    grid = [50.0 + i * dx for i in range(601)]
    integral = sum(p.pdf(t) * dx for t in grid)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_cdf_monotone_and_bounded():
    p = BMAPredictive(components=[
        BMAComponent(source="a", mu=80.0, sigma=2.0, weight=1.0),
        BMAComponent(source="b", mu=85.0, sigma=3.0, weight=1.0),
    ])
    assert p.cdf(40.0) == pytest.approx(0.0, abs=1e-6)
    assert p.cdf(140.0) == pytest.approx(1.0, abs=1e-6)
    # Monotone:
    last = -1.0
    for t in [60.0, 70.0, 80.0, 85.0, 90.0, 100.0]:
        v = p.cdf(t)
        assert v >= last
        last = v


# ─────────────────── Bucket probabilities ────────────────────────────────────

def test_bma_bucket_probabilities_match_single_gaussian_when_one_component():
    """1-component BMA must equal the single-Gaussian bucket integration."""
    p = BMAPredictive(components=[
        BMAComponent(source="a", mu=82.0, sigma=2.5, weight=1.0),
    ])
    buckets = [(None, 80.0), (80.0, 85.0), (85.0, None)]
    bma_probs = bma_bucket_probabilities(p, buckets)
    expected = [
        norm.cdf(80.0, 82.0, 2.5),
        norm.cdf(85.0, 82.0, 2.5) - norm.cdf(80.0, 82.0, 2.5),
        1.0 - norm.cdf(85.0, 82.0, 2.5),
    ]
    for got, want in zip(bma_probs, expected):
        assert got == pytest.approx(want, abs=1e-4)


def test_bma_bucket_probabilities_sum_to_one():
    p = BMAPredictive(components=[
        BMAComponent(source="a", mu=78.0, sigma=2.0, weight=1.0),
        BMAComponent(source="b", mu=86.0, sigma=3.0, weight=1.0),
    ])
    buckets = [(None, 75.0), (75.0, 80.0), (80.0, 85.0), (85.0, 90.0), (90.0, None)]
    probs = bma_bucket_probabilities(p, buckets)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert all(p_ >= 0 for p_ in probs)


def test_bma_bucket_probabilities_bimodal_distributes_mass_to_both_modes():
    """Two far-apart components should give meaningful mass to both modes
    instead of collapsing to one peak around the consensus mean."""
    p = BMAPredictive(components=[
        BMAComponent(source="cool_model", mu=78.0, sigma=1.5, weight=1.0),
        BMAComponent(source="warm_model", mu=88.0, sigma=1.5, weight=1.0),
    ])
    buckets = [(None, 75.0), (75.0, 80.0), (80.0, 85.0), (85.0, 90.0), (90.0, None)]
    probs = bma_bucket_probabilities(p, buckets)
    # Cool mode (75-80) and warm mode (85-90) should each have substantial mass.
    assert probs[1] > 0.20, f"cool-mode mass too low: {probs[1]}"
    assert probs[3] > 0.20, f"warm-mode mass too low: {probs[3]}"
    # Middle bucket (80-85) — consensus mean lives here but mixture should NOT
    # put most mass there; that's the whole point of BMA over single-Gaussian.
    assert probs[2] < probs[1] + probs[3], (
        "single-Gaussian-on-mean would put all mass in middle bucket; "
        "BMA must spread mass to both modes"
    )


# ─────────────────── Predictive constructor ──────────────────────────────────

def test_build_predictive_uses_lead_skill_mae_when_well_supported():
    p = build_bma_predictive(
        calibrated_means={"hrrr": 82.0, "nws": 81.0},
        weights_by_source={"hrrr": 0.6, "nws": 0.4},
        lead_skill_mae_by_source={"hrrr": 1.8, "nws": 2.4},
        lead_skill_n_obs_by_source={"hrrr": 100, "nws": 80},
    )
    assert p.fallback_used is False
    sigmas = {c.source: c.sigma for c in p.components}
    assert sigmas["hrrr"] == pytest.approx(1.8)
    assert sigmas["nws"] == pytest.approx(2.4)


def test_build_predictive_falls_back_to_prior_when_low_n():
    p = build_bma_predictive(
        calibrated_means={"hrrr": 82.0, "nws": 81.0},
        weights_by_source={"hrrr": 0.5, "nws": 0.5},
        lead_skill_mae_by_source={"hrrr": 0.5, "nws": 2.4},
        lead_skill_n_obs_by_source={"hrrr": 5, "nws": 80},  # hrrr below threshold
    )
    assert p.fallback_used is True
    sigmas = {c.source: c.sigma for c in p.components}
    assert sigmas["hrrr"] == pytest.approx(BMA_PRIOR_SIGMA_F)
    assert sigmas["nws"] == pytest.approx(2.4)


def test_build_predictive_drops_zero_weight_sources():
    p = build_bma_predictive(
        calibrated_means={"a": 80.0, "b": 85.0, "c": 90.0},
        weights_by_source={"a": 0.5, "b": 0.0, "c": 0.5},
    )
    sources = [c.source for c in p.components]
    assert "b" not in sources
    assert set(sources) == {"a", "c"}


def test_predictive_to_dict_serializable():
    p = build_bma_predictive(
        calibrated_means={"hrrr": 82.0, "ifs": 80.0},
        weights_by_source={"hrrr": 0.6, "ifs": 0.4},
        lead_skill_mae_by_source={"hrrr": 1.8, "ifs": 2.0},
        lead_skill_n_obs_by_source={"hrrr": 100, "ifs": 90},
    )
    d = predictive_to_dict(p)
    assert d["n_components"] == 2
    assert d["fallback_used"] is False
    assert len(d["components"]) == 2
    assert {c["source"] for c in d["components"]} == {"hrrr", "ifs"}
    # JSON-roundtrip safe (no numpy types lurking):
    import json
    json.dumps(d)
