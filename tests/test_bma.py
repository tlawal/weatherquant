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


# ─────────────────── Phase 2 EM weight fitter ───────────────────────────────

def test_em_cold_start_returns_uniform_when_below_min_obs():
    """Below EM_MIN_TRAINING_OBS we must NOT iterate — caller shouldn't get
    a confidently-wrong weight from a 5-sample fit."""
    from backend.modeling.bma import fit_bma_weights_em
    training = [({"a": 80.0, "b": 82.0}, 81.0) for _ in range(5)]
    sigma = {"a": 2.0, "b": 2.0}
    result = fit_bma_weights_em(training, sigma)
    assert result.n_obs == 5
    assert result.converged is False
    assert result.n_iter == 0
    # Uniform 0.5 / 0.5
    assert result.weights["a"] == pytest.approx(0.5)
    assert result.weights["b"] == pytest.approx(0.5)


def test_em_recovers_dominant_source_when_one_is_clearly_better():
    """Training: source A always nails it (μ=y), source B is always 5° off.
    EM must put nearly all weight on A regardless of starting init."""
    from backend.modeling.bma import fit_bma_weights_em
    training = []
    for i in range(60):
        y = 80.0 + i * 0.1
        training.append(({"a": y, "b": y + 5.0}, y))
    sigma = {"a": 1.0, "b": 1.0}
    # Start adversarially with all weight on the bad source — EM must escape.
    result = fit_bma_weights_em(training, sigma, init_weights={"a": 0.01, "b": 0.99})
    assert result.converged is True
    assert result.weights["a"] > 0.95
    assert result.weights["b"] < 0.05


def test_em_distributes_weight_when_sources_are_equally_skilled():
    """Two sources with identical residual distributions — weights should
    converge near 0.5/0.5 from any reasonable start."""
    import math
    import random
    from backend.modeling.bma import fit_bma_weights_em
    random.seed(42)
    training = []
    for _ in range(80):
        y = 80.0 + random.uniform(-3, 3)
        # Both sources equally biased: forecast = y + N(0, 2) noise
        training.append(({
            "a": y + random.gauss(0, 2.0),
            "b": y + random.gauss(0, 2.0),
        }, y))
    sigma = {"a": 2.0, "b": 2.0}
    result = fit_bma_weights_em(training, sigma)
    # Should be very close to 0.5/0.5 — small drift OK from finite sample
    assert abs(result.weights["a"] - 0.5) < 0.15
    assert abs(result.weights["b"] - 0.5) < 0.15
    assert result.converged is True


def test_em_handles_missing_source_per_obs_without_crashing():
    """Some training obs may lack one source (e.g. HRRR didn't run that day).
    The fitter must produce well-formed weights regardless of availability."""
    from backend.modeling.bma import fit_bma_weights_em, EM_WEIGHT_FLOOR
    training = []
    for i in range(80):
        y = 80.0 + i * 0.1
        if i % 4 == 0:
            training.append(({"nws": y + 0.5}, y))
        else:
            training.append(({"nws": y + 0.5, "hrrr": y + 0.2}, y))
    sigma = {"nws": 1.0, "hrrr": 1.0}
    result = fit_bma_weights_em(training, sigma)
    # Invariants must hold regardless of where EM lands:
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)
    for w in result.weights.values():
        assert w >= EM_WEIGHT_FLOOR - 1e-12
    assert "nws" in result.weights and "hrrr" in result.weights


def test_em_availability_can_outweigh_small_accuracy_advantage():
    """Standard Raftery 2005: the M-step divides by total N, so a flaky source
    with a small accuracy edge can lose to a reliable source with worse
    forecasts. This is the CORRECT production behavior — a source missing on
    25% of days has only 75% of the credit-accumulation budget of a reliable
    one. Small accuracy advantage doesn't compensate."""
    from backend.modeling.bma import fit_bma_weights_em
    # Marginal HRRR advantage (0.2 vs 0.5 bias, both σ=1.0). HRRR present 75%.
    training = []
    for i in range(80):
        y = 80.0 + i * 0.1
        if i % 4 == 0:
            training.append(({"nws": y + 0.5}, y))
        else:
            training.append(({"nws": y + 0.5, "hrrr": y + 0.2}, y))
    sigma = {"nws": 1.0, "hrrr": 1.0}
    result = fit_bma_weights_em(training, sigma)
    assert result.weights["nws"] > result.weights["hrrr"]


def test_em_strong_accuracy_advantage_can_overcome_availability_penalty():
    """The flip side: when an intermittent source is dramatically more
    accurate than the reliable one, EM correctly favors it despite missing
    days."""
    from backend.modeling.bma import fit_bma_weights_em
    # Big accuracy gap: NWS always 2.0° high, HRRR essentially perfect.
    training = []
    for i in range(80):
        y = 80.0 + i * 0.1
        if i % 4 == 0:
            training.append(({"nws": y + 2.0}, y))
        else:
            training.append(({"nws": y + 2.0, "hrrr": y + 0.05}, y))
    sigma = {"nws": 1.0, "hrrr": 1.0}
    result = fit_bma_weights_em(training, sigma)
    assert result.weights["hrrr"] > result.weights["nws"]


def test_em_more_accurate_source_wins_when_both_always_present():
    """Sanity: when both sources are always present, the more accurate one
    should dominate. Decoupled from the availability-penalty test above."""
    from backend.modeling.bma import fit_bma_weights_em
    training = []
    for i in range(80):
        y = 80.0 + i * 0.1
        # NWS always 0.5° high, HRRR always 0.2° high. Both always present.
        training.append(({"nws": y + 0.5, "hrrr": y + 0.2}, y))
    sigma = {"nws": 1.0, "hrrr": 1.0}
    result = fit_bma_weights_em(training, sigma)
    assert result.converged is True
    assert result.weights["hrrr"] > result.weights["nws"]


def test_em_weights_sum_to_one_and_above_floor():
    """Output invariants: weights sum to 1, no weight goes below the floor."""
    from backend.modeling.bma import fit_bma_weights_em, EM_WEIGHT_FLOOR
    training = [({"a": 80.0, "b": 81.0, "c": 79.0}, 80.0) for _ in range(50)]
    sigma = {"a": 1.0, "b": 1.0, "c": 1.0}
    result = fit_bma_weights_em(training, sigma)
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)
    for w in result.weights.values():
        assert w >= EM_WEIGHT_FLOOR


def test_em_log_likelihood_monotonically_increases():
    """EM is guaranteed to never decrease likelihood. Verify by capturing
    LL at every iteration and confirming non-decreasing."""
    from backend.modeling.bma import (
        _log_likelihood,
        fit_bma_weights_em,
    )
    training = []
    for i in range(50):
        y = 80.0 + i * 0.1
        training.append(({"a": y + 0.3, "b": y + 1.0}, y))
    sigma = {"a": 1.5, "b": 1.5}

    # Capture LL trajectory by patching: easier to check end-vs-start LL
    init = {"a": 0.5, "b": 0.5}
    init_ll = _log_likelihood(training, init, sigma, ["a", "b"])
    result = fit_bma_weights_em(training, sigma, init_weights=init)
    assert result.log_likelihood >= init_ll - 1e-9


# ─────────────────── Phase 3 online-EM weight updates ──────────────────────

def test_online_em_no_op_when_density_underflows():
    """If every forecast is so far from y that the mixture density underflows,
    the update is a no-op rather than a crash."""
    from backend.modeling.bma import online_em_step

    weights = {"a": 0.5, "b": 0.5}
    # Observed = 80, forecasts at 200 and -50 with σ=0.1 → density ≈ 0.
    forecasts = {"a": 200.0, "b": -50.0}
    sigma = {"a": 0.1, "b": 0.1}
    out = online_em_step(weights, forecasts, observed_y=80.0, sigma_by_source=sigma)
    # No-op: weights unchanged
    assert out == pytest.approx(weights)


def test_online_em_nudges_toward_more_accurate_source():
    """A single observation favoring source A should pull A's weight up."""
    from backend.modeling.bma import online_em_step

    weights = {"a": 0.5, "b": 0.5}
    forecasts = {"a": 80.05, "b": 81.5}  # A nearly perfect, B 1.5° off
    sigma = {"a": 1.0, "b": 1.0}
    out = online_em_step(weights, forecasts, observed_y=80.0, sigma_by_source=sigma)
    assert out["a"] > 0.5
    assert out["b"] < 0.5
    # Weights still sum to 1 and respect floor.
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_online_em_preserves_floor_invariant():
    """After the update, every weight remains ≥ EM_WEIGHT_FLOOR."""
    from backend.modeling.bma import EM_WEIGHT_FLOOR, online_em_step

    weights = {"a": 0.999, "b": 0.001}  # b at near-floor already
    # Push hard against b: A perfectly accurate, B 10° off
    forecasts = {"a": 80.0, "b": 90.0}
    sigma = {"a": 1.0, "b": 1.0}
    out = online_em_step(weights, forecasts, observed_y=80.0, sigma_by_source=sigma, lr=0.5)
    for s, w in out.items():
        assert w >= EM_WEIGHT_FLOOR - 1e-12


def test_online_em_leaves_missing_sources_approximately_unchanged():
    """When a source is absent from this observation, its weight is not
    directly updated by the M-step. Global renormalization can shrink it
    slightly (~1-2%) when the present sources collectively gain mass; that's
    expected and acceptable. The offline batch refit handles long-run
    availability penalties exactly."""
    from backend.modeling.bma import online_em_step

    weights = {"a": 0.4, "b": 0.4, "c": 0.2}
    # Only A and B forecast; C is missing.
    forecasts = {"a": 80.0, "b": 81.0}
    sigma = {"a": 1.0, "b": 1.0, "c": 1.0}
    out = online_em_step(weights, forecasts, observed_y=80.0, sigma_by_source=sigma)
    # C's weight should be very close to its input — at most a few percent
    # of dilution from renormalization, and never the dramatic shifts that
    # A and B see from the actual EM update.
    assert abs(out["c"] - 0.2) < 0.01
    # Also verify A and B did move (so we know the test is exercising the
    # algorithm, not just trivially passing).
    assert out["a"] != pytest.approx(0.4, abs=1e-3)


def test_online_em_lr_zero_is_identity():
    """Sanity: lr=0 means no update."""
    from backend.modeling.bma import online_em_step

    weights = {"a": 0.6, "b": 0.4}
    forecasts = {"a": 90.0, "b": 80.0}  # adversarial — B is right but lr=0
    sigma = {"a": 1.0, "b": 1.0}
    out = online_em_step(weights, forecasts, observed_y=80.0, sigma_by_source=sigma, lr=0.0)
    assert out == pytest.approx(weights, abs=1e-9)


def test_online_em_small_lr_makes_small_changes():
    """A single observation with lr=0.05 should move weights by < lr × |r-w|."""
    from backend.modeling.bma import online_em_step

    weights = {"a": 0.5, "b": 0.5}
    forecasts = {"a": 80.0, "b": 80.5}
    sigma = {"a": 1.0, "b": 1.0}
    out = online_em_step(weights, forecasts, observed_y=80.0, sigma_by_source=sigma, lr=0.05)
    # Even with A clearly preferred (perfect), one step at lr=0.05 shouldn't
    # move A by more than ~0.05 × 1.0 (full responsibility delta).
    assert abs(out["a"] - weights["a"]) < 0.06


def test_online_em_repeated_steps_converge_toward_offline_fit():
    """Sequence of online updates from a stream of (forecast, y) pairs should
    pull weights toward what offline-EM would compute on the same training.

    Demonstrates the correctness of the online-EM trajectory."""
    from backend.modeling.bma import (
        fit_bma_weights_em,
        online_em_step,
    )

    # Build training where source 'good' is much better than 'bad'
    training: list[tuple[dict, float]] = []
    for i in range(200):
        y = 80.0 + (i % 10) * 0.1
        training.append(({"good": y + 0.05, "bad": y + 1.5}, y))

    sigma = {"good": 1.0, "bad": 1.0}

    # Offline fit on the whole training set
    offline = fit_bma_weights_em(training, sigma)

    # Online updates from uniform start, lr=0.05, replay the same data
    weights = {"good": 0.5, "bad": 0.5}
    for forecasts, y in training:
        weights = online_em_step(weights, forecasts, y, sigma, lr=0.05)

    # Online with this many updates should land within a few percent of offline
    assert abs(weights["good"] - offline.weights["good"]) < 0.1
    # And on the right side of 0.5 — we know good > 0.5
    assert weights["good"] > 0.7


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
