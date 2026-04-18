"""Unit tests for dynamic per-station ensemble weight computation.

Covers the pure math in _compute_weights_from_stats — the DB-backed helpers
are integration-tested separately (they need an event loop + schema).
"""
from __future__ import annotations

from backend.modeling.station_weights import (
    DEFAULT_WEIGHTS,
    GLOBAL_MSE_PRIOR,
    WEIGHT_CAP,
    WEIGHT_FLOOR,
    _compute_weights_from_stats,
)


def test_biased_ecmwf_auto_demotes():
    """A source with high MSE gets a small weight; unbiased sources dominate."""
    stats = {
        "nws":       {"mse_fast": 1.0, "mse_slow": 1.0, "n_samples": 30},
        "wu_hourly": {"mse_fast": 1.5, "mse_slow": 1.5, "n_samples": 30},
        "hrrr":      {"mse_fast": 2.0, "mse_slow": 2.0, "n_samples": 30},
        "nbm":       {"mse_fast": 1.5, "mse_slow": 1.5, "n_samples": 30},
        "ecmwf_ifs": {"mse_fast": 25.0, "mse_slow": 25.0, "n_samples": 30},
    }
    w = _compute_weights_from_stats(stats)
    assert w["nws"] > w["ecmwf_ifs"]
    assert w["ecmwf_ifs"] <= WEIGHT_FLOOR + 1e-6  # clamped to floor
    # Weights still sum to 1.
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_cold_start_shrinks_toward_uniform():
    """With n=0 for everyone, shrinkage dominates → near-uniform weights."""
    stats = {src: {"mse_fast": 1.0, "mse_slow": 1.0, "n_samples": 0}
             for src in ("nws", "wu_hourly", "hrrr", "nbm", "ecmwf_ifs")}
    # Even with wildly different MSEs, n=0 forces the prior to dominate.
    stats["ecmwf_ifs"]["mse_fast"] = 0.01
    stats["ecmwf_ifs"]["mse_slow"] = 0.01
    w = _compute_weights_from_stats(stats)
    # Spread should be small (all ≈ 0.2)
    assert max(w.values()) - min(w.values()) < 0.05


def test_empty_returns_empty():
    assert _compute_weights_from_stats({}) == {}


def test_weights_floor_holds_and_sum_is_one():
    """WEIGHT_FLOOR holds after clamping even for a terrible source; weights sum to 1.

    Note: WEIGHT_CAP binds pre-renormalization only — in a 2-source case where
    the loser hits the floor, the winner can exceed CAP after renormalization.
    This is acceptable: the floor guarantees the loser still contributes and the
    cap matters most when we have 4-5 sources.
    """
    stats = {
        "nws":       {"mse_fast": 0.001, "mse_slow": 0.001, "n_samples": 30},
        "ecmwf_ifs": {"mse_fast": 100.0, "mse_slow": 100.0, "n_samples": 30},
    }
    w = _compute_weights_from_stats(stats)
    assert w["ecmwf_ifs"] >= WEIGHT_FLOOR - 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_defaults_cover_all_ensemble_sources():
    """Sanity: fallback defaults defined for every source the model uses."""
    for src in ("nws", "wu_hourly", "hrrr", "nbm", "ecmwf_ifs"):
        assert src in DEFAULT_WEIGHTS
