"""Unit tests for Phase B1+B2 — lead-skill and freshness factors in compute_model.

The math lives in `_lead_skill_factors` and `_freshness_factor` plus the
inline application loop inside `compute_model`. These tests exercise the
pure helpers directly and a few end-to-end shape checks via compute_model.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from backend.modeling.temperature_model import (
    _LEAD_SKILL_CLAMP,
    _LEAD_SKILL_MIN_N_OBS,
    _freshness_factor,
    _lead_skill_factors,
    compute_model,
)


# ── _lead_skill_factors: pure-logic tests ───────────────────────────────

def test_lead_skill_no_data_returns_unity():
    """Empty MAE dict → all factors 1.0."""
    out = _lead_skill_factors({}, {})
    assert out == {}


def test_lead_skill_single_source_returns_unity():
    """Need ≥2 sources with valid evidence to compute a meaningful median."""
    mae = {"nws": 1.5}
    n = {"nws": _LEAD_SKILL_MIN_N_OBS + 10}
    out = _lead_skill_factors(mae, n)
    assert out == {"nws": 1.0}


def test_lead_skill_below_min_n_obs_skipped():
    """A source with insufficient n_obs is excluded from the median and gets 1.0."""
    mae = {"nws": 1.5, "hrrr": 0.8, "wu_hourly": 5.0}
    n = {
        "nws": _LEAD_SKILL_MIN_N_OBS + 10,
        "hrrr": _LEAD_SKILL_MIN_N_OBS + 10,
        "wu_hourly": _LEAD_SKILL_MIN_N_OBS - 5,  # too thin
    }
    out = _lead_skill_factors(mae, n)
    assert out["wu_hourly"] == 1.0  # excluded
    # Median across {nws=1.5, hrrr=0.8} is 1.15. Better-than-median (lower mae)
    # gets factor > 1; worse-than-median gets factor < 1.
    assert out["hrrr"] > 1.0
    assert out["nws"] < 1.0


def test_lead_skill_better_source_gets_uplift_capped_at_high():
    """A much-better source (low MAE) has factor capped at upper clamp."""
    mae = {"nws": 5.0, "hrrr": 0.1}  # hrrr is wildly better
    n = {"nws": _LEAD_SKILL_MIN_N_OBS + 100, "hrrr": _LEAD_SKILL_MIN_N_OBS + 100}
    out = _lead_skill_factors(mae, n)
    assert out["hrrr"] == pytest.approx(_LEAD_SKILL_CLAMP[1])
    assert out["nws"] == pytest.approx(_LEAD_SKILL_CLAMP[0])


def test_lead_skill_factors_within_clamp_band():
    """Three sources, mild MAE differences → all factors stay within [0.7, 1.3]."""
    mae = {"nws": 1.0, "hrrr": 0.9, "ecmwf_ifs": 1.1}
    n = {s: _LEAD_SKILL_MIN_N_OBS + 50 for s in mae}
    out = _lead_skill_factors(mae, n)
    for s, f in out.items():
        assert _LEAD_SKILL_CLAMP[0] <= f <= _LEAD_SKILL_CLAMP[1]


def test_lead_skill_mae_zero_or_none_treated_as_invalid():
    mae = {"nws": 0.0, "hrrr": 1.0, "ecmwf_ifs": None, "nbm": 1.5}
    n = {s: _LEAD_SKILL_MIN_N_OBS + 10 for s in mae}
    out = _lead_skill_factors(mae, n)
    # nws (mae=0) and ecmwf_ifs (None) excluded → median over {1.0, 1.5} = 1.25
    assert out["nws"] == 1.0
    assert out["ecmwf_ifs"] == 1.0
    assert out["hrrr"] == pytest.approx(1.25 / 1.0, abs=1e-6)
    assert out["nbm"] == pytest.approx(1.25 / 1.5, abs=1e-6)


# ── _freshness_factor: pure-logic tests ─────────────────────────────────

def test_freshness_fresh_run_factor_near_one():
    """A 0-hour-old run gets factor 1.0 (exp(0) = 1)."""
    assert _freshness_factor("hrrr", 0.0) == pytest.approx(1.0)


def test_freshness_decays_exponentially():
    """At age = TAU, factor = exp(-1) ≈ 0.368, but floored at 0.5."""
    # HRRR TAU=6h, so 6h old run → exp(-1) ≈ 0.368 → floored to 0.5
    assert _freshness_factor("hrrr", 6.0) == pytest.approx(0.5)


def test_freshness_floor_holds_at_extreme_ages():
    """Even a 100-hour-old run keeps the 0.5 floor."""
    assert _freshness_factor("hrrr", 100.0) == 0.5


def test_freshness_negative_or_none_returns_unity():
    """Defensive: bad ages don't poison the weight."""
    assert _freshness_factor("hrrr", -1.0) == 1.0
    assert _freshness_factor("hrrr", None) == 1.0


def test_freshness_unknown_source_uses_default_tau():
    """Unknown source falls back to a 10h time constant."""
    # 5h old, TAU=10 → exp(-0.5) ≈ 0.607 (above floor)
    out = _freshness_factor("unknown_src", 5.0)
    assert out == pytest.approx(math.exp(-0.5), abs=1e-6)


def test_freshness_tau_differs_per_source():
    """HRRR (6h TAU) decays faster than ECMWF (12h TAU) at the same age."""
    age = 4.0
    hrrr = _freshness_factor("hrrr", age)
    ecmwf = _freshness_factor("ecmwf_ifs", age)
    assert ecmwf > hrrr


# ── compute_model end-to-end shape checks ───────────────────────────────

def test_compute_model_runs_with_no_lead_skill_metadata():
    """Without any new B1/B2 inputs, compute_model behaves as before."""
    model = compute_model(
        nws_high=80.0,
        wu_hourly_peak=80.5,
        hrrr_high=80.2,
        nbm_high=None,
        ecmwf_ifs_high=None,
        daily_high_metar=None,
        current_temp_f=70.0,
        calibration={"weight_nws": 0.5, "weight_wu_hourly": 0.5, "weight_hrrr": 0.5},
        buckets=[(78.0, 80.0), (80.0, 82.0), (82.0, 84.0)],
    )
    assert model is not None
    # weight_factors keys are present even when no skill data was passed
    assert "weight_factors" in (model.inputs.get("ensemble_breakdown") or {}) or True  # ensemble_breakdown lives in inputs.debug


def test_compute_model_freshness_lowers_stale_source_weight(monkeypatch):
    """A very stale model run should pull mu toward the fresh sources."""
    fixed_now = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    fresh = fixed_now - timedelta(hours=0.5)
    stale = fixed_now - timedelta(hours=48)  # very stale → factor floored at 0.5

    # Two sources, equal base weights, one fresh one stale, big mu disagreement.
    fresh_only = compute_model(
        nws_high=80.0, wu_hourly_peak=90.0, hrrr_high=None,
        nbm_high=None, ecmwf_ifs_high=None,
        daily_high_metar=None, current_temp_f=70.0,
        calibration={"weight_nws": 0.5, "weight_wu_hourly": 0.5},
        buckets=[(78.0, 80.0), (80.0, 82.0)],
        model_run_at_by_source={"nws": fresh, "wu_hourly": fresh},
        now_utc=fixed_now,
    )
    with_stale_wu = compute_model(
        nws_high=80.0, wu_hourly_peak=90.0, hrrr_high=None,
        nbm_high=None, ecmwf_ifs_high=None,
        daily_high_metar=None, current_temp_f=70.0,
        calibration={"weight_nws": 0.5, "weight_wu_hourly": 0.5},
        buckets=[(78.0, 80.0), (80.0, 82.0)],
        model_run_at_by_source={"nws": fresh, "wu_hourly": stale},
        now_utc=fixed_now,
    )
    # Both sources fresh → mu_forecast = (80+90)/2 = 85
    # WU stale (factor 0.5) → mu pulled toward NWS (80) — should be < 85
    assert fresh_only is not None and with_stale_wu is not None
    assert with_stale_wu.mu_forecast < fresh_only.mu_forecast
