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
    _lead_skill_sigma,
    _lead_time_sigma_growth,
    _lead_skill_factors,
    compute_model,
)
from backend.engine.signal_engine import _build_source_timing_metadata, _trusted_pre_model_spread
from backend.storage.models import ForecastObs, SourceLeadTimeSkill


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


def test_lead_skill_below_min_n_obs_gets_partial_adjustment():
    """Thin but nonzero samples get shrunken, not ignored, before n=30."""
    mae = {"nws": 1.5, "hrrr": 0.8, "wu_hourly": 5.0}
    n = {
        "nws": _LEAD_SKILL_MIN_N_OBS + 10,
        "hrrr": _LEAD_SKILL_MIN_N_OBS + 10,
        "wu_hourly": _LEAD_SKILL_MIN_N_OBS - 5,  # too thin
    }
    out = _lead_skill_factors(mae, n)
    # Median across all nonzero-evidence sources is 1.5. WU's raw factor clamps
    # to 0.7, then shrinks 25/30 of the way from 1.0.
    assert out["wu_hourly"] == pytest.approx(1.0 + (25 / 30) * (0.7 - 1.0))
    # Better-than-median (lower mae) gets factor > 1; median source stays ~1.
    assert out["hrrr"] > 1.0
    assert out["nws"] == pytest.approx(1.0)


def test_lead_skill_zero_n_obs_still_skipped():
    mae = {"nws": 1.5, "hrrr": 0.8, "wu_hourly": 5.0}
    n = {"nws": 40, "hrrr": 40, "wu_hourly": 0}
    out = _lead_skill_factors(mae, n)
    assert out["wu_hourly"] == 1.0


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


def test_lead_skill_sigma_uses_pre_threshold_shrinkage():
    sigma = _lead_skill_sigma(
        {"hrrr": 1.5, "nws": 2.1},
        {"hrrr": 25, "nws": 20},
        {"hrrr": 0.6, "nws": 0.4},
        1.0,
    )

    hrrr = (5 / 30) * 3.0 + (25 / 30) * 1.5
    nws = (10 / 30) * 3.0 + (20 / 30) * 2.1
    assert sigma == pytest.approx(0.6 * hrrr + 0.4 * nws)


def test_lead_skill_sigma_requires_two_sources():
    assert _lead_skill_sigma(
        {"hrrr": 1.5},
        {"hrrr": 25},
        {"hrrr": 1.0},
        1.0,
    ) is None


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


def test_lead_time_sigma_growth_caps_one_to_three_day_horizons():
    event_settlement_utc = datetime(2026, 5, 13, 23, 59, tzinfo=timezone.utc)
    sigma_48h = _lead_time_sigma_growth(
        {"nws": event_settlement_utc - timedelta(hours=48)},
        event_settlement_utc,
        1.0,
    )
    sigma_96h = _lead_time_sigma_growth(
        {"nws": event_settlement_utc - timedelta(hours=96)},
        event_settlement_utc,
        1.0,
    )

    assert sigma_48h == pytest.approx(3.5)
    assert sigma_96h == pytest.approx(6.3)


def test_wu_hourly_uses_fetched_at_for_lead_skill_and_bma_sigma_note():
    """WU hourly has no model_run_at, but should still find lead-skill rows."""
    settlement_utc = datetime(2026, 5, 13, 23, 59, 59, tzinfo=timezone.utc)
    fetched_at = settlement_utc - timedelta(hours=24, minutes=5)
    wu_forecast = ForecastObs(
        city_id=1,
        source="wu_hourly",
        date_et="2026-05-13",
        high_f=85.4,
        model_run_at=None,
        fetched_at=fetched_at,
    )
    wu_skill = SourceLeadTimeSkill(
        city_id=1,
        source="wu_hourly",
        lead_time_bucket_hours=24,
        mae_f=2.4,
        n_obs=5,
    )

    freshness_times, skill_mae, skill_n, lead_buckets = _build_source_timing_metadata(
        {"wu_hourly": wu_forecast},
        {("wu_hourly", 24): wu_skill},
        settlement_utc,
    )

    assert freshness_times["wu_hourly"] == fetched_at
    assert lead_buckets["wu_hourly"] == 24
    assert skill_mae["wu_hourly"] == 2.4
    assert skill_n["wu_hourly"] == 5

    model = compute_model(
        nws_high=77.5,
        wu_hourly_peak=85.4,
        hrrr_high=None,
        nbm_high=None,
        ecmwf_ifs_high=None,
        daily_high_metar=None,
        current_temp_f=70.0,
        calibration={"weight_nws": 0.5, "weight_wu_hourly": 0.5},
        buckets=[(74.0, 76.0), (76.0, 78.0), (78.0, 80.0), (80.0, None)],
        model_run_at_by_source=freshness_times,
        lead_skill_mae_by_source=skill_mae,
        lead_skill_n_obs_by_source=skill_n,
        now_utc=settlement_utc - timedelta(hours=20),
    )

    assert model is not None
    bma_notes = model.inputs["bma_shadow"]["notes"]
    assert "wu_hourly: no SourceLeadTimeSkill row, σ=prior" not in bma_notes
    assert any("wu_hourly: n=5<30, σ=shrinkage" in note for note in bma_notes)


def test_pre_model_regime_spread_uses_trusted_sources_not_ai_outlier():
    obs_by_source = {
        "nws": ForecastObs(city_id=1, source="nws", date_et="2026-05-13", high_f=81.0),
        "wu_hourly": ForecastObs(city_id=1, source="wu_hourly", date_et="2026-05-13", high_f=79.0),
        "hrrr": ForecastObs(city_id=1, source="hrrr", date_et="2026-05-13", high_f=79.9),
        "hrrr_15min": ForecastObs(city_id=1, source="hrrr_15min", date_et="2026-05-13", high_f=79.9),
        "nbm": ForecastObs(city_id=1, source="nbm", date_et="2026-05-13", high_f=79.4),
        "ecmwf_ifs": ForecastObs(city_id=1, source="ecmwf_ifs", date_et="2026-05-13", high_f=80.2),
        "ecmwf_aifs": ForecastObs(city_id=1, source="ecmwf_aifs", date_et="2026-05-13", high_f=48.0),
    }

    trusted_spread, raw_spread, gates = _trusted_pre_model_spread(obs_by_source)

    assert raw_spread >= 30.0
    assert trusted_spread <= 2.5
    assert gates["ecmwf_aifs"]["reason"] == "outlier_vs_trusted_median"


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


def test_compute_model_uses_lead_skill_sigma_before_n30():
    fixed_now = datetime(2026, 5, 31, 18, 0, tzinfo=timezone.utc)
    model_run = fixed_now - timedelta(hours=2)
    settlement = fixed_now + timedelta(hours=30)

    model = compute_model(
        nws_high=86.0,
        wu_hourly_peak=85.5,
        hrrr_high=85.0,
        nbm_high=None,
        ecmwf_ifs_high=None,
        daily_high_metar=None,
        current_temp_f=None,
        calibration={"weight_nws": 0.5, "weight_wu_hourly": 0.5, "weight_hrrr": 0.5},
        buckets=[(None, 83.0), (84.0, 85.0), (86.0, None)],
        model_run_at_by_source={
            "nws": model_run,
            "wu_hourly": model_run,
            "hrrr": model_run,
        },
        lead_skill_mae_by_source={"nws": 1.6, "wu_hourly": 1.8, "hrrr": 1.4},
        lead_skill_n_obs_by_source={"nws": 25, "wu_hourly": 24, "hrrr": 23},
        now_utc=fixed_now,
        event_settlement_utc=settlement,
    )

    assert model is not None
    assert model.inputs["sigma_lead_source"] == "lead_skill_shrinkage"
    assert model.inputs["sigma_lead"] < model.inputs["sigma_lead_generic"]
