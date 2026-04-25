"""Unit tests for Phase A4 — Kalman posterior σ blended into σ_final.

The blend lives in `_blend_kalman_sigma` so this test exercises the math
directly, without spinning up the full compute_model pipeline (which needs
forecasts, METAR obs, and calibration data).
"""
from __future__ import annotations

import math

import pytest

from backend.modeling.temperature_model import _blend_kalman_sigma


def test_blend_collapses_to_input_when_w_metar_zero():
    """At w_metar = 0 (early morning), Kalman contributes nothing."""
    out = _blend_kalman_sigma(
        sigma_final=2.0,
        w_metar=0.0,
        kalman_uncertainty=0.5,
        peak_hour_local=15.0,
        hour_local_fractional=8.0,
    )
    assert out == pytest.approx(2.0, abs=1e-9)


def test_blend_collapses_to_kalman_when_w_metar_one():
    """At w_metar = 1 (full ground-truth dominance), σ_final = kalman_σ."""
    # 0.5°F kalman uncertainty + 0.3*2.0=0.6°F drift → sqrt(0.25+0.36)=sqrt(0.61)≈0.781
    expected = math.sqrt(0.5 ** 2 + (0.3 * 2.0) ** 2)
    out = _blend_kalman_sigma(
        sigma_final=2.0,
        w_metar=1.0,
        kalman_uncertainty=0.5,
        peak_hour_local=15.0,
        hour_local_fractional=13.0,
    )
    assert out == pytest.approx(expected, abs=1e-6)


def test_blend_tightens_when_kalman_is_confident():
    """A small Kalman σ pulls σ_final down at non-zero METAR weight."""
    pre = 2.0
    out = _blend_kalman_sigma(
        sigma_final=pre,
        w_metar=0.7,
        kalman_uncertainty=0.2,  # very confident filter
        peak_hour_local=15.0,
        hour_local_fractional=14.5,  # close to peak → small drift
    )
    assert out < pre


def test_blend_widens_when_kalman_is_uncertain():
    """A large Kalman σ inflates σ_final."""
    pre = 1.5
    out = _blend_kalman_sigma(
        sigma_final=pre,
        w_metar=0.7,
        kalman_uncertainty=4.0,  # very uncertain filter
        peak_hour_local=15.0,
        hour_local_fractional=10.0,
    )
    assert out > pre


def test_blend_drift_term_grows_with_hours_to_peak():
    """Same Kalman σ, but earlier in the day → more drift uncertainty → wider blend."""
    early = _blend_kalman_sigma(
        sigma_final=1.0, w_metar=1.0,
        kalman_uncertainty=0.3,
        peak_hour_local=16.0, hour_local_fractional=8.0,  # 8h to peak
    )
    late = _blend_kalman_sigma(
        sigma_final=1.0, w_metar=1.0,
        kalman_uncertainty=0.3,
        peak_hour_local=16.0, hour_local_fractional=15.5,  # 0.5h to peak
    )
    assert early > late


def test_blend_uses_floor_of_half_hour_for_drift():
    """Past-peak (negative hours_to_peak) is floored at 0.5 — drift never zero."""
    out = _blend_kalman_sigma(
        sigma_final=0.0, w_metar=1.0,
        kalman_uncertainty=0.0,
        peak_hour_local=10.0, hour_local_fractional=15.0,  # 5h past peak
    )
    # All from drift: 0.3 * 0.5 = 0.15
    assert out == pytest.approx(0.15, abs=1e-6)


def test_blend_handles_missing_peak_hour():
    """When peak_hour_local is None, falls back to a 2h drift assumption."""
    out = _blend_kalman_sigma(
        sigma_final=0.0, w_metar=1.0,
        kalman_uncertainty=0.0,
        peak_hour_local=None, hour_local_fractional=10.0,
    )
    # 0.3 * 2.0 = 0.6
    assert out == pytest.approx(0.6, abs=1e-6)


def test_blend_unit_mult_scales_drift_for_celsius():
    """Drift term is in °F natively; unit_mult=5/9 converts for Celsius cities."""
    out_c = _blend_kalman_sigma(
        sigma_final=0.0, w_metar=1.0,
        kalman_uncertainty=0.0,
        peak_hour_local=15.0, hour_local_fractional=13.0,  # 2h to peak
        unit_mult=5.0 / 9.0,
    )
    # 0.3 * 2.0 * (5/9) = 0.6 * 5/9 ≈ 0.333
    assert out_c == pytest.approx(0.6 * 5.0 / 9.0, abs=1e-6)


def test_blend_tightens_monotonically_as_kalman_uncertainty_drops():
    """Holding everything else fixed, smaller Kalman uncertainty → smaller σ_final."""
    pre = 2.0
    sigmas = []
    for k_u in [3.0, 2.0, 1.0, 0.5, 0.1]:
        sigmas.append(_blend_kalman_sigma(
            sigma_final=pre, w_metar=0.7,
            kalman_uncertainty=k_u,
            peak_hour_local=15.0, hour_local_fractional=14.0,
        ))
    # Strictly decreasing as Kalman tightens.
    for a, b in zip(sigmas, sigmas[1:]):
        assert a > b
