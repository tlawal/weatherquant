"""Phase C3 — regime detector + Kelly multiplier."""
from __future__ import annotations

import pytest

from backend.modeling.regime import (
    RegimeLabel,
    detect_regime,
    regime_kelly_multiplier,
)


# ── detect_regime: pure-logic tests ─────────────────────────────────────

def test_quiet_day_classified_calm():
    """Tight ensemble + flat pressure + dry → CALM."""
    r = detect_regime(
        current_spread_f=0.5,
        historical_spreads_f=[0.4, 0.5, 0.6],
        pressure_tendency_inhg=0.005,
        has_precip=False,
    )
    assert r.label == RegimeLabel.CALM
    assert r.score < 0.25


def test_volatile_front_passage_classified_volatile():
    """Wide spread, rapidly growing, dropping pressure, precip → VOLATILE."""
    r = detect_regime(
        current_spread_f=6.0,
        historical_spreads_f=[2.0, 2.5, 3.0],
        pressure_tendency_inhg=-0.10,
        has_precip=True,
    )
    assert r.label == RegimeLabel.VOLATILE
    assert r.score >= 0.65


def test_mixed_signals_classified_normal():
    """Moderate spread + some growth → NORMAL band (not CALM, not VOLATILE)."""
    r = detect_regime(
        current_spread_f=3.5,
        historical_spreads_f=[2.0, 2.0, 2.5],
        pressure_tendency_inhg=0.03,
        has_precip=False,
    )
    assert r.label == RegimeLabel.NORMAL
    assert 0.25 <= r.score < 0.65


def test_missing_inputs_zero_weight_components():
    """All-None inputs collapse to score 0 → CALM (defensive default)."""
    r = detect_regime(current_spread_f=None)
    assert r.label == RegimeLabel.CALM
    assert r.score == 0.0


def test_score_monotonic_in_spread():
    """Higher spread → higher score, holding everything else fixed."""
    common = dict(historical_spreads_f=[1.0, 1.0, 1.0],
                  pressure_tendency_inhg=0.0, has_precip=False)
    s_low = detect_regime(current_spread_f=1.0, **common).score
    s_mid = detect_regime(current_spread_f=3.0, **common).score
    s_hi = detect_regime(current_spread_f=6.0, **common).score
    assert s_low < s_mid < s_hi


def test_growth_amplifies_score_vs_flat_history():
    """Same current spread, but rapid growth vs flat history → higher score."""
    flat = detect_regime(
        current_spread_f=4.0, historical_spreads_f=[4.0, 4.0, 4.0],
        pressure_tendency_inhg=0.0, has_precip=False,
    )
    grew = detect_regime(
        current_spread_f=4.0, historical_spreads_f=[1.0, 1.5, 2.0],
        pressure_tendency_inhg=0.0, has_precip=False,
    )
    assert grew.score > flat.score


def test_components_dict_exposes_subscores():
    """Telemetry: components dict captures each subscore for observability."""
    r = detect_regime(
        current_spread_f=4.0,
        historical_spreads_f=[1.0, 1.0, 1.0],
        pressure_tendency_inhg=0.06,
        has_precip=True,
    )
    for k in ("spread", "growth", "pressure", "precip", "current_spread_f", "historical_n"):
        assert k in r.components


# ── regime_kelly_multiplier: pure-logic tests ───────────────────────────

def test_kelly_multiplier_calm_returns_one():
    assert regime_kelly_multiplier(0.0) == pytest.approx(1.0)


def test_kelly_multiplier_volatile_returns_half():
    assert regime_kelly_multiplier(1.0) == pytest.approx(0.5)


def test_kelly_multiplier_linear_midpoint():
    assert regime_kelly_multiplier(0.5) == pytest.approx(0.75)


def test_kelly_multiplier_clamps_out_of_range():
    """Defensive clamp: scores outside [0,1] don't break sizing."""
    assert regime_kelly_multiplier(-0.5) == pytest.approx(1.0)
    assert regime_kelly_multiplier(1.5) == pytest.approx(0.5)


def test_kelly_multiplier_monotonically_decreasing():
    prev = regime_kelly_multiplier(0.0)
    for s in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        cur = regime_kelly_multiplier(s)
        assert cur <= prev
        prev = cur
