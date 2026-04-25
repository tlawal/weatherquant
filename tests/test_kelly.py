"""Unit tests for Kelly sizing + EV helpers.

Covers the pure math in backend/strategy/kelly.py.
"""
from __future__ import annotations

import pytest

from backend.strategy.kelly import (
    calculate_ev_per_share,
    calculate_expected_value,
    calculate_kelly_fraction,
)


# ── calculate_ev_per_share ────────────────────────────────────────────────

def test_ev_per_share_zero_when_p_equals_price():
    """At p == price the share is fairly priced; EV is exactly zero."""
    assert calculate_ev_per_share(0.50, 0.50) == 0.0
    assert calculate_ev_per_share(0.30, 0.30) == 0.0


def test_ev_per_share_positive_when_p_above_price():
    """Buying YES at a price below model probability is +EV."""
    ev = calculate_ev_per_share(0.60, 0.40)
    # p*(1-price) - (1-p)*price = 0.6*0.6 - 0.4*0.4 = 0.36 - 0.16 = 0.20
    assert ev == pytest.approx(0.20, abs=1e-6)


def test_ev_per_share_negative_when_p_below_price():
    ev = calculate_ev_per_share(0.30, 0.50)
    # 0.3*0.5 - 0.7*0.5 = 0.15 - 0.35 = -0.20
    assert ev == pytest.approx(-0.20, abs=1e-6)


def test_ev_per_share_equals_p_minus_price():
    """Algebraic identity: p*(1-q) - (1-p)*q == p - q."""
    for p, q in [(0.55, 0.42), (0.71, 0.30), (0.05, 0.10), (0.95, 0.80)]:
        assert calculate_ev_per_share(p, q) == pytest.approx(p - q, abs=1e-6)


def test_ev_per_share_monotone_in_p():
    """Holding price fixed, EV per share strictly increases in p."""
    price = 0.40
    prev = -1.0
    for p in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        ev = calculate_ev_per_share(p, price)
        assert ev > prev
        prev = ev


def test_ev_per_share_returns_zero_outside_valid_price_range():
    """price <= 0 or price >= 1 returns 0.0 (defensive guard)."""
    assert calculate_ev_per_share(0.50, 0.0) == 0.0
    assert calculate_ev_per_share(0.50, 1.0) == 0.0
    assert calculate_ev_per_share(0.50, -0.1) == 0.0
    assert calculate_ev_per_share(0.50, 1.1) == 0.0


# ── distinction from calculate_expected_value ─────────────────────────────

def test_per_share_vs_per_dollar_differ_in_denominator():
    """ev_per_share is per held YES share. calculate_expected_value is per $1 wagered.

    At p=0.60, price=0.40:
      per-share = 0.20  (you bought 1 share for $0.40, expect $0.60 back)
      per-$1   = (0.60 * (1/0.40 - 1)) - 0.40 = 0.60*1.5 - 0.40 = 0.90 - 0.40 = 0.50

    Per-$1 is larger because $1 wagered buys 2.5 shares at $0.40 each. The two
    quantities are NOT interchangeable — exit gates use per-share, Kelly uses per-$1.
    """
    p, price = 0.60, 0.40
    per_share = calculate_ev_per_share(p, price)
    per_dollar = calculate_expected_value(p, price)
    assert per_share == pytest.approx(0.20, abs=1e-6)
    assert per_dollar == pytest.approx(0.50, abs=1e-4)
    assert per_dollar > per_share  # always true when EV > 0 and price < 0.5


def test_per_dollar_equals_per_share_divided_by_price():
    """Dimensional sanity: ev_per_$1 ≈ ev_per_share / price."""
    for p, price in [(0.60, 0.40), (0.55, 0.30), (0.20, 0.10)]:
        per_share = calculate_ev_per_share(p, price)
        per_dollar = calculate_expected_value(p, price)
        assert per_dollar == pytest.approx(per_share / price, abs=1e-3)


# ── calculate_kelly_fraction (existing function — sanity ring-fence) ──────

def test_kelly_zero_when_no_edge():
    """At p == price, Kelly is zero."""
    assert calculate_kelly_fraction(0.50, 0.50) == 0.0


def test_kelly_positive_with_edge():
    """Full Kelly raw math: f* = (b*p - q)/b at p=0.6, price=0.4 → 0.333."""
    f = calculate_kelly_fraction(0.60, 0.40, fractional_kelly=1.0, max_position_size=1.0)
    assert f == pytest.approx(0.3333, abs=1e-3)


def test_kelly_capped_by_max_position_size():
    """max_position_size caps the fraction even at full Kelly."""
    f = calculate_kelly_fraction(0.95, 0.05, fractional_kelly=1.0, max_position_size=0.25)
    assert f == pytest.approx(0.25, abs=1e-6)
