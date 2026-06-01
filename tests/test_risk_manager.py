import pytest

from backend.config import Config
from backend.execution.risk_manager import compute_size


@pytest.fixture(autouse=True)
def small_bankroll_config(monkeypatch):
    monkeypatch.setattr(Config, "BANKROLL_CAP", 10.0, raising=False)
    monkeypatch.setattr(Config, "MAX_POSITION_PCT", 0.10, raising=False)
    monkeypatch.setattr(Config, "KELLY_FRACTION", 0.10, raising=False)
    monkeypatch.setattr(Config, "MAX_LIQUIDITY_PCT", 0.20, raising=False)
    monkeypatch.setattr(Config, "MIN_ORDERBOOK_DEPTH_DOLLARS", 2000.0, raising=False)
    monkeypatch.setattr(Config, "MIN_ORDER_NOTIONAL_DOLLARS", 1.0, raising=False)
    monkeypatch.setattr(Config, "MIN_NOTIONAL_BUMP_MAX_KELLY_MULTIPLE", 3.0, raising=False)


def test_small_bankroll_bumps_to_legal_min_when_liquidity_supports():
    result = compute_size(
        model_prob=0.60,
        limit_price=0.30,
        bankroll=10.0,
        open_exposure=0.0,
        ask_depth=100.0,
    )

    assert result.rejected is False
    assert result.min_notional_bump is True
    assert result.liquidity_haircut == pytest.approx(1.0)
    assert result.position_cap == pytest.approx(1.0)
    assert result.size * 0.30 >= 1.0


def test_min_notional_rejects_when_orderbook_cannot_absorb_one_dollar():
    result = compute_size(
        model_prob=0.60,
        limit_price=0.30,
        bankroll=10.0,
        open_exposure=0.0,
        ask_depth=5.0,
    )

    assert result.rejected is True
    assert result.reject_reason.startswith("min_notional_blocked")
    assert "liquidity_cap=$0.30" in result.reject_reason
    assert result.liquidity_haircut == pytest.approx(0.5)


def test_min_notional_rejects_when_bump_would_overbet_kelly():
    result = compute_size(
        model_prob=0.52,
        limit_price=0.50,
        bankroll=10.0,
        open_exposure=0.0,
        ask_depth=100.0,
    )

    assert result.rejected is True
    assert result.reject_reason.startswith("min_notional_overbet")
    assert "max=3.00x" in result.reject_reason


def test_invalid_sizing_price_rejected_before_division():
    result = compute_size(
        model_prob=0.80,
        limit_price=0.0,
        bankroll=10.0,
        open_exposure=0.0,
        ask_depth=100.0,
    )

    assert result.rejected is True
    assert result.reject_reason.startswith("invalid_price")
