import pytest

from backend.config import Config
from backend.execution.risk_manager import compute_size
from backend.strategy.posterior_kelly import posterior_aware_kelly


def test_posterior_kelly_haircuts_component_disagreement():
    result = posterior_aware_kelly(
        bma_shadow={
            "components": [
                {"source": "hrrr", "mu": 82.5, "sigma": 0.5, "weight": 0.45},
                {"source": "ifs", "mu": 86.0, "sigma": 0.7, "weight": 0.55},
            ],
        },
        low_f=82.0,
        high_f=83.0,
        yes_price=0.20,
        fractional_kelly=0.10,
        max_position_size=0.25,
    )

    assert result is not None
    assert result.aggregate_prob > 0.20
    assert result.aggregate_kelly_f > 0.0
    assert result.weighted_median_component_kelly_f == 0.0
    assert result.conservative_kelly_f == 0.0
    assert result.haircut_applied is True


def test_posterior_kelly_keeps_size_when_components_agree():
    result = posterior_aware_kelly(
        bma_shadow={
            "components": [
                {"source": "hrrr", "mu": 82.4, "sigma": 0.6, "weight": 0.5},
                {"source": "ifs", "mu": 82.6, "sigma": 0.6, "weight": 0.5},
            ],
        },
        low_f=82.0,
        high_f=83.0,
        yes_price=0.20,
        fractional_kelly=0.10,
        max_position_size=0.25,
    )

    assert result is not None
    assert result.conservative_kelly_f > 0.0
    assert result.haircut_applied is False


def test_compute_size_uses_posterior_kelly_override(monkeypatch):
    monkeypatch.setattr(Config, "BANKROLL_CAP", 100.0, raising=False)
    monkeypatch.setattr(Config, "MAX_POSITION_PCT", 0.25, raising=False)
    baseline = compute_size(
        model_prob=0.60,
        limit_price=0.30,
        bankroll=100.0,
        open_exposure=0.0,
        ask_depth=100.0,
    )
    posterior = compute_size(
        model_prob=0.60,
        limit_price=0.30,
        bankroll=100.0,
        open_exposure=0.0,
        ask_depth=100.0,
        kelly_fraction_override=0.04,
    )

    assert baseline.rejected is False
    assert posterior.rejected is False
    assert posterior.kelly_source == "posterior_bma_component_median"
    assert posterior.kelly_f == pytest.approx(0.04)
    assert posterior.size < baseline.size


def test_compute_size_rejects_zero_posterior_kelly(monkeypatch):
    monkeypatch.setattr(Config, "BANKROLL_CAP", 100.0, raising=False)
    monkeypatch.setattr(Config, "MAX_POSITION_PCT", 0.25, raising=False)
    result = compute_size(
        model_prob=0.60,
        limit_price=0.30,
        bankroll=100.0,
        open_exposure=0.0,
        ask_depth=100.0,
        kelly_fraction_override=0.0,
    )

    assert result.rejected is True
    assert result.kelly_source == "posterior_bma_component_median"
    assert result.reject_reason.startswith("negative_kelly")
