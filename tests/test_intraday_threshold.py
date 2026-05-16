import pytest

from backend.modeling.intraday_threshold import (
    bucket_probs_from_survival,
    enforce_monotone_survival,
    predict_intraday_threshold_probabilities,
)
from backend.modeling.settlement import canonical_bucket_ranges


def test_enforce_monotone_survival_uses_isotonic_adjustment():
    adjusted = enforce_monotone_survival({
        70.0: 0.92,
        72.0: 0.96,  # violates non-increasing survival
        74.0: 0.55,
    })

    assert adjusted[70.0] == pytest.approx(adjusted[72.0])
    assert adjusted[72.0] >= adjusted[74.0]
    assert all(0.0 <= p <= 1.0 for p in adjusted.values())


def test_bucket_probs_from_survival_preserves_floor_and_total_mass():
    buckets = canonical_bucket_ranges([
        (None, 68.0),
        (68.0, 69.0),
        (70.0, 71.0),
        (72.0, 73.0),
        (74.0, None),
    ])
    probs = bucket_probs_from_survival(
        buckets,
        {68.0: 1.0, 70.0: 0.98, 72.0: 0.90, 74.0: 0.35},
        observed_high=69.1,
    )

    assert probs[0] == pytest.approx(0.0)
    assert probs[1] == pytest.approx(0.02)
    assert probs[2] == pytest.approx(0.08)
    assert probs[3] == pytest.approx(0.55)
    assert probs[4] == pytest.approx(0.35)
    assert sum(probs) == pytest.approx(1.0)


def test_intraday_threshold_atlanta_style_shadow_reduces_low_buckets():
    buckets = canonical_bucket_ranges([
        (None, 65.0),
        (66.0, 67.0),
        (68.0, 69.0),
        (70.0, 71.0),
        (72.0, 73.0),
        (74.0, 75.0),
        (76.0, 77.0),
        (78.0, None),
    ])

    result = predict_intraday_threshold_probabilities(
        buckets=buckets,
        observed_high=69.1,
        current_temp_f=69.8,
        projected_high=73.25,
        consensus_high=74.73,
        sigma=2.78,
        remaining_rise=4.15,
        hour_local=11.25,
        peak_hour_local=15.68,
        trend_per_hr=1.58,
        trusted_spread=2.7,
        forecast_quality="ok",
    )

    assert result is not None
    assert result.alpha == 0.0
    assert result.probs[0] == pytest.approx(0.0)
    assert result.probs[1] == pytest.approx(0.0)
    # Display bucket 68-69 is canonical [68, 70). With observed_high=69.1, only
    # [69.1, 70) can still win, and the threshold model should make crossing 70
    # very likely on this warming profile.
    assert result.probs[2] < 0.02
    assert result.probs[3] < 0.08
    assert sum(result.probs) == pytest.approx(1.0)
    survival_values = [p for _, p in sorted(result.survival.items())]
    assert survival_values == sorted(survival_values, reverse=True)
