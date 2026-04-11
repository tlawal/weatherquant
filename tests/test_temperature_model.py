from datetime import datetime
from zoneinfo import ZoneInfo

import backend.modeling.temperature_model as temperature_model
from backend.engine.signal_engine import classify_city_state
from backend.modeling.adaptive import AdaptiveResult, KalmanState, StationTimePrediction
from backend.modeling.temperature_model import compute_model


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 4, 8, 19, 5, tzinfo=ZoneInfo("America/New_York"))
        return fixed.astimezone(tz) if tz else fixed


def test_compute_model_locks_current_bucket_after_peak(monkeypatch):
    monkeypatch.setattr(temperature_model, "datetime", _FixedDateTime)

    local_tz = ZoneInfo("America/New_York")
    adaptive = AdaptiveResult(
        kalman=KalmanState(
            smoothed_temp=66.2,
            temp_trend_per_min=-0.0125,
            uncertainty=0.2,
            n_observations=35,
            process_noise_factor=1.0,
        ),
        regression_slope=-0.02,
        regression_r2=0.81,
        regression_features_used=["time"],
        station_predictions=[
            StationTimePrediction(
                obs_time=datetime(2026, 4, 8, 19, 52, tzinfo=local_tz),
                predicted_temp=66.0,
                uncertainty=0.35,
                minutes_ahead=47.0,
                is_past=False,
                actual_temp=None,
                trend_per_hour=-0.8,
            ),
        ],
        predicted_daily_high=69.0,
        predicted_high_time=datetime(2026, 4, 8, 15, 20, tzinfo=local_tz),
        sigma_adjustment=1.0,
        peak_already_passed=True,
        composite_peak_timing="3:20 PM",
        peak_timing_source="actual_observed",
    )

    model = compute_model(
        nws_high=72.0,
        wu_daily_high=69.8,
        wu_hourly_peak=66.0,
        daily_high_metar=69.0,
        current_temp_f=66.2,
        calibration=None,
        buckets=[(68.0, 69.0), (70.0, 71.0), (72.0, 73.0), (74.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/New_York",
        observed_high=69.0,
        ml_features={
            "temp_slope_3h": -2.4,
            "avg_peak_timing_mins": 920.0,
            "day_of_year": 98,
        },
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=68.3,
        nbm_high=68.6,
    )

    assert model is not None
    assert model.lock_regime is True
    assert model.observed_bucket_idx == 0
    assert model.observed_bucket_upper_f == 70.0
    assert model.probs[0] > 0.99
    assert model.probs[1] < 0.01
    assert model.prob_hotter_bucket < 0.01
    assert model.prob_new_high_raw > 0.2
    assert classify_city_state(model.prob_hotter_bucket) == "resolved"


def test_lock_falls_back_when_adaptive_lags(monkeypatch):
    """Atlanta regression: at 19:17 ET the observed daily high was 78.8°F but
    adaptive.peak_already_passed hadn't flipped yet. The lock regime must
    still engage on the observation fallback path so the 76-77 bucket is
    zeroed out and the 78-79 bucket gets the bulk of the mass."""
    monkeypatch.setattr(temperature_model, "datetime", _FixedDateTime)

    local_tz = ZoneInfo("America/New_York")
    # Adaptive has NOT declared peak passed (the exact lag we are guarding).
    adaptive = AdaptiveResult(
        kalman=KalmanState(
            smoothed_temp=76.0,
            temp_trend_per_min=-0.005,
            uncertainty=0.25,
            n_observations=35,
            process_noise_factor=1.0,
        ),
        regression_slope=-0.01,
        regression_r2=0.6,
        regression_features_used=["time"],
        station_predictions=[
            StationTimePrediction(
                obs_time=datetime(2026, 4, 8, 19, 30, tzinfo=local_tz),
                predicted_temp=76.0,
                uncertainty=0.4,
                minutes_ahead=25.0,
                is_past=False,
                actual_temp=None,
                trend_per_hour=-0.3,
            ),
        ],
        predicted_daily_high=78.0,
        predicted_high_time=datetime(2026, 4, 8, 16, 15, tzinfo=local_tz),
        sigma_adjustment=1.0,
        peak_already_passed=False,  # <-- the key: adaptive lag
        composite_peak_timing="4:15 PM",
        peak_timing_source="forecast",
    )

    model = compute_model(
        nws_high=79.0,
        wu_daily_high=78.5,
        wu_hourly_peak=78.0,
        daily_high_metar=78.8,
        current_temp_f=76.0,
        calibration=None,
        buckets=[(74.0, 75.0), (76.0, 77.0), (78.0, 79.0), (80.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/New_York",
        observed_high=78.8,
        ml_features={
            "temp_slope_3h": -1.0,
            "avg_peak_timing_mins": 975.0,
            "day_of_year": 98,
        },
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=78.4,
        nbm_high=78.6,
    )

    assert model is not None
    # The lock must engage on the observation fallback path.
    assert model.lock_regime is True
    # Observed bucket is 78-79 (contains the 78.8 high).
    assert model.observed_bucket_idx == 2
    # The surpassed 76-77 bucket must be zero.
    assert model.probs[1] == 0.0
    # And the 74-75 bucket too.
    assert model.probs[0] == 0.0
    # The observed bucket should hold the dominant mass.
    assert model.probs[2] > 0.9
    assert model.prob_hotter_bucket < 0.05
