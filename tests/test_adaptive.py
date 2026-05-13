from datetime import datetime
from zoneinfo import ZoneInfo

from backend.modeling.adaptive import (
    KalmanState,
    compute_peak_timing,
    compute_station_predictions,
)


def _obs(hour: int, minute: int, temp_f: float) -> dict:
    return {
        "observed_at": datetime(2026, 5, 13, hour, minute, tzinfo=ZoneInfo("America/New_York")),
        "temp_f": temp_f,
    }


def test_negative_kalman_dip_before_consensus_peak_is_not_peak_passed():
    kalman = KalmanState(
        smoothed_temp=73.0,
        temp_trend_per_min=-0.06,
        uncertainty=0.6,
        n_observations=12,
        process_noise_factor=1.0,
    )
    todays_obs = [
        _obs(8, 52, 66.0),
        _obs(9, 52, 70.0),
        _obs(10, 52, 74.0),
        _obs(11, 52, 73.0),
    ]

    peak = compute_peak_timing(
        wu_hourly_peak_time="3:00 PM",
        historical_peak_mins=900.0,
        kalman=kalman,
        current_hour_local=11,
        todays_obs=todays_obs,
        forecast_high=80.0,
    )

    assert peak["peak_already_passed"] is False
    assert "temporary cooling dip" in peak["detail"]


def test_negative_kalman_after_consensus_peak_can_mark_peak_passed():
    kalman = KalmanState(
        smoothed_temp=73.0,
        temp_trend_per_min=-0.02,
        uncertainty=0.6,
        n_observations=12,
        process_noise_factor=1.0,
    )
    todays_obs = [
        _obs(11, 52, 72.0),
        _obs(13, 52, 75.5),
        _obs(15, 52, 75.0),
        _obs(16, 52, 73.5),
    ]

    peak = compute_peak_timing(
        wu_hourly_peak_time="3:00 PM",
        historical_peak_mins=900.0,
        kalman=kalman,
        current_hour_local=16,
        todays_obs=todays_obs,
        forecast_high=76.5,
    )

    assert peak["peak_already_passed"] is True
    assert peak["source"] == "actual_observed"


def test_remaining_rise_cap_is_diagnostic_when_below_pre_peak_consensus():
    now_local = datetime(2026, 5, 13, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    kalman = KalmanState(
        smoothed_temp=75.0,
        temp_trend_per_min=0.04,
        uncertainty=0.5,
        n_observations=10,
        process_noise_factor=1.0,
    )

    capped = compute_station_predictions(
        kalman=kalman,
        regression_slope=0.04,
        regression_r2=1.0,
        observation_minutes=[52],
        now_local=now_local,
        todays_obs=[],
        start_hour=15,
        end_hour=16,
        estimated_peak_mins=960.0,
        remaining_rise=1.0,
        forecast_high=None,
    )[0]
    guarded = compute_station_predictions(
        kalman=kalman,
        regression_slope=0.04,
        regression_r2=1.0,
        observation_minutes=[52],
        now_local=now_local,
        todays_obs=[],
        start_hour=15,
        end_hour=16,
        estimated_peak_mins=960.0,
        remaining_rise=1.0,
        forecast_high=82.0,
    )[0]

    assert capped.predicted_temp <= 76.0
    assert guarded.predicted_temp > 76.0
