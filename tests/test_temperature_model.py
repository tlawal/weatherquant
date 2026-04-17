from datetime import datetime
from zoneinfo import ZoneInfo

import backend.modeling.temperature_model as temperature_model
from backend.engine.signal_engine import classify_city_state
from backend.modeling.adaptive import (
    AdaptiveResult,
    KalmanState,
    StationTimePrediction,
    compute_peak_timing,
)
from backend.modeling.temperature_model import compute_model


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 4, 8, 19, 5, tzinfo=ZoneInfo("America/New_York"))
        return fixed.astimezone(tz) if tz else fixed


class _AfternoonDateTime(datetime):
    """14:00 local — before the classic 18:00 rollover guard kicks in."""

    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 4, 8, 14, 0, tzinfo=ZoneInfo("America/New_York"))
        return fixed.astimezone(tz) if tz else fixed


class _MorningDateTime(datetime):
    """09:00 local — before the 11 AM Kalman-nowcast gate."""

    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 4, 8, 9, 0, tzinfo=ZoneInfo("America/New_York"))
        return fixed.astimezone(tz) if tz else fixed


class _PacificEveningDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 4, 16, 18, 5, tzinfo=ZoneInfo("America/Los_Angeles"))
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
    assert model.probs[0] > 0.97
    assert model.probs[1] < 0.02
    assert model.prob_hotter_bucket <= 0.02
    assert model.prob_new_high_raw > 0.2
    assert model.lock_trigger_reason == "adaptive_peak_passed"
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
    assert model.lock_trigger_reason == "hard_cooling_gap"





def _adaptive_with_high(predicted_high: float, n_obs: int = 30) -> AdaptiveResult:
    """Helper: build a minimal AdaptiveResult with the given Kalman high."""
    local_tz = ZoneInfo("America/New_York")
    return AdaptiveResult(
        kalman=KalmanState(
            smoothed_temp=72.0,
            temp_trend_per_min=0.01,
            uncertainty=0.3,
            n_observations=n_obs,
            process_noise_factor=1.0,
        ),
        regression_slope=0.01,
        regression_r2=0.75,
        regression_features_used=["time"],
        station_predictions=[
            StationTimePrediction(
                obs_time=datetime(2026, 4, 8, 15, 52, tzinfo=local_tz),
                predicted_temp=predicted_high,
                uncertainty=0.5,
                minutes_ahead=60.0,
                is_past=False,
                actual_temp=None,
                trend_per_hour=0.6,
            ),
        ],
        predicted_daily_high=predicted_high,
        predicted_high_time=datetime(2026, 4, 8, 15, 20, tzinfo=local_tz),
        sigma_adjustment=1.0,
        peak_already_passed=False,
        composite_peak_timing="3:20 PM",
        peak_timing_source="kalman_trend",
    )


def test_ensemble_70_30_blend_post_11am(monkeypatch):
    """At 14:00 local with ≥11 AM gate open, mu_forecast must be
    70% multi-model + 30% Kalman. Exercise with Kalman under-shooting
    (the Atlanta regression that motivated this change)."""
    monkeypatch.setattr(temperature_model, "datetime", _AfternoonDateTime)

    adaptive = _adaptive_with_high(predicted_high=70.0, n_obs=30)

    model = compute_model(
        nws_high=84.0,
        wu_hourly_peak=84.5,
        daily_high_metar=None,
        current_temp_f=72.0,
        calibration=None,
        buckets=[(80.0, 81.0), (82.0, 83.0), (84.0, 85.0), (86.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/New_York",
        observed_high=None,
        ml_features=None,
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=83.5,
        nbm_high=83.8,
        ecmwf_ifs_high=84.2,
    )

    assert model is not None
    assert model.inputs["kalman_nowcast_active"] is True
    mu_multi = model.inputs["mu_multi_model"]
    mu_forecast = model.inputs["mu_forecast"]
    expected = 0.70 * mu_multi + 0.30 * 70.0
    assert abs(mu_forecast - expected) < 0.05, (
        f"mu_forecast={mu_forecast} expected≈{expected} (multi={mu_multi})"
    )
    assert model.inputs["ensemble_breakdown"]["mode"] == "70_multi_30_kalman"
    assert model.inputs["kalman_divergence_f"] is not None
    assert model.inputs["kalman_divergence_f"] > 3.0
    # Multi-model slice should still be near consensus (~84°)
    assert mu_multi > 83.0


def test_ensemble_pre_11am_multi_only(monkeypatch):
    """At 09:00 local Kalman must NOT influence mu_forecast — the 70/30
    gate is closed and mu_forecast == mu_multi_model exactly."""
    monkeypatch.setattr(temperature_model, "datetime", _MorningDateTime)

    adaptive = _adaptive_with_high(predicted_high=65.0, n_obs=12)

    model = compute_model(
        nws_high=84.0,
        wu_hourly_peak=84.5,
        daily_high_metar=None,
        current_temp_f=65.0,
        calibration=None,
        buckets=[(80.0, 81.0), (82.0, 83.0), (84.0, 85.0), (86.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/New_York",
        observed_high=None,
        ml_features=None,
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=83.5,
        nbm_high=83.8,
        ecmwf_ifs_high=84.2,
    )

    assert model is not None
    assert model.inputs["kalman_nowcast_active"] is False
    assert model.inputs["mu_forecast"] == model.inputs["mu_multi_model"]
    assert model.inputs["ensemble_breakdown"]["mode"] == "pre_11am_multi_only"
    assert model.inputs["ensemble_breakdown"]["kalman"] is None
    assert model.inputs["kalman_divergence_f"] is None


def test_ecmwf_ifs_included_in_sources_used(monkeypatch):
    """ECMWF IFS must appear in sources_used and contribute to mu_multi_model."""
    monkeypatch.setattr(temperature_model, "datetime", _AfternoonDateTime)

    model = compute_model(
        nws_high=80.0,
        wu_hourly_peak=80.5,
        daily_high_metar=None,
        current_temp_f=72.0,
        calibration=None,
        buckets=[(78.0, 79.0), (80.0, 81.0), (82.0, 83.0), (84.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/New_York",
        observed_high=None,
        ml_features=None,
        adaptive=None,
        latest_weather=None,
        hrrr_high=80.2,
        nbm_high=80.3,
        ecmwf_ifs_high=82.0,
    )

    assert model is not None
    assert "ecmwf_ifs" in model.inputs["sources_used"]
    assert model.inputs["ecmwf_ifs_high"] == 82.0
    # ECMWF IFS skews the multi-model mean slightly upward vs. the
    # no-ECMWF case; sanity check we're above 80.
    assert model.inputs["mu_multi_model"] > 80.0


def test_compute_peak_timing_normalizes_legacy_et_string_to_city_local():
    peak = compute_peak_timing(
        wu_hourly_peak_time="10:00 PM ET",
        wu_hourly_peak_mins=None,
        historical_peak_mins=None,
        kalman=None,
        current_hour_local=17,
        todays_obs=[],
        city_tz="America/Los_Angeles",
        reference_local_dt=datetime(2026, 4, 16, 17, 0, tzinfo=ZoneInfo("America/Los_Angeles")),
    )

    assert peak["estimated_peak_mins"] == 19 * 60
    assert peak["estimated_peak_time"] == "7:00 PM"


def test_la_regression_uses_official_floor_and_hard_cooling_lock(monkeypatch):
    monkeypatch.setattr(temperature_model, "datetime", _PacificEveningDateTime)

    local_tz = ZoneInfo("America/Los_Angeles")
    adaptive = AdaptiveResult(
        kalman=KalmanState(
            smoothed_temp=66.7,
            temp_trend_per_min=0.0355,
            uncertainty=0.2,
            n_observations=39,
            process_noise_factor=1.0,
        ),
        regression_slope=-0.0008,
        regression_r2=0.004,
        regression_features_used=["time", "precip_flag"],
        station_predictions=[
            StationTimePrediction(
                obs_time=datetime(2026, 4, 16, 18, 53, tzinfo=local_tz),
                predicted_temp=66.7,
                uncertainty=0.5,
                minutes_ahead=48.0,
                is_past=False,
                actual_temp=None,
                trend_per_hour=1.9,
            ),
            StationTimePrediction(
                obs_time=datetime(2026, 4, 16, 19, 53, tzinfo=local_tz),
                predicted_temp=66.7,
                uncertainty=1.3,
                minutes_ahead=108.0,
                is_past=False,
                actual_temp=None,
                trend_per_hour=1.2,
            ),
        ],
        predicted_daily_high=69.1,
        predicted_high_time=datetime(2026, 4, 16, 12, 53, tzinfo=local_tz),
        sigma_adjustment=0.85,
        peak_already_passed=False,
        composite_peak_timing="8:00 PM",
        peak_timing_source="wu_hourly+historical+kalman_trend",
    )

    model = compute_model(
        nws_high=70.0,
        wu_hourly_peak=63.0,
        daily_high_metar=69.1,
        current_temp_f=64.4,
        calibration=None,
        buckets=[(68.0, 69.0), (70.0, 71.0), (72.0, 73.0), (74.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/Los_Angeles",
        observed_high=69.1,
        raw_observed_high=69.8,
        ml_features={
            "temp_slope_3h": -4.7,
            "avg_peak_timing_mins": 960.0,
            "day_of_year": 106,
        },
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=69.8,
        nbm_high=66.5,
        ecmwf_ifs_high=69.4,
    )

    assert model is not None
    assert model.lock_regime is True
    assert model.lock_trigger_reason == "hard_cooling_gap"
    assert model.observed_bucket_idx == 0
    assert model.probs[0] > 0.95
    assert model.prob_hotter_bucket < 0.05
    assert model.inputs["observed_high"] == 69.1
    assert model.inputs["raw_observed_high"] == 69.8


def test_hard_cooling_lock_does_not_fire_when_current_temp_is_still_near_high(monkeypatch):
    monkeypatch.setattr(temperature_model, "datetime", _PacificEveningDateTime)

    adaptive = AdaptiveResult(
        kalman=KalmanState(
            smoothed_temp=68.9,
            temp_trend_per_min=0.03,
            uncertainty=0.2,
            n_observations=39,
            process_noise_factor=1.0,
        ),
        regression_slope=0.0,
        regression_r2=0.1,
        regression_features_used=["time"],
        station_predictions=[],
        predicted_daily_high=69.1,
        predicted_high_time=None,
        sigma_adjustment=1.0,
        peak_already_passed=False,
        composite_peak_timing="8:00 PM",
        peak_timing_source="kalman_trend",
    )

    model = compute_model(
        nws_high=70.0,
        wu_hourly_peak=69.5,
        daily_high_metar=69.1,
        current_temp_f=68.8,
        calibration=None,
        buckets=[(68.0, 69.0), (70.0, 71.0), (72.0, 73.0), (74.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/Los_Angeles",
        observed_high=69.1,
        raw_observed_high=69.8,
        ml_features={
            "temp_slope_3h": -0.2,
            "avg_peak_timing_mins": 960.0,
            "day_of_year": 106,
        },
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=69.8,
        nbm_high=69.4,
        ecmwf_ifs_high=69.6,
    )

    assert model is not None
    assert model.lock_regime is False
    assert model.lock_trigger_reason is None
