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


def test_ensemble_kalman_zero_weight_when_diverged(monkeypatch):
    """Atlanta regression: Kalman nowcast 70°F vs multi-model consensus ~84°F
    is a >10°F divergence on a ~1°F consensus spread. The ensemble must
    collapse to multi-model only — consensus wins when a single-station
    filter fights five physics models."""
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
    assert model.inputs["kalman_nowcast_active"] is False
    assert model.inputs["kalman_weight"] == 0.0
    mu_multi = model.inputs["mu_multi_model"]
    mu_forecast = model.inputs["mu_forecast"]
    assert mu_forecast == mu_multi, f"Kalman must not influence mu_forecast when badly diverged (got {mu_forecast} vs multi {mu_multi})"
    assert model.inputs["ensemble_breakdown"]["mode"].startswith("multi_only")
    assert model.inputs["kalman_divergence_f"] > 3.0
    assert mu_multi > 83.0


def test_ensemble_kalman_active_in_peak_window_with_consensus(monkeypatch):
    """When Kalman agrees with consensus and we are inside the ±2h peak
    window, the filter contributes non-zero weight."""
    monkeypatch.setattr(temperature_model, "datetime", _AfternoonDateTime)

    # Consensus ~84, Kalman predicts 84.2 — divergence well inside spread budget
    adaptive = _adaptive_with_high(predicted_high=84.2, n_obs=30)

    model = compute_model(
        nws_high=84.0,
        wu_hourly_peak=84.5,
        daily_high_metar=None,
        current_temp_f=82.0,
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
    assert model.inputs["kalman_weight"] > 0.0
    assert model.inputs["kalman_weight"] <= 0.45
    # mu_forecast is a blend: bracketed by multi and kalman
    mu_multi = model.inputs["mu_multi_model"]
    mu_forecast = model.inputs["mu_forecast"]
    assert min(mu_multi, 84.2) <= mu_forecast <= max(mu_multi, 84.2) + 0.01
    assert model.inputs["ensemble_breakdown"]["mode"].startswith("blend_multi_kalman_w")


def test_ensemble_kalman_outside_peak_window(monkeypatch):
    """At 09:00 local with peak at 15:20, we are >2h before peak; even
    a well-calibrated Kalman gets zero weight because the filter has no
    forecast skill at that horizon."""
    monkeypatch.setattr(temperature_model, "datetime", _MorningDateTime)

    adaptive = _adaptive_with_high(predicted_high=83.9, n_obs=20)

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
    assert model.inputs["kalman_weight"] == 0.0
    assert model.inputs["mu_forecast"] == model.inputs["mu_multi_model"]
    assert model.inputs["ensemble_breakdown"]["mode"] == "multi_only:outside_peak_window"
    assert model.inputs["ensemble_breakdown"]["kalman"] is not None  # still reported for diagnostics


def test_compute_kalman_weight_helper():
    """Direct unit test of the weight helper."""
    from backend.modeling.temperature_model import compute_kalman_weight

    # At peak, well-calibrated: near max_weight
    w = compute_kalman_weight(
        hour_local_fractional=15.33, peak_hour_local=15.33,
        kalman_divergence=1.0, spread=2.0, n_obs=25, peak_already_passed=False,
    )
    assert abs(w - 0.45) < 1e-6

    # Outside ±2h window: zero
    w = compute_kalman_weight(
        hour_local_fractional=11.0, peak_hour_local=15.7,
        kalman_divergence=1.0, spread=2.0, n_obs=25, peak_already_passed=False,
    )
    assert w == 0.0

    # Diverged >> spread: zero
    w = compute_kalman_weight(
        hour_local_fractional=15.0, peak_hour_local=15.5,
        kalman_divergence=10.0, spread=2.0, n_obs=25, peak_already_passed=False,
    )
    assert w == 0.0

    # Not enough observations: zero
    w = compute_kalman_weight(
        hour_local_fractional=15.3, peak_hour_local=15.3,
        kalman_divergence=0.5, spread=2.0, n_obs=5, peak_already_passed=False,
    )
    assert w == 0.0

    # Peak passed: halved
    w_active = compute_kalman_weight(
        hour_local_fractional=15.0, peak_hour_local=15.0,
        kalman_divergence=0.5, spread=2.0, n_obs=25, peak_already_passed=False,
    )
    w_passed = compute_kalman_weight(
        hour_local_fractional=15.0, peak_hour_local=15.0,
        kalman_divergence=0.5, spread=2.0, n_obs=25, peak_already_passed=True,
    )
    assert abs(w_passed - 0.5 * w_active) < 1e-6


def test_seattle_evening_cooling_lock_engages(monkeypatch):
    """Seattle regression: 17:47 local, observed high 55°F reached earlier,
    current 53.6°F and dropping with clear negative Kalman trend. Lock must
    engage on path 3 (strong-cooling override) even though hour<18 and
    adaptive.peak_already_passed hasn't flipped. Distribution must collapse
    to the 54-55 bucket (canonical [54, 56))."""
    class _SeattleDT(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = cls(2026, 4, 8, 17, 47, tzinfo=ZoneInfo("America/Los_Angeles"))
            return fixed.astimezone(tz) if tz else fixed

    monkeypatch.setattr(temperature_model, "datetime", _SeattleDT)

    local_tz = ZoneInfo("America/Los_Angeles")
    adaptive = AdaptiveResult(
        kalman=KalmanState(
            smoothed_temp=53.6,
            temp_trend_per_min=-0.015,  # ~-0.9°F/hr
            uncertainty=0.3,
            n_observations=28,
            process_noise_factor=1.0,
        ),
        regression_slope=-0.012,
        regression_r2=0.7,
        regression_features_used=["time"],
        station_predictions=[],
        predicted_daily_high=55.0,
        predicted_high_time=datetime(2026, 4, 8, 15, 30, tzinfo=local_tz),
        sigma_adjustment=0.85,
        peak_already_passed=False,  # <-- adaptive lag
        composite_peak_timing="3:30 PM",
        peak_timing_source="wu_hourly",
    )

    # Polymarket Seattle-style 2°F buckets. Canonical ranges expand to [lo, lo+2).
    model = compute_model(
        nws_high=57.0,
        wu_hourly_peak=56.5,
        daily_high_metar=55.0,
        current_temp_f=53.6,
        calibration=None,
        buckets=[(52.0, 53.0), (54.0, 55.0), (56.0, 57.0), (58.0, 59.0), (60.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/Los_Angeles",
        observed_high=55.0,
        ml_features=None,
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=56.8,
        nbm_high=57.2,
        ecmwf_ifs_high=57.5,
    )

    assert model is not None, "compute_model should produce a result"
    # Lock must engage via path 3 (strong cooling override pre-18:00).
    assert model.lock_regime is True, "lock must engage on strong cooling + deficit"
    # remaining_rise must be zeroed by the cooling clamp.
    assert model.remaining_rise == 0.0
    # The observed-bucket (54-55, canonical [54, 56)) dominates.
    observed_idx = model.observed_bucket_idx
    assert observed_idx is not None
    assert model.probs[observed_idx] > 0.9
    # Any bucket above observed gets near-zero mass.
    assert model.prob_hotter_bucket < 0.08


def test_remaining_rise_uses_kalman_trend_when_aggressive(monkeypatch):
    """Atlanta 11 AM scenario: current 77°F, Kalman trend 3.77°F/hr, peak
    15:20 → ~4h of heating ahead → trend-based rise ~15°F dominates the
    static 4°F table value and projected_high lands near consensus."""
    class _LateMorningDT(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = cls(2026, 4, 8, 11, 20, tzinfo=ZoneInfo("America/New_York"))
            return fixed.astimezone(tz) if tz else fixed

    monkeypatch.setattr(temperature_model, "datetime", _LateMorningDT)

    # Kalman sees 3.77°F/hr heating
    local_tz = ZoneInfo("America/New_York")
    adaptive = AdaptiveResult(
        kalman=KalmanState(
            smoothed_temp=77.2,
            temp_trend_per_min=3.77 / 60.0,  # °/min
            uncertainty=0.3,
            n_observations=25,
            process_noise_factor=1.0,
        ),
        regression_slope=4.46 / 60.0,
        regression_r2=0.76,
        regression_features_used=["time", "precip_flag"],
        station_predictions=[],
        predicted_daily_high=81.2,
        predicted_high_time=datetime(2026, 4, 8, 15, 20, tzinfo=local_tz),
        sigma_adjustment=0.85,
        peak_already_passed=False,
        composite_peak_timing="3:20 PM",
        peak_timing_source="wu_hourly+kalman_trend",
    )

    model = compute_model(
        nws_high=90.0,
        wu_hourly_peak=88.0,
        daily_high_metar=77.0,
        current_temp_f=77.0,
        calibration=None,
        buckets=[(80.0, 81.0), (82.0, 83.0), (84.0, 85.0), (86.0, 87.0), (88.0, 89.0), (90.0, None)],
        forecast_quality="ok",
        unit="F",
        city_tz="America/New_York",
        observed_high=77.0,
        ml_features=None,
        adaptive=adaptive,
        latest_weather=None,
        hrrr_high=87.6,
        nbm_high=88.0,
        ecmwf_ifs_high=89.3,
    )

    assert model is not None
    # remaining_rise must lift toward Kalman trend (>>4°F static), capped by max source + 2°F
    assert model.remaining_rise >= 10.0, f"remaining_rise {model.remaining_rise} — trend fix not applied"
    # projected_high reflects the lifted trajectory
    assert model.mu_projected >= 86.0
    # mu_forecast is mu_multi_model (Kalman is far diverged at 11:20 and outside peak window)
    assert model.inputs["kalman_weight"] == 0.0
    assert model.inputs["mu_forecast"] == model.inputs["mu_multi_model"]


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
