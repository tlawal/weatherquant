"""
Adaptive prediction engine — Kalman filter + multivariate rolling regression.

Produces station-time predictions (e.g., predicted temp at KATL :52 each hour)
and composite peak timing estimates.  Re-initialized from scratch each signal
cycle using the full day's 5-minute METAR observations.

Academic references:
  - Delle Monache et al. (2011) "Kalman Filter and Analog-Based Retrievals"
    Mon. Weather Rev. 139(10) — state estimation for near-surface temperature.
  - Hacker & Rife (2007) "A Practical Approach to Sequential Estimation of
    Systematic Error in NWP" — adaptive Kalman with weather-conditioned noise.
  - Mass & Brier (2015) "Two-Meter Temperature Forecasting with K-Nearest
    Neighbors" — multivariate surface obs improve short-range predictions.
  - Glahn & Lowry (1972) "Use of MOS in Objective Weather Forecasting"
    — our regression extends classic MOS with real-time surface features.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    smoothed_temp: float
    temp_trend_per_min: float       # °/min
    uncertainty: float
    n_observations: int
    process_noise_factor: float     # current weather-adapted noise level


@dataclass
class StationTimePrediction:
    obs_time: datetime              # e.g. 2026-03-25 15:52 local
    predicted_temp: float           # in city's unit (°F or °C)
    uncertainty: float
    minutes_ahead: float
    is_past: bool                   # True if observation already occurred
    actual_temp: Optional[float] = None   # filled in if obs_time is past
    trend_per_hour: Optional[float] = None  # °/hr at this point


@dataclass
class AdaptiveResult:
    kalman: KalmanState
    regression_slope: float         # °/min from latest window
    regression_r2: float
    regression_features_used: list[str]
    station_predictions: list[StationTimePrediction]
    predicted_daily_high: float
    predicted_high_time: Optional[datetime]
    sigma_adjustment: float         # multiplier for base sigma
    peak_already_passed: bool
    composite_peak_timing: Optional[str]   # e.g. "3:52 PM ET"
    peak_timing_source: Optional[str]      # e.g. "kalman_trend"
    remaining_rise_cap: Optional[float] = None   # ML-predicted remaining rise (°)
    diurnal_peak_estimate: Optional[float] = None  # T_max from fitted curve
    diurnal_fit_rmse: Optional[float] = None       # goodness of fit (°)
    diurnal_fit_active: bool = False                # whether curve was used


# ---------------------------------------------------------------------------
# Kalman filter (2-state: [temp, trend])
# ---------------------------------------------------------------------------

def _weather_process_noise(obs: dict) -> float:
    """Compute a process noise scale factor from extended weather fields.

    Higher factor → more uncertainty in the prediction step.
    Returns a multiplier (1.0 = baseline).
    """
    factor = 1.0

    # Wind speed: high wind → boundary layer mixing → less predictable
    wind_kt = obs.get("wind_speed_kt")
    if wind_kt is not None:
        if wind_kt > 20:
            factor *= 1.6
        elif wind_kt > 10:
            factor *= 1.3

    # Precipitation: rapid temp drops
    wx = obs.get("wx_string") or ""
    if any(tok in wx.upper() for tok in ("RA", "TS", "SH", "SN", "DZ", "GR")):
        factor *= 2.0

    # Cloud cover transitions (encoded: CLR=0 … OVC=4)
    cloud_val = obs.get("cloud_cover_val")
    if cloud_val is not None and cloud_val >= 3:  # BKN or OVC
        factor *= 1.4

    # Low humidity → wider diurnal range
    humidity = obs.get("humidity_pct")
    if humidity is not None and humidity < 30:
        factor *= 1.15

    return factor


def run_kalman(
    observations: list[dict],
    dt_key: str = "observed_at",
    temp_key: str = "temp_f",
) -> KalmanState:
    """Run a 2-state Kalman filter over a sequence of METAR observations.

    Each observation dict must have at least `dt_key` (datetime) and
    `temp_key` (float).  Extended weather fields (wind_speed_kt, wx_string,
    cloud_cover_val, humidity_pct) are optional — used for adaptive process
    noise.

    State vector: x = [temperature, temperature_rate_per_minute]
    """
    if not observations:
        return KalmanState(
            smoothed_temp=0.0, temp_trend_per_min=0.0,
            uncertainty=10.0, n_observations=0, process_noise_factor=1.0,
        )

    # Measurement noise (°F)² — ASOS sensors typically ±0.5°F
    R = 0.25  # (0.5)²

    # Initial state from first observation
    first = observations[0]
    x = np.array([first[temp_key], 0.0])  # [temp, trend]
    P = np.array([[4.0, 0.0],
                  [0.0, 0.01]])  # initial covariance

    last_dt: datetime = first[dt_key]
    noise_factor = 1.0

    for obs in observations[1:]:
        dt: datetime = obs[dt_key]
        temp: float = obs[temp_key]

        delta_min = (dt - last_dt).total_seconds() / 60.0
        if delta_min <= 0:
            continue

        # State transition: constant-velocity model
        F = np.array([[1.0, delta_min],
                      [0.0, 1.0]])

        # Weather-conditioned process noise
        noise_factor = _weather_process_noise(obs)
        q_temp = 0.05 * noise_factor * delta_min   # position noise
        q_trend = 0.0005 * noise_factor * delta_min  # trend noise
        Q = np.array([[q_temp, 0.0],
                      [0.0, q_trend]])

        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update (measurement matrix: we observe temperature only)
        H = np.array([[1.0, 0.0]])
        y = temp - (H @ x)[0]           # innovation
        S = (H @ P @ H.T)[0, 0] + R     # innovation covariance
        K = (P @ H.T) / S               # Kalman gain (2×1)
        x = x + K.flatten() * y
        P = P - np.outer(K.flatten(), H @ P)

        last_dt = dt

    return KalmanState(
        smoothed_temp=float(x[0]),
        temp_trend_per_min=float(x[1]),
        uncertainty=float(math.sqrt(P[0, 0])),
        n_observations=len(observations),
        process_noise_factor=float(noise_factor),
    )


# ---------------------------------------------------------------------------
# Multivariate rolling regression
# ---------------------------------------------------------------------------

_CLOUD_MAP = {"CLR": 0, "SKC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}


def _encode_cloud(cover: Optional[str]) -> Optional[float]:
    if cover is None:
        return None
    return float(_CLOUD_MAP.get(cover.upper().strip(), 2))


def run_regression(
    observations: list[dict],
    window_minutes: int = 60,
    dt_key: str = "observed_at",
    temp_key: str = "temp_f",
) -> tuple[float, float, list[str]]:
    """60-minute rolling OLS regression with optional weather features.

    Returns (slope_per_min, r_squared, features_used).
    Falls back to univariate (time-only) if extended fields are unavailable.
    """
    if len(observations) < 3:
        return 0.0, 0.0, []

    # Window: use only the most recent `window_minutes` of data
    latest_dt = observations[-1][dt_key]
    cutoff = latest_dt - timedelta(minutes=window_minutes)
    windowed = [o for o in observations if o[dt_key] >= cutoff]
    if len(windowed) < 3:
        windowed = observations[-3:]  # minimum for regression

    ref_dt = windowed[0][dt_key]
    y = np.array([o[temp_key] for o in windowed])

    # Feature matrix — always include time (minutes since window start)
    time_mins = np.array([(o[dt_key] - ref_dt).total_seconds() / 60.0 for o in windowed])
    features = [time_mins]
    feature_names = ["time"]

    # Extended weather features (only add if >50% of observations have the field)
    def _try_feature(key: str, transform=None):
        vals = []
        for o in windowed:
            v = o.get(key)
            if v is not None and transform:
                v = transform(v)
            vals.append(v)
        non_null = [v for v in vals if v is not None]
        if len(non_null) >= len(windowed) * 0.5:
            # Fill missing with mean
            mean_val = sum(non_null) / len(non_null)
            filled = np.array([v if v is not None else mean_val for v in vals])
            features.append(filled)
            feature_names.append(key)

    _try_feature("wind_speed_kt")
    _try_feature("humidity_pct")
    _try_feature("cloud_cover", transform=_encode_cloud)
    _try_feature("precip_flag", transform=lambda x: float(bool(x)))

    # Pressure tendency (change over window)
    pressures = [o.get("altimeter_inhg") for o in windowed]
    non_null_p = [(i, p) for i, p in enumerate(pressures) if p is not None]
    if len(non_null_p) >= len(windowed) * 0.5:
        # Compute running pressure change from first available
        base_p = non_null_p[0][1]
        tendency = []
        last_known = base_p
        for o in windowed:
            p = o.get("altimeter_inhg")
            if p is not None:
                last_known = p
            tendency.append(last_known - base_p)
        features.append(np.array(tendency))
        feature_names.append("pressure_tendency")

    # Build design matrix (with intercept)
    X = np.column_stack([np.ones(len(windowed))] + features)

    # OLS via numpy lstsq
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, []

    # R²
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = max(0.0, r2)

    # Slope is the coefficient for 'time' (index 1 in coeffs, since 0 is intercept)
    slope = float(coeffs[1]) if len(coeffs) > 1 else 0.0

    return slope, r2, feature_names


# ---------------------------------------------------------------------------
# Diurnal decay for extrapolation
# ---------------------------------------------------------------------------

def _diurnal_decay(minutes_ahead: float, now_hour: float, peak_hour: float) -> float:
    """Decay factor [0, 1] that reduces extrapolation rate as we approach peak time.

    Physical basis: the diurnal heating rate follows a half-sinusoid that
    peaks mid-morning and decays to zero at peak temperature time (typically
    2–4 PM).  A constant-rate linear extrapolation overestimates future
    temperatures because it ignores this deceleration.

    The decay factor approximates the ratio of actual integrated heating
    to linearly-extrapolated heating over the interval [now, target]:

      Before peak: rate declines quasi-linearly → avg multiplier ≈ 1 − f/2
                   where f = fraction of remaining heating window consumed.
      Past peak:   only apply the slope up to peak, then hold flat.

    References:
      - Parton & Nicholls (2012) J. Appl. Meteor. Climatol. 51, 612–630.
      - Mayer & Groom (2002) J. Atmos. Sci. 59, 1413–1424.
    """
    if minutes_ahead <= 0:
        return 1.0

    target_hour = now_hour + minutes_ahead / 60.0

    if target_hour > peak_hour:
        # Past peak: heating only occurs in [now, peak], with average
        # decay 0.5 over that window.  After peak, temperature is flat.
        mins_to_peak = max(0.0, (peak_hour - now_hour) * 60.0)
        return (mins_to_peak * 0.5) / minutes_ahead if minutes_ahead > 0 else 0.0

    # Before peak: rate declines linearly toward zero at peak.
    # Average multiplier over [now, target] ≈ 1 − f/2
    # where f is the fraction of the remaining heating window consumed.
    frac = (target_hour - now_hour) / (peak_hour - now_hour)
    return max(0.0, 1.0 - frac / 2.0)


# ---------------------------------------------------------------------------
# Station-time predictions
# ---------------------------------------------------------------------------

def compute_station_predictions(
    kalman: KalmanState,
    regression_slope: float,
    regression_r2: float,
    observation_minutes: list[int],
    now_local: datetime,
    todays_obs: list[dict],
    start_hour: int = 6,
    end_hour: int = 20,       # up to 7:52 PM (exclusive of 20:xx)
    dt_key: str = "observed_at",
    temp_key: str = "temp_f",
    city_tz: str = "America/New_York",
    estimated_peak_mins: Optional[float] = None,
    remaining_rise: Optional[float] = None,
    diurnal_model: Optional["DiurnalFit"] = None,
) -> list[StationTimePrediction]:
    """Compute predicted temps at each station observation time from start_hour through end_hour.

    Blends Kalman trend with regression slope, applying a diurnal decay
    factor so that the extrapolated warming rate decelerates toward zero
    at the estimated peak temperature time.  An optional remaining-rise
    cap prevents predictions from exceeding the Kalman temperature plus
    the ML-predicted remaining temperature rise.

    Past observation times are filled with actual observed values.
    """
    tz = ZoneInfo(city_tz)
    today = now_local.date()
    predictions: list[StationTimePrediction] = []

    # Build lookup of actual observations by (hour, minute) in LOCAL time
    actual_by_hm: dict[tuple[int, int], float] = {}
    for obs in todays_obs:
        _dt = obs[dt_key]
        if _dt.tzinfo is None:
            _dt = _dt.replace(tzinfo=timezone.utc)
        dt_local = _dt.astimezone(tz)
        if dt_local.date() == today and obs.get(temp_key) is not None:
            actual_by_hm[(dt_local.hour, dt_local.minute)] = obs[temp_key]

    # Generate station observation times across the afternoon window
    for hour in range(start_hour, end_hour):
        for minute in observation_minutes:
            obs_time = datetime(today.year, today.month, today.day,
                                hour, minute, tzinfo=tz)
            minutes_ahead = (obs_time - now_local).total_seconds() / 60.0
            is_past = minutes_ahead < -5  # allow 5 min grace

            # Check for actual observation (±2 min tolerance)
            actual = None
            for dm in range(-2, 3):
                check_m = (minute + dm) % 60
                check_h = hour + ((minute + dm) // 60)
                actual = actual_by_hm.get((check_h, check_m))
                if actual is not None:
                    break

            if is_past and actual is not None:
                # Past with actual observation
                predictions.append(StationTimePrediction(
                    obs_time=obs_time,
                    predicted_temp=actual,
                    uncertainty=0.0,
                    minutes_ahead=minutes_ahead,
                    is_past=True,
                    actual_temp=actual,
                    trend_per_hour=None,
                ))
            else:
                # Future or past without data — predict
                # Blend Kalman trend and regression slope weighted by R²
                kalman_weight = 0.6
                reg_weight = 0.4 * regression_r2  # scale by confidence
                total_w = kalman_weight + reg_weight
                if total_w > 0:
                    blended_slope = (
                        kalman_weight * kalman.temp_trend_per_min +
                        reg_weight * regression_slope
                    ) / total_w
                else:
                    blended_slope = kalman.temp_trend_per_min

                # Apply diurnal decay so the extrapolated warming rate
                # decelerates toward zero at the estimated peak time,
                # rather than projecting a constant rate indefinitely.
                peak_hour = (estimated_peak_mins or 900) / 60.0   # default 3 PM
                now_hour = now_local.hour + now_local.minute / 60.0
                decay = _diurnal_decay(minutes_ahead, now_hour, peak_hour)
                kalman_pred = kalman.smoothed_temp + blended_slope * minutes_ahead * decay

                # Blend with parametric diurnal curve when available
                target_hour = now_hour + minutes_ahead / 60.0
                if (diurnal_model is not None
                        and diurnal_model.is_reliable
                        and minutes_ahead > 0):
                    curve_pred = diurnal_model.predict(target_hour)
                    # Ramp: near-term trust Kalman, far-term trust curve
                    curve_weight = min(1.0, minutes_ahead / 120.0)
                    if diurnal_model.rmse > 2.0:
                        curve_weight *= 0.7
                    predicted = (1.0 - curve_weight) * kalman_pred + curve_weight * curve_pred
                else:
                    predicted = kalman_pred

                # Cap: never exceed current temp + ML-predicted remaining rise
                if remaining_rise is not None and remaining_rise >= 0:
                    max_predicted = kalman.smoothed_temp + remaining_rise
                    predicted = min(predicted, max_predicted)

                # Floor: during warming hours, don't predict below current temp
                if blended_slope > 0 and minutes_ahead > 0 and target_hour < peak_hour:
                    predicted = max(predicted, kalman.smoothed_temp)

                # Uncertainty grows with time ahead
                base_unc = kalman.uncertainty
                time_unc = abs(minutes_ahead) * 0.015 * kalman.process_noise_factor
                total_unc = math.sqrt(base_unc ** 2 + time_unc ** 2)

                trend_per_hour = blended_slope * 60.0 * decay

                predictions.append(StationTimePrediction(
                    obs_time=obs_time,
                    predicted_temp=round(predicted, 1),
                    uncertainty=round(total_unc, 1),
                    minutes_ahead=minutes_ahead,
                    is_past=is_past,
                    actual_temp=actual,
                    trend_per_hour=round(trend_per_hour, 1) if trend_per_hour else None,
                ))

    return predictions


# ---------------------------------------------------------------------------
# Composite peak timing
# ---------------------------------------------------------------------------

def compute_peak_timing(
    wu_hourly_peak_time: Optional[str],
    historical_peak_mins: Optional[float],
    kalman: Optional[KalmanState],
    current_hour_local: int,
    todays_obs: list[dict],
    dt_key: str = "observed_at",
    temp_key: str = "temp_f",
    city_tz: str = "America/New_York",
) -> dict:
    """Estimate when the daily peak temperature occurs/occurred.

    Fuses three sources:
      1. WU hourly forecast peak time (forward-looking)
      2. Historical average peak time (filtered for data quality)
      3. Kalman filter trend (real-time trajectory)

    Returns dict with: estimated_peak_time, confidence, source,
    peak_already_passed, detail.
    """
    tz = ZoneInfo(city_tz)
    result = {
        "estimated_peak_time": None,
        "estimated_peak_mins": None,  # minutes since midnight
        "confidence": "low",
        "source": "default",
        "peak_already_passed": False,
        "detail": "",
    }

    # Check if Kalman trend has been negative for extended period
    # → peak likely already passed.
    #
    # Lowered thresholds (n_observations >= 5, trend < -0.005°/min ≈ -0.3°/hr)
    # so the lock activates earlier in cities with sparse METAR cadence —
    # the previous gates of n>=10 and trend<-0.01 lagged by an hour or more
    # past the actual peak. We additionally require that the day's observed
    # max is at least 0.3°F above the current temperature, to avoid false
    # positives during a flat plateau.
    if (
        kalman
        and kalman.n_observations >= 5
        and kalman.temp_trend_per_min < -0.005
        and todays_obs
    ):
        max_obs = max(todays_obs, key=lambda o: o.get(temp_key, -999))
        latest_obs = todays_obs[-1]
        latest_temp = latest_obs.get(temp_key)
        max_temp = max_obs.get(temp_key)
        if (
            latest_temp is not None
            and max_temp is not None
            and (max_temp - latest_temp) >= 0.3
        ):
            max_dt = max_obs[dt_key]
            if max_dt.tzinfo is None:
                max_dt = max_dt.replace(tzinfo=timezone.utc)
            max_dt = max_dt.astimezone(tz)
            peak_mins = max_dt.hour * 60 + max_dt.minute
            result["estimated_peak_mins"] = peak_mins
            result["estimated_peak_time"] = max_dt.strftime("%-I:%M %p")
            result["peak_already_passed"] = True
            result["source"] = "actual_observed"
            result["confidence"] = "high"
            result["detail"] = (
                f"Kalman trend negative ({kalman.temp_trend_per_min * 60:.1f}°/hr) "
                f"for {kalman.n_observations} obs; "
                f"max {max_temp:.1f}°F is {max_temp - latest_temp:.1f}°F above "
                f"current {latest_temp:.1f}°F — peak already reached"
            )
            return result

    # Source estimates (minutes since midnight)
    estimates: list[tuple[float, float, str]] = []  # (mins, weight, label)

    # 1. WU hourly forecast peak time
    if wu_hourly_peak_time:
        wu_mins = _parse_time_to_mins(wu_hourly_peak_time)
        if wu_mins is not None:
            # Weight WU higher in morning, lower in afternoon
            wu_weight = 0.6 if current_hour_local < 14 else 0.3
            estimates.append((wu_mins, wu_weight, "wu_hourly"))

    # 2. Historical average peak timing
    if historical_peak_mins is not None and 600 < historical_peak_mins < 1200:
        hist_weight = 0.2  # always low — just a sanity check
        estimates.append((historical_peak_mins, hist_weight, "historical"))

    # 3. Kalman trend extrapolation (if still rising)
    if kalman and kalman.n_observations >= 5 and kalman.temp_trend_per_min > 0.005:
        # Extrapolate: how many more minutes until trend reaches zero?
        # Simple model: trend decays linearly based on typical diurnal pattern
        # Estimate ~2-3 hours until peak if trend is +0.02°/min at noon
        now_mins = current_hour_local * 60
        # Rough decay: peak occurs when rate drops to zero
        # Assume rate decays at ~0.005°/min per hour from current value
        decay_rate = 0.005 / 60.0  # per minute
        if decay_rate > 0 and kalman.temp_trend_per_min > 0:
            mins_to_peak = kalman.temp_trend_per_min / decay_rate
            mins_to_peak = min(mins_to_peak, 240)  # cap at 4 hours
            kalman_peak_mins = now_mins + mins_to_peak
            # Weight Kalman higher in afternoon (more data, trend more reliable)
            kalman_weight = 0.3 if current_hour_local < 14 else 0.5
            estimates.append((kalman_peak_mins, kalman_weight, "kalman_trend"))

    if not estimates:
        # Default: 3 PM
        result["estimated_peak_mins"] = 900
        result["estimated_peak_time"] = "3:00 PM"
        result["source"] = "default"
        result["confidence"] = "low"
        result["detail"] = "No data sources available — using 3 PM default"
        return result

    # Weighted average
    total_w = sum(w for _, w, _ in estimates)
    avg_mins = sum(m * w for m, w, _ in estimates) / total_w
    avg_mins = max(600, min(1200, avg_mins))  # clamp 10 AM – 8 PM

    result["estimated_peak_mins"] = round(avg_mins)
    h, m = divmod(int(avg_mins), 60)
    dummy = datetime(2000, 1, 1, h, m)
    result["estimated_peak_time"] = dummy.strftime("%-I:%M %p")
    result["source"] = "+".join(lbl for _, _, lbl in estimates)
    result["confidence"] = "high" if len(estimates) >= 2 else "medium"
    result["peak_already_passed"] = current_hour_local * 60 > avg_mins + 30
    result["detail"] = ", ".join(
        f"{lbl}: {int(mins // 60)}:{int(mins % 60):02d} (w={w:.2f})"
        for mins, w, lbl in estimates
    )

    return result


def _parse_time_to_mins(time_str: str) -> Optional[float]:
    """Parse '3:00 PM ET' or '15:00' to minutes since midnight."""
    import re
    time_str = time_str.strip()
    # Strip timezone suffix
    time_str = re.sub(r'\s*(ET|CT|MT|PT|EST|CST|MST|PST|EDT|CDT|MDT|PDT)\s*$', '', time_str, flags=re.I)
    time_str = time_str.strip()

    # Try 12-hour format: "3:00 PM"
    m = re.match(r'^(\d{1,2}):(\d{2})\s*(AM|PM)$', time_str, re.I)
    if m:
        h = int(m.group(1))
        mins = int(m.group(2))
        if m.group(3).upper() == "PM" and h != 12:
            h += 12
        elif m.group(3).upper() == "AM" and h == 12:
            h = 0
        return float(h * 60 + mins)

    # Try 24-hour format: "15:00"
    m = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)))

    return None


# ---------------------------------------------------------------------------
# Main entry point — called from signal_engine
# ---------------------------------------------------------------------------

def run_adaptive(
    todays_obs: list[dict],
    observation_minutes: list[int],
    now_local: datetime,
    city_tz: str = "America/New_York",
    wu_hourly_peak_time: Optional[str] = None,
    historical_peak_mins: Optional[float] = None,
    forecast_high: Optional[float] = None,
    ml_features: Optional[dict] = None,
) -> Optional[AdaptiveResult]:
    """Run the full adaptive prediction pipeline.

    Args:
        todays_obs: ALL MetarObs for today, as dicts with keys:
            observed_at, temp_f, and optionally wind_speed_kt, humidity_pct,
            cloud_cover, wx_string, altimeter_inhg, cloud_cover_val
        observation_minutes: station pattern minutes, e.g. [52]
        now_local: current datetime in city's local timezone
        city_tz: IANA timezone string
        wu_hourly_peak_time: e.g. "3:00 PM ET" from WU forecast
        historical_peak_mins: average historical peak in minutes since midnight
        forecast_high: fused NWS/WU daily high forecast (used for remaining-rise cap)
        ml_features: dict with temp_slope_3h, avg_peak_timing_mins, day_of_year
                     (used for ML remaining-rise prediction)
    """
    if not todays_obs or len(todays_obs) < 3:
        log.debug("adaptive: insufficient observations (%d)", len(todays_obs))
        return None

    # 1. Kalman filter over full day
    kalman = run_kalman(todays_obs)

    # 2. Rolling regression on recent window
    reg_slope, reg_r2, features_used = run_regression(todays_obs)

    # 3. Composite peak timing (computed before station predictions so
    #    the estimated peak time is available for diurnal decay)
    peak_info = compute_peak_timing(
        wu_hourly_peak_time=wu_hourly_peak_time,
        historical_peak_mins=historical_peak_mins,
        kalman=kalman,
        current_hour_local=now_local.hour,
        todays_obs=todays_obs,
        city_tz=city_tz,
    )

    # 4. ML remaining-rise cap (prevents runaway extrapolation)
    _remaining_rise: Optional[float] = None
    if ml_features is not None:
        try:
            from backend.modeling.residual_tracker import predict_remaining_rise
            _remaining_rise = predict_remaining_rise(
                hour_local=now_local.hour,
                current_temp_f=kalman.smoothed_temp,
                temp_slope_3h=ml_features.get("temp_slope_3h", 0.0),
                avg_peak_timing_mins=ml_features.get("avg_peak_timing_mins", 960.0),
                day_of_year=ml_features.get("day_of_year", now_local.timetuple().tm_yday),
            )
        except Exception:
            log.debug("adaptive: remaining-rise prediction failed, skipping cap")

    # 5. Fit parametric diurnal curve (Phase 2) for physically grounded
    #    far-term predictions.  Requires >= 6 observations and a forecast high.
    _diurnal_fit = None
    if len(todays_obs) >= 6 and forecast_high is not None:
        try:
            from backend.modeling.diurnal_model import fit_diurnal_curve
            _diurnal_fit = fit_diurnal_curve(
                todays_obs,
                forecast_high,
                peak_mins=peak_info.get("estimated_peak_mins"),
                city_tz=city_tz,
            )
            if _diurnal_fit is not None:
                log.debug(
                    "adaptive: diurnal fit T_max=%.1f peak=%.1fh rmse=%.2f n=%d reliable=%s",
                    _diurnal_fit.T_max, _diurnal_fit.t_peak,
                    _diurnal_fit.rmse, _diurnal_fit.n_obs_used,
                    _diurnal_fit.is_reliable,
                )
        except Exception:
            log.debug("adaptive: diurnal curve fit failed, using decay fallback")

    # 6. Station-time predictions (diurnal curve + decay + remaining-rise cap)
    predictions = compute_station_predictions(
        kalman=kalman,
        regression_slope=reg_slope,
        regression_r2=reg_r2,
        observation_minutes=observation_minutes,
        now_local=now_local,
        todays_obs=todays_obs,
        city_tz=city_tz,
        estimated_peak_mins=peak_info.get("estimated_peak_mins"),
        remaining_rise=_remaining_rise,
        diurnal_model=_diurnal_fit,
    )

    # 7. Predicted daily high from station predictions
    if predictions:
        best_pred = max(predictions, key=lambda p: p.predicted_temp)
        predicted_daily_high = best_pred.predicted_temp
        predicted_high_time = best_pred.obs_time
    else:
        predicted_daily_high = kalman.smoothed_temp
        predicted_high_time = None

    # 8. Sigma adjustment: if we have rich data, tighten uncertainty
    sigma_adj = 1.0
    if kalman.n_observations >= 20:
        sigma_adj *= 0.85
    if reg_r2 > 0.8:
        sigma_adj *= 0.9
    if len(features_used) >= 3:
        sigma_adj *= 0.95
    sigma_adj = max(0.6, sigma_adj)  # never over-tighten

    return AdaptiveResult(
        kalman=kalman,
        regression_slope=reg_slope,
        regression_r2=reg_r2,
        regression_features_used=features_used,
        station_predictions=predictions,
        predicted_daily_high=predicted_daily_high,
        predicted_high_time=predicted_high_time,
        sigma_adjustment=sigma_adj,
        peak_already_passed=peak_info["peak_already_passed"],
        composite_peak_timing=peak_info["estimated_peak_time"],
        peak_timing_source=peak_info["source"],
        remaining_rise_cap=_remaining_rise,
        diurnal_peak_estimate=_diurnal_fit.T_max if _diurnal_fit else None,
        diurnal_fit_rmse=_diurnal_fit.rmse if _diurnal_fit else None,
        diurnal_fit_active=(_diurnal_fit is not None and _diurnal_fit.is_reliable),
    )
