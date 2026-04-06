"""
Temperature forecast fusion + METAR intraday adjustment.

Fuses NWS API, WU daily, and WU hourly forecasts using calibrated weights.
Applies METAR-based intraday adjustment as the day progresses.

Key design decisions:
  - sigma never collapses below 1.0°F (minimum uncertainty)
  - METAR weight increases continuously through the day
  - When METAR-heavy (late day), sigma widens slightly to reflect
    that we're trusting a single ground observation more
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from scipy.stats import norm as _norm

from backend.modeling.distribution import bucket_probabilities, conditional_bucket_probabilities
from backend.modeling.residual_tracker import predict_remaining_rise, is_ml_model_loaded

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# ─── Time-of-day heuristics ───────────────────────────────────────────────────
# These represent expected *remaining* rise in temperature from current reading.
# Based on typical daily temperature curves for southeastern US cities.
_REMAINING_RISE_TABLE = [
    # (start_hour_local, end_hour_local, expected_remaining_rise_f)
    (0,  6,  14.0),   # overnight → large morning rise ahead
    (6,  9,  11.0),   # early morning → significant rise
    (9,  11,  8.0),   # mid-morning
    (11, 13,  4.0),   # late morning
    (13, 15,  2.0),   # early afternoon (peak warming period)
    (15, 17,  0.5),   # mid-afternoon (near peak)
    (17, 19,  0.0),   # late afternoon (likely at or past peak)
    (19, 24,  0.0),   # evening/night
]

# METAR weight by hour ET (observation weight vs. forecast weight)
# 0.0 = trust only forecasts; 1.0 = trust only METAR projection
_METAR_WEIGHT_TABLE = [
    (0,  9,  0.00),   # overnight/early morning — no reliable METAR signal yet
    (9,  12, 0.15),   # late morning
    (12, 14, 0.25),
    (14, 16, 0.40),
    (16, 18, 0.55),
    (18, 20, 0.70),   # late afternoon/evening — METAR is the dominant signal
    (20, 24, 0.99),
]


@dataclass
class ModelResult:
    mu: float
    sigma: float
    probs: list[float]
    mu_forecast: float
    mu_projected: float
    w_metar: float
    remaining_rise: float
    forecast_quality: str
    prob_new_high: float = 1.0
    inputs: dict = field(default_factory=dict)


def _interpolate_table(table: list, hour: int) -> float:
    for start, end, val in table:
        if start <= hour < end:
            return val
    return table[-1][2]


def _expected_remaining_rise(hour_local: int) -> float:
    return _interpolate_table(_REMAINING_RISE_TABLE, hour_local)


def _metar_weight(hour_local: int) -> float:
    return _interpolate_table(_METAR_WEIGHT_TABLE, hour_local)


def weather_adjusted_sigma(
    sigma_raw: float,
    cloud_cover_val: Optional[int] = None,
    humidity_pct: Optional[float] = None,
    wind_speed_kt: Optional[float] = None,
    wind_gust_kt: Optional[float] = None,
    pressure_tendency: Optional[float] = None,
    has_precip: bool = False,
    dewpoint_spread_f: Optional[float] = None,
) -> float:
    """Adjust sigma (forecast uncertainty) based on current weather conditions.

    Physical reasoning for each adjustment:
      - Overcast skies block shortwave radiation, capping diurnal heating
        → tighter temperature distribution.
      - High humidity / small dewpoint spread → moist air has higher Cp
        → smaller diurnal swing → lower uncertainty.
      - Strong sustained wind → deep boundary-layer mixing → spatially
        uniform temperatures → more predictable.
      - Large gust spread → turbulent micro-scale variability → wider.
      - Falling barometric pressure → approaching front / regime change
        → much wider uncertainty.
      - Active precipitation → evaporative cooling dominates the energy
        budget → temperature locked near wet-bulb → tighter distribution.

    Returns sigma_raw multiplied by a clamped factor in [0.5, 1.5].
    """
    factor = 1.0

    # Cloud cover (strongest single predictor of diurnal range)
    if cloud_cover_val is not None:
        # CLR=0: 1.0, FEW=1: 0.925, SCT=2: 0.85, BKN=3: 0.775, OVC=4: 0.70
        factor *= 1.0 - 0.075 * cloud_cover_val

    # Humidity / dewpoint spread
    if dewpoint_spread_f is not None:
        if dewpoint_spread_f < 5:
            factor *= 0.85
        elif dewpoint_spread_f < 10:
            factor *= 0.92
    elif humidity_pct is not None:
        if humidity_pct > 80:
            factor *= 0.85
        elif humidity_pct > 60:
            factor *= 0.92

    # Wind: sustained wind tightens, gusts widen
    if wind_speed_kt is not None:
        if wind_speed_kt > 15:
            factor *= 0.90
        elif wind_speed_kt > 8:
            factor *= 0.95
    if wind_gust_kt is not None and wind_speed_kt is not None:
        gust_spread = wind_gust_kt - wind_speed_kt
        if gust_spread > 10:
            factor *= 1.10

    # Pressure tendency (falling = front approaching = wider)
    if pressure_tendency is not None:
        if pressure_tendency < -0.06:
            factor *= 1.25
        elif pressure_tendency < -0.03:
            factor *= 1.10

    # Active precipitation constrains temps strongly
    if has_precip:
        factor *= 0.80

    return sigma_raw * max(0.5, min(1.5, factor))


def compute_model(
    nws_high: Optional[float],
    wu_daily_high: Optional[float],
    wu_hourly_peak: Optional[float],
    daily_high_metar: Optional[float],
    current_temp_f: Optional[float],
    calibration: Optional[dict],
    buckets: list[tuple[Optional[float], Optional[float]]],
    forecast_quality: str = "ok",
    unit: str = "F",
    city_tz: str = "America/New_York",
    observed_high: Optional[float] = None,
    ml_features: Optional[dict] = None,
    adaptive=None,
    latest_weather: Optional[dict] = None,
    hrrr_high: Optional[float] = None,
    gfs_high: Optional[float] = None,
) -> Optional[ModelResult]:
    """
    Fuse all forecast sources and compute temperature distribution + bucket probabilities.

    Args:
        nws_high: NWS API daily high forecast (units match 'unit')
        wu_daily_high: WU daily high scrape
        wu_hourly_peak: max(WU hourly temps)
        daily_high_metar: max observed temp today
        current_temp_f: latest METAR temperature
        calibration: dict with bias_nws, bias_wu_daily, bias_wu_hourly,
                     weight_nws, weight_wu_daily, weight_wu_hourly
        buckets: list of (lo, hi) bucket boundaries
        forecast_quality: "ok" | "degraded"
        unit: "F" or "C"

    Returns:
        ModelResult or None if insufficient data.
    """
    # Apply calibration bias corrections
    cal = calibration or {}
    bias_nws = cal.get("bias_nws", 0.0)
    bias_wud = cal.get("bias_wu_daily", 0.0)
    bias_wuh = cal.get("bias_wu_hourly", 0.0)
    w_nws = cal.get("weight_nws", 1/3)
    w_wud = cal.get("weight_wu_daily", 1/3)
    w_wuh = cal.get("weight_wu_hourly", 1/3)
    # WU Nighttime Rollover Protection
    # If it's past 18:00 local and the WU daily forecast deviates wildly from the actual METAR high so far,
    # it's highly likely WU has rolled over its "Today" block to display tomorrow's high. We discard it.
    local_tz = ZoneInfo(city_tz)
    now_local = datetime.now(local_tz)
    hour_local = now_local.hour
    if hour_local >= 18 and wu_daily_high is not None and daily_high_metar is not None:
        unit_dev = 6.0 if unit == "C" else 10.0
        if abs(wu_daily_high - daily_high_metar) > unit_dev:
            log.warning("model: dropping wu_daily_high (%.1f) due to likely nighttime rollover (metar=%.1f)", wu_daily_high, daily_high_metar)
            wu_daily_high = None

    calibrated = {}
    if nws_high is not None:
        calibrated["nws"] = (nws_high + bias_nws, w_nws)
    if wu_daily_high is not None:
        calibrated["wu_daily"] = (wu_daily_high + bias_wud, w_wud)
    if wu_hourly_peak is not None:
        calibrated["wu_hourly"] = (wu_hourly_peak + bias_wuh, w_wuh)
    if hrrr_high is not None:
        calibrated["hrrr"] = (hrrr_high + cal.get("bias_hrrr", 0.0), cal.get("weight_hrrr", 0.5))
    if gfs_high is not None:
        calibrated["gfs"] = (gfs_high + cal.get("bias_gfs", 0.0), cal.get("weight_gfs", 0.2))

    if not calibrated:
        log.warning("model: no forecast sources available — cannot compute model")
        return None

    # Weighted mean of available sources (re-normalize weights)
    total_weight = sum(w for _, w in calibrated.values())
    mu_forecast = sum(v * w for v, w in calibrated.values()) / total_weight

    # Scale factor for Celsius
    unit_mult = 5.0 / 9.0 if unit == "C" else 1.0

    # Uncertainty from disagreement
    vals = [v for v, _ in calibrated.values()]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        sigma_raw = max(1.0 * unit_mult, spread / 2.0)
    else:
        # Only one source — use a conservative base uncertainty
        sigma_raw = 2.5 * unit_mult

    # ── Weather-conditioned sigma adjustment ─────────────────────────────────
    # Adjust base sigma using current weather conditions (cloud cover,
    # humidity, wind, pressure tendency, precipitation).  This replaces
    # the regime-agnostic forecast-spread-only sigma with a physically
    # grounded heteroscedastic estimate.
    if latest_weather:
        wx = latest_weather
        sigma_raw = weather_adjusted_sigma(
            sigma_raw,
            cloud_cover_val=wx.get("cloud_cover_val"),
            humidity_pct=wx.get("humidity_pct"),
            wind_speed_kt=wx.get("wind_speed_kt"),
            wind_gust_kt=wx.get("wind_gust_kt"),
            pressure_tendency=wx.get("pressure_tendency"),
            has_precip=bool(wx.get("has_precip")),
            dewpoint_spread_f=wx.get("dewpoint_spread_f"),
        )

    # ── METAR intraday adjustment ──────────────────────────────────────────────
    w_metar = _metar_weight(hour_local)

    # Use ML-based remaining rise prediction if features are available
    _ml = ml_features or {}
    remaining_rise = predict_remaining_rise(
        hour_local=hour_local,
        current_temp_f=current_temp_f or 70.0,
        temp_slope_3h=_ml.get("temp_slope_3h", 0.0),
        avg_peak_timing_mins=_ml.get("avg_peak_timing_mins", 960.0),
        day_of_year=_ml.get("day_of_year", now_local.timetuple().tm_yday),
        unit_mult=unit_mult,
    )

    # Projected high = max(daily high so far, current + expected rise)
    projected_high = mu_forecast  # default to forecast if no METAR

    if daily_high_metar is not None:
        # Daily high observed so far is a floor
        obs_projected = daily_high_metar + remaining_rise
        projected_high = max(daily_high_metar, obs_projected)

        if current_temp_f is not None:
            # Current temp + rise may be higher than daily high so far
            current_projected = current_temp_f + remaining_rise
            projected_high = max(projected_high, current_projected)

    # After 5 PM ET with declining temps, cap projected_high —
    # no meaningful temperature rises expected after sunset.
    if hour_local >= 17 and daily_high_metar is not None:
        projected_ceiling = max(
            daily_high_metar,
            (current_temp_f or daily_high_metar) + 1.0,
        )
        projected_high = min(projected_high, projected_ceiling)

    # ── Adaptive prediction integration ──────────────────────────────────────
    # Use Kalman + regression predicted high as an additional projected_high
    # candidate, weighted by the adaptive engine's confidence.
    adaptive_high = None
    if adaptive is not None:
        adaptive_high = adaptive.predicted_daily_high
        if adaptive_high is not None and daily_high_metar is not None:
            # Adaptive high can't be below already-observed high
            adaptive_high = max(adaptive_high, daily_high_metar)
        if adaptive_high is not None:
            # Blend adaptive into projected_high: weight increases with data
            n_obs = adaptive.kalman.n_observations if adaptive.kalman else 0
            adaptive_w = min(0.4, n_obs * 0.02)  # caps at 0.4 with 20+ obs

            # Time-of-day cap: before peak heating, adaptive has insufficient
            # diurnal data and tends to under-predict the day's high.
            if hour_local < 11:
                adaptive_w = min(adaptive_w, 0.15)
            elif hour_local < 13:
                adaptive_w = min(adaptive_w, 0.25)

            # Consensus-divergence dampening: when adaptive disagrees with
            # the forecast consensus by more than sigma, reduce its influence.
            divergence = abs(adaptive_high - mu_forecast)
            divergence_threshold = max(2.0 * unit_mult, sigma_raw)
            if divergence > divergence_threshold:
                dampening = max(0.1, 1.0 - (divergence - divergence_threshold) / (3.0 * divergence_threshold))
                log.info(
                    "model: adaptive dampened %.2f -> %.2f (divergence=%.1f°, threshold=%.1f°)",
                    adaptive_w, adaptive_w * dampening, divergence, divergence_threshold,
                )
                adaptive_w *= dampening

            projected_high = (1.0 - adaptive_w) * projected_high + adaptive_w * adaptive_high

    # Weighted combination
    mu_final = (1.0 - w_metar) * mu_forecast + w_metar * projected_high

    # When heavily relying on METAR, widen sigma slightly to reflect
    # that a single ground observation is noisy
    sigma_final = sigma_raw * (1.0 + 0.2 * w_metar)

    # Apply adaptive sigma adjustment (tightens when trend data is rich)
    if adaptive is not None:
        sigma_final *= adaptive.sigma_adjustment

    sigma_final = max(1.0 * unit_mult, sigma_final)

    # ── Compute bucket probabilities ───────────────────────────────────────────
    if not buckets:
        log.warning("model: no buckets to compute probabilities for")
        probs = []
    elif observed_high is not None:
        # Daily high can only go up — zero out buckets already surpassed
        probs = conditional_bucket_probabilities(
            mu_final, sigma_final, buckets, floor=observed_high
        )
    else:
        probs = bucket_probabilities(mu_final, sigma_final, buckets)

    # ── Prob new high ────────────────────────────────────────────────────────────
    # P(final daily high > current observed max) using the model distribution.
    if daily_high_metar is not None and sigma_final > 0:
        prob_new_high = float(1.0 - _norm.cdf(daily_high_metar, mu_final, sigma_final))
    else:
        prob_new_high = 1.0  # no observation yet — high hasn't been established

    inputs = {
        "nws_high": nws_high,
        "wu_daily_high": wu_daily_high,
        "wu_hourly_peak": wu_hourly_peak,
        "hrrr_high": hrrr_high,
        "gfs_high": gfs_high,
        "daily_high_metar": daily_high_metar,
        "current_temp_f": current_temp_f,
        "mu_forecast": float(mu_forecast),
        "projected_high": float(projected_high),
        "w_metar": float(w_metar),
        "remaining_rise": remaining_rise,
        "hour_local": hour_local,
        "spread": float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0,
        "sigma_raw": float(sigma_raw),
        "sources_used": list(calibrated.keys()),
        "forecast_quality": forecast_quality,
        "prob_new_high": round(prob_new_high, 4),
        "observed_high": observed_high,
    }

    # Adaptive engine audit data
    if adaptive is not None:
        inputs["adaptive"] = {
            "kalman_temp": round(adaptive.kalman.smoothed_temp, 1),
            "kalman_trend_per_hr": round(adaptive.kalman.temp_trend_per_min * 60, 2),
            "kalman_n_obs": adaptive.kalman.n_observations,
            "regression_slope_per_hr": round(adaptive.regression_slope * 60, 2),
            "regression_r2": round(adaptive.regression_r2, 3),
            "regression_features": adaptive.regression_features_used,
            "predicted_daily_high": round(adaptive.predicted_daily_high, 1),
            "sigma_adjustment": round(adaptive.sigma_adjustment, 3),
            "peak_already_passed": adaptive.peak_already_passed,
            "composite_peak_timing": adaptive.composite_peak_timing,
            "peak_timing_source": adaptive.peak_timing_source,
        }

    return ModelResult(
        mu=float(mu_final),
        sigma=float(sigma_final),
        probs=probs,
        mu_forecast=float(mu_forecast),
        mu_projected=float(projected_high),
        w_metar=float(w_metar),
        remaining_rise=remaining_rise,
        forecast_quality=forecast_quality,
        prob_new_high=prob_new_high,
        inputs=inputs,
    )
