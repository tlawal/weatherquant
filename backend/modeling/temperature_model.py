"""
Temperature forecast fusion + METAR intraday adjustment.

Fuses five forecast sources using calibrated weights:
  - NWS API (US only)
  - WU hourly peak (via weather.com v1 API)
  - HRRR (via Open-Meteo, GFS+HRRR blend, US CONUS)
  - NBM (via Open-Meteo, NCEP National Blend of Models, US CONUS)
  - ECMWF IFS (via Open-Meteo, global 9–25 km, top-2 global NWP model)

The ensemble μ is primarily the multi-model mean. The Kalman nowcast
is blended in dynamically via `compute_kalman_weight`: it only earns
weight inside a ±2h tent around the anticipated peak and only when it
agrees with the multi-model panel within the panel's own spread.
Outside that window or when badly diverged the weight collapses to 0
and mu_forecast = mu_multi_model. METAR-based intraday adjustment is
applied downstream via the _METAR_WEIGHT_TABLE, unchanged.

Key design decisions:
  - sigma never collapses below 1.0°F (minimum uncertainty)
  - METAR weight increases continuously through the day
  - When METAR-heavy (late day), sigma widens slightly to reflect
    that we're trusting a single ground observation more
  - Kalman is a nowcast, not a forecast: weighted only where its short-
    horizon skill applies, and a secondary <=10% residual nudge into
    projected_high gated on the same positive weight.
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
from backend.modeling.settlement import (
    bucket_upper_bound,
    canonical_bucket_ranges,
    find_bucket_idx_for_value,
    hotter_bucket_floor,
)
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
    (20, 24, 0.85),   # capped at 0.85 — forecasts always retain ≥15% influence
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
    prob_hotter_bucket: float = 1.0
    prob_new_high_raw: float = 1.0
    lock_regime: bool = False
    observed_bucket_idx: Optional[int] = None
    observed_bucket_upper_f: Optional[float] = None
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


def compute_kalman_weight(
    hour_local_fractional: float,
    peak_hour_local: Optional[float],
    kalman_divergence: float,
    spread: float,
    n_obs: int,
    peak_already_passed: bool,
    max_weight: float = 0.45,
    half_window_hours: float = 2.0,
) -> float:
    """Ensemble weight for the Kalman nowcast slice of mu_forecast.

    The filter has short-horizon skill; it earns its keep in a ±2h tent
    centered on the anticipated peak. Outside that window the multi-model
    panel (ECMWF IFS, HRRR, NBM, NWS, WU) is strictly more skillful. When
    the Kalman nowcast disagrees with consensus by more than the panel's
    own spread, scale down sharply — a lone filter beating five physics
    models is almost always the filter missing a regime change.

    All temperature arguments are in the same unit (°F or °C). The helper
    only uses ratios, so it is unit-agnostic.

    Returns 0.0 when the filter is out-of-window, under-observed, or badly
    diverged. Never exceeds `max_weight`.
    """
    if n_obs < 10 or peak_hour_local is None:
        return 0.0
    window_factor = max(0.0, 1.0 - abs(hour_local_fractional - peak_hour_local) / half_window_hours)
    if window_factor <= 0.0:
        return 0.0
    div_budget = max(spread, 3.0)
    div_excess = max(0.0, kalman_divergence - div_budget)
    div_penalty = max(0.0, 1.0 - div_excess / div_budget)
    if peak_already_passed:
        window_factor *= 0.5
    return max_weight * window_factor * div_penalty


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sum_hotter_bucket_probabilities(probs: list[float], observed_bucket_idx: Optional[int]) -> float:
    if observed_bucket_idx is None:
        return 1.0 if probs else 0.0
    return float(sum(probs[observed_bucket_idx + 1:]))


def _reallocate_hotter_tail(
    probs: list[float],
    observed_bucket_idx: int,
    hotter_prob: float,
) -> list[float]:
    adjusted = [0.0 for _ in probs]
    adjusted[observed_bucket_idx] = max(0.0, 1.0 - hotter_prob)
    tail = probs[observed_bucket_idx + 1:]
    tail_sum = sum(tail)
    if hotter_prob > 0 and tail_sum > 0:
        for idx, prob in enumerate(tail, start=observed_bucket_idx + 1):
            adjusted[idx] = hotter_prob * prob / tail_sum
    return adjusted


def _late_day_lock_active(
    *,
    observed_high: Optional[float],
    current_temp_f: Optional[float],
    remaining_rise: float,
    adaptive,
    hour_local: int,
    unit_mult: float,
) -> bool:
    """Decide whether to engage the late-day lock regime.

    The lock concentrates probability on the bucket containing the observed
    high once we are confident the day's peak is in. We do not require the
    adaptive engine to have flagged peak_already_passed — that detector can
    lag by an hour or more in sparse-METAR cities. Instead we add an
    observation-grounded fallback path so the lock still engages once it is
    physically clear we are post-peak (after 18:00 local, no residual rise,
    current temp at least 0.5°F below observed high, trend non-rising).
    """
    if observed_high is None or current_temp_f is None:
        return False
    if remaining_rise > 0.25 * unit_mult:
        return False
    if current_temp_f > observed_high - (0.5 * unit_mult):
        return False

    trend_per_hr = (
        adaptive.kalman.temp_trend_per_min * 60.0
        if adaptive is not None and adaptive.kalman is not None
        else None
    )

    # Path 1 — adaptive engine declared peak passed (existing behavior).
    if adaptive is not None and adaptive.peak_already_passed:
        return hour_local >= 18 or trend_per_hr is None or trend_per_hr <= 0.25

    # Path 2 — observation fallback: physically post-peak even if adaptive lags.
    # All four floor checks above already passed, so we know:
    #   - observed_high and current_temp_f exist
    #   - remaining_rise <= 0.25°F
    #   - current_temp is at least 0.5°F below observed_high
    # The remaining condition is "we're past the typical late-day window AND
    # the trend isn't actively rising back up". Trend may be unknown (no
    # adaptive), in which case the time-of-day check alone is sufficient.
    if hour_local >= 18 and (trend_per_hr is None or trend_per_hr <= 0.25):
        return True

    # Path 3 — strong-cooling override for west-coast / late-peak cities.
    # Seattle regression: at 17:47 local with observed 55°F, current 53.6°F
    # (deficit 1.4°F) and a clearly negative Kalman trend, `peak_already_passed`
    # hadn't flipped and path 2's 18:00 gate was too late. When the trend is
    # firmly negative with reliable observations and we are at/past 15:00
    # local (after solar noon for all continental US cities), the physics is
    # unambiguous — lock.
    n_obs = adaptive.kalman.n_observations if adaptive is not None and adaptive.kalman is not None else 0
    deficit = observed_high - current_temp_f
    if (
        hour_local >= 15
        and trend_per_hr is not None
        and trend_per_hr <= -0.25
        and n_obs >= 10
        and deficit >= 1.0 * unit_mult
    ):
        return True

    return False


def _compute_lock_probs(
    *,
    canonical_buckets: list[tuple[Optional[float], Optional[float]]],
    observed_high: float,
    observed_bucket_idx: int,
    current_temp_f: Optional[float],
    adaptive,
    existing_probs: list[float],
    existing_hotter_prob: float,
    unit_mult: float,
) -> tuple[list[float], float, float, float, float]:
    trend_per_hr = adaptive.kalman.temp_trend_per_min * 60.0 if adaptive and adaptive.kalman else 0.0
    future_predictions = []
    if adaptive is not None:
        future_predictions = [
            pred for pred in (adaptive.station_predictions or [])
            if not pred.is_past and pred.minutes_ahead > 0
        ]

    if future_predictions:
        hottest_future = max(future_predictions, key=lambda pred: pred.predicted_temp)
        hottest_prediction = float(hottest_future.predicted_temp)
        hottest_uncertainty = float(hottest_future.uncertainty or (0.25 * unit_mult))
        lock_sigma = _clamp(hottest_uncertainty, 0.15 * unit_mult, 0.50 * unit_mult)

        deficit_from_high = observed_high - hottest_prediction
        if trend_per_hr <= 0.0 and deficit_from_high >= 1.5 * unit_mult:
            lock_sigma = min(lock_sigma, 0.15 * unit_mult)
        elif trend_per_hr <= 0.0 and deficit_from_high >= 0.5 * unit_mult:
            lock_sigma = min(lock_sigma, 0.25 * unit_mult)

        lock_mu = max(observed_high, hottest_prediction)
        lock_probs = conditional_bucket_probabilities(
            lock_mu,
            lock_sigma,
            canonical_buckets,
            floor=observed_high,
        )
        return lock_probs, _sum_hotter_bucket_probabilities(lock_probs, observed_bucket_idx), lock_mu, lock_sigma, max(observed_high, hottest_prediction)

    deficit_from_high = (observed_high - current_temp_f) if current_temp_f is not None else 0.0
    if trend_per_hr <= 0.0 and deficit_from_high >= 1.5 * unit_mult:
        cap = min(existing_hotter_prob, 0.02)
    elif trend_per_hr <= 0.25 and deficit_from_high >= 0.5 * unit_mult:
        cap = min(existing_hotter_prob, 0.05)
    else:
        cap = min(existing_hotter_prob, 0.10)

    fallback_sigma = 0.15 * unit_mult if deficit_from_high >= 1.5 * unit_mult and trend_per_hr <= 0.0 else 0.25 * unit_mult
    lock_probs = _reallocate_hotter_tail(existing_probs, observed_bucket_idx, cap)
    return lock_probs, cap, observed_high, fallback_sigma, observed_high


def compute_model(
    nws_high: Optional[float],
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
    nbm_high: Optional[float] = None,
    ecmwf_ifs_high: Optional[float] = None,
) -> Optional[ModelResult]:
    """
    Fuse all forecast sources and compute temperature distribution + bucket probabilities.

    Args:
        nws_high: NWS API daily high forecast (units match 'unit')
        wu_hourly_peak: max(WU hourly temps) from weather.com v1 API
        daily_high_metar: max observed temp today
        current_temp_f: latest METAR temperature
        calibration: dict with bias_nws, bias_wu_hourly,
                     weight_nws, weight_wu_hourly
        buckets: list of (lo, hi) bucket boundaries
        forecast_quality: "ok" | "degraded"
        unit: "F" or "C"

    Returns:
        ModelResult or None if insufficient data.
    """
    # Apply calibration bias corrections.
    # Historical CalibrationParams rows may still carry baked-in weight_wu_daily
    # values; we ignore them and renormalize defaults across the remaining sources.
    cal = calibration or {}
    bias_nws = cal.get("bias_nws", 0.0)
    bias_wuh = cal.get("bias_wu_hourly", 0.0)
    w_nws = cal.get("weight_nws", 0.5)
    w_wuh = cal.get("weight_wu_hourly", 0.5)
    local_tz = ZoneInfo(city_tz)
    now_local = datetime.now(local_tz)
    hour_local = now_local.hour

    calibrated = {}
    if nws_high is not None:
        calibrated["nws"] = (nws_high + bias_nws, w_nws)
    if wu_hourly_peak is not None:
        calibrated["wu_hourly"] = (wu_hourly_peak + bias_wuh, w_wuh)
    if hrrr_high is not None:
        calibrated["hrrr"] = (hrrr_high + cal.get("bias_hrrr", 0.0), cal.get("weight_hrrr", 0.5))
    if nbm_high is not None:
        calibrated["nbm"] = (nbm_high + cal.get("bias_nbm", 0.0), cal.get("weight_nbm", 0.2))
    if ecmwf_ifs_high is not None:
        calibrated["ecmwf_ifs"] = (
            ecmwf_ifs_high + cal.get("bias_ecmwf_ifs", 0.0),
            cal.get("weight_ecmwf_ifs", 0.5),
        )

    if not calibrated:
        log.warning("model: no forecast sources available — cannot compute model")
        return None

    # Multi-model weighted mean of available sources (re-normalize weights).
    total_weight = sum(w for _, w in calibrated.values())
    mu_multi_model = sum(v * w for v, w in calibrated.values()) / total_weight

    vals = [v for v, _ in calibrated.values()]
    spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    # ── Multi-model + dynamic Kalman blend ─────────────────────────────────
    # Kalman is a short-horizon nowcast, not a forecast. It earns weight
    # only in a ±2h tent around the anticipated peak and only when it
    # agrees with the multi-model panel within the panel's own spread.
    # Outside that window or when badly diverged, consensus wins — the
    # filter has no physics and no synoptic context.
    hour_local_fractional = now_local.hour + now_local.minute / 60.0
    kalman_blend_high: Optional[float] = None
    kalman_divergence: Optional[float] = None
    peak_hour_local: Optional[float] = None
    kalman_w: float = 0.0
    mode_reason = "no_adaptive"

    if adaptive is not None and getattr(adaptive, "predicted_daily_high", None) is not None:
        kalman_blend_high = float(adaptive.predicted_daily_high)
        if daily_high_metar is not None:
            kalman_blend_high = max(kalman_blend_high, daily_high_metar)
        kalman_divergence = abs(kalman_blend_high - mu_multi_model)
        if getattr(adaptive, "predicted_high_time", None) is not None:
            peak_dt_local = adaptive.predicted_high_time.astimezone(local_tz)
            peak_hour_local = peak_dt_local.hour + peak_dt_local.minute / 60.0
        n_obs = adaptive.kalman.n_observations if adaptive.kalman else 0
        kalman_w = compute_kalman_weight(
            hour_local_fractional=hour_local_fractional,
            peak_hour_local=peak_hour_local,
            kalman_divergence=kalman_divergence,
            spread=spread,
            n_obs=n_obs,
            peak_already_passed=adaptive.peak_already_passed,
        )
        if kalman_w > 0.0:
            mode_reason = "in_peak_window"
        elif n_obs < 10:
            mode_reason = "low_obs"
        elif peak_hour_local is None:
            mode_reason = "no_peak_time"
        elif abs(hour_local_fractional - (peak_hour_local or 0.0)) > 2.0:
            mode_reason = "outside_peak_window"
        else:
            mode_reason = "diverged"

    kalman_nowcast_active = kalman_w > 0.0
    if kalman_nowcast_active and kalman_blend_high is not None:
        mu_forecast = (1.0 - kalman_w) * mu_multi_model + kalman_w * kalman_blend_high
        mode = f"blend_multi_kalman_w{kalman_w:.2f}"
    else:
        mu_forecast = mu_multi_model
        mode = f"multi_only:{mode_reason}"

    ensemble_breakdown = {
        "multi_model": round(mu_multi_model, 2),
        "kalman": round(kalman_blend_high, 2) if kalman_blend_high is not None else None,
        "kalman_weight": round(kalman_w, 3),
        "peak_hour_local": round(peak_hour_local, 2) if peak_hour_local is not None else None,
        "mode": mode,
    }

    # Scale factor for Celsius
    unit_mult = 5.0 / 9.0 if unit == "C" else 1.0
    canonical_buckets = canonical_bucket_ranges(buckets)

    # Uncertainty from disagreement
    if len(vals) >= 2:
        sigma_raw = max(1.0 * unit_mult, spread / 2.0)
    else:
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

    # When the live trend says we have much more heating ahead than the
    # static table expects, lift `remaining_rise` to match the trajectory
    # — capped against the forecast panel's upper envelope so we never
    # project above max(sources) + 2°F headroom. This fixes the mid-morning
    # under-projection that previously locked `projected_high` near the
    # current reading on hot days (e.g. Atlanta at 11:00, 77°F, trend
    # 3.77°F/hr, peak 15:43 → trend_rise ~15°F vs static 4°F).
    if (
        adaptive is not None
        and adaptive.kalman is not None
        and adaptive.kalman.n_observations >= 10
        and adaptive.kalman.temp_trend_per_min > 0.0
        and peak_hour_local is not None
        and current_temp_f is not None
        and vals
    ):
        hours_to_peak = max(0.0, peak_hour_local - hour_local_fractional)
        trend_rise = adaptive.kalman.temp_trend_per_min * 60.0 * hours_to_peak
        headroom = max(0.0, max(vals) + 2.0 * unit_mult - current_temp_f)
        trend_rise = min(trend_rise, headroom)
        remaining_rise = max(remaining_rise, trend_rise)

    # Collapse remaining_rise to 0 when we are clearly past peak and
    # cooling. Physics: the daily high can only go up, and once the
    # observed high is firmly above current temp with a negative Kalman
    # trend, a fresh rise is not plausible (Seattle regression: 17:47
    # with observed 55°F, current 53.6°F, residual predictor was still
    # emitting ~0.4°F remaining rise and inflating projected_high).
    if (
        adaptive is not None
        and adaptive.kalman is not None
        and adaptive.kalman.n_observations >= 5
        and adaptive.kalman.temp_trend_per_min < -0.005  # ~-0.3°F/hr
        and daily_high_metar is not None
        and current_temp_f is not None
        and current_temp_f < daily_high_metar - 0.5 * unit_mult
    ):
        remaining_rise = 0.0

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

    # ── Adaptive residual blend into projected_high ──────────────────────────
    # Kalman is already the 30% slice of mu_forecast (70/30 blend above),
    # gated to hour_local >= 11. Here we apply only a light residual blend
    # (<=10%) into projected_high so the METAR intraday path can still be
    # nudged toward the Kalman nowcast without re-stacking the Kalman
    # weight on top of the ensemble slice.
    adaptive_high = None
    if adaptive is not None and getattr(adaptive, "predicted_daily_high", None) is not None:
        adaptive_high = adaptive.predicted_daily_high
        if daily_high_metar is not None:
            adaptive_high = max(adaptive_high, daily_high_metar)
        if kalman_nowcast_active and adaptive_high is not None:
            n_obs = adaptive.kalman.n_observations if adaptive.kalman else 0
            adaptive_w = min(0.10, n_obs * 0.01)  # caps at 0.10 with 10+ obs
            if hour_local >= 19:
                adaptive_w = min(adaptive_w, 0.03)  # high is locked
            elif hour_local >= 17:
                adaptive_w = min(adaptive_w, 0.05)
            projected_high = (1.0 - adaptive_w) * projected_high + adaptive_w * adaptive_high

    # Universal ceiling against forecast consensus (defense in depth).
    # Stale or anomalous observed-high values (e.g. a daily_high_metar
    # leaking across midnight, or a bad single observation) must not let
    # projected_high run past the live forecast panel by more than a
    # small headroom. Real hot-day spikes still pass — the cap is
    # max(sources) + 2°F — but 7°F+ artifacts get clipped.
    projected_high_capped = False
    if vals:
        consensus_ceiling = max(vals) + 2.0 * unit_mult
        if projected_high > consensus_ceiling:
            projected_high = consensus_ceiling
            projected_high_capped = True

    # After 5 PM with declining temps, hard-cap projected_high.
    # This is a physical constraint (no meaningful rises after sunset)
    # and must be applied AFTER adaptive blending to prevent re-inflation.
    if hour_local >= 17 and daily_high_metar is not None:
        projected_ceiling = max(
            daily_high_metar,
            (current_temp_f or daily_high_metar) + 1.0,
        )
        projected_high = min(projected_high, projected_ceiling)

    # ── Divergence-aware METAR weight reduction ──────────────────────────────
    # When the METAR projection diverges significantly from the multi-model
    # forecast consensus, the METAR data is likely incomplete (missed the peak
    # due to sparse observations, station gaps, etc.). Reduce w_metar and
    # inflate sigma to reflect this uncertainty.
    divergence = abs(projected_high - mu_forecast)
    divergence_f = divergence / unit_mult if unit_mult else divergence
    if divergence_f > 3.0:
        # Linear penalty: 0% at 3°F, 50% reduction at 10°F divergence
        penalty = min(1.0, (divergence_f - 3.0) / 7.0)
        w_metar *= (1.0 - penalty * 0.50)
        log.debug(
            "model: divergence penalty %.1f°F → w_metar %.3f → %.3f",
            divergence_f, _metar_weight(hour_local), w_metar,
        )

    # ── Observation-density gating ────────────────────────────────────────────
    # With few METAR observations (e.g. 8 vs expected ~24 for hourly station),
    # the daily high may have missed the actual peak. Scale down w_metar.
    if adaptive is not None and adaptive.kalman is not None:
        n_obs = adaptive.kalman.n_observations
        if n_obs < 12:
            density_factor = n_obs / 12.0  # 0.0 at 0 obs, 1.0 at 12+
            w_metar *= max(0.30, density_factor)
            log.debug(
                "model: sparse obs (%d) → density_factor %.2f → w_metar %.3f",
                n_obs, density_factor, w_metar,
            )

    # Weighted combination
    mu_final = (1.0 - w_metar) * mu_forecast + w_metar * projected_high

    # As METAR observations accumulate through the day (w_metar rises),
    # forecast spread becomes less relevant because ground truth dominates.
    # Blend from full forecast sigma toward a tight observation-based sigma.
    observation_sigma = max(1.0, remaining_rise + 0.5) * unit_mult
    # Inflate observation sigma when METAR and forecasts diverge —
    # disagreement means more uncertainty, not less.
    if divergence_f > 3.0:
        observation_sigma += divergence * 0.3
    sigma_final = (1.0 - w_metar) * sigma_raw + w_metar * observation_sigma

    # Apply adaptive sigma adjustment (tightens when trend data is rich)
    if adaptive is not None:
        sigma_final *= adaptive.sigma_adjustment

    sigma_final = max(1.0 * unit_mult, sigma_final)

    # ── Compute bucket probabilities ───────────────────────────────────────────
    if not canonical_buckets:
        log.warning("model: no buckets to compute probabilities for")
        probs = []
    elif observed_high is not None:
        probs = conditional_bucket_probabilities(
            mu_final, sigma_final, canonical_buckets, floor=observed_high
        )
    else:
        probs = bucket_probabilities(mu_final, sigma_final, canonical_buckets)

    observed_bucket_idx = find_bucket_idx_for_value(canonical_buckets, observed_high)
    observed_bucket_upper_f = bucket_upper_bound(canonical_buckets, observed_bucket_idx)

    if observed_high is not None and sigma_final > 0:
        prob_new_high_raw = float(1.0 - _norm.cdf(observed_high, mu_final, sigma_final))
    else:
        prob_new_high_raw = 1.0

    if observed_high is not None and observed_bucket_idx is not None:
        prob_hotter_bucket = _sum_hotter_bucket_probabilities(probs, observed_bucket_idx)
    else:
        prob_hotter_bucket = prob_new_high_raw

    lock_regime = _late_day_lock_active(
        observed_high=observed_high,
        current_temp_f=current_temp_f,
        remaining_rise=remaining_rise,
        adaptive=adaptive,
        hour_local=hour_local,
        unit_mult=unit_mult,
    )

    if lock_regime and observed_high is not None and observed_bucket_idx is not None:
        lock_probs, prob_hotter_bucket, lock_mu, lock_sigma, lock_projected_high = _compute_lock_probs(
            canonical_buckets=canonical_buckets,
            observed_high=observed_high,
            observed_bucket_idx=observed_bucket_idx,
            current_temp_f=current_temp_f,
            adaptive=adaptive,
            existing_probs=probs,
            existing_hotter_prob=prob_hotter_bucket,
            unit_mult=unit_mult,
        )
        probs = lock_probs
        mu_final = float(lock_mu)
        sigma_final = float(lock_sigma)
        projected_high = float(lock_projected_high)

    inputs = {
        "nws_high": nws_high,
        "wu_hourly_peak": wu_hourly_peak,
        "hrrr_high": hrrr_high,
        "nbm_high": nbm_high,
        "ecmwf_ifs_high": ecmwf_ifs_high,
        "daily_high_metar": daily_high_metar,
        "current_temp_f": current_temp_f,
        "mu_forecast": float(mu_forecast),
        "mu_multi_model": float(mu_multi_model),
        "ensemble_breakdown": ensemble_breakdown,
        "kalman_nowcast_active": kalman_nowcast_active,
        "kalman_weight": round(kalman_w, 3),
        "kalman_divergence_f": round(kalman_divergence, 2) if kalman_divergence is not None else None,
        "projected_high": float(projected_high),
        "projected_high_capped": projected_high_capped,
        "metar_forecast_divergence_f": round(divergence_f, 2),
        "w_metar": float(w_metar),
        "w_metar_base": float(_metar_weight(hour_local)),
        "remaining_rise": remaining_rise,
        "hour_local": hour_local,
        "spread": float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0,
        "sigma_raw": float(sigma_raw),
        "sources_used": list(calibrated.keys()),
        "forecast_quality": forecast_quality,
        "prob_new_high": round(prob_hotter_bucket, 4),
        "prob_hotter_bucket": round(prob_hotter_bucket, 4),
        "prob_new_high_raw": round(prob_new_high_raw, 4),
        "lock_regime": lock_regime,
        "observed_high": observed_high,
        "observed_bucket_idx": observed_bucket_idx,
        "observed_bucket_upper_f": observed_bucket_upper_f,
        "next_hotter_bucket_floor_f": hotter_bucket_floor(canonical_buckets, observed_bucket_idx),
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
        prob_new_high=prob_hotter_bucket,
        prob_hotter_bucket=prob_hotter_bucket,
        prob_new_high_raw=prob_new_high_raw,
        lock_regime=lock_regime,
        observed_bucket_idx=observed_bucket_idx,
        observed_bucket_upper_f=observed_bucket_upper_f,
        inputs=inputs,
    )
