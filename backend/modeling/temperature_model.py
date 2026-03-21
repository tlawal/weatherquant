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

from backend.modeling.distribution import bucket_probabilities

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# ─── Time-of-day heuristics ───────────────────────────────────────────────────
# These represent expected *remaining* rise in temperature from current reading.
# Based on typical daily temperature curves for southeastern US cities.
_REMAINING_RISE_TABLE = [
    # (start_hour_et, end_hour_et, expected_remaining_rise_f)
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
    (20, 24, 0.80),
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
    inputs: dict = field(default_factory=dict)


def _interpolate_table(table: list, hour: int) -> float:
    for start, end, val in table:
        if start <= hour < end:
            return val
    return table[-1][2]


def _expected_remaining_rise(hour_et: int) -> float:
    return _interpolate_table(_REMAINING_RISE_TABLE, hour_et)


def _metar_weight(hour_et: int) -> float:
    return _interpolate_table(_METAR_WEIGHT_TABLE, hour_et)


def compute_model(
    nws_high: Optional[float],
    wu_daily_high: Optional[float],
    wu_hourly_peak: Optional[float],
    daily_high_metar: Optional[float],
    current_temp_f: Optional[float],
    calibration: Optional[dict],
    buckets: list[tuple[Optional[float], Optional[float]]],
    forecast_quality: str = "ok",
) -> Optional[ModelResult]:
    """
    Fuse all forecast sources and compute temperature distribution + bucket probabilities.

    Args:
        nws_high: NWS API daily high forecast (°F)
        wu_daily_high: WU daily high scrape (°F)
        wu_hourly_peak: max(WU hourly temps) (°F)
        daily_high_metar: max observed temp today (°F)
        current_temp_f: latest METAR temperature (°F)
        calibration: dict with bias_nws, bias_wu_daily, bias_wu_hourly,
                     weight_nws, weight_wu_daily, weight_wu_hourly
        buckets: list of (lo, hi) bucket boundaries
        forecast_quality: "ok" | "degraded"

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

    calibrated = {}
    if nws_high is not None:
        calibrated["nws"] = (nws_high + bias_nws, w_nws)
    if wu_daily_high is not None:
        calibrated["wu_daily"] = (wu_daily_high + bias_wud, w_wud)
    if wu_hourly_peak is not None:
        calibrated["wu_hourly"] = (wu_hourly_peak + bias_wuh, w_wuh)

    if not calibrated:
        log.warning("model: no forecast sources available — cannot compute model")
        return None

    # Weighted mean of available sources (re-normalize weights)
    total_weight = sum(w for _, w in calibrated.values())
    mu_forecast = sum(v * w for v, w in calibrated.values()) / total_weight

    # Uncertainty from disagreement
    vals = [v for v, _ in calibrated.values()]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        sigma_raw = max(1.0, spread / 2.0)
    else:
        # Only one source — use a conservative base uncertainty
        sigma_raw = 2.5

    # ── METAR intraday adjustment ──────────────────────────────────────────────
    now_et = datetime.now(ET)
    hour_et = now_et.hour

    w_metar = _metar_weight(hour_et)
    remaining_rise = _expected_remaining_rise(hour_et)

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

    # Weighted combination
    mu_final = (1.0 - w_metar) * mu_forecast + w_metar * projected_high

    # When heavily relying on METAR, widen sigma slightly to reflect
    # that a single ground observation is noisy
    sigma_final = sigma_raw * (1.0 + 0.2 * w_metar)
    sigma_final = max(1.0, sigma_final)

    # ── Compute bucket probabilities ───────────────────────────────────────────
    if not buckets:
        log.warning("model: no buckets to compute probabilities for")
        probs = []
    else:
        probs = bucket_probabilities(mu_final, sigma_final, buckets)

    inputs = {
        "nws_high": nws_high,
        "wu_daily_high": wu_daily_high,
        "wu_hourly_peak": wu_hourly_peak,
        "daily_high_metar": daily_high_metar,
        "current_temp_f": current_temp_f,
        "mu_forecast": round(mu_forecast, 2),
        "projected_high": round(projected_high, 2),
        "w_metar": round(w_metar, 3),
        "remaining_rise": remaining_rise,
        "hour_et": hour_et,
        "spread": round(max(vals) - min(vals), 2) if len(vals) >= 2 else 0,
        "sigma_raw": round(sigma_raw, 3),
        "sources_used": list(calibrated.keys()),
        "forecast_quality": forecast_quality,
    }

    return ModelResult(
        mu=round(mu_final, 2),
        sigma=round(sigma_final, 3),
        probs=probs,
        mu_forecast=round(mu_forecast, 2),
        mu_projected=round(projected_high, 2),
        w_metar=round(w_metar, 3),
        remaining_rise=remaining_rise,
        forecast_quality=forecast_quality,
        inputs=inputs,
    )
