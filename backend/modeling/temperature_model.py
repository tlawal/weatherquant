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
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from scipy.stats import norm as _norm

from backend.config import Config
from backend.modeling.bma import (
    bma_conditional_bucket_probabilities,
    bma_bucket_probabilities,
    build_bma_predictive,
    predictive_to_dict,
)
from backend.modeling.distribution import bucket_probabilities, conditional_bucket_probabilities
from backend.modeling.intraday_threshold import predict_intraday_threshold_probabilities
from backend.modeling.settlement import (
    bucket_upper_bound,
    canonical_bucket_ranges,
    find_bucket_idx_for_value,
    hotter_bucket_floor,
)
from backend.modeling.residual_tracker import predict_remaining_rise, is_ml_model_loaded

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# Hard divergence cap (°F): when Kalman disagrees with the NWP panel by more
# than this threshold, its weight drops to zero unconditionally.  This prevents
# the PnL-killing pattern where one filter pulls μ 6-8°F away from 5 physics
# models that agree (Atlanta, LA/Seattle regression April 2026).
KALMAN_HARD_DIVERGENCE_CAP_F = 6.0

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
    # ── M1 BMA shadow output ──────────────────────────────────────────────
    # Computed alongside the legacy single-Gaussian path; does NOT drive
    # trades during shadow phase. Trade decisions still use mu/sigma/probs.
    # After 14 days of side-by-side CRPS comparison the shadow path can be
    # promoted by swapping mu/sigma/probs for bma_mu/bma_sigma/bma_probs.
    bma_mu: Optional[float] = None
    bma_sigma: Optional[float] = None
    bma_probs: Optional[list[float]] = None
    bma_meta: Optional[dict] = None


def _interpolate_table(table: list, hour: int) -> float:
    for start, end, val in table:
        if start <= hour < end:
            return val
    return table[-1][2]


def _expected_remaining_rise(hour_local: int) -> float:
    return _interpolate_table(_REMAINING_RISE_TABLE, hour_local)


def _metar_weight(hour_local: int) -> float:
    return _interpolate_table(_METAR_WEIGHT_TABLE, hour_local)


# ─── Phase B1+B2: lead-time skill & freshness factors ───────────────────────
# Per-source freshness time constants (hours). exp(-age_h / TAU) decays the
# weight for stale model runs. A WU hourly forecast lacks an authoritative
# model_run_at, so its TAU is set high — caller passes 1.0 (no decay) instead.
_FRESHNESS_TAU_HOURS = {
    "hrrr": 6.0,
    "hrrr_15min": 6.0,
    "ncep_hrrr_conus_15min": 6.0,
    "nbm": 8.0,
    "gfs": 12.0,
    "ecmwf_ifs": 12.0,
    "ecmwf_aifs": 12.0,
    # Bayesian-upgrade Q3 — AI-NWP additions, same 4×/day cadence as GFS/AIFS.
    "gfs_graphcast": 12.0,
    # Bayesian-upgrade Q3 §13 — NOAA AIWP archive (00z + 12z daily, ~8h post-init
    # latency). Slightly longer TAU because we get fresh runs only twice a day.
    "pangu_weather": 14.0,
    "fourcastnet_v2": 14.0,
    # Microsoft Aurora — same NOAA AIWP cadence (00z + 12z) and ~8h IFS-init
    # latency, so reuse the 14h decay constant.
    "aurora": 14.0,
    "nws": 10.0,
    "wu_hourly": 12.0,
    "wu_history": 12.0,
}
_AI_FORECAST_SOURCES = frozenset({
    "ecmwf_aifs",
    "gfs_graphcast",
    "pangu_weather",
    "fourcastnet_v2",
    "aurora",
})
_STALE_AI_LIVE_EXCLUDE_SOURCES = frozenset({
    "pangu_weather",
    "fourcastnet_v2",
    "aurora",
})
_STALE_AI_LIVE_EXCLUDE_HOURS = 24.0
_UNPROVEN_AI_WEIGHT_CAP = 0.05
_TRUSTED_REFERENCE_SOURCES = frozenset({
    "nws",
    "wu_hourly",
    "hrrr",
    "hrrr_15min",
    "nbm",
    "ecmwf_ifs",
})
_GENERAL_OUTLIER_THRESHOLD_F = 10.0
_AI_OUTLIER_THRESHOLD_F = 6.0
_AI_OUTLIER_MIN_STATION_SAMPLES = 30
_AI_OUTLIER_MAX_STATION_MAE_F = 4.0
_HRRR_15MIN_PARENT_MAX_DELTA_F = 4.0
_SAME_DAY_TIGHT_SPREAD_CAP_F = 2.5
_SAME_DAY_TIGHT_SIGMA_CAP_F = 2.75
_SAME_DAY_STATION_SIGMA_MIN_SAMPLES = 5
_SAME_DAY_STATION_SIGMA_MIN_F = 1.5
_SAME_DAY_STATION_SIGMA_MAX_F = 2.0
_SAME_DAY_TIGHT_CAP_MAX_HOURS = 18.0
# Hard cap on lead-skill weight swing — protects against thin-data buckets.
_LEAD_SKILL_CLAMP = (0.7, 1.3)
# Floor on freshness factor so a stale source still contributes something.
_FRESHNESS_FLOOR = 0.5
# Minimum n_obs in a SourceLeadTimeSkill bucket before we trust its MAE.
_LEAD_SKILL_MIN_N_OBS = 30
_LEAD_SKILL_PRIOR_SIGMA_F = 3.0


def _lead_time_sigma_growth(
    model_run_at_by_source: Optional[dict[str, datetime]],
    event_settlement_utc: Optional[datetime],
    unit_mult: float,
) -> float:
    """Bayesian-upgrade Q4 — lead-time-conditional σ growth.

    NOAA's empirical MAE growth on next-day high-temperature forecasts is
    roughly σ_lead(L) ≈ 1.5 + 0.05·L °F over L hours from model issue to
    event time. Uses the median lead across active sources so a single
    very-stale source can't dominate the inflation.

    Returns an additive σ (in the model's display unit) to combine in
    quadrature with the ensemble-disagreement σ. Returns 0 when no metadata
    is available — keeps behavior identical to the pre-Q4 path in that case.
    """
    if not model_run_at_by_source or event_settlement_utc is None:
        return 0.0
    target = event_settlement_utc
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    leads_h: list[float] = []
    for _mr in model_run_at_by_source.values():
        if _mr is None:
            continue
        mr = _mr if _mr.tzinfo is not None else _mr.replace(tzinfo=timezone.utc)
        leads_h.append(max(0.0, (target - mr).total_seconds() / 3600.0))
    if not leads_h:
        return 0.0
    leads_h.sort()
    median_lead = leads_h[len(leads_h) // 2]
    sigma_f = 1.5 + 0.05 * median_lead
    if median_lead <= 72.0:
        sigma_f = min(sigma_f, 3.5)
    return sigma_f * unit_mult


def _lead_skill_sigma(
    mae_by_source: Optional[dict[str, float]],
    n_obs_by_source: Optional[dict[str, int]],
    weights_by_source: dict[str, float],
    unit_mult: float,
) -> Optional[float]:
    """Empirical-Bayes residual sigma from lead-time skill rows.

    This is the legacy single-Gaussian analogue of BMA's per-component sigma:
    use each source's lead-bucket MAE immediately, but shrink it toward a
    3°F prior until the bucket reaches the mature n=30 threshold.
    """
    mae_map = mae_by_source or {}
    n_map = n_obs_by_source or {}
    rows: list[tuple[float, float]] = []
    prior = _LEAD_SKILL_PRIOR_SIGMA_F * unit_mult
    for src, mae in mae_map.items():
        try:
            mae_f = float(mae)
        except (TypeError, ValueError):
            continue
        if mae_f <= 0:
            continue
        try:
            n = int(n_map.get(src, 0) or 0)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            continue
        weight = float(weights_by_source.get(src, 0.0) or 0.0)
        if weight <= 0:
            continue
        confidence = min(1.0, n / _LEAD_SKILL_MIN_N_OBS)
        shrunk_sigma = (1.0 - confidence) * prior + confidence * mae_f
        rows.append((shrunk_sigma, weight))
    if len(rows) < 2:
        return None
    total_w = sum(w for _, w in rows)
    if total_w <= 0:
        return None
    return max(1.0 * unit_mult, sum(sigma * w for sigma, w in rows) / total_w)


def _spread_or_none(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    return max(vals) - min(vals)


def _station_source_has_good_evidence(
    src: str,
    station_source_meta: Optional[dict[str, dict]],
) -> bool:
    meta = (station_source_meta or {}).get(src) or {}
    try:
        n_samples = int(meta.get("n_samples") or 0)
    except (TypeError, ValueError):
        n_samples = 0
    mae = meta.get("mae_7d")
    if mae is None:
        mae = meta.get("mae_30d")
    try:
        mae_f = float(mae)
    except (TypeError, ValueError):
        mae_f = None
    return (
        n_samples >= _AI_OUTLIER_MIN_STATION_SAMPLES
        and mae_f is not None
        and mae_f <= _AI_OUTLIER_MAX_STATION_MAE_F
    )


def apply_forecast_source_quality_gates(
    source_values: dict[str, float],
    *,
    station_source_meta: Optional[dict[str, dict]] = None,
    unit_mult: float = 1.0,
) -> dict:
    """Quarantine physically implausible source members before live fusion.

    The trusted reference is the median of operational/non-experimental sources.
    Experimental AI members need local station evidence before they are allowed
    to deviate as much as the operational panel.
    """
    clean_values = {
        src: float(val)
        for src, val in source_values.items()
        if val is not None
    }
    raw_spread = _spread_or_none(clean_values.values())
    trusted_refs = [
        val for src, val in clean_values.items()
        if src in _TRUSTED_REFERENCE_SOURCES
    ]
    if len(trusted_refs) < 3:
        return {
            "live_values": clean_values,
            "raw_spread": raw_spread,
            "trusted_spread": raw_spread,
            "trusted_reference_median": None,
            "source_quality_gates": {},
        }

    ref = float(statistics.median(trusted_refs))
    gates: dict[str, dict] = {}
    if "hrrr_15min" in clean_values and "hrrr" in clean_values:
        parent = clean_values["hrrr"]
        val = clean_values["hrrr_15min"]
        threshold = _HRRR_15MIN_PARENT_MAX_DELTA_F * unit_mult
        delta = abs(val - parent)
        if delta > threshold:
            gates["hrrr_15min"] = {
                "reason": "companion_divergence_vs_hrrr",
                "value": round(val, 2),
                "reference_source": "hrrr",
                "reference": round(parent, 2),
                "delta": round(delta, 2),
                "threshold": round(threshold, 2),
            }
    live_values: dict[str, float] = {}
    for src, val in clean_values.items():
        if src in gates:
            continue
        threshold = _GENERAL_OUTLIER_THRESHOLD_F * unit_mult
        if (
            src in _AI_FORECAST_SOURCES
            and not _station_source_has_good_evidence(src, station_source_meta)
        ):
            threshold = _AI_OUTLIER_THRESHOLD_F * unit_mult
        delta = abs(val - ref)
        if delta > threshold:
            gates[src] = {
                "reason": "outlier_vs_trusted_median",
                "value": round(val, 2),
                "reference": round(ref, 2),
                "delta": round(delta, 2),
                "threshold": round(threshold, 2),
            }
            continue
        live_values[src] = val

    return {
        "live_values": live_values,
        "raw_spread": raw_spread,
        "trusted_spread": _spread_or_none(live_values.values()),
        "trusted_reference_median": ref,
        "source_quality_gates": gates,
    }


def _ensemble_sigma(
    values_and_weights: list[tuple[float, float]],
    fallback_sigma: float,
    sigma_floor: float,
) -> float:
    """Phase B5 — weight-aware σ from ensemble disagreement.

    For ≥3 forecasts with positive weight, σ = sqrt(weighted variance) about
    the weighted mean. This generalizes the spread/2 heuristic and respects
    per-source weights (so a 5°F outlier from a low-weight source widens σ
    less than the same outlier from a high-weight source).
    """
    rows = [(v, w) for v, w in values_and_weights if w > 0]
    if len(rows) < 3:
        return fallback_sigma
    total_w = sum(w for _, w in rows)
    if total_w <= 0:
        return fallback_sigma
    mu = sum(v * w for v, w in rows) / total_w
    var = sum(w * (v - mu) ** 2 for v, w in rows) / total_w
    return max(sigma_floor, math.sqrt(max(var, 0.0)))


def _freshness_factor(source: str, age_hours: float) -> float:
    """exp decay capped at floor; older runs contribute less to the ensemble."""
    if age_hours is None or age_hours < 0:
        return 1.0
    tau = _FRESHNESS_TAU_HOURS.get(source, 10.0)
    return max(_FRESHNESS_FLOOR, math.exp(-age_hours / tau))


def _lead_skill_factors(
    mae_by_source: dict[str, Optional[float]],
    n_obs_by_source: Optional[dict[str, int]] = None,
) -> dict[str, float]:
    """Per-source weight multiplier from lead-time MAE skill (clamped ±30%).

    For each source with evidence, multiplier = clamp(median_mae / mae, 0.7,
    1.3). Thin but nonzero data gets an empirical-Bayes partial adjustment:
    factor = 1 + confidence * (raw_factor - 1), confidence = n / threshold.
    Missing/zero-n data gets 1.0 and is excluded from the median.
    Need at least 2 sources with valid evidence to compute a meaningful median.
    """
    n_obs = n_obs_by_source or {}
    valid = {
        s: m for s, m in mae_by_source.items()
        if m is not None and m > 0 and n_obs.get(s, 0) > 0
    }
    if len(valid) < 2:
        return {s: 1.0 for s in mae_by_source}
    median_mae = statistics.median(valid.values())
    out: dict[str, float] = {}
    for s in mae_by_source:
        m = mae_by_source.get(s)
        if s in valid and median_mae > 0:
            ratio = median_mae / m
            raw_factor = max(_LEAD_SKILL_CLAMP[0], min(_LEAD_SKILL_CLAMP[1], ratio))
            confidence = min(1.0, max(0, int(n_obs.get(s, 0))) / _LEAD_SKILL_MIN_N_OBS)
            out[s] = 1.0 + confidence * (raw_factor - 1.0)
        else:
            out[s] = 1.0
    return out


def _blend_kalman_sigma(
    sigma_final: float,
    w_metar: float,
    kalman_uncertainty: float,
    peak_hour_local: Optional[float],
    hour_local_fractional: float,
    unit_mult: float = 1.0,
) -> float:
    """Mix the Kalman posterior σ into σ_final at the METAR weight.

    Kalman σ is the filter's posterior std (sqrt(P[0,0])) in the city's unit.
    Add a drift term proportional to hours-to-peak — early in the day there's
    more time for the trajectory to deviate from the smoothed estimate. Both
    components combine in quadrature; the result blends in at w_metar weight
    because Kalman is METAR-derived evidence.
    """
    if peak_hour_local is not None:
        hours_to_peak = max(0.5, peak_hour_local - hour_local_fractional)
    else:
        hours_to_peak = 2.0
    drift_sigma = 0.3 * hours_to_peak * unit_mult  # ~0.3°F/hr of trajectory drift
    kalman_sigma = math.sqrt(kalman_uncertainty ** 2 + drift_sigma ** 2)
    return math.sqrt(
        (1.0 - w_metar) * sigma_final ** 2 + w_metar * kalman_sigma ** 2
    )


def compute_kalman_weight(
    hour_local_fractional: float,
    peak_hour_local: Optional[float],
    kalman_divergence: float,
    spread: float,
    n_obs: int,
    peak_already_passed: bool,
    max_weight: float = 0.30,
    half_window_hours: float = 2.0,
) -> float:
    """Ensemble weight for the Kalman nowcast slice of mu_forecast.

    The filter has short-horizon skill; it earns its keep in a ±2h tent
    centered on the anticipated peak. Outside that window the multi-model
    panel (ECMWF IFS, HRRR, NBM, NWS, WU) is strictly more skillful. When
    the Kalman nowcast disagrees with consensus by more than the panel's
    own spread, scale down sharply — a lone filter beating five physics
    models is almost always the filter missing a regime change.

    Hard cap: divergence > 6°F → weight = 0 unconditionally.
    This prevents the Atlanta regression (NWP 88-90°F vs Kalman 81°F
    → blended ~85.7°F) from creating false edges on wrong buckets.

    All temperature arguments are in the same unit (°F or °C). The helper
    only uses ratios, so it is unit-agnostic.

    Returns 0.0 when the filter is out-of-window, under-observed, or badly
    diverged. Never exceeds `max_weight`.
    """
    if n_obs < 10 or peak_hour_local is None:
        return 0.0
    # Hard divergence cap: 6°F+ disagreement = filter is wrong, not the NWP panel
    if kalman_divergence > KALMAN_HARD_DIVERGENCE_CAP_F:
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
    trusted_consensus_high: Optional[float] = None,
) -> bool:
    """Decide whether to engage the late-day lock regime.

    The lock concentrates probability on the bucket containing the observed
    high once we are confident the day's peak is in. We do not require the
    adaptive engine to have flagged peak_already_passed — that detector can
    lag by an hour or false-trigger during intraday dips. Instead we require
    late-day timing plus observation-grounded evidence that the observed high
    is plausibly final.
    """
    if observed_high is None or current_temp_f is None:
        return False
    if hour_local < 15:
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
    n_obs = adaptive.kalman.n_observations if adaptive is not None and adaptive.kalman is not None else 0
    deficit = observed_high - current_temp_f

    # Path 1 — observation fallback: physically post-peak even if adaptive lags.
    # All four floor checks above already passed, so we know:
    #   - observed_high and current_temp_f exist
    #   - remaining_rise <= 0.25°F
    #   - current_temp is at least 0.5°F below observed_high
    # The remaining condition is "we're past the typical late-day window AND
    # the trend isn't actively rising back up". Trend may be unknown (no
    # adaptive), in which case the time-of-day check alone is sufficient.
    if hour_local >= 18 and (trend_per_hr is None or trend_per_hr <= 0.25):
        return True

    # Path 2 — strong-cooling override for west-coast / late-peak cities.
    # Seattle regression: at 17:47 local with observed 55°F, current 53.6°F
    # (deficit 1.4°F) and a clearly negative Kalman trend, `peak_already_passed`
    # hadn't flipped and path 2's 18:00 gate was too late. When the trend is
    # firmly negative with reliable observations, the observed high is near
    # trusted consensus, and we are at/past 15:00 local, the physics is
    # unambiguous enough to lock before the 18:00 fallback.
    observed_near_consensus = True
    if trusted_consensus_high is not None:
        observed_near_consensus = observed_high >= trusted_consensus_high - 2.0 * unit_mult
    if (
        15 <= hour_local < 18
        and trend_per_hr is not None
        and trend_per_hr <= -0.25
        and n_obs >= 10
        and deficit >= 1.0 * unit_mult
        and observed_near_consensus
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


def _intraday_threshold_live_alpha(
    *,
    observed_high: Optional[float],
    current_temp_f: Optional[float],
    hour_local: float,
    trusted_spread: float,
    forecast_quality: str,
    lock_regime: bool,
    regime_label: Optional[str],
    unit_mult: float,
) -> float:
    """Earned live blend weight for same-day threshold probabilities."""
    if not Config.INTRADAY_THRESHOLD_LIVE_BLEND_ENABLED:
        return 0.0
    if observed_high is None or forecast_quality != "ok":
        return 0.0
    if lock_regime:
        return 0.0

    alpha = 0.0
    if hour_local >= 10.0:
        alpha = 0.55
    if hour_local >= 11.5:
        alpha = 0.95
    if hour_local >= 13.0:
        alpha = 1.0
    try:
        if current_temp_f is not None and current_temp_f >= observed_high - (0.1 * unit_mult):
            alpha = max(alpha, 0.75)
    except TypeError:
        pass

    if str(regime_label or "").lower() == "volatile":
        alpha *= 0.70
    if trusted_spread > 4.0 * unit_mult:
        alpha *= 0.75

    cap = max(0.0, min(1.0, Config.INTRADAY_THRESHOLD_ALPHA_MAX))
    return round(max(0.0, min(cap, alpha)), 4)


def _blend_probability_vectors(
    base_probs: list[float],
    overlay_probs: list[float],
    alpha: float,
) -> list[float]:
    if not base_probs or not overlay_probs or len(base_probs) != len(overlay_probs) or alpha <= 0:
        return base_probs
    a = max(0.0, min(1.0, float(alpha)))
    blended = [
        max(0.0, (1.0 - a) * float(base) + a * float(overlay))
        for base, overlay in zip(base_probs, overlay_probs)
    ]
    total = sum(blended)
    return [p / total for p in blended] if total > 0 else blended


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
    hrrr_15min_high: Optional[float] = None,
    nbm_high: Optional[float] = None,
    ecmwf_ifs_high: Optional[float] = None,
    ecmwf_aifs_high: Optional[float] = None,
    # Bayesian-upgrade Q3: AI-NWP foundation models (experimental).
    gfs_graphcast_high: Optional[float] = None,
    # §13 — Pangu-Weather + FourCastNetv2-small from NOAA AIWP S3 archive.
    # Same shape (just another forecast high in city.unit). Sourced via
    # backend/ingestion/aiwp.py.
    pangu_weather_high: Optional[float] = None,
    fourcastnet_v2_high: Optional[float] = None,
    # Microsoft Aurora (Bodnar et al. 2024) via NOAA AIWP S3 archive.
    aurora_high: Optional[float] = None,
    model_run_at_by_source: Optional[dict[str, datetime]] = None,
    lead_skill_mae_by_source: Optional[dict[str, Optional[float]]] = None,
    lead_skill_n_obs_by_source: Optional[dict[str, int]] = None,
    now_utc: Optional[datetime] = None,
    # Bayesian-upgrade Q4: end-of-day settlement time used to compute lead
    # for σ growth. None = behave exactly as pre-Q4 (no inflation).
    event_settlement_utc: Optional[datetime] = None,
    # Bayesian-upgrade Q6: regime-aware σ inflation multiplier (1.0 = calm,
    # up to 2.0 = volatile). None = behave exactly as pre-Q6 (no inflation).
    regime_sigma_multiplier: Optional[float] = None,
    regime_label: Optional[str] = None,
    # M1 Phase 2: EM-fit BMA mixing weights from BMAWeights (per (city, lead)).
    # Caller supplies {source: weight} when a fit exists for the operative
    # lead bucket; build_bma_predictive uses these instead of the legacy
    # weights_by_source for matching sources, falling back to legacy for any
    # source not present in the fitted set. None = pure shadow on legacy
    # weights (Phase 1 behavior).
    fitted_bma_weights_by_source: Optional[dict[str, float]] = None,
    # Same-day threshold survival calibration. Caller supplies a materialized
    # remapper when enough city/station/hour/floor history exists.
    threshold_survival_calibrator: Optional[
        Callable[[dict[float, float]], tuple[dict[float, float], dict]]
    ] = None,
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
    # Per-station dynamic weights/biases (from ForecastDailyError EWMA) override
    # city-level calibration for any source that has enough samples.
    station_w: dict = cal.get("station_source_weights") or {}
    station_b: dict = cal.get("station_source_biases") or {}
    station_source_meta: dict = cal.get("station_source_meta") or {}

    def _debias(src: str, raw: float, legacy_default: float = 0.0) -> float:
        # Station bias is EWMA(forecast - observed), so subtract it.
        if src in station_b:
            return raw - float(station_b[src])
        # Legacy city-level bias is the *correction* (add it to the forecast).
        return raw + float(cal.get(f"bias_{src}", legacy_default))

    def _weight(src: str, default: float) -> float:
        if src in station_w:
            return float(station_w[src])
        return float(cal.get(f"weight_{src}", default))

    local_tz = ZoneInfo(city_tz)
    now_local = datetime.now(local_tz)
    hour_local = now_local.hour
    # Scale factor for Celsius
    unit_mult = 5.0 / 9.0 if unit == "C" else 1.0

    calibrated = {}
    if nws_high is not None:
        calibrated["nws"] = (_debias("nws", nws_high), _weight("nws", 0.5))
    if wu_hourly_peak is not None:
        calibrated["wu_hourly"] = (
            _debias("wu_hourly", wu_hourly_peak),
            _weight("wu_hourly", 0.5),
        )
    if hrrr_high is not None:
        calibrated["hrrr"] = (_debias("hrrr", hrrr_high), _weight("hrrr", 0.5))
    if hrrr_15min_high is not None:
        calibrated["hrrr_15min"] = (
            _debias("hrrr_15min", hrrr_15min_high),
            _weight("hrrr_15min", 0.35),
        )
    if nbm_high is not None:
        calibrated["nbm"] = (_debias("nbm", nbm_high), _weight("nbm", 0.2))
    if ecmwf_ifs_high is not None:
        calibrated["ecmwf_ifs"] = (
            _debias("ecmwf_ifs", ecmwf_ifs_high),
            _weight("ecmwf_ifs", 0.5),
        )
    if ecmwf_aifs_high is not None:
        calibrated["ecmwf_aifs"] = (
            _debias("ecmwf_aifs", ecmwf_aifs_high),
            _weight("ecmwf_aifs", 0.4),
        )
    # Bayesian-upgrade Q3: AI-NWP additions. Same base weight as AIFS while we
    # accumulate per-source skill — lead-skill factors will adjust automatically.
    if gfs_graphcast_high is not None:
        calibrated["gfs_graphcast"] = (
            _debias("gfs_graphcast", gfs_graphcast_high),
            _weight("gfs_graphcast", 0.4),
        )
    # §13 — NOAA AIWP-sourced AI members. Slightly lower base weight (0.35) to
    # reflect the 6-hour timestep limitation (true daily peak may sit between
    # forecast steps); lead-skill factors will refine this once residuals land.
    if pangu_weather_high is not None:
        calibrated["pangu_weather"] = (
            _debias("pangu_weather", pangu_weather_high),
            _weight("pangu_weather", 0.35),
        )
    if fourcastnet_v2_high is not None:
        calibrated["fourcastnet_v2"] = (
            _debias("fourcastnet_v2", fourcastnet_v2_high),
            _weight("fourcastnet_v2", 0.35),
        )
    # Microsoft Aurora (Bodnar et al. 2024) via NOAA AIWP. Same base weight as
    # the other AI-NWP members; lead-skill + freshness factors will fold in
    # once SourceLeadTimeSkill rows accumulate.
    if aurora_high is not None:
        calibrated["aurora"] = (
            _debias("aurora", aurora_high),
            _weight("aurora", 0.35),
        )

    if not calibrated:
        log.warning("model: no forecast sources available — cannot compute model")
        return None

    excluded_sources: dict[str, str] = {}
    raw_calibrated = dict(calibrated)
    source_quality = apply_forecast_source_quality_gates(
        {src: val for src, (val, _) in calibrated.items()},
        station_source_meta=station_source_meta,
        unit_mult=unit_mult,
    )
    source_quality_gates = source_quality["source_quality_gates"]
    if source_quality_gates:
        for src, gate in source_quality_gates.items():
            excluded_sources[src] = gate["reason"]
        calibrated = {
            src: calibrated[src]
            for src in source_quality["live_values"]
            if src in calibrated
        }
        if not calibrated:
            log.warning("model: all forecast sources excluded by source quality gates")
            return None

    # ── Phase B1+B2: lead-skill + freshness adjustments ───────────────────
    # Multiply each source's base weight by its lead-time skill factor (clamped
    # ±30%) and by an exp-decay freshness factor on model_run_at age. Sources
    # with no metadata get factor 1.0 (no adjustment).
    lead_factors = _lead_skill_factors(
        lead_skill_mae_by_source or {},
        lead_skill_n_obs_by_source or {},
    )
    now_ts = now_utc or datetime.now(timezone.utc)
    mra = model_run_at_by_source or {}
    weight_factors: dict[str, dict] = {}
    bma_calibrated: dict[str, tuple[float, float]] = {}
    for src in list(calibrated.keys()):
        mu_val, base_w = calibrated[src]
        lf = lead_factors.get(src, 1.0)
        mr = mra.get(src)
        if mr is not None:
            if mr.tzinfo is None:
                mr = mr.replace(tzinfo=timezone.utc)
            age_h = max(0.0, (now_ts - mr).total_seconds() / 3600.0)
            ff = _freshness_factor(src, age_h)
        else:
            age_h = None
            ff = 1.0
        diagnostic_effective = base_w * lf * ff
        bma_calibrated[src] = (mu_val, diagnostic_effective)
        effective = diagnostic_effective
        live_gate: Optional[str] = None

        if (
            src in _STALE_AI_LIVE_EXCLUDE_SOURCES
            and age_h is not None
            and age_h > _STALE_AI_LIVE_EXCLUDE_HOURS
        ):
            live_gate = "stale_ai_excluded"
            excluded_sources[src] = f"age_hours>{_STALE_AI_LIVE_EXCLUDE_HOURS:g}"
            del calibrated[src]
            effective = 0.0
        else:
            if src in _AI_FORECAST_SOURCES and src not in station_w:
                capped = min(effective, _UNPROVEN_AI_WEIGHT_CAP)
                if capped < effective:
                    live_gate = "unproven_ai_weight_cap"
                effective = capped
            calibrated[src] = (mu_val, effective)
        weight_factors[src] = {
            "base": round(base_w, 4),
            "lead_factor": round(lf, 3),
            "freshness_factor": round(ff, 3),
            "age_hours": round(age_h, 2) if age_h is not None else None,
            "diagnostic_effective": round(diagnostic_effective, 4),
            "effective": round(effective, 4),
            "live_gate": live_gate,
        }

    for src, gate in source_quality_gates.items():
        if src in weight_factors or src not in raw_calibrated:
            continue
        _raw_val, base_w = raw_calibrated[src]
        weight_factors[src] = {
            "base": round(base_w, 4),
            "lead_factor": 1.0,
            "freshness_factor": 1.0,
            "age_hours": None,
            "diagnostic_effective": round(base_w, 4),
            "effective": 0.0,
            "live_gate": gate["reason"],
            "source_quality_gate": gate,
        }

    if not calibrated:
        log.warning("model: all forecast sources excluded by live gates")
        return None

    # Multi-model weighted mean of available sources (re-normalize weights).
    total_weight = sum(w for _, w in calibrated.values())
    if total_weight <= 0.0:
        log.warning("model: non-positive live source weight after calibration gates")
        return None
    mu_multi_model = sum(v * w for v, w in calibrated.values()) / total_weight

    # ── M1 BMA shadow predictive (Phase 1) ───────────────────────────────────
    # Build a Gaussian-mixture diagnostic over all calibrated per-source means,
    # using SourceLeadTimeSkill MAE as σᵢ when available. BMA intentionally sees
    # stale/thin AI members for shadow diagnostics; live legacy probabilities
    # above can exclude or cap them until local evidence is credible.
    # SHADOW MODE: computed for comparison only — `mu_final`, `sigma_final`,
    # `probs` below remain driven by the legacy single-Gaussian path. Once we
    # have ≥14 days of side-by-side CRPS we'll promote the BMA outputs.
    try:
        bma_predictive = build_bma_predictive(
            calibrated_means={src: mu_val for src, (mu_val, _) in bma_calibrated.items()},
            weights_by_source={src: w for src, (_, w) in bma_calibrated.items()},
            lead_skill_mae_by_source=lead_skill_mae_by_source or {},
            lead_skill_n_obs_by_source=lead_skill_n_obs_by_source or {},
            sigma_unit_mult=(5.0 / 9.0 if unit == "C" else 1.0),
            fitted_weights_by_source=fitted_bma_weights_by_source,
        )
    except Exception:
        log.exception("model: BMA shadow predictive failed; legacy path unaffected")
        bma_predictive = None

    vals = [v for v, _ in calibrated.values()]
    spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
    raw_spread = source_quality.get("raw_spread")
    trusted_spread = spread if len(vals) >= 2 else 0.0

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
        "weight_factors": weight_factors,
    }

    canonical_buckets = canonical_bucket_ranges(buckets)

    # Uncertainty from disagreement (Phase B5).
    # Prefer weighted-variance σ across ≥3 sources when available — that's
    # the proper formalization of what spread/2 approximates and it respects
    # the same per-source weights used in the mean. Fall back to spread/2 (or
    # the no-evidence default) when we don't have enough sources.
    sigma_floor = 1.0 * unit_mult
    fallback = max(sigma_floor, spread / 2.0) if len(vals) >= 2 else 2.5 * unit_mult
    sigma_raw = _ensemble_sigma(
        list(calibrated.values()), fallback_sigma=fallback, sigma_floor=sigma_floor,
    )

    # ── Q4: lead-time-conditional σ growth ───────────────────────────────────
    # Adds NOAA-empirical 1.5 + 0.05·L °F (median lead across sources) in
    # quadrature to the ensemble disagreement. This fixes the prior
    # under-confidence at L > 24h: a 48h forecast carries ~3.9°F of lead
    # noise that the inter-source spread alone systematically misses.
    live_model_run_at_by_source = {
        src: ts for src, ts in (model_run_at_by_source or {}).items()
        if src in calibrated
    }
    live_weights_by_source = {
        src: weight for src, (_, weight) in calibrated.items()
    }
    sigma_lead_skill = _lead_skill_sigma(
        lead_skill_mae_by_source,
        lead_skill_n_obs_by_source,
        live_weights_by_source,
        unit_mult,
    )
    sigma_lead_generic = _lead_time_sigma_growth(
        live_model_run_at_by_source, event_settlement_utc, unit_mult,
    )
    sigma_lead_source = "lead_skill_shrinkage" if sigma_lead_skill is not None else "generic_lead_growth"
    sigma_lead = sigma_lead_skill if sigma_lead_skill is not None else sigma_lead_generic
    if sigma_lead > 0.0:
        sigma_raw = math.sqrt(sigma_raw * sigma_raw + sigma_lead * sigma_lead)

    # ── Q6: regime-aware σ inflation ─────────────────────────────────────────
    # Caller supplies a multiplier in [1.0, 1.5] from regime_sigma_inflation().
    # Applied multiplicatively (not in quadrature) because regime here
    # represents *systematic* widening of forecast residual variance, not an
    # additional independent noise source. Bounded for safety in case the
    # caller passes an outlier value.
    if regime_sigma_multiplier is not None:
        _mult = max(1.0, min(1.5, float(regime_sigma_multiplier)))
        sigma_raw = sigma_raw * _mult

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
    w_metar_base = _metar_weight(hour_local)
    w_metar = w_metar_base
    w_metar_gate: Optional[str] = None
    has_intraday_observation = (
        daily_high_metar is not None
        or observed_high is not None
        or current_temp_f is not None
    )
    if not has_intraday_observation:
        # Next-day/future market pages can render late at night with no station
        # observation for the event date. Do not let clock time alone tighten
        # sigma toward a same-day METAR projection that does not exist.
        w_metar = 0.0
        w_metar_gate = "no_intraday_observation"

    # Use ML-based remaining rise prediction if features are available
    _ml = ml_features or {}
    _latest_wx_for_ml = latest_weather or {}
    remaining_rise = predict_remaining_rise(
        hour_local=hour_local,
        current_temp_f=current_temp_f or 70.0,
        temp_slope_3h=_ml.get("temp_slope_3h", 0.0),
        avg_peak_timing_mins=_ml.get("avg_peak_timing_mins", 960.0),
        day_of_year=_ml.get("day_of_year", now_local.timetuple().tm_yday),
        humidity_pct=_latest_wx_for_ml.get("humidity_pct", 50.0),
        cloud_cover_val=_latest_wx_for_ml.get("cloud_cover_val", 0.0),
        wind_speed_kt=_latest_wx_for_ml.get("wind_speed_kt", 0.0),
        wind_gust_kt=_latest_wx_for_ml.get("wind_gust_kt", 0.0),
        dewpoint_spread_f=_latest_wx_for_ml.get("dewpoint_spread_f", 10.0),
        pressure_tendency_3h=_latest_wx_for_ml.get("pressure_tendency", 0.0),
        precip_flag=1.0 if _latest_wx_for_ml.get("has_precip") else 0.0,
        precip_recent_3h=1.0 if _latest_wx_for_ml.get("has_precip") else 0.0,
        regime_score_proxy=_ml.get("regime_score", 0.0),
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
        # Hard divergence cap: same 6°F rule as compute_kalman_weight.
        # If the adaptive high diverges > 6°F from mu_multi_model, skip
        # the residual blend entirely — the filter is misleading.
        adaptive_div = abs(adaptive_high - mu_multi_model) if adaptive_high is not None else 0.0
        if kalman_nowcast_active and adaptive_high is not None and adaptive_div <= KALMAN_HARD_DIVERGENCE_CAP_F:
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
            divergence_f, w_metar_base, w_metar,
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

    metar_projection_gate: Optional[str] = w_metar_gate
    projected_high_for_blend = projected_high
    projected_high_raw = projected_high
    fallback_peak_hour_local = None
    try:
        fallback_peak_hour_local = float(_ml.get("avg_peak_timing_mins", 960.0)) / 60.0
    except (TypeError, ValueError):
        fallback_peak_hour_local = None
    gate_peak_hour_local = (
        peak_hour_local if peak_hour_local is not None else fallback_peak_hour_local
    )
    before_peak = (
        gate_peak_hour_local is not None
        and hour_local_fractional < gate_peak_hour_local
    )
    observed_near_consensus = (
        daily_high_metar is not None
        and daily_high_metar >= mu_forecast - 2.0 * unit_mult
    )
    physically_locked = (
        (hour_local >= 17 and daily_high_metar is not None)
        or (
            adaptive is not None
            and adaptive.peak_already_passed
            and observed_near_consensus
        )
    )
    if (
        not kalman_nowcast_active
        and projected_high < mu_forecast - 3.0 * unit_mult
        and (before_peak or not physically_locked)
    ):
        metar_projection_gate = "below_consensus"
        projected_high_for_blend = mu_forecast

    # Weighted combination
    mu_final = (1.0 - w_metar) * mu_forecast + w_metar * projected_high_for_blend

    # As METAR observations accumulate through the day (w_metar rises),
    # forecast spread becomes less relevant because ground truth dominates.
    # Blend from full forecast sigma toward a tight observation-based sigma.
    observation_sigma = max(1.0, remaining_rise + 0.5) * unit_mult
    # Inflate observation sigma when METAR and forecasts diverge —
    # disagreement means more uncertainty, not less.
    divergence_for_blend = abs(projected_high_for_blend - mu_forecast)
    divergence_for_blend_f = (
        divergence_for_blend / unit_mult if unit_mult else divergence_for_blend
    )
    if divergence_for_blend_f > 3.0:
        observation_sigma += divergence_for_blend * 0.3
    sigma_final = (1.0 - w_metar) * sigma_raw + w_metar * observation_sigma

    # Apply adaptive sigma adjustment (tightens when trend data is rich)
    if adaptive is not None:
        sigma_final *= adaptive.sigma_adjustment

    # ── Kalman posterior uncertainty blend (Phase A4) ─────────────────────────
    # Once the Kalman filter has enough observations (n_obs >= 10), its
    # posterior std deviation P[0,0]^0.5 is meaningful. Blend it in via
    # _blend_kalman_sigma — kept as a separate helper so the math is testable.
    if (
        adaptive is not None
        and adaptive.kalman is not None
        and adaptive.kalman.n_observations >= 10
    ):
        sigma_final = _blend_kalman_sigma(
            sigma_final=sigma_final,
            w_metar=w_metar,
            kalman_uncertainty=adaptive.kalman.uncertainty,
            peak_hour_local=peak_hour_local,
            hour_local_fractional=hour_local_fractional,
            unit_mult=unit_mult,
        )

    sigma_final = max(1.0 * unit_mult, sigma_final)

    time_to_settlement_h: Optional[float] = None
    if event_settlement_utc is not None:
        _settle = event_settlement_utc
        if _settle.tzinfo is None:
            _settle = _settle.replace(tzinfo=timezone.utc)
        _now_for_settlement = now_ts
        if _now_for_settlement.tzinfo is None:
            _now_for_settlement = _now_for_settlement.replace(tzinfo=timezone.utc)
        time_to_settlement_h = max(
            0.0,
            (
                _settle.astimezone(timezone.utc)
                - _now_for_settlement.astimezone(timezone.utc)
            ).total_seconds() / 3600.0,
        )
    same_day_sigma_cap_applied = False
    same_day_sigma_cap_source = None
    same_day_sigma_cap_value = _SAME_DAY_TIGHT_SIGMA_CAP_F * unit_mult
    regime_is_volatile = str(regime_label or "").lower() == "volatile"
    if (
        time_to_settlement_h is not None
        and time_to_settlement_h <= _SAME_DAY_TIGHT_CAP_MAX_HOURS
        and trusted_spread <= _SAME_DAY_TIGHT_SPREAD_CAP_F * unit_mult
        and not regime_is_volatile
    ):
        sigma_cap = _SAME_DAY_TIGHT_SIGMA_CAP_F * unit_mult
        same_day_sigma_cap_source = "fixed_same_day_tight_consensus"
        station_sigma_cap = None
        if calibration:
            try:
                station_n_samples = int(calibration.get("station_n_samples") or 0)
            except (TypeError, ValueError):
                station_n_samples = 0
            station_error = calibration.get("station_rmse_f")
            if station_error is None:
                station_error = calibration.get("station_mae_f")
            try:
                station_error_f = float(station_error)
            except (TypeError, ValueError):
                station_error_f = None
            if (
                station_n_samples >= _SAME_DAY_STATION_SIGMA_MIN_SAMPLES
                and station_error_f is not None
                and station_error_f > 0
            ):
                station_sigma_cap = min(
                    _SAME_DAY_STATION_SIGMA_MAX_F,
                    max(_SAME_DAY_STATION_SIGMA_MIN_F, 1.35 * station_error_f),
                ) * unit_mult
        if station_sigma_cap is not None:
            sigma_cap = min(sigma_cap, station_sigma_cap)
            same_day_sigma_cap_source = "station_calibration"
        same_day_sigma_cap_value = sigma_cap
        if sigma_final > sigma_cap:
            sigma_final = sigma_cap
            same_day_sigma_cap_applied = True

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

    trusted_consensus_high = source_quality.get("trusted_reference_median")
    if trusted_consensus_high is None:
        trusted_consensus_high = mu_forecast
    lock_suppressed_reason: Optional[str] = None
    if (
        hour_local < 15
        and adaptive is not None
        and getattr(adaptive, "peak_already_passed", False)
        and observed_high is not None
        and current_temp_f is not None
        and remaining_rise <= 0.25 * unit_mult
        and current_temp_f <= observed_high - 0.5 * unit_mult
    ):
        lock_suppressed_reason = "pre_15_adaptive_peak_passed"
        log.warning(
            "model: suppressing impossible pre-15 lock hour=%s observed=%.1f current=%.1f consensus=%.1f",
            hour_local,
            observed_high,
            current_temp_f,
            trusted_consensus_high,
        )

    lock_regime = _late_day_lock_active(
        observed_high=observed_high,
        current_temp_f=current_temp_f,
        remaining_rise=remaining_rise,
        adaptive=adaptive,
        hour_local=hour_local,
        unit_mult=unit_mult,
        trusted_consensus_high=trusted_consensus_high,
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
        projected_high_for_blend = projected_high
        projected_high_raw = projected_high

    intraday_threshold_out = None
    intraday_threshold_live_alpha = 0.0
    legacy_probs_before_intraday_blend: Optional[list[float]] = None
    if canonical_buckets and observed_high is not None:
        try:
            trend_per_hr = None
            if adaptive is not None and adaptive.kalman is not None:
                trend_per_hr = adaptive.kalman.temp_trend_per_min * 60.0
            intraday_threshold_out = predict_intraday_threshold_probabilities(
                buckets=canonical_buckets,
                observed_high=observed_high,
                current_temp_f=current_temp_f,
                projected_high=float(projected_high_for_blend),
                consensus_high=float(mu_forecast),
                sigma=float(sigma_final),
                remaining_rise=float(remaining_rise),
                hour_local=float(hour_local_fractional),
                peak_hour_local=peak_hour_local,
                trend_per_hr=trend_per_hr,
                trusted_spread=float(trusted_spread),
                forecast_quality=forecast_quality,
                lock_regime=lock_regime,
                survival_calibrator=threshold_survival_calibrator,
            )
            if intraday_threshold_out is not None:
                intraday_threshold_live_alpha = _intraday_threshold_live_alpha(
                    observed_high=observed_high,
                    current_temp_f=current_temp_f,
                    hour_local=float(hour_local_fractional),
                    trusted_spread=float(trusted_spread),
                    forecast_quality=forecast_quality,
                    lock_regime=lock_regime,
                    regime_label=regime_label,
                    unit_mult=unit_mult,
                )
                if intraday_threshold_live_alpha > 0:
                    legacy_probs_before_intraday_blend = list(probs)
                    probs = _blend_probability_vectors(
                        probs,
                        intraday_threshold_out.probs,
                        intraday_threshold_live_alpha,
                    )
                    intraday_threshold_out.alpha = intraday_threshold_live_alpha
                    intraday_threshold_out.notes.append("live_blend_active")
                    if observed_bucket_idx is not None:
                        prob_hotter_bucket = _sum_hotter_bucket_probabilities(
                            probs,
                            observed_bucket_idx,
                        )
        except Exception:
            log.exception("model: intraday threshold shadow failed; legacy path unaffected")
            intraday_threshold_out = None
            intraday_threshold_live_alpha = 0.0

    inputs = {
        "nws_high": nws_high,
        "wu_hourly_peak": wu_hourly_peak,
        "hrrr_high": hrrr_high,
        "hrrr_15min_high": hrrr_15min_high,
        "nbm_high": nbm_high,
        "ecmwf_ifs_high": ecmwf_ifs_high,
        "ecmwf_aifs_high": ecmwf_aifs_high,
        "gfs_graphcast_high": gfs_graphcast_high,
        "pangu_weather_high": pangu_weather_high,
        "fourcastnet_v2_high": fourcastnet_v2_high,
        "aurora_high": aurora_high,
        "daily_high_metar": daily_high_metar,
        "current_temp_f": current_temp_f,
        "mu_forecast": float(mu_forecast),
        "mu_multi_model": float(mu_multi_model),
        "ensemble_breakdown": ensemble_breakdown,
        "kalman_nowcast_active": kalman_nowcast_active,
        "kalman_weight": round(kalman_w, 3),
        "kalman_divergence_f": round(kalman_divergence, 2) if kalman_divergence is not None else None,
        "projected_high": float(projected_high_for_blend),
        "projected_high_raw": float(projected_high_raw),
        "projected_high_for_blend": float(projected_high_for_blend),
        "projected_high_capped": projected_high_capped,
        "metar_projection_gate": metar_projection_gate,
        "metar_forecast_divergence_f": round(divergence_f, 2),
        "w_metar": float(w_metar),
        "w_metar_base": float(w_metar_base),
        "w_metar_gate": w_metar_gate,
        "remaining_rise": remaining_rise,
        "hour_local": hour_local,
        "spread": float(trusted_spread),
        "raw_spread": float(raw_spread) if raw_spread is not None else None,
        "trusted_spread": float(trusted_spread),
        "trusted_reference_median": source_quality.get("trusted_reference_median"),
        "source_quality_gates": source_quality_gates,
        "sigma_raw": float(sigma_raw),
        "sigma_lead": float(sigma_lead) if sigma_lead else 0.0,
        "sigma_lead_source": sigma_lead_source if sigma_lead else None,
        "sigma_lead_generic": float(sigma_lead_generic) if sigma_lead_generic else 0.0,
        "same_day_sigma_cap_applied": same_day_sigma_cap_applied,
        "same_day_sigma_cap_f": same_day_sigma_cap_value,
        "same_day_sigma_cap_source": same_day_sigma_cap_source,
        "time_to_settlement_h": (
            round(time_to_settlement_h, 2)
            if time_to_settlement_h is not None
            else None
        ),
        "regime_label": regime_label,
        "sources_used": list(calibrated.keys()),
        "excluded_sources": excluded_sources,
        "forecast_quality": forecast_quality,
        "prob_new_high": round(prob_hotter_bucket, 4),
        "prob_hotter_bucket": round(prob_hotter_bucket, 4),
        "prob_new_high_raw": round(prob_new_high_raw, 4),
        "lock_regime": lock_regime,
        "lock_suppressed_reason": lock_suppressed_reason,
        "observed_high": observed_high,
        "observed_bucket_idx": observed_bucket_idx,
        "observed_bucket_upper_f": observed_bucket_upper_f,
        "next_hotter_bucket_floor_f": hotter_bucket_floor(canonical_buckets, observed_bucket_idx),
    }

    if intraday_threshold_out is not None:
        inputs["intraday_threshold_shadow"] = intraday_threshold_out.to_dict()
        inputs["threshold_calibration"] = intraday_threshold_out.calibration
        inputs["intraday_threshold_live_alpha"] = round(intraday_threshold_live_alpha, 4)
        inputs["intraday_threshold_live_blend_enabled"] = bool(intraday_threshold_live_alpha > 0)
    if legacy_probs_before_intraday_blend is not None:
        inputs["legacy_probs_before_intraday_blend"] = [
            round(float(p), 6) for p in legacy_probs_before_intraday_blend
        ]

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
            "remaining_rise_cap_active": getattr(adaptive, "remaining_rise_cap_active", True),
            "remaining_rise_cap_reason": getattr(adaptive, "remaining_rise_cap_reason", None),
        }

    # ── M1 BMA shadow output (read-only; does not affect trade decisions) ───
    bma_mu_out: Optional[float] = None
    bma_sigma_out: Optional[float] = None
    bma_probs_out: Optional[list[float]] = None
    bma_meta_out: Optional[dict] = None
    if bma_predictive is not None and bma_predictive.components:
        try:
            bma_mu_out = float(bma_predictive.mean)
            bma_sigma_out = float(bma_predictive.sigma)
            if canonical_buckets:
                if observed_high is not None:
                    bma_probs_out = bma_conditional_bucket_probabilities(
                        bma_predictive, canonical_buckets, floor=observed_high
                    )
                else:
                    bma_probs_out = bma_bucket_probabilities(bma_predictive, canonical_buckets)
            bma_meta_out = predictive_to_dict(bma_predictive)
            bma_meta_out["conditioned_on_observed_high"] = (
                round(float(observed_high), 3) if observed_high is not None else None
            )
            # Diagnostic: how far the mixture mean drifts from the live legacy
            # weighted mean. This can be non-zero while BMA remains a shadow
            # diagnostic over sources that live probabilities exclude or cap.
            bma_meta_out["mu_delta_vs_legacy"] = round(
                bma_mu_out - float(mu_multi_model), 3
            )
            # Diagnostic: σ comparison. Mixture σ should generally be ≥ legacy
            # ensemble σ because it adds the within-source variance term that
            # _ensemble_sigma omits.
            bma_meta_out["sigma_delta_vs_legacy_raw"] = round(
                bma_sigma_out - float(sigma_raw), 3
            )
            _bma_sources = {
                c.get("source") for c in bma_meta_out.get("components", [])
                if isinstance(c, dict)
            }
            _excluded_in_bma = sorted(
                src for src in excluded_sources
                if src in _bma_sources
            )
            bma_meta_out["diagnostic_only"] = bool(_excluded_in_bma)
            bma_meta_out["diagnostic_excluded_sources"] = _excluded_in_bma
        except Exception:
            log.exception("model: BMA shadow output failed; legacy path unaffected")
            bma_mu_out = bma_sigma_out = bma_probs_out = None
            bma_meta_out = None

    # Persist BMA shadow output via the existing ModelSnapshot.inputs_json
    # path so we can compute CRPS comparisons offline against settled events.
    # No schema migration needed — inputs is already a JSON column.
    if bma_meta_out is not None:
        inputs["bma_shadow"] = {
            "mu": round(bma_mu_out, 3) if bma_mu_out is not None else None,
            "sigma": round(bma_sigma_out, 3) if bma_sigma_out is not None else None,
            "probs": [round(p, 6) for p in (bma_probs_out or [])],
            **bma_meta_out,
        }

    return ModelResult(
        mu=float(mu_final),
        sigma=float(sigma_final),
        probs=probs,
        mu_forecast=float(mu_forecast),
        mu_projected=float(projected_high_for_blend),
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
        bma_mu=bma_mu_out,
        bma_sigma=bma_sigma_out,
        bma_probs=bma_probs_out,
        bma_meta=bma_meta_out,
    )
