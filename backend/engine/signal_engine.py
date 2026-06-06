"""
Signal engine — computes per-bucket edge and selects candidate trades.

Edge = model_prob - market_prob - execution_cost

Execution cost model:
  half_spread + slippage_estimate based on depth
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from backend.config import Config
from backend.tz_utils import active_dates_for_city, city_local_date, city_local_now
from backend.modeling.distribution import edge as compute_edge
from backend.engine.market_sanity import evaluate_market_sanity
from backend.modeling.live_calibration import (
    load_live_bucket_diagnostic,
    load_threshold_survival_calibrator,
)
from backend.modeling.settlement import (
    bucket_upper_bound,
    canonical_bucket_ranges,
    find_bucket_idx_for_value,
)
from backend.modeling.temperature_model import (
    apply_forecast_source_quality_gates,
    compute_model,
    ModelResult,
)
from backend.modeling.calibration import get_calibration_async
from backend.modeling.calibration_engine import get_reliability_metrics, remap_probability
from backend.modeling.adaptive import run_adaptive
from backend.modeling.regime import (
    detect_regime,
    regime_kelly_multiplier,
    regime_sigma_inflation,
)
from backend.strategy.kelly import calculate_ev_per_share
from backend.strategy.posterior_kelly import posterior_aware_kelly
from backend.storage.db import get_session
from backend.storage.models import Bucket, Event, City
from backend.storage.repos import (
    bucket_lead_time,
    get_all_cities,
    get_buckets_for_event,
    get_calibration,
    get_daily_high_metar,
    get_event,
    get_latest_successful_forecast,
    get_latest_market_snapshot,
    get_latest_metar,
    get_latest_model_snapshot,
    get_lead_skills_for_city,
    get_recent_model_snapshots,
    get_resolution_high_metar,
    get_station_calibration,
    get_station_profile,
    get_temp_slope,
    get_avg_peak_timing_mins,
    get_todays_metar_obs,
    insert_model_snapshot,
    insert_signal,
    update_heartbeat,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def _forecast_skill_reference_time(obs) -> Optional[datetime]:
    """Timestamp used for live lead-skill and freshness lookups.

    Most NWP sources expose a model initialization time. API-style sources
    such as WU hourly do not, but their historical skill is scored by when the
    forecast was fetched, so use fetched_at as the live fallback.
    """
    for attr in ("model_run_at", "fetched_at"):
        ts = getattr(obs, attr, None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    return None


def _build_source_timing_metadata(
    src_to_obs: dict[str, object],
    lead_skills_by_key: dict,
    settlement_utc: datetime,
) -> tuple[dict[str, datetime], dict[str, Optional[float]], dict[str, int], dict[str, int]]:
    """Build per-source timestamps and lead-skill maps for compute_model().

    The returned timestamp map is passed to compute_model's legacy
    `model_run_at_by_source` argument, but it is intentionally a freshness
    reference time: model_run_at when available, fetched_at otherwise.
    """
    freshness_time_by_source: dict[str, datetime] = {}
    lead_skill_mae_by_source: dict[str, Optional[float]] = {}
    lead_skill_n_obs_by_source: dict[str, int] = {}
    lead_bucket_by_source: dict[str, int] = {}

    for src, obs in src_to_obs.items():
        if obs is None or getattr(obs, "high_f", None) is None:
            continue
        ref_time = _forecast_skill_reference_time(obs)
        if ref_time is None:
            continue

        freshness_time_by_source[src] = ref_time
        lead_h = max(0.0, (settlement_utc - ref_time).total_seconds() / 3600.0)
        bucket = bucket_lead_time(lead_h)
        lead_bucket_by_source[src] = bucket

        skill = lead_skills_by_key.get((src, bucket))
        if skill is not None:
            lead_skill_mae_by_source[src] = skill.mae_f
            lead_skill_n_obs_by_source[src] = skill.n_obs

    return (
        freshness_time_by_source,
        lead_skill_mae_by_source,
        lead_skill_n_obs_by_source,
        lead_bucket_by_source,
    )


def _trusted_pre_model_spread(
    src_to_obs: dict[str, object],
    *,
    station_source_meta: Optional[dict[str, dict]] = None,
    unit_mult: float = 1.0,
) -> tuple[Optional[float], Optional[float], dict]:
    """Return (trusted_spread, raw_spread, gates) for pre-model regime detection."""
    source_values: dict[str, float] = {}
    for src, obs in src_to_obs.items():
        if obs is None or getattr(obs, "high_f", None) is None:
            continue
        source_values[src] = float(getattr(obs, "high_f"))
    quality = apply_forecast_source_quality_gates(
        source_values,
        station_source_meta=station_source_meta,
        unit_mult=unit_mult,
    )
    return (
        quality.get("trusted_spread"),
        quality.get("raw_spread"),
        quality.get("source_quality_gates") or {},
    )


@dataclass
class BucketSignal:
    city_slug: str
    city_display: str
    unit: str
    event_id: int
    bucket_id: int
    bucket_idx: int
    label: str
    low_f: Optional[float]
    high_f: Optional[float]
    model_prob: float
    mkt_prob: float
    raw_edge: float
    exec_cost: float
    true_edge: float
    ev_per_share: float
    ev_at_bid: Optional[float]
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    yes_mid: Optional[float]
    spread: Optional[float]
    yes_ask_depth: float
    yes_bid_depth: float
    reason: dict = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    actionable: bool = False
    prob_new_high: float = 1.0
    prob_hotter_bucket: float = 1.0
    prob_new_high_raw: float = 1.0
    lock_regime: bool = False
    city_state: str = "early"
    resolution_mismatch: Optional[float] = None
    observed_bucket_idx: Optional[int] = None
    observed_bucket_upper_f: Optional[float] = None
    # Phase C3 — regime telemetry (attached per-city; same value for every
    # bucket in the city). None = regime detector did not run this cycle.
    regime_score: Optional[float] = None
    regime_label: Optional[str] = None


def classify_city_state(prob_new_high: float) -> str:
    """Classify city trading state based on probability of a new daily high."""
    if prob_new_high > 0.40:
        return "early"
    elif prob_new_high > 0.15:
        return "approaching"
    elif prob_new_high > 0.05:
        return "volatile"
    else:
        return "resolved"


def compute_twe(signals: list[BucketSignal]) -> float:
    """Trading Window Edge: sum of positive after-cost edges across liquid buckets.

    TWE = sum(max(0, true_edge_i)) for all buckets where mkt_prob_i >= 0.05

    Higher TWE = more total exploitable mispricing in the active trading window.
    Adapts to anomalous weather: more liquid buckets = wider window = higher TWE.
    """
    return round(sum(
        max(0.0, sig.true_edge)
        for sig in signals
        if sig.mkt_prob >= 0.05
    ), 4)


def _effective_probability_floor(
    *,
    settlement_high: Optional[float],
    raw_daily_high: Optional[float],
    current_temp: Optional[float],
) -> tuple[Optional[float], Optional[str]]:
    """Observed high floor used for live probability support.

    `settlement_high` remains the audit value for final resolution, but live
    same-day trading must respect the highest trusted observation currently in
    hand. Otherwise a resolution-minute lag can leave impossible lower buckets
    with model mass while the market correctly reprices them toward zero.
    """
    candidates: list[tuple[float, str]] = []
    for value, source in (
        (settlement_high, "settlement_high"),
        (raw_daily_high, "raw_daily_high"),
        (current_temp, "current_temp"),
    ):
        try:
            if value is not None and math.isfinite(float(value)):
                candidates.append((float(value), source))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def _execution_cost(spread: Optional[float], ask_depth: float) -> float:
    """
    Estimate total execution cost = half_spread + slippage.

    Slippage is depth-dependent: thin markets have more impact.
    """
    half_spread = (spread or 0.04) / 2  # default 2% each side if unknown
    if ask_depth > 200:
        slippage = 0.005
    elif ask_depth > 100:
        slippage = 0.010
    elif ask_depth > 50:
        slippage = 0.015
    else:
        slippage = 0.025  # thin market, high impact
    return float(round(half_spread + slippage, 4))


async def run_signal_engine() -> list[BucketSignal]:
    """
    Run the full signal engine for all enabled cities.

    Returns list of BucketSignal ordered by true_edge descending.
    """
    signals: list[BucketSignal] = []

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    for city in cities:
        for d in active_dates_for_city(city):
            city_signals = await _compute_city_signals(city, d)
            signals.extend(city_signals)

    # Sort by true_edge descending
    signals.sort(key=lambda s: s.true_edge, reverse=True)

    async with get_session() as sess:
        await update_heartbeat(sess, "run_model", success=True)

    return signals


def resolve_active_station(city: City, event: Optional[Event]) -> tuple[str, str]:
    """Resolve which observation station is active for this city/event today.

    Returns (station_id, source) where source is one of:
      - 'event_resolution_station' — overridden by the per-event Polymarket
        resolution source (parsed via _extract_resolution_url at ingest time)
      - 'city_default' — falls back to City.metar_station

    The override path lets us follow Polymarket's actual settlement station
    when it differs from our city default (e.g. Houston: city default KIAH
    vs market-resolved KHOU). The default path keeps existing behavior.
    """
    override = getattr(event, "resolution_station_id", None) if event is not None else None
    if override:
        return override.upper(), "event_resolution_station"
    return ((city.metar_station or "").upper(), "city_default")


async def _compute_city_signals(city: City, today_et: str) -> list[BucketSignal]:
    """Compute signals for all buckets in a city's today event."""
    async with get_session() as sess:
        event = await get_event(sess, city.id, today_et)
        if not event:
            return []

        if event.status not in ("ok",):
            return []

        active_station_id, active_station_source = resolve_active_station(city, event)

        buckets = await get_buckets_for_event(sess, event.id)
        # Live METAR + daily-high are valid only for the city's current local date.
        # For tomorrow / day-after events the late-day-lock predicate must not
        # see today's observed high — otherwise it collapses the next-day
        # distribution onto today's bucket (see plan §Fix 1).
        is_today = (today_et == city_local_date(city))
        metar = await get_latest_metar(sess, city.id) if is_today else None
        daily_high = (
            await get_daily_high_metar(
                sess, city.id, today_et,
                city_tz=getattr(city, "tz", "America/New_York"),
            )
            if is_today else None
        )

        nws_obs = await get_latest_successful_forecast(sess, city.id, "nws", today_et)
        wu_hourly_obs = await get_latest_successful_forecast(sess, city.id, "wu_hourly", today_et)
        wu_history_obs = await get_latest_successful_forecast(sess, city.id, "wu_history", today_et)
        hrrr_obs = await get_latest_successful_forecast(sess, city.id, "hrrr", today_et)
        hrrr_15min_obs = await get_latest_successful_forecast(sess, city.id, "hrrr_15min", today_et)
        nbm_obs = await get_latest_successful_forecast(sess, city.id, "nbm", today_et)
        ecmwf_ifs_obs = await get_latest_successful_forecast(sess, city.id, "ecmwf_ifs", today_et)
        ecmwf_aifs_obs = await get_latest_successful_forecast(sess, city.id, "ecmwf_aifs", today_et)
        # Bayesian-upgrade Q3 — additional AI-NWP foundation models (experimental).
        gfs_graphcast_obs = await get_latest_successful_forecast(sess, city.id, "gfs_graphcast", today_et)
        # §13 — NOAA AIWP-sourced AI members.
        pangu_weather_obs = await get_latest_successful_forecast(sess, city.id, "pangu_weather", today_et)
        fourcastnet_v2_obs = await get_latest_successful_forecast(sess, city.id, "fourcastnet_v2", today_et)
        # §17 — Microsoft Aurora (Swin transformer) via NOAA AIWP.
        aurora_obs = await get_latest_successful_forecast(sess, city.id, "aurora", today_et)

        cal = await get_calibration(sess, city.id)
        # Phase B1: lead-time skill table for ensemble weight adjustment.
        lead_skills_by_key = await get_lead_skills_for_city(sess, city.id)
        # NEW: Reliability metrics for probability remapping
        reliability_bins = await get_reliability_metrics(city.id)
        station_cal = (
            await get_station_calibration(sess, active_station_id)
            if active_station_id else None
        )

        # Station profile for resolution-aware high — keyed on the active
        # station (per-event override > city default).
        profile = await get_station_profile(sess, active_station_id) if active_station_id else None
        if profile and profile.observation_minutes:
            valid_minutes = json.loads(profile.observation_minutes)
            resolution_high = await get_resolution_high_metar(sess, city.id, today_et, valid_minutes, city_tz=getattr(city, "tz", "America/New_York"))
        else:
            valid_minutes = None
            resolution_high = None

        # ML features for the residual tracker
        temp_slope_3h = await get_temp_slope(sess, city.id, hours_back=3)
        avg_peak_mins = await get_avg_peak_timing_mins(
            sess, city.id, days_back=3, tz=ZoneInfo(getattr(city, "tz", "America/New_York"))
        )

        # Fetch ALL of today's 5-minute observations for adaptive engine
        todays_obs_rows = await get_todays_metar_obs(
            sess, city.id, today_et, city_tz=getattr(city, "tz", "America/New_York")
        )

    # Extract weather condition from METAR extended data for anomaly badges
    metar_condition = None
    if metar and metar.extended:
        metar_condition = getattr(metar.extended, "condition", None)

    if not buckets:
        log.debug("signal: %s — no buckets", city.city_slug)
        return []

    # Build bucket boundary list
    bucket_ranges = [(b.low_f, b.high_f) for b in buckets]
    canonical_ranges = canonical_bucket_ranges(bucket_ranges)

    # Resolve ground truth: take the MAX of every observation source we trust.
    # Using "first available in priority order" causes desync when the
    # primary source (WU history) lags behind METAR — the model would then
    # see a too-low observed high, fail to lock, and not zero out
    # already-surpassed buckets even though the gates correctly do.
    #
    # Freshness gate: wu_history is keyed by today_et but its stored high_f
    # can carry the prior local date's peak when WU hasn't updated. Reject
    # the value if its observation timestamp resolves to a different local
    # date. daily_high and resolution_high are already local-date scoped at
    # the SQL level (observed_at in tz-aware bounds).
    _city_tz = ZoneInfo(getattr(city, "tz", "America/New_York"))
    _wu_hist_high = None
    if wu_history_obs and wu_history_obs.high_f is not None:
        _wu_obs_time = None
        try:
            _wu_raw = json.loads(wu_history_obs.raw_json) if wu_history_obs.raw_json else {}
            _obs_time_str = _wu_raw.get("obs_time")
            if _obs_time_str:
                _wu_obs_time = datetime.fromisoformat(str(_obs_time_str).rstrip("Z")).replace(tzinfo=timezone.utc)
        except Exception:
            _wu_obs_time = None
        if _wu_obs_time is None or _wu_obs_time.astimezone(_city_tz).strftime("%Y-%m-%d") == today_et:
            _wu_hist_high = wu_history_obs.high_f

    # Resolution-minute gate: when the station profile declares valid
    # settlement minutes (e.g. KATL samples at :52), a raw METAR spike
    # *between* those minutes cannot lock a bucket — settlement will
    # resolve on the :52 reading, not the intra-minute peak. Exclude raw
    # daily_high from the floor in that case; it remains visible in the
    # debug block via raw_daily_high for diagnostics.
    if valid_minutes:
        _candidate_highs = [
            v for v in (_wu_hist_high, resolution_high)
            if v is not None
        ]
    else:
        _candidate_highs = [
            v for v in (_wu_hist_high, resolution_high, daily_high)
            if v is not None
        ]
    ground_truth_high = max(_candidate_highs) if _candidate_highs else None

    # Track which source supplied the winning value for audit/debug.
    ground_truth_source: Optional[str] = None
    if ground_truth_high is not None:
        if _wu_hist_high == ground_truth_high:
            ground_truth_source = "wu_history"
        elif resolution_high == ground_truth_high:
            ground_truth_source = "resolution_metar"
        else:
            ground_truth_source = "raw_metar"

    # Observed high floor for conditional probabilities:
    # ground_truth_high is the settlement-audit floor. The live probability floor
    # also includes raw/current observations because daily-high support is
    # monotone and the market reprices buckets as soon as the current high makes
    # them commercially impossible.
    observed_high_floor, probability_floor_source = _effective_probability_floor(
        settlement_high=ground_truth_high,
        raw_daily_high=daily_high,
        current_temp=metar.temp_f if metar else None,
    )
    if ground_truth_high is None and observed_high_floor is not None:
        ground_truth_source = probability_floor_source

    # Resolution mismatch: raw_high exceeds resolution_high
    resolution_mismatch = None
    if daily_high is not None and resolution_high is not None:
        mismatch = daily_high - resolution_high
        if mismatch >= 1.0:
            resolution_mismatch = round(mismatch, 1)

    # Build calibration dict
    cal_dict = None
    if cal:
        cal_dict = {
            "bias_nws": cal.bias_nws,
            "bias_wu_hourly": cal.bias_wu_hourly,
            "bias_hrrr": v if (v := getattr(cal, "bias_hrrr", None)) is not None else 0.0,
            "bias_nbm": v if (v := getattr(cal, "bias_nbm", None)) is not None else 0.0,
            "weight_nws": cal.weight_nws,
            "weight_wu_hourly": cal.weight_wu_hourly,
            "weight_hrrr": v if (v := getattr(cal, "weight_hrrr", None)) is not None else 0.5,
            "weight_nbm": v if (v := getattr(cal, "weight_nbm", None)) is not None else 0.2,
        }
    if station_cal is not None:
        if cal_dict is None:
            cal_dict = {}
        cal_dict["station_mae_f"] = getattr(station_cal, "mae_f", None)
        cal_dict["station_rmse_f"] = getattr(station_cal, "rmse_f", None)
        cal_dict["station_n_samples"] = getattr(station_cal, "n_samples", 0)

    # Overlay dynamic per-station weights + biases learned from forecast skill.
    # Station-level values (when available) replace the city-level calibration
    # for that source; compute_model falls back to cal_dict / defaults otherwise.
    station_source_meta: dict = {}
    try:
        from backend.modeling.station_weights import (
            load_source_skill_summary,
            load_station_source_weights,
        )
        station_weights, station_biases = await load_station_source_weights(
            active_station_id
        )
        station_source_meta = await load_source_skill_summary(active_station_id)
        if station_weights or station_biases or station_source_meta:
            if cal_dict is None:
                cal_dict = {}
            cal_dict["station_source_weights"] = station_weights
            cal_dict["station_source_biases"] = station_biases
            cal_dict["station_source_meta"] = station_source_meta
    except Exception:
        log.exception("signal: %s — station weight load failed", city.city_slug)

    _src_to_obs = {
        "nws": nws_obs, "wu_hourly": wu_hourly_obs, "hrrr": hrrr_obs,
        "hrrr_15min": hrrr_15min_obs, "nbm": nbm_obs, "ecmwf_ifs": ecmwf_ifs_obs,
        "ecmwf_aifs": ecmwf_aifs_obs,
        "gfs_graphcast": gfs_graphcast_obs,
        "pangu_weather": pangu_weather_obs,
        "fourcastnet_v2": fourcastnet_v2_obs,
        "aurora": aurora_obs,
    }

    # ── Adaptive prediction engine (Kalman + regression) ────────────────────
    city_tz_str = getattr(city, "tz", "America/New_York")
    now_local = city_local_now(city)
    adaptive_result = None

    # Extract WU hourly peak time for composite peak timing
    wu_hourly_peak_time = None
    if wu_hourly_obs and wu_hourly_obs.raw_json:
        try:
            wu_hourly_peak_time = json.loads(wu_hourly_obs.raw_json).get("peak_hour")
        except Exception:
            pass

    obs_dicts = []
    if valid_minutes and todays_obs_rows:
        # Convert MetarObs rows to dicts for the adaptive engine
        for row in todays_obs_rows:
            d = {
                "observed_at": row.observed_at,
                "temp_f": row.temp_f,
            }
            # Parse extended fields from raw_json if available
            if row.raw_json:
                try:
                    raw = json.loads(row.raw_json) if isinstance(row.raw_json, str) else row.raw_json
                    if isinstance(raw, dict):
                        d["wind_speed_kt"] = raw.get("wspd")
                        d["wx_string"] = raw.get("wxString")
                        d["altimeter_inhg"] = raw.get("altim")
                        cover = raw.get("cover")
                        d["cloud_cover"] = cover
                        cloud_map = {"CLR": 0, "SKC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}
                        d["cloud_cover_val"] = cloud_map.get((cover or "").upper(), None)
                        # Humidity and dewpoint from raw METAR (Magnus formula)
                        dewp = raw.get("dewp")
                        temp_c = raw.get("temp")
                        if dewp is not None and temp_c is not None:
                            try:
                                tc, dc = float(temp_c), float(dewp)
                                d["humidity_pct"] = 100 * math.exp((17.625 * dc) / (243.04 + dc)) / math.exp((17.625 * tc) / (243.04 + tc))
                                d["dewpoint_f"] = dc * 9.0 / 5.0 + 32.0
                            except Exception:
                                pass
                        d["wind_gust_kt"] = raw.get("wgst")
                        # Precipitation flag
                        wx = raw.get("wxString") or ""
                        d["precip_flag"] = any(tok in wx.upper() for tok in ("RA", "TS", "SH", "SN", "DZ"))
                except Exception:
                    pass
            obs_dicts.append(d)

        # Trusted same-day consensus for adaptive peak/cap guards. Use the
        # same source-quality gate as the live probability model so a bad AI
        # member cannot drag adaptive timing or remaining-rise caps.
        _adaptive_source_values = {
            src: float(obs.high_f)
            for src, obs in _src_to_obs.items()
            if obs is not None and getattr(obs, "high_f", None) is not None
        }
        _adaptive_quality = apply_forecast_source_quality_gates(
            _adaptive_source_values,
            station_source_meta=station_source_meta,
            unit_mult=1.0 if getattr(city, "unit", "F") == "F" else 5.0 / 9.0,
        )
        _adaptive_live_values = list((_adaptive_quality.get("live_values") or {}).values())
        if _adaptive_live_values:
            adaptive_forecast_high = float(statistics.median(_adaptive_live_values))
        else:
            _fc_highs = [
                s.high_f for s in [nws_obs, wu_hourly_obs]
                if s is not None and s.high_f is not None
            ]
            adaptive_forecast_high = sum(_fc_highs) / len(_fc_highs) if _fc_highs else None

        try:
            adaptive_result = run_adaptive(
                todays_obs=obs_dicts,
                observation_minutes=valid_minutes,
                now_local=now_local,
                city_tz=city_tz_str,
                wu_hourly_peak_time=wu_hourly_peak_time,
                historical_peak_mins=avg_peak_mins,
                forecast_high=adaptive_forecast_high,
                ml_features={
                    "temp_slope_3h": temp_slope_3h or 0.0,
                    "avg_peak_timing_mins": avg_peak_mins or 960.0,
                    "day_of_year": now_local.timetuple().tm_yday,
                },
            )
        except Exception:
            log.exception("signal: %s — adaptive engine failed", city.city_slug)

    # Build latest weather conditions dict for sigma adjustment
    _latest_wx: Optional[dict] = None
    if obs_dicts:
        _lw = obs_dicts[-1]
        _dp_spread = None
        if metar and metar.temp_f is not None and _lw.get("dewpoint_f") is not None:
            _dp_spread = max(0.0, metar.temp_f - _lw["dewpoint_f"])
        # Pressure tendency: diff between first and last obs altimeter
        _p_tendency = None
        _p_first = next((o.get("altimeter_inhg") for o in obs_dicts if o.get("altimeter_inhg") is not None), None)
        _p_last = _lw.get("altimeter_inhg")
        if _p_first is not None and _p_last is not None:
            _p_tendency = _p_last - _p_first
        _latest_wx = {
            "cloud_cover_val": _lw.get("cloud_cover_val"),
            "humidity_pct": _lw.get("humidity_pct"),
            "wind_speed_kt": _lw.get("wind_speed_kt"),
            "wind_gust_kt": _lw.get("wind_gust_kt"),
            "pressure_tendency": _p_tendency,
            "has_precip": _lw.get("precip_flag", False),
            "dewpoint_spread_f": _dp_spread,
        }

    # ── Phase B1+B2: per-source freshness + lead-skill metadata ──────────
    # Settlement = end of city local day (23:59:59 in city tz). Lead-time
    # buckets in SourceLeadTimeSkill snap to {0, 1, 3, 6, 12, 18, 24, 36, 48, 72}h.
    _settlement_local = datetime.strptime(today_et, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=ZoneInfo(getattr(city, "tz", "America/New_York"))
    )
    _settlement_utc = _settlement_local.astimezone(timezone.utc)
    _now_utc = datetime.now(timezone.utc)
    live_calibration_hour_bucket = max(0, min(23, int(now_local.hour)))
    live_calibration_floor_idx = -1
    if observed_high_floor is not None:
        _floor_idx = find_bucket_idx_for_value(canonical_ranges, observed_high_floor)
        live_calibration_floor_idx = int(_floor_idx) if _floor_idx is not None else -1

    _src_to_obs = {
        "nws": nws_obs, "wu_hourly": wu_hourly_obs, "hrrr": hrrr_obs,
        "hrrr_15min": hrrr_15min_obs, "nbm": nbm_obs, "ecmwf_ifs": ecmwf_ifs_obs,
        "ecmwf_aifs": ecmwf_aifs_obs,
        "gfs_graphcast": gfs_graphcast_obs,
        "pangu_weather": pangu_weather_obs,
        "fourcastnet_v2": fourcastnet_v2_obs,
        "aurora": aurora_obs,
    }
    (
        model_run_at_by_source,
        lead_skill_mae_by_source,
        lead_skill_n_obs_by_source,
        lead_bucket_by_source,
    ) = _build_source_timing_metadata(
        _src_to_obs,
        lead_skills_by_key,
        _settlement_utc,
    )

    # M1 Phase 2 — fetch EM-fit BMA weights for the operative lead bucket.
    # Pick the median lead bucket across active sources; that single fit
    # represents the panel's "typical" lead at trade time. Sources with
    # leads outside that bucket still get their fitted weight (graceful
    # approximation) and any source missing from the fitted set falls back
    # to the legacy lead-skill × freshness weight inside build_bma_predictive.
    fitted_bma_weights: Optional[dict[str, float]] = None
    if lead_bucket_by_source:
        try:
            from backend.modeling.bma_weights_repo import get_bma_weights_for_city

            _buckets_sorted = sorted(lead_bucket_by_source.values())
            _operative_bucket = _buckets_sorted[len(_buckets_sorted) // 2]
            fitted_bma_weights = await get_bma_weights_for_city(
                sess, city.id, _operative_bucket,
            )
        except Exception:
            log.exception(
                "signal: %s — BMA fitted weights fetch failed; legacy weights used",
                city.city_slug,
            )

    # ── Q6: regime detection BEFORE compute_model so σ inflation can flow ──
    # The pre-Q6 path detected regime AFTER the model and used the result
    # only for Kelly sizing. Detect early using the trusted-source spread so
    # a quarantined forecast member cannot falsely mark the day volatile.
    _pre_regime: Optional["RegimeResult"] = None
    try:
        _pre_spread, _pre_raw_spread, _pre_quality_gates = _trusted_pre_model_spread(
            _src_to_obs,
            station_source_meta=station_source_meta,
            unit_mult=5.0 / 9.0 if getattr(city, "unit", "F") == "C" else 1.0,
        )
        _recent_snaps_pre = await get_recent_model_snapshots(sess, event.id, limit=4)
        _pre_hist_spreads: list[float] = []
        for _s in _recent_snaps_pre:
            try:
                _ij = json.loads(_s.inputs_json) if _s.inputs_json else {}
                _sp = _ij.get("trusted_spread", _ij.get("spread"))
                if _sp is not None:
                    _pre_hist_spreads.append(float(_sp))
            except Exception:
                pass
            if len(_pre_hist_spreads) >= 3:
                break
        _pre_regime = detect_regime(
            current_spread_f=_pre_spread,
            historical_spreads_f=_pre_hist_spreads or None,
            pressure_tendency_inhg=(_latest_wx or {}).get("pressure_tendency"),
            has_precip=bool((_latest_wx or {}).get("has_precip")),
        )
    except Exception:
        log.exception(
            "signal: %s — pre-model regime detection failed; σ inflation skipped",
            city.city_slug,
        )
        _pre_regime = None

    _regime_sigma_mult: Optional[float] = (
        regime_sigma_inflation(_pre_regime.score) if _pre_regime else None
    )

    threshold_survival_calibrator = None
    threshold_calibration_context: dict = {
        "context_used": "identity",
        "city_id": city.id,
        "station_id": active_station_id,
        "hour_bucket": live_calibration_hour_bucket,
        "observed_floor_bucket_idx": live_calibration_floor_idx,
        "applied": False,
        "reason": "not_loaded",
    }
    try:
        async with get_session() as cal_sess:
            threshold_survival_calibrator, threshold_calibration_context = (
                await load_threshold_survival_calibrator(
                    cal_sess,
                    city_id=city.id,
                    station_id=active_station_id,
                    hour_bucket=live_calibration_hour_bucket,
                    observed_floor_bucket_idx=live_calibration_floor_idx,
                )
            )
    except Exception:
        log.exception("signal: %s — threshold calibration load failed", city.city_slug)
        threshold_calibration_context = {
            **threshold_calibration_context,
            "context_used": "error",
            "reason": "load_failed",
        }

    # Run temperature model
    model = compute_model(
        nws_high=nws_obs.high_f if nws_obs else None,
        wu_hourly_peak=wu_hourly_obs.high_f if wu_hourly_obs else None,
        hrrr_high=hrrr_obs.high_f if hrrr_obs else None,
        hrrr_15min_high=hrrr_15min_obs.high_f if hrrr_15min_obs else None,
        nbm_high=nbm_obs.high_f if nbm_obs else None,
        ecmwf_ifs_high=ecmwf_ifs_obs.high_f if ecmwf_ifs_obs else None,
        ecmwf_aifs_high=ecmwf_aifs_obs.high_f if ecmwf_aifs_obs else None,
        gfs_graphcast_high=gfs_graphcast_obs.high_f if gfs_graphcast_obs else None,
        pangu_weather_high=pangu_weather_obs.high_f if pangu_weather_obs else None,
        fourcastnet_v2_high=fourcastnet_v2_obs.high_f if fourcastnet_v2_obs else None,
        aurora_high=aurora_obs.high_f if aurora_obs else None,
        model_run_at_by_source=model_run_at_by_source or None,
        # Q4 — settlement time used for lead-time σ growth (NOAA empirical
        # 1.5 + 0.05·L °F, combined in quadrature with ensemble σ).
        event_settlement_utc=_settlement_utc,
        # Q6 — regime-aware σ inflation (1.0 calm → 2.0 volatile).
        regime_sigma_multiplier=_regime_sigma_mult,
        regime_label=_pre_regime.label.value if _pre_regime else None,
        lead_skill_mae_by_source=lead_skill_mae_by_source or None,
        lead_skill_n_obs_by_source=lead_skill_n_obs_by_source or None,
        # M1 Phase 2 — EM-fit BMA mixing weights (None when no fit yet).
        fitted_bma_weights_by_source=fitted_bma_weights,
        now_utc=_now_utc,
        daily_high_metar=ground_truth_high,
        current_temp_f=metar.temp_f if metar else None,
        calibration=cal_dict,
        buckets=bucket_ranges,
        forecast_quality=event.forecast_quality or "ok",
        unit=getattr(city, "unit", "F"),
        city_tz=city_tz_str,
        observed_high=observed_high_floor,
        ml_features={
            "temp_slope_3h": temp_slope_3h,
            "avg_peak_timing_mins": avg_peak_mins,
            "day_of_year": datetime.now(timezone.utc).timetuple().tm_yday,
            "regime_score": _pre_regime.score if _pre_regime else 0.0,
        },
        adaptive=adaptive_result,
        latest_weather=_latest_wx,
        threshold_survival_calibrator=threshold_survival_calibrator,
    )

    if model is None:
        log.warning("signal: %s — model returned None (insufficient data)", city.city_slug)
        return []

    prob_hotter_bucket = model.prob_hotter_bucket
    city_state = classify_city_state(prob_hotter_bucket)
    model.inputs["city_state"] = city_state
    # Stamp the resolved station + ground-truth source onto the snapshot so
    # /state/{city} and the dashboard can display which station/source is
    # actually driving today's model.
    model.inputs["active_station_id"] = active_station_id
    model.inputs["active_station_source"] = active_station_source
    model.inputs["ground_truth_high"] = ground_truth_high
    model.inputs["ground_truth_source"] = ground_truth_source
    model.inputs["raw_daily_high"] = daily_high
    model.inputs["probability_floor_high"] = observed_high_floor
    model.inputs["probability_floor_source"] = probability_floor_source
    model.inputs["settlement_floor_high"] = ground_truth_high
    model.inputs["observation_minutes"] = valid_minutes
    # Q7 — stamp regime telemetry on the snapshot so the backtest harness can
    # split per-regime metrics (Brier/win-rate on CALM vs VOLATILE days).
    model.inputs["regime_score"] = (
        round(_pre_regime.score, 3) if _pre_regime else None
    )
    model.inputs["regime_label"] = (
        _pre_regime.label.value if _pre_regime else None
    )
    model.inputs["regime_sigma_multiplier"] = (
        round(_regime_sigma_mult, 3) if _regime_sigma_mult is not None else None
    )
    if not model.inputs.get("threshold_calibration"):
        model.inputs["threshold_calibration"] = threshold_calibration_context
    elif threshold_calibration_context:
        model.inputs["threshold_calibration"] = {
            **threshold_calibration_context,
            **model.inputs["threshold_calibration"],
        }

    # Use a single session for all DB writes (model snapshot + per-bucket reads/inserts)
    signals: list[BucketSignal] = []
    async with get_session() as sess:
        # Persist model snapshot — capture id so each Signal row generated
        # in this pass can be tagged with its parent generation. This is
        # what lets the dashboard filter to "latest snapshot only".
        snapshot = await insert_model_snapshot(
            sess,
            event_id=event.id,
            mu=model.mu,
            sigma=model.sigma,
            probs_json=json.dumps(model.probs),
            inputs_json=json.dumps(model.inputs),
            forecast_quality=model.forecast_quality,
        )
        snapshot_id = snapshot.id if snapshot is not None else None

        # Phase C3 + Q6 — regime telemetry already computed pre-model so its
        # σ inflation could flow through compute_model. Reuse the result for
        # downstream Kelly multiplier + per-bucket telemetry.
        _regime = _pre_regime

        if city_state == "resolved":
            log.info(
                "signal: %s — resolved (prob_hotter_bucket=%.3f), writing non-actionable signals",
                city.city_slug,
                prob_hotter_bucket,
            )

        # Compute signal per bucket
        for i, bucket in enumerate(buckets):
            if i >= len(model.probs):
                continue

            model_prob_threshold_calibrated = model.probs[i]
            model_prob = model_prob_threshold_calibrated

            # If METAR high already exceeds this bucket's ceiling, probability is 0
            # (the final daily high can only go up, never down)
            bucket_ceiling = bucket_upper_bound(canonical_ranges, i)
            if ground_truth_high is not None and bucket_ceiling is not None:
                if ground_truth_high >= bucket_ceiling:
                    model_prob = 0.0

            # Get latest market snapshot (reuse session)
            mkt_snap = await get_latest_market_snapshot(sess, bucket.id)

            if not mkt_snap or mkt_snap.yes_mid is None:
                # No market data — count as signal with no actionable edge
                sig = BucketSignal(
                    city_slug=city.city_slug,
                    city_display=city.display_name,
                    unit=getattr(city, "unit", "F"),
                    event_id=event.id,
                    bucket_id=bucket.id,
                    bucket_idx=i,
                    label=bucket.label or f"Bucket {i}",
                    low_f=bucket.low_f,
                    high_f=bucket.high_f,
                    model_prob=float(round(model_prob, 4)),
                    mkt_prob=0.0,
                    raw_edge=0.0,
                    exec_cost=0.0,
                    true_edge=0.0,
                    ev_per_share=0.0,
                    ev_at_bid=None,
                    yes_bid=None,
                    yes_ask=None,
                    yes_mid=None,
                    spread=None,
                    yes_ask_depth=0.0,
                    yes_bid_depth=0.0,
                    gate_failures=["no_market_data"],
                    prob_new_high=prob_hotter_bucket,
                    prob_hotter_bucket=prob_hotter_bucket,
                    prob_new_high_raw=model.prob_new_high_raw,
                    lock_regime=model.lock_regime,
                    city_state=city_state,
                    resolution_mismatch=resolution_mismatch,
                    observed_bucket_idx=model.observed_bucket_idx,
                    observed_bucket_upper_f=model.observed_bucket_upper_f,
                    regime_score=(_regime.score if _regime else None),
                    regime_label=(_regime.label.value if _regime else None),
                )
                signals.append(sig)
                continue

            mkt_prob = mkt_snap.yes_mid
            ask_depth = mkt_snap.yes_ask_depth or 0.0
            bid_depth = mkt_snap.yes_bid_depth or 0.0
            spread = mkt_snap.spread
            exec_cost = _execution_cost(spread, ask_depth)

            bucket_live_calibration = await load_live_bucket_diagnostic(
                sess,
                city_id=city.id,
                station_id=active_station_id,
                hour_bucket=live_calibration_hour_bucket,
                observed_floor_bucket_idx=live_calibration_floor_idx,
                bucket_idx=i,
                prob=model_prob,
            )
            if bucket_live_calibration.get("applied"):
                model_prob = float(bucket_live_calibration["bucket_calibrated_prob"])

            # Apply probability calibration (remap based on historical reliability)
            calibrated_prob = remap_probability(model_prob, reliability_bins)

            # Edge calculation based on calibrated probability
            raw_edge_buy = calibrated_prob - mkt_prob
            true_edge = raw_edge_buy - exec_cost

            # Per-share EV (used by EDGE_DECAY exit gate).
            # ev_per_share uses the mid (entry-side reference); ev_at_bid uses
            # the bid (exit-side reference) since exits sell into the bid.
            ev_per_share = calculate_ev_per_share(calibrated_prob, mkt_prob)
            ev_at_bid = (
                calculate_ev_per_share(calibrated_prob, mkt_snap.yes_bid)
                if mkt_snap.yes_bid is not None
                else None
            )
            market_age_s = None
            fetched_at = getattr(mkt_snap, "fetched_at", None)
            if fetched_at is not None:
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                market_age_s = max(
                    0.0,
                    (datetime.now(timezone.utc) - fetched_at.astimezone(timezone.utc)).total_seconds(),
                )
            posterior_kelly_payload = None
            try:
                pk = posterior_aware_kelly(
                    bma_shadow=model.inputs.get("bma_shadow"),
                    low_f=bucket.low_f,
                    high_f=bucket.high_f,
                    yes_price=mkt_snap.yes_ask or mkt_prob,
                    fractional_kelly=Config.KELLY_FRACTION,
                    max_position_size=Config.MAX_POSITION_PCT,
                )
                if pk is not None:
                    posterior_kelly_payload = {
                        **pk.to_dict(),
                        "sizing_price": float(round(mkt_snap.yes_ask or mkt_prob, 4)),
                        "source": "bma_component_weighted_median",
                    }
            except Exception:
                log.debug("signal: posterior Kelly failed for bucket %s", bucket.id, exc_info=True)

            reason = {
                **model.inputs,
                "bucket_idx": i,
                "label": bucket.label,
                "model_prob_raw": float(round(model_prob, 4)),
                "model_prob_threshold_calibrated": float(round(model_prob_threshold_calibrated, 4)),
                "bucket_live_calibration": bucket_live_calibration,
                "model_prob_cal": float(round(calibrated_prob, 4)),
                "mkt_prob": float(round(mkt_prob, 4)),
                "raw_edge": float(round(raw_edge_buy, 4)),
                "exec_cost": float(round(exec_cost, 4)),
                "true_edge": float(round(true_edge, 4)),
                "ev_per_share": float(round(ev_per_share, 6)),
                "ev_at_bid": (float(round(ev_at_bid, 6)) if ev_at_bid is not None else None),
                "posterior_kelly": posterior_kelly_payload,
                "spread": spread,
                "ask_depth": ask_depth,
                "bid_depth": bid_depth,
                "market_snapshot_age_s": round(market_age_s, 1) if market_age_s is not None else None,
                "city_state": city_state,
                "prob_hotter_bucket": round(model.prob_hotter_bucket, 4),
                "prob_new_high_raw": round(model.prob_new_high_raw, 4),
                "lock_regime": model.lock_regime,
                "observed_bucket_idx": model.observed_bucket_idx,
                "observed_bucket_upper_f": model.observed_bucket_upper_f,
                "resolution_high": resolution_high,
                "raw_high": daily_high,
                "ground_truth_high": ground_truth_high,
                "ground_truth_source": ground_truth_source,
                "active_station_id": active_station_id,
                "active_station_source": active_station_source,
                "observation_minutes": valid_minutes,
                "resolution_mismatch": resolution_mismatch,
                "metar_condition": metar_condition,
            }

            gate_failures: list[str] = []
            if (
                posterior_kelly_payload is not None
                and posterior_kelly_payload.get("conservative_kelly_f", 0.0) <= 0.0
            ):
                gate_failures.append("posterior_kelly_no_size")
            threshold_calibration = reason.get("threshold_calibration") or {}
            try:
                threshold_cal_n = (
                    int(threshold_calibration.get("min_sample_count") or 0)
                    if threshold_calibration.get("context_used") == "city_station_hour_floor"
                    else 0
                )
            except (TypeError, ValueError):
                threshold_cal_n = 0
            market_sanity = evaluate_market_sanity(
                model_prob=calibrated_prob,
                market_prob=mkt_prob,
                exec_cost=exec_cost,
                model_true_edge=true_edge,
                market_snapshot_age_s=market_age_s,
                spread=spread,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                min_true_edge=Config.MIN_TRUE_EDGE,
                threshold_calibration_n=threshold_cal_n,
            )
            reason["market_sanity"] = market_sanity
            if market_sanity.get("blocked"):
                gate_failures.append(market_sanity["failure"])

            actionable = (
                true_edge >= Config.MIN_TRUE_EDGE
                and 0.02 <= mkt_prob <= 0.98  # avoid extreme markets
                and ask_depth >= Config.MIN_LIQUIDITY_SHARES
                and event.forecast_quality == "ok"
                and city_state != "resolved"
                and not gate_failures
            )

            sig = BucketSignal(
                city_slug=city.city_slug,
                city_display=city.display_name,
                unit=getattr(city, "unit", "F"),
                event_id=event.id,
                bucket_id=bucket.id,
                bucket_idx=i,
                label=bucket.label or f"Bucket {i}",
                low_f=bucket.low_f,
                high_f=bucket.high_f,
                model_prob=float(round(model_prob, 4)),
                mkt_prob=float(round(mkt_prob, 4)),
                raw_edge=round(raw_edge_buy, 4),
                exec_cost=round(exec_cost, 4),
                true_edge=round(true_edge, 4),
                ev_per_share=round(ev_per_share, 6),
                ev_at_bid=(round(ev_at_bid, 6) if ev_at_bid is not None else None),
                yes_bid=mkt_snap.yes_bid,
                yes_ask=mkt_snap.yes_ask,
                yes_mid=float(round(mkt_prob, 4)),
                spread=spread,
                yes_ask_depth=ask_depth,
                yes_bid_depth=bid_depth,
                reason=reason,
                gate_failures=gate_failures,
                actionable=actionable,
                prob_new_high=prob_hotter_bucket,
                prob_hotter_bucket=prob_hotter_bucket,
                prob_new_high_raw=model.prob_new_high_raw,
                lock_regime=model.lock_regime,
                city_state=city_state,
                resolution_mismatch=resolution_mismatch,
                observed_bucket_idx=model.observed_bucket_idx,
                observed_bucket_upper_f=model.observed_bucket_upper_f,
                regime_score=(_regime.score if _regime else None),
                regime_label=(_regime.label.value if _regime else None),
            )
            signals.append(sig)

            # Persist signal to DB (reuse session) — tag with snapshot_id so
            # the dashboard can filter to "rows from the latest generation".
            await insert_signal(
                sess,
                bucket_id=bucket.id,
                model_snapshot_id=snapshot_id,
                model_prob=sig.model_prob,
                mkt_prob=sig.mkt_prob,
                raw_edge=sig.raw_edge,
                exec_cost=sig.exec_cost,
                true_edge=sig.true_edge,
                reason_json=json.dumps(reason, default=str),
                gate_failures_json=json.dumps(sig.gate_failures),
            )

    return signals
