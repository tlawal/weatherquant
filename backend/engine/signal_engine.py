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
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from backend.config import Config
from backend.tz_utils import active_dates_for_city, city_local_date, city_local_now
from backend.modeling.distribution import edge as compute_edge
from backend.modeling.temperature_model import compute_model, ModelResult
from backend.modeling.calibration import get_calibration_async
from backend.modeling.calibration_engine import get_reliability_metrics, remap_probability
from backend.modeling.adaptive import run_adaptive
from backend.storage.db import get_session
from backend.storage.models import Bucket, Event, City
from backend.storage.repos import (
    get_all_cities,
    get_buckets_for_event,
    get_calibration,
    get_daily_high_metar,
    get_event,
    get_latest_forecast,
    get_latest_market_snapshot,
    get_latest_metar,
    get_latest_model_snapshot,
    get_resolution_high_metar,
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
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    yes_mid: Optional[float]
    spread: Optional[float]
    yes_ask_depth: float
    reason: dict = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    actionable: bool = False
    prob_new_high: float = 1.0
    city_state: str = "early"
    resolution_mismatch: Optional[float] = None


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


async def _compute_city_signals(city: City, today_et: str) -> list[BucketSignal]:
    """Compute signals for all buckets in a city's today event."""
    async with get_session() as sess:
        event = await get_event(sess, city.id, today_et)
        if not event:
            return []

        if event.status not in ("ok",):
            return []

        buckets = await get_buckets_for_event(sess, event.id)
        metar = await get_latest_metar(sess, city.id)
        daily_high = await get_daily_high_metar(sess, city.id, today_et, city_tz=getattr(city, "tz", "America/New_York"))

        nws_obs = await get_latest_forecast(sess, city.id, "nws", today_et)
        wu_daily_obs = await get_latest_forecast(sess, city.id, "wu_daily", today_et)
        wu_hourly_obs = await get_latest_forecast(sess, city.id, "wu_hourly", today_et)
        wu_history_obs = await get_latest_forecast(sess, city.id, "wu_history", today_et)
        hrrr_obs = await get_latest_forecast(sess, city.id, "hrrr", today_et)
        gfs_obs = await get_latest_forecast(sess, city.id, "gfs", today_et)

        cal = await get_calibration(sess, city.id)
        # NEW: Reliability metrics for probability remapping
        reliability_bins = await get_reliability_metrics(city.id)

        # Station profile for resolution-aware high
        profile = await get_station_profile(sess, city.metar_station) if city.metar_station else None
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

    if not buckets:
        log.debug("signal: %s — no buckets", city.city_slug)
        return []

    # Build bucket boundary list
    bucket_ranges = [(b.low_f, b.high_f) for b in buckets]

    # Resolve ground truth: prefer WU history (settlement source),
    # then resolution-filtered METAR, then raw METAR
    if wu_history_obs and wu_history_obs.high_f is not None:
        ground_truth_high = wu_history_obs.high_f
    elif resolution_high is not None:
        ground_truth_high = resolution_high
    else:
        ground_truth_high = daily_high  # raw MAX fallback

    # Observed high floor for conditional probabilities:
    # Use ground_truth_high if available, otherwise fall back to current METAR temp
    # (current temp is always a lower bound for the daily high)
    observed_high_floor = ground_truth_high
    if observed_high_floor is None and metar and metar.temp_f is not None:
        observed_high_floor = metar.temp_f

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
            "bias_wu_daily": cal.bias_wu_daily,
            "bias_wu_hourly": cal.bias_wu_hourly,
            "weight_nws": cal.weight_nws,
            "weight_wu_daily": cal.weight_wu_daily,
            "weight_wu_hourly": cal.weight_wu_hourly,
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

    if valid_minutes and todays_obs_rows:
        # Convert MetarObs rows to dicts for the adaptive engine
        obs_dicts = []
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

        # Fused forecast high for adaptive remaining-rise cap
        _fc_highs = [
            s.high_f for s in [nws_obs, wu_daily_obs, wu_hourly_obs]
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

    # Run temperature model
    model = compute_model(
        nws_high=nws_obs.high_f if nws_obs else None,
        wu_daily_high=wu_daily_obs.high_f if wu_daily_obs else None,
        wu_hourly_peak=wu_hourly_obs.high_f if wu_hourly_obs else None,
        hrrr_high=hrrr_obs.high_f if hrrr_obs else None,
        gfs_high=gfs_obs.high_f if gfs_obs else None,
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
        },
        adaptive=adaptive_result,
        latest_weather=_latest_wx,
    )

    if model is None:
        log.warning("signal: %s — model returned None (insufficient data)", city.city_slug)
        return []

    prob_new_high = model.prob_new_high
    city_state = classify_city_state(prob_new_high)

    if city_state == "resolved":
        log.info("signal: %s — resolved (prob_new_high=%.3f), skipping", city.city_slug, prob_new_high)
        return []

    # Use a single session for all DB writes (model snapshot + per-bucket reads/inserts)
    signals: list[BucketSignal] = []
    async with get_session() as sess:
        # Persist model snapshot
        await insert_model_snapshot(
            sess,
            event_id=event.id,
            mu=model.mu,
            sigma=model.sigma,
            probs_json=json.dumps(model.probs),
            inputs_json=json.dumps(model.inputs),
            forecast_quality=model.forecast_quality,
        )

        # Compute signal per bucket
        for i, bucket in enumerate(buckets):
            if i >= len(model.probs):
                continue

            model_prob = model.probs[i]

            # If METAR high already exceeds this bucket's ceiling, probability is 0
            # (the final daily high can only go up, never down)
            if ground_truth_high is not None and bucket.high_f is not None:
                if ground_truth_high >= bucket.high_f:
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
                    yes_bid=None,
                    yes_ask=None,
                    yes_mid=None,
                    spread=None,
                    yes_ask_depth=0.0,
                    gate_failures=["no_market_data"],
                    prob_new_high=prob_new_high,
                    city_state=city_state,
                    resolution_mismatch=resolution_mismatch,
                )
                signals.append(sig)
                continue

            mkt_prob = mkt_snap.yes_mid
            ask_depth = mkt_snap.yes_ask_depth or 0.0
            spread = mkt_snap.spread
            exec_cost = _execution_cost(spread, ask_depth)

            # Apply probability calibration (remap based on historical reliability)
            calibrated_prob = remap_probability(model_prob, reliability_bins)

            # Edge calculation based on calibrated probability
            raw_edge_buy = calibrated_prob - mkt_prob
            true_edge = raw_edge_buy - exec_cost

            reason = {
                **model.inputs,
                "bucket_idx": i,
                "label": bucket.label,
                "model_prob_raw": float(round(model_prob, 4)),
                "model_prob_cal": float(round(calibrated_prob, 4)),
                "mkt_prob": float(round(mkt_prob, 4)),
                "raw_edge": float(round(raw_edge_buy, 4)),
                "exec_cost": float(round(exec_cost, 4)),
                "true_edge": float(round(true_edge, 4)),
                "spread": spread,
                "ask_depth": ask_depth,
                "city_state": city_state,
                "resolution_high": resolution_high,
                "raw_high": daily_high,
                "observation_minutes": valid_minutes,
                "resolution_mismatch": resolution_mismatch,
            }

            actionable = (
                true_edge >= Config.MIN_TRUE_EDGE
                and 0.02 <= mkt_prob <= 0.98  # avoid extreme markets
                and ask_depth >= Config.MIN_LIQUIDITY_SHARES
                and event.forecast_quality == "ok"
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
                yes_bid=mkt_snap.yes_bid,
                yes_ask=mkt_snap.yes_ask,
                yes_mid=float(round(mkt_prob, 4)),
                spread=spread,
                yes_ask_depth=ask_depth,
                reason=reason,
                actionable=actionable,
                prob_new_high=prob_new_high,
                city_state=city_state,
                resolution_mismatch=resolution_mismatch,
            )
            signals.append(sig)

            # Persist signal to DB (reuse session)
            await insert_signal(
                sess,
                bucket_id=bucket.id,
                model_prob=sig.model_prob,
                mkt_prob=sig.mkt_prob,
                raw_edge=sig.raw_edge,
                exec_cost=sig.exec_cost,
                true_edge=sig.true_edge,
                reason_json=json.dumps(reason, default=str),
                gate_failures_json=json.dumps(sig.gate_failures),
            )

    return signals

