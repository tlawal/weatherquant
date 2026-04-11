"""
Dashboard routes — serves Jinja2 HTMX templates.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import Config
from backend.city_registry import get_city_priority, CITY_REGISTRY_BY_SLUG
from backend.market_context.service import serialize_market_context_snapshot
from backend.tz_utils import city_local_date, city_local_now, city_local_tomorrow, et_today
from backend.strategy.kelly import calculate_expected_value, calculate_kelly_fraction
from backend.modeling.calibration_engine import get_reliability_metrics

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_HERE, "templates")

templates = Jinja2Templates(directory=_TEMPLATES_DIR)
dashboard_router = APIRouter()


@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    log.info("dashboard: GET / hit")
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_all_cities,
        get_arming_state,
        get_daily_realized_pnl,
        get_all_positions,
        get_latest_signals,
        get_signals_for_latest_snapshot,
        get_position,
        get_recently_redeemed_events,
        get_unredeemed_resolved_events,
    )
    from backend.storage.models import Bucket, Event, City

    today_et = et_today()

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)
        arming = await get_arming_state(sess)
        daily_pnl = await get_daily_realized_pnl(sess, today_et)
        positions = await get_all_positions(sess)
        # Filter to "rows from the latest model snapshot per event" so a
        # half-finished signal-engine pass can never leave stale signals
        # alongside the freshly-written ones for the same bucket.
        raw_signals = await get_signals_for_latest_snapshot(sess, limit=200)

    log.info("dashboard: data fetched: cities=%d, arming=%s, pnl=%.2f", len(cities), arming.state, daily_pnl)

    # Build signal rows for the table
    signal_rows = []
    for sig in raw_signals:
        async with get_session() as sess:
            from sqlalchemy import select
            bucket_row = await sess.get(Bucket, sig.bucket_id)
            if not bucket_row:
                continue
            event_row = await sess.get(Event, bucket_row.event_id)
            if not event_row or event_row.date_et != today_et:
                continue
            city_row = await sess.get(City, event_row.city_id)

        reason = json.loads(sig.reason_json) if sig.reason_json else {}
        gate_failures = json.loads(sig.gate_failures_json) if sig.gate_failures_json else []
        if reason.get("city_state") == "resolved":
            continue

        slug = city_row.city_slug if city_row else ""
        signal_rows.append({
            "city_slug": slug,
            "city_display": city_row.display_name if city_row else "",
            "unit": city_row.unit if city_row else "F",
            "bucket_idx": bucket_row.bucket_idx if bucket_row else 0,
            "label": bucket_row.label or f"Bucket {bucket_row.bucket_idx}",
            "low_f": bucket_row.low_f,
            "high_f": bucket_row.high_f,
            "model_prob": sig.model_prob,
            "mkt_prob": sig.mkt_prob,
            "true_edge": sig.true_edge,
            "exec_cost": sig.exec_cost,
            "spread": reason.get("spread"),
            "ask_depth": reason.get("ask_depth"),
            "actionable": sig.true_edge >= 0.10 and not gate_failures,
            "gate_failures": gate_failures,
            "prob_new_high": reason.get("prob_hotter_bucket", reason.get("prob_new_high", 1.0)),
            "prob_hotter_bucket": reason.get("prob_hotter_bucket", reason.get("prob_new_high", 1.0)),
            "prob_new_high_raw": reason.get("prob_new_high_raw"),
            "lock_regime": reason.get("lock_regime", False),
            "observed_bucket_idx": reason.get("observed_bucket_idx"),
            "observed_bucket_upper_f": reason.get("observed_bucket_upper_f"),
            "city_state": reason.get("city_state", "early"),
            "resolution_high": reason.get("resolution_high"),
            "raw_high": reason.get("raw_high"),
            "observation_minutes": reason.get("observation_minutes"),
            "resolution_mismatch": reason.get("resolution_mismatch"),
        })

    # Deduplicate — keep latest signal per (city, bucket_idx)
    seen = {}
    deduped = []
    for row in signal_rows:
        key = (row["city_slug"], row["bucket_idx"])
        if key not in seen:
            seen[key] = True
            deduped.append(row)

    # Sort: timezone (east → west) then market probability descending within each city
    deduped.sort(key=lambda r: (get_city_priority(r["city_slug"]), -(r.get("mkt_prob") if r.get("mkt_prob") is not None else -1.0)))

    total_exposure = sum((p.net_qty * p.avg_cost) for p in positions if p.net_qty > 0)

    # Unredeemed winning positions — only events where the user actually holds
    # a winning position (positive net_qty on the winning bucket).
    async with get_session() as sess:
        unredeemed_events = await get_unredeemed_resolved_events(sess, require_position=True)

    unredeemed_wins = []
    for evt in unredeemed_events:
        if evt.winning_bucket_idx is None:
            continue
        async with get_session() as sess:
            city = await sess.get(City, evt.city_id)
            total_payout = 0.0
            winning_label = None
            for bucket in evt.buckets:
                if bucket.bucket_idx != evt.winning_bucket_idx:
                    continue
                winning_label = bucket.label
                pos = await get_position(sess, bucket.id)
                if pos and pos.net_qty > 0:
                    total_payout += pos.net_qty * 1.0
                break
        if total_payout > 0:
            unredeemed_wins.append({
                "event_id": evt.id,
                "city_name": city.display_name if city else "?",
                "date_et": evt.date_et,
                "winning_label": winning_label or "N/A",
                "total_payout": round(total_payout, 2),
            })

    # Recently redeemed events (last 7 days) for history/retry
    async with get_session() as sess:
        redeemed_events = await get_recently_redeemed_events(sess, days=7)

    recent_redeems = []
    for evt in redeemed_events:
        async with get_session() as sess:
            city = await sess.get(City, evt.city_id)
            winning_label = None
            for bucket in evt.buckets:
                if (evt.winning_bucket_idx is not None
                        and bucket.bucket_idx == evt.winning_bucket_idx):
                    winning_label = bucket.label
                    break
        recent_redeems.append({
            "event_id": evt.id,
            "city_name": city.display_name if city else "?",
            "date_et": evt.date_et,
            "winning_label": winning_label or "N/A",
            "redeemed_at": evt.redeemed_at.strftime("%b %d %H:%M") if evt.redeemed_at else "",
        })

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "signal_rows": deduped,
            "arming_state": arming.state,
            "daily_pnl": round(daily_pnl, 2),
            "total_exposure": round(total_exposure, 2),
            "open_positions": len([p for p in positions if p.net_qty > 0]),
            "cities": [c.city_slug for c in cities],
            "today_et": today_et,
            "unredeemed_wins": unredeemed_wins,
            "recent_redeems": recent_redeems,
        },
    )


@dashboard_router.get("/city/{city_slug}", response_class=HTMLResponse)
async def city_detail(request: Request, city_slug: str, date: str | None = None):
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_city_by_slug,
        get_event,
        get_buckets_for_event,
        get_latest_metar,
        get_latest_forecast,
        get_latest_successful_forecast,
        get_latest_model_snapshot,
        get_latest_signal_for_bucket,
        get_latest_market_snapshot,
        get_daily_high_metar,
        get_market_context_snapshot,
        get_station_profile,
        get_resolution_high_metar,
        get_avg_peak_timing,
        get_todays_extended_obs,
    )

    async with get_session() as sess:
        city = await get_city_by_slug(sess, city_slug)
        if not city:
            return HTMLResponse("<h1>City not found</h1>", status_code=404)

    # Determine "today" in the city's local timezone
    now_local = city_local_now(city)
    now_et = datetime.now(ZoneInfo("America/New_York"))
    et_tz = ZoneInfo("America/New_York")
    real_today_et = city_local_date(city)

    # Roll over to tomorrow's market if it's past 8 PM local time
    active_date_et = real_today_et
    if now_local.hour >= 20:
        active_date_et = city_local_tomorrow(city)

    target_date_et = date if date else active_date_et

    async with get_session() as sess:
        # Fetch available event dates for this city
        from sqlalchemy import select, distinct
        from backend.storage.models import Event
        date_query = select(distinct(Event.date_et)).where(Event.city_id == city.id).order_by(Event.date_et.desc())
        available_dates = (await sess.execute(date_query)).scalars().all()

        metar = await get_latest_metar(sess, city.id)
        # For the selected date, we also want the official high observed by METAR
        obs_high_f = await get_daily_high_metar(sess, city.id, target_date_et, city_tz=getattr(city, "tz", "America/New_York"))
        avg_peak_timing = await get_avg_peak_timing(sess, city.id, days_back=3, et_tz=et_tz)

        # Station profile for resolution-aware display
        station_profile = await get_station_profile(sess, city.metar_station) if city.metar_station else None
        resolution_high_f = None
        obs_minutes_list = None
        if station_profile and station_profile.observation_minutes:
            obs_minutes_list = json.loads(station_profile.observation_minutes)
            resolution_high_f = await get_resolution_high_metar(sess, city.id, target_date_et, obs_minutes_list, city_tz=getattr(city, "tz", "America/New_York"))

        wu_d = await get_latest_successful_forecast(sess, city.id, "wu_daily", target_date_et)
        wu_h = await get_latest_successful_forecast(sess, city.id, "wu_hourly", target_date_et)
        wu_history = await get_latest_successful_forecast(sess, city.id, "wu_history", target_date_et)
        hrrr_fc = await get_latest_successful_forecast(sess, city.id, "hrrr", target_date_et)
        nbm_fc = await get_latest_successful_forecast(sess, city.id, "nbm", target_date_et)
        ecmwf_ifs_fc = await get_latest_successful_forecast(sess, city.id, "ecmwf_ifs", target_date_et)
        
        primary_fc = None
        if city.is_us:
             primary_fc = await get_latest_forecast(sess, city.id, "nws", target_date_et)
        else:
             primary_fc = await get_latest_forecast(sess, city.id, "open_meteo", target_date_et)
        
        event = await get_event(sess, city.id, target_date_et)
        market_context_snapshot = await get_market_context_snapshot(sess, city.id, target_date_et)

    model = None
    buckets_with_signals = []

    if event:
        async with get_session() as sess:
            buckets = await get_buckets_for_event(sess, event.id)
            model = await get_latest_model_snapshot(sess, event.id)
            model_inputs = json.loads(model.inputs_json) if model and model.inputs_json else {}

            snapshot_id = model.id if model is not None else None
            for bucket in buckets:
                sig = await get_latest_signal_for_bucket(sess, bucket.id, snapshot_id=snapshot_id)
                if model and sig and sig.computed_at < model.computed_at:
                    sig = None
                mkt = await get_latest_market_snapshot(sess, bucket.id)

                probs = json.loads(model.probs_json) if model and model.probs_json else []
                model_prob = probs[bucket.bucket_idx] if bucket.bucket_idx < len(probs) else None

                yes_price = mkt.yes_ask if mkt else None
                ev = calculate_expected_value(model_prob, yes_price) if model_prob is not None and yes_price else None
                kelly_f = calculate_kelly_fraction(
                    model_prob, 
                    yes_price, 
                    fractional_kelly=Config.KELLY_FRACTION,
                    max_position_size=Config.MAX_POSITION_PCT
                ) if model_prob is not None and yes_price else None

                buckets_with_signals.append({
                    "bucket_idx": bucket.bucket_idx,
                    "bucket_id": bucket.id,
                    "label": bucket.label or f"Bucket {bucket.bucket_idx}",
                    "low_f": bucket.low_f,
                    "high_f": bucket.high_f,
                    "yes_token_id": bucket.yes_token_id,
                    "model_prob": round(model_prob, 4) if model_prob is not None else None,
                    "mkt_prob": mkt.yes_mid if mkt else None,
                    "yes_bid": mkt.yes_bid if mkt else None,
                    "yes_ask": mkt.yes_ask if mkt else None,
                    "spread": mkt.spread if mkt else None,
                    "ask_depth": mkt.yes_ask_depth if mkt else None,
                    "bid_depth": mkt.yes_bid_depth if mkt else None,
                    "true_edge": sig.true_edge if sig else None,
                    "ev": ev,
                    "kelly_f": kelly_f,
                    "exec_cost": sig.exec_cost if sig else None,
                    "actionable": (sig.true_edge >= 0.10) if sig else False,
                })

    model_inputs = json.loads(model.inputs_json) if model and model.inputs_json else {}
    probs_json = json.dumps(json.loads(model.probs_json) if model and model.probs_json else [])

    def _age(dt):
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - dt).total_seconds(), 0)

    def _fmt_time_et(dt) -> str | None:
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(et_tz).strftime("%-I:%M %p ET")

    wu_history_raw = json.loads(wu_history.raw_json) if (wu_history and wu_history.raw_json) else {}
    wu_hourly_raw = json.loads(wu_h.raw_json) if (wu_h and wu_h.raw_json) else {}
    hrrr_raw = json.loads(hrrr_fc.raw_json) if (hrrr_fc and hrrr_fc.raw_json) else {}

    # Build HRRR hourly forecast data for chart overlay
    hrrr_hourly = []
    hrrr_hourly_data = hrrr_raw.get("hourly") or {}
    if hrrr_hourly_data.get("times") and hrrr_hourly_data.get("temps"):
        for t, temp in zip(hrrr_hourly_data["times"], hrrr_hourly_data["temps"]):
            if temp is not None:
                hrrr_hourly.append({"time": t, "temp_f": temp})

    reliability_bins = await get_reliability_metrics(city.id)

    # ── Adaptive predictions (obs table + station-time predictions) ─────────
    import math
    from backend.modeling.adaptive import run_adaptive

    city_tz_str = getattr(city, "tz", "America/New_York")
    obs_table = []
    station_predictions = []
    adaptive_info = None

    if target_date_et == real_today_et:
        async with get_session() as sess:
            ext_obs_rows = await get_todays_extended_obs(
                sess, city.id, target_date_et, city_tz=city_tz_str
            )

        city_tz_obj = ZoneInfo(city_tz_str)

        # Build obs_table for template (full day, newest first)
        for row in reversed(ext_obs_rows):
            _oa = row.observed_at
            if _oa.tzinfo is None:
                _oa = _oa.replace(tzinfo=timezone.utc)
            dt_local = _oa.astimezone(city_tz_obj)
            ext = row.extended
            is_station_min = False
            if obs_minutes_list:
                is_station_min = any(
                    abs(dt_local.minute - m) <= 1 or abs(dt_local.minute - m) >= 59
                    for m in obs_minutes_list
                )
            obs_table.append({
                "time": dt_local.strftime("%-I:%M %p"),
                "time_sort": dt_local.isoformat(),
                "temp_f": row.temp_f,
                "dewpoint_f": ext.dewpoint_f if ext else None,
                "humidity_pct": round(ext.humidity_pct) if ext and ext.humidity_pct else None,
                "wind_dir": ext.wind_dir_deg if ext else None,
                "wind_speed_kt": ext.wind_speed_kt if ext else None,
                "wind_gust_kt": ext.wind_gust_kt if ext else None,
                "altimeter_inhg": ext.altimeter_inhg if ext else None,
                "precip_in": ext.precip_in if ext else None,
                "condition": ext.condition if ext else None,
                "is_station_min": is_station_min,
                "is_daily_high": row.temp_f == obs_high_f if obs_high_f is not None else False,
            })

        # Deduplicate: multiple pollers can insert the same observation
        seen_times = set()
        deduped = []
        for row in obs_table:
            if row["time_sort"] not in seen_times:
                seen_times.add(row["time_sort"])
                deduped.append(row)
        obs_table = deduped

        # Run adaptive engine for station-time predictions
        if obs_minutes_list and ext_obs_rows and len(ext_obs_rows) >= 3:
            obs_dicts = []
            for row in ext_obs_rows:
                _oa = row.observed_at
                if _oa.tzinfo is None:
                    _oa = _oa.replace(tzinfo=timezone.utc)
                d = {"observed_at": _oa, "temp_f": row.temp_f}
                ext = row.extended
                if ext:
                    d["wind_speed_kt"] = ext.wind_speed_kt
                    d["humidity_pct"] = ext.humidity_pct
                    d["cloud_cover"] = ext.cloud_cover
                    cloud_map = {"CLR": 0, "SKC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}
                    d["cloud_cover_val"] = cloud_map.get((ext.cloud_cover or "").upper(), None)
                    d["wx_string"] = ext.wx_string
                    d["altimeter_inhg"] = ext.altimeter_inhg
                    d["precip_flag"] = bool(ext.wx_string and any(
                        tok in (ext.wx_string or "").upper() for tok in ("RA", "TS", "SH", "SN", "DZ")
                    ))
                obs_dicts.append(d)

            wu_peak_time = wu_hourly_raw.get("peak_hour")
            # Fused forecast high for adaptive remaining-rise cap
            _fc_vals = [
                s.high_f for s in [primary_fc, wu_d, wu_h]
                if s is not None and getattr(s, "high_f", None) is not None
            ]
            _adaptive_fc_high = sum(_fc_vals) / len(_fc_vals) if _fc_vals else None
            try:
                adaptive = run_adaptive(
                    todays_obs=obs_dicts,
                    observation_minutes=obs_minutes_list,
                    now_local=now_local,
                    city_tz=city_tz_str,
                    wu_hourly_peak_time=wu_peak_time,
                    historical_peak_mins=None,
                    forecast_high=_adaptive_fc_high,
                    ml_features={
                        "temp_slope_3h": 0.0,
                        "avg_peak_timing_mins": 960.0,
                        "day_of_year": now_local.timetuple().tm_yday,
                    },
                )
                if adaptive:
                    for sp in adaptive.station_predictions:
                        dt_local = sp.obs_time.astimezone(city_tz_obj) if sp.obs_time.tzinfo else sp.obs_time
                        trend_arrow = ""
                        if sp.trend_per_hour is not None:
                            if sp.trend_per_hour > 0.5: trend_arrow = "^"
                            elif sp.trend_per_hour > 0.1: trend_arrow = "7"
                            elif sp.trend_per_hour > -0.1: trend_arrow = ">"
                            elif sp.trend_per_hour > -0.5: trend_arrow = "\\"
                            else: trend_arrow = "v"
                            # Interpolate HRRR forecast temp at this station time
                        hrrr_fc_temp = None
                        if hrrr_hourly:
                            sp_hour = dt_local.hour + dt_local.minute / 60.0
                            for i, h in enumerate(hrrr_hourly):
                                h_dt = datetime.fromisoformat(h["time"])
                                h_hour = h_dt.hour + h_dt.minute / 60.0
                                if h_hour >= sp_hour:
                                    if i > 0:
                                        prev = hrrr_hourly[i - 1]
                                        p_dt = datetime.fromisoformat(prev["time"])
                                        p_hour = p_dt.hour + p_dt.minute / 60.0
                                        frac = (sp_hour - p_hour) / (h_hour - p_hour) if h_hour != p_hour else 0.5
                                        hrrr_fc_temp = round(prev["temp_f"] + frac * (h["temp_f"] - prev["temp_f"]), 1)
                                    else:
                                        hrrr_fc_temp = h["temp_f"]
                                    break

                        station_predictions.append({
                            "time": dt_local.strftime("%-I:%M %p"),
                            "actual_temp": sp.actual_temp,
                            "predicted_temp": sp.predicted_temp,
                            "uncertainty": sp.uncertainty,
                            "is_past": sp.is_past,
                            "trend_per_hour": sp.trend_per_hour,
                            "trend_arrow": trend_arrow,
                            "is_predicted_high": (sp.predicted_temp == adaptive.predicted_daily_high),
                            "hrrr_forecast_temp": hrrr_fc_temp,
                        })
                    # Compute weather sigma factor for display
                    _wx_sigma_factor = None
                    if obs_dicts:
                        _lw = obs_dicts[-1]
                        from backend.modeling.temperature_model import weather_adjusted_sigma
                        _wx_sigma_factor = round(weather_adjusted_sigma(
                            1.0,  # pass 1.0 to get the raw factor
                            cloud_cover_val=_lw.get("cloud_cover_val"),
                            humidity_pct=_lw.get("humidity_pct"),
                            wind_speed_kt=_lw.get("wind_speed_kt"),
                            wind_gust_kt=_lw.get("wind_gust_kt"),
                            has_precip=bool(_lw.get("precip_flag")),
                        ), 2)
                    adaptive_info = {
                        "kalman_temp": round(adaptive.kalman.smoothed_temp, 1),
                        "kalman_trend_per_hr": round(adaptive.kalman.temp_trend_per_min * 60, 2),
                        "kalman_n_obs": adaptive.kalman.n_observations,
                        "kalman_uncertainty": round(adaptive.kalman.uncertainty, 2),
                        "regression_slope_per_hr": round(adaptive.regression_slope * 60, 2),
                        "regression_r2": round(adaptive.regression_r2, 3),
                        "regression_features": adaptive.regression_features_used,
                        "predicted_daily_high": round(adaptive.predicted_daily_high, 1),
                        "predicted_high_time": adaptive.predicted_high_time.astimezone(city_tz_obj).strftime("%-I:%M %p") if adaptive.predicted_high_time else None,
                        "sigma_adjustment": round(adaptive.sigma_adjustment, 3),
                        "peak_already_passed": adaptive.peak_already_passed,
                        "composite_peak_timing": adaptive.composite_peak_timing,
                        "peak_timing_source": adaptive.peak_timing_source,
                        "remaining_rise_cap": round(adaptive.remaining_rise_cap, 1) if adaptive.remaining_rise_cap is not None else None,
                        "weather_sigma_factor": _wx_sigma_factor,
                        "diurnal_fit_active": adaptive.diurnal_fit_active,
                        "diurnal_fit_rmse": round(adaptive.diurnal_fit_rmse, 1) if adaptive.diurnal_fit_rmse is not None else None,
                        "diurnal_peak_estimate": round(adaptive.diurnal_peak_estimate, 1) if adaptive.diurnal_peak_estimate is not None else None,
                    }
            except Exception as e:
                log.warning("city_detail: adaptive engine failed for %s: %s", city_slug, e)

    return templates.TemplateResponse(
        "city.html",
        {
            "request": request,
            "city": city,
            "today_et": target_date_et,
            "real_today_et": real_today_et,
            "city_tomorrow": city_local_tomorrow(city) if city_local_now(city).hour >= 20 else None,
            "available_dates": available_dates,
            "obs_high_f": obs_high_f,
            "avg_peak_timing": avg_peak_timing,
            "resolution_high_f": resolution_high_f,
            "resolution_mismatch": round(obs_high_f - resolution_high_f, 1) if (obs_high_f is not None and resolution_high_f is not None and obs_high_f - resolution_high_f >= 1.0) else None,
            "observation_minutes": obs_minutes_list,
            "station_confidence": station_profile.confidence if station_profile else None,
            "station_frequency": station_profile.observation_frequency if station_profile else None,
            "resolution_source_url": event.resolution_source_url if event else None,
            "nws_timeseries_url": f"https://www.weather.gov/wrh/timeseries?site={city.metar_station.lower()}" if city.metar_station else None,
            "resolution_station_id": event.resolution_station_id if event else None,
            "active_station_id": model_inputs.get("active_station_id") if isinstance(model_inputs, dict) else None,
            "active_station_source": model_inputs.get("active_station_source") if isinstance(model_inputs, dict) else None,
            "ground_truth_high": model_inputs.get("ground_truth_high") if isinstance(model_inputs, dict) else None,
            "ground_truth_source": model_inputs.get("ground_truth_source") if isinstance(model_inputs, dict) else None,
            "settlement_source_verified": event.settlement_source_verified if event else None,
            "metar": {
                "temp_f": metar.temp_f if (metar and target_date_et == real_today_et) else None,
                "daily_high_f": obs_high_f,
                "observed_at": metar.observed_at if (metar and target_date_et == real_today_et) else None,
                "report_at": metar.report_at if (metar and target_date_et == real_today_et) else None,
                "station": metar.metar_station if metar else None,
                "raw_text": metar.raw_text if (metar and target_date_et == real_today_et) else None,
                "age_s": _age(metar.observed_at if (metar and target_date_et == real_today_et) else None),
                "source_url": (
                    f"https://aviationweather.gov/api/data/metar?ids={city.metar_station}&format=json&latest=1"
                    if city.metar_station
                    else f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}&longitude={city.lon}&current_weather=true"
                ) if city else None,
            },
            "forecasts": {
                "primary": {
                    "source": "nws" if city.is_us else "Open-Meteo",
                    "high_f": primary_fc.high_f if primary_fc else None,
                    "age_s": _age(primary_fc.fetched_at if primary_fc else None),
                    "collected_at": _fmt_time_et(primary_fc.fetched_at if primary_fc else None),
                    "url": f"https://api.weather.gov/gridpoints/{city.nws_office}/{city.nws_grid_x},{city.nws_grid_y}/forecast" if city.is_us else f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}&longitude={city.lon}&hourly=temperature_2m&forecast_days=1"
                },
                "wu_daily": {
                    "high_f": max((v for v in [wu_d.high_f if wu_d else None, obs_high_f] if v is not None), default=None),
                    "age_s": _age(wu_d.fetched_at if wu_d else None),
                    "collected_at": _fmt_time_et(wu_d.fetched_at if wu_d else None),
                    "url": f"https://www.wunderground.com/weather/{city.metar_station}" if city.metar_station else None
                },
                "wu_hourly": {
                    "high_f": wu_h.high_f if wu_h else None,
                    "age_s": _age(wu_h.fetched_at if wu_h else None),
                    "url": f"https://www.wunderground.com/hourly/{city.metar_station}/date/{target_date_et}" if city.metar_station else None,
                    "peak_hour": wu_hourly_raw.get("peak_hour"),
                    "collected_at": _fmt_time_et(wu_h.fetched_at if wu_h else None),
                },
                "wu_history": {
                    "high_f": wu_history.high_f if wu_history else None,
                    "age_s": _age(wu_history.fetched_at if wu_history else None),
                    "url": f"https://www.wunderground.com/history/daily/{city.metar_station}/date/{target_date_et}" if city.metar_station else None,
                    "obs_time": wu_history_raw.get("obs_time"),
                    "collected_at": _fmt_time_et(wu_history.fetched_at if wu_history else None),
                },
                "hrrr": {
                    "high_f": hrrr_fc.high_f if hrrr_fc else None,
                    "age_s": _age(hrrr_fc.fetched_at if hrrr_fc else None),
                    "collected_at": _fmt_time_et(hrrr_fc.fetched_at if hrrr_fc else None),
                    "url": f"https://open-meteo.com/en/docs?latitude={city.lat}&longitude={city.lon}&hourly=temperature_2m&models=gfs_hrrr&temperature_unit=fahrenheit&forecast_days=1" if city.lat else None,
                },
                "nbm": {
                    "high_f": nbm_fc.high_f if nbm_fc else None,
                    "age_s": _age(nbm_fc.fetched_at if nbm_fc else None),
                    "collected_at": _fmt_time_et(nbm_fc.fetched_at if nbm_fc else None),
                    "url": f"https://open-meteo.com/en/docs?latitude={city.lat}&longitude={city.lon}&hourly=temperature_2m&models=ncep_nbm_conus&temperature_unit=fahrenheit&forecast_days=1" if city.lat else None,
                },
                "ecmwf_ifs": {
                    "high_f": ecmwf_ifs_fc.high_f if ecmwf_ifs_fc else None,
                    "age_s": _age(ecmwf_ifs_fc.fetched_at if ecmwf_ifs_fc else None),
                    "collected_at": _fmt_time_et(ecmwf_ifs_fc.fetched_at if ecmwf_ifs_fc else None),
                    "url": (
                        f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}"
                        f"&longitude={city.lon}&hourly=temperature_2m&models=ecmwf_ifs"
                        f"&forecast_days=1&temperature_unit=fahrenheit"
                    ) if city.lat is not None else None,
                },
            },
            "event": event,
            "model": {
                "mu": model.mu if model else None,
                "sigma": model.sigma if model else None,
                "probs_json": probs_json,
                "inputs": model_inputs,
            } if model else None,
            "now_hour_et": now_local.hour,
            "city_tomorrow": city_local_tomorrow(city),
            "city_tz": getattr(city, "tz", "America/New_York"),
            "buckets": buckets_with_signals,
            "reliability_json": json.dumps([
                {"expected": b.expected_prob, "observed": b.observed_prob, "count": b.count}
                for b in reliability_bins
            ]),
            "reliability_total_samples": sum(b.count for b in reliability_bins),
            "obs_table": obs_table,
            "obs_table_json": json.dumps(obs_table),
            "station_predictions": station_predictions,
            "station_predictions_json": json.dumps(station_predictions),
            "hrrr_hourly_json": json.dumps(hrrr_hourly),
            "adaptive_info": adaptive_info,
            "market_context_snapshot": serialize_market_context_snapshot(market_context_snapshot),
            "market_context_llm_ready": Config.market_context_llm_ready(),
        },
    )


@dashboard_router.get("/admin/cities", response_class=HTMLResponse)
async def cities_admin(request: Request):
    from backend.storage.db import get_session
    from backend.storage.repos import get_all_cities, get_all_heartbeats

    async with get_session() as sess:
        cities = await get_all_cities(sess)
        heartbeats = await get_all_heartbeats(sess)

    hb_map = {hb.job_name: hb for hb in heartbeats}

    return templates.TemplateResponse(
        "cities_admin.html",
        {
            "request": request,
            "cities": cities,
            "heartbeats": hb_map,
        },
    )


@dashboard_router.get("/htmx/signals-table", response_class=HTMLResponse)
async def htmx_signals_table(request: Request):
    """HTMX partial — refreshes only the signals table body."""
    from backend.storage.db import get_session
    from backend.storage.repos import get_signals_for_latest_snapshot
    from backend.storage.models import Bucket, Event, City

    today_et = et_today()
    async with get_session() as sess:
        raw_signals = await get_signals_for_latest_snapshot(sess, limit=200)

    rows = []
    seen = {}
    for sig in raw_signals:
        async with get_session() as sess:
            b = await sess.get(Bucket, sig.bucket_id)
            if not b:
                continue
            ev = await sess.get(Event, b.event_id)
            if not ev or ev.date_et != today_et:
                continue
            c = await sess.get(City, ev.city_id)

        key = (c.city_slug if c else "", b.bucket_idx)
        if key in seen:
            continue
        seen[key] = True

        reason = json.loads(sig.reason_json) if sig.reason_json else {}
        gate_failures = json.loads(sig.gate_failures_json) if sig.gate_failures_json else []
        if reason.get("city_state") == "resolved":
            continue

        slug = c.city_slug if c else ""
        rows.append({
            "city_slug": slug,
            "city_display": c.display_name if c else "",
            "unit": c.unit if c else "F",
            "bucket_idx": b.bucket_idx,
            "label": b.label or f"Bucket {b.bucket_idx}",
            "low_f": b.low_f,
            "high_f": b.high_f,
            "model_prob": sig.model_prob,
            "mkt_prob": sig.mkt_prob,
            "true_edge": sig.true_edge,
            "spread": reason.get("spread"),
            "ask_depth": reason.get("ask_depth"),
            "actionable": sig.true_edge >= 0.10 and not gate_failures,
            "gate_failures": gate_failures,
            "prob_new_high": reason.get("prob_hotter_bucket", reason.get("prob_new_high", 1.0)),
            "prob_hotter_bucket": reason.get("prob_hotter_bucket", reason.get("prob_new_high", 1.0)),
            "prob_new_high_raw": reason.get("prob_new_high_raw"),
            "lock_regime": reason.get("lock_regime", False),
            "observed_bucket_idx": reason.get("observed_bucket_idx"),
            "observed_bucket_upper_f": reason.get("observed_bucket_upper_f"),
            "city_state": reason.get("city_state", "early"),
            "resolution_high": reason.get("resolution_high"),
            "raw_high": reason.get("raw_high"),
            "observation_minutes": reason.get("observation_minutes"),
            "resolution_mismatch": reason.get("resolution_mismatch"),
        })

    rows.sort(key=lambda r: (get_city_priority(r["city_slug"]), -(r.get("mkt_prob") if r.get("mkt_prob") is not None else -1.0)))
    return templates.TemplateResponse("partials/signals_table.html", {"request": request, "signal_rows": rows})


@dashboard_router.get("/redemptions", response_class=HTMLResponse)
async def redemptions_page(request: Request):
    """Full-page view of all events with positions and their redemption status."""
    return templates.TemplateResponse("redemptions.html", {"request": request})
