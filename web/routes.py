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
    )
    from backend.storage.models import Bucket, Event, City

    today_et = et_today()

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)
        arming = await get_arming_state(sess)
        daily_pnl = await get_daily_realized_pnl(sess, today_et)
        positions = await get_all_positions(sess)
        raw_signals = await get_latest_signals(sess, limit=200)

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
            "prob_new_high": reason.get("prob_new_high", 1.0),
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

    # Sort: timezone (east → west) then edge descending within each city
    deduped.sort(key=lambda r: (get_city_priority(r["city_slug"]), -r["true_edge"]))

    total_exposure = sum((p.net_qty * p.avg_cost) for p in positions if p.net_qty > 0)

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
        get_latest_model_snapshot,
        get_latest_signal_for_bucket,
        get_latest_market_snapshot,
        get_daily_high_metar,
        get_station_profile,
        get_resolution_high_metar,
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

    target_date_et = date if date else real_today_et

    async with get_session() as sess:
        # Fetch available event dates for this city
        from sqlalchemy import select, distinct
        from backend.storage.models import Event
        date_query = select(distinct(Event.date_et)).where(Event.city_id == city.id).order_by(Event.date_et.desc())
        available_dates = (await sess.execute(date_query)).scalars().all()

        metar = await get_latest_metar(sess, city.id)
        # For the selected date, we also want the official high observed by METAR
        obs_high_f = await get_daily_high_metar(sess, city.id, target_date_et)

        # Station profile for resolution-aware display
        station_profile = await get_station_profile(sess, city.metar_station) if city.metar_station else None
        resolution_high_f = None
        obs_minutes_list = None
        if station_profile and station_profile.observation_minutes:
            obs_minutes_list = json.loads(station_profile.observation_minutes)
            resolution_high_f = await get_resolution_high_metar(sess, city.id, target_date_et, obs_minutes_list)

        wu_d = await get_latest_forecast(sess, city.id, "wu_daily", target_date_et)
        wu_h = await get_latest_forecast(sess, city.id, "wu_hourly", target_date_et)
        wu_history = await get_latest_forecast(sess, city.id, "wu_history", target_date_et)
        
        primary_fc = None
        if city.is_us:
             primary_fc = await get_latest_forecast(sess, city.id, "nws", target_date_et)
        else:
             primary_fc = await get_latest_forecast(sess, city.id, "open_meteo", target_date_et)
        
        event = await get_event(sess, city.id, target_date_et)

    model = None
    buckets_with_signals = []

    if event:
        async with get_session() as sess:
            buckets = await get_buckets_for_event(sess, event.id)
            model = await get_latest_model_snapshot(sess, event.id)

        for bucket in buckets:
            async with get_session() as sess:
                sig = await get_latest_signal_for_bucket(sess, bucket.id)
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
                "label": bucket.label or f"Bucket {bucket.bucket_idx}",
                "low_f": bucket.low_f,
                "high_f": bucket.high_f,
                "model_prob": round(model_prob, 4) if model_prob is not None else None,
                "mkt_prob": mkt.yes_mid if mkt else None,
                "yes_bid": mkt.yes_bid if mkt else None,
                "yes_ask": mkt.yes_ask if mkt else None,
                "spread": mkt.spread if mkt else None,
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

    return templates.TemplateResponse(
        "city.html",
        {
            "request": request,
            "city": city,
            "today_et": target_date_et,
            "real_today_et": real_today_et,
            "available_dates": available_dates,
            "obs_high_f": obs_high_f,
            "resolution_high_f": resolution_high_f,
            "resolution_mismatch": round(obs_high_f - resolution_high_f, 1) if (obs_high_f is not None and resolution_high_f is not None and obs_high_f - resolution_high_f >= 1.0) else None,
            "observation_minutes": obs_minutes_list,
            "station_confidence": station_profile.confidence if station_profile else None,
            "station_frequency": station_profile.observation_frequency if station_profile else None,
            "resolution_source_url": event.resolution_source_url if event else None,
            "resolution_station_id": event.resolution_station_id if event else None,
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
                    "url": f"https://api.weather.gov/gridpoints/{city.nws_office}/{city.nws_grid_x},{city.nws_grid_y}/forecast" if city.is_us else f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}&longitude={city.lon}&hourly=temperature_2m&forecast_days=1"
                },
                "wu_daily": {
                    "high_f": max((v for v in [wu_d.high_f if wu_d else None, obs_high_f] if v is not None), default=None),
                    "age_s": _age(wu_d.fetched_at if wu_d else None),
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
                for b in (reliability_bins := await get_reliability_metrics(city.id))
            ]),
            "reliability_total_samples": sum(b.count for b in reliability_bins),
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
    from backend.storage.repos import get_latest_signals
    from backend.storage.models import Bucket, Event, City

    today_et = et_today()
    async with get_session() as sess:
        raw_signals = await get_latest_signals(sess, limit=200)

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

        slug = c.city_slug if c else ""
        rows.append({
            "city_slug": slug,
            "city_display": c.display_name if c else "",
            "unit": c.unit if c else "F",
            "bucket_idx": b.bucket_idx,
            "label": b.label or f"Bucket {b.bucket_idx}",
            "model_prob": sig.model_prob,
            "mkt_prob": sig.mkt_prob,
            "true_edge": sig.true_edge,
            "spread": reason.get("spread"),
            "ask_depth": reason.get("ask_depth"),
            "actionable": sig.true_edge >= 0.10 and not gate_failures,
            "gate_failures": gate_failures,
            "prob_new_high": reason.get("prob_new_high", 1.0),
            "city_state": reason.get("city_state", "early"),
            "resolution_high": reason.get("resolution_high"),
            "raw_high": reason.get("raw_high"),
            "observation_minutes": reason.get("observation_minutes"),
            "resolution_mismatch": reason.get("resolution_mismatch"),
        })

    rows.sort(key=lambda r: (get_city_priority(r["city_slug"]), -r["true_edge"]))
    return templates.TemplateResponse("partials/signals_table.html", {"request": request, "signal_rows": rows})
