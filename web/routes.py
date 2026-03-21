"""
Dashboard routes — serves Jinja2 HTMX templates.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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

    today_et = date.today().isoformat()

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

        signal_rows.append({
            "city_slug": city_row.city_slug if city_row else "",
            "city_display": city_row.display_name if city_row else "",
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
        })

    # Deduplicate — keep latest signal per (city, bucket_idx)
    seen = {}
    deduped = []
    for row in signal_rows:
        key = (row["city_slug"], row["bucket_idx"])
        if key not in seen:
            seen[key] = True
            deduped.append(row)

    deduped.sort(key=lambda r: r["true_edge"], reverse=True)

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
async def city_detail(request: Request, city_slug: str):
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
    )

    today_et = date.today().isoformat()

    async with get_session() as sess:
        city = await get_city_by_slug(sess, city_slug)
        if not city:
            return HTMLResponse("<h1>City not found</h1>", status_code=404)

        metar = await get_latest_metar(sess, city.id)
        nws = await get_latest_forecast(sess, city.id, "nws", today_et)
        wu_d = await get_latest_forecast(sess, city.id, "wu_daily", today_et)
        wu_h = await get_latest_forecast(sess, city.id, "wu_hourly", today_et)
        wu_history = await get_latest_forecast(sess, city.id, "wu_history", today_et)
        event = await get_event(sess, city.id, today_et)

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

    return templates.TemplateResponse(
        "city.html",
        {
            "request": request,
            "city": city,
            "today_et": today_et,
            "metar": {
                "temp_f": metar.temp_f if metar else None,
                "daily_high_f": metar.daily_high_f if metar else None,
                "observed_at": metar.observed_at.isoformat() if metar else None,
                "age_s": _age(metar.fetched_at if metar else None),
            },
            "daily_high_f": wu_history.high_f if wu_history else None,
            "forecasts": {
                "nws": {"high_f": nws.high_f if nws else None, "age_s": _age(nws.fetched_at if nws else None)},
                "wu_daily": {"high_f": wu_d.high_f if wu_d else None, "age_s": _age(wu_d.fetched_at if wu_d else None)},
                "wu_hourly": {"high_f": wu_h.high_f if wu_h else None, "age_s": _age(wu_h.fetched_at if wu_h else None)},
            },
            "event": event,
            "model": {
                "mu": model.mu if model else None,
                "sigma": model.sigma if model else None,
                "probs_json": probs_json,
                "inputs": model_inputs,
            } if model else None,
            "buckets": buckets_with_signals,
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

    today_et = date.today().isoformat()
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

        rows.append({
            "city_slug": c.city_slug if c else "",
            "city_display": c.display_name if c else "",
            "bucket_idx": b.bucket_idx,
            "label": b.label or f"Bucket {b.bucket_idx}",
            "model_prob": sig.model_prob,
            "mkt_prob": sig.mkt_prob,
            "true_edge": sig.true_edge,
            "spread": reason.get("spread"),
            "ask_depth": reason.get("ask_depth"),
            "actionable": sig.true_edge >= 0.10 and not gate_failures,
            "gate_failures": gate_failures,
        })

    rows.sort(key=lambda r: r["true_edge"], reverse=True)
    return templates.TemplateResponse("partials/signals_table.html", {"request": request, "signal_rows": rows})
