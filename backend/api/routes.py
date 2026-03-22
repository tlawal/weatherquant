"""
FastAPI REST routes — all read and write endpoints.

Read endpoints: no auth required.
Write endpoints: require X-Admin-Token header.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.deps import require_admin
from backend.config import Config
from backend.execution import arming as arming_mod
from backend.storage.db import get_session
from backend.storage.repos import (
    append_audit,
    get_all_cities,
    get_all_heartbeats,
    get_all_positions,
    get_arming_state,
    get_audit_log,
    get_buckets_for_event,
    get_calibration,
    get_daily_realized_pnl,
    get_event,
    get_latest_forecast,
    get_latest_market_snapshot,
    get_latest_metar,
    get_latest_model_snapshot,
    get_latest_signals,
    get_recent_orders,
)

log = logging.getLogger(__name__)

router = APIRouter()

_start_time = datetime.now(timezone.utc)


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    uptime_s = (datetime.now(timezone.utc) - _start_time).total_seconds()

    # DB may not be ready yet if startup is still running in the background
    try:
        async with get_session() as sess:
            heartbeats = await get_all_heartbeats(sess)
        db_ready = True
    except RuntimeError:
        # init_db() hasn't completed yet — return a minimal healthy response
        # so Railway's healthcheck passes while startup is in progress
        return {
            "status": "initializing",
            "uptime_seconds": round(uptime_s, 1),
            "version": "weatherquant-v1",
        }
    except Exception as e:
        log.warning("health: db check failed: %s", e)
        return {
            "status": "degraded",
            "uptime_seconds": round(uptime_s, 1),
            "version": "weatherquant-v1",
            "error": str(e),
        }

    worker_age = None
    for hb in heartbeats:
        if hb.job_name == "scheduler_alive":
            age = (datetime.now(timezone.utc) - hb.last_run_at).total_seconds()
            worker_age = round(age, 1)

    return {
        "status": "ok",
        "uptime_seconds": round(uptime_s, 1),
        "worker_heartbeat_age_s": worker_age,
        "version": "weatherquant-v1",
        "auto_trade_default": Config.AUTO_TRADE_DEFAULT,
        "bankroll_cap": Config.BANKROLL_CAP,
    }



# ─── Live Current Temperature (on-demand fetch) ───────────────────────────────

@router.get("/api/current-temp/{city_slug}")
async def get_current_temp(city_slug: str):
    """Fetch live current temperature for a city on demand.
    US cities: aviationweather.gov METAR API.
    International cities: Open-Meteo current_weather API.
    """
    import aiohttp
    from datetime import datetime, timezone

    async with get_session() as sess:
        from backend.storage.repos import get_city_by_slug
        city = await get_city_by_slug(sess, city_slug)

    if not city:
        raise HTTPException(status_code=404, detail="City not found")

    timeout = aiohttp.ClientTimeout(total=10)
    headers = {"User-Agent": "WeatherQuant/1.0"}

    try:
        if city.is_us and city.metar_station:
            url = f"https://aviationweather.gov/api/data/metar?ids={city.metar_station}&format=json&latest=1"
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as http:
                async with http.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)

            if not data:
                raise HTTPException(status_code=503, detail="No METAR data returned")

            obs = data[0]
            temp_c = obs.get("temp")
            if temp_c is None:
                raise HTTPException(status_code=503, detail="No temperature in METAR")

            temp_c = float(temp_c)
            temp_f = round(temp_c * 9 / 5 + 32, 1)

            # obsTime can be epoch int or ISO string
            obs_time_raw = obs.get("obsTime")
            report_time_raw = obs.get("reportTime")

            def _parse_time(raw):
                if raw is None:
                    return None
                try:
                    if isinstance(raw, (int, float)):
                        return datetime.fromtimestamp(int(raw), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    return datetime.fromisoformat(str(raw).rstrip("Z")).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    return str(raw)

            return {
                "temp_f": temp_f,
                "temp_c": round(temp_c, 1),
                "observed_at": _parse_time(obs_time_raw),
                "report_at": _parse_time(report_time_raw),
                "station": obs.get("stationId") or city.metar_station,
                "raw_text": obs.get("rawOb"),
                "source": "aviationweather.gov",
                "source_url": url,
                "unit": city.unit or "F",
            }

        elif not city.is_us and city.lat and city.lon:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}&longitude={city.lon}&current_weather=true"
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            cw = data.get("current_weather")
            if not cw:
                raise HTTPException(status_code=503, detail="No current_weather in Open-Meteo response")

            temp_c = float(cw.get("temperature", 0))
            temp_display = temp_c if (city.unit or "C") == "C" else round(temp_c * 9 / 5 + 32, 1)
            obs_time = cw.get("time", "")

            return {
                "temp_f": temp_display,
                "temp_c": round(temp_c, 1),
                "observed_at": obs_time,
                "report_at": obs_time,
                "station": city.metar_station or "OM",
                "raw_text": None,
                "source": "open-meteo.com",
                "source_url": url,
                "unit": city.unit or "C",
            }

        else:
            raise HTTPException(status_code=400, detail="City has no METAR station or coordinates")

    except HTTPException:
        raise
    except Exception as e:
        log.warning("current-temp: fetch failed for %s: %s", city_slug, e)
        raise HTTPException(status_code=503, detail=f"Fetch failed: {e}")


# ─── Cities ───────────────────────────────────────────────────────────────────

@router.get("/cities")
async def list_cities():
    async with get_session() as sess:
        cities = await get_all_cities(sess)
        heartbeats = await get_all_heartbeats(sess)

    hb_map = {hb.job_name: hb for hb in heartbeats}
    today_et = date.today().isoformat()

    result = []
    for city in cities:
        async with get_session() as sess:
            metar = await get_latest_metar(sess, city.id)
            nws = await get_latest_forecast(sess, city.id, "nws", today_et)
            wu_d = await get_latest_forecast(sess, city.id, "wu_daily", today_et)
            event = await get_event(sess, city.id, today_et)

        result.append({
            "city_slug": city.city_slug,
            "display_name": city.display_name,
            "metar_station": city.metar_station,
            "enabled": city.enabled,
            "metar_temp_f": metar.temp_f if metar else None,
            "metar_daily_high_f": metar.daily_high_f if metar else None,
            "metar_age_s": round(
                (datetime.now(timezone.utc) - metar.fetched_at).total_seconds(), 0
            ) if metar else None,
            "nws_high_f": nws.high_f if nws else None,
            "wu_daily_high_f": wu_d.high_f if wu_d else None,
            "event_status": event.status if event else None,
            "forecast_quality": event.forecast_quality if event else None,
            "trading_enabled": event.trading_enabled if event else False,
            "settlement_verified": event.settlement_source_verified if event else False,
        })

    return result


# ─── State per city ───────────────────────────────────────────────────────────

@router.get("/state/{city_slug}")
async def get_city_state(city_slug: str):
    today_et = date.today().isoformat()

    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        metar = await get_latest_metar(sess, city.id)
        nws = await get_latest_forecast(sess, city.id, "nws", today_et)
        wu_d = await get_latest_forecast(sess, city.id, "wu_daily", today_et)
        wu_h = await get_latest_forecast(sess, city.id, "wu_hourly", today_et)
        wu_hist = await get_latest_forecast(sess, city.id, "wu_history", today_et)
        event = await get_event(sess, city.id, today_et)
        model = None if not event else await get_latest_model_snapshot(sess, event.id)

    def _age_s(dt_):
        if not dt_:
            return None
        return round((datetime.now(timezone.utc) - dt_).total_seconds(), 0)

    # Use WU History as ground truth if available; fallback to METAR daily high
    daily_high = wu_hist.high_f if wu_hist and wu_hist.high_f is not None else (metar.daily_high_f if metar else None)

    return {
        "city_slug": city_slug,
        "display_name": city.display_name,
        "metar_station": city.metar_station,
        "current_temp_f": metar.temp_f if metar else None,
        "daily_high_f": daily_high,
        "metar_observed_at": metar.observed_at.isoformat() if metar else None,
        "metar_age_s": _age_s(metar.fetched_at if metar else None),
        "forecasts": {
            "nws": {
                "high_f": nws.high_f if nws else None,
                "age_s": _age_s(nws.fetched_at if nws else None),
                "error": nws.parse_error if nws else None,
            },
            "wu_daily": {
                "high_f": wu_d.high_f if wu_d else None,
                "age_s": _age_s(wu_d.fetched_at if wu_d else None),
                "error": wu_d.parse_error if wu_d else None,
            },
            "wu_hourly": {
                "high_f": wu_h.high_f if wu_h else None,
                "age_s": _age_s(wu_h.fetched_at if wu_h else None),
                "error": wu_h.parse_error if wu_h else None,
            },
        },
        "forecast_quality": event.forecast_quality if event else None,
        "wu_scrape_error": event.wu_scrape_error if event else None,
        "model": {
            "mu": model.mu if model else None,
            "sigma": model.sigma if model else None,
            "probs": json.loads(model.probs_json) if model and model.probs_json else None,
            "inputs": json.loads(model.inputs_json) if model and model.inputs_json else None,
            "computed_at": model.computed_at.isoformat() if model else None,
        } if model else None,
        "event_status": event.status if event else "no_event",
        "settlement_source": event.settlement_source if event else None,
        "settlement_verified": event.settlement_source_verified if event else False,
        "trading_enabled": event.trading_enabled if event else False,
    }


# ─── Markets ─────────────────────────────────────────────────────────────────

@router.get("/markets/{city_slug}")
async def get_markets(city_slug: str):
    today_et = date.today().isoformat()
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        event = await get_event(sess, city.id, today_et)
        if not event:
            return {"event": None, "buckets": []}

        buckets = await get_buckets_for_event(sess, event.id)

    result = []
    for b in buckets:
        async with get_session() as sess:
            snap = await get_latest_market_snapshot(sess, b.id)
        result.append({
            "bucket_id": b.id,
            "bucket_idx": b.bucket_idx,
            "label": b.label,
            "low_f": b.low_f,
            "high_f": b.high_f,
            "yes_token_id": b.yes_token_id,
            "yes_bid": snap.yes_bid if snap else None,
            "yes_ask": snap.yes_ask if snap else None,
            "yes_mid": snap.yes_mid if snap else None,
            "spread": snap.spread if snap else None,
            "ask_depth": snap.yes_ask_depth if snap else None,
            "fetched_at": snap.fetched_at.isoformat() if snap else None,
        })

    return {
        "event": {
            "id": event.id,
            "slug": event.gamma_slug,
            "status": event.status,
            "settlement_source": event.settlement_source,
            "settlement_verified": event.settlement_source_verified,
            "trading_enabled": event.trading_enabled,
            "forecast_quality": event.forecast_quality,
        },
        "buckets": result,
    }


# ─── Signals ────────────────────────────────────────────────────────────────

@router.get("/signals")
async def get_all_signals():
    """Return latest signals across all cities, ranked by true_edge."""
    async with get_session() as sess:
        raw_signals = await get_latest_signals(sess, limit=200)

    out = []
    for sig in raw_signals:
        async with get_session() as sess:
            from backend.storage.models import Bucket, Event, City
            from sqlalchemy import select
            bucket_row = await sess.get(Bucket, sig.bucket_id)
            if not bucket_row:
                continue
            event_row = await sess.get(Event, bucket_row.event_id)
            city_row = await sess.get(City, event_row.city_id) if event_row else None

        out.append({
            "signal_id": sig.id,
            "city_slug": city_row.city_slug if city_row else None,
            "city_display": city_row.display_name if city_row else None,
            "bucket_id": sig.bucket_id,
            "bucket_idx": bucket_row.bucket_idx if bucket_row else None,
            "label": bucket_row.label if bucket_row else None,
            "low_f": bucket_row.low_f if bucket_row else None,
            "high_f": bucket_row.high_f if bucket_row else None,
            "model_prob": sig.model_prob,
            "mkt_prob": sig.mkt_prob,
            "raw_edge": sig.raw_edge,
            "exec_cost": sig.exec_cost,
            "true_edge": sig.true_edge,
            "reason": json.loads(sig.reason_json) if sig.reason_json else {},
            "computed_at": sig.computed_at.isoformat(),
        })

    out.sort(key=lambda x: x["true_edge"], reverse=True)
    return out


@router.get("/signals/{city_slug}")
async def get_city_signals(city_slug: str):
    today_et = date.today().isoformat()
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        event = await get_event(sess, city.id, today_et)
        if not event:
            return []
        buckets = await get_buckets_for_event(sess, event.id)

    result = []
    for b in buckets:
        async with get_session() as sess:
            from backend.storage.repos import get_latest_signal_for_bucket
            sig = await get_latest_signal_for_bucket(sess, b.id)
        if sig:
            result.append({
                "bucket_idx": b.bucket_idx,
                "label": b.label,
                "low_f": b.low_f,
                "high_f": b.high_f,
                "model_prob": sig.model_prob,
                "mkt_prob": sig.mkt_prob,
                "true_edge": sig.true_edge,
                "exec_cost": sig.exec_cost,
                "gate_failures": json.loads(sig.gate_failures_json) if sig.gate_failures_json else [],
                "reason": json.loads(sig.reason_json) if sig.reason_json else {},
                "computed_at": sig.computed_at.isoformat(),
            })

    result.sort(key=lambda x: x["true_edge"], reverse=True)
    return result


# ─── Positions ────────────────────────────────────────────────────────────────

@router.get("/positions")
async def get_positions():
    today_et = date.today().isoformat()
    async with get_session() as sess:
        positions = await get_all_positions(sess)
        daily_pnl = await get_daily_realized_pnl(sess, today_et)

    result = []
    for p in positions:
        async with get_session() as sess:
            from backend.storage.models import Bucket
            b = await sess.get(Bucket, p.bucket_id)
        result.append({
            "bucket_id": p.bucket_id,
            "label": b.label if b else None,
            "side": p.side,
            "net_qty": p.net_qty,
            "avg_cost": p.avg_cost,
            "last_mkt_price": p.last_mkt_price,
            "unrealized_pnl": p.unrealized_pnl,
            "realized_pnl": p.realized_pnl,
            "updated_at": p.updated_at.isoformat(),
        })

    return {
        "positions": result,
        "daily_realized_pnl": round(daily_pnl, 4),
        "total_unrealized_pnl": round(sum(p.unrealized_pnl for p in positions), 4),
    }


# ─── Orders ───────────────────────────────────────────────────────────────────

@router.get("/orders")
async def get_orders(limit: int = 50):
    async with get_session() as sess:
        orders = await get_recent_orders(sess, limit=limit)
    return [
        {
            "id": o.id,
            "bucket_id": o.bucket_id,
            "side": o.side,
            "qty": o.qty,
            "limit_price": o.limit_price,
            "status": o.status,
            "fill_price": o.fill_price,
            "fill_qty": o.fill_qty,
            "clob_order_id": o.clob_order_id,
            "created_at": o.created_at.isoformat(),
        }
        for o in orders
    ]


# ─── Audit Log ────────────────────────────────────────────────────────────────

@router.get("/audit")
async def get_audit(limit: int = 100, action: Optional[str] = None):
    async with get_session() as sess:
        entries = await get_audit_log(sess, limit=limit, action_filter=action)
    return [
        {
            "id": e.id,
            "ts": e.ts.isoformat(),
            "actor": e.actor,
            "action": e.action,
            "ok": e.ok,
            "error_msg": e.error_msg,
            "payload": json.loads(e.payload_json) if e.payload_json else None,
        }
        for e in entries
    ]


# ─── Calibration ─────────────────────────────────────────────────────────────

@router.get("/calibration/{city_slug}")
async def get_city_calibration(city_slug: str):
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        cal = await get_calibration(sess, city.id)
    if not cal:
        return {"city_slug": city_slug, "calibration": None}
    return {
        "city_slug": city_slug,
        "bias_nws": cal.bias_nws,
        "bias_wu_daily": cal.bias_wu_daily,
        "bias_wu_hourly": cal.bias_wu_hourly,
        "weight_nws": cal.weight_nws,
        "weight_wu_daily": cal.weight_wu_daily,
        "weight_wu_hourly": cal.weight_wu_hourly,
        "n_samples": cal.n_samples,
        "last_realized_high": cal.last_realized_high,
        "updated_at": cal.updated_at.isoformat() if cal.updated_at else None,
    }


# ─── Arming State ─────────────────────────────────────────────────────────────

@router.get("/arming")
async def get_arming():
    return await arming_mod.get_state()


class ArmingConfirmRequest(BaseModel):
    token: str
    secret: str


@router.post("/arming/request")
async def request_arming(request: Request, actor: str = Depends(require_admin)):
    ip = request.client.host if request.client else "unknown"
    ok, message, token = await arming_mod.request_arming(actor=f"{actor}@{ip}")
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message, "token": token}


@router.post("/arming/confirm")
async def confirm_arming(
    body: ArmingConfirmRequest,
    request: Request,
    actor: str = Depends(require_admin),
):
    ip = request.client.host if request.client else "unknown"
    ok, message = await arming_mod.confirm_arming(
        token=body.token, secret=body.secret, actor=f"{actor}@{ip}"
    )
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message, "state": "ARMED"}


@router.post("/arming/disable")
async def disable_arming(request: Request, actor: str = Depends(require_admin)):
    ip = request.client.host if request.client else "unknown"

    # Cancel all open CLOB orders on kill switch
    from backend.ingestion.polymarket_clob import get_clob
    from backend.storage.repos import get_open_orders, update_order_status
    async with get_session() as sess:
        open_orders = await get_open_orders(sess)

    clob = get_clob()
    cancelled = 0
    for order in open_orders:
        if order.clob_order_id and clob:
            await clob.cancel_order(order.clob_order_id)
        async with get_session() as sess:
            await update_order_status(sess, order.id, "cancelled", cancel_reason="kill_switch")
        cancelled += 1

    ok, message = await arming_mod.disarm(actor=f"{actor}@{ip}", reason="manual_kill_switch")
    return {"ok": True, "message": message, "orders_cancelled": cancelled}


# ─── Manual Trade ─────────────────────────────────────────────────────────────

class ManualTradeRequest(BaseModel):
    city_slug: str
    bucket_idx: int
    side: str  # "buy_yes" | "buy_no"
    qty: Optional[float] = None  # None = auto-size
    limit_price: Optional[float] = None
    dry_run: bool = False


@router.post("/trade")
async def manual_trade(
    body: ManualTradeRequest,
    actor: str = Depends(require_admin),
):
    """Manual trade endpoint — builds a BucketSignal from current state and executes."""
    from backend.engine.signal_engine import BucketSignal
    from backend.execution.trader import execute_signal
    from backend.ingestion.polymarket_clob import get_clob

    today_et = date.today().isoformat()

    async with get_session() as sess:
        city = await _get_city_or_404(sess, body.city_slug)
        event = await get_event(sess, city.id, today_et)
        if not event:
            raise HTTPException(status_code=404, detail="No event today for this city")

        buckets = await get_buckets_for_event(sess, event.id)

    bucket = next((b for b in buckets if b.bucket_idx == body.bucket_idx), None)
    if not bucket:
        raise HTTPException(status_code=404, detail=f"Bucket {body.bucket_idx} not found")

    async with get_session() as sess:
        from backend.storage.repos import get_latest_signal_for_bucket, get_latest_market_snapshot
        sig_row = await get_latest_signal_for_bucket(sess, bucket.id)
        mkt_snap = await get_latest_market_snapshot(sess, bucket.id)

    model_prob = sig_row.model_prob if sig_row else 0.5
    mkt_prob = mkt_snap.yes_mid if mkt_snap else 0.5
    yes_ask = body.limit_price or (mkt_snap.yes_ask if mkt_snap else None) or 0.5
    true_edge = sig_row.true_edge if sig_row else 0.0

    signal = BucketSignal(
        city_slug=body.city_slug,
        city_display=city.display_name,
        event_id=event.id,
        bucket_id=bucket.id,
        bucket_idx=body.bucket_idx,
        label=bucket.label or f"Bucket {body.bucket_idx}",
        low_f=bucket.low_f,
        high_f=bucket.high_f,
        model_prob=model_prob,
        mkt_prob=mkt_prob,
        raw_edge=model_prob - mkt_prob,
        exec_cost=sig_row.exec_cost if sig_row else 0.02,
        true_edge=true_edge,
        yes_bid=mkt_snap.yes_bid if mkt_snap else None,
        yes_ask=yes_ask,
        yes_mid=mkt_prob,
        spread=mkt_snap.spread if mkt_snap else None,
        yes_ask_depth=mkt_snap.yes_ask_depth if mkt_snap else 0.0,
        actionable=True,  # manual override
    )

    # For manual trades, bypass the edge gate but keep all safety gates
    clob = get_clob()
    bankroll = Config.BANKROLL_CAP
    if clob and clob.can_trade:
        balance = await clob.get_balance()
        if balance:
            bankroll = min(balance, Config.BANKROLL_CAP)

    result = await execute_signal(signal, bankroll=bankroll, actor=actor, dry_run=body.dry_run)
    return result


# ─── Config ──────────────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    min_true_edge: Optional[float] = None
    max_daily_loss: Optional[float] = None
    bankroll_cap: Optional[float] = None


@router.post("/config")
async def update_config(body: ConfigUpdate, actor: str = Depends(require_admin)):
    """Update runtime risk thresholds."""
    updates = {}
    if body.min_true_edge is not None:
        if not (0.0 <= body.min_true_edge <= 0.5):
            raise HTTPException(status_code=400, detail="min_true_edge must be in [0, 0.5]")
        Config.MIN_TRUE_EDGE = body.min_true_edge
        updates["min_true_edge"] = body.min_true_edge

    if body.max_daily_loss is not None:
        if not (0.1 <= body.max_daily_loss <= 10.0):
            raise HTTPException(status_code=400, detail="max_daily_loss must be in [0.1, 10]")
        Config.MAX_DAILY_LOSS = body.max_daily_loss
        updates["max_daily_loss"] = body.max_daily_loss

    if body.bankroll_cap is not None:
        if body.bankroll_cap > 100:
            raise HTTPException(status_code=400, detail="bankroll_cap cannot exceed $100 in v1")
        Config.BANKROLL_CAP = body.bankroll_cap
        updates["bankroll_cap"] = body.bankroll_cap

    async with get_session() as sess:
        await append_audit(sess, actor=actor, action="config_updated", payload=updates)

    return {"ok": True, "updated": updates}


@router.post("/cities/{city_slug}/toggle")
async def toggle_city(city_slug: str, actor: str = Depends(require_admin)):
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        city.enabled = not city.enabled
        await sess.commit()
        enabled = city.enabled
        await append_audit(
            sess, actor=actor, action="city_toggled",
            payload={"city_slug": city_slug, "enabled": enabled}
        )
    return {"city_slug": city_slug, "enabled": enabled}


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _get_city_or_404(sess, city_slug: str):
    from backend.storage.repos import get_city_by_slug
    city = await get_city_by_slug(sess, city_slug)
    if not city:
        raise HTTPException(status_code=404, detail=f"City not found: {city_slug}")
    return city
