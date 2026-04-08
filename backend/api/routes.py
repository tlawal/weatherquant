"""
FastAPI REST routes — all read and write endpoints.

Read endpoints: no auth required.
Write endpoints: require X-Admin-Token header.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.deps import require_admin
from backend.city_registry import CITY_REGISTRY_BY_SLUG
from backend.tz_utils import city_local_date, et_today
from backend.config import Config
from backend.engine.signal_engine import classify_city_state
from backend.execution import arming as arming_mod
from backend.market_context.adapter import MarketContextLLMError
from backend.market_context.service import (
    MarketContextBuildError,
    get_market_context_snapshot_payload,
    refresh_market_context_snapshot,
)
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
    get_position,
    get_recent_orders,
    get_recently_redeemed_events,
    get_unredeemed_resolved_events,
    get_unresolved_events_with_positions,
)

log = logging.getLogger(__name__)

router = APIRouter()

_start_time = datetime.now(timezone.utc)


def _age_seconds(dt_: Optional[datetime]) -> Optional[float]:
    """Return rounded age in seconds, treating naive DB timestamps as UTC."""
    if not dt_:
        return None
    if dt_.tzinfo is None:
        dt_ = dt_.replace(tzinfo=timezone.utc)
    else:
        dt_ = dt_.astimezone(timezone.utc)
    return round((datetime.now(timezone.utc) - dt_).total_seconds(), 0)


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
            worker_age = _age_seconds(hb.last_run_at)

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
        # 1. Primary Attempt for all metar cities: api.weather.gov
        if city.metar_station:
            try:
                nws_url = f"https://api.weather.gov/stations/{city.metar_station}/observations/latest"
                async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "WeatherQuant/1.0", "Accept": "application/geo+json"}) as http:
                    async with http.get(nws_url) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            props = data.get("properties", {})
                            temp_c_val = (props.get("temperature") or {}).get("value")
                            if temp_c_val is not None:
                                temp_c = float(temp_c_val)
                                temp_f = temp_c if (city.unit or "C") == "C" else round(temp_c * 9 / 5 + 32, 1)
                                if city.unit != "C": temp_f = round(temp_c * 9 / 5 + 32, 1)

                                ts_str = props.get("timestamp")
                                if ts_str:
                                    from zoneinfo import ZoneInfo
                                    dt_et = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York"))
                                    obs_time = dt_et.strftime("%-I:%M %p ET")
                                else:
                                    obs_time = ""
                                return {
                                    "temp_f": temp_f,
                                    "temp_c": round(temp_c, 1),
                                    "observed_at": obs_time,
                                    "report_at": obs_time,
                                    "station": city.metar_station,
                                    "raw_text": props.get("rawMessage"),
                                    "source": "api.weather.gov",
                                    "source_url": nws_url,
                                    "unit": city.unit or ("F" if city.is_us else "C"),
                                }
            except Exception as nws_err:
                log.info("NWS latest obs failed for %s, falling back: %s", city_slug, nws_err)

        # 2. Secondary Attempt: aviationweather (US) or Open-Meteo (Intl)
        if city.is_us and city.metar_station:
            try:
                url = f"https://aviationweather.gov/api/data/metar?ids={city.metar_station}&format=json&latest=1"
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as http:
                    async with http.get(url) as resp:
                        resp.raise_for_status()
                        data = await resp.json(content_type=None)

                if not data:
                    raise ValueError("No METAR data returned")

                obs = data[0]
                temp_c = obs.get("temp")
                if temp_c is None:
                    raise ValueError("No temperature in METAR")

                temp_c = float(temp_c)
                temp_f = round(temp_c * 9 / 5 + 32, 1)

                obs_time_raw = obs.get("obsTime")
                report_time_raw = obs.get("reportTime")

                def _parse_time(raw):
                    if raw is None:
                        return None
                    try:
                        from zoneinfo import ZoneInfo
                        et = ZoneInfo("America/New_York")
                        if isinstance(raw, (int, float)):
                            dt = datetime.fromtimestamp(int(raw), tz=timezone.utc).astimezone(et)
                        else:
                            dt = datetime.fromisoformat(str(raw).rstrip("Z")).replace(tzinfo=timezone.utc).astimezone(et)
                        return dt.strftime("%-I:%M %p ET")
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
            except Exception as metar_err:
                log.error("METAR fetch failed for US city %s: %s", city_slug, metar_err)
                # US cities: METAR is the ONLY source. No fallback to Open-Meteo/OWM
                # which can be 1-2°F off — enough to cause bracket misclassification.
                return {
                    "temp_f": None,
                    "temp_c": None,
                    "observed_at": None,
                    "report_at": None,
                    "station": city.metar_station,
                    "raw_text": f"METAR unavailable: {metar_err}",
                    "source": "aviationweather.gov (retrying)",
                    "source_url": url,
                    "unit": "F",
                }

        if not city.is_us and city.lat and city.lon:
            try:
                url = f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}&longitude={city.lon}&current_weather=true"
                async with aiohttp.ClientSession(timeout=timeout) as http:
                    async with http.get(url) as resp:
                        if resp.status == 429:
                            raise ValueError("Open-Meteo Rate Limited (429)")
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
                    "unit": city.unit or ("F" if city.is_us else "C"),
                }
            except Exception as om_err:
                log.warning("Open-Meteo fetch failed for %s, falling back to OpenWeatherMap: %s", city_slug, om_err)

                owm_url = f"https://api.openweathermap.org/data/2.5/weather?lat={city.lat}&lon={city.lon}&appid=de79374f3007b36700415b6679d810b1&units=metric"
                async with aiohttp.ClientSession(timeout=timeout) as http:
                    async with http.get(owm_url) as resp:
                        if resp.status == 429:
                            return {
                                "temp_f": None,
                                "temp_c": None,
                                "observed_at": None,
                                "report_at": None,
                                "station": "Rate Limited",
                                "raw_text": "Both Open-Meteo and OpenWeather APIs Rate Limited (HTTP 429)",
                                "source": "API Fallbacks",
                                "source_url": owm_url,
                                "unit": city.unit or ("F" if city.is_us else "C"),
                            }
                        resp.raise_for_status()
                        data = await resp.json()

                temp_c = float(data.get("main", {}).get("temp", 0))
                temp_display = temp_c if (city.unit or "C") == "C" else round(temp_c * 9 / 5 + 32, 1)
                
                obs_time_unix = data.get("dt")
                if obs_time_unix:
                    obs_time = datetime.fromtimestamp(obs_time_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                else:
                    obs_time = ""

                return {
                    "temp_f": temp_display,
                    "temp_c": round(temp_c, 1),
                    "observed_at": obs_time,
                    "report_at": obs_time,
                    "station": data.get("name", "OWM"),
                    "raw_text": None,
                    "source": "openweathermap.org (fallback)",
                    "source_url": owm_url,
                    "unit": city.unit or ("F" if city.is_us else "C"),
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

    result = []
    for city in cities:
        today_city = city_local_date(city)
        async with get_session() as sess:
            metar = await get_latest_metar(sess, city.id)
            nws = await get_latest_forecast(sess, city.id, "nws", today_city)
            wu_d = await get_latest_forecast(sess, city.id, "wu_daily", today_city)
            event = await get_event(sess, city.id, today_city)

        result.append({
            "city_slug": city.city_slug,
            "display_name": city.display_name,
            "metar_station": city.metar_station,
            "enabled": city.enabled,
            "metar_temp_f": metar.temp_f if metar else None,
            "metar_daily_high_f": metar.daily_high_f if metar else None,
            "metar_age_s": _age_seconds(metar.fetched_at) if metar else None,
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
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        today_city = city_local_date(city)
        metar = await get_latest_metar(sess, city.id)
        nws = await get_latest_forecast(sess, city.id, "nws", today_city)
        wu_d = await get_latest_forecast(sess, city.id, "wu_daily", today_city)
        wu_h = await get_latest_forecast(sess, city.id, "wu_hourly", today_city)
        wu_hist = await get_latest_forecast(sess, city.id, "wu_history", today_city)
        event = await get_event(sess, city.id, today_city)
        model = None if not event else await get_latest_model_snapshot(sess, event.id)

    # Use WU History as ground truth if available; fallback to METAR daily high
    daily_high = wu_hist.high_f if wu_hist and wu_hist.high_f is not None else (metar.daily_high_f if metar else None)

    model_inputs = json.loads(model.inputs_json) if model and model.inputs_json else {}
    reg = CITY_REGISTRY_BY_SLUG.get(city_slug, {})
    prob_hotter_bucket = model_inputs.get("prob_hotter_bucket", model_inputs.get("prob_new_high"))
    city_state = model_inputs.get("city_state") or classify_city_state(
        prob_hotter_bucket if prob_hotter_bucket is not None else 1.0
    )

    return {
        "city_slug": city_slug,
        "display_name": city.display_name,
        "metar_station": city.metar_station,
        "current_temp_f": metar.temp_f if metar else None,
        "daily_high_f": daily_high,
        "metar_observed_at": metar.observed_at.isoformat() if metar else None,
        "metar_age_s": _age_seconds(metar.fetched_at if metar else None),
        "forecasts": {
            "nws": {
                "high_f": nws.high_f if nws else None,
                "age_s": _age_seconds(nws.fetched_at if nws else None),
                "error": nws.parse_error if nws else None,
            },
            "wu_daily": {
                "high_f": wu_d.high_f if wu_d else None,
                "age_s": _age_seconds(wu_d.fetched_at if wu_d else None),
                "error": wu_d.parse_error if wu_d else None,
            },
            "wu_hourly": {
                "high_f": wu_h.high_f if wu_h else None,
                "age_s": _age_seconds(wu_h.fetched_at if wu_h else None),
                "error": wu_h.parse_error if wu_h else None,
            },
        },
        "forecast_quality": event.forecast_quality if event else None,
        "wu_scrape_error": event.wu_scrape_error if event else None,
        "prob_new_high": prob_hotter_bucket,
        "prob_hotter_bucket": prob_hotter_bucket,
        "prob_new_high_raw": model_inputs.get("prob_new_high_raw"),
        "lock_regime": model_inputs.get("lock_regime"),
        "observed_bucket_idx": model_inputs.get("observed_bucket_idx"),
        "observed_bucket_upper_f": model_inputs.get("observed_bucket_upper_f"),
        "city_state": city_state,
        "utc_offset": reg.get("utc_offset"),
        "model": {
            "mu": model.mu if model else None,
            "sigma": model.sigma if model else None,
            "probs": json.loads(model.probs_json) if model and model.probs_json else None,
            "inputs": model_inputs or None,
            "computed_at": model.computed_at.isoformat() if model else None,
        } if model else None,
        "event_status": event.status if event else "no_event",
        "settlement_source": event.settlement_source if event else None,
        "settlement_verified": event.settlement_source_verified if event else False,
        "trading_enabled": event.trading_enabled if event else False,
    }


# ─── Market Context ───────────────────────────────────────────────────────────

@router.get("/api/market-context/{city_slug}")
async def get_market_context(city_slug: str, date: Optional[str] = None):
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
    target_date = date or city_local_date(city)

    snapshot = await get_market_context_snapshot_payload(city_slug, target_date)
    return {
        "city_slug": city_slug,
        "date_et": target_date,
        "llm_ready": Config.market_context_llm_ready(),
        "snapshot": snapshot,
    }


@router.post("/api/market-context/{city_slug}/refresh")
async def refresh_market_context(
    city_slug: str,
    date: Optional[str] = None,
    actor: str = Depends(require_admin),
):
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
    target_date = date or city_local_date(city)

    try:
        snapshot = await refresh_market_context_snapshot(city_slug, target_date)
        async with get_session() as sess:
            await append_audit(
                sess,
                actor=actor,
                action="market_context_refresh",
                payload={"city_slug": city_slug, "date_et": target_date, "status": snapshot.get("generation_status")},
                ok=snapshot.get("generation_status") == "success",
                error_msg=snapshot.get("last_error"),
            )
        return {
            "ok": snapshot.get("generation_status") == "success",
            "city_slug": city_slug,
            "date_et": target_date,
            "snapshot": snapshot,
        }
    except MarketContextLLMError as exc:
        async with get_session() as sess:
            await append_audit(
                sess,
                actor=actor,
                action="market_context_refresh",
                payload={"city_slug": city_slug, "date_et": target_date},
                ok=False,
                error_msg=str(exc),
            )
        raise HTTPException(status_code=503, detail=str(exc))
    except MarketContextBuildError as exc:
        async with get_session() as sess:
            await append_audit(
                sess,
                actor=actor,
                action="market_context_refresh",
                payload={"city_slug": city_slug, "date_et": target_date},
                ok=False,
                error_msg=str(exc),
            )
        raise HTTPException(status_code=400, detail=str(exc))


# ─── Markets ─────────────────────────────────────────────────────────────────

@router.get("/markets/{city_slug}")
async def get_markets(city_slug: str):
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        today_city = city_local_date(city)
        event = await get_event(sess, city.id, today_city)
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

        reason = json.loads(sig.reason_json) if sig.reason_json else {}
        slug = city_row.city_slug if city_row else None
        reg = CITY_REGISTRY_BY_SLUG.get(slug or "", {})
        out.append({
            "signal_id": sig.id,
            "city_slug": slug,
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
            "prob_new_high": reason.get("prob_hotter_bucket", reason.get("prob_new_high")),
            "prob_hotter_bucket": reason.get("prob_hotter_bucket", reason.get("prob_new_high")),
            "prob_new_high_raw": reason.get("prob_new_high_raw"),
            "lock_regime": reason.get("lock_regime"),
            "observed_bucket_idx": reason.get("observed_bucket_idx"),
            "observed_bucket_upper_f": reason.get("observed_bucket_upper_f"),
            "city_state": reason.get("city_state"),
            "utc_offset": reg.get("utc_offset"),
            "reason": reason,
            "computed_at": sig.computed_at.isoformat(),
        })

    out.sort(key=lambda x: x["mkt_prob"] if x.get("mkt_prob") is not None else -1.0, reverse=True)
    return out


@router.get("/signals/{city_slug}")
async def get_city_signals(city_slug: str):
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        today_city = city_local_date(city)
        event = await get_event(sess, city.id, today_city)
        if not event:
            return []
        buckets = await get_buckets_for_event(sess, event.id)

    result = []
    for b in buckets:
        async with get_session() as sess:
            from backend.storage.repos import get_latest_signal_for_bucket
            sig = await get_latest_signal_for_bucket(sess, b.id)
        if sig:
            reason = json.loads(sig.reason_json) if sig.reason_json else {}
            result.append({
                "bucket_idx": b.bucket_idx,
                "label": b.label,
                "low_f": b.low_f,
                "high_f": b.high_f,
                "model_prob": sig.model_prob,
                "mkt_prob": sig.mkt_prob,
                "true_edge": sig.true_edge,
                "exec_cost": sig.exec_cost,
                "prob_new_high": reason.get("prob_hotter_bucket", reason.get("prob_new_high")),
                "prob_hotter_bucket": reason.get("prob_hotter_bucket", reason.get("prob_new_high")),
                "prob_new_high_raw": reason.get("prob_new_high_raw"),
                "lock_regime": reason.get("lock_regime"),
                "observed_bucket_idx": reason.get("observed_bucket_idx"),
                "observed_bucket_upper_f": reason.get("observed_bucket_upper_f"),
                "city_state": reason.get("city_state"),
                "gate_failures": json.loads(sig.gate_failures_json) if sig.gate_failures_json else [],
                "reason": reason,
                "computed_at": sig.computed_at.isoformat(),
            })

    result.sort(key=lambda x: x["mkt_prob"] if x.get("mkt_prob") is not None else -1.0, reverse=True)
    return result


# ─── Positions ────────────────────────────────────────────────────────────────

@router.get("/positions")
async def get_positions():
    today = et_today()
    async with get_session() as sess:
        positions = await get_all_positions(sess)
        daily_pnl = await get_daily_realized_pnl(sess, today)

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


@router.get("/api/unredeemed-wins")
async def unredeemed_wins():
    """Unredeemed winning positions from resolved events."""
    async with get_session() as sess:
        events = await get_unredeemed_resolved_events(sess)

    result = []
    for event in events:
        async with get_session() as sess:
            from backend.storage.models import City
            city = await sess.get(City, event.city_id)
            buckets_info = []
            total_payout = 0.0
            has_conditions = False
            for bucket in event.buckets:
                if not bucket.condition_id:
                    continue
                has_conditions = True
                pos = await get_position(sess, bucket.id)
                is_winner = (
                    event.winning_bucket_idx is not None
                    and bucket.bucket_idx == event.winning_bucket_idx
                )
                net_qty = pos.net_qty if pos else 0.0
                avg_cost = pos.avg_cost if pos else 0.0
                payout = net_qty * 1.0 if is_winner else 0.0
                buckets_info.append({
                    "bucket_idx": bucket.bucket_idx,
                    "label": bucket.label,
                    "net_qty": net_qty,
                    "avg_cost": avg_cost,
                    "is_winner": is_winner,
                    "expected_payout": round(payout, 4),
                })
                total_payout += payout

        if has_conditions:
            result.append({
                "event_id": event.id,
                "city_name": city.display_name if city else "Unknown",
                "city_slug": city.city_slug if city else "",
                "date_et": event.date_et,
                "winning_bucket_idx": event.winning_bucket_idx,
                "resolved_at": event.resolved_at.isoformat() if event.resolved_at else None,
                "buckets": buckets_info,
                "total_expected_payout": round(total_payout, 4),
            })

    return {"unredeemed": result}


@router.post("/api/check-resolved")
async def check_resolved(actor: str = Depends(require_admin)):
    """Trigger resolution check against Gamma API for all unresolved events."""
    from backend.execution.redeemer import check_resolved_markets
    count = await check_resolved_markets()
    return {"ok": True, "newly_resolved": count}


@router.post("/api/redeem/{event_id}")
async def redeem_event(event_id: int, force: bool = False, actor: str = Depends(require_admin)):
    """Manually redeem positions for a single resolved event. Use ?force=true to retry."""
    from backend.execution.redeemer import redeem_single_event
    try:
        result = await redeem_single_event(event_id, actor=f"admin:{actor}", force=force)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/redeem-by-condition")
async def redeem_by_condition(condition_id: str, actor: str = Depends(require_admin)):
    """Redeem positions directly by condition ID (no DB event needed). For manual trades.

    Queries the Polymarket data API to find token IDs, then queries on-chain
    ERC1155 balances and passes actual amounts to NegRiskAdapter.redeemPositions.
    """
    import aiohttp
    from eth_account import Account

    if not Config.POLYMARKET_PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="No private key configured")

    account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    sender = account.address

    NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
    REDEEM_SELECTOR = bytes.fromhex("dbeccb23")
    BALANCE_OF_SEL = bytes.fromhex("00fdd58e")
    POLYGON_RPC = Config.POLYGON_RPC_URL

    # Find YES/NO token IDs from Polymarket data API
    addr = Config.FUNDER_ADDRESS or sender
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        # Fetch positions to find token IDs for this condition
        yes_token_id = None
        no_token_id = None
        try:
            url = f"https://data-api.polymarket.com/positions?user={addr}"
            async with http.get(url) as resp:
                if resp.status == 200:
                    positions = await resp.json()
                    for pos in positions:
                        if pos.get("conditionId") == condition_id:
                            yes_token_id = pos.get("asset")
                            no_token_id = pos.get("oppositeAsset")
                            break
        except Exception:
            pass

        if not yes_token_id:
            raise HTTPException(status_code=400, detail="Could not find token IDs for this condition")

        # Query on-chain ERC1155 balances (use addr = proxy/funder that holds tokens)
        async def balance_of(token_id_str):
            padded_owner = bytes.fromhex(addr.replace("0x", "").zfill(64))
            padded_token = int(token_id_str).to_bytes(32, "big")
            calldata = "0x" + (BALANCE_OF_SEL + padded_owner + padded_token).hex()
            r = await (await http.post(POLYGON_RPC, json={
                "jsonrpc": "2.0", "id": 1, "method": "eth_call",
                "params": [{"to": NEG_RISK_ADAPTER, "data": calldata}, "latest"],
            })).json()
            return int(r["result"], 16) if "result" in r else 0

        yes_bal = await balance_of(yes_token_id)
        no_bal = await balance_of(no_token_id) if no_token_id else 0

        if yes_bal == 0 and no_bal == 0:
            raise HTTPException(status_code=400, detail=f"No token balance found (yes={yes_bal}, no={no_bal})")

        # Build calldata: redeemPositions(conditionId, [yes_amount, no_amount])
        cid_bytes = bytes.fromhex(condition_id.replace("0x", ""))
        calldata = (
            REDEEM_SELECTOR
            + cid_bytes.rjust(32, b"\x00")
            + (64).to_bytes(32, "big")
            + (2).to_bytes(32, "big")
            + yes_bal.to_bytes(32, "big")
            + no_bal.to_bytes(32, "big")
        )

        nonce_resp = await (await http.post(POLYGON_RPC, json={
            "jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionCount",
            "params": [sender, "latest"],
        })).json()
        gas_resp = await (await http.post(POLYGON_RPC, json={
            "jsonrpc": "2.0", "id": 2, "method": "eth_gasPrice", "params": [],
        })).json()

        nonce = int(nonce_resp["result"], 16)
        gas_price = int(gas_resp["result"], 16)

        tx = {
            "to": NEG_RISK_ADAPTER,
            "value": 0,
            "gas": 300_000,
            "gasPrice": gas_price,
            "nonce": nonce,
            "chainId": Config.CHAIN_ID,
            "data": calldata,
        }
        signed = account.sign_transaction(tx)
        send_resp = await (await http.post(POLYGON_RPC, json={
            "jsonrpc": "2.0", "id": 3, "method": "eth_sendRawTransaction",
            "params": ["0x" + signed.raw_transaction.hex()],
        })).json()

    if "error" in send_resp:
        raise HTTPException(status_code=500, detail=f"TX failed: {send_resp['error']}")

    return {
        "ok": True,
        "tx_hash": send_resp.get("result"),
        "sender": sender,
        "condition_id": condition_id,
        "amounts": [yes_bal, no_bal],
        "to": NEG_RISK_ADAPTER,
    }


@router.get("/api/redemptions")
async def redemptions_list():
    """All events that have positions, with resolution and on-chain status."""
    import aiohttp

    NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
    GET_DETERMINED_SEL = "7ae2e67b"  # getDetermined(bytes32)
    POLYGON_RPC = Config.POLYGON_RPC_URL

    # Gather all relevant events: unresolved with positions + resolved unredeemed + recently redeemed
    async with get_session() as sess:
        unresolved = await get_unresolved_events_with_positions(sess)
        unredeemed = await get_unredeemed_resolved_events(sess)
        redeemed = await get_recently_redeemed_events(sess, days=30)

    all_events = {e.id: e for e in unresolved + unredeemed + redeemed}
    events = sorted(all_events.values(), key=lambda e: e.date_et, reverse=True)

    # Batch check on-chain determination for all condition_ids
    condition_ids = set()
    for evt in events:
        for b in evt.buckets:
            if b.condition_id:
                condition_ids.add(b.condition_id)

    determined_map = {}
    if condition_ids:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            # Step 1: fetch negRiskMarketID for each event from Gamma API (or default to condition ID if not NegRisk)
            cid_to_market_id = {}
            for evt in events:
                if not evt.gamma_event_id:
                    continue
                try:
                    url = f"https://gamma-api.polymarket.com/events/{evt.gamma_event_id}"
                    async with http.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, list) and len(data) > 0:
                                data = data[0]
                            market_id = data.get("negRiskMarketID")
                            
                            # Map this market ID to all buckets in this event
                            for b in evt.buckets:
                                if b.condition_id and market_id:
                                    cid_to_market_id[b.condition_id] = market_id
                except Exception:
                    pass

            # Step 2: call getDetermined(negRiskMarketID)
            for cid in condition_ids:
                # Fall back to checking the condition_id if negRiskMarketID is absent (standard CTF)
                mid = cid_to_market_id.get(cid, cid)
                if not mid:
                    determined_map[cid] = None
                    continue
                try:
                    mid_hex = mid.replace("0x", "")
                    data = f"0x{GET_DETERMINED_SEL}{mid_hex.zfill(64)}"
                    resp = await (await http.post(POLYGON_RPC, json={
                        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
                        "params": [{"to": NEG_RISK_ADAPTER, "data": data}, "latest"],
                    })).json()
                    if "result" in resp:
                        determined_map[cid] = int(resp["result"], 16) != 0
                    else:
                        determined_map[cid] = None
                except Exception:
                    determined_map[cid] = None

    rows = []
    for evt in events:
        async with get_session() as sess:
            from backend.storage.models import City
            city = await sess.get(City, evt.city_id)
            buckets_info = []
            for bucket in evt.buckets:
                if not bucket.condition_id:
                    continue
                pos = await get_position(sess, bucket.id)
                is_winner = (
                    evt.winning_bucket_idx is not None
                    and bucket.bucket_idx == evt.winning_bucket_idx
                )
                buckets_info.append({
                    "bucket_idx": bucket.bucket_idx,
                    "label": bucket.label,
                    "condition_id": bucket.condition_id,
                    "net_qty": pos.net_qty if pos else 0,
                    "avg_cost": pos.avg_cost if pos else 0,
                    "is_winner": is_winner,
                    "on_chain_determined": determined_map.get(bucket.condition_id),
                })

        # Overall on-chain status: determined if ANY bucket with position is determined
        any_determined = any(
            b["on_chain_determined"] for b in buckets_info
            if b["net_qty"] > 0 and b["on_chain_determined"] is not None
        )

        if evt.redeemed_at:
            status = "redeemed"
        elif evt.resolved_at and any_determined:
            status = "redeemable"
        elif evt.resolved_at:
            status = "resolved_not_determined"
        else:
            status = "open"

        rows.append({
            "event_id": evt.id,
            "city_name": city.display_name if city else "?",
            "city_slug": city.city_slug if city else "",
            "date_et": evt.date_et,
            "gamma_event_id": evt.gamma_event_id,
            "status": status,
            "resolved_at": evt.resolved_at.isoformat() if evt.resolved_at else None,
            "redeemed_at": evt.redeemed_at.isoformat() if evt.redeemed_at else None,
            "winning_bucket_idx": evt.winning_bucket_idx,
            "buckets": buckets_info,
        })

    # ── Fetch on-chain positions from Polymarket data API (catches manual trades) ──
    db_condition_ids = set()
    for r in rows:
        for b in r["buckets"]:
            if b["condition_id"]:
                db_condition_ids.add(b["condition_id"])

    addr = Config.FUNDER_ADDRESS
    if not addr and Config.POLYMARKET_PRIVATE_KEY:
        try:
            from eth_account import Account
            addr = Account.from_key(Config.POLYMARKET_PRIVATE_KEY).address
        except Exception:
            pass

    if addr:
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                url = f"https://data-api.polymarket.com/positions?user={addr}"
                async with http.get(url) as resp:
                    if resp.status == 200:
                        api_positions = await resp.json()
                    else:
                        api_positions = []

                for pos in api_positions:
                    cid = pos.get("conditionId", "")
                    if not cid or cid in db_condition_ids:
                        continue
                    size = float(pos.get("size", 0))
                    if size <= 0:
                        continue

                    # Check on-chain determination — getDetermined takes negRiskMarketID
                    determined = None
                    try:
                        # Fetch Market ID from Event mapping
                        mid = cid_to_market_id.get(cid)
                        # If not in DB mapping, try to fetch directly from Gamma API
                        gamma_event_id = pos.get("eventId")
                        if not mid and gamma_event_id:
                            url = f"https://gamma-api.polymarket.com/events/{gamma_event_id}"
                            async with http.get(url) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    if isinstance(data, list) and len(data) > 0:
                                        data = data[0]
                                    mid = data.get("negRiskMarketID")
                        
                        # Fallback to condition id if it's not NegRisk
                        mid = mid or cid

                        if mid:
                            mid_hex = mid.replace("0x", "")
                            call_data = f"0x{GET_DETERMINED_SEL}{mid_hex.zfill(64)}"
                            rpc_resp = await (await http.post(POLYGON_RPC, json={
                                "jsonrpc": "2.0", "id": 1, "method": "eth_call",
                                "params": [{"to": NEG_RISK_ADAPTER, "data": call_data}, "latest"],
                            })).json()
                            if "result" in rpc_resp:
                                determined = int(rpc_resp["result"], 16) != 0
                    except Exception:
                        pass

                    redeemable_api = pos.get("redeemable", False)
                    if redeemable_api and determined:
                        status = "redeemable"
                    elif redeemable_api:
                        status = "resolved_not_determined"
                    else:
                        status = "open"

                    rows.append({
                        "event_id": None,
                        "source": "on_chain",
                        "city_name": pos.get("title", "Manual Position"),
                        "city_slug": "",
                        "date_et": pos.get("endDate", ""),
                        "gamma_event_id": pos.get("eventId"),
                        "event_slug": pos.get("eventSlug", ""),
                        "status": status,
                        "resolved_at": None,
                        "redeemed_at": None,
                        "winning_bucket_idx": None,
                        "buckets": [{
                            "bucket_idx": 0,
                            "label": f"{pos.get('outcome', 'YES')} — {pos.get('title', '')}",
                            "condition_id": cid,
                            "net_qty": size,
                            "avg_cost": float(pos.get("avgPrice", 0)),
                            "is_winner": pos.get("curPrice", 0) == 1,
                            "on_chain_determined": determined,
                        }],
                    })
        except Exception as e:
            log.warning("redemptions: failed to fetch on-chain positions: %s", e)

    return {"events": rows}


@router.get("/api/redeem-diag")
async def redeem_diagnostics(condition_id: str = None):
    """Comprehensive redemption diagnostics. Pass ?condition_id= for per-market details."""
    import aiohttp
    from eth_account import Account
    from backend.execution.chain_utils import get_wallet_address

    signer = Account.from_key(Config.POLYMARKET_PRIVATE_KEY).address if Config.POLYMARKET_PRIVATE_KEY else None
    token_holder = get_wallet_address() if Config.POLYMARKET_PRIVATE_KEY else None

    result = {
        "signer_eoa": signer,
        "funder_proxy": Config.FUNDER_ADDRESS or None,
        "token_holder": token_holder,
        "neg_risk_adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
    }

    if not condition_id or not token_holder:
        return result

    NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
    BALANCE_OF_SEL = bytes.fromhex("00fdd58e")
    GET_DETERMINED_SEL = "7ae2e67b"
    POLYGON_RPC = Config.POLYGON_RPC_URL

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        # Fetch market data from CLOB API for questionId and token IDs
        question_id = None
        yes_token_id = None
        no_token_id = None
        try:
            url = f"https://clob.polymarket.com/markets/{condition_id}"
            async with http.get(url) as resp:
                if resp.status == 200:
                    mkt = await resp.json()
                    question_id = mkt.get("question_id")
                    result["clob_market"] = {
                        "question_id": question_id,
                        "question": mkt.get("question"),
                        "neg_risk": mkt.get("neg_risk"),
                        "active": mkt.get("active"),
                    }
        except Exception as e:
            result["clob_error"] = str(e)

        # Also check data API for positions
        try:
            url = f"https://data-api.polymarket.com/positions?user={token_holder}"
            async with http.get(url) as resp:
                if resp.status == 200:
                    positions = await resp.json()
                    for pos in positions:
                        if pos.get("conditionId") == condition_id:
                            yes_token_id = pos.get("asset")
                            no_token_id = pos.get("oppositeAsset")
                            result["data_api_position"] = {
                                "size": pos.get("size"),
                                "redeemable": pos.get("redeemable"),
                                "curPrice": pos.get("curPrice"),
                                "outcome": pos.get("outcome"),
                                "yes_token_id": yes_token_id,
                                "no_token_id": no_token_id,
                            }
                            break
        except Exception as e:
            result["data_api_error"] = str(e)

        # getDetermined(negRiskMarketID)
        market_id = question_id  # Default to question_id as fallback
        try:
            # We can find the negRiskMarketID by querying the gamma API directly if we had gamma_id, 
            # but since we only have clob question_id here, we can query gamma API via conditionId
            url = f"https://gamma-api.polymarket.com/events?condition_id={condition_id}"
            async with http.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        market_id = data[0].get("negRiskMarketID", question_id)
        except Exception:
            pass
            
        if market_id:
            try:
                mid_hex = market_id.replace("0x", "")
                call_data = f"0x{GET_DETERMINED_SEL}{mid_hex.zfill(64)}"
                rpc_resp = await (await http.post(POLYGON_RPC, json={
                    "jsonrpc": "2.0", "id": 1, "method": "eth_call",
                    "params": [{"to": NEG_RISK_ADAPTER, "data": call_data}, "latest"],
                })).json()
                if "result" in rpc_resp:
                    result["on_chain_determined"] = int(rpc_resp["result"], 16) != 0
                else:
                    result["on_chain_determined"] = None
                    result["rpc_error"] = rpc_resp.get("error")
            except Exception as e:
                result["determined_error"] = str(e)

        # ERC1155 balances
        async def _bal(token_id_str):
            padded_owner = bytes.fromhex(token_holder.replace("0x", "").zfill(64))
            padded_token = int(token_id_str).to_bytes(32, "big")
            cd = "0x" + (BALANCE_OF_SEL + padded_owner + padded_token).hex()
            r = await (await http.post(POLYGON_RPC, json={
                "jsonrpc": "2.0", "id": 1, "method": "eth_call",
                "params": [{"to": NEG_RISK_ADAPTER, "data": cd}, "latest"],
            })).json()
            return int(r["result"], 16) if "result" in r else 0

        if yes_token_id:
            result["yes_balance_raw"] = await _bal(yes_token_id)
            result["yes_balance_shares"] = result["yes_balance_raw"] / 1_000_000
        if no_token_id:
            result["no_balance_raw"] = await _bal(no_token_id)
            result["no_balance_shares"] = result["no_balance_raw"] / 1_000_000

        # Diagnosis
        determined = result.get("on_chain_determined")
        yes_bal = result.get("yes_balance_raw", 0)
        no_bal = result.get("no_balance_raw", 0)
        if determined and (yes_bal > 0 or no_bal > 0):
            result["diagnosis"] = "Ready to redeem"
        elif not determined:
            result["diagnosis"] = "getDetermined=false — market not yet resolved on NegRiskAdapter"
        elif yes_bal == 0 and no_bal == 0:
            result["diagnosis"] = "Zero token balance — already redeemed or tokens not held by this address"
        else:
            result["diagnosis"] = "Unknown state"

    return result


@router.get("/api/position/{city_slug}/{bucket_idx}")
async def get_bucket_position(city_slug: str, bucket_idx: int, date_et: str | None = None):
    """Get current position for a specific bucket."""
    from backend.storage.repos import get_position
    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        if date_et:
            active_date = date_et
        else:
            from backend.tz_utils import city_local_now, city_local_tomorrow
            now_local = city_local_now(city)
            active_date = city_local_tomorrow(city) if now_local.hour >= 20 else city_local_date(city)
        event = await get_event(sess, city.id, active_date)
        if not event:
            return {"net_qty": 0, "avg_cost": 0, "side": "yes", "unrealized_pnl": 0}
        buckets = await get_buckets_for_event(sess, event.id)

    bucket = next((b for b in buckets if b.bucket_idx == bucket_idx), None)
    if not bucket:
        return {"net_qty": 0, "avg_cost": 0, "side": "yes", "unrealized_pnl": 0}

    async with get_session() as sess:
        pos = await get_position(sess, bucket.id)

    if not pos or pos.net_qty <= 0:
        return {"net_qty": 0, "avg_cost": 0, "side": "yes", "unrealized_pnl": 0}
    return {
        "net_qty": pos.net_qty,
        "avg_cost": round(pos.avg_cost, 4),
        "side": pos.side,
        "unrealized_pnl": round(pos.unrealized_pnl, 4),
        "realized_pnl": round(pos.realized_pnl, 4),
    }


# ─── Position Sync ────────────────────────────────────────────────────────────

@router.post("/api/sync-positions")
async def sync_positions_from_api(actor: str = Depends(require_admin)):
    """Sync DB positions from Polymarket data API (restores incorrectly zeroed positions)."""
    import aiohttp
    from backend.execution.chain_utils import get_wallet_address

    addr = get_wallet_address()
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        url = f"https://data-api.polymarket.com/positions?user={addr}"
        async with http.get(url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=f"Polymarket API returned {resp.status}")
            api_positions = await resp.json()

    corrections = []
    for api_pos in api_positions:
        size = float(api_pos.get("size", 0))
        if size <= 0:
            continue
        asset = api_pos.get("asset", "")
        avg_price = float(api_pos.get("avgPrice", 0))
        title = api_pos.get("title", "")

        # Find matching bucket by yes_token_id
        async with get_session() as sess:
            from sqlalchemy import select
            from backend.storage.models import Bucket, Position
            result = await sess.execute(
                select(Bucket).where(Bucket.yes_token_id == asset)
            )
            bucket = result.scalar_one_or_none()
            if not bucket:
                continue

            pos = await get_position(sess, bucket.id)
            if pos and abs(pos.net_qty - size) < 0.01:
                continue  # already correct

            if pos:
                old_qty = pos.net_qty
                pos.net_qty = size
                pos.avg_cost = avg_price
                if pos.last_mkt_price:
                    pos.unrealized_pnl = size * (pos.last_mkt_price - avg_price)
                await sess.commit()
            else:
                new_pos = Position(
                    bucket_id=bucket.id,
                    side="yes",
                    net_qty=size,
                    avg_cost=avg_price,
                    last_mkt_price=avg_price,
                    unrealized_pnl=0.0,
                )
                sess.add(new_pos)
                old_qty = 0
                await sess.commit()

            corrections.append({
                "bucket_id": bucket.id,
                "title": title,
                "old_qty": old_qty,
                "new_qty": size,
                "avg_price": avg_price,
            })

    return {"ok": True, "synced": len(corrections), "corrections": corrections}


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


# ─── Wallet Balance ──────────────────────────────────────────────────────────

@router.get("/api/balance")
async def get_wallet_balance():
    """Return USDC balance and portfolio exposure for header display."""
    from backend.ingestion.polymarket_clob import get_clob

    clob = get_clob()
    balance = None
    if clob and clob.can_trade:
        balance = await clob.get_balance()

    # Fallback: Polymarket data API if CLOB balance unavailable
    if balance is None:
        addr = Config.FUNDER_ADDRESS
        if not addr and Config.POLYMARKET_PRIVATE_KEY:
            try:
                from eth_account import Account
                addr = Account.from_key(Config.POLYMARKET_PRIVATE_KEY).address
            except Exception:
                pass
        if addr:
            try:
                import aiohttp
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as http:
                    url = f"https://data-api.polymarket.com/value?user={addr}"
                    async with http.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            balance = float(data.get("value", 0))
            except Exception:
                pass

    async with get_session() as sess:
        positions = await get_all_positions(sess)

    open_exposure = sum(
        (p.net_qty * p.avg_cost) for p in positions if p.net_qty > 0
    )
    unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    open_count = len([p for p in positions if p.net_qty > 0])

    return {
        "balance": round(balance, 2) if balance is not None else None,
        "open_exposure": round(open_exposure, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "open_positions": open_count,
        "portfolio_value": round((balance or 0) + open_exposure, 2),
    }


# ─── Live Orderbook ──────────────────────────────────────────────────────────

@router.get("/api/orderbook/{city_slug}/{bucket_idx}")
async def get_orderbook(city_slug: str, bucket_idx: int, date_et: str | None = None):
    """Live orderbook + balance for the trading panel."""
    from backend.ingestion.polymarket_clob import get_clob

    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        if date_et:
            active_date = date_et
        else:
            from backend.tz_utils import city_local_now, city_local_tomorrow
            now_local = city_local_now(city)
            active_date = city_local_tomorrow(city) if now_local.hour >= 20 else city_local_date(city)
        event = await get_event(sess, city.id, active_date)
        if not event:
            raise HTTPException(status_code=404, detail=f"No event for this city on {active_date}")
        buckets = await get_buckets_for_event(sess, event.id)

    bucket = next((b for b in buckets if b.bucket_idx == bucket_idx), None)
    if not bucket:
        raise HTTPException(status_code=404, detail=f"Bucket {bucket_idx} not found")
    if not bucket.yes_token_id:
        return {"error": "No token ID for this bucket", "bids": [], "asks": []}

    clob = get_clob()
    raw_book = None
    balance = None
    if clob:
        raw_book = await clob.get_order_book(bucket.yes_token_id)
        if clob.can_trade:
            balance = await clob.get_balance()

    bids = []
    asks = []
    yes_bid = None
    yes_ask = None
    yes_bid_depth = 0.0
    yes_ask_depth = 0.0

    if raw_book:
        raw_bids = raw_book.get("bids") or []
        raw_asks = raw_book.get("asks") or []

        for b in raw_bids:
            price = float(b.get("price", 0))
            size = float(b.get("size", 0))
            if price > 0:
                bids.append({"price": price, "size": size})
        for a in raw_asks:
            price = float(a.get("price", 0))
            size = float(a.get("size", 0))
            if price > 0:
                asks.append({"price": price, "size": size})

        bids.sort(key=lambda x: x["price"], reverse=True)
        asks.sort(key=lambda x: x["price"])

        if bids:
            yes_bid = bids[0]["price"]
            yes_bid_depth = sum(b["size"] for b in bids if b["price"] >= yes_bid)
        if asks:
            yes_ask = asks[0]["price"]
            yes_ask_depth = sum(a["size"] for a in asks if a["price"] <= yes_ask)

    spread = round(yes_ask - yes_bid, 4) if (yes_ask and yes_bid) else None
    yes_mid = round((yes_ask + yes_bid) / 2, 4) if (yes_ask and yes_bid) else None

    # Also fetch latest stored signal for model info
    async with get_session() as sess:
        sig = await get_latest_market_snapshot(sess, bucket.id)

    return {
        "bucket_idx": bucket_idx,
        "label": bucket.label,
        "low_f": bucket.low_f,
        "high_f": bucket.high_f,
        "yes_token_id": bucket.yes_token_id,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_mid": yes_mid,
        "spread": spread,
        "yes_bid_depth": round(yes_bid_depth, 2),
        "yes_ask_depth": round(yes_ask_depth, 2),
        "bids": bids[:10],
        "asks": asks[:10],
        "balance": round(balance, 2) if balance is not None else None,
        "stored_ask": sig.yes_ask if sig else None,
        "stored_bid": sig.yes_bid if sig else None,
    }


# ─── Manual Trade ─────────────────────────────────────────────────────────────

class ManualTradeRequest(BaseModel):
    city_slug: str
    bucket_idx: int
    bucket_id: Optional[int] = None  # canonical PK — when present, bypasses date guessing
    side: str  # "buy_yes" | "buy_no" | "sell_yes" | "sell_no"
    qty: Optional[float] = None  # None = auto-size
    limit_price: Optional[float] = None
    order_type: str = "limit"  # "limit" | "market"


@router.post("/trade")
async def manual_trade(
    body: ManualTradeRequest,
    actor: str = Depends(require_admin),
):
    """Manual trade endpoint — builds a BucketSignal from current state and executes."""
    from backend.engine.signal_engine import BucketSignal
    from backend.execution.trader import execute_signal
    from backend.ingestion.polymarket_clob import get_clob
    from backend.tz_utils import city_local_now, city_local_tomorrow

    async with get_session() as sess:
        city = await _get_city_or_404(sess, body.city_slug)

        if body.bucket_id is not None:
            # Canonical path: resolve bucket directly by PK — no date guessing
            from backend.storage.repos import get_bucket_by_id, get_event_by_id
            bucket = await get_bucket_by_id(sess, body.bucket_id)
            if not bucket:
                raise HTTPException(status_code=404, detail=f"Bucket id={body.bucket_id} not found")
            event = await get_event_by_id(sess, bucket.event_id)
            if not event:
                raise HTTPException(status_code=404, detail=f"Event for bucket id={body.bucket_id} not found")
            if event.city_id != city.id:
                raise HTTPException(status_code=400, detail=f"Bucket id={body.bucket_id} does not belong to city {body.city_slug}")
        else:
            # Legacy path: resolve by city_slug + bucket_idx + 8 PM rollover
            now_local = city_local_now(city)
            if now_local.hour >= 20:
                active_date = city_local_tomorrow(city)
            else:
                active_date = city_local_date(city)

            event = await get_event(sess, city.id, active_date)
            if not event:
                raise HTTPException(status_code=404, detail=f"No event for {body.city_slug} on {active_date}")

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
    true_edge = sig_row.true_edge if sig_row else 0.0
    is_sell = body.side.startswith("sell_")

    signal = BucketSignal(
        city_slug=body.city_slug,
        city_display=city.display_name,
        unit=getattr(city, "unit", "F"),
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
        yes_ask=mkt_snap.yes_ask if mkt_snap else None,
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

    clob_side = "SELL" if is_sell else "BUY"
    result = await execute_signal(
        signal,
        bankroll=bankroll,
        actor=actor,
        manual=True,
        qty_override=body.qty,
        order_type=body.order_type,
        side=clob_side,
        limit_price_override=body.limit_price
    )
    return result


@router.get("/trade/orders/{city_slug}")
async def get_active_orders(city_slug: str, date_et: str | None = None, actor: str = Depends(require_admin)):
    """Get active Polymarket CLOB limit orders for the current city."""
    from backend.ingestion.polymarket_clob import get_clob
    from backend.tz_utils import city_local_now, city_local_date, city_local_tomorrow
    from backend.storage.repos import get_buckets_for_event

    async with get_session() as sess:
        city = await _get_city_or_404(sess, city_slug)
        if date_et:
            active_date = date_et
        else:
            now_local = city_local_now(city)
            active_date = city_local_tomorrow(city) if now_local.hour >= 20 else city_local_date(city)
        event = await get_event(sess, city.id, active_date)
        if not event:
            return []

        buckets = await get_buckets_for_event(sess, event.id)
        if not buckets:
            return []
            
    clob = get_clob()
    if not clob:
        return []
        
    condition_ids = list({b.condition_id for b in buckets if b.condition_id})
    if not condition_ids:
        return []

    # NegRisk multi-outcome markets have a distinct condition_id per bucket.
    # py_clob_client's OpenOrderParams filters to a single market, so fan out
    # and merge the results.
    results = await asyncio.gather(
        *[clob.get_open_orders(market=cid) for cid in condition_ids],
        return_exceptions=True,
    )
    orders: list[dict] = []
    for r in results:
        if isinstance(r, list):
            orders.extend(r)

    token_to_idx: dict[str, int] = {}
    for b in buckets:
        if b.yes_token_id:
            token_to_idx[b.yes_token_id] = b.bucket_idx
        if b.no_token_id:
            token_to_idx[b.no_token_id] = b.bucket_idx

    log = logging.getLogger("routes_active_orders")
    enriched = []
    for o in orders:
        asset_id = o.get('asset_id')
        idx = token_to_idx.get(asset_id)
        if idx is None:
            log.warning("Order has unknown asset_id: %s", asset_id)
            continue
        try:
            enriched.append({
                "id": o.get("id"),
                "bucket_idx": idx,
                "side": o.get("side"),
                "price": float(o.get("price", 0)),
                "size": float(o.get("original_size", 0)),
                "size_matched": float(o.get("size_matched", 0)),
                "created_at": o.get("created_at"),
            })
        except (ValueError, TypeError) as e:
            log.error("Error parsing order: %s, %s", o, e)
            continue

    log.info("active orders: n_markets=%d n_orders=%d", len(condition_ids), len(enriched))
    return enriched


@router.delete("/trade/orders/{order_id}")
async def cancel_order(order_id: str, actor: str = Depends(require_admin)):
    """Cancel an active Polymarket CLOB limit order."""
    from backend.ingestion.polymarket_clob import get_clob
    clob = get_clob()
    if not clob:
        raise HTTPException(500, "CLOB client unavailable")
        
    success = await clob.cancel_order(order_id)
    if not success:
        raise HTTPException(400, "Failed to cancel order or order already filled")
        
    return {"success": True}


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
