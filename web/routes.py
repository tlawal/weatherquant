"""
Dashboard routes — serves Jinja2 HTMX templates.
"""
from __future__ import annotations

import json
import logging
import os
from time import monotonic
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import Config
from backend.city_registry import get_city_priority, CITY_REGISTRY_BY_SLUG
from backend.market_context.service import serialize_market_context_snapshot, _resolve_realized_high_with_source
from backend.tz_utils import city_local_date, city_local_now, city_local_tomorrow, city_local_day_after_tomorrow, et_today
from backend.strategy.kelly import calculate_ev_per_share, calculate_expected_value, calculate_kelly_fraction
from backend.modeling.calibration_engine import get_reliability_metrics, get_reliability_diagnostics
from backend.modeling.settlement import canonical_bucket_ranges
from backend.ingestion.model_metadata import fetch_openmeteo_metadata, _OM_META_ENDPOINTS
from collections import defaultdict

log = logging.getLogger(__name__)


# ── Non-weather METAR conditions (no anomaly badge needed) ────────────────
_NORMAL_CONDITIONS = frozenset({
    "Fair", "Partly Cloudy", "Mostly Cloudy", "Cloudy", "Clear",
})


def _build_city_groups(
    signal_rows: list[dict],
    max_initial_buckets: int = 8,
) -> list[dict]:
    """Group flat signal rows by city, compute city-level aggregates (TWE, etc.).

    Returns list of city group dicts sorted by timezone priority then TWE desc.
    """
    groups_map: dict[str, dict] = {}

    for row in signal_rows:
        slug = row["city_slug"]
        if slug not in groups_map:
            reason = row.get("_reason") or {}
            ensemble = reason.get("ensemble_breakdown") or {}
            groups_map[slug] = {
                "city_slug": slug,
                "city_display": row["city_display"],
                "unit": row.get("unit", "F"),
                "city_state": row.get("city_state", "early"),
                "prob_up": row.get("prob_hotter_bucket", row.get("prob_new_high", 1.0)),
                "prob_new_high_raw": row.get("prob_new_high_raw"),
                "lock_regime": row.get("lock_regime", False),
                "consensus_temp": reason.get("mu_forecast"),
                "kalman_divergence": reason.get("kalman_divergence_f"),
                "kalman_nowcast_active": reason.get("kalman_nowcast_active", False),
                "ensemble_breakdown": ensemble,
                "metar_condition": reason.get("metar_condition"),
                "resolution_high": reason.get("resolution_high"),
                "raw_high": reason.get("raw_high"),
                "observation_minutes": reason.get("observation_minutes"),
                "resolution_mismatch": reason.get("resolution_mismatch"),
                "buckets": [],
            }
        groups_map[slug]["buckets"].append(row)

    for slug, group in groups_map.items():
        buckets = group["buckets"]
        # TWE: sum of positive after-cost edges on liquid buckets (mkt_prob >= 5%)
        group["twe"] = round(sum(
            max(0.0, b.get("true_edge", 0.0))
            for b in buckets
            if (b.get("mkt_prob") or 0) >= 0.05
        ), 4)
        # Sort buckets by bucket_idx for sparkline and display
        buckets.sort(key=lambda b: b.get("bucket_idx", 0))
        group["sparkline_probs"] = [b.get("model_prob", 0) or 0 for b in buckets]
        group["max_initial"] = max_initial_buckets
        group["has_more"] = len(buckets) > max_initial_buckets
        group["twe_highlight"] = group["twe"] >= 0.08  # green highlight threshold
        group["is_anomaly"] = (
            group["metar_condition"] is not None
            and group["metar_condition"] not in _NORMAL_CONDITIONS
        )

    return sorted(
        groups_map.values(),
        key=lambda g: (get_city_priority(g["city_slug"]), -g["twe"]),
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_HERE, "templates")

templates = Jinja2Templates(directory=_TEMPLATES_DIR)


def humanize_age(seconds):
    """Format an age in seconds as 'Xh Ym ago' / 'Xm Ys ago' / 'Xs ago'.

    Falls back to '—' for None so templates can pipe raw values without
    extra existence checks.
    """
    if seconds is None:
        return "—"
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 0:
        s = 0
    if s < 60:
        return f"{s}s ago"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s ago" if s else f"{m}m ago"
    h, m = divmod(m, 60)
    return f"{h}h {m}m ago" if m else f"{h}h ago"


templates.env.filters["humanize_age"] = humanize_age

CURRENT_TEMP_MAX_AGE_S = 2 * 60 * 60


def _age_seconds(dt, now: datetime) -> int | None:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((now - dt).total_seconds())


def _fresh_current_temp_row(row, *, now: datetime, max_age_s: int = CURRENT_TEMP_MAX_AGE_S):
    """Return row only when it is fresh enough for live current-temp display."""
    age_s = _age_seconds(getattr(row, "observed_at", None), now) if row else None
    if age_s is None or age_s < 0 or age_s > max_age_s:
        return None
    return row


def _select_freshest_current_temp(
    rows,
    *,
    now: datetime,
    max_age_s: int = CURRENT_TEMP_MAX_AGE_S,
):
    fresh = [
        row for row in rows
        if _fresh_current_temp_row(row, now=now, max_age_s=max_age_s) is not None
    ]
    if not fresh:
        return None
    def _observed_at_utc(row):
        dt = getattr(row, "observed_at", None)
        if not dt:
            return datetime.min.replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    return max(fresh, key=_observed_at_utc)


async def _fetch_live_current_temp_for_station(city, station_id: str) -> str | None:
    """Fetch and persist a fresh current observation for a station.

    Weather.gov is preferred for the NWS API side; AviationWeather is a fallback.
    Returns the source written, or None when both upstreams fail.
    """
    import aiohttp
    from backend.ingestion.metar import (
        _insert_or_merge_metar_observation,
        _parse_nws_obs_time,
        _parse_nws_temp,
        _parse_obs_time,
        _parse_temp,
        parse_aviationweather_extended,
        parse_nws_extended,
    )

    station = (station_id or "").upper()
    if not station:
        return None

    timeout = aiohttp.ClientTimeout(total=8)

    try:
        headers = {"User-Agent": "WeatherQuant/1.0", "Accept": "application/geo+json"}
        url = f"https://api.weather.gov/stations/{station}/observations/latest"
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as http:
            async with http.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    props = data.get("properties", {})
                    temp = _parse_nws_temp(props)
                    if temp is not None:
                        temp_c, temp_f = temp
                        obs_time = _parse_nws_obs_time(props)
                        await _insert_or_merge_metar_observation(
                            city=city,
                            station_id=station,
                            observed_at=obs_time,
                            report_at=obs_time,
                            temp_c=temp_c,
                            temp_f=temp_f,
                            raw_text=props.get("rawMessage"),
                            raw_json=json.dumps({"source": "nws_obs", **props}, default=str),
                            ext_data=parse_nws_extended(props),
                            source="nws_api",
                        )
                        return "nws_api"
                else:
                    log.debug("current_temp: weather.gov HTTP %d for %s", resp.status, station)
    except Exception as e:
        log.info("current_temp: weather.gov failed for %s: %s", station, e)

    try:
        headers = {"User-Agent": "WeatherQuant/1.0"}
        url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json&latest=1"
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.debug("current_temp: aviationweather HTTP %d for %s", resp.status, station)
                    return None
                data = await resp.json(content_type=None)

        if not isinstance(data, list) or not data:
            return None
        obs = data[0]
        temp = _parse_temp(obs)
        if temp is None:
            return None
        temp_c, temp_f = temp
        obs_time = _parse_obs_time(obs)
        report_at = _parse_obs_time({"obsTime": obs.get("reportTime")}) if obs.get("reportTime") else obs_time
        await _insert_or_merge_metar_observation(
            city=city,
            station_id=station,
            observed_at=obs_time,
            report_at=report_at,
            temp_c=temp_c,
            temp_f=temp_f,
            raw_text=obs.get("rawOb"),
            raw_json=json.dumps(obs, default=str),
            ext_data=parse_aviationweather_extended(obs, temp_c=temp_c),
            source="aviation",
        )
        return "aviation"
    except Exception as e:
        log.info("current_temp: aviationweather failed for %s: %s", station, e)
        return None


def _format_current_temp_dual(
    madis_metar,
    nws_metar,
    madis_obs,
    city,
    *,
    target_date_et=None,
    real_today_et=None,
    now=None,
    active_station: str | None = None,
) -> dict | None:
    """Build dual-source (MADIS + NWS) Current Temp payload for US cities.

    Returns None for non-US cities, stale dates, or when neither source has
    a row yet. Each side carries temp / obs time / age / station / raw METAR /
    source URL so the template can render a ⓘ tooltip with "↗ View Source".
    `primary` names whichever side is fresher (ties → MADIS per unified-arch
    decision); `delta_s` is the age gap (int seconds, or None).

    Module-level so unit tests can call it without spinning up the full
    FastAPI route. `now` defaults to datetime.now(UTC) for live callers.
    """
    if not getattr(city, "is_us", False):
        return None
    # Viewing a past/future date: suppress the live Current Temp dual card.
    if target_date_et is not None and real_today_et is not None and target_date_et != real_today_et:
        return None

    now = now or datetime.now(timezone.utc)
    city_tz_obj = ZoneInfo(getattr(city, "tz", "America/New_York"))

    def _age(dt):
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return round((now - dt).total_seconds(), 0)

    def _local(dt):
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            return dt.astimezone(city_tz_obj).strftime("%-I:%M %p %Z")
        except Exception:
            return None

    # ── MADIS side ──────────────────────────────────────────────────────────
    madis_side = {
        "temp_f": None, "obs_time_local": None, "age_s": None,
        "station": None, "raw_text": None,
        "source_url": None, "source_label": "MADIS HFMETAR",
        "status": None,
    }
    if madis_metar:
        madis_side["temp_f"] = madis_metar.temp_f
        madis_side["obs_time_local"] = _local(madis_metar.observed_at)
        madis_side["age_s"] = _age(madis_metar.observed_at)
        madis_side["station"] = madis_metar.metar_station
        # netCDF payload has no raw METAR string; rendered as — in template.
        madis_side["raw_text"] = getattr(madis_metar, "raw_text", None)
    elif active_station:
        madis_side["station"] = active_station.upper()
        madis_side["status"] = "No fresh MADIS row"
    # source_file lives on the legacy MadisObs row — still useful for the link.
    source_file = getattr(madis_obs, "source_file", None) if madis_obs else None
    if source_file:
        madis_side["source_url"] = (
            "https://madis-data.ncep.noaa.gov/madisPublic1/data/"
            f"LDAD/hfmetar/netCDF/{source_file}"
        )
    else:
        madis_side["source_url"] = (
            "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/"
        )

    # ── NWS side ────────────────────────────────────────────────────────────
    nws_side = {
        "temp_f": None, "obs_time_local": None, "age_s": None,
        "station": None, "raw_text": None,
        "source_url": None, "source_label": "NWS API",
        "status": None,
    }
    if nws_metar:
        nws_side["temp_f"] = nws_metar.temp_f
        nws_side["obs_time_local"] = _local(nws_metar.observed_at)
        nws_side["age_s"] = _age(nws_metar.observed_at)
        nws_side["station"] = nws_metar.metar_station
        nws_side["raw_text"] = getattr(nws_metar, "raw_text", None)
        nws_source = getattr(nws_metar, "source", None)
        if nws_source == "aviation":
            nws_side["source_label"] = "AviationWeather"
        if nws_metar.metar_station and nws_source == "aviation":
            nws_side["source_url"] = (
                f"https://aviationweather.gov/api/data/metar?ids={nws_metar.metar_station}"
                "&format=json&latest=1"
            )
        elif nws_metar.metar_station:
            nws_side["source_url"] = (
                f"https://api.weather.gov/stations/{nws_metar.metar_station}"
                "/observations/latest"
            )
    elif active_station:
        nws_side["station"] = active_station.upper()
        nws_side["status"] = "No fresh NWS/API row"

    # Neither side has data — caller treats this like non-US (no panel).
    if madis_side["age_s"] is None and nws_side["age_s"] is None:
        return None

    # Primary = fresher side (lowest age_s). Ties break to MADIS — it's the
    # nominal primary source per the unified-architecture decision.
    if madis_side["age_s"] is not None and nws_side["age_s"] is not None:
        primary = "madis" if madis_side["age_s"] <= nws_side["age_s"] else "nws"
        delta_s = abs(int(madis_side["age_s"]) - int(nws_side["age_s"]))
    elif madis_side["age_s"] is not None:
        primary = "madis"
        delta_s = None
    else:
        primary = "nws"
        delta_s = None

    return {
        "madis": madis_side,
        "nws": nws_side,
        "primary": primary,
        "delta_s": delta_s,
    }


dashboard_router = APIRouter()
_DASHBOARD_CONTEXT_CACHE: dict[str, dict] = {}


@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    log.info("dashboard: GET / hit")
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_all_cities,
        get_arming_state,
        get_daily_realized_pnl,
        get_open_position_summary,
    )

    today_et = et_today()
    cache_ttl_s = max(0, int(Config.DASHBOARD_CACHE_TTL_SECONDS or 0))
    cache_key = f"dashboard:{today_et}"
    cached = _DASHBOARD_CONTEXT_CACHE.get(cache_key)
    now_mono = monotonic()
    if cache_ttl_s and cached and cached.get("expires_at", 0) > now_mono:
        context = dict(cached["context"])
        context["request"] = request
        context["dashboard_cache_hit"] = True
        return templates.TemplateResponse("dashboard.html", context)

    # Single outer session for the whole request — one connection check-out
    # rather than one per signal × N signals. The previous N-session pattern
    # was leaking aiosqlite daemon threads under SQLite + NullPool: each
    # `async with get_session()` opens a fresh connection (NullPool has no
    # pooling), and under concurrent cron load some of those connections
    # would orphan their daemon threads, eventually exhausting OS thread
    # limits and 500ing the dashboard with `dialect.connect()` failures.
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)
        arming = await get_arming_state(sess)
        daily_pnl = await get_daily_realized_pnl(sess, today_et)
        position_summary = await get_open_position_summary(sess)

        log.info("dashboard: data fetched: cities=%d, arming=%s, pnl=%.2f", len(cities), arming.state, daily_pnl)

    context = {
        "request": request,
        "signal_rows": [],
        "city_groups": [],
        "arming_state": arming.state,
        "daily_pnl": round(daily_pnl, 2),
        "total_exposure": round(float(position_summary["total_exposure"]), 2),
        "open_positions": int(position_summary["open_positions"]),
        "cities": [c.city_slug for c in cities],
        "today_et": today_et,
        "dashboard_cache_hit": False,
    }
    if cache_ttl_s:
        cache_context = {k: v for k, v in context.items() if k != "request"}
        _DASHBOARD_CONTEXT_CACHE.clear()
        _DASHBOARD_CONTEXT_CACHE[cache_key] = {
            "expires_at": monotonic() + cache_ttl_s,
            "context": cache_context,
        }

    return templates.TemplateResponse("dashboard.html", context)


@dashboard_router.get("/city/{city_slug}", response_class=HTMLResponse)
async def city_detail(request: Request, city_slug: str, date: str | None = None):
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_city_by_slug,
        get_event,
        get_buckets_for_event,
        get_latest_metar,
        get_latest_forecast,
        get_latest_successful_forecasts_bulk,
        get_latest_model_snapshot,
        get_latest_signals_for_buckets,
        get_latest_market_snapshots_bulk,
        get_daily_high_metar,
        get_market_context_snapshot,
        get_station_profile,
        get_resolution_high_metar,
        get_avg_peak_timing,
        get_todays_extended_obs,
        bucket_lead_time,
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

    # Default to today; users can select tomorrow via the date dropdown (3-day horizon)
    active_date_et = real_today_et

    target_date_et = date if date else active_date_et

    async with get_session() as sess:
        # Fetch available event dates for this city
        from sqlalchemy import select, distinct
        from backend.storage.models import Event
        date_query = select(distinct(Event.date_et)).where(Event.city_id == city.id).order_by(Event.date_et.desc())
        available_dates = (await sess.execute(date_query)).scalars().all()

        current_temp_station = (city.metar_station or "").upper() if city.metar_station else None
        metar = await get_latest_metar(sess, city.id)
        # Dual-source Current Temp: latest-per-source MetarObs rows for US cities.
        # MADIS writes source="madis"; Weather.gov writes source="nws_api";
        # AviationWeather fallback writes source="aviation".
        madis_obs = None  # legacy MadisObs row — still used for source_file URL
        madis_metar = None  # MetarObs(source="madis")
        nws_api_metar = None  # MetarObs(source="nws_api")
        aviation_metar = None  # MetarObs(source="aviation")
        nws_metar = None  # Freshest NWS/API-side row
        if city.is_us:
            from backend.storage.repos import (
                get_latest_madis_obs,
                get_latest_metar_by_source,
            )
            madis_obs = await get_latest_madis_obs(sess, city.id)
            madis_metar = await get_latest_metar_by_source(
                sess, city.id, "madis", current_temp_station
            )
            nws_api_metar = await get_latest_metar_by_source(
                sess, city.id, "nws_api", current_temp_station
            )
            aviation_metar = await get_latest_metar_by_source(
                sess, city.id, "aviation", current_temp_station
            )
            nws_metar = _select_freshest_current_temp(
                [nws_api_metar, aviation_metar], now=datetime.now(timezone.utc)
            )
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

        forecast_sources = [
            "wu_hourly", "wu_history", "hrrr", "hrrr_15min", "nbm",
            "ecmwf_ifs", "ecmwf_aifs", "gfs_graphcast", "pangu_weather",
            "fourcastnet_v2", "aurora",
        ]
        forecasts_by_source = await get_latest_successful_forecasts_bulk(
            sess, city.id, forecast_sources, target_date_et,
        )
        wu_h = forecasts_by_source.get("wu_hourly")
        wu_history = forecasts_by_source.get("wu_history")
        hrrr_fc = forecasts_by_source.get("hrrr")
        hrrr_15min_fc = forecasts_by_source.get("hrrr_15min")
        nbm_fc = forecasts_by_source.get("nbm")
        ecmwf_ifs_fc = forecasts_by_source.get("ecmwf_ifs")
        ecmwf_aifs_fc = forecasts_by_source.get("ecmwf_aifs")
        # Bayesian-upgrade Q3/U3 — additional AI-NWP foundation models (experimental).
        gfs_graphcast_fc = forecasts_by_source.get("gfs_graphcast")
        # §13 — NOAA AIWP-sourced AI members (Pangu-Weather, FourCastNetv2-small).
        pangu_weather_fc = forecasts_by_source.get("pangu_weather")
        fourcastnet_v2_fc = forecasts_by_source.get("fourcastnet_v2")
        # §17 — Microsoft Aurora (Swin transformer) via NOAA AIWP.
        aurora_fc = forecasts_by_source.get("aurora")
        # Per-source skill (dynamic weight, MAE, bias, yesterday's error) for tooltips.
        try:
            from backend.modeling.station_weights import load_source_skill_summary
            source_skill = await load_source_skill_summary(city.metar_station)
        except Exception:
            source_skill = {}
        
        primary_fc = None
        if city.is_us:
             primary_fc = await get_latest_forecast(sess, city.id, "nws", target_date_et)
        else:
             primary_fc = await get_latest_forecast(sess, city.id, "open_meteo", target_date_et)

        # Lead-time skills for Phase 4 verification panel
        from backend.storage.models import SourceLeadTimeSkill
        lt_rows = (
            await sess.execute(
                select(SourceLeadTimeSkill)
                .where(SourceLeadTimeSkill.city_id == city.id)
                .order_by(SourceLeadTimeSkill.source, SourceLeadTimeSkill.lead_time_bucket_hours)
            )
        ).scalars().all()
        lead_time_skills = []
        for r in lt_rows:
            lead_time_skills.append({
                "source": r.source,
                "lead_time_bucket_hours": r.lead_time_bucket_hours,
                "mae_f": r.mae_f,
                "bias_f": r.bias_f,
                "n_obs": r.n_obs,
                "computed_at": r.computed_at.isoformat() if r.computed_at else None,
            })

        event = await get_event(sess, city.id, target_date_et)
        market_context_snapshot = await get_market_context_snapshot(sess, city.id, target_date_et)

        # Settlement high with source tracking
        settlement_result = await _resolve_realized_high_with_source(
            sess, city=city, date_et=target_date_et,
            observation_minutes=obs_minutes_list or [],
        )
        # Format obs_time in city TZ
        _sh_obs_time = settlement_result.get("obs_time")
        _sh_obs_time_local = None
        if _sh_obs_time:
            try:
                # Safety: if DB returns naive datetime, assume UTC
                if _sh_obs_time.tzinfo is None:
                    _sh_obs_time = _sh_obs_time.replace(tzinfo=timezone.utc)
                city_tz_obj = ZoneInfo(getattr(city, "tz", "America/New_York"))
                _sh_obs_time_local = _sh_obs_time.astimezone(city_tz_obj).strftime("%-I:%M %p %Z")
            except Exception:
                pass

    if city.is_us and current_temp_station and target_date_et == real_today_et:
        now_utc = datetime.now(timezone.utc)
        madis_metar = _fresh_current_temp_row(madis_metar, now=now_utc)
        nws_metar = _select_freshest_current_temp(
            [nws_api_metar, aviation_metar], now=now_utc
        )

        if nws_metar is None:
            log.debug(
                "city_detail: no fresh NWS current temp for %s; using cached observations",
                current_temp_station,
            )

    model = None
    buckets_with_signals = []

    if event:
        async with get_session() as sess:
            buckets = await get_buckets_for_event(sess, event.id)
            model = await get_latest_model_snapshot(sess, event.id)
            model_inputs = json.loads(model.inputs_json) if model and model.inputs_json else {}

            # M1 Phase 1.5 — pull BMA shadow probs once so the per-bucket loop
            # below can attach a per-bucket bma_prob without reparsing JSON.
            bma_shadow = model_inputs.get("bma_shadow") if isinstance(model_inputs, dict) else None
            bma_probs = (bma_shadow or {}).get("probs") or []
            intraday_shadow = model_inputs.get("intraday_threshold_shadow") if isinstance(model_inputs, dict) else None
            intraday_probs = (intraday_shadow or {}).get("probs") or []
            canonical_ranges = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
            observed_high_for_rules = None
            if isinstance(model_inputs, dict):
                observed_high_for_rules = model_inputs.get("observed_high")
                if observed_high_for_rules is None:
                    observed_high_for_rules = model_inputs.get("ground_truth_high")
            try:
                observed_high_for_rules = (
                    float(observed_high_for_rules)
                    if observed_high_for_rules is not None
                    else None
                )
            except (TypeError, ValueError):
                observed_high_for_rules = None

            snapshot_id = model.id if model is not None else None
            bucket_ids = [bucket.id for bucket in buckets]
            signals_by_bucket = await get_latest_signals_for_buckets(
                sess, bucket_ids, snapshot_id=snapshot_id,
            )
            markets_by_bucket = await get_latest_market_snapshots_bulk(sess, bucket_ids)
            now_for_market_age = datetime.now(timezone.utc)
            for bucket in buckets:
                sig = signals_by_bucket.get(bucket.id)
                if model and sig and sig.computed_at < model.computed_at:
                    sig = None
                mkt = markets_by_bucket.get(bucket.id)
                sig_reason = {}
                gate_failures = []
                if sig and sig.reason_json:
                    try:
                        sig_reason = json.loads(sig.reason_json)
                    except Exception:
                        sig_reason = {}
                if sig and sig.gate_failures_json:
                    try:
                        gate_failures = json.loads(sig.gate_failures_json)
                    except Exception:
                        gate_failures = []

                probs = json.loads(model.probs_json) if model and model.probs_json else []
                model_prob = probs[bucket.bucket_idx] if bucket.bucket_idx < len(probs) else None
                # Per-bucket BMA shadow prob (None when no shadow data or bucket index out of range).
                bma_prob = bma_probs[bucket.bucket_idx] if bucket.bucket_idx < len(bma_probs) else None
                intraday_prob = (
                    intraday_probs[bucket.bucket_idx]
                    if bucket.bucket_idx < len(intraday_probs)
                    else None
                )
                canonical_low = None
                canonical_high = None
                if bucket.bucket_idx < len(canonical_ranges):
                    canonical_low, canonical_high = canonical_ranges[bucket.bucket_idx]
                bucket_surpassed = (
                    observed_high_for_rules is not None
                    and canonical_high is not None
                    and observed_high_for_rules >= canonical_high
                )
                resolution_rule_label = None
                if canonical_high is not None:
                    resolution_rule_label = f"settles below {canonical_high:.1f}°{getattr(city, 'unit', 'F')}"

                yes_price = mkt.yes_ask if mkt else None
                ev = calculate_expected_value(model_prob, yes_price) if model_prob is not None and yes_price else None
                kelly_f = calculate_kelly_fraction(
                    model_prob,
                    yes_price,
                    fractional_kelly=Config.KELLY_FRACTION,
                    max_position_size=Config.MAX_POSITION_PCT
                ) if model_prob is not None and yes_price else None
                market_age_s = None
                if mkt and mkt.fetched_at:
                    fetched_at = mkt.fetched_at
                    if fetched_at.tzinfo is None:
                        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                    market_age_s = max(0, int((now_for_market_age - fetched_at).total_seconds()))

                ask_depth = mkt.yes_ask_depth if mkt else None
                bid_depth = mkt.yes_bid_depth if mkt else None
                spread = mkt.spread if mkt else None
                exec_cost = sig.exec_cost if sig else None
                true_edge = sig.true_edge if sig else None
                max_safe_shares = (
                    round(max(0.0, ask_depth * Config.MAX_LIQUIDITY_PCT), 2)
                    if ask_depth is not None else None
                )
                target_trade_shares = (
                    round(min(max_safe_shares, 25.0), 2)
                    if max_safe_shares is not None else None
                )
                impact_adjusted_ev = true_edge
                why_not_tradable = list(gate_failures or [])
                if mkt is None:
                    why_not_tradable.append("no_market_snapshot")
                if market_age_s is not None and market_age_s > 300:
                    why_not_tradable.append(f"market_snapshot_stale_{market_age_s}s")
                if spread is not None and spread > Config.MAX_SPREAD:
                    why_not_tradable.append(
                        f"spread_{spread:.3f}_gt_{Config.MAX_SPREAD:.3f}"
                    )
                if ask_depth is not None and ask_depth < Config.MIN_LIQUIDITY_SHARES:
                    why_not_tradable.append(
                        f"ask_depth_{ask_depth:.0f}_lt_{Config.MIN_LIQUIDITY_SHARES:.0f}"
                    )
                if true_edge is not None and true_edge < Config.MIN_TRUE_EDGE:
                    why_not_tradable.append(
                        f"true_edge_{true_edge:.3f}_lt_{Config.MIN_TRUE_EDGE:.3f}"
                    )
                if mkt and mkt.yes_ask is not None and mkt.yes_ask >= Config.MAX_ENTRY_PRICE:
                    why_not_tradable.append(
                        f"ask_{mkt.yes_ask:.3f}_gte_max_entry_{Config.MAX_ENTRY_PRICE:.3f}"
                    )
                display_ev_per_share = sig_reason.get("ev_per_share")
                if display_ev_per_share is None and model_prob is not None and mkt and mkt.yes_mid is not None:
                    display_ev_per_share = calculate_ev_per_share(model_prob, mkt.yes_mid)
                display_ev_at_bid = sig_reason.get("ev_at_bid")
                if display_ev_at_bid is None and model_prob is not None and mkt and mkt.yes_bid is not None:
                    display_ev_at_bid = calculate_ev_per_share(model_prob, mkt.yes_bid)
                # Preserve order while avoiding duplicate reasons from the signal
                # engine and this route-level display layer.
                why_not_tradable = list(dict.fromkeys(why_not_tradable))
                actionable = (
                    true_edge is not None
                    and true_edge >= Config.MIN_TRUE_EDGE
                    and mkt is not None
                    and (mkt.yes_mid is not None and 0.02 <= mkt.yes_mid <= 0.98)
                    and (ask_depth or 0.0) >= Config.MIN_LIQUIDITY_SHARES
                    and not why_not_tradable
                )

                buckets_with_signals.append({
                    "bucket_idx": bucket.bucket_idx,
                    "bucket_id": bucket.id,
                    "label": bucket.label or f"Bucket {bucket.bucket_idx}",
                    "low_f": bucket.low_f,
                    "high_f": bucket.high_f,
                    "canonical_low_f": canonical_low,
                    "canonical_high_f": canonical_high,
                    "observed_high_f": observed_high_for_rules,
                    "surpassed": bucket_surpassed,
                    "resolution_rule_label": resolution_rule_label,
                    "yes_token_id": bucket.yes_token_id,
                    "model_prob": round(model_prob, 4) if model_prob is not None else None,
                    "bma_prob": round(bma_prob, 4) if bma_prob is not None else None,
                    "intraday_prob": round(intraday_prob, 4) if intraday_prob is not None else None,
                    "mkt_prob": mkt.yes_mid if mkt else None,
                    "yes_bid": mkt.yes_bid if mkt else None,
                    "yes_ask": mkt.yes_ask if mkt else None,
                    "spread": spread,
                    "ask_depth": ask_depth,
                    "bid_depth": bid_depth,
                    "market_snapshot_age_s": market_age_s,
                    "market_snapshot_at": mkt.fetched_at.isoformat() if mkt and mkt.fetched_at else None,
                    "true_edge": true_edge,
                    "raw_edge": sig.raw_edge if sig else None,
                    "ev": ev,
                    "kelly_f": kelly_f,
                    "exec_cost": exec_cost,
                    "ev_per_share": display_ev_per_share,
                    "ev_at_bid": display_ev_at_bid,
                    "impact_adjusted_ev": impact_adjusted_ev,
                    "max_safe_shares": max_safe_shares,
                    "target_trade_shares": target_trade_shares,
                    "entry_ladder": {
                        "passive_bid": mkt.yes_bid if mkt else None,
                        "inside": (
                            round(((mkt.yes_bid or 0) + (mkt.yes_ask or 0)) / 2, 4)
                            if mkt and mkt.yes_bid is not None and mkt.yes_ask is not None
                            else None
                        ),
                        "marketable_ask": mkt.yes_ask if mkt else None,
                    },
                    "exit_ladder": {
                        "bid": mkt.yes_bid if mkt else None,
                        "bid_depth": bid_depth,
                        "ev_at_bid": display_ev_at_bid,
                    },
                    "why_not_tradable": why_not_tradable,
                    "gate_failures": gate_failures,
                    "bucket_live_calibration": (
                        sig_reason.get("bucket_live_calibration")
                        if isinstance(sig_reason.get("bucket_live_calibration"), dict)
                        else None
                    ),
                    "market_sanity": (
                        sig_reason.get("market_sanity")
                        if isinstance(sig_reason.get("market_sanity"), dict)
                        else None
                    ),
                    "actionable": actionable,
                })

    model_inputs = json.loads(model.inputs_json) if model and model.inputs_json else {}
    probs_json = json.dumps(json.loads(model.probs_json) if model and model.probs_json else [])

    def _float_or_none(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _read_json_file(path):
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text())
        except Exception as exc:
            return {"error": f"metadata_parse_failed: {exc}"}

    def _residual_ml_status() -> dict:
        try:
            from backend.modeling.residual_paths import (
                residual_metadata_path,
                residual_model_path,
                residual_shadow_metadata_path,
                residual_shadow_model_path,
            )
        except Exception as exc:
            return {
                "loaded": False,
                "shadow_available": False,
                "status": "unavailable",
                "error": str(exc),
            }

        promoted_meta_raw = _read_json_file(residual_metadata_path())
        shadow_meta_raw = _read_json_file(residual_shadow_metadata_path())
        promoted_meta = promoted_meta_raw if isinstance(promoted_meta_raw, dict) else {}
        shadow_meta = shadow_meta_raw if isinstance(shadow_meta_raw, dict) else {}
        loaded = residual_model_path().exists()
        shadow_available = residual_shadow_model_path().exists()
        if loaded:
            status = "promoted"
        elif shadow_available:
            status = "shadow"
        else:
            status = "fallback"
        return {
            "loaded": loaded,
            "shadow_available": shadow_available,
            "status": status,
            "promoted_metadata": promoted_meta,
            "shadow_metadata": shadow_meta,
            "test_mae": _float_or_none((promoted_meta or {}).get("test_mae")),
            "baseline_mae": _float_or_none((promoted_meta or {}).get("baseline_mae")),
            "rain_subset_mae": _float_or_none((promoted_meta or {}).get("rain_subset_mae")),
            "shadow_test_mae": _float_or_none((shadow_meta or {}).get("test_mae")),
            "shadow_baseline_mae": _float_or_none((shadow_meta or {}).get("baseline_mae")),
            "shadow_rain_subset_mae": _float_or_none((shadow_meta or {}).get("rain_subset_mae")),
            "promotion_blockers": (
                (shadow_meta or {}).get("promotion_blockers")
                or (promoted_meta or {}).get("promotion_blockers")
                or []
            ),
        }

    def _accuracy_status() -> dict:
        threshold_cal = (
            model_inputs.get("threshold_calibration")
            if isinstance(model_inputs, dict) and isinstance(model_inputs.get("threshold_calibration"), dict)
            else {}
        )
        sanity_rows = [
            row
            for row in (b.get("market_sanity") for b in buckets_with_signals)
            if isinstance(row, dict)
        ]
        bucket_cal_rows = [
            row
            for row in (b.get("bucket_live_calibration") for b in buckets_with_signals)
            if isinstance(row, dict)
        ]
        posterior_edges = [
            _float_or_none(row.get("posterior_edge"))
            for row in sanity_rows
            if _float_or_none(row.get("posterior_edge")) is not None
        ]
        weights = [
            _float_or_none(row.get("weight"))
            for row in sanity_rows
            if _float_or_none(row.get("weight")) is not None
        ]
        gaps = [
            _float_or_none(row.get("gap"))
            for row in sanity_rows
            if _float_or_none(row.get("gap")) is not None
        ]
        threshold_sample_count = (
            threshold_cal.get("min_sample_count")
            or threshold_cal.get("max_sample_count")
            or 0
        )
        return {
            "threshold": {
                "applied": bool(threshold_cal.get("applied")),
                "context_used": threshold_cal.get("context_used") or "identity",
                "sample_count": int(threshold_sample_count or 0),
                "threshold_count": int(
                    threshold_cal.get("threshold_count")
                    or threshold_cal.get("thresholds_applied")
                    or 0
                ),
                "brier_raw": _float_or_none(threshold_cal.get("brier_raw")),
                "brier_cal": _float_or_none(threshold_cal.get("brier_cal")),
                "rps_raw": _float_or_none(threshold_cal.get("rps_raw")),
                "rps_cal": _float_or_none(threshold_cal.get("rps_cal")),
                "reason": threshold_cal.get("reason"),
            },
            "bucket_live": {
                "rows": len(bucket_cal_rows),
                "applied_count": sum(1 for row in bucket_cal_rows if row.get("applied")),
                "max_sample_count": max(
                    [int(row.get("sample_count") or 0) for row in bucket_cal_rows] or [0]
                ),
                "avg_brier": (
                    sum(_float_or_none(row.get("brier")) or 0.0 for row in bucket_cal_rows if _float_or_none(row.get("brier")) is not None)
                    / max(1, sum(1 for row in bucket_cal_rows if _float_or_none(row.get("brier")) is not None))
                ) if bucket_cal_rows else None,
            },
            "market_sanity": {
                "rows": len(sanity_rows),
                "blocked_count": sum(1 for row in sanity_rows if row.get("blocked")),
                "max_weight": max(weights) if weights else None,
                "worst_posterior_edge": min(posterior_edges) if posterior_edges else None,
                "largest_gap": max(gaps) if gaps else None,
                "failure": next((row.get("failure") for row in sanity_rows if row.get("failure")), None),
            },
            "residual_ml": _residual_ml_status(),
        }

    accuracy_status = _accuracy_status()

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

    # City local-tz info for the dual-format clock display below.
    _city_tz_name = getattr(city, "tz", None) or "America/New_York"
    _city_tz = ZoneInfo(_city_tz_name)
    _city_tz_label = "ET" if _city_tz_name == "America/New_York" else (
        datetime.now(_city_tz).strftime("%Z") or _city_tz_name.split("/")[-1]
    )

    def _fmt_utc(dt) -> str | None:
        """Format UTC time alongside the city's local time, e.g. '12:00 UTC (08:00 ET)'."""
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        utc_str = dt.astimezone(timezone.utc).strftime("%H:%M")
        local_str = dt.astimezone(_city_tz).strftime("%H:%M")
        return f"{utc_str} UTC ({local_str} {_city_tz_label})"

    def _model_run_age(dt) -> str | None:
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        minutes = int(delta.total_seconds() // 60)
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"

    def _forecast_reference_time(fc) -> datetime | None:
        if not fc:
            return None
        return getattr(fc, "model_run_at", None) or getattr(fc, "fetched_at", None)

    def _lead_time_hours(model_run_at: datetime | None) -> int | None:
        if not model_run_at:
            return None
        if model_run_at.tzinfo is None:
            model_run_at = model_run_at.replace(tzinfo=timezone.utc)
        event_end = datetime.strptime(target_date_et, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=_city_tz
        )
        return max(0, int((event_end.astimezone(timezone.utc) - model_run_at).total_seconds() // 3600))

    lead_skill_by_key = {
        (row["source"], row["lead_time_bucket_hours"]): row
        for row in lead_time_skills
    }

    forecast_obs_for_lead = {
        "nws": primary_fc if city.is_us else None,
        "wu_hourly": wu_h,
        "hrrr": hrrr_fc,
        "hrrr_15min": hrrr_15min_fc,
        "nbm": nbm_fc,
        "ecmwf_ifs": ecmwf_ifs_fc,
        "ecmwf_aifs": ecmwf_aifs_fc,
        "gfs_graphcast": gfs_graphcast_fc,
        "pangu_weather": pangu_weather_fc,
        "fourcastnet_v2": fourcastnet_v2_fc,
        "aurora": aurora_fc,
    }
    current_lead_skills = []
    for src, fc in forecast_obs_for_lead.items():
        if not fc or getattr(fc, "high_f", None) is None:
            continue
        ref_time = _forecast_reference_time(fc)
        lead_hours = _lead_time_hours(ref_time)
        if lead_hours is None:
            continue
        lead_bucket = bucket_lead_time(lead_hours)
        skill = lead_skill_by_key.get((src, lead_bucket))
        if not skill or skill.get("mae_f") is None:
            continue
        current_lead_skills.append({
            "source": src,
            "lead_time_hours": lead_hours,
            "lead_time_bucket_hours": lead_bucket,
            "mae_f": skill.get("mae_f"),
            "bias_f": skill.get("bias_f"),
            "n_obs": skill.get("n_obs") or 0,
        })
    best_current_lead_skill = (
        min(current_lead_skills, key=lambda s: s["mae_f"])
        if current_lead_skills else None
    )

    # `_format_current_temp_dual` lives at module scope (testable) — see above.

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
    reliability_diag = await get_reliability_diagnostics(city.id)

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
                s.high_f for s in [primary_fc, wu_h]
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

    # Fetch station calibration for compact card
    station_cal = None
    if city.metar_station:
        from backend.storage.repos import get_station_calibration
        async with get_session() as sess:
            station_cal_row = await get_station_calibration(sess, city.metar_station)
        if station_cal_row:
            station_cal = {
                "station_id": station_cal_row.station_id,
                "mae_f": station_cal_row.mae_f,
                "bias_f": station_cal_row.bias_f,
                "rmse_f": station_cal_row.rmse_f,
                "n_samples": station_cal_row.n_samples,
                "pct_days_traded": station_cal_row.pct_days_traded,
                "tradeability": station_cal_row.tradeability,
                "best_source": station_cal_row.best_source,
                "best_source_mae": station_cal_row.best_source_mae,
                "best_overall_source": station_cal_row.best_source,
                "best_overall_source_mae": station_cal_row.best_source_mae,
                "best_current_lead": best_current_lead_skill,
                "mae_ecmwf_f": station_cal_row.mae_ecmwf_f,
                "mae_gfs_hrrr_f": station_cal_row.mae_gfs_hrrr_f,
                "mae_nws_f": station_cal_row.mae_nws_f,
                "winner": station_cal_row.winner,
            }

    model_snapshot_current_temp_lag_s = None
    if metar and metar.observed_at and model and model.computed_at:
        metar_obs_at = metar.observed_at
        model_computed_at = model.computed_at
        if metar_obs_at.tzinfo is None:
            metar_obs_at = metar_obs_at.replace(tzinfo=timezone.utc)
        if model_computed_at.tzinfo is None:
            model_computed_at = model_computed_at.replace(tzinfo=timezone.utc)
        model_snapshot_current_temp_lag_s = max(
            0.0,
            (metar_obs_at.astimezone(timezone.utc) - model_computed_at.astimezone(timezone.utc)).total_seconds(),
        )

    obs_proximity = None
    if city.is_us:
        try:
            from backend.execution.obs_proximity import build_obs_proximity_status

            def _obs_ref_from_inputs() -> float | None:
                if not isinstance(model_inputs, dict):
                    return None
                adaptive = model_inputs.get("adaptive") if isinstance(model_inputs.get("adaptive"), dict) else {}
                for raw in (
                    model_inputs.get("current_temp_f"),
                    adaptive.get("predicted_daily_high"),
                    model_inputs.get("projected_high_for_blend"),
                    model_inputs.get("projected_high"),
                    model_inputs.get("observed_high"),
                    model_inputs.get("ground_truth_high"),
                ):
                    try:
                        if raw is not None:
                            return float(raw)
                    except (TypeError, ValueError):
                        continue
                return None

            obs_reference_temp = _obs_ref_from_inputs()
            if obs_reference_temp is None and metar and target_date_et == real_today_et:
                obs_reference_temp = metar.temp_f

            obs_station = None
            if isinstance(model_inputs, dict):
                obs_station = model_inputs.get("active_station_id")
            obs_station = obs_station or current_temp_station

            obs_proximity = build_obs_proximity_status(
                city_slug=city.city_slug,
                station_id=obs_station,
                now_local=now_local,
                observation_minutes=obs_minutes_list,
                bucket_specs=buckets_with_signals,
                reference_temp_f=obs_reference_temp,
                enabled=Config.OBS_EXIT_ENABLED,
                is_us=bool(city.is_us),
                window_minutes=Config.OBS_EXIT_WINDOW_MINUTES,
                temp_sensitivity_threshold_f=Config.TEMP_SENSITIVITY_THRESHOLD_F,
            )
        except Exception as e:
            log.warning("city_detail: OBS_PROXIMITY status failed for %s: %s", city_slug, e)

    wallet_leaderboard = {
        "enabled": Config.WALLET_TRACKER_ENABLED,
        "rows": [],
        "current_market": [],
        "global_leaders": [],
        "city_leaders": [],
        "bucket_consensus": [],
        "confluence": {"status": "unavailable", "badge": "NO RANKED FLOW", "reason": "wallet_tracker_unavailable"},
        "display_limit": Config.WALLET_TRACKER_DISPLAY_LIMIT,
        "status": "unavailable",
        "reason": "wallet_tracker_unavailable",
        "message": "Unable to load wallets - check wallet tracker configuration or public trade ingestion.",
        "disclaimer": (
            "Wallet leaderboard is read-only public-market analytics. "
            "It is not a copy-trading signal and does not trigger automated trades."
        ),
    }
    smart_money_context = {
        "status": "unavailable",
        "badge": "NO RANKED FLOW",
        "reason": "wallet_tracker_unavailable",
    }
    try:
        from backend.market_context.wallet_tracker import (
            get_wallet_leaderboard_payload,
        )

        wallet_leaderboard = await get_wallet_leaderboard_payload(
            city.city_slug,
            target_date_et,
            buckets=buckets_with_signals,
        )
        smart_money_context = wallet_leaderboard.get("confluence") or {
            "status": "unavailable",
            "reason": "smart_money_context_unavailable",
        }
    except Exception as e:
        log.warning("city_detail: wallet tracker payload failed for %s: %s", city_slug, e)
        wallet_leaderboard["status"] = "error"
        wallet_leaderboard["reason"] = "wallet_tracker_load_error"
        wallet_leaderboard["message"] = (
            "Unable to load wallets - check wallet tracker configuration or public trade ingestion."
        )
        smart_money_context = {
            "status": "unavailable",
            "badge": "NO RANKED FLOW",
            "reason": "wallet_tracker_load_error",
        }

    return templates.TemplateResponse(
        "city.html",
        {
            "request": request,
            "city": city,
            "today_et": target_date_et,
            "real_today_et": real_today_et,
            "city_tomorrow": city_local_tomorrow(city),
            "city_day_after_tomorrow": city_local_day_after_tomorrow(city),
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
            "model_snapshot_current_temp_lag_s": model_snapshot_current_temp_lag_s,
            "model_snapshot_stale_current_temp": (
                model_snapshot_current_temp_lag_s is not None
                and model_snapshot_current_temp_lag_s > 30.0
            ),
            "settlement_source_verified": event.settlement_source_verified if event else None,
            "settlement_high": {
                "high_f": settlement_result.get("high_f"),
                "rounded_settlement_f": settlement_result.get("rounded_settlement_f"),
                "source_used": settlement_result.get("source_used"),
                "obs_time_local": _sh_obs_time_local,
                "source_url": {
                    "tgftp": f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{city.metar_station}.TXT" if city.metar_station else None,
                    "wu_history": f"https://www.wunderground.com/history/daily/{city.metar_station}/date/{target_date_et}" if city.metar_station else None,
                    "resolution_metar": f"https://aviationweather.gov/api/data/metar?ids={city.metar_station}&format=json&latest=1" if city.metar_station else None,
                    "raw_metar": f"https://aviationweather.gov/api/data/metar?ids={city.metar_station}&format=json&latest=1" if city.metar_station else None,
                }.get(settlement_result.get("source_used") or ""),
            },
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
                "obs_time_local": (
                    lambda: (
                        metar.observed_at.replace(tzinfo=timezone.utc).astimezone(
                            ZoneInfo(getattr(city, "tz", "America/New_York"))
                        ).strftime("%-I:%M %p %Z")
                        if metar.observed_at.tzinfo is None
                        else metar.observed_at.astimezone(
                            ZoneInfo(getattr(city, "tz", "America/New_York"))
                        ).strftime("%-I:%M %p %Z")
                    )()
                ) if (metar and metar.observed_at and target_date_et == real_today_et) else None,
            },
            # Dual-source Current Temp — MADIS + NWS API side-by-side (US cities only)
            "current_temp_dual": _format_current_temp_dual(
                madis_metar, nws_metar, madis_obs, city,
                target_date_et=target_date_et,
                real_today_et=real_today_et,
                active_station=current_temp_station,
            ) if city.is_us else None,
            "forecasts": {
                "primary": {
                    "source": "nws" if city.is_us else "Open-Meteo",
                    "high_f": primary_fc.high_f if primary_fc else None,
                    "age_s": _age(primary_fc.fetched_at if primary_fc else None),
                    "collected_at": _fmt_time_et(primary_fc.fetched_at if primary_fc else None),
                    "model_run_at": _fmt_utc(primary_fc.model_run_at if primary_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(primary_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(primary_fc)),
                    "url": f"https://api.weather.gov/gridpoints/{city.nws_office}/{city.nws_grid_x},{city.nws_grid_y}/forecast" if city.is_us else f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}&longitude={city.lon}&hourly=temperature_2m&start_date={target_date_et}&end_date={target_date_et}",
                    "skill": source_skill.get("nws"),
                },
                "wu_hourly": {
                    "high_f": wu_h.high_f if wu_h else None,
                    "age_s": _age(wu_h.fetched_at if wu_h else None),
                    "url": f"https://www.wunderground.com/hourly/{city.metar_station}/date/{target_date_et}" if city.metar_station else None,
                    "peak_hour": wu_hourly_raw.get("peak_hour"),
                    "collected_at": _fmt_time_et(wu_h.fetched_at if wu_h else None),
                    "model_run_at": _fmt_utc(wu_h.model_run_at if wu_h else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(wu_h)),
                    "model_run_age": _model_run_age(_forecast_reference_time(wu_h)),
                    "skill": source_skill.get("wu_hourly"),
                },
                "wu_history": {
                    "high_f": wu_history.high_f if wu_history else None,
                    "age_s": _age(wu_history.fetched_at if wu_history else None),
                    "url": f"https://www.wunderground.com/history/daily/{city.metar_station}/date/{target_date_et}" if city.metar_station else None,
                    "obs_time": wu_history_raw.get("obs_time"),
                    "collected_at": _fmt_time_et(wu_history.fetched_at if wu_history else None),
                    "model_run_at": _fmt_utc(wu_history.model_run_at if wu_history else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(wu_history)),
                    "model_run_age": _model_run_age(_forecast_reference_time(wu_history)),
                },
                "hrrr": {
                    "high_f": hrrr_fc.high_f if hrrr_fc else None,
                    "age_s": _age(hrrr_fc.fetched_at if hrrr_fc else None),
                    "collected_at": _fmt_time_et(hrrr_fc.fetched_at if hrrr_fc else None),
                    "model_run_at": _fmt_utc(hrrr_fc.model_run_at if hrrr_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(hrrr_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(hrrr_fc)),
                    "url": f"https://open-meteo.com/en/docs?latitude={city.lat}&longitude={city.lon}&hourly=temperature_2m&models=gfs_hrrr&temperature_unit=fahrenheit&start_date={target_date_et}&end_date={target_date_et}" if city.lat else None,
                    "skill": source_skill.get("hrrr"),
                },
                "nbm": {
                    "high_f": nbm_fc.high_f if nbm_fc else None,
                    "age_s": _age(nbm_fc.fetched_at if nbm_fc else None),
                    "collected_at": _fmt_time_et(nbm_fc.fetched_at if nbm_fc else None),
                    "model_run_at": _fmt_utc(nbm_fc.model_run_at if nbm_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(nbm_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(nbm_fc)),
                    "url": f"https://open-meteo.com/en/docs?latitude={city.lat}&longitude={city.lon}&hourly=temperature_2m&models=ncep_nbm_conus&temperature_unit=fahrenheit&start_date={target_date_et}&end_date={target_date_et}" if city.lat else None,
                    "skill": source_skill.get("nbm"),
                },
                "ecmwf_ifs": {
                    "high_f": ecmwf_ifs_fc.high_f if ecmwf_ifs_fc else None,
                    "age_s": _age(ecmwf_ifs_fc.fetched_at if ecmwf_ifs_fc else None),
                    "collected_at": _fmt_time_et(ecmwf_ifs_fc.fetched_at if ecmwf_ifs_fc else None),
                    "model_run_at": _fmt_utc(ecmwf_ifs_fc.model_run_at if ecmwf_ifs_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(ecmwf_ifs_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(ecmwf_ifs_fc)),
                    "url": (
                        f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}"
                        f"&longitude={city.lon}&hourly=temperature_2m&models=ecmwf_ifs"
                        f"&start_date={target_date_et}&end_date={target_date_et}&temperature_unit=fahrenheit"
                    ) if city.lat is not None else None,
                    "skill": source_skill.get("ecmwf_ifs"),
                },
                "ecmwf_aifs": {
                    "high_f": ecmwf_aifs_fc.high_f if ecmwf_aifs_fc else None,
                    "age_s": _age(ecmwf_aifs_fc.fetched_at if ecmwf_aifs_fc else None),
                    "collected_at": _fmt_time_et(ecmwf_aifs_fc.fetched_at if ecmwf_aifs_fc else None),
                    "model_run_at": _fmt_utc(ecmwf_aifs_fc.model_run_at if ecmwf_aifs_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(ecmwf_aifs_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(ecmwf_aifs_fc)),
                    "url": (
                        f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}"
                        f"&longitude={city.lon}&hourly=temperature_2m&models=ecmwf_aifs025_single"
                        f"&start_date={target_date_et}&end_date={target_date_et}&temperature_unit=fahrenheit"
                    ) if city.lat is not None else None,
                    "skill": source_skill.get("ecmwf_aifs"),
                    "experimental": True,
                },
                "hrrr_15min": {
                    "high_f": hrrr_15min_fc.high_f if hrrr_15min_fc else None,
                    "age_s": _age(hrrr_15min_fc.fetched_at if hrrr_15min_fc else None),
                    "collected_at": _fmt_time_et(hrrr_15min_fc.fetched_at if hrrr_15min_fc else None),
                    "model_run_at": _fmt_utc(hrrr_15min_fc.model_run_at if hrrr_15min_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(hrrr_15min_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(hrrr_15min_fc)),
                    "url": (
                        f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}"
                        f"&longitude={city.lon}&hourly=temperature_2m&models=ncep_hrrr_conus_15min"
                        f"&start_date={target_date_et}&end_date={target_date_et}&temperature_unit=fahrenheit"
                    ) if city.lat is not None else None,
                    "skill": source_skill.get("hrrr_15min"),
                },
                # Bayesian-upgrade Q3/U3 — DeepMind GraphCast (experimental, AI model).
                "gfs_graphcast": {
                    "high_f": gfs_graphcast_fc.high_f if gfs_graphcast_fc else None,
                    "age_s": _age(gfs_graphcast_fc.fetched_at if gfs_graphcast_fc else None),
                    "collected_at": _fmt_time_et(gfs_graphcast_fc.fetched_at if gfs_graphcast_fc else None),
                    "model_run_at": _fmt_utc(gfs_graphcast_fc.model_run_at if gfs_graphcast_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(gfs_graphcast_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(gfs_graphcast_fc)),
                    "url": (
                        f"https://api.open-meteo.com/v1/forecast?latitude={city.lat}"
                        f"&longitude={city.lon}&hourly=temperature_2m&models=gfs_graphcast025"
                        f"&start_date={target_date_et}&end_date={target_date_et}&temperature_unit=fahrenheit"
                    ) if city.lat is not None else None,
                    "skill": source_skill.get("gfs_graphcast"),
                    "experimental": True,
                },
                # §13 — Huawei Pangu-Weather, sourced from NOAA AIWP S3 archive.
                # IFS-initialized, 6-hour timestep. URL points to the AWS Open Data
                # registry page; no per-city query URL since this is a 3 GB NetCDF.
                "pangu_weather": {
                    "high_f": pangu_weather_fc.high_f if pangu_weather_fc else None,
                    "age_s": _age(pangu_weather_fc.fetched_at if pangu_weather_fc else None),
                    "collected_at": _fmt_time_et(pangu_weather_fc.fetched_at if pangu_weather_fc else None),
                    "model_run_at": _fmt_utc(pangu_weather_fc.model_run_at if pangu_weather_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(pangu_weather_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(pangu_weather_fc)),
                    "url": "https://registry.opendata.aws/aiwp/",
                    "skill": source_skill.get("pangu_weather"),
                    "experimental": True,
                },
                # §13 — NVIDIA FourCastNet v2-small, also from NOAA AIWP archive.
                "fourcastnet_v2": {
                    "high_f": fourcastnet_v2_fc.high_f if fourcastnet_v2_fc else None,
                    "age_s": _age(fourcastnet_v2_fc.fetched_at if fourcastnet_v2_fc else None),
                    "collected_at": _fmt_time_et(fourcastnet_v2_fc.fetched_at if fourcastnet_v2_fc else None),
                    "model_run_at": _fmt_utc(fourcastnet_v2_fc.model_run_at if fourcastnet_v2_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(fourcastnet_v2_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(fourcastnet_v2_fc)),
                    "url": "https://registry.opendata.aws/aiwp/",
                    "skill": source_skill.get("fourcastnet_v2"),
                    "experimental": True,
                },
                # §17 — Microsoft Aurora (Swin transformer), NOAA AIWP archive.
                "aurora": {
                    "high_f": aurora_fc.high_f if aurora_fc else None,
                    "age_s": _age(aurora_fc.fetched_at if aurora_fc else None),
                    "collected_at": _fmt_time_et(aurora_fc.fetched_at if aurora_fc else None),
                    "model_run_at": _fmt_utc(aurora_fc.model_run_at if aurora_fc else None),
                    "lead_time_hours": _lead_time_hours(_forecast_reference_time(aurora_fc)),
                    "model_run_age": _model_run_age(_forecast_reference_time(aurora_fc)),
                    "url": "https://registry.opendata.aws/aiwp/",
                    "skill": source_skill.get("aurora"),
                    "experimental": True,
                },
            },
            "event": event,
            "model": {
                "mu": model.mu if model else None,
                "sigma": model.sigma if model else None,
                "probs_json": probs_json,
                "inputs": model_inputs,
                # U1 — surface "Last computed" timestamp in the Model Forecast box.
                # Uses the same dual UTC + city-local format as the per-source
                # Model run popovers (_fmt_utc above is closure-scoped to city).
                "computed_at": _fmt_utc(model.computed_at) if model else None,
            } if model else None,
            "now_hour_et": now_local.hour,
            "city_tomorrow": city_local_tomorrow(city),
            "city_tz": getattr(city, "tz", "America/New_York"),
            "buckets": buckets_with_signals,
            "twe": round(sum(
                max(0.0, b["true_edge"])
                for b in buckets_with_signals
                if b.get("true_edge") is not None and (b.get("mkt_prob") or 0) >= 0.05
            ), 4),
            "reliability_json": json.dumps([
                {"expected": b.expected_prob, "observed": b.observed_prob, "count": b.count}
                for b in reliability_bins
            ]),
            "reliability_total_samples": sum(b.count for b in reliability_bins),
            "reliability_diag": reliability_diag,
            "obs_table": obs_table,
            "obs_table_json": json.dumps(obs_table),
            "station_predictions": station_predictions,
            "station_predictions_json": json.dumps(station_predictions),
            "hrrr_hourly_json": json.dumps(hrrr_hourly),
            "adaptive_info": adaptive_info,
            "obs_proximity": obs_proximity,
            "accuracy_status": accuracy_status,
            "wallet_leaderboard": wallet_leaderboard,
            "smart_money_context": smart_money_context,
            "station_cal": station_cal,
            "market_context_snapshot": serialize_market_context_snapshot(market_context_snapshot),
            "market_context_llm_ready": Config.market_context_llm_ready(),
            "lead_time_skills": lead_time_skills,
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


@dashboard_router.get("/admin/db", response_class=HTMLResponse)
async def db_admin(request: Request):
    return templates.TemplateResponse("db_admin.html", {"request": request})


@dashboard_router.get("/htmx/signals-table", response_class=HTMLResponse)
async def htmx_signals_table(request: Request):
    """HTMX partial — refreshes only the signals table body."""
    from backend.storage.db import get_session
    from backend.storage.repos import get_dashboard_signal_rows

    async with get_session() as sess:
        signal_context_rows = await get_dashboard_signal_rows(
            sess, limit=1000, date_et=et_today(),
        )

        rows = []
        seen = {}
        for row in signal_context_rows:
            sig = row["signal"]
            b = row["bucket"]
            c = row["city"]

            key = (c.city_slug, b.bucket_idx)
            if key in seen:
                continue
            seen[key] = True

            reason = json.loads(sig.reason_json) if sig.reason_json else {}
            gate_failures = json.loads(sig.gate_failures_json) if sig.gate_failures_json else []
            if reason.get("city_state") == "resolved":
                continue

            slug = c.city_slug
            rows.append({
                "city_slug": slug,
                "city_display": c.display_name,
                "unit": c.unit,
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
                "_reason": reason,
            })

    city_groups = _build_city_groups(rows)
    return templates.TemplateResponse("partials/signals_table.html", {"request": request, "city_groups": city_groups})


@dashboard_router.get("/htmx/redeem-summary", response_class=HTMLResponse)
async def htmx_redeem_summary(request: Request):
    """HTMX partial for settlement/redeem panels on the root dashboard."""
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_recently_redeemed_event_summaries,
        get_unredeemed_winning_position_summaries,
    )

    async with get_session() as sess:
        unredeemed_wins = await get_unredeemed_winning_position_summaries(sess)
        recent_redeems = await get_recently_redeemed_event_summaries(sess, days=7)
    return templates.TemplateResponse(
        "partials/redeem_summary.html",
        {
            "request": request,
            "unredeemed_wins": unredeemed_wins,
            "recent_redeems": recent_redeems,
        },
    )


@dashboard_router.get("/stations", response_class=HTMLResponse)
async def stations_page(request: Request):
    """Full-page view of all station calibrations with Leaflet map."""
    from backend.storage.db import get_session
    from backend.storage.repos import get_arming_state

    async with get_session() as sess:
        arming = await get_arming_state(sess)

    return templates.TemplateResponse(
        "stations.html",
        {
            "request": request,
            "arming_state": arming.state,
        },
    )


@dashboard_router.get("/redemptions", response_class=HTMLResponse)
async def redemptions_page(request: Request):
    """Full-page view of all events with positions and their redemption status."""
    return templates.TemplateResponse("redemptions.html", {"request": request})


# ─── Alpha Dashboard (Section 6 Layer 6) ──────────────────────────────────────
# Brier(model) − Brier(market) over settled events. The only metric that
# directly measures whether we add value over the market. Drives BMA promotion.

@dashboard_router.get("/calibration/edge", response_class=HTMLResponse)
async def calibration_edge_page(request: Request, days_back: int = 30):
    """Alpha dashboard — legacy vs BMA shadow vs market, by Brier."""
    from backend.modeling.edge_metrics import compute_edge_metrics

    metrics = await compute_edge_metrics(days_back=days_back)
    return templates.TemplateResponse(
        "calibration_edge.html",
        {
            "request": request,
            "metrics": metrics,
            "days_back": days_back,
            "metrics_json": json.dumps(metrics, default=str),
        },
    )


@dashboard_router.get("/api/calibration/edge.json")
async def calibration_edge_json(days_back: int = 30):
    """JSON API for the alpha dashboard (powers the chip on /)."""
    from backend.modeling.edge_metrics import compute_edge_metrics
    return await compute_edge_metrics(days_back=days_back)


@dashboard_router.get("/api/positions/{position_id}/exit-events")
async def position_exit_events(position_id: int):
    """Phase B4 — structured exit-event journal for a position.

    Returns every cascade decision the exit engine made on this position,
    with the full pre-trade context. Drives the position-journal UI.
    """
    import json as _json
    from sqlalchemy import select
    from backend.storage.db import get_session
    from backend.storage.models import ExitEvent

    async with get_session() as sess:
        result = await sess.execute(
            select(ExitEvent)
            .where(ExitEvent.position_id == position_id)
            .order_by(ExitEvent.ts.asc())
        )
        rows = list(result.scalars().all())

    events = []
    for r in rows:
        try:
            reason = _json.loads(r.reason_json) if r.reason_json else None
        except Exception:
            reason = None
        events.append({
            "id": r.id,
            "ts": r.ts.isoformat() if r.ts else None,
            "trigger_level": r.trigger_level,
            "trigger_reason": r.trigger_reason,
            "ev_at_bid_pre": r.ev_at_bid_pre,
            "ev_at_bid_post": r.ev_at_bid_post,
            "true_edge_pre": r.true_edge_pre,
            "true_edge_post": r.true_edge_post,
            "market_bid": r.market_bid,
            "market_ask": r.market_ask,
            "shares_exited": r.shares_exited,
            "shares_remaining": r.shares_remaining,
            "model_snapshot_id": r.model_snapshot_id,
            "reason": reason,
        })
    return {"position_id": position_id, "events": events}


@dashboard_router.get("/positions/{position_id}/journal", response_class=HTMLResponse)
async def position_journal_page(request: Request, position_id: int):
    """Read-only journal view of all exit events for a position."""
    import json as _json
    from sqlalchemy import select
    from backend.storage.db import get_session
    from backend.storage.models import ExitEvent, Position

    async with get_session() as sess:
        pos = (await sess.execute(
            select(Position).where(Position.id == position_id)
        )).scalar_one_or_none()
        events = list((await sess.execute(
            select(ExitEvent)
            .where(ExitEvent.position_id == position_id)
            .order_by(ExitEvent.ts.asc())
        )).scalars().all())

    rows_html = []
    for e in events:
        try:
            reason = _json.loads(e.reason_json) if e.reason_json else {}
        except Exception:
            reason = {}
        rows_html.append(
            f"<tr>"
            f"<td>{e.ts.isoformat(timespec='seconds') if e.ts else ''}</td>"
            f"<td><b>{e.trigger_level}</b></td>"
            f"<td>{e.trigger_reason}</td>"
            f"<td>{f'{e.ev_at_bid_pre:.4f}' if e.ev_at_bid_pre is not None else '—'}</td>"
            f"<td>{f'{e.market_bid:.3f}' if e.market_bid is not None else '—'}</td>"
            f"<td>{f'{e.market_ask:.3f}' if e.market_ask is not None else '—'}</td>"
            f"<td>{e.shares_exited:.1f}</td>"
            f"<td>{e.shares_remaining:.1f}</td>"
            f"<td><code style='font-size:11px'>{reason.get('execution_status','—')}</code></td>"
            f"</tr>"
        )
    pos_label = (
        f"Position {pos.id} · bucket {pos.bucket_id} · entry ${pos.avg_cost:.3f}"
        if pos else f"Position {position_id} (not found)"
    )
    body = (
        "<style>body{font-family:system-ui;padding:18px;max-width:1100px;margin:auto}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}"
        "th{background:#f4f4f4}h1{font-size:18px;margin-bottom:4px}"
        "p{color:#666;margin-top:0;margin-bottom:14px}</style>"
        f"<h1>Exit Journal — {pos_label}</h1>"
        f"<p>{len(events)} exit decision(s) recorded.</p>"
        "<table><thead><tr>"
        "<th>Timestamp (UTC)</th><th>Level</th><th>Reason</th>"
        "<th>EV@bid</th><th>Bid</th><th>Ask</th>"
        "<th>Exited</th><th>Remaining</th><th>Status</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )
    return HTMLResponse(body)


@dashboard_router.get("/herbie-timing", response_class=HTMLResponse)
async def herbie_timing_page(request: Request):
    """Phase C4 — read-only per-source latency + accuracy comparison vs Open-Meteo.

    Side-channel evidence for the day-90 promote/kill decision (plan §C4).
    Empty state is expected until herbie-data is installed and 24h+ has elapsed.
    """
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select, func
    from backend.storage.db import get_session
    from backend.storage.models import HerbieForecastTiming

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    async with get_session() as sess:
        rows = (
            await sess.execute(
                select(
                    HerbieForecastTiming.source,
                    func.count(HerbieForecastTiming.id).label("n"),
                    func.avg(HerbieForecastTiming.latency_delta_seconds).label("mean_latency_s"),
                    func.avg(HerbieForecastTiming.abs_diff_f).label("mean_abs_diff_f"),
                    func.avg(HerbieForecastTiming.herbie_mae_at_resolution).label("herbie_mae"),
                    func.avg(HerbieForecastTiming.open_meteo_mae_at_resolution).label("om_mae"),
                )
                .where(HerbieForecastTiming.herbie_fetched_at >= cutoff)
                .group_by(HerbieForecastTiming.source)
                .order_by(HerbieForecastTiming.source.asc())
            )
        ).all()

    def _fmt(v, fmt="{:.2f}"):
        return fmt.format(v) if v is not None else "—"

    rows_html = []
    for r in rows:
        latency_min = (r.mean_latency_s / 60.0) if r.mean_latency_s is not None else None
        rows_html.append(
            f"<tr>"
            f"<td><b>{r.source}</b></td>"
            f"<td>{r.n}</td>"
            f"<td>{_fmt(latency_min)} min</td>"
            f"<td>{_fmt(r.mean_abs_diff_f)} °F</td>"
            f"<td>{_fmt(r.herbie_mae)} °F</td>"
            f"<td>{_fmt(r.om_mae)} °F</td>"
            f"</tr>"
        )

    if not rows_html:
        rows_html = [
            "<tr><td colspan='6' style='text-align:center;color:#888;padding:24px'>"
            "No Herbie samples yet. Side-channel harness is enabled but has not yet "
            "captured a model run. First HRRR sample expected within 60 min; first "
            "IFS / AIFS within 6h of next 00 / 06 / 12 / 18z run.</td></tr>"
        ]

    body = (
        "<style>body{font-family:system-ui;padding:18px;max-width:1100px;margin:auto}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}"
        "th{background:#f4f4f4}h1{font-size:18px;margin-bottom:4px}"
        "p{color:#666;margin-top:0;margin-bottom:14px}"
        "code{background:#f4f4f4;padding:1px 4px;border-radius:3px}</style>"
        "<h1>Herbie side-channel — 30-day comparison</h1>"
        "<p>Latency and accuracy of direct Herbie fetches vs the production Open-Meteo path. "
        "Promote/kill rules per source documented in plan §C4. "
        "Positive latency = Herbie slower than Open-Meteo.</p>"
        "<table><thead><tr>"
        "<th>Source</th><th>n</th><th>Mean latency Δ</th>"
        "<th>Mean |Δhigh_f|</th><th>Herbie MAE</th><th>Open-Meteo MAE</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )
    return HTMLResponse(body)


@dashboard_router.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    """Strategy configuration, Kelly sizing, model schedules, and probability heatmap."""
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_arming_state,
        get_signals_for_latest_snapshot,
        get_all_cities,
    )
    from backend.storage.models import Bucket, Event
    from sqlalchemy import select

    async with get_session() as sess:
        arming = await get_arming_state(sess)
        raw_signals = await get_signals_for_latest_snapshot(
            sess, limit=1000, date_et=None,
        )
        cities = await get_all_cities(sess)

        bucket_ids = {sig.bucket_id for sig in raw_signals}
        if bucket_ids:
            bucket_rows = list((await sess.execute(
                select(Bucket).where(Bucket.id.in_(bucket_ids))
            )).scalars().all())
        else:
            bucket_rows = []
        bucket_map = {b.id: b for b in bucket_rows}

        event_ids = {b.event_id for b in bucket_rows}
        if event_ids:
            event_rows = list((await sess.execute(
                select(Event).where(Event.id.in_(event_ids))
            )).scalars().all())
        else:
            event_rows = []
        event_map = {e.id: e for e in event_rows}

    city_map = {c.id: c for c in cities}

    # Build bucket probability heatmap data grouped by city
    heatmap_data = {}  # city_slug -> list of bucket dicts
    for sig in raw_signals:
        bucket = bucket_map.get(sig.bucket_id)
        if not bucket:
            continue
        event = event_map.get(bucket.event_id)
        if not event:
            continue
        city = city_map.get(event.city_id)
        if not city or event.date_et != city_local_date(city):
            continue
        slug = city.city_slug
        if slug not in heatmap_data:
            heatmap_data[slug] = {"display_name": city.display_name, "buckets": []}
        heatmap_data[slug]["buckets"].append({
            "label": bucket.label or f"{bucket.low_f}-{bucket.high_f}",
            "low_f": bucket.low_f,
            "high_f": bucket.high_f,
            "bucket_idx": bucket.bucket_idx,
            "model_prob": round(sig.model_prob, 4),
            "mkt_prob": round(sig.mkt_prob if sig.mkt_prob else 0, 4),
            "true_edge": round(sig.true_edge, 4),
        })

    # Sort buckets within each city by bucket_idx
    for slug in heatmap_data:
        heatmap_data[slug]["buckets"].sort(key=lambda b: b["bucket_idx"])

    config_snapshot = {
        "kelly_fraction": Config.KELLY_FRACTION,
        "max_entry_price": Config.MAX_ENTRY_PRICE,
        "max_spread": Config.MAX_SPREAD,
        "min_true_edge": Config.MIN_TRUE_EDGE,
        "bankroll_cap": Config.BANKROLL_CAP,
        "max_daily_loss": Config.MAX_DAILY_LOSS,
        "max_positions_per_event": Config.MAX_POSITIONS_PER_EVENT,
        "trading_window_close_et": Config.TRADING_WINDOW_CLOSE_ET,
        # Exit engine params
        "quick_flip_target": Config.QUICK_FLIP_TARGET,
        "urgent_exit_max_spread": Config.URGENT_EXIT_MAX_SPREAD,
        "consensus_debounce_runs": Config.CONSENSUS_DEBOUNCE_RUNS,
        "expiry_discount": Config.EXPIRY_DISCOUNT,
        "obs_exit_enabled": Config.OBS_EXIT_ENABLED,
        "obs_exit_window_minutes": Config.OBS_EXIT_WINDOW_MINUTES,
        "temp_sensitivity_threshold_f": Config.TEMP_SENSITIVITY_THRESHOLD_F,
        "obs_min_profit_cents": Config.OBS_MIN_PROFIT_CENTS,
        "obs_reentry_cooldown_minutes": Config.OBS_REENTRY_COOLDOWN_MINUTES,
        "obs_min_depth_usd": Config.OBS_MIN_DEPTH_USD,
        "obs_max_orderbook_imbalance": Config.OBS_MAX_ORDERBOOK_IMBALANCE,
        # Telegram
        "telegram_enabled": Config.TELEGRAM_ENABLED,
    }

    # ── Build live model schedule data from Open-Meteo metadata ──────────────
    _MODEL_REGISTRY = [
        {"key": "ecmwf_ifs", "name": "ECMWF IFS", "resolution": "9 km", "alpha": "6–8h lag", "stub": False},
        {"key": "hrrr", "name": "HRRR CONUS", "resolution": "3 km", "alpha": "1–2h lag", "stub": False},
        {"key": "nbm", "name": "NCEP NBM", "resolution": "2.5 km", "alpha": "2–3h lag", "stub": False},
        {"key": "hrrr_15min", "name": "HRRR CONUS 15min*", "resolution": "3 km", "alpha": "1–2h lag", "stub": False},
        {"key": "gfs", "name": "GFS*", "resolution": "13 km", "alpha": "3–5h lag", "stub": True},
    ]

    now_utc = datetime.now(timezone.utc)

    def _fmt_age(ts: int | None) -> str | None:
        if not ts:
            return None
        delta = now_utc - datetime.fromtimestamp(ts, tz=timezone.utc)
        minutes = int(delta.total_seconds() // 60)
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"

    def _fmt_ts(ts: int | None) -> str | None:
        if not ts:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC")

    def _human_interval(sec: int | None) -> str:
        if not sec:
            return "unknown"
        if sec == 3600:
            return "1h"
        if sec == 21600:
            return "6h"
        if sec < 3600:
            return f"{sec // 60}m"
        return f"{sec // 3600}h"

    model_schedule_data = []
    for m in _MODEL_REGISTRY:
        meta = await fetch_openmeteo_metadata(m["key"]) if not m["stub"] else None
        init_ts = meta.get("last_run_initialisation_time") if meta else None
        avail_ts = meta.get("last_run_availability_time") if meta else None
        temp_res = meta.get("temporal_resolution_seconds") if meta else None
        upd_int = meta.get("update_interval_seconds") if meta else None

        # Status based on staleness
        status = "unknown"
        if m["stub"]:
            status = "stub"
        elif avail_ts:
            age_min = (now_utc - datetime.fromtimestamp(avail_ts, tz=timezone.utc)).total_seconds() / 60
            if age_min < 20:
                status = "live"
            elif age_min < 60:
                status = "delayed"
            else:
                status = "stale"

        model_schedule_data.append({
            "name": m["name"],
            "key": m["key"],
            "resolution": m["resolution"],
            "alpha": m["alpha"],
            "stub": m["stub"],
            "last_model_run": _fmt_ts(init_ts),
            "update_available": _fmt_ts(avail_ts),
            "update_age": _fmt_age(avail_ts),
            "temporal_resolution": _human_interval(temp_res),
            "temporal_resolution_sec": temp_res,
            "update_frequency": _human_interval(upd_int),
            "update_frequency_sec": upd_int,
            "api_link": _OM_META_ENDPOINTS.get(m["key"], ""),
            "status": status,
        })

    return templates.TemplateResponse(
        "strategies.html",
        {
            "request": request,
            "arming_state": arming.state,
            "config": config_snapshot,
            "heatmap_data": heatmap_data,
            "model_schedule_data": model_schedule_data,
        },
    )


# ─── Backtest ────────────────────────────────────────────────────────────────

@dashboard_router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    """Walk-forward backtester dashboard."""
    from backend.storage.db import get_session
    from backend.storage.models import BacktestRun
    from backend.backtesting.engine import (
        BacktestParams,
        get_coverage_breakdown,
    )
    from sqlalchemy import select, desc

    async with get_session() as sess:
        runs_q = select(BacktestRun).order_by(desc(BacktestRun.created_at)).limit(20)
        runs = (await sess.execute(runs_q)).scalars().all()

    # Three-tier coverage report — what the page actually needs to render
    # honest counts and the empty-state notice.
    coverage = await get_coverage_breakdown()

    arming_state = "DISARMED"
    try:
        from backend.storage.repos import get_arming_state
        async with get_session() as sess:
            arm = await get_arming_state(sess)
            arming_state = arm.state
    except Exception:
        pass

    from dataclasses import asdict

    return templates.TemplateResponse(
        "backtest.html",
        {
            "request": request,
            "arming_state": arming_state,
            "runs": runs,
            "coverage": coverage,
            "default_params": asdict(BacktestParams()),
        },
    )


@dashboard_router.get("/api/backtest/coverage")
async def api_backtest_coverage():
    """Three-tier coverage report for the /backtest dashboard banner."""
    from backend.backtesting.engine import get_coverage_breakdown
    return await get_coverage_breakdown()


@dashboard_router.post("/api/backtest/run")
async def api_run_backtest(request: Request):
    """Launch a backtest run (async background task)."""
    import asyncio
    from backend.backtesting.engine import (
        BacktestEngine,
        BacktestParams,
        count_resolved_sources,
    )
    from backend.storage.models import BacktestRun
    from backend.storage.db import get_session
    from dataclasses import fields, asdict

    body = await request.json()

    # Build params from request, only accepting known fields
    valid_keys = {f.name for f in fields(BacktestParams)}
    filtered = {}
    for k, v in body.items():
        if k in valid_keys:
            filtered[k] = v
    params = BacktestParams(**filtered)

    # Create run record
    async with get_session() as sess:
        run = BacktestRun(
            params_json=json.dumps(asdict(params)),
            start_date="pending",
            end_date="pending",
            status="running",
        )
        sess.add(run)
        await sess.commit()
        await sess.refresh(run)
        run_id = run.id

    # Debug visibility: report how many resolved events exist from each source
    # before we spin up the background engine.
    try:
        counts = await count_resolved_sources()
        print(
            f"[backtest] run_id={run_id} "
            f"resolved_local={counts['resolved_local']} "
            f"resolved_gamma={counts['resolved_gamma']}"
        )
    except Exception as e:
        print(f"[backtest] run_id={run_id} resolved-source-count error: {e}")

    # Run in background
    async def _bg():
        engine = BacktestEngine(params)
        await engine.run(run_id=run_id)

    asyncio.create_task(_bg())

    return {"run_id": run_id, "status": "running"}


@dashboard_router.get("/api/backtest/{run_id:int}")
async def api_get_backtest(run_id: int):
    """Poll backtest status/results."""
    from backend.storage.models import BacktestRun
    from backend.storage.db import get_session

    async with get_session() as sess:
        run = await sess.get(BacktestRun, run_id)

    if not run:
        return {"error": "not_found"}

    result = {"status": run.status, "run_id": run.id}

    if run.status == "completed" and run.results_json:
        result.update(json.loads(run.results_json))
    elif run.status == "failed":
        result["error"] = run.error_msg

    return result


@dashboard_router.post("/api/backtest/enrich-gamma")
async def api_enrich_gamma():
    """Fetch resolved weather markets from Gamma API to enrich local DB."""
    from backend.backtesting.engine import enrich_from_gamma
    result = await enrich_from_gamma()
    return {
        "fetched": result.fetched,
        "weather_matched": result.weather_matched,
        "stored": result.stored,
        "matched_events": result.matched_events,
        "matched_metar": result.matched_metar,
        "matched_forecast": result.matched_forecast,
        "last_enriched_at": result.last_enriched_at.isoformat(),
        # Back-compat field for any older clients
        "enriched": result.matched_events,
    }


@dashboard_router.get("/api/backtest/enrichment-status")
async def api_backtest_enrichment_status():
    """Return the "last enriched" summary used by the banner on /backtest."""
    from backend.backtesting.engine import get_enrichment_status
    return await get_enrichment_status()
