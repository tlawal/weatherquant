"""
Per-station 30-day rolling forecast calibration.

Compares fused ensemble forecasts vs. observed METAR daily highs
to compute MAE, bias, RMSE, and tradeability per ICAO station.
Inspired by br0br0/weather-oracle calibration tables.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select, func, distinct

from backend.storage.db import get_session
from backend.storage.models import (
    City, Event, ForecastObs, MetarObs, ModelSnapshot, Order, Bucket, Position,
)

log = logging.getLogger(__name__)

# Tradeability thresholds (°F MAE)
_GREEN_THRESHOLD = 1.5
_RED_THRESHOLD = 3.0


def _classify_tradeability(mae: float) -> str:
    if mae <= _GREEN_THRESHOLD:
        return "GREEN"
    elif mae <= _RED_THRESHOLD:
        return "AMBER"
    return "RED"


async def _get_daily_highs(
    city_id: int,
    city_tz: str,
    dates: list[str],
) -> dict[str, float]:
    """Get observed METAR daily high for each date_et."""
    highs: dict[str, float] = {}
    tz = ZoneInfo(city_tz)

    async with get_session() as sess:
        for date_et in dates:
            start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
            end_dt = start_dt + timedelta(days=1)
            result = await sess.execute(
                select(func.max(MetarObs.temp_f))
                .where(
                    MetarObs.city_id == city_id,
                    MetarObs.temp_f.isnot(None),
                    MetarObs.observed_at >= start_dt,
                    MetarObs.observed_at < end_dt,
                )
            )
            val = result.scalar_one_or_none()
            if val is not None:
                highs[date_et] = float(val)

    return highs


async def _get_forecast_highs(
    city_id: int,
    dates: list[str],
    sources: list[str] = ("nws", "wu_hourly", "hrrr", "nbm"),
) -> dict[str, dict[str, float]]:
    """For each date, get each source's latest forecast high_f.

    Returns: {date_et: {source: high_f}}
    """
    result: dict[str, dict[str, float]] = defaultdict(dict)

    async with get_session() as sess:
        for source in sources:
            for date_et in dates:
                row = await sess.execute(
                    select(ForecastObs.high_f)
                    .where(
                        ForecastObs.city_id == city_id,
                        ForecastObs.source == source,
                        ForecastObs.date_et == date_et,
                        ForecastObs.high_f.isnot(None),
                    )
                    .order_by(ForecastObs.fetched_at.desc())
                    .limit(1)
                )
                val = row.scalar_one_or_none()
                if val is not None:
                    result[date_et][source] = float(val)

    return dict(result)


async def _get_model_snapshot_highs(
    city_id: int,
    dates: list[str],
) -> dict[str, float]:
    """Get the model's fused forecast (mu) from the latest ModelSnapshot per event/date."""
    result: dict[str, float] = {}

    async with get_session() as sess:
        for date_et in dates:
            # Find event for city+date
            event_q = await sess.execute(
                select(Event.id).where(
                    Event.city_id == city_id,
                    Event.date_et == date_et,
                )
            )
            event_id = event_q.scalar_one_or_none()
            if event_id is None:
                continue

            # Latest model snapshot
            snap_q = await sess.execute(
                select(ModelSnapshot.mu)
                .where(ModelSnapshot.event_id == event_id)
                .order_by(ModelSnapshot.computed_at.desc())
                .limit(1)
            )
            mu = snap_q.scalar_one_or_none()
            if mu is not None:
                result[date_et] = float(mu)

    return result


async def _get_pct_days_traded(
    city_id: int,
    dates: list[str],
) -> float:
    """Compute % of dates that had filled orders."""
    traded_dates = set()

    async with get_session() as sess:
        for date_et in dates:
            result = await sess.execute(
                select(func.count(Order.id))
                .join(Bucket, Bucket.id == Order.bucket_id)
                .join(Event, Event.id == Bucket.event_id)
                .where(
                    Event.city_id == city_id,
                    Event.date_et == date_et,
                    Order.status == "filled",
                )
            )
            count = result.scalar_one_or_none() or 0
            if count > 0:
                traded_dates.add(date_et)

    if not dates:
        return 0.0
    return round(len(traded_dates) / len(dates) * 100, 1)


async def compute_station_calibration(
    city: City,
    *,
    min_samples: int = 1,
    max_days: int = 30,
) -> Optional[dict]:
    """Compute rolling calibration metrics for a single city/station.

    Args:
        city: The city to compute for.
        min_samples: Minimum observed days required. Default 1 (fallback mode);
            was implicitly 3 previously. With 1, any station that has at least one
            day of obs + forecast gets a row — `n_samples < 30` signals fallback.
        max_days: Size of rolling window. Default 30. Fewer obs than max_days is
            normal in fallback mode.

    Returns a dict of kwargs for upsert_station_calibration, or None if
    even the relaxed threshold is not met.
    """
    station_id = city.metar_station
    if not station_id:
        return None

    city_tz = getattr(city, "tz", "America/New_York")
    now = datetime.now(ZoneInfo(city_tz))
    today_str = now.strftime("%Y-%m-%d")

    # Build list of past max_days dates (excluding today — not settled yet)
    dates = [
        (now - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, max_days + 1)
    ]

    # Fetch observed highs
    obs_highs = await _get_daily_highs(city.id, city_tz, dates)
    if len(obs_highs) < min_samples:
        log.info(
            "station_cal: %s only %d obs highs (< %d min), skipping",
            station_id, len(obs_highs), min_samples,
        )
        return None

    # Fetch per-source forecast highs (includes ecmwf_ifs for model comparison)
    sources = ["nws", "wu_hourly", "hrrr", "hrrr_15min", "nbm", "ecmwf_ifs"]
    fc_highs = await _get_forecast_highs(city.id, dates, sources)

    # Fetch model mu (fused ensemble)
    model_highs = await _get_model_snapshot_highs(city.id, dates)

    # ── Compute per-source errors ───────────────────────────────────────────
    source_errors: dict[str, list[float]] = defaultdict(list)
    fused_errors: list[float] = []

    for date_et in dates:
        obs = obs_highs.get(date_et)
        if obs is None:
            continue

        # Per-source
        fc = fc_highs.get(date_et, {})
        for src in sources:
            if src in fc:
                source_errors[src].append(fc[src] - obs)

        # Fused ensemble (model mu)
        fused = model_highs.get(date_et)
        if fused is not None:
            fused_errors.append(fused - obs)
        else:
            # fallback: average all available forecasts for this date
            if fc:
                avg_fc = sum(fc.values()) / len(fc)
                fused_errors.append(avg_fc - obs)

    if not fused_errors:
        log.info(
            "station_cal: %s no fused errors (obs_dates=%d, fc_dates=%d)",
            station_id, len(obs_highs), len(fc_highs),
        )
        return None

    # ── Aggregate metrics ──────────────────────────────────────────────────
    mae = sum(abs(e) for e in fused_errors) / len(fused_errors)
    bias = sum(fused_errors) / len(fused_errors)
    rmse = math.sqrt(sum(e * e for e in fused_errors) / len(fused_errors))

    # Per-source MAE
    source_mae: dict[str, float] = {}
    for src, errs in source_errors.items():
        if errs:
            source_mae[src] = round(sum(abs(e) for e in errs) / len(errs), 2)

    best_source = min(source_mae, key=source_mae.get, default=None) if source_mae else None
    best_source_mae = source_mae.get(best_source) if best_source else None

    # Per-model MAE for 5-way comparison (ECMWF vs GFS+HRRR vs NWS vs WU vs NBM)
    mae_ecmwf = source_mae.get("ecmwf_ifs")
    mae_gfs_hrrr = source_mae.get("hrrr")     # hrrr = GFS+HRRR blend via Open-Meteo
    mae_nws = source_mae.get("nws")            # NWS WFO human-adjusted forecast
    mae_wu_hourly = source_mae.get("wu_hourly")  # Weather Underground Hourly
    mae_nbm = source_mae.get("nbm")              # NCEP National Blend of Models

    # Determine 5-way winner (lowest MAE; TIE if within 0.05°F)
    candidates: dict[str, float] = {}
    if mae_ecmwf is not None:
        candidates["ECMWF"] = mae_ecmwf
    if mae_gfs_hrrr is not None:
        candidates["GFS_HRRR"] = mae_gfs_hrrr
    if mae_nws is not None:
        candidates["NWS"] = mae_nws
    if mae_wu_hourly is not None:
        candidates["WU_HOURLY"] = mae_wu_hourly
    if mae_nbm is not None:
        candidates["NBM"] = mae_nbm

    winner = None
    if candidates:
        min_mae = min(candidates.values())
        winners = [k for k, v in candidates.items() if abs(v - min_mae) < 0.05]
        winner = "TIE" if len(winners) > 1 else winners[0]

    # Tradeability
    tradeability = _classify_tradeability(mae)

    # % days traded
    pct_traded = await _get_pct_days_traded(city.id, dates)

    # is_fallback is derived client-side from n_samples < 30
    # but we include days_window_used for future configurability
    return {
        "city_slug": city.city_slug,
        "city_name": city.display_name,
        "lat": city.lat,
        "lon": city.lon,
        "mae_f": round(mae, 2),
        "bias_f": round(bias, 2),
        "rmse_f": round(rmse, 2),
        "n_samples": len(fused_errors),
        "pct_days_traded": pct_traded,
        "tradeability": tradeability,
        "best_source": best_source,
        "best_source_mae": best_source_mae,
        "source_mae_json": json.dumps(source_mae) if source_mae else None,
        "mae_ecmwf_f": mae_ecmwf,
        "mae_gfs_hrrr_f": mae_gfs_hrrr,
        "mae_nws_f": mae_nws,
        "mae_wu_hourly_f": mae_wu_hourly,
        "mae_nbm_f": mae_nbm,
        "winner": winner,
    }


async def refresh_all_station_calibrations(min_samples: int = 1) -> int:
    """Recompute rolling calibration for all enabled cities.

    Args:
        min_samples: Minimum observed days required per station. Default 1
            allows fallback-mode rows for stations with < 30 days of history.
            Set higher (e.g. 3 or 10) for stricter quality gating.

    Returns number of stations updated.
    """
    from backend.storage.repos import get_all_cities, upsert_station_calibration

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    from backend.modeling.station_weights import (
        backfill_forecast_daily_errors,
        update_station_weights,
    )

    updated = 0
    for city in cities:
        if not city.metar_station:
            continue
        try:
            cal_data = await compute_station_calibration(city, min_samples=min_samples)
            if cal_data is None:
                continue
            async with get_session() as sess:
                await upsert_station_calibration(sess, city.metar_station, **cal_data)
            updated += 1
            log.info(
                "station_cal: %s mae=%.2f bias=%+.2f tradeability=%s samples=%d",
                city.metar_station, cal_data["mae_f"], cal_data["bias_f"],
                cal_data["tradeability"], cal_data["n_samples"],
            )
            # Refresh ForecastDailyError + dynamic per-source weights.
            try:
                city_tz = getattr(city, "tz", "America/New_York")
                await backfill_forecast_daily_errors(
                    city.metar_station, city.id, city_tz
                )
                await update_station_weights(city.metar_station)
            except Exception:
                log.exception(
                    "station_weights: %s update failed", city.metar_station
                )
        except Exception as e:
            log.error("station_cal: %s failed: %s", city.metar_station, e, exc_info=True)

    log.info("station_cal: refreshed %d stations", updated)
    return updated
