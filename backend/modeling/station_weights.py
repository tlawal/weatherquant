"""
Dynamic per-station ensemble weights from forecast skill.

Each (station, source) pair tracks a fast EWMA (α=0.5, ~1-day half-life) and
slow EWMA (α=0.067, ~10-day half-life) of squared error, blended into an
effective variance. Weights are inverse-variance, normalized across available
sources, with empirical-Bayes shrinkage toward a uniform prior when n_samples
is low (so cold-start stations fall back gracefully to default weights).

See plan: /Users/larry/.claude/plans/i-m-looking-at-https-weatherquant-up-rai-rustling-brooks.md
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select, func
from backend.storage.db import get_session
from backend.storage.models import (
    Event,
    ForecastDailyError,
    ForecastObs,
    MetarObs,
    StationSourceWeight,
)

log = logging.getLogger(__name__)

# EWMA alphas — fast reacts in ~1 day, slow in ~10 days
ALPHA_FAST = 0.5
ALPHA_SLOW = 0.067
ALPHA_BIAS = 0.1
# Blend between fast and slow MSE (higher = more reactive)
BLEND_FAST = 0.4
# Empirical-Bayes shrinkage constant (n needed for half weight toward own data)
SHRINKAGE_K = 7
# Global prior MSE (°F²) used while station has little history
GLOBAL_MSE_PRIOR = 9.0  # ≈ 3°F RMSE, typical day-ahead NWP error
# Clamp bounds on final normalized weight so no source dominates or zeros
WEIGHT_FLOOR = 0.05
WEIGHT_CAP = 0.60
# Live-trading safety gates. Station weights and biases are learned from daily
# errors, and 3-5 rows is enough to be actively harmful. Keep live probabilities
# on defaults until the station/source pair has a minimally credible sample.
MIN_LIVE_STATION_SOURCE_SAMPLES = 10
MAX_LIVE_STATION_BIAS_ABS_F = 3.0
MAX_LIVE_STATION_BIAS_MAE_7D_F = 4.0
# Forecast sources participating in the live ensemble. Keep this aligned with
# station_calibration.py and signal_engine.py so every traded source can receive
# per-station bias/weight corrections once it has errors.
ENSEMBLE_SOURCES = (
    "nws", "wu_hourly", "hrrr", "hrrr_15min", "nbm", "ecmwf_ifs",
    "ecmwf_aifs", "gfs_graphcast", "pangu_weather", "fourcastnet_v2", "aurora",
)
# Default fallback weights (match prior hardcoded defaults in temperature_model.py)
DEFAULT_WEIGHTS: dict[str, float] = {
    "nws": 0.5,
    "wu_hourly": 0.5,
    "hrrr": 0.5,
    "hrrr_15min": 0.35,
    "nbm": 0.2,
    "ecmwf_ifs": 0.5,
    "ecmwf_aifs": 0.4,
    "gfs_graphcast": 0.4,
    "pangu_weather": 0.35,
    "fourcastnet_v2": 0.35,
    "aurora": 0.35,
}


def _clamp_station_bias(bias_f: Optional[float]) -> float:
    """Clamp dynamic station bias corrections to a sane live-trading range."""
    bias = float(bias_f or 0.0)
    return max(-MAX_LIVE_STATION_BIAS_ABS_F, min(MAX_LIVE_STATION_BIAS_ABS_F, bias))


def _use_live_station_weight(row: StationSourceWeight) -> bool:
    """Whether a station/source row is mature enough to move live weights."""
    return int(row.n_samples or 0) >= MIN_LIVE_STATION_SOURCE_SAMPLES and float(row.weight or 0.0) > 0.0


def _use_live_station_bias(row: StationSourceWeight) -> bool:
    """Whether a station/source row is mature and accurate enough to debias."""
    if not _use_live_station_weight(row):
        return False
    if row.mae_7d is not None and float(row.mae_7d) > MAX_LIVE_STATION_BIAS_MAE_7D_F:
        return False
    return True


def _compute_weights_from_stats(
    per_source_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Turn per-source {mse_fast, mse_slow, n_samples} into normalized weights.

    Pure function — no DB. Used by both the settlement updater and tests.
    """
    if not per_source_stats:
        return {}

    raw: dict[str, float] = {}
    for src, st in per_source_stats.items():
        n = int(st.get("n_samples", 0))
        mse_hat = (
            BLEND_FAST * st.get("mse_fast", GLOBAL_MSE_PRIOR)
            + (1 - BLEND_FAST) * st.get("mse_slow", GLOBAL_MSE_PRIOR)
        )
        lam = n / (n + SHRINKAGE_K)
        var_used = lam * mse_hat + (1 - lam) * GLOBAL_MSE_PRIOR
        raw[src] = 1.0 / (var_used + 1e-6)

    total = sum(raw.values())
    if total <= 0:
        return {}
    normalized = {src: w / total for src, w in raw.items()}

    # Clamp and renormalize
    clamped = {
        src: max(WEIGHT_FLOOR, min(WEIGHT_CAP, w))
        for src, w in normalized.items()
    }
    total2 = sum(clamped.values())
    return {src: w / total2 for src, w in clamped.items()} if total2 > 0 else {}


async def update_station_weights(station_id: str, *, lookback_days: int = 60) -> int:
    """Rebuild per-source EWMAs and weights for one station from ForecastDailyError.

    We iterate daily errors chronologically to compute EWMAs from scratch each
    run — simple and deterministic given only ~60 rows per source. Returns the
    number of source rows written.
    """
    async with get_session() as sess:
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()
        rows = (
            await sess.execute(
                select(ForecastDailyError)
                .where(
                    ForecastDailyError.station_id == station_id,
                    ForecastDailyError.date_et >= cutoff,
                )
                .order_by(ForecastDailyError.date_et.asc())
            )
        ).scalars().all()

    if not rows:
        return 0

    by_source: dict[str, list[ForecastDailyError]] = defaultdict(list)
    for r in rows:
        by_source[r.source].append(r)

    per_source_stats: dict[str, dict[str, float]] = {}
    per_source_extras: dict[str, dict[str, Optional[float]]] = {}

    for src, errs in by_source.items():
        mse_fast = 0.0
        mse_slow = 0.0
        bias = 0.0
        for i, r in enumerate(errs):
            e = r.err_f
            if i == 0:
                mse_fast = e * e
                mse_slow = e * e
                bias = e
            else:
                mse_fast = ALPHA_FAST * e * e + (1 - ALPHA_FAST) * mse_fast
                mse_slow = ALPHA_SLOW * e * e + (1 - ALPHA_SLOW) * mse_slow
                bias = ALPHA_BIAS * e + (1 - ALPHA_BIAS) * bias

        # 7-day and 30-day MAE for tooltip
        last7 = errs[-7:]
        last30 = errs[-30:]
        mae_7 = sum(abs(e.err_f) for e in last7) / len(last7) if last7 else None
        mae_30 = sum(abs(e.err_f) for e in last30) / len(last30) if last30 else None
        yesterday_err = errs[-1].err_f if errs else None

        per_source_stats[src] = {
            "mse_fast": mse_fast,
            "mse_slow": mse_slow,
            "n_samples": len(errs),
        }
        per_source_extras[src] = {
            "bias_f": bias,
            "mae_7d": mae_7,
            "mae_30d": mae_30,
            "yesterday_err_f": yesterday_err,
        }

    weights = _compute_weights_from_stats(per_source_stats)

    async with get_session() as sess:
        existing = (
            await sess.execute(
                select(StationSourceWeight).where(
                    StationSourceWeight.station_id == station_id
                )
            )
        ).scalars().all()
        by_src = {w.source: w for w in existing}

        for src, stats in per_source_stats.items():
            extras = per_source_extras[src]
            weight = weights.get(src, 0.0)
            row = by_src.get(src)
            if row is None:
                row = StationSourceWeight(station_id=station_id, source=src)
                sess.add(row)
            row.weight = weight
            row.mse_fast = stats["mse_fast"]
            row.mse_slow = stats["mse_slow"]
            row.n_samples = int(stats["n_samples"])
            row.bias_f = float(extras["bias_f"] or 0.0)
            row.mae_7d = extras["mae_7d"]
            row.mae_30d = extras["mae_30d"]
            row.yesterday_err_f = extras["yesterday_err_f"]
        await sess.commit()

    log.info(
        "station_weights: %s updated — sources=%s",
        station_id,
        {s: round(w, 3) for s, w in weights.items()},
    )
    return len(per_source_stats)


async def load_station_source_weights(
    station_id: Optional[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (weights, biases) dicts for use by compute_model.

    Falls back to default weights / zero bias when station has no row or a source
    has n_samples < MIN_LIVE_STATION_SOURCE_SAMPLES. The model side also applies
    DEFAULT_WEIGHTS when a source is missing from the returned dict.
    """
    weights: dict[str, float] = {}
    biases: dict[str, float] = {}
    if not station_id:
        return weights, biases

    async with get_session() as sess:
        rows = (
            await sess.execute(
                select(StationSourceWeight).where(
                    StationSourceWeight.station_id == station_id
                )
            )
        ).scalars().all()

    for r in rows:
        if _use_live_station_weight(r):
            weights[r.source] = float(r.weight)
        if _use_live_station_bias(r):
            biases[r.source] = _clamp_station_bias(r.bias_f)
    return weights, biases


def _day_window(date_et: str, city_tz: str) -> tuple[datetime, datetime, datetime]:
    tz = ZoneInfo(city_tz)
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)
    return start_dt, end_dt, end_dt.astimezone(timezone.utc)


async def load_source_skill_summary(
    station_id: Optional[str],
) -> dict[str, dict[str, Optional[float]]]:
    """Return per-source skill dict for the tooltip/API.

    {source: {weight, bias_f, mae_7d, mae_30d, yesterday_err_f, n_samples}}
    """
    out: dict[str, dict[str, Optional[float]]] = {}
    if not station_id:
        return out
    async with get_session() as sess:
        rows = (
            await sess.execute(
                select(StationSourceWeight).where(
                    StationSourceWeight.station_id == station_id
                )
            )
        ).scalars().all()
    for r in rows:
        out[r.source] = {
            "weight": float(r.weight) if r.weight is not None else None,
            "bias_f": float(r.bias_f) if r.bias_f is not None else None,
            "mae_7d": r.mae_7d,
            "mae_30d": r.mae_30d,
            "yesterday_err_f": r.yesterday_err_f,
            "n_samples": int(r.n_samples),
        }
    return out


async def backfill_forecast_daily_errors(
    station_id: str,
    city_id: int,
    city_tz: str,
    *,
    max_days: int = 60,
) -> int:
    """Backfill ForecastDailyError rows from ForecastObs + MetarObs daily highs.

    Writes one row per (station, date_et, source) pair that has both a forecast
    high and an observed high available. Safe to re-run — upserts on the unique
    constraint (station_id, date_et, source).
    """
    tz = ZoneInfo(city_tz)
    now = datetime.now(tz)
    dates = [
        (now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, max_days + 1)
    ]

    # Observed highs per date. Prefer the event's explicit resolution station,
    # then WU history, then any raw METAR daily max for the city. This aligns
    # station weights with the settlement basis instead of whichever raw ASOS
    # reading happened to be largest.
    obs_highs: dict[str, float] = {}
    async with get_session() as sess:
        event_rows = (
            await sess.execute(
                select(Event).where(
                    Event.city_id == city_id,
                    Event.date_et.in_(dates),
                )
            )
        ).scalars().all()
        events_by_date = {e.date_et: e for e in event_rows}
        event_end_by_date: dict[str, datetime] = {}

        for d in dates:
            start_dt, end_dt, event_end_utc = _day_window(d, city_tz)
            event_end_by_date[d] = event_end_utc
            event = events_by_date.get(d)
            target_station = (
                (event.resolution_station_id if event else None)
                or station_id
                or ""
            ).upper()
            val = (
                await sess.execute(
                    select(func.max(MetarObs.temp_f)).where(
                        MetarObs.city_id == city_id,
                        MetarObs.metar_station == target_station,
                        MetarObs.temp_f.isnot(None),
                        MetarObs.observed_at >= start_dt,
                        MetarObs.observed_at < end_dt,
                    )
                )
            ).scalar_one_or_none()
            if val is None:
                val = (
                    await sess.execute(
                        select(ForecastObs.high_f)
                        .where(
                            ForecastObs.city_id == city_id,
                            ForecastObs.source == "wu_history",
                            ForecastObs.date_et == d,
                            ForecastObs.high_f.isnot(None),
                        )
                        .order_by(ForecastObs.fetched_at.desc(), ForecastObs.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
            if val is None:
                val = (
                    await sess.execute(
                        select(func.max(MetarObs.temp_f)).where(
                            MetarObs.city_id == city_id,
                            MetarObs.temp_f.isnot(None),
                            MetarObs.observed_at >= start_dt,
                            MetarObs.observed_at < end_dt,
                        )
                    )
                ).scalar_one_or_none()
            if val is not None:
                obs_highs[d] = float(val)

        if not obs_highs:
            return 0

        # Forecast highs per (source, date) — latest fetch available before the
        # event day ends. This prevents late/revised rows from rewriting history.
        fc: dict[tuple[str, str], float] = {}
        for src in ENSEMBLE_SOURCES:
            for d in dates:
                cutoff_utc = event_end_by_date.get(d)
                val = (
                    await sess.execute(
                        select(ForecastObs.high_f)
                        .where(
                            ForecastObs.city_id == city_id,
                            ForecastObs.source == src,
                            ForecastObs.date_et == d,
                            ForecastObs.high_f.isnot(None),
                            ForecastObs.fetched_at <= cutoff_utc,
                        )
                        .order_by(ForecastObs.fetched_at.desc(), ForecastObs.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if val is not None:
                    fc[(src, d)] = float(val)

        # Existing rows → true upsert/update instead of skipping duplicates.
        existing = (
            await sess.execute(
                select(ForecastDailyError).where(
                    ForecastDailyError.station_id == station_id,
                    ForecastDailyError.date_et.in_(dates),
                )
            )
        ).scalars().all()
        existing_by_key = {(r.source, r.date_et): r for r in existing}

        written = 0
        for (src, d), fval in fc.items():
            obs = obs_highs.get(d)
            if obs is None:
                continue
            err = fval - obs
            row = existing_by_key.get((src, d))
            if row is None:
                row = ForecastDailyError(
                    station_id=station_id,
                    date_et=d,
                    source=src,
                )
                sess.add(row)
            row.forecast_high_f = fval
            row.observed_high_f = obs
            row.err_f = err
            written += 1
        if written:
            await sess.commit()
        return written
