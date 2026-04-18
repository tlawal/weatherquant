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
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.db import get_session
from backend.storage.models import (
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
# Forecast sources participating in the ensemble
ENSEMBLE_SOURCES = ("nws", "wu_hourly", "hrrr", "nbm", "ecmwf_ifs")
# Default fallback weights (match prior hardcoded defaults in temperature_model.py)
DEFAULT_WEIGHTS: dict[str, float] = {
    "nws": 0.5,
    "wu_hourly": 0.5,
    "hrrr": 0.5,
    "nbm": 0.2,
    "ecmwf_ifs": 0.5,
}


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
    has n_samples < 3 (cold-start protection). The model side also applies
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
        if r.n_samples >= 3 and r.weight > 0:
            weights[r.source] = float(r.weight)
            biases[r.source] = float(r.bias_f or 0.0)
    return weights, biases


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

    # Observed highs per date
    obs_highs: dict[str, float] = {}
    async with get_session() as sess:
        for d in dates:
            start_dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=tz)
            end_dt = start_dt + timedelta(days=1)
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

        # Forecast highs per (source, date) — latest fetch
        fc: dict[tuple[str, str], float] = {}
        for src in ENSEMBLE_SOURCES:
            for d in dates:
                val = (
                    await sess.execute(
                        select(ForecastObs.high_f)
                        .where(
                            ForecastObs.city_id == city_id,
                            ForecastObs.source == src,
                            ForecastObs.date_et == d,
                            ForecastObs.high_f.isnot(None),
                        )
                        .order_by(ForecastObs.fetched_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if val is not None:
                    fc[(src, d)] = float(val)

        # Existing rows → skip duplicates
        existing = (
            await sess.execute(
                select(
                    ForecastDailyError.source,
                    ForecastDailyError.date_et,
                ).where(ForecastDailyError.station_id == station_id)
            )
        ).all()
        existing_set = {(s, d) for s, d in existing}

        written = 0
        for (src, d), fval in fc.items():
            obs = obs_highs.get(d)
            if obs is None or (src, d) in existing_set:
                continue
            sess.add(
                ForecastDailyError(
                    station_id=station_id,
                    date_et=d,
                    source=src,
                    forecast_high_f=fval,
                    observed_high_f=obs,
                    err_f=fval - obs,
                )
            )
            written += 1
        if written:
            await sess.commit()
        return written
