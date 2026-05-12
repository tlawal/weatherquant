"""DB plumbing for fitted BMA weights (M1 Phase 2).

- `get_bma_weights_for_city(sess, city_id, lead_bucket_hours)` — read fitted
  weights at trade time. Returns `{source: weight}` or None when no fit exists
  for this (city, lead_bucket) — caller falls back to legacy weights then.

- `upsert_bma_weights(sess, city_id, lead_bucket_hours, fit_result, sigma_by_source)`
  — replace-on-write for the fitter. Atomic per (city, lead_bucket): deletes
  any prior rows, then inserts the new fit. This avoids stale-source rows
  hanging around when the source set shrinks (e.g. a model goes offline).

- `load_training_data_for_city(sess, city_id, lead_bucket_hours, days_back)`
  — assemble (forecasts_dict, observed_y) tuples from settled events.

The loader joins ForecastObs (one row per source × event) to settlement highs
by date_et + city. Settlement source priority matches the existing
`compute_source_lead_time_skills` in calibration_engine.py: MetarObs daily
high first, then wu_history fallback.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modeling.bma import BMAFitResult, BMA_MIN_N_FOR_SIGMA
from backend.storage.models import BMAWeights, City, Event, ForecastObs

log = logging.getLogger(__name__)


async def get_bma_weights_for_city(
    sess: AsyncSession,
    city_id: int,
    lead_bucket_hours: int,
) -> Optional[dict[str, float]]:
    """Return fitted {source: weight} or None when no fit exists yet."""
    rows = (
        await sess.execute(
            select(BMAWeights).where(
                BMAWeights.city_id == city_id,
                BMAWeights.lead_time_bucket_hours == lead_bucket_hours,
            )
        )
    ).scalars().all()
    if not rows:
        return None
    return {r.source: float(r.weight) for r in rows}


async def upsert_bma_weights(
    sess: AsyncSession,
    *,
    city_id: int,
    lead_bucket_hours: int,
    fit_result: BMAFitResult,
    sigma_by_source: dict[str, float],
) -> int:
    """Replace any prior fit for (city_id, lead_bucket) with `fit_result`.

    Returns the number of rows written. Caller commits.
    """
    await sess.execute(
        delete(BMAWeights).where(
            BMAWeights.city_id == city_id,
            BMAWeights.lead_time_bucket_hours == lead_bucket_hours,
        )
    )

    n_written = 0
    for source, weight in fit_result.weights.items():
        sigma = float(sigma_by_source.get(source, 0.0))
        if sigma <= 0:
            # Defensive: skip sources without a σ (would crash the fuse-time
            # consumer). Should not occur in practice — the fitter discards
            # σ-less sources before iteration.
            log.warning(
                "bma_weights: skipping write for city_id=%s source=%s lead=%s — no σ",
                city_id, source, lead_bucket_hours,
            )
            continue
        sess.add(BMAWeights(
            city_id=city_id,
            source=source,
            lead_time_bucket_hours=lead_bucket_hours,
            weight=float(weight),
            sigma_f=sigma,
            n_obs=fit_result.n_obs,
            log_likelihood=fit_result.log_likelihood,
            n_iter=fit_result.n_iter,
            converged=fit_result.converged,
            fitted_at=datetime.now(timezone.utc),
        ))
        n_written += 1
    return n_written


# ──────────────────── Training-data loader ──────────────────────────────────

# Lead-bucket assignment matches calibration_engine._LEAD_TIME_BUCKETS so the
# σᵢ from SourceLeadTimeSkill aligns with the wᵢ produced here.
_LEAD_BUCKETS = (72, 48, 36, 24, 18, 12, 6, 3, 1, 0)


async def _settlement_high_for_event(
    sess: AsyncSession, city_id: int, date_et: str
) -> Optional[float]:
    """Resolve the continuous settlement high used for BMA training.

    Delegates to the lead-skill settlement oracle so BMA σ/weights and
    SourceLeadTimeSkill are trained against the same target. A Polymarket
    winner mismatch returns None so disputed dates do not nudge continuous
    temperature weights.
    """
    from backend.modeling.calibration_engine import resolve_canonical_settlement_high

    city = await sess.get(City, city_id)
    if city is None:
        return None
    event = (
        await sess.execute(
            select(Event).where(
                Event.city_id == city_id,
                Event.date_et == date_et,
            )
        )
    ).scalar_one_or_none()
    if event is None:
        return None
    result = await resolve_canonical_settlement_high(
        sess, city=city, event=event, validate_polymarket_winner=True,
    )
    if result.get("matches_polymarket_winner") is False:
        return None
    high = result.get("high_f")
    return float(high) if high is not None else None


async def load_training_data_for_city(
    sess: AsyncSession,
    city_id: int,
    lead_bucket_hours: int,
    days_back: int = 90,
) -> list[tuple[dict[str, float], float]]:
    """Build (forecasts_dict, observed_y) tuples for EM fitting.

    For each past event in the lookback window:
      1. Compute the canonical settlement high.
      2. Compute the fixed decision checkpoint for `lead_bucket_hours`.
      3. Keep one forecast per source: latest row available by that checkpoint.

    This mirrors SourceLeadTimeSkill so BMA fitting does not treat frequent
    refresh cadence as extra independent evidence.
    """
    from backend.modeling.calibration_engine import (
        FORECAST_SKILL_SOURCES,
        select_latest_forecasts_by_checkpoint,
    )

    city = await sess.get(City, city_id)
    if city is None:
        return []
    city_tz = getattr(city, "tz", "America/New_York") or "America/New_York"
    now_local = datetime.now(ZoneInfo(city_tz))
    today_et = now_local.strftime("%Y-%m-%d")
    cutoff = (now_local - timedelta(days=days_back)).strftime("%Y-%m-%d")

    events = (
        await sess.execute(
            select(Event).where(
                Event.city_id == city_id,
                Event.date_et < today_et,
                Event.date_et >= cutoff,
            )
        )
    ).scalars().all()

    training: list[tuple[dict[str, float], float]] = []

    for event in events:
        observed_y = await _settlement_high_for_event(sess, city_id, event.date_et)
        if observed_y is None:
            continue

        event_end_local = datetime.strptime(event.date_et, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59,
            tzinfo=ZoneInfo(city_tz),
        )
        event_end_utc = event_end_local.astimezone(timezone.utc)
        checkpoint_utc = event_end_utc - timedelta(hours=lead_bucket_hours)

        forecasts = (
            await sess.execute(
                select(ForecastObs).where(
                    ForecastObs.city_id == city_id,
                    ForecastObs.date_et == event.date_et,
                    ForecastObs.source.in_(FORECAST_SKILL_SOURCES),
                    ForecastObs.high_f.isnot(None),
                )
            )
        ).scalars().all()

        per_source_latest = select_latest_forecasts_by_checkpoint(
            list(forecasts), checkpoint_utc,
        )

        if not per_source_latest:
            continue

        forecasts_dict: dict[str, float] = {
            src: float(fc.high_f) for src, fc in per_source_latest.items() if fc.high_f is not None
        }
        if forecasts_dict:
            training.append((forecasts_dict, observed_y))

    return training


# ──────────────────── Sigma loader (shared with predictive) ─────────────────

async def apply_online_em_for_event(
    sess: AsyncSession,
    *,
    city_id: int,
    event_id: int,
    date_et: str,
    lr: float = 0.05,
) -> dict[int, int]:
    """Apply one online-EM step per (lead_bucket) using this event's forecasts
    + settlement. Touches every BMAWeights row whose lead_bucket has at least
    one matching forecast for this event.

    Returns: {lead_bucket: n_sources_updated} — useful for logging.

    Caller is responsible for committing and for marking the event as
    processed (Event.bma_online_processed_at). This function only mutates
    BMAWeights.
    """
    from backend.modeling.bma import online_em_step
    from backend.modeling.calibration_engine import (
        FORECAST_SKILL_SOURCES,
        select_latest_forecasts_by_checkpoint,
    )
    from backend.storage.models import BMAWeights, ForecastObs, Event

    # Settlement high (same priority as Phase 2 loader).
    observed_y = await _settlement_high_for_event(sess, city_id, date_et)
    if observed_y is None:
        return {}

    event = (await sess.execute(select(Event).where(Event.id == event_id))).scalar_one_or_none()
    if event is None:
        return {}

    city = await sess.get(City, city_id)
    city_tz = getattr(city, "tz", "America/New_York") if city else "America/New_York"
    event_end_local = datetime.strptime(date_et, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59,
        tzinfo=ZoneInfo(city_tz or "America/New_York"),
    )
    event_end_utc = event_end_local.astimezone(timezone.utc)

    forecasts = (
        await sess.execute(
            select(ForecastObs).where(
                ForecastObs.city_id == city_id,
                ForecastObs.date_et == date_et,
                ForecastObs.source.in_(FORECAST_SKILL_SOURCES),
                ForecastObs.high_f.isnot(None),
            )
        )
    ).scalars().all()

    # Group: {lead_bucket: {source: μ_ik}} using fixed decision checkpoints.
    by_bucket: dict[int, dict[str, float]] = {}
    for bucket in _LEAD_BUCKETS:
        checkpoint_utc = event_end_utc - timedelta(hours=bucket)
        selected = select_latest_forecasts_by_checkpoint(
            list(forecasts), checkpoint_utc,
        )
        if selected:
            by_bucket[bucket] = {
                src: float(fc.high_f)
                for src, fc in selected.items()
                if fc.high_f is not None
            }

    summary: dict[int, int] = {}
    for lead_bucket, forecasts_dict in by_bucket.items():
        # Pull current weights + σ for this bucket.
        rows = (
            await sess.execute(
                select(BMAWeights).where(
                    BMAWeights.city_id == city_id,
                    BMAWeights.lead_time_bucket_hours == lead_bucket,
                )
            )
        ).scalars().all()
        if not rows:
            # No fitted weights at this bucket yet — nothing to update online.
            # The nightly batch will create them once enough data accumulates.
            continue

        current_weights = {r.source: float(r.weight) for r in rows}
        sigma_by_source = {r.source: float(r.sigma_f) for r in rows}

        updated = online_em_step(
            current_weights=current_weights,
            forecasts=forecasts_dict,
            observed_y=observed_y,
            sigma_by_source=sigma_by_source,
            lr=lr,
        )

        # Write back. Only mutate weights — σ stays whatever the offline
        # fitter wrote (Phase 3 intentionally does not update σ; that's M3
        # plan territory).
        n_updated = 0
        for r in rows:
            new_w = updated.get(r.source)
            if new_w is not None and abs(new_w - r.weight) > 1e-9:
                r.weight = float(new_w)
                r.n_obs = int(r.n_obs or 0) + 1
                r.fitted_at = datetime.now(timezone.utc)
                n_updated += 1
        summary[lead_bucket] = n_updated

    return summary


async def load_sigma_by_source_for_city(
    sess: AsyncSession,
    city_id: int,
    lead_bucket_hours: int,
) -> dict[str, float]:
    """Pull σᵢ for each source from SourceLeadTimeSkill at this lead bucket.

    Sources without a row at this lead are excluded from the fit (the fitter
    drops sources missing a σ). Caller can decide to include them with a prior
    σ if desired.
    """
    from backend.storage.models import SourceLeadTimeSkill

    rows = (
        await sess.execute(
            select(SourceLeadTimeSkill).where(
                SourceLeadTimeSkill.city_id == city_id,
                SourceLeadTimeSkill.lead_time_bucket_hours == lead_bucket_hours,
            )
        )
    ).scalars().all()
    return {
        r.source: float(r.mae_f)
        for r in rows
        if (
            r.mae_f is not None
            and r.mae_f > 0
            and int(r.n_obs or 0) >= BMA_MIN_N_FOR_SIGMA
        )
    }
