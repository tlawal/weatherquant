"""
Probability calibration engine — analyzes model reliability and remaps probabilities.

This module compares historical model-predicted probabilities against actual outcomes
to identify overconfidence or bias, allowing for re-calibration of live signals.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

from sqlalchemy import select, desc, func
from backend.storage.db import get_session
from backend.storage.models import ModelSnapshot, Event, ForecastObs, Bucket, MetarObs, City
from backend.modeling.settlement import canonical_bucket_ranges, find_bucket_idx_for_value

log = logging.getLogger(__name__)

@dataclass
class ReliabilityBin:
    min_prob: float
    max_prob: float
    count: int = 0
    hits: int = 0
    
    @property
    def expected_prob(self) -> float:
        return (self.min_prob + self.max_prob) / 2
    
    @property
    def observed_prob(self) -> float:
        return self.hits / self.count if self.count > 0 else 0.0

async def get_reliability_metrics(city_id: int, days_back: int = 90) -> List[ReliabilityBin]:
    """
    Compute reliability metrics (observed vs expected probability) for a city.

    Ground truth for the realized daily high is resolved per-event in this order:
      1. MetarObs daily max within the event's local-tz day  (primary — most reliable)
      2. wu_history ForecastObs.high_f for the same date_et  (fallback)

    The previous version gated on `Event.resolved_at IS NOT NULL` (set by the
    Polymarket redeemer). That coupled calibration to trade/settlement activity —
    events without a Polymarket market, or events where the redeemer cron hadn't
    run, were invisible to calibration. We now only require:
      - Event.date_et < today (must be a past day)
      - Event has a ModelSnapshot (have a prediction to score)
      - Event has Bucket rows (bucket boundaries to decide "which bucket won")
      - Ground-truth high exists via MetarObs or wu_history
    """
    # Create bins: 0-10%, 10-20%, ..., 90-100%
    bins = [ReliabilityBin(i/10, (i+1)/10) for i in range(10)]

    today_et = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    async with get_session() as sess:
        # Resolve city timezone once — needed for the MetarObs local-day window.
        city_tz_row = await sess.execute(select(City.tz).where(City.id == city_id))
        city_tz = city_tz_row.scalar_one_or_none() or "America/New_York"
        try:
            tz = ZoneInfo(city_tz)
        except Exception:
            tz = ZoneInfo("America/New_York")

        # Latest ModelSnapshot per event.
        latest_snap_sub = (
            select(
                ModelSnapshot.event_id,
                func.max(ModelSnapshot.id).label("max_snap_id"),
            )
            .group_by(ModelSnapshot.event_id)
            .subquery()
        )

        # Past events with a snapshot — no resolved_at requirement.
        query = (
            select(ModelSnapshot, Event.id, Event.date_et)
            .join(latest_snap_sub, ModelSnapshot.id == latest_snap_sub.c.max_snap_id)
            .join(Event, ModelSnapshot.event_id == Event.id)
            .where(Event.city_id == city_id)
            .where(Event.date_et < today_et)
            .where(Event.date_et >= cutoff)
            .order_by(desc(Event.date_et))
            .limit(200)
        )
        results = (await sess.execute(query)).all()

        # Pre-fetch buckets for each event (avoid N+1 if possible — simple loop ok for ≤200).
        event_ids = list({r[1] for r in results})
        event_buckets: Dict[int, list] = {}
        for eid in event_ids:
            b_query = select(Bucket).where(Bucket.event_id == eid).order_by(Bucket.bucket_idx)
            event_buckets[eid] = (await sess.execute(b_query)).scalars().all()

        async def _resolved_high(date_et: str) -> Optional[float]:
            """METAR daily max (primary) or wu_history (fallback)."""
            try:
                start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
            except ValueError:
                return None
            end_dt = start_dt + timedelta(days=1)
            metar_q = await sess.execute(
                select(func.max(MetarObs.temp_f)).where(
                    MetarObs.city_id == city_id,
                    MetarObs.temp_f.isnot(None),
                    MetarObs.observed_at >= start_dt,
                    MetarObs.observed_at < end_dt,
                )
            )
            v = metar_q.scalar_one_or_none()
            if v is not None:
                return float(v)
            wu_q = await sess.execute(
                select(ForecastObs.high_f)
                .where(
                    ForecastObs.city_id == city_id,
                    ForecastObs.date_et == date_et,
                    ForecastObs.source == "wu_history",
                    ForecastObs.high_f.isnot(None),
                )
                .order_by(desc(ForecastObs.id))
                .limit(1)
            )
            v = wu_q.scalar_one_or_none()
            return float(v) if v is not None else None

        if not results:
            log.debug("calibration: city_id=%d no past events with snapshots in last %dd",
                      city_id, days_back)
            return bins

        matched = 0
        for snap, event_id, date_et in results:
            buckets = event_buckets.get(event_id, [])
            if not buckets:
                continue
            rh = await _resolved_high(date_et)
            if rh is None:
                continue
            try:
                probs = json.loads(snap.probs_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if len(probs) != len(buckets):
                continue
            canonical = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
            hit = find_bucket_idx_for_value(canonical, rh)
            if hit is None:
                continue
            matched += 1
            for i, p in enumerate(probs):
                bin_idx = min(int(p * 10), 9)
                bins[bin_idx].count += 1
                if i == hit:
                    bins[bin_idx].hits += 1

        total = sum(b.count for b in bins)
        log.info(
            "calibration: city_id=%d events_in_window=%d matched=%d total_samples=%d bins_with_data=%d",
            city_id, len(results), matched, total, sum(1 for b in bins if b.count > 0),
        )
    return bins

async def get_reliability_diagnostics(city_id: int) -> dict:
    """
    Report exactly which calibration filter is short-handed for this city.

    Filters now (post-decoupling):
      - Event.date_et < today  AND  ModelSnapshot exists
      - Event has Bucket rows
      - MetarObs daily high OR wu_history exists for the event date

    `resolved_events` / `last_resolved_at` are retained as INFORMATIONAL only
    — they track redeemer activity but no longer gate calibration.
    """
    today_et = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async with get_session() as sess:
        total_events = (await sess.execute(
            select(func.count(Event.id)).where(Event.city_id == city_id)
        )).scalar_one()

        past_events = (await sess.execute(
            select(func.count(Event.id)).where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

        # Past events with at least one ModelSnapshot.
        past_with_snap = (await sess.execute(
            select(func.count(func.distinct(Event.id)))
            .join(ModelSnapshot, ModelSnapshot.event_id == Event.id)
            .where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

        # Past events with at least one Bucket row (bucket boundaries required).
        past_with_buckets = (await sess.execute(
            select(func.count(func.distinct(Event.id)))
            .join(Bucket, Bucket.event_id == Event.id)
            .where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

        # Past event dates with a MetarObs row. Approximation: match by
        # UTC-date string of observed_at against date_et. Not perfectly TZ-aware
        # but close enough for a diagnostic count (the actual calibration uses
        # the proper city-tz window).
        # For cross-dialect safety we pull distinct event dates + distinct
        # observed-at dates and intersect in Python.
        past_event_dates_rows = (await sess.execute(
            select(func.distinct(Event.date_et))
            .where(Event.city_id == city_id, Event.date_et < today_et)
        )).all()
        past_event_dates = {r[0] for r in past_event_dates_rows}

        # Pull raw observed_at timestamps where temp_f is not null, bucket into
        # YYYY-MM-DD in Python. UTC-day approximation is fine for a diagnostic
        # count (true calibration resolves via the city-tz window).
        metar_times = (await sess.execute(
            select(MetarObs.observed_at).where(
                MetarObs.city_id == city_id,
                MetarObs.temp_f.isnot(None),
            )
        )).scalars().all()
        metar_days = {t.strftime("%Y-%m-%d") for t in metar_times if t is not None}
        past_dates_with_metar_high = len(past_event_dates & metar_days)

        with_wu = (await sess.execute(
            select(func.count(func.distinct(ForecastObs.date_et))).where(
                ForecastObs.city_id == city_id,
                ForecastObs.source == "wu_history",
                ForecastObs.high_f.isnot(None),
            )
        )).scalar_one()

        resolved = (await sess.execute(
            select(func.count(Event.id)).where(
                Event.city_id == city_id,
                Event.date_et < today_et,
                Event.resolved_at.isnot(None),
            )
        )).scalar_one()

        last_resolved = (await sess.execute(
            select(func.max(Event.resolved_at)).where(Event.city_id == city_id)
        )).scalar_one()

        last_event_date = (await sess.execute(
            select(func.max(Event.date_et)).where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

    return {
        "total_events": int(total_events or 0),
        "past_events": int(past_events or 0),
        "past_events_with_snapshot": int(past_with_snap or 0),
        "past_events_with_buckets": int(past_with_buckets or 0),
        "past_dates_with_metar_high": int(past_dates_with_metar_high or 0),
        "dates_with_wu_history": int(with_wu or 0),
        "resolved_events": int(resolved or 0),  # informational
        "last_resolved_at": last_resolved.isoformat() if last_resolved else None,
        "last_event_date": last_event_date if last_event_date else None,
        # back-compat alias so older template copies don't KeyError during rolling deploy
        "events_with_snapshot": int(past_with_snap or 0),
    }


def remap_probability(prob: float, bins: List[ReliabilityBin]) -> float:
    """Adjust a raw model probability using isotonic regression calibration.

    Isotonic regression (Gneiting & Raftery 2007) learns a monotone
    non-parametric mapping from predicted → observed probabilities.
    It works well with small samples and guarantees reliability
    (calibrated predictions are monotonically related to raw ones).

    Falls back to identity (no correction) when < 15 total samples.
    """
    if not bins or all(b.count == 0 for b in bins):
        return prob

    # Build training data from reliability bins
    xs: List[float] = []  # predicted probabilities (bin centers)
    ys: List[float] = []  # observed probabilities
    ws: List[float] = []  # sample weights
    total_samples = 0

    for b in bins:
        if b.count >= 2:  # need at least 2 samples per bin to be meaningful
            xs.append(b.expected_prob)
            ys.append(b.observed_prob)
            ws.append(float(b.count))
            total_samples += b.count

    if total_samples < 15 or len(xs) < 3:
        return prob  # insufficient data — identity pass-through

    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(
            y_min=0.0, y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        ir.fit(xs, ys, sample_weight=ws)

        calibrated = float(ir.predict([prob])[0])

        # Dampen the correction — blend 70% calibrated + 30% raw
        # to avoid overfitting on limited historical data
        dampened = 0.70 * calibrated + 0.30 * prob
        return max(0.0, min(1.0, dampened))
    except Exception:
        # sklearn not available or fit failed — fall back to identity
        log.debug("isotonic calibration failed, using raw probability", exc_info=True)
        return prob


# ─── Lead-Time Skill Analysis ─────────────────────────────────────────────────

_LEAD_TIME_BUCKETS = [72, 48, 36, 24, 18, 12, 6, 3, 1, 0]


def _bucket_lead_time(hours: float) -> int:
    """Round lead time to the nearest bucket."""
    for bucket in _LEAD_TIME_BUCKETS:
        if hours >= bucket:
            return bucket
    return 0


async def compute_source_lead_time_skills(
    city_id: int,
    days_back: int = 90,
    min_obs_per_bucket: int = 5,
) -> dict:
    """Compute MAE and bias per forecast source at each lead-time bucket.

    Joins ForecastObs (with model_run_at) to resolved Event settlement highs.
    Lead time = hours between model_run_at and the end of the event day (23:59:59
    local time, approximated as midnight ET + timezone offset).

    Returns a dict keyed by (source, lead_time_bucket) with {mae, bias, n}.
    """
    from backend.storage.models import SourceLeadTimeSkill
    from backend.storage.repos import get_daily_high_metar

    today_et = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    async with get_session() as sess:
        # Get resolved events for this city in the lookback window
        event_stmt = (
            select(Event)
            .where(
                Event.city_id == city_id,
                Event.date_et < today_et,
                Event.date_et >= cutoff,
                Event.winning_bucket_idx.isnot(None),
            )
        )
        events = (await sess.execute(event_stmt)).scalars().all()

        if not events:
            log.info("lead_time_skill: no resolved events for city_id=%s in last %d days", city_id, days_back)
            return {}

        # Collect forecast obs for these events that have model_run_at
        results: dict = {}
        for event in events:
            # Settlement high: try MetarObs first, then wu_history
            obs_high = await get_daily_high_metar(sess, city_id, event.date_et)
            if obs_high is None:
                # Fallback to wu_history
                wu_stmt = (
                    select(ForecastObs)
                    .where(
                        ForecastObs.city_id == city_id,
                        ForecastObs.source == "wu_history",
                        ForecastObs.date_et == event.date_et,
                        ForecastObs.high_f.isnot(None),
                    )
                    .order_by(desc(ForecastObs.fetched_at))
                )
                wu_rec = (await sess.execute(wu_stmt)).scalars().first()
                obs_high = wu_rec.high_f if wu_rec else None

            if obs_high is None:
                continue

            # Get all forecast obs with model_run_at for this event
            fc_stmt = (
                select(ForecastObs)
                .where(
                    ForecastObs.city_id == city_id,
                    ForecastObs.date_et == event.date_et,
                    ForecastObs.model_run_at.isnot(None),
                    ForecastObs.high_f.isnot(None),
                )
            )
            forecasts = (await sess.execute(fc_stmt)).scalars().all()

            for fc in forecasts:
                # Approximate event end time as midnight ET of date_et
                event_end = datetime.strptime(event.date_et, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59, tzinfo=ZoneInfo("America/New_York")
                )
                # Ensure model_run_at has timezone info
                model_run = fc.model_run_at
                if model_run.tzinfo is None:
                    model_run = model_run.replace(tzinfo=timezone.utc)

                lead_hours = (event_end - model_run).total_seconds() / 3600
                if lead_hours < -1:  # Sanity check: model run shouldn't be after event end
                    continue
                lead_hours = max(0, lead_hours)

                bucket = _bucket_lead_time(lead_hours)
                key = (fc.source, bucket)

                if key not in results:
                    results[key] = {"errors": [], "n": 0}

                error = fc.high_f - obs_high
                results[key]["errors"].append(error)
                results[key]["n"] += 1

        # Compute MAE and bias per bucket
        skills = {}
        for (source, bucket), data in results.items():
            if data["n"] < min_obs_per_bucket:
                continue
            errors = data["errors"]
            mae = sum(abs(e) for e in errors) / len(errors)
            bias = sum(errors) / len(errors)
            skills[(source, bucket)] = {
                "source": source,
                "lead_time_bucket_hours": bucket,
                "mae_f": round(mae, 2),
                "bias_f": round(bias, 2),
                "n_obs": data["n"],
            }

            # Upsert to database
            existing = await sess.execute(
                select(SourceLeadTimeSkill).where(
                    SourceLeadTimeSkill.city_id == city_id,
                    SourceLeadTimeSkill.source == source,
                    SourceLeadTimeSkill.lead_time_bucket_hours == bucket,
                )
            )
            existing = existing.scalar_one_or_none()

            if existing:
                existing.mae_f = round(mae, 2)
                existing.bias_f = round(bias, 2)
                existing.n_obs = data["n"]
                existing.computed_at = datetime.now(timezone.utc)
            else:
                skill = SourceLeadTimeSkill(
                    city_id=city_id,
                    source=source,
                    lead_time_bucket_hours=bucket,
                    mae_f=round(mae, 2),
                    bias_f=round(bias, 2),
                    n_obs=data["n"],
                )
                sess.add(skill)

        await sess.commit()

        log.info(
            "lead_time_skill: city_id=%s computed %d source/bucket combos from %d events",
            city_id, len(skills), len(events),
        )
        return skills

