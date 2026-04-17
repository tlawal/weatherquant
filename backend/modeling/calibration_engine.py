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
    """
    Adjust a raw model probability based on historical reliability bins.
    Uses simple linear interpolation between bin centers.
    """
    if not bins or all(b.count == 0 for b in bins):
        return prob
        
    bin_idx = min(int(prob * 10), 9)
    target_bin = bins[bin_idx]
    
    if target_bin.count < 5: # Need a minimum forest of samples
        return prob
        
    # Simple multiplier: if observed is 0.7 and expected is 0.9, we multiply by 0.7/0.9
    # This is a naive 'Platt-like' scaling.
    correction = target_bin.observed_prob / target_bin.expected_prob if target_bin.expected_prob > 0 else 1.0
    
    # Dampen the correction to avoid wild swings from small samples
    weight = min(target_bin.count / 20, 1.0)
    final_prob = (1.0 - weight) * prob + weight * (prob * correction)
    
    return max(0.0, min(1.0, final_prob))
