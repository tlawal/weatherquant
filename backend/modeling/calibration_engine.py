"""
Probability calibration engine — analyzes model reliability and remaps probabilities.

This module compares historical model-predicted probabilities against actual outcomes
to identify overconfidence or bias, allowing for re-calibration of live signals.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

from sqlalchemy import select, desc, func
from backend.storage.db import get_session
from backend.storage.models import ModelSnapshot, Event, ForecastObs, Bucket
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

async def get_reliability_metrics(city_id: int, days_back: int = 30) -> List[ReliabilityBin]:
    """
    Compute reliability metrics (observed vs expected probability) for a city.
    Analyzes historical ModelSnapshots vs ForecastObs(wu_history).
    """
    # Create bins: 0-10%, 10-20%, ..., 90-100%
    bins = [ReliabilityBin(i/10, (i+1)/10) for i in range(10)]
    
    async with get_session() as sess:
        # Subquery: latest ModelSnapshot per event (one per event, not hundreds)
        latest_snap_sub = (
            select(
                ModelSnapshot.event_id,
                func.max(ModelSnapshot.id).label("max_snap_id"),
            )
            .group_by(ModelSnapshot.event_id)
            .subquery()
        )

        # Subquery: latest wu_history ForecastObs per (city_id, date_et)
        latest_wu_sub = (
            select(
                ForecastObs.city_id,
                ForecastObs.date_et,
                func.max(ForecastObs.id).label("max_fo_id"),
            )
            .where(ForecastObs.source == "wu_history")
            .group_by(ForecastObs.city_id, ForecastObs.date_et)
            .subquery()
        )

        # Only settled past events (exclude today)
        from datetime import date, timezone, datetime
        today_et = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        query = (
            select(ModelSnapshot, ForecastObs.high_f, Event.id)
            .join(latest_snap_sub, ModelSnapshot.id == latest_snap_sub.c.max_snap_id)
            .join(Event, ModelSnapshot.event_id == Event.id)
            .join(
                latest_wu_sub,
                (latest_wu_sub.c.city_id == Event.city_id)
                & (latest_wu_sub.c.date_et == Event.date_et),
            )
            .join(
                ForecastObs,
                ForecastObs.id == latest_wu_sub.c.max_fo_id,
            )
            .where(Event.city_id == city_id)
            .where(Event.date_et < today_et)
            .where(Event.resolved_at.isnot(None))
            .order_by(desc(Event.date_et))
            .limit(200)
        )

        results = (await sess.execute(query)).all()
        if len(results) == 0:
            log.debug("calibration: city_id=%d found 0 settled events with wu_history", city_id)
        else:
            log.info("calibration: city_id=%d found %d settled events with wu_history", city_id, len(results))

        # We also need the bucket boundaries for each event to check for "hits"
        event_ids = list(set([r[2] for r in results]))
        event_buckets = {}
        for eid in event_ids:
            b_query = select(Bucket).where(Bucket.event_id == eid).order_by(Bucket.bucket_idx)
            event_buckets[eid] = (await sess.execute(b_query)).scalars().all()

    for snap, realized_high, event_id in results:
        if realized_high is None:
            continue
            
        probs = json.loads(snap.probs_json)
        buckets = event_buckets.get(event_id, [])
        
        if len(probs) != len(buckets):
            continue
            
        rh_val = realized_high
        if rh_val is None:
            continue
        rh: float = float(rh_val)
        
        canonical = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
        hit = find_bucket_idx_for_value(canonical, rh)
        hit_idx = hit if hit is not None else -1
        
        if hit_idx == -1:
            continue
            
        # Add to bins
        for i, p in enumerate(probs):
            bin_idx = min(int(p * 10), 9)
            bins[bin_idx].count += 1
            if i == hit_idx:
                bins[bin_idx].hits += 1
                
    total = sum(b.count for b in bins)
    log.info("calibration: city_id=%d total_samples=%d bins_with_data=%d",
             city_id, total, sum(1 for b in bins if b.count > 0))
    return bins

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
