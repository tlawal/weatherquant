"""Alpha dashboard — Brier(model) − Brier(market).

Section 6 Layer 6 of the Bayesian upgrade plan: the only metric that directly
measures whether we add value over the market. Without this, BMA promotion has
no quantitative basis.

Three forecasting paths are scored against settled outcomes:

- **Legacy model** — `ModelSnapshot.probs_json` (drives trades today).
- **BMA shadow** — `ModelSnapshot.inputs_json["bma_shadow"]["probs"]`
  (computed since `e4835da`, does not drive trades).
- **Market** — latest `MarketSnapshot.yes_mid` per bucket.

For each settled `Event` (`winning_bucket_idx IS NOT NULL`):

  outcome_b = 1 if b.bucket_idx == event.winning_bucket_idx else 0
  brier_b   = (prob_b − outcome_b)²

Aggregate per source:
  mean_brier      = Σ brier / N
  edge_vs_market  = mean_brier_market − mean_brier_source     (positive = wins)

This module is pure read-side: no DB writes, no scheduler hooks. Called from
`/calibration/edge` and `/api/calibration/edge.json`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.backtesting.metrics import compute_brier
from backend.storage.db import get_session
from backend.storage.models import (
    Bucket,
    City,
    Event,
    MarketSnapshot,
    ModelSnapshot,
)

log = logging.getLogger(__name__)


# Minimum samples in any subgroup before we report a metric. Below this we
# return null in the per-city / per-day buckets to avoid noisy headline numbers.
DEFAULT_MIN_N_BUCKETS = 10

# Headline edge is reported in basis points (×10000) for readability:
# Brier difference of 0.0036 → "+36 bps".
BPS = 10000


@dataclass
class _BucketScore:
    """One settled bucket scored by all three sources."""
    event_id: int
    city_slug: str
    date_et: str                     # YYYY-MM-DD
    bucket_idx: int
    outcome: int                     # 0 or 1
    legacy_prob: Optional[float]     # None if probs_json missing this idx
    bma_prob: Optional[float]        # None if no bma_shadow yet
    market_prob: Optional[float]     # None if no MarketSnapshot
    bma_between_share: Optional[float] = None  # for per-regime breakouts


# ───────────────────────── Aggregation helpers ──────────────────────────────

def _safe_brier(pairs: list[tuple[float, int]]) -> Optional[float]:
    """Mean Brier score; None when no samples (don't pretend zero)."""
    if not pairs:
        return None
    bs, _ = compute_brier(pairs)
    return float(bs)


def _aggregate(scores: list[_BucketScore]) -> dict:
    """Roll up a flat list of bucket scores into the dashboard schema."""
    legacy_pairs = [(s.legacy_prob, s.outcome) for s in scores if s.legacy_prob is not None]
    bma_pairs = [(s.bma_prob, s.outcome) for s in scores if s.bma_prob is not None]
    market_pairs = [(s.market_prob, s.outcome) for s in scores if s.market_prob is not None]

    legacy_brier = _safe_brier(legacy_pairs)
    bma_brier = _safe_brier(bma_pairs)
    market_brier = _safe_brier(market_pairs)

    def _edge(source_brier: Optional[float]) -> Optional[float]:
        if source_brier is None or market_brier is None:
            return None
        return market_brier - source_brier

    return {
        "legacy": {
            "brier": round(legacy_brier, 6) if legacy_brier is not None else None,
            "edge_vs_market": round(_edge(legacy_brier), 6) if _edge(legacy_brier) is not None else None,
            "edge_bps": round(_edge(legacy_brier) * BPS, 1) if _edge(legacy_brier) is not None else None,
            "n": len(legacy_pairs),
        },
        "bma": {
            "brier": round(bma_brier, 6) if bma_brier is not None else None,
            "edge_vs_market": round(_edge(bma_brier), 6) if _edge(bma_brier) is not None else None,
            "edge_bps": round(_edge(bma_brier) * BPS, 1) if _edge(bma_brier) is not None else None,
            "n": len(bma_pairs),
        },
        "market": {
            "brier": round(market_brier, 6) if market_brier is not None else None,
            "edge_vs_market": 0.0,
            "edge_bps": 0.0,
            "n": len(market_pairs),
        },
    }


def _consecutive_days_bma_wins(by_day: list[dict]) -> int:
    """Count how many of the *most recent* days BMA had lower Brier than legacy.

    Resets to zero on the first day BMA didn't win; days with insufficient data
    in either source are skipped (don't reset the streak, but don't count
    toward it either).
    """
    streak = 0
    for row in reversed(by_day):
        lb = row.get("legacy_brier")
        bb = row.get("bma_brier")
        if lb is None or bb is None:
            continue
        if bb < lb:
            streak += 1
        else:
            break
    return streak


# ───────────────────────── Pure scoring core ────────────────────────────────

def _score_event(
    *,
    event: dict,
    legacy_probs: list[float],
    bma_probs: Optional[list[float]],
    market_by_bucket: dict[int, Optional[float]],
    bma_between_share: Optional[float] = None,
) -> list[_BucketScore]:
    """Pure scoring helper — no DB. Used directly by unit tests.

    `event` must include keys: id, city_slug, date_et, winning_bucket_idx,
    and `buckets` (list of {bucket_idx}).
    """
    scores: list[_BucketScore] = []
    winning_idx = event["winning_bucket_idx"]
    for b in event["buckets"]:
        idx = b["bucket_idx"]
        outcome = 1 if idx == winning_idx else 0

        legacy_prob = legacy_probs[idx] if 0 <= idx < len(legacy_probs) else None
        bma_prob = (
            bma_probs[idx] if bma_probs is not None and 0 <= idx < len(bma_probs) else None
        )
        market_prob = market_by_bucket.get(idx)

        scores.append(_BucketScore(
            event_id=event["id"],
            city_slug=event["city_slug"],
            date_et=event["date_et"],
            bucket_idx=idx,
            outcome=outcome,
            legacy_prob=legacy_prob,
            bma_prob=bma_prob,
            market_prob=market_prob,
            bma_between_share=bma_between_share,
        ))
    return scores


# ───────────────────────── DB-backed entry point ────────────────────────────

async def _score_resolved_events(
    sess: AsyncSession,
    *,
    cutoff: datetime,
) -> list[_BucketScore]:
    """Iterate settled events since `cutoff` and produce one _BucketScore
    per (event, bucket). Skips events without a model snapshot."""
    result = await sess.execute(
        select(Event, City)
        .join(City, City.id == Event.city_id)
        .where(
            Event.winning_bucket_idx.isnot(None),
            Event.resolved_at.isnot(None),
            Event.resolved_at >= cutoff,
        )
        .order_by(Event.resolved_at.asc())
    )
    rows = list(result.all())

    scores: list[_BucketScore] = []
    for event, city in rows:
        # Latest snapshot is fine — signal engine stops producing new snapshots
        # once an event resolves, so latest ≈ latest-before-resolved_at.
        snap = (
            await sess.execute(
                select(ModelSnapshot)
                .where(ModelSnapshot.event_id == event.id)
                .order_by(desc(ModelSnapshot.computed_at))
                .limit(1)
            )
        ).scalar_one_or_none()
        if snap is None:
            continue

        try:
            legacy_probs = json.loads(snap.probs_json or "[]")
        except Exception:
            log.debug("edge_metrics: bad probs_json on snapshot id=%s", snap.id)
            legacy_probs = []

        bma_probs: Optional[list[float]] = None
        bma_between: Optional[float] = None
        try:
            inputs = json.loads(snap.inputs_json or "{}") if snap.inputs_json else {}
            bma = inputs.get("bma_shadow") if isinstance(inputs, dict) else None
            if isinstance(bma, dict):
                bma_probs = bma.get("probs")
                bma_between = bma.get("between_share")
        except Exception:
            log.debug("edge_metrics: bad inputs_json on snapshot id=%s", snap.id)

        # Per-bucket market price (latest MarketSnapshot before resolution).
        buckets = (
            await sess.execute(
                select(Bucket)
                .where(Bucket.event_id == event.id)
                .order_by(Bucket.bucket_idx.asc())
            )
        ).scalars().all()
        market_by_bucket: dict[int, Optional[float]] = {}
        for b in buckets:
            mkt = (
                await sess.execute(
                    select(MarketSnapshot)
                    .where(MarketSnapshot.bucket_id == b.id)
                    .order_by(desc(MarketSnapshot.fetched_at))
                    .limit(1)
                )
            ).scalar_one_or_none()
            market_by_bucket[b.bucket_idx] = mkt.yes_mid if mkt and mkt.yes_mid is not None else None

        scores.extend(_score_event(
            event={
                "id": event.id,
                "city_slug": city.city_slug,
                "date_et": event.date_et,
                "winning_bucket_idx": event.winning_bucket_idx,
                "buckets": [{"bucket_idx": b.bucket_idx} for b in buckets],
            },
            legacy_probs=legacy_probs,
            bma_probs=bma_probs,
            market_by_bucket=market_by_bucket,
            bma_between_share=bma_between,
        ))

    return scores


# ───────────────────────── Public API ───────────────────────────────────────

async def compute_edge_metrics(
    days_back: int = 30,
    min_n_buckets: int = DEFAULT_MIN_N_BUCKETS,
    sess: Optional[AsyncSession] = None,
) -> dict:
    """Aggregate Brier scores + edge-vs-market over the last `days_back` days.

    Returns a dict shaped for /calibration/edge (HTML + JSON). Empty subgroups
    appear as {n: <count>, brier: null} when below `min_n_buckets`.

    Pass `sess` to use the caller's session (for tests); otherwise a new
    session is opened from the global pool.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    if sess is None:
        async with get_session() as s:
            scores = await _score_resolved_events(s, cutoff=cutoff)
    else:
        scores = await _score_resolved_events(sess, cutoff=cutoff)

    # Aggregate overall
    overall = _aggregate(scores)

    # Per-city
    by_city: dict[str, dict] = {}
    cities_seen: dict[str, list[_BucketScore]] = {}
    for s in scores:
        cities_seen.setdefault(s.city_slug, []).append(s)
    for slug, group in sorted(cities_seen.items()):
        if len(group) < min_n_buckets:
            by_city[slug] = {"n": len(group), "below_min_threshold": True}
        else:
            by_city[slug] = _aggregate(group)
            by_city[slug]["n"] = len(group)

    # Per-day timeseries
    days_seen: dict[str, list[_BucketScore]] = {}
    for s in scores:
        days_seen.setdefault(s.date_et, []).append(s)
    by_day: list[dict] = []
    for d in sorted(days_seen.keys()):
        group = days_seen[d]
        agg = _aggregate(group)
        by_day.append({
            "date": d,
            "n": len(group),
            "legacy_brier": agg["legacy"]["brier"],
            "bma_brier": agg["bma"]["brier"],
            "market_brier": agg["market"]["brier"],
            "legacy_edge_bps": agg["legacy"]["edge_bps"],
            "bma_edge_bps": agg["bma"]["edge_bps"],
        })

    # Per-regime: high inter-source disagreement vs low
    high_dis = [s for s in scores if (s.bma_between_share or 0.0) > 0.5]
    low_dis = [s for s in scores if (s.bma_between_share or 0.0) <= 0.5]
    by_regime = {
        "high_disagreement": (
            _aggregate(high_dis) if len(high_dis) >= min_n_buckets
            else {"n": len(high_dis), "below_min_threshold": True}
        ),
        "low_disagreement": (
            _aggregate(low_dis) if len(low_dis) >= min_n_buckets
            else {"n": len(low_dis), "below_min_threshold": True}
        ),
    }

    promotion_signal = {
        "current_edge_legacy_bps": overall["legacy"]["edge_bps"],
        "current_edge_bma_bps": overall["bma"]["edge_bps"],
        "bma_better_than_legacy": (
            overall["bma"]["brier"] is not None
            and overall["legacy"]["brier"] is not None
            and overall["bma"]["brier"] < overall["legacy"]["brier"]
        ),
        "consecutive_days_bma_wins": _consecutive_days_bma_wins(by_day),
        "n_events_unique": len({s.event_id for s in scores}),
    }

    return {
        "lookback_days": days_back,
        "n_events": len({s.event_id for s in scores}),
        "n_buckets": len(scores),
        "by_source": overall,
        "by_city": by_city,
        "by_day": by_day,
        "by_regime": by_regime,
        "promotion_signal": promotion_signal,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
