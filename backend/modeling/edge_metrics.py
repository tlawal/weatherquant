"""Alpha dashboard — Brier(model) − Brier(market) plus CRPS.

Section 6 Layer 6 of the Bayesian upgrade plan: the only metric that directly
measures whether we add value over the market. Without this, BMA promotion has
no quantitative basis.

Three forecasting paths are scored against settled outcomes:

- **Legacy model** — `ModelSnapshot.probs_json` (drives trades today).
- **BMA shadow** — `ModelSnapshot.inputs_json["bma_shadow"]["probs"]`
  (computed since `e4835da`, does not drive trades).
- **Market** — `MarketSnapshot.yes_mid` per bucket at a fixed checkpoint.

For each scored `Event` (Gamma-confirmed or provisional local settlement):

  outcome_b = 1 if b.bucket_idx == event.winning_bucket_idx else 0
  brier_b   = (prob_b − outcome_b)²

Aggregate per source:
  mean_brier      = Σ brier / N
  edge_vs_market  = mean_brier_market − mean_brier_source     (positive = wins)

CRPS is also computed per event from the resolved continuous daily high:
legacy uses the normal `mu/sigma`, BMA uses its Gaussian mixture components,
and market uses normalized bucket prices over bucket representatives. Lower
CRPS is better; CRPS edge is market_crps − source_crps in °F.

This module is pure read-side: no DB writes, no scheduler hooks. Called from
`/calibration/edge` and `/api/calibration/edge.json`.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from scipy.stats import norm

from backend.backtesting.metrics import compute_brier
from backend.modeling.calibration_engine import resolve_canonical_settlement_high
from backend.modeling.settlement import canonical_bucket_ranges, find_bucket_idx_for_value
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

# Score what the model/market knew before the event was effectively settled.
# Noon local is late enough to include same-day forecast updates and market
# liquidity, but early enough to avoid scoring post-lock/settlement hindsight.
SCORE_CHECKPOINT_HOUR_LOCAL = 12


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
    settlement_status: str = "gamma_confirmed"  # gamma_confirmed | provisional_local
    score_checkpoint_utc: Optional[str] = None


@dataclass
class _DistributionScore:
    """One settled event scored as a full predictive distribution."""
    event_id: int
    city_slug: str
    date_et: str
    resolved_high_f: float
    legacy_crps: Optional[float]
    bma_crps: Optional[float]
    market_crps: Optional[float]
    settlement_status: str = "gamma_confirmed"
    score_checkpoint_utc: Optional[str] = None
    bma_between_share: Optional[float] = None


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


def _safe_mean(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _aggregate_crps(scores: list[_DistributionScore]) -> dict:
    """Roll up event-level CRPS into the same source schema as Brier."""
    legacy_vals = [s.legacy_crps for s in scores if s.legacy_crps is not None]
    bma_vals = [s.bma_crps for s in scores if s.bma_crps is not None]
    market_vals = [s.market_crps for s in scores if s.market_crps is not None]

    legacy_crps = _safe_mean(legacy_vals)
    bma_crps = _safe_mean(bma_vals)
    market_crps = _safe_mean(market_vals)

    def _edge(source_crps: Optional[float]) -> Optional[float]:
        if source_crps is None or market_crps is None:
            return None
        return market_crps - source_crps

    return {
        "legacy": {
            "crps": round(legacy_crps, 6) if legacy_crps is not None else None,
            "edge_vs_market": round(_edge(legacy_crps), 6) if _edge(legacy_crps) is not None else None,
            "edge_cdeg": round(_edge(legacy_crps) * 100, 1) if _edge(legacy_crps) is not None else None,
            "n": len(legacy_vals),
        },
        "bma": {
            "crps": round(bma_crps, 6) if bma_crps is not None else None,
            "edge_vs_market": round(_edge(bma_crps), 6) if _edge(bma_crps) is not None else None,
            "edge_cdeg": round(_edge(bma_crps) * 100, 1) if _edge(bma_crps) is not None else None,
            "n": len(bma_vals),
        },
        "market": {
            "crps": round(market_crps, 6) if market_crps is not None else None,
            "edge_vs_market": 0.0,
            "edge_cdeg": 0.0,
            "n": len(market_vals),
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


# ───────────────────── Continuous distribution scoring ──────────────────────

def _normal_crps(mu: float, sigma: float, observed: float) -> Optional[float]:
    """Closed-form CRPS for a Gaussian predictive distribution.

    Lower is better; units are degrees Fahrenheit. Formula:
    σ[z(2Φ(z)-1) + 2φ(z) - 1/sqrt(π)], z=(y-μ)/σ.
    """
    try:
        mu_f = float(mu)
        sigma_f = float(sigma)
        y = float(observed)
    except (TypeError, ValueError):
        return None
    if sigma_f <= 0 or not math.isfinite(sigma_f):
        return None
    z = (y - mu_f) / sigma_f
    score = sigma_f * (
        z * (2.0 * float(norm.cdf(z)) - 1.0)
        + 2.0 * float(norm.pdf(z))
        - 1.0 / math.sqrt(math.pi)
    )
    return float(max(0.0, score))


def _a_term(delta: float, sigma: float) -> float:
    if sigma <= 0:
        return abs(delta)
    z = delta / sigma
    return (
        2.0 * sigma * float(norm.pdf(z))
        + delta * (2.0 * float(norm.cdf(z)) - 1.0)
    )


def _bma_mixture_crps(components: list[dict], observed: float) -> Optional[float]:
    """CRPS for a finite Gaussian mixture.

    Uses the identity:
      CRPS(F,y) = ΣᵢwᵢA(y-μᵢ,σᵢ²) - 0.5ΣᵢΣⱼwᵢwⱼA(μᵢ-μⱼ,σᵢ²+σⱼ²)
    """
    parsed: list[tuple[float, float, float]] = []
    for comp in components or []:
        try:
            mu = float(comp.get("mu"))
            sigma = float(comp.get("sigma"))
            weight = float(comp.get("weight"))
        except (TypeError, ValueError, AttributeError):
            continue
        if sigma <= 0 or weight <= 0:
            continue
        parsed.append((mu, sigma, weight))
    total_w = sum(w for _, _, w in parsed)
    if total_w <= 0:
        return None
    parsed = [(mu, sigma, w / total_w) for mu, sigma, w in parsed]
    y = float(observed)

    first = sum(w * _a_term(y - mu, sigma) for mu, sigma, w in parsed)
    second = 0.0
    for mu_i, sigma_i, w_i in parsed:
        for mu_j, sigma_j, w_j in parsed:
            second += w_i * w_j * _a_term(
                mu_i - mu_j,
                math.sqrt(sigma_i * sigma_i + sigma_j * sigma_j),
            )
    return float(max(0.0, first - 0.5 * second))


def _bucket_representatives(
    buckets: list[dict],
) -> dict[int, float]:
    finite_widths: list[float] = []
    for b in buckets:
        lo = b.get("low_f")
        hi = b.get("high_f")
        if lo is not None and hi is not None:
            finite_widths.append(float(hi) - float(lo))
    width = sorted(finite_widths)[len(finite_widths) // 2] if finite_widths else 2.0
    width = max(1.0, float(width))

    reps: dict[int, float] = {}
    for b in buckets:
        idx = b.get("bucket_idx")
        if idx is None:
            continue
        lo = b.get("low_f")
        hi = b.get("high_f")
        if lo is not None and hi is not None:
            rep = (float(lo) + float(hi)) / 2.0
        elif hi is not None:
            rep = float(hi) - width / 2.0
        elif lo is not None:
            rep = float(lo) + width / 2.0
        else:
            continue
        reps[int(idx)] = rep
    return reps


def _discrete_crps(
    probs_by_bucket: dict[int, Optional[float]],
    reps_by_bucket: dict[int, float],
    observed: float,
) -> Optional[float]:
    rows: list[tuple[float, float]] = []
    for idx, prob in probs_by_bucket.items():
        if prob is None or idx not in reps_by_bucket:
            continue
        try:
            p = float(prob)
        except (TypeError, ValueError):
            continue
        if p > 0:
            rows.append((float(reps_by_bucket[idx]), p))
    total_p = sum(p for _, p in rows)
    if total_p <= 0:
        return None
    rows = [(value, p / total_p) for value, p in rows]
    y = float(observed)
    first = sum(p * abs(value - y) for value, p in rows)
    second = 0.0
    for value_i, p_i in rows:
        for value_j, p_j in rows:
            second += p_i * p_j * abs(value_i - value_j)
    return float(max(0.0, first - 0.5 * second))


def _score_distribution_event(
    *,
    event: dict,
    legacy_mu: Optional[float],
    legacy_sigma: Optional[float],
    bma_shadow: Optional[dict],
    market_by_bucket: dict[int, Optional[float]],
    resolved_high_f: float,
) -> _DistributionScore:
    buckets = event.get("buckets") or []
    legacy_crps = _normal_crps(legacy_mu, legacy_sigma, resolved_high_f)

    bma_crps = None
    bma_between = None
    if isinstance(bma_shadow, dict):
        bma_between = bma_shadow.get("between_share")
        components = bma_shadow.get("components")
        if isinstance(components, list) and components:
            bma_crps = _bma_mixture_crps(components, resolved_high_f)
        if bma_crps is None:
            bma_crps = _normal_crps(
                bma_shadow.get("mean") or bma_shadow.get("mu"),
                bma_shadow.get("sigma"),
                resolved_high_f,
            )

    reps = _bucket_representatives(buckets)
    market_crps = _discrete_crps(market_by_bucket, reps, resolved_high_f)

    return _DistributionScore(
        event_id=event["id"],
        city_slug=event["city_slug"],
        date_et=event["date_et"],
        resolved_high_f=float(resolved_high_f),
        legacy_crps=legacy_crps,
        bma_crps=bma_crps,
        market_crps=market_crps,
        settlement_status=event.get("settlement_status", "gamma_confirmed"),
        score_checkpoint_utc=event.get("score_checkpoint_utc"),
        bma_between_share=(
            float(bma_between) if bma_between is not None else None
        ),
    )


# ───────────────────────── Pure scoring core ────────────────────────────────

def _score_event(
    *,
    event: dict,
    legacy_probs: list[float],
    bma_probs: Optional[list[float]],
    market_by_bucket: dict[int, Optional[float]],
    bma_between_share: Optional[float] = None,
    settlement_status: str = "gamma_confirmed",
    score_checkpoint_utc: Optional[str] = None,
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
            settlement_status=settlement_status,
            score_checkpoint_utc=score_checkpoint_utc,
        ))
    return scores


# ───────────────────────── DB-backed entry point ────────────────────────────

def _empty_diagnostics() -> dict:
    return {
        "score_checkpoint_hour_local": SCORE_CHECKPOINT_HOUR_LOCAL,
        "past_events": 0,
        "local_settled_events": 0,
        "gamma_confirmed_events": 0,
        "provisional_local_events": 0,
        "scored_events": 0,
        "events_without_local_settlement": 0,
        "events_without_buckets": 0,
        "events_missing_model_snapshot": 0,
        "events_with_any_missing_market_snapshot": 0,
        "missing_market_snapshot_buckets": 0,
        "events_without_crps_settlement_high": 0,
    }


def _event_score_checkpoint_utc(event: Event, city: City) -> datetime:
    city_tz = ZoneInfo(getattr(city, "tz", None) or "America/New_York")
    local_day = datetime.strptime(event.date_et, "%Y-%m-%d").replace(
        hour=SCORE_CHECKPOINT_HOUR_LOCAL,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=city_tz,
    )
    return local_day.astimezone(timezone.utc)


def _is_past_local_event(event: Event, city: City, *, now_utc: datetime) -> bool:
    city_tz = ZoneInfo(getattr(city, "tz", None) or "America/New_York")
    today_local = now_utc.astimezone(city_tz).date()
    event_date = datetime.strptime(event.date_et, "%Y-%m-%d").date()
    return event_date < today_local


async def _derive_provisional_winner(
    sess: AsyncSession,
    *,
    event: Event,
    city: City,
    buckets: list[Bucket],
) -> Optional[int]:
    settlement = await resolve_canonical_settlement_high(
        sess,
        city=city,
        event=event,
        validate_polymarket_winner=False,
    )
    high_f = settlement.get("high_f")
    if high_f is None:
        return None
    ranges = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
    return find_bucket_idx_for_value(ranges, float(high_f))


async def _latest_model_snapshot_before(
    sess: AsyncSession,
    *,
    event_id: int,
    checkpoint_utc: datetime,
) -> Optional[ModelSnapshot]:
    result = await sess.execute(
        select(ModelSnapshot)
        .where(
            ModelSnapshot.event_id == event_id,
            ModelSnapshot.computed_at <= checkpoint_utc,
        )
        .order_by(desc(ModelSnapshot.computed_at), desc(ModelSnapshot.id))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _latest_market_snapshots_before(
    sess: AsyncSession,
    *,
    bucket_ids: list[int],
    checkpoint_utc: datetime,
) -> dict[int, MarketSnapshot]:
    if not bucket_ids:
        return {}
    sub = (
        select(
            MarketSnapshot.bucket_id,
            func.max(MarketSnapshot.fetched_at).label("max_ts"),
        )
        .where(
            MarketSnapshot.bucket_id.in_(bucket_ids),
            MarketSnapshot.fetched_at <= checkpoint_utc,
        )
        .group_by(MarketSnapshot.bucket_id)
        .subquery()
    )
    rows = (
        await sess.execute(
            select(MarketSnapshot).join(
                sub,
                (MarketSnapshot.bucket_id == sub.c.bucket_id)
                & (MarketSnapshot.fetched_at == sub.c.max_ts),
            )
        )
    ).scalars().all()
    return {row.bucket_id: row for row in rows}


async def _score_resolved_events(
    sess: AsyncSession,
    *,
    cutoff: datetime,
) -> tuple[list[_BucketScore], list[_DistributionScore], dict]:
    """Score past events since `cutoff`.

    Gamma-confirmed outcomes are preferred when available. Otherwise, past
    events with a canonical local settlement high are scored as
    `provisional_local` so the alpha dashboard can work before UMA/Gamma has
    closed every market. Model and market probabilities are taken from a fixed
    pre-resolution checkpoint to avoid hindsight.
    """
    diagnostics = _empty_diagnostics()
    cutoff_date = cutoff.date().isoformat()
    now_utc = datetime.now(timezone.utc)

    result = await sess.execute(
        select(Event, City)
        .join(City, City.id == Event.city_id)
        .where(Event.date_et >= cutoff_date)
        .order_by(Event.date_et.asc(), City.city_slug.asc())
    )
    rows = list(result.all())
    if not rows:
        return [], [], diagnostics

    past_rows = [
        (event, city)
        for event, city in rows
        if _is_past_local_event(event, city, now_utc=now_utc)
    ]
    diagnostics["past_events"] = len(past_rows)
    if not past_rows:
        return [], [], diagnostics

    events = [event for event, _ in past_rows]
    city_by_event_id = {event.id: city for event, city in past_rows}
    event_ids = [event.id for event in events]
    bucket_rows = (
        await sess.execute(
            select(Bucket)
            .where(Bucket.event_id.in_(event_ids))
            .order_by(Bucket.event_id, Bucket.bucket_idx)
        )
    ).scalars().all()
    buckets_by_event_id: dict[int, list[Bucket]] = {}
    for bucket in bucket_rows:
        buckets_by_event_id.setdefault(bucket.event_id, []).append(bucket)

    scores: list[_BucketScore] = []
    distribution_scores: list[_DistributionScore] = []
    for event in events:
        city = city_by_event_id[event.id]
        buckets = buckets_by_event_id.get(event.id, [])
        if not buckets:
            diagnostics["events_without_buckets"] += 1
            continue

        settlement_high_f: Optional[float] = None
        settlement = await resolve_canonical_settlement_high(
            sess,
            city=city,
            event=event,
            validate_polymarket_winner=False,
        )
        if settlement.get("high_f") is not None:
            settlement_high_f = float(settlement["high_f"])

        if event.winning_bucket_idx is not None and event.resolved_at is not None:
            winning_bucket_idx = event.winning_bucket_idx
            settlement_status = "gamma_confirmed"
            diagnostics["gamma_confirmed_events"] += 1
            diagnostics["local_settled_events"] += 1
        else:
            if settlement_high_f is None:
                diagnostics["events_without_local_settlement"] += 1
                continue
            ranges = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
            winning_bucket_idx = find_bucket_idx_for_value(ranges, settlement_high_f)
            settlement_status = "provisional_local"
            diagnostics["provisional_local_events"] += 1
            diagnostics["local_settled_events"] += 1

        checkpoint_utc = _event_score_checkpoint_utc(event, city)
        snap = await _latest_model_snapshot_before(
            sess,
            event_id=event.id,
            checkpoint_utc=checkpoint_utc,
        )
        if snap is None:
            diagnostics["events_missing_model_snapshot"] += 1
            continue

        try:
            legacy_probs = json.loads(snap.probs_json or "[]")
        except Exception:
            log.debug("edge_metrics: bad probs_json on snapshot id=%s", snap.id)
            legacy_probs = []

        bma_probs: Optional[list[float]] = None
        bma_between: Optional[float] = None
        bma: Optional[dict] = None
        try:
            inputs = json.loads(snap.inputs_json or "{}") if snap.inputs_json else {}
            bma = inputs.get("bma_shadow") if isinstance(inputs, dict) else None
            if isinstance(bma, dict):
                bma_probs = bma.get("probs")
                bma_between = bma.get("between_share")
            else:
                bma = None
        except Exception:
            log.debug("edge_metrics: bad inputs_json on snapshot id=%s", snap.id)
            bma = None

        market_rows_by_bucket_id = await _latest_market_snapshots_before(
            sess,
            bucket_ids=[b.id for b in buckets],
            checkpoint_utc=checkpoint_utc,
        )
        missing_market = 0
        market_by_bucket = {
            b.bucket_idx: (
                market_rows_by_bucket_id[b.id].yes_mid
                if b.id in market_rows_by_bucket_id else None
            )
            for b in buckets
        }
        for b in buckets:
            if b.id not in market_rows_by_bucket_id:
                missing_market += 1
        if missing_market:
            diagnostics["events_with_any_missing_market_snapshot"] += 1
            diagnostics["missing_market_snapshot_buckets"] += missing_market

        event_payload = {
            "id": event.id,
            "city_slug": city.city_slug,
            "date_et": event.date_et,
            "winning_bucket_idx": winning_bucket_idx,
            "buckets": [
                {
                    "bucket_idx": b.bucket_idx,
                    "low_f": b.low_f,
                    "high_f": b.high_f,
                }
                for b in buckets
            ],
            "settlement_status": settlement_status,
            "score_checkpoint_utc": checkpoint_utc.isoformat(),
        }

        scores.extend(_score_event(
            event={
                **event_payload,
                "buckets": [{"bucket_idx": b.bucket_idx} for b in buckets],
            },
            legacy_probs=legacy_probs,
            bma_probs=bma_probs,
            market_by_bucket=market_by_bucket,
            bma_between_share=bma_between,
            settlement_status=settlement_status,
            score_checkpoint_utc=checkpoint_utc.isoformat(),
        ))
        if settlement_high_f is not None:
            distribution_scores.append(_score_distribution_event(
                event=event_payload,
                legacy_mu=snap.mu,
                legacy_sigma=snap.sigma,
                bma_shadow=bma,
                market_by_bucket=market_by_bucket,
                resolved_high_f=settlement_high_f,
            ))
        else:
            diagnostics["events_without_crps_settlement_high"] += 1
        diagnostics["scored_events"] += 1

    return scores, distribution_scores, diagnostics


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

    diagnostics = _empty_diagnostics()

    if sess is None:
        async with get_session() as s:
            scored = await _score_resolved_events(s, cutoff=cutoff)
    else:
        scored = await _score_resolved_events(sess, cutoff=cutoff)

    distribution_scores: list[_DistributionScore] = []
    if isinstance(scored, tuple) and len(scored) == 3:
        scores, distribution_scores, diagnostics = scored
    elif isinstance(scored, tuple):
        scores, diagnostics = scored
    else:
        # Backward-compatible with older tests/mocks that return just scores.
        scores = scored
        diagnostics["scored_events"] = len({s.event_id for s in scores})

    # Aggregate overall
    overall = _aggregate(scores)
    crps_overall = _aggregate_crps(distribution_scores)
    event_ids_by_status: dict[str, set[int]] = {}
    for s in scores:
        event_ids_by_status.setdefault(s.settlement_status, set()).add(s.event_id)
    settlement_status_counts = {
        status: len(event_ids)
        for status, event_ids in sorted(event_ids_by_status.items())
    }

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

    crps_city_groups: dict[str, list[_DistributionScore]] = {}
    for s in distribution_scores:
        crps_city_groups.setdefault(s.city_slug, []).append(s)
    crps_by_city: dict[str, dict] = {}
    min_n_events = max(1, math.ceil(min_n_buckets / 6))
    for slug, group in sorted(crps_city_groups.items()):
        if len(group) < min_n_events:
            crps_by_city[slug] = {"n": len(group), "below_min_threshold": True}
        else:
            crps_by_city[slug] = _aggregate_crps(group)
            crps_by_city[slug]["n"] = len(group)

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

    crps_days_seen: dict[str, list[_DistributionScore]] = {}
    for s in distribution_scores:
        crps_days_seen.setdefault(s.date_et, []).append(s)
    crps_by_day: list[dict] = []
    for d in sorted(crps_days_seen.keys()):
        group = crps_days_seen[d]
        agg = _aggregate_crps(group)
        crps_by_day.append({
            "date": d,
            "n": len(group),
            "legacy_crps": agg["legacy"]["crps"],
            "bma_crps": agg["bma"]["crps"],
            "market_crps": agg["market"]["crps"],
            "legacy_edge_cdeg": agg["legacy"]["edge_cdeg"],
            "bma_edge_cdeg": agg["bma"]["edge_cdeg"],
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
    crps_high_dis = [s for s in distribution_scores if (s.bma_between_share or 0.0) > 0.5]
    crps_low_dis = [s for s in distribution_scores if (s.bma_between_share or 0.0) <= 0.5]
    crps_by_regime = {
        "high_disagreement": (
            _aggregate_crps(crps_high_dis) if len(crps_high_dis) >= min_n_events
            else {"n": len(crps_high_dis), "below_min_threshold": True}
        ),
        "low_disagreement": (
            _aggregate_crps(crps_low_dis) if len(crps_low_dis) >= min_n_events
            else {"n": len(crps_low_dis), "below_min_threshold": True}
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
        "bma_crps_better_than_legacy": (
            crps_overall["bma"]["crps"] is not None
            and crps_overall["legacy"]["crps"] is not None
            and crps_overall["bma"]["crps"] < crps_overall["legacy"]["crps"]
        ),
        "consecutive_days_bma_wins": _consecutive_days_bma_wins(by_day),
        "n_events_unique": len({s.event_id for s in scores}),
    }

    return {
        "lookback_days": days_back,
        "n_events": len({s.event_id for s in scores}),
        "n_buckets": len(scores),
        "settlement_status_counts": settlement_status_counts,
        "diagnostics": diagnostics,
        "by_source": overall,
        "crps": {
            "by_source": crps_overall,
            "by_city": crps_by_city,
            "by_day": crps_by_day,
            "by_regime": crps_by_regime,
            "n_events": len(distribution_scores),
            "description": "Continuous ranked probability score in degrees Fahrenheit; lower is better. Market CRPS approximates bucket prices as a normalized discrete distribution over bucket representatives.",
        },
        "by_city": by_city,
        "by_day": by_day,
        "by_regime": by_regime,
        "promotion_signal": promotion_signal,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
