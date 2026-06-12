"""Shadow market-flow validation.

Evaluates whether stored smart-wallet / CLOB flow features predict subsequent
price moves or final settlement outcomes. This is intentionally read-only:
execution must keep these features shadow-only until this report shows
out-of-sample lift with enough samples.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import Bucket, City, Event, MarketFlowFeature, MarketSnapshot


@dataclass(frozen=True)
class FlowValidationParams:
    days_back: int = 14
    window_minutes: int = 15
    horizon_minutes: int = 15
    min_samples: int = 100
    max_rows: int = 5000
    large_move_threshold: float = 0.01

    def normalized(self) -> "FlowValidationParams":
        return FlowValidationParams(
            days_back=max(1, min(int(self.days_back), 365)),
            window_minutes=max(1, min(int(self.window_minutes), 240)),
            horizon_minutes=max(1, min(int(self.horizon_minutes), 240)),
            min_samples=max(2, min(int(self.min_samples), 10000)),
            max_rows=max(10, min(int(self.max_rows), 50000)),
            large_move_threshold=max(0.001, min(float(self.large_move_threshold), 0.25)),
        )


@dataclass(frozen=True)
class FlowValidationSample:
    bucket_id: int
    city_slug: str
    date_et: str
    bucket_idx: int
    computed_at: str
    signed_net_notional: float
    imbalance: float
    vpin: float
    toxicity_score: float
    top_wallet_weighted_flow: float
    direction_confidence: float
    price_now: float | None
    price_future: float | None
    price_delta: float | None
    next_up: int | None
    final_yes: int | None


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _price(snapshot: MarketSnapshot) -> float | None:
    if snapshot.yes_mid is not None:
        return float(snapshot.yes_mid)
    if snapshot.yes_bid is not None and snapshot.yes_ask is not None:
        return (float(snapshot.yes_bid) + float(snapshot.yes_ask)) / 2.0
    if snapshot.yes_bid is not None:
        return float(snapshot.yes_bid)
    if snapshot.yes_ask is not None:
        return float(snapshot.yes_ask)
    return None


def _roc_auc(scores: list[float], outcomes: list[int]) -> float | None:
    """AUC with average ranks for ties. Returns None for one-class data."""
    if len(scores) != len(outcomes) or not scores:
        return None
    n_pos = sum(1 for outcome in outcomes if outcome == 1)
    n_neg = len(outcomes) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranked = sorted(zip(scores, outcomes), key=lambda item: item[0])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(ranked):
        end = idx + 1
        while end < len(ranked) and ranked[end][0] == ranked[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for j in range(idx, end):
            if ranked[j][1] == 1:
                rank_sum_pos += avg_rank
        idx = end
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(float(auc), 6)


def _directional_hit_rate(scores: list[float], outcomes: list[int]) -> dict[str, Any]:
    usable = [(s, o) for s, o in zip(scores, outcomes) if abs(s) > 1e-12]
    if not usable:
        return {
            "n": 0,
            "hit_rate": None,
            "base_rate_up": None,
            "majority_baseline": None,
            "lift_vs_majority": None,
        }
    hits = sum(1 for score, outcome in usable if int(score > 0) == int(outcome))
    base_rate = sum(outcome for _, outcome in usable) / len(usable)
    majority = max(base_rate, 1.0 - base_rate)
    hit_rate = hits / len(usable)
    return {
        "n": len(usable),
        "hit_rate": round(hit_rate, 6),
        "base_rate_up": round(base_rate, 6),
        "majority_baseline": round(majority, 6),
        "lift_vs_majority": round(hit_rate - majority, 6),
    }


def _score_report(samples: list[dict[str, Any]], score_key: str, outcome_key: str) -> dict[str, Any]:
    rows = [
        (float(row[score_key]), int(row[outcome_key]))
        for row in samples
        if row.get(score_key) is not None and row.get(outcome_key) is not None
    ]
    scores = [score for score, _ in rows]
    outcomes = [outcome for _, outcome in rows]
    return {
        "score": score_key,
        "outcome": outcome_key,
        "n": len(rows),
        "auc": _roc_auc(scores, outcomes),
        "directional": _directional_hit_rate(scores, outcomes),
    }


def _large_move_report(samples: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    rows = []
    for row in samples:
        delta = row.get("price_delta")
        score = row.get(score_key)
        if delta is None or score is None:
            continue
        rows.append((float(score), int(abs(float(delta)) >= threshold)))
    scores = [score for score, _ in rows]
    outcomes = [outcome for _, outcome in rows]
    return {
        "score": score_key,
        "outcome": f"abs_price_delta_ge_{threshold:.3f}",
        "n": len(rows),
        "auc": _roc_auc(scores, outcomes),
        "event_rate": round(sum(outcomes) / len(outcomes), 6) if outcomes else None,
    }


def build_flow_validation_report(
    sample_rows: list[dict[str, Any]],
    *,
    params: FlowValidationParams,
) -> dict[str, Any]:
    """Build promotion diagnostics from precomputed sample dictionaries."""
    params = params.normalized()
    price_samples = [row for row in sample_rows if row.get("next_up") is not None]
    final_samples = [row for row in sample_rows if row.get("final_yes") is not None]
    directional_scores = [
        _score_report(price_samples, "signed_net_notional", "next_up"),
        _score_report(price_samples, "imbalance", "next_up"),
        _score_report(price_samples, "top_wallet_weighted_flow", "next_up"),
    ]
    settlement_scores = [
        _score_report(final_samples, "signed_net_notional", "final_yes"),
        _score_report(final_samples, "imbalance", "final_yes"),
        _score_report(final_samples, "top_wallet_weighted_flow", "final_yes"),
    ]
    toxicity_scores = [
        _large_move_report(price_samples, "vpin", params.large_move_threshold),
        _large_move_report(price_samples, "toxicity_score", params.large_move_threshold),
    ]
    best_directional = max(
        directional_scores,
        key=lambda item: item["auc"] if item["auc"] is not None else -1.0,
        default=None,
    )
    best_hit = max(
        (
            item
            for item in directional_scores
            if item["directional"]["lift_vs_majority"] is not None
        ),
        key=lambda item: item["directional"]["lift_vs_majority"],
        default=None,
    )

    blockers: list[str] = []
    if len(price_samples) < params.min_samples:
        blockers.append(f"price_move_samples_below_min:{len(price_samples)}<{params.min_samples}")
    best_auc = best_directional["auc"] if best_directional else None
    best_lift = (
        best_hit["directional"]["lift_vs_majority"]
        if best_hit
        else None
    )
    if best_auc is None or best_auc < 0.55:
        blockers.append("next_price_move_auc_below_0.55")
    if best_lift is None or best_lift < 0.03:
        blockers.append("directional_hit_lift_below_3pct")

    recommendation = "candidate_for_shadow_promotion" if not blockers else "keep_shadow_only"
    return {
        "params": asdict(params),
        "n_flow_rows": len(sample_rows),
        "n_price_move_samples": len(price_samples),
        "n_final_outcome_samples": len(final_samples),
        "directional_scores": directional_scores,
        "settlement_scores": settlement_scores,
        "toxicity_scores": toxicity_scores,
        "best_directional_score": best_directional,
        "best_directional_hit_lift": best_hit,
        "promotion": {
            "recommendation": recommendation,
            "allowed_for_execution": False,
            "blockers": blockers,
            "rule": (
                "Promotion requires at least min_samples, next-price AUC >= 0.55, "
                "directional hit-rate lift >= 3 percentage points, and a separate "
                "out-of-sample review before any execution gate is enabled."
            ),
        },
    }


async def evaluate_market_flow_features(
    session: AsyncSession,
    params: FlowValidationParams | None = None,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Join flow features to later order-book moves and settlement outcomes."""
    params = (params or FlowValidationParams()).normalized()
    as_of = _as_utc(as_of or datetime.now(timezone.utc))
    cutoff = as_of - timedelta(days=params.days_back)
    flow_rows = (
        await session.execute(
            select(MarketFlowFeature, Bucket, Event, City)
            .join(Bucket, Bucket.id == MarketFlowFeature.bucket_id)
            .join(Event, Event.id == Bucket.event_id)
            .join(City, City.id == Event.city_id)
            .where(
                MarketFlowFeature.window_minutes == params.window_minutes,
                MarketFlowFeature.computed_at >= cutoff,
            )
            .order_by(MarketFlowFeature.computed_at.desc())
            .limit(params.max_rows)
        )
    ).all()

    if not flow_rows:
        return build_flow_validation_report([], params=params)

    bucket_ids = sorted({int(bucket.id) for _, bucket, _, _ in flow_rows})
    min_ts = min(_as_utc(flow.computed_at) for flow, _, _, _ in flow_rows) - timedelta(minutes=10)
    max_ts = max(_as_utc(flow.computed_at) for flow, _, _, _ in flow_rows) + timedelta(
        minutes=params.horizon_minutes + 120
    )
    snapshot_rows = (
        await session.execute(
            select(MarketSnapshot)
            .where(
                MarketSnapshot.bucket_id.in_(bucket_ids),
                MarketSnapshot.fetched_at >= min_ts,
                MarketSnapshot.fetched_at <= max_ts,
            )
            .order_by(MarketSnapshot.bucket_id, MarketSnapshot.fetched_at)
        )
    ).scalars().all()

    snapshots_by_bucket: dict[int, list[tuple[datetime, float]]] = {}
    for snapshot in snapshot_rows:
        price = _price(snapshot)
        if price is None:
            continue
        snapshots_by_bucket.setdefault(int(snapshot.bucket_id), []).append(
            (_as_utc(snapshot.fetched_at), price)
        )

    samples: list[dict[str, Any]] = []
    for flow, bucket, event, city in flow_rows:
        computed_at = _as_utc(flow.computed_at)
        snapshots = snapshots_by_bucket.get(int(bucket.id), [])
        times = [ts for ts, _ in snapshots]
        price_now = None
        price_future = None
        price_delta = None
        next_up = None
        idx_now = bisect_right(times, computed_at) - 1
        idx_future = bisect_left(times, computed_at + timedelta(minutes=params.horizon_minutes))
        if idx_now >= 0:
            price_now = snapshots[idx_now][1]
        if idx_future < len(snapshots):
            price_future = snapshots[idx_future][1]
        if price_now is not None and price_future is not None:
            price_delta = price_future - price_now
            next_up = int(price_delta > 0)

        final_yes = None
        if event.winning_bucket_idx is not None:
            final_yes = int(int(event.winning_bucket_idx) == int(bucket.bucket_idx))

        samples.append(
            {
                "bucket_id": int(bucket.id),
                "city_slug": city.city_slug,
                "date_et": event.date_et,
                "bucket_idx": int(bucket.bucket_idx),
                "computed_at": computed_at.isoformat(),
                "signed_net_notional": float(flow.signed_net_notional or 0.0),
                "imbalance": float(flow.imbalance or 0.0),
                "vpin": float(flow.vpin or 0.0),
                "toxicity_score": float(flow.toxicity_score or 0.0),
                "top_wallet_weighted_flow": float(flow.top_wallet_weighted_flow or 0.0),
                "direction_confidence": float(flow.direction_confidence or 0.0),
                "price_now": price_now,
                "price_future": price_future,
                "price_delta": price_delta,
                "next_up": next_up,
                "final_yes": final_yes,
            }
        )

    report = build_flow_validation_report(samples, params=params)
    report["sample_preview"] = [
        asdict(FlowValidationSample(**row))
        for row in samples[:25]
    ]
    return report
