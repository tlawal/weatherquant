"""Offline validation helpers for the wallet tracker.

This module intentionally performs no live Polymarket calls. Feed it stored or
mocked wallet leaderboard snapshots and resolved bucket outcomes to evaluate
whether the read-only wallet signal has predictive value.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WalletTrackerValidationResult:
    snapshots: int
    top_wallet_persistence: float
    leaderboard_turnover: float
    bucket_hit_rate: float | None
    divergence_hit_rate: float | None
    by_regime: dict[str, dict[str, float]]
    notes: tuple[str, ...]


def _ranked_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        snapshot.get("current_market")
        or snapshot.get("rows")
        or snapshot.get("global_leaders")
        or []
    )


def _top_wallets(snapshot: dict[str, Any], n: int = 5) -> set[str]:
    rows = _ranked_rows(snapshot)
    return {
        str(row.get("wallet_address") or "").lower()
        for row in rows[:n]
        if row.get("wallet_address")
    }


def _favored_bucket(snapshot: dict[str, Any]) -> int | None:
    consensus = snapshot.get("bucket_consensus") or []
    if consensus:
        positive = [
            row for row in consensus
            if row.get("bucket_idx") is not None
            and float(row.get("weighted_flow") or row.get("net_notional_usd") or 0.0) > 0
        ]
        if positive:
            best = max(
                positive,
                key=lambda row: (
                    float(row.get("weighted_flow") or 0.0),
                    float(row.get("net_notional_usd") or 0.0),
                ),
            )
            return int(best["bucket_idx"])

    rows = _ranked_rows(snapshot)
    flow_by_bucket: dict[int, float] = defaultdict(float)
    for row in rows:
        idx = row.get("bucket_idx")
        if idx is None:
            continue
        try:
            flow_by_bucket[int(idx)] += float(
                row.get("net_notional_usd")
                or row.get("net_flow_usd")
                or 0.0
            )
        except (TypeError, ValueError):
            continue
    positive = {idx: flow for idx, flow in flow_by_bucket.items() if flow > 0}
    return max(positive, key=lambda idx: positive[idx]) if positive else None


def evaluate_wallet_tracker_snapshots(
    snapshots: list[dict[str, Any]],
    resolved_outcomes: dict[tuple[str, str], int],
) -> WalletTrackerValidationResult:
    """Evaluate stored/mock wallet snapshots against resolved outcomes.

    Expected snapshot keys:
      - city_slug
      - date
      - current_market or rows: wallet_tracker smart-money rows
      - bucket_consensus: optional V2 bucket consensus rows
      - smart_money_context or confluence: optional model/smart-money payload
      - regime: optional CALM/NORMAL/VOLATILE label

    TODO: Once historical public trade ingestion is complete, add forward-only
    cohorting so a wallet's rank is measured using only data known at that time.
    """
    if not snapshots:
        return WalletTrackerValidationResult(
            snapshots=0,
            top_wallet_persistence=0.0,
            leaderboard_turnover=0.0,
            bucket_hit_rate=None,
            divergence_hit_rate=None,
            by_regime={},
            notes=("No snapshots supplied.",),
        )

    persistence_scores = []
    turnover_scores = []
    bucket_hits = []
    divergence_hits = []
    regime_hits: dict[str, list[float]] = defaultdict(list)

    prior_top: set[str] | None = None
    for snapshot in snapshots:
        top = _top_wallets(snapshot)
        if prior_top is not None:
            overlap = len(top & prior_top) / max(1, len(top | prior_top))
            persistence_scores.append(overlap)
            turnover_scores.append(1.0 - overlap)
        prior_top = top

        key = (str(snapshot.get("city_slug")), str(snapshot.get("date")))
        resolved_idx = resolved_outcomes.get(key)
        favored_idx = _favored_bucket(snapshot)
        if resolved_idx is not None and favored_idx is not None:
            hit = 1.0 if favored_idx == resolved_idx else 0.0
            bucket_hits.append(hit)
            regime = str(snapshot.get("regime") or "UNKNOWN").upper()
            regime_hits[regime].append(hit)

        divergence = snapshot.get("smart_money_context") or snapshot.get("confluence") or {}
        if resolved_idx is not None and divergence.get("status") == "available":
            smart_idx = divergence.get("smart_money_bucket_idx")
            model_idx = divergence.get("model_bucket_idx")
            if smart_idx is not None and model_idx is not None and smart_idx != model_idx:
                divergence_hits.append(1.0 if smart_idx == resolved_idx else 0.0)

    by_regime = {
        regime: {
            "samples": float(len(vals)),
            "bucket_hit_rate": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }
        for regime, vals in regime_hits.items()
    }

    return WalletTrackerValidationResult(
        snapshots=len(snapshots),
        top_wallet_persistence=round(sum(persistence_scores) / len(persistence_scores), 4)
        if persistence_scores else 0.0,
        leaderboard_turnover=round(sum(turnover_scores) / len(turnover_scores), 4)
        if turnover_scores else 0.0,
        bucket_hit_rate=round(sum(bucket_hits) / len(bucket_hits), 4) if bucket_hits else None,
        divergence_hit_rate=round(sum(divergence_hits) / len(divergence_hits), 4)
        if divergence_hits else None,
        by_regime=by_regime,
        notes=(
            "Offline scaffold only; supply stored public trade history for robust validation.",
            "Divergence value is only measured when smart-money and model buckets differ.",
        ),
    )
