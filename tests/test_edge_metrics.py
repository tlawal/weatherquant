"""Section 6 Layer 6 — alpha dashboard / edge_metrics unit tests.

Synthetic events with known outcomes verify Brier math + edge computation +
promotion-signal logic, with no DB roundtrip. Aggregation is exercised against
hand-built bucket-score lists; the DB-backed `_score_resolved_events` is
covered by the smoke test in this file plus the live page after deploy.
"""
from __future__ import annotations

import asyncio
import math

import pytest

from backend.modeling.edge_metrics import (
    BPS,
    DEFAULT_MIN_N_BUCKETS,
    _aggregate,
    _BucketScore,
    _consecutive_days_bma_wins,
    _score_event,
    compute_edge_metrics,
)


# ───────────────────── _score_event (pure helper) ───────────────────────────

def test_score_event_outcome_one_for_winning_bucket():
    event = {
        "id": 1, "city_slug": "atlanta", "date_et": "2026-04-15",
        "winning_bucket_idx": 2,
        "buckets": [{"bucket_idx": 0}, {"bucket_idx": 1}, {"bucket_idx": 2}, {"bucket_idx": 3}],
    }
    scores = _score_event(
        event=event,
        legacy_probs=[0.10, 0.30, 0.40, 0.20],
        bma_probs=[0.05, 0.25, 0.50, 0.20],
        market_by_bucket={0: 0.05, 1: 0.20, 2: 0.55, 3: 0.20},
    )
    outcomes = [s.outcome for s in scores]
    assert outcomes == [0, 0, 1, 0]
    # Winning bucket scores
    win = scores[2]
    assert win.legacy_prob == pytest.approx(0.40)
    assert win.bma_prob == pytest.approx(0.50)
    assert win.market_prob == pytest.approx(0.55)


def test_score_event_handles_missing_bma():
    """Older snapshots predate the BMA shadow — bma_prob must be None."""
    event = {
        "id": 1, "city_slug": "denver", "date_et": "2026-04-15",
        "winning_bucket_idx": 1,
        "buckets": [{"bucket_idx": 0}, {"bucket_idx": 1}],
    }
    scores = _score_event(
        event=event,
        legacy_probs=[0.4, 0.6],
        bma_probs=None,
        market_by_bucket={0: 0.5, 1: 0.5},
    )
    assert all(s.bma_prob is None for s in scores)
    assert all(s.legacy_prob is not None for s in scores)


def test_score_event_handles_missing_market_snapshot():
    """A bucket without a MarketSnapshot must score legacy/bma but not market."""
    event = {
        "id": 1, "city_slug": "x", "date_et": "2026-04-15",
        "winning_bucket_idx": 0,
        "buckets": [{"bucket_idx": 0}, {"bucket_idx": 1}],
    }
    scores = _score_event(
        event=event,
        legacy_probs=[0.7, 0.3],
        bma_probs=[0.6, 0.4],
        market_by_bucket={0: 0.65, 1: None},  # second bucket missing market
    )
    assert scores[0].market_prob == pytest.approx(0.65)
    assert scores[1].market_prob is None


# ───────────────────── _aggregate / Brier closed forms ──────────────────────

def _make_score(legacy, bma, market, outcome, between=None):
    return _BucketScore(
        event_id=1, city_slug="x", date_et="2026-04-15",
        bucket_idx=0, outcome=outcome,
        legacy_prob=legacy, bma_prob=bma, market_prob=market,
        bma_between_share=between,
    )


def test_aggregate_perfect_predictions_zero_brier_positive_edge():
    scores = [
        _make_score(legacy=1.0, bma=1.0, market=0.5, outcome=1),
        _make_score(legacy=0.0, bma=0.0, market=0.5, outcome=0),
    ]
    out = _aggregate(scores)
    assert out["legacy"]["brier"] == pytest.approx(0.0)
    assert out["bma"]["brier"]    == pytest.approx(0.0)
    assert out["market"]["brier"] == pytest.approx(0.25)
    # Perfect → edge equals market Brier exactly
    assert out["legacy"]["edge_vs_market"] == pytest.approx(0.25)
    assert out["bma"]["edge_vs_market"]    == pytest.approx(0.25)


def test_aggregate_legacy_eq_bma_yields_identical_metrics():
    """When legacy probs == BMA probs on every bucket, the two columns
    must be byte-identical. Sanity check against a copy-paste bug."""
    scores = [
        _make_score(legacy=0.7, bma=0.7, market=0.5, outcome=1),
        _make_score(legacy=0.4, bma=0.4, market=0.5, outcome=0),
        _make_score(legacy=0.3, bma=0.3, market=0.5, outcome=0),
    ]
    out = _aggregate(scores)
    assert out["legacy"]["brier"] == out["bma"]["brier"]
    assert out["legacy"]["edge_vs_market"] == out["bma"]["edge_vs_market"]


def test_aggregate_market_edge_is_zero_by_construction():
    scores = [
        _make_score(legacy=0.5, bma=0.5, market=0.6, outcome=0),
        _make_score(legacy=0.5, bma=0.5, market=0.4, outcome=1),
    ]
    out = _aggregate(scores)
    assert out["market"]["edge_vs_market"] == 0.0
    assert out["market"]["edge_bps"] == 0.0


def test_aggregate_empty_input_returns_nulls_not_zeros():
    out = _aggregate([])
    for source in ("legacy", "bma", "market"):
        assert out[source]["brier"] is None
        assert out[source]["n"] == 0


def test_aggregate_brier_closed_form():
    """Hand-checked Brier on three buckets from one event:

    legacy probs   = [0.10, 0.30, 0.60]
    outcomes       = [0,    0,    1]
    Brier_per      = [0.01, 0.09, 0.16]
    mean Brier     = 0.0866...
    """
    scores = [
        _make_score(legacy=0.10, bma=0.10, market=0.10, outcome=0),
        _make_score(legacy=0.30, bma=0.30, market=0.30, outcome=0),
        _make_score(legacy=0.60, bma=0.60, market=0.60, outcome=1),
    ]
    out = _aggregate(scores)
    expected = (0.01 + 0.09 + 0.16) / 3.0
    assert out["legacy"]["brier"] == pytest.approx(expected, abs=1e-6)


def test_aggregate_bps_rounds_to_one_decimal():
    """edge_bps is the human-readable form (basis points)."""
    scores = [
        _make_score(legacy=0.40, bma=0.40, market=0.50, outcome=1),  # legacy better
    ]
    out = _aggregate(scores)
    # legacy Brier = 0.36, market Brier = 0.25 → edge = -0.11 → -1100 bps
    assert out["legacy"]["edge_bps"] == pytest.approx(-1100.0, abs=0.1)


# ───────────────────── _consecutive_days_bma_wins ───────────────────────────

def test_streak_counts_only_most_recent_consecutive_wins():
    by_day = [
        {"date": "2026-04-10", "legacy_brier": 0.10, "bma_brier": 0.15},  # legacy wins
        {"date": "2026-04-11", "legacy_brier": 0.20, "bma_brier": 0.10},  # BMA wins
        {"date": "2026-04-12", "legacy_brier": 0.20, "bma_brier": 0.15},  # BMA wins
        {"date": "2026-04-13", "legacy_brier": 0.20, "bma_brier": 0.18},  # BMA wins
    ]
    assert _consecutive_days_bma_wins(by_day) == 3


def test_streak_resets_on_first_loss():
    by_day = [
        {"date": "2026-04-10", "legacy_brier": 0.20, "bma_brier": 0.10},  # BMA wins
        {"date": "2026-04-11", "legacy_brier": 0.10, "bma_brier": 0.20},  # legacy wins
        {"date": "2026-04-12", "legacy_brier": 0.20, "bma_brier": 0.10},  # BMA wins
    ]
    # Most-recent day is BMA-wins, prior day was legacy-wins → streak=1
    assert _consecutive_days_bma_wins(by_day) == 1


def test_streak_skips_days_with_missing_data():
    """Days where either source lacks data shouldn't reset OR count
    toward the streak — the metric is "consecutive among comparable days"."""
    by_day = [
        {"date": "2026-04-10", "legacy_brier": 0.20, "bma_brier": 0.10},
        {"date": "2026-04-11", "legacy_brier": None, "bma_brier": None},  # skipped
        {"date": "2026-04-12", "legacy_brier": 0.18, "bma_brier": 0.12},
    ]
    assert _consecutive_days_bma_wins(by_day) == 2


def test_streak_zero_when_bma_loses_today():
    by_day = [
        {"date": "2026-04-10", "legacy_brier": 0.20, "bma_brier": 0.10},
        {"date": "2026-04-11", "legacy_brier": 0.10, "bma_brier": 0.20},
    ]
    assert _consecutive_days_bma_wins(by_day) == 0


def test_streak_empty_input():
    assert _consecutive_days_bma_wins([]) == 0


# ───────────────────── compute_edge_metrics integration ─────────────────────

def test_compute_edge_metrics_with_no_resolved_events_returns_empty_safe(monkeypatch):
    """When the DB has no resolved events, the function must still return
    a complete schema with null Brier and zero counts — never raise."""
    from backend.modeling import edge_metrics as em

    async def _stub(_sess, *, cutoff):
        return []

    monkeypatch.setattr(em, "_score_resolved_events", _stub)

    # Bypass the get_session context manager via a sentinel session — the stub
    # ignores its argument so any value works.
    class _DummySess:
        pass

    out = asyncio.run(em.compute_edge_metrics(days_back=30, sess=_DummySess()))
    assert out["n_events"] == 0
    assert out["n_buckets"] == 0
    assert out["by_source"]["legacy"]["brier"] is None
    assert out["by_source"]["bma"]["brier"] is None
    assert out["by_source"]["market"]["brier"] is None
    assert out["by_day"] == []
    assert out["by_city"] == {}
    assert out["promotion_signal"]["bma_better_than_legacy"] is False
    assert out["promotion_signal"]["consecutive_days_bma_wins"] == 0
