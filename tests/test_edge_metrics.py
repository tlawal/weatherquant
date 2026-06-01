"""Section 6 Layer 6 — alpha dashboard / edge_metrics unit tests.

Synthetic events with known outcomes verify Brier math + edge computation +
promotion-signal logic, with no DB roundtrip. Aggregation is exercised against
hand-built bucket-score lists; the DB-backed `_score_resolved_events` is
covered by the smoke test in this file plus the live page after deploy.
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.modeling.edge_metrics import (
    BPS,
    DEFAULT_MIN_N_BUCKETS,
    _aggregate,
    _aggregate_crps,
    _BucketScore,
    _bma_mixture_crps,
    _consecutive_days_bma_wins,
    _DistributionScore,
    _discrete_crps,
    _normal_crps,
    _score_event,
    _score_distribution_event,
    compute_edge_metrics,
)
from backend.storage.models import (
    Base,
    Bucket,
    City,
    Event,
    MarketSnapshot,
    MetarObs,
    ModelSnapshot,
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


# ───────────────────── CRPS helpers ─────────────────────────────────────────

def test_normal_crps_closed_form_is_low_when_observed_near_mu():
    centered = _normal_crps(80.0, 2.0, 80.0)
    missed = _normal_crps(80.0, 2.0, 86.0)

    assert centered == pytest.approx(2.0 * (2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)))
    assert missed > centered


def test_bma_mixture_crps_matches_single_normal_component():
    components = [{"mu": 82.0, "sigma": 2.5, "weight": 1.0}]

    assert _bma_mixture_crps(components, 83.0) == pytest.approx(
        _normal_crps(82.0, 2.5, 83.0)
    )


def test_discrete_crps_rewards_mass_near_realized_high():
    reps = {0: 80.0, 1: 84.0, 2: 88.0}
    good = _discrete_crps({0: 0.05, 1: 0.85, 2: 0.10}, reps, 84.0)
    bad = _discrete_crps({0: 0.85, 1: 0.10, 2: 0.05}, reps, 84.0)

    assert good < bad


def test_score_distribution_event_scores_legacy_bma_and_market():
    event = {
        "id": 7,
        "city_slug": "atlanta",
        "date_et": "2026-05-30",
        "buckets": [
            {"bucket_idx": 0, "low_f": None, "high_f": 81.0},
            {"bucket_idx": 1, "low_f": 82.0, "high_f": 83.0},
            {"bucket_idx": 2, "low_f": 84.0, "high_f": None},
        ],
        "settlement_status": "provisional_local",
        "score_checkpoint_utc": "2026-05-30T16:00:00+00:00",
    }
    score = _score_distribution_event(
        event=event,
        legacy_mu=82.0,
        legacy_sigma=2.0,
        bma_shadow={
            "between_share": 0.25,
            "components": [
                {"mu": 82.0, "sigma": 1.8, "weight": 0.7},
                {"mu": 85.0, "sigma": 2.2, "weight": 0.3},
            ],
        },
        market_by_bucket={0: 0.10, 1: 0.60, 2: 0.30},
        resolved_high_f=83.0,
    )

    assert score.event_id == 7
    assert score.settlement_status == "provisional_local"
    assert score.legacy_crps is not None
    assert score.bma_crps is not None
    assert score.market_crps is not None
    assert score.bma_between_share == pytest.approx(0.25)


def test_aggregate_crps_reports_edge_in_centi_degrees():
    scores = [
        _DistributionScore(
            event_id=1,
            city_slug="x",
            date_et="2026-05-30",
            resolved_high_f=83.0,
            legacy_crps=0.8,
            bma_crps=0.6,
            market_crps=1.1,
        ),
        _DistributionScore(
            event_id=2,
            city_slug="x",
            date_et="2026-05-31",
            resolved_high_f=84.0,
            legacy_crps=1.0,
            bma_crps=0.9,
            market_crps=1.2,
        ),
    ]

    out = _aggregate_crps(scores)
    assert out["legacy"]["crps"] == pytest.approx(0.9)
    assert out["bma"]["crps"] == pytest.approx(0.75)
    assert out["market"]["crps"] == pytest.approx(1.15)
    assert out["legacy"]["edge_cdeg"] == pytest.approx(25.0)
    assert out["bma"]["edge_cdeg"] == pytest.approx(40.0)


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

async def _setup_edge_db(tmp_path):
    db_path = tmp_path / "edge_metrics_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, session_factory


def _past_date(days_back: int = 3) -> str:
    return (
        datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days_back)
    ).strftime("%Y-%m-%d")


def _checkpoint(date_et: str) -> datetime:
    return datetime.strptime(date_et, "%Y-%m-%d").replace(
        hour=12,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(timezone.utc)


async def _seed_scored_event(session, *, date_et: str, resolved: bool = False):
    city = City(
        city_slug="atlanta",
        display_name="Atlanta",
        metar_station="KATL",
        enabled=True,
        is_us=True,
        unit="F",
        tz="America/New_York",
    )
    session.add(city)
    await session.flush()

    event = Event(
        city_id=city.id,
        date_et=date_et,
        status="ok",
        trading_enabled=True,
        winning_bucket_idx=3 if resolved else None,
        resolved_at=(_checkpoint(date_et) + timedelta(hours=8)) if resolved else None,
    )
    session.add(event)
    await session.flush()

    ranges = [(None, 75.0), (76.0, 77.0), (78.0, 79.0), (80.0, 81.0)]
    buckets = []
    for idx, (low_f, high_f) in enumerate(ranges):
        bucket = Bucket(
            event_id=event.id,
            bucket_idx=idx,
            label=f"b{idx}",
            low_f=low_f,
            high_f=high_f,
            condition_id=f"cond-{event.id}-{idx}",
        )
        session.add(bucket)
        buckets.append(bucket)
    await session.flush()

    checkpoint = _checkpoint(date_et)
    session.add(MetarObs(
        city_id=city.id,
        metar_station="KATL",
        observed_at=checkpoint + timedelta(hours=5),
        temp_f=80.1,
        source="aviation",
    ))
    session.add(ModelSnapshot(
        event_id=event.id,
        computed_at=checkpoint - timedelta(hours=1),
        mu=80.0,
        sigma=2.0,
        probs_json=json.dumps([0.10, 0.20, 0.30, 0.40]),
        inputs_json=json.dumps({
            "bma_shadow": {
                "mean": 80.4,
                "sigma": 2.1,
                "between_share": 0.2,
                "probs": [0.05, 0.15, 0.25, 0.55],
                "components": [
                    {"source": "hrrr", "mu": 80.2, "sigma": 1.9, "weight": 0.6},
                    {"source": "nws", "mu": 80.8, "sigma": 2.4, "weight": 0.4},
                ],
            }
        }),
    ))
    session.add(ModelSnapshot(
        event_id=event.id,
        computed_at=checkpoint + timedelta(hours=8),
        mu=80.0,
        sigma=0.2,
        probs_json=json.dumps([0.0, 0.0, 0.0, 1.0]),
        inputs_json=json.dumps({"bma_shadow": {"probs": [0.0, 0.0, 0.0, 1.0]}}),
    ))
    for idx, bucket in enumerate(buckets):
        session.add(MarketSnapshot(
            bucket_id=bucket.id,
            fetched_at=checkpoint - timedelta(minutes=30),
            yes_bid=0.20,
            yes_ask=0.30,
            yes_mid=[0.05, 0.15, 0.30, 0.50][idx],
            yes_bid_depth=100.0,
            yes_ask_depth=100.0,
            spread=0.10,
        ))
    await session.commit()
    return city, event

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
    assert out["promotion_signal"]["bma_crps_better_than_legacy"] is False
    assert out["promotion_signal"]["consecutive_days_bma_wins"] == 0
    assert out["crps"]["by_source"]["legacy"]["crps"] is None


def test_compute_edge_metrics_scores_provisional_local_settlements(tmp_path):
    async def _case():
        engine, session_factory = await _setup_edge_db(tmp_path)
        try:
            async with session_factory() as session:
                await _seed_scored_event(session, date_et=_past_date(), resolved=False)
                out = await compute_edge_metrics(days_back=30, sess=session)
                assert out["n_events"] == 1
                assert out["settlement_status_counts"]["provisional_local"] == 1
                assert out["diagnostics"]["provisional_local_events"] == 1
                assert out["diagnostics"]["gamma_confirmed_events"] == 0
                # Uses the noon pre-resolution snapshot, not the later near-perfect
                # post-resolution snapshot.
                assert out["by_source"]["legacy"]["brier"] == pytest.approx(0.125)
                assert out["crps"]["n_events"] == 1
                assert out["crps"]["by_source"]["legacy"]["crps"] is not None
                assert out["crps"]["by_source"]["bma"]["crps"] is not None
                assert out["crps"]["by_source"]["market"]["crps"] is not None
                assert out["promotion_signal"]["bma_crps_better_than_legacy"] in (True, False)
        finally:
            await engine.dispose()

    asyncio.run(_case())


def test_compute_edge_metrics_labels_gamma_confirmed_settlements(tmp_path):
    async def _case():
        engine, session_factory = await _setup_edge_db(tmp_path)
        try:
            async with session_factory() as session:
                await _seed_scored_event(session, date_et=_past_date(), resolved=True)
                out = await compute_edge_metrics(days_back=30, sess=session)
                assert out["n_events"] == 1
                assert out["settlement_status_counts"]["gamma_confirmed"] == 1
                assert out["diagnostics"]["gamma_confirmed_events"] == 1
        finally:
            await engine.dispose()

    asyncio.run(_case())
