"""Unit tests for the EDGE_DECAY exit gate (Phase A2).

Covers the pure logic of `_edge_decay_triggered` plus in-memory cache
trimming. The DB-level integration (warming the cache from EVHistory rows
and persisting on each cycle) is exercised by the cascade integration
tests in Phase A6.
"""
from __future__ import annotations

import pytest

import backend.execution.exit_engine as ee
from backend.config import Config


@pytest.fixture(autouse=True)
def _reset_ev_cache():
    """Each test gets a clean in-memory EV cache."""
    saved = ee._ev_cache.copy()
    ee._ev_cache.clear()
    yield
    ee._ev_cache.clear()
    ee._ev_cache.update(saved)


# ── _edge_decay_triggered: pure-logic tests ──────────────────────────────

def test_edge_decay_triggered_false_when_history_empty():
    assert ee._edge_decay_triggered(bucket_id=42) is False


def test_edge_decay_triggered_false_when_below_debounce_count():
    """A single negative observation isn't enough — must persist for N runs."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[42] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * (n - 1)
    assert ee._edge_decay_triggered(bucket_id=42) is False


def test_edge_decay_triggered_true_when_last_n_all_below_threshold():
    """N consecutive observations at or below threshold trigger the gate."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[42] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n
    assert ee._edge_decay_triggered(bucket_id=42) is True


def test_edge_decay_triggered_true_at_exact_threshold():
    """`<=` boundary: an observation equal to the threshold counts as decayed."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[42] = [Config.EDGE_DECAY_THRESHOLD] * n
    assert ee._edge_decay_triggered(bucket_id=42) is True


def test_edge_decay_one_recent_rebound_suppresses_trigger():
    """A single positive observation in the trailing window resets the gate."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    # First N-1 are bad, but the most recent one rebounds above threshold.
    hist = [Config.EDGE_DECAY_THRESHOLD - 0.01] * (n - 1) + [0.05]
    ee._ev_cache[42] = hist
    assert ee._edge_decay_triggered(bucket_id=42) is False


def test_edge_decay_only_evaluates_trailing_window():
    """Old positive observations don't immunize against a new run of negatives."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    # Older positives, then N consecutive negatives at the tail.
    hist = [0.10, 0.08, 0.05] + [Config.EDGE_DECAY_THRESHOLD - 0.01] * n
    ee._ev_cache[42] = hist
    assert ee._edge_decay_triggered(bucket_id=42) is True


def test_edge_decay_isolated_per_bucket():
    """Cache is keyed by bucket_id — one bucket's history can't leak into another."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[1] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n
    ee._ev_cache[2] = [0.10] * n
    assert ee._edge_decay_triggered(bucket_id=1) is True
    assert ee._edge_decay_triggered(bucket_id=2) is False
    assert ee._edge_decay_triggered(bucket_id=999) is False  # untracked bucket


# ── Cascade integration tests (Phase A6) ────────────────────────────────
# Exercise _run_exit_cascade_for_position end-to-end with the new EDGE_DECAY
# gate and the EV-corroboration suppression on URGENT. Plumbing for DB lookups
# is monkey-patched out so the test stays a fast unit test.

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.engine.signal_engine import BucketSignal


def _make_signal(
    *,
    bucket_id: int = 100,
    ev_at_bid: float | None = 0.05,
    yes_bid: float = 0.30,
    yes_bid_depth: float = 50.0,
    spread: float = 0.02,
    model_prob: float = 0.40,
    bucket_idx: int = 3,
    high_f: float = 78.0,
    low_f: float = 75.0,
    city_state: str = "early",
) -> BucketSignal:
    """Synthetic BucketSignal with sane defaults for cascade tests."""
    return BucketSignal(
        city_slug="atlanta",
        city_display="Atlanta",
        unit="F",
        event_id=1,
        bucket_id=bucket_id,
        bucket_idx=bucket_idx,
        label="75-78°F",
        low_f=low_f,
        high_f=high_f,
        model_prob=model_prob,
        mkt_prob=0.35,
        raw_edge=0.05,
        exec_cost=0.02,
        true_edge=0.03,
        ev_per_share=0.05,
        ev_at_bid=ev_at_bid,
        yes_bid=yes_bid,
        yes_ask=yes_bid + spread,
        yes_mid=yes_bid + spread / 2,
        spread=spread,
        yes_ask_depth=50.0,
        yes_bid_depth=yes_bid_depth,
        reason={"current_temp_f": 70.0, "raw_high": None},
        city_state=city_state,
    )


def _make_position(
    *,
    bucket_id: int = 100,
    avg_cost: float = 0.25,  # bid=0.30 → +5¢, below the 8¢ tier-1 target
    net_qty: float = 100.0,
    age_seconds: int = 7200,
    moon_bag_qty: float = 0.0,
    tier_1_exited: bool = False,
):
    """Synthetic position attached only via duck-typing — cascade reads attrs."""
    return SimpleNamespace(
        bucket_id=bucket_id,
        avg_cost=avg_cost,
        net_qty=net_qty,
        original_qty=net_qty,
        moon_bag_qty=moon_bag_qty,
        tier_1_exited=tier_1_exited,
        tier_2_exited=False,
        max_bid_seen=0.0,
        trailing_stop_price=None,
        entry_time=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )


@pytest.fixture
def stub_db(monkeypatch):
    """Replace get_session and get_city_by_slug so cascade can run without a DB.

    The trailing-stop update inside the cascade also runs SQL via get_session;
    we hand it a context manager whose session ignores writes.
    """
    @asynccontextmanager
    async def _fake_session():
        class _Sess:
            async def execute(self, *a, **k): return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
            async def commit(self): pass
            def add(self, *a, **k): pass
        yield _Sess()

    async def _fake_city_lookup(sess, slug):
        return SimpleNamespace(tz="America/New_York", city_slug=slug)

    monkeypatch.setattr("backend.execution.exit_engine.get_session", _fake_session)
    monkeypatch.setattr("backend.execution.exit_engine.get_city_by_slug", _fake_city_lookup)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_cascade_edge_decay_fires_after_n_consecutive_negative_evs(stub_db):
    """EV decays over EDGE_DECAY_DEBOUNCE_RUNS runs → EDGE_DECAY exit fires."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    bucket_id = 100
    ee._ev_cache[bucket_id] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n

    pos = _make_position(bucket_id=bucket_id, age_seconds=7200)
    signal = _make_signal(bucket_id=bucket_id, ev_at_bid=Config.EDGE_DECAY_THRESHOLD - 0.01)

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is not None
    assert result["level"] == "EDGE_DECAY"
    assert result["reason"] == "ev_decayed"


def test_cascade_consensus_shift_with_positive_ev_suppresses_both_exits(stub_db):
    """Held bucket still +EV → EDGE_DECAY doesn't fire AND URGENT is EV-suppressed."""
    bucket_id = 100
    consensus_bucket_id = 101  # different bucket
    # No EV decay history — held bucket is fine.
    ee._ev_cache[bucket_id] = [0.05, 0.04, 0.06]

    pos = _make_position(bucket_id=bucket_id, age_seconds=7200)
    signal = _make_signal(bucket_id=bucket_id, ev_at_bid=0.04, model_prob=0.40)
    consensus_sig = _make_signal(
        bucket_id=consensus_bucket_id, ev_at_bid=0.10, model_prob=0.55, bucket_idx=4,
    )

    result = _run(ee._run_exit_cascade_for_position(
        pos, signal, consensus_bucket_id, consensus_sig,
    ))
    assert result is None  # no exit fires


def test_cascade_consensus_shift_with_decayed_ev_fires_edge_decay_first(stub_db):
    """When both gates would fire, EDGE_DECAY supersedes URGENT (it's earlier in the cascade)."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    bucket_id = 100
    consensus_bucket_id = 101
    # EV history is fully decayed.
    ee._ev_cache[bucket_id] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n

    pos = _make_position(bucket_id=bucket_id, age_seconds=7200)
    signal = _make_signal(bucket_id=bucket_id, ev_at_bid=Config.EDGE_DECAY_THRESHOLD - 0.01)
    consensus_sig = _make_signal(
        bucket_id=consensus_bucket_id, ev_at_bid=0.10, model_prob=0.55, bucket_idx=4,
    )

    result = _run(ee._run_exit_cascade_for_position(
        pos, signal, consensus_bucket_id, consensus_sig,
    ))
    assert result is not None
    assert result["level"] == "EDGE_DECAY"  # NOT URGENT


def test_cascade_edge_decay_suppressed_when_position_too_young(stub_db):
    """Even with decayed EV history, age guard prevents premature EDGE_DECAY."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    bucket_id = 100
    ee._ev_cache[bucket_id] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n

    too_young = Config.EDGE_DECAY_MIN_POSITION_AGE_SECONDS - 60
    pos = _make_position(bucket_id=bucket_id, age_seconds=too_young)
    signal = _make_signal(bucket_id=bucket_id, ev_at_bid=Config.EDGE_DECAY_THRESHOLD - 0.01)

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is None  # age gate held


def test_cascade_edge_decay_suppressed_when_bid_below_floor(stub_db):
    """Even with decayed EV, refuse to sell into a dead book (bid < EDGE_DECAY_MIN_BID)."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    bucket_id = 100
    ee._ev_cache[bucket_id] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n

    pos = _make_position(bucket_id=bucket_id, age_seconds=7200)
    too_thin = Config.EDGE_DECAY_MIN_BID - 0.005
    signal = _make_signal(
        bucket_id=bucket_id,
        ev_at_bid=Config.EDGE_DECAY_THRESHOLD - 0.01,
        yes_bid=too_thin,
    )

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is None  # bid floor held


def test_cascade_no_action_when_signal_has_no_ev_at_bid(stub_db):
    """Missing market data (ev_at_bid is None) cannot trigger EDGE_DECAY."""
    bucket_id = 100
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[bucket_id] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n

    pos = _make_position(bucket_id=bucket_id, age_seconds=7200)
    signal = _make_signal(bucket_id=bucket_id, ev_at_bid=None, yes_bid=0.30)

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is None
