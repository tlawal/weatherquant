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
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

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
    entry_ev_at_bid: float | None = 0.04,
    entry_type: str = "AUTOMATIC",
    entry_strategy: str = "auto_edge",
    entry_model_prob: float = 0.40,
    entry_market_prob: float = 0.35,
):
    """Synthetic position attached only via duck-typing — cascade reads attrs."""
    entry_decision = {
        "entry_type": entry_type,
        "entry_strategy": entry_strategy,
        "model_prob": entry_model_prob,
        "market_prob": entry_market_prob,
        "ev_at_bid": entry_ev_at_bid,
        "yes_bid": entry_model_prob - entry_ev_at_bid if entry_ev_at_bid is not None else None,
        "source_highs": {
            "nws_high": 83.0,
            "hrrr_high": 82.0,
        },
    }
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
        entry_type=entry_type,
        entry_strategy=entry_strategy,
        strategy=entry_strategy,
        entry_decision_json=json.dumps(entry_decision),
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
    """EV decays and model probability drops → EDGE_DECAY exit fires."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    bucket_id = 100
    ee._ev_cache[bucket_id] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n

    pos = _make_position(bucket_id=bucket_id, age_seconds=7200)
    signal = _make_signal(
        bucket_id=bucket_id,
        ev_at_bid=Config.EDGE_DECAY_THRESHOLD - 0.01,
        model_prob=0.35,
    )

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is not None
    assert result["level"] == "EDGE_DECAY"
    assert result["reason"] == "ev_decayed"
    assert result["diagnostics"]["entry_ev_at_bid"] == pytest.approx(0.04)
    assert result["diagnostics"]["ev_drop"] >= Config.EDGE_DECAY_MIN_EV_DROP
    assert result["diagnostics"]["model_prob_drop"] >= Config.EDGE_DECAY_MIN_MODEL_PROB_DROP


def test_edge_decay_suppresses_market_reprice_without_model_deterioration():
    """A higher bid can make hold EV negative; that alone is not model edge decay."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    bucket_id = 100
    ee._ev_cache[bucket_id] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n
    pos = _make_position(
        bucket_id=bucket_id,
        avg_cost=0.40,
        age_seconds=7200,
        entry_ev_at_bid=0.04,
        entry_model_prob=0.40,
    )
    signal = _make_signal(
        bucket_id=bucket_id,
        ev_at_bid=Config.EDGE_DECAY_THRESHOLD - 0.01,
        yes_bid=0.415,
        model_prob=0.40,
    )

    allowed, diagnostics = ee._edge_decay_exit_allowed(
        pos, signal, age_s=7200, bid=signal.yes_bid
    )
    assert allowed is False
    assert diagnostics["blocked_reason"] == "no_model_deterioration"
    assert diagnostics["model_prob_drop"] == pytest.approx(0.0)
    assert diagnostics["bid_delta"] > 0


def test_cascade_edge_decay_does_not_sell_manual_negative_edge_that_improved(stub_db):
    """Atlanta regression: a manual negative-EV entry cannot exit just because EV stays negative."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    bucket_id = 100
    ee._ev_cache[bucket_id] = [-0.18, -0.15, -0.13][-n:]

    pos = _make_position(
        bucket_id=bucket_id,
        age_seconds=47 * 60,
        avg_cost=0.32,
        net_qty=5.0,
        entry_type="MANUAL",
        entry_strategy="manual_scalp",
        entry_ev_at_bid=-0.1933,
        entry_model_prob=0.1267,
        entry_market_prob=0.32,
    )
    signal = _make_signal(
        bucket_id=bucket_id,
        ev_at_bid=-0.1665,
        yes_bid=0.31,
        model_prob=0.1785,
    )
    signal.true_edge = -0.1665

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is None


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
    signal = _make_signal(
        bucket_id=bucket_id,
        ev_at_bid=Config.EDGE_DECAY_THRESHOLD - 0.01,
        model_prob=0.35,
    )
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


def test_expiry_likely_winner_passive_sell_not_ten_cent_dump(stub_db, monkeypatch):
    """Near close, a likely winner may be passively offered near par, not dumped 10c below bid."""
    now = datetime(2026, 5, 13, 19, 49, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("backend.execution.exit_engine.city_local_now", lambda city: now)
    pos = _make_position(avg_cost=0.95, net_qty=2.0, age_seconds=9000)
    signal = _make_signal(
        low_f=80.0,
        high_f=81.0,
        yes_bid=0.995,
        spread=0.003,
        model_prob=0.90,
    )
    signal.reason = {"raw_high": 80.6}

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is not None
    assert result["level"] == "PROFIT"
    assert result["reason"] == "expiry_passive_winner"
    assert result["price"] >= 0.99


def test_expiry_likely_winner_clamps_passive_price_to_clob_max(stub_db, monkeypatch):
    """Near-par bids above $0.99 must not produce invalid CLOB limit prices."""
    now = datetime(2026, 5, 13, 19, 49, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("backend.execution.exit_engine.city_local_now", lambda city: now)
    pos = _make_position(avg_cost=0.95, net_qty=1.0, age_seconds=9000)
    signal = _make_signal(
        low_f=74.0,
        high_f=75.0,
        yes_bid=0.998,
        spread=0.0,
        model_prob=0.90,
    )
    signal.reason = {"raw_high": 75.0}

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is not None
    assert result["level"] == "PROFIT"
    assert result["reason"] == "expiry_passive_winner"
    assert result["price"] == 0.99
    assert result["diagnostics"]["pre_cap_bid"] == 0.998
    assert result["diagnostics"]["reference_price"] == 0.998
    assert result["diagnostics"]["order_price"] == 0.99
    assert result["diagnostics"]["price_adjustment"]["reason"] == "clamped_to_clob_limit_range"


def test_expiry_likely_winner_holds_when_bid_below_passive_floor(stub_db, monkeypatch):
    """A likely winner with a sub-par bid is held to redeem instead of force-sold."""
    now = datetime(2026, 5, 13, 19, 49, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("backend.execution.exit_engine.city_local_now", lambda city: now)
    pos = _make_position(avg_cost=0.95, net_qty=2.0, age_seconds=9000)
    signal = _make_signal(
        low_f=80.0,
        high_f=81.0,
        yes_bid=0.970,
        spread=0.005,
        model_prob=0.90,
    )
    signal.reason = {"raw_high": 80.6}

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is not None
    assert result["level"] == "HOLD"
    assert result["no_order"] is True
    assert result["status"] == "HOLD_TO_REDEEM"


def test_expiry_ambiguous_risk_exit_uses_small_discount(stub_db, monkeypatch):
    """Ambiguous near-close buckets can risk-exit, but use the 1-2c cap, not the old 10c dump."""
    now = datetime(2026, 5, 13, 19, 49, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("backend.execution.exit_engine.city_local_now", lambda city: now)
    pos = _make_position(avg_cost=0.60, net_qty=10.0, age_seconds=9000)
    signal = _make_signal(
        low_f=80.0,
        high_f=81.0,
        yes_bid=0.50,
        spread=0.02,
        model_prob=0.20,
        ev_at_bid=-0.05,
    )
    signal.reason = {"raw_high": 75.0}

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is not None
    assert result["level"] == "EXPIRY"
    assert result["reason"] == "market_close_risk_exit"
    assert result["price"] == pytest.approx(0.48)


def test_expiry_loss_exit_blocked_when_bucket_still_positive_ev(stub_db, monkeypatch):
    """Positive EV at bid blocks non-emergency expiry loss exits."""
    now = datetime(2026, 5, 13, 19, 49, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("backend.execution.exit_engine.city_local_now", lambda city: now)
    pos = _make_position(avg_cost=0.60, net_qty=10.0, age_seconds=9000)
    signal = _make_signal(
        low_f=80.0,
        high_f=81.0,
        yes_bid=0.50,
        spread=0.02,
        model_prob=0.20,
        ev_at_bid=0.05,
    )
    signal.reason = {"raw_high": 75.0}

    result = _run(ee._run_exit_cascade_for_position(pos, signal, None, None))
    assert result is not None
    assert result["level"] == "HOLD"
    assert result["no_order"] is True
    assert result["status"] == "EXIT_BLOCKED"
