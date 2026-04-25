"""Phase B4 — ExitEvent structured journal.

Smoke-test the ExitEvent ORM model and the _emit_exit_event helper.
The web route is exercised manually; here we just confirm rows persist
correctly under the same monkeypatched session pattern used by the
cascade integration tests.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import backend.execution.exit_engine as ee
from backend.engine.signal_engine import BucketSignal


def _make_signal(**kw) -> BucketSignal:
    defaults = dict(
        city_slug="atlanta", city_display="Atlanta", unit="F",
        event_id=1, bucket_id=100, bucket_idx=3, label="75-78°F",
        low_f=75.0, high_f=78.0,
        model_prob=0.40, mkt_prob=0.35,
        raw_edge=0.05, exec_cost=0.02, true_edge=0.03,
        ev_per_share=0.05, ev_at_bid=0.04,
        yes_bid=0.30, yes_ask=0.32, yes_mid=0.31, spread=0.02,
        yes_ask_depth=50.0, yes_bid_depth=50.0,
        reason={"current_temp_f": 70.0, "raw_high": None},
        city_state="early",
    )
    defaults.update(kw)
    return BucketSignal(**defaults)


def _make_position(**kw):
    defaults = dict(
        id=42, bucket_id=100, avg_cost=0.25, net_qty=100.0,
        original_qty=100.0, moon_bag_qty=0.0,
        tier_1_exited=False, tier_2_exited=False,
        max_bid_seen=0.0, trailing_stop_price=None,
        entry_time=datetime.now(timezone.utc) - timedelta(seconds=7200),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_emit_exit_event_writes_row(monkeypatch):
    """Helper writes one ExitEvent with the cascade dict serialized in reason_json."""
    captured: list = []

    @asynccontextmanager
    async def _fake_session():
        class _Sess:
            def add(self, obj):
                captured.append(obj)
            async def commit(self):
                pass
        yield _Sess()

    monkeypatch.setattr(ee, "get_session", _fake_session)

    pos = _make_position()
    sig = _make_signal()
    cascade = {"level": "EDGE_DECAY", "price": 0.30, "reason": "ev_decayed", "qty_override": 100.0}

    asyncio.run(ee._emit_exit_event(
        pos, sig, cascade,
        shares_exited=100.0,
        execution_status="filled",
    ))

    assert len(captured) == 1
    row = captured[0]
    assert row.position_id == 42
    assert row.bucket_id == 100
    assert row.trigger_level == "EDGE_DECAY"
    assert row.trigger_reason == "ev_decayed"
    assert row.ev_at_bid_pre == pytest.approx(0.04)
    assert row.market_bid == pytest.approx(0.30)
    assert row.market_ask == pytest.approx(0.32)
    assert row.shares_exited == pytest.approx(100.0)
    assert row.shares_remaining == pytest.approx(0.0)

    payload = json.loads(row.reason_json)
    assert payload["execution_status"] == "filled"
    assert payload["cascade"]["level"] == "EDGE_DECAY"


def test_emit_exit_event_failure_path_records_zero_shares(monkeypatch):
    captured: list = []

    @asynccontextmanager
    async def _fake_session():
        class _Sess:
            def add(self, obj):
                captured.append(obj)
            async def commit(self):
                pass
        yield _Sess()

    monkeypatch.setattr(ee, "get_session", _fake_session)

    pos = _make_position()
    sig = _make_signal()
    cascade = {"level": "URGENT", "price": 0.29, "reason": "consensus_shifted"}

    asyncio.run(ee._emit_exit_event(
        pos, sig, cascade,
        shares_exited=0.0,
        execution_status="rejected",
        error="insufficient_balance",
    ))

    assert len(captured) == 1
    row = captured[0]
    assert row.shares_exited == 0.0
    assert row.shares_remaining == pytest.approx(100.0)
    payload = json.loads(row.reason_json)
    assert payload["execution_status"] == "rejected"
    assert payload["error"] == "insufficient_balance"


def test_emit_exit_event_swallows_db_errors(monkeypatch):
    """A DB failure on the journal write must not crash the exit cascade."""
    @asynccontextmanager
    async def _broken_session():
        raise RuntimeError("db offline")
        yield  # pragma: no cover

    monkeypatch.setattr(ee, "get_session", _broken_session)

    # Should not raise.
    asyncio.run(ee._emit_exit_event(
        _make_position(), _make_signal(),
        {"level": "EDGE_DECAY", "price": 0.30, "reason": "ev_decayed"},
        shares_exited=10.0,
        execution_status="filled",
    ))
