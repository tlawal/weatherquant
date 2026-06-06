import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import backend.engine.gating as gating
from backend.config import Config
from backend.engine.signal_engine import BucketSignal


def _run(coro):
    return asyncio.run(coro)


def _signal(**overrides):
    base = dict(
        city_slug="atlanta",
        city_display="Atlanta",
        unit="F",
        event_id=1,
        bucket_id=10,
        bucket_idx=4,
        label="84-85°F",
        low_f=84.0,
        high_f=85.0,
        model_prob=0.80,
        mkt_prob=0.30,
        raw_edge=0.50,
        exec_cost=0.02,
        true_edge=0.48,
        ev_per_share=0.40,
        ev_at_bid=0.35,
        yes_bid=0.29,
        yes_ask=0.31,
        yes_mid=0.30,
        spread=0.02,
        yes_ask_depth=200.0,
        yes_bid_depth=200.0,
        reason={},
        actionable=True,
    )
    base.update(overrides)
    return BucketSignal(**base)


def test_sell_exit_bypasses_entry_only_gates():
    event = SimpleNamespace(id=1, status="bad", settlement_source_verified=False, forecast_quality="degraded")
    signal = _signal(true_edge=-1.0, yes_ask=0.99, mkt_prob=0.99, spread=0.99, yes_ask_depth=0.0)

    result = _run(gating.run_all_gates(signal, event, city_id=1, side="SELL", emit_log=False))

    assert result.passed is True
    assert result.failures == []


def test_buy_max_entry_price_uses_yes_ask(monkeypatch):
    class DummyResult:
        def scalars(self):
            return self

        def all(self):
            return []

    class DummySession:
        async def execute(self, *_args, **_kwargs):
            return DummyResult()

        async def get(self, *_args, **_kwargs):
            return SimpleNamespace(tz="America/New_York")

    class DummyContext:
        async def __aenter__(self):
            return DummySession()

        async def __aexit__(self, *_args):
            return False

    async def fake_arming_state(_sess):
        return SimpleNamespace(state="ARMED")

    async def fake_latest_metar(_sess, _city_id):
        return SimpleNamespace(fetched_at=datetime.now(timezone.utc))

    async def fake_daily_realized_pnl(_sess, _today):
        return 0.0

    async def fake_all_positions(_sess):
        return []

    async def fake_daily_high_metar(*_args, **_kwargs):
        return None

    async def fake_portfolio_risk(**_kwargs):
        return []

    monkeypatch.setattr(gating, "get_session", lambda: DummyContext())
    monkeypatch.setattr(gating, "get_arming_state", fake_arming_state)
    monkeypatch.setattr(gating, "get_latest_metar", fake_latest_metar)
    monkeypatch.setattr(gating, "get_daily_realized_pnl", fake_daily_realized_pnl)
    monkeypatch.setattr(gating, "get_all_positions", fake_all_positions)
    monkeypatch.setattr(gating, "get_daily_high_metar", fake_daily_high_metar)
    monkeypatch.setattr(
        "backend.execution.portfolio_risk.check_portfolio_risk",
        fake_portfolio_risk,
    )
    monkeypatch.setattr(Config, "MAX_ENTRY_PRICE", 0.36, raising=False)
    monkeypatch.setattr(Config, "MIN_LIQUIDITY_SHARES", 100.0, raising=False)
    monkeypatch.setattr(Config, "MIN_TRUE_EDGE", 0.05, raising=False)
    monkeypatch.setattr(Config, "MAX_SPREAD", 0.10, raising=False)
    monkeypatch.setattr(Config, "METAR_STALE_TTL_SECONDS", 300, raising=False)
    monkeypatch.setattr(Config, "MAX_DAILY_LOSS", 10.0, raising=False)
    monkeypatch.setattr(Config, "MAX_POSITIONS_PER_EVENT", 3, raising=False)

    event = SimpleNamespace(
        id=1,
        status="ok",
        settlement_source_verified=True,
        settlement_source="WU",
        forecast_quality="ok",
    )
    signal = _signal(mkt_prob=0.30, yes_ask=0.50, true_edge=0.25)

    result = _run(gating.run_all_gates(signal, event, city_id=1, side="BUY", emit_log=False))

    assert result.passed is False
    assert any("GATE_MAX_PRICE: entry_price=0.5000" in f for f in result.failures)
