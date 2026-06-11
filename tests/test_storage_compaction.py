import json
from datetime import datetime, timezone

from backend.config import Config
from backend.market_context.wallet_tracker import MarketRef, PublicTrade, _trade_to_db_kwargs
from backend.storage.repos import compact_signal_reason_json, insert_forecast_obs
from backend.storage.models import Base, ForecastObs
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
import asyncio


def _run(coro):
    return asyncio.run(coro)


def test_compact_signal_reason_preserves_operational_fields(monkeypatch):
    monkeypatch.setattr(Config, "STORE_FULL_SIGNAL_REASON_JSON", False)
    monkeypatch.setattr(Config, "SIGNAL_REASON_MAX_JSON_BYTES", 6000)
    reason = {
        "ev_at_bid": 0.047,
        "mu_forecast": 92.3,
        "prob_hotter_bucket": 0.64,
        "regime_score": 0.7,
        "microstructure_shadow": {"toxicity_score": 0.2, "huge": ["x" * 200] * 20},
        "ensemble_breakdown": {"massive": "x" * 10000},
    }
    compact = json.loads(compact_signal_reason_json(reason))
    assert compact["ev_at_bid"] == 0.047
    assert compact["mu_forecast"] == 92.3
    assert compact["prob_hotter_bucket"] == 0.64
    assert compact["regime_score"] == 0.7
    assert "microstructure_shadow" in compact
    assert "ensemble_breakdown" not in compact


def test_wallet_trade_raw_payload_disabled_by_default(monkeypatch):
    monkeypatch.setattr(Config, "STORE_RAW_WALLET_PAYLOADS", False)
    trade = PublicTrade(
        wallet_address="0xabc",
        market_slug="atlanta-weather",
        condition_id="cond",
        asset_id="asset",
        side="BUY",
        size=10.0,
        price=0.25,
        timestamp=datetime(2026, 6, 11, tzinfo=timezone.utc),
        transaction_hash="0xhash",
        raw={"duplicated": "payload"},
    )
    market = MarketRef(
        city_slug="atlanta",
        date="2026-06-11",
        market_slug="atlanta-weather",
        condition_id="cond",
    )
    assert _trade_to_db_kwargs(trade, market)["raw_json"] is None


def test_forecast_insert_keeps_compact_wu_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "STORE_RAW_FORECAST_PAYLOADS", False)

    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'compact.db'}")
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            await insert_forecast_obs(
                session,
                city_id=1,
                source="wu_history",
                date_et="2026-06-11",
                high_f=91.0,
                raw_json=json.dumps({
                    "obs_time": "2026-06-11T20:52:00Z",
                    "high_f": 91.0,
                    "giant_payload": "x" * 10000,
                }),
                parse_error="x" * 1000,
            )
            row = (await session.execute(select(ForecastObs))).scalar_one()
        await engine.dispose()
        return row

    row = _run(scenario())
    payload = json.loads(row.raw_json)
    assert payload["obs_time"] == "2026-06-11T20:52:00Z"
    assert payload["high_f"] == 91.0
    assert "giant_payload" not in payload
    assert len(row.parse_error) <= 256
