import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.storage.db as storage_db
from backend.config import Config
from backend.market_context.wallet_tracker import (
    MarketRef,
    PublicTrade,
    build_wallet_leaderboard_payload,
    calculate_consistency_score,
    compute_smart_money_divergence,
    compute_wallet_metrics,
    infer_strategy_style,
    truncate_wallet_address,
    update_wallet_rankings,
)
from backend.storage.models import Base, City, WalletStat
from backend.storage.repos import get_wallet_stats_for_city, upsert_wallet_stat


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "wallet_tracker_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


def _trade(wallet: str, condition_id: str, side: str, size: float, price: float, ts: datetime) -> PublicTrade:
    return PublicTrade(
        wallet_address=wallet.lower(),
        condition_id=condition_id,
        side=side,
        size=size,
        price=price,
        timestamp=ts,
    )


def test_wallet_address_truncation():
    addr = "0x1234567890abcdef1234567890abcdef12345678"
    assert truncate_wallet_address(addr) == "0x1234...5678"
    assert truncate_wallet_address(addr, enabled=False) == addr
    assert truncate_wallet_address(None) == ""


def test_consistency_score_rewards_repeatable_quality():
    strong = calculate_consistency_score(
        sharpe_like=1.8,
        profitable_days_pct=0.9,
        win_rate=0.8,
        volume_usd=5000,
        activity_consistency=0.8,
    )
    weak = calculate_consistency_score(
        sharpe_like=-0.5,
        profitable_days_pct=0.25,
        win_rate=0.3,
        volume_usd=120,
        activity_consistency=0.2,
    )
    assert 0 <= weak < strong <= 1


def test_filter_wallets_below_min_volume_and_trades():
    now = datetime(2026, 5, 16, 12, tzinfo=timezone.utc)
    markets = [
        MarketRef(city_slug="atlanta", date="2026-05-16", condition_id="cond-a", bucket_idx=1, current_price=0.6),
        MarketRef(city_slug="atlanta", date="2026-05-15", condition_id="cond-b", bucket_idx=1, current_price=0.6),
    ]
    trades = [
        _trade("0x1234567890abcdef1234567890abcdef12345678", "cond-a", "BUY", 1, 0.1, now),
    ]
    assert compute_wallet_metrics(
        trades,
        markets,
        min_volume_usd=100,
        min_trades=3,
        min_active_days=1,
    ) == []


def test_compute_wallet_metrics_positive_wallet():
    wallet = "0x1234567890abcdef1234567890abcdef12345678"
    now = datetime(2026, 5, 16, 12, tzinfo=timezone.utc)
    markets = [
        MarketRef(city_slug="atlanta", date="2026-05-16", condition_id="cond-a", bucket_idx=2, bucket_label="88-90F", current_price=0.8),
        MarketRef(city_slug="atlanta", date="2026-05-15", condition_id="cond-b", bucket_idx=3, bucket_label="90-92F", current_price=1.0),
    ]
    trades = [
        _trade(wallet, "cond-a", "BUY", 200, 0.30, now),
        _trade(wallet, "cond-a", "SELL", 100, 0.55, now + timedelta(minutes=30)),
        _trade(wallet, "cond-b", "BUY", 150, 0.40, now - timedelta(days=1)),
        _trade(wallet, "cond-b", "SELL", 150, 0.75, now - timedelta(days=1, minutes=-60)),
    ]
    metrics = compute_wallet_metrics(
        trades,
        markets,
        min_volume_usd=10,
        min_trades=3,
        min_active_days=2,
        as_of_date="2026-05-16",
    )
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.wallet_address == wallet
    assert metric.realized_pnl > 0
    assert metric.unrealized_pnl > 0
    assert metric.consistency_score is not None and metric.consistency_score > 0


def test_strategy_style_inference():
    wallet = "0x1234567890abcdef1234567890abcdef12345678"
    base = datetime(2026, 5, 16, 12, 52, tzinfo=timezone.utc)
    trades = [
        _trade(wallet, "cond-a", "BUY", 10, 0.2, base + timedelta(minutes=i * 4))
        for i in range(3)
    ] + [
        _trade(wallet, "cond-a", "SELL", 10, 0.25, base + timedelta(minutes=20 + i * 4))
        for i in range(3)
    ]
    assert infer_strategy_style(
        trades,
        avg_hold_minutes=20,
        bucket_indices={1},
        observation_minutes=[52],
    ) == "Observation Flipper"
    assert infer_strategy_style(
        trades,
        avg_hold_minutes=20,
        bucket_indices={1},
        observation_minutes=None,
    ) == "Scalper"
    assert infer_strategy_style(
        trades[:3],
        avg_hold_minutes=240,
        bucket_indices={1, 2, 3},
        observation_minutes=None,
    ) == "Bucket Rotator"


def test_wallet_stat_upsert_is_idempotent(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))

    async def run_test():
        async with session_factory() as session:
            kwargs = {
                "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
                "city_slug": "atlanta",
                "condition_id": "cond-a",
                "date": "2026-05-16",
                "trade_count": 3,
                "volume_usd": 100.0,
                "realized_pnl": 10.0,
                "unrealized_pnl": 0.0,
                "consistency_score": 0.5,
            }
            first = await upsert_wallet_stat(session, **kwargs)
            second = await upsert_wallet_stat(session, **{**kwargs, "volume_usd": 125.0})
            rows = (await session.execute(select(WalletStat))).scalars().all()
            fetched = await get_wallet_stats_for_city(session, "atlanta", "2026-05-16")
            assert first.id == second.id
            assert len(rows) == 1
            assert fetched[0].volume_usd == 125.0

    _run(run_test())
    _run(engine.dispose())


def test_scheduler_update_handles_empty_market_data(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    monkeypatch.setattr(Config, "WALLET_TRACKER_ENABLED", True, raising=False)
    monkeypatch.setattr(Config, "WALLET_TRACKER_START_CITY", "atlanta", raising=False)

    async def run_test():
        async with session_factory() as session:
            session.add(
                City(
                    city_slug="atlanta",
                    display_name="Atlanta",
                    metar_station="KATL",
                    enabled=True,
                    is_us=True,
                    unit="F",
                    tz="America/New_York",
                )
            )
            await session.commit()
        summary = await update_wallet_rankings()
        assert summary.enabled is True
        assert summary.cities_scanned == 1
        assert summary.condition_ids_scanned == 0
        assert summary.wallets_updated == 0

    _run(run_test())
    _run(engine.dispose())


def test_dashboard_payload_helper_returns_safe_rows():
    row = SimpleNamespace(
        wallet_address="0x1234567890abcdef1234567890abcdef12345678",
        city_slug="atlanta",
        condition_id="cond-a",
        date="2026-05-16",
        trade_count=4,
        volume_usd=250.0,
        realized_pnl=12.5,
        unrealized_pnl=3.0,
        win_rate=0.75,
        avg_hold_minutes=42.0,
        avg_entry_price=0.3,
        avg_exit_price=0.5,
        profitable_days_pct=1.0,
        sharpe_like=1.2,
        consistency_score=0.82,
        regime="CALM",
        inferred_style="Scalper",
        bucket_idx=2,
        bucket_label="88-90F",
        net_position_qty=10.0,
        net_flow_usd=25.0,
        last_trade_ts=datetime(2026, 5, 16, 12, tzinfo=timezone.utc),
        market_slug="highest-temperature-in-atlanta",
    )
    payload = build_wallet_leaderboard_payload([row], enabled=True, limit=5)
    assert payload["status"] == "ok"
    assert payload["rows"][0]["display_address"] == "0x1234...5678"
    assert payload["rows"][0]["profile_url"].endswith(row.wallet_address.lower())
    assert "does not trigger automated trades" in payload["disclaimer"]


def test_smart_money_divergence_available_and_unavailable():
    buckets = [
        {"bucket_idx": 1, "label": "86-88F", "model_prob": 0.2},
        {"bucket_idx": 2, "label": "88-90F", "model_prob": 0.55},
        {"bucket_idx": 3, "label": "90-92F", "model_prob": 0.25},
    ]
    rows = [
        {"bucket_idx": 1, "bucket_label": "86-88F", "net_flow_usd": 120.0},
        {"bucket_idx": 3, "bucket_label": "90-92F", "net_flow_usd": 90.0},
    ]
    result = compute_smart_money_divergence(buckets, rows)
    assert result["status"] == "available"
    assert result["model_bucket_idx"] == 2
    assert result["smart_money_bucket_idx"] == 1
    assert result["divergence"] == "MEDIUM"

    unavailable = compute_smart_money_divergence([], rows)
    assert unavailable["status"] == "unavailable"
