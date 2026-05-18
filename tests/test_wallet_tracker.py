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
    WalletExposureMetric,
    build_bucket_consensus,
    build_wallet_leaderboard_payload,
    calculate_consistency_score,
    classify_model_confluence,
    compute_smart_money_divergence,
    compute_wallet_exposures,
    compute_wallet_metrics,
    compute_wallet_skill_scores,
    dedupe_public_trades,
    get_weather_smart_money_payload,
    infer_strategy_style,
    truncate_wallet_address,
    update_wallet_rankings,
    wilson_lower_bound,
)
from backend.storage.models import (
    Base,
    City,
    WalletMarketExposure,
    WalletSkillScore,
    WalletStat,
    WalletTrade,
)
from backend.storage.repos import (
    get_wallet_stats_for_city,
    upsert_wallet_market_exposure,
    upsert_wallet_skill_score,
    upsert_wallet_stat,
    upsert_wallet_trade,
)


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


def test_trade_deduping_uses_transaction_wallet_condition_side_asset():
    wallet = "0x1234567890abcdef1234567890abcdef12345678"
    now = datetime(2026, 5, 16, 12, tzinfo=timezone.utc)
    first = PublicTrade(
        wallet_address=wallet,
        condition_id="cond-a",
        side="BUY",
        size=10,
        price=0.4,
        timestamp=now,
        asset_id="asset-a",
        transaction_hash="0xtx",
    )
    duplicate = PublicTrade(
        wallet_address=wallet.upper(),
        condition_id="cond-a",
        side="BUY",
        size=10,
        price=0.4,
        timestamp=now + timedelta(seconds=1),
        asset_id="asset-a",
        transaction_hash="0xtx",
    )
    distinct = PublicTrade(
        wallet_address=wallet,
        condition_id="cond-a",
        side="SELL",
        size=10,
        price=0.55,
        timestamp=now + timedelta(minutes=2),
        asset_id="asset-a",
        transaction_hash="0xtx",
    )

    deduped = dedupe_public_trades([distinct, duplicate, first])
    assert len(deduped) == 2
    assert [t.side for t in deduped] == ["BUY", "SELL"]


def test_wilson_lower_bound_penalizes_tiny_perfect_records():
    tiny_perfect = wilson_lower_bound(4, 4)
    broad_strong = wilson_lower_bound(74, 80)

    assert tiny_perfect < 0.6
    assert broad_strong > tiny_perfect


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


def test_current_event_exposure_extracts_bucket_position():
    wallet = "0x1234567890abcdef1234567890abcdef12345678"
    now = datetime(2026, 5, 16, 12, tzinfo=timezone.utc)
    markets = [
        MarketRef(
            city_slug="atlanta",
            date="2026-05-16",
            condition_id="cond-a",
            market_slug="atlanta-high",
            bucket_idx=2,
            bucket_label="88-89F",
            current_price=0.65,
        )
    ]
    trades = [
        _trade(wallet, "cond-a", "BUY", 100, 0.40, now),
        _trade(wallet, "cond-a", "SELL", 25, 0.50, now + timedelta(minutes=20)),
    ]

    exposures = compute_wallet_exposures(trades, markets)

    assert len(exposures) == 1
    assert exposures[0].bucket_label == "88-89F"
    assert exposures[0].net_position_qty == 75
    assert exposures[0].net_notional_usd == 27.5
    assert exposures[0].trade_count == 2
    assert exposures[0].volume_usd == 52.5
    assert exposures[0].realized_pnl > 0
    assert exposures[0].unrealized_pnl > 0


def test_global_and_city_skill_scores_use_credibility_adjusted_quality():
    wallet = "0x1234567890abcdef1234567890abcdef12345678"
    other = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    base = datetime(2026, 5, 16, 12, tzinfo=timezone.utc)
    markets = [
        MarketRef(city_slug="atlanta", date="2026-05-14", condition_id="atl-1", resolved_winning_bucket_idx=1),
        MarketRef(city_slug="atlanta", date="2026-05-15", condition_id="atl-2", resolved_winning_bucket_idx=1),
        MarketRef(city_slug="boston", date="2026-05-16", condition_id="bos-1", resolved_winning_bucket_idx=2),
    ]
    exposures = [
        WalletExposureMetric(wallet, "atlanta", "2026-05-14", "atl-1", None, 1, "88-89F", 0, 40, None, 0, 60, base),
        WalletExposureMetric(wallet, "atlanta", "2026-05-15", "atl-2", None, 1, "88-89F", 0, 50, None, 0, 50, base),
        WalletExposureMetric(wallet, "boston", "2026-05-16", "bos-1", None, 2, "90-91F", 0, 30, None, 0, -10, base),
        WalletExposureMetric(other, "atlanta", "2026-05-14", "atl-1", None, 1, "88-89F", 0, 10, None, 0, 20, base),
    ]

    global_scores = compute_wallet_skill_scores(
        exposures,
        markets,
        scope="global",
        min_resolved_markets=3,
        min_volume_usd=1,
        min_active_days=2,
    )
    city_scores = compute_wallet_skill_scores(
        exposures,
        markets,
        scope="city",
        city_slug="atlanta",
        min_resolved_markets=2,
        min_volume_usd=1,
        min_active_days=2,
    )

    assert len(global_scores) == 1
    assert global_scores[0].wallet_address == wallet
    assert global_scores[0].resolved_markets == 3
    assert global_scores[0].win_rate == round(2 / 3, 4)
    assert global_scores[0].wilson_win_rate < global_scores[0].win_rate
    assert len(city_scores) == 1
    assert city_scores[0].city_slug == "atlanta"
    assert city_scores[0].rank == 1


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


def test_wallet_v2_upserts_are_idempotent(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    ts = datetime(2026, 5, 16, 12, tzinfo=timezone.utc)

    async def run_test():
        async with session_factory() as session:
            trade_kwargs = {
                "dedupe_key": "tx|wallet|cond|buy",
                "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
                "city_slug": "atlanta",
                "date": "2026-05-16",
                "condition_id": "cond-a",
                "side": "BUY",
                "size": 10.0,
                "price": 0.4,
                "notional_usd": 4.0,
                "trade_ts": ts,
            }
            exposure_kwargs = {
                "wallet_address": trade_kwargs["wallet_address"],
                "city_slug": "atlanta",
                "date": "2026-05-16",
                "condition_id": "cond-a",
                "net_position_qty": 10.0,
                "net_notional_usd": 4.0,
                "trade_count": 1,
                "volume_usd": 4.0,
                "avg_entry_price": 0.4,
                "realized_pnl": 0.0,
                "unrealized_pnl": 1.0,
                "last_trade_ts": ts,
                "last_updated_ts": ts,
            }
            skill_kwargs = {
                "wallet_address": trade_kwargs["wallet_address"],
                "scope": "global",
                "city_slug": "",
                "window_days": 90,
                "adjusted_score": 0.5,
                "rank": 1,
                "win_rate": 1.0,
                "wilson_win_rate": 0.7,
                "resolved_markets": 5,
                "total_markets": 6,
                "total_volume_usd": 100.0,
                "realized_pnl": 25.0,
                "roi": 0.25,
                "profit_factor": 25.0,
                "avg_notional_usd": 20.0,
                "active_days": 5,
                "last_active_ts": ts,
                "last_updated_ts": ts,
            }

            first_trade = await upsert_wallet_trade(session, **trade_kwargs)
            second_trade = await upsert_wallet_trade(session, **{**trade_kwargs, "price": 0.45})
            first_exposure = await upsert_wallet_market_exposure(session, **exposure_kwargs)
            second_exposure = await upsert_wallet_market_exposure(
                session,
                **{**exposure_kwargs, "net_notional_usd": 4.5, "volume_usd": 4.5},
            )
            first_skill = await upsert_wallet_skill_score(session, **skill_kwargs)
            second_skill = await upsert_wallet_skill_score(
                session,
                **{**skill_kwargs, "adjusted_score": 0.6},
            )

            trades = (await session.execute(select(WalletTrade))).scalars().all()
            exposures = (await session.execute(select(WalletMarketExposure))).scalars().all()
            skills = (await session.execute(select(WalletSkillScore))).scalars().all()
            assert first_trade.id == second_trade.id
            assert first_exposure.id == second_exposure.id
            assert first_skill.id == second_skill.id
            assert len(trades) == len(exposures) == len(skills) == 1
            assert trades[0].price == 0.45
            assert exposures[0].net_notional_usd == 4.5
            assert exposures[0].volume_usd == 4.5
            assert skills[0].adjusted_score == 0.6

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


def test_weather_smart_money_payload_returns_current_global_city_rows(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))
    monkeypatch.setattr(Config, "WALLET_TRACKER_ENABLED", True, raising=False)
    monkeypatch.setattr(Config, "WALLET_TRACKER_MIN_ADJUSTED_SCORE", 0.2, raising=False)
    ts = datetime.now(timezone.utc) - timedelta(minutes=30)
    wallet = "0x1234567890abcdef1234567890abcdef12345678"

    async def run_test():
        async with session_factory() as session:
            await upsert_wallet_skill_score(
                session,
                wallet_address=wallet,
                scope="global",
                city_slug="",
                window_days=Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
                adjusted_score=0.86,
                rank=1,
                win_rate=1.0,
                wilson_win_rate=0.72,
                resolved_markets=12,
                total_markets=14,
                total_volume_usd=5000.0,
                realized_pnl=800.0,
                roi=0.16,
                profit_factor=8.0,
                avg_notional_usd=357.0,
                active_days=8,
                last_active_ts=ts,
                last_updated_ts=ts,
            )
            await upsert_wallet_skill_score(
                session,
                wallet_address=wallet,
                scope="city",
                city_slug="atlanta",
                window_days=Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
                adjusted_score=0.80,
                rank=2,
                win_rate=1.0,
                wilson_win_rate=0.61,
                resolved_markets=7,
                total_markets=8,
                total_volume_usd=2000.0,
                realized_pnl=300.0,
                roi=0.15,
                profit_factor=6.0,
                avg_notional_usd=250.0,
                active_days=5,
                last_active_ts=ts,
                last_updated_ts=ts,
            )
            await upsert_wallet_market_exposure(
                session,
                wallet_address=wallet,
                city_slug="atlanta",
                date="2026-05-16",
                market_slug="atlanta-high",
                condition_id="cond-a",
                bucket_idx=2,
                bucket_label="88-89F",
                net_position_qty=120.0,
                net_notional_usd=72.0,
                trade_count=3,
                volume_usd=180.0,
                avg_entry_price=0.6,
                realized_pnl=0.0,
                unrealized_pnl=12.0,
                last_trade_ts=ts,
                last_updated_ts=ts,
            )

        payload = await get_weather_smart_money_payload(
            "atlanta",
            "2026-05-16",
            buckets=[
                {"bucket_idx": 1, "label": "86-87F", "model_prob": 0.25},
                {"bucket_idx": 2, "label": "88-89F", "model_prob": 0.55},
            ],
            limit=50,
        )
        assert payload["status"] == "ok"
        assert payload["current_market"][0]["global_rank"] == 1
        assert payload["current_market"][0]["city_rank"] == 2
        assert payload["bucket_consensus"][1]["ranked_wallets_long"] == 1
        assert payload["confluence"]["badge"] == "CONFIRMS MODEL"
        assert payload["global_leaders"][0]["win_rate_label"] == "100% over 12"

    _run(run_test())
    _run(engine.dispose())


def test_weather_smart_money_payload_disabled_is_safe(monkeypatch):
    monkeypatch.setattr(Config, "WALLET_TRACKER_ENABLED", False, raising=False)

    payload = _run(get_weather_smart_money_payload("atlanta", "2026-05-16", buckets=[]))

    assert payload["enabled"] is False
    assert payload["current_market"] == []
    assert payload["global_leaders"] == []
    assert payload["city_leaders"] == []


def test_bucket_consensus_and_model_confluence_classification():
    buckets = [
        {"bucket_idx": 1, "label": "86-87F", "model_prob": 0.2},
        {"bucket_idx": 2, "label": "88-89F", "model_prob": 0.6},
        {"bucket_idx": 3, "label": "90-91F", "model_prob": 0.2},
    ]
    rows = [
        {
            "bucket_idx": 2,
            "net_position_qty": 100.0,
            "net_notional_usd": 60.0,
            "avg_entry_price": 0.6,
            "global_score": 0.8,
        },
        {
            "bucket_idx": 3,
            "net_position_qty": 50.0,
            "net_notional_usd": 35.0,
            "avg_entry_price": 0.7,
            "global_score": 0.9,
        },
    ]

    consensus = build_bucket_consensus(buckets, rows)
    confluence = classify_model_confluence(buckets, consensus)

    assert consensus[1]["ranked_wallets_long"] == 1
    assert consensus[1]["avg_entry_price"] == 0.6
    assert confluence["badge"] == "CONFIRMS MODEL"


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
