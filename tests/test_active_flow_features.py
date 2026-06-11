import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.market_context.flow_features as flow_features
from backend.config import Config
from backend.market_context.wallet_tracker import PublicTrade
from backend.storage.models import Base, Bucket, City, Event, MarketFlowFeature, Position, WalletTrade


def _run(coro):
    return asyncio.run(coro)


def test_refresh_active_market_flow_features_writes_shadow_rows(tmp_path, monkeypatch):
    async def scenario():
        db_path = tmp_path / "flow.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        session_factory = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            city = City(city_slug="atlanta", display_name="Atlanta", enabled=True)
            session.add(city)
            await session.flush()
            event = Event(city_id=city.id, date_et="2026-06-11", gamma_slug="atlanta-weather")
            session.add(event)
            await session.flush()
            bucket = Bucket(
                event_id=event.id,
                bucket_idx=3,
                label="83-84",
                condition_id="condition-1",
            )
            session.add(bucket)
            await session.flush()
            session.add(Position(bucket_id=bucket.id, side="yes", net_qty=2.0, avg_cost=0.25))
            await session.commit()
            bucket_id = bucket.id

        @asynccontextmanager
        async def fake_get_session():
            async with session_factory() as session:
                yield session

        class FakeAdapter:
            async def fetch_trades_for_markets(self, condition_ids, *, limit=None):
                assert list(condition_ids) == ["condition-1"]
                assert limit == 1000
                return [
                    PublicTrade(
                        wallet_address="0xabc",
                        condition_id="condition-1",
                        side="BUY",
                        size=10.0,
                        price=0.4,
                        timestamp=datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
                    )
                ]

        monkeypatch.setattr(flow_features, "get_session", fake_get_session)
        monkeypatch.setattr(Config, "MARKET_FLOW_REFRESH_ENABLED", True)
        monkeypatch.setattr(Config, "MARKET_FLOW_FETCH_LIMIT", 1000)
        monkeypatch.setattr(Config, "STORE_RAW_WALLET_PAYLOADS", False)

        summary = await flow_features.refresh_active_market_flow_features(
            adapter=FakeAdapter(),
            bucket_ids=[bucket_id],
            as_of=datetime(2026, 6, 11, 14, 1, tzinfo=timezone.utc),
        )

        async with session_factory() as session:
            flow_rows = (
                await session.execute(
                    select(MarketFlowFeature).order_by(MarketFlowFeature.window_minutes)
                )
            ).scalars().all()
            trade_rows = (await session.execute(select(WalletTrade))).scalars().all()

        await engine.dispose()
        return summary, flow_rows, trade_rows

    summary, flow_rows, trade_rows = _run(scenario())
    assert summary.enabled is True
    assert summary.targets == 1
    assert summary.conditions == 1
    assert summary.trades_fetched == 1
    assert summary.trades_written == 1
    assert summary.feature_rows_written == 3
    assert [row.window_minutes for row in flow_rows] == [5, 15, 60]
    assert flow_rows[1].signed_net_notional == 4.0
    assert flow_rows[1].direction_source == "data_api_side"
    assert len(trade_rows) == 1
    assert trade_rows[0].raw_json is None
