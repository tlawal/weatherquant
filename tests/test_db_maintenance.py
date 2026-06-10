import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.storage.maintenance import (
    PROTECTED_TABLES,
    RetentionPolicy,
    build_db_size_report,
    run_retention_maintenance,
)
from backend.storage.models import Base


def _run(coro):
    return asyncio.run(coro)


def test_retention_policy_protects_trade_and_model_history():
    protected = set(PROTECTED_TABLES)
    assert {
        "events",
        "buckets",
        "orders",
        "fills",
        "positions",
        "closed_trades",
        "forecast_daily_errors",
        "source_lead_time_skills",
        "station_calibrations",
        "model_artifacts",
        "wallet_stats",
        "wallet_skill_scores",
    }.issubset(protected)
    assert "market_snapshots" not in protected
    assert "market_flow_features" not in protected
    assert "signals" not in protected


def test_retention_policy_clamps_unsafe_inputs():
    policy = RetentionPolicy(
        market_snapshot_days=-10,
        market_flow_days=0,
        raw_payload_days=1,
        signal_days=2,
        prune_signals=True,
        batch_size=999999,
    ).normalized()
    assert policy.market_snapshot_days == 3
    assert policy.market_flow_days == 1
    assert policy.raw_payload_days == 7
    assert policy.signal_days == 14
    assert policy.prune_signals is True
    assert policy.batch_size == 20000


def test_sqlite_maintenance_is_noop(tmp_path):
    async def scenario():
        db_path = tmp_path / "maintenance.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        session_factory = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            size_report = await build_db_size_report(session)
            prune_report = await run_retention_maintenance(session, dry_run=True)
        await engine.dispose()
        return size_report, prune_report

    size_report, prune_report = _run(scenario())
    assert size_report["supported"] is False
    assert size_report["dialect"] == "sqlite"
    assert prune_report["supported"] is False
    assert prune_report["actions"] == []
    assert "closed_trades" in prune_report["protected_tables"]
