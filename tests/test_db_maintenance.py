import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.storage.maintenance import (
    PROTECTED_TABLES,
    RetentionPolicy,
    build_db_size_report,
    build_cold_export_metadata,
    evaluate_db_storage_alerts,
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
        forecast_obs_days=3,
        wallet_trade_days=3,
        wallet_exposure_days=3,
        model_input_days=3,
        prune_signals=True,
        batch_size=999999,
    ).normalized()
    assert policy.market_snapshot_days == 3
    assert policy.market_flow_days == 1
    assert policy.raw_payload_days == 7
    assert policy.signal_days == 14
    assert policy.forecast_obs_days == 90
    assert policy.wallet_trade_days == 30
    assert policy.wallet_exposure_days == 14
    assert policy.model_input_days == 7
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


def test_db_storage_alerts_detect_volume_and_table_growth():
    mb = 1024 * 1024
    report = {
        "supported": True,
        "as_of": "2026-06-11T00:00:00+00:00",
        "totals": {
            "database_bytes": 3600 * mb,
            "wal_bytes": 250 * mb,
            "hot_store_bytes": 3850 * mb,
        },
        "tables": [
            {"table_name": "wallet_trades", "total_bytes": 900 * mb},
            {"table_name": "forecast_obs", "total_bytes": 300 * mb},
        ],
    }
    previous = {
        "table_bytes": {
            "wallet_trades": 760 * mb,
            "forecast_obs": 295 * mb,
        }
    }

    result = evaluate_db_storage_alerts(
        report,
        volume_limit_mb=5000,
        volume_alert_pct=0.70,
        top_table_alert_mb=750,
        table_growth_alert_mb=100,
        previous_snapshot=previous,
    )

    alert_types = {alert["type"] for alert in result["alerts"]}
    assert result["snapshot"]["usage_pct"] == 0.77
    assert "volume_usage" in alert_types
    assert "top_table_size" in alert_types
    assert "table_growth" in alert_types


def test_cold_export_metadata_allows_only_firehose_tables():
    meta = build_cold_export_metadata(
        "market_flow_features",
        days=99999,
        limit_rows=0,
        batch_size=999999,
    )
    assert meta["table"] == "market_flow_features"
    assert meta["cutoff_column"] == "computed_at"
    assert meta["cutoff_kind"] == "timestamp"
    assert meta["days"] == 3650
    assert meta["limit_rows"] == 1
    assert meta["batch_size"] == 20000

    try:
        build_cold_export_metadata("closed_trades", days=30)
    except ValueError as exc:
        assert "Unsupported cold export table" in str(exc)
        assert "wallet_trades" in str(exc)
    else:
        raise AssertionError("closed_trades must not be exposed by cold export")
