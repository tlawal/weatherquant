import asyncio
from contextlib import asynccontextmanager

import backend.storage.db as db_module
import backend.storage.maintenance as maintenance_module
import backend.worker.scheduler as scheduler
from backend.config import Config


def _run(coro):
    return asyncio.run(coro)


def test_db_retention_job_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(Config, "DB_RETENTION_ENABLED", False)

    async def boom_get_session():
        raise AssertionError("get_session should not be called when retention is disabled")

    monkeypatch.setattr(db_module, "get_session", boom_get_session)
    _run(scheduler.job_db_retention_maintenance())


def test_db_retention_job_uses_configured_policy(monkeypatch):
    seen = {}

    monkeypatch.setattr(Config, "DB_RETENTION_ENABLED", True)
    monkeypatch.setattr(Config, "DB_RETENTION_MARKET_SNAPSHOT_DAYS", 5)
    monkeypatch.setattr(Config, "DB_RETENTION_MARKET_FLOW_DAYS", 7)
    monkeypatch.setattr(Config, "DB_RETENTION_RAW_PAYLOAD_DAYS", 14)
    monkeypatch.setattr(Config, "DB_RETENTION_SIGNAL_DAYS", 45)
    monkeypatch.setattr(Config, "DB_RETENTION_FORECAST_OBS_DAYS", 180)
    monkeypatch.setattr(Config, "DB_RETENTION_WALLET_TRADE_DAYS", 120)
    monkeypatch.setattr(Config, "DB_RETENTION_WALLET_EXPOSURE_DAYS", 45)
    monkeypatch.setattr(Config, "DB_RETENTION_MODEL_INPUT_DAYS", 14)
    monkeypatch.setattr(Config, "DB_RETENTION_PRUNE_SIGNALS", False)
    monkeypatch.setattr(Config, "DB_RETENTION_BATCH_SIZE", 5000)

    @asynccontextmanager
    async def fake_get_session():
        yield object()

    async def fake_run_retention_maintenance(session, *, dry_run, policy):
        seen["session"] = session
        seen["dry_run"] = dry_run
        seen["policy"] = policy
        return {
            "supported": True,
            "dialect": "postgresql",
            "actions": [
                {"name": "delete_old_market_snapshots_for_inactive_events", "candidate_rows": 9, "affected_rows": 3},
                {"name": "delete_old_wallet_trades_beyond_skill_window", "candidate_rows": 2, "affected_rows": 0},
            ],
        }

    monkeypatch.setattr(db_module, "get_session", fake_get_session)
    monkeypatch.setattr(
        maintenance_module,
        "run_retention_maintenance",
        fake_run_retention_maintenance,
    )

    _run(scheduler.job_db_retention_maintenance())

    assert seen["dry_run"] is False
    policy = seen["policy"]
    assert policy.market_snapshot_days == 5
    assert policy.market_flow_days == 7
    assert policy.raw_payload_days == 14
    assert policy.signal_days == 45
    assert policy.forecast_obs_days == 180
    assert policy.wallet_trade_days == 120
    assert policy.wallet_exposure_days == 45
    assert policy.model_input_days == 14
    assert policy.prune_signals is False
    assert policy.batch_size == 5000
