import asyncio
from datetime import datetime, timezone

import backend.worker.scheduler as scheduler


def _run(coro):
    return asyncio.run(coro)


def test_recent_signal_cache_reuses_generation(monkeypatch):
    calls = {"n": 0}

    async def fake_job_run_model():
        calls["n"] += 1
        return ["fresh"]

    monkeypatch.setattr(scheduler, "_latest_signals", ["cached"], raising=False)
    monkeypatch.setattr(scheduler, "_latest_signals_at", datetime.now(timezone.utc), raising=False)
    monkeypatch.setattr(scheduler, "job_run_model", fake_job_run_model)

    result = _run(scheduler._get_recent_or_run_signals(max_age_s=90.0))

    assert result == ["cached"]
    assert calls["n"] == 0


def test_stale_signal_cache_refreshes_generation(monkeypatch):
    calls = {"n": 0}

    async def fake_job_run_model():
        calls["n"] += 1
        return ["fresh"]

    monkeypatch.setattr(scheduler, "_latest_signals", ["cached"], raising=False)
    monkeypatch.setattr(scheduler, "_latest_signals_at", datetime(2020, 1, 1, tzinfo=timezone.utc), raising=False)
    monkeypatch.setattr(scheduler, "job_run_model", fake_job_run_model)

    result = _run(scheduler._get_recent_or_run_signals(max_age_s=90.0))

    assert result == ["fresh"]
    assert calls["n"] == 1


def test_scheduler_registers_storage_and_flow_maintenance_jobs():
    sched = scheduler.create_scheduler()
    job_ids = {job.id for job in sched.get_jobs()}
    assert "db_retention_maintenance" in job_ids
    assert "db_storage_alerts" in job_ids
    assert "refresh_market_flow_features" in job_ids
