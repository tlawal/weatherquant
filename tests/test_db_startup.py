import asyncio

import backend.storage.db as db_module
from backend.config import Config


def _run(coro):
    return asyncio.run(coro)


def test_init_db_can_skip_legacy_migrations(monkeypatch, tmp_path):
    db_path = tmp_path / "startup.db"
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr(Config, "DB_STARTUP_LEGACY_MIGRATIONS_ENABLED", False)

    async def fail_ddl(_ddl: str):
        raise AssertionError("legacy DDL should be skipped")

    async def noop_seed():
        return None

    monkeypatch.setattr(db_module, "_run_ddl", fail_ddl)
    monkeypatch.setattr(db_module, "_seed_initial_data", noop_seed)

    async def scenario():
        try:
            await db_module.init_db()
        finally:
            await db_module.close_db()
            db_module._engine = None
            db_module._session_factory = None

    _run(scenario())
