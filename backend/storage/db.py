"""
Async database engine and session management.

Usage:
    async with get_session() as session:
        result = await session.execute(...)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import Config
from backend.storage.models import Base

log = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    url = Config.DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    is_sqlite = "sqlite" in url.lower()

    kwargs: dict = {
        "echo": False,
        "pool_pre_ping": True,
    }

    if not is_sqlite:
        # PostgreSQL — connection pooling
        kwargs.update(
            {
                "pool_size": 10,
                "max_overflow": 5,
                "pool_timeout": 30,
                "pool_recycle": 1800,
            }
        )
    else:
        # SQLite — single connection, no pooling
        kwargs["connect_args"] = {"check_same_thread": False}

    log.info("db: connecting url_prefix=%s", url[:40])
    return create_async_engine(url, **kwargs)


async def init_db() -> None:
    """Create tables and seed initial data. Called once on startup."""
    global _engine, _session_factory

    _engine = _build_engine()
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Patch cities table for Phase 13 manually (since Alembic isn't present)
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE cities ADD COLUMN is_us BOOLEAN NOT NULL DEFAULT true;"))
        except Exception as e:
            # Expected if column already exists
            pass
            
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE cities ADD COLUMN unit VARCHAR(1) NOT NULL DEFAULT 'F';"))
        except Exception:
            pass

        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE cities ADD COLUMN lat FLOAT;"))
        except Exception:
            pass

        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE cities ADD COLUMN lon FLOAT;"))
        except Exception:
            pass

        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE metar_obs ADD COLUMN report_at TIMESTAMP WITH TIME ZONE;"))
        except Exception:
            pass

        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE metar_obs ADD COLUMN raw_text TEXT;"))
        except Exception:
            pass
            
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE buckets ALTER COLUMN label TYPE VARCHAR(256);"))
        except Exception:
            pass

        # Event fields
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE events ADD COLUMN forecast_quality VARCHAR(16) NOT NULL DEFAULT 'ok';"))
        except Exception:
            pass
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE events ADD COLUMN wu_scrape_error TEXT;"))
        except Exception:
            pass

        # ForecastObs fields
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE forecast_obs ADD COLUMN raw_json TEXT;"))
        except Exception:
            pass
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE forecast_obs ADD COLUMN parse_error TEXT;"))
        except Exception:
            pass
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE forecast_obs ADD COLUMN raw_payload_hash VARCHAR(64);"))
        except Exception:
            pass

        # Order fields
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE orders ADD COLUMN signal_id INTEGER;"))
        except Exception:
            pass
        try:
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE orders ADD COLUMN gates_json TEXT;"))
        except Exception:
            pass

    await _seed_initial_data()
    log.info("db: init complete")


async def _seed_initial_data() -> None:
    """Insert initial cities and arming state if tables are empty."""
    from backend.storage.models import ArmingState, City

    async with get_session() as session:
        # Arming state — always exactly 1 row
        from sqlalchemy import select

        existing_arming = await session.execute(select(ArmingState))
        if existing_arming.scalar_one_or_none() is None:
            session.add(ArmingState(id=1, state="DISARMED"))
            await session.commit()
            log.info("db: seeded arming_state")

        # Cities
        from backend.storage.repos import upsert_city
        for c in Config.INITIAL_CITIES:
            await upsert_city(session, c)

        log.info("db: seeded %d cities", len(Config.INITIAL_CITIES))


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for a database session with auto-commit/rollback."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized — call init_db() first")

    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def close_db() -> None:
    """Dispose the engine on shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        log.info("db: engine disposed")
