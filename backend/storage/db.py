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
        existing_cities = await session.execute(select(City))
        if existing_cities.scalars().all():
            return  # already seeded

        for c in Config.INITIAL_CITIES:
            session.add(
                City(
                    city_slug=c["city_slug"],
                    display_name=c["display_name"],
                    metar_station=c.get("metar_station"),
                    nws_office=c.get("nws_office"),
                    nws_grid_x=c.get("nws_grid_x"),
                    nws_grid_y=c.get("nws_grid_y"),
                    wu_state=c.get("wu_state"),
                    wu_city=c.get("wu_city"),
                    enabled=c.get("enabled", False),
                )
            )
        await session.commit()
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
