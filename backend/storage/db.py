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
        # SQLite — use NullPool to avoid connection sharing issues,
        # and set a generous busy timeout so concurrent writers wait
        # instead of immediately raising "database is locked".
        from sqlalchemy.pool import NullPool
        kwargs["poolclass"] = NullPool
        kwargs["connect_args"] = {
            "check_same_thread": False,
            "timeout": 30,  # seconds to wait for lock (default is 5)
        }

    log.info("db: connecting url_prefix=%s", url[:40])
    engine = create_async_engine(url, **kwargs)

    if is_sqlite:
        # Enable WAL mode — allows concurrent readers during writes.
        # Also set synchronous=NORMAL (safe with WAL, much faster).
        import sqlalchemy

        @sqlalchemy.event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return engine


async def _run_ddl(ddl: str) -> None:
    """Run a single DDL statement in its own transaction. Silently ignore failures
    (e.g., column already exists). Critical: never shares a transaction with other DDL."""
    try:
        from sqlalchemy import text
        async with _engine.begin() as conn:
            await conn.execute(text(ddl))
    except Exception as e:
        log.debug("ddl: skipped (already applied or not supported): %s — %s", ddl[:60], e)


async def init_db() -> None:
    """Create tables and seed initial data. Called once on startup."""
    global _engine, _session_factory

    _engine = _build_engine()
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )

    # Step 1: create all tables (own transaction)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db: tables created/verified")

    # Step 2: schema migrations — each in its own transaction so one failure
    # never aborts another (PostgreSQL aborts the whole tx on any error)
    # cities
    await _run_ddl("ALTER TABLE cities ADD COLUMN is_us BOOLEAN NOT NULL DEFAULT true")
    await _run_ddl("ALTER TABLE cities ADD COLUMN unit VARCHAR(1) NOT NULL DEFAULT 'F'")
    await _run_ddl("ALTER TABLE cities ADD COLUMN lat FLOAT")
    await _run_ddl("ALTER TABLE cities ADD COLUMN lon FLOAT")
    await _run_ddl("ALTER TABLE cities ADD COLUMN tz VARCHAR(64) NOT NULL DEFAULT 'America/New_York'")

    # metar_obs
    await _run_ddl("ALTER TABLE metar_obs ADD COLUMN report_at TIMESTAMP WITH TIME ZONE")
    await _run_ddl("ALTER TABLE metar_obs ADD COLUMN raw_text TEXT")
    await _run_ddl("ALTER TABLE metar_obs ADD COLUMN raw_json TEXT")
    await _run_ddl("ALTER TABLE metar_obs ADD COLUMN parse_error TEXT")
    await _run_ddl("ALTER TABLE metar_obs ADD COLUMN raw_payload_hash VARCHAR(64)")

    # buckets
    await _run_ddl("ALTER TABLE buckets ALTER COLUMN label TYPE VARCHAR(256)")

    # events
    await _run_ddl("ALTER TABLE events ADD COLUMN forecast_quality VARCHAR(16) NOT NULL DEFAULT 'ok'")
    await _run_ddl("ALTER TABLE events ADD COLUMN wu_scrape_error TEXT")
    await _run_ddl("ALTER TABLE events ADD COLUMN resolution_source_url TEXT")
    await _run_ddl("ALTER TABLE events ADD COLUMN resolution_station_id VARCHAR(16)")
    await _run_ddl("ALTER TABLE events ADD COLUMN resolved_at TIMESTAMP WITH TIME ZONE")
    await _run_ddl("ALTER TABLE events ADD COLUMN winning_bucket_idx INTEGER")
    await _run_ddl("ALTER TABLE events ADD COLUMN redeemed_at TIMESTAMP WITH TIME ZONE")

    # forecast_obs
    await _run_ddl("ALTER TABLE forecast_obs ADD COLUMN raw_json TEXT")
    await _run_ddl("ALTER TABLE forecast_obs ADD COLUMN parse_error TEXT")
    await _run_ddl("ALTER TABLE forecast_obs ADD COLUMN raw_payload_hash VARCHAR(64)")

    # orders
    await _run_ddl("ALTER TABLE orders ADD COLUMN signal_id INTEGER")
    await _run_ddl("ALTER TABLE orders ADD COLUMN gates_json TEXT")

    # arming_state
    await _run_ddl("ALTER TABLE arming_state ADD COLUMN auto_redeem_enabled BOOLEAN NOT NULL DEFAULT 0")

    # positions
    await _run_ddl("ALTER TABLE positions ADD COLUMN entry_type VARCHAR(16)")
    await _run_ddl("ALTER TABLE positions ADD COLUMN strategy VARCHAR(64)")
    await _run_ddl("ALTER TABLE positions ADD COLUMN governing_exit_conditions TEXT")
    await _run_ddl("ALTER TABLE positions ADD COLUMN current_exit_status VARCHAR(256)")
    await _run_ddl("ALTER TABLE positions ADD COLUMN entry_time TIMESTAMP WITH TIME ZONE")
    await _run_ddl("ALTER TABLE positions ADD COLUMN entry_price FLOAT")

    # signals — generation tag to filter to "latest snapshot only"
    await _run_ddl("ALTER TABLE signals ADD COLUMN model_snapshot_id INTEGER")
    await _run_ddl("CREATE INDEX IF NOT EXISTS ix_signal_snapshot ON signals (model_snapshot_id)")

    # calibration_params — HRRR/GFS support
    await _run_ddl("ALTER TABLE calibration_params ADD COLUMN bias_hrrr FLOAT DEFAULT 0.0")
    await _run_ddl("ALTER TABLE calibration_params ADD COLUMN bias_gfs FLOAT DEFAULT 0.0")
    await _run_ddl("ALTER TABLE calibration_params ADD COLUMN weight_hrrr FLOAT DEFAULT 0.5")
    await _run_ddl("ALTER TABLE calibration_params ADD COLUMN weight_gfs FLOAT DEFAULT 0.2")
    # Migrate existing rows from old 1/3 defaults to new differentiated weights
    await _run_ddl("UPDATE calibration_params SET weight_hrrr = 0.5 WHERE weight_hrrr = 0.333")
    await _run_ddl("UPDATE calibration_params SET weight_gfs = 0.2 WHERE weight_gfs = 0.333")

    # calibration_params — GFS → NBM rename
    await _run_ddl("ALTER TABLE calibration_params ADD COLUMN bias_nbm FLOAT DEFAULT 0.0")
    await _run_ddl("ALTER TABLE calibration_params ADD COLUMN weight_nbm FLOAT DEFAULT 0.2")
    await _run_ddl("UPDATE calibration_params SET bias_nbm = bias_gfs WHERE bias_nbm = 0.0 AND bias_gfs != 0.0")
    await _run_ddl("UPDATE calibration_params SET weight_nbm = weight_gfs WHERE weight_nbm = 0.2 AND weight_gfs != 0.2")

    # metar_obs — source column for TGFTP vs aviation distinction
    await _run_ddl("ALTER TABLE metar_obs ADD COLUMN source VARCHAR(16) NOT NULL DEFAULT 'aviation'")
    await _run_ddl("CREATE INDEX IF NOT EXISTS ix_metar_source_station_ts ON metar_obs (source, metar_station, observed_at)")

    # madis_obs — MADIS HFMETAR benchmarking table
    await _run_ddl("""
        CREATE TABLE IF NOT EXISTS madis_obs (
            id SERIAL PRIMARY KEY,
            city_id INTEGER NOT NULL REFERENCES cities(id),
            metar_station VARCHAR(8) NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            temp_c FLOAT,
            temp_f FLOAT,
            source_file VARCHAR(64)
        )
    """)
    await _run_ddl("CREATE INDEX IF NOT EXISTS ix_madis_station_ts ON madis_obs (metar_station, observed_at)")

    # metar_obs_extended — add columns expected by routes.py
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS humidity_pct FLOAT")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS wind_dir_deg INTEGER")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS wind_speed_kt FLOAT")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS wind_gust_kt FLOAT")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS altimeter_inhg FLOAT")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS precip_in FLOAT")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS cloud_cover VARCHAR(4)")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS wx_string VARCHAR(64)")
    await _run_ddl("ALTER TABLE metar_obs_extended ADD COLUMN IF NOT EXISTS condition VARCHAR(32)")

    # station_calibrations — per-source MAE breakdown
    await _run_ddl("ALTER TABLE station_calibrations ADD COLUMN source_mae_json TEXT")
    await _run_ddl("ALTER TABLE station_calibrations ADD COLUMN rmse_f FLOAT DEFAULT 0.0")
    # station_calibrations — per-model MAE comparison (ECMWF vs GFS+HRRR vs NWS)
    await _run_ddl("ALTER TABLE station_calibrations ADD COLUMN mae_ecmwf_f FLOAT")
    await _run_ddl("ALTER TABLE station_calibrations ADD COLUMN mae_gfs_hrrr_f FLOAT")
    await _run_ddl("ALTER TABLE station_calibrations ADD COLUMN mae_nws_f FLOAT")
    await _run_ddl("ALTER TABLE station_calibrations ADD COLUMN winner VARCHAR(10)")

    # Step 3: seed initial data
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
