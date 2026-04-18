"""
WeatherQuant — main entry point.

Selects api or worker mode based on SERVICE_TYPE env var.
Both modes initialize the DB; only api serves HTTP.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from backend.config import Config

# ─── Logging setup ────────────────────────────────────────────────────────────
_log_format = (
    '{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
    if Config.LOG_JSON
    else "%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format=_log_format,
    stream=sys.stdout,
)
log = logging.getLogger("main")


def _log_config_warnings() -> None:
    warnings = Config.validate()
    for w in warnings:
        log.warning("config: %s", w)


async def _load_runtime_config() -> None:
    """Load persisted runtime config overrides from DB into Config class vars."""
    try:
        from backend.storage.db import get_session
        from backend.storage.repos import get_runtime_config
        async with get_session() as sess:
            params = await get_runtime_config(sess)
        if not params:
            return
        _FLOAT_FIELDS = {
            "min_true_edge": "MIN_TRUE_EDGE",
            "max_daily_loss": "MAX_DAILY_LOSS",
            "bankroll_cap": "BANKROLL_CAP",
            "kelly_fraction": "KELLY_FRACTION",
            "max_entry_price": "MAX_ENTRY_PRICE",
            "max_spread": "MAX_SPREAD",
            "quick_flip_target": "QUICK_FLIP_TARGET",
            "urgent_exit_max_spread": "URGENT_EXIT_MAX_SPREAD",
            "expiry_discount": "EXPIRY_DISCOUNT",
        }
        _INT_FIELDS = {
            "consensus_debounce_runs": "CONSENSUS_DEBOUNCE_RUNS",
        }
        for key, attr in _FLOAT_FIELDS.items():
            if key in params:
                setattr(Config, attr, float(params[key]))
        for key, attr in _INT_FIELDS.items():
            if key in params:
                setattr(Config, attr, int(params[key]))
        log.info("startup: loaded runtime config overrides: %s", list(params.keys()))
    except Exception as e:
        log.warning("startup: could not load runtime config from DB (safe to ignore on first boot): %s", e)


async def _run_startup_backfills() -> None:
    """Run idempotent startup backfills before normal background work begins."""
    from backend.ingestion.metar import backfill_recent_nws_extended

    try:
        repaired = await backfill_recent_nws_extended()
        log.info("startup: recent NWS extended backfill repaired=%d", repaired)
    except Exception as e:
        log.exception("startup: NWS extended backfill failed: %s", e)


async def run_api(start_worker: bool = False) -> None:
    """Start API server (FastAPI + Uvicorn)."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    from backend.storage.db import close_db, init_db
    from backend.api.routes import router as api_router
    from backend.ingestion.polymarket_clob import CLOBClient, set_clob

    app = FastAPI(title="WeatherQuant", version="1.0.0")

    # Flag so /health can report "initializing" until DB is ready
    _ready = {"db": False}

    async def _background_startup():
        try:
            log.info("api: background init starting (db + CLOB)")
            await init_db()
            await _load_runtime_config()
            await _run_startup_backfills()
            _ready["db"] = True
            log.info("api: db init complete")

            clob = CLOBClient()
            await clob.start()
            set_clob(clob)
            log.info("api: CLOB client ready (can_trade=%s)", clob.can_trade)

            if start_worker:
                from backend.worker.scheduler import start_scheduler
                await start_scheduler()
                log.info("api: scheduler started in background 'all' mode")

        except Exception as e:
            log.exception("api: background startup failed: %s", e)

    @app.on_event("startup")
    async def startup():
        log.info("api: server binding — launching background init")
        _log_config_warnings()
        # Fire-and-forget: don't block the server from binding
        asyncio.create_task(_background_startup())

    @app.on_event("shutdown")
    async def shutdown():
        if start_worker:
            try:
                from backend.worker.scheduler import stop_scheduler
                await stop_scheduler()
                log.info("api: scheduler stopped")
            except Exception as e:
                log.exception("api: error stopping scheduler: %s", e)

        from backend.ingestion.polymarket_clob import get_clob
        clob = get_clob()
        if clob:
            await clob.close()
        await close_db()
        log.info("api: shutdown complete")

    # Patch the /health route to reflect init state
    # We do this by storing the ready dict where the route can see it
    app.state.ready = _ready

    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "..", "web", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Register API routes
    app.include_router(api_router)

    # Register dashboard routes
    try:
        from web.routes import dashboard_router
        app.include_router(dashboard_router)
    except ImportError:
        log.warning("api: web.routes not found — dashboard disabled")

    port = int(os.environ.get("PORT", 8000))
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level=Config.LOG_LEVEL.lower()
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_worker() -> None:
    """Start background worker (scheduler + auto-trader)."""
    from backend.storage.db import close_db, init_db
    from backend.ingestion.polymarket_clob import CLOBClient, set_clob
    from backend.worker.scheduler import start_scheduler, stop_scheduler

    log.info("worker: starting up")
    _log_config_warnings()
    await init_db()
    await _load_runtime_config()
    await _run_startup_backfills()

    clob = CLOBClient()
    await clob.start()
    set_clob(clob)
    log.info("worker: CLOB client ready (can_trade=%s)", clob.can_trade)

    scheduler = await start_scheduler()

    # Run forever until interrupted
    try:
        while True:
            await asyncio.sleep(30)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("worker: shutting down")
    finally:
        await stop_scheduler()
        clob_inst = get_clob()
        if clob_inst:
            await clob_inst.close()
        await close_db()
        log.info("worker: shutdown complete")


async def run_all() -> None:
    """Start both API and Worker in the same process."""
    log.info("starting in 'all' mode (api + worker)")
    await run_api(start_worker=True)


def main() -> None:
    service = Config.SERVICE_TYPE
    log.info("WeatherQuant starting as service_type=%s", service)

    if service == "worker":
        asyncio.run(run_worker())
    elif service == "all":
        asyncio.run(run_all())
    else:
        asyncio.run(run_api())


if __name__ == "__main__":
    from backend.ingestion.polymarket_clob import get_clob
    main()
