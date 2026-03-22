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
