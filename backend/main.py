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


async def run_api() -> None:
    """Start API server (FastAPI + Uvicorn)."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    from backend.storage.db import close_db, init_db
    from backend.api.routes import router as api_router
    from backend.ingestion.polymarket_clob import CLOBClient, set_clob

    app = FastAPI(title="WeatherQuant", version="1.0.0")

    @app.on_event("startup")
    async def startup():
        log.info("api: starting up")
        _log_config_warnings()
        await init_db()

        # Initialize CLOB client (read-only if no creds)
        clob = CLOBClient()
        await clob.start()
        set_clob(clob)
        log.info("api: CLOB client ready (can_trade=%s)", clob.can_trade)

    @app.on_event("shutdown")
    async def shutdown():
        from backend.ingestion.polymarket_clob import get_clob
        clob = get_clob()
        if clob:
            await clob.close()
        await close_db()
        log.info("api: shutdown complete")

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


def main() -> None:
    service = Config.SERVICE_TYPE
    log.info("WeatherQuant starting as service_type=%s", service)

    if service == "worker":
        asyncio.run(run_worker())
    else:
        asyncio.run(run_api())


if __name__ == "__main__":
    from backend.ingestion.polymarket_clob import get_clob
    main()
