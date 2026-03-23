"""
APScheduler-based worker — runs all ingestion and trading loops.

Designed to run as a separate process from the API server.
All jobs write to Postgres; API server reads from Postgres.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from backend.config import Config

log = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _run_with_heartbeat(job_name: str, coro_fn) -> None:
    """Wrap a job coroutine with error catching and heartbeat update."""
    from backend.storage.db import get_session
    from backend.storage.repos import update_heartbeat

    try:
        await coro_fn()
        async with get_session() as sess:
            await update_heartbeat(sess, job_name, success=True)
    except Exception as e:
        log.error("scheduler: job=%s FAILED: %s", job_name, e, exc_info=True)
        try:
            async with get_session() as sess:
                await update_heartbeat(sess, job_name, success=False, error=str(e))
        except Exception:
            pass


async def job_fetch_metar():
    from backend.ingestion.metar import fetch_metar_all
    await fetch_metar_all()


async def job_fetch_nws():
    from backend.ingestion.forecasts import fetch_nws_all
    await fetch_nws_all()


async def job_fetch_wu():
    from backend.ingestion.forecasts import fetch_wu_all
    await fetch_wu_all()


async def job_fetch_open_meteo():
    from backend.ingestion.forecasts import fetch_open_meteo_all
    await fetch_open_meteo_all()


async def job_fetch_gamma():
    from backend.ingestion.polymarket_gamma import fetch_gamma_all
    await fetch_gamma_all()


async def job_fetch_clob():
    from backend.ingestion.polymarket_clob import fetch_clob_orderbooks, get_clob
    clob = get_clob()
    if not clob:
        return
    await fetch_clob_orderbooks(clob)


async def job_run_model():
    """Run signal engine and store results."""
    from backend.engine.signal_engine import run_signal_engine
    signals = await run_signal_engine()
    log.info(
        "scheduler: signal_engine produced %d signals (%d actionable)",
        len(signals),
        sum(1 for s in signals if s.actionable),
    )
    return signals


async def job_run_auto_trader():
    """Execute top signals if armed."""
    from backend.engine.signal_engine import run_signal_engine
    from backend.execution.trader import execute_top_signals
    from backend.ingestion.polymarket_clob import get_clob
    from backend.execution.arming import is_armed

    if not await is_armed():
        return

    clob = get_clob()
    bankroll = Config.BANKROLL_CAP
    if clob and clob.can_trade:
        balance = await clob.get_balance()
        if balance:
            bankroll = min(balance, Config.BANKROLL_CAP)

    signals = await run_signal_engine()
    results = await execute_top_signals(
        signals, bankroll=bankroll, max_trades=Config.MAX_POSITIONS_PER_EVENT
    )
    if results:
        log.info("scheduler: auto_trader executed %d trades", len(results))


async def job_reconcile_orders():
    """Check fill status of open orders."""
    from backend.storage.db import get_session
    from backend.storage.repos import get_open_orders, update_order_status
    from backend.ingestion.polymarket_clob import get_clob
    from datetime import timedelta

    async with get_session() as sess:
        open_orders = await get_open_orders(sess)

    clob = get_clob()
    for order in open_orders:
        # Auto-cancel orders older than 5 minutes
        age = (datetime.now(timezone.utc) - order.created_at).total_seconds()
        if age > 300:
            if clob and order.clob_order_id:
                await clob.cancel_order(order.clob_order_id)
            async with get_session() as sess:
                await update_order_status(
                    sess, order.id, "cancelled", cancel_reason="age_timeout_5min"
                )
            log.warning("reconcile: auto-cancelled stale order id=%d age=%.0fs", order.id, age)


async def job_auto_check_disarm():
    from backend.execution.arming import auto_check_disarm
    await auto_check_disarm()


async def job_discover_cities():
    from backend.ingestion.polymarket_gamma import discover_cities
    await discover_cities()


async def job_refresh_station_profiles():
    from backend.ingestion.station_pattern import refresh_all_station_profiles
    await refresh_all_station_profiles()


async def job_heartbeat():
    """Write a heartbeat so API server can detect worker liveness."""
    from backend.storage.db import get_session
    from backend.storage.repos import update_heartbeat
    async with get_session() as sess:
        await update_heartbeat(sess, "scheduler_alive", success=True)


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="America/New_York")

    def add(job_fn, seconds: int, name: str = None):
        scheduler.add_job(
            _run_with_heartbeat,
            trigger=IntervalTrigger(seconds=seconds),
            args=[name or job_fn.__name__, job_fn],
            id=name or job_fn.__name__,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )

    add(job_heartbeat,           seconds=30,   name="scheduler_alive")
    add(job_fetch_metar,         seconds=60,   name="fetch_metar")
    add(job_fetch_nws,           seconds=900,  name="fetch_nws")   # 15 min
    add(job_fetch_open_meteo,    seconds=900,  name="fetch_open_meteo")
    add(job_fetch_wu,            seconds=300,  name="fetch_wu")    # 5 min
    add(job_fetch_gamma,         seconds=120,  name="fetch_gamma") # 2 min
    add(job_fetch_clob,          seconds=30,   name="fetch_clob")
    add(job_run_model,           seconds=60,   name="run_model")
    add(job_run_auto_trader,     seconds=60,   name="run_auto_trader")
    add(job_reconcile_orders,    seconds=30,   name="reconcile_orders")
    add(job_auto_check_disarm,   seconds=60,   name="auto_check_disarm")
    add(job_discover_cities,     seconds=86400, name="discover_cities")  # 24h
    add(job_refresh_station_profiles, seconds=86400, name="refresh_station_profiles")  # 24h

    return scheduler


async def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    _scheduler = create_scheduler()
    _scheduler.start()
    log.info("scheduler: started with %d jobs", len(_scheduler.get_jobs()))

    # Run initial fetches immediately on startup
    await asyncio.sleep(2)  # brief delay for DB to settle
    for coro_fn, name in [
        (job_fetch_gamma, "fetch_gamma"),
        (job_fetch_metar, "fetch_metar"),
        (job_fetch_nws, "fetch_nws"),
        (job_fetch_open_meteo, "fetch_open_meteo"),
        (job_refresh_station_profiles, "refresh_station_profiles"),
    ]:
        try:
            await _run_with_heartbeat(name, coro_fn)
        except Exception as e:
            log.warning("scheduler: startup job %s failed: %s", name, e)

    return _scheduler


async def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        log.info("scheduler: stopped")
