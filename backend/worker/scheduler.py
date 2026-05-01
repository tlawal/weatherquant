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


async def job_fetch_open_meteo_models():
    from backend.ingestion.forecasts import fetch_open_meteo_models_all
    await fetch_open_meteo_models_all()


async def job_fetch_om_hrrr_rapid():
    """Phase B3 — adaptive HRRR cadence.

    HRRR runs hourly with availability typically ~45 min past each top-of-hour.
    Polling every 15 min on a fixed cadence wastes up to 14 min after each new
    run lands. This job fires every 5 min but only does work inside the
    expected availability window (40–65 min past the hour). Outside the window
    it's a no-op so the baseline 900s job still covers the rest of the hour.
    """
    from datetime import datetime, timezone
    from backend.ingestion.forecasts import fetch_open_meteo_models_all

    minutes_past = datetime.now(timezone.utc).minute
    if not (40 <= minutes_past <= 65):
        return
    await fetch_open_meteo_models_all(source_filter={"hrrr", "hrrr_15min"})


async def job_fetch_aiwp_pangu():
    """Phase Q3 §13 — Pangu-Weather from NOAA AIWP S3 archive."""
    from backend.ingestion.aiwp import fetch_pangu
    await fetch_pangu()


async def job_fetch_aiwp_fourcastnet_v2():
    """Phase Q3 §13 — FourCastNet v2-small from NOAA AIWP S3 archive."""
    from backend.ingestion.aiwp import fetch_fourcastnet_v2
    await fetch_fourcastnet_v2()


async def job_fetch_herbie_hrrr():
    from backend.ingestion.herbie_side_channel import fetch_herbie_hrrr
    await fetch_herbie_hrrr()


async def job_fetch_herbie_nbm():
    from backend.ingestion.herbie_side_channel import fetch_herbie_nbm
    await fetch_herbie_nbm()


async def job_fetch_herbie_ifs():
    from backend.ingestion.herbie_side_channel import fetch_herbie_ifs
    await fetch_herbie_ifs()


async def job_fetch_herbie_aifs():
    from backend.ingestion.herbie_side_channel import fetch_herbie_aifs
    await fetch_herbie_aifs()


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


async def job_run_night_owl():
    """Runs overnight to look for fresh 00z/06z model edges."""
    from backend.strategy.night_owl import run_night_owl
    from backend.execution.arming import is_armed
    if not await is_armed():
        return
    try:
        await run_night_owl()
    except Exception as e:
        log.exception("job_run_night_owl error: %s", e)


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


async def job_run_exit_engine():
    """Evaluate open positions for Quick Flip or Emergency Exits."""
    from backend.execution.exit_engine import run_exit_engine
    await run_exit_engine()


async def job_discover_cities():
    from backend.ingestion.polymarket_gamma import discover_cities
    await discover_cities()


async def job_refresh_station_profiles():
    from backend.ingestion.station_pattern import refresh_all_station_profiles
    await refresh_all_station_profiles()


async def job_refresh_missing_station_profiles():
    from backend.ingestion.station_pattern import refresh_missing_station_profiles
    await refresh_missing_station_profiles()


async def job_fetch_metar_smart():
    from backend.ingestion.metar import fetch_metar_smart
    await fetch_metar_smart()


async def job_fetch_tgftp_metar():
    from backend.ingestion.tgftp_metar import fetch_tgftp_all
    await fetch_tgftp_all()


async def job_fetch_madis():
    from backend.ingestion.madis_hfmetar import fetch_madis_latest
    await fetch_madis_latest()


async def job_check_resolved():
    from backend.execution.redeemer import check_resolved_markets
    await check_resolved_markets()


async def job_auto_redeem():
    from backend.execution.redeemer import run_auto_redeem
    await run_auto_redeem()


async def job_refresh_station_calibrations():
    from backend.modeling.station_calibration import refresh_all_station_calibrations
    await refresh_all_station_calibrations()


async def job_sync_positions():
    """Automatically synchronizes database positions with on-chain truth periodically."""
    from backend.execution.position_sync import sync_positions_from_chain
    await sync_positions_from_chain()


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
    add(job_fetch_open_meteo_models, seconds=900, name="fetch_om_models")  # 15 min
    add(job_fetch_om_hrrr_rapid, seconds=300, name="fetch_om_hrrr_rapid")  # 5 min, gated 40–65 min past the hour
    # Phase C4 — Herbie side-channel harness (no-op if herbie-data not installed).
    # Phase Q3 §13 — NOAA AIWP S3 archive (Pangu-Weather + FourCastNet v2-small).
    # Files upload ~5–8h post-init; 1-hour polling catches each new run within
    # the hour. Idempotency in fetch_aiwp_model() makes this cheap when nothing
    # is new — both jobs become no-ops between actual file landings.
    add(job_fetch_aiwp_pangu,           seconds=3600, name="fetch_aiwp_pangu")
    add(job_fetch_aiwp_fourcastnet_v2,  seconds=3600, name="fetch_aiwp_fourcastnet_v2")
    add(job_fetch_herbie_hrrr,   seconds=900,  name="fetch_herbie_hrrr")   # 15 min — hourly model
    add(job_fetch_herbie_nbm,    seconds=1800, name="fetch_herbie_nbm")    # 30 min
    add(job_fetch_herbie_ifs,    seconds=7200, name="fetch_herbie_ifs")    # 2 h — 4 runs/day
    add(job_fetch_herbie_aifs,   seconds=7200, name="fetch_herbie_aifs")   # 2 h — 4 runs/day
    add(job_fetch_wu,            seconds=300,  name="fetch_wu")    # 5 min
    add(job_fetch_gamma,         seconds=120,  name="fetch_gamma") # 2 min
    add(job_fetch_clob,          seconds=30,   name="fetch_clob")
    add(job_run_model,           seconds=60,   name="run_model")
    add(job_run_auto_trader,     seconds=60,   name="run_auto_trader")
    add(job_run_exit_engine,     seconds=300,  name="run_exit_engine")     # 5 min cascade
    add(job_run_night_owl,       seconds=300,  name="run_night_owl")       # 5 min cascade over night
    add(job_reconcile_orders,    seconds=30,   name="reconcile_orders")
    add(job_auto_check_disarm,   seconds=60,   name="auto_check_disarm")
    add(job_discover_cities,     seconds=86400, name="discover_cities")  # 24h
    add(job_refresh_station_profiles, seconds=3600, name="refresh_station_profiles")  # 1h
    add(job_fetch_metar_smart,       seconds=30,    name="fetch_metar_smart")
    add(job_fetch_tgftp_metar,      seconds=60,    name="fetch_tgftp_metar")
    add(job_fetch_madis,            seconds=300,   name="fetch_madis")
    add(job_check_resolved,          seconds=300,   name="check_resolved")  # 5 min
    add(job_auto_redeem,             seconds=43200, name="auto_redeem")  # 12h
    add(job_refresh_station_calibrations, seconds=21600, name="refresh_station_cal")  # 6h
    add(job_sync_positions,          seconds=600,   name="sync_positions") # 10 min

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
        (job_fetch_tgftp_metar, "fetch_tgftp_metar"),
        (job_fetch_madis, "fetch_madis"),
        (job_fetch_nws, "fetch_nws"),
        (job_fetch_wu, "fetch_wu"),
        (job_fetch_open_meteo, "fetch_open_meteo"),
        (job_fetch_open_meteo_models, "fetch_om_models"),
        (job_refresh_missing_station_profiles, "refresh_missing_station_profiles"),
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
