"""
Night Owl Strategy Orchestrator

Runs between 23:00 and 06:00 ET to exploit stale orderbooks after NWP models 
process the 00z and 06z cycles. During this window, Polymarket orderbooks 
sleep but deterministic models (HRRR, GFS, NBM) refresh with new insight for 
tomorrow's peak temperature.
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.engine.signal_engine import run_signal_engine
from backend.execution.arming import is_armed
from backend.execution.trader import execute_top_signals
from backend.ingestion.polymarket_clob import get_clob
from backend.config import Config

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


async def run_night_owl() -> None:
    """Evaluate and execute the Night Owl strategy if within time window."""
    if not await is_armed():
        return

    now_et = datetime.now(ET)
    
    # Only run in the 23:00 to 06:00 window (00z runs ~1-3am, 06z runs ~7-9am but we cut off at 6am)
    if not (now_et.hour >= 23 or now_et.hour < 6):
        return

    clob = get_clob()
    bankroll = Config.BANKROLL_CAP
    if clob and clob.can_trade:
        balance = await clob.get_balance()
        if balance is not None:
            bankroll = min(balance, Config.BANKROLL_CAP)

    log.info("night_owl: evaluating 00z/06z models for overnight alpha")
    
    # Generate signals across all cities
    signals = await run_signal_engine()
    
    # We pass 'night_owl' strategy context to execute_top_signals 
    # to bypass the standard 19:00 TRADING_WINDOW_CLOSE_ET gate in gating.py
    results = await execute_top_signals(
        signals, 
        bankroll=bankroll, 
        max_trades=Config.MAX_POSITIONS_PER_EVENT, 
        strategy="night_owl"
    )
    
    if results:
        success = [r for r in results if r.get("status") == "filled"]
        if success:
            log.info("night_owl: executed %d successful trades overnight", len(success))
