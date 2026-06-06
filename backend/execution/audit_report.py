"""Daily quant audit report assembly.

The report is deliberately read-only: it summarizes current DB state,
performance, model/runtime health, and known code-level findings so the
dashboard/API can render one operator-facing audit artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Config
from backend.execution.performance import (
    get_performance_summary,
    run_closed_trade_monte_carlo,
)
from backend.modeling.residual_tracker import is_ml_model_loaded
from backend.storage.models import ClosedTrade, Position, WorkerHeartbeat
from backend.storage.repos import get_all_positions, get_arming_state


CODE_AUDIT_FINDINGS = [
    {
        "area": "performance",
        "finding": "Old closed trades can be excluded from active stats without deleting audit rows.",
        "status": "implemented",
        "impact": "Prevents obsolete pre-upgrade trades from distorting PnL, Sharpe, drawdown, and MC risk.",
    },
    {
        "area": "execution_gates",
        "finding": "Risk-reducing SELL exits bypass entry-only gates.",
        "status": "implemented",
        "impact": "Prevents daily loss, arming, max-entry, and edge gates from trapping losing inventory.",
    },
    {
        "area": "execution_price",
        "finding": "BUY max-entry checks use executable yes_ask instead of mid.",
        "status": "implemented",
        "impact": "Reduces false actionable signals and adverse spread capture.",
    },
    {
        "area": "scheduler_db_pressure",
        "finding": "Signal generation is cached briefly and signal rows are committed once per event.",
        "status": "implemented",
        "impact": "Reduces duplicate model snapshots and asyncpg connection pressure.",
    },
    {
        "area": "residual_ml",
        "finding": "Promoted residual artifact hydrates before scheduler startup and predicts with named features.",
        "status": "implemented",
        "impact": "Restores trained remaining-rise model path and suppresses sklearn feature-name warnings.",
    },
    {
        "area": "smart_money",
        "finding": "Wallet flow remains shadow-only until on-chain direction and lead/lag features are validated.",
        "status": "shadow",
        "impact": "Avoids overfitting public wallet data while surfacing candidate alpha diagnostics.",
    },
]


RESEARCH_NOTES = [
    {
        "topic": "prediction market microstructure",
        "takeaway": "Direction and informed-flow attribution must be grounded in fill/order data, not feed impressions.",
        "application": "Keep wallet flow shadow-only until on-chain direction and lead/lag validation are available.",
    },
    {
        "topic": "probabilistic weather postprocessing",
        "takeaway": "Optimize CRPS/Brier and reliability by context, not point-error MAE alone.",
        "application": "Promote BMA/calibration overlays only by out-of-sample proper-score lift.",
    },
    {
        "topic": "optimal stopping",
        "takeaway": "Exit thresholds should be conditional on spread, depth, time-to-resolution, and model deterioration.",
        "application": "Segment Quick Flip/Ladder/EDGE_DECAY stats by entry edge and market regime before retuning.",
    },
]


async def build_daily_audit_report(sess: AsyncSession) -> dict[str, Any]:
    summary = await get_performance_summary(sess)
    mc_active = await run_closed_trade_monte_carlo(sess, include_excluded=False)
    arming = await get_arming_state(sess)
    positions = await get_all_positions(sess)
    open_positions = [p for p in positions if p.net_qty > 0]
    closed_total = (await sess.execute(select(func.count(ClosedTrade.id)))).scalar_one()
    closed_excluded = (
        await sess.execute(
            select(func.count(ClosedTrade.id)).where(
                ClosedTrade.excluded_from_stats == True  # noqa: E712
            )
        )
    ).scalar_one()
    heartbeats = (
        await sess.execute(
            select(WorkerHeartbeat).order_by(WorkerHeartbeat.last_run_at.desc()).limit(25)
        )
    ).scalars().all()

    now = datetime.now(timezone.utc)
    heartbeat_rows = []
    stale_heartbeats = []
    for hb in heartbeats:
        last = hb.last_run_at
        if last and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_s = (now - last).total_seconds() if last else None
        row = {
            "job_name": hb.job_name,
            "last_run_at": last.isoformat() if last else None,
            "age_s": round(age_s, 1) if age_s is not None else None,
            "success": not bool(hb.last_error),
            "run_count": hb.run_count,
            "error_count": hb.error_count,
            "error": hb.last_error,
        }
        heartbeat_rows.append(row)
        if age_s is None or age_s > 600 or hb.last_error:
            stale_heartbeats.append(row)

    return {
        "generated_at": now.isoformat(),
        "baseline": summary.get("baseline"),
        "arming": {
            "state": arming.state,
            "auto_trade_default": Config.AUTO_TRADE_DEFAULT,
            "bankroll_cap": Config.BANKROLL_CAP,
        },
        "sections": {
            "1_daily_code_math_audit_summary": CODE_AUDIT_FINDINGS,
            "2_live_log_trade_performance_forensics": {
                "performance_summary": summary,
                "closed_trades_total": int(closed_total or 0),
                "closed_trades_excluded": int(closed_excluded or 0),
                "open_positions": len(open_positions),
                "heartbeat_issues": stale_heartbeats,
            },
            "3_exit_execution_policy_audit": {
                "implemented_controls": [
                    "SELL exits bypass entry-only gates",
                    "BUY entries cap on executable ask",
                    "Ledger flags highlight impossible closed-trade rows",
                ],
                "next_measurement": "Segment exit PnL by entry edge, exit reason, spread, depth, and time-to-close.",
            },
            "4_monte_carlo_risk_simulation": mc_active,
            "5_new_signals_parameter_recommendations": [
                "Shadow executable spread/depth imbalance and order-book staleness.",
                "Shadow smart-wallet flow with cohort score, lead/lag, toxicity, and direction-source fields.",
                "Promote BMA or threshold calibration only after proper-score lift by city/station/hour/floor.",
            ],
            "6_academic_research_deep_dive": RESEARCH_NOTES,
            "7_actionable_code_changes": CODE_AUDIT_FINDINGS,
            "8_quant_projections_next_day_monitoring_plan": {
                "monitor": [
                    "No asyncpg non-checked-in connection warnings",
                    "No repeated APScheduler max_instances skips for model/exit jobs",
                    "Residual ML loaded true when Postgres artifact exists",
                    "Performance summary active baseline excludes reset trades",
                    "No sell exit blocked by buy-only gates",
                ],
                "go_no_go": "Keep auto-trading DISARMED until a full day of clean monitoring and active-baseline stats are sane.",
            },
        },
        "runtime_health": {
            "residual_ml_loaded": is_ml_model_loaded(),
            "heartbeats": heartbeat_rows,
        },
    }
