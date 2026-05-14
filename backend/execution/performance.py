"""Closed-trade performance ledger and summary analytics."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.backtesting.metrics import compute_max_drawdown, compute_profit_factor, compute_sharpe
from backend.storage.models import Bucket, City, ClosedTrade, Event, ExitEvent, Fill, Order, Position


def _json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _weighted_avg(items: list[tuple[float, float]]) -> tuple[float, float]:
    qty = sum(max(0.0, q) for q, _ in items)
    if qty <= 0:
        return 0.0, 0.0
    notional = sum(max(0.0, q) * p for q, p in items)
    return qty, notional / qty


def _hours_between(start: datetime | None, end: datetime | None) -> float | None:
    if not start or not end:
        return None
    try:
        return round((end - start).total_seconds() / 3600.0, 3)
    except Exception:
        return None


def _serialize_trade(t: ClosedTrade) -> dict[str, Any]:
    return {
        "id": t.id,
        "position_id": t.position_id,
        "bucket_id": t.bucket_id,
        "event_id": t.event_id,
        "city_slug": t.city_slug,
        "date_et": t.date_et,
        "bucket_idx": t.bucket_idx,
        "bucket_label": t.bucket_label,
        "entry_type": t.entry_type,
        "entry_strategy": t.entry_strategy,
        "exit_level": t.exit_level,
        "exit_reason": t.exit_reason,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "shares": t.shares,
        "avg_entry_price": t.avg_entry_price,
        "avg_exit_price": t.avg_exit_price,
        "fees": t.fees,
        "realized_pnl": t.realized_pnl,
        "final_outcome": t.final_outcome,
        "hold_to_redeem_value": t.hold_to_redeem_value,
        "foregone_pnl": t.foregone_pnl,
        "hold_time_hours": t.hold_time_hours,
        "entry_model_prob": t.entry_model_prob,
        "entry_market_prob": t.entry_market_prob,
        "entry_true_edge": t.entry_true_edge,
        "exit_market_bid": t.exit_market_bid,
        "exit_market_ask": t.exit_market_ask,
        "station_mae_f": t.station_mae_f,
        "time_window": t.time_window,
    }


async def materialize_closed_trades(sess: AsyncSession) -> int:
    """Build/update one closed-trade row for each locally closed position."""
    rows = (await sess.execute(
        select(Position, Bucket, Event, City)
        .join(Bucket, Position.bucket_id == Bucket.id)
        .join(Event, Bucket.event_id == Event.id)
        .join(City, Event.city_id == City.id)
        .where(Position.net_qty <= 0)
        .where(Position.entry_time.is_not(None))
    )).all()

    changed = 0
    now = datetime.now(timezone.utc)
    for pos, bucket, event, city in rows:
        orders = (await sess.execute(
            select(Order).where(Order.bucket_id == bucket.id).order_by(Order.created_at.asc())
        )).scalars().all()
        fills: list[tuple[Order, Fill]] = []
        for order in orders:
            for fill in (await sess.execute(
                select(Fill).where(Fill.order_id == order.id).order_by(Fill.filled_at.asc())
            )).scalars().all():
                fills.append((order, fill))

        buy_items = [(f.qty, f.price) for o, f in fills if o.side.startswith("buy")]
        sell_items = [(f.qty, f.price) for o, f in fills if o.side.startswith("sell")]
        buy_qty, avg_entry = _weighted_avg(buy_items)
        sell_qty, avg_exit = _weighted_avg(sell_items)
        shares = max(buy_qty, pos.original_qty or 0.0)
        avg_entry = avg_entry or pos.entry_price or pos.avg_cost or 0.0
        if avg_exit <= 0 and shares > 0 and pos.realized_pnl is not None:
            avg_exit = max(0.0, avg_entry + (pos.realized_pnl / shares))
        fees = sum(float(f.fee or 0.0) for _, f in fills)

        exit_events = (await sess.execute(
            select(ExitEvent)
            .where(ExitEvent.position_id == pos.id)
            .order_by(ExitEvent.ts.desc())
        )).scalars().all()
        exit_event = next((e for e in exit_events if (e.shares_exited or 0.0) > 0), exit_events[0] if exit_events else None)
        exit_time = None
        sell_fill_times = [f.filled_at for o, f in fills if o.side.startswith("sell")]
        if sell_fill_times:
            exit_time = max(sell_fill_times)
        elif exit_event:
            exit_time = exit_event.ts
        else:
            exit_time = pos.updated_at

        entry_snapshot = _json_loads(pos.entry_decision_json)
        source_payload = {
            "entry_decision": entry_snapshot,
            "exit_events": [_json_loads(e.reason_json) for e in exit_events[:10]],
        }
        won = event.winning_bucket_idx is not None and bucket.bucket_idx == event.winning_bucket_idx
        final_outcome = "win" if won else ("loss" if event.winning_bucket_idx is not None else "unknown")
        hold_to_redeem_value = shares * (1.0 if won else 0.0) if event.winning_bucket_idx is not None else None
        exit_value = shares * avg_exit if avg_exit is not None else None
        foregone = (
            round(hold_to_redeem_value - exit_value, 4)
            if hold_to_redeem_value is not None and exit_value is not None
            else None
        )
        realized_pnl = float(pos.realized_pnl or 0.0)
        if not realized_pnl and shares and avg_exit is not None:
            realized_pnl = shares * (avg_exit - avg_entry) - fees

        existing = (await sess.execute(
            select(ClosedTrade).where(ClosedTrade.position_id == pos.id)
        )).scalar_one_or_none()
        if existing is None:
            existing = ClosedTrade(position_id=pos.id, bucket_id=bucket.id, event_id=event.id)
            sess.add(existing)

        existing.city_slug = city.city_slug
        existing.date_et = str(event.date_et)
        existing.bucket_idx = bucket.bucket_idx
        existing.bucket_label = bucket.label or f"bucket {bucket.bucket_idx}"
        existing.entry_type = pos.entry_type
        existing.entry_strategy = pos.entry_strategy or entry_snapshot.get("entry_strategy") or pos.strategy
        existing.exit_level = exit_event.trigger_level if exit_event else None
        existing.exit_reason = exit_event.trigger_reason if exit_event else "local_closed"
        existing.entry_time = pos.entry_time
        existing.exit_time = exit_time
        existing.shares = float(shares or 0.0)
        existing.avg_entry_price = float(avg_entry or 0.0)
        existing.avg_exit_price = float(avg_exit or 0.0) if avg_exit is not None else None
        existing.fees = float(fees or 0.0)
        existing.realized_pnl = float(realized_pnl or 0.0)
        existing.final_outcome = final_outcome
        existing.hold_to_redeem_value = hold_to_redeem_value
        existing.foregone_pnl = foregone
        existing.hold_time_hours = _hours_between(pos.entry_time, exit_time)
        existing.entry_model_prob = entry_snapshot.get("model_prob")
        existing.entry_market_prob = entry_snapshot.get("market_prob")
        existing.entry_true_edge = entry_snapshot.get("true_edge")
        existing.exit_market_bid = exit_event.market_bid if exit_event else None
        existing.exit_market_ask = exit_event.market_ask if exit_event else None
        existing.station_mae_f = entry_snapshot.get("station_mae_f")
        existing.time_window = entry_snapshot.get("time_window")
        existing.source_json = json.dumps(source_payload, default=str)
        existing.updated_at = now
        changed += 1

    if changed:
        await sess.commit()
    return changed


async def get_closed_trade_rows(sess: AsyncSession, limit: int = 200) -> list[dict[str, Any]]:
    await materialize_closed_trades(sess)
    rows = (await sess.execute(
        select(ClosedTrade)
        .order_by(ClosedTrade.exit_time.desc().nullslast(), ClosedTrade.id.desc())
        .limit(limit)
    )).scalars().all()
    return [_serialize_trade(t) for t in rows]


def _breakdown(trades: list[ClosedTrade], attr: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
    for t in trades:
        grouped[str(getattr(t, attr, None) or "unknown")].append(t)
    out = []
    for key, rows in grouped.items():
        pnl = [float(t.realized_pnl or 0.0) for t in rows]
        wins = sum(1 for p in pnl if p > 0)
        out.append({
            "key": key,
            "n": len(rows),
            "total_pnl": round(sum(pnl), 4),
            "win_rate": round(wins / len(rows), 4) if rows else 0.0,
            "avg_pnl": round(sum(pnl) / len(rows), 4) if rows else 0.0,
        })
    out.sort(key=lambda r: (r["total_pnl"], r["n"]), reverse=True)
    return out


async def get_performance_summary(sess: AsyncSession) -> dict[str, Any]:
    await materialize_closed_trades(sess)
    trades = (await sess.execute(
        select(ClosedTrade).order_by(ClosedTrade.exit_time.asc().nullslast(), ClosedTrade.id.asc())
    )).scalars().all()
    pnl = [float(t.realized_pnl or 0.0) for t in trades]
    total = len(trades)
    wins = sum(1 for p in pnl if p > 0)
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        key = (t.exit_time.date().isoformat() if t.exit_time else t.date_et)
        daily_pnl[key] += float(t.realized_pnl or 0.0)
    equity = []
    running = 0.0
    for key in sorted(daily_pnl):
        running += daily_pnl[key]
        equity.append(running)
    max_dd, max_dd_pct = compute_max_drawdown(equity)
    daily_returns = list(daily_pnl.values())
    hold_times = [t.hold_time_hours for t in trades if t.hold_time_hours is not None]
    foregone = [t.foregone_pnl for t in trades if t.foregone_pnl is not None]
    pf = compute_profit_factor(pnl)
    return {
        "sample_status": "provisional" if total < 30 else "statistically_useful",
        "total_trades": total,
        "winning_trades": wins,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "total_pnl": round(sum(pnl), 4),
        "avg_pnl_per_trade": round(sum(pnl) / total, 4) if total else 0.0,
        "profit_factor": "inf" if isinstance(pf, float) and math.isinf(pf) else pf,
        "sharpe_ratio": compute_sharpe(daily_returns),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "avg_hold_time_hours": round(sum(hold_times) / len(hold_times), 3) if hold_times else 0.0,
        "total_foregone_pnl": round(sum(foregone), 4) if foregone else 0.0,
        "breakdowns": {
            "city": _breakdown(trades, "city_slug"),
            "entry_strategy": _breakdown(trades, "entry_strategy"),
            "exit_reason": _breakdown(trades, "exit_reason"),
            "time_window": _breakdown(trades, "time_window"),
        },
    }
