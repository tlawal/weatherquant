"""Realized-outcome calibration for exit policy.

This module converts closed-trade and exit-event history into evidence for
Quick Flip, EDGE_DECAY, urgent exits, and trailing stops. It is read-only:
recommendations are diagnostic until sample sizes are large enough and the
operator explicitly promotes a parameter change.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.execution.performance import materialize_closed_trades
from backend.storage.models import ClosedTrade, ExitEvent


@dataclass(frozen=True)
class ExitCalibrationParams:
    days_back: int = 90
    min_samples: int = 8
    include_excluded: bool = False

    def normalized(self) -> "ExitCalibrationParams":
        return ExitCalibrationParams(
            days_back=max(1, min(int(self.days_back), 730)),
            min_samples=max(2, min(int(self.min_samples), 1000)),
            include_excluded=bool(self.include_excluded),
        )


def _json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _regime_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.65:
        return "high"
    if value >= 0.35:
        return "medium"
    return "low"


def _extract_diagnostics(exit_event: dict[str, Any] | None) -> dict[str, Any]:
    payload = exit_event or {}
    cascade = payload.get("cascade") if isinstance(payload.get("cascade"), dict) else {}
    diagnostics = cascade.get("diagnostics") if isinstance(cascade.get("diagnostics"), dict) else {}
    trailing = diagnostics.get("trailing") if isinstance(diagnostics.get("trailing"), dict) else {}
    edge_decay = diagnostics if cascade.get("level") == "EDGE_DECAY" else {}
    return {
        "cascade": cascade,
        "diagnostics": diagnostics,
        "trailing": trailing,
        "edge_decay": edge_decay,
        "regime_score": (
            _num(trailing.get("regime_score"))
            or _num(diagnostics.get("regime_score"))
        ),
        "micro_vol_30m": (
            _num(trailing.get("micro_vol_30m"))
            or _num(diagnostics.get("micro_vol_30m"))
        ),
        "toxicity_score": _num(diagnostics.get("toxicity_score")),
        "trail_distance": _num(trailing.get("distance")),
        "edge_required_runs": _num(diagnostics.get("required_runs")),
        "edge_ev_drop": _num(diagnostics.get("ev_drop")),
        "edge_required_min_drop": _num(diagnostics.get("required_min_drop")),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "sample_status": "empty",
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "avg_foregone_pnl": None,
            "median_foregone_pnl": None,
            "avg_hold_hours": None,
            "avg_regime_score": None,
            "avg_micro_vol_30m": None,
        }
    pnls = [float(row.get("realized_pnl") or 0.0) for row in rows]
    foregone = [
        float(row["foregone_pnl"])
        for row in rows
        if row.get("foregone_pnl") is not None
    ]
    hold = [
        float(row["hold_time_hours"])
        for row in rows
        if row.get("hold_time_hours") is not None
    ]
    regimes = [
        float(row["regime_score"])
        for row in rows
        if row.get("regime_score") is not None
    ]
    micro_vols = [
        float(row["micro_vol_30m"])
        for row in rows
        if row.get("micro_vol_30m") is not None
    ]
    wins = sum(1 for pnl in pnls if pnl > 0)
    return {
        "n": len(rows),
        "sample_status": "low_n" if len(rows) < 30 else "useful",
        "total_pnl": round(sum(pnls), 4),
        "avg_pnl": round(sum(pnls) / len(pnls), 4),
        "win_rate": round(wins / len(rows), 4),
        "avg_foregone_pnl": round(sum(foregone) / len(foregone), 4) if foregone else None,
        "median_foregone_pnl": round(float(median(foregone)), 4) if foregone else None,
        "avg_hold_hours": round(sum(hold) / len(hold), 3) if hold else None,
        "avg_regime_score": round(sum(regimes) / len(regimes), 4) if regimes else None,
        "avg_micro_vol_30m": round(sum(micro_vols) / len(micro_vols), 6) if micro_vols else None,
    }


def _group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    out = []
    for label, items in grouped.items():
        out.append({"label": label, **_summarize(items)})
    out.sort(key=lambda item: (item["n"], item["total_pnl"]), reverse=True)
    return out


def _recommendations(rows: list[dict[str, Any]], min_samples: int) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    by_reason = {row["label"]: row for row in _group(rows, "exit_reason")}
    quick = by_reason.get("quick_flip")
    if quick and quick["n"] >= min_samples:
        avg_foregone = quick.get("avg_foregone_pnl")
        if avg_foregone is not None and avg_foregone > 0.03:
            recs.append({
                "policy": "quick_flip_target",
                "action": "raise_or_add_regime_condition",
                "confidence": "moderate" if quick["n"] >= 30 else "low",
                "reason": "Quick Flip exits are giving up positive settlement value on average.",
                "evidence": quick,
            })
        elif avg_foregone is not None and avg_foregone < -0.03:
            recs.append({
                "policy": "quick_flip_target",
                "action": "keep_or_expand_for_similar_regimes",
                "confidence": "moderate" if quick["n"] >= 30 else "low",
                "reason": "Quick Flip exits are saving value versus holding to settlement.",
                "evidence": quick,
            })

    for reason in ("trailing_stop", "tier_1_50pct", "tier_2_75pct"):
        row = by_reason.get(reason)
        if not row or row["n"] < min_samples:
            continue
        avg_foregone = row.get("avg_foregone_pnl")
        if avg_foregone is not None and avg_foregone > 0.03:
            recs.append({
                "policy": "dynamic_trailing_distance",
                "action": "widen_in_matching_regimes",
                "confidence": "moderate" if row["n"] >= 30 else "low",
                "reason": f"{reason} exits are leaving too much settlement value.",
                "evidence": row,
            })
        elif avg_foregone is not None and avg_foregone < -0.03:
            recs.append({
                "policy": "dynamic_trailing_distance",
                "action": "keep_or_tighten_in_matching_regimes",
                "confidence": "moderate" if row["n"] >= 30 else "low",
                "reason": f"{reason} exits are protecting value versus holding.",
                "evidence": row,
            })

    edge = by_reason.get("ev_decayed")
    if edge and edge["n"] >= min_samples:
        avg_foregone = edge.get("avg_foregone_pnl")
        if avg_foregone is not None and avg_foregone > 0.03:
            recs.append({
                "policy": "edge_decay",
                "action": "increase_debounce_or_required_drop",
                "confidence": "moderate" if edge["n"] >= 30 else "low",
                "reason": "EDGE_DECAY exits are too aggressive versus settlement value.",
                "evidence": edge,
            })
        elif avg_foregone is not None and avg_foregone < -0.03:
            recs.append({
                "policy": "edge_decay",
                "action": "keep_thresholds_pending_more_samples",
                "confidence": "moderate" if edge["n"] >= 30 else "low",
                "reason": "EDGE_DECAY exits are saving value versus holding.",
                "evidence": edge,
            })

    if not recs:
        recs.append({
            "policy": "all_exit_parameters",
            "action": "collect_more_realized_trades",
            "confidence": "low",
            "reason": "No exit bucket has enough realized evidence and foregone-PnL signal to retune safely.",
        })
    return recs


def build_exit_calibration_report(
    rows: list[dict[str, Any]],
    *,
    params: ExitCalibrationParams,
) -> dict[str, Any]:
    params = params.normalized()
    by_reason = _group(rows, "exit_reason")
    by_level = _group(rows, "exit_level")
    by_regime = _group(rows, "regime_bin")
    by_reason_regime = _group(rows, "reason_regime")
    recs = _recommendations(rows, params.min_samples)
    return {
        "params": asdict(params),
        "sample": _summarize(rows),
        "breakdowns": {
            "exit_reason": by_reason,
            "exit_level": by_level,
            "regime_bin": by_regime,
            "exit_reason_x_regime": by_reason_regime,
        },
        "recommendations": recs,
        "promotion": {
            "allowed_for_live_parameter_change": False,
            "rule": (
                "Retune only after a reason/regime cell has at least min_samples, "
                "stable negative/positive foregone-PnL evidence, and no degradation "
                "in drawdown or missed-winner exits in a replay backtest."
            ),
        },
    }


async def evaluate_exit_policy_calibration(
    session: AsyncSession,
    params: ExitCalibrationParams | None = None,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    params = (params or ExitCalibrationParams()).normalized()
    as_of = as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    cutoff = as_of - timedelta(days=params.days_back)

    await materialize_closed_trades(session)
    q = select(ClosedTrade).where(ClosedTrade.exit_time >= cutoff)
    if not params.include_excluded:
        q = q.where(ClosedTrade.excluded_from_stats == False)  # noqa: E712
    trades = (await session.execute(q.order_by(ClosedTrade.exit_time.asc()))).scalars().all()
    position_ids = [int(t.position_id) for t in trades]
    events_by_position: dict[int, list[ExitEvent]] = {}
    if position_ids:
        exit_events = (
            await session.execute(
                select(ExitEvent)
                .where(ExitEvent.position_id.in_(position_ids))
                .order_by(ExitEvent.position_id, ExitEvent.ts.desc())
            )
        ).scalars().all()
        for event in exit_events:
            events_by_position.setdefault(int(event.position_id), []).append(event)

    rows: list[dict[str, Any]] = []
    for trade in trades:
        exit_event = next(
            (
                event
                for event in events_by_position.get(int(trade.position_id), [])
                if (event.shares_exited or 0.0) > 0
            ),
            events_by_position.get(int(trade.position_id), [None])[0],
        )
        payload = _json_loads(exit_event.reason_json if exit_event else None)
        diag = _extract_diagnostics(payload)
        exit_reason = trade.exit_reason or (exit_event.trigger_reason if exit_event else None) or "unknown"
        exit_level = trade.exit_level or (exit_event.trigger_level if exit_event else None) or "unknown"
        regime_score = diag["regime_score"]
        regime = _regime_bin(regime_score)
        rows.append({
            "closed_trade_id": trade.id,
            "position_id": trade.position_id,
            "city_slug": trade.city_slug,
            "date_et": trade.date_et,
            "exit_level": exit_level,
            "exit_reason": exit_reason,
            "regime_score": regime_score,
            "regime_bin": regime,
            "reason_regime": f"{exit_reason}:{regime}",
            "micro_vol_30m": diag["micro_vol_30m"],
            "toxicity_score": diag["toxicity_score"],
            "trail_distance": diag["trail_distance"],
            "edge_required_runs": diag["edge_required_runs"],
            "edge_ev_drop": diag["edge_ev_drop"],
            "edge_required_min_drop": diag["edge_required_min_drop"],
            "realized_pnl": float(trade.realized_pnl or 0.0),
            "foregone_pnl": trade.foregone_pnl,
            "hold_time_hours": trade.hold_time_hours,
            "shares": trade.shares,
            "avg_entry_price": trade.avg_entry_price,
            "avg_exit_price": trade.avg_exit_price,
            "final_outcome": trade.final_outcome,
            "entry_true_edge": trade.entry_true_edge,
        })

    report = build_exit_calibration_report(rows, params=params)
    report["sample_preview"] = rows[:25]
    return report
