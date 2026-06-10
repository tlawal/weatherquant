"""Order-book and shadow flow helpers for execution diagnostics.

These helpers are deliberately side-effect free. The live engine can use the
book-sweep estimates for slippage and trailing diagnostics, while wallet/CLOB
flow remains shadow-only until it has out-of-sample evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import pstdev
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class FillSimulation:
    avg_price: float | None
    filled_size: float
    unfilled_size: float
    worst_price: float | None
    notional: float


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def parse_book_levels(raw_levels: Iterable[Any], *, side: str) -> list[dict[str, float]]:
    """Normalize CLOB levels and sort them in executable order."""
    levels: list[dict[str, float]] = []
    for row in raw_levels or []:
        if isinstance(row, dict):
            price = _to_float(row.get("price"))
            size = _to_float(row.get("size"))
        else:
            try:
                price = _to_float(row[0])
                size = _to_float(row[1])
            except (TypeError, IndexError):
                continue
        if price <= 0 or size <= 0:
            continue
        levels.append({"price": round(price, 4), "size": round(size, 6)})
    reverse = side.lower() == "bid"
    return sorted(levels, key=lambda item: item["price"], reverse=reverse)


def depth_at_touch(levels: Sequence[dict[str, float]]) -> float:
    if not levels:
        return 0.0
    best = levels[0]["price"]
    return round(sum(l["size"] for l in levels if abs(l["price"] - best) < 1e-9), 6)


def depth_within_cents(
    levels: Sequence[dict[str, float]],
    *,
    side: str,
    cents: float,
) -> float:
    if not levels:
        return 0.0
    best = levels[0]["price"]
    width = max(0.0, float(cents)) / 100.0
    if side.lower() == "bid":
        total = sum(l["size"] for l in levels if l["price"] >= best - width)
    else:
        total = sum(l["size"] for l in levels if l["price"] <= best + width)
    return round(total, 6)


def book_imbalance(bid_depth: float | None, ask_depth: float | None) -> float | None:
    bid = _to_float(bid_depth)
    ask = _to_float(ask_depth)
    total = bid + ask
    if total <= 0:
        return None
    return round((bid - ask) / total, 4)


def simulate_fill(levels: Sequence[dict[str, float]], shares: float) -> FillSimulation:
    remaining = max(0.0, _to_float(shares))
    if remaining <= 0:
        return FillSimulation(None, 0.0, 0.0, None, 0.0)

    filled = 0.0
    notional = 0.0
    worst_price: float | None = None
    for level in levels:
        if remaining <= 1e-9:
            break
        take = min(remaining, _to_float(level.get("size")))
        if take <= 0:
            continue
        price = _to_float(level.get("price"))
        filled += take
        notional += take * price
        worst_price = price
        remaining -= take

    avg = (notional / filled) if filled > 0 else None
    return FillSimulation(
        avg_price=round(avg, 4) if avg is not None else None,
        filled_size=round(filled, 6),
        unfilled_size=round(max(0.0, remaining), 6),
        worst_price=round(worst_price, 4) if worst_price is not None else None,
        notional=round(notional, 6),
    )


def rolling_mid_volatility(snapshots: Sequence[Any]) -> float:
    """Population stdev of consecutive mid-price changes in price units."""
    mids: list[float] = []
    for snap in snapshots or []:
        mid = _to_float(getattr(snap, "yes_mid", None), default=float("nan"))
        if math.isfinite(mid) and mid > 0:
            mids.append(mid)
    if len(mids) < 3:
        return 0.0
    deltas = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    return round(float(pstdev(deltas)), 6) if len(deltas) >= 2 else 0.0


def dynamic_trailing_distance(
    micro_vol_30m: float | None,
    regime_score: float | None,
    *,
    tier2_exited: bool = False,
) -> float:
    vol = max(0.0, _to_float(micro_vol_30m))
    regime = min(max(_to_float(regime_score), 0.0), 1.0)
    distance = (0.05 + 2.0 * vol) * (1.0 + 0.5 * regime)
    if tier2_exited:
        distance *= 0.75
    return round(min(max(distance, 0.03), 0.12), 4)


def compute_shadow_flow_features(
    trades: Iterable[Any],
    *,
    as_of: datetime | None = None,
    window_minutes: int = 15,
    top_wallet_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute shadow-only signed flow and VPIN-like toxicity.

    Trade side is treated as directional only when supplied by a trade feed or
    an already-normalized wallet record. Public book deltas alone are not used.
    """
    as_of = as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    start = as_of - timedelta(minutes=max(1, int(window_minutes)))
    top_wallet_scores = top_wallet_scores or {}

    buy_notional = 0.0
    sell_notional = 0.0
    weighted_flow = 0.0
    seen = 0
    for trade in trades or []:
        ts = getattr(trade, "timestamp", None) or getattr(trade, "trade_ts", None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < start or ts > as_of:
            continue
        side = str(getattr(trade, "side", "") or "").upper()
        if side not in {"BUY", "SELL"}:
            continue
        notional = _to_float(
            getattr(trade, "notional", None),
            default=_to_float(getattr(trade, "notional_usd", None)),
        )
        if notional <= 0:
            size = _to_float(getattr(trade, "size", None))
            price = _to_float(getattr(trade, "price", None))
            notional = abs(size * price)
        if notional <= 0:
            continue
        sign = 1.0 if side == "BUY" else -1.0
        if side == "BUY":
            buy_notional += notional
        else:
            sell_notional += notional
        wallet = str(getattr(trade, "wallet_address", "") or "").lower()
        weighted_flow += sign * notional * float(top_wallet_scores.get(wallet, 1.0))
        seen += 1

    total = buy_notional + sell_notional
    imbalance = ((buy_notional - sell_notional) / total) if total > 0 else 0.0
    vpin = abs(buy_notional - sell_notional) / total if total > 0 else 0.0
    confidence = 0.0 if seen == 0 else min(1.0, math.log1p(total) / math.log1p(1000.0))
    return {
        "window_minutes": int(window_minutes),
        "buy_notional": round(buy_notional, 4),
        "sell_notional": round(sell_notional, 4),
        "signed_net_notional": round(buy_notional - sell_notional, 4),
        "imbalance": round(imbalance, 4),
        "vpin": round(vpin, 4),
        "toxicity_score": round(vpin * confidence, 4),
        "top_wallet_weighted_flow": round(weighted_flow, 4),
        "direction_source": "data_api_side" if seen else "unavailable",
        "direction_confidence": round(confidence, 4),
        "trade_count": seen,
        "computed_at": as_of.isoformat(),
    }
