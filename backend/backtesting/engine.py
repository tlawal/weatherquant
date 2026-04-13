"""
Walk-forward backtester — replays stored model/market snapshots against
resolved Polymarket outcomes.

Usage:
    engine = BacktestEngine(BacktestParams())
    result = await engine.run()
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Optional

import aiohttp
from sqlalchemy import select, desc, func

from backend.backtesting.metrics import (
    BacktestMetrics,
    build_equity_curve,
    compute_brier,
    compute_max_drawdown,
    compute_profit_factor,
    compute_reliability_bins,
    compute_sharpe,
)
from backend.engine.signal_engine import _execution_cost
from backend.strategy.kelly import calculate_kelly_fraction
from backend.storage.db import get_session
from backend.storage.models import (
    BacktestRun,
    BacktestTrade,
    Bucket,
    City,
    Event,
    MarketSnapshot,
    ModelSnapshot,
)

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"

# Month name → number for slug parsing
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_SLUG_RE = re.compile(
    r"highest-temperature-in-(.+?)-on-([a-z]+)-(\d{1,2})-(\d{4})"
)


# ─── Parameters ──────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    """All tunable parameters for a backtest run."""
    bankroll: float = 10.0
    kelly_fraction: float = 0.10
    max_position_pct: float = 0.10
    min_true_edge: float = 0.10
    max_entry_price: float = 0.36
    max_spread: float = 0.04
    min_liquidity_shares: float = 10.0
    max_positions_per_event: int = 2
    night_owl_enabled: bool = True
    night_owl_start_hour: int = 23
    night_owl_end_hour: int = 6
    quick_flip_target: float = 0.05
    walk_forward_train_days: int = 21
    walk_forward_test_days: int = 7


# ─── Internal types ──────────────────────────────────────────────────────────

@dataclass
class SimTrade:
    """One simulated trade."""
    city_slug: str
    date_et: str
    bucket_idx: int
    bucket_label: str
    model_prob: float
    mkt_prob: float
    true_edge: float
    side: str         # "buy_yes"
    entry_price: float
    shares: float
    cost: float
    won: Optional[bool] = None
    pnl: float = 0.0
    exit_reason: Optional[str] = None


@dataclass
class Portfolio:
    """Running portfolio state during backtest."""
    bankroll: float
    equity: float
    positions_per_event: dict = field(default_factory=lambda: defaultdict(int))

    def available(self) -> float:
        return max(0.0, self.bankroll)


# ─── Gamma API enrichment ────────────────────────────────────────────────────

def parse_gamma_slug(slug: str) -> tuple[Optional[str], Optional[str]]:
    """Parse 'highest-temperature-in-atlanta-on-april-12-2026' → ('atlanta', '2026-04-12')."""
    m = _SLUG_RE.search(slug)
    if not m:
        return None, None
    city_part = m.group(1)
    month_name = m.group(2).lower()
    day = int(m.group(3))
    year = int(m.group(4))
    month_num = _MONTH_MAP.get(month_name)
    if not month_num:
        return None, None
    date_et = f"{year}-{month_num:02d}-{day:02d}"
    return city_part, date_et


def extract_winning_bucket(markets: list[dict]) -> Optional[int]:
    """From a Gamma event's markets array, find which bucket index resolved YES.

    Markets are sorted by bucket index in Gamma; the winning one has
    outcome == 'Yes' or the highest resolution value.
    """
    if not markets:
        return None

    for i, mkt in enumerate(markets):
        outcome = mkt.get("outcome")
        if outcome and str(outcome).lower() == "yes":
            return i
        # Some formats use "winner" boolean
        if mkt.get("winner") is True:
            return i

    return None


async def enrich_from_gamma() -> int:
    """Fetch closed weather markets from Gamma API, fill missing winning_bucket_idx."""
    enriched = 0
    offset = 0
    headers = {"User-Agent": "WeatherQuant-Backtest/1.0"}
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as http:
        while True:
            url = f"{GAMMA_API}/events"
            params = {
                "closed": "true",
                "tag": "Weather",
                "limit": "100",
                "offset": str(offset),
            }
            try:
                resp = await http.get(url, params=params)
                data = await resp.json(content_type=None)
            except Exception as e:
                log.warning("gamma enrich: request failed at offset=%d: %s", offset, e)
                break

            if not data or not isinstance(data, list):
                break

            for event_data in data:
                slug = event_data.get("slug", "")
                if "highest-temperature-in-" not in slug:
                    continue

                city_slug, date_et = parse_gamma_slug(slug)
                if not city_slug or not date_et:
                    continue

                event_markets = event_data.get("markets", [])
                winning_idx = extract_winning_bucket(event_markets)
                if winning_idx is None:
                    continue

                # Try to match to local DB
                async with get_session() as sess:
                    query = (
                        select(Event)
                        .join(City, Event.city_id == City.id)
                        .where(City.city_slug == city_slug)
                        .where(Event.date_et == date_et)
                    )
                    result = await sess.execute(query)
                    local_event = result.scalar_one_or_none()

                    if local_event and local_event.winning_bucket_idx is None:
                        local_event.winning_bucket_idx = winning_idx
                        if local_event.resolved_at is None:
                            local_event.resolved_at = datetime.now(timezone.utc)
                        await sess.commit()
                        enriched += 1

            offset += 100
            if len(data) < 100:
                break

    log.info("gamma enrich: updated %d events with resolution data", enriched)
    return enriched


# ─── Data loading ────────────────────────────────────────────────────────────

async def fetch_resolved_events() -> list[dict]:
    """Load all resolved events with their buckets, model snapshots, and market snapshots."""
    async with get_session() as sess:
        # Get events with known outcomes
        query = (
            select(Event, City)
            .join(City, Event.city_id == City.id)
            .where(Event.winning_bucket_idx.isnot(None))
            .order_by(Event.date_et)
        )
        rows = (await sess.execute(query)).all()

        events_data = []
        for event, city in rows:
            # Load buckets
            bq = select(Bucket).where(Bucket.event_id == event.id).order_by(Bucket.bucket_idx)
            buckets = (await sess.execute(bq)).scalars().all()
            if not buckets:
                continue

            # Load latest model snapshot per event (the one that would drive trading)
            msq = (
                select(ModelSnapshot)
                .where(ModelSnapshot.event_id == event.id)
                .order_by(desc(ModelSnapshot.computed_at))
                .limit(1)
            )
            snap = (await sess.execute(msq)).scalar_one_or_none()
            if not snap:
                continue

            # Load market snapshots for each bucket (latest before model snapshot)
            mkt_snaps = {}
            for b in buckets:
                mkq = (
                    select(MarketSnapshot)
                    .where(MarketSnapshot.bucket_id == b.id)
                    .where(MarketSnapshot.fetched_at <= snap.computed_at)
                    .order_by(desc(MarketSnapshot.fetched_at))
                    .limit(1)
                )
                ms = (await sess.execute(mkq)).scalar_one_or_none()
                if ms:
                    mkt_snaps[b.bucket_idx] = {
                        "yes_mid": ms.yes_mid,
                        "yes_bid": ms.yes_bid,
                        "yes_ask": ms.yes_ask,
                        "spread": ms.spread,
                        "ask_depth": ms.yes_ask_depth or 0.0,
                    }

            # Load ALL market snapshots for quick-flip detection
            all_mkt_snaps = {}
            for b in buckets:
                amq = (
                    select(MarketSnapshot)
                    .where(MarketSnapshot.bucket_id == b.id)
                    .where(MarketSnapshot.fetched_at > snap.computed_at)
                    .order_by(MarketSnapshot.fetched_at)
                )
                later_snaps = (await sess.execute(amq)).scalars().all()
                all_mkt_snaps[b.bucket_idx] = [
                    {"yes_bid": s.yes_bid, "fetched_at": s.fetched_at}
                    for s in later_snaps
                    if s.yes_bid is not None
                ]

            probs = json.loads(snap.probs_json) if snap.probs_json else []

            events_data.append({
                "event_id": event.id,
                "city_slug": city.city_slug,
                "city_display": city.display_name,
                "date_et": event.date_et,
                "winning_bucket_idx": event.winning_bucket_idx,
                "buckets": [
                    {
                        "idx": b.bucket_idx,
                        "label": b.label or f"Bucket {b.bucket_idx}",
                        "low_f": b.low_f,
                        "high_f": b.high_f,
                    }
                    for b in buckets
                ],
                "model_probs": probs,
                "model_mu": snap.mu,
                "model_sigma": snap.sigma,
                "market_data": mkt_snaps,
                "later_market_data": all_mkt_snaps,
                "snapshot_time": snap.computed_at,
            })

    log.info("backtest: loaded %d resolved events", len(events_data))
    return events_data


# ─── Trade simulation ────────────────────────────────────────────────────────

def simulate_entry(
    event_data: dict,
    bucket_idx: int,
    params: BacktestParams,
    portfolio: Portfolio,
) -> Optional[SimTrade]:
    """Evaluate one bucket for a trade entry. Returns SimTrade if taken."""
    probs = event_data["model_probs"]
    mkt = event_data["market_data"].get(bucket_idx)
    bucket_info = next((b for b in event_data["buckets"] if b["idx"] == bucket_idx), None)

    if not mkt or not bucket_info or bucket_idx >= len(probs):
        return None

    model_prob = probs[bucket_idx]
    mkt_prob = mkt["yes_mid"]
    spread = mkt.get("spread")
    ask_depth = mkt.get("ask_depth", 0.0)

    if mkt_prob is None or mkt_prob <= 0:
        return None

    exec_cost = _execution_cost(spread, ask_depth)
    true_edge = model_prob - mkt_prob - exec_cost

    # Apply gates
    if true_edge < params.min_true_edge:
        return None
    if mkt_prob < 0.02 or mkt_prob > 0.98:
        return None
    if ask_depth < params.min_liquidity_shares:
        return None

    entry_price = mkt.get("yes_ask") or mkt_prob
    if entry_price > params.max_entry_price:
        return None
    if spread is not None and spread > params.max_spread:
        return None

    # Check position limits
    event_key = event_data["event_id"]
    if portfolio.positions_per_event[event_key] >= params.max_positions_per_event:
        return None

    # Kelly sizing
    kelly_f = calculate_kelly_fraction(
        model_prob=model_prob,
        yes_price=entry_price,
        fractional_kelly=params.kelly_fraction,
        max_position_size=params.max_position_pct,
    )
    if kelly_f <= 0:
        return None

    effective_bankroll = portfolio.available()
    kelly_size = kelly_f * effective_bankroll
    position_cap = effective_bankroll * params.max_position_pct
    final_size = min(kelly_size, position_cap, effective_bankroll)

    shares = math.floor((final_size / entry_price) * 100) / 100
    cost = round(shares * entry_price, 4)
    if cost < 0.50:  # minimum trade size
        return None

    # Execute
    portfolio.bankroll -= cost
    portfolio.positions_per_event[event_key] += 1

    return SimTrade(
        city_slug=event_data["city_slug"],
        date_et=event_data["date_et"],
        bucket_idx=bucket_idx,
        bucket_label=bucket_info["label"],
        model_prob=round(model_prob, 4),
        mkt_prob=round(mkt_prob, 4),
        true_edge=round(true_edge, 4),
        side="buy_yes",
        entry_price=round(entry_price, 4),
        shares=shares,
        cost=cost,
    )


def simulate_exits(
    trades: list[SimTrade],
    events_map: dict[tuple[str, str], dict],
    params: BacktestParams,
) -> None:
    """Resolve all open trades: quick-flip, resolution, or expiry."""
    for trade in trades:
        if trade.won is not None:
            continue  # already resolved

        key = (trade.city_slug, trade.date_et)
        event_data = events_map.get(key)
        if not event_data:
            # No event data — mark as loss
            trade.won = False
            trade.pnl = -trade.cost
            trade.exit_reason = "no_data"
            continue

        winning_idx = event_data["winning_bucket_idx"]
        later_mkt = event_data.get("later_market_data", {}).get(trade.bucket_idx, [])

        # 1. Quick flip check — did bid ever reach entry + target?
        flip_price = trade.entry_price + params.quick_flip_target
        for snap in later_mkt:
            if snap["yes_bid"] and snap["yes_bid"] >= flip_price:
                trade.won = True
                payout = trade.shares * snap["yes_bid"]
                fee = payout * 0.02
                trade.pnl = round(payout - fee - trade.cost, 4)
                trade.exit_reason = "quick_flip"
                break

        if trade.won is not None:
            continue

        # 2. Resolution — hold to maturity
        if trade.bucket_idx == winning_idx:
            trade.won = True
            payout = trade.shares * 1.0  # YES resolves at $1
            fee = payout * 0.02
            trade.pnl = round(payout - fee - trade.cost, 4)
            trade.exit_reason = "resolved_win"
        else:
            trade.won = False
            trade.pnl = round(-trade.cost, 4)  # YES resolves at $0
            trade.exit_reason = "resolved_loss"


# ─── Walk-forward optimization ───────────────────────────────────────────────

def _run_single_pass(
    events: list[dict],
    params: BacktestParams,
) -> tuple[list[SimTrade], dict[str, float]]:
    """Run a single backtest pass (no walk-forward). Returns trades + daily P&L."""
    portfolio = Portfolio(bankroll=params.bankroll, equity=params.bankroll)
    trades: list[SimTrade] = []

    # Sort events by date
    sorted_events = sorted(events, key=lambda e: e["date_et"])
    events_map = {(e["city_slug"], e["date_et"]): e for e in sorted_events}

    for event_data in sorted_events:
        probs = event_data.get("model_probs", [])
        # Rank buckets by expected edge (highest first)
        bucket_edges = []
        for i in range(len(probs)):
            mkt = event_data["market_data"].get(i)
            if mkt and mkt.get("yes_mid"):
                edge = probs[i] - mkt["yes_mid"]
                bucket_edges.append((edge, i))
        bucket_edges.sort(reverse=True)

        for _, bucket_idx in bucket_edges:
            trade = simulate_entry(event_data, bucket_idx, params, portfolio)
            if trade:
                trades.append(trade)

    # Resolve all trades
    events_map_full = {(e["city_slug"], e["date_et"]): e for e in sorted_events}
    simulate_exits(trades, events_map_full, params)

    # Return bankroll from resolved trades
    for trade in trades:
        if trade.pnl is not None:
            portfolio.bankroll += trade.cost + trade.pnl

    # Build daily P&L
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        daily_pnl[t.date_et] += t.pnl

    return trades, dict(daily_pnl)


def optimize_params(
    train_events: list[dict],
    base_params: BacktestParams,
) -> BacktestParams:
    """Grid search over parameter space on training data.

    min_true_edge is EXCLUDED — the probability calibration engine already adapts
    the model surface, making threshold optimization prone to overfitting.
    """
    best_sharpe = -float("inf")
    best_params = base_params

    kelly_grid = [0.05, 0.10, 0.15, 0.20]
    entry_price_grid = [0.30, 0.36, 0.45]
    flip_target_grid = [0.03, 0.05, 0.08]

    for kf, mep, qft in product(kelly_grid, entry_price_grid, flip_target_grid):
        trial = BacktestParams(
            **{
                **asdict(base_params),
                "kelly_fraction": kf,
                "max_entry_price": mep,
                "quick_flip_target": qft,
            }
        )
        trades, daily_pnl = _run_single_pass(train_events, trial)
        if not daily_pnl:
            continue
        sharpe = compute_sharpe(list(daily_pnl.values()))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = trial

    return best_params


# ─── Main engine ─────────────────────────────────────────────────────────────

class BacktestEngine:
    """Walk-forward backtester using stored model/market snapshots."""

    def __init__(self, params: BacktestParams):
        self.params = params

    async def run(self, run_id: Optional[int] = None) -> dict:
        """Full backtest pipeline. Returns serializable results dict."""
        try:
            events = await fetch_resolved_events()
            if not events:
                return self._fail(run_id, "No resolved events found. Try 'Enrich from Gamma' first.")

            sorted_events = sorted(events, key=lambda e: e["date_et"])
            dates = [e["date_et"] for e in sorted_events]
            start_date = dates[0]
            end_date = dates[-1]

            # Decide mode: walk-forward if enough data, else single pass
            train_days = self.params.walk_forward_train_days
            test_days = self.params.walk_forward_test_days
            unique_dates = sorted(set(dates))

            if len(unique_dates) >= train_days + test_days:
                all_trades, daily_pnl = self._walk_forward(sorted_events, unique_dates)
            else:
                all_trades, daily_pnl = _run_single_pass(sorted_events, self.params)

            # Build results
            result = self._build_results(all_trades, daily_pnl, start_date, end_date)

            # Persist to DB
            if run_id is not None:
                await self._persist(run_id, result, all_trades)

            return result

        except Exception as e:
            log.exception("backtest: engine failed")
            return self._fail(run_id, str(e))

    def _walk_forward(
        self,
        sorted_events: list[dict],
        unique_dates: list[str],
    ) -> tuple[list[SimTrade], dict[str, float]]:
        """Rolling window walk-forward: train on N days, test on next M days."""
        train_days = self.params.walk_forward_train_days
        test_days = self.params.walk_forward_test_days

        all_trades: list[SimTrade] = []
        all_daily_pnl: dict[str, float] = defaultdict(float)

        i = 0
        while i + train_days + test_days <= len(unique_dates):
            train_date_set = set(unique_dates[i: i + train_days])
            test_date_set = set(unique_dates[i + train_days: i + train_days + test_days])

            train_events = [e for e in sorted_events if e["date_et"] in train_date_set]
            test_events = [e for e in sorted_events if e["date_et"] in test_date_set]

            if not train_events or not test_events:
                i += test_days
                continue

            # Optimize on training window
            best_params = optimize_params(train_events, self.params)

            # Evaluate on test window with optimized params (no lookahead)
            trades, daily_pnl = _run_single_pass(test_events, best_params)
            all_trades.extend(trades)
            for d, pnl in daily_pnl.items():
                all_daily_pnl[d] += pnl

            i += test_days

        return all_trades, dict(all_daily_pnl)

    def _build_results(
        self,
        trades: list[SimTrade],
        daily_pnl: dict[str, float],
        start_date: str,
        end_date: str,
    ) -> dict:
        """Compute all metrics and build the full results payload."""
        total_pnl = sum(t.pnl for t in trades)
        winners = [t for t in trades if t.won]
        win_rate = len(winners) / len(trades) if trades else 0.0

        # Brier score: model_prob vs binary outcome
        brier_preds = [(t.model_prob, 1 if t.won else 0) for t in trades]
        brier, bss = compute_brier(brier_preds)

        # Equity curve
        equity_curve = build_equity_curve(daily_pnl, self.params.bankroll)
        equity_values = [pt["equity"] for pt in equity_curve]
        max_dd, max_dd_pct = compute_max_drawdown(equity_values)

        # Sharpe
        daily_returns = list(daily_pnl.values())
        sharpe = compute_sharpe(daily_returns)

        # Reliability diagram
        reliability = compute_reliability_bins(brier_preds)

        # Profit factor
        pf = compute_profit_factor([t.pnl for t in trades])

        # Per-city breakdown
        per_city: dict[str, dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "edges": []})
        for t in trades:
            pc = per_city[t.city_slug]
            pc["trades"] += 1
            if t.won:
                pc["wins"] += 1
            pc["pnl"] += t.pnl
            pc["edges"].append(t.true_edge)

        per_city_final = {}
        for slug, data in per_city.items():
            per_city_final[slug] = {
                "trades": data["trades"],
                "win_rate": round(data["wins"] / data["trades"], 4) if data["trades"] else 0,
                "pnl": round(data["pnl"], 4),
                "avg_edge": round(sum(data["edges"]) / len(data["edges"]), 4) if data["edges"] else 0,
            }

        avg_edge = sum(t.true_edge for t in trades) / len(trades) if trades else 0.0

        metrics = BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(winners),
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 4),
            avg_pnl_per_trade=round(total_pnl / len(trades), 4) if trades else 0.0,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=round(sharpe, 4),
            brier_score=brier,
            brier_skill_score=bss,
            avg_true_edge=round(avg_edge, 4),
            profit_factor=pf,
        )

        # Trade log for UI
        trade_log = [
            {
                "city_slug": t.city_slug,
                "date_et": t.date_et,
                "bucket_idx": t.bucket_idx,
                "bucket_label": t.bucket_label,
                "model_prob": t.model_prob,
                "mkt_prob": t.mkt_prob,
                "true_edge": t.true_edge,
                "side": t.side,
                "entry_price": t.entry_price,
                "shares": t.shares,
                "cost": t.cost,
                "won": t.won,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ]

        return {
            "status": "completed",
            "start_date": start_date,
            "end_date": end_date,
            "params": asdict(self.params),
            "metrics": asdict(metrics),
            "equity_curve": equity_curve,
            "daily_pnl": [{"date": d, "pnl": round(p, 4)} for d, p in sorted(daily_pnl.items())],
            "reliability_bins": reliability,
            "per_city": per_city_final,
            "trade_log": trade_log,
        }

    async def _persist(self, run_id: int, result: dict, trades: list[SimTrade]) -> None:
        """Write results back to backtest_runs + backtest_trades."""
        metrics = result["metrics"]
        async with get_session() as sess:
            run = await sess.get(BacktestRun, run_id)
            if not run:
                return

            run.status = "completed"
            run.start_date = result["start_date"]
            run.end_date = result["end_date"]
            run.total_trades = metrics["total_trades"]
            run.winning_trades = metrics["winning_trades"]
            run.total_pnl = metrics["total_pnl"]
            run.max_drawdown = metrics["max_drawdown"]
            run.sharpe_ratio = metrics["sharpe_ratio"]
            run.brier_score = metrics["brier_score"]
            run.brier_skill_score = metrics["brier_skill_score"]
            run.win_rate = metrics["win_rate"]
            run.avg_edge = metrics["avg_true_edge"]
            run.results_json = json.dumps(result)

            for t in trades:
                sess.add(BacktestTrade(
                    run_id=run_id,
                    city_slug=t.city_slug,
                    date_et=t.date_et,
                    bucket_idx=t.bucket_idx,
                    bucket_label=t.bucket_label,
                    model_prob=t.model_prob,
                    mkt_prob=t.mkt_prob,
                    true_edge=t.true_edge,
                    side=t.side,
                    entry_price=t.entry_price,
                    shares=t.shares,
                    cost=t.cost,
                    won=t.won,
                    pnl=t.pnl,
                    exit_reason=t.exit_reason,
                ))

            await sess.commit()

    def _fail(self, run_id: Optional[int], error: str) -> dict:
        """Mark run as failed if persisted."""
        if run_id is not None:
            import asyncio
            asyncio.ensure_future(self._persist_failure(run_id, error))
        return {"status": "failed", "error": error}

    async def _persist_failure(self, run_id: int, error: str) -> None:
        async with get_session() as sess:
            run = await sess.get(BacktestRun, run_id)
            if run:
                run.status = "failed"
                run.error_msg = error
                await sess.commit()
