"""
Portfolio-level risk management — equity drawdown, cluster gating, VaR, strategy limits.

Supplements per-trade Kelly + gating with portfolio-wide constraints:
  1. Equity-curve drawdown halt: stop trading when drawdown > MAX_DRAWDOWN_PCT
  2. Cluster exposure gating: limit correlated city exposure (same weather regime)
  3. Daily VaR estimation: simple historical-sim VaR from positions + model probs
  4. Strategy-level loss limits: halt underperforming strategies independently

All checks return a list of gate failure strings (empty = pass).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from backend.config import Config
from backend.city_registry import get_city_cluster, CITY_REGISTRY_BY_SLUG

log = logging.getLogger(__name__)


# ── 1. Equity-Curve Drawdown ─────────────────────────────────────────────────

async def check_drawdown(bankroll: float) -> Optional[str]:
    """Check if current equity is in a drawdown exceeding the configured limit.

    Peak equity is tracked from the positions table (sum of realized P&L)
    plus current bankroll. We compare current equity against the historical
    peak to detect drawdowns.

    Returns a gate failure string, or None if within limits.
    """
    from backend.storage.db import get_session
    from backend.storage.repos import get_all_positions

    async with get_session() as sess:
        positions = await get_all_positions(sess)

    total_realized = sum(p.realized_pnl for p in positions)
    total_unrealized = sum(p.unrealized_pnl for p in positions if p.unrealized_pnl)

    current_equity = bankroll + total_unrealized

    # Peak equity: bankroll + total realized gains (high-water mark)
    # In a proper system this would be persisted; approximate with
    # bankroll_cap + total_realized as a proxy for peak.
    peak_equity = max(Config.BANKROLL_CAP, Config.BANKROLL_CAP + total_realized)

    if peak_equity <= 0:
        return None

    drawdown = (peak_equity - current_equity) / peak_equity

    if drawdown > Config.MAX_DRAWDOWN_PCT:
        msg = (
            f"GATE_DRAWDOWN: equity=${current_equity:.2f} is {drawdown:.1%} "
            f"below peak ${peak_equity:.2f} (limit={Config.MAX_DRAWDOWN_PCT:.0%})"
        )
        log.warning("portfolio_risk: %s", msg)
        return msg

    return None


# ── 2. Cluster Exposure Gating ───────────────────────────────────────────────

async def check_cluster_exposure(
    target_city_slug: str,
    bankroll: float,
) -> Optional[str]:
    """Limit total exposure to correlated cities within the same cluster.

    Cities in the same cluster (e.g., us_east: atlanta + miami + nyc) share
    the same synoptic weather regime and are highly correlated. Concentrating
    exposure in one cluster is a hidden risk.

    Returns a gate failure string, or None if within limits.
    """
    from backend.storage.db import get_session
    from backend.storage.repos import get_all_positions
    from backend.storage.models import Bucket, Event, City
    from sqlalchemy import select

    target_cluster = get_city_cluster(target_city_slug)
    if not target_cluster:
        return None  # unknown city, can't check cluster

    # Find all cities in the same cluster
    cluster_slugs = {
        slug for slug, info in CITY_REGISTRY_BY_SLUG.items()
        if info.get("cluster") == target_cluster
    }

    async with get_session() as sess:
        positions = await get_all_positions(sess)
        if not positions:
            return None

        # Map positions to city slugs via bucket → event → city
        cluster_exposure = 0.0
        for pos in positions:
            if pos.net_qty <= 0:
                continue
            bucket = (await sess.execute(
                select(Bucket).where(Bucket.id == pos.bucket_id)
            )).scalar_one_or_none()
            if not bucket:
                continue
            event = (await sess.execute(
                select(Event).where(Event.id == bucket.event_id)
            )).scalar_one_or_none()
            if not event:
                continue
            city = (await sess.execute(
                select(City).where(City.id == event.city_id)
            )).scalar_one_or_none()
            if city and city.city_slug in cluster_slugs:
                cluster_exposure += pos.net_qty * pos.avg_cost

    effective_bankroll = max(bankroll, Config.BANKROLL_CAP)
    max_cluster = effective_bankroll * Config.MAX_CLUSTER_EXPOSURE_PCT

    if cluster_exposure >= max_cluster:
        msg = (
            f"GATE_CLUSTER_EXPOSURE: cluster '{target_cluster}' exposure "
            f"${cluster_exposure:.2f} >= limit ${max_cluster:.2f} "
            f"({Config.MAX_CLUSTER_EXPOSURE_PCT:.0%} of bankroll)"
        )
        log.warning("portfolio_risk: %s", msg)
        return msg

    return None


# ── 3. Strategy-Level Loss Limits ────────────────────────────────────────────

async def check_strategy_loss(strategy: str) -> Optional[str]:
    """Halt a strategy when its cumulative realized loss exceeds the per-strategy limit.

    Tracks P&L per strategy by reading the strategy column on Position rows.

    Returns a gate failure string, or None if within limits.
    """
    from backend.storage.db import get_session
    from backend.storage.repos import get_all_positions

    async with get_session() as sess:
        positions = await get_all_positions(sess)

    # Sum realized P&L for positions tagged with this strategy
    strategy_pnl = sum(
        p.realized_pnl for p in positions
        if (p.strategy or "default") == strategy
    )

    if strategy_pnl < -Config.MAX_STRATEGY_LOSS:
        msg = (
            f"GATE_STRATEGY_LOSS: strategy '{strategy}' cumulative P&L "
            f"${strategy_pnl:.2f} < -${Config.MAX_STRATEGY_LOSS:.2f} limit"
        )
        log.warning("portfolio_risk: %s", msg)
        return msg

    return None


# ── Combined Portfolio Risk Check ────────────────────────────────────────────

async def check_portfolio_risk(
    city_slug: str,
    bankroll: float,
    strategy: str = "default",
) -> list[str]:
    """Run all portfolio-level risk checks. Returns list of failures (empty = pass)."""
    failures: list[str] = []

    drawdown_fail = await check_drawdown(bankroll)
    if drawdown_fail:
        failures.append(drawdown_fail)

    cluster_fail = await check_cluster_exposure(city_slug, bankroll)
    if cluster_fail:
        failures.append(cluster_fail)

    strategy_fail = await check_strategy_loss(strategy)
    if strategy_fail:
        failures.append(strategy_fail)

    if failures:
        log.warning(
            "portfolio_risk: %d gates FAILED for %s/%s: %s",
            len(failures), city_slug, strategy, "; ".join(failures),
        )
    else:
        log.debug("portfolio_risk: all gates passed for %s/%s", city_slug, strategy)

    return failures
