"""
Risk manager — Kelly-based position sizing with hard caps.

Design philosophy:
  - Half-Kelly to reduce variance and avoid ruin
  - Three-way minimum: Kelly, position cap, liquidity cap
  - Absolute minimum size of $1.00 (reject dust)
  - Hard bankroll ceiling of $10 for v1
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from backend.config import Config
from backend.strategy.kelly import calculate_kelly_fraction, calculate_expected_value

log = logging.getLogger(__name__)


def estimate_slippage(shares: float, ask_depth: float, base_bps: float = 50.0) -> float:
    """Linear market impact model for thin Polymarket orderbooks.

    Returns fractional price impact (0.01 = 1% slippage).
    For $2k-notional buckets, even a $1 order can move price 2-5%.

    Model: slippage = (shares / depth) * base_bps / 10000
    Capped at 5% to prevent extreme estimates on zero-depth books.
    """
    if ask_depth <= 0:
        return 0.03  # 3% default for unknown depth
    impact = (shares / ask_depth) * (base_bps / 10000.0)
    return min(impact, 0.05)  # cap at 5%


@dataclass
class SizingResult:
    size: float
    rejected: bool
    reject_reason: Optional[str]
    kelly_f: float
    kelly_size: float
    position_cap: float
    liquidity_cap: float
    bankroll_remaining: float


def compute_size(
    model_prob: float,
    limit_price: float,
    bankroll: float,
    open_exposure: float,
    ask_depth: float,
) -> SizingResult:
    """
    Compute position size using half-Kelly with hard caps.

    Args:
        model_prob: output from temperature model (probability this bucket resolves YES)
        limit_price: the price we plan to pay per share (≈ yes_ask)
        bankroll: total trading bankroll ($)
        open_exposure: cost basis of currently open positions ($)
        ask_depth: total shares available at ask (liquidity)

    Returns:
        SizingResult — check .rejected before using .size
    """
    # Cap bankroll to hard ceiling (safety: never trust passed-in bankroll > cap)
    effective_bankroll = min(bankroll, Config.BANKROLL_CAP)
    
    # Halve position size for thin orderbooks
    liquidity_value = ask_depth * limit_price
    if liquidity_value < Config.MIN_ORDERBOOK_DEPTH_DOLLARS:
        effective_bankroll *= 0.5
        
    bankroll_remaining = effective_bankroll - open_exposure

    if bankroll_remaining <= 0:
        return _rejected(f"bankroll_exhausted: remaining=${bankroll_remaining:.2f}", 0, 0, 0, bankroll_remaining)

    # ── Kelly fraction ────────────────────────────────────────────────────────
    kelly_f = calculate_kelly_fraction(
        model_prob=model_prob,
        yes_price=limit_price,
        fractional_kelly=Config.KELLY_FRACTION,
        max_position_size=Config.MAX_POSITION_PCT,
    )

    if kelly_f <= 0:
        return _rejected(
            f"negative_kelly: f={kelly_f:.4f} (no positive edge in sizing)",
            kelly_f, 0, 0, bankroll_remaining
        )

    # Note: calculate_kelly_fraction already applied Config.KELLY_FRACTION
    # and capped it at MAX_POSITION_PCT.
    kelly_size = kelly_f * effective_bankroll

    # ── Hard caps ────────────────────────────────────────────────────────────
    position_cap = effective_bankroll * Config.MAX_POSITION_PCT
    liquidity_cap = ask_depth * Config.MAX_LIQUIDITY_PCT * limit_price  # $ value

    final_size = min(kelly_size, position_cap, liquidity_cap, bankroll_remaining)

    # Convert from $ to shares at limit_price
    shares = math.floor((final_size / limit_price) * 100) / 100

    # Apply slippage estimate — deduct expected impact cost
    slippage_pct = estimate_slippage(shares, ask_depth)
    effective_price = limit_price * (1.0 + slippage_pct)
    dollar_cost = round(shares * effective_price, 2)

    if dollar_cost < 1.00:
        return _rejected(
            f"size_too_small: dollar_cost=${dollar_cost:.2f} < $1.00 minimum",
            kelly_f, kelly_size, position_cap, bankroll_remaining
        )

    log.info(
        "sizing: kelly_f=%.4f kelly_size=$%.2f position_cap=$%.2f "
        "liquidity_cap=$%.2f slippage=%.2f%% final_size=$%.2f (%.2f shares @ ${:.4f})",
        kelly_f, kelly_size, position_cap, liquidity_cap,
        slippage_pct * 100, dollar_cost, shares, limit_price,
    )

    return SizingResult(
        size=shares,
        rejected=False,
        reject_reason=None,
        kelly_f=round(kelly_f, 4),
        kelly_size=round(kelly_size, 2),
        position_cap=round(position_cap, 2),
        liquidity_cap=round(liquidity_cap, 2),
        bankroll_remaining=round(bankroll_remaining, 2),
    )


def _rejected(reason: str, kelly_f: float, kelly_size: float, position_cap: float, remaining: float) -> SizingResult:
    log.warning("sizing: REJECTED — %s", reason)
    return SizingResult(
        size=0.0,
        rejected=True,
        reject_reason=reason,
        kelly_f=round(kelly_f, 4),
        kelly_size=round(kelly_size, 2),
        position_cap=round(position_cap, 2),
        liquidity_cap=0.0,
        bankroll_remaining=round(remaining, 2),
    )
