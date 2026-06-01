"""
Risk manager — Kelly-based position sizing with hard caps.

Design philosophy:
  - Half-Kelly to reduce variance and avoid ruin
  - Three-way minimum: Kelly, position cap, liquidity cap
  - Absolute minimum notional from Polymarket CLOB (reject dust)
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

_SHARE_PRECISION = 100.0
_ROUNDING_TOLERANCE_DOLLARS = 0.01


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
    kelly_source: str = "single_probability"
    min_order_notional: float = 1.0
    liquidity_haircut: float = 1.0
    min_notional_bump: bool = False


def compute_size(
    model_prob: float,
    limit_price: float,
    bankroll: float,
    open_exposure: float,
    ask_depth: float,
    regime_multiplier: float = 1.0,
    kelly_fraction_override: Optional[float] = None,
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
    if not (0 < limit_price < 1):
        return _rejected(
            f"invalid_price: limit_price={limit_price}",
            0, 0, 0, 0,
        )

    # Cap bankroll to hard ceiling (safety: never trust passed-in bankroll > cap)
    effective_bankroll = min(bankroll, Config.BANKROLL_CAP)
    min_notional = max(0.0, float(getattr(Config, "MIN_ORDER_NOTIONAL_DOLLARS", 1.0)))

    bankroll_remaining = effective_bankroll - open_exposure

    if bankroll_remaining <= 0:
        return _rejected(
            f"bankroll_exhausted: remaining=${bankroll_remaining:.2f}",
            0, 0, 0, bankroll_remaining, min_order_notional=min_notional,
        )

    # Hard caps that should stay anchored to the actual bankroll. The liquidity
    # haircut below scales Kelly desire only; otherwise a $10 bankroll with a
    # 10% position cap can be silently shrunk below Polymarket's $1 floor.
    position_cap = effective_bankroll * Config.MAX_POSITION_PCT
    liquidity_cap = max(0.0, ask_depth * Config.MAX_LIQUIDITY_PCT * limit_price)
    liquidity_value = ask_depth * limit_price
    needed_depth_cap = max(min_notional, position_cap)
    liquidity_haircut = 1.0
    if (
        Config.MIN_ORDERBOOK_DEPTH_DOLLARS > 0
        and liquidity_value < Config.MIN_ORDERBOOK_DEPTH_DOLLARS
        and liquidity_cap + _ROUNDING_TOLERANCE_DOLLARS < needed_depth_cap
    ):
        liquidity_haircut = 0.5

    # ── Kelly fraction ────────────────────────────────────────────────────────
    # Phase C3 — regime_multiplier scales the base Kelly down in volatile
    # regimes (front passing, ensemble disagreement growing, active wx).
    # Clamped to [0.5, 1.0] in regime_kelly_multiplier; defensive clamp here too.
    rm = max(0.5, min(1.0, regime_multiplier))
    if kelly_fraction_override is not None:
        kelly_f = max(
            0.0,
            min(Config.MAX_POSITION_PCT, float(kelly_fraction_override) * rm),
        )
        kelly_source = "posterior_bma_component_median"
    else:
        kelly_f = calculate_kelly_fraction(
            model_prob=model_prob,
            yes_price=limit_price,
            fractional_kelly=Config.KELLY_FRACTION * rm,
            max_position_size=Config.MAX_POSITION_PCT,
        )
        kelly_source = "single_probability"

    if kelly_f <= 0:
        return _rejected(
            f"negative_kelly: f={kelly_f:.4f} (no positive edge in sizing)",
            kelly_f, 0, position_cap, bankroll_remaining,
            kelly_source=kelly_source,
            liquidity_cap=liquidity_cap,
            min_order_notional=min_notional,
            liquidity_haircut=liquidity_haircut,
        )

    # Note: calculate_kelly_fraction already applied Config.KELLY_FRACTION
    # and capped it at MAX_POSITION_PCT.
    kelly_size = kelly_f * effective_bankroll * liquidity_haircut

    final_size = min(kelly_size, position_cap, liquidity_cap, bankroll_remaining)
    min_notional_bump = False

    if min_notional > 0 and final_size + _ROUNDING_TOLERANCE_DOLLARS < min_notional:
        blockers = []
        if position_cap + _ROUNDING_TOLERANCE_DOLLARS < min_notional:
            blockers.append(f"position_cap=${position_cap:.2f}")
        if liquidity_cap + _ROUNDING_TOLERANCE_DOLLARS < min_notional:
            blockers.append(f"liquidity_cap=${liquidity_cap:.2f}")
        if bankroll_remaining + _ROUNDING_TOLERANCE_DOLLARS < min_notional:
            blockers.append(f"bankroll_remaining=${bankroll_remaining:.2f}")

        if blockers:
            return _rejected(
                "min_notional_blocked: "
                f"need ${min_notional:.2f}, " + ", ".join(blockers),
                kelly_f, kelly_size, position_cap, bankroll_remaining,
                kelly_source=kelly_source,
                liquidity_cap=liquidity_cap,
                min_order_notional=min_notional,
                liquidity_haircut=liquidity_haircut,
            )

        max_bump_multiple = max(
            1.0,
            float(getattr(Config, "MIN_NOTIONAL_BUMP_MAX_KELLY_MULTIPLE", 3.0)),
        )
        if kelly_size <= 0 or min_notional > kelly_size * max_bump_multiple + _ROUNDING_TOLERANCE_DOLLARS:
            overbet_multiple = math.inf if kelly_size <= 0 else min_notional / kelly_size
            return _rejected(
                "min_notional_overbet: "
                f"need ${min_notional:.2f}, kelly_size=${kelly_size:.2f}, "
                f"multiple={overbet_multiple:.2f}x > max={max_bump_multiple:.2f}x",
                kelly_f, kelly_size, position_cap, bankroll_remaining,
                kelly_source=kelly_source,
                liquidity_cap=liquidity_cap,
                min_order_notional=min_notional,
                liquidity_haircut=liquidity_haircut,
            )

        final_size = min_notional
        min_notional_bump = True

    # Convert from $ to shares at limit_price
    raw_shares = final_size / limit_price
    if min_notional > 0 and final_size <= min_notional + _ROUNDING_TOLERANCE_DOLLARS:
        shares = math.ceil(raw_shares * _SHARE_PRECISION) / _SHARE_PRECISION
    else:
        shares = math.floor(raw_shares * _SHARE_PRECISION) / _SHARE_PRECISION

    order_notional = shares * limit_price

    # Apply slippage estimate — deduct expected impact cost
    slippage_pct = estimate_slippage(shares, ask_depth)
    effective_price = limit_price * (1.0 + slippage_pct)
    dollar_cost = round(shares * effective_price, 2)

    if min_notional > 0 and order_notional + 1e-9 < min_notional:
        return _rejected(
            f"size_too_small: order_notional=${order_notional:.2f} < ${min_notional:.2f} minimum",
            kelly_f, kelly_size, position_cap, bankroll_remaining,
            kelly_source=kelly_source,
            liquidity_cap=liquidity_cap,
            min_order_notional=min_notional,
            liquidity_haircut=liquidity_haircut,
            min_notional_bump=min_notional_bump,
        )

    log.info(
        "sizing: kelly_f=%.4f kelly_size=$%.2f position_cap=$%.2f "
        "liquidity_cap=$%.2f liquidity_haircut=%.2f min_bump=%s "
        "slippage=%.2f%% final_size=$%.2f (%.2f shares @ ${:.4f})",
        kelly_f, kelly_size, position_cap, liquidity_cap,
        liquidity_haircut, min_notional_bump,
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
        kelly_source=kelly_source,
        min_order_notional=round(min_notional, 2),
        liquidity_haircut=round(liquidity_haircut, 2),
        min_notional_bump=min_notional_bump,
    )


def _rejected(
    reason: str,
    kelly_f: float,
    kelly_size: float,
    position_cap: float,
    remaining: float,
    kelly_source: str = "single_probability",
    liquidity_cap: float = 0.0,
    min_order_notional: float = 1.0,
    liquidity_haircut: float = 1.0,
    min_notional_bump: bool = False,
) -> SizingResult:
    log.warning("sizing: REJECTED — %s", reason)
    return SizingResult(
        size=0.0,
        rejected=True,
        reject_reason=reason,
        kelly_f=round(kelly_f, 4),
        kelly_size=round(kelly_size, 2),
        position_cap=round(position_cap, 2),
        liquidity_cap=round(liquidity_cap, 2),
        bankroll_remaining=round(remaining, 2),
        kelly_source=kelly_source,
        min_order_notional=round(min_order_notional, 2),
        liquidity_haircut=round(liquidity_haircut, 2),
        min_notional_bump=min_notional_bump,
    )
