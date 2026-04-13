"""
Backtest performance metrics — Sharpe, Brier, drawdown, reliability.

All functions are pure (no DB access) so they're easy to test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    brier_score: float = 0.0
    brier_skill_score: float = 0.0
    avg_true_edge: float = 0.0
    profit_factor: float = 0.0
    avg_hold_time_hours: float = 0.0


def compute_sharpe(daily_returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns.

    Uses sqrt(365) for crypto/prediction-market style (trades every day).
    """
    if len(daily_returns) < 2:
        return 0.0
    mean_r = sum(daily_returns) / len(daily_returns) - risk_free_rate
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean_r / std) * math.sqrt(365)


def compute_brier(
    predictions: list[tuple[float, int]],
) -> tuple[float, float]:
    """Brier Score and Brier Skill Score.

    Args:
        predictions: list of (model_prob, actual_outcome) where outcome is 0 or 1.

    Returns:
        (brier_score, brier_skill_score)
        BS  = (1/N) * sum((prob_i - outcome_i)^2)  — lower is better
        BSS = 1 - BS / BS_climatology               — higher is better, 1.0 = perfect
    """
    if not predictions:
        return 0.0, 0.0

    n = len(predictions)
    bs = sum((p - o) ** 2 for p, o in predictions) / n

    # Climatological Brier: using base rate as constant prediction
    base_rate = sum(o for _, o in predictions) / n
    bs_clim = base_rate * (1.0 - base_rate)

    bss = 1.0 - (bs / bs_clim) if bs_clim > 0 else 0.0

    return round(bs, 6), round(bss, 4)


def compute_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Max drawdown in absolute $ and as a percentage.

    Args:
        equity_curve: list of portfolio values over time.

    Returns:
        (max_drawdown_dollars, max_drawdown_pct)
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        dd_pct = dd / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    return round(max_dd, 4), round(max_dd_pct, 4)


def compute_reliability_bins(
    predictions: list[tuple[float, int]],
    n_bins: int = 10,
) -> list[dict]:
    """Reliability diagram data: bin predictions by decile.

    Returns list of dicts:
        {bin_center, mean_predicted, observed_frequency, count}
    """
    bins: list[dict] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        center = (lo + hi) / 2.0
        in_bin = [(p, o) for p, o in predictions if lo <= p < hi or (i == n_bins - 1 and p == hi)]
        if in_bin:
            mean_pred = sum(p for p, _ in in_bin) / len(in_bin)
            obs_freq = sum(o for _, o in in_bin) / len(in_bin)
        else:
            mean_pred = center
            obs_freq = 0.0
        bins.append({
            "bin_center": round(center, 2),
            "mean_predicted": round(mean_pred, 4),
            "observed_frequency": round(obs_freq, 4),
            "count": len(in_bin),
        })
    return bins


def compute_profit_factor(trades_pnl: list[float]) -> float:
    """Gross wins / gross losses. Returns inf if no losses."""
    gross_win = sum(p for p in trades_pnl if p > 0)
    gross_loss = abs(sum(p for p in trades_pnl if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return round(gross_win / gross_loss, 4)


def build_equity_curve(
    daily_pnl: dict[str, float],
    initial_bankroll: float,
) -> list[dict]:
    """Build equity curve from daily P&L dict.

    Args:
        daily_pnl: {date_str: pnl_dollars} sorted by date.
        initial_bankroll: starting capital.

    Returns:
        List of {date, equity, drawdown} dicts.
    """
    curve: list[dict] = []
    equity = initial_bankroll
    peak = equity

    for date_str in sorted(daily_pnl.keys()):
        equity += daily_pnl[date_str]
        if equity > peak:
            peak = equity
        dd = peak - equity
        curve.append({
            "date": date_str,
            "equity": round(equity, 4),
            "drawdown": round(dd, 4),
        })

    return curve
