"""Phase C3 — atmospheric-regime detector + Kelly multiplier.

Composite indicator from short-horizon volatility signals:
  - ensemble disagreement (current spread)
  - spread *growth* across the last 3 model cycles (front passing through)
  - pressure tendency from METAR (synoptic forcing)
  - precipitation flag (active wx event)

Output: a continuous regime_score in [0,1] (0=calm, 1=volatile) plus a
discrete `RegimeLabel` and an observability `components` dict.

The score feeds `regime_kelly_multiplier` which scales the base Kelly
fraction down in volatile conditions. Quiet days size larger; front-passage
days size smaller — same edge, less variance contribution.

Pure functions only. No DB access. The caller fetches inputs (METAR
extended, recent ModelSnapshot spreads) and supplies them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RegimeLabel(str, Enum):
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"


@dataclass
class RegimeResult:
    label: RegimeLabel
    score: float  # [0,1] — higher = more volatile
    components: dict = field(default_factory=dict)


# Component weights — must sum to 1.0. Disagreement and growth dominate
# because they're directly observable in our own data; pressure_tendency
# and precip add corroborating wx context but are noisier.
_W_SPREAD = 0.35
_W_GROWTH = 0.30
_W_PRESSURE = 0.20
_W_PRECIP = 0.15

# Regime label thresholds.
_CALM_BELOW = 0.25
_VOLATILE_AT_OR_ABOVE = 0.65

# Kelly multiplier endpoints.
_KELLY_MULT_CALM = 1.0
_KELLY_MULT_VOLATILE = 0.5


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def _spread_score(current_spread_f: float) -> float:
    """spread <1°F → 0.0 (calm); spread >5°F → 1.0 (volatile); linear in between."""
    if current_spread_f is None:
        return 0.0
    return _clamp01((current_spread_f - 1.0) / 4.0)


def _growth_score(current_spread_f: float, historical_spreads: list[float]) -> float:
    """Spread growth vs the median of the last few cycles.

    growth = current - median(history). 0°F or negative → 0.0; ≥2°F → 1.0.
    Returns 0.0 when we don't have enough history to judge.
    """
    if current_spread_f is None or not historical_spreads:
        return 0.0
    sorted_h = sorted(historical_spreads)
    n = len(sorted_h)
    median = sorted_h[n // 2] if n % 2 == 1 else 0.5 * (sorted_h[n // 2 - 1] + sorted_h[n // 2])
    growth = current_spread_f - median
    return _clamp01(growth / 2.0)


def _pressure_score(pressure_tendency_inhg: Optional[float]) -> float:
    """|tendency| ≥ 0.06 inHg/3h (≈ 2 hPa/3h, NWS definition of "rapid") → 1.0."""
    if pressure_tendency_inhg is None:
        return 0.0
    return _clamp01(abs(pressure_tendency_inhg) / 0.06)


def _precip_score(has_precip: bool) -> float:
    return 1.0 if has_precip else 0.0


def detect_regime(
    *,
    current_spread_f: Optional[float],
    historical_spreads_f: Optional[list[float]] = None,
    pressure_tendency_inhg: Optional[float] = None,
    has_precip: bool = False,
) -> RegimeResult:
    """Composite regime score and label from short-horizon volatility signals.

    All inputs are optional — missing inputs zero-weight that component but
    don't poison the rest of the score.
    """
    spread_s = _spread_score(current_spread_f if current_spread_f is not None else 0.0)
    growth_s = _growth_score(
        current_spread_f if current_spread_f is not None else 0.0,
        historical_spreads_f or [],
    )
    pressure_s = _pressure_score(pressure_tendency_inhg)
    precip_s = _precip_score(has_precip)

    score = (
        _W_SPREAD * spread_s
        + _W_GROWTH * growth_s
        + _W_PRESSURE * pressure_s
        + _W_PRECIP * precip_s
    )
    score = _clamp01(score)

    if score < _CALM_BELOW:
        label = RegimeLabel.CALM
    elif score >= _VOLATILE_AT_OR_ABOVE:
        label = RegimeLabel.VOLATILE
    else:
        label = RegimeLabel.NORMAL

    return RegimeResult(
        label=label,
        score=round(score, 3),
        components={
            "spread": round(spread_s, 3),
            "growth": round(growth_s, 3),
            "pressure": round(pressure_s, 3),
            "precip": round(precip_s, 3),
            "current_spread_f": current_spread_f,
            "historical_n": len(historical_spreads_f or []),
        },
    )


def regime_kelly_multiplier(score: float) -> float:
    """Map score [0,1] → Kelly multiplier [1.0, 0.5] linearly.

    Calm regime → full base Kelly. Volatile regime → halved Kelly.
    Use as `effective_kelly = base_kelly * regime_kelly_multiplier(score)`.
    """
    s = _clamp01(score)
    return _KELLY_MULT_CALM + (_KELLY_MULT_VOLATILE - _KELLY_MULT_CALM) * s
