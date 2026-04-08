"""
Per-city forecast calibration — EWMA bias correction.

Learns per-city biases by comparing forecasts against realized daily highs
after settlement. v1 uses EWMA bias correction with fixed weights.
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.storage.db import get_session
from backend.storage.repos import get_calibration, upsert_calibration

log = logging.getLogger(__name__)

# EWMA learning rate for bias corrections
_ALPHA = 0.3


def get_calibration_dict(city_id: int) -> dict:
    """Synchronous helper for use in background jobs — returns default if not set."""
    return {
        "bias_nws": 0.0,
        "bias_wu_daily": 0.0,
        "bias_wu_hourly": 0.0,
        "bias_hrrr": 0.0,
        "bias_nbm": 0.0,
        "weight_nws": 1 / 3,
        "weight_wu_daily": 1 / 3,
        "weight_wu_hourly": 1 / 3,
        "weight_hrrr": 0.5,
        "weight_nbm": 0.2,
    }


async def get_calibration_async(city_id: int) -> dict:
    """Async version — reads from DB."""
    async with get_session() as sess:
        cal = await get_calibration(sess, city_id)
    if cal is None:
        return get_calibration_dict(city_id)
    return {
        "bias_nws": cal.bias_nws,
        "bias_wu_daily": cal.bias_wu_daily,
        "bias_wu_hourly": cal.bias_wu_hourly,
        "bias_hrrr": v if (v := getattr(cal, "bias_hrrr", None)) is not None else 0.0,
        "bias_nbm": v if (v := getattr(cal, "bias_nbm", None)) is not None else 0.0,
        "weight_nws": cal.weight_nws,
        "weight_wu_daily": cal.weight_wu_daily,
        "weight_wu_hourly": cal.weight_wu_hourly,
        "weight_hrrr": v if (v := getattr(cal, "weight_hrrr", None)) is not None else 0.5,
        "weight_nbm": v if (v := getattr(cal, "weight_nbm", None)) is not None else 0.2,
    }


async def update_calibration(
    city_id: int,
    realized_high_f: float,
    nws_forecast: Optional[float],
    wu_daily_forecast: Optional[float],
    wu_hourly_forecast: Optional[float],
    hrrr_forecast: Optional[float] = None,
    nbm_forecast: Optional[float] = None,
) -> None:
    """
    Update per-city bias corrections after settlement.

    Applies EWMA update: new_bias = alpha * error + (1-alpha) * old_bias
    where error = realized_high - (forecast + old_bias)
    """
    async with get_session() as sess:
        cal = await get_calibration(sess, city_id)

    if cal is None:
        # Initialize with defaults
        async with get_session() as sess:
            cal = await upsert_calibration(sess, city_id=city_id)

    updates: dict = {}

    if nws_forecast is not None:
        error = realized_high_f - (nws_forecast + cal.bias_nws)
        updates["bias_nws"] = _ALPHA * error + (1 - _ALPHA) * cal.bias_nws
        log.info("calibration: city_id=%d nws_error=%.2f new_bias_nws=%.3f", city_id, error, updates["bias_nws"])

    if wu_daily_forecast is not None:
        error = realized_high_f - (wu_daily_forecast + cal.bias_wu_daily)
        updates["bias_wu_daily"] = _ALPHA * error + (1 - _ALPHA) * cal.bias_wu_daily
        log.info("calibration: city_id=%d wu_daily_error=%.2f new_bias_wu_daily=%.3f", city_id, error, updates["bias_wu_daily"])

    if wu_hourly_forecast is not None:
        error = realized_high_f - (wu_hourly_forecast + cal.bias_wu_hourly)
        updates["bias_wu_hourly"] = _ALPHA * error + (1 - _ALPHA) * cal.bias_wu_hourly
        log.info("calibration: city_id=%d wu_hourly_error=%.2f new_bias_wu_hourly=%.3f", city_id, error, updates["bias_wu_hourly"])

    if hrrr_forecast is not None:
        old_bias = getattr(cal, "bias_hrrr", 0.0) or 0.0
        error = realized_high_f - (hrrr_forecast + old_bias)
        updates["bias_hrrr"] = _ALPHA * error + (1 - _ALPHA) * old_bias
        log.info("calibration: city_id=%d hrrr_error=%.2f new_bias_hrrr=%.3f", city_id, error, updates["bias_hrrr"])

    if nbm_forecast is not None:
        old_bias = getattr(cal, "bias_nbm", 0.0) or 0.0
        error = realized_high_f - (nbm_forecast + old_bias)
        updates["bias_nbm"] = _ALPHA * error + (1 - _ALPHA) * old_bias
        log.info("calibration: city_id=%d nbm_error=%.2f new_bias_nbm=%.3f", city_id, error, updates["bias_nbm"])

    updates["n_samples"] = (cal.n_samples or 0) + 1
    updates["last_realized_high"] = realized_high_f

    async with get_session() as sess:
        await upsert_calibration(sess, city_id=city_id, **updates)
