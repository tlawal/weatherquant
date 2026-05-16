"""Observation-proximity exit helpers.

Pure decision helpers for the OBS_PROXIMITY exit layer.  They intentionally
avoid database access so the live exit engine, dashboard, tests, and backtest
can share the same boundary and orderbook logic.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

from backend.modeling.settlement import canonical_bucket_ranges, find_bucket_idx_for_value


BucketLike = dict[str, Any] | object


def _get(bucket: BucketLike, key: str, default=None):
    if isinstance(bucket, dict):
        return bucket.get(key, default)
    return getattr(bucket, key, default)


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_observation_minutes(minutes: Sequence[int] | str | None) -> list[int]:
    if minutes is None:
        return []
    if isinstance(minutes, str):
        import json

        try:
            parsed = json.loads(minutes)
        except Exception:
            return []
        minutes = parsed if isinstance(parsed, list) else []
    out: list[int] = []
    for raw in minutes:
        try:
            minute = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= minute <= 59 and minute not in out:
            out.append(minute)
    return sorted(out)


def next_observation_time(now_local: datetime, observation_minutes: Sequence[int] | str | None) -> Optional[datetime]:
    """Return the next scheduled station observation time in local time."""
    minutes = normalize_observation_minutes(observation_minutes)
    if not minutes:
        return None

    candidates: list[datetime] = []
    base = now_local.replace(second=0, microsecond=0)
    for hour_offset in range(0, 25):
        hour_base = base + timedelta(hours=hour_offset)
        for minute in minutes:
            candidate = hour_base.replace(minute=minute)
            if candidate >= now_local:
                candidates.append(candidate)
    return min(candidates) if candidates else None


def orderbook_depth_usd(yes_bid: float | None, yes_bid_depth: float | None) -> Optional[float]:
    bid = _float_or_none(yes_bid)
    depth = _float_or_none(yes_bid_depth)
    if bid is None or depth is None:
        return None
    return max(0.0, bid) * max(0.0, depth)


def orderbook_imbalance(yes_bid_depth: float | None, yes_ask_depth: float | None) -> Optional[float]:
    bid_depth = max(0.0, _float_or_none(yes_bid_depth) or 0.0)
    ask_depth = max(0.0, _float_or_none(yes_ask_depth) or 0.0)
    total = bid_depth + ask_depth
    if total <= 0:
        return None
    return max(bid_depth, ask_depth) / total


def _bucket_ranges(bucket_specs: Sequence[BucketLike]) -> list[tuple[Optional[float], Optional[float]]]:
    return [
        (_float_or_none(_get(b, "low_f")), _float_or_none(_get(b, "high_f")))
        for b in sorted(bucket_specs, key=lambda b: int(_get(b, "bucket_idx", _get(b, "idx", 0)) or 0))
    ]


def _bucket_label(bucket_specs: Sequence[BucketLike], bucket_idx: Optional[int]) -> Optional[str]:
    if bucket_idx is None:
        return None
    for bucket in bucket_specs:
        idx = _get(bucket, "bucket_idx", _get(bucket, "idx"))
        if idx == bucket_idx:
            label = _get(bucket, "label")
            return str(label) if label is not None else f"Bucket {bucket_idx}"
    return f"Bucket {bucket_idx}"


def nearest_boundary_distance(
    bucket_specs: Sequence[BucketLike],
    temp_f: float | None,
) -> Optional[float]:
    temp = _float_or_none(temp_f)
    if temp is None:
        return None
    boundaries: set[float] = set()
    for low_f, high_f in canonical_bucket_ranges(_bucket_ranges(bucket_specs)):
        if low_f is not None:
            boundaries.add(float(low_f))
        if high_f is not None:
            boundaries.add(float(high_f))
    if not boundaries:
        return None
    return min(abs(temp - b) for b in boundaries)


def sensitivity_badge(
    boundary_distance_f: float | None,
    threshold_f: float,
    shift_changes_bucket: bool = False,
) -> str:
    if shift_changes_bucket:
        return "HIGH"
    if boundary_distance_f is None:
        return "LOW"
    threshold = max(0.01, float(threshold_f))
    if boundary_distance_f <= threshold:
        return "HIGH"
    if boundary_distance_f <= threshold * 2.0:
        return "MEDIUM"
    return "LOW"


def build_obs_proximity_status(
    *,
    city_slug: str,
    station_id: str | None,
    now_local: datetime,
    observation_minutes: Sequence[int] | str | None,
    bucket_specs: Sequence[BucketLike],
    reference_temp_f: float | None,
    held_bucket_idx: int | None = None,
    enabled: bool,
    is_us: bool,
    window_minutes: int,
    temp_sensitivity_threshold_f: float,
) -> dict:
    """Build the shared observation-proximity state payload."""
    reference_temp = _float_or_none(reference_temp_f)
    next_obs = next_observation_time(now_local, observation_minutes)
    minutes_to_next = (
        (next_obs - now_local).total_seconds() / 60.0
        if next_obs is not None
        else None
    )
    canonical = canonical_bucket_ranges(_bucket_ranges(bucket_specs)) if bucket_specs else []
    base_bucket_idx = find_bucket_idx_for_value(canonical, reference_temp)
    plus_bucket_idx = find_bucket_idx_for_value(canonical, reference_temp + 1.0) if reference_temp is not None else None
    minus_bucket_idx = find_bucket_idx_for_value(canonical, reference_temp - 1.0) if reference_temp is not None else None
    boundary_distance = nearest_boundary_distance(bucket_specs, reference_temp)
    shift_changes_bucket = (
        base_bucket_idx is not None
        and (plus_bucket_idx != base_bucket_idx or minus_bucket_idx != base_bucket_idx)
    )
    within_boundary = (
        boundary_distance is not None
        and boundary_distance <= temp_sensitivity_threshold_f
    )
    fragile = bool(within_boundary or shift_changes_bucket)
    held_matches_reference = (
        held_bucket_idx is None
        or base_bucket_idx is None
        or held_bucket_idx == base_bucket_idx
    )
    in_window = (
        minutes_to_next is not None
        and 0.0 <= minutes_to_next <= float(window_minutes)
    )
    badge = sensitivity_badge(boundary_distance, temp_sensitivity_threshold_f, shift_changes_bucket)

    return {
        "city": city_slug,
        "station": station_id,
        "enabled": bool(enabled),
        "is_us": bool(is_us),
        "observation_minutes": normalize_observation_minutes(observation_minutes),
        "next_observation_time": next_obs.isoformat() if next_obs else None,
        "next_observation_label": next_obs.strftime("%-I:%M %p %Z") if next_obs else None,
        "minutes_to_next_obs": round(minutes_to_next, 2) if minutes_to_next is not None else None,
        "countdown_label": f"{max(0, int(round(minutes_to_next)))}m" if minutes_to_next is not None else None,
        "current_temp": reference_temp,
        "current_bucket": _bucket_label(bucket_specs, held_bucket_idx),
        "current_bucket_idx": held_bucket_idx,
        "reference_bucket_idx": base_bucket_idx,
        "reference_bucket": _bucket_label(bucket_specs, base_bucket_idx),
        "plus_1f_bucket_idx": plus_bucket_idx,
        "plus_1f_bucket": _bucket_label(bucket_specs, plus_bucket_idx),
        "minus_1f_bucket_idx": minus_bucket_idx,
        "minus_1f_bucket": _bucket_label(bucket_specs, minus_bucket_idx),
        "boundary_distance_f": round(boundary_distance, 3) if boundary_distance is not None else None,
        "within_boundary_threshold": within_boundary,
        "shift_changes_bucket": shift_changes_bucket,
        "held_matches_reference_bucket": held_matches_reference,
        "fragile": fragile and held_matches_reference,
        "sensitivity_badge": badge,
        "in_window": in_window,
        "armed": bool(enabled and is_us and in_window and fragile and held_matches_reference),
    }


def evaluate_obs_proximity_exit(
    *,
    city_slug: str,
    station_id: str | None,
    now_local: datetime,
    observation_minutes: Sequence[int] | str | None,
    bucket_specs: Sequence[BucketLike],
    held_bucket_idx: int,
    reference_temp_f: float | None,
    yes_bid: float | None,
    yes_ask: float | None,
    yes_bid_depth: float | None,
    yes_ask_depth: float | None,
    net_pnl_per_share: float,
    current_edge: float | None,
    enabled: bool,
    is_us: bool,
    window_minutes: int,
    temp_sensitivity_threshold_f: float,
    min_profit_cents: float,
    min_depth_usd: float,
    max_orderbook_imbalance: float,
    cooldown_active: bool,
) -> dict:
    """Return a complete OBS_PROXIMITY exit decision payload."""
    status = build_obs_proximity_status(
        city_slug=city_slug,
        station_id=station_id,
        now_local=now_local,
        observation_minutes=observation_minutes,
        bucket_specs=bucket_specs,
        reference_temp_f=reference_temp_f,
        held_bucket_idx=held_bucket_idx,
        enabled=enabled,
        is_us=is_us,
        window_minutes=window_minutes,
        temp_sensitivity_threshold_f=temp_sensitivity_threshold_f,
    )
    depth_usd = orderbook_depth_usd(yes_bid, yes_bid_depth)
    imbalance = orderbook_imbalance(yes_bid_depth, yes_ask_depth)
    profit_cents = float(net_pnl_per_share) * 100.0

    skip_reason = None
    if not enabled:
        skip_reason = "disabled"
    elif not is_us:
        skip_reason = "non_us_city"
    elif not station_id:
        skip_reason = "missing_station"
    elif not status["observation_minutes"]:
        skip_reason = "missing_station_pattern"
    elif not status["in_window"]:
        skip_reason = "outside_observation_window"
    elif status["current_temp"] is None:
        skip_reason = "missing_current_temp_or_nowcast"
    elif not status["held_matches_reference_bucket"]:
        skip_reason = "held_bucket_not_reference_bucket"
    elif not status["fragile"]:
        skip_reason = "not_boundary_fragile"
    elif profit_cents < float(min_profit_cents):
        skip_reason = "profit_below_min"
    elif depth_usd is None or depth_usd < float(min_depth_usd):
        skip_reason = "bid_depth_below_min"
    elif imbalance is None or imbalance > float(max_orderbook_imbalance):
        skip_reason = "orderbook_imbalance_too_high"
    elif cooldown_active:
        skip_reason = "cooldown_active"

    action = "EXIT" if skip_reason is None else "SKIP"
    return {
        **status,
        "yes_bid": _float_or_none(yes_bid),
        "yes_ask": _float_or_none(yes_ask),
        "yes_bid_depth": _float_or_none(yes_bid_depth),
        "yes_ask_depth": _float_or_none(yes_ask_depth),
        "orderbook_depth_usd": round(depth_usd, 2) if depth_usd is not None else None,
        "imbalance": round(imbalance, 4) if imbalance is not None else None,
        "current_edge": current_edge,
        "mark_to_market_profit_cents": round(profit_cents, 3),
        "min_profit_cents": float(min_profit_cents),
        "min_depth_usd": float(min_depth_usd),
        "max_orderbook_imbalance": float(max_orderbook_imbalance),
        "cooldown_active": bool(cooldown_active),
        "final_action": action,
        "skip_reason": skip_reason,
    }
