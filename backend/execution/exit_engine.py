"""
Exit Engine — manages risk and takes profit for open positions.

Interval: 300 seconds (5 min)
5-level cascade:
    1. EMERGENCY: METAR obs contradicts bucket by >= 3°F (Market sell)
    1b. EDGE_DECAY: ev_at_bid <= EDGE_DECAY_THRESHOLD for EDGE_DECAY_DEBOUNCE_RUNS
        consecutive runs, plus a material deterioration from the stored entry
        EV baseline and model/source thesis (Limit sell at bid). Fires before
        URGENT so EV-based exits take priority over structural-shift exits.
    2. URGENT: Model consensus shifted to different bucket (Limit sell bid - 1c)
       — debounced: requires CONSENSUS_DEBOUNCE_RUNS consecutive shifts
       — spread-guarded: suppressed when spread > URGENT_EXIT_MAX_SPREAD
    3. PROFIT: Quick Flip! Position is at entry + QUICK_FLIP_TARGET (Limit sell at bid)
    4. EXPIRY: hold likely winners to redeem or passively sell near par; risk-exit
       ambiguous/losing buckets with a small discount only.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, delete, desc

from backend.config import Config
from backend.storage.db import get_session
from backend.storage.repos import (
    append_audit,
    get_all_positions,
    get_buckets_for_event,
    get_city_by_slug,
    get_open_orders,
    get_station_profile,
)
from backend.engine.signal_engine import run_signal_engine, BucketSignal
from backend.execution.trader import execute_signal, normalize_limit_price_for_clob
from backend.execution.obs_proximity import evaluate_obs_proximity_exit, normalize_observation_minutes
from backend.tz_utils import city_local_now

log = logging.getLogger(__name__)

# Polymarket taker fee roughly 2%
FEE_RATE = 0.02


async def _emit_exit_event(
    pos,
    signal: BucketSignal,
    cascade: dict,
    *,
    shares_exited: float,
    execution_status: str,
    error: Optional[str] = None,
) -> None:
    """Phase B4 — write a structured ExitEvent row for every cascade decision.

    Captures the full pre-trade context (EV, edge, bid/ask, cascade dict) so
    we can replay "what did exit_engine see" without parsing free-text logs.
    """
    import json as _json
    from backend.storage.models import ExitEvent

    payload = {
        "cascade": cascade,
        "execution_status": execution_status,
        **({"error": error} if error else {}),
        "city_slug": signal.city_slug,
        "bucket_label": signal.label,
    }
    try:
        async with get_session() as sess:
            sess.add(
                ExitEvent(
                    position_id=getattr(pos, "id", 0) or 0,
                    bucket_id=pos.bucket_id,
                    trigger_level=cascade["level"],
                    trigger_reason=cascade["reason"],
                    ev_at_bid_pre=signal.ev_at_bid,
                    ev_at_bid_post=None,
                    true_edge_pre=getattr(signal, "true_edge", None),
                    true_edge_post=None,
                    market_bid=signal.yes_bid,
                    market_ask=signal.yes_ask,
                    shares_exited=float(shares_exited),
                    shares_remaining=float(max(0.0, (pos.net_qty or 0.0) - shares_exited)),
                    model_snapshot_id=None,
                    reason_json=_json.dumps(payload, default=str),
                )
            )
            await sess.commit()
    except Exception:
        log.exception("exit_engine: failed to emit ExitEvent for bucket %s", pos.bucket_id)

# ── DB-backed consensus history (survives deploys) ──────────────────────────
# In-memory cache is populated from DB on first call and kept in sync.
_consensus_cache: dict[int, list[int]] = {}
_consensus_cache_loaded = False

# ── DB-backed EV history (Phase A2 — EDGE_DECAY exit gate) ───────────────────
# Per-bucket recent ev_at_bid values. Same survives-deploys pattern as consensus.
_ev_cache: dict[int, list[float]] = {}
_ev_cache_loaded = False

# OBS_PROXIMITY debounce. This is intentionally in-memory and best-effort:
# pending SELL-order checks still prevent duplicate live orders, while this
# blocks repeated OBS decisions between scheduler cycles without a schema change.
_obs_exit_cache: dict[int, datetime] = {}


async def _load_consensus_cache() -> None:
    """Warm the in-memory cache from ConsensusHistory rows."""
    global _consensus_cache, _consensus_cache_loaded
    if _consensus_cache_loaded:
        return
    from backend.storage.models import ConsensusHistory
    async with get_session() as sess:
        rows = (await sess.execute(
            select(ConsensusHistory)
            .order_by(ConsensusHistory.recorded_at.asc())
            .limit(500)  # cap memory
        )).scalars().all()
    cache: dict[int, list[int]] = {}
    for r in rows:
        cache.setdefault(r.event_id, []).append(r.bucket_id)
    _consensus_cache = cache
    _consensus_cache_loaded = True
    log.info("consensus_cache: loaded %d events, %d total rows from DB",
             len(cache), sum(len(v) for v in cache.values()))


async def _record_consensus(event_id: int, bucket_id: int) -> None:
    """Write consensus to DB and update in-memory cache."""
    from backend.storage.models import ConsensusHistory
    hist = _consensus_cache.setdefault(event_id, [])
    hist.append(bucket_id)
    if len(hist) > 10:
        hist.pop(0)
    async with get_session() as sess:
        sess.add(ConsensusHistory(
            event_id=event_id,
            bucket_id=bucket_id,
            recorded_at=datetime.now(timezone.utc),
        ))
        # Prune old rows for this event (keep last 10)
        all_rows = (await sess.execute(
            select(ConsensusHistory.id)
            .where(ConsensusHistory.event_id == event_id)
            .order_by(desc(ConsensusHistory.recorded_at))
        )).scalars().all()
        if len(all_rows) > 10:
            stale_ids = all_rows[10:]
            await sess.execute(
                delete(ConsensusHistory)
                .where(ConsensusHistory.id.in_(stale_ids))
            )
        await sess.commit()


async def _load_ev_cache() -> None:
    """Warm the in-memory EV-history cache from the DB on first call."""
    global _ev_cache, _ev_cache_loaded
    if _ev_cache_loaded:
        return
    from backend.storage.models import EVHistory
    async with get_session() as sess:
        rows = (await sess.execute(
            select(EVHistory)
            .order_by(EVHistory.recorded_at.asc())
            .limit(1000)  # cap memory; per-bucket trim happens on write
        )).scalars().all()
    cache: dict[int, list[float]] = {}
    for r in rows:
        cache.setdefault(r.bucket_id, []).append(r.ev_at_bid)
    # Trim each bucket's history to the last EDGE_DECAY_HISTORY_KEEP entries.
    keep = Config.EDGE_DECAY_HISTORY_KEEP
    for bid, lst in cache.items():
        if len(lst) > keep:
            cache[bid] = lst[-keep:]
    _ev_cache = cache
    _ev_cache_loaded = True
    log.info("ev_cache: loaded %d buckets, %d total rows from DB",
             len(cache), sum(len(v) for v in cache.values()))


async def _record_ev(bucket_id: int, ev_at_bid: float, yes_bid: float | None,
                     model_prob: float | None) -> None:
    """Append an EV observation for a bucket to DB + in-memory cache."""
    from backend.storage.models import EVHistory
    hist = _ev_cache.setdefault(bucket_id, [])
    hist.append(ev_at_bid)
    keep = Config.EDGE_DECAY_HISTORY_KEEP
    if len(hist) > keep:
        del hist[:-keep]
    async with get_session() as sess:
        sess.add(EVHistory(
            bucket_id=bucket_id,
            ev_at_bid=ev_at_bid,
            yes_bid=yes_bid,
            model_prob=model_prob,
            recorded_at=datetime.now(timezone.utc),
        ))
        # Prune old rows for this bucket (keep last N)
        all_rows = (await sess.execute(
            select(EVHistory.id)
            .where(EVHistory.bucket_id == bucket_id)
            .order_by(desc(EVHistory.recorded_at))
        )).scalars().all()
        if len(all_rows) > keep:
            stale_ids = all_rows[keep:]
            await sess.execute(
                delete(EVHistory).where(EVHistory.id.in_(stale_ids))
            )
        await sess.commit()


def _edge_decay_triggered(bucket_id: int) -> bool:
    """True if the last EDGE_DECAY_DEBOUNCE_RUNS recorded EVs are all <= threshold."""
    hist = _ev_cache.get(bucket_id, [])
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    if len(hist) < n:
        return False
    recent = hist[-n:]
    return all(e <= Config.EDGE_DECAY_THRESHOLD for e in recent)


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_entry_decision(pos) -> dict:
    raw = getattr(pos, "entry_decision_json", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


_SOURCE_HIGH_KEYS = (
    "nws_high",
    "wu_hourly_peak",
    "hrrr_high",
    "hrrr_15min_high",
    "nbm_high",
    "ecmwf_ifs_high",
    "ecmwf_aifs_high",
    "gfs_graphcast_high",
    "pangu_weather_high",
    "fourcastnet_v2_high",
    "aurora_high",
)


def _source_highs(reason: dict | None) -> dict[str, float]:
    reason = reason or {}
    highs: dict[str, float] = {}
    for key in _SOURCE_HIGH_KEYS:
        val = _safe_float(reason.get(key))
        if val is not None:
            highs[key] = val
    return highs


def _bucket_miss_distance_f(signal: BucketSignal, temp_f: float | None) -> float | None:
    """Distance outside the held bucket; 0 means the forecast is inside it."""
    temp = _safe_float(temp_f)
    if temp is None:
        return None
    low = _safe_float(signal.low_f)
    high = _safe_float(signal.high_f)
    if low is not None and temp < low:
        return low - temp
    if high is not None and temp > high:
        return temp - high
    return 0.0


def _source_forecast_deteriorations(signal: BucketSignal, entry_sources: dict, current_sources: dict) -> dict[str, float]:
    """Per-source increase in distance from the held bucket since entry."""
    deteriorations: dict[str, float] = {}
    for key, current in current_sources.items():
        entry = _safe_float(entry_sources.get(key))
        if entry is None:
            continue
        entry_distance = _bucket_miss_distance_f(signal, entry)
        current_distance = _bucket_miss_distance_f(signal, current)
        if entry_distance is None or current_distance is None:
            continue
        deterioration = current_distance - entry_distance
        if deterioration > 0:
            deteriorations[key] = round(deterioration, 3)
    return deteriorations


def _entry_ev_at_bid(entry_decision: dict) -> float | None:
    ev = _safe_float(entry_decision.get("ev_at_bid"))
    if ev is not None:
        return ev
    model_prob = _safe_float(entry_decision.get("model_prob"))
    yes_bid = _safe_float(entry_decision.get("yes_bid"))
    if model_prob is not None and yes_bid is not None:
        return model_prob - yes_bid
    return None


def _entry_strategy(pos, entry_decision: dict) -> str:
    return str(
        getattr(pos, "entry_strategy", None)
        or entry_decision.get("entry_strategy")
        or getattr(pos, "strategy", None)
        or ""
    ).strip().lower()


def _edge_decay_diagnostics(pos, signal: BucketSignal, entry_decision: dict) -> dict:
    entry_sources = entry_decision.get("source_highs")
    if not isinstance(entry_sources, dict):
        entry_sources = {}
    current_sources = _source_highs(signal.reason)
    source_deltas = {}
    for key, current in current_sources.items():
        entry = _safe_float(entry_sources.get(key))
        if entry is not None:
            source_deltas[key] = round(current - entry, 3)
    source_deteriorations = _source_forecast_deteriorations(signal, entry_sources, current_sources)

    entry_ev = _entry_ev_at_bid(entry_decision)
    current_ev = _safe_float(signal.ev_at_bid)
    entry_model_prob = _safe_float(entry_decision.get("model_prob"))
    current_model_prob = _safe_float(signal.model_prob)
    model_prob_drop = (
        round(entry_model_prob - current_model_prob, 6)
        if current_model_prob is not None and entry_model_prob is not None else None
    )
    entry_market_prob = _safe_float(entry_decision.get("market_prob"))
    current_market_prob = _safe_float(signal.mkt_prob)
    entry_true_edge = _safe_float(entry_decision.get("true_edge"))
    current_true_edge = _safe_float(signal.true_edge)
    entry_bid = _safe_float(entry_decision.get("yes_bid"))
    current_bid = _safe_float(signal.yes_bid)

    diagnostics = {
        "entry_snapshot_available": bool(entry_decision),
        "entry_model_snapshot_unavailable": not bool(entry_decision),
        "entry_strategy": _entry_strategy(pos, entry_decision) or None,
        "entry_type": getattr(pos, "entry_type", None) or entry_decision.get("entry_type"),
        "entry_ev_at_bid": entry_ev,
        "current_ev_at_bid": current_ev,
        "ev_delta": (
            round(current_ev - entry_ev, 6)
            if current_ev is not None and entry_ev is not None else None
        ),
        "entry_model_prob": entry_model_prob,
        "current_model_prob": current_model_prob,
        "model_prob_delta": (
            round(current_model_prob - entry_model_prob, 6)
            if current_model_prob is not None and entry_model_prob is not None else None
        ),
        "model_prob_drop": model_prob_drop,
        "entry_market_prob": entry_market_prob,
        "current_market_prob": current_market_prob,
        "market_prob_delta": (
            round(current_market_prob - entry_market_prob, 6)
            if current_market_prob is not None and entry_market_prob is not None else None
        ),
        "entry_true_edge": entry_true_edge,
        "current_true_edge": current_true_edge,
        "true_edge_delta": (
            round(current_true_edge - entry_true_edge, 6)
            if current_true_edge is not None and entry_true_edge is not None else None
        ),
        "entry_bid": entry_bid,
        "current_bid": current_bid,
        "bid_delta": (
            round(current_bid - entry_bid, 6)
            if current_bid is not None and entry_bid is not None else None
        ),
        "source_high_deltas": source_deltas,
        "source_forecast_deterioration_f": source_deteriorations,
        "max_source_forecast_deterioration_f": (
            max(source_deteriorations.values()) if source_deteriorations else 0.0
        ),
        "recent_ev_at_bid": [
            round(e, 6)
            for e in _ev_cache.get(pos.bucket_id, [])[-Config.EDGE_DECAY_DEBOUNCE_RUNS:]
        ],
    }
    return diagnostics


def _edge_decay_exit_allowed(pos, signal: BucketSignal, age_s: float, bid: float) -> tuple[bool, dict]:
    """Return whether EV deterioration is strong enough to trigger EDGE_DECAY."""
    entry_decision = _load_entry_decision(pos)
    diagnostics = _edge_decay_diagnostics(pos, signal, entry_decision)
    diagnostics["threshold"] = Config.EDGE_DECAY_THRESHOLD
    diagnostics["required_min_drop"] = Config.EDGE_DECAY_MIN_EV_DROP
    diagnostics["required_entry_min_ev"] = Config.EDGE_DECAY_ENTRY_MIN_EV
    diagnostics["require_model_deterioration"] = Config.EDGE_DECAY_REQUIRE_MODEL_DETERIORATION
    diagnostics["required_model_prob_drop"] = Config.EDGE_DECAY_MIN_MODEL_PROB_DROP
    diagnostics["required_source_temp_deterioration_f"] = Config.EDGE_DECAY_MIN_SOURCE_TEMP_DETERIORATION_F

    if signal.ev_at_bid is None:
        diagnostics["blocked_reason"] = "missing_current_ev_at_bid"
        return False, diagnostics
    if not _edge_decay_triggered(pos.bucket_id):
        diagnostics["blocked_reason"] = "ev_not_debounced"
        return False, diagnostics
    if bid < Config.EDGE_DECAY_MIN_BID:
        diagnostics["blocked_reason"] = "bid_below_floor"
        return False, diagnostics

    entry_strategy = str(diagnostics.get("entry_strategy") or "")
    entry_type = str(diagnostics.get("entry_type") or "").upper()
    is_manual = entry_type == "MANUAL" or entry_strategy.startswith("manual_")
    is_manual_scalp = entry_strategy in {"manual_scalp", "scalp"}
    min_age = (
        Config.MANUAL_EDGE_DECAY_MIN_AGE_SECONDS
        if is_manual else Config.EDGE_DECAY_MIN_POSITION_AGE_SECONDS
    )
    diagnostics["manual_position"] = is_manual
    diagnostics["min_age_seconds"] = min_age

    if age_s < min_age:
        diagnostics["blocked_reason"] = "position_too_young"
        return False, diagnostics

    entry_ev = diagnostics.get("entry_ev_at_bid")
    current_ev = diagnostics.get("current_ev_at_bid")
    if entry_ev is None:
        diagnostics["blocked_reason"] = "entry_model_snapshot_unavailable"
        return False, diagnostics
    if current_ev is None:
        diagnostics["blocked_reason"] = "missing_current_ev_at_bid"
        return False, diagnostics

    ev_drop = entry_ev - current_ev
    diagnostics["ev_drop"] = round(ev_drop, 6)
    if is_manual:
        if not is_manual_scalp:
            diagnostics["blocked_reason"] = "manual_strategy_exempt"
            return False, diagnostics
        if ev_drop < Config.EDGE_DECAY_MIN_EV_DROP:
            diagnostics["blocked_reason"] = "manual_ev_not_worse_by_min_drop"
            return False, diagnostics
    else:
        if entry_ev < Config.EDGE_DECAY_ENTRY_MIN_EV:
            diagnostics["blocked_reason"] = "auto_entry_ev_below_min"
            return False, diagnostics
        if ev_drop < Config.EDGE_DECAY_MIN_EV_DROP:
            diagnostics["blocked_reason"] = "auto_ev_not_worse_by_min_drop"
            return False, diagnostics

    model_prob_drop = _safe_float(diagnostics.get("model_prob_drop"), 0.0) or 0.0
    source_temp_deterioration = _safe_float(
        diagnostics.get("max_source_forecast_deterioration_f"), 0.0
    ) or 0.0
    model_deteriorated = (
        model_prob_drop >= Config.EDGE_DECAY_MIN_MODEL_PROB_DROP
        or source_temp_deterioration >= Config.EDGE_DECAY_MIN_SOURCE_TEMP_DETERIORATION_F
    )
    diagnostics["model_deteriorated"] = model_deteriorated
    if Config.EDGE_DECAY_REQUIRE_MODEL_DETERIORATION and not model_deteriorated:
        diagnostics["blocked_reason"] = "no_model_deterioration"
        return False, diagnostics

    diagnostics["blocked_reason"] = None
    return True, diagnostics


def _stable_consensus(event_id: int, held_bucket_id: int, multiplier: int = 1) -> int | None:
    """Return the consensus bucket_id only if the last N runs consistently
    agree on a DIFFERENT bucket than the held one. Returns None if still noisy."""
    hist = _consensus_cache.get(event_id, [])
    n = Config.CONSENSUS_DEBOUNCE_RUNS * multiplier
    if len(hist) < n:
        return None  # not enough data to be confident
    recent = hist[-n:]
    # All recent runs must agree on the same bucket, AND it must differ from held
    if all(b == recent[0] for b in recent) and recent[0] != held_bucket_id:
        return recent[0]  # stable shift confirmed
    return None  # noisy or still matches held — suppress URGENT


def _bucket_contains_temp(signal: BucketSignal, temp_f: float | None) -> bool:
    """Return whether a final/observed high is inside the held bucket."""
    if temp_f is None:
        return False
    if signal.low_f is not None and temp_f < signal.low_f:
        return False
    if signal.high_f is not None and temp_f > signal.high_f:
        return False
    return True


def _best_observed_high(signal: BucketSignal, fallback: float | None = None) -> float | None:
    """Prefer final/canonical highs when present, then live observed highs."""
    reason = signal.reason or {}
    for key in (
        "resolution_high_f",
        "resolution_high",
        "ground_truth_high_f",
        "ground_truth_high",
        "canonical_high_f",
        "canonical_high",
        "raw_high",
        "observed_high_f",
        "observed_high",
    ):
        val = reason.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return fallback


def _is_likely_expiry_winner(signal: BucketSignal, bid: float, obs_high: float | None) -> bool:
    """Late-session winner classifier used only for exit decisions."""
    if _bucket_contains_temp(signal, obs_high):
        return True
    if signal.model_prob >= Config.EXPIRY_WINNER_HOLD_MIN_PROB:
        return True
    model_contradicts = signal.model_prob < Config.URGENT_MIN_EXIT_MODEL_PROB
    if bid >= Config.EXPIRY_MARKET_WIN_MIN_BID and not model_contradicts:
        return True
    return False


def _loss_exit_blocked_by_positive_ev(pos, signal: BucketSignal, price: float) -> bool:
    """Guard against non-emergency loss exits while the held bucket is still +EV."""
    return (
        price < (pos.avg_cost or 0.0)
        and signal.ev_at_bid is not None
        and signal.ev_at_bid > 0.0
    )


def _obs_reference_temp(signal: BucketSignal, current_temp: float | None, obs_high: float | None) -> float | None:
    """Best available near-term temperature reference for OBS_PROXIMITY."""
    reason = signal.reason or {}
    adaptive = reason.get("adaptive") if isinstance(reason.get("adaptive"), dict) else {}
    candidates = (
        current_temp,
        adaptive.get("predicted_daily_high"),
        reason.get("projected_high_for_blend"),
        reason.get("projected_high"),
        obs_high,
    )
    for value in candidates:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _obs_cooldown_active(bucket_id: int, now_utc: datetime) -> bool:
    last = _obs_exit_cache.get(bucket_id)
    if last is None:
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    cooldown = timedelta(minutes=Config.OBS_REENTRY_COOLDOWN_MINUTES)
    return now_utc - last.astimezone(timezone.utc) < cooldown


async def _audit_obs_decision(decision: dict) -> None:
    """Persist an OBS_PROXIMITY decision payload without risking the cascade."""
    try:
        async with get_session() as sess:
            await append_audit(
                sess,
                actor="exit_engine",
                action="obs_proximity_decision",
                payload=decision,
                ok=True,
            )
    except Exception:
        log.debug("exit_engine: failed to audit OBS_PROXIMITY decision", exc_info=True)


async def _run_exit_cascade_for_position(
    pos,
    signal: BucketSignal,
    consensus_bucket_id: Optional[int],
    consensus_sig: Optional[BucketSignal] = None,
    market_leader_sig: Optional[BucketSignal] = None,
) -> dict | None:
    """Evaluate the 4-level cascade for a single position."""
    
    # ── Price sanity checks ──
    bid = signal.yes_bid or 0.0
    if bid <= 0:
        return None  # No phantom exits on stale/zero books
    
    if pos.avg_cost <= 0:
        return None
        
    async with get_session() as sess:
        city = await get_city_by_slug(sess, signal.city_slug)
        if not city:
            return None
            
    now_local = city_local_now(city)
    
    # Calculate net profit (accounting for 2% fee)
    gross_pnl_per_share = bid - pos.avg_cost
    fee_per_share = bid * FEE_RATE
    net_pnl_per_share = gross_pnl_per_share - fee_per_share
    
    # Extract METAR observations
    current_temp = signal.reason.get("current_temp_f")
    raw_high = signal.reason.get("raw_high")
    obs_high = _best_observed_high(signal, raw_high if raw_high is not None else current_temp)
    
    # ── 1. EMERGENCY ──
    # METAR obs contradicts bucket by >= 3°F causing impossible situation or deep loss
    if obs_high is not None and signal.high_f is not None:
        if obs_high > signal.high_f + 1.0:
            # We already busted this bucket. It's essentially worth 0. Try to salvage anything.
            log.warning("exit: EMERGENCY %s - busted high (obs=%.1f > bucket=%.1f)", signal.city_slug, obs_high, signal.high_f)
            return {"level": "EMERGENCY", "price": bid, "reason": "busted_high"}
            
    if obs_high is not None and signal.low_f is not None and signal.model_prob < 0.05 and signal.city_state in ("resolved", "volatile"):
        if obs_high < signal.low_f - 3.0 and now_local.hour >= 18:
            log.warning("exit: EMERGENCY %s - deep miss (obs=%.1f < bucket=%.1f)", signal.city_slug, obs_high, signal.low_f)
            return {"level": "EMERGENCY", "price": bid, "reason": "deep_miss"}

    # Position age (used by EDGE_DECAY and URGENT gates).
    age_s = (
        (now_local.astimezone(timezone.utc) - pos.entry_time.astimezone(timezone.utc)).total_seconds()
        if pos.entry_time else float('inf')
    )

    # ── 1b. EDGE_DECAY ── (EV-based, fires before URGENT)
    # Exit when ev_at_bid has stayed at or below threshold for N consecutive runs.
    # Held bucket is no longer +EV; sell the non-moon-bag portion at the bid.
    edge_decay_allowed, edge_decay_diag = _edge_decay_exit_allowed(pos, signal, age_s, bid)
    if edge_decay_allowed:
        held_is_model_leader = bool(consensus_sig and consensus_sig.bucket_id == pos.bucket_id)
        held_is_market_leader = bool(market_leader_sig and market_leader_sig.bucket_id == pos.bucket_id)
        edge_decay_diag["held_is_model_leader"] = held_is_model_leader
        edge_decay_diag["held_is_market_leader"] = held_is_market_leader
        edge_decay_diag["model_leader_bucket_id"] = getattr(consensus_sig, "bucket_id", None)
        edge_decay_diag["market_leader_bucket_id"] = getattr(market_leader_sig, "bucket_id", None)
        if (
            Config.EDGE_DECAY_PROTECT_LEADING_BUCKET
            and (
                (
                    held_is_model_leader
                    and signal.model_prob >= Config.EDGE_DECAY_LEADER_MIN_MODEL_PROB
                )
                or held_is_market_leader
            )
        ):
            log.info(
                "exit: EDGE_DECAY suppressed %s — held bucket still leads "
                "(model_leader=%s market_leader=%s model_prob=%.3f ev_at_bid=%.4f)",
                signal.city_slug,
                held_is_model_leader,
                held_is_market_leader,
                signal.model_prob,
                signal.ev_at_bid,
            )
            edge_decay_diag["blocked_reason"] = "held_bucket_still_leading"
            edge_decay_allowed = False

    if edge_decay_allowed:
        sell_qty = pos.net_qty - (pos.moon_bag_qty or 0.0)
        if sell_qty <= 0:
            log.info(
                "exit: EDGE_DECAY suppressed %s — only moon-bag remaining (%.1f shares)",
                signal.city_slug, pos.net_qty,
            )
        else:
            recent = _ev_cache.get(pos.bucket_id, [])[-Config.EDGE_DECAY_DEBOUNCE_RUNS:]
            log.info(
                "exit: EDGE_DECAY %s — ev_at_bid deteriorated from %s to %.4f "
                "(drop=%s, recent=%s, age_s=%d). Selling %.1f shares at bid=%.3f.",
                signal.city_slug, edge_decay_diag.get("entry_ev_at_bid"),
                signal.ev_at_bid,
                edge_decay_diag.get("ev_drop"),
                [round(e, 4) for e in recent], int(age_s), sell_qty, bid,
            )
            return {
                "level": "EDGE_DECAY",
                "price": bid,
                "reason": "ev_decayed",
                "qty_override": sell_qty,
                "diagnostics": edge_decay_diag,
            }
    elif edge_decay_diag.get("blocked_reason") not in {"ev_not_debounced", "missing_current_ev_at_bid"}:
        log.info(
            "exit: EDGE_DECAY suppressed %s — %s (entry_ev=%s current_ev=%s recent=%s)",
            signal.city_slug,
            edge_decay_diag.get("blocked_reason"),
            edge_decay_diag.get("entry_ev_at_bid"),
            edge_decay_diag.get("current_ev_at_bid"),
            edge_decay_diag.get("recent_ev_at_bid"),
        )

    # ── 2. URGENT ── (debounced + spread-guarded + confidence-gated + EV-corroborated)
    # Model consensus shifted to different bucket, we are holding a non-consensus bucket.
    if consensus_bucket_id and pos.bucket_id != consensus_bucket_id and consensus_sig:
        spread = signal.spread or 0.0
        bid_depth = signal.yes_bid_depth or 0.0
        held_prob = signal.model_prob
        cons_prob = consensus_sig.model_prob

        # EV corroboration: if the held bucket is still meaningfully +EV at the bid,
        # a consensus-shift alone is not enough to URGENT-exit. EDGE_DECAY (above)
        # already handles the case where EV has actually decayed; URGENT now only
        # fires on a structural shift that's NOT contradicted by EV.
        ev_at_bid = signal.ev_at_bid

        if ev_at_bid is not None and ev_at_bid > Config.EDGE_DECAY_THRESHOLD:
            log.info(
                "exit: URGENT suppressed %s — consensus shifted but held bucket still +EV "
                "(ev_at_bid=%.4f > %.4f, consensus→bucket_id=%s)",
                signal.city_slug, ev_at_bid, Config.EDGE_DECAY_THRESHOLD, consensus_bucket_id,
            )
        elif age_s < Config.URGENT_MIN_POSITION_AGE_SECONDS:
            log.info("exit: URGENT suppressed %s — position age %ds < %ds", signal.city_slug, int(age_s), Config.URGENT_MIN_POSITION_AGE_SECONDS)
        elif held_prob >= Config.URGENT_MIN_EXIT_MODEL_PROB and held_prob >= cons_prob * 0.40:
            log.info("exit: URGENT suppressed %s — probability gate (held=%.2f, cons=%.2f)", signal.city_slug, held_prob, cons_prob)
        elif bid_depth < Config.URGENT_MIN_BID_DEPTH:
            log.warning("exit: URGENT suppressed %s — thin bid depth %.1f < %.1f", signal.city_slug, bid_depth, Config.URGENT_MIN_BID_DEPTH)
        elif spread > Config.URGENT_EXIT_MAX_SPREAD:
            log.warning("exit: URGENT suppressed %s — spread %.3f > %.3f (consensus→%s)", signal.city_slug, spread, Config.URGENT_EXIT_MAX_SPREAD, consensus_bucket_id)
        else:
            # URGENT exit: only sell non-moon-bag portion
            sell_qty = pos.net_qty - (pos.moon_bag_qty or 0.0)
            if sell_qty <= 0:
                log.info("exit: URGENT suppressed %s — only moon-bag remaining (%.1f shares)", signal.city_slug, pos.net_qty)
            else:
                # Check depth for passive vs aggressive sell
                if bid_depth < sell_qty * 2:
                    sell_price = bid  # Passive limit sell at the bid exactly to not sweep thin books
                else:
                    sell_price = max(0.01, bid - 0.01)  # Aggressive: bid - 1c

                log.info("exit: URGENT %s - consensus shifted to bucket_id=%s. Exiting %.1f shares (keeping %.1f moon-bag).",
                         signal.city_slug, consensus_bucket_id, sell_qty, pos.moon_bag_qty or 0.0)
                return {"level": "URGENT", "price": sell_price, "reason": "consensus_shifted",
                        "qty_override": sell_qty}

    # ── 2b. OBS_PROXIMITY ──
    # Near the next official station observation, take marked profits on a
    # fragile held bucket before a +/-1F official readout can flip the winner.
    reason = signal.reason or {}
    station_id = (
        reason.get("active_station_id")
        or reason.get("station_id")
        or getattr(city, "metar_station", None)
    )
    if station_id:
        station_id = str(station_id).upper()

    observation_minutes = normalize_observation_minutes(reason.get("observation_minutes"))
    if (
        Config.OBS_EXIT_ENABLED
        and getattr(city, "is_us", False)
        and station_id
        and not observation_minutes
    ):
        try:
            async with get_session() as sess:
                profile = await get_station_profile(sess, station_id)
            if profile and profile.observation_minutes:
                observation_minutes = normalize_observation_minutes(profile.observation_minutes)
        except Exception:
            log.debug("exit: OBS_PROXIMITY station-profile lookup failed for %s", station_id, exc_info=True)

    bucket_specs = []
    if Config.OBS_EXIT_ENABLED and getattr(city, "is_us", False) and observation_minutes:
        try:
            async with get_session() as sess:
                buckets = await get_buckets_for_event(sess, signal.event_id)
            bucket_specs = [
                {
                    "bucket_idx": b.bucket_idx,
                    "label": b.label or f"Bucket {b.bucket_idx}",
                    "low_f": b.low_f,
                    "high_f": b.high_f,
                }
                for b in buckets
            ]
        except Exception:
            log.debug("exit: OBS_PROXIMITY bucket lookup failed for event %s", signal.event_id, exc_info=True)

    obs_decision = evaluate_obs_proximity_exit(
        city_slug=signal.city_slug,
        station_id=station_id,
        now_local=now_local,
        observation_minutes=observation_minutes,
        bucket_specs=bucket_specs,
        held_bucket_idx=signal.bucket_idx,
        reference_temp_f=_obs_reference_temp(signal, current_temp, obs_high),
        yes_bid=signal.yes_bid,
        yes_ask=signal.yes_ask,
        yes_bid_depth=signal.yes_bid_depth,
        yes_ask_depth=signal.yes_ask_depth,
        net_pnl_per_share=net_pnl_per_share,
        current_edge=signal.ev_at_bid if signal.ev_at_bid is not None else signal.true_edge,
        enabled=Config.OBS_EXIT_ENABLED,
        is_us=bool(getattr(city, "is_us", False)),
        window_minutes=Config.OBS_EXIT_WINDOW_MINUTES,
        temp_sensitivity_threshold_f=Config.TEMP_SENSITIVITY_THRESHOLD_F,
        min_profit_cents=Config.OBS_MIN_PROFIT_CENTS,
        min_depth_usd=Config.OBS_MIN_DEPTH_USD,
        max_orderbook_imbalance=Config.OBS_MAX_ORDERBOOK_IMBALANCE,
        cooldown_active=_obs_cooldown_active(pos.bucket_id, now_local.astimezone(timezone.utc)),
    )

    if obs_decision["final_action"] == "EXIT":
        sell_qty = pos.net_qty - (pos.moon_bag_qty or 0.0)
        if sell_qty <= 0:
            obs_decision["final_action"] = "SKIP"
            obs_decision["skip_reason"] = "only_moon_bag_remaining"
        else:
            obs_decision["shares_to_exit"] = float(sell_qty)

    log.info(
        "exit: OBS_PROXIMITY decision city=%s station=%s action=%s reason=%s "
        "temp=%s bucket=%s plus=%s minus=%s boundary=%s min_to_obs=%s "
        "edge=%s pnl_c=%s depth_usd=%s imbalance=%s",
        obs_decision.get("city"),
        obs_decision.get("station"),
        obs_decision.get("final_action"),
        obs_decision.get("skip_reason"),
        obs_decision.get("current_temp"),
        obs_decision.get("current_bucket"),
        obs_decision.get("plus_1f_bucket"),
        obs_decision.get("minus_1f_bucket"),
        obs_decision.get("boundary_distance_f"),
        obs_decision.get("minutes_to_next_obs"),
        obs_decision.get("current_edge"),
        obs_decision.get("mark_to_market_profit_cents"),
        obs_decision.get("orderbook_depth_usd"),
        obs_decision.get("imbalance"),
    )
    await _audit_obs_decision(obs_decision)

    if obs_decision["final_action"] == "EXIT":
        _obs_exit_cache[pos.bucket_id] = now_local.astimezone(timezone.utc)
        log.info(
            "exit: OBS_PROXIMITY %s - fragile bucket near station obs; selling %.1f at bid=%.3f",
            signal.city_slug, obs_decision["shares_to_exit"], bid,
        )
        return {
            "level": "OBS_PROXIMITY",
            "price": bid,
            "reason": "observation_proximity_fragile_bucket",
            "qty_override": obs_decision["shares_to_exit"],
            "obs_decision": obs_decision,
        }

    # ── Update trailing stop high-water mark ──
    # Ratchet up max_bid_seen and trailing stop on every cycle
    if bid > (pos.max_bid_seen or 0.0):
        async with get_session() as sess:
            from sqlalchemy import update
            import backend.storage.models as m
            await sess.execute(
                update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                .values(
                    max_bid_seen=bid,
                    trailing_stop_price=max(bid - 0.05, pos.trailing_stop_price or 0.0) if pos.tier_1_exited else None,
                )
            )
            await sess.commit()

    # ── 3. PROFIT — Tiered partial exits + trailing stop ──
    tier_1_target = pos.avg_cost + 0.08  # +8¢ → sell 50%
    tier_2_target = pos.avg_cost + 0.15  # +15¢ → sell 25% more
    original_qty = pos.original_qty if pos.original_qty > 0 else pos.net_qty

    # Tier 1: Sell 50% at +8¢
    if not pos.tier_1_exited and bid >= tier_1_target:
        tier_1_qty = round(original_qty * 0.50, 2)
        tier_1_qty = min(tier_1_qty, pos.net_qty)  # can't sell more than we have
        if tier_1_qty > 0:
            log.info("exit: PROFIT Tier-1 %s — selling 50%% (%.1f shares) at +8¢ (bid=%.3f, entry=%.3f)",
                     signal.city_slug, tier_1_qty, bid, pos.avg_cost)
            return {"level": "PROFIT", "price": bid, "reason": "tier_1_50pct",
                    "qty_override": tier_1_qty,
                    "post_exit_update": {"tier_1_exited": True,
                                          "moon_bag_qty": round(original_qty * 0.25, 2),
                                          "trailing_stop_price": bid - 0.05,
                                          "max_bid_seen": bid}}

    # Tier 2: Sell 25% at +15¢
    if pos.tier_1_exited and not pos.tier_2_exited and bid >= tier_2_target:
        tier_2_qty = round(original_qty * 0.25, 2)
        tier_2_qty = min(tier_2_qty, pos.net_qty - (pos.moon_bag_qty or 0.0))
        if tier_2_qty > 0:
            log.info("exit: PROFIT Tier-2 %s — selling 25%% (%.1f shares) at +15¢ (bid=%.3f)",
                     signal.city_slug, tier_2_qty, bid)
            return {"level": "PROFIT", "price": bid, "reason": "tier_2_25pct",
                    "qty_override": tier_2_qty,
                    "post_exit_update": {"tier_2_exited": True}}

    # Trailing stop: After Tier 1, if bid drops below trailing stop, exit non-moon portion
    if pos.tier_1_exited and pos.trailing_stop_price and bid < pos.trailing_stop_price:
        trailing_qty = pos.net_qty - (pos.moon_bag_qty or 0.0)
        if trailing_qty > 0:
            log.info("exit: PROFIT Trailing-Stop %s — bid %.3f < stop %.3f. Selling %.1f shares.",
                     signal.city_slug, bid, pos.trailing_stop_price, trailing_qty)
            return {"level": "PROFIT", "price": bid, "reason": "trailing_stop",
                    "qty_override": trailing_qty}

    # Legacy quick-flip for positions that haven't been initialized with tiers
    if not pos.tier_1_exited and not pos.original_qty:
        target_price = pos.avg_cost + Config.QUICK_FLIP_TARGET
        if bid >= target_price:
            log.info("exit: PROFIT QuickFlip %s! (entry=%.3f, bid=%.3f, net_pnl_share=%.3f)", signal.city_slug, pos.avg_cost, bid, net_pnl_per_share)
            return {"level": "PROFIT", "price": bid, "reason": "quick_flip"}

    # ── 4. EXPIRY ──
    # 30 min before market close (7:30 PM local for daily high markets).
    # Hold likely winners to redeem. Only risk-exit ambiguous/losing buckets,
    # and never use the old 10c dump unless explicitly configured below the
    # risk-exit cap.
    if now_local.hour == 19 and now_local.minute >= 30:
        likely_winner = _is_likely_expiry_winner(signal, bid, obs_high)
        if likely_winner:
            if bid >= Config.EXPIRY_PASSIVE_SELL_MIN_BID:
                reference_price = max(Config.EXPIRY_PASSIVE_SELL_MIN_BID, bid)
                sell_price = normalize_limit_price_for_clob(reference_price)
                price_adjustment = None
                if abs(reference_price - sell_price) >= 0.0005:
                    price_adjustment = {
                        "reason": "clamped_to_clob_limit_range",
                        "reference_price": round(reference_price, 4),
                        "order_price": round(sell_price, 4),
                    }
                log.info(
                    "exit: EXPIRY_PASSIVE %s — likely winner; passively offering at %.3f "
                    "(reference=%.3f, bid=%.3f, model=%.3f, obs_high=%s)",
                    signal.city_slug, sell_price, reference_price, bid, signal.model_prob, obs_high,
                )
                return {
                    "level": "PROFIT",
                    "price": sell_price,
                    "reason": "expiry_passive_winner",
                    "reference_price": reference_price,
                    "diagnostics": {
                        "pre_cap_bid": bid,
                        "reference_price": round(reference_price, 4),
                        "order_price": round(sell_price, 4),
                        "price_adjustment": price_adjustment,
                    },
                }
            log.info(
                "exit: HOLD_TO_REDEEM %s — likely winner; bid %.3f below passive sell floor %.3f",
                signal.city_slug, bid, Config.EXPIRY_PASSIVE_SELL_MIN_BID,
            )
            return {
                "level": "HOLD",
                "price": bid,
                "reason": "likely_winner_hold_to_redeem",
                "no_order": True,
                "status": "HOLD_TO_REDEEM",
            }

        discount = min(Config.EXPIRY_DISCOUNT, Config.EXPIRY_RISK_EXIT_MAX_DISCOUNT)
        risk_exit_price = max(0.01, bid - discount)
        if _loss_exit_blocked_by_positive_ev(pos, signal, risk_exit_price):
            log.info(
                "exit: EXPIRY suppressed %s — risk price %.3f below avg %.3f while ev_at_bid=%.4f",
                signal.city_slug, risk_exit_price, pos.avg_cost, signal.ev_at_bid,
            )
            return {
                "level": "HOLD",
                "price": bid,
                "reason": "positive_ev_blocks_expiry_loss_exit",
                "no_order": True,
                "status": "EXIT_BLOCKED",
            }
        log.info(
            "exit: EXPIRY_RISK %s — ambiguous/losing near close; bid=%.3f discount=%.3f price=%.3f",
            signal.city_slug, bid, discount, risk_exit_price,
        )
        return {"level": "EXPIRY", "price": risk_exit_price, "reason": "market_close_risk_exit"}

    return None


async def run_exit_engine() -> None:
    """Run the 4-level exit engine cascade on all open positions."""
    log.info("exit_engine: evaluating open positions")
    
    async with get_session() as sess:
        positions = await get_all_positions(sess)
        open_orders = await get_open_orders(sess)
        
    active_positions = [p for p in positions if p.net_qty > 0]
    if not active_positions:
        log.debug("exit_engine: no active positions")
        return
        
    # Find buckets with pending SELL orders so we don't double-exit
    pending_sell_buckets = {o.bucket_id for o in open_orders if o.side == "sell_yes"}
    
    # Get latest signals to inform exits
    signals = await run_signal_engine()
    sig_map = {s.bucket_id: s for s in signals}
    
    # Warm consensus + EV caches from DB on first run (survives deploys)
    await _load_consensus_cache()
    await _load_ev_cache()

    # Find consensus bucket for each event (highest model prob)
    event_consensus = {}
    for s in signals:
        if s.event_id not in event_consensus or s.model_prob > event_consensus[s.event_id].model_prob:
            event_consensus[s.event_id] = s

    # Find market-leading bucket for each event (highest mid, falling back to bid).
    # This is only a veto for EDGE_DECAY; it does not create trade signals.
    event_market_leader = {}
    for s in signals:
        price = s.yes_mid if s.yes_mid is not None else s.yes_bid
        if price is None:
            continue
        current = event_market_leader.get(s.event_id)
        current_price = (
            current.yes_mid if current and current.yes_mid is not None
            else (current.yes_bid if current else None)
        )
        if current is None or current_price is None or price > current_price:
            event_market_leader[s.event_id] = s

    # Persist consensus to DB + update in-memory cache
    for event_id, sig in event_consensus.items():
        await _record_consensus(event_id, sig.bucket_id)

    # Persist per-position EV-at-bid history (drives the EDGE_DECAY gate).
    # Skip positions with no signal or no bid-side EV (no market data).
    for pos in active_positions:
        signal = sig_map.get(pos.bucket_id)
        if signal is None or signal.ev_at_bid is None:
            continue
        await _record_ev(
            bucket_id=pos.bucket_id,
            ev_at_bid=signal.ev_at_bid,
            yes_bid=signal.yes_bid,
            model_prob=signal.model_prob,
        )

    exits_triggered = 0

    for pos in active_positions:
        if pos.bucket_id in pending_sell_buckets:
            continue

        signal = sig_map.get(pos.bucket_id)
        if not signal:
            continue

        # Use debounced consensus — only fires if last N runs consistently agree
        # on a different bucket than the held position.
        consensus_sig = event_consensus.get(signal.event_id)
        multiplier = 1
        if consensus_sig and abs(consensus_sig.bucket_idx - signal.bucket_idx) == 1:
            multiplier = Config.URGENT_ADJACENT_DEBOUNCE_MULTIPLIER

        stable_consensus_id = _stable_consensus(signal.event_id, pos.bucket_id, multiplier)

        market_leader_sig = event_market_leader.get(signal.event_id)
        cascade = await _run_exit_cascade_for_position(
            pos,
            signal,
            stable_consensus_id,
            consensus_sig,
            market_leader_sig,
        )
        if cascade:
            if cascade.get("no_order"):
                status = cascade.get("status") or cascade["level"]
                exits_triggered += 1
                log.info(
                    "exit_engine: %s for bucket %d — no order placed (%s)",
                    status, pos.bucket_id, cascade["reason"],
                )
                async with get_session() as sess:
                    from sqlalchemy import update
                    import backend.storage.models as m
                    await sess.execute(
                        update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                        .values(current_exit_status=f"{status}: {cascade['reason']}")
                    )
                    await sess.commit()
                await _emit_exit_event(
                    pos, signal, cascade,
                    shares_exited=0.0,
                    execution_status="no_order",
                )
                continue

            exits_triggered += 1
            sell_price = round(cascade["price"], 3)
            exit_qty = cascade.get("qty_override", pos.net_qty)
            log.info(
                "exit_engine: triggering %s exit for bucket %d (shares=%.1f of %.1f). Limit=%.3f.", 
                cascade["level"], pos.bucket_id, exit_qty, pos.net_qty, sell_price
            )
            
            # Save exit triggered status
            async with get_session() as sess:
                from sqlalchemy import update
                import backend.storage.models as m
                await sess.execute(
                    update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                    .values(current_exit_status=f"{cascade['level']} exit triggered: {cascade['reason']} ({exit_qty:.1f} shares)")
                )
                await sess.commit()

            result = await execute_signal(
                signal=signal,
                bankroll=0.0, # Not used for sells
                actor="exit_engine",
                manual=True, # Bypasses sizing & most gates
                order_type="limit",
                side="SELL",
                limit_price_override=sell_price,
                qty_override=exit_qty,
            )
            
            exec_status = result.get("status", "unknown")
            if result.get("price_adjustment"):
                diagnostics = cascade.setdefault("diagnostics", {})
                diagnostics["price_adjustment"] = result.get("price_adjustment")
                diagnostics["reference_price"] = result.get("reference_price")
                diagnostics["order_price"] = result.get("order_price")
            fill_payload = result.get("fill") if isinstance(result.get("fill"), dict) else {}
            shares_actually_exited = (
                float(fill_payload.get("qty") or exit_qty)
                if exec_status == "filled" else 0.0
            )
            await _emit_exit_event(
                pos, signal, cascade,
                shares_exited=shares_actually_exited,
                execution_status=exec_status,
                error=None if exec_status in ("filled", "timeout", "open") else str(result.get("error", "")),
            )

            # ── Handle result ──
            if result.get("status") in ("filled", "timeout", "open"):
                # Apply post-exit tier state updates (tier flags, trailing stop, etc.)
                post_update = cascade.get("post_exit_update")
                if post_update and result.get("status") == "filled":
                    async with get_session() as sess:
                        from sqlalchemy import update
                        import backend.storage.models as m
                        await sess.execute(
                            update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                            .values(**post_update)
                        )
                        await sess.commit()
                        log.info("exit_engine: applied post-exit update for bucket %d: %s", pos.bucket_id, post_update)

                try:
                    from backend.notifications.telegram import notify_exit_triggered
                    await notify_exit_triggered(
                        city_slug=signal.city_slug,
                        level=cascade["level"],
                        reason=cascade["reason"],
                        price=sell_price,
                        shares=exit_qty,
                        details=cascade.get("diagnostics"),
                    )
                except Exception:
                    log.debug("Telegram exit notification failed (non-critical)", exc_info=True)
            else:
                # Exit order failed — update position status and notify
                err = result.get("error", "unknown")
                log.warning(
                    "exit_engine: %s exit for bucket %d FAILED — %s",
                    cascade["level"], pos.bucket_id, err,
                )
                async with get_session() as sess:
                    from sqlalchemy import update
                    import backend.storage.models as m
                    await sess.execute(
                        update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                        .values(current_exit_status=f"{cascade['level']} exit FAILED: {err[:80]}")
                    )
                    await sess.commit()
                try:
                    from backend.notifications.telegram import notify_exit_failed
                    await notify_exit_failed(
                        city_slug=signal.city_slug,
                        level=cascade["level"],
                        reason=cascade["reason"],
                        price=sell_price,
                        shares=exit_qty,
                        error=err,
                        details=cascade.get("diagnostics"),
                    )
                except Exception:
                    log.debug("Telegram exit-failed notification failed (non-critical)", exc_info=True)
        else:
            # Save active monitoring status with tier awareness
            bid = signal.yes_bid or 0.0
            if pos.tier_1_exited:
                moon_info = f" | Moon-bag: {pos.moon_bag_qty:.1f} shares" if pos.moon_bag_qty else ""
                trail_info = f" | Trail: ${pos.trailing_stop_price:.3f}" if pos.trailing_stop_price else ""
                status = f"Monitoring: Tier-1 locked{' + Tier-2 locked' if pos.tier_2_exited else ''}{moon_info}{trail_info} (bid ${bid:.2f})"
            else:
                target_price = pos.avg_cost + 0.08
                status = f"Monitoring: await Tier-1 +8¢ (target ${target_price:.2f}, bid ${bid:.2f}) or Shift"
            async with get_session() as sess:
                from sqlalchemy import update
                import backend.storage.models as m
                await sess.execute(
                    update(m.Position).where(m.Position.bucket_id == pos.bucket_id)
                    .values(current_exit_status=status)
                )
                await sess.commit()
            
    log.info("exit_engine: complete (%d exits triggered)", exits_triggered)
