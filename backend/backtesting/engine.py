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
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
from sqlalchemy import String, desc, func, select

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
from backend.execution.obs_proximity import evaluate_obs_proximity_exit
from backend.modeling.calibration_engine import resolve_canonical_settlement_high
from backend.modeling.settlement import canonical_bucket_ranges, find_bucket_idx_for_value
from backend.strategy.kelly import calculate_kelly_fraction
from backend.storage.db import get_session
from backend.storage.models import (
    BacktestResolvedEvent,
    BacktestRun,
    BacktestTrade,
    Bucket,
    City,
    Event,
    ForecastObs,
    MarketSnapshot,
    MetarObs,
    ModelSnapshot,
)

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
# Gamma currently caps event pages at 100 rows. If we request 500, it still
# returns 100; pagination must use the real cap or enrichment stops after page 1.
GAMMA_PAGE_SIZE = 100

# Month name → number for slug parsing
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_SLUG_RE = re.compile(
    r"(highest|lowest)-temperature-in-(.+?)-on-([a-z]+)-(\d{1,2})(?:-(\d{4}))?"
)

# Gamma + local cities generally use matching short slugs (nyc/sf/la). These
# aliases only apply when a long form appears on one side (e.g. Gamma
# occasionally uses `new-york` instead of `nyc`). Both directions are handled.
_CITY_ALIASES: dict[str, str] = {
    "new-york": "nyc",
    "new-york-city": "nyc",
    "san-francisco": "sf",
    "los-angeles": "la",
    "washington-dc": "dc",
    "washington-d-c": "dc",
    # Reverse aliases (Gamma sometimes uses abbreviations)
    "atl": "atlanta",
    "chi": "chicago",
    "hou": "houston",
    "dal": "dallas",
    "den": "denver",
    "sea": "seattle",
    "mia": "miami",
    "aus": "austin",
    # Long-form Polymarket slugs
    "los-angeles-ca": "la",
    "san-francisco-ca": "sf",
    "new-york-ny": "nyc",
    "washington-d-c-": "dc",
}

# Bucket range parsers (reused from backend/ingestion/polymarket_gamma.py patterns)
_BUCKET_RANGE_RE = re.compile(
    r"(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)\s*°?\s*[FCfc]?"
)
_BUCKET_ABOVE_RE = re.compile(
    r"(?:(?:above|higher than|over|≥|>=)\s*(\d+\.?\d*)"
    r"|(\d+\.?\d*)\s*°?\s*[FCfc]?\s*or\s*(?:higher|above|more))",
    re.I,
)
_BUCKET_BELOW_RE = re.compile(
    r"(?:(?:below|lower than|under|<)\s*(\d+\.?\d*)"
    r"|(\d+\.?\d*)\s*°?\s*[FCfc]?\s*or\s*(?:below|lower|under|less))",
    re.I,
)


# ─── Parameters ──────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    """All tunable parameters for a backtest run."""
    bankroll: float = 100.0
    kelly_fraction: float = 0.10
    max_position_pct: float = 0.20
    min_true_edge: float = 0.03
    max_entry_price: float = 0.60
    max_spread: float = 0.20
    min_liquidity_shares: float = 0.0
    max_positions_per_event: int = 2
    night_owl_enabled: bool = True
    night_owl_start_hour: int = 23
    night_owl_end_hour: int = 6
    quick_flip_target: float = 0.05
    walk_forward_train_days: int = 21
    walk_forward_test_days: int = 7
    obs_exit_enabled: bool = False
    obs_exit_window_minutes: int = 20
    temp_sensitivity_threshold_f: float = 1.0
    obs_min_profit_cents: float = 5.0
    obs_min_depth_usd: float = 100.0
    obs_max_orderbook_imbalance: float = 0.72


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
    # Q7 — regime label snapshot at trade open (calm | normal | volatile | None)
    # Read from the originating ModelSnapshot.inputs_json["regime_label"]; lets
    # _build_results stratify Brier / win-rate by regime so the operator can
    # gate volatile-day trades when realized calibration is poor.
    regime_label: Optional[str] = None


@dataclass
class Portfolio:
    """Running portfolio state during backtest."""
    bankroll: float
    equity: float
    positions_per_event: dict = field(default_factory=lambda: defaultdict(int))

    def available(self) -> float:
        return max(0.0, self.bankroll)


_GATE_REASON_LABELS = {
    "missing_market": "Missing market snapshot",
    "invalid_market_probability": "Invalid market probability",
    "edge_below_min": "Edge below threshold",
    "market_prob_out_of_range": "Market probability out of range",
    "insufficient_ask_depth": "Not enough ask depth",
    "entry_price_above_max": "Entry price above cap",
    "spread_above_max": "Spread too wide",
    "max_positions_per_event": "Event position limit",
    "kelly_non_positive": "Kelly size <= 0",
    "trade_too_small": "Trade below minimum size",
}


def _new_gate_diagnostics(params: BacktestParams) -> dict:
    """Mutable diagnostics collector for strategy gate decisions."""
    return {
        "candidates_evaluated": 0,
        "trades_taken": 0,
        "rejected_total": 0,
        "reasons": defaultdict(int),
        "constraint_hits": defaultdict(int),
        "per_city": defaultdict(lambda: {"candidates": 0, "trades": 0, "rejected": 0}),
        "best_candidate": None,
        "best_rejected": None,
        "near_misses": [],
        "thresholds": {
            "min_true_edge": params.min_true_edge,
            "max_entry_price": params.max_entry_price,
            "max_spread": params.max_spread,
            "min_liquidity_shares": params.min_liquidity_shares,
            "max_positions_per_event": params.max_positions_per_event,
            "min_trade_cost": 0.50,
        },
    }


def _candidate_snapshot(
    event_data: dict,
    bucket_idx: int,
    *,
    model_prob: Optional[float] = None,
    mkt_prob: Optional[float] = None,
    true_edge: Optional[float] = None,
    spread: Optional[float] = None,
    ask_depth: Optional[float] = None,
    entry_price: Optional[float] = None,
    reason: Optional[str] = None,
    violations: Optional[list[str]] = None,
    actual: Optional[float] = None,
    required: Optional[float] = None,
) -> dict:
    bucket_info = next((b for b in event_data.get("buckets", []) if b.get("idx") == bucket_idx), None)
    out = {
        "city_slug": event_data.get("city_slug"),
        "date_et": event_data.get("date_et"),
        "bucket_idx": bucket_idx,
        "bucket_label": (bucket_info or {}).get("label") or f"Bucket {bucket_idx}",
        "model_prob": round(model_prob, 4) if model_prob is not None else None,
        "market_prob": round(mkt_prob, 4) if mkt_prob is not None else None,
        "true_edge": round(true_edge, 4) if true_edge is not None else None,
        "spread": round(spread, 4) if spread is not None else None,
        "ask_depth": round(ask_depth, 2) if ask_depth is not None else None,
        "entry_price": round(entry_price, 4) if entry_price is not None else None,
        "reason": reason,
        "reason_label": _GATE_REASON_LABELS.get(reason, reason) if reason else None,
        "violations": violations or [],
        "actual": round(actual, 4) if actual is not None else None,
        "required": round(required, 4) if required is not None else None,
    }
    if actual is not None and required is not None:
        out["gap"] = round(abs(actual - required), 4)
    return out


def _record_gate_candidate(diagnostics: Optional[dict], event_data: dict, candidate: dict) -> None:
    if diagnostics is None:
        return
    diagnostics["candidates_evaluated"] += 1
    city = event_data.get("city_slug") or "unknown"
    diagnostics["per_city"][city]["candidates"] += 1
    edge = candidate.get("true_edge")
    best = diagnostics.get("best_candidate")
    if edge is not None and (best is None or edge > (best.get("true_edge") or -999)):
        diagnostics["best_candidate"] = dict(candidate)


def _record_gate_rejection(
    diagnostics: Optional[dict],
    event_data: dict,
    candidate: dict,
    reason: str,
    violations: Optional[list[str]] = None,
) -> None:
    if diagnostics is None:
        return
    diagnostics["rejected_total"] += 1
    diagnostics["reasons"][reason] += 1
    for violation in violations or [reason]:
        diagnostics["constraint_hits"][violation] += 1
    city = event_data.get("city_slug") or "unknown"
    diagnostics["per_city"][city]["rejected"] += 1

    rejected = dict(candidate)
    rejected["reason"] = reason
    rejected["reason_label"] = _GATE_REASON_LABELS.get(reason, reason)
    rejected["violations"] = violations or [reason]
    edge = rejected.get("true_edge")
    best = diagnostics.get("best_rejected")
    if edge is not None and (best is None or edge > (best.get("true_edge") or -999)):
        diagnostics["best_rejected"] = rejected

    near_misses = diagnostics["near_misses"]
    near_misses.append(rejected)
    near_misses.sort(key=lambda row: row.get("true_edge") if row.get("true_edge") is not None else -999, reverse=True)
    del near_misses[8:]


def _record_gate_accept(diagnostics: Optional[dict], event_data: dict, candidate: dict) -> None:
    if diagnostics is None:
        return
    diagnostics["trades_taken"] += 1
    city = event_data.get("city_slug") or "unknown"
    diagnostics["per_city"][city]["trades"] += 1


def _gate_actual_required(reason: str, *, actuals: dict, params: BacktestParams) -> tuple[Optional[float], Optional[float]]:
    if reason == "edge_below_min":
        return actuals.get("true_edge"), params.min_true_edge
    if reason == "entry_price_above_max":
        return actuals.get("entry_price"), params.max_entry_price
    if reason == "spread_above_max":
        return actuals.get("spread"), params.max_spread
    if reason == "insufficient_ask_depth":
        return actuals.get("ask_depth"), params.min_liquidity_shares
    if reason == "trade_too_small":
        return actuals.get("cost"), 0.50
    return None, None


def _finalize_gate_diagnostics(diagnostics: Optional[dict]) -> dict:
    if not diagnostics:
        return {}

    reasons = [
        {"reason": reason, "label": _GATE_REASON_LABELS.get(reason, reason), "count": count}
        for reason, count in diagnostics["reasons"].items()
    ]
    reasons.sort(key=lambda row: row["count"], reverse=True)

    constraint_hits = [
        {"reason": reason, "label": _GATE_REASON_LABELS.get(reason, reason), "count": count}
        for reason, count in diagnostics["constraint_hits"].items()
    ]
    constraint_hits.sort(key=lambda row: row["count"], reverse=True)

    per_city = {
        city: dict(data)
        for city, data in sorted(
            diagnostics["per_city"].items(),
            key=lambda item: (item[1].get("trades", 0), item[1].get("candidates", 0)),
            reverse=True,
        )
    }

    return {
        "candidates_evaluated": diagnostics["candidates_evaluated"],
        "trades_taken": diagnostics["trades_taken"],
        "rejected_total": diagnostics["rejected_total"],
        "top_reasons": reasons,
        "constraint_hits": constraint_hits,
        "per_city": per_city,
        "best_candidate": diagnostics.get("best_candidate"),
        "best_rejected": diagnostics.get("best_rejected"),
        "near_misses": diagnostics.get("near_misses", []),
        "thresholds": diagnostics.get("thresholds", {}),
        "description": (
            "Strategy gate diagnostics show why modeled bucket opportunities did or did not become simulated trades. "
            "Top reasons use the first blocking gate; constraint hits count every violated gate on the candidate."
        ),
    }


# ─── Gamma API enrichment ────────────────────────────────────────────────────

@dataclass
class EnrichmentResult:
    """Summary of one Gamma enrichment pass."""
    fetched: int              # total closed Gamma event rows pulled
    weather_matched: int      # passed slug filter
    stored: int               # inserted/updated in backtest_resolved_events
    matched_events: int       # cross-referenced an existing events row
    matched_metar: int        # found a MetarObs actual for that city/date
    matched_forecast: int     # found forecast data (ModelSnapshot or ForecastObs)
    last_enriched_at: datetime


def parse_gamma_slug(
    slug: str,
    fallback_year: Optional[int] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a Polymarket event slug.

    Examples:
        'highest-temperature-in-atlanta-on-april-12-2026'
            → ('high', 'atlanta', '2026-04-12')
        'lowest-temperature-in-new-york-on-april-12-2026'
            → ('low', 'new-york', '2026-04-12')
        'highest-temperature-in-nyc-on-december-30'  (no year)
            → ('high', 'new-york-city', f'{fallback_year}-12-30') if fallback given

    City aliases (nyc, new-york, sf, …) are normalized to canonical slugs.
    """
    if not slug:
        return None, None, None
    m = _SLUG_RE.search(slug)
    if not m:
        return None, None, None
    kind = "high" if m.group(1) == "highest" else "low"
    city_part = m.group(2)
    city_part = _CITY_ALIASES.get(city_part, city_part)
    month_num = _MONTH_MAP.get(m.group(3).lower())
    if not month_num:
        return None, None, None
    try:
        day = int(m.group(4))
    except ValueError:
        return None, None, None
    year_str = m.group(5)
    if year_str:
        try:
            year = int(year_str)
        except ValueError:
            return None, None, None
    elif fallback_year is not None:
        year = fallback_year
    else:
        return None, None, None
    date_et = f"{year:04d}-{month_num:02d}-{day:02d}"
    return kind, city_part, date_et


def _derive_parent_slug(market_slug: str) -> str:
    """Extract the parent event slug from a bucket market slug.

    'highest-temperature-in-atlanta-on-april-12-2026-65-69'
        → 'highest-temperature-in-atlanta-on-april-12-2026'
    """
    if not market_slug:
        return ""
    m = _SLUG_RE.search(market_slug)
    return m.group(0) if m else market_slug


def _parse_bucket_range(label: str) -> tuple[Optional[float], Optional[float]]:
    """Extract (lo_f, hi_f) from a bucket market question/slug.

    Handles '65-69°F', 'above 90°F', '60°F or lower', etc.
    Returns (None, None) if unparseable.
    """
    if not label:
        return None, None
    m = _BUCKET_RANGE_RE.search(label)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    m = _BUCKET_ABOVE_RE.search(label)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            return float(raw), None  # open-ended upper
        except (TypeError, ValueError):
            pass
    m = _BUCKET_BELOW_RE.search(label)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            return None, float(raw)  # open-ended lower
        except (TypeError, ValueError):
            pass
    return None, None


def _extract_winner(market: dict) -> tuple[Optional[str], Optional[float]]:
    """From a Polymarket market row, return (winning_outcome, final_price).

    Polymarket encodes resolution in `outcomePrices` — a JSON-string of
    [\"1\",\"0\"] or [\"0\",\"1\"]. The winning outcome is whichever index
    equals "1".
    """
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
    except (ValueError, TypeError):
        return None, None
    if not outcomes or len(outcomes) != len(prices):
        return None, None
    for name, p in zip(outcomes, prices):
        if p >= 0.99:
            return str(name), float(p)
    return None, None


async def _lookup_metar_actual(
    sess,
    city_slug: str,
    date_et: str,
    kind: str,
) -> Optional[float]:
    """Return the observed max/min temperature (°F) for a city on a given ET date.

    kind: 'high' → max(temp_f), 'low' → min(temp_f).

    Uses timezone-aware date windows instead of fragile substr/cast string
    matching, ensuring correct results across DB dialects and timezones.
    """
    city = (await sess.execute(
        select(City).where(City.city_slug == city_slug)
    )).scalar_one_or_none()
    if not city:
        return None

    # Build timezone-aware day window for the city
    from zoneinfo import ZoneInfo
    city_tz_str = getattr(city, 'tz', 'America/New_York') or 'America/New_York'
    try:
        tz = ZoneInfo(city_tz_str)
    except Exception:
        tz = ZoneInfo('America/New_York')

    try:
        start_dt = datetime.strptime(date_et, '%Y-%m-%d').replace(tzinfo=tz)
    except ValueError:
        return None
    from datetime import timedelta
    end_dt = start_dt + timedelta(days=1)

    agg = func.max(MetarObs.temp_f) if kind == "high" else func.min(MetarObs.temp_f)
    row = (await sess.execute(
        select(agg)
        .where(MetarObs.city_id == city.id)
        .where(MetarObs.temp_f.isnot(None))
        .where(MetarObs.observed_at >= start_dt)
        .where(MetarObs.observed_at < end_dt)
    )).scalar_one_or_none()

    if row is None:
        # Fall back: check daily_high_f column
        row = (await sess.execute(
            select(MetarObs.daily_high_f)
            .where(MetarObs.city_id == city.id)
            .where(MetarObs.daily_high_f.isnot(None))
            .where(MetarObs.observed_at >= start_dt)
            .where(MetarObs.observed_at < end_dt)
            .limit(1)
        )).scalar_one_or_none()

    try:
        return float(row) if row is not None else None
    except (TypeError, ValueError):
        return None


async def _lookup_forecast_prob(
    sess,
    city_slug: str,
    date_et: str,
    winning_idx: Optional[int],
    matched_event_id: Optional[int],
) -> tuple[bool, Optional[float]]:
    """Find the model's forecast probability for the winning bucket.

    Returns (has_forecast_data, model_prob_for_winner).
    - has_forecast_data: True if we have ANY ForecastObs or ModelSnapshot for
      this city/date.
    - model_prob_for_winner: the stored probability assigned to the winning
      bucket, or None if we cannot identify it.
    """
    prob: Optional[float] = None
    has_data = False

    if matched_event_id is not None and winning_idx is not None:
        snap = (await sess.execute(
            select(ModelSnapshot)
            .where(ModelSnapshot.event_id == matched_event_id)
            .order_by(desc(ModelSnapshot.computed_at))
            .limit(1)
        )).scalar_one_or_none()
        if snap and snap.probs_json:
            has_data = True
            try:
                probs = json.loads(snap.probs_json)
                if 0 <= winning_idx < len(probs):
                    prob = float(probs[winning_idx])
            except (ValueError, TypeError):
                pass

    if not has_data:
        city = (await sess.execute(
            select(City).where(City.city_slug == city_slug)
        )).scalar_one_or_none()
        if city:
            row = (await sess.execute(
                select(ForecastObs.id)
                .where(ForecastObs.city_id == city.id)
                .where(ForecastObs.date_et == date_et)
                .limit(1)
            )).scalar_one_or_none()
            if row is not None:
                has_data = True

    return has_data, prob


def _event_end_year(event_data: dict) -> Optional[int]:
    """Extract the year from an event's endDate/closedTime (ISO-8601 UTC)."""
    for key in ("endDate", "closedTime", "startDate"):
        raw = event_data.get(key)
        if not raw:
            continue
        try:
            # endDate is like "2025-12-30T12:00:00Z" — first 4 chars are year
            return int(str(raw)[:4])
        except (ValueError, TypeError):
            continue
    return None


async def enrich_from_gamma() -> EnrichmentResult:
    """Fetch closed weather markets from Gamma, resolve winners, cross-match.

    Uses `GET /events?closed=true&tag_slug=daily-temperature` which returns
    parent events with bucket markets already nested. For each event we:
      1. Parse the city + date (falling back to endDate if the slug omits year)
      2. Identify the winning bucket via each market's `outcomePrices` JSON
      3. Upsert a row into `backtest_resolved_events` and back-fill the local
         Event.winning_bucket_idx when a matching event exists.
      4. Cross-reference MetarObs / ForecastObs for enrichment metadata.
    """
    fetched = 0
    weather_matched = 0
    stored = 0
    matched_events = 0
    matched_metar = 0
    matched_forecast = 0
    now = datetime.now(timezone.utc)

    headers = {"User-Agent": "WeatherQuant/1.0 (contact@weatherquant.local)"}
    timeout = aiohttp.ClientTimeout(total=30)

    # Collect (event_slug, markets, event_data) from Gamma
    events_to_process: list[tuple[str, list[dict], dict]] = []

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as http:
        offset = 0
        page_size = GAMMA_PAGE_SIZE
        seen_event_slugs: set[str] = set()
        while True:
            url = f"{GAMMA_API}/events"
            params = {
                "closed": "true",
                "limit": str(page_size),
                "offset": str(offset),
                "tag_slug": "daily-temperature",
            }
            try:
                resp = await http.get(url, params=params)
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            except Exception as e:
                log.warning("gamma enrich: request failed offset=%d: %s", offset, e)
                break

            if not data or not isinstance(data, list):
                break

            new_rows_on_page = 0
            for ev in data:
                slug = ev.get("slug") or ""
                if slug in seen_event_slugs:
                    continue
                seen_event_slugs.add(slug)
                new_rows_on_page += 1
                fetched += 1
                if not (
                    "highest-temperature-in-" in slug
                    or "lowest-temperature-in-" in slug
                ):
                    continue
                weather_matched += 1
                markets = ev.get("markets") or []
                if not markets:
                    continue
                events_to_process.append((slug, markets, ev))

            if len(data) < page_size or new_rows_on_page == 0:
                break
            offset += len(data)

    # Persist + cross-reference
    async with get_session() as sess:
        for event_slug, markets, event_data in events_to_process:
            fallback_year = _event_end_year(event_data)
            kind, city_slug, date_et = parse_gamma_slug(event_slug, fallback_year)
            if not city_slug or not date_et:
                continue

            # Sort bucket markets by parsed low bound so the index order
            # matches the bot's own bucket convention (low → high).
            ranked: list[tuple[float, Optional[float], dict]] = []
            for m in markets:
                label = m.get("question") or m.get("slug", "")
                lo, hi = _parse_bucket_range(label)
                sort_key = lo if lo is not None else (hi if hi is not None else 1e9)
                ranked.append((sort_key, hi, m))
            ranked.sort(key=lambda r: r[0])

            buckets_out: list[dict] = []
            winning_idx: Optional[int] = None
            winning_label: Optional[str] = None
            final_price: Optional[float] = None
            resolved_outcome: Optional[str] = None

            for idx, (lo_sort, _hi, m) in enumerate(ranked):
                label = m.get("question") or m.get("slug", "")
                lo, hi = _parse_bucket_range(label)
                buckets_out.append({
                    "idx": idx,
                    "label": label,
                    "lo": lo,
                    "hi": hi,
                    "slug": m.get("slug", ""),
                })
                name, price = _extract_winner(m)
                if name is None or price is None:
                    continue
                if str(name).strip().lower() == "yes":
                    winning_idx = idx
                    winning_label = label
                    final_price = price
                    resolved_outcome = name

            match_status = "unmatched"
            matched_event_id: Optional[int] = None
            matched_actual_f: Optional[float] = None
            matched_forecast_prob: Optional[float] = None

            # 1) Existing local events row
            ev = (await sess.execute(
                select(Event).join(City, Event.city_id == City.id)
                .where(City.city_slug == city_slug)
                .where(Event.date_et == date_et)
            )).scalar_one_or_none()
            if ev is not None:
                matched_event_id = ev.id
                match_status = "matched_event"
                matched_events += 1
                if ev.winning_bucket_idx is None and winning_idx is not None:
                    ev.winning_bucket_idx = winning_idx
                    if ev.resolved_at is None:
                        ev.resolved_at = now

            # 2) MetarObs actual
            actual = await _lookup_metar_actual(sess, city_slug, date_et, kind or "high")
            if actual is not None:
                matched_actual_f = actual
                if match_status == "unmatched":
                    match_status = "matched_metar"
                matched_metar += 1

            # 3) Forecast data
            has_forecast, fprob = await _lookup_forecast_prob(
                sess, city_slug, date_et, winning_idx, matched_event_id
            )
            if fprob is not None:
                matched_forecast_prob = fprob
            if has_forecast:
                if match_status == "unmatched":
                    match_status = "matched_forecast"
                matched_forecast += 1

            # Upsert BacktestResolvedEvent
            existing = (await sess.execute(
                select(BacktestResolvedEvent)
                .where(BacktestResolvedEvent.event_slug == event_slug)
            )).scalar_one_or_none()

            market_slug = markets[0].get("slug", "") if markets else ""
            buckets_json = json.dumps(buckets_out)

            if existing is not None:
                existing.market_slug = market_slug
                existing.city_slug = city_slug
                existing.date_et = date_et
                existing.market_kind = kind or "high"
                existing.winning_bucket_idx = winning_idx
                existing.winning_bucket_label = winning_label
                existing.final_price = final_price
                existing.resolved_outcome = resolved_outcome
                existing.buckets_json = buckets_json
                existing.matched_event_id = matched_event_id
                existing.matched_metar_actual_f = matched_actual_f
                existing.matched_forecast_prob = matched_forecast_prob
                existing.match_status = match_status
                existing.enriched_at = now
                if existing.closed_at is None:
                    existing.closed_at = now
            else:
                sess.add(BacktestResolvedEvent(
                    event_slug=event_slug,
                    market_slug=market_slug,
                    city_slug=city_slug,
                    date_et=date_et,
                    market_kind=kind or "high",
                    winning_bucket_idx=winning_idx,
                    winning_bucket_label=winning_label,
                    final_price=final_price,
                    resolved_outcome=resolved_outcome,
                    buckets_json=buckets_json,
                    matched_event_id=matched_event_id,
                    matched_metar_actual_f=matched_actual_f,
                    matched_forecast_prob=matched_forecast_prob,
                    match_status=match_status,
                    enriched_at=now,
                    closed_at=now,
                ))
            stored += 1

        await sess.commit()

    log.info(
        "gamma enrich: fetched=%d weather=%d stored=%d events=%d metar=%d forecast=%d",
        fetched, weather_matched, stored, matched_events, matched_metar, matched_forecast,
    )
    return EnrichmentResult(
        fetched=fetched,
        weather_matched=weather_matched,
        stored=stored,
        matched_events=matched_events,
        matched_metar=matched_metar,
        matched_forecast=matched_forecast,
        last_enriched_at=now,
    )


# ─── Data loading ────────────────────────────────────────────────────────────

async def get_enrichment_status() -> dict:
    """Return summary stats for the backtest dashboard's "last enriched" banner."""
    async with get_session() as sess:
        latest = (await sess.execute(
            select(BacktestResolvedEvent)
            .order_by(desc(BacktestResolvedEvent.enriched_at))
            .limit(1)
        )).scalar_one_or_none()
        total = (await sess.execute(
            select(func.count(BacktestResolvedEvent.id))
        )).scalar_one() or 0
        matched = (await sess.execute(
            select(func.count(BacktestResolvedEvent.id))
            .where(BacktestResolvedEvent.match_status != "unmatched")
        )).scalar_one() or 0
        matched_event = (await sess.execute(
            select(func.count(BacktestResolvedEvent.id))
            .where(BacktestResolvedEvent.match_status == "matched_event")
        )).scalar_one() or 0
        matched_metar = (await sess.execute(
            select(func.count(BacktestResolvedEvent.id))
            .where(BacktestResolvedEvent.match_status == "matched_metar")
        )).scalar_one() or 0
        matched_forecast = (await sess.execute(
            select(func.count(BacktestResolvedEvent.id))
            .where(BacktestResolvedEvent.match_status == "matched_forecast")
        )).scalar_one() or 0

    return {
        "last_enriched_at": latest.enriched_at.isoformat() if latest else None,
        "total_markets": int(total),
        "matched_markets": int(matched),
        "matched_events": int(matched_event),
        "matched_metar": int(matched_metar),
        "matched_forecast": int(matched_forecast),
        "unmatched_markets": max(0, int(total) - int(matched)),
    }


async def count_resolved_sources() -> dict:
    """Quick counts used for the /api/backtest/run debug print."""
    async with get_session() as sess:
        local = (await sess.execute(
            select(func.count(Event.id))
            .where(Event.winning_bucket_idx.isnot(None))
        )).scalar_one() or 0
        gamma = (await sess.execute(
            select(func.count(BacktestResolvedEvent.id))
            .where(BacktestResolvedEvent.winning_bucket_idx.isnot(None))
        )).scalar_one() or 0
    return {"resolved_local": int(local), "resolved_gamma": int(gamma)}


def _event_is_past_local_day(event: Event, city: City, *, now_utc: datetime | None = None) -> bool:
    now = now_utc or datetime.now(timezone.utc)
    try:
        city_tz = ZoneInfo(getattr(city, "tz", None) or "America/New_York")
        event_date = datetime.strptime(event.date_et, "%Y-%m-%d").date()
    except Exception:
        return False
    return event_date < now.astimezone(city_tz).date()


def _winner_idx_from_high(buckets: list[Bucket], high_f: float | None) -> int | None:
    if high_f is None or not buckets:
        return None
    ranges = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
    return find_bucket_idx_for_value(ranges, float(high_f))


async def _prefetch_observed_highs(
    sess,
    event_city_rows: list[tuple[Event, City]],
) -> dict[tuple[int, str], float]:
    """Bulk-load METAR daily highs keyed by (city_id, local YYYY-MM-DD)."""
    if not event_city_rows:
        return {}
    city_ids = sorted({int(city.id) for _, city in event_city_rows})
    date_vals = []
    for event, _ in event_city_rows:
        try:
            date_vals.append(datetime.strptime(event.date_et, "%Y-%m-%d").date())
        except Exception:
            continue
    if not city_ids or not date_vals:
        return {}

    # Broad UTC window; local-date assignment happens in Python per city TZ.
    start_utc = datetime.combine(min(date_vals), datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=1)
    end_utc = datetime.combine(max(date_vals), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=2)
    tz_by_city_id: dict[int, ZoneInfo] = {}
    for _, city in event_city_rows:
        if city.id in tz_by_city_id:
            continue
        try:
            tz_by_city_id[int(city.id)] = ZoneInfo(getattr(city, "tz", None) or "America/New_York")
        except Exception:
            tz_by_city_id[int(city.id)] = ZoneInfo("America/New_York")

    rows = (
        await sess.execute(
            select(MetarObs.city_id, MetarObs.observed_at, MetarObs.temp_f)
            .where(
                MetarObs.city_id.in_(city_ids),
                MetarObs.temp_f.isnot(None),
                MetarObs.observed_at >= start_utc,
                MetarObs.observed_at < end_utc,
            )
        )
    ).all()

    highs: dict[tuple[int, str], float] = {}
    for city_id, observed_at, temp_f in rows:
        if observed_at is None or temp_f is None:
            continue
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        tz = tz_by_city_id.get(int(city_id), ZoneInfo("America/New_York"))
        date_key = observed_at.astimezone(tz).date().isoformat()
        key = (int(city_id), date_key)
        temp = float(temp_f)
        if key not in highs or temp > highs[key]:
            highs[key] = temp
    return highs


async def _derive_local_resolution(
    sess,
    *,
    event: Event,
    city: City,
    buckets: list[Bucket],
    observed_highs: dict[tuple[int, str], float] | None = None,
    allow_slow_fallback: bool = True,
) -> tuple[int | None, float | None, str | None]:
    """Derive a provisional local winner from station/WU settlement data.

    Polymarket/Gamma outcomes are preferred when present, but the bot already
    stores enough station history to score past tracked events before Gamma has
    matched them. This is what makes `/backtest` useful during live operation.
    """
    if event.winning_bucket_idx is not None:
        high = (observed_highs or {}).get((int(city.id), event.date_et))
        if high is None and allow_slow_fallback:
            settlement = await resolve_canonical_settlement_high(
                sess,
                city=city,
                event=event,
                validate_polymarket_winner=False,
            )
            high = settlement.get("high_f")
        return int(event.winning_bucket_idx), (float(high) if high is not None else None), "gamma_confirmed"

    if not _event_is_past_local_day(event, city):
        return None, None, None

    source_used = "station_metar"
    high = (observed_highs or {}).get((int(city.id), event.date_et))
    if high is None and allow_slow_fallback:
        settlement = await resolve_canonical_settlement_high(
            sess,
            city=city,
            event=event,
            validate_polymarket_winner=False,
        )
        high = settlement.get("high_f")
        source_used = settlement.get("source_used") or "local_settlement"
    if high is None:
        return None, None, None
    winner_idx = _winner_idx_from_high(buckets, float(high))
    if winner_idx is None:
        return None, float(high), f"{source_used}_unmapped"
    return int(winner_idx), float(high), source_used


def _as_float(raw) -> Optional[float]:
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_probability(raw) -> Optional[float]:
    val = _as_float(raw)
    if val is None or val <= 0.0 or val >= 1.0:
        return None
    return val


def _normalise_entry_market_row(row: dict) -> dict:
    """Build a usable entry orderbook snapshot from stored MarketSnapshot fields.

    Older rows can have null/zero `yes_mid` even when bid/ask is stored. The
    strategy simulator needs a market probability, so derive mid from bid/ask
    before deciding the row is untradable.
    """
    yes_bid = _as_probability(row.get("yes_bid"))
    yes_ask = _as_probability(row.get("yes_ask"))
    yes_mid = _as_probability(row.get("yes_mid"))
    if yes_mid is None:
        if yes_bid is not None and yes_ask is not None:
            yes_mid = (yes_bid + yes_ask) / 2.0
        elif yes_ask is not None:
            yes_mid = yes_ask
        elif yes_bid is not None:
            yes_mid = yes_bid

    spread = _as_float(row.get("spread"))
    if spread is None and yes_bid is not None and yes_ask is not None:
        spread = max(0.0, yes_ask - yes_bid)

    return {
        "yes_mid": yes_mid,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "spread": spread,
        "bid_depth": _as_float(row.get("yes_bid_depth")) or 0.0,
        "ask_depth": _as_float(row.get("yes_ask_depth")) or 0.0,
    }


def _aware_utc(dt: datetime | None) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _city_tz(city: City) -> ZoneInfo:
    try:
        return ZoneInfo(getattr(city, "tz", None) or "America/New_York")
    except Exception:
        return ZoneInfo("America/New_York")


def _select_actionable_model_snapshot(
    event: Event,
    city: City,
    snapshots: list[ModelSnapshot],
) -> Optional[ModelSnapshot]:
    """Pick a tradable decision checkpoint instead of the latest post-close row."""
    if not snapshots:
        return None
    try:
        event_date = datetime.strptime(event.date_et, "%Y-%m-%d").date()
    except Exception:
        return max(snapshots, key=lambda snap: _aware_utc(snap.computed_at) or datetime.min.replace(tzinfo=timezone.utc))

    tz = _city_tz(city)
    annotated = []
    for snap in snapshots:
        computed_at = _aware_utc(getattr(snap, "computed_at", None))
        if computed_at is None:
            continue
        annotated.append((computed_at, computed_at.astimezone(tz), snap))
    if not annotated:
        return None
    annotated.sort(key=lambda row: row[0])

    # Regular quick-flip checkpoint: latest same-day snapshot from 7-10 AM local.
    morning = [
        row for row in annotated
        if row[1].date() == event_date and 7 <= (row[1].hour + row[1].minute / 60.0) <= 10
    ]
    if morning:
        return morning[-1][2]

    # Night Owl checkpoint: latest snapshot from previous 11 PM through 6 AM local.
    overnight = [
        row for row in annotated
        if (
            (row[1].date() == event_date - timedelta(days=1) and row[1].hour >= 23)
            or (row[1].date() == event_date and row[1].hour < 6)
        )
    ]
    if overnight:
        return overnight[-1][2]

    # If the market was first tracked late, use the earliest regular-session row
    # before pre-close. This avoids selecting post-resolution 0.001/0.999 books.
    regular_session = [
        row for row in annotated
        if row[1].date() == event_date and 6 <= (row[1].hour + row[1].minute / 60.0) <= 15
    ]
    if regular_session:
        return regular_session[0][2]

    pre_close = [
        row for row in annotated
        if row[1].date() == event_date and (row[1].hour + row[1].minute / 60.0) < 19.5
    ]
    if pre_close:
        return pre_close[-1][2]

    # Last resort: latest row before or on the event's local date. Never prefer a
    # post-event-day row over an earlier potentially tradable snapshot.
    non_future = [row for row in annotated if row[1].date() <= event_date]
    return (non_future[-1][2] if non_future else annotated[0][2])


async def get_coverage_breakdown() -> dict:
    """Three-tier coverage report driving the /backtest banner.

    Tiers:
      - replayable:    local Event with winning_bucket_idx + ≥1 ModelSnapshot
                       AND every bucket has ≥1 MarketSnapshot. Full simulation.
      - brier_only:    same as replayable minus the MarketSnapshot requirement.
                       Forecast accuracy can be measured but no trade simulation.
      - outcome_only:  Gamma-resolved markets with no usable local snapshots.
                       Reference data only — the simulator skips these.

    Also reports raw model_snapshots/market_snapshots counts so that an empty
    DB lights up "scheduler not running" diagnostics on the page itself.
    """
    train = BacktestParams.walk_forward_train_days
    test = BacktestParams.walk_forward_test_days
    needed_for_walkforward = int(train) + int(test)

    async with get_session() as sess:
        # Raw snapshot row counts — fast, helps spot a dead scheduler
        n_model = (await sess.execute(
            select(func.count(ModelSnapshot.id))
        )).scalar_one() or 0
        n_market = (await sess.execute(
            select(func.count(MarketSnapshot.id))
        )).scalar_one() or 0

        # Events with at least one ModelSnapshot. A local station-derived
        # winner makes past tracked events replayable even before Gamma/UMA has
        # cross-matched the market outcome into Event.winning_bucket_idx.
        events_with_model_q = (
            select(Event, City)
            .join(City, Event.city_id == City.id)
            .where(Event.winning_bucket_idx.isnot(None))
            .where(Event.id.in_(select(ModelSnapshot.event_id).distinct()))
        )
        local_events_with_model_q = (
            select(Event, City)
            .join(City, Event.city_id == City.id)
            .where(Event.id.in_(select(ModelSnapshot.event_id).distinct()))
        )
        confirmed_events = (await sess.execute(events_with_model_q)).all()
        local_events_with_model = (await sess.execute(local_events_with_model_q)).all()
        n_with_model = len(local_events_with_model)
        event_ids = [event.id for event, _ in local_events_with_model]
        buckets_by_event_id: dict[int, list[Bucket]] = {}
        covered_counts_by_event_id: dict[int, int] = {}
        if event_ids:
            bucket_rows = (
                await sess.execute(
                    select(Bucket)
                    .where(Bucket.event_id.in_(event_ids))
                    .order_by(Bucket.event_id, Bucket.bucket_idx)
                )
            ).scalars().all()
            for bucket in bucket_rows:
                buckets_by_event_id.setdefault(bucket.event_id, []).append(bucket)

            coverage_rows = (
                await sess.execute(
                    select(
                        Bucket.event_id,
                        func.count(func.distinct(MarketSnapshot.bucket_id)).label("covered"),
                    )
                    .join(MarketSnapshot, MarketSnapshot.bucket_id == Bucket.id)
                    .where(Bucket.event_id.in_(event_ids))
                    .group_by(Bucket.event_id)
                )
            ).all()
            covered_counts_by_event_id = {
                int(event_id): int(covered or 0)
                for event_id, covered in coverage_rows
            }
        observed_highs = await _prefetch_observed_highs(sess, local_events_with_model)

        # Of those, how many have ≥1 MarketSnapshot per bucket?
        # Tradable iff: every bucket of the event has at least one snapshot.
        replayable = 0
        replayable_dates: list[str] = []
        brier_only = 0
        local_station_resolved = 0
        local_outcome_pending = 0
        for event, city in local_events_with_model:
            bucket_rows = buckets_by_event_id.get(event.id, [])
            if not bucket_rows:
                continue
            winner_idx, _, status = await _derive_local_resolution(
                sess,
                event=event,
                city=city,
                buckets=list(bucket_rows),
                observed_highs=observed_highs,
                allow_slow_fallback=False,
            )
            if winner_idx is None:
                local_outcome_pending += 1
                continue
            if status != "gamma_confirmed":
                local_station_resolved += 1

            bucket_count = len(bucket_rows)
            if bucket_count == 0:
                continue
            covered_buckets = covered_counts_by_event_id.get(event.id, 0)
            if covered_buckets >= bucket_count:
                replayable += 1
                if event.date_et:
                    replayable_dates.append(event.date_et)
            else:
                brier_only += 1

        # Gamma-resolved markets minus what's already covered locally
        gamma_resolved = (await sess.execute(
            select(func.count(BacktestResolvedEvent.id))
            .where(BacktestResolvedEvent.winning_bucket_idx.isnot(None))
        )).scalar_one() or 0
        outcome_only = max(0, int(gamma_resolved) - replayable - brier_only)

    earliest = min(replayable_dates) if replayable_dates else None
    latest = max(replayable_dates) if replayable_dates else None

    return {
        "replayable": int(replayable),
        "brier_only": int(brier_only),
        "outcome_only": int(outcome_only),
        "model_snapshots": int(n_model),
        "market_snapshots": int(n_market),
        "earliest_snapshot": earliest,
        "latest_snapshot": latest,
        "local_events_with_model": int(n_with_model),
        "local_gamma_confirmed": int(len(confirmed_events)),
        "local_station_resolved": int(local_station_resolved),
        "local_outcome_pending": int(local_outcome_pending),
        "needs_more_days": int(replayable) < needed_for_walkforward,
        "walkforward_threshold": needed_for_walkforward,
    }


async def fetch_resolved_events() -> list[dict]:
    """Load all resolved events with their buckets, model snapshots, and market snapshots.

    Only returns events that have the data needed to replay simulation
    (ModelSnapshot + MarketSnapshot per bucket). BacktestResolvedEvent rows
    without a matching local Event cannot be simulated, but they already
    backfill Event.winning_bucket_idx during enrich_from_gamma() — so
    more events become "resolved" after enrichment without code changes here.
    """
    async with get_session() as sess:
        # Get events with known or locally derivable outcomes. We do not require
        # Event.winning_bucket_idx because Gamma enrichment can lag local station
        # data by days, while backtesting only needs the settled bucket.
        query = (
            select(Event, City)
            .join(City, Event.city_id == City.id)
            .where(Event.id.in_(select(ModelSnapshot.event_id).distinct()))
            .order_by(Event.date_et)
        )
        rows = (await sess.execute(query)).all()
        observed_highs = await _prefetch_observed_highs(sess, rows)
        event_ids = [event.id for event, _ in rows]
        buckets_by_event_id: dict[int, list[Bucket]] = {}
        latest_snap_by_event_id: dict[int, ModelSnapshot] = {}
        if event_ids:
            bucket_rows = (
                await sess.execute(
                    select(Bucket)
                    .where(Bucket.event_id.in_(event_ids))
                    .order_by(Bucket.event_id, Bucket.bucket_idx)
                )
            ).scalars().all()
            for bucket in bucket_rows:
                buckets_by_event_id.setdefault(bucket.event_id, []).append(bucket)

            snap_rows = (
                await sess.execute(
                    select(ModelSnapshot)
                    .where(ModelSnapshot.event_id.in_(event_ids))
                    .order_by(ModelSnapshot.event_id, ModelSnapshot.computed_at, ModelSnapshot.id)
                )
            ).scalars().all()
            snapshots_by_event_id: dict[int, list[ModelSnapshot]] = defaultdict(list)
            for snap in snap_rows:
                snapshots_by_event_id[int(snap.event_id)].append(snap)
            city_by_event_id = {int(event.id): city for event, city in rows}
            event_by_id = {int(event.id): event for event, _ in rows}
            latest_snap_by_event_id = {}
            for event_id, snap_list in snapshots_by_event_id.items():
                event = event_by_id.get(event_id)
                city = city_by_event_id.get(event_id)
                if event is None or city is None:
                    continue
                selected = _select_actionable_model_snapshot(event, city, snap_list)
                if selected is not None:
                    latest_snap_by_event_id[event_id] = selected

        entry_market_by_bucket_id: dict[int, dict] = {}
        later_max_bid_by_bucket_id: dict[int, float] = {}
        snap_ids = [snap.id for snap in latest_snap_by_event_id.values()]
        if snap_ids:
            snap_times_sub = (
                select(
                    ModelSnapshot.event_id.label("event_id"),
                    ModelSnapshot.computed_at.label("computed_at"),
                )
                .where(ModelSnapshot.id.in_(snap_ids))
                .subquery()
            )
            entry_market_sub = (
                select(
                    MarketSnapshot.bucket_id.label("bucket_id"),
                    MarketSnapshot.yes_mid.label("yes_mid"),
                    MarketSnapshot.yes_bid.label("yes_bid"),
                    MarketSnapshot.yes_ask.label("yes_ask"),
                    MarketSnapshot.spread.label("spread"),
                    MarketSnapshot.yes_bid_depth.label("yes_bid_depth"),
                    MarketSnapshot.yes_ask_depth.label("yes_ask_depth"),
                    func.row_number().over(
                        partition_by=MarketSnapshot.bucket_id,
                        order_by=(MarketSnapshot.fetched_at.desc(), MarketSnapshot.id.desc()),
                    ).label("rn"),
                )
                .select_from(MarketSnapshot)
                .join(Bucket, Bucket.id == MarketSnapshot.bucket_id)
                .join(snap_times_sub, Bucket.event_id == snap_times_sub.c.event_id)
                .where(MarketSnapshot.fetched_at <= snap_times_sub.c.computed_at)
                .subquery()
            )
            entry_rows = (
                await sess.execute(
                    select(entry_market_sub).where(entry_market_sub.c.rn == 1)
                )
            ).mappings().all()
            for row in entry_rows:
                entry_market_by_bucket_id[int(row["bucket_id"])] = _normalise_entry_market_row(row)

            later_rows = (
                await sess.execute(
                    select(
                        MarketSnapshot.bucket_id,
                        func.max(MarketSnapshot.yes_bid).label("max_bid"),
                    )
                    .select_from(MarketSnapshot)
                    .join(Bucket, Bucket.id == MarketSnapshot.bucket_id)
                    .join(snap_times_sub, Bucket.event_id == snap_times_sub.c.event_id)
                    .where(
                        MarketSnapshot.fetched_at > snap_times_sub.c.computed_at,
                        MarketSnapshot.yes_bid.isnot(None),
                    )
                    .group_by(MarketSnapshot.bucket_id)
                )
            ).all()
            later_max_bid_by_bucket_id = {
                int(bucket_id): float(max_bid)
                for bucket_id, max_bid in later_rows
                if max_bid is not None
            }

        events_data = []
        for event, city in rows:
            buckets = buckets_by_event_id.get(event.id, [])
            if not buckets:
                continue
            winner_idx, resolved_high_f, settlement_status = await _derive_local_resolution(
                sess,
                event=event,
                city=city,
                buckets=list(buckets),
                observed_highs=observed_highs,
                allow_slow_fallback=False,
            )
            if winner_idx is None:
                continue

            # Latest model snapshot per event (the one that would drive trading)
            snap = latest_snap_by_event_id.get(event.id)
            if not snap:
                continue

            # Load market snapshots for each bucket (latest before model snapshot)
            mkt_snaps = {}
            for b in buckets:
                ms = entry_market_by_bucket_id.get(b.id)
                if ms:
                    mkt_snaps[b.bucket_idx] = ms

            # Summarized later market path for quick-flip detection. Full
            # snapshot replay is too expensive for live use once the table has
            # millions of rows; max bid preserves "did the target trade?"
            # without scanning every later orderbook row per bucket.
            all_mkt_snaps = {}
            for b in buckets:
                max_bid = later_max_bid_by_bucket_id.get(b.id)
                all_mkt_snaps[b.bucket_idx] = (
                    [{
                        "yes_bid": max_bid,
                        "yes_ask": None,
                        "yes_bid_depth": None,
                        "yes_ask_depth": None,
                        "spread": None,
                        "fetched_at": None,
                    }]
                    if max_bid is not None else []
                )

            probs = json.loads(snap.probs_json) if snap.probs_json else []

            # Q7 — pull regime label from the snapshot inputs (set by signal_engine
            # at compute time). Lets _build_results stratify metrics by regime.
            snap_inputs: dict = {}
            _snap_regime_label: Optional[str] = None
            try:
                _inp = json.loads(snap.inputs_json) if snap.inputs_json else {}
                snap_inputs = _inp if isinstance(_inp, dict) else {}
                _rl = snap_inputs.get("regime_label")
                if _rl:
                    _snap_regime_label = str(_rl)
            except Exception:
                pass

            events_data.append({
                "event_id": event.id,
                "city_slug": city.city_slug,
                "city_display": city.display_name,
                "city_tz": city.tz or "America/New_York",
                "is_us": bool(city.is_us),
                "metar_station": city.metar_station,
                "date_et": event.date_et,
                "winning_bucket_idx": winner_idx,
                "settlement_status": settlement_status or "local_settlement",
                "resolved_high_f": resolved_high_f,
                "buckets": [
                    {
                        "idx": b.bucket_idx,
                        "bucket_idx": b.bucket_idx,
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
                "model_inputs": snap_inputs,
                "snapshot_time": snap.computed_at,
                "regime_label": _snap_regime_label,
            })

    log.info("backtest: loaded %d resolved events", len(events_data))
    return events_data


# ─── Trade simulation ────────────────────────────────────────────────────────

def _estimate_slippage(shares: float, ask_depth: float, base_bps: float = 50.0) -> float:
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


def simulate_entry(
    event_data: dict,
    bucket_idx: int,
    params: BacktestParams,
    portfolio: Portfolio,
    diagnostics: Optional[dict] = None,
) -> Optional[SimTrade]:
    """Evaluate one bucket for a trade entry. Returns SimTrade if taken."""
    probs = event_data["model_probs"]
    mkt = event_data["market_data"].get(bucket_idx)
    bucket_info = next((b for b in event_data["buckets"] if b["idx"] == bucket_idx), None)

    if not mkt or not bucket_info or bucket_idx >= len(probs):
        candidate = _candidate_snapshot(event_data, bucket_idx, reason="missing_market")
        _record_gate_candidate(diagnostics, event_data, candidate)
        _record_gate_rejection(diagnostics, event_data, candidate, "missing_market")
        return None

    model_prob = probs[bucket_idx]
    mkt_prob = mkt["yes_mid"]
    spread = mkt.get("spread")
    ask_depth = mkt.get("ask_depth", 0.0)

    if mkt_prob is None or mkt_prob <= 0:
        candidate = _candidate_snapshot(
            event_data,
            bucket_idx,
            model_prob=model_prob,
            mkt_prob=mkt_prob,
            spread=spread,
            ask_depth=ask_depth,
            reason="invalid_market_probability",
        )
        _record_gate_candidate(diagnostics, event_data, candidate)
        _record_gate_rejection(diagnostics, event_data, candidate, "invalid_market_probability")
        return None

    exec_cost = _execution_cost(spread, ask_depth)
    true_edge = model_prob - mkt_prob - exec_cost
    entry_price = mkt.get("yes_ask") or mkt_prob
    actuals = {
        "true_edge": true_edge,
        "entry_price": entry_price,
        "spread": spread,
        "ask_depth": ask_depth,
    }
    candidate = _candidate_snapshot(
        event_data,
        bucket_idx,
        model_prob=model_prob,
        mkt_prob=mkt_prob,
        true_edge=true_edge,
        spread=spread,
        ask_depth=ask_depth,
        entry_price=entry_price,
    )
    _record_gate_candidate(diagnostics, event_data, candidate)

    # Apply observable gates and keep all constraint hits for diagnostics. The
    # primary rejection reason remains the first gate that would block execution.
    violations: list[str] = []
    if true_edge < params.min_true_edge:
        violations.append("edge_below_min")
    if mkt_prob < 0.02 or mkt_prob > 0.98:
        violations.append("market_prob_out_of_range")
    if ask_depth < params.min_liquidity_shares:
        violations.append("insufficient_ask_depth")
    if entry_price > params.max_entry_price:
        violations.append("entry_price_above_max")
    if spread is not None and spread > params.max_spread:
        violations.append("spread_above_max")
    if violations:
        reason = violations[0]
        actual, required = _gate_actual_required(reason, actuals=actuals, params=params)
        rejected = {
            **candidate,
            "reason": reason,
            "reason_label": _GATE_REASON_LABELS.get(reason, reason),
            "violations": violations,
            "actual": round(actual, 4) if actual is not None else None,
            "required": round(required, 4) if required is not None else None,
        }
        if actual is not None and required is not None:
            rejected["gap"] = round(abs(actual - required), 4)
        _record_gate_rejection(diagnostics, event_data, rejected, reason, violations)
        return None

    # Apply slippage model: linear market impact for thin Polymarket books
    slippage_pct = _estimate_slippage(1.0, ask_depth)  # estimate for 1 share first

    # Check position limits
    event_key = event_data["event_id"]
    if portfolio.positions_per_event[event_key] >= params.max_positions_per_event:
        rejected = {**candidate, "reason": "max_positions_per_event"}
        _record_gate_rejection(diagnostics, event_data, rejected, "max_positions_per_event")
        return None

    # Kelly sizing
    kelly_f = calculate_kelly_fraction(
        model_prob=model_prob,
        yes_price=entry_price,
        fractional_kelly=params.kelly_fraction,
        max_position_size=params.max_position_pct,
    )
    if kelly_f <= 0:
        rejected = {**candidate, "reason": "kelly_non_positive"}
        _record_gate_rejection(diagnostics, event_data, rejected, "kelly_non_positive")
        return None

    effective_bankroll = portfolio.available()
    kelly_size = kelly_f * effective_bankroll
    position_cap = effective_bankroll * params.max_position_pct
    final_size = min(kelly_size, position_cap, effective_bankroll)

    shares = math.floor((final_size / entry_price) * 100) / 100

    # Refine slippage with actual share count
    slippage_pct = _estimate_slippage(shares, ask_depth)
    entry_price_slipped = entry_price * (1.0 + slippage_pct)

    cost = round(shares * entry_price_slipped, 4)
    if cost < 0.50:  # minimum trade size
        actuals["cost"] = cost
        actual, required = _gate_actual_required("trade_too_small", actuals=actuals, params=params)
        rejected = {
            **candidate,
            "reason": "trade_too_small",
            "reason_label": _GATE_REASON_LABELS["trade_too_small"],
            "actual": round(actual, 4) if actual is not None else None,
            "required": round(required, 4) if required is not None else None,
        }
        if actual is not None and required is not None:
            rejected["gap"] = round(abs(actual - required), 4)
        _record_gate_rejection(diagnostics, event_data, rejected, "trade_too_small")
        return None

    # Execute
    portfolio.bankroll -= cost
    portfolio.positions_per_event[event_key] += 1
    _record_gate_accept(
        diagnostics,
        event_data,
        {
            **candidate,
            "entry_price": round(entry_price_slipped, 4),
            "shares": shares,
            "cost": cost,
        },
    )

    return SimTrade(
        city_slug=event_data["city_slug"],
        date_et=event_data["date_et"],
        bucket_idx=bucket_idx,
        bucket_label=bucket_info["label"],
        model_prob=round(model_prob, 4),
        mkt_prob=round(mkt_prob, 4),
        true_edge=round(true_edge, 4),
        side="buy_yes",
        entry_price=round(entry_price_slipped, 4),
        shares=shares,
        cost=cost,
        regime_label=event_data.get("regime_label"),
    )


def _obs_reference_temp_from_inputs(model_inputs: dict) -> Optional[float]:
    if not isinstance(model_inputs, dict):
        return None
    adaptive = model_inputs.get("adaptive") if isinstance(model_inputs.get("adaptive"), dict) else {}
    for raw in (
        model_inputs.get("current_temp_f"),
        adaptive.get("predicted_daily_high"),
        model_inputs.get("projected_high_for_blend"),
        model_inputs.get("projected_high"),
        model_inputs.get("observed_high"),
        model_inputs.get("ground_truth_high"),
    ):
        try:
            if raw is not None:
                return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _snapshot_local_time(snap: dict, city_tz: str) -> Optional[datetime]:
    fetched_at = snap.get("fetched_at")
    if not isinstance(fetched_at, datetime):
        return None
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    try:
        tz = ZoneInfo(city_tz or "America/New_York")
    except Exception:
        tz = ZoneInfo("America/New_York")
    return fetched_at.astimezone(tz)


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

        # 1. Optional observation-proximity replay. This is a lightweight
        # approximation using the stored model inputs and later orderbook
        # snapshots, not a full intraday model-state replay.
        if params.obs_exit_enabled:
            model_inputs = event_data.get("model_inputs") or {}
            reference_temp = _obs_reference_temp_from_inputs(model_inputs)
            observation_minutes = (
                model_inputs.get("observation_minutes")
                if isinstance(model_inputs, dict)
                else None
            )
            station_id = (
                model_inputs.get("active_station_id")
                if isinstance(model_inputs, dict)
                else None
            ) or event_data.get("metar_station")

            for snap in later_mkt:
                bid = snap.get("yes_bid")
                if bid is None:
                    continue
                now_local = _snapshot_local_time(
                    snap,
                    event_data.get("city_tz") or "America/New_York",
                )
                if now_local is None:
                    continue
                net_pnl_per_share = float(bid) - trade.entry_price - (float(bid) * 0.02)
                decision = evaluate_obs_proximity_exit(
                    city_slug=trade.city_slug,
                    station_id=station_id,
                    now_local=now_local,
                    observation_minutes=observation_minutes,
                    bucket_specs=event_data.get("buckets", []),
                    held_bucket_idx=trade.bucket_idx,
                    reference_temp_f=reference_temp,
                    yes_bid=bid,
                    yes_ask=snap.get("yes_ask"),
                    yes_bid_depth=snap.get("yes_bid_depth"),
                    yes_ask_depth=snap.get("yes_ask_depth"),
                    net_pnl_per_share=net_pnl_per_share,
                    current_edge=trade.true_edge,
                    enabled=True,
                    is_us=bool(event_data.get("is_us", True)),
                    window_minutes=params.obs_exit_window_minutes,
                    temp_sensitivity_threshold_f=params.temp_sensitivity_threshold_f,
                    min_profit_cents=params.obs_min_profit_cents,
                    min_depth_usd=params.obs_min_depth_usd,
                    max_orderbook_imbalance=params.obs_max_orderbook_imbalance,
                    cooldown_active=False,
                )
                if decision.get("final_action") == "EXIT":
                    trade.won = True
                    payout = trade.shares * float(bid)
                    fee = payout * 0.02
                    trade.pnl = round(payout - fee - trade.cost, 4)
                    trade.exit_reason = "obs_proximity"
                    break

        if trade.won is not None:
            continue

        # 2. Quick flip check — did bid ever reach entry + target?
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

        # 3. Resolution — hold to maturity
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
    diagnostics: Optional[dict] = None,
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
            elif diagnostics is not None:
                simulate_entry(event_data, i, params, portfolio, diagnostics=diagnostics)
        bucket_edges.sort(reverse=True)

        for _, bucket_idx in bucket_edges:
            trade = simulate_entry(event_data, bucket_idx, params, portfolio, diagnostics=diagnostics)
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


def _mean_brier(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    return float(sum((float(p) - int(outcome)) ** 2 for p, outcome in pairs) / len(pairs))


def _source_score(label: str, pairs: list[tuple[float, int]], market_brier: float | None) -> dict:
    brier = _mean_brier(pairs)
    edge = None if brier is None or market_brier is None else market_brier - brier
    return {
        "label": label,
        "brier": round(brier, 6) if brier is not None else None,
        "edge_vs_market": round(edge, 6) if edge is not None else None,
        "edge_bps": round(edge * 10000, 1) if edge is not None else None,
        "n": len(pairs),
    }


def _build_bma_comparison(events: list[dict]) -> dict:
    """Compare legacy live probabilities, BMA shadow, and market prices.

    This is event/bucket scoring, independent of whether the simulated strategy
    chose to trade that bucket. It answers: "which probability surface was
    better calibrated against the resolved winner?"
    """
    overall_pairs = {"legacy": [], "bma": [], "market": []}
    by_city_pairs: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(
        lambda: {"legacy": [], "bma": [], "market": []}
    )
    by_status: dict[str, int] = defaultdict(int)
    bma_event_count = 0

    for event in events:
        winner_idx = event.get("winning_bucket_idx")
        if winner_idx is None:
            continue
        status = event.get("settlement_status") or "unknown"
        by_status[status] += 1
        legacy_probs = event.get("model_probs") or []
        inputs = event.get("model_inputs") or {}
        bma = inputs.get("bma_shadow") if isinstance(inputs, dict) else None
        bma_probs = bma.get("probs") if isinstance(bma, dict) else None
        if isinstance(bma_probs, list):
            bma_event_count += 1
        market_data = event.get("market_data") or {}
        city_slug = event.get("city_slug") or "unknown"

        for bucket in event.get("buckets") or []:
            idx = int(bucket.get("idx", bucket.get("bucket_idx", -1)))
            if idx < 0:
                continue
            outcome = 1 if idx == winner_idx else 0
            if idx < len(legacy_probs):
                try:
                    p = float(legacy_probs[idx])
                    overall_pairs["legacy"].append((p, outcome))
                    by_city_pairs[city_slug]["legacy"].append((p, outcome))
                except (TypeError, ValueError):
                    pass
            if isinstance(bma_probs, list) and idx < len(bma_probs):
                try:
                    p = float(bma_probs[idx])
                    overall_pairs["bma"].append((p, outcome))
                    by_city_pairs[city_slug]["bma"].append((p, outcome))
                except (TypeError, ValueError):
                    pass
            mkt = market_data.get(idx) or {}
            try:
                mp = float(mkt.get("yes_mid"))
                overall_pairs["market"].append((mp, outcome))
                by_city_pairs[city_slug]["market"].append((mp, outcome))
            except (TypeError, ValueError, AttributeError):
                pass

    market_brier = _mean_brier(overall_pairs["market"])
    overall = {
        src: _source_score(src, pairs, market_brier)
        for src, pairs in overall_pairs.items()
    }
    if overall["market"]["brier"] is not None:
        overall["market"]["edge_vs_market"] = 0.0
        overall["market"]["edge_bps"] = 0.0

    by_city: dict[str, dict] = {}
    for slug, pairs_by_source in sorted(by_city_pairs.items()):
        city_market = _mean_brier(pairs_by_source["market"])
        city_scores = {
            src: _source_score(src, pairs, city_market)
            for src, pairs in pairs_by_source.items()
        }
        if city_scores["market"]["brier"] is not None:
            city_scores["market"]["edge_vs_market"] = 0.0
            city_scores["market"]["edge_bps"] = 0.0
        by_city[slug] = city_scores

    legacy_brier = overall["legacy"]["brier"]
    bma_brier = overall["bma"]["brier"]
    bma_delta = None
    if legacy_brier is not None and bma_brier is not None:
        bma_delta = round(float(legacy_brier) - float(bma_brier), 6)

    if bma_delta is None:
        recommendation = "BMA shadow has insufficient scored samples."
    elif bma_delta > 0:
        recommendation = "BMA shadow is beating the live legacy surface on Brier; promote only after this persists across independent days."
    elif bma_delta < 0:
        recommendation = "BMA shadow is worse than the live legacy surface; keep BMA shadow-only and inspect over-weighted sources."
    else:
        recommendation = "BMA shadow and live legacy are tied on current scored buckets."

    return {
        "overall": overall,
        "by_city": by_city,
        "bma_minus_legacy_brier": (
            round(float(bma_brier) - float(legacy_brier), 6)
            if legacy_brier is not None and bma_brier is not None else None
        ),
        "legacy_minus_bma_brier": bma_delta,
        "bma_better_than_legacy": bool(bma_delta is not None and bma_delta > 0),
        "events_scored": len({(e.get("city_slug"), e.get("date_et")) for e in events if e.get("winning_bucket_idx") is not None}),
        "bma_events_scored": bma_event_count,
        "settlement_status_counts": dict(sorted(by_status.items())),
        "recommendation": recommendation,
    }


def _build_forecast_source_diagnostics(events: list[dict]) -> dict:
    """Aggregate BMA component forecast errors by source."""
    rows: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "abs_error": 0.0, "bias": 0.0, "weight": 0.0, "sigma": 0.0}
    )
    for event in events:
        resolved_high = event.get("resolved_high_f")
        if resolved_high is None:
            continue
        inputs = event.get("model_inputs") or {}
        bma = inputs.get("bma_shadow") if isinstance(inputs, dict) else None
        comps = bma.get("components") if isinstance(bma, dict) else None
        if not isinstance(comps, list):
            continue
        for comp in comps:
            try:
                source = str(comp.get("source") or "unknown")
                mu = float(comp.get("mu"))
                weight = float(comp.get("weight") or 0.0)
                sigma = float(comp.get("sigma") or 0.0)
                err = mu - float(resolved_high)
            except (TypeError, ValueError, AttributeError):
                continue
            row = rows[source]
            row["n"] += 1
            row["abs_error"] += abs(err)
            row["bias"] += err
            row["weight"] += weight
            row["sigma"] += sigma

    out = []
    for source, row in rows.items():
        n = int(row["n"])
        if n <= 0:
            continue
        out.append({
            "source": source,
            "n": n,
            "mae_f": round(row["abs_error"] / n, 3),
            "bias_f": round(row["bias"] / n, 3),
            "avg_bma_weight": round(row["weight"] / n, 4),
            "avg_sigma_f": round(row["sigma"] / n, 3),
        })
    out.sort(key=lambda r: (r["mae_f"], -r["n"]))
    return {
        "components": out,
        "best_source": out[0]["source"] if out else None,
        "worst_source": out[-1]["source"] if out else None,
        "description": "BMA component forecast error at the backtest model snapshot; lower MAE is better, positive bias means too hot.",
    }


def _build_actionable_recommendations(
    *,
    metrics: BacktestMetrics,
    per_city: dict,
    bma_comparison: dict,
    source_diagnostics: dict,
    gate_diagnostics: Optional[dict] = None,
) -> list[dict]:
    recommendations: list[dict] = []

    if metrics.total_trades == 0:
        gate_diagnostics = gate_diagnostics or {}
        top_reason = (gate_diagnostics.get("top_reasons") or [{}])[0]
        best_rejected = gate_diagnostics.get("best_rejected") or {}
        detail = (
            "No simulated trades passed the configured gates; calibration can still be scored from resolved events."
        )
        if top_reason:
            detail = (
                f"{top_reason.get('count', 0)} candidates stopped first at "
                f"{top_reason.get('label', top_reason.get('reason', 'unknown gate'))}. "
            )
            if best_rejected:
                edge = best_rejected.get("true_edge")
                city = best_rejected.get("city_slug") or "unknown"
                bucket = best_rejected.get("bucket_label") or f"bucket {best_rejected.get('bucket_idx')}"
                if edge is not None:
                    detail += (
                        f"Best rejected candidate was {city} {bucket} at {edge * 100:.1f}% true edge. "
                    )
                actual = best_rejected.get("actual")
                required = best_rejected.get("required")
                reason_label = best_rejected.get("reason_label")
                if actual is not None and required is not None and reason_label:
                    detail += f"{reason_label}: actual {actual:.3f}, required {required:.3f}. "
            detail += "Use the gate diagnostics table before loosening thresholds."
        recommendations.append({
            "severity": "high",
            "area": "strategy",
            "title": "No simulated trades passed the gates",
            "detail": detail,
        })
    elif metrics.sharpe_ratio < 0:
        recommendations.append({
            "severity": "high",
            "area": "strategy",
            "title": "Strategy Sharpe is negative",
            "detail": "Tighten entry gates or disable the worst city buckets before increasing size.",
        })

    city_rows = [
        (slug, data)
        for slug, data in per_city.items()
        if (data.get("trades") or 0) > 0
    ]
    if city_rows:
        worst = min(city_rows, key=lambda item: item[1].get("pnl", 0.0))
        best = max(city_rows, key=lambda item: item[1].get("pnl", 0.0))
        recommendations.append({
            "severity": "medium",
            "area": "city",
            "title": f"Best city: {best[0]} / worst city: {worst[0]}",
            "detail": (
                f"{best[0]} P&L ${best[1].get('pnl', 0.0):.2f}; "
                f"{worst[0]} P&L ${worst[1].get('pnl', 0.0):.2f}. "
                "Use this to gate per-city risk until sample size is larger."
            ),
        })

    if bma_comparison.get("bma_better_than_legacy"):
        recommendations.append({
            "severity": "medium",
            "area": "model",
            "title": "BMA shadow is outperforming live legacy probabilities",
            "detail": "Keep collecting independent days; promotion should require persistent Brier and CRPS advantage, not one backtest run.",
        })
    elif bma_comparison.get("legacy_minus_bma_brier") is not None:
        recommendations.append({
            "severity": "medium",
            "area": "model",
            "title": "Keep BMA shadow-only for now",
            "detail": bma_comparison.get("recommendation") or "BMA has not cleared the live-surface promotion bar.",
        })

    comps = source_diagnostics.get("components") or []
    if comps:
        best = comps[0]
        worst = comps[-1]
        recommendations.append({
            "severity": "low",
            "area": "forecast_sources",
            "title": f"Source check: {best['source']} best, {worst['source']} worst",
            "detail": (
                f"{best['source']} MAE {best['mae_f']:.2f}°F; "
                f"{worst['source']} MAE {worst['mae_f']:.2f}°F. "
                "Downweight persistently high-MAE sources in BMA/legacy blending."
            ),
        })

    return recommendations


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
            gate_diagnostics = _new_gate_diagnostics(self.params)

            if len(unique_dates) >= train_days + test_days:
                all_trades, daily_pnl = self._walk_forward(
                    sorted_events,
                    unique_dates,
                    diagnostics=gate_diagnostics,
                )
            else:
                all_trades, daily_pnl = _run_single_pass(
                    sorted_events,
                    self.params,
                    diagnostics=gate_diagnostics,
                )

            # Build results
            result = self._build_results(
                all_trades,
                daily_pnl,
                start_date,
                end_date,
                sorted_events,
                gate_diagnostics=gate_diagnostics,
            )

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
        diagnostics: Optional[dict] = None,
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
            trades, daily_pnl = _run_single_pass(test_events, best_params, diagnostics=diagnostics)
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
        events: list[dict],
        gate_diagnostics: Optional[dict] = None,
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

        # Q7 — per-regime breakouts. Lets the operator see Brier/win-rate/PnL
        # split by atmospheric regime so volatile-day failure modes are visible.
        # Trades without a recorded regime label fall under "unknown".
        per_regime: dict[str, dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "preds": []}
        )
        for t in trades:
            key = t.regime_label or "unknown"
            pr = per_regime[key]
            pr["trades"] += 1
            if t.won:
                pr["wins"] += 1
            pr["pnl"] += t.pnl
            pr["preds"].append((t.model_prob, 1 if t.won else 0))

        per_regime_final: dict[str, dict] = {}
        for label, data in per_regime.items():
            n = data["trades"]
            if n == 0:
                continue
            try:
                _brier, _bss = compute_brier(data["preds"])
            except Exception:
                _brier, _bss = None, None
            per_regime_final[label] = {
                "trades": n,
                "win_rate": round(data["wins"] / n, 4),
                "pnl": round(data["pnl"], 4),
                "avg_pnl": round(data["pnl"] / n, 4),
                "brier": round(_brier, 4) if _brier is not None else None,
                "brier_skill_score": (
                    round(_bss, 4) if _bss is not None else None
                ),
            }

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

        bma_comparison = _build_bma_comparison(events)
        source_diagnostics = _build_forecast_source_diagnostics(events)
        finalized_gate_diagnostics = _finalize_gate_diagnostics(gate_diagnostics)
        recommendations = _build_actionable_recommendations(
            metrics=metrics,
            per_city=per_city_final,
            bma_comparison=bma_comparison,
            source_diagnostics=source_diagnostics,
            gate_diagnostics=finalized_gate_diagnostics,
        )

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
            "per_regime": per_regime_final,  # Q7
            "bma_comparison": bma_comparison,
            "forecast_sources": source_diagnostics,
            "gate_diagnostics": finalized_gate_diagnostics,
            "recommendations": recommendations,
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
