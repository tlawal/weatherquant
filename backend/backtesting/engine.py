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

@dataclass
class EnrichmentResult:
    """Summary of one Gamma enrichment pass."""
    fetched: int              # total closed markets pulled
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
    """
    city = (await sess.execute(
        select(City).where(City.city_slug == city_slug)
    )).scalar_one_or_none()
    if not city:
        return None

    # Date filter: ET calendar day. We approximate using the stored observed_at
    # in UTC; this is coarse but adequate for a day-bucket match.
    agg = func.max(MetarObs.temp_f) if kind == "high" else func.min(MetarObs.temp_f)
    row = (await sess.execute(
        select(agg)
        .where(MetarObs.city_id == city.id)
        .where(func.substr(func.cast(MetarObs.observed_at, String), 1, 10) == date_et)
    )).scalar_one_or_none()

    if row is None:
        # Fall back: match by daily_high_f if we stored any row for that day
        row = (await sess.execute(
            select(MetarObs.daily_high_f)
            .where(MetarObs.city_id == city.id)
            .where(func.substr(func.cast(MetarObs.observed_at, String), 1, 10) == date_et)
            .where(MetarObs.daily_high_f.isnot(None))
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
        page_size = 500
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

            for ev in data:
                fetched += 1
                slug = ev.get("slug") or ""
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

            if len(data) < page_size:
                break
            offset += page_size

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

    return {
        "last_enriched_at": latest.enriched_at.isoformat() if latest else None,
        "total_markets": int(total),
        "matched_markets": int(matched),
        "matched_events": int(matched_event),
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


async def fetch_resolved_events() -> list[dict]:
    """Load all resolved events with their buckets, model snapshots, and market snapshots.

    Only returns events that have the data needed to replay simulation
    (ModelSnapshot + MarketSnapshot per bucket). BacktestResolvedEvent rows
    without a matching local Event cannot be simulated, but they already
    backfill Event.winning_bucket_idx during enrich_from_gamma() — so
    more events become "resolved" after enrichment without code changes here.
    """
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
