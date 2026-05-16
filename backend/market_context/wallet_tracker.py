"""Read-only public-wallet analytics for Polymarket weather markets.

This module is informational only. It must not place orders, size positions,
or feed automated execution decisions.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import aiohttp

from backend.config import Config

log = logging.getLogger(__name__)

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
PROFILE_URL = "https://polymarket.com/profile/{wallet_address}"
_USER_AGENT = {"User-Agent": "WeatherQuant/1.0 (wallet-tracker; read-only)"}


def _warn_if_execution_caller() -> None:
    """Log loudly if this read-only module is called from execution code."""
    for frame in inspect.stack()[1:8]:
        filename = frame.filename.replace("\\", "/")
        if "/backend/execution/" in filename:
            log.error(
                "wallet_tracker: read-only analytics called from execution path: %s:%s",
                frame.filename,
                frame.lineno,
            )
            return


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def truncate_wallet_address(address: str | None, enabled: bool = True) -> str:
    if not address:
        return ""
    if not enabled:
        return address
    address = str(address)
    if len(address) <= 12:
        return address
    return f"{address[:6]}...{address[-4:]}"


def wallet_profile_url(address: str | None) -> str | None:
    if not address:
        return None
    return PROFILE_URL.format(wallet_address=str(address).lower())


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class MarketRef:
    city_slug: str
    date: str
    condition_id: str
    market_slug: str | None = None
    bucket_idx: int | None = None
    bucket_label: str | None = None
    current_price: float | None = None
    resolved_winning_bucket_idx: int | None = None
    regime: str | None = None


@dataclass(frozen=True)
class PublicTrade:
    wallet_address: str
    condition_id: str
    side: str
    size: float
    price: float
    timestamp: datetime
    asset_id: str | None = None
    outcome: str | None = None
    outcome_index: int | None = None
    market_slug: str | None = None
    event_slug: str | None = None
    transaction_hash: str | None = None
    profile_name: str | None = None

    @property
    def notional(self) -> float:
        return abs(self.size * self.price)

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "PublicTrade | None":
        wallet = str(row.get("proxyWallet") or row.get("wallet") or "").lower()
        condition_id = str(row.get("conditionId") or row.get("market") or "")
        if not wallet or not condition_id:
            return None
        side = str(row.get("side") or "").upper()
        if side not in {"BUY", "SELL"}:
            return None
        size = _to_float(row.get("size"))
        price = _to_float(row.get("price"))
        if size <= 0 or price < 0:
            return None
        outcome_index = row.get("outcomeIndex")
        try:
            outcome_index = int(outcome_index) if outcome_index is not None else None
        except (TypeError, ValueError):
            outcome_index = None
        return cls(
            wallet_address=wallet,
            condition_id=condition_id,
            side=side,
            size=size,
            price=price,
            timestamp=_coerce_timestamp(row.get("timestamp") or row.get("match_time")),
            asset_id=str(row.get("asset") or row.get("asset_id") or "") or None,
            outcome=str(row.get("outcome") or "") or None,
            outcome_index=outcome_index,
            market_slug=str(row.get("slug") or "") or None,
            event_slug=str(row.get("eventSlug") or "") or None,
            transaction_hash=str(row.get("transactionHash") or row.get("transaction_hash") or "") or None,
            profile_name=str(row.get("name") or row.get("pseudonym") or "") or None,
        )


@dataclass(frozen=True)
class WalletMetric:
    wallet_address: str
    city_slug: str
    date: str
    condition_id: str
    market_slug: str | None
    bucket_idx: int | None
    bucket_label: str | None
    trade_count: int
    volume_usd: float
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float | None
    avg_hold_minutes: float | None
    avg_entry_price: float | None
    avg_exit_price: float | None
    profitable_days_pct: float | None
    sharpe_like: float | None
    consistency_score: float | None
    regime: str | None
    inferred_style: str
    last_trade_ts: datetime | None
    net_position_qty: float
    net_flow_usd: float

    def to_db_kwargs(self) -> dict[str, Any]:
        return {
            "wallet_address": self.wallet_address,
            "city_slug": self.city_slug,
            "market_slug": self.market_slug,
            "condition_id": self.condition_id,
            "date": self.date,
            "trade_count": self.trade_count,
            "volume_usd": self.volume_usd,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "win_rate": self.win_rate,
            "avg_hold_minutes": self.avg_hold_minutes,
            "avg_entry_price": self.avg_entry_price,
            "avg_exit_price": self.avg_exit_price,
            "profitable_days_pct": self.profitable_days_pct,
            "sharpe_like": self.sharpe_like,
            "consistency_score": self.consistency_score,
            "regime": self.regime,
            "inferred_style": self.inferred_style,
            "bucket_idx": self.bucket_idx,
            "bucket_label": self.bucket_label,
            "net_position_qty": self.net_position_qty,
            "net_flow_usd": self.net_flow_usd,
            "last_trade_ts": self.last_trade_ts,
            "last_updated_ts": datetime.now(timezone.utc),
        }


@dataclass(frozen=True)
class WalletTrackerSummary:
    enabled: bool
    cities_scanned: int = 0
    condition_ids_scanned: int = 0
    trades_fetched: int = 0
    wallets_updated: int = 0
    top_wallet_score: float | None = None
    errors: tuple[str, ...] = ()


class PolymarketDataApiTradeAdapter:
    """Adapter for documented public Polymarket profile trade endpoints."""

    def __init__(
        self,
        timeout_s: float = 12.0,
        fetch_pause_seconds: float | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.fetch_pause_seconds = (
            Config.WALLET_TRACKER_FETCH_PAUSE_SECONDS
            if fetch_pause_seconds is None
            else fetch_pause_seconds
        )

    async def fetch_trades_for_markets(
        self,
        condition_ids: Iterable[str],
        *,
        limit: int | None = None,
    ) -> list[PublicTrade]:
        ids = [cid for cid in dict.fromkeys(condition_ids) if cid]
        if not ids:
            return []
        limit = min(limit or Config.WALLET_TRACKER_FETCH_LIMIT, 10000)
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        trades: list[PublicTrade] = []
        async with aiohttp.ClientSession(timeout=timeout, headers=_USER_AGENT) as http:
            for offset in range(0, len(ids), 20):
                chunk = ids[offset:offset + 20]
                params = {
                    "market": ",".join(chunk),
                    "limit": str(limit),
                    "offset": "0",
                    "takerOnly": "true",
                }
                url = f"{DATA_API}/trades"
                log.info(
                    "wallet_tracker: fetching public trades markets=%d limit=%d",
                    len(chunk),
                    limit,
                )
                try:
                    async with http.get(url, params=params) as resp:
                        if resp.status != 200:
                            log.warning("wallet_tracker: data-api trades returned %d", resp.status)
                            continue
                        data = await resp.json(content_type=None)
                except Exception as exc:
                    log.warning("wallet_tracker: public trade fetch failed: %s", exc)
                    continue
                rows = data if isinstance(data, list) else data.get("data", [])
                row_iter = rows if isinstance(rows, list) else []
                for row in row_iter:
                    if isinstance(row, dict):
                        trade = PublicTrade.from_api(row)
                        if trade:
                            trades.append(trade)
                if self.fetch_pause_seconds > 0:
                    await asyncio.sleep(self.fetch_pause_seconds)
        return trades

    async def fetch_public_profile(self, wallet_address: str) -> dict[str, Any] | None:
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=_USER_AGENT) as http:
            try:
                async with http.get(
                    f"{GAMMA_API}/public-profile",
                    params={"address": wallet_address},
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json(content_type=None)
                    return data if isinstance(data, dict) else None
            except Exception as exc:
                log.debug("wallet_tracker: profile fetch failed wallet=%s err=%s", wallet_address, exc)
                return None


def calculate_sharpe_like(returns: list[float]) -> float:
    if len(returns) < 2:
        return 1.0 if returns and returns[0] > 0 else 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std <= 0:
        return 1.0 if mean_r > 0 else 0.0
    return mean_r / std


def calculate_consistency_score(
    *,
    sharpe_like: float,
    profitable_days_pct: float,
    win_rate: float,
    volume_usd: float,
    activity_consistency: float,
) -> float:
    normalized_sharpe_like = _clamp((sharpe_like + 1.0) / 3.0)
    normalized_log_volume = _clamp(math.log1p(max(0.0, volume_usd)) / math.log1p(5000.0))
    score = (
        0.35 * normalized_sharpe_like
        + 0.25 * _clamp(profitable_days_pct)
        + 0.20 * _clamp(win_rate)
        + 0.10 * normalized_log_volume
        + 0.10 * _clamp(activity_consistency)
    )
    return round(_clamp(score), 4)


def infer_strategy_style(
    trades: list[PublicTrade],
    *,
    avg_hold_minutes: float | None,
    bucket_indices: set[int],
    observation_minutes: list[int] | None = None,
) -> str:
    if len(trades) < 3:
        return "Unknown"
    if observation_minutes:
        near_obs = 0
        for trade in trades:
            minute = trade.timestamp.minute
            if any(min(abs(minute - obs), 60 - abs(minute - obs)) <= 8 for obs in observation_minutes):
                near_obs += 1
        if near_obs / len(trades) >= 0.5 and {t.side for t in trades} == {"BUY", "SELL"}:
            return "Observation Flipper"
    if avg_hold_minutes is not None and avg_hold_minutes <= 45 and len(trades) >= 5:
        return "Scalper"
    if len(bucket_indices) >= 3 and max(bucket_indices) - min(bucket_indices) <= 4:
        return "Bucket Rotator"
    sell_ratio = sum(1 for t in trades if t.side == "SELL") / max(1, len(trades))
    if sell_ratio <= 0.2 and avg_hold_minutes is not None and avg_hold_minutes >= 240:
        return "Early Forecaster"
    if sell_ratio <= 0.2:
        return "Late Holder"
    return "Unknown"


def _weighted_avg(items: list[tuple[float, float]]) -> float | None:
    qty = sum(max(0.0, q) for q, _ in items)
    if qty <= 0:
        return None
    return sum(q * price for q, price in items) / qty


def _metric_passes_filters(
    metric: WalletMetric,
    *,
    min_volume_usd: float,
    min_trades: int,
    min_active_days: int,
    active_days: int,
    history_days: int,
) -> bool:
    if metric.volume_usd < min_volume_usd:
        return False
    if metric.trade_count < min_trades:
        return False
    if history_days >= min_active_days and active_days < min_active_days:
        return False
    if (metric.realized_pnl + metric.unrealized_pnl) <= 0:
        return False
    avg_notional = metric.volume_usd / max(1, metric.trade_count)
    if avg_notional < 1.0:
        return False
    return True


def compute_wallet_metrics(
    trades: list[PublicTrade],
    market_refs: list[MarketRef],
    *,
    min_volume_usd: float = 100.0,
    min_trades: int = 3,
    min_active_days: int = 2,
    max_wallets: int | None = None,
    as_of_date: str | None = None,
    observation_minutes: list[int] | None = None,
) -> list[WalletMetric]:
    market_by_condition = {m.condition_id: m for m in market_refs if m.condition_id}
    history_dates = {m.date for m in market_refs if m.date}
    if not market_by_condition:
        return []
    as_of_date = as_of_date or max(history_dates)

    by_wallet: dict[str, list[PublicTrade]] = defaultdict(list)
    for trade in trades:
        if trade.condition_id in market_by_condition:
            by_wallet[trade.wallet_address.lower()].append(trade)

    metrics: list[tuple[WalletMetric, int]] = []
    for wallet_address, wallet_trades in by_wallet.items():
        wallet_trades.sort(key=lambda t: t.timestamp)
        trade_count = len(wallet_trades)
        volume_usd = sum(t.notional for t in wallet_trades)
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        close_count = 0
        profitable_close_count = 0
        hold_minutes: list[float] = []
        entry_items: list[tuple[float, float]] = []
        exit_items: list[tuple[float, float]] = []
        date_pnl: dict[str, float] = defaultdict(float)
        date_volume: dict[str, float] = defaultdict(float)
        bucket_indices: set[int] = set()
        condition_stats: dict[str, dict[str, Any]] = {}

        trades_by_condition: dict[str, list[PublicTrade]] = defaultdict(list)
        for trade in wallet_trades:
            trades_by_condition[trade.condition_id].append(trade)

        for condition_id, condition_trades in trades_by_condition.items():
            market = market_by_condition[condition_id]
            if market.bucket_idx is not None:
                bucket_indices.add(market.bucket_idx)
            lots: list[list[Any]] = []
            condition_realized = 0.0
            buys_usd = 0.0
            sells_usd = 0.0
            bought_qty = 0.0
            sold_qty = 0.0

            for trade in condition_trades:
                date_volume[market.date] += trade.notional
                if trade.side == "BUY":
                    lots.append([trade.size, trade.price, trade.timestamp])
                    entry_items.append((trade.size, trade.price))
                    buys_usd += trade.notional
                    bought_qty += trade.size
                    continue

                exit_items.append((trade.size, trade.price))
                sells_usd += trade.notional
                sold_qty += trade.size
                remaining = trade.size
                while remaining > 0 and lots:
                    lot_qty, lot_price, lot_ts = lots[0]
                    matched = min(float(lot_qty), remaining)
                    pnl = matched * (trade.price - float(lot_price))
                    condition_realized += pnl
                    close_count += 1
                    if pnl > 0:
                        profitable_close_count += 1
                    hold_minutes.append(
                        max(0.0, (trade.timestamp - lot_ts).total_seconds() / 60.0)
                    )
                    lot_qty = float(lot_qty) - matched
                    remaining -= matched
                    if lot_qty <= 1e-9:
                        lots.pop(0)
                    else:
                        lots[0][0] = lot_qty

            remaining_qty = sum(float(q) for q, _, _ in lots)
            condition_unrealized = 0.0
            if market.current_price is not None:
                condition_unrealized = sum(
                    float(q) * (float(market.current_price) - float(price))
                    for q, price, _ in lots
                )
            condition_volume = sum(t.notional for t in condition_trades)
            net_flow_usd = buys_usd - sells_usd
            realized_pnl += condition_realized
            unrealized_pnl += condition_unrealized
            date_pnl[market.date] += condition_realized + condition_unrealized
            condition_stats[condition_id] = {
                "market": market,
                "volume_usd": condition_volume,
                "net_position_qty": remaining_qty,
                "net_flow_usd": net_flow_usd,
                "score_weight": abs(net_flow_usd) + abs(remaining_qty * (market.current_price or 0.0)),
            }

        active_days = sum(1 for v in date_volume.values() if v > 0)
        returns = [
            date_pnl[d] / max(1.0, date_volume[d])
            for d in sorted(date_volume.keys())
            if date_volume[d] > 0
        ]
        profitable_days_pct = (
            sum(1 for d in date_volume if date_volume[d] > 0 and date_pnl[d] > 0) / active_days
            if active_days else 0.0
        )
        sharpe_like = calculate_sharpe_like(returns)
        win_rate = (
            profitable_close_count / close_count
            if close_count
            else profitable_days_pct
        )
        activity_consistency = (
            active_days / max(1, min(len(history_dates), Config.WALLET_TRACKER_LOOKBACK_DAYS))
            if history_dates else 0.0
        )
        consistency_score = calculate_consistency_score(
            sharpe_like=sharpe_like,
            profitable_days_pct=profitable_days_pct,
            win_rate=win_rate,
            volume_usd=volume_usd,
            activity_consistency=activity_consistency,
        )
        dominant = max(
            condition_stats.values(),
            key=lambda s: (s["score_weight"], s["volume_usd"]),
        )
        market = dominant["market"]
        avg_hold = sum(hold_minutes) / len(hold_minutes) if hold_minutes else None
        avg_entry = _weighted_avg(entry_items)
        avg_exit = _weighted_avg(exit_items)
        metric = WalletMetric(
            wallet_address=wallet_address,
            city_slug=market.city_slug,
            date=as_of_date,
            condition_id=market.condition_id,
            market_slug=market.market_slug,
            bucket_idx=market.bucket_idx,
            bucket_label=market.bucket_label,
            trade_count=trade_count,
            volume_usd=round(volume_usd, 4),
            realized_pnl=round(realized_pnl, 4),
            unrealized_pnl=round(unrealized_pnl, 4),
            win_rate=round(win_rate, 4),
            avg_hold_minutes=round(avg_hold, 2) if avg_hold is not None else None,
            avg_entry_price=round(avg_entry, 4) if avg_entry is not None else None,
            avg_exit_price=round(avg_exit, 4) if avg_exit is not None else None,
            profitable_days_pct=round(profitable_days_pct, 4),
            sharpe_like=round(sharpe_like, 4),
            consistency_score=consistency_score,
            regime=market.regime,
            inferred_style=infer_strategy_style(
                wallet_trades,
                avg_hold_minutes=avg_hold,
                bucket_indices=bucket_indices,
                observation_minutes=observation_minutes,
            ),
            last_trade_ts=max((t.timestamp for t in wallet_trades), default=None),
            net_position_qty=round(float(dominant["net_position_qty"]), 4),
            net_flow_usd=round(float(dominant["net_flow_usd"]), 4),
        )
        if _metric_passes_filters(
            metric,
            min_volume_usd=min_volume_usd,
            min_trades=min_trades,
            min_active_days=min_active_days,
            active_days=active_days,
            history_days=len(history_dates),
        ):
            metrics.append((metric, active_days))

    metrics.sort(
        key=lambda item: (
            item[0].consistency_score or 0.0,
            item[0].volume_usd,
            item[0].realized_pnl + item[0].unrealized_pnl,
        ),
        reverse=True,
    )
    return [m for m, _ in metrics[:max_wallets]]


async def _discover_market_refs_for_city(city: Any, *, as_of_date: str) -> list[MarketRef]:
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_buckets_for_event,
        get_latest_market_snapshots_bulk,
        get_recent_events_for_city,
    )

    market_refs: list[MarketRef] = []
    async with get_session() as sess:
        events = await get_recent_events_for_city(
            sess,
            city.id,
            before_or_on_date_et=as_of_date,
            limit=max(1, Config.WALLET_TRACKER_LOOKBACK_DAYS),
        )
        bucket_rows = []
        for event in events:
            buckets = await get_buckets_for_event(sess, event.id)
            for bucket in buckets:
                if bucket.condition_id:
                    bucket_rows.append((event, bucket))
        snapshots = await get_latest_market_snapshots_bulk(sess, [b.id for _, b in bucket_rows])

    for event, bucket in bucket_rows:
        snap = snapshots.get(bucket.id)
        current_price = snap.yes_mid if snap and snap.yes_mid is not None else None
        if event.winning_bucket_idx is not None:
            current_price = 1.0 if bucket.bucket_idx == event.winning_bucket_idx else 0.0
        market_refs.append(
            MarketRef(
                city_slug=city.city_slug,
                date=event.date_et,
                condition_id=bucket.condition_id,
                market_slug=event.gamma_slug,
                bucket_idx=bucket.bucket_idx,
                bucket_label=bucket.label,
                current_price=current_price,
                resolved_winning_bucket_idx=event.winning_bucket_idx,
            )
        )
    return market_refs


async def _wallet_tracker_cities() -> list[Any]:
    from backend.storage.db import get_session
    from backend.storage.repos import get_all_cities, get_city_by_slug

    configured = (Config.WALLET_TRACKER_START_CITY or "atlanta").strip().lower()
    async with get_session() as sess:
        if configured == "all":
            return await get_all_cities(sess, enabled_only=True)
        cities = []
        for slug in [s.strip() for s in configured.split(",") if s.strip()]:
            city = await get_city_by_slug(sess, slug)
            if city and city.enabled:
                cities.append(city)
        return cities


async def update_wallet_rankings(
    adapter: PolymarketDataApiTradeAdapter | None = None,
) -> WalletTrackerSummary:
    _warn_if_execution_caller()
    if not Config.WALLET_TRACKER_READ_ONLY:
        log.error("wallet_tracker: WALLET_TRACKER_READ_ONLY=false; refusing to run")
        return WalletTrackerSummary(enabled=False, errors=("read_only_disabled",))
    if not Config.WALLET_TRACKER_ENABLED:
        return WalletTrackerSummary(enabled=False)

    from backend.storage.db import get_session
    from backend.storage.repos import upsert_wallet_stat
    from backend.tz_utils import city_local_date

    adapter = adapter or PolymarketDataApiTradeAdapter()
    cities = await _wallet_tracker_cities()
    total_conditions = 0
    total_trades = 0
    total_wallets = 0
    top_score: float | None = None
    errors: list[str] = []

    for city in cities:
        as_of_date = city_local_date(city)
        market_refs = await _discover_market_refs_for_city(city, as_of_date=as_of_date)
        condition_ids = [m.condition_id for m in market_refs]
        total_conditions += len(condition_ids)
        if not condition_ids:
            log.info("wallet_tracker: city=%s no condition_ids", city.city_slug)
            continue
        try:
            trades = await adapter.fetch_trades_for_markets(condition_ids)
        except Exception as exc:
            msg = f"{city.city_slug}:{type(exc).__name__}"
            log.warning("wallet_tracker: city=%s trade fetch failed: %s", city.city_slug, exc)
            errors.append(msg)
            trades = []
        total_trades += len(trades)
        metrics = compute_wallet_metrics(
            trades,
            market_refs,
            min_volume_usd=Config.WALLET_TRACKER_MIN_VOLUME_USD,
            min_trades=Config.WALLET_TRACKER_MIN_TRADES,
            min_active_days=Config.WALLET_TRACKER_MIN_ACTIVE_DAYS,
            max_wallets=Config.WALLET_TRACKER_MAX_WALLETS_PER_CITY,
            as_of_date=as_of_date,
        )
        async with get_session() as sess:
            for metric in metrics:
                await upsert_wallet_stat(sess, **metric.to_db_kwargs())
        total_wallets += len(metrics)
        if metrics:
            top_score = max(top_score or 0.0, metrics[0].consistency_score or 0.0)
        log.info(
            "wallet_tracker: city=%s condition_ids=%d trades=%d wallets=%d top_score=%s errors=%d",
            city.city_slug,
            len(condition_ids),
            len(trades),
            len(metrics),
            metrics[0].consistency_score if metrics else None,
            len(errors),
        )

    return WalletTrackerSummary(
        enabled=True,
        cities_scanned=len(cities),
        condition_ids_scanned=total_conditions,
        trades_fetched=total_trades,
        wallets_updated=total_wallets,
        top_wallet_score=top_score,
        errors=tuple(errors),
    )


def serialize_wallet_stat_row(row: Any, *, truncate_addresses: bool = True) -> dict[str, Any]:
    address = str(getattr(row, "wallet_address", "") or "").lower()
    last_trade_ts = getattr(row, "last_trade_ts", None)
    if isinstance(last_trade_ts, datetime):
        last_trade = last_trade_ts.isoformat()
    else:
        last_trade = last_trade_ts
    return {
        "wallet_address": address,
        "display_address": truncate_wallet_address(address, enabled=truncate_addresses),
        "profile_url": wallet_profile_url(address),
        "city_slug": getattr(row, "city_slug", None),
        "market_slug": getattr(row, "market_slug", None),
        "condition_id": getattr(row, "condition_id", None),
        "date": getattr(row, "date", None),
        "trade_count": getattr(row, "trade_count", 0) or 0,
        "volume_usd": round(float(getattr(row, "volume_usd", 0.0) or 0.0), 2),
        "realized_pnl": round(float(getattr(row, "realized_pnl", 0.0) or 0.0), 2),
        "unrealized_pnl": round(float(getattr(row, "unrealized_pnl", 0.0) or 0.0), 2),
        "win_rate": getattr(row, "win_rate", None),
        "avg_hold_minutes": getattr(row, "avg_hold_minutes", None),
        "avg_entry_price": getattr(row, "avg_entry_price", None),
        "avg_exit_price": getattr(row, "avg_exit_price", None),
        "profitable_days_pct": getattr(row, "profitable_days_pct", None),
        "sharpe_like": getattr(row, "sharpe_like", None),
        "consistency_score": getattr(row, "consistency_score", None),
        "regime": getattr(row, "regime", None),
        "inferred_style": getattr(row, "inferred_style", None) or "Unknown",
        "bucket_idx": getattr(row, "bucket_idx", None),
        "bucket_label": getattr(row, "bucket_label", None),
        "net_position_qty": round(float(getattr(row, "net_position_qty", 0.0) or 0.0), 4),
        "net_flow_usd": round(float(getattr(row, "net_flow_usd", 0.0) or 0.0), 2),
        "last_trade_ts": last_trade,
    }


def build_wallet_leaderboard_payload(
    rows: list[Any],
    *,
    enabled: bool,
    truncate_addresses: bool = True,
    limit: int = 10,
) -> dict[str, Any]:
    seen: set[str] = set()
    serialized = []
    for row in rows:
        address = str(getattr(row, "wallet_address", "") or "").lower()
        if not address or address in seen:
            continue
        seen.add(address)
        serialized.append(
            serialize_wallet_stat_row(row, truncate_addresses=truncate_addresses)
        )
        if len(serialized) >= limit:
            break
    return {
        "enabled": enabled,
        "rows": serialized,
        "status": "ok" if serialized else ("disabled" if not enabled else "empty"),
        "disclaimer": (
            "Wallet leaderboard is read-only public-market analytics. "
            "It is not a copy-trading signal and does not trigger automated trades."
        ),
    }


async def get_wallet_leaderboard_payload(
    city_slug: str,
    date_et: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    _warn_if_execution_caller()
    from backend.storage.db import get_session
    from backend.storage.repos import get_wallet_stats_for_city

    limit = limit or Config.WALLET_TRACKER_MAX_WALLETS_PER_CITY
    if not Config.WALLET_TRACKER_ENABLED:
        return build_wallet_leaderboard_payload(
            [],
            enabled=False,
            truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
            limit=limit,
        )
    async with get_session() as sess:
        rows = await get_wallet_stats_for_city(sess, city_slug, date_et=date_et, limit=max(100, limit))
    return build_wallet_leaderboard_payload(
        rows,
        enabled=True,
        truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
        limit=limit,
    )


def compute_smart_money_divergence(
    buckets: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model_candidates = [
        b for b in buckets
        if b.get("bucket_idx") is not None and b.get("model_prob") is not None
    ]
    if not model_candidates:
        return {"status": "unavailable", "reason": "model_bucket_unavailable"}
    model_bucket = max(model_candidates, key=lambda b: b.get("model_prob") or 0.0)

    flow_by_bucket: dict[int, float] = defaultdict(float)
    label_by_bucket: dict[int, str] = {}
    for row in leaderboard_rows:
        idx = row.get("bucket_idx")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        flow_by_bucket[idx_int] += float(row.get("net_flow_usd") or 0.0)
        if row.get("bucket_label"):
            label_by_bucket[idx_int] = row["bucket_label"]

    positive = {idx: flow for idx, flow in flow_by_bucket.items() if flow > 0}
    if not positive:
        return {"status": "unavailable", "reason": "smart_money_flow_unavailable"}

    smart_idx = max(positive, key=lambda idx: positive[idx])
    model_idx = int(model_bucket["bucket_idx"])
    distance = abs(model_idx - smart_idx)
    divergence = "LOW" if distance == 0 else ("MEDIUM" if distance == 1 else "HIGH")
    return {
        "status": "available",
        "model_bucket_idx": model_idx,
        "model_bucket_label": model_bucket.get("label"),
        "smart_money_bucket_idx": smart_idx,
        "smart_money_bucket_label": label_by_bucket.get(smart_idx),
        "smart_money_net_flow_usd": round(positive[smart_idx], 2),
        "divergence": divergence,
    }
