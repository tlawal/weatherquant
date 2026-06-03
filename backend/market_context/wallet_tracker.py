"""Read-only public-wallet analytics for Polymarket weather markets.

This module is informational only. It must not place orders, size positions,
or feed automated execution decisions.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
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
_MAX_RAW_DEDUPE_KEY_LEN = 240


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
    raw: dict[str, Any] | None = None

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
            raw=row,
        )


@dataclass(frozen=True)
class WalletExposureMetric:
    wallet_address: str
    city_slug: str
    date: str
    condition_id: str
    market_slug: str | None
    bucket_idx: int | None
    bucket_label: str | None
    net_position_qty: float
    net_notional_usd: float
    avg_entry_price: float | None
    realized_pnl: float
    unrealized_pnl: float
    last_trade_ts: datetime | None
    trade_count: int = 0
    volume_usd: float = 0.0

    def to_db_kwargs(self) -> dict[str, Any]:
        return {
            "wallet_address": self.wallet_address,
            "city_slug": self.city_slug,
            "date": self.date,
            "market_slug": self.market_slug,
            "condition_id": self.condition_id,
            "bucket_idx": self.bucket_idx,
            "bucket_label": self.bucket_label,
            "net_position_qty": self.net_position_qty,
            "net_notional_usd": self.net_notional_usd,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "last_trade_ts": self.last_trade_ts,
            "trade_count": self.trade_count,
            "volume_usd": self.volume_usd,
            "last_updated_ts": datetime.now(timezone.utc),
        }


@dataclass(frozen=True)
class WalletSkillMetric:
    wallet_address: str
    scope: str
    city_slug: str
    window_days: int
    adjusted_score: float
    rank: int | None
    win_rate: float | None
    wilson_win_rate: float | None
    resolved_markets: int
    total_markets: int
    total_volume_usd: float
    realized_pnl: float
    roi: float | None
    profit_factor: float | None
    avg_notional_usd: float | None
    active_days: int
    last_active_ts: datetime | None

    def to_db_kwargs(self) -> dict[str, Any]:
        return {
            "wallet_address": self.wallet_address,
            "scope": self.scope,
            "city_slug": self.city_slug,
            "window_days": self.window_days,
            "adjusted_score": self.adjusted_score,
            "rank": self.rank,
            "win_rate": self.win_rate,
            "wilson_win_rate": self.wilson_win_rate,
            "resolved_markets": self.resolved_markets,
            "total_markets": self.total_markets,
            "total_volume_usd": self.total_volume_usd,
            "realized_pnl": self.realized_pnl,
            "roi": self.roi,
            "profit_factor": self.profit_factor,
            "avg_notional_usd": self.avg_notional_usd,
            "active_days": self.active_days,
            "last_active_ts": self.last_active_ts,
            "last_updated_ts": datetime.now(timezone.utc),
        }


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
        page_size = min(500, limit)
        condition_chunk_size = max(1, min(20, Config.WALLET_TRACKER_CONDITION_CHUNK_SIZE))
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        trades: list[PublicTrade] = []
        async with aiohttp.ClientSession(timeout=timeout, headers=_USER_AGENT) as http:
            for chunk_offset in range(0, len(ids), condition_chunk_size):
                chunk = ids[chunk_offset:chunk_offset + condition_chunk_size]
                fetched_for_chunk = 0
                for page_offset in range(0, limit, page_size):
                    params = {
                        "market": ",".join(chunk),
                        "limit": str(page_size),
                        "offset": str(page_offset),
                    }
                    if Config.WALLET_TRACKER_TAKER_ONLY:
                        params["takerOnly"] = "true"
                    url = f"{DATA_API}/trades"
                    log.info(
                        "wallet_tracker: fetching public trades markets=%d page_offset=%d page_size=%d chunk_size=%d",
                        len(chunk),
                        page_offset,
                        page_size,
                        condition_chunk_size,
                    )
                    try:
                        async with http.get(url, params=params) as resp:
                            if resp.status != 200:
                                log.warning("wallet_tracker: data-api trades returned %d", resp.status)
                                break
                            data = await resp.json(content_type=None)
                    except Exception as exc:
                        log.warning("wallet_tracker: public trade fetch failed: %s", exc)
                        break
                    rows = data if isinstance(data, list) else data.get("data", [])
                    row_iter = rows if isinstance(rows, list) else []
                    if not row_iter:
                        break
                    for row in row_iter:
                        if isinstance(row, dict):
                            trade = PublicTrade.from_api(row)
                            if trade:
                                trades.append(trade)
                    fetched_for_chunk += len(row_iter)
                    if len(row_iter) < page_size or fetched_for_chunk >= limit:
                        break
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


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = wins / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return round(max(0.0, (centre - margin) / denom), 4)


def profit_factor(pnls: list[float]) -> float | None:
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss <= 0:
        return round(gross_win, 4) if gross_win > 0 else None
    return round(gross_win / gross_loss, 4)


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


def trade_dedupe_identity(trade: PublicTrade) -> str:
    tx = trade.transaction_hash or ""
    if tx:
        return "|".join(
            [
                tx,
                trade.wallet_address.lower(),
                trade.condition_id,
                trade.side,
                trade.asset_id or "",
                f"{trade.size:.8f}",
                f"{trade.price:.8f}",
            ]
        )
    return "|".join(
        [
            trade.wallet_address.lower(),
            trade.condition_id,
            trade.side,
            trade.asset_id or "",
            trade.timestamp.isoformat(),
            f"{trade.size:.8f}",
            f"{trade.price:.8f}",
        ]
    )


def trade_dedupe_key(trade: PublicTrade) -> str:
    """Return a stable DB-safe key for one public trade.

    Polymarket weather token ids can be long enough that the raw trade
    identity exceeds the historical wallet_trades.dedupe_key VARCHAR(256)
    column. Preserve existing short-key behavior for already-ingested rows,
    but hash any long identity before it reaches storage.
    """
    identity = trade_dedupe_identity(trade)
    if len(identity) <= _MAX_RAW_DEDUPE_KEY_LEN:
        return identity
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def dedupe_public_trades(trades: list[PublicTrade]) -> list[PublicTrade]:
    seen: set[str] = set()
    deduped: list[PublicTrade] = []
    for trade in sorted(trades, key=lambda t: t.timestamp):
        key = trade_dedupe_key(trade)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trade)
    return deduped


def compute_wallet_exposures(
    trades: list[PublicTrade],
    market_refs: list[MarketRef],
) -> list[WalletExposureMetric]:
    market_by_condition = {m.condition_id: m for m in market_refs if m.condition_id}
    by_wallet_condition: dict[tuple[str, str], list[PublicTrade]] = defaultdict(list)
    for trade in dedupe_public_trades(trades):
        if trade.condition_id in market_by_condition:
            by_wallet_condition[(trade.wallet_address.lower(), trade.condition_id)].append(trade)

    exposures: list[WalletExposureMetric] = []
    for (wallet_address, condition_id), condition_trades in by_wallet_condition.items():
        market = market_by_condition[condition_id]
        condition_trades.sort(key=lambda t: t.timestamp)
        lots: list[list[Any]] = []
        realized_pnl = 0.0
        buys_usd = 0.0
        sells_usd = 0.0
        entry_items: list[tuple[float, float]] = []

        for trade in condition_trades:
            if trade.side == "BUY":
                lots.append([trade.size, trade.price, trade.timestamp])
                entry_items.append((trade.size, trade.price))
                buys_usd += trade.notional
                continue

            sells_usd += trade.notional
            remaining = trade.size
            while remaining > 0 and lots:
                lot_qty, lot_price, _lot_ts = lots[0]
                matched = min(float(lot_qty), remaining)
                realized_pnl += matched * (trade.price - float(lot_price))
                lot_qty = float(lot_qty) - matched
                remaining -= matched
                if lot_qty <= 1e-9:
                    lots.pop(0)
                else:
                    lots[0][0] = lot_qty

        net_qty = sum(float(q) for q, _, _ in lots)
        avg_entry = _weighted_avg([(float(q), float(price)) for q, price, _ in lots]) or _weighted_avg(entry_items)
        current_price = market.current_price
        unrealized = (
            sum(float(q) * (float(current_price) - float(price)) for q, price, _ in lots)
            if current_price is not None
            else 0.0
        )
        exposures.append(
            WalletExposureMetric(
                wallet_address=wallet_address,
                city_slug=market.city_slug,
                date=market.date,
                condition_id=condition_id,
                market_slug=market.market_slug,
                bucket_idx=market.bucket_idx,
                bucket_label=market.bucket_label,
                net_position_qty=round(net_qty, 4),
                net_notional_usd=round(buys_usd - sells_usd, 4),
                avg_entry_price=round(avg_entry, 4) if avg_entry is not None else None,
                realized_pnl=round(realized_pnl, 4),
                unrealized_pnl=round(unrealized, 4),
                last_trade_ts=max((t.timestamp for t in condition_trades), default=None),
                trade_count=len(condition_trades),
                volume_usd=round(sum(t.notional for t in condition_trades), 4),
            )
        )
    return exposures


def _trade_to_db_kwargs(trade: PublicTrade, market: MarketRef) -> dict[str, Any]:
    return {
        "dedupe_key": trade_dedupe_key(trade),
        "wallet_address": trade.wallet_address.lower(),
        "city_slug": market.city_slug,
        "date": market.date,
        "market_slug": trade.market_slug or market.market_slug,
        "condition_id": market.condition_id,
        "asset_id": trade.asset_id,
        "bucket_idx": market.bucket_idx,
        "bucket_label": market.bucket_label,
        "side": trade.side,
        "size": trade.size,
        "price": trade.price,
        "notional_usd": trade.notional,
        "trade_ts": trade.timestamp,
        "transaction_hash": trade.transaction_hash,
        "raw_json": json.dumps(trade.raw, default=str) if trade.raw else None,
    }


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


def compute_wallet_skill_scores(
    exposures: list[WalletExposureMetric],
    market_refs: list[MarketRef],
    *,
    scope: str,
    city_slug: str = "",
    window_days: int = 90,
    min_resolved_markets: int = 3,
    min_volume_usd: float = 100.0,
    min_active_days: int = 2,
) -> list[WalletSkillMetric]:
    resolved_conditions = {
        m.condition_id
        for m in market_refs
        if m.resolved_winning_bucket_idx is not None
    }
    by_wallet: dict[str, list[WalletExposureMetric]] = defaultdict(list)
    for exposure in exposures:
        if scope == "city" and exposure.city_slug != city_slug:
            continue
        by_wallet[exposure.wallet_address.lower()].append(exposure)

    raw_scores: list[WalletSkillMetric] = []
    for wallet_address, rows in by_wallet.items():
        total_markets = len({r.condition_id for r in rows})
        resolved_rows = [r for r in rows if r.condition_id in resolved_conditions]
        resolved_markets = len({r.condition_id for r in resolved_rows})
        total_volume = sum(
            float(getattr(r, "volume_usd", 0.0) or 0.0) or abs(r.net_notional_usd)
            for r in rows
        )
        active_days = len({r.date for r in rows})
        if resolved_markets < min_resolved_markets:
            continue
        if total_volume < min_volume_usd:
            continue
        if active_days < min_active_days:
            continue

        pnls = [r.realized_pnl + r.unrealized_pnl for r in resolved_rows]
        resolved_pnl = sum(pnls)
        if resolved_pnl <= 0:
            continue
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / resolved_markets if resolved_markets else 0.0
        wilson = wilson_lower_bound(wins, resolved_markets)
        roi = resolved_pnl / total_volume if total_volume > 0 else None
        pf = profit_factor(pnls)
        avg_notional = total_volume / max(1, total_markets)
        volume_component = _clamp(math.log1p(total_volume) / math.log1p(25000.0))
        resolved_component = _clamp(resolved_markets / 25.0)
        active_component = _clamp(active_days / 14.0)
        roi_component = _clamp((roi or 0.0) / 0.5)
        adjusted_score = round(
            _clamp(
                0.35 * wilson
                + 0.20 * roi_component
                + 0.15 * volume_component
                + 0.15 * resolved_component
                + 0.15 * active_component
            ),
            4,
        )
        raw_scores.append(
            WalletSkillMetric(
                wallet_address=wallet_address,
                scope=scope,
                city_slug=city_slug if scope == "city" else "",
                window_days=window_days,
                adjusted_score=adjusted_score,
                rank=None,
                win_rate=round(win_rate, 4),
                wilson_win_rate=wilson,
                resolved_markets=resolved_markets,
                total_markets=total_markets,
                total_volume_usd=round(total_volume, 4),
                realized_pnl=round(resolved_pnl, 4),
                roi=round(roi, 4) if roi is not None else None,
                profit_factor=pf,
                avg_notional_usd=round(avg_notional, 4),
                active_days=active_days,
                last_active_ts=max((r.last_trade_ts for r in rows if r.last_trade_ts), default=None),
            )
        )

    raw_scores.sort(
        key=lambda s: (
            s.adjusted_score,
            s.wilson_win_rate or 0.0,
            s.total_volume_usd,
            s.last_active_ts or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return [
        WalletSkillMetric(**{**score.__dict__, "rank": idx})
        for idx, score in enumerate(raw_scores, start=1)
    ]


async def _discover_market_refs_for_city(
    city: Any,
    *,
    as_of_date: str,
    lookback_days: int | None = None,
    exact_date: bool = False,
) -> list[MarketRef]:
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_event,
        get_buckets_for_event,
        get_latest_market_snapshots_bulk,
        get_recent_events_for_city,
    )

    market_refs: list[MarketRef] = []
    async with get_session() as sess:
        if exact_date:
            event = await get_event(sess, city.id, as_of_date)
            events = [event] if event else []
        else:
            events = await get_recent_events_for_city(
                sess,
                city.id,
                before_or_on_date_et=as_of_date,
                limit=max(1, lookback_days or Config.WALLET_TRACKER_LOOKBACK_DAYS),
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


async def _wallet_tracker_cities(city_slugs: Iterable[str] | None = None) -> list[Any]:
    from backend.storage.db import get_session
    from backend.storage.repos import get_all_cities, get_city_by_slug

    explicit_slugs = [
        str(s).strip().lower()
        for s in (city_slugs or [])
        if str(s).strip()
    ]
    configured = ",".join(explicit_slugs) if explicit_slugs else (
        Config.WALLET_TRACKER_START_CITY or "atlanta"
    ).strip().lower()
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
    *,
    city_slugs: Iterable[str] | None = None,
    as_of_date: str | None = None,
    lookback_days: int | None = None,
    exact_date: bool = False,
    write_global_skills: bool = True,
) -> WalletTrackerSummary:
    _warn_if_execution_caller()
    if not Config.WALLET_TRACKER_READ_ONLY:
        log.error("wallet_tracker: WALLET_TRACKER_READ_ONLY=false; refusing to run")
        return WalletTrackerSummary(enabled=False, errors=("read_only_disabled",))
    if not Config.WALLET_TRACKER_ENABLED:
        return WalletTrackerSummary(enabled=False)

    from backend.storage.db import get_session
    from backend.storage.repos import (
        bulk_upsert_wallet_market_exposures,
        bulk_upsert_wallet_skill_scores,
        bulk_upsert_wallet_stats,
        bulk_upsert_wallet_trades,
    )
    from backend.tz_utils import city_local_date

    adapter = adapter or PolymarketDataApiTradeAdapter()
    cities = await _wallet_tracker_cities(city_slugs)
    total_conditions = 0
    total_trades = 0
    total_wallets = 0
    top_score: float | None = None
    errors: list[str] = []
    all_market_refs: list[MarketRef] = []
    all_exposures: list[WalletExposureMetric] = []

    for city in cities:
        city_as_of_date = as_of_date or city_local_date(city)
        market_refs = await _discover_market_refs_for_city(
            city,
            as_of_date=city_as_of_date,
            lookback_days=lookback_days,
            exact_date=exact_date,
        )
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
        trades = dedupe_public_trades(trades)
        total_trades += len(trades)
        market_by_condition = {m.condition_id: m for m in market_refs}
        exposures = compute_wallet_exposures(trades, market_refs)
        all_market_refs.extend(market_refs)
        all_exposures.extend(exposures)
        async with get_session() as sess:
            trade_rows = [
                _trade_to_db_kwargs(trade, market)
                for trade in trades
                if (market := market_by_condition.get(trade.condition_id)) is not None
            ]
            await bulk_upsert_wallet_trades(sess, trade_rows)
            await bulk_upsert_wallet_market_exposures(
                sess,
                [exposure.to_db_kwargs() for exposure in exposures],
            )
            await sess.commit()
        metrics = compute_wallet_metrics(
            trades,
            market_refs,
            min_volume_usd=Config.WALLET_TRACKER_MIN_VOLUME_USD,
            min_trades=Config.WALLET_TRACKER_MIN_TRADES,
            min_active_days=Config.WALLET_TRACKER_MIN_ACTIVE_DAYS,
            max_wallets=Config.WALLET_TRACKER_MAX_WALLETS_PER_CITY,
            as_of_date=city_as_of_date,
        )
        async with get_session() as sess:
            await bulk_upsert_wallet_stats(
                sess,
                [metric.to_db_kwargs() for metric in metrics],
            )
            await sess.commit()
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

    if all_exposures:
        global_skills = (
            compute_wallet_skill_scores(
                all_exposures,
                all_market_refs,
                scope="global",
                window_days=Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
                min_resolved_markets=Config.WALLET_TRACKER_MIN_RESOLVED_MARKETS,
                min_volume_usd=Config.WALLET_TRACKER_MIN_VOLUME_USD,
                min_active_days=Config.WALLET_TRACKER_MIN_ACTIVE_DAYS,
            )
            if write_global_skills else []
        )
        city_skill_rows: list[WalletSkillMetric] = []
        for city in cities:
            city_skill_rows.extend(
                compute_wallet_skill_scores(
                    all_exposures,
                    all_market_refs,
                    scope="city",
                    city_slug=city.city_slug,
                    window_days=Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
                    min_resolved_markets=Config.WALLET_TRACKER_MIN_RESOLVED_MARKETS,
                    min_volume_usd=Config.WALLET_TRACKER_MIN_VOLUME_USD,
                    min_active_days=Config.WALLET_TRACKER_MIN_ACTIVE_DAYS,
                )
            )
        async with get_session() as sess:
            await bulk_upsert_wallet_skill_scores(
                sess,
                [skill.to_db_kwargs() for skill in global_skills + city_skill_rows],
            )
            await sess.commit()

    return WalletTrackerSummary(
        enabled=True,
        cities_scanned=len(cities),
        condition_ids_scanned=total_conditions,
        trades_fetched=total_trades,
        wallets_updated=total_wallets,
        top_wallet_score=top_score,
        errors=tuple(errors),
    )


async def refresh_wallet_rankings_for_city_date(
    city_slug: str,
    date_et: str,
    adapter: PolymarketDataApiTradeAdapter | None = None,
    *,
    include_global_skills: bool = False,
    include_history_skills: bool = False,
) -> WalletTrackerSummary:
    """Refresh read-only wallet analytics for one city page/date.

    Manual city refreshes should not overwrite global skill rankings with a
    one-city slice unless the caller explicitly opts in. Historical skill
    refreshes rebuild city-specific wallet skill from recent resolved markets
    so the current-market table can show city rank/accuracy/PnL context.
    """
    current = await update_wallet_rankings(
        adapter=adapter,
        city_slugs=[city_slug],
        as_of_date=date_et,
        lookback_days=1,
        exact_date=True,
        write_global_skills=False,
    )
    if not include_history_skills:
        return current

    history = await update_wallet_rankings(
        adapter=adapter,
        city_slugs=[city_slug],
        as_of_date=date_et,
        lookback_days=Config.WALLET_TRACKER_LOOKBACK_DAYS,
        exact_date=False,
        write_global_skills=include_global_skills,
    )
    return WalletTrackerSummary(
        enabled=current.enabled and history.enabled,
        cities_scanned=max(current.cities_scanned, history.cities_scanned),
        condition_ids_scanned=current.condition_ids_scanned + history.condition_ids_scanned,
        trades_fetched=current.trades_fetched + history.trades_fetched,
        wallets_updated=max(current.wallets_updated, history.wallets_updated),
        top_wallet_score=max(
            current.top_wallet_score or 0.0,
            history.top_wallet_score or 0.0,
        ) or None,
        errors=tuple([*current.errors, *history.errors]),
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


def serialize_wallet_skill_score(row: Any, *, truncate_addresses: bool = True) -> dict[str, Any]:
    address = str(getattr(row, "wallet_address", "") or "").lower()
    last_active_ts = getattr(row, "last_active_ts", None)
    return {
        "wallet_address": address,
        "display_address": truncate_wallet_address(address, enabled=truncate_addresses),
        "profile_url": wallet_profile_url(address),
        "scope": getattr(row, "scope", None),
        "city_slug": getattr(row, "city_slug", None),
        "window_days": getattr(row, "window_days", None),
        "rank": getattr(row, "rank", None),
        "adjusted_score": getattr(row, "adjusted_score", None),
        "win_rate": getattr(row, "win_rate", None),
        "wilson_win_rate": getattr(row, "wilson_win_rate", None),
        "resolved_markets": getattr(row, "resolved_markets", 0) or 0,
        "total_markets": getattr(row, "total_markets", 0) or 0,
        "total_volume_usd": round(float(getattr(row, "total_volume_usd", 0.0) or 0.0), 2),
        "realized_pnl": round(float(getattr(row, "realized_pnl", 0.0) or 0.0), 2),
        "roi": getattr(row, "roi", None),
        "profit_factor": getattr(row, "profit_factor", None),
        "avg_notional_usd": getattr(row, "avg_notional_usd", None),
        "active_days": getattr(row, "active_days", 0) or 0,
        "last_active_ts": last_active_ts.isoformat() if isinstance(last_active_ts, datetime) else last_active_ts,
        "win_rate_label": (
            f"{round((getattr(row, 'win_rate', 0) or 0) * 100):.0f}% over {getattr(row, 'resolved_markets', 0) or 0}"
            if getattr(row, "win_rate", None) is not None else "—"
        ),
    }


def _recency_weight(last_trade_ts: datetime | None, now: datetime | None = None) -> float:
    if not last_trade_ts:
        return 0.25
    now = now or datetime.now(timezone.utc)
    if last_trade_ts.tzinfo is None:
        last_trade_ts = last_trade_ts.replace(tzinfo=timezone.utc)
    age_minutes = max(0.0, (now - last_trade_ts.astimezone(timezone.utc)).total_seconds() / 60.0)
    return _clamp(math.exp(-age_minutes / 240.0), 0.15, 1.0)


def _notional_weight(net_notional_usd: float) -> float:
    return _clamp(math.log1p(abs(net_notional_usd)) / math.log1p(5000.0), 0.0, 1.0)


def serialize_current_exposure_row(
    exposure: Any,
    *,
    global_skill: Any | None,
    city_skill: Any | None,
    legacy_city_stat: Any | None = None,
    legacy_city_rank: int | None = None,
    truncate_addresses: bool = True,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    has_global_skill = global_skill is not None
    global_score = (
        float(getattr(global_skill, "adjusted_score", 0.0) or 0.0)
        if has_global_skill else 0.0
    )
    is_ranked = has_global_skill and global_score >= Config.WALLET_TRACKER_MIN_ADJUSTED_SCORE
    city_score = float(getattr(city_skill, "adjusted_score", 0.0) or 0.0) if city_skill else 0.75
    net_notional = float(getattr(exposure, "net_notional_usd", 0.0) or 0.0)
    net_qty = float(getattr(exposure, "net_position_qty", 0.0) or 0.0)
    if abs(net_notional) <= 1e-9 and abs(net_qty) <= 1e-9:
        return None
    last_trade_ts = getattr(exposure, "last_trade_ts", None)
    recency = _recency_weight(last_trade_ts, now=now)
    has_legacy_city_stat = legacy_city_stat is not None
    legacy_score = (
        float(getattr(legacy_city_stat, "consistency_score", 0.0) or 0.0)
        if has_legacy_city_stat else 0.0
    )
    if is_ranked:
        alpha_score = global_score * city_score * _notional_weight(net_notional) * recency
        skill_source = "wallet_skill_scores"
    elif has_legacy_city_stat and legacy_score >= Config.WALLET_TRACKER_MIN_ADJUSTED_SCORE:
        is_ranked = True
        city_score = legacy_score
        alpha_score = legacy_score * _notional_weight(net_notional) * recency
        skill_source = "wallet_stats_city_fallback"
    else:
        # Show live positioning even before skill is mature, but keep the
        # alpha score deliberately small so unproven wallets do not dominate
        # smart-money/model confluence.
        alpha_score = 0.05 * _notional_weight(net_notional) * recency
        skill_source = "unscored_current_exposure"
    address = str(getattr(exposure, "wallet_address", "") or "").lower()
    if isinstance(last_trade_ts, datetime):
        if last_trade_ts.tzinfo is None:
            last_trade_ts = last_trade_ts.replace(tzinfo=timezone.utc)
        now_dt = now or datetime.now(timezone.utc)
        last_trade_age_min = max(0, int((now_dt - last_trade_ts.astimezone(timezone.utc)).total_seconds() // 60))
        last_trade = last_trade_ts.isoformat()
    else:
        last_trade_age_min = None
        last_trade = last_trade_ts
    direction = "LONG" if net_qty > 0 else ("EXITING" if net_notional < 0 else "FLAT")
    fallback_volume = float(getattr(legacy_city_stat, "volume_usd", 0.0) or 0.0) if legacy_city_stat else 0.0
    fallback_pnl = (
        float(getattr(legacy_city_stat, "realized_pnl", 0.0) or 0.0)
        + float(getattr(legacy_city_stat, "unrealized_pnl", 0.0) or 0.0)
        if legacy_city_stat else 0.0
    )
    fallback_roi = fallback_pnl / fallback_volume if fallback_volume > 0 else None
    return {
        "wallet_address": address,
        "display_address": truncate_wallet_address(address, enabled=truncate_addresses),
        "profile_url": wallet_profile_url(address),
        "global_rank": getattr(global_skill, "rank", None),
        "city_rank": getattr(city_skill, "rank", None) if city_skill else legacy_city_rank,
        "global_score": global_score if has_global_skill else None,
        "city_score": getattr(city_skill, "adjusted_score", None) if city_skill else (legacy_score if has_legacy_city_stat else None),
        "alpha_score": round(alpha_score, 4),
        "is_ranked": bool(is_ranked),
        "skill_source": skill_source,
        "win_rate": getattr(global_skill, "win_rate", None) if global_skill else getattr(legacy_city_stat, "win_rate", None),
        "wilson_win_rate": getattr(global_skill, "wilson_win_rate", None) if global_skill else None,
        "resolved_markets": getattr(global_skill, "resolved_markets", 0) if global_skill else getattr(legacy_city_stat, "trade_count", 0) if legacy_city_stat else 0,
        "roi": getattr(global_skill, "roi", None) if global_skill else fallback_roi,
        "profit_factor": getattr(global_skill, "profit_factor", None) if global_skill else None,
        "bucket_idx": getattr(exposure, "bucket_idx", None),
        "bucket_label": getattr(exposure, "bucket_label", None),
        "condition_id": getattr(exposure, "condition_id", None),
        "market_slug": getattr(exposure, "market_slug", None),
        "direction": direction,
        "net_position_qty": round(net_qty, 4),
        "net_notional_usd": round(net_notional, 2),
        "trade_count": getattr(exposure, "trade_count", 0) or 0,
        "volume_usd": round(float(getattr(exposure, "volume_usd", 0.0) or 0.0), 2),
        "avg_entry_price": getattr(exposure, "avg_entry_price", None),
        "realized_pnl": getattr(exposure, "realized_pnl", None),
        "unrealized_pnl": getattr(exposure, "unrealized_pnl", None),
        "last_trade_ts": last_trade,
        "last_trade_age_min": last_trade_age_min,
    }


def build_bucket_consensus(
    buckets: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    consensus_by_idx: dict[int, dict[str, Any]] = {}
    for bucket in buckets:
        idx = bucket.get("bucket_idx")
        if idx is None:
            continue
        idx_int = int(idx)
        consensus_by_idx[idx_int] = {
            "bucket_idx": idx_int,
            "bucket_label": bucket.get("label"),
            "model_prob": bucket.get("model_prob"),
            "wallets_long": 0,
            "ranked_wallets_long": 0,
            "net_notional_usd": 0.0,
            "net_position_qty": 0.0,
            "weighted_flow": 0.0,
            "avg_entry_price": None,
            "_entry_num": 0.0,
            "_entry_den": 0.0,
        }
    for row in current_rows:
        idx = row.get("bucket_idx")
        if idx is None or idx not in consensus_by_idx:
            continue
        net_qty = float(row.get("net_position_qty") or 0.0)
        net_notional = float(row.get("net_notional_usd") or row.get("net_flow_usd") or 0.0)
        if net_qty <= 0 and net_notional <= 0:
            continue
        c = consensus_by_idx[idx]
        c["wallets_long"] += 1
        if row.get("is_ranked", True):
            c["ranked_wallets_long"] += 1
        c["net_notional_usd"] += net_notional
        c["net_position_qty"] += net_qty
        score_weight = float(
            row.get("global_score")
            or row.get("consistency_score")
            or row.get("alpha_score")
            or 1.0
        )
        c["weighted_flow"] += net_notional * score_weight
        avg_entry = row.get("avg_entry_price")
        if avg_entry is not None and net_qty > 0:
            c["_entry_num"] += float(avg_entry) * net_qty
            c["_entry_den"] += net_qty
    consensus = []
    for c in consensus_by_idx.values():
        if c["_entry_den"] > 0:
            c["avg_entry_price"] = round(c["_entry_num"] / c["_entry_den"], 4)
        c["net_notional_usd"] = round(c["net_notional_usd"], 2)
        c["net_position_qty"] = round(c["net_position_qty"], 4)
        c["weighted_flow"] = round(c["weighted_flow"], 4)
        c.pop("_entry_num", None)
        c.pop("_entry_den", None)
        consensus.append(c)
    consensus.sort(key=lambda x: x["bucket_idx"])
    return consensus


def classify_model_confluence(
    buckets: list[dict[str, Any]],
    bucket_consensus: list[dict[str, Any]],
) -> dict[str, Any]:
    model_candidates = [
        b for b in buckets
        if b.get("bucket_idx") is not None and b.get("model_prob") is not None
    ]
    if not model_candidates:
        return {"status": "unavailable", "badge": "NO MODEL", "reason": "model_bucket_unavailable"}
    model_bucket = max(model_candidates, key=lambda b: b.get("model_prob") or 0.0)
    flow_candidates = [
        c for c in bucket_consensus
        if c.get("ranked_wallets_long", 0) > 0 and c.get("weighted_flow", 0.0) > 0
    ]
    if not flow_candidates:
        return {
            "status": "unavailable",
            "badge": "NO RANKED FLOW",
            "reason": "no_ranked_wallet_flow",
            "model_bucket_idx": model_bucket.get("bucket_idx"),
            "model_bucket_label": model_bucket.get("label"),
        }
    smart_bucket = max(flow_candidates, key=lambda c: (c.get("weighted_flow") or 0.0, c.get("net_notional_usd") or 0.0))
    model_idx = int(model_bucket["bucket_idx"])
    smart_idx = int(smart_bucket["bucket_idx"])
    distance = abs(model_idx - smart_idx)
    if distance == 0:
        badge = "CONFIRMS MODEL"
    elif distance == 1:
        badge = "ADJACENT"
    else:
        badge = "DIVERGES"
    return {
        "status": "available",
        "badge": badge,
        "model_bucket_idx": model_idx,
        "model_bucket_label": model_bucket.get("label"),
        "smart_money_bucket_idx": smart_idx,
        "smart_money_bucket_label": smart_bucket.get("bucket_label"),
        "smart_money_net_flow_usd": smart_bucket.get("net_notional_usd"),
        "ranked_wallets_long": smart_bucket.get("ranked_wallets_long"),
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


def _wallet_tracker_message(status: str, reason: str | None = None) -> str:
    if status == "disabled" or reason == "wallet_tracker_disabled":
        return "Wallet tracker is disabled by configuration."
    if status == "error":
        return "Unable to load wallets - check wallet tracker configuration or public trade ingestion."
    return "No wallet trades have been stored for this city/date yet."


def _latest_row_timestamp(*row_groups: list[Any]) -> str | None:
    latest: datetime | None = None
    for rows in row_groups:
        for row in rows or []:
            for attr in ("last_updated_ts", "last_trade_ts", "last_active_ts"):
                value = getattr(row, attr, None)
                if isinstance(value, datetime):
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=timezone.utc)
                    value = value.astimezone(timezone.utc)
                    if latest is None or value > latest:
                        latest = value
    return latest.isoformat() if latest else None


def _serialize_wallet_stat_as_current_row(
    row: Any,
    *,
    rank: int,
    truncate_addresses: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Map the v1 wallet_stats read model into the V2 current-market shape."""
    base = serialize_wallet_stat_row(row, truncate_addresses=truncate_addresses)
    last_trade_ts = getattr(row, "last_trade_ts", None)
    if isinstance(last_trade_ts, datetime):
        if last_trade_ts.tzinfo is None:
            last_trade_ts = last_trade_ts.replace(tzinfo=timezone.utc)
        now_dt = now or datetime.now(timezone.utc)
        last_trade_age_min = max(
            0,
            int((now_dt - last_trade_ts.astimezone(timezone.utc)).total_seconds() // 60),
        )
    else:
        last_trade_age_min = None
    consistency = float(getattr(row, "consistency_score", 0.0) or 0.0)
    net_flow = float(getattr(row, "net_flow_usd", 0.0) or 0.0)
    net_qty = float(getattr(row, "net_position_qty", 0.0) or 0.0)
    volume = float(getattr(row, "volume_usd", 0.0) or 0.0)
    return {
        **base,
        "global_rank": rank,
        "city_rank": rank,
        "global_score": consistency,
        "city_score": consistency,
        "alpha_score": round(consistency * _notional_weight(net_flow or volume), 4),
        "wilson_win_rate": None,
        "resolved_markets": getattr(row, "trade_count", 0) or 0,
        "roi": (
            round(float(getattr(row, "realized_pnl", 0.0) or 0.0) / volume, 4)
            if volume > 0 else None
        ),
        "profit_factor": None,
        "direction": "LONG" if net_qty > 0 or net_flow > 0 else ("EXITING" if net_flow < 0 else "FLAT"),
        "net_notional_usd": round(net_flow, 2),
        "last_trade_age_min": last_trade_age_min,
        "source": "wallet_stats_fallback",
    }


def _serialize_wallet_stat_as_leader(
    row: Any,
    *,
    scope: str,
    rank: int,
    truncate_addresses: bool = True,
) -> dict[str, Any]:
    base = serialize_wallet_stat_row(row, truncate_addresses=truncate_addresses)
    volume = float(getattr(row, "volume_usd", 0.0) or 0.0)
    realized = float(getattr(row, "realized_pnl", 0.0) or 0.0)
    win_rate = getattr(row, "win_rate", None)
    trade_count = getattr(row, "trade_count", 0) or 0
    return {
        "wallet_address": base["wallet_address"],
        "display_address": base["display_address"],
        "profile_url": base["profile_url"],
        "scope": scope,
        "city_slug": getattr(row, "city_slug", None) if scope == "city" else "",
        "window_days": Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
        "rank": rank,
        "adjusted_score": getattr(row, "consistency_score", None),
        "win_rate": win_rate,
        "wilson_win_rate": None,
        "resolved_markets": trade_count,
        "total_markets": trade_count,
        "total_volume_usd": round(volume, 2),
        "realized_pnl": round(realized, 2),
        "roi": round(realized / volume, 4) if volume > 0 else None,
        "profit_factor": None,
        "avg_notional_usd": round(volume / max(1, trade_count), 2) if trade_count else None,
        "active_days": 1,
        "last_active_ts": base["last_trade_ts"],
        "win_rate_label": (
            f"{round(float(win_rate) * 100):.0f}% over {trade_count}"
            if win_rate is not None else "—"
        ),
        "source": "wallet_stats_fallback",
    }


def _serialize_wallet_stat_leaders(
    rows: list[Any],
    *,
    scope: str,
    limit: int,
    truncate_addresses: bool = True,
) -> list[dict[str, Any]]:
    leaders: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        address = str(getattr(row, "wallet_address", "") or "").lower()
        if not address or address in seen:
            continue
        seen.add(address)
        leaders.append(
            _serialize_wallet_stat_as_leader(
                row,
                scope=scope,
                rank=len(leaders) + 1,
                truncate_addresses=truncate_addresses,
            )
        )
        if len(leaders) >= limit:
            break
    return leaders


# Regression note:
# V1 displayed wallets from wallet_stats through get_wallet_stats_for_city().
# Smart Money V2 changed city_detail to call get_wallet_leaderboard_payload()
# with buckets=..., and that short-circuited into the new V2 payload builder.
# The V2 builder only read wallet_skill_scores and wallet_market_exposures, and
# serialize_current_exposure_row() required a global skill row before rendering
# any exposure. In production, those normalized V2 tables may be empty, not yet
# backfilled, or filtered below threshold while wallet_stats still has usable
# rows. The displayed current rows became empty, and ranked flow disappeared
# because bucket consensus was derived from those same suppressed current rows.
# Keep wallet_stats as the compatibility floor until normalized V2 ingestion has
# a complete backfill. This module remains read-only analytics only.
def _empty_weather_smart_money_payload(
    *,
    enabled: bool,
    status: str,
    reason: str | None = None,
    buckets: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    bucket_consensus = build_bucket_consensus(buckets or [], [])
    confluence = classify_model_confluence(buckets or [], bucket_consensus)
    if reason and confluence.get("status") == "unavailable":
        confluence["reason"] = reason
    return {
        "enabled": enabled,
        "status": status,
        "reason": reason,
        "message": _wallet_tracker_message(status, reason),
        "mode": "weather_smart_money_v2",
        "current_source": "none",
        "coverage": {
            "current_source": "none",
            "current_market_wallets": 0,
            "bucket_consensus_buckets": len(bucket_consensus),
            "legacy_current_rows": 0,
            "legacy_city_rows": 0,
            "legacy_global_rows": 0,
            "exposure_rows": 0,
            "global_skill_rows": 0,
            "city_skill_rows": 0,
            "display_limit": limit or Config.WALLET_TRACKER_DISPLAY_LIMIT,
            "window_days": Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
            "last_refresh": None,
        },
        "rows": [],
        "current_market": [],
        "global_leaders": [],
        "city_leaders": [],
        "bucket_consensus": bucket_consensus,
        "confluence": confluence,
        "display_limit": limit or Config.WALLET_TRACKER_DISPLAY_LIMIT,
        "window_days": Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
        "disclaimer": (
            "Wallet leaderboard is read-only public-market analytics. "
            "It is not a copy-trading signal and does not trigger automated trades."
        ),
    }


async def get_weather_smart_money_payload(
    city_slug: str,
    date_et: str,
    *,
    buckets: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return V2 smart-money payload for a city/date page.

    This is read-only context. It intentionally has no dependency on execution
    modules and must not be used as an automated trading signal.
    """
    _warn_if_execution_caller()
    from backend.storage.db import get_session
    from backend.storage.repos import (
        get_wallet_stats_for_city,
        get_wallet_stats_leaderboard,
        get_wallet_market_exposures_for_event,
        get_wallet_skill_scores,
    )
    from sqlalchemy import func, select
    from backend.storage.models import WalletTrade

    limit = limit or Config.WALLET_TRACKER_DISPLAY_LIMIT
    buckets = buckets or []
    fallback_limit = max(100, limit * 10)
    if not Config.WALLET_TRACKER_ENABLED:
        return _empty_weather_smart_money_payload(
            enabled=False,
            status="disabled",
            reason="wallet_tracker_disabled",
            buckets=buckets,
            limit=limit,
        )

    async with get_session() as sess:
        legacy_current_rows = await get_wallet_stats_for_city(
            sess,
            city_slug,
            date_et=date_et,
            limit=fallback_limit,
        )
        legacy_city_rows = await get_wallet_stats_leaderboard(
            sess,
            city_slug=city_slug,
            limit=fallback_limit,
        )
        legacy_global_rows = await get_wallet_stats_leaderboard(
            sess,
            limit=fallback_limit,
        )
        global_rows = await get_wallet_skill_scores(
            sess,
            scope="global",
            city_slug="",
            window_days=Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
            limit=limit,
        )
        city_rows = await get_wallet_skill_scores(
            sess,
            scope="city",
            city_slug=city_slug,
            window_days=Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
            limit=limit,
        )
        exposure_rows = await get_wallet_market_exposures_for_event(
            sess,
            city_slug,
            date_et,
            limit=max(500, limit * 20),
        )
        trade_coverage = (
            await sess.execute(
                select(
                    func.count(WalletTrade.id),
                    func.count(func.distinct(WalletTrade.wallet_address)),
                    func.count(func.distinct(WalletTrade.condition_id)),
                ).where(
                    WalletTrade.city_slug == city_slug,
                    WalletTrade.date == date_et,
                )
            )
        ).one()

    global_by_wallet = {
        str(row.wallet_address).lower(): row
        for row in global_rows
    }
    city_by_wallet = {
        str(row.wallet_address).lower(): row
        for row in city_rows
    }
    legacy_city_by_wallet: dict[str, tuple[int, Any]] = {}
    for idx, row in enumerate(legacy_city_rows, start=1):
        address = str(getattr(row, "wallet_address", "") or "").lower()
        if address and address not in legacy_city_by_wallet:
            legacy_city_by_wallet[address] = (idx, row)

    current_rows = []
    for exposure in exposure_rows:
        address = str(getattr(exposure, "wallet_address", "") or "").lower()
        legacy_rank, legacy_stat = legacy_city_by_wallet.get(address, (None, None))
        row = serialize_current_exposure_row(
            exposure,
            global_skill=global_by_wallet.get(address),
            city_skill=city_by_wallet.get(address),
            legacy_city_stat=legacy_stat,
            legacy_city_rank=legacy_rank,
            truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
        )
        if row:
            current_rows.append(row)
    current_rows.sort(
        key=lambda row: (
            row.get("alpha_score") or 0.0,
            abs(float(row.get("net_notional_usd") or 0.0)),
            row.get("global_score") or 0.0,
        ),
        reverse=True,
    )
    current_rows = current_rows[:limit]
    current_source = "wallet_market_exposures"
    if not current_rows and legacy_current_rows:
        current_rows = [
            _serialize_wallet_stat_as_current_row(
                row,
                rank=idx,
                truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
            )
            for idx, row in enumerate(legacy_current_rows[:limit], start=1)
        ]
        current_source = "wallet_stats_fallback"
    elif not current_rows:
        current_source = "none"
    bucket_consensus = build_bucket_consensus(buckets, current_rows)
    confluence = classify_model_confluence(buckets, bucket_consensus)
    global_leaders = [
        serialize_wallet_skill_score(
            row,
            truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
        )
        for row in global_rows[:limit]
    ]
    if not global_leaders:
        global_leaders = _serialize_wallet_stat_leaders(
            legacy_global_rows,
            scope="global",
            limit=limit,
            truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
        )
    city_leaders = [
        serialize_wallet_skill_score(
            row,
            truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
        )
        for row in city_rows[:limit]
    ]
    if not city_leaders:
        city_leaders = _serialize_wallet_stat_leaders(
            legacy_city_rows,
            scope="city",
            limit=limit,
            truncate_addresses=Config.WALLET_TRACKER_TRUNCATE_ADDRESSES,
        )
    has_wallet_data = bool(current_rows or global_leaders or city_leaders)
    status = "ok" if has_wallet_data else "empty"
    reason = None if has_wallet_data else "no_wallet_data_for_city_date"
    coverage = {
        "current_source": current_source,
        "current_market_wallets": len(current_rows),
        "bucket_consensus_buckets": len(bucket_consensus),
        "legacy_current_rows": len(legacy_current_rows),
        "legacy_city_rows": len(legacy_city_rows),
        "legacy_global_rows": len(legacy_global_rows),
        "wallet_trade_rows": int(trade_coverage[0] or 0),
        "wallets_scanned": int(trade_coverage[1] or 0),
        "condition_ids_scanned": int(trade_coverage[2] or 0),
        "exposure_rows": len(exposure_rows),
        "global_skill_rows": len(global_rows),
        "city_skill_rows": len(city_rows),
        "display_limit": limit,
        "window_days": Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
        "last_refresh": _latest_row_timestamp(
            legacy_current_rows,
            legacy_city_rows,
            legacy_global_rows,
            exposure_rows,
            global_rows,
            city_rows,
        ),
    }

    return {
        "enabled": True,
        "status": status,
        "reason": reason,
        "message": _wallet_tracker_message(status, reason),
        "mode": "weather_smart_money_v2",
        "current_source": current_source,
        "coverage": coverage,
        "rows": current_rows,
        "current_market": current_rows,
        "global_leaders": global_leaders,
        "city_leaders": city_leaders,
        "bucket_consensus": bucket_consensus,
        "confluence": confluence,
        "display_limit": limit,
        "window_days": Config.WALLET_TRACKER_SKILL_WINDOW_DAYS,
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
    buckets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _warn_if_execution_caller()
    from backend.storage.db import get_session
    from backend.storage.repos import get_wallet_stats_for_city

    if buckets is not None:
        return await get_weather_smart_money_payload(
            city_slug,
            date_et,
            buckets=buckets,
            limit=limit,
        )

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
