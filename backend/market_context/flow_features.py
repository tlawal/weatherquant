"""Lightweight active-market shadow flow refresh.

This module keeps CLOB/Data API flow features fresh for held markets without
running the full smart-wallet leaderboard scan. The output remains shadow-only:
execution code may display diagnostics, but must not gate trades on these rows
until out-of-sample promotion tests pass.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from backend.config import Config
from backend.execution.microstructure import compute_shadow_flow_features
from backend.market_context.wallet_tracker import (
    MarketRef,
    PolymarketDataApiTradeAdapter,
    PublicTrade,
    dedupe_public_trades,
    _trade_to_db_kwargs,
)
from backend.storage.db import get_session
from backend.storage.repos import (
    bulk_upsert_wallet_trades,
    get_market_flow_refresh_targets,
    insert_market_flow_feature,
)
from backend.tz_utils import et_today

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveFlowRefreshSummary:
    enabled: bool
    targets: int = 0
    conditions: int = 0
    trades_fetched: int = 0
    trades_written: int = 0
    feature_rows_written: int = 0
    errors: tuple[str, ...] = ()


def _target_to_market_ref(target: dict) -> MarketRef:
    return MarketRef(
        city_slug=str(target.get("city_slug") or ""),
        date=str(target.get("date_et") or ""),
        condition_id=str(target.get("condition_id") or ""),
        bucket_id=int(target["bucket_id"]) if target.get("bucket_id") is not None else None,
        market_slug=target.get("market_slug"),
        bucket_idx=target.get("bucket_idx"),
        bucket_label=target.get("bucket_label"),
    )


async def refresh_active_market_flow_features(
    adapter: PolymarketDataApiTradeAdapter | None = None,
    *,
    bucket_ids: Iterable[int] | None = None,
    open_positions_only: bool | None = None,
    windows: tuple[int, ...] = (5, 15, 60),
    as_of: datetime | None = None,
    limit: int | None = None,
) -> ActiveFlowRefreshSummary:
    """Fetch public trades and write shadow flow rows for active targets."""
    if not Config.MARKET_FLOW_REFRESH_ENABLED:
        return ActiveFlowRefreshSummary(enabled=False)

    as_of = as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    open_only = (
        Config.MARKET_FLOW_REFRESH_OPEN_POSITIONS_ONLY
        if open_positions_only is None
        else bool(open_positions_only)
    )
    target_limit = limit or Config.MARKET_FLOW_ACTIVE_BUCKETS_LIMIT

    async with get_session() as sess:
        targets = await get_market_flow_refresh_targets(
            sess,
            bucket_ids=[int(x) for x in bucket_ids] if bucket_ids else None,
            open_positions_only=open_only,
            min_date_et=et_today(),
            limit=target_limit,
        )

    if not targets:
        return ActiveFlowRefreshSummary(enabled=True)

    condition_ids = [str(t["condition_id"]) for t in targets if t.get("condition_id")]
    adapter = adapter or PolymarketDataApiTradeAdapter(timeout_s=8.0)
    errors: list[str] = []
    try:
        trades = await adapter.fetch_trades_for_markets(
            condition_ids,
            limit=max(1, Config.MARKET_FLOW_FETCH_LIMIT),
        )
    except Exception as exc:
        log.warning("market_flow: active public-trade fetch failed: %s", exc)
        trades = []
        errors.append(f"fetch:{type(exc).__name__}")
    trades = dedupe_public_trades(trades)
    trades_by_condition: dict[str, list[PublicTrade]] = {}
    for trade in trades:
        trades_by_condition.setdefault(str(trade.condition_id), []).append(trade)

    market_refs = {
        str(target["condition_id"]): _target_to_market_ref(target)
        for target in targets
        if target.get("condition_id")
    }
    trade_rows = [
        _trade_to_db_kwargs(trade, market_refs[trade.condition_id])
        for trade in trades
        if trade.condition_id in market_refs
    ]

    feature_rows_written = 0
    trades_written = 0
    async with get_session() as sess:
        trades_written = await bulk_upsert_wallet_trades(sess, trade_rows)
        for target in targets:
            condition_id = str(target.get("condition_id") or "")
            bucket_id = target.get("bucket_id")
            if not condition_id or bucket_id is None:
                continue
            market_trades = trades_by_condition.get(condition_id, [])
            for window_minutes in windows:
                features = compute_shadow_flow_features(
                    market_trades,
                    as_of=as_of,
                    window_minutes=int(window_minutes),
                )
                await insert_market_flow_feature(
                    sess,
                    commit=False,
                    bucket_id=int(bucket_id),
                    computed_at=as_of,
                    window_minutes=int(window_minutes),
                    signed_net_notional=features["signed_net_notional"],
                    buy_notional=features["buy_notional"],
                    sell_notional=features["sell_notional"],
                    imbalance=features["imbalance"],
                    vpin=features["vpin"],
                    toxicity_score=features["toxicity_score"],
                    top_wallet_weighted_flow=features["top_wallet_weighted_flow"],
                    direction_source=features["direction_source"],
                    direction_confidence=features["direction_confidence"],
                    raw_json=json.dumps(features, default=str, separators=(",", ":")),
                )
                feature_rows_written += 1
        await sess.commit()

    log.info(
        "market_flow: refreshed targets=%d conditions=%d trades=%d features=%d errors=%d",
        len(targets),
        len(set(condition_ids)),
        len(trades),
        feature_rows_written,
        len(errors),
    )
    return ActiveFlowRefreshSummary(
        enabled=True,
        targets=len(targets),
        conditions=len(set(condition_ids)),
        trades_fetched=len(trades),
        trades_written=trades_written,
        feature_rows_written=feature_rows_written,
        errors=tuple(errors),
    )
