from __future__ import annotations

import asyncio
import logging
import aiohttp
from datetime import datetime, timezone
from backend.storage.db import get_session
from backend.storage.repos import get_position

log = logging.getLogger(__name__)

async def _fetch_api_positions(addr: str, timeout_s: float) -> list[dict]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        url = f"https://data-api.polymarket.com/positions?user={addr}"
        async with http.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Polymarket API returned {resp.status}")
            data = await resp.json()
            return data if isinstance(data, list) else []


async def sync_positions_from_chain(
    api_positions: list[dict] | None = None,
    *,
    http_timeout_s: float = 15.0,
    total_timeout_s: float | None = None,
) -> dict:
    """Sync DB positions from Polymarket data API.

    Matches positions by YES token, NO token, or condition ID so manual trades
    update the same DB-backed exposure used by the header and redemptions page.
    """
    from backend.execution.chain_utils import get_wallet_address

    async def _sync() -> dict:
        positions = api_positions
        if positions is None:
            addr = get_wallet_address()
            try:
                positions = await _fetch_api_positions(addr, http_timeout_s)
            except Exception as e:
                log.error("sync_positions: %s", e)
                return {"ok": False, "error": str(e)}

        corrections = []
        for api_pos in positions or []:
            size = float(api_pos.get("size", 0) or 0)
            if size <= 0:
                continue
            asset = str(api_pos.get("asset", "") or "")
            condition_id = str(api_pos.get("conditionId", "") or "")
            outcome = str(api_pos.get("outcome", "") or "").strip().lower()
            avg_price = float(api_pos.get("avgPrice", 0) or 0)
            cur_price = float(api_pos.get("curPrice", 0) or 0)
            title = api_pos.get("title", "")

            async with get_session() as sess:
                from sqlalchemy import or_, select
                from backend.storage.models import Bucket, Position

                result = await sess.execute(
                    select(Bucket).where(
                        or_(
                            Bucket.yes_token_id == asset,
                            Bucket.no_token_id == asset,
                            Bucket.condition_id == condition_id,
                        )
                    )
                )
                buckets = list(result.scalars().all())
                if not buckets:
                    continue

                bucket = next((b for b in buckets if b.yes_token_id == asset), None)
                side = "yes"
                if bucket is None:
                    bucket = next((b for b in buckets if b.no_token_id == asset), None)
                    if bucket is not None:
                        side = "no"
                if bucket is None:
                    bucket = buckets[0]
                    side = "no" if outcome == "no" else "yes"

                pos = await get_position(sess, bucket.id)
                last_mkt_price = cur_price if cur_price > 0 else avg_price
                if (
                    pos
                    and abs(pos.net_qty - size) < 0.01
                    and abs(pos.avg_cost - avg_price) < 0.0001
                    and pos.side == side
                ):
                    continue

                if pos:
                    old_qty = pos.net_qty
                    pos.side = side
                    pos.net_qty = size
                    pos.avg_cost = avg_price
                    pos.last_mkt_price = last_mkt_price
                    pos.unrealized_pnl = size * (last_mkt_price - avg_price)
                    if not pos.entry_type:
                        pos.entry_type = "MANUAL"
                    if not pos.entry_time:
                        pos.entry_time = datetime.now(timezone.utc)
                    if not pos.entry_price:
                        pos.entry_price = avg_price
                else:
                    new_pos = Position(
                        bucket_id=bucket.id,
                        side=side,
                        net_qty=size,
                        avg_cost=avg_price,
                        last_mkt_price=last_mkt_price,
                        unrealized_pnl=size * (last_mkt_price - avg_price),
                        entry_type="MANUAL",
                        entry_time=datetime.now(timezone.utc),
                        entry_price=avg_price,
                    )
                    sess.add(new_pos)
                    old_qty = 0
                await sess.commit()

                corrections.append({
                    "bucket_id": bucket.id,
                    "side": side,
                    "condition_id": condition_id,
                    "title": title,
                    "old_qty": old_qty,
                    "new_qty": size,
                    "avg_price": avg_price,
                })

        if corrections:
            log.info("sync_positions: updated %d positions from chain", len(corrections))

        return {"ok": True, "synced": len(corrections), "corrections": corrections}

    if total_timeout_s is not None:
        try:
            return await asyncio.wait_for(_sync(), timeout=total_timeout_s)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "position sync timed out"}
    return await _sync()


async def schedule_position_sync_retries(
    *,
    attempts: int = 3,
    initial_delay_s: float = 2.0,
    interval_s: float = 4.0,
) -> None:
    """Best-effort post-trade sync retries for Polymarket indexer lag."""
    for attempt in range(attempts):
        await asyncio.sleep(initial_delay_s if attempt == 0 else interval_s)
        try:
            await sync_positions_from_chain(http_timeout_s=6.0, total_timeout_s=7.0)
        except Exception:
            log.debug("post_trade_sync: retry failed", exc_info=True)
