from __future__ import annotations

import logging
import aiohttp
from backend.storage.db import get_session
from backend.storage.repos import get_position

log = logging.getLogger(__name__)

async def sync_positions_from_chain() -> dict:
    """Sync DB positions from Polymarket data API (restores incorrectly zeroed positions)."""
    from backend.execution.chain_utils import get_wallet_address

    addr = get_wallet_address()
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        url = f"https://data-api.polymarket.com/positions?user={addr}"
        async with http.get(url) as resp:
            if resp.status != 200:
                log.error("sync_positions: Polymarket API returned %d", resp.status)
                return {"ok": False, "error": f"Polymarket API returned {resp.status}"}
            api_positions = await resp.json()

    corrections = []
    for api_pos in api_positions:
        size = float(api_pos.get("size", 0))
        if size <= 0:
            continue
        asset = api_pos.get("asset", "")
        avg_price = float(api_pos.get("avgPrice", 0))
        title = api_pos.get("title", "")

        # Find matching bucket by yes_token_id
        async with get_session() as sess:
            from sqlalchemy import select
            from backend.storage.models import Bucket, Position
            result = await sess.execute(
                select(Bucket).where(Bucket.yes_token_id == asset)
            )
            bucket = result.scalar_one_or_none()
            if not bucket:
                continue

            pos = await get_position(sess, bucket.id)
            if pos and abs(pos.net_qty - size) < 0.01:
                continue  # already correct

            if pos:
                old_qty = pos.net_qty
                pos.net_qty = size
                pos.avg_cost = avg_price
                if pos.last_mkt_price:
                    pos.unrealized_pnl = size * (pos.last_mkt_price - avg_price)
                await sess.commit()
            else:
                new_pos = Position(
                    bucket_id=bucket.id,
                    side="yes",
                    net_qty=size,
                    avg_cost=avg_price,
                    last_mkt_price=avg_price,
                    unrealized_pnl=0.0,
                )
                sess.add(new_pos)
                old_qty = 0
                await sess.commit()

            corrections.append({
                "bucket_id": bucket.id,
                "title": title,
                "old_qty": old_qty,
                "new_qty": size,
                "avg_price": avg_price,
            })

    if corrections:
        log.info("sync_positions: updated %d positions from chain", len(corrections))

    return {"ok": True, "synced": len(corrections), "corrections": corrections}
