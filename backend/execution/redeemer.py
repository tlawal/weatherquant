"""
Auto-redemption of resolved Polymarket weather markets.

Two phases:
  1. check_resolved_markets() — poll Gamma API for resolution status
  2. redeem_positions() — call redeemPositions() on NegRiskAdapter on-chain
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp

from backend.config import Config
from backend.storage.db import get_session
from backend.storage.repos import (
    get_position,
    get_unredeemed_resolved_events,
    get_unresolved_events_with_positions,
    upsert_position,
)

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_RPC = Config.POLYGON_RPC_URL
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
_TIMEOUT = aiohttp.ClientTimeout(total=15)


async def check_resolved_markets() -> int:
    """
    Check if any events with open positions have resolved on Polymarket.
    Returns count of newly resolved events.
    """
    async with get_session() as sess:
        events = await get_unresolved_events_with_positions(sess)

    if not events:
        return 0

    resolved_count = 0
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
        for event in events:
            if not event.gamma_event_id:
                continue
            try:
                url = f"{GAMMA_API}/events/{event.gamma_event_id}"
                async with http.get(url) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()

                # Gamma API: "closed" means resolved, check market outcomes
                if not data.get("closed"):
                    continue

                # Find winning bucket by checking which market resolved YES
                markets = data.get("markets", [])
                winning_idx = None
                for mkt in markets:
                    outcome = mkt.get("outcome")
                    resolved_to = mkt.get("resolvedTo")
                    if resolved_to == "Yes" or resolved_to == 1 or resolved_to == "1":
                        # Match by groupItemTitle or position in list
                        # Markets are ordered by bucket index
                        idx = markets.index(mkt)
                        winning_idx = idx
                        break

                async with get_session() as sess:
                    from sqlalchemy import update
                    from backend.storage.models import Event as EventModel
                    await sess.execute(
                        update(EventModel)
                        .where(EventModel.id == event.id)
                        .values(
                            resolved_at=datetime.now(timezone.utc),
                            winning_bucket_idx=winning_idx,
                        )
                    )
                    await sess.commit()

                resolved_count += 1
                log.info(
                    "redeemer: event %d (%s) resolved, winning_bucket_idx=%s",
                    event.id, event.date_et, winning_idx,
                )
            except Exception as e:
                log.warning("redeemer: check event %d failed: %s", event.id, e)

    return resolved_count


async def redeem_positions() -> int:
    """
    Redeem winning positions for resolved events.
    Calls redeemPositions(conditionId, indexSets) on NegRiskAdapter per bucket.
    Returns count of redeemed events.
    """
    async with get_session() as sess:
        events = await get_unredeemed_resolved_events(sess)

    if not events:
        return 0

    if not Config.POLYMARKET_PRIVATE_KEY:
        log.warning("redeemer: no private key configured, skipping redemption")
        return 0

    from eth_account import Account
    account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    sender = account.address
    chain_id = Config.CHAIN_ID

    # redeemPositions(bytes32 conditionId, uint256[] indexSets)
    # selector = keccak256("redeemPositions(bytes32,uint256[])") = 0xdbeccb23
    SELECTOR = bytes.fromhex("dbeccb23")
    # indexSets: [1, 2] for binary YES/NO outcomes (2^0=YES, 2^1=NO)
    INDEX_SETS = [1, 2]

    redeemed_count = 0
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
        for event in events:
            try:
                # Only redeem buckets where we hold a position
                buckets_to_redeem = []
                async with get_session() as sess:
                    for bucket in event.buckets:
                        if bucket.condition_id:
                            pos = await get_position(sess, bucket.id)
                            if pos and pos.net_qty > 0:
                                buckets_to_redeem.append(bucket)

                if not buckets_to_redeem:
                    # No positions to redeem — just mark as redeemed
                    async with get_session() as sess:
                        from sqlalchemy import update as sql_update
                        from backend.storage.models import Event as EventModel
                        await sess.execute(
                            sql_update(EventModel)
                            .where(EventModel.id == event.id)
                            .values(redeemed_at=datetime.now(timezone.utc))
                        )
                        await sess.commit()
                    continue

                # Get nonce and gas price once per event
                nonce_payload = {
                    "jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionCount",
                    "params": [sender, "latest"],
                }
                gas_payload = {
                    "jsonrpc": "2.0", "id": 2, "method": "eth_gasPrice", "params": [],
                }

                async with http.post(POLYGON_RPC, json=nonce_payload) as r:
                    nonce_result = await r.json()
                if "error" in nonce_result:
                    log.error("redeemer: nonce RPC error: %s", nonce_result["error"])
                    continue
                nonce = int(nonce_result["result"], 16)

                async with http.post(POLYGON_RPC, json=gas_payload) as r:
                    gas_result = await r.json()
                if "error" in gas_result:
                    log.error("redeemer: gasPrice RPC error: %s", gas_result["error"])
                    continue
                gas_price = int(gas_result["result"], 16)

                tx_hashes = []
                for bucket in buckets_to_redeem:
                    condition_id = bytes.fromhex(bucket.condition_id.replace("0x", ""))
                    # ABI encode: conditionId (bytes32), offset to indexSets (64),
                    # length of indexSets, then each uint256
                    calldata = (
                        SELECTOR
                        + condition_id.rjust(32, b"\x00")
                        + (64).to_bytes(32, "big")  # offset to dynamic array
                        + len(INDEX_SETS).to_bytes(32, "big")
                        + b"".join(i.to_bytes(32, "big") for i in INDEX_SETS)
                    )

                    tx = {
                        "to": NEG_RISK_ADAPTER,
                        "value": 0,
                        "gas": 200_000,
                        "gasPrice": gas_price,
                        "nonce": nonce,
                        "chainId": chain_id,
                        "data": calldata,
                    }
                    signed = account.sign_transaction(tx)
                    send_payload = {
                        "jsonrpc": "2.0", "id": 3, "method": "eth_sendRawTransaction",
                        "params": ["0x" + signed.raw_transaction.hex()],
                    }
                    async with http.post(POLYGON_RPC, json=send_payload) as r:
                        send_result = await r.json()

                    if "error" in send_result:
                        log.error(
                            "redeemer: redeem tx failed for bucket %d: %s",
                            bucket.id, send_result["error"],
                        )
                        continue

                    tx_hash = send_result.get("result")
                    tx_hashes.append(tx_hash)
                    nonce += 1
                    log.info("redeemer: redeem tx sent for bucket %d: %s", bucket.id, tx_hash)

                if not tx_hashes:
                    continue

                # Mark event as redeemed and zero out positions
                async with get_session() as sess:
                    from sqlalchemy import update as sql_update
                    from backend.storage.models import Event as EventModel
                    await sess.execute(
                        sql_update(EventModel)
                        .where(EventModel.id == event.id)
                        .values(redeemed_at=datetime.now(timezone.utc))
                    )

                    # Zero out positions and realize PnL
                    for bucket in event.buckets:
                        pos = await get_position(sess, bucket.id)
                        if pos and pos.net_qty > 0:
                            is_winner = (
                                event.winning_bucket_idx is not None
                                and bucket.bucket_idx == event.winning_bucket_idx
                            )
                            # Winner redeems at $1.00, loser at $0.00
                            redeem_price = 1.0 if is_winner else 0.0
                            await upsert_position(
                                sess,
                                bucket_id=bucket.id,
                                side="yes",
                                fill_qty=-pos.net_qty,
                                fill_price=redeem_price,
                                last_mkt_price=redeem_price,
                            )

                    await sess.commit()

                redeemed_count += 1
                from backend.storage.repos import append_audit
                async with get_session() as sess:
                    await append_audit(
                        sess,
                        actor="auto_redeemer",
                        action="positions_redeemed",
                        payload={
                            "event_id": event.id,
                            "date_et": event.date_et,
                            "tx_hashes": tx_hashes,
                            "winning_bucket_idx": event.winning_bucket_idx,
                        },
                    )

            except Exception as e:
                log.error("redeemer: redeem event %d failed: %s", event.id, e, exc_info=True)

    return redeemed_count


async def run_auto_redeem() -> None:
    """Combined check + redeem pass. Called by scheduler."""
    resolved = await check_resolved_markets()
    if resolved:
        log.info("redeemer: found %d newly resolved events", resolved)

    redeemed = await redeem_positions()
    if redeemed:
        log.info("redeemer: redeemed %d events", redeemed)
