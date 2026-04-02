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
    append_audit,
    get_event_by_id,
    get_position,
    get_unredeemed_resolved_events,
    get_unresolved_events_with_gamma_id,
    get_unresolved_events_with_positions,
    upsert_position,
)

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_RPC = Config.POLYGON_RPC_URL
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
_TIMEOUT = aiohttp.ClientTimeout(total=15)

from backend.execution.chain_utils import erc1155_balance as _erc1155_balance


async def check_resolved_markets() -> int:
    """
    Check if any events with open positions have resolved on Polymarket.
    Returns count of newly resolved events.
    """
    async with get_session() as sess:
        events = await get_unresolved_events_with_gamma_id(sess)

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


async def redeem_single_event(event_id: int, actor: str = "auto_redeemer", force: bool = False) -> dict:
    """
    Redeem positions for a single resolved event.
    Returns {"ok": True, "tx_hashes": [...], "event_id": ...} on success.
    Raises ValueError for validation failures, RuntimeError for tx/config failures.
    Use force=True to retry an event already marked as redeemed.
    """
    async with get_session() as sess:
        event = await get_event_by_id(sess, event_id)

    if not event:
        raise ValueError(f"Event {event_id} not found")
    if event.resolved_at is None:
        raise ValueError(f"Event {event_id} not yet resolved")
    if event.redeemed_at is not None and not force:
        raise ValueError(f"Event {event_id} already redeemed")
    if not Config.POLYMARKET_PRIVATE_KEY:
        raise RuntimeError("No private key configured")

    from eth_account import Account
    account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    sender = account.address
    chain_id = Config.CHAIN_ID

    # NegRiskAdapter redeemPositions(bytes32 conditionId, uint256[] amounts)
    REDEEM_SELECTOR = bytes.fromhex("dbeccb23")

    log.info("redeemer: sender=%s, target=NegRiskAdapter, event=%d", sender, event_id)

    # Redeem all buckets with condition_id — on-chain call redeems whatever tokens
    # the wallet holds, regardless of DB position state
    buckets_to_redeem = [b for b in event.buckets if b.condition_id]

    if not buckets_to_redeem:
        async with get_session() as sess:
            from sqlalchemy import update as sql_update
            from backend.storage.models import Event as EventModel
            await sess.execute(
                sql_update(EventModel)
                .where(EventModel.id == event.id)
                .values(redeemed_at=datetime.now(timezone.utc))
            )
            await sess.commit()
        return {"ok": True, "tx_hashes": [], "event_id": event_id, "note": "no positions to redeem"}

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
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
            raise RuntimeError(f"nonce RPC error: {nonce_result['error']}")
        nonce = int(nonce_result["result"], 16)

        async with http.post(POLYGON_RPC, json=gas_payload) as r:
            gas_result = await r.json()
        if "error" in gas_result:
            raise RuntimeError(f"gasPrice RPC error: {gas_result['error']}")
        gas_price = int(gas_result["result"], 16)

        tx_hashes = []
        for bucket in buckets_to_redeem:
            condition_id = bytes.fromhex(bucket.condition_id.replace("0x", ""))

            # Query on-chain ERC1155 balances for YES and NO tokens
            yes_bal = 0
            no_bal = 0
            if bucket.yes_token_id:
                yes_bal = await _erc1155_balance(http, NEG_RISK_ADAPTER, sender, bucket.yes_token_id)
            if bucket.no_token_id:
                no_bal = await _erc1155_balance(http, NEG_RISK_ADAPTER, sender, bucket.no_token_id)

            if yes_bal == 0 and no_bal == 0:
                log.info("redeemer: bucket %d has 0 balance, skipping", bucket.id)
                continue

            # NegRiskAdapter redeemPositions(conditionId, amounts)
            amounts = [yes_bal, no_bal]
            calldata = (
                REDEEM_SELECTOR
                + condition_id.rjust(32, b"\x00")
                + (64).to_bytes(32, "big")  # offset to dynamic array
                + (2).to_bytes(32, "big")
                + yes_bal.to_bytes(32, "big")
                + no_bal.to_bytes(32, "big")
            )
            log.info("redeemer: bucket %d amounts=[%d, %d]", bucket.id, yes_bal, no_bal)

            tx = {
                "to": NEG_RISK_ADAPTER,
                "value": 0,
                "gas": 300_000,
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
        raise RuntimeError("All redemption transactions failed")

    # Mark event as redeemed and zero out positions
    async with get_session() as sess:
        from sqlalchemy import update as sql_update
        from backend.storage.models import Event as EventModel
        await sess.execute(
            sql_update(EventModel)
            .where(EventModel.id == event.id)
            .values(redeemed_at=datetime.now(timezone.utc))
        )

        for bucket in event.buckets:
            pos = await get_position(sess, bucket.id)
            if pos and pos.net_qty > 0:
                is_winner = (
                    event.winning_bucket_idx is not None
                    and bucket.bucket_idx == event.winning_bucket_idx
                )
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

    async with get_session() as sess:
        await append_audit(
            sess,
            actor=actor,
            action="positions_redeemed",
            payload={
                "event_id": event.id,
                "date_et": event.date_et,
                "tx_hashes": tx_hashes,
                "winning_bucket_idx": event.winning_bucket_idx,
            },
        )

    return {"ok": True, "tx_hashes": tx_hashes, "event_id": event_id}


async def redeem_positions() -> int:
    """
    Redeem winning positions for all resolved events.
    Calls redeem_single_event() for each unredeemed resolved event.
    Returns count of redeemed events.
    """
    async with get_session() as sess:
        events = await get_unredeemed_resolved_events(sess)

    if not events:
        return 0

    if not Config.POLYMARKET_PRIVATE_KEY:
        log.warning("redeemer: no private key configured, skipping redemption")
        return 0

    redeemed_count = 0
    for event in events:
        try:
            result = await redeem_single_event(event.id, actor="auto_redeemer")
            if result.get("tx_hashes"):
                redeemed_count += 1
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
