"""
Polymarket CLOB client — order books and order placement.

Reuses py-clob-client pattern from btc-15m-quant.
Handles authentication, order books, limit order placement.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from backend.config import Config
from backend.tz_utils import city_local_date
from backend.storage.db import get_session
from backend.storage.repos import (
    get_all_cities,
    get_buckets_for_event,
    get_event,
    insert_market_snapshot,
    update_heartbeat,
)

log = logging.getLogger(__name__)

CLOB_HOST = Config.POLYMARKET_HOST
GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_RPC = Config.POLYGON_RPC_URL
_TIMEOUT = aiohttp.ClientTimeout(total=10)
_HEADERS = {"User-Agent": "WeatherQuant/1.0 (contact@weatherquant.local)"}


@dataclass
class BucketOrderBook:
    bucket_id: int
    yes_token_id: str
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    yes_mid: Optional[float] = None
    yes_bid_depth: float = 0.0
    yes_ask_depth: float = 0.0
    spread: Optional[float] = None
    fetched_ok: bool = False


class CLOBClient:
    """Thin async wrapper around Polymarket CLOB REST API."""

    def __init__(self) -> None:
        self._client = None
        self.can_trade: bool = False
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        """Initialize aiohttp session and CLOB auth."""
        self._session = aiohttp.ClientSession(
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        # Initialize py-clob-client for authenticated endpoints
        pk = Config.POLYMARKET_PRIVATE_KEY
        if pk and len(pk) > 20:
            try:
                from py_clob_client.client import ClobClient
                self._client = ClobClient(
                    host=Config.POLYMARKET_HOST,
                    key=pk,
                    chain_id=Config.CHAIN_ID,
                    funder=Config.FUNDER_ADDRESS or None,
                )
                creds = self._client.create_or_derive_api_creds()
                self._client.set_api_creds(creds)
                self.can_trade = True
                log.info("clob: credentials derived successfully")
            except Exception as e:
                log.error("clob: credential derivation failed: %s", e)
                self.can_trade = False
        else:
            log.warning("clob: POLYMARKET_PRIVATE_KEY not set — read-only mode")

        # Ensure USDC + conditional token allowances (non-fatal)
        if self.can_trade:
            await self.ensure_allowance()
            await self.ensure_conditional_allowance()

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("CLOBClient.start() not called")
        return self._session

    async def get_order_book(self, token_id: str) -> Optional[dict]:
        """Fetch raw order book for a single token."""
        url = f"{CLOB_HOST}/book?token_id={token_id}"
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except Exception as e:
            log.debug("clob: order book %s error: %s", token_id[:16], e)
            return None

    async def get_bucket_orderbook(self, bucket_id: int, yes_token_id: str) -> BucketOrderBook:
        """Fetch order book for a YES token and compute metrics."""
        ob = BucketOrderBook(bucket_id=bucket_id, yes_token_id=yes_token_id)
        data = await self.get_order_book(yes_token_id)
        if not data:
            return ob

        bids = data.get("bids") or []
        asks = data.get("asks") or []

        if bids:
            ob.yes_bid = max(float(b.get("price", 0)) for b in bids)
            ob.yes_bid_depth = sum(float(b.get("size", 0)) for b in bids if float(b.get("price", 0)) >= ob.yes_bid)
        if asks:
            ob.yes_ask = min(float(a.get("price", 0)) for a in asks if float(a.get("price", 0)) > 0)
            ob.yes_ask_depth = sum(float(a.get("size", 0)) for a in asks if float(a.get("price", 0)) <= (ob.yes_ask or 999))

        if ob.yes_bid and ob.yes_ask:
            ob.yes_mid = round((ob.yes_bid + ob.yes_ask) / 2, 4)
            ob.spread = round(ob.yes_ask - ob.yes_bid, 4)

        ob.fetched_ok = True
        return ob

    async def place_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" | "SELL"
        size: float,
        price: float,
    ) -> Optional[dict]:
        """Place a limit order. Returns order dict or None on failure."""
        if not self.can_trade:
            log.error("clob: cannot place order — no credentials")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs
            loop = asyncio.get_event_loop()
            args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
            )
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._client.create_and_post_order(args)),
                timeout=15.0,
            )
            log.info(
                "clob: order placed token=%s side=%s size=%.2f price=%.4f result=%s",
                token_id[:16], side, size, price, result,
            )
            return result
        except asyncio.TimeoutError:
            log.error("clob: place_limit_order timed out")
            return None
        except Exception as e:
            log.error("clob: place_limit_order failed: %s", e)
            return None

    async def place_market_order(
        self,
        token_id: str,
        side: str,  # "BUY" | "SELL"
        amount: float,  # $ for BUY, shares for SELL
    ) -> Optional[dict]:
        """Place a FOK market order. Returns order dict or None on failure."""
        if not self.can_trade:
            log.error("clob: cannot place order — no credentials")
            return None

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            loop = asyncio.get_event_loop()
            args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side,
            )

            def _create_and_post():
                order = self._client.create_market_order(args)
                return self._client.post_order(order, orderType=OrderType.FOK)

            result = await asyncio.wait_for(
                loop.run_in_executor(None, _create_and_post),
                timeout=15.0,
            )
            log.info(
                "clob: market order placed token=%s side=%s amount=%.2f result=%s",
                token_id[:16], side, amount, result,
            )
            return result
        except asyncio.TimeoutError:
            log.error("clob: place_market_order timed out")
            return None
        except Exception as e:
            log.error("clob: place_market_order failed: %s", e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by CLOB order ID."""
        if not self.can_trade:
            return False
        try:
            from py_clob_client.clob_types import OpenOrderParams
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._client.cancel(order_id)
                ),
                timeout=10.0,
            )
            log.info("clob: cancelled order %s", order_id)
            return True
        except Exception as e:
            log.error("clob: cancel_order %s failed: %s", order_id, e)
            return False

    async def get_balance(self) -> Optional[float]:
        """Fetch USDC collateral balance."""
        if not self.can_trade:
            return None
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._client.get_balance_allowance(params)),
                timeout=8.0,
            )
            if isinstance(result, dict):
                raw = result.get("balance") or result.get("available") or 0
                try:
                    x = float(raw)
                    return x / 1_000_000 if x >= 1e5 else x
                except (ValueError, TypeError):
                    return None
        except Exception as e:
            log.warning("clob: get_balance failed: %s", e)
            return None

    async def ensure_allowance(self) -> None:
        """Approve USDC spending for both Polymarket exchange contracts if allowance is 0."""
        if not self.can_trade:
            return

        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        from py_clob_client.config import get_contract_config

        loop = asyncio.get_event_loop()
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._client.get_balance_allowance(params)),
                timeout=8.0,
            )
        except Exception as e:
            log.warning("clob: allowance check failed: %s", e)
            return

        allowance = 0
        if isinstance(result, dict):
            try:
                allowance = int(result.get("allowance", 0))
            except (ValueError, TypeError):
                pass

        if allowance > 0:
            log.info("clob: USDC allowance already set (%d)", allowance)
            return

        log.warning("clob: USDC allowance is 0 — sending on-chain approve txs")

        try:
            chain_id = Config.CHAIN_ID
            cfg = get_contract_config(chain_id, neg_risk=False)
            cfg_neg = get_contract_config(chain_id, neg_risk=True)
            usdc_addr = cfg.collateral
            # Polymarket requires USDC approval for 3 contracts:
            # CTF Exchange, Neg Risk Exchange, and Neg Risk Adapter
            NEG_RISK_ADAPTER = {137: "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"}
            spenders = [cfg.exchange, cfg_neg.exchange]
            if chain_id in NEG_RISK_ADAPTER:
                spenders.append(NEG_RISK_ADAPTER[chain_id])

            from eth_account import Account
            account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
            sender = account.address
            max_uint256 = 2**256 - 1

            # Get initial nonce
            nonce_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionCount",
                "params": [sender, "latest"],
            }
            gas_payload = {
                "jsonrpc": "2.0", "id": 2, "method": "eth_gasPrice", "params": [],
            }
            async with self.session.post(POLYGON_RPC, json=nonce_payload) as r:
                nonce_result = await r.json()
            if "error" in nonce_result:
                log.error("clob: approve RPC nonce error: %s", nonce_result["error"])
                return
            nonce = int(nonce_result["result"], 16)

            async with self.session.post(POLYGON_RPC, json=gas_payload) as r:
                gas_result = await r.json()
            if "error" in gas_result:
                log.error("clob: approve RPC gasPrice error: %s", gas_result["error"])
                return
            gas_price = int(gas_result["result"], 16)

            # ERC20 approve(address,uint256) selector = 0x095ea7b3
            for spender in spenders:
                calldata = (
                    bytes.fromhex("095ea7b3")
                    + bytes.fromhex(spender[2:].lower().zfill(64))
                    + max_uint256.to_bytes(32, "big")
                )
                tx = {
                    "to": usdc_addr,
                    "value": 0,
                    "gas": 60_000,
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
                async with self.session.post(POLYGON_RPC, json=send_payload) as r:
                    send_result = await r.json()

                if "error" in send_result:
                    log.error("clob: approve tx failed for %s: %s", spender, send_result["error"])
                else:
                    tx_hash = send_result.get("result")
                    log.info("clob: approve tx sent for %s: %s", spender, tx_hash)
                nonce += 1

            # Refresh CLOB server's cached allowance
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._client.update_balance_allowance(params)),
                timeout=8.0,
            )
            log.info("clob: allowance cache refreshed")
        except Exception as e:
            log.warning("clob: on-chain approve failed (non-fatal, trading may still work): %s", e)

    async def ensure_conditional_allowance(self) -> None:
        """Approve conditional token transfers for SELL orders (ERC1155 setApprovalForAll)."""
        if not self.can_trade:
            return

        from py_clob_client.config import get_contract_config

        chain_id = Config.CHAIN_ID
        cfg = get_contract_config(chain_id, neg_risk=False)
        conditional_token = cfg.conditional_tokens

        # Check if already approved by trying a small call — for now just send approvals
        # (setApprovalForAll is idempotent, safe to re-send)
        spenders = [cfg.exchange]
        cfg_neg = get_contract_config(chain_id, neg_risk=True)
        spenders.append(cfg_neg.exchange)
        NEG_RISK_ADAPTER = {137: "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"}
        if chain_id in NEG_RISK_ADAPTER:
            spenders.append(NEG_RISK_ADAPTER[chain_id])

        try:
            from eth_account import Account
            account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
            sender = account.address

            nonce_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionCount",
                "params": [sender, "latest"],
            }
            gas_payload = {
                "jsonrpc": "2.0", "id": 2, "method": "eth_gasPrice", "params": [],
            }
            async with self.session.post(POLYGON_RPC, json=nonce_payload) as r:
                nonce_result = await r.json()
            if "error" in nonce_result:
                log.error("clob: conditional approve RPC nonce error: %s", nonce_result["error"])
                return
            nonce = int(nonce_result["result"], 16)

            async with self.session.post(POLYGON_RPC, json=gas_payload) as r:
                gas_result = await r.json()
            if "error" in gas_result:
                log.error("clob: conditional approve RPC gasPrice error: %s", gas_result["error"])
                return
            gas_price = int(gas_result["result"], 16)

            # ERC1155 setApprovalForAll(address,bool) selector = 0xa22cb465
            for spender in spenders:
                calldata = (
                    bytes.fromhex("a22cb465")
                    + bytes.fromhex(spender[2:].lower().zfill(64))
                    + (1).to_bytes(32, "big")  # true
                )
                tx = {
                    "to": conditional_token,
                    "value": 0,
                    "gas": 60_000,
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
                async with self.session.post(POLYGON_RPC, json=send_payload) as r:
                    send_result = await r.json()

                if "error" in send_result:
                    log.error("clob: conditional approve tx failed for %s: %s", spender, send_result["error"])
                else:
                    tx_hash = send_result.get("result")
                    log.info("clob: conditional approve tx sent for %s: %s", spender, tx_hash)
                nonce += 1

        except Exception as e:
            log.warning("clob: conditional token approve failed (non-fatal): %s", e)


async def fetch_clob_orderbooks(clob: CLOBClient) -> None:
    """Fetch order books for all watched buckets and persist snapshots."""
    from backend.tz_utils import city_local_now, city_local_tomorrow

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    # Collect all orderbook data first (HTTP calls, no DB)
    snapshots_to_insert = []
    for city in cities:
        # Match the 8 PM rollover logic from fetch_gamma_all —
        # after 8 PM local, the active market is *tomorrow's* event.
        now_local = city_local_now(city)
        if now_local.hour >= 20:
            active_date = city_local_tomorrow(city)
        else:
            active_date = city_local_date(city)

        async with get_session() as sess:
            event = await get_event(sess, city.id, active_date)
            if not event or event.status != "ok":
                continue
            buckets = await get_buckets_for_event(sess, event.id)

        for bucket in buckets:
            if not bucket.yes_token_id:
                continue
            ob = await clob.get_bucket_orderbook(bucket.id, bucket.yes_token_id)
            if not ob.fetched_ok:
                continue
            snapshots_to_insert.append({
                "bucket_id": bucket.id,
                "yes_bid": ob.yes_bid,
                "yes_ask": ob.yes_ask,
                "yes_mid": ob.yes_mid,
                "yes_bid_depth": ob.yes_bid_depth,
                "yes_ask_depth": ob.yes_ask_depth,
                "spread": ob.spread,
            })
            # Brief pause between token fetches
            await asyncio.sleep(0.3)

    # Batch-insert all snapshots in a single session/transaction
    async with get_session() as sess:
        for snap_data in snapshots_to_insert:
            await insert_market_snapshot(sess, **snap_data)
        await update_heartbeat(sess, "fetch_clob", success=True)


# Global singleton — set by main.py startup
_clob_client: Optional[CLOBClient] = None


def get_clob() -> Optional[CLOBClient]:
    return _clob_client


def set_clob(client: CLOBClient) -> None:
    global _clob_client
    _clob_client = client
