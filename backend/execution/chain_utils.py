"""Shared on-chain query utilities for Polygon / Polymarket."""
from __future__ import annotations

import aiohttp

from backend.config import Config

POLYGON_RPC = Config.POLYGON_RPC_URL
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
_BALANCE_OF_SEL = bytes.fromhex("00fdd58e")  # ERC1155 balanceOf(address,uint256)


async def erc1155_balance(
    http: aiohttp.ClientSession, contract: str, owner: str, token_id: str,
) -> int:
    """Query ERC1155 balanceOf on-chain. Returns raw uint256."""
    padded_owner = bytes.fromhex(owner.replace("0x", "").zfill(64))
    token_int = int(token_id)
    padded_token = token_int.to_bytes(32, "big")
    calldata = "0x" + (_BALANCE_OF_SEL + padded_owner + padded_token).hex()
    resp = await (await http.post(POLYGON_RPC, json={
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [{"to": contract, "data": calldata}, "latest"],
    })).json()
    if "result" in resp:
        return int(resp["result"], 16)
    return 0


def get_wallet_address() -> str:
    """Derive wallet address from the configured private key."""
    from eth_account import Account
    return Account.from_key(Config.POLYMARKET_PRIVATE_KEY).address
