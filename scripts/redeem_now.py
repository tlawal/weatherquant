"""
One-shot NegRisk redemption script for Polymarket positions.

Usage:
    python -m scripts.redeem_now --condition-id 0xf407680f... [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
REDEEM_SELECTOR = bytes.fromhex("dbeccb23")       # redeemPositions(bytes32,uint256[])
BALANCE_OF_SEL = bytes.fromhex("00fdd58e")         # ERC1155 balanceOf(address,uint256)
GET_DETERMINED_SEL = bytes.fromhex("7ae2e67b")     # getDetermined(bytes32)

POLYGON_RPC = os.environ.get("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
CHAIN_ID = int(os.environ.get("CHAIN_ID", "137"))


async def rpc_call(http: aiohttp.ClientSession, to: str, data: str) -> str | None:
    resp = await (await http.post(POLYGON_RPC, json={
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    })).json()
    return resp.get("result")


async def erc1155_balance(http: aiohttp.ClientSession, contract: str, owner: str, token_id: str) -> int:
    padded_owner = bytes.fromhex(owner.replace("0x", "").zfill(64))
    padded_token = int(token_id).to_bytes(32, "big")
    calldata = "0x" + (BALANCE_OF_SEL + padded_owner + padded_token).hex()
    result = await rpc_call(http, contract, calldata)
    return int(result, 16) if result else 0


async def get_determined(http: aiohttp.ClientSession, question_id: str) -> bool:
    qid_hex = question_id.replace("0x", "")
    calldata = "0x" + (GET_DETERMINED_SEL + bytes.fromhex(qid_hex.zfill(64))).hex()
    result = await rpc_call(http, NEG_RISK_ADAPTER, calldata)
    return int(result, 16) != 0 if result else False


async def main(condition_id: str, dry_run: bool):
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    funder = os.environ.get("FUNDER_ADDRESS", "")

    if not private_key:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in environment")
        sys.exit(1)

    from eth_account import Account
    account = Account.from_key(private_key)
    eoa = account.address
    token_holder = funder or eoa

    print(f"EOA (signer):    {eoa}")
    print(f"Token holder:    {token_holder}")
    print(f"Condition ID:    {condition_id}")
    print(f"NegRiskAdapter:  {NEG_RISK_ADAPTER}")
    print(f"Chain ID:        {CHAIN_ID}")
    print(f"RPC:             {POLYGON_RPC}")
    print()

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        # Step 1: Fetch questionId and token IDs from CLOB API
        print("── Step 1: Fetch market data from CLOB API ──")
        url = f"https://clob.polymarket.com/markets/{condition_id}"
        async with http.get(url) as resp:
            if resp.status != 200:
                print(f"ERROR: CLOB API returned {resp.status}")
                sys.exit(1)
            mkt = await resp.json()

        question_id = mkt.get("question_id")
        print(f"Question ID:     {question_id}")
        print(f"Question:        {mkt.get('question')}")
        print(f"Neg risk:        {mkt.get('neg_risk')}")
        print()

        # Step 2: Fetch token IDs from data API positions
        print("── Step 2: Fetch token IDs from data API ──")
        url = f"https://data-api.polymarket.com/positions?user={token_holder}"
        yes_token_id = None
        no_token_id = None
        async with http.get(url) as resp:
            if resp.status == 200:
                positions = await resp.json()
                for pos in positions:
                    if pos.get("conditionId") == condition_id:
                        yes_token_id = pos.get("asset")
                        no_token_id = pos.get("oppositeAsset")
                        print(f"Size:            {pos.get('size')}")
                        print(f"Redeemable:      {pos.get('redeemable')}")
                        print(f"Outcome:         {pos.get('outcome')}")
                        print(f"YES token ID:    {yes_token_id}")
                        print(f"NO token ID:     {no_token_id}")
                        break
                else:
                    print("WARNING: Position not found in data API")

        if not yes_token_id:
            # Try CLOB market tokens as fallback
            tokens = mkt.get("tokens", [])
            for t in tokens:
                if t.get("outcome") == "Yes":
                    yes_token_id = t.get("token_id")
                elif t.get("outcome") == "No":
                    no_token_id = t.get("token_id")
            if yes_token_id:
                print(f"YES token (CLOB): {yes_token_id}")
                print(f"NO token (CLOB):  {no_token_id}")
            else:
                print("ERROR: Could not determine token IDs")
                sys.exit(1)
        print()

        # Step 3: Check getDetermined
        print("── Step 3: Check on-chain determination ──")
        if question_id:
            determined = await get_determined(http, question_id)
            print(f"getDetermined:   {determined}")
        else:
            determined = False
            print("WARNING: No questionId, cannot check getDetermined")

        if not determined:
            print("WARNING: Market not determined on-chain. Redemption may fail or pay 0.")
        print()

        # Step 4: Query ERC1155 balances
        print("── Step 4: Query on-chain ERC1155 balances ──")
        yes_bal = await erc1155_balance(http, NEG_RISK_ADAPTER, token_holder, yes_token_id) if yes_token_id else 0
        no_bal = await erc1155_balance(http, NEG_RISK_ADAPTER, token_holder, no_token_id) if no_token_id else 0
        print(f"YES balance:     {yes_bal} raw ({yes_bal / 1_000_000:.6f} shares)")
        print(f"NO balance:      {no_bal} raw ({no_bal / 1_000_000:.6f} shares)")

        if yes_bal == 0 and no_bal == 0:
            print("\nERROR: Zero balance — nothing to redeem.")
            print("Possible causes: already redeemed, tokens on different address, or wrong contract.")
            sys.exit(1)
        print()

        # Step 5: Build redemption calldata
        print("── Step 5: Build redeemPositions calldata ──")
        cid_bytes = bytes.fromhex(condition_id.replace("0x", ""))
        calldata = (
            REDEEM_SELECTOR
            + cid_bytes.rjust(32, b"\x00")
            + (64).to_bytes(32, "big")      # offset to dynamic array
            + (2).to_bytes(32, "big")        # array length
            + yes_bal.to_bytes(32, "big")
            + no_bal.to_bytes(32, "big")
        )
        print(f"Calldata:        0x{calldata.hex()[:80]}...")
        print(f"Amounts:         [{yes_bal}, {no_bal}]")
        print()

        if dry_run:
            print("DRY RUN — not sending transaction.")
            return

        # Step 6: Send transaction
        print("── Step 6: Send transaction ──")
        nonce_resp = await (await http.post(POLYGON_RPC, json={
            "jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionCount",
            "params": [eoa, "latest"],
        })).json()
        gas_resp = await (await http.post(POLYGON_RPC, json={
            "jsonrpc": "2.0", "id": 2, "method": "eth_gasPrice", "params": [],
        })).json()

        nonce = int(nonce_resp["result"], 16)
        gas_price = int(gas_resp["result"], 16)

        tx = {
            "to": NEG_RISK_ADAPTER,
            "value": 0,
            "gas": 500_000,
            "gasPrice": gas_price,
            "nonce": nonce,
            "chainId": CHAIN_ID,
            "data": calldata,
        }
        print(f"Nonce:           {nonce}")
        print(f"Gas price:       {gas_price} ({gas_price / 1e9:.1f} gwei)")

        signed = account.sign_transaction(tx)
        send_resp = await (await http.post(POLYGON_RPC, json={
            "jsonrpc": "2.0", "id": 3, "method": "eth_sendRawTransaction",
            "params": ["0x" + signed.raw_transaction.hex()],
        })).json()

        if "error" in send_resp:
            print(f"ERROR: {send_resp['error']}")
            sys.exit(1)

        tx_hash = send_resp.get("result")
        print(f"TX sent:         {tx_hash}")
        print(f"Polygonscan:     https://polygonscan.com/tx/{tx_hash}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redeem NegRisk positions on Polymarket")
    parser.add_argument("--condition-id", required=True, help="Condition ID (hex, 0x-prefixed)")
    parser.add_argument("--dry-run", action="store_true", help="Show diagnostics without sending TX")
    args = parser.parse_args()
    asyncio.run(main(args.condition_id, args.dry_run))
