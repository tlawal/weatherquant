import asyncio
import os
import logging
from lib2to3.pgen2 import token
from datetime import datetime, timezone
import aiohttp
from eth_abi import encode
from eth_account.messages import encode_typed_data
import sys

# Ensure backend imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import Config
from backend.execution.chain_utils import get_wallet_address, _BALANCE_OF_SEL

log = logging.getLogger(__name__)

NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
REDEEM_SELECTOR = bytes.fromhex("dbeccb23")
GET_DETERMINED_SELECTOR = bytes.fromhex("ccb005ae")

async def erc1155_balance(http: aiohttp.ClientSession, contract: str, owner: str, token_id: str) -> int:
    padded_owner = bytes.fromhex(owner.replace("0x", "").zfill(64))
    token_int = int(token_id)
    padded_token = token_int.to_bytes(32, "big")
    calldata = "0x" + (_BALANCE_OF_SEL + padded_owner + padded_token).hex()
    resp = await http.post(Config.POLYGON_RPC_URL, json={
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [{"to": contract, "data": calldata}, "latest"],
    })
    res_json = await resp.json()
    if "result" in res_json:
        return int(res_json["result"], 16)
    return 0

def build_safe_signature(account, proxy_wallet: str, to: str, value: int, data: bytes, safe_nonce: int, chain_id: int) -> bytes:
    """Build Gnosis Safe EIP-712 signature for proxy execTransaction."""
    domain = {"verifyingContract": proxy_wallet, "chainId": chain_id}
    types = {
        "EIP712Domain": [{"name": "verifyingContract", "type": "address"}, {"name": "chainId", "type": "uint256"}],
        "SafeTx": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "nonce", "type": "uint256"}
        ]
    }
    
    # Standard Call operation=0, 0 gas options (Safe delegates execution costs to EOA sender)
    message = {
        "to": to,
        "value": value,
        "data": "0x" + data.hex(),
        "operation": 0,
        "safeTxGas": 0,
        "baseGas": 0,
        "gasPrice": 0,
        "gasToken": "0x0000000000000000000000000000000000000000",
        "refundReceiver": "0x0000000000000000000000000000000000000000",
        "nonce": safe_nonce
    }
    
    eip712_json = {"types": types, "primaryType": "SafeTx", "domain": domain, "message": message}
    signable = encode_typed_data(full_message=eip712_json)
    signed = account.sign_message(signable)
    
    # Gnosis Safe uses r, s, v byte packing
    r = signed.r.to_bytes(32, byteorder='big')
    s = signed.s.to_bytes(32, byteorder='big')
    return r + s + bytes([signed.v])

async def sweep_market(condition_id_hex: str, yes_token_id: str, no_token_id: str):
    """Diagnose, verify, and sweep the condition on NegRiskAdapter via Proxy."""
    if not Config.POLYMARKET_PRIVATE_KEY:
        print("Set POLYMARKET_PRIVATE_KEY in .env!")
        return

    from eth_account import Account
    account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
    sender_eoa = account.address
    proxy_wallet = get_wallet_address()
    chain_id = Config.CHAIN_ID
    
    print(f"--- Sweeping NegRisk ({condition_id_hex}) ---")
    print(f"EOA Sender:   {sender_eoa}")
    print(f"Proxy Wallet: {proxy_wallet}")

    async def _rpc(method, params):
        async with aiohttp.ClientSession() as s:
            r = await s.post(Config.POLYGON_RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
            return (await r.json()).get("result")

    condition_id_bytes = bytes.fromhex(condition_id_hex.replace("0x", ""))

    # 1. Check getDetermined
    get_det_data = "0x" + (GET_DETERMINED_SELECTOR + condition_id_bytes).hex()
    res_det = await _rpc("eth_call", [{"to": NEG_RISK_ADAPTER, "data": get_det_data}, "latest"])
    is_determined = bool(int(res_det, 16)) if res_det and res_det != "0x" else False
    print(f"getDetermined: {is_determined}")

    if not is_determined:
        print("ERROR: Market is not determined on the NegRiskAdapter yet.")
        print("Wait for the condition to be fully reported/resolved on the Adapter.")
        print("Retrying now will cleanly fail and burn gas. Exiting.")
        return

    # 2. Check Balances on Proxy Wallet
    async with aiohttp.ClientSession() as s:
        yes_bal = await erc1155_balance(s, NEG_RISK_ADAPTER, proxy_wallet, yes_token_id) if yes_token_id else 0
        no_bal = await erc1155_balance(s, NEG_RISK_ADAPTER, proxy_wallet, no_token_id) if no_token_id else 0

    print(f"Balances on Proxy -> YES: {yes_bal / 1e6} | NO: {no_bal / 1e6}")

    if yes_bal == 0 and no_bal == 0:
        print("Nothing to sweep. Balances are 0.")
        return

    # 3. Construct Proxy execTransaction
    # `redeemPositions(bytes32 _conditionId, uint256[] _amounts)` -> requires dynamically sized indexsets?
    # No, Polymarket NegRiskAdapter redeemPositions uses offset = 64 (0x40), length = 2, yes_bal, no_bal.
    redeem_data = (
        REDEEM_SELECTOR
        + condition_id_bytes.rjust(32, b"\x00")
        + (64).to_bytes(32, "big")  
        + (2).to_bytes(32, "big")
        + yes_bal.to_bytes(32, "big")
        + no_bal.to_bytes(32, "big")
    )

    safe_nonce_hex = await _rpc("eth_call", [{"to": proxy_wallet, "data": "0xaffed0e0"}, "latest"])
    safe_nonce = int(safe_nonce_hex, 16) if safe_nonce_hex and safe_nonce_hex != "0x" else 0
    print(f"Safe Nonce: {safe_nonce}")

    sig = build_safe_signature(account, proxy_wallet, NEG_RISK_ADAPTER, 0, redeem_data, safe_nonce, chain_id)
    
    # ABI encode execTransaction
    EXEC_TX_SELECTOR = bytes.fromhex("6a761202")
    encoded_args = encode(
        ['address', 'uint256', 'bytes', 'uint8', 'uint256', 'uint256', 'uint256', 'address', 'address', 'bytes'],
        [NEG_RISK_ADAPTER, 0, redeem_data, 0, 0, 0, 0, "0x0000000000000000000000000000000000000000", "0x0000000000000000000000000000000000000000", sig]
    )
    proxy_calldata = EXEC_TX_SELECTOR + encoded_args

    # 4. Fire Transaction
    gas_price_hex = await _rpc("eth_gasPrice", [])
    eoa_nonce_hex = await _rpc("eth_getTransactionCount", [sender_eoa, "latest"])
    
    tx = {
        "to": proxy_wallet,
        "value": 0,
        "gas": 400_000,
        "gasPrice": int(gas_price_hex, 16),
        "nonce": int(eoa_nonce_hex, 16),
        "chainId": chain_id,
        "data": proxy_calldata,
    }
    
    signed_tx = account.sign_transaction(tx)
    tx_hash = await _rpc("eth_sendRawTransaction", ["0x" + signed_tx.raw_transaction.hex()])
    print(f"Transaction broadcasting... Hash: {tx_hash}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", default="0xf407680f8a8aaeae83098a6e8f55e180592063e521f3597ccb36f22041de0511", help="Condition ID to redeem")
    parser.add_argument("--yes", default="68464819232927294590340903739217335680789125025992638975000171096382629434455", help="Yes Token ID")
    parser.add_argument("--no", default="48881063161752679858635940589274673280694928305036466602881671880256012823733", help="No Token ID")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sweep_market(args.condition_id, args.yes, args.no))
