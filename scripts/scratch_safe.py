import asyncio
import os
from dotenv import load_dotenv
from eth_account import Account
import aiohttp
from eth_abi import encode

async def test_build():
    load_dotenv()
    pk = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not pk:
        print("no pk")
        return
    account = Account.from_key(pk)
    
    proxy_wallet = os.getenv("FUNDER_ADDRESS", "0x7AbA1F81034d418A4DED1613626cA7573FD85153")
    to = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
    data = bytes.fromhex("dbeccb23") # mock calldata
    
    from scripts.safe_utils import generate_safe_signature
    chain_id = int(os.getenv("CHAIN_ID", 137))
    
    # 1. get safe nonce via RPC
    async with aiohttp.ClientSession() as http:
        rpc = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        nonce_payload = {
            "jsonrpc": "2.0", "id": 1, "method": "eth_call",
            "params": [{"to": proxy_wallet, "data": "0xaffed0e0"}, "latest"] # nonce()
        }
        res = await http.post(rpc, json=nonce_payload)
        res_json = await res.json()
        safe_nonce = int(res_json["result"], 16)
        print("safe_nonce", safe_nonce)
        
        # 2. build signature
        sig = generate_safe_signature(account, proxy_wallet, to, 0, data, 0, 0, 0, 0, "0x0000000000000000000000000000000000000000", "0x0000000000000000000000000000000000000000", safe_nonce, chain_id)
        
        # 3. encode
        EXEC_TX_SELECTOR = bytes.fromhex("6a761202")
        encoded_args = encode(
            ['address', 'uint256', 'bytes', 'uint8', 'uint256', 'uint256', 'uint256', 'address', 'address', 'bytes'],
            [to, 0, data, 0, 0, 0, 0, "0x0000000000000000000000000000000000000000", "0x0000000000000000000000000000000000000000", sig]
        )
        proxy_calldata = "0x" + (EXEC_TX_SELECTOR + encoded_args).hex()
        
        # 4. Try eth_estimateGas to verify the payload is correct
        est = {
            "jsonrpc": "2.0", "id": 2, "method": "eth_estimateGas",
            "params": [{"from": account.address, "to": proxy_wallet, "data": proxy_calldata}]
        }
        res_est = await http.post(rpc, json=est)
        print("Estimate result:", await res_est.json())

if __name__ == "__main__":
    asyncio.run(test_build())
