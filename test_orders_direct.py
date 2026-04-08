import os
import asyncio
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OpenOrderParams

host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
key = os.getenv("POLYMARKET_PRIVATE_KEY")
chain_id = 137

clob = ClobClient(host, key=key, chain_id=chain_id)
clob.set_api_creds(clob.create_or_derive_api_creds())

def run():
    res = clob.get_orders(OpenOrderParams(market="0x8f3cb84572c55b7cb2e35478f73076d0401acd3790ef2060996ce8686b41a91a"))
    print("open_orders length:", len(res))
    if len(res) > 0:
        print("first order keys:", res[0].keys())
        import pprint
        pprint.pprint(res[0])

run()
