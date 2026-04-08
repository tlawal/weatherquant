import asyncio
from backend.ingestion.polymarket_clob import get_clob
import logging

logging.basicConfig(level=logging.INFO)

async def run():
    clob = get_clob()
    if not clob:
        print("CLOB NOT INITIALIZED")
        return
    print("balance:", await clob.get_balance())
    res = await clob.get_open_orders("0x8f3cb84572c55b7cb2e35478f73076d0401acd3790ef2060996ce8686b41a91a")
    print("open_orders length:", len(res))
    if len(res) > 0:
        print("first order keys:", res[0].keys())
        import pprint
        pprint.pprint(res[0])

asyncio.run(run())
