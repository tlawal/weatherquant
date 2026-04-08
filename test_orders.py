from dotenv import load_dotenv
load_dotenv('/Users/larry/code/weatherquant/.env')
import asyncio
from backend.config import Config
# Clear instance so it reloads from the new env vars
Config._instance = None
from backend.ingestion.polymarket_clob import get_clob
clob = get_clob()

async def run():
    if not clob:
        print("CLOB initialization failed")
        return
    print("balance:", await clob.get_balance())
    res = await clob.get_open_orders("0x8f3cb84572c55b7cb2e35478f73076d0401acd3790ef2060996ce8686b41a91a")
    print("open_orders length:", len(res))
    if len(res) > 0:
        print("first order keys:", res[0].keys())
        print("first order:", res[0])

asyncio.run(run())
