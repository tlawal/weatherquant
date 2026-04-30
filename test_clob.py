import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.polymarket_clob import get_clob
from py_clob_client_v2.clob_types import OpenOrderParams

async def main():
    c = get_clob()
    await c._ensure_initialized()
    orders = c._client.get_orders(OpenOrderParams(market="0x8f3cb84572c55b7cb2e35478f73076d0401acd3790ef2060996ce8686b41a91a"))
    print(orders)

asyncio.run(main())
