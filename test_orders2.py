import subprocess
import json
import os
import asyncio

def load_railway_env():
    out = subprocess.check_output(['railway', 'variables', '--json'])
    vars = json.loads(out)
    for k, v in vars.items():
        if isinstance(v, str):
            os.environ[k] = v
        else:
            os.environ[k] = str(v)

load_railway_env()
from backend.config import Config
Config._instance = None
from backend.ingestion.polymarket_clob import get_clob
clob = get_clob()

async def run():
    print("balance:", await clob.get_balance())
    res = await clob.get_open_orders("0x8f3cb84572c55b7cb2e35478f73076d0401acd3790ef2060996ce8686b41a91a")
    print("open_orders length:", len(res))
    if len(res) > 0:
        print("first order keys:", res[0].keys())
        import pprint
        pprint.pprint(res[0])

asyncio.run(run())
