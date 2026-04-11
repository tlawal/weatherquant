import asyncio
from backend.storage.db import get_session, init_db
from backend.storage.repos import upsert_city
from sqlalchemy import select

async def main():
    await init_db()
    async with get_session() as sess:
        missing = [
            ("sf", "KSFO", "san-francisco", "ca", True),
            ("la", "KLAX", "los-angeles", "ca", True),
            ("austin", "KAUS", "austin", "tx", True),
            ("denver", "KBKF", "denver", "co", True),
            ("houston", "KHOU", "houston", "tx", True),
            ("taipei", "RCTP", "taipei", "tw", True),
            ("hong-kong", "VHHH", "hong-kong", "hk", True),
        ]
        for slug, metar, wu_city, wu_state, enabled in missing:
            await upsert_city(
                sess,
                {
                    "city_slug": slug,
                    "metar_station": metar,
                    "wu_city": wu_city,
                    "wu_state": wu_state,
                    "enabled": enabled
                }
            )
            print(f"Updated {slug}")

if __name__ == "__main__":
    asyncio.run(main())
