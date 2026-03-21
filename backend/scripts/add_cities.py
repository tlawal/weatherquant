import asyncio
from backend.storage.db import get_session, init_db
from backend.storage.repos import upsert_city
from backend.storage.models import City
import logging

logging.basicConfig(level=logging.INFO)

cities_to_add = [
    # US (Fahrenheit)
    ("nyc", "NYC", True, "F"),
    ("chicago", "Chicago", True, "F"),
    ("atlanta", "Atlanta", True, "F"),
    ("miami", "Miami", True, "F"),
    ("dallas", "Dallas", True, "F"),
    ("seattle", "Seattle", True, "F"),
    ("sf", "SF", True, "F"),
    ("la", "LA", True, "F"),
    ("austin", "Austin", True, "F"),
    ("denver", "Denver", True, "F"),
    ("houston", "Houston", True, "F"),
    
    # Europe (Celsius)
    ("london", "London", False, "C"),
    ("paris", "Paris", False, "C"),
    ("madrid", "Madrid", False, "C"),
    ("munich", "Munich", False, "C"),
    ("warsaw", "Warsaw", False, "C"),
    ("milan", "Milan", False, "C"),
    
    # Asia (Celsius)
    ("shanghai", "Shanghai", False, "C"),
    ("beijing", "Beijing", False, "C"),
    ("shenzhen", "Shenzhen", False, "C"),
    ("chongqing", "Chongqing", False, "C"),
    ("wuhan", "Wuhan", False, "C"),
    ("chengdu", "Chengdu", False, "C"),
    ("seoul", "Seoul", False, "C"),
    ("tokyo", "Tokyo", False, "C"),
    ("taipei", "Taipei", False, "C"),
    ("hong-kong", "Hong Kong", False, "C"),
    ("singapore", "Singapore", False, "C"),
    
    # Other (Celsius)
    ("toronto", "Toronto", False, "C"),
    ("sao-paulo", "Sao Paulo", False, "C"),
    ("buenos-aires", "Buenos Aires", False, "C"),
    ("wellington", "Wellington", False, "C"),
    ("tel-aviv", "Tel Aviv", False, "C"),
    ("lucknow", "Lucknow", False, "C"),
]

async def main():
    await init_db()
    async with get_session() as sess:
        for slug, name, is_us, unit in cities_to_add:
            try:
                await upsert_city(
                    sess,
                    {
                        "city_slug": slug,
                        "display_name": name,
                        "enabled": False,
                        "is_us": is_us,
                        "unit": unit,
                    }
                )
                print(f"Added {name}")
            except Exception as e:
                print(f"Failed to add {name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
