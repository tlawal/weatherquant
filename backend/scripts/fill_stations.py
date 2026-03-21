import asyncio
import httpx
import re
from yarl import URL
from backend.storage.db import get_session, init_db
from backend.storage.repos import get_all_cities, upsert_city

async def main():
    await init_db()
    
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=False)
        
    for city in cities:
        print(f"Checking {city.city_slug}...")
        url = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city.city_slug}-on-march-21-2026"
        url2 = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city.city_slug}-on-march-22-2026"
        url3 = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city.city_slug}-on-march-23-2026"
        
        event_data = None
        for endpoint in [url, url2, url3]:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(endpoint, timeout=10)
                    if resp.status_code == 200:
                        event_data = resp.json()
                        break
            except Exception:
                pass
                
        if not event_data:
            print(f"  -> Could not find active event for {city.city_slug}")
            continue
            
        markets = event_data.get("markets", [])
        if not markets:
            continue
            
        description = markets[0].get("description", "")
        # Look for https://www.wunderground.com/history/daily/...
        m = re.search(r"wunderground\.com/history/daily/([a-z0-9A-Z_/-]+)", description)
        if not m:
            print(f"  -> No WU URL found in description for {city.city_slug}")
            continue
            
        path_parts = m.group(1).rstrip("/.").split("/")
        # e.g., ['us', 'ga', 'atlanta', 'KATL'] or ['gb', 'london', 'EGLL']
        station = path_parts[-1].upper()
        wu_city = path_parts[-2] if len(path_parts) > 1 else ""
        wu_state = path_parts[-3] if len(path_parts) > 2 else ""
        
        # Some are just short like: gb/london/EGLL -> state='gb', city='london'
        # Just use what we got and enable it
        print(f"  -> Extracted station={station}, wu_city={wu_city}, wu_state={wu_state}")
        
        async with get_session() as sess:
            await upsert_city(
                sess,
                {
                    "city_slug": city.city_slug,
                    "metar_station": station,
                    "wu_city": wu_city,
                    "wu_state": wu_state,
                    "enabled": True
                }
            )

if __name__ == "__main__":
    asyncio.run(main())
