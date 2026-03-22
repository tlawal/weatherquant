import asyncio
import httpx
from backend.storage.db import get_session, init_db
from backend.storage.repos import get_all_cities, upsert_city

async def main():
    await init_db()
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=False)
        
    for city in cities:
        if not city.is_us:
            continue
            
        print(f"Checking {city.city_slug} (METAR: {city.metar_station})...")
        if not city.metar_station:
            continue
            
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                # 1. Get Lat/Lon from METAR station
                url1 = f"https://aviationweather.gov/api/data/stationinfo?ids={city.metar_station}&format=json"
                resp1 = await client.get(url1, timeout=10)
                if resp1.status_code != 200 or not resp1.json():
                    print(f"  -> Failed to get station info")
                    continue
                
                info = resp1.json()[0]
                lat, lon = info["lat"], info["lon"]
                
                # 2. Get Gridpoints from NWS
                url2 = f"https://api.weather.gov/points/{lat},{lon}"
                headers = {"User-Agent": "WeatherQuant/1.0 (contact@weatherquant.local)"}
                resp2 = await client.get(url2, headers=headers, timeout=10)
                if resp2.status_code != 200:
                    print(f"  -> Failed to get NWS points ({resp2.status_code})")
                    continue
                    
                props = resp2.json().get("properties", {})
                office = props.get("gridId")
                grid_x = props.get("gridX")
                grid_y = props.get("gridY")
                
                print(f"  -> NWS: {office} {grid_x},{grid_y}")
                
                # 3. Update DB
                async with get_session() as sess:
                    await upsert_city(
                        sess,
                        {
                            "city_slug": city.city_slug,
                            "nws_office": office,
                            "nws_grid_x": grid_x,
                            "nws_grid_y": grid_y,
                        }
                    )
            except Exception as e:
                print(f"  -> Error: {e}")
                
        # Sleep to avoid NWS rate limits
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
