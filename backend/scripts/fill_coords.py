"""
Script to populate latitude and longitude for all registered cities
using the aviationweather.gov Station Info API based on their METAR station.
"""

import asyncio
import json
import logging

import httpx
from sqlalchemy import select

from backend.config import Config
from backend.storage.db import get_session, init_db
from backend.storage.models import City
from backend.storage.repos import get_all_cities, upsert_city

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STATION_API = "https://aviationweather.gov/api/data/stationinfo"

async def enrich_city_coords(client: httpx.AsyncClient, city: City) -> dict:
    data = {"city_slug": city.city_slug}
    if not city.metar_station:
        log.warning(f"No METAR station for {city.city_slug}")
        return data

    try:
        resp = await client.get(
            STATION_API,
            params={"ids": city.metar_station, "format": "json"}
        )
        resp.raise_for_status()
        payload = resp.json()
        
        if payload and isinstance(payload, list) and len(payload) > 0:
            station_info = payload[0]
            lat = station_info.get("lat")
            lon = station_info.get("lon")
            if lat is not None and lon is not None:
                data["lat"] = float(lat)
                data["lon"] = float(lon)
                log.info(f"{city.city_slug} mapped to {lat}, {lon}")
            else:
                log.error(f"No lat/lon found in payload for {city.city_slug}")
        else:
            log.error(f"Invalid or empty payload for {city.city_slug}")
    except Exception as e:
        log.exception(f"Failed to fetch coords for {city.city_slug}: {e}")
        
    return data

async def main():
    await init_db()
    
    async with get_session() as session:
        cities = await get_all_cities(session)
        log.info(f"Found {len(cities)} cities in database.")
    
    updates = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for c in cities:
            update = await enrich_city_coords(client, c)
            if "lat" in update:
                updates.append(update)
            await asyncio.sleep(0.2) # Rate limit protection

    async with get_session() as session:
        for u in updates:
            await upsert_city(session, u)
        await session.commit()
        
        # Now read all cities and export to JSON for config.py seeding
        all_cities = await get_all_cities(session)
        export = []
        for c in all_cities:
            export.append({
                "city_slug": c.city_slug,
                "display_name": c.display_name,
                "nws_office": c.nws_office,
                "nws_grid_x": c.nws_grid_x,
                "nws_grid_y": c.nws_grid_y,
                "lat": c.lat,
                "lon": c.lon,
                "metar_station": c.metar_station,
                "wu_state": c.wu_state,
                "wu_city": c.wu_city,
                "enabled": int(c.enabled),
                "is_us": int(c.is_us),
                "unit": c.unit,
            })
            
    with open("/tmp/weather_cities.json", "w") as f:
        json.dump(export, f, indent=4)
        
    log.info(f"Updated {len(updates)} cities. config.py export saved to /tmp/weather_cities.json")

if __name__ == "__main__":
    asyncio.run(main())
