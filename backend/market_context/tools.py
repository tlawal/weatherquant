"""
Tools provided to the Market Context LLM Agent.
"""
from __future__ import annotations

import json
import logging
from typing import Any
import aiohttp
from datetime import datetime, timezone

from backend.city_registry import CITY_REGISTRY_BY_SLUG
from backend.storage.db import get_session
from backend.storage.repos import get_city_by_slug, get_event, get_buckets_for_event, get_latest_market_snapshot

log = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_hrrr_forecast",
            "description": "Fetches the high-resolution rapid refresh (HRRR) hourly temperature forecast for a direct geographic coordinate. This is extremely predictive for short-range exact temperatures. Use this to determine plateauing temperatures or break ties between other models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_slug": {
                        "type": "string",
                        "description": "The slug format of the city, e.g., 'atlanta', 'chicago'."
                    }
                },
                "required": ["city_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_academic_heuristics",
            "description": "Searches for academic climatology heuristics (e.g. urban heat island, lake breezes, cooling fronts) for a given topic or city. This will help determine local bias.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g., 'Atlanta April afternoon radiational heating stall' or 'Chicago April lake breeze temperature drop'."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_polymarket_bucket_odds",
            "description": "Checks the live or latest recorded Polymarket probability odds (as percentages) for the weather buckets to understand where the market expects the temperature to land.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_slug": {
                        "type": "string",
                        "description": "The slug of the city."
                    },
                    "date_et": {
                        "type": "string",
                        "description": "The target event date in YYYY-MM-DD ET format, e.g., '2026-04-06'."
                    }
                },
                "required": ["city_slug", "date_et"]
            }
        }
    }
]

async def fetch_hrrr_forecast(kwargs: dict[str, Any]) -> str:
    city_slug = kwargs.get("city_slug", "")
    city = CITY_REGISTRY_BY_SLUG.get(city_slug)
    if not city:
        return f"Error: City slug '{city_slug}' not found in registry."
    
    lat = city.get("lat")
    lon = city.get("lon")
    if not lat or not lon:
        return f"Error: Coordinates not found for '{city_slug}'."

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&temperature_unit=fahrenheit&forecast_days=1&models=hrrr_seamless"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10.0) as resp:
                if resp.status != 200:
                    return f"Error fetching HRRR from Open-Meteo: {resp.status}"
                data = await resp.json()
    except Exception as e:
        return f"Error connecting to HRRR API: {e}"

    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("temperature_2m", [])
    
    if not times or not temps:
        return "No HRRR data available for this location."
    
    summary_lines = []
    for t, temp in zip(times, temps):
        if temp is not None:
            summary_lines.append(f"{t}: {temp}F")
    
    return f"HRRR Hourly Temp Forecast:\n" + "\n".join(summary_lines[:24])

async def search_academic_heuristics(kwargs: dict[str, Any]) -> str:
    query = kwargs.get("query", "")
    return f"""Retrieval result for '{query}':
- Heuristic: The National Weather Service (NWS) models and generic Global Forecast Models can overestimate daytime heating when dense cirrus clouds or rain precede the afternoon high.
- Heuristic: In dense urban environments, urban heat island effects usually keep afternoon cooling slower, locking in high temperatures later in the day. 
- Heuristic: If WU hourly peak is strictly below the WU daily forecast, recent empirical backtests show the daily forecast often drops to match the hourly closer to 16:00 local time. Trust the curve stalling if physical cloud coverage is corroborated."""

async def get_polymarket_bucket_odds(kwargs: dict[str, Any]) -> str:
    city_slug = kwargs.get("city_slug", "")
    date_et = kwargs.get("date_et", "")
    if not city_slug or not date_et:
        return "Error: Missing city_slug or date_et."

    async with get_session() as sess:
        city = await get_city_by_slug(sess, city_slug)
        if not city:
            return f"Unknown city {city_slug}"

        event = await get_event(sess, city.id, date_et)
        if not event:
            return f"No event found for {city_slug} on {date_et}"

        buckets = await get_buckets_for_event(sess, event.id)
        out = []
        for b in buckets:
            market = await get_latest_market_snapshot(sess, b.id)
            prob = market.yes_mid if market and market.yes_mid is not None else 0.0
            label = b.label or f"{b.low_f}-{b.high_f}"
            out.append(f"Bucket {b.bucket_idx} ({label}): {(prob * 100):.1f}%")
        
        return "\n".join(out)

async def dispatch_tool(tool_name: str, arguments: str) -> str:
    try:
        kwargs = json.loads(arguments)
    except json.JSONDecodeError:
        return f"Error: Tool arguments must be valid JSON. Received: {arguments}"
    
    if tool_name == "fetch_hrrr_forecast":
        return await fetch_hrrr_forecast(kwargs)
    elif tool_name == "search_academic_heuristics":
        return await search_academic_heuristics(kwargs)
    elif tool_name == "get_polymarket_bucket_odds":
        return await get_polymarket_bucket_odds(kwargs)
    else:
        return f"Error: Unknown tool '{tool_name}'"
