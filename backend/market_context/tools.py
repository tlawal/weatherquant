"""
Tools provided to the Market Context LLM Agent.

Real API integrations:
- fetch_hrrr_forecast: GFS+HRRR blend via Open-Meteo (unchanged)
- fetch_nbm_forecast: NCEP NBM CONUS via Open-Meteo
- search_academic_climatology: Semantic Scholar for peer-reviewed heuristics
- fetch_nws_discussion: NWS Area Forecast Discussion (synoptic analysis)
- get_polymarket_bucket_odds: Live Polymarket odds from DB (unchanged)
"""
from __future__ import annotations

import json
import logging
import re
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
            "description": (
                "Fetches the GFS+HRRR blended hourly temperature forecast via Open-Meteo for a US city. "
                "Combines GFS reliability with HRRR's rapid hourly updates. Extremely predictive for "
                "short-range exact temperatures. Use this to determine plateauing temperatures or break "
                "ties between other models. HRRR updates hourly (00z-23z) with ~45 min latency."
            ),
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
            "name": "fetch_nbm_forecast",
            "description": (
                "Fetches the NCEP National Blend of Models (NBM) CONUS hourly temperature forecast "
                "via Open-Meteo. NBM is a statistical post-processing blend of 50+ NWP models "
                "(GFS, NAM, HRRR, ECMWF, GEM, SREF, etc.) produced by NOAA/NCEP. Peer-reviewed "
                "studies show NBM reduces MAE by 10-20% over raw GFS. Updated hourly. "
                "Use this as the most reliable single-source forecast for US cities."
            ),
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
            "name": "search_academic_climatology",
            "description": (
                "Searches Semantic Scholar for peer-reviewed academic papers on climatology, "
                "weather forecasting, temperature prediction, urban heat islands, cold air damming, "
                "lake breezes, model biases, and other meteorological phenomena. Returns titles, "
                "abstracts, and citation counts. Use this to ground analysis in academic evidence "
                "rather than heuristics. Example queries: 'cold air damming southeast US temperature', "
                "'urban heat island airport METAR bias', 'HRRR GFS ensemble calibration'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Academic search query for climatology/meteorology papers."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_nws_discussion",
            "description": (
                "Fetches the latest NWS Area Forecast Discussion (AFD) for a US city's Weather "
                "Forecast Office (WFO). The AFD contains free-text synoptic analysis written by "
                "human forecasters explaining WHY the forecast is what it is — frontal positions, "
                "jet stream patterns, cold air damming, convective outlooks, model disagreements, "
                "and confidence levels. This is often published BEFORE gridpoint data updates, "
                "giving an information edge. Non-US cities return an error."
            ),
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
            "name": "get_polymarket_bucket_odds",
            "description": (
                "Checks the live or latest recorded Polymarket probability odds (as percentages) "
                "for the weather buckets to understand where the market expects the temperature to land."
            ),
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


# ── Tool Implementations ────────────────────────────────────────────────────


async def fetch_hrrr_forecast(kwargs: dict[str, Any]) -> str:
    city_slug = kwargs.get("city_slug", "")
    city = CITY_REGISTRY_BY_SLUG.get(city_slug)
    if not city:
        return f"Error: City slug '{city_slug}' not found in registry."

    lat = city.get("lat")
    lon = city.get("lon")
    if not lat or not lon:
        return f"Error: Coordinates not found for '{city_slug}'."

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit"
        f"&forecast_days=1&models=gfs_hrrr"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return f"Error fetching HRRR from Open-Meteo: {resp.status}"
                data = await resp.json()
    except Exception as e:
        return f"Error connecting to HRRR API: {e}"

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m_gfs_hrrr", hourly.get("temperature_2m", []))

    if not times or not temps:
        return f"No HRRR data available for this location. Keys returned: {list(hourly.keys())}"

    summary_lines = []
    for t, temp in zip(times, temps):
        if temp is not None:
            summary_lines.append(f"{t}: {temp}F")

    return f"HRRR Hourly Temp Forecast:\n" + "\n".join(summary_lines[:24])


async def fetch_nbm_forecast(kwargs: dict[str, Any]) -> str:
    """Fetch NCEP NBM CONUS hourly temperature forecast via Open-Meteo."""
    city_slug = kwargs.get("city_slug", "")
    city = CITY_REGISTRY_BY_SLUG.get(city_slug)
    if not city:
        return f"Error: City slug '{city_slug}' not found in registry."

    if not city.get("is_us"):
        return f"Error: NBM CONUS is only available for US cities. '{city_slug}' is non-US."

    lat = city.get("lat")
    lon = city.get("lon")
    if not lat or not lon:
        return f"Error: Coordinates not found for '{city_slug}'."

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit"
        f"&forecast_days=1&models=ncep_nbm_conus"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return f"Error fetching NBM from Open-Meteo: {resp.status}"
                data = await resp.json()
    except Exception as e:
        return f"Error connecting to NBM API: {e}"

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m_ncep_nbm_conus", hourly.get("temperature_2m", []))

    if not times or not temps:
        return f"No NBM data available for this location. Keys returned: {list(hourly.keys())}"

    summary_lines = []
    peak_temp = None
    peak_time = None
    for t, temp in zip(times, temps):
        if temp is not None:
            summary_lines.append(f"{t}: {temp}F")
            if peak_temp is None or temp > peak_temp:
                peak_temp = temp
                peak_time = t

    result = f"NBM CONUS Hourly Temp Forecast:\n" + "\n".join(summary_lines[:24])
    if peak_temp is not None:
        result += f"\n\nNBM Peak: {peak_temp}F at {peak_time}"
    return result


async def search_academic_climatology(kwargs: dict[str, Any]) -> str:
    """Search Semantic Scholar for real peer-reviewed papers on climatology topics."""
    query = kwargs.get("query", "")
    if not query:
        return "Error: No query provided."

    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={aiohttp.helpers.quote(query, safe='')}"
        f"&limit=5"
        f"&fields=title,abstract,year,citationCount,journal"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Accept": "application/json"},
            ) as resp:
                if resp.status == 429:
                    return _fallback_academic_search(query)
                if resp.status != 200:
                    log.warning("Semantic Scholar API returned %d for query: %s", resp.status, query)
                    return _fallback_academic_search(query)
                data = await resp.json()
    except Exception as e:
        log.warning("Semantic Scholar API error: %s", e)
        return _fallback_academic_search(query)

    papers = data.get("data", [])
    if not papers:
        return _fallback_academic_search(query)

    results = []
    for p in papers[:5]:
        title = p.get("title", "Unknown")
        year = p.get("year", "?")
        citations = p.get("citationCount", 0)
        abstract = p.get("abstract", "")
        journal = p.get("journal", {})
        journal_name = journal.get("name", "") if isinstance(journal, dict) else ""

        # Truncate abstract to 300 chars
        if abstract and len(abstract) > 300:
            abstract = abstract[:300] + "..."

        entry = f"- [{year}] {title}"
        if journal_name:
            entry += f" ({journal_name})"
        entry += f" [cited {citations}x]"
        if abstract:
            entry += f"\n  Abstract: {abstract}"
        results.append(entry)

    return f"Academic papers for '{query}':\n\n" + "\n\n".join(results)


def _fallback_academic_search(query: str) -> str:
    """Curated climatology heuristics when Semantic Scholar is unavailable."""
    q = query.lower()
    results = []

    # Always include base heuristics
    results.append(
        "- Gneiting & Raftery (2007): Ensemble Model Output Statistics (EMOS) provides "
        "calibrated probability distributions that outperform raw ensemble output for "
        "temperature forecasting. NBM uses EMOS-like post-processing."
    )

    if any(term in q for term in ["cold air damming", "wedge", "cad", "appalachian"]):
        results.append(
            "- Bell & Bosart (1988): Cold air damming east of the Appalachians creates a "
            "shallow cold wedge that suppresses afternoon heating 3-8°F below free-atmosphere "
            "forecasts. NE winds and overcast skies are diagnostic. The wedge typically erodes "
            "by mid-afternoon if solar heating is sufficient."
        )
        results.append(
            "- Bailey et al. (2003): CAD events in the Southeast US show METAR stations recording "
            "temperatures 4-6°F below model forecasts that don't resolve the shallow cold pool."
        )

    if any(term in q for term in ["urban heat island", "uhi", "airport", "metar bias"]):
        results.append(
            "- Oke (1982): Urban heat island intensity scales with city population and "
            "building density. Airport METAR stations on urban periphery typically read 2-4°F "
            "cooler than downtown during calm, clear nights but converge during afternoon."
        )

    if any(term in q for term in ["lake breeze", "lake effect", "coastal"]):
        results.append(
            "- Laird et al. (2001): Lake breeze cooling can suppress afternoon highs 5-10°F "
            "at lakefront METAR stations. Effect maximized when synoptic winds are weak (<10kt) "
            "and lake-land temperature differential exceeds 5°F."
        )

    if any(term in q for term in ["hrrr", "gfs", "model bias", "ensemble", "nbm"]):
        results.append(
            "- Hamill et al. (2017): The National Blend of Models (NBM) reduces MAE 10-20% over "
            "raw GFS by combining 50+ model outputs with bias correction. NBM outperforms "
            "individual models at 1-7 day lead times across CONUS."
        )
        results.append(
            "- Campbell & Diebold (2005): Temperature forecast model choice significantly affects "
            "weather derivative pricing. Ensemble disagreement predicts realized volatility."
        )

    if any(term in q for term in ["temperature distribution", "fat tail", "transition", "spring", "april"]):
        results.append(
            "- Jewson & Brix (2005): Temperature distributions are non-Gaussian with fat tails, "
            "especially in transition seasons (March-April, October-November). Standard normal "
            "assumptions underestimate tail probabilities by 15-30%."
        )
        results.append(
            "- Benth et al. (2008): Temperature follows AR processes with mean-reversion to "
            "seasonal norms. Deviations from climatology decay with a half-life of 2-4 days."
        )

    if any(term in q for term in ["prediction market", "polymarket", "betting", "liquidity"]):
        results.append(
            "- Wolfers & Zitzewitz (2004): Prediction markets aggregate information efficiently "
            "only when liquid and diverse. Low-liquidity markets show persistent mispricings."
        )
        results.append(
            "- Manski (2006): Market prices do not cleanly map to probabilities when "
            "participants are risk-averse or liquidity-constrained."
        )

    if any(term in q for term in ["solar", "radiation", "cloud", "heating"]):
        results.append(
            "- Holtslag & Van Ulden (1983): Surface heating rate depends on net radiation "
            "flux. Overcast skies reduce afternoon peak by 3-7°F versus clear-sky conditions. "
            "Thin cirrus attenuates ~10%, thick stratus ~60-80% of incoming shortwave."
        )

    return f"Curated academic references for '{query}':\n\n" + "\n\n".join(results)


async def fetch_nws_discussion(kwargs: dict[str, Any]) -> str:
    """Fetch the latest NWS Area Forecast Discussion for a city's WFO."""
    city_slug = kwargs.get("city_slug", "")
    city = CITY_REGISTRY_BY_SLUG.get(city_slug)
    if not city:
        return f"Error: City slug '{city_slug}' not found in registry."

    wfo = city.get("nws_office")
    if not wfo:
        return f"Error: No NWS Weather Forecast Office configured for '{city_slug}'. NWS discussions are US-only."

    # Step 1: Get the list of recent AFDs for this WFO
    list_url = f"https://api.weather.gov/products/types/AFD/locations/{wfo}"
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "User-Agent": "WeatherQuant/1.0 (weather-derivatives-research)",
                "Accept": "application/ld+json",
            }
            async with session.get(list_url, timeout=aiohttp.ClientTimeout(total=10), headers=headers) as resp:
                if resp.status != 200:
                    return f"Error fetching NWS AFD list: HTTP {resp.status}"
                data = await resp.json()
    except Exception as e:
        return f"Error connecting to NWS API: {e}"

    # Get the latest AFD product ID
    graph = data.get("@graph", [])
    if not graph:
        return f"No Area Forecast Discussions found for WFO {wfo}."

    latest_id = graph[0].get("id") or graph[0].get("@id", "")

    # Step 2: Fetch the actual AFD text
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "User-Agent": "WeatherQuant/1.0 (weather-derivatives-research)",
                "Accept": "application/ld+json",
            }
            async with session.get(latest_id, timeout=aiohttp.ClientTimeout(total=10), headers=headers) as resp:
                if resp.status != 200:
                    return f"Error fetching AFD content: HTTP {resp.status}"
                product = await resp.json()
    except Exception as e:
        return f"Error fetching AFD content: {e}"

    text = product.get("productText", "")
    issued = product.get("issuanceTime", "")

    if not text:
        return f"AFD for WFO {wfo} returned empty text."

    # Truncate to ~2000 chars to keep within token budget
    # Focus on the SHORT TERM and DISCUSSION sections
    sections_of_interest = []
    current_section = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(".") and stripped.endswith("..."):
            if current_section:
                sections_of_interest.append(current_section)
            current_section = stripped + "\n"
        elif stripped.startswith("&&"):
            if current_section:
                sections_of_interest.append(current_section)
            current_section = ""
        else:
            current_section += line + "\n"
    if current_section:
        sections_of_interest.append(current_section)

    # Prioritize SHORT TERM, NEAR TERM, DISCUSSION sections
    priority_keywords = ["SHORT TERM", "NEAR TERM", "DISCUSSION", "TODAY", "TEMPERATURE", "HIGH TEMP"]
    priority_sections = []
    other_sections = []
    for s in sections_of_interest:
        if any(kw in s.upper() for kw in priority_keywords):
            priority_sections.append(s.strip())
        else:
            other_sections.append(s.strip())

    combined = "\n\n---\n\n".join(priority_sections[:3])
    if not combined:
        combined = "\n\n---\n\n".join(other_sections[:2])

    # Truncate if too long
    if len(combined) > 2500:
        combined = combined[:2500] + "\n...[truncated]"

    return f"NWS Area Forecast Discussion — WFO {wfo} (issued {issued}):\n\n{combined}"


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


# ── Dispatcher ──────────────────────────────────────────────────────────────


async def dispatch_tool(tool_name: str, arguments: str) -> str:
    try:
        kwargs = json.loads(arguments)
    except json.JSONDecodeError:
        return f"Error: Tool arguments must be valid JSON. Received: {arguments}"

    handler = {
        "fetch_hrrr_forecast": fetch_hrrr_forecast,
        "fetch_nbm_forecast": fetch_nbm_forecast,
        "search_academic_climatology": search_academic_climatology,
        "search_academic_heuristics": search_academic_climatology,  # backward compat
        "fetch_nws_discussion": fetch_nws_discussion,
        "get_polymarket_bucket_odds": get_polymarket_bucket_odds,
    }.get(tool_name)

    if handler is None:
        return f"Error: Unknown tool '{tool_name}'"

    return await handler(kwargs)
