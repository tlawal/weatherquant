"""
METAR ingestion — polls aviationweather.gov every 60 seconds.

Ground truth for real-time temperature observations.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import aiohttp

from backend.tz_utils import city_local_date
from backend.storage.db import get_session
from backend.storage.models import City
from backend.storage.repos import (
    get_all_cities,
    get_daily_high_metar,
    get_event,
    get_metar_obs_by_key,
    get_recent_metar_obs_missing_extended,
    insert_metar_obs,
    upsert_metar_obs_extended,
    update_heartbeat,
)

log = logging.getLogger(__name__)

METAR_URL = "https://aviationweather.gov/api/data/metar"

_USER_AGENT = "WeatherQuant/1.0 (contact@weatherquant.local)"
_TIMEOUT = aiohttp.ClientTimeout(total=15)
_KNOTS_PER_MPS = 1.9438444924406
_KNOTS_PER_KPH = 0.539956803
_KNOTS_PER_MPH = 0.868976242
_INHG_PER_PA = 0.000295299830714
_INHG_PER_HPA = 0.0295299830714
_IN_PER_MM = 0.0393700787402
_FT_PER_M = 3.28083989501
_SM_PER_M = 0.000621371192237


def _c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _round_value(value: float, digits: int = 1) -> float:
    return round(float(value), digits)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _parse_temp(obs: dict) -> Optional[tuple[float, float]]:
    """Return (temp_c, temp_f) or None if not parseable."""
    # Primary: temp field (degrees C)
    temp_c = obs.get("temp")
    if temp_c is not None:
        try:
            tc = float(temp_c)
            return tc, _c_to_f(tc)
        except (ValueError, TypeError):
            pass

    # Fallback: temperature string in raw METAR
    raw = obs.get("rawOb", "") or ""
    import re
    m = re.search(r"\b(M?)(\d{2})/(M?\d{2})\b", raw)
    if m:
        try:
            sign = -1 if m.group(1) == "M" else 1
            tc = sign * float(m.group(2))
            return tc, _c_to_f(tc)
        except (ValueError, TypeError):
            pass

    return None


def _parse_obs_time(obs: dict) -> Optional[datetime]:
    """Parse observation time from METAR response."""
    raw_ts = obs.get("obsTime") or obs.get("reportTime")
    if raw_ts:
        try:
            # Can be epoch int or ISO string
            if isinstance(raw_ts, (int, float)):
                return datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
            return datetime.fromisoformat(str(raw_ts).rstrip("Z")).replace(
                tzinfo=timezone.utc
            )
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _humidity_from_magnus(temp_c: float, dewp_c: float) -> float:
    """Compute relative humidity (%) via the Magnus formula."""
    import math
    a, b = 17.625, 243.04
    rh = 100.0 * math.exp((a * dewp_c) / (b + dewp_c)) / math.exp((a * temp_c) / (b + temp_c))
    return round(max(0.0, min(100.0, rh)), 1)


# Map wx_string + cloud cover to human-readable condition
_WX_CONDITION_PATTERNS = [
    (("thunder", "ts", "tstm", "lightning"), "Thunderstorm"),
    (("hail", "small hail", "gr", "gs"), "Hail"),
    (("freezing rain", "fzra", "ice pellets", "sleet"), "Freezing Rain"),
    (("snow", "sn", "blowing snow"), "Snow"),
    (("showers", "shower", "shra"), "Showers"),
    (("drizzle", "dz"), "Drizzle"),
    (("rain", "ra"), "Rain"),
    (("fog", "fg"), "Fog"),
    (("mist", "br"), "Mist"),
    (("haze", "hz"), "Haze"),
    (("smoke", "fu"), "Smoke"),
]

_COVER_CONDITION = {
    "CLR": "Fair", "SKC": "Fair", "FEW": "Fair",
    "SCT": "Partly Cloudy", "BKN": "Mostly Cloudy", "OVC": "Cloudy",
}

_NWS_CLOUD_COVER_MAP = {
    "CLEAR": "CLR",
    "CLR": "CLR",
    "SKC": "SKC",
    "FEW": "FEW",
    "SCATTERED": "SCT",
    "SCT": "SCT",
    "BROKEN": "BKN",
    "BKN": "BKN",
    "OVERCAST": "OVC",
    "OVC": "OVC",
}

def _extract_qv_value(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        return _coerce_float(payload.get("value"))
    return _coerce_float(payload)


def _normalize_unit_code(unit_code: Optional[str]) -> str:
    return (unit_code or "").strip().lower()


def _convert_speed_to_knots(value: float, unit_code: Optional[str]) -> Optional[float]:
    unit = _normalize_unit_code(unit_code)
    if "m_s-1" in unit:
        return _round_value(value * _KNOTS_PER_MPS)
    if "km_h-1" in unit or "km/h" in unit:
        return _round_value(value * _KNOTS_PER_KPH)
    if "mi_h-1" in unit or "mph" in unit:
        return _round_value(value * _KNOTS_PER_MPH)
    if "kt" in unit or "knot" in unit:
        return _round_value(value)
    return None


def _convert_pressure_to_inhg(value: float, unit_code: Optional[str]) -> Optional[float]:
    unit = _normalize_unit_code(unit_code)
    if unit.endswith(":pa") or unit.endswith("/pa") or unit == "pa":
        return round(value * _INHG_PER_PA, 2)
    if "hectopa" in unit or "hpa" in unit or unit.endswith(":mb") or unit == "mb":
        return round(value * _INHG_PER_HPA, 2)
    if "inhg" in unit or "in_hg" in unit:
        return round(value, 2)
    return None


def _convert_pressure_to_mb(value: float, unit_code: Optional[str]) -> Optional[float]:
    unit = _normalize_unit_code(unit_code)
    if unit.endswith(":pa") or unit.endswith("/pa") or unit == "pa":
        return _round_value(value / 100.0)
    if "hectopa" in unit or "hpa" in unit or unit.endswith(":mb") or unit == "mb":
        return _round_value(value)
    if "inhg" in unit or "in_hg" in unit:
        return _round_value(value / _INHG_PER_HPA)
    return None


def _convert_length_to_inches(value: float, unit_code: Optional[str]) -> Optional[float]:
    unit = _normalize_unit_code(unit_code)
    if unit.endswith(":mm") or unit == "mm":
        return round(value * _IN_PER_MM, 2)
    if unit.endswith(":cm") or unit == "cm":
        return round(value * _IN_PER_MM * 10, 2)
    if unit.endswith(":m") or unit == "m":
        return round(value * _IN_PER_MM * 1000, 2)
    if "inch" in unit or unit.endswith(":in") or unit == "in":
        return round(value, 2)
    return None


def _convert_length_to_feet(value: float, unit_code: Optional[str]) -> Optional[float]:
    unit = _normalize_unit_code(unit_code)
    if unit.endswith(":m") or unit == "m":
        return _coerce_int(value * _FT_PER_M)
    if "foot" in unit or unit.endswith(":ft") or unit == "ft":
        return _coerce_int(value)
    return None


def _convert_visibility_to_sm(value: float, unit_code: Optional[str]) -> Optional[float]:
    unit = _normalize_unit_code(unit_code)
    if unit.endswith(":m") or unit == "m":
        return _round_value(value * _SM_PER_M)
    if "mile" in unit or unit.endswith(":sm") or unit == "sm":
        return _round_value(value)
    return None


def _extract_qv_converted(payload: Any, converter) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    value = _extract_qv_value(payload)
    if value is None:
        return None
    return converter(value, payload.get("unitCode"))


def _normalize_cloud_cover(cloud_cover: Optional[str]) -> Optional[str]:
    if not cloud_cover:
        return None
    return _NWS_CLOUD_COVER_MAP.get(str(cloud_cover).strip().upper())


def _format_present_weather(present_weather: Any) -> Optional[str]:
    if not isinstance(present_weather, list):
        return None

    parts: list[str] = []
    for entry in present_weather:
        if not isinstance(entry, dict):
            continue
        token_parts = [
            str(entry.get(key)).strip()
            for key in ("coverage", "intensity", "modifier", "weather")
            if entry.get(key)
        ]
        if token_parts:
            parts.append(" ".join(token_parts))

    if not parts:
        return None

    return "; ".join(parts)[:64]


def _derive_condition(
    wx_string: Optional[str],
    cloud_cover: Optional[str],
    text_description: Optional[str] = None,
) -> Optional[str]:
    """Derive a short human-readable condition from weather/cloud signals."""
    haystack = " ".join(
        part for part in [wx_string, text_description] if part
    ).lower()

    if haystack:
        for tokens, label in _WX_CONDITION_PATTERNS:
            if any(token in haystack for token in tokens):
                return label
    if cloud_cover:
        return _COVER_CONDITION.get(cloud_cover.upper())
    if haystack and any(token in haystack for token in ("fair", "clear", "sunny")):
        return "Fair"
    return None


def parse_aviationweather_extended(obs: dict, temp_c: Optional[float] = None) -> dict:
    """Parse extended METAR fields from aviationweather.gov JSON response.

    Returns a dict suitable for upsert_metar_obs_extended().
    """
    ext: dict = {}

    # Dewpoint
    dewp_c = obs.get("dewp")
    if dewp_c is not None:
        dc = _coerce_float(dewp_c)
        if dc is not None:
            ext["dewpoint_c"] = round(dc, 1)
            ext["dewpoint_f"] = _c_to_f(dc)
            if temp_c is not None:
                ext["humidity_pct"] = _humidity_from_magnus(float(temp_c), dc)

    # Wind
    wdir = _coerce_int(obs.get("wdir"))
    if wdir is not None:
        ext["wind_dir_deg"] = wdir

    wspd = _coerce_float(obs.get("wspd"))
    if wspd is not None:
        ext["wind_speed_kt"] = wspd

    wgst = _coerce_float(obs.get("wgst"))
    if wgst is not None:
        ext["wind_gust_kt"] = wgst

    # Pressure
    altim = _coerce_float(obs.get("altim"))
    if altim is not None:
        ext["altimeter_inhg"] = round(altim, 2)

    slp = _coerce_float(obs.get("slp"))
    if slp is not None:
        ext["sea_level_pressure_mb"] = round(slp, 1)

    # Visibility
    visib = _coerce_float(obs.get("visib"))
    if visib is not None:
        ext["visibility_sm"] = round(visib, 1)

    # Cloud cover — use the dominant (highest) cover level
    clouds = obs.get("clouds")
    cover = obs.get("cover")
    if isinstance(clouds, list) and clouds:
        # clouds is array of {cover: "BKN", base: 5000}
        # Use the first layer's cover as the dominant one
        first = clouds[0]
        ext["cloud_cover"] = first.get("cover")
        base = first.get("base")
        base_ft = _coerce_int(base)
        if base_ft is not None:
            ext["cloud_base_ft"] = base_ft
    elif cover:
        ext["cloud_cover"] = str(cover)

    # Weather string (present weather)
    wx = obs.get("wxString")
    if wx:
        ext["wx_string"] = str(wx)[:64]

    # Derive human-readable condition
    condition = _derive_condition(ext.get("wx_string"), ext.get("cloud_cover"))
    if condition:
        ext["condition"] = condition

    return ext


def parse_nws_extended(props: dict) -> dict:
    """Parse extended observation fields from api.weather.gov properties."""
    ext: dict = {}

    dewpoint_c = _extract_qv_value(props.get("dewpoint"))
    if dewpoint_c is not None:
        ext["dewpoint_c"] = _round_value(dewpoint_c)
        ext["dewpoint_f"] = _c_to_f(dewpoint_c)

    humidity = _extract_qv_value(props.get("relativeHumidity"))
    if humidity is not None:
        ext["humidity_pct"] = _round_value(humidity)

    wind_dir = _extract_qv_value(props.get("windDirection"))
    if wind_dir is not None:
        ext["wind_dir_deg"] = _coerce_int(wind_dir)

    wind_speed = _extract_qv_converted(props.get("windSpeed"), _convert_speed_to_knots)
    if wind_speed is not None:
        ext["wind_speed_kt"] = wind_speed

    wind_gust = _extract_qv_converted(props.get("windGust"), _convert_speed_to_knots)
    if wind_gust is not None:
        ext["wind_gust_kt"] = wind_gust

    altimeter = _extract_qv_converted(props.get("barometricPressure"), _convert_pressure_to_inhg)
    if altimeter is not None:
        ext["altimeter_inhg"] = altimeter

    sea_level = _extract_qv_converted(props.get("seaLevelPressure"), _convert_pressure_to_mb)
    if sea_level is not None:
        ext["sea_level_pressure_mb"] = sea_level

    visibility = _extract_qv_converted(props.get("visibility"), _convert_visibility_to_sm)
    if visibility is not None:
        ext["visibility_sm"] = visibility

    precip = _extract_qv_converted(props.get("precipitationLastHour"), _convert_length_to_inches)
    if precip is not None:
        ext["precip_in"] = precip

    cloud_layers = props.get("cloudLayers")
    if isinstance(cloud_layers, list) and cloud_layers:
        for layer in cloud_layers:
            if not isinstance(layer, dict):
                continue
            cover = _normalize_cloud_cover(layer.get("amount"))
            base_ft = _extract_qv_converted(layer.get("base"), _convert_length_to_feet)
            if cover:
                ext["cloud_cover"] = cover
            if base_ft is not None:
                ext["cloud_base_ft"] = _coerce_int(base_ft)
            if cover or base_ft is not None:
                break

    wx_string = _format_present_weather(props.get("presentWeather"))
    text_description = props.get("textDescription")
    if not wx_string and text_description:
        wx_string = str(text_description)[:64]
    if wx_string:
        ext["wx_string"] = wx_string[:64]

    condition = _derive_condition(
        ext.get("wx_string"),
        ext.get("cloud_cover"),
        str(text_description) if text_description else None,
    )
    if condition:
        ext["condition"] = condition[:32]

    return ext


def _parse_nws_temp(props: dict) -> Optional[tuple[float, float]]:
    temp_c = _extract_qv_value(props.get("temperature"))
    if temp_c is None:
        return None
    return float(temp_c), _c_to_f(temp_c)


def _parse_nws_obs_time(props: dict) -> datetime:
    ts_str = props.get("timestamp")
    if ts_str:
        try:
            return datetime.fromisoformat(str(ts_str).rstrip("Z")).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


async def _insert_or_merge_metar_observation(
    city: City,
    station_id: str,
    observed_at: datetime,
    report_at: datetime,
    temp_c: float,
    temp_f: float,
    raw_text: Optional[str],
    raw_json: str,
    ext_data: dict,
) -> None:
    today_local = city_local_date(city)

    async with get_session() as sess:
        metar_row = await get_metar_obs_by_key(sess, city.id, station_id, observed_at)
        if metar_row is None:
            prev_high = await get_daily_high_metar(
                sess,
                city.id,
                today_local,
                city_tz=getattr(city, "tz", "America/New_York"),
            )
            daily_high = max((v for v in [temp_f, prev_high] if v is not None), default=temp_f)
            metar_row = await insert_metar_obs(
                sess,
                city_id=city.id,
                metar_station=station_id,
                observed_at=observed_at,
                report_at=report_at,
                temp_c=temp_c,
                temp_f=temp_f,
                daily_high_f=daily_high,
                raw_text=raw_text,
                raw_json=raw_json,
                source="aviation",
            )

        if ext_data:
            try:
                await upsert_metar_obs_extended(sess, metar_obs_id=metar_row.id, **ext_data)
            except Exception as e:
                log.debug("metar_ext: failed for %s at %s: %s", station_id, observed_at, e)


async def backfill_recent_nws_extended(hours: int = 24) -> int:
    """Backfill missing extended rows for recent NWS observations."""
    repaired = 0
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with get_session() as sess:
        rows = await get_recent_metar_obs_missing_extended(sess, since)
        for row in rows:
            raw_json = row.raw_json
            if not raw_json:
                continue

            try:
                raw = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            except Exception:
                log.debug("nws_backfill: invalid raw_json for metar_obs_id=%s", row.id)
                continue

            if not isinstance(raw, dict) or raw.get("source") != "nws_obs":
                continue

            ext_data = parse_nws_extended(raw)
            if not ext_data:
                continue

            await upsert_metar_obs_extended(sess, metar_obs_id=row.id, **ext_data)
            repaired += 1

    if repaired:
        log.info("nws_backfill: repaired %d recent extended observation rows", repaired)
    else:
        log.info("nws_backfill: no recent rows needed repair")
    return repaired


async def fetch_metar_all() -> None:
    """Fetch Current Weather for all enabled cities. US uses METAR; Intl uses Open-Meteo."""
    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    if not cities:
        log.warning("metar: no enabled cities configured")
        return

    us_cities = [c for c in cities if c.is_us and c.metar_station]
    # International cities with ICAO metar_station also use the global aviationweather.gov feed
    intl_metar_cities = [c for c in cities if not c.is_us and c.metar_station]
    # International cities without a station code use Open-Meteo
    intl_openmeteo_cities = [c for c in cities if not c.is_us and not c.metar_station and c.lat and c.lon]

    # Per-event override stations: poll Polymarket's resolution station too
    # whenever it differs from the city default. Stored under the same city_id.
    us_extra = await _resolve_extra_event_stations(us_cities) if us_cities else []
    intl_extra = await _resolve_extra_event_stations(intl_metar_cities) if intl_metar_cities else []

    if us_cities or us_extra:
        await _fetch_us_metars(us_cities, extra_pairs=us_extra)

    if intl_metar_cities or intl_extra:
        # ICAO stations are global — aviationweather.gov covers all of them
        await _fetch_us_metars(intl_metar_cities, extra_pairs=intl_extra)

    # Supplementary: weather.gov observations as fallback/cross-validation
    nws_obs_cities = us_cities + intl_metar_cities
    if nws_obs_cities:
        try:
            await _fetch_nws_observations(nws_obs_cities)
        except Exception as e:
            log.warning("nws_obs: supplementary fetch failed: %s", e)

    if intl_openmeteo_cities:
        await _fetch_intl_open_meteo(intl_openmeteo_cities)


async def _resolve_extra_event_stations(cities: list[City]) -> list[tuple[City, str]]:
    """For each city, return (city, override_station) pairs where the active
    event names a resolution station that differs from the city default.

    These are polled in addition to the default station so the model can
    settle on Polymarket's actual resolution source. Observations are stored
    under the same city_id but with the override station_id, so existing
    queries (get_daily_high_metar / get_resolution_high_metar) see both
    streams.
    """
    pairs: list[tuple[City, str]] = []
    for city in cities:
        if not city.metar_station:
            continue
        default_station = city.metar_station.upper()
        today_local = city_local_date(city)
        async with get_session() as sess:
            event = await get_event(sess, city.id, today_local)
        override = getattr(event, "resolution_station_id", None) if event is not None else None
        if not override:
            continue
        override = override.upper()
        if override == default_station:
            continue
        pairs.append((city, override))
    return pairs


async def _fetch_us_metars(cities: list[City], extra_pairs: Optional[list[tuple[City, str]]] = None) -> None:
    """Fetch METARs for the given cities (using each city's default station),
    plus any optional extra (city, override_station) pairs that should be
    stored under the same city_id."""
    station_map: dict[str, City] = {c.metar_station.upper(): c for c in cities if c.metar_station}
    if extra_pairs:
        for city, station in extra_pairs:
            station_map.setdefault(station.upper(), city)
    if not station_map:
        return
    stations = ",".join(station_map.keys())
    url = f"{METAR_URL}?ids={stations}&format=json&latest=1"

    try:
        async with aiohttp.ClientSession(
            timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
        ) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.error("metar: HTTP %d from %s", resp.status, url)
                    return
                data = await resp.json(content_type=None)
    except Exception as e:
        log.error("metar: fetch failed: %s", e)
        return

    if not isinstance(data, list):
        return

    for obs in data:
        station_id = (obs.get("stationId") or obs.get("station") or "").upper()
        city = station_map.get(station_id)
        if not city:
            continue

        temp = _parse_temp(obs)
        if temp is None:
            continue
        temp_c, temp_f = temp

        obs_time = _parse_obs_time(obs)
        raw_str = json.dumps(obs, default=str)
        report_at = _parse_obs_time({"obsTime": obs.get("reportTime")}) if obs.get("reportTime") else obs_time
        raw_text = obs.get("rawOb")
        ext_data = parse_aviationweather_extended(obs, temp_c=temp_c)

        await _insert_or_merge_metar_observation(
            city=city,
            station_id=station_id,
            observed_at=obs_time,
            report_at=report_at,
            temp_c=temp_c,
            temp_f=temp_f,
            raw_text=raw_text,
            raw_json=raw_str,
            ext_data=ext_data,
        )

    await _mark_heartbeat_success()


async def _fetch_nws_observations(cities: list[City]) -> None:
    """Supplementary: fetch latest observation from weather.gov for US cities."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
    async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=headers) as http:
        for city in cities:
            try:
                url = f"https://api.weather.gov/stations/{city.metar_station}/observations/latest"
                async with http.get(url) as resp:
                    if resp.status != 200:
                        log.debug("nws_obs: HTTP %d for %s", resp.status, city.metar_station)
                        continue
                    data = await resp.json(content_type=None)

                props = data.get("properties", {})
                temp = _parse_nws_temp(props)
                if temp is None:
                    continue
                temp_c, temp_f = temp
                obs_time = _parse_nws_obs_time(props)
                station_id = (city.metar_station or "").upper()

                await _insert_or_merge_metar_observation(
                    city=city,
                    station_id=station_id,
                    observed_at=obs_time,
                    report_at=obs_time,
                    temp_c=temp_c,
                    temp_f=temp_f,
                    raw_text=props.get("rawMessage"),
                    raw_json=json.dumps({"source": "nws_obs", **props}, default=str),
                    ext_data=parse_nws_extended(props),
                )
                await asyncio.sleep(0.3)  # rate limit courtesy
            except Exception as e:
                log.error("nws_obs: %s failed: %s", city.metar_station, e)


async def _fetch_intl_open_meteo(cities: list[City]) -> None:
    """Fetch current weather for international cities via Open-Meteo."""
    # Group by lat/lon to use Open-Meteo's bulk API
    lats = ",".join(str(c.lat) for c in cities)
    lons = ",".join(str(c.lon) for c in cities)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&current_weather=true"

    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as http:
            async with http.get(url) as resp:
                if resp.status != 200:
                    log.error("open-meteo: HTTP %d from %s", resp.status, url)
                    return
                data = await resp.json()
    except Exception as e:
        log.error("open-meteo: fetch failed: %s", e)
        return

    # Open-Meteo returns a list if multiple locations are requested
    if not isinstance(data, list):
        data = [data]

    for i, obs_data in enumerate(data):
        if i >= len(cities):
            break
        city = cities[i]
        today_local = city_local_date(city)
        curr = obs_data.get("current_weather")
        if not curr:
            continue

        temp_c = curr.get("temperature")
        if temp_c is None:
            continue

        # User said for intl cities use C but my internal daily_high_f column handles whatever is passed.
        # Actually for intl cities I should store C in the F column as well if that's what's used.
        temp_internal = float(temp_c)
        if city.unit == "F":
             temp_internal = _c_to_f(temp_internal)

        obs_time_raw = curr.get("time") # ISO string
        try:
            obs_time = datetime.fromisoformat(obs_time_raw).replace(tzinfo=timezone.utc)
        except Exception:
            obs_time = datetime.now(timezone.utc)

        raw_str = json.dumps(obs_data)

        async with get_session() as sess:
            prev_high = await get_daily_high_metar(sess, city.id, today_local, city_tz=getattr(city, "tz", "America/New_York"))
            daily_high = max(
                (v for v in [temp_internal, prev_high] if v is not None), default=temp_internal
            )

            await insert_metar_obs(
                sess,
                city_id=city.id,
                metar_station=city.metar_station or "OM",
                observed_at=obs_time,
                report_at=obs_time, # Open-Meteo usually just has one time
                temp_c=temp_c,
                temp_f=temp_internal,
                daily_high_f=daily_high,
                raw_text=None,
                raw_json=raw_str,
                source="open_meteo",
            )


def should_poll_station(
    observation_minutes: list[int], now_minute: int,
    pre_window: int = 2, post_window: int = 10,
    window: int | None = None,
) -> bool:
    """Return True if now_minute is within [-pre_window, +post_window] of any observation minute.

    Asymmetric window: poll from 2 min before to 10 min after each observation
    minute, covering delayed appearances on aviationweather.gov.
    Legacy `window` param is accepted but ignored.
    """
    for m in observation_minutes:
        diff = (now_minute - m) % 60
        if diff <= post_window or diff >= (60 - pre_window):
            return True
    return False


async def fetch_metar_smart() -> None:
    """Fetch METAR only for stations near their observation window."""
    from backend.storage.repos import get_station_profile

    now = datetime.now(timezone.utc)
    now_minute = now.minute

    async with get_session() as sess:
        cities = await get_all_cities(sess, enabled_only=True)

    # All cities with METAR station (US and intl ICAO) use smart polling
    metar_cities = [c for c in cities if c.metar_station]
    in_window = []
    for city in metar_cities:
        async with get_session() as sess:
            profile = await get_station_profile(sess, city.metar_station)
        if not profile or not profile.observation_minutes or (profile.confidence or 0) < 0.7:
            in_window.append(city)  # no confident profile → always poll
            continue
        valid_minutes = json.loads(profile.observation_minutes)
        if should_poll_station(valid_minutes, now_minute):
            in_window.append(city)

    # Per-event override stations: include any station Polymarket points at,
    # gated by the override station's own profile (or always poll if unknown).
    extra_pairs: list[tuple[City, str]] = []
    if metar_cities:
        all_overrides = await _resolve_extra_event_stations(metar_cities)
        for city, override in all_overrides:
            async with get_session() as sess:
                profile = await get_station_profile(sess, override)
            if not profile or not profile.observation_minutes or (profile.confidence or 0) < 0.7:
                extra_pairs.append((city, override))
                continue
            valid_minutes = json.loads(profile.observation_minutes)
            if should_poll_station(valid_minutes, now_minute):
                extra_pairs.append((city, override))

    if in_window or extra_pairs:
        await _fetch_us_metars(in_window, extra_pairs=extra_pairs)
        log.info(
            "smart_poll: fetched %d/%d METAR stations (+%d event-overrides) at minute :%02d",
            len(in_window), len(metar_cities), len(extra_pairs), now_minute,
        )

    # International cities without METAR station use Open-Meteo (always polled)
    intl_openmeteo_cities = [c for c in cities if not c.is_us and not c.metar_station and c.lat and c.lon]
    if intl_openmeteo_cities:
        await _fetch_intl_open_meteo(intl_openmeteo_cities)


async def _mark_heartbeat_success() -> None:
    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_metar", success=True)


async def _mark_heartbeat_error(error: str) -> None:
    async with get_session() as sess:
        await update_heartbeat(sess, "fetch_metar", success=False, error=error)
