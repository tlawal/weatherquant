"""
Repository layer — all database queries are here.

Business logic MUST NOT use raw SQL or direct ORM queries.
Every public function returns typed Python objects, never raw Row objects.
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import desc, func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Config
from backend.storage.models import (
    ArmingState,
    AuditLog,
    Bucket,
    CalibrationParams,
    City,
    ClosedTrade,
    Event,
    Fill,
    ForecastObs,
    MarketContextSnapshot,
    MarketFlowFeature,
    MarketSnapshot,
    MadisObs,
    MetarObs,
    MetarObsExtended,
    ModelArtifact,
    ModelSnapshot,
    Order,
    Position,
    RuntimeConfig,
    Signal,
    SourceLeadTimeSkill,
    StationCalibration,
    StationProfile,
    WalletMarketExposure,
    WalletSkillScore,
    WalletStat,
    WalletTrade,
    WorkerHeartbeat,
)


def _chunks(items: list[Any], size: int = 1000):
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _json_dumps_compact(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, separators=(",", ":"))
    except Exception:
        return None


def _bounded_string(value: Any, max_len: int = 256) -> str | None:
    if value is None:
        return None
    s = str(value)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _compact_forecast_raw_payload(source: str | None, raw_json: Any) -> str | None:
    """Keep only metadata needed by downstream logic unless full raw storage is enabled."""
    if raw_json is None or Config.STORE_RAW_FORECAST_PAYLOADS:
        return raw_json
    payload = _json_loads_maybe(raw_json)
    if not isinstance(payload, dict):
        return None
    keep_keys = (
        "source",
        "obs_time",
        "peak_hour",
        "high_f",
        "temp_f",
        "model_run_at",
        "generated_at",
        "station_id",
        "valid_time",
    )
    compact = {k: payload.get(k) for k in keep_keys if payload.get(k) is not None}
    if source:
        compact.setdefault("source", source)
    return _json_dumps_compact(compact) if compact else None


_SIGNAL_REASON_KEEP_KEYS = {
    "active_station_id",
    "adaptive_sigma_f",
    "ask_depth",
    "bid_depth",
    "bucket_live_calibration",
    "city_state",
    "consensus_bucket_idx",
    "current_temp_f",
    "daily_high_metar",
    "ev_at_bid",
    "ev_per_share",
    "exec_cost",
    "kalman_divergence_f",
    "kalman_nowcast_active",
    "lock_regime",
    "market_sanity",
    "market_snapshot_at",
    "market_snapshot_id",
    "metar_condition",
    "microstructure_shadow",
    "model_prob_raw",
    "model_snapshot_id",
    "mu_forecast",
    "mu_multi_model",
    "observation_minutes",
    "observed_bucket_idx",
    "observed_bucket_upper_f",
    "posterior_kelly",
    "prob_hotter_bucket",
    "prob_new_high",
    "prob_new_high_raw",
    "projected_high",
    "projected_high_for_blend",
    "raw_high",
    "regime_label",
    "regime_score",
    "resolution_high",
    "resolution_high_f",
    "resolution_mismatch",
    "sigma_raw",
    "source_quality",
    "source_quality_gates",
    "spread",
    "station_mae_f",
    "threshold_calibration",
    "time_to_settlement_h",
}


def _shrink_reason_value(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, str):
                out[k] = _bounded_string(v, 192)
            elif isinstance(v, (int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, list):
                out[k] = [_shrink_reason_value(x) for x in v[:12]]
            elif isinstance(v, dict):
                out[k] = _shrink_reason_value(v)
        return out
    if isinstance(value, list):
        return [_shrink_reason_value(v) for v in value[:12]]
    if isinstance(value, str):
        return _bounded_string(value, 192)
    return value


def compact_signal_reason_json(reason: Any) -> str | None:
    if reason is None:
        return None
    if Config.STORE_FULL_SIGNAL_REASON_JSON:
        raw = _json_dumps_compact(reason)
    else:
        payload = _json_loads_maybe(reason)
        if not isinstance(payload, dict):
            return None
        compact = {
            key: _shrink_reason_value(payload[key])
            for key in _SIGNAL_REASON_KEEP_KEYS
            if key in payload and payload[key] is not None
        }
        raw = _json_dumps_compact(compact)
    if raw is None:
        return None
    max_bytes = max(512, int(Config.SIGNAL_REASON_MAX_JSON_BYTES or 6000))
    if len(raw.encode("utf-8")) <= max_bytes:
        return raw
    payload = _json_loads_maybe(raw)
    if isinstance(payload, dict):
        for key in ("microstructure_shadow", "source_quality", "source_quality_gates", "threshold_calibration"):
            payload.pop(key, None)
        raw = _json_dumps_compact(payload)
    return raw if raw and len(raw.encode("utf-8")) <= max_bytes else raw[:max_bytes]


def compact_gate_failures_json(gate_failures: Any) -> str | None:
    payload = _json_loads_maybe(gate_failures)
    if payload is None:
        payload = gate_failures
    if isinstance(payload, list):
        payload = [_bounded_string(item, 160) for item in payload[:20]]
    return _json_dumps_compact(payload)


# ─── Cities ───────────────────────────────────────────────────────────────────

async def get_all_cities(session: AsyncSession, enabled_only: bool = False) -> list[City]:
    from backend.city_registry import get_city_priority

    q = select(City)
    if enabled_only:
        q = q.where(City.enabled.is_(True))
    result = await session.execute(q)
    cities = list(result.scalars().all())
    cities.sort(key=lambda c: get_city_priority(c.city_slug))
    return cities


async def get_city_by_slug(session: AsyncSession, city_slug: str) -> Optional[City]:
    result = await session.execute(
        select(City).where(City.city_slug == city_slug)
    )
    return result.scalar_one_or_none()


async def upsert_city(session: AsyncSession, city_data: dict) -> City:
    slug = city_data["city_slug"]
    city = await get_city_by_slug(session, slug)
    if city is None:
        city = City(**city_data)
        session.add(city)
    else:
        for k, v in city_data.items():
            if hasattr(city, k):
                setattr(city, k, v)
    await session.commit()
    await session.refresh(city)
    return city


# ─── Events ───────────────────────────────────────────────────────────────────

async def get_event(
    session: AsyncSession, city_id: int, date_et: str
) -> Optional[Event]:
    result = await session.execute(
        select(Event)
        .where(Event.city_id == city_id, Event.date_et == date_et)
    )
    return result.scalar_one_or_none()


async def get_event_by_id(session: AsyncSession, event_id: int) -> Optional[Event]:
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Event)
        .where(Event.id == event_id)
        .options(selectinload(Event.buckets))
    )
    return result.scalar_one_or_none()


async def upsert_event(session: AsyncSession, city_id: int, date_et: str, **kwargs) -> Event:
    event = await get_event(session, city_id, date_et)
    if event is None:
        event = Event(city_id=city_id, date_et=date_et, **kwargs)
        session.add(event)
    else:
        for k, v in kwargs.items():
            if hasattr(event, k):
                setattr(event, k, v)
    await session.commit()
    await session.refresh(event)
    return event


async def get_event_with_buckets(
    session: AsyncSession, city_id: int, date_et: str
) -> Optional[Event]:
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Event)
        .options(selectinload(Event.buckets))
        .where(Event.city_id == city_id, Event.date_et == date_et)
    )
    return result.scalar_one_or_none()


async def get_recent_events_for_city(
    session: AsyncSession,
    city_id: int,
    before_or_on_date_et: str | None = None,
    limit: int = 14,
) -> list[Event]:
    q = select(Event).where(Event.city_id == city_id)
    if before_or_on_date_et:
        q = q.where(Event.date_et <= before_or_on_date_et)
    q = q.order_by(desc(Event.date_et)).limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


# ─── Buckets ──────────────────────────────────────────────────────────────────

async def get_buckets_for_event(session: AsyncSession, event_id: int) -> list[Bucket]:
    result = await session.execute(
        select(Bucket)
        .where(Bucket.event_id == event_id)
        .order_by(Bucket.bucket_idx)
    )
    return list(result.scalars().all())


async def get_bucket_by_id(session: AsyncSession, bucket_id: int) -> Optional[Bucket]:
    result = await session.execute(select(Bucket).where(Bucket.id == bucket_id))
    return result.scalar_one_or_none()


async def upsert_bucket(
    session: AsyncSession, event_id: int, bucket_idx: int, **kwargs
) -> Bucket:
    result = await session.execute(
        select(Bucket).where(
            Bucket.event_id == event_id,
            Bucket.bucket_idx == bucket_idx,
        )
    )
    bucket = result.scalar_one_or_none()
    if bucket is None:
        bucket = Bucket(event_id=event_id, bucket_idx=bucket_idx, **kwargs)
        session.add(bucket)
    else:
        for k, v in kwargs.items():
            if hasattr(bucket, k):
                setattr(bucket, k, v)
    await session.commit()
    await session.refresh(bucket)
    return bucket


# ─── METAR ────────────────────────────────────────────────────────────────────

async def insert_metar_obs(session: AsyncSession, **kwargs) -> MetarObs:
    obs = MetarObs(**kwargs)
    session.add(obs)
    await session.commit()
    return obs


async def insert_metar_obs_extended(session: AsyncSession, **kwargs) -> MetarObsExtended:
    ext = MetarObsExtended(**kwargs)
    session.add(ext)
    await session.commit()
    return ext


async def get_metar_obs_by_key(
    session: AsyncSession,
    city_id: int,
    metar_station: str,
    observed_at: datetime,
    source: Optional[str] = None,
) -> Optional[MetarObs]:
    conditions = [
        MetarObs.city_id == city_id,
        MetarObs.metar_station == metar_station,
        MetarObs.observed_at == observed_at,
    ]
    if source is not None:
        conditions.append(MetarObs.source == source)
    result = await session.execute(
        select(MetarObs).where(*conditions).limit(1)
    )
    return result.scalar_one_or_none()


async def upsert_metar_obs_extended(
    session: AsyncSession,
    metar_obs_id: int,
    **kwargs,
) -> MetarObsExtended:
    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None and hasattr(MetarObsExtended, key)
    }

    result = await session.execute(
        select(MetarObsExtended).where(MetarObsExtended.metar_obs_id == metar_obs_id).limit(1)
    )
    ext = result.scalar_one_or_none()
    if ext is None:
        ext = MetarObsExtended(metar_obs_id=metar_obs_id, **clean_kwargs)
        session.add(ext)
    else:
        for key, value in clean_kwargs.items():
            if getattr(ext, key) is None:
                setattr(ext, key, value)

    await session.commit()
    await session.refresh(ext)
    return ext


async def get_recent_metar_obs_missing_extended(
    session: AsyncSession,
    since: datetime,
) -> list[MetarObs]:
    result = await session.execute(
        select(MetarObs)
        .outerjoin(MetarObsExtended, MetarObsExtended.metar_obs_id == MetarObs.id)
        .where(
            MetarObs.observed_at >= since,
            MetarObs.raw_json.isnot(None),
            MetarObsExtended.id.is_(None),
        )
        .order_by(MetarObs.observed_at.desc())
    )
    return list(result.scalars().all())


async def get_todays_extended_obs(
    session: AsyncSession,
    city_id: int,
    date_et: str,
    city_tz: str = "America/New_York",
) -> list[MetarObs]:
    """Return ALL MetarObs with eagerly-loaded extended fields for a given local date.

    Each returned MetarObs has its `.extended` relationship populated (or None).
    """
    from sqlalchemy.orm import selectinload

    tz = ZoneInfo(city_tz)
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)

    result = await session.execute(
        select(MetarObs)
        .options(selectinload(MetarObs.extended))
        .where(
            MetarObs.city_id == city_id,
            MetarObs.temp_f.isnot(None),
            MetarObs.observed_at >= start_dt,
            MetarObs.observed_at < end_dt,
        )
        .order_by(MetarObs.observed_at)
    )
    return list(result.scalars().all())


async def get_latest_metar(
    session: AsyncSession, city_id: int
) -> Optional[MetarObs]:
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(MetarObs)
        .options(selectinload(MetarObs.extended))
        .where(MetarObs.city_id == city_id)
        .order_by(desc(MetarObs.observed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_metar_by_source(
    session: AsyncSession,
    city_id: int,
    source: str,
    metar_station: Optional[str] = None,
) -> Optional[MetarObs]:
    """Latest MetarObs for a city filtered by source (e.g. 'tgftp', 'aviation')."""
    conditions = [MetarObs.city_id == city_id, MetarObs.source == source]
    if metar_station:
        conditions.append(MetarObs.metar_station == metar_station.upper())
    result = await session.execute(
        select(MetarObs)
        .where(*conditions)
        .order_by(desc(MetarObs.observed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_daily_high_metar_obs(
    session: AsyncSession, city_id: int, date_et: str,
    city_tz: str = "America/New_York",
    source: Optional[str] = None,
) -> Optional[MetarObs]:
    """Return the MetarObs row that achieved the daily high temp_f for the given local date.

    When source is provided, only consider observations from that source.
    Returns the most recent observation at the max temperature.
    """
    tz = ZoneInfo(city_tz)
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)

    # First get the max temp_f
    q_max = select(func.max(MetarObs.temp_f)).where(
        MetarObs.city_id == city_id,
        MetarObs.observed_at >= start_dt,
        MetarObs.observed_at < end_dt,
    )
    if source is not None:
        q_max = q_max.where(MetarObs.source == source)

    max_result = await session.execute(q_max)
    max_temp = max_result.scalar_one_or_none()
    if max_temp is None:
        return None

    # Then get the most recent row at that max temp
    q_row = select(MetarObs).where(
        MetarObs.city_id == city_id,
        MetarObs.observed_at >= start_dt,
        MetarObs.observed_at < end_dt,
        MetarObs.temp_f == max_temp,
    )
    if source is not None:
        q_row = q_row.where(MetarObs.source == source)

    q_row = q_row.order_by(desc(MetarObs.observed_at)).limit(1)
    result = await session.execute(q_row)
    return result.scalar_one_or_none()


async def get_daily_high_metar(
    session: AsyncSession, city_id: int, date_et: str,
    city_tz: str = "America/New_York",
    source: Optional[str] = None,
) -> Optional[float]:
    """Max temp_f observed for a city on the given local date.

    When source is provided, only consider observations from that source
    (e.g. 'tgftp' or 'aviation'). When None, aggregate across all sources.
    """
    tz = ZoneInfo(city_tz)
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)

    q = select(func.max(MetarObs.temp_f)).where(
        MetarObs.city_id == city_id,
        MetarObs.observed_at >= start_dt,
        MetarObs.observed_at < end_dt,
    )
    if source is not None:
        q = q.where(MetarObs.source == source)

    result = await session.execute(q)
    return result.scalar_one_or_none()


async def get_avg_peak_timing(
    session: AsyncSession, city_id: int, days_back: int = 3, et_tz: ZoneInfo = ZoneInfo("America/New_York")
) -> Optional[str]:
    """Calculate the average time of day the maximum temperature was reached over the last N days."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back + 1)
    
    result = await session.execute(
        select(MetarObs)
        .where(
            MetarObs.city_id == city_id,
            MetarObs.temp_f.isnot(None),
            MetarObs.observed_at >= since
        )
    )
    obs_list = list(result.scalars().all())
    if not obs_list:
        return None
        
    from collections import defaultdict
    daily_obs = defaultdict(list)
    for obs in obs_list:
        dt_local = obs.observed_at.astimezone(et_tz)
        daily_obs[dt_local.date()].append(obs)
        
    today_date = datetime.now(et_tz).date()
    peak_minutes = []
    
    for d, obs_for_day in daily_obs.items():
        if d == today_date and datetime.now(et_tz).hour < 20:
            continue  # don't include today if it's not late enough
            
        best_obs = max(obs_for_day, key=lambda o: o.temp_f)
        best_dt = best_obs.observed_at.astimezone(et_tz)
        minutes_since_midnight = best_dt.hour * 60 + best_dt.minute
        peak_minutes.append(minutes_since_midnight)
        
    if not peak_minutes:
        return None
        
    avg_mins = sum(peak_minutes) / len(peak_minutes)
    avg_hour = int(avg_mins // 60)
    avg_min = int(avg_mins % 60)
    
    # Format to "X:XX PM ET"
    dummy_dt = datetime.now(et_tz).replace(hour=avg_hour, minute=avg_min)
    return dummy_dt.strftime("%-I:%M %p ET")


async def get_temp_slope(
    session: AsyncSession, city_id: int, hours_back: int = 3
) -> float:
    """Calculate the temperature change (°F) over the last N hours from METAR observations."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours_back)
    
    result = await session.execute(
        select(MetarObs.temp_f, MetarObs.observed_at)
        .where(
            MetarObs.city_id == city_id,
            MetarObs.temp_f.isnot(None),
            MetarObs.observed_at >= since
        )
        .order_by(MetarObs.observed_at)
    )
    rows = result.all()
    if len(rows) < 2:
        return 0.0
    
    oldest_temp = rows[0][0]
    newest_temp = rows[-1][0]
    return float(newest_temp - oldest_temp)


async def get_avg_peak_timing_mins(
    session: AsyncSession, city_id: int, days_back: int = 3, tz: ZoneInfo = ZoneInfo("America/New_York")
) -> float:
    """Return the average minutes-since-midnight that the daily peak was reached over the last N days."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back + 1)
    
    result = await session.execute(
        select(MetarObs)
        .where(
            MetarObs.city_id == city_id,
            MetarObs.temp_f.isnot(None),
            MetarObs.observed_at >= since
        )
    )
    obs_list = list(result.scalars().all())
    if not obs_list:
        return 960.0  # default 4 PM
    
    from collections import defaultdict
    daily_obs = defaultdict(list)
    for obs in obs_list:
        dt_local = obs.observed_at.astimezone(tz)
        daily_obs[dt_local.date()].append(obs)
    
    today_date = datetime.now(tz).date()
    peak_minutes = []
    for d, obs_for_day in daily_obs.items():
        if d == today_date:
            continue
        best = max(obs_for_day, key=lambda o: o.temp_f)
        best_local = best.observed_at.astimezone(tz)
        peak_minutes.append(best_local.hour * 60 + best_local.minute)
    
    if not peak_minutes:
        return 960.0
    return sum(peak_minutes) / len(peak_minutes)


# ─── Station Profiles ────────────────────────────────────────────────────

async def get_station_profile(
    session: AsyncSession, metar_station: str
) -> Optional[StationProfile]:
    result = await session.execute(
        select(StationProfile).where(StationProfile.metar_station == metar_station)
    )
    return result.scalar_one_or_none()


async def upsert_station_profile(
    session: AsyncSession,
    metar_station: str,
    observation_minutes: str,
    observation_frequency: str,
    samples_analyzed: int,
    confidence: float,
) -> StationProfile:
    profile = await get_station_profile(session, metar_station)
    now = datetime.now(timezone.utc)
    if profile is None:
        profile = StationProfile(
            metar_station=metar_station,
            observation_minutes=observation_minutes,
            observation_frequency=observation_frequency,
            samples_analyzed=samples_analyzed,
            confidence=confidence,
            last_analyzed_at=now,
        )
        session.add(profile)
    else:
        profile.observation_minutes = observation_minutes
        profile.observation_frequency = observation_frequency
        profile.samples_analyzed = samples_analyzed
        profile.confidence = confidence
        profile.last_analyzed_at = now
    await session.commit()
    await session.refresh(profile)
    return profile


async def get_resolution_high_metar(
    session: AsyncSession,
    city_id: int,
    date_et: str,
    valid_minutes: list[int],
    tolerance: int = 1,
    city_tz: str = "America/New_York",
) -> Optional[float]:
    """Max temp_f only at valid observation timestamps (±tolerance minutes)."""
    tz = ZoneInfo(city_tz)
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)

    # Expand valid minutes with tolerance
    expanded = set()
    for m in valid_minutes:
        for offset in range(-tolerance, tolerance + 1):
            expanded.add((m + offset) % 60)

    from sqlalchemy import extract
    result = await session.execute(
        select(func.max(MetarObs.temp_f))
        .where(
            MetarObs.city_id == city_id,
            MetarObs.observed_at >= start_dt,
            MetarObs.observed_at < end_dt,
            extract("minute", MetarObs.observed_at).in_(list(expanded)),
        )
    )
    return result.scalar_one_or_none()


async def get_metar_obs_for_station(
    session: AsyncSession,
    metar_station: str,
    since: datetime,
) -> list[MetarObs]:
    """Get METAR observations for a station since a given datetime."""
    result = await session.execute(
        select(MetarObs)
        .where(
            MetarObs.metar_station == metar_station,
            MetarObs.observed_at >= since,
        )
        .order_by(MetarObs.observed_at)
    )
    return list(result.scalars().all())


async def get_todays_metar_obs(
    session: AsyncSession,
    city_id: int,
    date_et: str,
    city_tz: str = "America/New_York",
) -> list[MetarObs]:
    """Return ALL MetarObs for a city on the given local date, ordered by time.

    Used by the adaptive engine to feed the full day's 5-minute observations
    into the Kalman filter and regression.
    """
    tz = ZoneInfo(city_tz)
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)

    result = await session.execute(
        select(MetarObs)
        .where(
            MetarObs.city_id == city_id,
            MetarObs.temp_f.isnot(None),
            MetarObs.observed_at >= start_dt,
            MetarObs.observed_at < end_dt,
        )
        .order_by(MetarObs.observed_at)
    )
    return list(result.scalars().all())


async def get_metar_obs_for_city_since(
    session: AsyncSession,
    city_id: int,
    since: datetime,
) -> list[MetarObs]:
    result = await session.execute(
        select(MetarObs)
        .where(
            MetarObs.city_id == city_id,
            MetarObs.temp_f.isnot(None),
            MetarObs.observed_at >= since,
        )
        .order_by(MetarObs.observed_at)
    )
    return list(result.scalars().all())


# ─── Forecasts ────────────────────────────────────────────────────────────────

async def insert_forecast_obs(session: AsyncSession, **kwargs) -> ForecastObs:
    kwargs["raw_json"] = _compact_forecast_raw_payload(kwargs.get("source"), kwargs.get("raw_json"))
    if not Config.STORE_RAW_FORECAST_PAYLOADS and kwargs.get("parse_error") is not None:
        kwargs["parse_error"] = _bounded_string(kwargs.get("parse_error"), 256)
    obs = ForecastObs(**kwargs)
    session.add(obs)
    await session.commit()
    return obs


async def get_latest_forecast(
    session: AsyncSession, city_id: int, source: str, date_et: str
) -> Optional[ForecastObs]:
    result = await session.execute(
        select(ForecastObs)
        .where(
            ForecastObs.city_id == city_id,
            ForecastObs.source == source,
            ForecastObs.date_et == date_et,
        )
        .order_by(desc(ForecastObs.fetched_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_successful_forecast(
    session: AsyncSession, city_id: int, source: str, date_et: str
) -> Optional[ForecastObs]:
    """Like get_latest_forecast but only returns rows where high_f is not NULL."""
    result = await session.execute(
        select(ForecastObs)
        .where(
            ForecastObs.city_id == city_id,
            ForecastObs.source == source,
            ForecastObs.date_et == date_et,
            ForecastObs.high_f.isnot(None),
        )
        .order_by(desc(ForecastObs.fetched_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_successful_forecasts_bulk(
    session: AsyncSession,
    city_id: int,
    sources: list[str],
    date_et: str,
) -> dict[str, ForecastObs]:
    """Fetch the latest successful ForecastObs for each source in one query."""
    if not sources:
        return {}
    sub = (
        select(
            ForecastObs.source.label("source"),
            func.max(ForecastObs.fetched_at).label("max_ts"),
        )
        .where(
            ForecastObs.city_id == city_id,
            ForecastObs.date_et == date_et,
            ForecastObs.source.in_(sources),
            ForecastObs.high_f.isnot(None),
        )
        .group_by(ForecastObs.source)
        .subquery()
    )
    result = await session.execute(
        select(ForecastObs)
        .join(
            sub,
            (ForecastObs.source == sub.c.source)
            & (ForecastObs.fetched_at == sub.c.max_ts),
        )
        .where(
            ForecastObs.city_id == city_id,
            ForecastObs.date_et == date_et,
            ForecastObs.source.in_(sources),
            ForecastObs.high_f.isnot(None),
        )
    )
    return {row.source: row for row in result.scalars().all()}


async def get_recent_successful_forecasts(
    session: AsyncSession,
    city_id: int,
    source: str,
    limit: int = 10,
    before_or_on_date_et: str | None = None,
) -> list[ForecastObs]:
    q = (
        select(ForecastObs)
        .where(
            ForecastObs.city_id == city_id,
            ForecastObs.source == source,
            ForecastObs.high_f.isnot(None),
        )
        .order_by(desc(ForecastObs.date_et), desc(ForecastObs.fetched_at))
    )
    if before_or_on_date_et:
        q = q.where(ForecastObs.date_et <= before_or_on_date_et)
    result = await session.execute(q.limit(limit))
    return list(result.scalars().all())


# ─── Lead-time skill ──────────────────────────────────────────────────────────

_LEAD_TIME_BUCKETS = (72, 48, 36, 24, 18, 12, 6, 3, 1, 0)


def bucket_lead_time(hours: float) -> int:
    """Snap a continuous lead time (hours) to the nearest tracked bucket."""
    for b in _LEAD_TIME_BUCKETS:
        if hours >= b:
            return b
    return 0


async def get_lead_skills_for_city(
    session: AsyncSession, city_id: int
) -> dict[tuple[str, int], SourceLeadTimeSkill]:
    """All SourceLeadTimeSkill rows for a city, keyed by (source, bucket_hours)."""
    result = await session.execute(
        select(SourceLeadTimeSkill).where(SourceLeadTimeSkill.city_id == city_id)
    )
    return {
        (row.source, row.lead_time_bucket_hours): row
        for row in result.scalars().all()
    }


# ─── Market Snapshots ─────────────────────────────────────────────────────────

async def insert_market_snapshot(
    session: AsyncSession,
    *,
    commit: bool = True,
    **kwargs,
) -> MarketSnapshot:
    snap = MarketSnapshot(**kwargs)
    session.add(snap)
    if commit:
        await session.commit()
        await session.refresh(snap)
    else:
        await session.flush()
    return snap


async def get_latest_market_snapshot(
    session: AsyncSession, bucket_id: int
) -> Optional[MarketSnapshot]:
    result = await session.execute(
        select(MarketSnapshot)
        .where(MarketSnapshot.bucket_id == bucket_id)
        .order_by(desc(MarketSnapshot.fetched_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_market_snapshots_bulk(
    session: AsyncSession, bucket_ids: list[int]
) -> dict[int, MarketSnapshot]:
    """Fetch the most recent MarketSnapshot for each bucket_id in one query."""
    if not bucket_ids:
        return {}
    # Subquery: latest fetched_at per bucket
    sub = (
        select(
            MarketSnapshot.bucket_id,
            func.max(MarketSnapshot.fetched_at).label("max_ts"),
        )
        .where(MarketSnapshot.bucket_id.in_(bucket_ids))
        .group_by(MarketSnapshot.bucket_id)
        .subquery()
    )
    result = await session.execute(
        select(MarketSnapshot).join(
            sub,
            (MarketSnapshot.bucket_id == sub.c.bucket_id)
            & (MarketSnapshot.fetched_at == sub.c.max_ts),
        )
    )
    return {s.bucket_id: s for s in result.scalars().all()}


async def get_market_snapshots_for_bucket(
    session: AsyncSession,
    bucket_id: int,
    since: datetime | None = None,
    limit: int | None = None,
) -> list[MarketSnapshot]:
    q = select(MarketSnapshot).where(MarketSnapshot.bucket_id == bucket_id)
    if since:
        q = q.where(MarketSnapshot.fetched_at >= since)
    q = q.order_by(MarketSnapshot.fetched_at)
    if limit:
        q = q.limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


async def get_market_snapshots_for_event(
    session: AsyncSession,
    event_id: int,
    since: datetime | None = None,
) -> list[MarketSnapshot]:
    q = (
        select(MarketSnapshot)
        .join(Bucket, Bucket.id == MarketSnapshot.bucket_id)
        .where(Bucket.event_id == event_id)
        .order_by(MarketSnapshot.fetched_at)
    )
    if since:
        q = q.where(MarketSnapshot.fetched_at >= since)
    result = await session.execute(q)
    return list(result.scalars().all())


# ─── Shadow Market Flow Features ─────────────────────────────────────────────

async def insert_market_flow_feature(
    session: AsyncSession,
    *,
    commit: bool = True,
    **kwargs,
) -> MarketFlowFeature:
    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if hasattr(MarketFlowFeature, key)
    }
    row = MarketFlowFeature(**clean_kwargs)
    session.add(row)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def get_latest_market_flow_features_bulk(
    session: AsyncSession,
    bucket_ids: list[int],
    *,
    window_minutes: int | None = None,
) -> dict[int, MarketFlowFeature]:
    if not bucket_ids:
        return {}
    filters = [MarketFlowFeature.bucket_id.in_(bucket_ids)]
    if window_minutes is not None:
        filters.append(MarketFlowFeature.window_minutes == window_minutes)
    sub = (
        select(
            MarketFlowFeature.bucket_id,
            func.max(MarketFlowFeature.computed_at).label("max_ts"),
        )
        .where(*filters)
        .group_by(MarketFlowFeature.bucket_id)
        .subquery()
    )
    result = await session.execute(
        select(MarketFlowFeature).join(
            sub,
            (MarketFlowFeature.bucket_id == sub.c.bucket_id)
            & (MarketFlowFeature.computed_at == sub.c.max_ts),
        ).where(*filters)
    )
    return {row.bucket_id: row for row in result.scalars().all()}


# ─── Wallet Stats ─────────────────────────────────────────────────────────────

async def upsert_wallet_stat(
    session: AsyncSession,
    *,
    commit: bool = True,
    **kwargs,
) -> WalletStat:
    """Postgres-safe idempotent upsert for read-only wallet analytics."""
    wallet_address = str(kwargs["wallet_address"]).lower()
    city_slug = str(kwargs["city_slug"])
    condition_id = str(kwargs["condition_id"])
    date_value = str(kwargs["date"])

    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if hasattr(WalletStat, key)
    }
    clean_kwargs.update(
        {
            "wallet_address": wallet_address,
            "city_slug": city_slug,
            "condition_id": condition_id,
            "date": date_value,
        }
    )

    result = await session.execute(
        select(WalletStat).where(
            WalletStat.wallet_address == wallet_address,
            WalletStat.city_slug == city_slug,
            WalletStat.condition_id == condition_id,
            WalletStat.date == date_value,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = WalletStat(**clean_kwargs)
        session.add(row)
    else:
        for key, value in clean_kwargs.items():
            setattr(row, key, value)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def bulk_upsert_wallet_stats(
    session: AsyncSession,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    clean_rows: list[dict[str, Any]] = []
    keys: list[tuple[str, str, str, str]] = []
    for kwargs in rows:
        wallet_address = str(kwargs["wallet_address"]).lower()
        city_slug = str(kwargs["city_slug"])
        condition_id = str(kwargs["condition_id"])
        date_value = str(kwargs["date"])
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if hasattr(WalletStat, key)
        }
        clean_kwargs.update(
            {
                "wallet_address": wallet_address,
                "city_slug": city_slug,
                "condition_id": condition_id,
                "date": date_value,
            }
        )
        clean_rows.append(clean_kwargs)
        keys.append((wallet_address, city_slug, condition_id, date_value))

    existing: dict[tuple[str, str, str, str], WalletStat] = {}
    for chunk in _chunks(keys):
        result = await session.execute(
            select(WalletStat).where(
                tuple_(
                    WalletStat.wallet_address,
                    WalletStat.city_slug,
                    WalletStat.condition_id,
                    WalletStat.date,
                ).in_(chunk)
            )
        )
        for row in result.scalars().all():
            existing[
                (
                    str(row.wallet_address).lower(),
                    str(row.city_slug),
                    str(row.condition_id),
                    str(row.date),
                )
            ] = row

    for clean_kwargs, key in zip(clean_rows, keys):
        row = existing.get(key)
        if row is None:
            session.add(WalletStat(**clean_kwargs))
        else:
            for attr, value in clean_kwargs.items():
                setattr(row, attr, value)
    await session.flush()
    return len(clean_rows)


async def get_wallet_stats_for_city(
    session: AsyncSession,
    city_slug: str,
    date_et: str | None = None,
    limit: int = 100,
) -> list[WalletStat]:
    q = select(WalletStat).where(WalletStat.city_slug == city_slug)
    if date_et:
        q = q.where(WalletStat.date == date_et)
    q = q.order_by(
        WalletStat.consistency_score.desc().nullslast(),
        WalletStat.volume_usd.desc(),
        WalletStat.last_trade_ts.desc().nullslast(),
    ).limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


async def get_wallet_stats_leaderboard(
    session: AsyncSession,
    *,
    city_slug: str | None = None,
    date_et: str | None = None,
    limit: int = 500,
) -> list[WalletStat]:
    q = select(WalletStat)
    if city_slug:
        q = q.where(WalletStat.city_slug == city_slug)
    if date_et:
        q = q.where(WalletStat.date == date_et)
    q = q.order_by(
        WalletStat.consistency_score.desc().nullslast(),
        WalletStat.volume_usd.desc(),
        WalletStat.last_trade_ts.desc().nullslast(),
    ).limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


async def upsert_wallet_trade(
    session: AsyncSession,
    *,
    commit: bool = True,
    **kwargs,
) -> WalletTrade:
    dedupe_key = str(kwargs["dedupe_key"])
    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if hasattr(WalletTrade, key)
    }
    clean_kwargs["wallet_address"] = str(clean_kwargs["wallet_address"]).lower()

    result = await session.execute(
        select(WalletTrade).where(WalletTrade.dedupe_key == dedupe_key)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = WalletTrade(**clean_kwargs)
        session.add(row)
    else:
        for key, value in clean_kwargs.items():
            setattr(row, key, value)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def bulk_upsert_wallet_trades(
    session: AsyncSession,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    clean_rows: list[dict[str, Any]] = []
    keys: list[str] = []
    for kwargs in rows:
        dedupe_key = str(kwargs["dedupe_key"])
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if hasattr(WalletTrade, key)
        }
        clean_kwargs["wallet_address"] = str(clean_kwargs["wallet_address"]).lower()
        clean_rows.append(clean_kwargs)
        keys.append(dedupe_key)

    existing: dict[str, WalletTrade] = {}
    for chunk in _chunks(keys):
        result = await session.execute(
            select(WalletTrade).where(WalletTrade.dedupe_key.in_(chunk))
        )
        for row in result.scalars().all():
            existing[str(row.dedupe_key)] = row

    for clean_kwargs, key in zip(clean_rows, keys):
        row = existing.get(key)
        if row is None:
            session.add(WalletTrade(**clean_kwargs))
        else:
            for attr, value in clean_kwargs.items():
                setattr(row, attr, value)
    await session.flush()
    return len(clean_rows)


async def upsert_wallet_market_exposure(
    session: AsyncSession,
    *,
    commit: bool = True,
    **kwargs,
) -> WalletMarketExposure:
    wallet_address = str(kwargs["wallet_address"]).lower()
    condition_id = str(kwargs["condition_id"])
    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if hasattr(WalletMarketExposure, key)
    }
    clean_kwargs.update({"wallet_address": wallet_address, "condition_id": condition_id})

    result = await session.execute(
        select(WalletMarketExposure).where(
            WalletMarketExposure.wallet_address == wallet_address,
            WalletMarketExposure.condition_id == condition_id,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = WalletMarketExposure(**clean_kwargs)
        session.add(row)
    else:
        for key, value in clean_kwargs.items():
            setattr(row, key, value)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def bulk_upsert_wallet_market_exposures(
    session: AsyncSession,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    clean_rows: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    for kwargs in rows:
        wallet_address = str(kwargs["wallet_address"]).lower()
        condition_id = str(kwargs["condition_id"])
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if hasattr(WalletMarketExposure, key)
        }
        clean_kwargs.update({"wallet_address": wallet_address, "condition_id": condition_id})
        clean_rows.append(clean_kwargs)
        keys.append((wallet_address, condition_id))

    existing: dict[tuple[str, str], WalletMarketExposure] = {}
    for chunk in _chunks(keys):
        result = await session.execute(
            select(WalletMarketExposure).where(
                tuple_(
                    WalletMarketExposure.wallet_address,
                    WalletMarketExposure.condition_id,
                ).in_(chunk)
            )
        )
        for row in result.scalars().all():
            existing[(str(row.wallet_address).lower(), str(row.condition_id))] = row

    for clean_kwargs, key in zip(clean_rows, keys):
        row = existing.get(key)
        if row is None:
            session.add(WalletMarketExposure(**clean_kwargs))
        else:
            for attr, value in clean_kwargs.items():
                setattr(row, attr, value)
    await session.flush()
    return len(clean_rows)


async def upsert_wallet_skill_score(
    session: AsyncSession,
    *,
    commit: bool = True,
    **kwargs,
) -> WalletSkillScore:
    wallet_address = str(kwargs["wallet_address"]).lower()
    scope = str(kwargs["scope"])
    city_slug = str(kwargs.get("city_slug") or "")
    window_days = int(kwargs.get("window_days") or 90)
    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if hasattr(WalletSkillScore, key)
    }
    clean_kwargs.update(
        {
            "wallet_address": wallet_address,
            "scope": scope,
            "city_slug": city_slug,
            "window_days": window_days,
        }
    )

    result = await session.execute(
        select(WalletSkillScore).where(
            WalletSkillScore.wallet_address == wallet_address,
            WalletSkillScore.scope == scope,
            WalletSkillScore.city_slug == city_slug,
            WalletSkillScore.window_days == window_days,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = WalletSkillScore(**clean_kwargs)
        session.add(row)
    else:
        for key, value in clean_kwargs.items():
            setattr(row, key, value)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def bulk_upsert_wallet_skill_scores(
    session: AsyncSession,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    clean_rows: list[dict[str, Any]] = []
    keys: list[tuple[str, str, str, int]] = []
    for kwargs in rows:
        wallet_address = str(kwargs["wallet_address"]).lower()
        scope = str(kwargs["scope"])
        city_slug = str(kwargs.get("city_slug") or "")
        window_days = int(kwargs.get("window_days") or 90)
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if hasattr(WalletSkillScore, key)
        }
        clean_kwargs.update(
            {
                "wallet_address": wallet_address,
                "scope": scope,
                "city_slug": city_slug,
                "window_days": window_days,
            }
        )
        clean_rows.append(clean_kwargs)
        keys.append((wallet_address, scope, city_slug, window_days))

    existing: dict[tuple[str, str, str, int], WalletSkillScore] = {}
    for chunk in _chunks(keys):
        result = await session.execute(
            select(WalletSkillScore).where(
                tuple_(
                    WalletSkillScore.wallet_address,
                    WalletSkillScore.scope,
                    WalletSkillScore.city_slug,
                    WalletSkillScore.window_days,
                ).in_(chunk)
            )
        )
        for row in result.scalars().all():
            existing[
                (
                    str(row.wallet_address).lower(),
                    str(row.scope),
                    str(row.city_slug),
                    int(row.window_days),
                )
            ] = row

    for clean_kwargs, key in zip(clean_rows, keys):
        row = existing.get(key)
        if row is None:
            session.add(WalletSkillScore(**clean_kwargs))
        else:
            for attr, value in clean_kwargs.items():
                setattr(row, attr, value)
    await session.flush()
    return len(clean_rows)


async def get_wallet_skill_scores(
    session: AsyncSession,
    *,
    scope: str,
    city_slug: str = "",
    window_days: int = 90,
    limit: int = 50,
) -> list[WalletSkillScore]:
    q = select(WalletSkillScore).where(
        WalletSkillScore.scope == scope,
        WalletSkillScore.city_slug == city_slug,
        WalletSkillScore.window_days == window_days,
    )
    q = q.order_by(
        WalletSkillScore.adjusted_score.desc(),
        WalletSkillScore.total_volume_usd.desc(),
        WalletSkillScore.last_active_ts.desc().nullslast(),
    ).limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


async def get_wallet_market_exposures_for_event(
    session: AsyncSession,
    city_slug: str,
    date_et: str,
    limit: int = 500,
) -> list[WalletMarketExposure]:
    q = (
        select(WalletMarketExposure)
        .where(
            WalletMarketExposure.city_slug == city_slug,
            WalletMarketExposure.date == date_et,
        )
        .order_by(
            WalletMarketExposure.last_trade_ts.desc().nullslast(),
            WalletMarketExposure.net_notional_usd.desc(),
        )
        .limit(limit)
    )
    result = await session.execute(q)
    return list(result.scalars().all())


# ─── Model Snapshots ──────────────────────────────────────────────────────────

async def insert_model_snapshot(session: AsyncSession, commit: bool = True, **kwargs) -> ModelSnapshot:
    snap = ModelSnapshot(**kwargs)
    session.add(snap)
    if commit:
        await session.commit()
        await session.refresh(snap)
    else:
        await session.flush()
    return snap


async def get_latest_model_snapshot(
    session: AsyncSession, event_id: int
) -> Optional[ModelSnapshot]:
    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.event_id == event_id)
        .order_by(desc(ModelSnapshot.computed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_recent_model_snapshots(
    session: AsyncSession,
    event_id: int,
    limit: int = 10,
) -> list[ModelSnapshot]:
    result = await session.execute(
        select(ModelSnapshot)
        .where(ModelSnapshot.event_id == event_id)
        .order_by(desc(ModelSnapshot.computed_at))
        .limit(limit)
    )
    return list(result.scalars().all())


async def upsert_model_artifact(
    session: AsyncSession,
    *,
    name: str,
    model_bytes: bytes,
    metadata_json: str | None = None,
) -> ModelArtifact:
    result = await session.execute(
        select(ModelArtifact).where(ModelArtifact.name == name)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = ModelArtifact(
            name=name,
            model_bytes=model_bytes,
            metadata_json=metadata_json,
            updated_at=datetime.now(timezone.utc),
        )
        session.add(row)
    else:
        row.model_bytes = model_bytes
        row.metadata_json = metadata_json
        row.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(row)
    return row


async def get_model_artifact(
    session: AsyncSession,
    name: str,
) -> ModelArtifact | None:
    result = await session.execute(
        select(ModelArtifact).where(ModelArtifact.name == name)
    )
    return result.scalar_one_or_none()


# ─── Market Context ───────────────────────────────────────────────────────────

async def get_market_context_snapshot(
    session: AsyncSession,
    city_id: int,
    date_et: str,
) -> Optional[MarketContextSnapshot]:
    result = await session.execute(
        select(MarketContextSnapshot)
        .where(
            MarketContextSnapshot.city_id == city_id,
            MarketContextSnapshot.date_et == date_et,
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def upsert_market_context_snapshot(
    session: AsyncSession,
    city_id: int,
    date_et: str,
    **kwargs,
) -> MarketContextSnapshot:
    snap = await get_market_context_snapshot(session, city_id, date_et)
    if snap is None:
        snap = MarketContextSnapshot(city_id=city_id, date_et=date_et, **kwargs)
        session.add(snap)
    else:
        for k, v in kwargs.items():
            if hasattr(snap, k):
                setattr(snap, k, v)
    await session.commit()
    await session.refresh(snap)
    return snap


# ─── Signals ──────────────────────────────────────────────────────────────────

async def insert_signal(session: AsyncSession, commit: bool = True, **kwargs) -> Signal:
    kwargs["reason_json"] = compact_signal_reason_json(kwargs.get("reason_json"))
    kwargs["gate_failures_json"] = compact_gate_failures_json(kwargs.get("gate_failures_json"))
    sig = Signal(**kwargs)
    session.add(sig)
    if commit:
        await session.commit()
    return sig


async def get_latest_signals(
    session: AsyncSession, limit: int = 50
) -> list[Signal]:
    result = await session.execute(
        select(Signal).order_by(desc(Signal.computed_at)).limit(limit)
    )
    return list(result.scalars().all())


async def get_signals_for_latest_snapshot(
    session: AsyncSession, limit: int = 200, date_et: str | None = None
) -> list[Signal]:
    """Return Signal rows belonging to the **latest model_snapshot per event**.

    The dashboard mixes signals across all enabled cities; without this
    filter, a fresh model rerun briefly lives alongside the previous run's
    rows (any bucket whose new generation hasn't been written yet still
    surfaces its old generation). Filtering on
    `signals.model_snapshot_id == max(model_snapshots.id) per event_id`
    eliminates that staleness.

    Falls back gracefully for legacy Signal rows where model_snapshot_id is
    NULL: those are still returned by computed_at ordering, so the
    dashboard never goes blank during the rollout window.
    """
    # Subquery: latest snapshot id per event
    latest_stmt = select(
        ModelSnapshot.event_id.label("event_id"),
        func.max(ModelSnapshot.id).label("max_id"),
    )
    if date_et:
        latest_stmt = (
            latest_stmt
            .join(Event, Event.id == ModelSnapshot.event_id)
            .where(Event.date_et == date_et)
        )
    latest_per_event = latest_stmt.group_by(ModelSnapshot.event_id).subquery()

    conditions = [
        (Signal.model_snapshot_id == latest_per_event.c.max_id)
        | (Signal.model_snapshot_id.is_(None))
    ]
    if date_et:
        conditions.append(Event.date_et == date_et)

    stmt = (
        select(Signal)
        .join(Bucket, Bucket.id == Signal.bucket_id)
        .join(Event, Event.id == Bucket.event_id)
        .join(
            latest_per_event,
            latest_per_event.c.event_id == Bucket.event_id,
        )
        .where(*conditions)
        .order_by(desc(Signal.computed_at))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_latest_signal_for_bucket(
    session: AsyncSession,
    bucket_id: int,
    snapshot_id: Optional[int] = None,
) -> Optional[Signal]:
    """Most recent Signal for a bucket, optionally pinned to a snapshot.

    Pass `snapshot_id` (e.g. from `get_latest_model_snapshot(event_id)`) to
    guarantee the row belongs to the latest generation. Falls back to the
    plain "latest by computed_at" behavior if no row exists for that
    snapshot (e.g. legacy rows with NULL model_snapshot_id).
    """
    if snapshot_id is not None:
        result = await session.execute(
            select(Signal)
            .where(
                Signal.bucket_id == bucket_id,
                Signal.model_snapshot_id == snapshot_id,
            )
            .order_by(desc(Signal.computed_at))
            .limit(1)
        )
        snap_row = result.scalar_one_or_none()
        if snap_row is not None:
            return snap_row

    result = await session.execute(
        select(Signal)
        .where(Signal.bucket_id == bucket_id)
        .order_by(desc(Signal.computed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_signals_for_buckets(
    session: AsyncSession,
    bucket_ids: list[int],
    snapshot_id: Optional[int] = None,
) -> dict[int, Signal]:
    """Fetch latest Signal rows for bucket_ids, optionally pinned to one snapshot."""
    if not bucket_ids:
        return {}

    async def _fetch(snapshot_filter: Optional[int], ids: list[int]) -> dict[int, Signal]:
        if not ids:
            return {}
        filters = [Signal.bucket_id.in_(ids)]
        if snapshot_filter is not None:
            filters.append(Signal.model_snapshot_id == snapshot_filter)
        sub = (
            select(
                Signal.bucket_id.label("bucket_id"),
                func.max(Signal.computed_at).label("max_ts"),
            )
            .where(*filters)
            .group_by(Signal.bucket_id)
            .subquery()
        )
        result = await session.execute(
            select(Signal)
            .join(
                sub,
                (Signal.bucket_id == sub.c.bucket_id)
                & (Signal.computed_at == sub.c.max_ts),
            )
            .where(*filters)
        )
        return {row.bucket_id: row for row in result.scalars().all()}

    signals = await _fetch(snapshot_id, bucket_ids)
    if snapshot_id is not None:
        missing = [bucket_id for bucket_id in bucket_ids if bucket_id not in signals]
        signals.update(await _fetch(None, missing))
    return signals


# ─── Orders & Fills ───────────────────────────────────────────────────────────

async def insert_order(session: AsyncSession, **kwargs) -> Order:
    order = Order(**kwargs)
    session.add(order)
    await session.commit()
    await session.refresh(order)
    return order


async def update_order_status(
    session: AsyncSession,
    order_id: int,
    status: str,
    **kwargs,
) -> None:
    await session.execute(
        update(Order)
        .where(Order.id == order_id)
        .values(status=status, **kwargs)
    )
    await session.commit()


async def get_open_orders(session: AsyncSession) -> list[Order]:
    result = await session.execute(
        select(Order).where(Order.status.in_(["pending", "open"]))
    )
    return list(result.scalars().all())


async def get_open_sell_orders_for_buckets(
    session: AsyncSession,
    bucket_ids: list[int],
) -> dict[int, list[Order]]:
    if not bucket_ids:
        return {}
    result = await session.execute(
        select(Order)
        .where(
            Order.bucket_id.in_(bucket_ids),
            Order.side.in_(["sell_yes", "sell_no"]),
            Order.status.in_(["pending", "open", "retrying"]),
        )
        .order_by(Order.created_at.desc())
    )
    grouped: dict[int, list[Order]] = {}
    for order in result.scalars().all():
        grouped.setdefault(order.bucket_id, []).append(order)
    return grouped


async def insert_fill(session: AsyncSession, **kwargs) -> Fill:
    fill = Fill(**kwargs)
    session.add(fill)
    await session.commit()
    return fill


async def get_recent_orders(
    session: AsyncSession, limit: int = 50
) -> list[Order]:
    result = await session.execute(
        select(Order).order_by(desc(Order.created_at)).limit(limit)
    )
    return list(result.scalars().all())


# ─── Positions ────────────────────────────────────────────────────────────────

async def get_position(
    session: AsyncSession, bucket_id: int
) -> Optional[Position]:
    result = await session.execute(
        select(Position).where(Position.bucket_id == bucket_id)
    )
    return result.scalar_one_or_none()


async def upsert_position(
    session: AsyncSession,
    bucket_id: int,
    side: str,
    fill_qty: float,
    fill_price: float,
    last_mkt_price: float | None = None,
    entry_type: str | None = None,
    strategy: str | None = None,
    entry_strategy: str | None = None,
    entry_decision_json: str | None = None,
    governing_exit_conditions: str | None = None,
    current_exit_status: str | None = None,
) -> Position:
    """Update position with a fill. fill_qty positive=buy, negative=sell."""
    pos = await get_position(session, bucket_id)
    now = datetime.now(timezone.utc)
    if pos is None:
        pos = Position(
            bucket_id=bucket_id,
            side=side,
            net_qty=max(fill_qty, 0),
            avg_cost=fill_price if fill_qty > 0 else 0.0,
            last_mkt_price=last_mkt_price or fill_price,
            entry_type=entry_type,
            strategy=strategy,
            entry_strategy=entry_strategy,
            entry_decision_json=entry_decision_json,
            governing_exit_conditions=governing_exit_conditions,
            current_exit_status=current_exit_status,
            entry_time=now if fill_qty > 0 else None,
            entry_price=fill_price if fill_qty > 0 else None,
            original_qty=max(fill_qty, 0),
        )
        session.add(pos)
    else:
        old_qty = pos.net_qty
        new_qty = old_qty + fill_qty
        if fill_qty > 0:
            # BUY: weighted average cost
            total_cost = old_qty * pos.avg_cost + fill_qty * fill_price
            pos.avg_cost = total_cost / new_qty if new_qty > 0 else 0.0
        else:
            # SELL: realize PnL
            sold = abs(fill_qty)
            pos.realized_pnl += sold * (fill_price - pos.avg_cost)
        pos.net_qty = max(new_qty, 0)
        
        # If building a larger position, update the entry metadata
        if fill_qty > 0:
            if not pos.entry_time:
                pos.entry_time = now
            if not pos.entry_price:
                pos.entry_price = fill_price
            if entry_type:
                pos.entry_type = entry_type
            if strategy:
                pos.strategy = strategy
            if entry_strategy:
                pos.entry_strategy = entry_strategy
            if entry_decision_json:
                pos.entry_decision_json = entry_decision_json
            if governing_exit_conditions:
                pos.governing_exit_conditions = governing_exit_conditions
            if current_exit_status:
                pos.current_exit_status = current_exit_status
            pos.original_qty = (pos.original_qty or 0.0) + fill_qty

        if last_mkt_price is not None:
            pos.last_mkt_price = last_mkt_price
        if pos.net_qty > 0 and pos.last_mkt_price:
            pos.unrealized_pnl = pos.net_qty * (pos.last_mkt_price - pos.avg_cost)
        else:
            pos.unrealized_pnl = 0.0

    await session.commit()
    await session.refresh(pos)
    return pos


async def get_all_positions(session: AsyncSession) -> list[Position]:
    result = await session.execute(
        select(Position).where(Position.net_qty > 0)
    )
    return list(result.scalars().all())


async def get_daily_realized_pnl(session: AsyncSession, date_et: str) -> float:
    """Sum of realized PnL for positions associated with today's events."""
    # Join Position → Bucket → Event to filter by date
    result = await session.execute(
        select(func.sum(Position.realized_pnl))
        .join(Bucket, Position.bucket_id == Bucket.id)
        .join(Event, Bucket.event_id == Event.id)
        .where(Event.date_et == date_et)
    )
    return float(result.scalar_one_or_none() or 0.0)


# ─── Arming State ─────────────────────────────────────────────────────────────

async def get_arming_state(session: AsyncSession) -> ArmingState:
    """Always returns the singleton arming state (id=1)."""
    result = await session.execute(select(ArmingState).where(ArmingState.id == 1))
    arming = result.scalar_one_or_none()
    if arming is None:
        arming = ArmingState(id=1, state="DISARMED")
        session.add(arming)
        await session.commit()
    return arming


async def update_arming_state(session: AsyncSession, **kwargs) -> ArmingState:
    await session.execute(
        update(ArmingState).where(ArmingState.id == 1).values(**kwargs)
    )
    await session.commit()
    return await get_arming_state(session)


# ─── Audit Log ────────────────────────────────────────────────────────────────

async def append_audit(
    session: AsyncSession,
    actor: str,
    action: str,
    payload: dict | None = None,
    ok: bool = True,
    error_msg: str | None = None,
) -> AuditLog:
    entry = AuditLog(
        actor=actor,
        action=action,
        payload_json=json.dumps(payload) if payload else None,
        ok=ok,
        error_msg=error_msg,
    )
    session.add(entry)
    await session.commit()
    return entry


async def get_audit_log(
    session: AsyncSession, limit: int = 100, action_filter: str | None = None
) -> list[AuditLog]:
    q = select(AuditLog).order_by(desc(AuditLog.ts))
    if action_filter:
        q = q.where(AuditLog.action == action_filter)
    q = q.limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


# ─── Calibration ──────────────────────────────────────────────────────────────

async def get_calibration(
    session: AsyncSession, city_id: int
) -> Optional[CalibrationParams]:
    result = await session.execute(
        select(CalibrationParams).where(CalibrationParams.city_id == city_id)
    )
    return result.scalar_one_or_none()


async def upsert_calibration(
    session: AsyncSession, city_id: int, **kwargs
) -> CalibrationParams:
    cal = await get_calibration(session, city_id)
    if cal is None:
        cal = CalibrationParams(city_id=city_id, **kwargs)
        session.add(cal)
    else:
        for k, v in kwargs.items():
            if hasattr(cal, k):
                setattr(cal, k, v)
    await session.commit()
    await session.refresh(cal)
    return cal


# ─── Worker Heartbeat ─────────────────────────────────────────────────────────

async def update_heartbeat(
    session: AsyncSession,
    job_name: str,
    success: bool = True,
    error: str | None = None,
) -> None:
    result = await session.execute(
        select(WorkerHeartbeat).where(WorkerHeartbeat.job_name == job_name)
    )
    hb = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)

    if hb is None:
        hb = WorkerHeartbeat(job_name=job_name)
        session.add(hb)

    hb.last_run_at = now
    hb.run_count = (hb.run_count or 0) + 1
    if success:
        hb.last_success_at = now
        hb.last_error = None
    else:
        hb.error_count = (hb.error_count or 0) + 1
        hb.last_error = error

    await session.commit()


async def get_all_heartbeats(session: AsyncSession) -> list[WorkerHeartbeat]:
    result = await session.execute(select(WorkerHeartbeat))
    return list(result.scalars().all())


async def get_heartbeat(session: AsyncSession, job_name: str) -> WorkerHeartbeat | None:
    """Get heartbeat for a specific worker job by name."""
    result = await session.execute(
        select(WorkerHeartbeat).where(WorkerHeartbeat.job_name == job_name)
    )
    return result.scalar_one_or_none()


# ─── Resolution / Redemption ─────────────────────────────────────────────────

async def get_unresolved_events_with_positions(session: AsyncSession) -> list[Event]:
    """Events that have open positions but haven't been marked resolved."""
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Event)
        .join(Bucket, Bucket.event_id == Event.id)
        .join(Position, Position.bucket_id == Bucket.id)
        .where(Event.resolved_at.is_(None), Position.net_qty > 0)
        .options(selectinload(Event.buckets))
        .distinct()
    )
    return list(result.scalars().all())


async def get_unresolved_events_with_gamma_id(session: AsyncSession) -> list[Event]:
    """All unresolved events that exist on Polymarket (have gamma_event_id)."""
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Event)
        .where(Event.resolved_at.is_(None), Event.gamma_event_id.isnot(None))
        .options(selectinload(Event.buckets))
    )
    return list(result.scalars().all())


async def get_unredeemed_resolved_events(
    session: AsyncSession,
    require_position: bool = False,
) -> list[Event]:
    """Events that are resolved but not yet redeemed.

    If require_position=True, only return events that have at least one bucket
    with a Position whose net_qty > 0 (i.e. the user actually holds something
    redeemable). This prevents phantom rows for resolved markets the user never
    traded.
    """
    from sqlalchemy.orm import selectinload
    stmt = (
        select(Event)
        .where(Event.resolved_at.isnot(None), Event.redeemed_at.is_(None))
        .options(selectinload(Event.buckets))
    )
    if require_position:
        has_pos = (
            select(Position.id)
            .join(Bucket, Bucket.id == Position.bucket_id)
            .where(Bucket.event_id == Event.id, Position.net_qty > 0)
            .exists()
        )
        stmt = stmt.where(has_pos)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_recently_redeemed_events(session: AsyncSession, days: int = 7) -> list[Event]:
    """Events redeemed within the last N days."""
    from datetime import timedelta
    from sqlalchemy.orm import selectinload
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await session.execute(
        select(Event)
        .where(Event.redeemed_at.isnot(None), Event.redeemed_at >= cutoff)
        .options(selectinload(Event.buckets))
    )
    return list(result.scalars().all())


# ─── Station Calibrations ─────────────────────────────────────────────────────

async def get_station_calibration(
    session: AsyncSession, station_id: str
) -> Optional[StationCalibration]:
    result = await session.execute(
        select(StationCalibration).where(StationCalibration.station_id == station_id)
    )
    return result.scalar_one_or_none()


async def get_station_calibration_by_slug(
    session: AsyncSession, city_slug: str
) -> Optional[StationCalibration]:
    result = await session.execute(
        select(StationCalibration).where(StationCalibration.city_slug == city_slug)
    )
    return result.scalar_one_or_none()


async def get_all_station_calibrations(
    session: AsyncSession,
) -> list[StationCalibration]:
    result = await session.execute(
        select(StationCalibration).order_by(StationCalibration.mae_f)
    )
    return list(result.scalars().all())


async def upsert_station_calibration(
    session: AsyncSession, station_id: str, **kwargs
) -> StationCalibration:
    cal = await get_station_calibration(session, station_id)
    if cal is None:
        cal = StationCalibration(station_id=station_id, **kwargs)
        session.add(cal)
    else:
        for k, v in kwargs.items():
            if hasattr(cal, k):
                setattr(cal, k, v)
    await session.commit()
    await session.refresh(cal)
    return cal


# ─── MADIS HFMETAR ─────────────────────────────────────────────────────────────

async def get_madis_obs_by_key(
    session: AsyncSession, metar_station: str, observed_at: datetime
) -> Optional[MadisObs]:
    """Check if a MadisObs row exists for (station, observed_at)."""
    result = await session.execute(
        select(MadisObs).where(
            MadisObs.metar_station == metar_station,
            MadisObs.observed_at == observed_at,
        )
    )
    return result.scalar_one_or_none()


async def insert_madis_obs(session: AsyncSession, **kwargs) -> MadisObs:
    obs = MadisObs(**kwargs)
    session.add(obs)
    await session.commit()
    await session.refresh(obs)
    return obs


async def get_latest_madis_obs(
    session: AsyncSession, city_id: int
) -> Optional[MadisObs]:
    """Latest MadisObs for a city (for UI benchmarking display)."""
    result = await session.execute(
        select(MadisObs)
        .where(MadisObs.city_id == city_id)
        .order_by(desc(MadisObs.observed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


# ─── Runtime Config ───────────────────────────────────────────────────────────

async def get_runtime_config(session: AsyncSession) -> dict:
    """Return the persisted runtime config overrides dict (may be empty)."""
    result = await session.execute(select(RuntimeConfig).where(RuntimeConfig.id == 1))
    row = result.scalar_one_or_none()
    if row is None or not row.params_json:
        return {}
    try:
        return json.loads(row.params_json)
    except Exception:
        return {}


async def save_runtime_config(session: AsyncSession, params: dict, updated_by: str = "admin") -> None:
    """Upsert the singleton runtime config row with the given params dict."""
    result = await session.execute(select(RuntimeConfig).where(RuntimeConfig.id == 1))
    row = result.scalar_one_or_none()
    if row is None:
        row = RuntimeConfig(id=1, params_json=json.dumps(params), updated_by=updated_by)
        session.add(row)
    else:
        row.params_json = json.dumps(params)
        row.updated_by = updated_by
    await session.commit()
