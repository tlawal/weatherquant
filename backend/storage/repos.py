"""
Repository layer — all database queries are here.

Business logic MUST NOT use raw SQL or direct ORM queries.
Every public function returns typed Python objects, never raw Row objects.
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import (
    ArmingState,
    AuditLog,
    Bucket,
    CalibrationParams,
    City,
    Event,
    Fill,
    ForecastObs,
    MarketContextSnapshot,
    MarketSnapshot,
    MetarObs,
    MetarObsExtended,
    ModelSnapshot,
    Order,
    Position,
    Signal,
    StationCalibration,
    StationProfile,
    WorkerHeartbeat,
)


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
) -> Optional[MetarObs]:
    result = await session.execute(
        select(MetarObs).where(
            MetarObs.city_id == city_id,
            MetarObs.metar_station == metar_station,
            MetarObs.observed_at == observed_at,
        ).limit(1)
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
    session: AsyncSession, city_id: int, source: str
) -> Optional[MetarObs]:
    """Latest MetarObs for a city filtered by source (e.g. 'tgftp', 'aviation')."""
    result = await session.execute(
        select(MetarObs)
        .where(MetarObs.city_id == city_id, MetarObs.source == source)
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


# ─── Market Snapshots ─────────────────────────────────────────────────────────

async def insert_market_snapshot(session: AsyncSession, **kwargs) -> MarketSnapshot:
    snap = MarketSnapshot(**kwargs)
    session.add(snap)
    await session.commit()
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


# ─── Model Snapshots ──────────────────────────────────────────────────────────

async def insert_model_snapshot(session: AsyncSession, **kwargs) -> ModelSnapshot:
    snap = ModelSnapshot(**kwargs)
    session.add(snap)
    await session.commit()
    await session.refresh(snap)
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

async def insert_signal(session: AsyncSession, **kwargs) -> Signal:
    sig = Signal(**kwargs)
    session.add(sig)
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
    session: AsyncSession, limit: int = 200
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
    latest_per_event = (
        select(
            ModelSnapshot.event_id.label("event_id"),
            func.max(ModelSnapshot.id).label("max_id"),
        )
        .group_by(ModelSnapshot.event_id)
        .subquery()
    )

    # Join Signal -> Bucket -> Event so we can match snapshots by event
    stmt = (
        select(Signal)
        .join(Bucket, Bucket.id == Signal.bucket_id)
        .join(
            latest_per_event,
            latest_per_event.c.event_id == Bucket.event_id,
        )
        .where(
            (Signal.model_snapshot_id == latest_per_event.c.max_id)
            | (Signal.model_snapshot_id.is_(None))
        )
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
            governing_exit_conditions=governing_exit_conditions,
            current_exit_status=current_exit_status,
            entry_time=now if fill_qty > 0 else None,
            entry_price=fill_price if fill_qty > 0 else None,
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
            if governing_exit_conditions:
                pos.governing_exit_conditions = governing_exit_conditions
            if current_exit_status:
                pos.current_exit_status = current_exit_status

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
