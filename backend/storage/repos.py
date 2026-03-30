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
    MarketSnapshot,
    MetarObs,
    MetarObsExtended,
    ModelSnapshot,
    Order,
    Position,
    Signal,
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


# ─── Buckets ──────────────────────────────────────────────────────────────────

async def get_buckets_for_event(session: AsyncSession, event_id: int) -> list[Bucket]:
    result = await session.execute(
        select(Bucket)
        .where(Bucket.event_id == event_id)
        .order_by(Bucket.bucket_idx)
    )
    return list(result.scalars().all())


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
    result = await session.execute(
        select(MetarObs)
        .where(MetarObs.city_id == city_id)
        .order_by(desc(MetarObs.observed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_daily_high_metar(
    session: AsyncSession, city_id: int, date_et: str,
    city_tz: str = "America/New_York",
) -> Optional[float]:
    """Max temp_f observed for a city on the given local date."""
    tz = ZoneInfo(city_tz)
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)

    result = await session.execute(
        select(func.max(MetarObs.temp_f))
        .where(
            MetarObs.city_id == city_id,
            MetarObs.observed_at >= start_dt,
            MetarObs.observed_at < end_dt,
        )
    )
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


# ─── Model Snapshots ──────────────────────────────────────────────────────────

async def insert_model_snapshot(session: AsyncSession, **kwargs) -> ModelSnapshot:
    snap = ModelSnapshot(**kwargs)
    session.add(snap)
    await session.commit()
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


async def get_latest_signal_for_bucket(
    session: AsyncSession, bucket_id: int
) -> Optional[Signal]:
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
) -> Position:
    """Update position with a fill. fill_qty positive=buy, negative=sell."""
    pos = await get_position(session, bucket_id)
    if pos is None:
        pos = Position(
            bucket_id=bucket_id,
            side=side,
            net_qty=max(fill_qty, 0),
            avg_cost=fill_price if fill_qty > 0 else 0.0,
            last_mkt_price=last_mkt_price or fill_price,
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


async def get_unredeemed_resolved_events(session: AsyncSession) -> list[Event]:
    """Events that are resolved but not yet redeemed."""
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Event)
        .where(Event.resolved_at.isnot(None), Event.redeemed_at.is_(None))
        .options(selectinload(Event.buckets))
    )
    return list(result.scalars().all())
