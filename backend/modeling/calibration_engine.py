"""
Probability calibration engine — analyzes model reliability and remaps probabilities.

This module compares historical model-predicted probabilities against actual outcomes
to identify overconfidence or bias, allowing for re-calibration of live signals.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

from sqlalchemy import extract, select, desc, func
from backend.storage.db import get_session
from backend.storage.models import (
    ModelSnapshot, Event, ForecastObs, Bucket, MetarObs, City, StationProfile,
)
from backend.modeling.settlement import canonical_bucket_ranges, find_bucket_idx_for_value

log = logging.getLogger(__name__)

@dataclass
class ReliabilityBin:
    min_prob: float
    max_prob: float
    count: int = 0
    hits: int = 0
    
    @property
    def expected_prob(self) -> float:
        return (self.min_prob + self.max_prob) / 2
    
    @property
    def observed_prob(self) -> float:
        return self.hits / self.count if self.count > 0 else 0.0

async def get_reliability_metrics(city_id: int, days_back: int = 90) -> List[ReliabilityBin]:
    """
    Compute reliability metrics (observed vs expected probability) for a city.

    Ground truth for the realized daily high is resolved per-event in this order:
      1. MetarObs daily max within the event's local-tz day  (primary — most reliable)
      2. wu_history ForecastObs.high_f for the same date_et  (fallback)

    The previous version gated on `Event.resolved_at IS NOT NULL` (set by the
    Polymarket redeemer). That coupled calibration to trade/settlement activity —
    events without a Polymarket market, or events where the redeemer cron hadn't
    run, were invisible to calibration. We now only require:
      - Event.date_et < today (must be a past day)
      - Event has a ModelSnapshot (have a prediction to score)
      - Event has Bucket rows (bucket boundaries to decide "which bucket won")
      - Ground-truth high exists via MetarObs or wu_history
    """
    # Create bins: 0-10%, 10-20%, ..., 90-100%
    bins = [ReliabilityBin(i/10, (i+1)/10) for i in range(10)]

    today_et = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    async with get_session() as sess:
        # Resolve city timezone once — needed for the MetarObs local-day window.
        city_tz_row = await sess.execute(select(City.tz).where(City.id == city_id))
        city_tz = city_tz_row.scalar_one_or_none() or "America/New_York"
        try:
            tz = ZoneInfo(city_tz)
        except Exception:
            tz = ZoneInfo("America/New_York")

        # Latest ModelSnapshot per event for this city's lookback window only.
        latest_snap_sub = (
            select(
                ModelSnapshot.event_id,
                func.max(ModelSnapshot.id).label("max_snap_id"),
            )
            .join(Event, Event.id == ModelSnapshot.event_id)
            .where(Event.city_id == city_id)
            .where(Event.date_et < today_et)
            .where(Event.date_et >= cutoff)
            .group_by(ModelSnapshot.event_id)
            .subquery()
        )

        # Past events with a snapshot — no resolved_at requirement.
        query = (
            select(ModelSnapshot, Event.id, Event.date_et)
            .join(latest_snap_sub, ModelSnapshot.id == latest_snap_sub.c.max_snap_id)
            .join(Event, ModelSnapshot.event_id == Event.id)
            .where(Event.city_id == city_id)
            .where(Event.date_et < today_et)
            .where(Event.date_et >= cutoff)
            .order_by(desc(Event.date_et))
            .limit(200)
        )
        results = (await sess.execute(query)).all()

        event_ids = list({r[1] for r in results})
        event_buckets: Dict[int, list] = {}
        if event_ids:
            b_rows = (
                await sess.execute(
                    select(Bucket)
                    .where(Bucket.event_id.in_(event_ids))
                    .order_by(Bucket.event_id, Bucket.bucket_idx)
                )
            ).scalars().all()
            for bucket in b_rows:
                event_buckets.setdefault(bucket.event_id, []).append(bucket)

        async def _resolved_high(date_et: str) -> Optional[float]:
            """METAR daily max (primary) or wu_history (fallback)."""
            try:
                start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
            except ValueError:
                return None
            end_dt = start_dt + timedelta(days=1)
            metar_q = await sess.execute(
                select(func.max(MetarObs.temp_f)).where(
                    MetarObs.city_id == city_id,
                    MetarObs.temp_f.isnot(None),
                    MetarObs.observed_at >= start_dt,
                    MetarObs.observed_at < end_dt,
                )
            )
            v = metar_q.scalar_one_or_none()
            if v is not None:
                return float(v)
            wu_q = await sess.execute(
                select(ForecastObs.high_f)
                .where(
                    ForecastObs.city_id == city_id,
                    ForecastObs.date_et == date_et,
                    ForecastObs.source == "wu_history",
                    ForecastObs.high_f.isnot(None),
                )
                .order_by(desc(ForecastObs.id))
                .limit(1)
            )
            v = wu_q.scalar_one_or_none()
            return float(v) if v is not None else None

        if not results:
            log.debug("calibration: city_id=%d no past events with snapshots in last %dd",
                      city_id, days_back)
            return bins

        matched = 0
        for snap, event_id, date_et in results:
            buckets = event_buckets.get(event_id, [])
            if not buckets:
                continue
            rh = await _resolved_high(date_et)
            if rh is None:
                continue
            try:
                probs = json.loads(snap.probs_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if len(probs) != len(buckets):
                continue
            canonical = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
            hit = find_bucket_idx_for_value(canonical, rh)
            if hit is None:
                continue
            matched += 1
            for i, p in enumerate(probs):
                bin_idx = min(int(p * 10), 9)
                bins[bin_idx].count += 1
                if i == hit:
                    bins[bin_idx].hits += 1

        total = sum(b.count for b in bins)
        log.info(
            "calibration: city_id=%d events_in_window=%d matched=%d total_samples=%d bins_with_data=%d",
            city_id, len(results), matched, total, sum(1 for b in bins if b.count > 0),
        )
    return bins

async def get_reliability_diagnostics(city_id: int) -> dict:
    """
    Report exactly which calibration filter is short-handed for this city.

    Filters now (post-decoupling):
      - Event.date_et < today  AND  ModelSnapshot exists
      - Event has Bucket rows
      - MetarObs daily high OR wu_history exists for the event date

    `resolved_events` / `last_resolved_at` are retained as INFORMATIONAL only
    — they track redeemer activity but no longer gate calibration.
    """
    today_et = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async with get_session() as sess:
        total_events = (await sess.execute(
            select(func.count(Event.id)).where(Event.city_id == city_id)
        )).scalar_one()

        past_events = (await sess.execute(
            select(func.count(Event.id)).where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

        # Past events with at least one ModelSnapshot.
        past_with_snap = (await sess.execute(
            select(func.count(func.distinct(Event.id)))
            .join(ModelSnapshot, ModelSnapshot.event_id == Event.id)
            .where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

        # Past events with at least one Bucket row (bucket boundaries required).
        past_with_buckets = (await sess.execute(
            select(func.count(func.distinct(Event.id)))
            .join(Bucket, Bucket.event_id == Event.id)
            .where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

        # Past event dates with a MetarObs row. Approximation: match by
        # UTC-date string of observed_at against date_et. Not perfectly TZ-aware
        # but close enough for a diagnostic count (the actual calibration uses
        # the proper city-tz window).
        # For cross-dialect safety we pull distinct event dates + distinct
        # observed-at dates and intersect in Python.
        past_event_dates_rows = (await sess.execute(
            select(func.distinct(Event.date_et))
            .where(Event.city_id == city_id, Event.date_et < today_et)
        )).all()
        past_event_dates = {r[0] for r in past_event_dates_rows}

        # Pull raw observed_at timestamps where temp_f is not null, bucket into
        # YYYY-MM-DD in Python. UTC-day approximation is fine for a diagnostic
        # count (true calibration resolves via the city-tz window).
        metar_times = (await sess.execute(
            select(MetarObs.observed_at).where(
                MetarObs.city_id == city_id,
                MetarObs.temp_f.isnot(None),
            )
        )).scalars().all()
        metar_days = {t.strftime("%Y-%m-%d") for t in metar_times if t is not None}
        past_dates_with_metar_high = len(past_event_dates & metar_days)

        with_wu = (await sess.execute(
            select(func.count(func.distinct(ForecastObs.date_et))).where(
                ForecastObs.city_id == city_id,
                ForecastObs.source == "wu_history",
                ForecastObs.high_f.isnot(None),
            )
        )).scalar_one()

        resolved = (await sess.execute(
            select(func.count(Event.id)).where(
                Event.city_id == city_id,
                Event.date_et < today_et,
                Event.resolved_at.isnot(None),
            )
        )).scalar_one()

        last_resolved = (await sess.execute(
            select(func.max(Event.resolved_at)).where(Event.city_id == city_id)
        )).scalar_one()

        last_event_date = (await sess.execute(
            select(func.max(Event.date_et)).where(
                Event.city_id == city_id,
                Event.date_et < today_et,
            )
        )).scalar_one()

    return {
        "total_events": int(total_events or 0),
        "past_events": int(past_events or 0),
        "past_events_with_snapshot": int(past_with_snap or 0),
        "past_events_with_buckets": int(past_with_buckets or 0),
        "past_dates_with_metar_high": int(past_dates_with_metar_high or 0),
        "dates_with_wu_history": int(with_wu or 0),
        "resolved_events": int(resolved or 0),  # informational
        "last_resolved_at": last_resolved.isoformat() if last_resolved else None,
        "last_event_date": last_event_date if last_event_date else None,
        # back-compat alias so older template copies don't KeyError during rolling deploy
        "events_with_snapshot": int(past_with_snap or 0),
    }


def remap_probability(prob: float, bins: List[ReliabilityBin]) -> float:
    """Adjust a raw model probability using isotonic regression calibration.

    Isotonic regression (Gneiting & Raftery 2007) learns a monotone
    non-parametric mapping from predicted → observed probabilities.
    It works well with small samples and guarantees reliability
    (calibrated predictions are monotonically related to raw ones).

    Falls back to identity (no correction) when < 15 total samples.
    """
    if not bins or all(b.count == 0 for b in bins):
        return prob

    # Build training data from reliability bins
    xs: List[float] = []  # predicted probabilities (bin centers)
    ys: List[float] = []  # observed probabilities
    ws: List[float] = []  # sample weights
    total_samples = 0

    for b in bins:
        if b.count >= 2:  # need at least 2 samples per bin to be meaningful
            xs.append(b.expected_prob)
            ys.append(b.observed_prob)
            ws.append(float(b.count))
            total_samples += b.count

    if total_samples < 15 or len(xs) < 3:
        return prob  # insufficient data — identity pass-through

    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(
            y_min=0.0, y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        ir.fit(xs, ys, sample_weight=ws)

        calibrated = float(ir.predict([prob])[0])

        # Dampen the correction — blend 70% calibrated + 30% raw
        # to avoid overfitting on limited historical data
        dampened = 0.70 * calibrated + 0.30 * prob
        return max(0.0, min(1.0, dampened))
    except Exception:
        # sklearn not available or fit failed — fall back to identity
        log.debug("isotonic calibration failed, using raw probability", exc_info=True)
        return prob


# ─── Lead-Time Skill Analysis ─────────────────────────────────────────────────

_LEAD_TIME_BUCKETS = [72, 48, 36, 24, 18, 12, 6, 3, 1, 0]
FORECAST_SKILL_SOURCES = frozenset({
    "nws",
    "wu_hourly",
    "open_meteo",
    "hrrr",
    "hrrr_15min",
    "nbm",
    "ecmwf_ifs",
    "ecmwf_aifs",
    "gfs_graphcast",
    "pangu_weather",
    "fourcastnet_v2",
    "aurora",
})


def _bucket_lead_time(hours: float) -> int:
    """Round lead time to the nearest bucket."""
    for bucket in _LEAD_TIME_BUCKETS:
        if hours >= bucket:
            return bucket
    return 0


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _event_end_utc(date_et: str, city_tz: str) -> datetime:
    tz = ZoneInfo(city_tz or "America/New_York")
    local = datetime.strptime(date_et, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=tz,
    )
    return local.astimezone(timezone.utc)


async def _station_observation_minutes(
    sess,
    station_id: Optional[str],
) -> Optional[list[int]]:
    if not station_id:
        return None
    row = (
        await sess.execute(
            select(StationProfile).where(
                StationProfile.metar_station == station_id.upper(),
            )
        )
    ).scalar_one_or_none()
    if row is None or not row.observation_minutes:
        return None
    try:
        vals = json.loads(row.observation_minutes)
    except Exception:
        return None
    out = []
    for v in vals or []:
        try:
            out.append(int(v) % 60)
        except (TypeError, ValueError):
            continue
    return out or None


async def _station_high_for_date(
    sess,
    *,
    city_id: int,
    date_et: str,
    city_tz: str,
    station_id: Optional[str],
    valid_minutes: Optional[list[int]] = None,
    tolerance: int = 1,
) -> Optional[float]:
    if not station_id:
        return None
    tz = ZoneInfo(city_tz or "America/New_York")
    start_dt = datetime.strptime(date_et, "%Y-%m-%d").replace(tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)
    stmt = select(func.max(MetarObs.temp_f)).where(
        MetarObs.city_id == city_id,
        MetarObs.metar_station == station_id.upper(),
        MetarObs.temp_f.isnot(None),
        MetarObs.observed_at >= start_dt,
        MetarObs.observed_at < end_dt,
    )
    if valid_minutes:
        expanded = {
            (m + offset) % 60
            for m in valid_minutes
            for offset in range(-tolerance, tolerance + 1)
        }
        stmt = stmt.where(extract("minute", MetarObs.observed_at).in_(list(expanded)))
    val = (await sess.execute(stmt)).scalar_one_or_none()
    return float(val) if val is not None else None


async def _wu_history_high_for_date(
    sess,
    *,
    city_id: int,
    date_et: str,
) -> Optional[float]:
    row = (
        await sess.execute(
            select(ForecastObs)
            .where(
                ForecastObs.city_id == city_id,
                ForecastObs.source == "wu_history",
                ForecastObs.date_et == date_et,
                ForecastObs.high_f.isnot(None),
            )
            .order_by(desc(ForecastObs.fetched_at), desc(ForecastObs.id))
            .limit(1)
        )
    ).scalar_one_or_none()
    return float(row.high_f) if row and row.high_f is not None else None


async def resolve_canonical_settlement_high(
    sess,
    *,
    city: City,
    event: Event,
    validate_polymarket_winner: bool = True,
) -> dict:
    """Resolve the continuous high used for source MAE/bias scoring.

    This mirrors market settlement as closely as local data allows: prefer the
    event's explicit resolution station, respect known station observation
    minutes, and use WU history only when station data is unavailable. When
    Polymarket's categorical winner exists, use it as a validation check rather
    than converting bucket ranges into fake continuous temperatures.
    """
    city_tz = getattr(city, "tz", "America/New_York") or "America/New_York"
    station_id = (
        getattr(event, "resolution_station_id", None)
        or getattr(city, "metar_station", None)
    )
    station_id = station_id.upper() if station_id else None
    valid_minutes = await _station_observation_minutes(sess, station_id)

    high_f = None
    source_used = None
    if valid_minutes:
        high_f = await _station_high_for_date(
            sess,
            city_id=city.id,
            date_et=event.date_et,
            city_tz=city_tz,
            station_id=station_id,
            valid_minutes=valid_minutes,
        )
        source_used = "resolution_metar" if high_f is not None else None

    if high_f is None and not valid_minutes:
        high_f = await _station_high_for_date(
            sess,
            city_id=city.id,
            date_et=event.date_et,
            city_tz=city_tz,
            station_id=station_id,
        )
        source_used = "station_metar" if high_f is not None else None

    if high_f is None:
        high_f = await _wu_history_high_for_date(
            sess, city_id=city.id, date_et=event.date_et,
        )
        source_used = "wu_history" if high_f is not None else None

    out = {
        "high_f": high_f,
        "source_used": source_used,
        "station_id": station_id,
        "valid_minutes": valid_minutes,
        "derived_bucket_idx": None,
        "winner_bucket_idx": event.winning_bucket_idx,
        "matches_polymarket_winner": None,
    }
    if (
        high_f is not None
        and validate_polymarket_winner
        and event.winning_bucket_idx is not None
    ):
        buckets = (
            await sess.execute(
                select(Bucket)
                .where(Bucket.event_id == event.id)
                .order_by(Bucket.bucket_idx)
            )
        ).scalars().all()
        if buckets:
            ranges = canonical_bucket_ranges([(b.low_f, b.high_f) for b in buckets])
            derived_idx = find_bucket_idx_for_value(ranges, float(high_f))
            out["derived_bucket_idx"] = derived_idx
            out["matches_polymarket_winner"] = (
                derived_idx == event.winning_bucket_idx
            )
    return out


def select_latest_forecasts_by_checkpoint(
    forecasts: list[ForecastObs],
    checkpoint_utc: datetime,
) -> dict[str, ForecastObs]:
    """Pick one available forecast per source at a fixed decision checkpoint."""
    checkpoint = _as_utc(checkpoint_utc) or checkpoint_utc
    selected: dict[str, ForecastObs] = {}

    def sort_key(fc: ForecastObs) -> tuple[datetime, datetime, int]:
        fetched = (
            _as_utc(getattr(fc, "fetched_at", None))
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        model_run = (
            _as_utc(getattr(fc, "model_run_at", None))
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        return (fetched, model_run, int(getattr(fc, "id", 0) or 0))

    for fc in forecasts:
        if fc.high_f is None or fc.source not in FORECAST_SKILL_SOURCES:
            continue
        fetched_at = _as_utc(getattr(fc, "fetched_at", None))
        model_run_at = _as_utc(getattr(fc, "model_run_at", None))
        if fetched_at is None or fetched_at > checkpoint:
            continue
        if model_run_at is not None and model_run_at > checkpoint:
            continue
        prev = selected.get(fc.source)
        if prev is None or sort_key(fc) > sort_key(prev):
            selected[fc.source] = fc
    return selected


async def compute_source_lead_time_skills(
    city_id: int,
    days_back: int = 90,
    min_obs_per_bucket: int = 1,
    return_diagnostics: bool = False,
):
    """Compute MAE and bias per forecast source at each lead-time bucket.

    Scores fixed decision checkpoints before settlement. For each
    (date, checkpoint, source), exactly one forecast is scored: the latest row
    available by that checkpoint. This keeps frequently refreshed sources
    valuable live without letting refresh cadence inflate historical sample
    size.

    Args:
        return_diagnostics: when True, returns a dict shaped for the admin
            recompute endpoint (counts + reason-codes for empty buckets) so
            the operator can see *why* the table isn't populating. When
            False (default, scheduler path), returns the skill map for
            backwards compat.

    Returns:
        - When `return_diagnostics=False` (default): `dict` keyed by
          (source, lead_time_bucket) with {mae, bias, n}.
        - When `return_diagnostics=True`: a flat dict with
          `events_resolved`, `events_with_settlement`,
          `forecast_obs_with_model_run_at`, `source_bucket_combos_attempted`,
          `source_bucket_combos_below_min_n`, `source_bucket_combos_written`,
          `missing_reasons` (per "<source>:<bucket>" key explaining why empty),
          and `skills` (the same map as the non-diagnostic return).
    """
    # Counters for diagnostic-mode return — incremented as we walk the data.
    diag = {
        "events_resolved": 0,
        "events_past": 0,
        "events_with_settlement": 0,
        "events_without_settlement": 0,
        "events_scored": 0,
        "settlement_mismatches": 0,
        "forecast_obs_with_model_run_at": 0,
        "forecast_obs_candidates": 0,
        "forecast_obs_scored": 0,
        "source_bucket_combos_attempted": 0,
        "source_bucket_combos_below_min_n": 0,
        "source_bucket_combos_written": 0,
        "missing_reasons": {},
        "settlement_mismatch_details": [],
        "skills": {},
    }

    def _wrap(skills_map: dict):
        """Pick the right return shape based on return_diagnostics."""
        if return_diagnostics:
            diag["skills"] = {
                f"{src}:{bucket}h": v
                for (src, bucket), v in skills_map.items()
            }
            return diag
        return skills_map

    async with get_session() as sess:
        from backend.storage.models import SourceLeadTimeSkill

        city = await sess.get(City, city_id)
        if city is None:
            if return_diagnostics:
                diag["missing_reasons"]["city"] = "not_found"
            return _wrap({})

        city_tz = getattr(city, "tz", "America/New_York") or "America/New_York"
        today_local = datetime.now(ZoneInfo(city_tz)).strftime("%Y-%m-%d")
        cutoff = (
            datetime.now(ZoneInfo(city_tz)) - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        diag["events_resolved"] = int((await sess.execute(
            select(func.count(Event.id)).where(
                Event.city_id == city_id,
                Event.date_et < today_local,
                Event.date_et >= cutoff,
                Event.winning_bucket_idx.isnot(None),
            )
        )).scalar_one() or 0)

        # Past events are scoreable even before Gamma/Polymarket has populated
        # winning_bucket_idx; the continuous target comes from settlement obs.
        event_stmt = (
            select(Event)
            .where(
                Event.city_id == city_id,
                Event.date_et < today_local,
                Event.date_et >= cutoff,
            )
            .order_by(Event.date_et.asc())
        )
        events = (await sess.execute(event_stmt)).scalars().all()
        diag["events_past"] = len(events)

        if not events:
            log.info("lead_time_skill: no past events for city_id=%s in last %d days", city_id, days_back)
            return _wrap({})

        results: dict = {}
        for event in events:
            settlement = await resolve_canonical_settlement_high(
                sess, city=city, event=event, validate_polymarket_winner=True,
            )
            obs_high = settlement.get("high_f")
            if obs_high is None:
                diag["events_without_settlement"] += 1
                continue
            diag["events_with_settlement"] += 1
            if settlement.get("matches_polymarket_winner") is False:
                diag["settlement_mismatches"] += 1
                if return_diagnostics:
                    diag["settlement_mismatch_details"].append({
                        "date_et": event.date_et,
                        "high_f": round(float(obs_high), 2),
                        "derived_bucket_idx": settlement.get("derived_bucket_idx"),
                        "winner_bucket_idx": settlement.get("winner_bucket_idx"),
                        "source_used": settlement.get("source_used"),
                    })
                continue

            fc_stmt = (
                select(ForecastObs)
                .where(
                    ForecastObs.city_id == city_id,
                    ForecastObs.date_et == event.date_et,
                    ForecastObs.source.in_(FORECAST_SKILL_SOURCES),
                    ForecastObs.high_f.isnot(None),
                )
            )
            forecasts = (await sess.execute(fc_stmt)).scalars().all()
            diag["forecast_obs_candidates"] += len(forecasts)
            diag["forecast_obs_with_model_run_at"] += sum(
                1 for fc in forecasts if fc.model_run_at is not None
            )
            if not forecasts:
                continue

            event_end_utc = _event_end_utc(event.date_et, city_tz)
            event_scored = False
            for bucket in _LEAD_TIME_BUCKETS:
                checkpoint_utc = event_end_utc - timedelta(hours=bucket)
                selected = select_latest_forecasts_by_checkpoint(
                    list(forecasts), checkpoint_utc,
                )
                for source, fc in selected.items():
                    key = (source, bucket)
                    if key not in results:
                        results[key] = {"errors": [], "dates": set()}
                    results[key]["errors"].append(float(fc.high_f) - float(obs_high))
                    results[key]["dates"].add(event.date_et)
                    diag["forecast_obs_scored"] += 1
                    event_scored = True
            if event_scored:
                diag["events_scored"] += 1

        diag["source_bucket_combos_attempted"] = len(results)

        # Compute MAE and bias per bucket
        skills = {}
        for (source, bucket), data in results.items():
            n_obs = len(data["dates"])
            if n_obs < min_obs_per_bucket:
                diag["source_bucket_combos_below_min_n"] += 1
                if return_diagnostics:
                    diag["missing_reasons"][f"{source}:{bucket}h"] = (
                        f"n={n_obs}<{min_obs_per_bucket}"
                    )
                continue
            errors = data["errors"]
            mae = sum(abs(e) for e in errors) / len(errors)
            bias = sum(errors) / len(errors)
            skills[(source, bucket)] = {
                "source": source,
                "lead_time_bucket_hours": bucket,
                "mae_f": round(mae, 2),
                "bias_f": round(bias, 2),
                "n_obs": n_obs,
            }

            # Upsert to database
            existing = await sess.execute(
                select(SourceLeadTimeSkill).where(
                    SourceLeadTimeSkill.city_id == city_id,
                    SourceLeadTimeSkill.source == source,
                    SourceLeadTimeSkill.lead_time_bucket_hours == bucket,
                )
            )
            existing = existing.scalar_one_or_none()

            if existing:
                existing.mae_f = round(mae, 2)
                existing.bias_f = round(bias, 2)
                existing.n_obs = n_obs
                existing.computed_at = datetime.now(timezone.utc)
            else:
                skill = SourceLeadTimeSkill(
                    city_id=city_id,
                    source=source,
                    lead_time_bucket_hours=bucket,
                    mae_f=round(mae, 2),
                    bias_f=round(bias, 2),
                    n_obs=n_obs,
                )
                sess.add(skill)

        diag["source_bucket_combos_written"] = len(skills)

        await sess.commit()

        log.info(
            "lead_time_skill: city_id=%s computed %d source/bucket combos from %d past events",
            city_id, len(skills), len(events),
        )
        return _wrap(skills)
