"""Live probability calibration for same-day temperature bucket markets.

This module materializes two calibration layers:

* Threshold survival calibration for P(final high >= threshold).
* Per-bucket/hour/floor reliability diagnostics for displayed bucket probs.

The signal path reads compact materialized rows. The expensive historical
scoring runs on the scheduler and admin diagnostics.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modeling.calibration_engine import resolve_canonical_settlement_high
from backend.modeling.intraday_threshold import (
    bucket_probs_from_survival,
    enforce_monotone_survival,
)
from backend.modeling.settlement import canonical_bucket_ranges, find_bucket_idx_for_value
from backend.storage.db import get_session
from backend.storage.models import (
    Bucket,
    City,
    Event,
    LiveBucketCalibration,
    ModelSnapshot,
    ThresholdCalibration,
)

log = logging.getLogger(__name__)

PROB_BINS = tuple((i / 10.0, (i + 1) / 10.0) for i in range(10))
THRESHOLD_EXACT_MIN_N = 50
THRESHOLD_CITY_HOUR_MIN_N = 75
LIVE_BUCKET_EXACT_MIN_N = 40


@dataclass
class _BinStats:
    count: int = 0
    hits: int = 0
    pred_sum: float = 0.0

    @property
    def pred_mean(self) -> float:
        return self.pred_sum / self.count if self.count else 0.0

    @property
    def observed_rate(self) -> float:
        return self.hits / self.count if self.count else 0.0


@dataclass
class _ThresholdAgg:
    preds: list[float] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)
    bins: dict[int, _BinStats] = field(default_factory=dict)

    def add(self, pred: float, outcome: int) -> None:
        p = _clamp01(pred)
        bin_idx = _prob_bin(p)
        self.preds.append(p)
        self.outcomes.append(1 if outcome else 0)
        stats = self.bins.setdefault(bin_idx, _BinStats())
        stats.count += 1
        stats.hits += 1 if outcome else 0
        stats.pred_sum += p


@dataclass
class _BucketAgg:
    count: int = 0
    hits: int = 0
    pred_sum: float = 0.0
    squared_err_sum: float = 0.0

    def add(self, pred: float, hit: bool) -> None:
        p = _clamp01(pred)
        y = 1.0 if hit else 0.0
        self.count += 1
        self.hits += int(hit)
        self.pred_sum += p
        self.squared_err_sum += (p - y) ** 2


@dataclass
class _SnapshotSample:
    city_id: int
    station_id: str
    hour_bucket: int
    floor_idx: int
    survival: dict[float, float]
    probs: list[float]
    canonical_buckets: list[tuple[Optional[float], Optional[float]]]
    winner_idx: int
    settlement_high: float
    observed_high: Optional[float]


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _prob_bin(prob: float) -> int:
    return max(0, min(9, int(_clamp01(prob) * 10.0)))


def _station_key(station_id: Optional[str]) -> str:
    return (station_id or "").strip().upper()


def _hour_bucket(value: Any) -> int:
    try:
        return max(0, min(23, int(float(value))))
    except (TypeError, ValueError):
        return -1


def _floor_idx(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _brier(preds: list[float], outcomes: list[int]) -> Optional[float]:
    if not preds or len(preds) != len(outcomes):
        return None
    return sum((_clamp01(p) - float(y)) ** 2 for p, y in zip(preds, outcomes)) / len(preds)


def ordered_rps(probs: list[float], winner_idx: int) -> Optional[float]:
    """Ranked probability score for ordered temperature buckets."""
    if not probs or winner_idx < 0 or winner_idx >= len(probs) or len(probs) < 2:
        return None
    total = sum(max(0.0, float(p)) for p in probs)
    if total <= 0:
        return None
    norm_probs = [max(0.0, float(p)) / total for p in probs]
    score = 0.0
    cdf = 0.0
    for idx in range(len(norm_probs) - 1):
        cdf += norm_probs[idx]
        obs_cdf = 1.0 if winner_idx <= idx else 0.0
        score += (cdf - obs_cdf) ** 2
    return score / (len(norm_probs) - 1)


def _bins_payload(agg: _ThresholdAgg) -> list[dict[str, Any]]:
    out = []
    for idx in range(10):
        stats = agg.bins.get(idx, _BinStats())
        lo, hi = PROB_BINS[idx]
        out.append({
            "bin": idx,
            "min_prob": lo,
            "max_prob": hi,
            "count": stats.count,
            "hits": stats.hits,
            "predicted_mean": round(stats.pred_mean, 6) if stats.count else None,
            "observed_rate": round(stats.observed_rate, 6) if stats.count else None,
        })
    return out


def remap_probability_from_bins(prob: float, bins: list[dict[str, Any]], *, min_bin_n: int = 3) -> tuple[float, dict]:
    """Empirical reliability remap with shrinkage toward the raw probability."""
    p = _clamp01(prob)
    idx = _prob_bin(p)
    selected = next((b for b in bins if int(b.get("bin", -1)) == idx), None)
    if not selected:
        return p, {"applied": False, "reason": "missing_bin", "prob_bin": idx}
    n = int(selected.get("count") or 0)
    obs = selected.get("observed_rate")
    if n < min_bin_n or obs is None:
        return p, {"applied": False, "reason": "insufficient_bin_n", "prob_bin": idx, "n": n}
    obs_rate = _clamp01(float(obs))
    weight = min(0.80, n / (n + 20.0))
    calibrated = _clamp01((1.0 - weight) * p + weight * obs_rate)
    return calibrated, {
        "applied": True,
        "prob_bin": idx,
        "n": n,
        "observed_rate": round(obs_rate, 6),
        "weight": round(weight, 4),
        "raw_prob": round(p, 6),
        "calibrated_prob": round(calibrated, 6),
    }


def apply_threshold_calibration(
    survival: dict[float, float],
    threshold_rows: dict[float, ThresholdCalibration],
) -> tuple[dict[float, float], dict[str, Any]]:
    """Apply materialized threshold calibration to a survival map."""
    if not survival or not threshold_rows:
        return survival, {"applied": False, "reason": "no_threshold_rows"}
    calibrated: dict[float, float] = {}
    per_threshold: dict[str, Any] = {}
    applied = 0
    sample_counts = []
    brier_raw_values: list[float] = []
    brier_cal_values: list[float] = []
    rps_raw_values: list[float] = []
    rps_cal_values: list[float] = []
    for threshold, raw_prob in survival.items():
        row = threshold_rows.get(float(threshold))
        if row is None or not row.bins_json:
            calibrated[float(threshold)] = _clamp01(raw_prob)
            continue
        try:
            bins = json.loads(row.bins_json)
        except Exception:
            calibrated[float(threshold)] = _clamp01(raw_prob)
            continue
        new_prob, diag = remap_probability_from_bins(raw_prob, bins)
        calibrated[float(threshold)] = new_prob
        if diag.get("applied"):
            applied += 1
            sample_counts.append(int(row.n_samples or 0))
        if row.brier_raw is not None:
            brier_raw_values.append(float(row.brier_raw))
        if row.brier_cal is not None:
            brier_cal_values.append(float(row.brier_cal))
        if row.rps_raw is not None:
            rps_raw_values.append(float(row.rps_raw))
        if row.rps_cal is not None:
            rps_cal_values.append(float(row.rps_cal))
        per_threshold[str(_fmt_threshold(float(threshold)))] = {
            **diag,
            "threshold_f": float(threshold),
            "row_n": int(row.n_samples or 0),
            "brier_raw": row.brier_raw,
            "brier_cal": row.brier_cal,
            "rps_raw": row.rps_raw,
            "rps_cal": row.rps_cal,
        }
    return enforce_monotone_survival(calibrated), {
        "applied": applied > 0,
        "thresholds_applied": applied,
        "min_sample_count": min(sample_counts) if sample_counts else 0,
        "brier_raw": (
            round(sum(brier_raw_values) / len(brier_raw_values), 6)
            if brier_raw_values else None
        ),
        "brier_cal": (
            round(sum(brier_cal_values) / len(brier_cal_values), 6)
            if brier_cal_values else None
        ),
        "rps_raw": (
            round(sum(rps_raw_values) / len(rps_raw_values), 6)
            if rps_raw_values else None
        ),
        "rps_cal": (
            round(sum(rps_cal_values) / len(rps_cal_values), 6)
            if rps_cal_values else None
        ),
        "per_threshold": per_threshold,
    }


def _fmt_threshold(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def make_threshold_survival_calibrator(
    threshold_rows: dict[float, ThresholdCalibration],
    *,
    context_meta: dict[str, Any],
) -> Callable[[dict[float, float]], tuple[dict[float, float], dict[str, Any]]]:
    def _calibrate(survival: dict[float, float]) -> tuple[dict[float, float], dict[str, Any]]:
        calibrated, diag = apply_threshold_calibration(survival, threshold_rows)
        diag = {**context_meta, **diag}
        return calibrated, diag

    return _calibrate


async def load_threshold_survival_calibrator(
    sess: AsyncSession,
    *,
    city_id: int,
    station_id: Optional[str],
    hour_bucket: int,
    observed_floor_bucket_idx: int,
) -> tuple[Optional[Callable[[dict[float, float]], tuple[dict[float, float], dict[str, Any]]]], dict[str, Any]]:
    """Load the best available threshold calibration backoff context."""
    station = _station_key(station_id)
    candidates = [
        {
            "station_id": station,
            "hour_bucket": int(hour_bucket),
            "observed_floor_bucket_idx": int(observed_floor_bucket_idx),
            "min_n": THRESHOLD_EXACT_MIN_N,
            "level": "city_station_hour_floor",
        },
        {
            "station_id": "",
            "hour_bucket": int(hour_bucket),
            "observed_floor_bucket_idx": int(observed_floor_bucket_idx),
            "min_n": THRESHOLD_EXACT_MIN_N,
            "level": "city_hour_floor",
        },
        {
            "station_id": "",
            "hour_bucket": int(hour_bucket),
            "observed_floor_bucket_idx": -1,
            "min_n": THRESHOLD_CITY_HOUR_MIN_N,
            "level": "city_hour",
        },
    ]
    for candidate in candidates:
        rows = (
            await sess.execute(
                select(ThresholdCalibration).where(
                    ThresholdCalibration.city_id == city_id,
                    ThresholdCalibration.station_id == candidate["station_id"],
                    ThresholdCalibration.hour_bucket == candidate["hour_bucket"],
                    ThresholdCalibration.observed_floor_bucket_idx == candidate["observed_floor_bucket_idx"],
                    ThresholdCalibration.n_samples >= candidate["min_n"],
                )
            )
        ).scalars().all()
        if rows:
            threshold_rows = {float(row.threshold_f): row for row in rows}
            meta = {
                "context_used": candidate["level"],
                "city_id": city_id,
                "station_id": candidate["station_id"],
                "hour_bucket": candidate["hour_bucket"],
                "observed_floor_bucket_idx": candidate["observed_floor_bucket_idx"],
                "min_required_n": candidate["min_n"],
                "threshold_count": len(rows),
                "min_sample_count": min(int(row.n_samples or 0) for row in rows),
                "max_sample_count": max(int(row.n_samples or 0) for row in rows),
            }
            return make_threshold_survival_calibrator(threshold_rows, context_meta=meta), meta
    return None, {
        "context_used": "identity",
        "city_id": city_id,
        "station_id": station,
        "hour_bucket": int(hour_bucket),
        "observed_floor_bucket_idx": int(observed_floor_bucket_idx),
        "applied": False,
        "reason": "no_calibration_context",
    }


async def load_live_bucket_diagnostic(
    sess: AsyncSession,
    *,
    city_id: int,
    station_id: Optional[str],
    hour_bucket: int,
    observed_floor_bucket_idx: int,
    bucket_idx: int,
    prob: float,
) -> dict[str, Any]:
    """Return exact-context per-bucket reliability and optional remap."""
    bin_idx = _prob_bin(prob)
    station = _station_key(station_id)
    row = (
        await sess.execute(
            select(LiveBucketCalibration).where(
                LiveBucketCalibration.city_id == city_id,
                LiveBucketCalibration.station_id == station,
                LiveBucketCalibration.hour_bucket == int(hour_bucket),
                LiveBucketCalibration.observed_floor_bucket_idx == int(observed_floor_bucket_idx),
                LiveBucketCalibration.bucket_idx == int(bucket_idx),
                LiveBucketCalibration.prob_bin == bin_idx,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return {
            "applied": False,
            "reason": "missing_exact_context",
            "raw_model_prob": round(_clamp01(prob), 6),
            "threshold_calibrated_prob": round(_clamp01(prob), 6),
            "prob_bin": bin_idx,
            "sample_count": 0,
        }
    sample_count = int(row.n_samples or 0)
    observed_rate = row.observed_rate
    adjusted = _clamp01(prob)
    applied = False
    if sample_count >= LIVE_BUCKET_EXACT_MIN_N and observed_rate is not None:
        weight = min(0.70, sample_count / (sample_count + 30.0))
        adjusted = _clamp01((1.0 - weight) * adjusted + weight * float(observed_rate))
        applied = True
    return {
        "applied": applied,
        "raw_model_prob": round(_clamp01(prob), 6),
        "threshold_calibrated_prob": round(_clamp01(prob), 6),
        "bucket_calibrated_prob": round(adjusted, 6),
        "bucket_idx": int(bucket_idx),
        "prob_bin": bin_idx,
        "sample_count": sample_count,
        "hits": int(row.hits or 0),
        "empirical_hit_rate": round(float(observed_rate), 6) if observed_rate is not None else None,
        "predicted_mean": round(float(row.predicted_mean), 6) if row.predicted_mean is not None else None,
        "brier": round(float(row.brier), 6) if row.brier is not None else None,
    }


async def _load_samples_for_city(sess: AsyncSession, city: City, *, days_back: int) -> list[_SnapshotSample]:
    city_tz = getattr(city, "tz", "America/New_York") or "America/New_York"
    today_local = datetime.now(ZoneInfo(city_tz)).strftime("%Y-%m-%d")
    cutoff = (datetime.now(ZoneInfo(city_tz)) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    events = (
        await sess.execute(
            select(Event)
            .where(
                Event.city_id == city.id,
                Event.date_et < today_local,
                Event.date_et >= cutoff,
            )
            .order_by(Event.date_et.asc())
        )
    ).scalars().all()
    samples: list[_SnapshotSample] = []
    for event in events:
        settlement = await resolve_canonical_settlement_high(
            sess, city=city, event=event, validate_polymarket_winner=True,
        )
        settlement_high = settlement.get("high_f")
        if settlement_high is None or settlement.get("matches_polymarket_winner") is False:
            continue
        bucket_rows = (
            await sess.execute(
                select(Bucket)
                .where(Bucket.event_id == event.id)
                .order_by(Bucket.bucket_idx)
            )
        ).scalars().all()
        if not bucket_rows:
            continue
        canonical = canonical_bucket_ranges([(b.low_f, b.high_f) for b in bucket_rows])
        winner_idx = find_bucket_idx_for_value(canonical, float(settlement_high))
        if winner_idx is None:
            continue
        snaps = (
            await sess.execute(
                select(ModelSnapshot)
                .where(ModelSnapshot.event_id == event.id)
                .order_by(ModelSnapshot.computed_at.asc())
            )
        ).scalars().all()
        deduped: dict[tuple[int, int], ModelSnapshot] = {}
        for snap in snaps:
            try:
                inputs = json.loads(snap.inputs_json) if snap.inputs_json else {}
                shadow = inputs.get("intraday_threshold_shadow") or {}
                survival_raw = shadow.get("survival") or {}
                probs = json.loads(snap.probs_json) if snap.probs_json else []
            except Exception:
                continue
            if not survival_raw or not probs:
                continue
            hour = _hour_bucket((shadow.get("features") or {}).get("hour_local", inputs.get("hour_local")))
            floor = _floor_idx(inputs.get("observed_bucket_idx"))
            if hour < 0:
                continue
            deduped[(hour, floor)] = snap
        for snap in deduped.values():
            try:
                inputs = json.loads(snap.inputs_json) if snap.inputs_json else {}
                shadow = inputs.get("intraday_threshold_shadow") or {}
                survival = {
                    float(k): _clamp01(v)
                    for k, v in (shadow.get("survival") or {}).items()
                }
                probs = [float(p) for p in json.loads(snap.probs_json)]
            except Exception:
                continue
            if len(probs) != len(canonical) or not survival:
                continue
            samples.append(_SnapshotSample(
                city_id=city.id,
                station_id=_station_key(inputs.get("active_station_id")),
                hour_bucket=_hour_bucket((shadow.get("features") or {}).get("hour_local", inputs.get("hour_local"))),
                floor_idx=_floor_idx(inputs.get("observed_bucket_idx")),
                survival=survival,
                probs=probs,
                canonical_buckets=canonical,
                winner_idx=int(winner_idx),
                settlement_high=float(settlement_high),
                observed_high=(float(inputs["observed_high"]) if inputs.get("observed_high") is not None else None),
            ))
    return samples


def _threshold_contexts(sample: _SnapshotSample) -> list[tuple[str, int, int]]:
    return [
        (sample.station_id, sample.hour_bucket, sample.floor_idx),
        ("", sample.hour_bucket, sample.floor_idx),
        ("", sample.hour_bucket, -1),
    ]


async def refresh_live_calibrations_for_city(city_id: int, *, days_back: int = 90) -> dict[str, Any]:
    """Rebuild threshold and live-bucket calibration materializations for one city."""
    async with get_session() as sess:
        city = await sess.get(City, city_id)
        if city is None:
            return {"city_id": city_id, "status": "missing_city"}
        samples = await _load_samples_for_city(sess, city, days_back=days_back)

        threshold_aggs: dict[tuple[str, int, int, float], _ThresholdAgg] = {}
        rps_context: dict[tuple[str, int, int], dict[str, list[float]]] = {}
        bucket_aggs: dict[tuple[str, int, int, int, int], _BucketAgg] = {}

        for sample in samples:
            for station_id, hour, floor in _threshold_contexts(sample):
                ctx = (station_id, hour, floor)
                raw_rps = ordered_rps(sample.probs, sample.winner_idx)
                if raw_rps is not None:
                    rps_context.setdefault(ctx, {"raw": [], "cal": []})["raw"].append(raw_rps)
                for threshold, pred in sample.survival.items():
                    key = (station_id, hour, floor, float(threshold))
                    agg = threshold_aggs.setdefault(key, _ThresholdAgg())
                    agg.add(pred, int(sample.settlement_high >= float(threshold)))

            for bucket_idx, prob in enumerate(sample.probs):
                key = (
                    sample.station_id,
                    sample.hour_bucket,
                    sample.floor_idx,
                    int(bucket_idx),
                    _prob_bin(prob),
                )
                bucket_aggs.setdefault(key, _BucketAgg()).add(prob, bucket_idx == sample.winner_idx)

        # Compute calibrated RPS once threshold bin maps are known.
        rows_by_context_threshold: dict[tuple[str, int, int, float], list[dict[str, Any]]] = {
            key: _bins_payload(agg) for key, agg in threshold_aggs.items()
        }
        for sample in samples:
            for station_id, hour, floor in _threshold_contexts(sample):
                ctx = (station_id, hour, floor)
                calibrated_survival = {}
                for threshold, pred in sample.survival.items():
                    bins = rows_by_context_threshold.get((station_id, hour, floor, float(threshold)), [])
                    calibrated_survival[threshold] = remap_probability_from_bins(pred, bins)[0]
                calibrated_probs = bucket_probs_from_survival(
                    sample.canonical_buckets,
                    enforce_monotone_survival(calibrated_survival),
                    observed_high=sample.observed_high,
                )
                cal_rps = ordered_rps(calibrated_probs, sample.winner_idx)
                if cal_rps is not None:
                    rps_context.setdefault(ctx, {"raw": [], "cal": []})["cal"].append(cal_rps)

        await sess.execute(delete(ThresholdCalibration).where(ThresholdCalibration.city_id == city_id))
        await sess.execute(delete(LiveBucketCalibration).where(LiveBucketCalibration.city_id == city_id))

        now = datetime.now(timezone.utc)
        threshold_rows = 0
        for (station_id, hour, floor, threshold), agg in threshold_aggs.items():
            bins = _bins_payload(agg)
            cal_preds = [
                remap_probability_from_bins(pred, bins)[0]
                for pred in agg.preds
            ]
            rps = rps_context.get((station_id, hour, floor), {"raw": [], "cal": []})
            sess.add(ThresholdCalibration(
                city_id=city_id,
                station_id=station_id,
                hour_bucket=hour,
                observed_floor_bucket_idx=floor,
                threshold_f=threshold,
                n_samples=len(agg.preds),
                brier_raw=_brier(agg.preds, agg.outcomes),
                brier_cal=_brier(cal_preds, agg.outcomes),
                rps_raw=(sum(rps["raw"]) / len(rps["raw"]) if rps["raw"] else None),
                rps_cal=(sum(rps["cal"]) / len(rps["cal"]) if rps["cal"] else None),
                bins_json=json.dumps(bins),
                updated_at=now,
            ))
            threshold_rows += 1

        bucket_rows = 0
        for (station_id, hour, floor, bucket_idx, prob_bin), agg in bucket_aggs.items():
            if agg.count <= 0:
                continue
            sess.add(LiveBucketCalibration(
                city_id=city_id,
                station_id=station_id,
                hour_bucket=hour,
                observed_floor_bucket_idx=floor,
                bucket_idx=bucket_idx,
                prob_bin=prob_bin,
                n_samples=agg.count,
                hits=agg.hits,
                predicted_mean=agg.pred_sum / agg.count,
                observed_rate=agg.hits / agg.count,
                brier=agg.squared_err_sum / agg.count,
                updated_at=now,
            ))
            bucket_rows += 1
        await sess.commit()

    return {
        "city_id": city_id,
        "city_slug": city.city_slug,
        "samples": len(samples),
        "threshold_rows": threshold_rows,
        "bucket_rows": bucket_rows,
        "days_back": days_back,
    }


async def refresh_all_live_calibrations(*, days_back: int = 90) -> dict[str, Any]:
    async with get_session() as sess:
        cities = (
            await sess.execute(select(City).where(City.enabled.is_(True)).order_by(City.city_slug))
        ).scalars().all()
    summaries = []
    for city in cities:
        try:
            summaries.append(await refresh_live_calibrations_for_city(city.id, days_back=days_back))
        except Exception as exc:
            log.exception("live_calibration: city=%s refresh failed: %s", city.city_slug, exc)
            summaries.append({"city_id": city.id, "city_slug": city.city_slug, "error": str(exc)})
    return {
        "cities": len(cities),
        "summaries": summaries,
        "threshold_rows": sum(int(s.get("threshold_rows") or 0) for s in summaries),
        "bucket_rows": sum(int(s.get("bucket_rows") or 0) for s in summaries),
        "samples": sum(int(s.get("samples") or 0) for s in summaries),
    }


async def threshold_calibration_diagnostics(city_id: Optional[int] = None) -> dict[str, Any]:
    async with get_session() as sess:
        stmt = select(ThresholdCalibration)
        if city_id is not None:
            stmt = stmt.where(ThresholdCalibration.city_id == city_id)
        rows = (await sess.execute(stmt)).scalars().all()
    if not rows:
        return {"rows": 0, "cities": 0, "min_n": None, "max_n": None}
    return {
        "rows": len(rows),
        "cities": len({r.city_id for r in rows}),
        "min_n": min(int(r.n_samples or 0) for r in rows),
        "max_n": max(int(r.n_samples or 0) for r in rows),
        "updated_at": max(r.updated_at for r in rows if r.updated_at).isoformat(),
        "eligible_rows_n50": sum(1 for r in rows if int(r.n_samples or 0) >= THRESHOLD_EXACT_MIN_N),
        "avg_brier_raw": sum((r.brier_raw or 0.0) for r in rows if r.brier_raw is not None) / max(1, sum(1 for r in rows if r.brier_raw is not None)),
        "avg_brier_cal": sum((r.brier_cal or 0.0) for r in rows if r.brier_cal is not None) / max(1, sum(1 for r in rows if r.brier_cal is not None)),
    }


async def live_bucket_calibration_diagnostics(city_id: Optional[int] = None) -> dict[str, Any]:
    async with get_session() as sess:
        stmt = select(LiveBucketCalibration)
        if city_id is not None:
            stmt = stmt.where(LiveBucketCalibration.city_id == city_id)
        rows = (await sess.execute(stmt)).scalars().all()
    if not rows:
        return {"rows": 0, "cities": 0, "min_n": None, "max_n": None}
    return {
        "rows": len(rows),
        "cities": len({r.city_id for r in rows}),
        "min_n": min(int(r.n_samples or 0) for r in rows),
        "max_n": max(int(r.n_samples or 0) for r in rows),
        "updated_at": max(r.updated_at for r in rows if r.updated_at).isoformat(),
        "eligible_rows_n40": sum(1 for r in rows if int(r.n_samples or 0) >= LIVE_BUCKET_EXACT_MIN_N),
        "avg_brier": sum((r.brier or 0.0) for r in rows if r.brier is not None) / max(1, sum(1 for r in rows if r.brier is not None)),
    }
