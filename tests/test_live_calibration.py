import asyncio
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.storage.db as storage_db
from backend.modeling.live_calibration import (
    apply_threshold_calibration,
    live_bucket_calibration_diagnostics,
    load_live_bucket_diagnostic,
    load_threshold_survival_calibrator,
    ordered_rps,
)
from backend.storage.models import Base, LiveBucketCalibration, ThresholdCalibration


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "live_calibration_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


def _bins(observed_rate: float, count: int = 10) -> str:
    rows = []
    for idx in range(10):
        rows.append({
            "bin": idx,
            "min_prob": idx / 10,
            "max_prob": (idx + 1) / 10,
            "count": count,
            "hits": int(round(observed_rate * count)),
            "predicted_mean": idx / 10 + 0.05,
            "observed_rate": observed_rate,
        })
    return json.dumps(rows)


def test_apply_threshold_calibration_remaps_and_preserves_monotone_survival():
    rows = {
        80.0: type("Row", (), {
            "bins_json": _bins(0.90),
            "n_samples": 50,
            "brier_raw": 0.10,
            "brier_cal": 0.08,
            "rps_raw": 0.20,
            "rps_cal": 0.15,
        })(),
        82.0: type("Row", (), {
            "bins_json": _bins(0.20),
            "n_samples": 50,
            "brier_raw": 0.20,
            "brier_cal": 0.12,
            "rps_raw": 0.30,
            "rps_cal": 0.25,
        })(),
    }

    calibrated, diag = apply_threshold_calibration({80.0: 0.24, 82.0: 0.86}, rows)

    assert diag["applied"] is True
    assert diag["thresholds_applied"] == 2
    assert diag["brier_raw"] == pytest.approx(0.15)
    assert diag["brier_cal"] == pytest.approx(0.10)
    assert diag["rps_raw"] == pytest.approx(0.25)
    assert diag["rps_cal"] == pytest.approx(0.20)
    assert calibrated[80.0] >= calibrated[82.0]
    assert all(0.0 <= p <= 1.0 for p in calibrated.values())


def test_ordered_rps_scores_ordered_bucket_distribution():
    assert ordered_rps([0.0, 1.0, 0.0], 1) == pytest.approx(0.0)
    assert ordered_rps([1.0, 0.0, 0.0], 2) > 0.0


def test_threshold_calibrator_uses_exact_context_when_sample_count_is_enough(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))

    async def run_test():
        async with session_factory() as sess:
            sess.add(ThresholdCalibration(
                city_id=1,
                station_id="KATL",
                hour_bucket=10,
                observed_floor_bucket_idx=2,
                threshold_f=80.0,
                n_samples=50,
                brier_raw=0.20,
                brier_cal=0.15,
                rps_raw=0.10,
                rps_cal=0.08,
                bins_json=_bins(0.85),
                updated_at=datetime.now(timezone.utc),
            ))
            sess.add(ThresholdCalibration(
                city_id=1,
                station_id="",
                hour_bucket=10,
                observed_floor_bucket_idx=-1,
                threshold_f=80.0,
                n_samples=75,
                brier_raw=0.20,
                brier_cal=0.19,
                rps_raw=0.10,
                rps_cal=0.10,
                bins_json=_bins(0.10),
                updated_at=datetime.now(timezone.utc),
            ))
            await sess.commit()

        async with session_factory() as sess:
            calibrator, meta = await load_threshold_survival_calibrator(
                sess,
                city_id=1,
                station_id="KATL",
                hour_bucket=10,
                observed_floor_bucket_idx=2,
            )
            assert calibrator is not None
            assert meta["context_used"] == "city_station_hour_floor"
            calibrated, diag = calibrator({80.0: 0.24})
            assert diag["applied"] is True
            assert calibrated[80.0] > 0.24

    _run(run_test())
    _run(engine.dispose())


def test_live_bucket_diagnostic_applies_only_exact_high_n_context(tmp_path, monkeypatch):
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))

    async def run_test():
        async with session_factory() as sess:
            sess.add(LiveBucketCalibration(
                city_id=1,
                station_id="KATL",
                hour_bucket=12,
                observed_floor_bucket_idx=3,
                bucket_idx=4,
                prob_bin=2,
                n_samples=40,
                hits=20,
                predicted_mean=0.25,
                observed_rate=0.50,
                brier=0.10,
                updated_at=datetime.now(timezone.utc),
            ))
            await sess.commit()

        async with session_factory() as sess:
            diag = await load_live_bucket_diagnostic(
                sess,
                city_id=1,
                station_id="KATL",
                hour_bucket=12,
                observed_floor_bucket_idx=3,
                bucket_idx=4,
                prob=0.24,
            )
            assert diag["applied"] is True
            assert diag["sample_count"] == 40
            assert diag["bucket_calibrated_prob"] > 0.24

        summary = await live_bucket_calibration_diagnostics(city_id=1)
        assert summary["rows"] == 1
        assert summary["eligible_rows_n40"] == 1

    _run(run_test())
    _run(engine.dispose())
