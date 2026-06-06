"""
ML Trainer — extracts historical METAR observations from the app database,
engineers features (temp slope, peak timing persistence, seasonality),
and trains a GradientBoostingRegressor to predict remaining temperature rise.

Usage:
    python -m backend.modeling.ml_trainer
"""
import json
import logging
import os
import sqlite3
import asyncio

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from dotenv import load_dotenv

from zoneinfo import ZoneInfo
from backend.modeling.residual_paths import (
    residual_metadata_path,
    residual_model_path,
    residual_shadow_metadata_path,
    residual_shadow_model_path,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("ml_trainer")

ROOT_DIR = Path(__file__).parent.parent.parent


def _sqlite_path_from_url(url: str) -> Path | None:
    if not url:
        return None
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if not url.startswith(prefix):
            continue
        raw_path = url[len(prefix):]
        if not raw_path or raw_path == ":memory:":
            return None
        path = Path(raw_path)
        return path if path.is_absolute() else ROOT_DIR / path
    return None


def _resolve_db_path() -> Path:
    explicit_path = os.environ.get("ML_TRAINER_DB_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path).expanduser()

    db_url_path = _sqlite_path_from_url(os.environ.get("DATABASE_URL", "").strip())
    if db_url_path is not None:
        return db_url_path

    candidates = [
        ROOT_DIR / "data" / "state.db",
        ROOT_DIR / "data" / "weatherquant.db",
        ROOT_DIR / "state.db",
        ROOT_DIR / "weatherquant.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DB_PATH = _resolve_db_path()
MODEL_PATH = residual_model_path()
METADATA_PATH = residual_metadata_path()
SHADOW_MODEL_PATH = residual_shadow_model_path()
SHADOW_METADATA_PATH = residual_shadow_metadata_path()
PROMOTE_RESIDUAL_ML = os.environ.get("PROMOTE_RESIDUAL_ML", "").strip().lower() in {
    "1", "true", "yes", "on"
}
MIN_PROMOTION_MAE_IMPROVEMENT_F = 0.20
MIN_RAIN_SUBSET_IMPROVEMENT_F = 0.15
MAX_RAIN_SUBSET_WORSE_F = 0.05
RAIN_SUBSET_MIN_N = 30

# Static baseline for comparison (the old _REMAINING_RISE_TABLE)
_OLD_TABLE = [
    (0, 6, 14.0), (6, 9, 11.0), (9, 11, 8.0), (11, 13, 4.0),
    (13, 15, 2.0), (15, 17, 0.5), (17, 19, 0.0), (19, 24, 0.0),
]

def _old_remaining_rise(hour: int) -> float:
    for s, e, v in _OLD_TABLE:
        if s <= hour < e:
            return v
    return 0.0


def _cloud_cover_value(cover) -> float:
    cover_map = {"CLR": 0, "SKC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}
    if cover is None:
        return 0.0
    return float(cover_map.get(str(cover).strip().upper(), 0))


def _has_precip(wx_string=None, condition=None, precip_in=None) -> float:
    try:
        if precip_in is not None and float(precip_in) > 0:
            return 1.0
    except (TypeError, ValueError):
        pass
    haystack = f"{wx_string or ''} {condition or ''}".upper()
    tokens = ("RA", "TS", "SH", "SN", "DZ", "RAIN", "STORM", "SNOW", "DRIZZLE")
    return 1.0 if any(tok in haystack for tok in tokens) else 0.0


def _fill_numeric(df: pd.DataFrame, column: str, default: float) -> None:
    if column not in df.columns:
        df[column] = default
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)


def _add_recent_precip_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city_id", "observed_at"]).copy()
    df["precip_recent_3h"] = 0.0
    for _city_id, idx in df.groupby("city_id").groups.items():
        group = df.loc[idx].sort_values("observed_at")
        rolled = (
            group.set_index("observed_at")["precip_flag"]
            .rolling("3h", min_periods=1)
            .max()
            .to_numpy()
        )
        df.loc[group.index, "precip_recent_3h"] = rolled
    return df


async def _load_metar_dataframe_from_app_db() -> pd.DataFrame:
    """Load METAR rows through the app DB layer for Railway/Postgres."""
    from sqlalchemy import select

    from backend.storage.db import get_session, init_db
    from backend.storage.models import City, MetarObs, MetarObsExtended

    await init_db()
    async with get_session() as sess:
        result = await sess.execute(
            select(
                MetarObs.city_id,
                MetarObs.observed_at,
                MetarObs.temp_f,
                City.city_slug,
                City.tz,
                MetarObsExtended.dewpoint_f,
                MetarObsExtended.humidity_pct,
                MetarObsExtended.wind_speed_kt,
                MetarObsExtended.wind_gust_kt,
                MetarObsExtended.altimeter_inhg,
                MetarObsExtended.precip_in,
                MetarObsExtended.cloud_cover,
                MetarObsExtended.wx_string,
                MetarObsExtended.condition,
            )
            .join(City, MetarObs.city_id == City.id)
            .outerjoin(MetarObsExtended, MetarObsExtended.metar_obs_id == MetarObs.id)
            .where(MetarObs.temp_f.is_not(None))
            .order_by(MetarObs.city_id, MetarObs.observed_at)
        )
        rows = result.all()

    return pd.DataFrame(
        [
            {
                "city_id": row.city_id,
                "observed_at": row.observed_at,
                "temp_f": row.temp_f,
                "city_slug": row.city_slug,
                "tz": row.tz,
                "dewpoint_f": row.dewpoint_f,
                "humidity_pct": row.humidity_pct,
                "wind_speed_kt": row.wind_speed_kt,
                "wind_gust_kt": row.wind_gust_kt,
                "altimeter_inhg": row.altimeter_inhg,
                "precip_in": row.precip_in,
                "cloud_cover": row.cloud_cover,
                "wx_string": row.wx_string,
                "condition": row.condition,
            }
            for row in rows
        ]
    )


def _load_metar_dataframe() -> pd.DataFrame | None:
    if DB_PATH.exists():
        log.info(f"Connecting to SQLite DB at {DB_PATH}")
        conn = sqlite3.connect(str(DB_PATH))

        # Fetch all MetarObs with city timezone
        query = """
        SELECT
            m.city_id,
            m.observed_at,
            m.temp_f,
            c.city_slug,
            c.tz,
            x.dewpoint_f,
            x.humidity_pct,
            x.wind_speed_kt,
            x.wind_gust_kt,
            x.altimeter_inhg,
            x.precip_in,
            x.cloud_cover,
            x.wx_string,
            x.condition
        FROM metar_obs m
        JOIN cities c ON m.city_id = c.id
        LEFT JOIN metar_obs_extended x ON x.metar_obs_id = m.id
        WHERE m.temp_f IS NOT NULL
        ORDER BY m.city_id, m.observed_at
        """
        log.info("Loading METAR observations from SQLite database...")
        df = pd.read_sql_query(query, conn, parse_dates=["observed_at"])
        conn.close()
        return df

    if os.environ.get("DATABASE_URL"):
        log.info("SQLite training DB not found at %s; loading from app DATABASE_URL", DB_PATH)
        return asyncio.run(_load_metar_dataframe_from_app_db())

    log.error(f"Database not found at {DB_PATH}. Cannot train.")
    return None


def extract_features_and_train():
    # 1. Fetch all MetarObs with city timezone
    df = _load_metar_dataframe()
    if df is None:
        return

    if df.empty:
        log.warning("No METAR observations found. Cannot train.")
        return

    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)

    # Convert to local timezone per city
    def to_local(row):
        tz_str = row["tz"] if row["tz"] else "America/New_York"
        try:
            return row["observed_at"].tz_convert(ZoneInfo(tz_str))
        except Exception:
            return row["observed_at"].tz_convert(ZoneInfo("America/New_York"))

    log.info("Converting timestamps to local timezones...")
    # Per-city timezones produce an object dtype Series when mixed in one
    # DataFrame, so pandas' `.dt` accessor is not safe here.
    df["local_time"] = df.apply(to_local, axis=1)
    df["date_local"] = df["local_time"].apply(lambda dt: dt.date())
    df["hour_local"] = df["local_time"].apply(lambda dt: dt.hour)
    df["minute_local"] = df["local_time"].apply(lambda dt: dt.minute)
    df["day_of_year"] = df["local_time"].apply(lambda dt: dt.timetuple().tm_yday)
    df["minutes_since_midnight"] = df["hour_local"] * 60 + df["minute_local"]
    df["cloud_cover_val"] = df["cloud_cover"].apply(_cloud_cover_value) if "cloud_cover" in df.columns else 0.0
    df["precip_flag"] = df.apply(
        lambda row: _has_precip(
            row.get("wx_string"),
            row.get("condition"),
            row.get("precip_in"),
        ),
        axis=1,
    )
    _fill_numeric(df, "humidity_pct", 50.0)
    _fill_numeric(df, "wind_speed_kt", 0.0)
    _fill_numeric(df, "wind_gust_kt", 0.0)
    _fill_numeric(df, "altimeter_inhg", np.nan)
    _fill_numeric(df, "dewpoint_f", np.nan)
    df["dewpoint_spread_f"] = (df["temp_f"] - df["dewpoint_f"]).clip(lower=0.0)
    df["dewpoint_spread_f"] = df["dewpoint_spread_f"].fillna(10.0)
    df = _add_recent_precip_feature(df)

    log.info(f"Loaded {len(df)} total METAR observations across {df['city_id'].nunique()} cities.")

    # 2. Daily high and peak time per city/date
    daily_stats = df.groupby(["city_id", "date_local"]).agg(
        daily_high_metar=("temp_f", "max")
    ).reset_index()

    idx = df.groupby(["city_id", "date_local"])["temp_f"].idxmax()
    peak_times = df.loc[idx, ["city_id", "date_local", "minutes_since_midnight"]]
    peak_times = peak_times.rename(columns={"minutes_since_midnight": "peak_minutes_since_midnight"})
    daily_stats = pd.merge(daily_stats, peak_times, on=["city_id", "date_local"])

    # 3. Rolling 3-day average of peak timing (shifted to avoid data leakage)
    daily_stats["date_local"] = pd.to_datetime(daily_stats["date_local"])
    daily_stats = daily_stats.sort_values(["city_id", "date_local"])
    daily_stats["avg_peak_timing_mins"] = daily_stats.groupby("city_id")["peak_minutes_since_midnight"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    daily_stats["avg_peak_timing_mins"] = daily_stats["avg_peak_timing_mins"].fillna(960)  # default 4 PM

    # 4. Merge back to observations
    df["date_local"] = pd.to_datetime(df["date_local"])
    df = pd.merge(df, daily_stats[["city_id", "date_local", "daily_high_metar", "avg_peak_timing_mins"]],
                  on=["city_id", "date_local"])

    # 5. Filter to daytime hours only (6 AM – 10 PM)
    df = df[(df["hour_local"] >= 6) & (df["hour_local"] <= 22)].copy()

    # 6. Temperature slope over last 3 hours via merge_asof
    df = df.sort_values(["city_id", "observed_at"])
    df_3h_ago = df[["observed_at", "city_id", "temp_f"]].copy()
    df_3h_ago["observed_at"] = df_3h_ago["observed_at"] + pd.Timedelta(hours=3)
    df_3h_ago = df_3h_ago.rename(columns={"temp_f": "temp_f_3h_ago"})

    df = pd.merge_asof(
        df.sort_values("observed_at"),
        df_3h_ago.sort_values("observed_at"),
        on="observed_at",
        by="city_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=30)
    )
    df["temp_f_3h_ago"] = df["temp_f_3h_ago"].fillna(df["temp_f"])
    df["temp_slope_3h"] = df["temp_f"] - df["temp_f_3h_ago"]

    df_press_3h_ago = df[["observed_at", "city_id", "altimeter_inhg"]].copy()
    df_press_3h_ago["observed_at"] = df_press_3h_ago["observed_at"] + pd.Timedelta(hours=3)
    df_press_3h_ago = df_press_3h_ago.rename(columns={"altimeter_inhg": "altimeter_3h_ago"})
    df = pd.merge_asof(
        df.sort_values("observed_at"),
        df_press_3h_ago.sort_values("observed_at"),
        on="observed_at",
        by="city_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=45),
    )
    df["pressure_tendency_3h"] = (
        pd.to_numeric(df["altimeter_inhg"], errors="coerce")
        - pd.to_numeric(df["altimeter_3h_ago"], errors="coerce")
    ).fillna(0.0)
    df["regime_score_proxy"] = (
        0.35 * (df["cloud_cover_val"] >= 3).astype(float)
        + 0.25 * (df["precip_recent_3h"] > 0).astype(float)
        + 0.20 * (df["pressure_tendency_3h"].abs() >= 0.03).astype(float)
        + 0.20 * (df["wind_gust_kt"] >= 20).astype(float)
    ).clip(upper=1.0)

    # 7. Target: remaining rise = daily_high - current_temp, clipped at 0
    df["y_remaining_rise"] = (df["daily_high_metar"] - df["temp_f"]).clip(lower=0.0)
    feature_defaults = {
        "humidity_pct": 50.0,
        "cloud_cover_val": 0.0,
        "wind_speed_kt": 0.0,
        "wind_gust_kt": 0.0,
        "dewpoint_spread_f": 10.0,
        "pressure_tendency_3h": 0.0,
        "precip_flag": 0.0,
        "precip_recent_3h": 0.0,
        "regime_score_proxy": 0.0,
    }
    for col, default in feature_defaults.items():
        _fill_numeric(df, col, default)
    df = df.dropna(subset=["temp_slope_3h", "avg_peak_timing_mins", "y_remaining_rise"])

    if len(df) < 50:
        log.warning(f"Only {len(df)} usable samples — not enough to train. Need more METAR history.")
        return

    # 8. Features
    features = [
        "hour_local",
        "temp_f",
        "temp_slope_3h",
        "avg_peak_timing_mins",
        "day_of_year",
        "humidity_pct",
        "cloud_cover_val",
        "wind_speed_kt",
        "wind_gust_kt",
        "dewpoint_spread_f",
        "pressure_tendency_3h",
        "precip_flag",
        "precip_recent_3h",
        "regime_score_proxy",
    ]
    X = df[features]
    y = df["y_remaining_rise"]

    # Chronological, date-grouped holdout. A random row split leaks same-day
    # diurnal structure because morning and afternoon observations from the same
    # city/date can land on both sides of the split.
    unique_dates = sorted(pd.to_datetime(df["date_local"]).dt.date.unique())
    if len(unique_dates) < 5:
        log.warning(
            "Only %d unique local dates — not enough for leakage-safe chronological validation.",
            len(unique_dates),
        )
        return
    split_idx = max(1, int(len(unique_dates) * 0.8))
    split_idx = min(split_idx, len(unique_dates) - 1)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])
    train_mask = pd.to_datetime(df["date_local"]).dt.date.isin(train_dates)
    test_mask = pd.to_datetime(df["date_local"]).dt.date.isin(test_dates)
    train_groups = set(zip(df.loc[train_mask, "city_id"], pd.to_datetime(df.loc[train_mask, "date_local"]).dt.date))
    test_groups = set(zip(df.loc[test_mask, "city_id"], pd.to_datetime(df.loc[test_mask, "date_local"]).dt.date))
    leaked_groups = sorted(train_groups & test_groups)
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    if X_train.empty or X_test.empty:
        log.warning(
            "Chronological split produced empty train/test partition (train=%d, test=%d).",
            len(X_train), len(X_test),
        )
        return

    # 9. Train
    log.info(
        "Training GradientBoostingRegressor on %d train samples / %d test samples "
        "(%d train dates / %d test dates)...",
        len(X_train), len(X_test), len(train_dates), len(test_dates),
    )
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 10. Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    # Baseline MAE: what the old static table would have predicted
    baseline_preds = X_test["hour_local"].apply(_old_remaining_rise)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    test_df = df.loc[test_mask].copy()

    def _subset_metrics(mask: pd.Series) -> dict:
        subset_n = int(mask.sum())
        if subset_n <= 0:
            return {"n": 0, "test_mae": None, "baseline_mae": None, "improvement_f": None}
        subset_model_mae = mean_absolute_error(y_test.loc[mask], test_preds[mask.to_numpy()])
        subset_baseline_mae = mean_absolute_error(y_test.loc[mask], baseline_preds.loc[mask])
        return {
            "n": subset_n,
            "test_mae": round(float(subset_model_mae), 3),
            "baseline_mae": round(float(subset_baseline_mae), 3),
            "improvement_f": round(float(subset_baseline_mae - subset_model_mae), 3),
        }

    rain_mask = (
        (test_df["precip_flag"] > 0)
        | (test_df["precip_recent_3h"] > 0)
        | ((test_df["humidity_pct"] >= 75) & (test_df["cloud_cover_val"] >= 3))
    )
    regime_mask = test_df["regime_score_proxy"] >= 0.5
    rain_subset = _subset_metrics(rain_mask)
    regime_subset = _subset_metrics(regime_mask)

    log.info(f"=== Results ===")
    log.info(f"ML Model  — Train MAE: {train_mae:.2f}°F  |  Test MAE: {test_mae:.2f}°F")
    log.info(f"Old Table — Baseline MAE: {baseline_mae:.2f}°F")
    improvement_f = baseline_mae - test_mae
    improvement_pct = (1 - test_mae / baseline_mae) * 100 if baseline_mae > 0 else 0.0
    log.info(f"Improvement: {improvement_f:.2f}°F lower MAE ({improvement_pct:.1f}% reduction)")

    # Feature importances
    importances = model.feature_importances_
    for col, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        log.info(f"  Feature [{col}]: {imp:.4f}")

    promotion_blockers: list[str] = []
    if improvement_f < MIN_PROMOTION_MAE_IMPROVEMENT_F:
        promotion_blockers.append(
            f"overall_improvement {improvement_f:.2f}F < {MIN_PROMOTION_MAE_IMPROVEMENT_F:.2f}F"
        )
    if leaked_groups:
        promotion_blockers.append(f"city_date_leakage_detected n={len(leaked_groups)}")
    rain_improvement = rain_subset.get("improvement_f")
    if int(rain_subset.get("n") or 0) >= RAIN_SUBSET_MIN_N:
        if rain_improvement is not None and rain_improvement < MIN_RAIN_SUBSET_IMPROVEMENT_F:
            promotion_blockers.append(
                f"rain_subset_improvement {rain_improvement:.2f}F < {MIN_RAIN_SUBSET_IMPROVEMENT_F:.2f}F"
            )
    elif rain_improvement is not None and rain_improvement < -MAX_RAIN_SUBSET_WORSE_F:
        promotion_blockers.append(
            f"rain_subset_worse {rain_improvement:.2f}F < -{MAX_RAIN_SUBSET_WORSE_F:.2f}F"
        )
    regime_improvement = regime_subset.get("improvement_f")
    if int(regime_subset.get("n") or 0) >= RAIN_SUBSET_MIN_N:
        if regime_improvement is not None and regime_improvement < -MAX_RAIN_SUBSET_WORSE_F:
            promotion_blockers.append(
                f"regime_subset_worse {regime_improvement:.2f}F < -{MAX_RAIN_SUBSET_WORSE_F:.2f}F"
            )
    promotion_ready = not promotion_blockers
    output_model_path = MODEL_PATH if (PROMOTE_RESIDUAL_ML and promotion_ready) else SHADOW_MODEL_PATH
    output_meta_path = METADATA_PATH if (PROMOTE_RESIDUAL_ML and promotion_ready) else SHADOW_METADATA_PATH

    # 11. Save. Default is shadow-only so a training run cannot silently change
    # live remaining-rise predictions loaded by residual_tracker.py.
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model_path)
    log.info(f"Saved {'promoted' if output_model_path == MODEL_PATH else 'shadow'} model to {output_model_path}")
    if PROMOTE_RESIDUAL_ML and not promotion_ready:
        log.warning(
            "PROMOTE_RESIDUAL_ML requested but blocked: %s",
            "; ".join(promotion_blockers),
        )

    meta = {
        "features": features,
        "train_mae": round(float(train_mae), 3),
        "test_mae": round(float(test_mae), 3),
        "baseline_mae": round(float(baseline_mae), 3),
        "improvement_f": round(float(improvement_f), 3),
        "improvement_pct": round(float(improvement_pct), 1),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "n_cities": int(df["city_id"].nunique()),
        "split_method": "chronological_date_holdout",
        "train_dates": [str(d) for d in sorted(train_dates)],
        "test_dates": [str(d) for d in sorted(test_dates)],
        "date_group_leakage_detected": bool(leaked_groups),
        "leaked_city_date_groups": [f"{city_id}:{date_value}" for city_id, date_value in leaked_groups[:20]],
        "rain_subset_mae": rain_subset,
        "regime_subset_mae": regime_subset,
        "feature_importances": {
            col: round(float(imp), 5)
            for col, imp in sorted(zip(features, importances), key=lambda x: -x[1])
        },
        "promotion_ready": bool(promotion_ready),
        "promoted": bool(output_model_path == MODEL_PATH),
        "promotion_blockers": promotion_blockers,
        "promotion_threshold_mae_improvement_f": MIN_PROMOTION_MAE_IMPROVEMENT_F,
        "rain_subset_min_n": RAIN_SUBSET_MIN_N,
        "rain_subset_min_improvement_f": MIN_RAIN_SUBSET_IMPROVEMENT_F,
    }
    output_meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Saved metadata to {output_meta_path}")
    if output_model_path == MODEL_PATH:
        from backend.modeling.residual_artifacts import save_promoted_residual_artifact_to_db

        saved_to_db = asyncio.run(save_promoted_residual_artifact_to_db())
        if saved_to_db:
            log.info("Saved promoted model artifact to Postgres")
        else:
            log.warning("Promoted model was not saved to Postgres")
    log.info("Done.")


if __name__ == "__main__":
    extract_features_and_train()
