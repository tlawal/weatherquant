"""
ML Trainer — extracts historical METAR observations from SQLite,
engineers features (temp slope, peak timing persistence, seasonality),
and trains a GradientBoostingRegressor to predict remaining temperature rise.

Usage:
    python -m backend.modeling.ml_trainer
"""
import json
import logging
import sqlite3

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("ml_trainer")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "state.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent.parent.parent / "state.db"
MODEL_PATH = Path(__file__).parent / "residual_model.pkl"
METADATA_PATH = Path(__file__).parent / "residual_model_meta.json"

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


def extract_features_and_train():
    log.info(f"Connecting to SQLite DB at {DB_PATH}")
    if not DB_PATH.exists():
        log.error(f"Database not found at {DB_PATH}. Cannot train.")
        return
    conn = sqlite3.connect(str(DB_PATH))

    # 1. Fetch all MetarObs with city timezone
    query = """
    SELECT 
        m.city_id,
        m.observed_at,
        m.temp_f,
        c.city_slug,
        c.tz
    FROM metar_obs m
    JOIN cities c ON m.city_id = c.id
    WHERE m.temp_f IS NOT NULL
    ORDER BY m.city_id, m.observed_at
    """
    log.info("Loading METAR observations from database...")
    df = pd.read_sql_query(query, conn, parse_dates=["observed_at"])
    conn.close()

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
    df["local_time"] = df.apply(to_local, axis=1)
    df["date_local"] = df["local_time"].dt.date
    df["hour_local"] = df["local_time"].dt.hour
    df["minute_local"] = df["local_time"].dt.minute
    df["day_of_year"] = df["local_time"].dt.dayofyear
    df["minutes_since_midnight"] = df["hour_local"] * 60 + df["minute_local"]

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

    # 7. Target: remaining rise = daily_high - current_temp, clipped at 0
    df["y_remaining_rise"] = (df["daily_high_metar"] - df["temp_f"]).clip(lower=0.0)
    df = df.dropna(subset=["temp_slope_3h", "avg_peak_timing_mins", "y_remaining_rise"])

    if len(df) < 50:
        log.warning(f"Only {len(df)} usable samples — not enough to train. Need more METAR history.")
        return

    # 8. Features
    features = ["hour_local", "temp_f", "temp_slope_3h", "avg_peak_timing_mins", "day_of_year"]
    X = df[features]
    y = df["y_remaining_rise"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 9. Train
    log.info(f"Training GradientBoostingRegressor on {len(X_train)} samples ({len(X_test)} test)...")
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

    log.info(f"=== Results ===")
    log.info(f"ML Model  — Train MAE: {train_mae:.2f}°F  |  Test MAE: {test_mae:.2f}°F")
    log.info(f"Old Table — Baseline MAE: {baseline_mae:.2f}°F")
    log.info(f"Improvement: {baseline_mae - test_mae:.2f}°F lower MAE ({(1 - test_mae / baseline_mae) * 100:.1f}% reduction)")

    # Feature importances
    importances = model.feature_importances_
    for col, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        log.info(f"  Feature [{col}]: {imp:.4f}")

    # 11. Save
    joblib.dump(model, MODEL_PATH)
    log.info(f"Saved model to {MODEL_PATH}")

    meta = {
        "features": features,
        "train_mae": round(float(train_mae), 3),
        "test_mae": round(float(test_mae), 3),
        "baseline_mae": round(float(baseline_mae), 3),
        "improvement_pct": round(float((1 - test_mae / baseline_mae) * 100), 1),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "n_cities": int(df["city_id"].nunique()),
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    log.info(f"Saved metadata to {METADATA_PATH}")
    log.info("Done.")


if __name__ == "__main__":
    extract_features_and_train()
