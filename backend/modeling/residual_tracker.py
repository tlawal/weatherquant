"""
Residual Tracker — ML-based remaining rise predictor.

Loads a pre-trained GradientBoostingRegressor (trained by ml_trainer.py)
and predicts remaining temperature rise given current conditions.

Falls back to the old static table if no trained model is available.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "residual_model.pkl"
METADATA_PATH = Path(__file__).parent / "residual_model_meta.json"

# ─── Static fallback (identical to old _REMAINING_RISE_TABLE) ────────────────
_FALLBACK_TABLE = [
    (0, 6, 14.0), (6, 9, 11.0), (9, 11, 8.0), (11, 13, 4.0),
    (13, 15, 2.0), (15, 17, 0.5), (17, 19, 0.0), (19, 24, 0.0),
]

def _fallback_remaining_rise(hour_local: int) -> float:
    for s, e, v in _FALLBACK_TABLE:
        if s <= hour_local < e:
            return v
    return 0.0


# ─── Singleton model loader ─────────────────────────────────────────────────
_model = None
_features: list[str] = []
_loaded = False


def _ensure_model():
    global _model, _features, _loaded
    if _loaded:
        return
    _loaded = True

    if not MODEL_PATH.exists():
        log.warning("residual_tracker: no trained model found at %s — using static fallback", MODEL_PATH)
        return

    try:
        _model = joblib.load(MODEL_PATH)
        if METADATA_PATH.exists():
            meta = json.loads(METADATA_PATH.read_text())
            _features = meta.get("features", [])
            log.info(
                "residual_tracker: loaded model (test_mae=%.2f°F, %d samples, trained %s)",
                meta.get("test_mae", -1), meta.get("n_samples", 0), meta.get("trained_at", "?")
            )
        else:
            _features = ["hour_local", "temp_f", "temp_slope_3h", "avg_peak_timing_mins", "day_of_year"]
            log.info("residual_tracker: loaded model (no metadata found)")
    except Exception:
        log.exception("residual_tracker: failed to load model — using static fallback")
        _model = None


def predict_remaining_rise(
    hour_local: int,
    current_temp_f: float,
    temp_slope_3h: float = 0.0,
    avg_peak_timing_mins: float = 960.0,
    day_of_year: int = 80,
    unit_mult: float = 1.0,
) -> float:
    """
    Predict the remaining temperature rise from the current observation
    to the day's ultimate high.

    Returns a value in the same units as current_temp_f (°F or °C after unit_mult).
    """
    _ensure_model()

    if _model is not None:
        import numpy as np
        features = {
            "hour_local": hour_local,
            "temp_f": current_temp_f,
            "temp_slope_3h": temp_slope_3h,
            "avg_peak_timing_mins": avg_peak_timing_mins,
            "day_of_year": day_of_year,
        }
        # Construct feature vector in the order the model expects
        ordered_features = _features if _features else list(features.keys())
        X = np.array([[features.get(f, 0.0) for f in ordered_features]])

        try:
            pred = float(_model.predict(X)[0])
            return max(0.0, pred * unit_mult)
        except Exception:
            log.exception("residual_tracker: prediction failed — falling back to static table")

    # Static fallback
    return _fallback_remaining_rise(hour_local) * unit_mult


def is_ml_model_loaded() -> bool:
    """Check if the ML model is actively being used (vs. static fallback)."""
    _ensure_model()
    return _model is not None
