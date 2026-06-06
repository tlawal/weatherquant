import numpy as np
import pandas as pd

from backend.modeling.ml_trainer import (
    _add_recent_precip_feature,
    _cloud_cover_value,
    _has_precip,
)


def test_precip_feature_detects_weather_strings_and_amounts():
    assert _has_precip(wx_string="-RA", condition=None, precip_in=None) == 1.0
    assert _has_precip(wx_string=None, condition="Thunderstorm", precip_in=None) == 1.0
    assert _has_precip(wx_string=None, condition=None, precip_in=0.01) == 1.0
    assert _has_precip(wx_string="CLR", condition="Clear", precip_in=0.0) == 0.0


def test_cloud_cover_value_orders_sky_cover():
    assert _cloud_cover_value("CLR") == 0.0
    assert _cloud_cover_value("SCT") == 2.0
    assert _cloud_cover_value("OVC") == 4.0
    assert _cloud_cover_value(None) == 0.0


def test_recent_precip_feature_rolls_per_city_over_three_hours():
    base = pd.Timestamp("2026-06-01T12:00:00Z")
    df = pd.DataFrame({
        "city_id": [1, 1, 1, 1, 2],
        "observed_at": [
            base,
            base + pd.Timedelta(hours=1),
            base + pd.Timedelta(hours=2),
            base + pd.Timedelta(hours=4),
            base + pd.Timedelta(hours=2),
        ],
        "precip_flag": [1.0, 0.0, 0.0, 0.0, 0.0],
    })

    out = _add_recent_precip_feature(df)

    city1 = out[out["city_id"] == 1].sort_values("observed_at")
    assert city1["precip_recent_3h"].to_list() == [1.0, 1.0, 1.0, 0.0]
    city2 = out[out["city_id"] == 2]
    assert np.allclose(city2["precip_recent_3h"], [0.0])
