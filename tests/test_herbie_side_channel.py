from datetime import datetime, timezone

import numpy as np
import xarray as xr

from backend.ingestion.herbie_side_channel import (
    _as_utc_aware,
    _herbie_run_arg,
    _nearest_grid_value_f,
)


def test_herbie_run_arg_converts_aware_utc_to_naive():
    aware = datetime(2026, 6, 3, 13, tzinfo=timezone.utc)

    got = _herbie_run_arg(aware)

    assert got == datetime(2026, 6, 3, 13)
    assert got.tzinfo is None


def test_as_utc_aware_treats_naive_datetimes_as_utc():
    naive = datetime(2026, 6, 3, 13)

    got = _as_utc_aware(naive)

    assert got == datetime(2026, 6, 3, 13, tzinfo=timezone.utc)


def test_nearest_grid_value_handles_2d_lat_lon_and_wrapped_longitude():
    ds = xr.Dataset(
        data_vars={
            "t2m": (("y", "x"), np.array([[290.0, 295.0], [297.0, 300.0]])),
        },
        coords={
            "latitude": (("y", "x"), np.array([[33.0, 33.0], [34.0, 34.0]])),
            "longitude": (("y", "x"), np.array([[275.0, 276.0], [275.0, 276.0]])),
        },
    )

    assert _nearest_grid_value_f(ds, lat=33.9, lon=-84.1) == 300.0
