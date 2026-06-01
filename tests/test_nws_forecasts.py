from types import SimpleNamespace

import pytest

from backend.ingestion.forecasts import _extract_nws_high_for_date


def _city():
    return SimpleNamespace(tz="America/New_York")


def _forecast(periods):
    return {"properties": {"periods": periods}}


def test_nws_extract_requires_requested_local_date():
    data = _forecast([
        {
            "number": 1,
            "name": "Tuesday",
            "isDaytime": True,
            "startTime": "2026-06-02T06:00:00-04:00",
            "endTime": "2026-06-02T18:00:00-04:00",
            "temperature": 88,
            "temperatureUnit": "F",
        }
    ])

    high_f, meta, parse_error = _extract_nws_high_for_date(
        data, _city(), "2026-06-01"
    )

    assert high_f is None
    assert parse_error == "target_date_not_in_nws_periods"
    assert meta["requested_date"] == "2026-06-01"
    assert meta["available_daytime_period_dates"] == ["2026-06-02"]


def test_nws_extract_uses_matching_requested_date_only():
    data = _forecast([
        {
            "number": 1,
            "name": "Monday",
            "isDaytime": True,
            "startTime": "2026-06-01T06:00:00-04:00",
            "endTime": "2026-06-01T18:00:00-04:00",
            "temperature": 86,
            "temperatureUnit": "F",
        },
        {
            "number": 2,
            "name": "Tuesday",
            "isDaytime": True,
            "startTime": "2026-06-02T06:00:00-04:00",
            "endTime": "2026-06-02T18:00:00-04:00",
            "temperature": 94,
            "temperatureUnit": "F",
        },
    ])

    high_f, meta, parse_error = _extract_nws_high_for_date(
        data, _city(), "2026-06-01"
    )

    assert high_f == pytest.approx(86.0)
    assert parse_error is None
    assert meta["matched_period_name"] == "Monday"
    assert meta["matched_local_date"] == "2026-06-01"
