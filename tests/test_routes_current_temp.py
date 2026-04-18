"""Unit tests for the dual-source Current Temp card helpers in web/routes.py.

These tests call the module-level helpers directly (no FastAPI / HTTP round-trip,
no DB) using SimpleNamespace stand-ins for the ORM rows. Time is frozen via the
`now` kwarg so age assertions are deterministic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from web.routes import _format_current_temp_dual, humanize_age


# ── humanize_age ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, "—"),
        (0, "0s ago"),
        (45, "45s ago"),
        (59, "59s ago"),
        (60, "1m ago"),
        (120, "2m ago"),
        (125, "2m 5s ago"),
        (3599, "59m 59s ago"),
        (3600, "1h ago"),
        (3900, "1h 5m ago"),
        (7261, "2h 1m ago"),  # 2*3600 + 60 + 1 — seconds drop at hour granularity
        (-3, "0s ago"),  # negative → clamped
        ("nope", "—"),  # garbage → "—"
    ],
)
def test_humanize_age(seconds, expected):
    assert humanize_age(seconds) == expected


# ── _format_current_temp_dual ───────────────────────────────────────────────

_NOW = datetime(2026, 4, 18, 22, 30, 0, tzinfo=timezone.utc)
_TODAY_ET = "2026-04-18"


def _metar(temp_f, obs_at, station="KATL", raw_text=None):
    return SimpleNamespace(
        temp_f=temp_f,
        observed_at=obs_at,
        metar_station=station,
        raw_text=raw_text,
    )


def _city(is_us=True, tz="America/New_York"):
    return SimpleNamespace(is_us=is_us, tz=tz)


def test_dual_returns_none_for_non_us_city():
    madis = _metar(75.0, _NOW)
    nws = _metar(74.0, _NOW)
    result = _format_current_temp_dual(
        madis, nws, None, _city(is_us=False),
        target_date_et=_TODAY_ET, real_today_et=_TODAY_ET, now=_NOW,
    )
    assert result is None


def test_dual_returns_none_for_stale_date():
    madis = _metar(75.0, _NOW)
    nws = _metar(74.0, _NOW)
    result = _format_current_temp_dual(
        madis, nws, None, _city(),
        target_date_et="2026-04-17", real_today_et=_TODAY_ET, now=_NOW,
    )
    assert result is None


def test_dual_returns_none_when_neither_source_has_data():
    result = _format_current_temp_dual(
        None, None, None, _city(),
        target_date_et=_TODAY_ET, real_today_et=_TODAY_ET, now=_NOW,
    )
    assert result is None


def test_dual_madis_fresher_marks_primary_and_computes_delta():
    # MADIS = 2 min old, NWS = 10 min old → MADIS fresher by 480s.
    madis_obs_at = datetime(2026, 4, 18, 22, 28, 0, tzinfo=timezone.utc)
    nws_obs_at = datetime(2026, 4, 18, 22, 20, 0, tzinfo=timezone.utc)

    madis = _metar(77.0, madis_obs_at, station="KATL")
    nws = _metar(75.2, nws_obs_at, station="KATL", raw_text="KATL 182220Z ...")
    legacy_madis_obs = SimpleNamespace(source_file="20260418_2200.gz")

    result = _format_current_temp_dual(
        madis, nws, legacy_madis_obs, _city(),
        target_date_et=_TODAY_ET, real_today_et=_TODAY_ET, now=_NOW,
    )

    assert result is not None
    assert result["primary"] == "madis"
    assert result["delta_s"] == 480
    assert result["madis"]["temp_f"] == 77.0
    assert result["madis"]["age_s"] == 120
    assert result["madis"]["station"] == "KATL"
    assert result["madis"]["source_url"].endswith("20260418_2200.gz")
    assert result["nws"]["temp_f"] == 75.2
    assert result["nws"]["age_s"] == 600
    assert result["nws"]["raw_text"] == "KATL 182220Z ..."
    assert result["nws"]["source_url"] == (
        "https://api.weather.gov/stations/KATL/observations/latest"
    )


def test_dual_nws_fresher_when_madis_stale():
    # MADIS = 15 min old, NWS = 3 min old → NWS fresher by 720s.
    madis_obs_at = datetime(2026, 4, 18, 22, 15, 0, tzinfo=timezone.utc)
    nws_obs_at = datetime(2026, 4, 18, 22, 27, 0, tzinfo=timezone.utc)
    madis = _metar(77.0, madis_obs_at)
    nws = _metar(75.2, nws_obs_at, raw_text="KATL ...")
    result = _format_current_temp_dual(
        madis, nws, None, _city(),
        target_date_et=_TODAY_ET, real_today_et=_TODAY_ET, now=_NOW,
    )
    assert result is not None
    assert result["primary"] == "nws"
    assert result["delta_s"] == 720


def test_dual_only_nws_present_marks_nws_primary_and_nulls_madis_side():
    nws_obs_at = datetime(2026, 4, 18, 22, 25, 0, tzinfo=timezone.utc)
    nws = _metar(74.0, nws_obs_at, raw_text="KATL ...")
    result = _format_current_temp_dual(
        None, nws, None, _city(),
        target_date_et=_TODAY_ET, real_today_et=_TODAY_ET, now=_NOW,
    )
    assert result is not None
    assert result["primary"] == "nws"
    assert result["delta_s"] is None
    assert result["madis"]["temp_f"] is None
    assert result["madis"]["age_s"] is None
    assert result["nws"]["age_s"] == 300


def test_dual_only_madis_present_marks_madis_primary():
    madis_obs_at = datetime(2026, 4, 18, 22, 29, 0, tzinfo=timezone.utc)
    madis = _metar(77.0, madis_obs_at)
    result = _format_current_temp_dual(
        madis, None, None, _city(),
        target_date_et=_TODAY_ET, real_today_et=_TODAY_ET, now=_NOW,
    )
    assert result is not None
    assert result["primary"] == "madis"
    assert result["delta_s"] is None
    assert result["nws"]["temp_f"] is None


def test_dual_equal_ages_ties_break_to_madis():
    same_time = datetime(2026, 4, 18, 22, 28, 0, tzinfo=timezone.utc)
    madis = _metar(77.0, same_time)
    nws = _metar(75.2, same_time)
    result = _format_current_temp_dual(
        madis, nws, None, _city(),
        target_date_et=_TODAY_ET, real_today_et=_TODAY_ET, now=_NOW,
    )
    assert result["primary"] == "madis"
    assert result["delta_s"] == 0
