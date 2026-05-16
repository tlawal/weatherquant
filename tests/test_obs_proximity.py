from datetime import datetime
from zoneinfo import ZoneInfo

from backend.execution.obs_proximity import (
    build_obs_proximity_status,
    evaluate_obs_proximity_exit,
    sensitivity_badge,
)


BUCKETS = [
    {"idx": 0, "label": "Below 75", "low_f": None, "high_f": 75.0},
    {"idx": 1, "label": "75 to 80", "low_f": 75.0, "high_f": 80.0},
    {"idx": 2, "label": "80 or above", "low_f": 80.0, "high_f": None},
]


def _decision(**overrides):
    params = {
        "city_slug": "nyc",
        "station_id": "KNYC",
        "now_local": datetime(2026, 5, 16, 12, 40, tzinfo=ZoneInfo("America/New_York")),
        "observation_minutes": [52],
        "bucket_specs": BUCKETS,
        "held_bucket_idx": 1,
        "reference_temp_f": 79.4,
        "yes_bid": 0.40,
        "yes_ask": 0.43,
        "yes_bid_depth": 400.0,
        "yes_ask_depth": 350.0,
        "net_pnl_per_share": 0.08,
        "current_edge": 0.06,
        "enabled": True,
        "is_us": True,
        "window_minutes": 20,
        "temp_sensitivity_threshold_f": 1.0,
        "min_profit_cents": 5.0,
        "min_depth_usd": 100.0,
        "max_orderbook_imbalance": 0.72,
        "cooldown_active": False,
    }
    params.update(overrides)
    return evaluate_obs_proximity_exit(**params)


def test_no_exit_outside_observation_window():
    decision = _decision(now_local=datetime(2026, 5, 16, 12, 20, tzinfo=ZoneInfo("America/New_York")))

    assert decision["final_action"] == "SKIP"
    assert decision["skip_reason"] == "outside_observation_window"


def test_no_exit_if_not_near_bucket_boundary():
    decision = _decision(reference_temp_f=77.5)

    assert decision["final_action"] == "SKIP"
    assert decision["skip_reason"] == "not_boundary_fragile"
    assert decision["sensitivity_badge"] == "LOW"


def test_exit_if_one_degree_crosses_boundary_and_profitable():
    decision = _decision()

    assert decision["final_action"] == "EXIT"
    assert decision["skip_reason"] is None
    assert decision["current_bucket"] == "75 to 80"
    assert decision["plus_1f_bucket"] == "80 or above"
    assert decision["mark_to_market_profit_cents"] == 8.0


def test_no_exit_if_orderbook_depth_too_low():
    decision = _decision(yes_bid_depth=100.0)

    assert decision["final_action"] == "SKIP"
    assert decision["skip_reason"] == "bid_depth_below_min"


def test_no_exit_if_orderbook_imbalance_exceeds_threshold():
    decision = _decision(yes_bid_depth=400.0, yes_ask_depth=100.0)

    assert decision["final_action"] == "SKIP"
    assert decision["skip_reason"] == "orderbook_imbalance_too_high"


def test_debounce_prevents_repeated_exits():
    decision = _decision(cooldown_active=True)

    assert decision["final_action"] == "SKIP"
    assert decision["skip_reason"] == "cooldown_active"


def test_dashboard_helper_sensitivity_badges():
    assert sensitivity_badge(3.0, 1.0, False) == "LOW"
    assert sensitivity_badge(1.5, 1.0, False) == "MEDIUM"
    assert sensitivity_badge(0.5, 1.0, False) == "HIGH"
    assert sensitivity_badge(3.0, 1.0, True) == "HIGH"

    status = build_obs_proximity_status(
        city_slug="nyc",
        station_id="KNYC",
        now_local=datetime(2026, 5, 16, 12, 40, tzinfo=ZoneInfo("America/New_York")),
        observation_minutes=[52],
        bucket_specs=BUCKETS,
        reference_temp_f=79.4,
        enabled=True,
        is_us=True,
        window_minutes=20,
        temp_sensitivity_threshold_f=1.0,
    )
    assert status["sensitivity_badge"] == "HIGH"
    assert status["armed"] is True
