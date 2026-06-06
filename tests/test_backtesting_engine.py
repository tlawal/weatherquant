from datetime import datetime, timezone
from types import SimpleNamespace

from backend.backtesting.engine import (
    BacktestParams,
    GAMMA_PAGE_SIZE,
    Portfolio,
    SimTrade,
    _build_bma_comparison,
    _build_exit_breakdown,
    _build_forecast_source_diagnostics,
    _break_even_win_rate,
    _finalize_gate_diagnostics,
    _new_gate_diagnostics,
    _normalise_entry_market_row,
    _select_actionable_model_snapshot,
    _winner_idx_from_high,
    simulate_entry,
    simulate_exits,
)


def test_winner_idx_from_high_uses_canonical_integer_bucket_bounds():
    buckets = [
        SimpleNamespace(low_f=None, high_f=77.0),
        SimpleNamespace(low_f=78.0, high_f=79.0),
        SimpleNamespace(low_f=80.0, high_f=81.0),
        SimpleNamespace(low_f=82.0, high_f=None),
    ]

    assert _winner_idx_from_high(buckets, 79.0) == 1
    assert _winner_idx_from_high(buckets, 80.0) == 2


def test_bma_comparison_scores_shadow_against_legacy_and_market():
    events = [
        {
            "city_slug": "atlanta",
            "date_et": "2026-06-01",
            "winning_bucket_idx": 1,
            "settlement_status": "station_metar",
            "model_probs": [0.20, 0.55, 0.25],
            "model_inputs": {"bma_shadow": {"probs": [0.10, 0.75, 0.15]}},
            "market_data": {
                0: {"yes_mid": 0.25},
                1: {"yes_mid": 0.45},
                2: {"yes_mid": 0.30},
            },
            "buckets": [{"idx": 0}, {"idx": 1}, {"idx": 2}],
        }
    ]

    out = _build_bma_comparison(events)

    assert out["events_scored"] == 1
    assert out["bma_better_than_legacy"] is True
    assert out["overall"]["bma"]["brier"] < out["overall"]["legacy"]["brier"]
    assert out["overall"]["bma"]["edge_bps"] > out["overall"]["legacy"]["edge_bps"]


def test_forecast_source_diagnostics_rank_component_mae_and_bias():
    events = [
        {
            "resolved_high_f": 80.0,
            "model_inputs": {
                "bma_shadow": {
                    "components": [
                        {"source": "hrrr", "mu": 80.5, "sigma": 1.5, "weight": 0.6},
                        {"source": "ifs", "mu": 84.0, "sigma": 2.5, "weight": 0.4},
                    ]
                }
            },
        },
        {
            "resolved_high_f": 82.0,
            "model_inputs": {
                "bma_shadow": {
                    "components": [
                        {"source": "hrrr", "mu": 81.5, "sigma": 1.5, "weight": 0.7},
                        {"source": "ifs", "mu": 85.0, "sigma": 2.5, "weight": 0.3},
                    ]
                }
            },
        },
    ]

    out = _build_forecast_source_diagnostics(events)

    assert out["best_source"] == "hrrr"
    assert out["worst_source"] == "ifs"
    hrrr = next(row for row in out["components"] if row["source"] == "hrrr")
    assert hrrr["mae_f"] == 0.5
    assert hrrr["bias_f"] == 0.0


def test_backtest_params_default_walkforward_threshold_stays_28_days():
    params = BacktestParams()

    assert params.walk_forward_train_days + params.walk_forward_test_days == 28


def test_backtest_params_defaults_are_exploratory_for_warmed_live_data():
    params = BacktestParams()

    assert params.strategy_profile == "ev_baseline"
    assert params.bankroll == 100.0
    assert params.max_position_pct == 0.20
    assert params.min_true_edge == 0.03
    assert params.min_model_prob == 0.0
    assert params.min_entry_price == 0.0
    assert params.max_entry_price == 0.60
    assert params.max_spread == 0.20
    assert params.min_liquidity_shares == 0.0


def test_gamma_page_size_matches_gamma_api_cap_for_pagination():
    assert GAMMA_PAGE_SIZE == 100


def test_break_even_win_rate_uses_net_settlement_payout():
    assert _break_even_win_rate(0.15) == 0.1531
    assert _break_even_win_rate(0.85) == 0.8673
    assert _break_even_win_rate(1.10) == 1.0


def test_high_win_rate_filters_reject_low_model_probability_first_when_edge_passes():
    params = BacktestParams(
        strategy_profile="high_win_rate",
        min_true_edge=0.02,
        min_model_prob=0.80,
        min_entry_price=0.65,
        max_entry_price=0.92,
        max_spread=0.10,
        min_liquidity_shares=0,
    )
    diagnostics = _new_gate_diagnostics(params)
    event_data = {
        "event_id": 1,
        "city_slug": "atlanta",
        "date_et": "2026-06-01",
        "buckets": [{"idx": 0, "label": "82-83F"}],
        "model_probs": [0.78],
        "market_data": {
            0: {
                "yes_mid": 0.60,
                "yes_ask": 0.65,
                "spread": 0.02,
                "ask_depth": 100.0,
            }
        },
    }

    trade = simulate_entry(event_data, 0, params, Portfolio(bankroll=100, equity=100), diagnostics)
    out = _finalize_gate_diagnostics(diagnostics)

    assert trade is None
    assert out["top_reasons"][0]["reason"] == "model_prob_below_min"
    assert out["best_rejected"]["required"] == 0.8


def test_high_win_rate_filters_reject_entry_price_below_floor():
    params = BacktestParams(
        strategy_profile="high_win_rate",
        min_true_edge=0.02,
        min_model_prob=0.80,
        min_entry_price=0.65,
        max_entry_price=0.92,
        max_spread=0.10,
        min_liquidity_shares=0,
    )
    diagnostics = _new_gate_diagnostics(params)
    event_data = {
        "event_id": 1,
        "city_slug": "atlanta",
        "date_et": "2026-06-01",
        "buckets": [{"idx": 0, "label": "82-83F"}],
        "model_probs": [0.82],
        "market_data": {
            0: {
                "yes_mid": 0.40,
                "yes_ask": 0.40,
                "spread": 0.02,
                "ask_depth": 100.0,
            }
        },
    }

    trade = simulate_entry(event_data, 0, params, Portfolio(bankroll=100, equity=100), diagnostics)
    out = _finalize_gate_diagnostics(diagnostics)

    assert trade is None
    assert out["top_reasons"][0]["reason"] == "entry_price_below_min"


def test_entry_market_row_derives_mid_and_spread_from_bid_ask():
    out = _normalise_entry_market_row({
        "yes_mid": None,
        "yes_bid": 0.20,
        "yes_ask": 0.24,
        "spread": None,
        "yes_bid_depth": 12,
        "yes_ask_depth": 18,
    })

    assert out["yes_mid"] == 0.22
    assert round(out["spread"], 6) == 0.04
    assert out["bid_depth"] == 12.0
    assert out["ask_depth"] == 18.0


def test_select_actionable_snapshot_prefers_same_day_morning_over_post_close():
    event = SimpleNamespace(date_et="2026-06-01")
    city = SimpleNamespace(tz="America/New_York")
    overnight = SimpleNamespace(id=1, computed_at=datetime(2026, 6, 1, 3, 30, tzinfo=timezone.utc))
    morning = SimpleNamespace(id=2, computed_at=datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc))
    post_close = SimpleNamespace(id=3, computed_at=datetime(2026, 6, 2, 2, 0, tzinfo=timezone.utc))

    selected = _select_actionable_model_snapshot(event, city, [overnight, morning, post_close])

    assert selected.id == 2


def test_simulate_entry_records_gate_diagnostics_for_rejected_candidate():
    params = BacktestParams(min_true_edge=0.10, max_entry_price=0.36, min_liquidity_shares=10)
    diagnostics = _new_gate_diagnostics(params)
    event_data = {
        "event_id": 1,
        "city_slug": "atlanta",
        "date_et": "2026-06-01",
        "buckets": [{"idx": 0, "label": "82-83F"}],
        "model_probs": [0.20],
        "market_data": {
            0: {
                "yes_mid": 0.15,
                "yes_ask": 0.16,
                "spread": 0.01,
                "ask_depth": 25.0,
            }
        },
    }

    trade = simulate_entry(event_data, 0, params, Portfolio(bankroll=10, equity=10), diagnostics)
    out = _finalize_gate_diagnostics(diagnostics)

    assert trade is None
    assert out["candidates_evaluated"] == 1
    assert out["trades_taken"] == 0
    assert out["top_reasons"][0]["reason"] == "edge_below_min"
    assert out["best_rejected"]["city_slug"] == "atlanta"
    assert out["best_rejected"]["bucket_label"] == "82-83F"


def test_simulate_entry_records_gate_acceptance():
    params = BacktestParams(min_true_edge=0.10, max_entry_price=0.36, min_liquidity_shares=10)
    diagnostics = _new_gate_diagnostics(params)
    event_data = {
        "event_id": 1,
        "city_slug": "atlanta",
        "date_et": "2026-06-01",
        "buckets": [{"idx": 0, "label": "82-83F"}],
        "model_probs": [0.70],
        "market_data": {
            0: {
                "yes_mid": 0.20,
                "yes_ask": 0.21,
                "spread": 0.02,
                "ask_depth": 100.0,
            }
        },
    }

    trade = simulate_entry(event_data, 0, params, Portfolio(bankroll=10, equity=10), diagnostics)
    out = _finalize_gate_diagnostics(diagnostics)

    assert trade is not None
    assert out["candidates_evaluated"] == 1
    assert out["trades_taken"] == 1
    assert out["rejected_total"] == 0


def test_simulate_exits_records_quick_flip_hold_time_and_hold_comparison():
    params = BacktestParams(quick_flip_target=0.05)
    trade = SimTrade(
        city_slug="atlanta",
        date_et="2026-06-01",
        bucket_idx=0,
        bucket_label="82-83F",
        model_prob=0.70,
        mkt_prob=0.20,
        true_edge=0.45,
        side="buy_yes",
        entry_price=0.20,
        shares=10.0,
        cost=2.0,
    )
    events_map = {
        ("atlanta", "2026-06-01"): {
            "winning_bucket_idx": 1,
            "snapshot_time": datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
            "later_market_data": {
                0: [{
                    "yes_bid": 0.26,
                    "fetched_at": datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc),
                }]
            },
        }
    }

    simulate_exits([trade], events_map, params)
    breakdown = _build_exit_breakdown([trade])

    assert trade.exit_reason == "quick_flip"
    assert trade.hold_time_hours == 2.0
    assert trade.hold_to_resolution_won is False
    assert trade.hold_to_resolution_pnl == -2.0
    assert breakdown["by_exit"][0]["label"] == "quick_flip"
    assert breakdown["by_exit"][0]["quick_flip_vs_hold_pnl_delta"] > 0
