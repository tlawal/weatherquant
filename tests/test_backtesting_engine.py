from datetime import datetime, timezone
from types import SimpleNamespace

from backend.backtesting.engine import (
    BacktestParams,
    Portfolio,
    _build_bma_comparison,
    _build_forecast_source_diagnostics,
    _finalize_gate_diagnostics,
    _new_gate_diagnostics,
    _normalise_entry_market_row,
    _select_actionable_model_snapshot,
    _winner_idx_from_high,
    simulate_entry,
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
