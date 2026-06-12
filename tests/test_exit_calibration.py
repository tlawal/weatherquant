from backend.backtesting.exit_calibration import (
    ExitCalibrationParams,
    build_exit_calibration_report,
)


def test_exit_calibration_empty_collects_more_data():
    report = build_exit_calibration_report(
        [],
        params=ExitCalibrationParams(min_samples=4),
    )
    assert report["sample"]["n"] == 0
    assert report["recommendations"][0]["action"] == "collect_more_realized_trades"
    assert report["promotion"]["allowed_for_live_parameter_change"] is False


def test_exit_calibration_flags_quick_flip_that_gives_up_settlement_value():
    rows = [
        {
            "exit_reason": "quick_flip",
            "exit_level": "PROFIT",
            "regime_bin": "low",
            "reason_regime": "quick_flip:low",
            "realized_pnl": 0.20,
            "foregone_pnl": 0.08,
            "hold_time_hours": 1.0,
            "regime_score": 0.2,
        },
        {
            "exit_reason": "quick_flip",
            "exit_level": "PROFIT",
            "regime_bin": "low",
            "reason_regime": "quick_flip:low",
            "realized_pnl": 0.10,
            "foregone_pnl": 0.06,
            "hold_time_hours": 1.2,
            "regime_score": 0.2,
        },
    ]
    report = build_exit_calibration_report(
        rows,
        params=ExitCalibrationParams(min_samples=2),
    )
    rec = report["recommendations"][0]
    assert rec["policy"] == "quick_flip_target"
    assert rec["action"] == "raise_or_add_regime_condition"
    assert rec["evidence"]["avg_foregone_pnl"] == 0.07


def test_exit_calibration_flags_edge_decay_as_too_aggressive():
    rows = [
        {
            "exit_reason": "ev_decayed",
            "exit_level": "EDGE_DECAY",
            "regime_bin": "high",
            "reason_regime": "ev_decayed:high",
            "realized_pnl": -0.05,
            "foregone_pnl": 0.10,
            "hold_time_hours": 2.0,
            "regime_score": 0.8,
        },
        {
            "exit_reason": "ev_decayed",
            "exit_level": "EDGE_DECAY",
            "regime_bin": "high",
            "reason_regime": "ev_decayed:high",
            "realized_pnl": -0.02,
            "foregone_pnl": 0.04,
            "hold_time_hours": 2.5,
            "regime_score": 0.7,
        },
    ]
    report = build_exit_calibration_report(
        rows,
        params=ExitCalibrationParams(min_samples=2),
    )
    rec = report["recommendations"][0]
    assert rec["policy"] == "edge_decay"
    assert rec["action"] == "increase_debounce_or_required_drop"
    assert report["breakdowns"]["regime_bin"][0]["label"] == "high"
