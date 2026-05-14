from datetime import datetime, timezone
from types import SimpleNamespace

from backend.execution.performance import _serialize_trade


def test_serialize_closed_trade_exposes_foregone_pnl():
    trade = SimpleNamespace(
        id=1,
        position_id=42,
        bucket_id=100,
        event_id=7,
        city_slug="atlanta",
        date_et="2026-05-13",
        bucket_idx=3,
        bucket_label="80-81°F",
        entry_type="MANUAL",
        entry_strategy="manual_hold_to_redeem",
        exit_level="EXPIRY",
        exit_reason="market_close",
        entry_time=datetime(2026, 5, 13, 17, 29, tzinfo=timezone.utc),
        exit_time=datetime(2026, 5, 13, 23, 49, tzinfo=timezone.utc),
        shares=2.0,
        avg_entry_price=0.95,
        avg_exit_price=0.895,
        fees=0.0,
        realized_pnl=-0.11,
        final_outcome="win",
        hold_to_redeem_value=2.0,
        foregone_pnl=0.21,
        hold_time_hours=6.33,
        entry_model_prob=0.9,
        entry_market_prob=0.95,
        entry_true_edge=0.02,
        exit_market_bid=0.995,
        exit_market_ask=1.0,
        station_mae_f=1.1,
        time_window="late_day",
    )

    row = _serialize_trade(trade)
    assert row["entry_strategy"] == "manual_hold_to_redeem"
    assert row["exit_reason"] == "market_close"
    assert row["foregone_pnl"] == 0.21
    assert row["final_outcome"] == "win"
