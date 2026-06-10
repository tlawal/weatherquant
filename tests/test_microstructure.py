from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.execution.microstructure import (
    book_imbalance,
    compute_shadow_flow_features,
    depth_within_cents,
    dynamic_trailing_distance,
    parse_book_levels,
    rolling_mid_volatility,
    simulate_fill,
)


def test_book_sweep_fill_uses_depth_weighted_average():
    asks = parse_book_levels(
        [
            {"price": "0.42", "size": "5"},
            {"price": "0.43", "size": "10"},
            {"price": "0.50", "size": "100"},
        ],
        side="ask",
    )

    fill = simulate_fill(asks, 10)

    assert fill.filled_size == 10
    assert fill.unfilled_size == 0
    assert fill.avg_price == 0.425
    assert fill.worst_price == 0.43


def test_depth_and_imbalance_features():
    bids = parse_book_levels(
        [
            {"price": 0.45, "size": 4},
            {"price": 0.44, "size": 6},
            {"price": 0.40, "size": 100},
        ],
        side="bid",
    )
    asks = parse_book_levels(
        [
            {"price": 0.46, "size": 2},
            {"price": 0.49, "size": 8},
        ],
        side="ask",
    )

    assert depth_within_cents(bids, side="bid", cents=1) == 10
    assert depth_within_cents(asks, side="ask", cents=3) == 10
    assert book_imbalance(10, 2) == 0.6667


def test_dynamic_trailing_widens_in_volatile_regime_and_tightens_after_tier2():
    calm = dynamic_trailing_distance(0.0, 0.0, tier2_exited=False)
    volatile = dynamic_trailing_distance(0.025, 0.8, tier2_exited=False)
    tier2 = dynamic_trailing_distance(0.025, 0.8, tier2_exited=True)

    assert calm == 0.05
    assert volatile > calm
    assert tier2 < volatile
    assert 0.03 <= tier2 <= 0.12


def test_rolling_mid_volatility_uses_consecutive_changes():
    now = datetime.now(timezone.utc)
    snaps = [
        SimpleNamespace(yes_mid=0.40, fetched_at=now - timedelta(minutes=3)),
        SimpleNamespace(yes_mid=0.41, fetched_at=now - timedelta(minutes=2)),
        SimpleNamespace(yes_mid=0.39, fetched_at=now - timedelta(minutes=1)),
        SimpleNamespace(yes_mid=0.42, fetched_at=now),
    ]

    assert rolling_mid_volatility(snaps) > 0


def test_shadow_flow_handles_empty_and_one_sided_flow():
    now = datetime.now(timezone.utc)
    empty = compute_shadow_flow_features([], as_of=now, window_minutes=5)
    assert empty["direction_source"] == "unavailable"
    assert empty["toxicity_score"] == 0.0

    trades = [
        SimpleNamespace(
            wallet_address="0xabc",
            side="BUY",
            size=10,
            price=0.40,
            notional=4.0,
            timestamp=now - timedelta(minutes=1),
        )
    ]
    one_sided = compute_shadow_flow_features(trades, as_of=now, window_minutes=5)

    assert one_sided["direction_source"] == "data_api_side"
    assert one_sided["signed_net_notional"] == 4.0
    assert one_sided["imbalance"] == 1.0
    assert one_sided["vpin"] == 1.0
