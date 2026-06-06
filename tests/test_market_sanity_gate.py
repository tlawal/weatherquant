from backend.engine.market_sanity import evaluate_market_sanity, market_quality_weight


def test_market_sanity_blocks_high_model_edge_with_negative_posterior_edge():
    diag = evaluate_market_sanity(
        model_prob=0.55,
        market_prob=0.35,
        exec_cost=0.18,
        model_true_edge=0.12,
        market_snapshot_age_s=30,
        spread=0.02,
        bid_depth=50,
        ask_depth=50,
        min_true_edge=0.10,
        threshold_calibration_n=20,
    )

    assert diag["weight"] == 0.35
    assert diag["blocked"] is True
    assert diag["failure"].startswith("GATE_MARKET_SANITY:")


def test_market_sanity_thin_stale_market_does_not_block_on_gap_alone():
    diag = evaluate_market_sanity(
        model_prob=0.45,
        market_prob=0.05,
        exec_cost=0.02,
        model_true_edge=0.38,
        market_snapshot_age_s=600,
        spread=0.12,
        bid_depth=1,
        ask_depth=1,
        min_true_edge=0.10,
        threshold_calibration_n=0,
    )

    assert diag["weight"] == 0.0
    assert diag["gap"] > 0.20
    assert diag["gap_ok"] is True
    assert diag["blocked"] is False


def test_market_quality_weight_respects_depth_spread_age_thresholds():
    assert market_quality_weight(
        market_snapshot_age_s=45,
        spread=0.025,
        bid_depth=20,
        ask_depth=25,
    ) == 0.35
    assert market_quality_weight(
        market_snapshot_age_s=120,
        spread=0.055,
        bid_depth=10,
        ask_depth=10,
    ) == 0.15
    assert market_quality_weight(
        market_snapshot_age_s=301,
        spread=0.02,
        bid_depth=100,
        ask_depth=100,
    ) == 0.0
