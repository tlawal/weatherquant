from backend.modeling.distribution import conditional_bucket_probabilities
from backend.modeling.settlement import (
    canonical_bucket_ranges,
    find_bucket_idx_for_value,
    round_temperature_half_up,
)


def test_canonical_bucket_ranges_apply_half_up_integer_settlement():
    raw = [(None, 68.0), (68.0, 69.0), (70.0, 71.0), (72.0, None)]
    assert canonical_bucket_ranges(raw) == [
        (None, 67.5),
        (67.5, 69.5),
        (69.5, 71.5),
        (71.5, None),
    ]


def test_canonical_bucket_ranges_preserve_already_contiguous_bounds():
    raw = [(72.0, 74.0), (74.0, 76.0), (76.0, 78.0)]
    assert canonical_bucket_ranges(raw) == raw


def test_exact_boundary_maps_to_prior_inclusive_bucket():
    canonical = canonical_bucket_ranges([(68.0, 69.0), (70.0, 71.0), (72.0, 73.0)])
    assert find_bucket_idx_for_value(canonical, 69.0) == 0
    assert find_bucket_idx_for_value(canonical, 70.0) == 1
    assert find_bucket_idx_for_value(canonical, 69.5) == 1


def test_half_degree_rounds_up_to_official_integer_bucket():
    canonical = canonical_bucket_ranges([
        (82.0, 83.0),
        (84.0, 85.0),
        (86.0, 87.0),
    ])

    assert round_temperature_half_up(84.4) == 84
    assert round_temperature_half_up(84.5) == 85
    assert round_temperature_half_up(84.6) == 85
    assert find_bucket_idx_for_value(canonical, 83.4) == 0
    assert find_bucket_idx_for_value(canonical, 83.5) == 1
    assert find_bucket_idx_for_value(canonical, 84.6) == 1


def test_conditioning_keeps_mass_inside_current_bucket():
    canonical = canonical_bucket_ranges([(68.0, 69.0), (70.0, 71.0), (72.0, 73.0), (74.0, None)])
    probs = conditional_bucket_probabilities(
        68.97117647058823,
        1.224,
        canonical,
        floor=68.0,
    )
    assert probs[0] > probs[1]
    assert probs[0] > 0.5
