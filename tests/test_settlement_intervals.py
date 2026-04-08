from backend.modeling.distribution import conditional_bucket_probabilities
from backend.modeling.settlement import canonical_bucket_ranges, find_bucket_idx_for_value


def test_canonical_bucket_ranges_fill_inclusive_integer_gaps():
    raw = [(None, 68.0), (68.0, 69.0), (70.0, 71.0), (72.0, None)]
    assert canonical_bucket_ranges(raw) == [
        (None, 68.0),
        (68.0, 70.0),
        (70.0, 72.0),
        (72.0, None),
    ]


def test_canonical_bucket_ranges_preserve_already_contiguous_bounds():
    raw = [(72.0, 74.0), (74.0, 76.0), (76.0, 78.0)]
    assert canonical_bucket_ranges(raw) == raw


def test_exact_boundary_maps_to_prior_inclusive_bucket():
    canonical = canonical_bucket_ranges([(68.0, 69.0), (70.0, 71.0), (72.0, 73.0)])
    assert find_bucket_idx_for_value(canonical, 69.0) == 0
    assert find_bucket_idx_for_value(canonical, 70.0) == 1


def test_conditioning_keeps_mass_inside_current_bucket():
    canonical = canonical_bucket_ranges([(68.0, 69.0), (70.0, 71.0), (72.0, 73.0), (74.0, None)])
    probs = conditional_bucket_probabilities(
        68.97117647058823,
        1.224,
        canonical,
        floor=69.0,
    )
    assert probs[0] > probs[1]
    assert probs[0] > 0.5
