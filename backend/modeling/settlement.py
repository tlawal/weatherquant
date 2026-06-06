"""
Canonical settlement interval helpers for temperature buckets.

Stored bucket bounds are display-oriented and come directly from parsed market
labels.  For Polymarket integer temperature markets, the official value is an
integer temperature.  If the station/source reports decimals, values are
rounded half-up first: 84.5°F -> 85°F, 84.4°F -> 84°F.  A displayed bucket like
"84-85°F" therefore covers raw observations [83.5, 85.5), not [84, 86).

This module converts ordered bucket bounds into non-overlapping,
exclusive-upper raw-temperature intervals used for probability math and
outcome mapping.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence


BucketRange = tuple[Optional[float], Optional[float]]


def canonical_bucket_ranges(buckets: Sequence[BucketRange]) -> list[BucketRange]:
    """Return canonical exclusive-upper settlement intervals.

    Rules:
      - integer inclusive buckets use official half-up rounding semantics
        (84-85 => [83.5, 85.5), 86+ => [85.5, +inf))
      - open-below integer buckets end at the next bucket's half-up lower
        boundary
      - finite buckets use the next bucket's lower bound when that reveals the
        actual settlement threshold (e.g. 68-69 followed by 70-71 => [68, 70))
      - already-contiguous buckets remain unchanged (e.g. 72-74, 74-76)
      - the final finite bucket falls back to +1 when the parsed bounds look
        like inclusive integer endpoints and no hotter threshold is available
    """
    normalized = [(float(lo) if lo is not None else None, float(hi) if hi is not None else None) for lo, hi in buckets]
    if _looks_like_integer_settlement_buckets(normalized):
        return _half_up_integer_bucket_ranges(normalized)

    future_low: Optional[float] = None
    canonical_rev: list[BucketRange] = []

    for lo, hi in reversed(normalized):
        if hi is None:
            canonical_hi = None
        elif future_low is not None and future_low > hi:
            canonical_hi = future_low
        elif future_low is not None and future_low == hi:
            canonical_hi = hi
        elif lo is not None and _is_integerish(lo) and _is_integerish(hi) and (hi - lo) <= 1.0:
            canonical_hi = hi + 1.0
        else:
            canonical_hi = hi

        canonical_rev.append((lo, canonical_hi))
        if lo is not None:
            future_low = lo

    return list(reversed(canonical_rev))


def find_bucket_idx_for_value(buckets: Sequence[BucketRange], value: Optional[float]) -> Optional[int]:
    """Return the bucket index containing value under canonical semantics."""
    if value is None:
        return None
    for idx, (lo, hi) in enumerate(buckets):
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return idx
    return None


def bucket_upper_bound(buckets: Sequence[BucketRange], bucket_idx: Optional[int]) -> Optional[float]:
    if bucket_idx is None or bucket_idx < 0 or bucket_idx >= len(buckets):
        return None
    return buckets[bucket_idx][1]


def hotter_bucket_floor(buckets: Sequence[BucketRange], bucket_idx: Optional[int]) -> Optional[float]:
    if bucket_idx is None:
        return None
    hotter_idx = bucket_idx + 1
    if hotter_idx >= len(buckets):
        return None
    return buckets[hotter_idx][0]


def _is_integerish(value: float) -> bool:
    return abs(value - round(value)) < 1e-9


def round_temperature_half_up(value: Optional[float]) -> Optional[int]:
    """Round an official decimal temperature to integer settlement form.

    Python's built-in ``round`` is bankers rounding, so 84.5 would round to 84.
    Polymarket-style weather settlements use half-up integer semantics when
    the official source reports decimals.
    """
    if value is None:
        return None
    return int(math.floor(float(value) + 0.5))


def _looks_like_integer_settlement_buckets(buckets: Sequence[BucketRange]) -> bool:
    """Detect Polymarket-style integer-inclusive display buckets."""
    finite_integer_pairs = [
        (lo, hi)
        for lo, hi in buckets
        if lo is not None
        and hi is not None
        and _is_integerish(lo)
        and _is_integerish(hi)
    ]
    if not finite_integer_pairs:
        return False
    # 82-83, 84-85, etc. are integer settlement buckets. 72-74, 74-76 are
    # treated as already-contiguous continuous thresholds unless neighboring
    # gaps prove otherwise.
    return any((hi - lo) <= 1.0 for lo, hi in finite_integer_pairs)


def _half_up_integer_bucket_ranges(buckets: Sequence[BucketRange]) -> list[BucketRange]:
    """Convert integer-inclusive labels into raw half-up intervals."""
    out: list[BucketRange] = []
    for idx, (lo, hi) in enumerate(buckets):
        next_low = next(
            (
                future_lo
                for future_lo, _future_hi in buckets[idx + 1:]
                if future_lo is not None
            ),
            None,
        )
        canonical_lo: Optional[float]
        canonical_hi: Optional[float]

        if lo is None:
            canonical_lo = None
        elif _is_integerish(lo):
            canonical_lo = float(lo) - 0.5
        else:
            canonical_lo = lo

        if hi is None:
            canonical_hi = None
        elif next_low is not None and _is_integerish(next_low):
            canonical_hi = float(next_low) - 0.5
        elif _is_integerish(hi):
            canonical_hi = float(hi) + 0.5
        else:
            canonical_hi = hi

        out.append((canonical_lo, canonical_hi))
    return out
