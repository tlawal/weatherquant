"""
Canonical settlement interval helpers for temperature buckets.

Stored bucket bounds are display-oriented and come directly from parsed market
labels.  For Polymarket integer ranges like "68-69°F", the true settlement
interval is [68, 70), not [68, 69).  This module converts ordered bucket
bounds into non-overlapping, exclusive-upper intervals used for probability
math and outcome mapping.
"""
from __future__ import annotations

from typing import Optional, Sequence


BucketRange = tuple[Optional[float], Optional[float]]


def canonical_bucket_ranges(buckets: Sequence[BucketRange]) -> list[BucketRange]:
    """Return canonical exclusive-upper settlement intervals.

    Rules:
      - open-below buckets keep their parsed upper bound
      - finite buckets use the next bucket's lower bound when that reveals the
        actual settlement threshold (e.g. 68-69 followed by 70-71 => [68, 70))
      - already-contiguous buckets remain unchanged (e.g. 72-74, 74-76)
      - the final finite bucket falls back to +1 when the parsed bounds look
        like inclusive integer endpoints and no hotter threshold is available
    """
    normalized = [(float(lo) if lo is not None else None, float(hi) if hi is not None else None) for lo, hi in buckets]
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
