"""Unit tests for the EDGE_DECAY exit gate (Phase A2).

Covers the pure logic of `_edge_decay_triggered` plus in-memory cache
trimming. The DB-level integration (warming the cache from EVHistory rows
and persisting on each cycle) is exercised by the cascade integration
tests in Phase A6.
"""
from __future__ import annotations

import pytest

import backend.execution.exit_engine as ee
from backend.config import Config


@pytest.fixture(autouse=True)
def _reset_ev_cache():
    """Each test gets a clean in-memory EV cache."""
    saved = ee._ev_cache.copy()
    ee._ev_cache.clear()
    yield
    ee._ev_cache.clear()
    ee._ev_cache.update(saved)


# ── _edge_decay_triggered: pure-logic tests ──────────────────────────────

def test_edge_decay_triggered_false_when_history_empty():
    assert ee._edge_decay_triggered(bucket_id=42) is False


def test_edge_decay_triggered_false_when_below_debounce_count():
    """A single negative observation isn't enough — must persist for N runs."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[42] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * (n - 1)
    assert ee._edge_decay_triggered(bucket_id=42) is False


def test_edge_decay_triggered_true_when_last_n_all_below_threshold():
    """N consecutive observations at or below threshold trigger the gate."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[42] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n
    assert ee._edge_decay_triggered(bucket_id=42) is True


def test_edge_decay_triggered_true_at_exact_threshold():
    """`<=` boundary: an observation equal to the threshold counts as decayed."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[42] = [Config.EDGE_DECAY_THRESHOLD] * n
    assert ee._edge_decay_triggered(bucket_id=42) is True


def test_edge_decay_one_recent_rebound_suppresses_trigger():
    """A single positive observation in the trailing window resets the gate."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    # First N-1 are bad, but the most recent one rebounds above threshold.
    hist = [Config.EDGE_DECAY_THRESHOLD - 0.01] * (n - 1) + [0.05]
    ee._ev_cache[42] = hist
    assert ee._edge_decay_triggered(bucket_id=42) is False


def test_edge_decay_only_evaluates_trailing_window():
    """Old positive observations don't immunize against a new run of negatives."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    # Older positives, then N consecutive negatives at the tail.
    hist = [0.10, 0.08, 0.05] + [Config.EDGE_DECAY_THRESHOLD - 0.01] * n
    ee._ev_cache[42] = hist
    assert ee._edge_decay_triggered(bucket_id=42) is True


def test_edge_decay_isolated_per_bucket():
    """Cache is keyed by bucket_id — one bucket's history can't leak into another."""
    n = Config.EDGE_DECAY_DEBOUNCE_RUNS
    ee._ev_cache[1] = [Config.EDGE_DECAY_THRESHOLD - 0.01] * n
    ee._ev_cache[2] = [0.10] * n
    assert ee._edge_decay_triggered(bucket_id=1) is True
    assert ee._edge_decay_triggered(bucket_id=2) is False
    assert ee._edge_decay_triggered(bucket_id=999) is False  # untracked bucket
