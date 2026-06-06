import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.execution.performance import (
    _serialize_trade,
    exclude_oldest_closed_trades,
    get_closed_trade_rows,
    get_performance_summary,
    run_closed_trade_monte_carlo,
)
from backend.storage.models import Base, ClosedTrade


def _run(coro):
    return asyncio.run(coro)


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
        excluded_from_stats=False,
        excluded_reason=None,
        excluded_at=None,
    )

    row = _serialize_trade(trade)
    assert row["entry_strategy"] == "manual_hold_to_redeem"
    assert row["exit_reason"] == "market_close"
    assert row["foregone_pnl"] == 0.21
    assert row["final_outcome"] == "win"
    assert row["excluded_from_stats"] is False
    assert row["ledger_quality_ok"] is True


def test_serialize_closed_trade_flags_nonzero_pnl_with_zero_shares():
    trade = SimpleNamespace(
        id=2,
        position_id=43,
        bucket_id=101,
        event_id=8,
        city_slug="atlanta",
        date_et="2026-06-05",
        bucket_idx=4,
        bucket_label="84-85°F",
        entry_type="MANUAL",
        entry_strategy="manual_scalp",
        exit_level="PROFIT",
        exit_reason="tier_2_25pct",
        entry_time=datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 6, 5, 20, 0, tzinfo=timezone.utc),
        shares=0.0,
        avg_entry_price=0.65,
        avg_exit_price=0.87,
        fees=0.0,
        realized_pnl=1.0874,
        final_outcome="unknown",
        hold_to_redeem_value=None,
        foregone_pnl=None,
        hold_time_hours=4.0,
        entry_model_prob=None,
        entry_market_prob=None,
        entry_true_edge=None,
        exit_market_bid=0.88,
        exit_market_ask=0.92,
        station_mae_f=None,
        time_window=None,
        excluded_from_stats=True,
        excluded_reason="post_upgrade_baseline_reset",
        excluded_at=datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc),
    )

    row = _serialize_trade(trade)

    assert row["excluded_from_stats"] is True
    assert "nonzero_pnl_with_zero_shares" in row["ledger_quality_flags"]
    assert "unresolved_final_outcome" in row["ledger_quality_flags"]
    assert row["ledger_quality_ok"] is False


@pytest.mark.parametrize("include_excluded,expected_total,expected_n", [
    (False, 1.5, 2),
    (True, -8.5, 12),
])
def test_exclude_oldest_closed_trades_filters_summary_and_rows(tmp_path, include_excluded, expected_total, expected_n):
    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'perf.db'}")
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            for idx in range(12):
                pnl = -1.0 if idx < 10 else (1.0 if idx == 10 else 0.5)
                session.add(
                    ClosedTrade(
                        position_id=idx + 1,
                        bucket_id=idx + 1,
                        event_id=idx + 1,
                        city_slug="atlanta",
                        date_et=f"2026-05-{idx + 1:02d}",
                        bucket_idx=idx,
                        bucket_label=f"bucket {idx}",
                        exit_time=datetime(2026, 5, idx + 1, tzinfo=timezone.utc),
                        shares=1.0,
                        avg_entry_price=0.5,
                        avg_exit_price=0.5 + pnl,
                        realized_pnl=pnl,
                    )
                )
            await session.commit()

            result = await exclude_oldest_closed_trades(
                session,
                count=10,
                reason="post_upgrade_baseline_reset",
            )
            summary = await get_performance_summary(session, include_excluded=include_excluded)
            rows = await get_closed_trade_rows(session, limit=20, include_excluded=include_excluded)

        await engine.dispose()
        return result, summary, rows

    result, summary, rows = _run(scenario())

    assert result["excluded_count"] == 10
    assert result["excluded_ids"] == list(range(1, 11))
    assert summary["total_trades"] == expected_n
    assert summary["total_pnl"] == pytest.approx(expected_total)
    assert len(rows) == expected_n


def test_monte_carlo_serializes_when_active_baseline_has_no_losses(tmp_path):
    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'mc.db'}")
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            for idx, pnl in enumerate([1.0874, 0.4375], start=1):
                session.add(
                    ClosedTrade(
                        position_id=idx,
                        bucket_id=idx,
                        event_id=idx,
                        city_slug="atlanta",
                        date_et=f"2026-06-0{idx}",
                        bucket_idx=idx,
                        bucket_label=f"bucket {idx}",
                        exit_time=datetime(2026, 6, idx, tzinfo=timezone.utc),
                        shares=5.0,
                        avg_entry_price=0.5,
                        avg_exit_price=0.5 + pnl / 5.0,
                        realized_pnl=pnl,
                    )
                )
            await session.commit()

            result = await run_closed_trade_monte_carlo(
                session,
                paths=1000,
                horizon_trades=5,
                bankroll=10.0,
            )

        await engine.dispose()
        return result

    result = _run(scenario())

    assert result["sample"]["profit_factor"] == "inf"
    json.dumps(result, allow_nan=False)
