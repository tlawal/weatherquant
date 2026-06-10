"""Postgres size reporting and conservative retention maintenance.

The cleanup policy deliberately preserves trade, fill, position, event,
calibration, model-artifact, and wallet-skill history. It only targets bulky
high-frequency ephemera and raw payload blobs after normalized columns exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


PROTECTED_TABLES = (
    "cities",
    "events",
    "buckets",
    "orders",
    "fills",
    "positions",
    "closed_trades",
    "forecast_daily_errors",
    "station_profiles",
    "station_source_weights",
    "station_calibrations",
    "source_lead_time_skills",
    "threshold_calibrations",
    "live_bucket_calibrations",
    "bma_weights",
    "calibration_params",
    "model_artifacts",
    "wallet_stats",
    "wallet_market_exposures",
    "wallet_skill_scores",
)


@dataclass(frozen=True)
class RetentionPolicy:
    market_snapshot_days: int = 21
    market_flow_days: int = 14
    raw_payload_days: int = 30
    signal_days: int = 60
    prune_signals: bool = False
    batch_size: int = 5000

    def normalized(self) -> "RetentionPolicy":
        return RetentionPolicy(
            market_snapshot_days=max(3, min(int(self.market_snapshot_days), 365)),
            market_flow_days=max(1, min(int(self.market_flow_days), 365)),
            raw_payload_days=max(7, min(int(self.raw_payload_days), 365)),
            signal_days=max(14, min(int(self.signal_days), 365)),
            prune_signals=bool(self.prune_signals),
            batch_size=max(100, min(int(self.batch_size), 20000)),
        )


def _dialect(session: AsyncSession) -> str:
    bind = session.get_bind()
    return getattr(getattr(bind, "dialect", None), "name", "unknown")


def _cutoff(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


async def build_db_size_report(session: AsyncSession) -> dict[str, Any]:
    """Return per-table size/dead-tuple metadata for Postgres."""
    dialect = _dialect(session)
    if dialect != "postgresql":
        return {
            "supported": False,
            "dialect": dialect,
            "reason": "Postgres size metadata is only available on PostgreSQL.",
            "protected_tables": list(PROTECTED_TABLES),
        }

    rows = (await session.execute(text(
        """
        SELECT
            relname AS table_name,
            n_live_tup::bigint AS live_rows_est,
            n_dead_tup::bigint AS dead_rows_est,
            pg_total_relation_size(relid)::bigint AS total_bytes,
            pg_table_size(relid)::bigint AS table_bytes,
            pg_indexes_size(relid)::bigint AS index_bytes,
            pg_size_pretty(pg_total_relation_size(relid)) AS total_pretty,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY pg_total_relation_size(relid) DESC
        """
    ))).mappings().all()
    totals = {
        "total_bytes": sum(int(r["total_bytes"] or 0) for r in rows),
        "table_bytes": sum(int(r["table_bytes"] or 0) for r in rows),
        "index_bytes": sum(int(r["index_bytes"] or 0) for r in rows),
        "dead_rows_est": sum(int(r["dead_rows_est"] or 0) for r in rows),
    }
    return {
        "supported": True,
        "dialect": dialect,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "protected_tables": list(PROTECTED_TABLES),
        "tables": [dict(r) for r in rows],
    }


async def _count(session: AsyncSession, sql: str, params: dict[str, Any]) -> int:
    value = await session.scalar(text(sql), params)
    return int(value or 0)


async def _execute_batched(
    session: AsyncSession,
    sql: str,
    params: dict[str, Any],
    *,
    batch_size: int,
) -> int:
    total = 0
    while True:
        result = await session.execute(text(sql), {**params, "batch_size": batch_size})
        await session.commit()
        rowcount = result.rowcount
        affected = int(rowcount) if rowcount is not None and rowcount >= 0 else 0
        total += affected
        if affected < batch_size:
            break
    return total


async def run_retention_maintenance(
    session: AsyncSession,
    *,
    dry_run: bool = True,
    policy: RetentionPolicy | None = None,
) -> dict[str, Any]:
    """Run or dry-run safe retention maintenance.

    This intentionally does not delete normalized trade/performance/model
    history. It prunes or nulls high-volume operational data only.
    """
    dialect = _dialect(session)
    policy = (policy or RetentionPolicy()).normalized()
    if dialect != "postgresql":
        return {
            "supported": False,
            "dialect": dialect,
            "dry_run": dry_run,
            "reason": "Retention maintenance is Postgres-only in production.",
            "protected_tables": list(PROTECTED_TABLES),
            "policy": policy.__dict__,
            "actions": [],
        }

    now = datetime.now(timezone.utc)
    params = {
        "market_snapshot_cutoff": _cutoff(policy.market_snapshot_days),
        "market_flow_cutoff": _cutoff(policy.market_flow_days),
        "raw_payload_cutoff": _cutoff(policy.raw_payload_days),
        "signal_cutoff": _cutoff(policy.signal_days),
    }

    action_specs: list[dict[str, Any]] = [
        {
            "name": "delete_old_market_flow_features",
            "table": "market_flow_features",
            "mode": "delete",
            "count_sql": "SELECT count(*) FROM market_flow_features WHERE computed_at < :market_flow_cutoff",
            "exec_sql": """
                WITH doomed AS (
                    SELECT id FROM market_flow_features
                    WHERE computed_at < :market_flow_cutoff
                    ORDER BY computed_at ASC
                    LIMIT :batch_size
                )
                DELETE FROM market_flow_features WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"market_flow_cutoff": params["market_flow_cutoff"]},
            "rationale": "Shadow microstructure rows are high-frequency ephemera; older rows should be exported before long-term research, not kept in the hot DB.",
        },
        {
            "name": "delete_old_market_snapshots_for_inactive_events",
            "table": "market_snapshots",
            "mode": "delete",
            "count_sql": """
                SELECT count(*)
                FROM market_snapshots ms
                JOIN buckets b ON b.id = ms.bucket_id
                JOIN events e ON e.id = b.event_id
                WHERE ms.fetched_at < :market_snapshot_cutoff
                  AND e.date_et < to_char((now() AT TIME ZONE 'America/New_York')::date, 'YYYY-MM-DD')
                  AND NOT EXISTS (
                    SELECT 1 FROM positions p
                    WHERE p.bucket_id = ms.bucket_id AND COALESCE(p.net_qty, 0) > 0
                  )
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT ms.id
                    FROM market_snapshots ms
                    JOIN buckets b ON b.id = ms.bucket_id
                    JOIN events e ON e.id = b.event_id
                    WHERE ms.fetched_at < :market_snapshot_cutoff
                      AND e.date_et < to_char((now() AT TIME ZONE 'America/New_York')::date, 'YYYY-MM-DD')
                      AND NOT EXISTS (
                        SELECT 1 FROM positions p
                        WHERE p.bucket_id = ms.bucket_id AND COALESCE(p.net_qty, 0) > 0
                      )
                    ORDER BY ms.fetched_at ASC
                    LIMIT :batch_size
                )
                DELETE FROM market_snapshots WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"market_snapshot_cutoff": params["market_snapshot_cutoff"]},
            "rationale": "Book snapshots are useful short-horizon execution data; resolved inactive markets do not need unlimited hot snapshots.",
        },
        {
            "name": "null_old_metar_raw_payloads",
            "table": "metar_obs",
            "mode": "null_raw_payload",
            "count_sql": """
                SELECT count(*) FROM metar_obs
                WHERE fetched_at < :raw_payload_cutoff
                  AND (raw_json IS NOT NULL OR raw_text IS NOT NULL)
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT id FROM metar_obs
                    WHERE fetched_at < :raw_payload_cutoff
                      AND (raw_json IS NOT NULL OR raw_text IS NOT NULL)
                    ORDER BY fetched_at ASC
                    LIMIT :batch_size
                )
                UPDATE metar_obs
                SET raw_json = NULL, raw_text = NULL
                WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"raw_payload_cutoff": params["raw_payload_cutoff"]},
            "rationale": "Normalized temp/high fields and extended parsed fields remain; bulky raw source payloads leave the hot DB.",
        },
        {
            "name": "null_old_forecast_raw_payloads",
            "table": "forecast_obs",
            "mode": "null_raw_payload",
            "count_sql": """
                SELECT count(*) FROM forecast_obs
                WHERE fetched_at < :raw_payload_cutoff
                  AND (raw_json IS NOT NULL OR parse_error IS NOT NULL)
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT id FROM forecast_obs
                    WHERE fetched_at < :raw_payload_cutoff
                      AND (raw_json IS NOT NULL OR parse_error IS NOT NULL)
                    ORDER BY fetched_at ASC
                    LIMIT :batch_size
                )
                UPDATE forecast_obs
                SET raw_json = NULL, parse_error = NULL
                WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"raw_payload_cutoff": params["raw_payload_cutoff"]},
            "rationale": "Preserves source/date/model_run/high_f/raw_hash; removes payload text after normalization.",
        },
        {
            "name": "null_old_wallet_trade_raw_payloads",
            "table": "wallet_trades",
            "mode": "null_raw_payload",
            "count_sql": """
                SELECT count(*) FROM wallet_trades
                WHERE inserted_at < :raw_payload_cutoff
                  AND raw_json IS NOT NULL
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT id FROM wallet_trades
                    WHERE inserted_at < :raw_payload_cutoff
                      AND raw_json IS NOT NULL
                    ORDER BY inserted_at ASC
                    LIMIT :batch_size
                )
                UPDATE wallet_trades
                SET raw_json = NULL
                WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"raw_payload_cutoff": params["raw_payload_cutoff"]},
            "rationale": "Keeps normalized wallet flow/trade fields while shedding duplicated API payload blobs.",
        },
    ]

    if policy.prune_signals:
        action_specs.append(
            {
                "name": "delete_old_unreferenced_signals_for_inactive_events",
                "table": "signals",
                "mode": "delete_optional",
                "count_sql": """
                    SELECT count(*)
                    FROM signals s
                    JOIN buckets b ON b.id = s.bucket_id
                    JOIN events e ON e.id = b.event_id
                    WHERE s.computed_at < :signal_cutoff
                      AND e.date_et < to_char((now() AT TIME ZONE 'America/New_York')::date, 'YYYY-MM-DD')
                      AND NOT EXISTS (SELECT 1 FROM orders o WHERE o.signal_id = s.id)
                      AND NOT EXISTS (
                        SELECT 1 FROM positions p
                        WHERE p.bucket_id = s.bucket_id AND COALESCE(p.net_qty, 0) > 0
                      )
                """,
                "exec_sql": """
                    WITH doomed AS (
                        SELECT s.id
                        FROM signals s
                        JOIN buckets b ON b.id = s.bucket_id
                        JOIN events e ON e.id = b.event_id
                        WHERE s.computed_at < :signal_cutoff
                          AND e.date_et < to_char((now() AT TIME ZONE 'America/New_York')::date, 'YYYY-MM-DD')
                          AND NOT EXISTS (SELECT 1 FROM orders o WHERE o.signal_id = s.id)
                          AND NOT EXISTS (
                            SELECT 1 FROM positions p
                            WHERE p.bucket_id = s.bucket_id AND COALESCE(p.net_qty, 0) > 0
                          )
                        ORDER BY s.computed_at ASC
                        LIMIT :batch_size
                    )
                    DELETE FROM signals WHERE id IN (SELECT id FROM doomed)
                """,
                "params": {"signal_cutoff": params["signal_cutoff"]},
                "rationale": "Optional only. Latest signal generations and any order-linked history remain protected.",
            }
        )

    actions: list[dict[str, Any]] = []
    for spec in action_specs:
        count = await _count(session, spec["count_sql"], spec["params"])
        affected = count
        if not dry_run and count > 0:
            affected = await _execute_batched(
                session,
                spec["exec_sql"],
                spec["params"],
                batch_size=policy.batch_size,
            )
        actions.append(
            {
                "name": spec["name"],
                "table": spec["table"],
                "mode": spec["mode"],
                "dry_run": dry_run,
                "candidate_rows": count,
                "affected_rows": 0 if dry_run else affected,
                "rationale": spec["rationale"],
            }
        )

    return {
        "supported": True,
        "dialect": dialect,
        "dry_run": dry_run,
        "as_of": now.isoformat(),
        "policy": policy.__dict__,
        "protected_tables": list(PROTECTED_TABLES),
        "actions": actions,
        "notes": [
            "Run dry_run=true first and inspect candidate rows.",
            "After large deletes on Postgres, run VACUUM (ANALYZE) or wait for autovacuum to reclaim planner stats; disk pages may not return to the OS immediately.",
            "Export older market_snapshots/market_flow_features to cold storage before pruning if they are needed for long-horizon microstructure research.",
        ],
    }
