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
    "wallet_skill_scores",
)


@dataclass(frozen=True)
class RetentionPolicy:
    market_snapshot_days: int = 21
    market_flow_days: int = 14
    raw_payload_days: int = 30
    signal_days: int = 60
    forecast_obs_days: int = 180
    wallet_trade_days: int = 120
    wallet_exposure_days: int = 45
    model_input_days: int = 14
    prune_signals: bool = False
    batch_size: int = 5000

    def normalized(self) -> "RetentionPolicy":
        return RetentionPolicy(
            market_snapshot_days=max(3, min(int(self.market_snapshot_days), 365)),
            market_flow_days=max(1, min(int(self.market_flow_days), 365)),
            raw_payload_days=max(7, min(int(self.raw_payload_days), 365)),
            signal_days=max(14, min(int(self.signal_days), 365)),
            forecast_obs_days=max(90, min(int(self.forecast_obs_days), 730)),
            wallet_trade_days=max(30, min(int(self.wallet_trade_days), 730)),
            wallet_exposure_days=max(14, min(int(self.wallet_exposure_days), 365)),
            model_input_days=max(7, min(int(self.model_input_days), 365)),
            prune_signals=bool(self.prune_signals),
            batch_size=max(100, min(int(self.batch_size), 20000)),
        )


@dataclass(frozen=True)
class ColdExportSpec:
    table: str
    cutoff_column: str
    cutoff_kind: str
    rationale: str


ALLOWED_COLD_EXPORTS: dict[str, ColdExportSpec] = {
    "market_snapshots": ColdExportSpec(
        table="market_snapshots",
        cutoff_column="fetched_at",
        cutoff_kind="timestamp",
        rationale="CLOB depth/price history for execution and VPIN-style research.",
    ),
    "market_flow_features": ColdExportSpec(
        table="market_flow_features",
        cutoff_column="computed_at",
        cutoff_kind="timestamp",
        rationale="Shadow flow/imbalance/toxicity feature history for out-of-sample validation.",
    ),
    "wallet_trades": ColdExportSpec(
        table="wallet_trades",
        cutoff_column="trade_ts",
        cutoff_kind="timestamp",
        rationale="Normalized public wallet-trade firehose for longer-window smart-money backtests.",
    ),
    "wallet_market_exposures": ColdExportSpec(
        table="wallet_market_exposures",
        cutoff_column="date",
        cutoff_kind="date",
        rationale="Derived wallet exposure state for research snapshots.",
    ),
    "forecast_obs": ColdExportSpec(
        table="forecast_obs",
        cutoff_column="date_et",
        cutoff_kind="date",
        rationale="Raw forecast observations beyond the hot rolling skill window.",
    ),
}


def get_cold_export_spec(table: str) -> ColdExportSpec:
    key = (table or "").strip().lower()
    try:
        return ALLOWED_COLD_EXPORTS[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(ALLOWED_COLD_EXPORTS))
        raise ValueError(f"Unsupported cold export table {table!r}; allowed: {allowed}") from exc


def build_cold_export_metadata(
    table: str,
    *,
    days: int,
    limit_rows: int | None = None,
    batch_size: int = 5000,
) -> dict[str, Any]:
    """Validate and describe a hot-DB cold export request."""
    spec = get_cold_export_spec(table)
    days_clamped = max(1, min(int(days), 3650))
    batch_clamped = max(100, min(int(batch_size), 20000))
    row_limit = None if limit_rows is None else max(1, min(int(limit_rows), 10_000_000))
    cutoff_dt = _cutoff(days_clamped)
    cutoff_value: Any
    if spec.cutoff_kind == "date":
        cutoff_value = cutoff_dt.strftime("%Y-%m-%d")
    else:
        cutoff_value = cutoff_dt
    return {
        "table": spec.table,
        "cutoff_column": spec.cutoff_column,
        "cutoff_kind": spec.cutoff_kind,
        "cutoff_value": cutoff_value,
        "days": days_clamped,
        "limit_rows": row_limit,
        "batch_size": batch_clamped,
        "rationale": spec.rationale,
    }


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
    database_bytes = int(
        await session.scalar(text("SELECT pg_database_size(current_database())")) or 0
    )
    wal_bytes = 0
    try:
        wal_bytes = int(
            await session.scalar(text("SELECT COALESCE(sum(size), 0) FROM pg_ls_waldir()"))
            or 0
        )
    except Exception:
        wal_bytes = 0
    totals["database_bytes"] = database_bytes
    totals["wal_bytes"] = wal_bytes
    totals["hot_store_bytes"] = database_bytes + wal_bytes
    return {
        "supported": True,
        "dialect": dialect,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "protected_tables": list(PROTECTED_TABLES),
        "tables": [dict(r) for r in rows],
    }


def evaluate_db_storage_alerts(
    size_report: dict[str, Any],
    *,
    volume_limit_mb: int = 5000,
    volume_alert_pct: float = 0.70,
    top_table_alert_mb: int = 750,
    table_growth_alert_mb: int = 100,
    previous_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate DB storage alerts from a size report and optional prior snapshot."""
    if not size_report.get("supported"):
        return {
            "supported": False,
            "alerts": [],
            "snapshot": None,
            "reason": size_report.get("reason", "unsupported"),
        }

    totals = size_report.get("totals") or {}
    volume_limit_bytes = max(1, int(volume_limit_mb)) * 1024 * 1024
    hot_store_bytes = int(
        totals.get("hot_store_bytes")
        or totals.get("database_bytes")
        or totals.get("total_bytes")
        or 0
    )
    usage_pct = hot_store_bytes / volume_limit_bytes
    tables = size_report.get("tables") or []
    top_table = tables[0] if tables else None
    top_table_name = top_table.get("table_name") if top_table else None
    top_table_bytes = int(top_table.get("total_bytes") or 0) if top_table else 0
    table_bytes = {
        str(row.get("table_name")): int(row.get("total_bytes") or 0)
        for row in tables
        if row.get("table_name")
    }

    alerts: list[dict[str, Any]] = []
    if usage_pct >= float(volume_alert_pct):
        alerts.append(
            {
                "level": "critical" if usage_pct >= 0.90 else "warning",
                "type": "volume_usage",
                "message": (
                    f"Postgres hot store is {usage_pct:.1%} of configured "
                    f"{int(volume_limit_mb)} MB limit"
                ),
                "usage_pct": round(usage_pct, 4),
                "hot_store_bytes": hot_store_bytes,
                "volume_limit_bytes": volume_limit_bytes,
            }
        )

    top_table_threshold = max(1, int(top_table_alert_mb)) * 1024 * 1024
    if top_table and top_table_bytes >= top_table_threshold:
        alerts.append(
            {
                "level": "warning",
                "type": "top_table_size",
                "message": (
                    f"Top table {top_table_name} is "
                    f"{top_table_bytes / 1024 / 1024:.1f} MB"
                ),
                "table": top_table_name,
                "table_bytes": top_table_bytes,
                "threshold_bytes": top_table_threshold,
            }
        )

    previous_tables = (previous_snapshot or {}).get("table_bytes") or {}
    growth_threshold = max(1, int(table_growth_alert_mb)) * 1024 * 1024
    growth_rows = []
    for table_name, current_bytes in table_bytes.items():
        previous_bytes = int(previous_tables.get(table_name) or 0)
        growth = current_bytes - previous_bytes
        if previous_bytes > 0 and growth >= growth_threshold:
            growth_rows.append((table_name, growth, current_bytes, previous_bytes))
    for table_name, growth, current_bytes, previous_bytes in sorted(
        growth_rows,
        key=lambda item: item[1],
        reverse=True,
    )[:5]:
        alerts.append(
            {
                "level": "warning",
                "type": "table_growth",
                "message": (
                    f"Table {table_name} grew "
                    f"{growth / 1024 / 1024:.1f} MB since last storage snapshot"
                ),
                "table": table_name,
                "growth_bytes": growth,
                "current_bytes": current_bytes,
                "previous_bytes": previous_bytes,
                "threshold_bytes": growth_threshold,
            }
        )

    snapshot = {
        "as_of": size_report.get("as_of"),
        "hot_store_bytes": hot_store_bytes,
        "database_bytes": int(totals.get("database_bytes") or 0),
        "wal_bytes": int(totals.get("wal_bytes") or 0),
        "volume_limit_bytes": volume_limit_bytes,
        "usage_pct": round(usage_pct, 6),
        "top_table": top_table_name,
        "top_table_bytes": top_table_bytes,
        "table_bytes": table_bytes,
    }
    return {
        "supported": True,
        "alerts": alerts,
        "snapshot": snapshot,
        "usage_pct": round(usage_pct, 6),
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
        "forecast_obs_cutoff_date": (
            now - timedelta(days=policy.forecast_obs_days)
        ).strftime("%Y-%m-%d"),
        "wallet_trade_cutoff": _cutoff(policy.wallet_trade_days),
        "wallet_exposure_cutoff_date": (
            now - timedelta(days=policy.wallet_exposure_days)
        ).strftime("%Y-%m-%d"),
        "model_input_cutoff": _cutoff(policy.model_input_days),
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
        {
            "name": "null_old_model_snapshot_inputs",
            "table": "model_snapshots",
            "mode": "null_debug_payload",
            "count_sql": """
                SELECT count(*)
                FROM model_snapshots ms
                JOIN events e ON e.id = ms.event_id
                WHERE ms.computed_at < :model_input_cutoff
                  AND ms.inputs_json IS NOT NULL
                  AND e.date_et < to_char((now() AT TIME ZONE 'America/New_York')::date, 'YYYY-MM-DD')
                  AND NOT EXISTS (
                    SELECT 1 FROM buckets b
                    JOIN positions p ON p.bucket_id = b.id
                    WHERE b.event_id = ms.event_id AND COALESCE(p.net_qty, 0) > 0
                  )
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT ms.id
                    FROM model_snapshots ms
                    JOIN events e ON e.id = ms.event_id
                    WHERE ms.computed_at < :model_input_cutoff
                      AND ms.inputs_json IS NOT NULL
                      AND e.date_et < to_char((now() AT TIME ZONE 'America/New_York')::date, 'YYYY-MM-DD')
                      AND NOT EXISTS (
                        SELECT 1 FROM buckets b
                        JOIN positions p ON p.bucket_id = b.id
                        WHERE b.event_id = ms.event_id AND COALESCE(p.net_qty, 0) > 0
                      )
                    ORDER BY ms.computed_at ASC
                    LIMIT :batch_size
                )
                UPDATE model_snapshots
                SET inputs_json = NULL
                WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"model_input_cutoff": params["model_input_cutoff"]},
            "rationale": "Keeps timestamp, mu, sigma, probability vector, and quality while removing old debug input JSON from inactive markets.",
        },
        {
            "name": "delete_old_forecast_obs_beyond_skill_window",
            "table": "forecast_obs",
            "mode": "delete_rolling_history",
            "count_sql": """
                SELECT count(*)
                FROM forecast_obs
                WHERE date_et < :forecast_obs_cutoff_date
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT id
                    FROM forecast_obs
                    WHERE date_et < :forecast_obs_cutoff_date
                    ORDER BY date_et ASC, fetched_at ASC
                    LIMIT :batch_size
                )
                DELETE FROM forecast_obs WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"forecast_obs_cutoff_date": params["forecast_obs_cutoff_date"]},
            "rationale": "Lead-time skill and live calibration use rolling windows; aggregate error tables preserve older learning while the hot DB keeps the most relevant normalized forecasts.",
        },
        {
            "name": "delete_old_wallet_trades_beyond_skill_window",
            "table": "wallet_trades",
            "mode": "delete_rolling_history",
            "count_sql": """
                SELECT count(*)
                FROM wallet_trades
                WHERE trade_ts < :wallet_trade_cutoff
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT id
                    FROM wallet_trades
                    WHERE trade_ts < :wallet_trade_cutoff
                    ORDER BY trade_ts ASC
                    LIMIT :batch_size
                )
                DELETE FROM wallet_trades WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"wallet_trade_cutoff": params["wallet_trade_cutoff"]},
            "rationale": "Wallet skill is finite-window and aggregate WalletStat/WalletSkillScore rows preserve the signal; stale public trade firehose rows are not production-critical.",
        },
        {
            "name": "delete_old_wallet_market_exposures",
            "table": "wallet_market_exposures",
            "mode": "delete_rolling_exposure",
            "count_sql": """
                SELECT count(*)
                FROM wallet_market_exposures
                WHERE date < :wallet_exposure_cutoff_date
            """,
            "exec_sql": """
                WITH doomed AS (
                    SELECT id
                    FROM wallet_market_exposures
                    WHERE date < :wallet_exposure_cutoff_date
                    ORDER BY date ASC, last_updated_ts ASC
                    LIMIT :batch_size
                )
                DELETE FROM wallet_market_exposures WHERE id IN (SELECT id FROM doomed)
            """,
            "params": {"wallet_exposure_cutoff_date": params["wallet_exposure_cutoff_date"]},
            "rationale": "Exposure rows are point-in-time wallet state. Current/recent rows inform shadow flow; old exposure state can be regenerated from retained trade windows or aggregates.",
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
