"""Backfill events.resolution_station_id by re-parsing Polymarket Gamma events.

For every Event with gamma_event_id (or gamma_slug) and a NULL or stale
resolution_station_id, refetch the Gamma event payload and re-run
_extract_resolution_url(). The parser was extended to also recognize
Weather Underground station-keyed URLs and a small alias map of US
airport names → ICAO codes (e.g. "William P. Hobby" → KHOU), so events
that previously slipped through with NULL station IDs can now be
populated without waiting for the next ingest cycle.

Idempotent — running it twice produces the same result.

Usage:
    python -m backend.scripts.backfill_resolution_stations --dry-run
    python -m backend.scripts.backfill_resolution_stations --commit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

import aiohttp
from sqlalchemy import select

from backend.ingestion.polymarket_gamma import (
    GAMMA_API,
    _HEADERS,
    _TIMEOUT,
    _extract_resolution_url,
)
from backend.storage.db import get_session
from backend.storage.models import City, Event

log = logging.getLogger(__name__)


async def _fetch_event_payload(
    http: aiohttp.ClientSession, event: Event
) -> Optional[dict]:
    """Try gamma_event_id first, then fall back to gamma_slug."""
    if event.gamma_event_id:
        url = f"{GAMMA_API}/events/{event.gamma_event_id}"
        try:
            async with http.get(url) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
        except Exception as e:
            log.debug("backfill: gamma_event_id fetch failed for %s: %s", event.id, e)

    if event.gamma_slug:
        url = f"{GAMMA_API}/events/slug/{event.gamma_slug}"
        try:
            async with http.get(url) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
        except Exception as e:
            log.debug("backfill: gamma_slug fetch failed for %s: %s", event.id, e)

    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Persist the resolved station IDs (default is dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing.",
    )
    parser.add_argument(
        "--include-overrides",
        action="store_true",
        help=(
            "Re-parse and overwrite even when resolution_station_id is "
            "already set. Use this after extending the parser/alias map."
        ),
    )
    args = parser.parse_args()

    commit = args.commit and not args.dry_run

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from backend.storage.db import init_db
    await init_db()

    async with get_session() as sess:
        stmt = select(Event).where(Event.gamma_event_id.isnot(None))
        if not args.include_overrides:
            stmt = stmt.where(Event.resolution_station_id.is_(None))
        result = await sess.execute(stmt)
        events: list[Event] = list(result.scalars().all())

        city_rows = await sess.execute(select(City))
        cities_by_id: dict[int, City] = {c.id: c for c in city_rows.scalars().all()}

    log.info(
        "backfill: %d candidate events (include_overrides=%s, commit=%s)",
        len(events),
        args.include_overrides,
        commit,
    )

    if not events:
        return

    summary: list[tuple[str, str, Optional[str], Optional[str]]] = []
    changed = 0

    async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_HEADERS) as http:
        for event in events:
            city = cities_by_id.get(event.city_id)
            slug = city.city_slug if city else f"city_id={event.city_id}"

            payload = await _fetch_event_payload(http, event)
            if not payload:
                summary.append((slug, event.date_et, event.resolution_station_id, "FETCH_FAILED"))
                continue

            url, station = _extract_resolution_url(payload)
            before = event.resolution_station_id
            after = (station or "").upper() or None

            summary.append((slug, event.date_et, before, after))

            if after and after != before:
                changed += 1
                if commit:
                    async with get_session() as write_sess:
                        write_event = await write_sess.get(Event, event.id)
                        if write_event is not None:
                            write_event.resolution_station_id = after
                            if url and not write_event.resolution_source_url:
                                write_event.resolution_source_url = url[:500]
                            await write_sess.commit()

            await asyncio.sleep(0.1)  # be polite

    print(f"\nbackfill summary ({len(summary)} events, {changed} changes):")
    print(f"{'city':<14} {'date':<12} {'before':<8} {'after':<8}")
    for slug, date_et, before, after in summary:
        print(f"{slug:<14} {date_et:<12} {str(before or '—'):<8} {str(after or '—'):<8}")

    if not commit:
        print("\nDry run — pass --commit to persist.")


if __name__ == "__main__":
    asyncio.run(main())
