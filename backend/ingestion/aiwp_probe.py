"""Watch NOAA's AIWP S3 archive for new model prefixes (especially FCN3).

NOAA's AIWP open-data S3 bucket gets new AI weather models added periodically:
GraphCast and Pangu landed in 2024, FourCastNet v2-small followed, Aurora
arrived in 2025. FCN3 (FourCastNet v3) is the highest-leverage candidate
to land next; integrating it is a one-line addition to `AIWP_MODELS` in
`aiwp.py` once the S3 prefix appears.

Rather than manually checking the bucket monthly, this scheduled probe
lists top-level prefixes weekly and emits a WARNING log line if any new
prefix appears that isn't already in our `AIWP_MODELS` registry. The
operator (or a log alert) sees the warning and can flip the new model on
in one line.

Cheap to run: one HTTP request per week, no auth required, no compute.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Iterable

import aiohttp

from backend.ingestion.aiwp import AIWP_MODELS, AIWP_S3_BASE

log = logging.getLogger(__name__)


# Prefixes we already know about. These are the top-level S3 folder names
# (e.g. "PANG_v100_IFS", "FOUR_v200_GFS"). Any new prefix not in this set
# gets reported as a candidate for integration.
def _known_prefixes() -> set[str]:
    """Build the set of S3 prefixes our AIWP fetcher already handles.

    Each registered model has both GFS- and IFS-initialized variants on
    S3; we only fetch one variant per model but track both as known so
    we don't alert on the unused twin.
    """
    known: set[str] = set()
    for model_code, version, _ic in AIWP_MODELS.values():
        # Both initialization variants are on S3 even if we only fetch one.
        known.add(f"{model_code}_{version}_GFS")
        known.add(f"{model_code}_{version}_IFS")
    # Models we've seen on the bucket but don't currently integrate
    # (skipped in `aiwp.py` comments). Don't alert on these either.
    known.update({
        "FOUR_v100_GFS",       # FourCastNet v1 — superseded by v2-small
        "FOUR_v100_IFS",       # FourCastNet v1 — superseded
        "GRAP_v100_GFS",       # GraphCast — we have it via Open-Meteo instead
        "GRAP_v100_IFS",       # GraphCast — same
    })
    return known


# Regex matching valid AIWP top-level prefix shape: 4-letter model code,
# underscore, version (v\d+), underscore, IC source. Filters out S3
# bucket directories that aren't model folders ("Derived/", "parquet/",
# "colab_resources/" etc).
_PREFIX_RE = re.compile(r"^[A-Z]{4}_v\d{3}_(GFS|IFS)/$")


async def list_aiwp_top_level_prefixes(timeout_s: int = 30) -> list[str]:
    """List S3 top-level prefixes under the AIWP bucket via the
    REST `?delimiter=/` endpoint. No auth required; bucket is public.

    Returns a list of prefixes shaped like 'PANG_v100_IFS/' (note the
    trailing slash). Caller filters as needed.
    """
    url = f"{AIWP_S3_BASE}/?delimiter=/"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            body = await resp.text()
    # Quick-and-dirty XML parse — S3 ListBucket V1 response has each
    # CommonPrefix as `<Prefix>NAME/</Prefix>`.
    return re.findall(r"<Prefix>([^<]+)</Prefix>", body)


async def probe_aiwp_for_new_models() -> dict:
    """Probe S3 once and report any prefix shape we don't already handle.

    Returns:
        dict with keys:
            - all_prefixes: every top-level S3 folder we found
            - known: subset already in AIWP_MODELS or explicit-skip list
            - candidates: shape-matching new prefixes worth integrating
              (e.g. 'FOUR_v300_IFS' if NOAA adds FCN3)
            - other: prefixes that didn't match the model-folder shape
              (e.g. 'Derived/', 'parquet/' — informational only)

    Logs WARNING when `candidates` is non-empty so a log-tail alert can
    trigger a notification.
    """
    try:
        all_prefixes = await list_aiwp_top_level_prefixes()
    except Exception as e:
        log.warning("aiwp_probe: S3 list failed: %s", e)
        return {"error": str(e), "candidates": []}

    # Strip trailing slashes for comparison.
    all_stripped = [p.rstrip("/") for p in all_prefixes]
    known = _known_prefixes()

    candidates: list[str] = []
    other: list[str] = []
    known_seen: list[str] = []

    for prefix in all_stripped:
        prefix_with_slash = f"{prefix}/"
        if not _PREFIX_RE.match(prefix_with_slash):
            other.append(prefix)
            continue
        if prefix in known:
            known_seen.append(prefix)
        else:
            candidates.append(prefix)

    if candidates:
        # Most actionable line in this whole module — log at WARNING so
        # it surfaces in normal log-tailing without per-deploy filter
        # configuration. Operator (or any log-based alert) treats this
        # as: "go integrate the new model in aiwp.py:AIWP_MODELS".
        log.warning(
            "aiwp_probe: NEW MODEL PREFIX detected on NOAA S3 — candidates=%s. "
            "Integrate by adding to AIWP_MODELS in backend/ingestion/aiwp.py",
            candidates,
        )
    else:
        log.info(
            "aiwp_probe: no new model prefixes (known=%d, total=%d)",
            len(known_seen), len(all_stripped),
        )

    return {
        "all_prefixes": all_stripped,
        "known": known_seen,
        "candidates": candidates,
        "other": other,
    }


# ── CLI for one-shot manual probe ──────────────────────────────────────────

if __name__ == "__main__":
    """Run from a shell to manually probe S3 right now:

        python -m backend.ingestion.aiwp_probe
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = asyncio.run(probe_aiwp_for_new_models())
    print()
    print("=== AIWP probe result ===")
    print(f"All prefixes ({len(result.get('all_prefixes', []))}):")
    for p in sorted(result.get("all_prefixes", [])):
        print(f"  {p}")
    if result.get("candidates"):
        print(f"\n>>> NEW CANDIDATES ({len(result['candidates'])}): "
              f"{result['candidates']}")
    else:
        print("\nNo new model candidates.")
