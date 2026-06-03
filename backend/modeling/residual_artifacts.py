"""Persist and hydrate residual ML model artifacts through Postgres.

This uses the existing Postgres volume instead of requiring a filesystem
volume on the app service. Runtime startup hydrates the DB artifact to the
local model path so the prediction path remains synchronous and cheap.
"""
from __future__ import annotations

import logging

from backend.modeling.residual_paths import residual_metadata_path, residual_model_path

log = logging.getLogger(__name__)

RESIDUAL_ARTIFACT_NAME = "residual_remaining_rise"


async def save_promoted_residual_artifact_to_db() -> bool:
    """Save the promoted residual model file and metadata into Postgres."""
    from backend.storage.db import get_session, init_db
    from backend.storage.repos import upsert_model_artifact

    model_path = residual_model_path()
    metadata_path = residual_metadata_path()
    if not model_path.exists():
        log.warning("residual_artifact: promoted model file missing at %s", model_path)
        return False

    await init_db()
    metadata_json = metadata_path.read_text() if metadata_path.exists() else None
    async with get_session() as sess:
        await upsert_model_artifact(
            sess,
            name=RESIDUAL_ARTIFACT_NAME,
            model_bytes=model_path.read_bytes(),
            metadata_json=metadata_json,
        )
    log.info("residual_artifact: saved promoted model artifact to Postgres")
    return True


async def hydrate_promoted_residual_artifact_from_db() -> bool:
    """Write the promoted residual model from Postgres to the runtime path."""
    from backend.storage.db import get_session
    from backend.storage.repos import get_model_artifact

    async with get_session() as sess:
        artifact = await get_model_artifact(sess, RESIDUAL_ARTIFACT_NAME)
    if artifact is None:
        log.info("residual_artifact: no promoted model artifact found in Postgres")
        return False

    model_path = residual_model_path()
    metadata_path = residual_metadata_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(bytes(artifact.model_bytes))
    if artifact.metadata_json:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(str(artifact.metadata_json))
    try:
        from backend.modeling.residual_tracker import reset_model_cache

        reset_model_cache()
    except Exception:
        log.exception("residual_artifact: failed to reset residual model cache")
    log.info(
        "residual_artifact: hydrated promoted model artifact from Postgres to %s",
        model_path,
    )
    return True
