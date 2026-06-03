import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.storage.db as storage_db
from backend.modeling.residual_artifacts import (
    RESIDUAL_ARTIFACT_NAME,
    hydrate_promoted_residual_artifact_from_db,
)
from backend.storage.models import Base
from backend.storage.repos import upsert_model_artifact


def _run(coro):
    return asyncio.run(coro)


async def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "residual_artifacts_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(storage_db, "_engine", engine)
    monkeypatch.setattr(storage_db, "_session_factory", session_factory)
    return engine, session_factory


def test_hydrate_promoted_residual_artifact_from_db(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    monkeypatch.setenv("RESIDUAL_MODEL_DIR", str(model_dir))
    engine, session_factory = _run(_setup_test_db(tmp_path, monkeypatch))

    async def run_test():
        async with session_factory() as session:
            await upsert_model_artifact(
                session,
                name=RESIDUAL_ARTIFACT_NAME,
                model_bytes=b"model-bytes",
                metadata_json='{"test_mae": 1.23}',
            )

        hydrated = await hydrate_promoted_residual_artifact_from_db()

        assert hydrated is True
        assert (model_dir / "residual_model.pkl").read_bytes() == b"model-bytes"
        assert (model_dir / "residual_model_meta.json").read_text() == '{"test_mae": 1.23}'

    _run(run_test())
    _run(engine.dispose())
