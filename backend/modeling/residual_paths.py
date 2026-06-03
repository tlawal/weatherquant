"""Filesystem paths for the residual ML model artifacts.

The default path stays in backend/modeling for local development. Production
should set RESIDUAL_MODEL_DIR to a persistent Railway volume mount such as
/data/models so promoted models survive image deploys.
"""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_DIR = Path(__file__).parent


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return default
    return Path(value).expanduser()


def residual_model_dir() -> Path:
    return _path_from_env("RESIDUAL_MODEL_DIR", DEFAULT_MODEL_DIR)


def residual_model_path() -> Path:
    return _path_from_env(
        "RESIDUAL_MODEL_PATH",
        residual_model_dir() / "residual_model.pkl",
    )


def residual_metadata_path() -> Path:
    return _path_from_env(
        "RESIDUAL_MODEL_META_PATH",
        residual_model_dir() / "residual_model_meta.json",
    )


def residual_shadow_model_path() -> Path:
    return _path_from_env(
        "RESIDUAL_SHADOW_MODEL_PATH",
        residual_model_dir() / "residual_model_shadow.pkl",
    )


def residual_shadow_metadata_path() -> Path:
    return _path_from_env(
        "RESIDUAL_SHADOW_MODEL_META_PATH",
        residual_model_dir() / "residual_model_shadow_meta.json",
    )
