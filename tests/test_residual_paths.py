from pathlib import Path

from backend.modeling import residual_paths


def test_residual_model_dir_defaults_to_modeling_package(monkeypatch):
    monkeypatch.delenv("RESIDUAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("RESIDUAL_MODEL_PATH", raising=False)

    assert residual_paths.residual_model_dir() == residual_paths.DEFAULT_MODEL_DIR
    assert residual_paths.residual_model_path() == (
        residual_paths.DEFAULT_MODEL_DIR / "residual_model.pkl"
    )


def test_residual_model_dir_can_point_to_persistent_volume(monkeypatch):
    monkeypatch.setenv("RESIDUAL_MODEL_DIR", "/data/models")
    monkeypatch.delenv("RESIDUAL_MODEL_PATH", raising=False)
    monkeypatch.delenv("RESIDUAL_MODEL_META_PATH", raising=False)

    assert residual_paths.residual_model_path() == Path("/data/models/residual_model.pkl")
    assert residual_paths.residual_metadata_path() == Path("/data/models/residual_model_meta.json")


def test_residual_model_path_override_wins(monkeypatch):
    monkeypatch.setenv("RESIDUAL_MODEL_DIR", "/data/models")
    monkeypatch.setenv("RESIDUAL_MODEL_PATH", "/tmp/custom.pkl")

    assert residual_paths.residual_model_path() == Path("/tmp/custom.pkl")
