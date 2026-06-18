"""Model registry: model catalog driven by model_registry.yaml.

Replaces the hard-coded catalog in templates.py. A model is "available" once
its weight files are on disk under MODELS_DIR.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .domain import ModelOption
from .paths import LOCAL_QWEN_0_5B, MODEL_REGISTRY_FILE, model_dir_for_repo


@lru_cache(maxsize=1)
def _registry_entries() -> list[dict[str, object]]:
    data = yaml.safe_load(MODEL_REGISTRY_FILE.read_text(encoding="utf-8"))
    return list(data.get("models", []))


def _model_present(model_dir: Path) -> bool:
    """A model is usable once config.json and a weight file are on disk."""
    if not (model_dir / "config.json").exists():
        return False
    return any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").exists()


def _resolve_path(repo_id: str) -> tuple[Path | None, bool]:
    cache_dir = model_dir_for_repo(repo_id)
    # Backward compat: the originally cached 0.5B may sit at LOCAL_QWEN_0_5B.
    if not _model_present(cache_dir) and _model_present(LOCAL_QWEN_0_5B):
        if repo_id.endswith("Qwen3.5-0.8B"):
            return LOCAL_QWEN_0_5B, True
    available = _model_present(cache_dir)
    return (cache_dir if available else None), available


def get_model_catalog() -> list[ModelOption]:
    catalog: list[ModelOption] = []
    for entry in _registry_entries():
        repo_id = str(entry["repo_id"])
        local_path, available = _resolve_path(repo_id)
        catalog.append(
            ModelOption(
                id=str(entry["id"]),
                name=str(entry["name"]),
                parameter_count=str(entry["parameter_count"]),
                learning_value=str(entry["learning_value"]),
                lf_template=str(entry["lf_template"]),
                local_path=str(local_path) if local_path else None,
                available=available,
                recommended=bool(entry.get("recommended", False)),
                repo_id=repo_id,
                download_size_label=entry.get("download_size_label"),  # type: ignore[arg-type]
            )
        )
    return catalog


def get_model(model_id: str) -> ModelOption:
    for model in get_model_catalog():
        if model.id == model_id:
            return model
    raise KeyError(model_id)
