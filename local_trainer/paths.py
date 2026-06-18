from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"
SAMPLES_DIR = ROOT_DIR / "samples"
RUNTIME_DIR = ROOT_DIR / "runtime"
DATASET_DIR = RUNTIME_DIR / "datasets"
RUNS_DIR = RUNTIME_DIR / "runs"
WORKBENCH_DB = RUNTIME_DIR / "workbench.db"
LLAMA_FACTORY_DIR = ROOT_DIR / "LLaMA-Factory"
MODELS_DIR = ROOT_DIR / "models"
MODEL_REGISTRY_FILE = Path(__file__).resolve().parent / "model_registry.yaml"
LOCAL_QWEN_0_5B = MODELS_DIR / "Qwen" / "Qwen3___5-0___8B"


def model_dir_for_repo(repo_id: str) -> Path:
    """Where ModelScope places a downloaded model under MODELS_DIR.

    Current ModelScope layout: ``cache_dir/<repo_id>`` with dots turned into ``___``.
    Also checks legacy ``cache_dir/models/<repo_id>`` for backward compat.
    """
    new_path = MODELS_DIR / repo_id.replace(".", "___")
    if new_path.exists():
        return new_path
    # Legacy layout (older modelscope added a models/ subdirectory)
    legacy_path = MODELS_DIR / "models" / repo_id.replace(".", "___")
    if legacy_path.exists():
        return legacy_path
    # Default to new layout for fresh downloads
    return new_path


def ensure_runtime_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)