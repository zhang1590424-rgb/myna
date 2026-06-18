from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"
SAMPLES_DIR = ROOT_DIR / "samples"
RUNTIME_DIR = ROOT_DIR / "runtime"
DATASET_DIR = RUNTIME_DIR / "datasets"
RUNS_DIR = RUNTIME_DIR / "runs"
LLAMA_FACTORY_DIR = ROOT_DIR / "LLaMA-Factory"
MODELS_DIR = ROOT_DIR / "models"
LOCAL_QWEN_0_5B = MODELS_DIR / "models" / "Qwen" / "Qwen2___5-0___5B-Instruct"


def model_dir_for_repo(repo_id: str) -> Path:
    """Where ModelScope (1.37 layout) places a downloaded model under MODELS_DIR.

    Cache layout is ``cache_dir/models/<repo_id>`` with dots turned into ``___``.
    """
    return MODELS_DIR / "models" / repo_id.replace(".", "___")


def ensure_runtime_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
