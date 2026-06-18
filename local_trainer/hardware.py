from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def llamafactory_cli() -> str:
    """Resolve the llamafactory-cli launcher.

    Prefers the script sitting next to the current Python (the venv bin dir),
    so real training works even when the venv is not on PATH. Falls back to
    PATH lookup, then the bare command name.
    """
    candidate = Path(sys.executable).with_name("llamafactory-cli")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("llamafactory-cli")
    return found or "llamafactory-cli"


def detect_device() -> str:
    """Return the best available training/inference device."""
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_precision(device: str) -> dict[str, bool]:
    """Pick a safe precision per device.

    MPS/CPU use fp32 (bf16/fp16 both False): M0 verified fp16 training is
    unstable on MPS. Only CUDA gets bf16.
    """
    if device == "cuda":
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": False}


def torch_dtype(device: str):
    """Inference dtype matching the training precision policy."""
    import torch

    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def training_env() -> dict[str, str]:
    """Environment variables to inject into the training subprocess."""
    env = dict(os.environ)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env
