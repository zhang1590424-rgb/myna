from __future__ import annotations

import time

from .domain import EnvironmentStatus, ModelStatus
from .model_registry import get_model_catalog

_env_cache: EnvironmentStatus | None = None
_env_cache_ts: float = 0.0
_ENV_CACHE_TTL: float = 10.0  # 10 秒内复用缓存


def collect_environment_status(*, force: bool = False) -> EnvironmentStatus:
    global _env_cache, _env_cache_ts
    now = time.monotonic()
    if not force and _env_cache is not None and (now - _env_cache_ts) < _ENV_CACHE_TTL:
        return _env_cache
    status = _collect_environment_status_impl()
    _env_cache = status
    _env_cache_ts = now
    return status


def invalidate_environment_cache() -> None:
    """下载完成或环境变更时主动失效缓存。"""
    global _env_cache, _env_cache_ts
    _env_cache = None
    _env_cache_ts = 0.0


def _collect_environment_status_impl() -> EnvironmentStatus:
    llamafactory_ok = _can_import("llamafactory")
    torch_mps_ok = _torch_mps_available()
    models = [
        ModelStatus(id=model.id, name=model.name, available=model.available, note=model.learning_value)
        for model in get_model_catalog()
    ]

    ready_items = [True, llamafactory_ok, any(model.available for model in models)]
    progress = round(sum(1 for item in ready_items if item) / len(ready_items) * 100)
    if llamafactory_ok and any(model.available for model in models):
        if torch_mps_ok:
            message = "运行组件和本地模型已就绪，可以开始体验。"
        else:
            message = "运行组件和本地模型已就绪，但没有检测到 MPS，真实训练可能会变慢。"
    else:
        message = "还缺少训练组件或本地模型，首次使用需要先准备。"

    return EnvironmentStatus(
        python_ok=True,
        llamafactory_ok=llamafactory_ok,
        torch_mps_ok=torch_mps_ok,
        model_status=models,
        progress=progress,
        message=message,
    )


def _can_import(package: str) -> bool:
    try:
        __import__(package)
    except Exception:
        return False
    return True


def _torch_mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False
