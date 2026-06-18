#!/usr/bin/env bash
# Checks whether the local training service can run on this Mac.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x ".venv/bin/python" ]; then
  printf '[FAIL] 缺少 .venv，请先运行 scripts/install.command\n' >&2
  exit 1
fi

.venv/bin/python - <<'PY'
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

failures: list[str] = []
warnings: list[str] = []


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    warnings.append(message)
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


root = Path.cwd()

if sys.platform == "darwin":
    ok("当前系统是 macOS")
else:
    warn("当前系统不是 macOS，本项目主要按 Apple Silicon Mac 验证")

if sys.version_info >= (3, 10):
    ok(f"Python {sys.version.split()[0]}")
else:
    fail("Python 版本低于 3.10")

for package in ["fastapi", "uvicorn", "pydantic", "yaml", "modelscope"]:
    try:
        importlib.import_module(package)
        ok(f"Python 依赖可导入：{package}")
    except Exception as exc:
        fail(f"Python 依赖不可导入：{package} ({exc})")

try:
    importlib.import_module("llamafactory")
    ok("LLaMA-Factory 可导入")
except Exception as exc:
    fail(f"LLaMA-Factory 不可导入：{exc}")

try:
    import torch

    if torch.backends.mps.is_available():
        ok("检测到 PyTorch MPS，可使用 Apple GPU")
    else:
        warn("未检测到 PyTorch MPS，真实训练可能明显变慢")
except Exception as exc:
    warn(f"无法检查 PyTorch MPS：{exc}")

try:
    from local_trainer.model_registry import get_model_catalog

    models = get_model_catalog()
    available = [model.name for model in models if model.available]
    if available:
        ok("已有本地模型：" + "、".join(available))
    else:
        warn("暂无本地模型；可启动页面后在准备环境页下载")
except Exception as exc:
    fail(f"模型清单检查失败：{exc}")

free_gb = shutil.disk_usage(root).free / (1024**3)
if free_gb >= 20:
    ok(f"磁盘可用空间 {free_gb:.1f} GB")
elif free_gb >= 8:
    warn(f"磁盘可用空间 {free_gb:.1f} GB；可以试用轻量模型，但建议预留 20 GB 以上")
else:
    fail(f"磁盘可用空间仅 {free_gb:.1f} GB，不足以稳定下载模型和训练")

if failures:
    print(f"[SUMMARY] doctor failed: {len(failures)} 个阻塞项，{len(warnings)} 个提醒项")
    raise SystemExit(1)

print(f"[SUMMARY] doctor passed: 0 个阻塞项，{len(warnings)} 个提醒项")
PY
