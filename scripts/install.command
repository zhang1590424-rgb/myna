#!/usr/bin/env bash
# Agent-friendly installer for 小白训练师.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLAMA_FACTORY_REPO="${LLAMA_FACTORY_REPO:-https://github.com/hiyouga/LLaMA-Factory.git}"
LLAMA_FACTORY_REF="${LLAMA_FACTORY_REF:-8792f06}"

log() {
  printf '[install] %s\n' "$1"
}

fail() {
  printf '[install][error] %s\n' "$1" >&2
  exit 1
}

if [ "$(uname -s)" != "Darwin" ]; then
  log "当前不是 macOS；可以安装，但本项目主要按 Apple Silicon Mac 验证。"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "没有找到 ${PYTHON_BIN}。请先安装 Python 3.10+。"

"$PYTHON_BIN" - <<'PY' || fail "Python 版本过低，请使用 Python 3.10+。"
import sys

if sys.version_info < (3, 10):
    raise SystemExit(1)
PY

if [ ! -x ".venv/bin/python" ]; then
  log "创建 Python 虚拟环境 .venv。"
  "$PYTHON_BIN" -m venv .venv
else
  log "复用已有 .venv。"
fi

log "安装项目依赖。"
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

if [ ! -d "LLaMA-Factory" ]; then
  command -v git >/dev/null 2>&1 || fail "没有找到 git，无法下载 LLaMA-Factory。"
  log "下载 LLaMA-Factory（ref: ${LLAMA_FACTORY_REF}）。"
  git clone --filter=blob:none "$LLAMA_FACTORY_REPO" LLaMA-Factory
  git -C LLaMA-Factory fetch --depth 1 origin "$LLAMA_FACTORY_REF" || true
  git -C LLaMA-Factory checkout --detach "$LLAMA_FACTORY_REF" || log "无法切到指定 ref，继续使用默认分支。"
else
  log "复用已有 LLaMA-Factory 目录，不修改其中内容。"
fi

log "安装 LLaMA-Factory 到当前虚拟环境。"
.venv/bin/python -m pip install -e ./LLaMA-Factory

log "安装完成。建议继续运行：scripts/doctor.command"
