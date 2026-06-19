#!/usr/bin/env bash
# Agent-friendly updater for Myna.
# 拉取最新代码、更新依赖、按需同步训练引擎版本。用户的模型、数据和训练记录不受影响。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf '[update] %s\n' "$1"
}

fail() {
  printf '[update][error] %s\n' "$1" >&2
  printf '按回车键关闭。'
  read -r _ || true
  exit 1
}

printf '================================================\n'
printf '  Myna · 升级到最新版\n'
printf '================================================\n\n'
log "你下载的模型、上传的数据和训练记录都不会被改动。"

command -v git >/dev/null 2>&1 || fail "没有找到 git，无法拉取更新。"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  fail "当前目录不是 git 仓库，无法自动升级。请重新用 git clone 安装，或让 AI 助手帮你处理。"

# 升级前记录关键文件状态，用于判断是否需要重装依赖 / 同步训练引擎。
req_hash_before=""
[ -f requirements.txt ] && req_hash_before="$(shasum requirements.txt | awk '{print $1}')"
ref_before=""
[ -f scripts/install.command ] && \
  ref_before="$(grep -E '^LLAMA_FACTORY_REF=' scripts/install.command | head -n1 | cut -d'"' -f2 || true)"

if [ -n "$(git status --porcelain)" ]; then
  fail "检测到本地有未提交的改动，自动升级可能覆盖它们。请先备份这些改动，或让 AI 助手帮你处理后再升级。"
fi

current_branch="$(git rev-parse --abbrev-ref HEAD)"
log "拉取最新代码（分支：${current_branch}）。"
git pull --ff-only || fail "拉取更新失败。可能是网络问题或本地与远程有分歧，请让 AI 助手帮你排查。"

# 升级后再次读取关键文件状态。
req_hash_after=""
[ -f requirements.txt ] && req_hash_after="$(shasum requirements.txt | awk '{print $1}')"
ref_after=""
[ -f scripts/install.command ] && \
  ref_after="$(grep -E '^LLAMA_FACTORY_REF=' scripts/install.command | head -n1 | cut -d'"' -f2 || true)"

if [ ! -x ".venv/bin/python" ]; then
  log "未找到运行环境 .venv，调用安装脚本完成初始化。"
  bash scripts/install.command || fail "安装运行环境失败。"
else
  if [ "$req_hash_before" != "$req_hash_after" ]; then
    log "检测到依赖清单有更新，重新安装 Python 依赖。"
  else
    log "更新 Python 依赖（确保与最新代码匹配）。"
  fi
  .venv/bin/python -m pip install -r requirements.txt || fail "更新 Python 依赖失败。"

  # 训练引擎版本有变化时才同步，避免动用户已下好的 LLaMA-Factory。
  if [ -n "$ref_after" ] && [ "$ref_before" != "$ref_after" ] && [ -d "LLaMA-Factory/.git" ]; then
    log "训练引擎版本有更新（${ref_before:-未知} -> ${ref_after}），同步 LLaMA-Factory。"
    git -C LLaMA-Factory fetch --depth 1 origin "$ref_after" || true
    if git -C LLaMA-Factory checkout --detach "$ref_after"; then
      .venv/bin/python -m pip install -e ./LLaMA-Factory || fail "重新安装 LLaMA-Factory 失败。"
    else
      log "无法切换到新版训练引擎，保持现状。如训练异常，请重新运行 scripts/install.command。"
    fi
  fi
fi

printf '\n'
log "升级完成。建议接着运行 scripts/doctor.command 确认环境正常。"
log "之后双击 Myna.app 或运行 scripts/start.command 即可使用最新版本。"
printf '按回车键关闭。'
read -r _ || true
