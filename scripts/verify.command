#!/usr/bin/env bash
# Runs automated checks and verifies the local HTTP service.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-4180}"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}"
PYTHON=".venv/bin/python"
STARTED_PID=""

fail() {
  printf '[verify][error] %s\n' "$1" >&2
  exit 1
}

cleanup() {
  if [ -n "$STARTED_PID" ]; then
    kill "$STARTED_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

[ -x "$PYTHON" ] || fail "缺少 .venv，请先运行 scripts/install.command"

printf '[verify] 运行单元测试。\n'
"$PYTHON" -m unittest discover -s tests

printf '[verify] 运行 ruff。\n'
"$PYTHON" -m ruff check local_trainer tests

if lsof -i TCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  printf '[verify] 端口 %s 已有服务，复用它做 API 检查。\n' "$PORT"
else
  printf '[verify] 临时启动本地服务。\n'
  "$PYTHON" -m uvicorn local_trainer.main:app --host "$HOST" --port "$PORT" >/tmp/myna-verify.log 2>&1 &
  STARTED_PID=$!
fi

for _ in $(seq 1 60); do
  if curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

curl -fsS "${URL}/api/health" >/dev/null || fail "健康检查失败"
curl -fsS "${URL}/api/templates" >/dev/null || fail "模板 API 检查失败"
curl -fsS "${URL}/api/models" >/dev/null || fail "模型 API 检查失败"
curl -fsS "${URL}/api/environment" >/dev/null || fail "环境 API 检查失败"

printf '[verify] 通过：测试、ruff、本地 API 都正常。\n'
