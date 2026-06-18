#!/usr/bin/env bash
# Starts the local web service and opens the browser.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-4180}"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}"
PYTHON=".venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  printf '[start][error] 没有找到运行环境 .venv。请先运行 scripts/install.command\n' >&2
  printf '按回车键关闭。'
  read -r _ || true
  exit 1
fi

printf '================================================\n'
printf '  Myna · 本机 Web 控制台\n'
printf '================================================\n\n'

if lsof -i TCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  if curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
    printf '[start] 服务已经在运行，打开浏览器：%s\n' "$URL"
    open "$URL"
    printf '按回车键关闭本窗口（不会停止已运行的服务）。'
    read -r _ || true
    exit 0
  fi
  printf '[start][error] 端口 %s 已被占用，但不是 Myna 服务。请换端口或关闭占用进程。\n' "$PORT" >&2
  exit 1
fi

printf '[start] 启动本地服务：%s\n' "$URL"
"$PYTHON" -m uvicorn local_trainer.main:app --host "$HOST" --port "$PORT" &
SERVER_PID=$!

cleanup() {
  printf '\n[start] 正在停止服务。\n'
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

printf '[start] 等待服务就绪。\n'
for _ in $(seq 1 60); do
  if curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
    printf '[start] 已就绪，打开浏览器：%s\n' "$URL"
    open "$URL"
    break
  fi
  sleep 0.5
done

if ! curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
  printf '[start][error] 服务启动超时，请运行 scripts/doctor.command 检查环境。\n' >&2
  exit 1
fi

printf '\n使用中请保持本窗口开着。关闭本窗口即可停止服务。\n'
wait "$SERVER_PID"
