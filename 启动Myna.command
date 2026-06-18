#!/usr/bin/env bash
# 用户可见启动入口。实际逻辑放在 scripts/start.command，方便 Agent 复用。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${ROOT_DIR}/scripts/start.command"
