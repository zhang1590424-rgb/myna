#!/usr/bin/env bash
# Builds the Dock-pinnable macOS launcher app for Myna.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

APP_DIR="Myna.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

command -v swiftc >/dev/null 2>&1 || {
  printf '[macos-app][error] 没有找到 swiftc，请先安装 Xcode Command Line Tools。\n' >&2
  exit 1
}

printf '[macos-app] 生成图标。\n'
.venv/bin/python macos/make_icon.py

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
cp macos/Info.plist "${CONTENTS_DIR}/Info.plist"

printf '[macos-app] 编译启动器。\n'
BUILT_BINARIES=()
for ARCH in arm64 x86_64; do
  if swiftc \
    -target "${ARCH}-apple-macos12.0" \
    -O \
    -framework AppKit \
    macos/MynaLauncher.swift \
    -o "${TMP_DIR}/Myna-${ARCH}"; then
    BUILT_BINARIES+=("${TMP_DIR}/Myna-${ARCH}")
  fi
done

if [ "${#BUILT_BINARIES[@]}" -eq 0 ]; then
  printf '[macos-app][error] 启动器编译失败。\n' >&2
  exit 1
fi

if [ "${#BUILT_BINARIES[@]}" -gt 1 ]; then
  lipo -create "${BUILT_BINARIES[@]}" -output "${MACOS_DIR}/Myna"
else
  cp "${BUILT_BINARIES[0]}" "${MACOS_DIR}/Myna"
fi

chmod +x "${MACOS_DIR}/Myna"
codesign --force --deep --sign - "$APP_DIR"
printf '[macos-app] 完成：%s\n' "$APP_DIR"
