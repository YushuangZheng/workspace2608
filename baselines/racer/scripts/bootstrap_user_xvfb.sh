#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_VERSION='2:21.1.4-2ubuntu1.7~22.04.16'
PACKAGE_FILE='xvfb_2%3a21.1.4-2ubuntu1.7~22.04.16_amd64.deb'
PACKAGE_SHA256='f857020815dd091b4331cc45b37550a92480178c2931c76b486af2202e4271ec'
BINARY_SHA256='cf71d78eab8321f7c568271053af449f966a75422140117ea1f244bf72a7e81e'
CACHE_ROOT='/data/yukun/.cache/racer'
PACKAGE_DIR="$CACHE_ROOT/xvfb-ubuntu-packages"
EXTRACT_ROOT="$CACHE_ROOT/xvfb-ubuntu-root"
PACKAGE_PATH="$PACKAGE_DIR/$PACKAGE_FILE"
BINARY_PATH="$EXTRACT_ROOT/usr/bin/Xvfb"

mkdir -p "$PACKAGE_DIR" "$EXTRACT_ROOT"

if [[ ! -f "$PACKAGE_PATH" ]]; then
  command -v apt-get >/dev/null || {
    echo 'ERROR: apt-get is required to download the Ubuntu-matched Xvfb package.' >&2
    exit 1
  }
  (
    cd "$PACKAGE_DIR"
    apt-get download "xvfb=$PACKAGE_VERSION"
  )
fi

printf '%s  %s\n' "$PACKAGE_SHA256" "$PACKAGE_PATH" | sha256sum --check --status || {
  echo "ERROR: Xvfb package checksum mismatch: $PACKAGE_PATH" >&2
  exit 1
}

dpkg-deb -x "$PACKAGE_PATH" "$EXTRACT_ROOT"

printf '%s  %s\n' "$BINARY_SHA256" "$BINARY_PATH" | sha256sum --check --status || {
  echo "ERROR: extracted Xvfb checksum mismatch: $BINARY_PATH" >&2
  exit 1
}

echo "User-space Xvfb ready: $BINARY_PATH"
