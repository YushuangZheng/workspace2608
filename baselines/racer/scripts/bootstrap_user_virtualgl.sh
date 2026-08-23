#!/usr/bin/env bash
set -Eeuo pipefail

VERSION='3.1.5'
PACKAGE_NAME="virtualgl_${VERSION}_amd64.deb"
PACKAGE_SHA256='df3f7788ce41b182a47c0d298e5cd6d2d63579522cb41825970b7726e825485e'
PACKAGE_URL="https://github.com/VirtualGL/virtualgl/releases/download/${VERSION}/${PACKAGE_NAME}"
CACHE_ROOT=${RACER_VGL_CACHE_ROOT:-/data/yukun/.cache/racer}
DOWNLOAD_DIR="$CACHE_ROOT/downloads"
PACKAGE_PATH="$DOWNLOAD_DIR/$PACKAGE_NAME"
INSTALL_ROOT=${RACER_VGL_ROOT:-$CACHE_ROOT/virtualgl-${VERSION}-root}
VGLRUN="$INSTALL_ROOT/opt/VirtualGL/bin/vglrun"
EGLINFO="$INSTALL_ROOT/opt/VirtualGL/bin/eglinfo"
VGL_LIBDIR="$INSTALL_ROOT/usr/lib"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v curl >/dev/null || fail 'curl is unavailable.'
command -v dpkg-deb >/dev/null || fail 'dpkg-deb is unavailable.'
command -v sha256sum >/dev/null || fail 'sha256sum is unavailable.'

mkdir -p "$DOWNLOAD_DIR"
if [[ -f "$PACKAGE_PATH" ]]; then
  echo "$PACKAGE_SHA256  $PACKAGE_PATH" | sha256sum --check --status || \
    fail "cached package checksum mismatch: $PACKAGE_PATH"
else
  partial="$PACKAGE_PATH.partial.$$"
  trap 'status=$?; if [[ -n "${partial:-}" && -f "$partial" ]]; then unlink "$partial"; fi; exit "$status"' EXIT
  curl --fail --location --retry 4 --retry-all-errors \
    --connect-timeout 15 --output "$partial" "$PACKAGE_URL"
  echo "$PACKAGE_SHA256  $partial" | sha256sum --check --status || \
    fail 'downloaded VirtualGL package checksum mismatch.'
  mv "$partial" "$PACKAGE_PATH"
  partial=''
  trap - EXIT
fi

if [[ ! -x "$VGLRUN" || ! -x "$EGLINFO" || ! -f "$VGL_LIBDIR/libvglfaker.so" ]]; then
  [[ ! -e "$INSTALL_ROOT" ]] || \
    fail "incomplete VirtualGL extraction exists; inspect before replacing: $INSTALL_ROOT"
  stage=$(mktemp -d "$CACHE_ROOT/virtualgl-${VERSION}-stage.XXXXXX")
  trap 'status=$?; if [[ -n "${stage:-}" && -d "$stage" ]]; then rm -rf -- "$stage"; fi; exit "$status"' EXIT
  dpkg-deb --extract "$PACKAGE_PATH" "$stage"
  [[ -x "$stage/opt/VirtualGL/bin/vglrun" ]] || fail 'extracted vglrun is missing.'
  [[ -x "$stage/opt/VirtualGL/bin/eglinfo" ]] || fail 'extracted eglinfo is missing.'
  [[ -f "$stage/usr/lib/libvglfaker.so" ]] || \
    fail 'extracted libvglfaker.so is missing.'
  mv "$stage" "$INSTALL_ROOT"
  stage=''
  trap - EXIT
fi

actual_version=$(dpkg-deb --field "$PACKAGE_PATH" Version | sed 's/-.*$//')
[[ "$actual_version" == "$VERSION" ]] || \
  fail "unexpected VirtualGL version: ${actual_version:-unknown}"

printf 'VirtualGL %s ready at %s\n' "$VERSION" "$INSTALL_ROOT"
printf 'Package SHA-256: %s\n' "$PACKAGE_SHA256"
printf 'vglrun: %s\n' "$VGLRUN"
printf 'eglinfo: %s\n' "$EGLINFO"
printf 'library directory: %s\n' "$VGL_LIBDIR"
