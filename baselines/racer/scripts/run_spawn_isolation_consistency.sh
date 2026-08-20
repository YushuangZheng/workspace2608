#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACER_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
UPSTREAM="$RACER_ROOT/upstream"
ACTOR_PY='/data/yukun/miniconda3/envs/dynamac-racer/bin/python'
COPPELIASIM_ROOT='/data/yukun/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
XVFB='/data/yukun/.cache/racer/xvfb-ubuntu-root/usr/bin/Xvfb'
RUN_ID=${RACER_RUN_ID:?RACER_RUN_ID is required}
DISPLAY_ID=${RACER_DISPLAY_ID:-97}
TASK=${RACER_TASKS:-place_cups}
EPISODE=${RACER_START_EPISODE:-0}
RUNTIME_DIR="$RACER_ROOT/runtime/$RUN_ID"
DIRECT_JSON="$RUNTIME_DIR/direct_observation.json"
ISOLATED_JSON="$RUNTIME_DIR/isolated_observation.json"
REPORT_JSON="$RUNTIME_DIR/observation_comparison.json"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$DISPLAY_ID" =~ ^[0-9]+$ ]] || fail 'RACER_DISPLAY_ID must be numeric.'
[[ "$EPISODE" =~ ^[0-9]+$ ]] || fail 'RACER_START_EPISODE must be nonnegative.'
[[ "$TASK" =~ ^[a-z0-9_]+$ ]] || fail 'consistency gate requires exactly one task.'
[[ -x "$ACTOR_PY" ]] || fail "actor interpreter missing: $ACTOR_PY"
[[ -x "$XVFB" ]] || fail "user-space Xvfb missing: $XVFB"
[[ ! -e "$RUNTIME_DIR" ]] || fail "runtime target already exists: $RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR/direct" "$RUNTIME_DIR/isolated"

xvfb_pid=''
cleanup() {
  status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$xvfb_pid" ]]; then
    kill -TERM "$xvfb_pid" 2>/dev/null || true
    wait "$xvfb_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

DISPLAY_SOCKET="/tmp/.X11-unix/X$DISPLAY_ID"
DISPLAY_LOCK="/tmp/.X${DISPLAY_ID}-lock"
[[ ! -e "$DISPLAY_LOCK" && ! -S "$DISPLAY_SOCKET" ]] || \
  fail "X display :$DISPLAY_ID is already in use"

env -u LD_LIBRARY_PATH XDG_CACHE_HOME=/data/yukun/.cache/racer \
  "$XVFB" ":$DISPLAY_ID" -screen 0 1280x1024x24 -ac \
  +extension GLX +iglx +render -noreset \
  >"$RUNTIME_DIR/xvfb.log" 2>&1 &
xvfb_pid=$!
for _ in $(seq 1 100); do
  [[ -S "$DISPLAY_SOCKET" ]] && break
  kill -0 "$xvfb_pid" 2>/dev/null || fail "Xvfb exited; see $RUNTIME_DIR/xvfb.log"
  sleep 0.1
done
[[ -S "$DISPLAY_SOCKET" ]] || fail 'Xvfb did not become ready in 10 seconds.'
DISPLAY=":$DISPLAY_ID" xdpyinfo >"$RUNTIME_DIR/xdpyinfo.txt" 2>&1 || \
  fail "xdpyinfo failed; see $RUNTIME_DIR/xdpyinfo.txt"
DISPLAY=":$DISPLAY_ID" XDG_CACHE_HOME=/data/yukun/.cache/racer \
  LIBGL_ALWAYS_SOFTWARE=1 glxinfo -B >"$RUNTIME_DIR/glxinfo.txt" 2>&1 || \
  fail "software GLX context creation failed; see $RUNTIME_DIR/glxinfo.txt"

capture_environment=(
  env
  -u LIBGL_DRIVERS_PATH
  -u MESA_LOADER_DRIVER_OVERRIDE
  DISPLAY=":$DISPLAY_ID"
  XDG_CACHE_HOME=/data/yukun/.cache/racer
  COPPELIASIM_ROOT="$COPPELIASIM_ROOT"
  LD_LIBRARY_PATH="$COPPELIASIM_ROOT"
  QT_PLUGIN_PATH="$COPPELIASIM_ROOT"
  QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT/platforms"
  QT_QPA_PLATFORM=xcb
  QT_XCB_GL_INTEGRATION=xcb_glx
  LIBGL_ALWAYS_SOFTWARE=1
)

cd "$UPSTREAM"
"${capture_environment[@]}" "$ACTOR_PY" -u "$SCRIPT_DIR/capture_direct_observation.py" \
  --task "$TASK" --dataset-root "$UPSTREAM/racer/data/rlbench/test" \
  --episode "$EPISODE" --episode-length 30 --output "$DIRECT_JSON" \
  >"$RUNTIME_DIR/direct_capture.log" 2>&1

"${capture_environment[@]}" \
  RACER_ISOLATION_RUNTIME_DIR="$RUNTIME_DIR/isolated" \
  "$ACTOR_PY" -u "$SCRIPT_DIR/capture_isolated_observation.py" \
  --task "$TASK" --dataset-root "$UPSTREAM/racer/data/rlbench/test" \
  --episode "$EPISODE" --episode-length 30 --output "$ISOLATED_JSON" \
  >"$RUNTIME_DIR/isolated_capture.log" 2>&1

"$ACTOR_PY" "$SCRIPT_DIR/compare_observation_fingerprints.py" \
  --direct "$DIRECT_JSON" --isolated "$ISOLATED_JSON" --output "$REPORT_JSON" \
  >"$RUNTIME_DIR/comparison_stdout.json"

echo "RACER direct and spawn-isolated initialization observations match exactly."
echo "Comparison: $REPORT_JSON"
