#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACER_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
UPSTREAM="$RACER_ROOT/upstream"
ACTOR_PY='/data/yukun/miniconda3/envs/dynamac-racer/bin/python'
COPPELIASIM_ROOT='/data/yukun/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
XVFB='/data/yukun/.cache/racer/xvfb-ubuntu-root/usr/bin/Xvfb'
OWNED_PROCESS_AUDIT="$SCRIPT_DIR/audit_owned_processes.py"
RUN_ID=${RACER_RUN_ID:?RACER_RUN_ID is required}
DISPLAY_ID=${RACER_DISPLAY_ID:-97}
CAPTURE_TIMEOUT_SECONDS=${RACER_CAPTURE_TIMEOUT_SECONDS:-600}
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
[[ "$CAPTURE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  fail 'RACER_CAPTURE_TIMEOUT_SECONDS must be positive.'
[[ "$TASK" =~ ^[a-z0-9_]+$ ]] || fail 'consistency gate requires exactly one task.'
[[ -x "$ACTOR_PY" ]] || fail "actor interpreter missing: $ACTOR_PY"
[[ -x "$XVFB" ]] || fail "user-space Xvfb missing: $XVFB"
command -v timeout >/dev/null || fail 'GNU timeout is unavailable.'
[[ -f "$OWNED_PROCESS_AUDIT" ]] || fail 'owned-process audit helper is missing.'
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
  audit_status=1
  for _ in $(seq 1 10); do
    if env -u RACER_RUN_ID -u RACER_OWNER_TOKEN \
      "$ACTOR_PY" "$OWNED_PROCESS_AUDIT" --value "$RUN_ID" \
      --ignore-pid "$$" --output "$RUNTIME_DIR/cleanup_process_audit.json" \
      >/dev/null 2>&1; then
      audit_status=0
      break
    fi
    sleep 0.5
  done
  if (( audit_status != 0 )); then
    echo "ERROR: consistency-owned processes remain after cleanup; see $RUNTIME_DIR/cleanup_process_audit.json" >&2
    (( status != 0 )) || status=1
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

env -u LD_LIBRARY_PATH RACER_OWNER_TOKEN="${RUN_ID}_xvfb" \
  XDG_CACHE_HOME=/data/yukun/.cache/racer \
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

printf 'capture_timeout_seconds\t%s\n' "$CAPTURE_TIMEOUT_SECONDS" \
  >"$RUNTIME_DIR/time_limits.tsv"

audit_owner() {
  local owner_token=$1
  local output=$2
  env -u RACER_RUN_ID -u RACER_OWNER_TOKEN \
    "$ACTOR_PY" "$OWNED_PROCESS_AUDIT" \
      --environment-key RACER_OWNER_TOKEN --value "$owner_token" \
      --output "$output"
}

cd "$UPSTREAM"
DIRECT_OWNER="${RUN_ID}_direct_capture"
direct_status=0
if "${capture_environment[@]}" RACER_OWNER_TOKEN="$DIRECT_OWNER" \
  timeout --signal=TERM --kill-after=30s "${CAPTURE_TIMEOUT_SECONDS}s" \
  "$ACTOR_PY" -u "$SCRIPT_DIR/capture_direct_observation.py" \
    --task "$TASK" --dataset-root "$UPSTREAM/racer/data/rlbench/test" \
    --episode "$EPISODE" --episode-length 30 --output "$DIRECT_JSON" \
    >"$RUNTIME_DIR/direct_capture.log" 2>&1; then
  direct_status=0
else
  direct_status=$?
fi
if ! audit_owner "$DIRECT_OWNER" "$RUNTIME_DIR/direct_capture_process_audit.json" \
  >"$RUNTIME_DIR/direct_capture_process_audit_stdout.json"; then
  fail "direct capture left owned processes; see $RUNTIME_DIR/direct_capture_process_audit.json"
fi
(( direct_status == 0 )) || \
  fail "direct capture exited with status $direct_status (124 means timeout); see $RUNTIME_DIR/direct_capture.log"

ISOLATED_OWNER="${RUN_ID}_isolated_capture"
isolated_status=0
if "${capture_environment[@]}" RACER_OWNER_TOKEN="$ISOLATED_OWNER" \
  RACER_ISOLATION_RUNTIME_DIR="$RUNTIME_DIR/isolated" \
  timeout --signal=TERM --kill-after=30s "${CAPTURE_TIMEOUT_SECONDS}s" \
  "$ACTOR_PY" -u "$SCRIPT_DIR/capture_isolated_observation.py" \
    --task "$TASK" --dataset-root "$UPSTREAM/racer/data/rlbench/test" \
    --episode "$EPISODE" --episode-length 30 --output "$ISOLATED_JSON" \
    >"$RUNTIME_DIR/isolated_capture.log" 2>&1; then
  isolated_status=0
else
  isolated_status=$?
fi
if ! audit_owner "$ISOLATED_OWNER" "$RUNTIME_DIR/isolated_capture_process_audit.json" \
  >"$RUNTIME_DIR/isolated_capture_process_audit_stdout.json"; then
  fail "isolated capture left owned processes; see $RUNTIME_DIR/isolated_capture_process_audit.json"
fi
(( isolated_status == 0 )) || \
  fail "isolated capture exited with status $isolated_status (124 means timeout); see $RUNTIME_DIR/isolated_capture.log"

"$ACTOR_PY" "$SCRIPT_DIR/compare_observation_fingerprints.py" \
  --direct "$DIRECT_JSON" --isolated "$ISOLATED_JSON" --output "$REPORT_JSON" \
  >"$RUNTIME_DIR/comparison_stdout.json"

echo "RACER direct and spawn-isolated initialization observations match exactly."
echo "Comparison: $REPORT_JSON"
