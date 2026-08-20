#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACER_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKTREE_ROOT=$(git -C "$RACER_ROOT" rev-parse --show-toplevel)
ACTOR_PY='/data/yukun/miniconda3/envs/dynamac-racer/bin/python'
SESSION_PROBE="$SCRIPT_DIR/tmux_session_has_live_pane.sh"
SUPERVISOR_ID=${RACER_SUPERVISOR_ID:-virtualgl_egl_$(date +%Y%m%d_%H%M%S)}
SUPERVISOR_RUNTIME="$RACER_ROOT/runtime/$SUPERVISOR_ID"
STATUS_FILE="$SUPERVISOR_RUNTIME/status.tsv"
PIPELINE_LOG="$SUPERVISOR_RUNTIME/pipeline.log"
EGL_GATE_RUN_ID="${SUPERVISOR_ID}_egl_gate"
ISOLATION_CONSISTENCY_RUN_ID="${SUPERVISOR_ID}_isolation_consistency"
ISOLATION_GATE_RUN_ID="${SUPERVISOR_ID}_isolation_gate"
FULL_RUN_ID="${SUPERVISOR_ID}_full"
EGL_GATE_LOG_NAME='egl_gate_place_cups_ep0'
ISOLATION_GATE_LOG_NAME='spawn_isolation_gate_place_cups_ep0'
LM_GPU=1
VLM_GPUS='2,3'
ACTOR_GPU=4
LM_PORT=${RACER_LM_PORT:-18001}
VLM_PORT=${RACER_VLM_PORT:-21003}
DISPLAY_ID=${RACER_DISPLAY_ID:-97}
WAIT_SESSIONS=(dynamac_spr_full_20260821 dynamac_guardian_full_20260821)
terminal_state=0

mkdir -p "$SUPERVISOR_RUNTIME"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

set_state() {
  local state=$1
  local detail=$2
  local temporary="$STATUS_FILE.tmp.$$"
  printf 'timestamp\t%s\nstate\t%s\ndetail\t%s\n' \
    "$(date --iso-8601=seconds)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$STATUS_FILE"
  echo "[$(date --iso-8601=seconds)] state=$state detail=$detail"
}

on_exit() {
  local status=$?
  trap - EXIT
  if (( status != 0 && terminal_state == 0 )); then
    set_state failed "supervisor exited with status $status"
  fi
  exit "$status"
}
trap on_exit EXIT

fail_terminal() {
  local state=$1
  shift
  terminal_state=1
  set_state "$state" "$*"
  exit 1
}

claim_once() {
  local name=$1
  local marker="$SUPERVISOR_RUNTIME/${name}.claimed"
  if ! (set -o noclobber; printf '%s\n' "$(date --iso-8601=seconds)" >"$marker") \
      2>/dev/null; then
    fail_terminal duplicate_attempt_blocked \
      "$name was already claimed for supervisor $SUPERVISOR_ID; no retry is allowed"
  fi
}

[[ "$WORKTREE_ROOT" == '/data/yukun/worktrees/essay2608-racer-egl' ]] || \
  fail_terminal invalid_worktree "unexpected worktree: $WORKTREE_ROOT"
[[ "$(git -C "$WORKTREE_ROOT" branch --show-current)" == 'repro/racer-egl' ]] || \
  fail_terminal invalid_branch 'expected repro/racer-egl'
[[ -z "$(git -C "$WORKTREE_ROOT" status --porcelain --untracked-files=no)" ]] || \
  fail_terminal dirty_worktree 'tracked worktree changes detected'
[[ -x "$ACTOR_PY" ]] || fail_terminal missing_actor_python "$ACTOR_PY"
[[ -x "$SESSION_PROBE" ]] || fail_terminal missing_session_probe "$SESSION_PROBE"

for run_id in "$EGL_GATE_RUN_ID" "$ISOLATION_CONSISTENCY_RUN_ID" \
  "$ISOLATION_GATE_RUN_ID" "$FULL_RUN_ID"; do
  [[ ! -e "$RACER_ROOT/runtime/$run_id" ]] || \
    fail_terminal target_exists "runtime/$run_id"
  [[ ! -e "$RACER_ROOT/results/$run_id" ]] || \
    fail_terminal target_exists "results/$run_id"
done

set_state waiting_for_baselines 'waiting for SPR and Guardian tmux sessions to terminate'
while :; do
  active=()
  for session in "${WAIT_SESSIONS[@]}"; do
    if "$SESSION_PROBE" "$session"; then
      active+=("$session")
    fi
  done
  ((${#active[@]} == 0)) && break
  echo "Still waiting for tmux sessions: ${active[*]}"
  sleep 30
done

gpu_is_idle() {
  local gpu=$1
  local processes
  processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  [[ -z "$processes" ]]
}

set_state waiting_for_gpus 'waiting for GPUs 1-4 to remain idle for three checks'
idle_checks=0
while (( idle_checks < 3 )); do
  all_idle=1
  for gpu in 1 2 3 4; do
    gpu_is_idle "$gpu" || all_idle=0
  done
  if (( all_idle == 1 )); then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  echo "GPU idle checks: $idle_checks/3"
  (( idle_checks == 3 )) || sleep 10
done

common_environment=(
  RACER_LM_GPU="$LM_GPU"
  RACER_VLM_GPUS="$VLM_GPUS"
  RACER_ACTOR_GPU="$ACTOR_GPU"
  RACER_LM_PORT="$LM_PORT"
  RACER_VLM_PORT="$VLM_PORT"
  RACER_DISPLAY_ID="$DISPLAY_ID"
)

selected_backend=''
selected_egl_device=''
egl_failure_detail=''

set_state resolving_egl_device "mapping physical NVIDIA GPU $ACTOR_GPU"
if EGL_DEVICE=$(
    "$ACTOR_PY" "$SCRIPT_DIR/resolve_virtualgl_device.py" \
      --gpu "$ACTOR_GPU" --print-device \
      2>"$SUPERVISOR_RUNTIME/egl_device_mapping_error.log"
  ) && "$ACTOR_PY" "$SCRIPT_DIR/resolve_virtualgl_device.py" --gpu "$ACTOR_GPU" \
    >"$SUPERVISOR_RUNTIME/egl_device_map.json"; then
  echo "Selected $EGL_DEVICE for physical NVIDIA GPU $ACTOR_GPU."
  set_state egl_gate_running 'one fixed place_cups episode 0 with zero evaluator retries; full run remains locked'
  if env "${common_environment[@]}" \
    RACER_GL_BACKEND=virtualgl-egl \
    RACER_EGL_DEVICE="$EGL_DEVICE" \
    RACER_RUN_ID="$EGL_GATE_RUN_ID" \
    RACER_TASKS=place_cups \
    RACER_START_EPISODE=0 \
    RACER_EVAL_EPISODES=1 \
    RACER_RETRY_FOR_INVALID_ACTION_ERROR=0 \
    RACER_LOG_NAME="$EGL_GATE_LOG_NAME" \
    bash "$SCRIPT_DIR/run_three_task_eval.sh" \
    >"$SUPERVISOR_RUNTIME/egl_gate_driver.log" 2>&1; then
    EGL_GATE_METRICS="$RACER_ROOT/results/$EGL_GATE_RUN_ID/$EGL_GATE_LOG_NAME/metrics.json"
    EGL_GATE_ACTOR_LOG="$RACER_ROOT/runtime/$EGL_GATE_RUN_ID/actor_eval.log"
    if "$ACTOR_PY" "$SCRIPT_DIR/validate_single_episode.py" \
      --metrics "$EGL_GATE_METRICS" --actor-log "$EGL_GATE_ACTOR_LOG" \
      >"$SUPERVISOR_RUNTIME/egl_gate_validation.json"; then
      selected_backend='virtualgl-egl'
      selected_egl_device="$EGL_DEVICE"
      set_state egl_gate_passed '1/1 episode completed with natural status 0 and no retry'
    else
      egl_failure_detail="strict EGL gate validation failed; see $SUPERVISOR_RUNTIME/egl_gate_validation.json"
    fi
  else
    egl_failure_detail="EGL episode process failed; see $SUPERVISOR_RUNTIME/egl_gate_driver.log"
  fi
else
  egl_failure_detail="EGL device mapping failed; see $SUPERVISOR_RUNTIME/egl_device_mapping_error.log"
fi

if [[ -z "$selected_backend" ]]; then
  claim_once spawn_isolation_fallback
  set_state egl_gate_failed_starting_isolation_consistency \
    "$egl_failure_detail; running the one allowed spawn-isolation fallback path"
  if ! env "${common_environment[@]}" \
    RACER_RUN_ID="$ISOLATION_CONSISTENCY_RUN_ID" \
    RACER_TASKS=place_cups \
    RACER_START_EPISODE=0 \
    bash "$SCRIPT_DIR/run_spawn_isolation_consistency.sh" \
    >"$SUPERVISOR_RUNTIME/isolation_consistency_driver.log" 2>&1; then
    fail_terminal isolation_consistency_failed \
      "full run locked; no retry; see $SUPERVISOR_RUNTIME/isolation_consistency_driver.log"
  fi

  set_state isolation_consistency_passed \
    'direct and isolated initialization observations match exactly; worker closed naturally with status 0'
  set_state isolation_gate_running \
    'the only spawn-isolated place_cups episode 0 attempt; zero evaluator retries; full run remains locked'
  if ! env "${common_environment[@]}" \
    RACER_GL_BACKEND=spawn-isolated-software \
    RACER_RUN_ID="$ISOLATION_GATE_RUN_ID" \
    RACER_TASKS=place_cups \
    RACER_START_EPISODE=0 \
    RACER_EVAL_EPISODES=1 \
    RACER_RETRY_FOR_INVALID_ACTION_ERROR=0 \
    RACER_LOG_NAME="$ISOLATION_GATE_LOG_NAME" \
    bash "$SCRIPT_DIR/run_three_task_eval.sh" \
    >"$SUPERVISOR_RUNTIME/isolation_gate_driver.log" 2>&1; then
    fail_terminal isolation_gate_failed \
      "full run locked; no retry; see $SUPERVISOR_RUNTIME/isolation_gate_driver.log"
  fi

  ISOLATION_GATE_METRICS="$RACER_ROOT/results/$ISOLATION_GATE_RUN_ID/$ISOLATION_GATE_LOG_NAME/metrics.json"
  ISOLATION_GATE_ACTOR_LOG="$RACER_ROOT/runtime/$ISOLATION_GATE_RUN_ID/actor_eval.log"
  if ! "$ACTOR_PY" "$SCRIPT_DIR/validate_single_episode.py" \
    --metrics "$ISOLATION_GATE_METRICS" --actor-log "$ISOLATION_GATE_ACTOR_LOG" \
    >"$SUPERVISOR_RUNTIME/isolation_gate_validation.json"; then
    fail_terminal isolation_gate_validation_failed \
      "full run locked; no retry; see $SUPERVISOR_RUNTIME/isolation_gate_validation.json"
  fi
  selected_backend='spawn-isolated-software'
  set_state isolation_gate_passed \
    '1/1 isolated episode completed with natural evaluator and simulator-worker status 0; no retry'
fi

for gpu in 1 2 3 4; do
  gpu_is_idle "$gpu" || fail_terminal gpu_reoccupied "GPU $gpu became busy after gate"
done

backend_environment=(RACER_GL_BACKEND="$selected_backend")
if [[ "$selected_backend" == 'virtualgl-egl' ]]; then
  backend_environment+=(RACER_EGL_DEVICE="$selected_egl_device")
  FULL_LOG_NAME='official_ckpt_three_task_virtualgl_egl'
else
  FULL_LOG_NAME='official_ckpt_three_task_spawn_isolated'
fi

set_state full_running "three tasks x 25 fixed episodes; backend=$selected_backend"
if ! env "${common_environment[@]}" "${backend_environment[@]}" \
  RACER_RUN_ID="$FULL_RUN_ID" \
  RACER_TASKS=place_cups,place_wine_at_rack_location,sweep_to_dustpan_of_size \
  RACER_START_EPISODE=0 \
  RACER_EVAL_EPISODES=25 \
  RACER_RETRY_FOR_INVALID_ACTION_ERROR=5 \
  RACER_LOG_NAME="$FULL_LOG_NAME" \
  bash "$SCRIPT_DIR/run_three_task_eval.sh" \
  >"$SUPERVISOR_RUNTIME/full_driver.log" 2>&1; then
  fail_terminal full_failed "see $SUPERVISOR_RUNTIME/full_driver.log"
fi

FULL_OUTPUT="$RACER_ROOT/results/$FULL_RUN_ID/$FULL_LOG_NAME"
for artifact in metrics.json comparison.json comparison.csv comparison.md paper_vs_reproduction.png; do
  [[ -s "$FULL_OUTPUT/$artifact" ]] || \
    fail_terminal full_validation_failed "missing or empty $FULL_OUTPUT/$artifact"
done

terminal_state=1
set_state complete "backend=$selected_backend; gate and 3x25 completed; outputs: $FULL_OUTPUT"
