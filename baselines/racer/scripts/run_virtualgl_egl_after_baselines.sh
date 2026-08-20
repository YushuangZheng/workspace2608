#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACER_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKTREE_ROOT=$(git -C "$RACER_ROOT" rev-parse --show-toplevel)
ACTOR_PY='/data/yukun/miniconda3/envs/dynamac-racer/bin/python'
SUPERVISOR_ID=${RACER_SUPERVISOR_ID:-virtualgl_egl_$(date +%Y%m%d_%H%M%S)}
SUPERVISOR_RUNTIME="$RACER_ROOT/runtime/$SUPERVISOR_ID"
STATUS_FILE="$SUPERVISOR_RUNTIME/status.tsv"
PIPELINE_LOG="$SUPERVISOR_RUNTIME/pipeline.log"
GATE_RUN_ID="${SUPERVISOR_ID}_gate"
FULL_RUN_ID="${SUPERVISOR_ID}_full"
GATE_LOG_NAME='egl_gate_place_cups_ep0'
FULL_LOG_NAME='official_ckpt_three_task_virtualgl_egl'
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

[[ "$WORKTREE_ROOT" == '/data/yukun/worktrees/essay2608-racer-egl' ]] || \
  fail_terminal invalid_worktree "unexpected worktree: $WORKTREE_ROOT"
[[ "$(git -C "$WORKTREE_ROOT" branch --show-current)" == 'repro/racer-egl' ]] || \
  fail_terminal invalid_branch 'expected repro/racer-egl'
[[ -z "$(git -C "$WORKTREE_ROOT" status --porcelain --untracked-files=no)" ]] || \
  fail_terminal dirty_worktree 'tracked worktree changes detected'
[[ -x "$ACTOR_PY" ]] || fail_terminal missing_actor_python "$ACTOR_PY"

for run_id in "$GATE_RUN_ID" "$FULL_RUN_ID"; do
  [[ ! -e "$RACER_ROOT/runtime/$run_id" ]] || \
    fail_terminal target_exists "runtime/$run_id"
  [[ ! -e "$RACER_ROOT/results/$run_id" ]] || \
    fail_terminal target_exists "results/$run_id"
done

set_state waiting_for_baselines 'waiting for SPR and Guardian tmux sessions to terminate'
while :; do
  active=()
  for session in "${WAIT_SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
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

set_state resolving_egl_device "mapping physical NVIDIA GPU $ACTOR_GPU"
EGL_DEVICE=$(
  "$ACTOR_PY" "$SCRIPT_DIR/resolve_virtualgl_device.py" \
    --gpu "$ACTOR_GPU" --print-device
) || fail_terminal egl_mapping_failed "could not map GPU $ACTOR_GPU"
"$ACTOR_PY" "$SCRIPT_DIR/resolve_virtualgl_device.py" --gpu "$ACTOR_GPU" \
  >"$SUPERVISOR_RUNTIME/egl_device_map.json" || \
  fail_terminal egl_mapping_failed "could not record GPU $ACTOR_GPU mapping"
echo "Selected $EGL_DEVICE for physical NVIDIA GPU $ACTOR_GPU."

common_environment=(
  RACER_GL_BACKEND=virtualgl-egl
  RACER_EGL_DEVICE="$EGL_DEVICE"
  RACER_LM_GPU="$LM_GPU"
  RACER_VLM_GPUS="$VLM_GPUS"
  RACER_ACTOR_GPU="$ACTOR_GPU"
  RACER_LM_PORT="$LM_PORT"
  RACER_VLM_PORT="$VLM_PORT"
  RACER_DISPLAY_ID="$DISPLAY_ID"
)

set_state egl_gate_running 'fixed place_cups episode 0; full run remains locked'
if ! env "${common_environment[@]}" \
  RACER_RUN_ID="$GATE_RUN_ID" \
  RACER_TASKS=place_cups \
  RACER_START_EPISODE=0 \
  RACER_EVAL_EPISODES=1 \
  RACER_LOG_NAME="$GATE_LOG_NAME" \
  bash "$SCRIPT_DIR/run_three_task_eval.sh" \
  >"$SUPERVISOR_RUNTIME/gate_driver.log" 2>&1; then
  fail_terminal egl_gate_failed "see $SUPERVISOR_RUNTIME/gate_driver.log"
fi

GATE_METRICS="$RACER_ROOT/results/$GATE_RUN_ID/$GATE_LOG_NAME/metrics.json"
GATE_ACTOR_LOG="$RACER_ROOT/runtime/$GATE_RUN_ID/actor_eval.log"
if ! "$ACTOR_PY" "$SCRIPT_DIR/validate_single_episode.py" \
  --metrics "$GATE_METRICS" --actor-log "$GATE_ACTOR_LOG" \
  >"$SUPERVISOR_RUNTIME/gate_validation.json"; then
  fail_terminal egl_gate_validation_failed \
    "see $SUPERVISOR_RUNTIME/gate_validation.json"
fi

set_state egl_gate_passed '1/1 episode completed with natural status 0; unlocking 3x25'
for gpu in 1 2 3 4; do
  gpu_is_idle "$gpu" || fail_terminal gpu_reoccupied "GPU $gpu became busy after gate"
done

set_state full_running 'three tasks x 25 fixed episodes'
if ! env "${common_environment[@]}" \
  RACER_RUN_ID="$FULL_RUN_ID" \
  RACER_TASKS=place_cups,place_wine_at_rack_location,sweep_to_dustpan_of_size \
  RACER_START_EPISODE=0 \
  RACER_EVAL_EPISODES=25 \
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
set_state complete "gate and 3x25 completed; outputs: $FULL_OUTPUT"
