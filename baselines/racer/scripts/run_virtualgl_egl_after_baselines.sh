#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACER_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKTREE_ROOT=$(git -C "$RACER_ROOT" rev-parse --show-toplevel)
ACTOR_PY='/data/yukun/miniconda3/envs/dynamac-racer/bin/python'
SESSION_PROBE="$SCRIPT_DIR/tmux_session_has_live_pane.sh"
OWNED_PROCESS_AUDIT="$SCRIPT_DIR/audit_owned_processes.py"
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
EPISODE_GATE_TIMEOUT_SECONDS=${RACER_EPISODE_GATE_TIMEOUT_SECONDS:-3600}
CAPTURE_TIMEOUT_SECONDS=${RACER_CAPTURE_TIMEOUT_SECONDS:-600}
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

verify_expected_checkout() {
  local current_head
  current_head=$(git -C "$WORKTREE_ROOT" rev-parse HEAD) || return 1
  [[ "$current_head" == "$EXPECTED_HEAD" ]] || return 1
  [[ "$(git -C "$WORKTREE_ROOT" branch --show-current)" == 'repro/racer-egl' ]] || \
    return 1
  [[ -z "$(git -C "$WORKTREE_ROOT" status --porcelain --untracked-files=no)" ]]
}

audit_stage() {
  local stage=$1
  local run_id=$2
  env -u RACER_RUN_ID -u RACER_OWNER_TOKEN \
    "$ACTOR_PY" "$OWNED_PROCESS_AUDIT" --value "$run_id" \
      --output "$SUPERVISOR_RUNTIME/${stage}_process_audit.json" \
      >"$SUPERVISOR_RUNTIME/${stage}_process_audit_stdout.json"
}

[[ "$WORKTREE_ROOT" == '/data/yukun/worktrees/essay2608-racer-egl' ]] || \
  fail_terminal invalid_worktree "unexpected worktree: $WORKTREE_ROOT"
EXPECTED_HEAD=$(git -C "$WORKTREE_ROOT" rev-parse HEAD) || \
  fail_terminal expected_head_unavailable 'could not resolve startup HEAD'
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || \
  fail_terminal expected_head_invalid "$EXPECTED_HEAD"
printf '%s\n' "$EXPECTED_HEAD" >"$SUPERVISOR_RUNTIME/expected_head.txt"
[[ "$(git -C "$WORKTREE_ROOT" branch --show-current)" == 'repro/racer-egl' ]] || \
  fail_terminal invalid_branch 'expected repro/racer-egl'
[[ -z "$(git -C "$WORKTREE_ROOT" status --porcelain --untracked-files=no)" ]] || \
  fail_terminal dirty_worktree 'tracked worktree changes detected'
[[ -x "$ACTOR_PY" ]] || fail_terminal missing_actor_python "$ACTOR_PY"
[[ -x "$SESSION_PROBE" ]] || fail_terminal missing_session_probe "$SESSION_PROBE"
[[ -f "$OWNED_PROCESS_AUDIT" ]] || \
  fail_terminal missing_owned_process_audit "$OWNED_PROCESS_AUDIT"
command -v timeout >/dev/null || fail_terminal missing_timeout 'GNU timeout is unavailable'
[[ "$EPISODE_GATE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  fail_terminal invalid_episode_timeout "$EPISODE_GATE_TIMEOUT_SECONDS"
[[ "$CAPTURE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  fail_terminal invalid_capture_timeout "$CAPTURE_TIMEOUT_SECONDS"
verify_expected_checkout || \
  fail_terminal startup_checkout_mismatch "expected clean repro/racer-egl at $EXPECTED_HEAD"
printf 'expected_head\t%s\nepisode_gate_timeout_seconds\t%s\ncapture_timeout_seconds\t%s\n' \
  "$EXPECTED_HEAD" "$EPISODE_GATE_TIMEOUT_SECONDS" "$CAPTURE_TIMEOUT_SECONDS" \
  >"$SUPERVISOR_RUNTIME/frozen_contract.tsv"

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

verify_expected_checkout || \
  fail_terminal checkout_changed_while_waiting \
    "expected clean repro/racer-egl at $EXPECTED_HEAD after dependency wait"

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

verify_expected_checkout || \
  fail_terminal checkout_changed_before_gate \
    "expected clean repro/racer-egl at $EXPECTED_HEAD before episode gates"

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
  set_state egl_gate_running \
    "one fixed place_cups episode 0; success=true; zero retries; timeout=${EPISODE_GATE_TIMEOUT_SECONDS}s; full run locked"
  egl_gate_status=0
  if timeout --signal=TERM --kill-after=30s "${EPISODE_GATE_TIMEOUT_SECONDS}s" \
    env "${common_environment[@]}" \
    RACER_GL_BACKEND=virtualgl-egl \
    RACER_EGL_DEVICE="$EGL_DEVICE" \
    RACER_RUN_ID="$EGL_GATE_RUN_ID" \
    RACER_TASKS=place_cups \
    RACER_START_EPISODE=0 \
    RACER_EVAL_EPISODES=1 \
    RACER_RETRY_FOR_INVALID_ACTION_ERROR=0 \
    RACER_GATE_EVIDENCE_REQUIRED=1 \
    RACER_LOG_NAME="$EGL_GATE_LOG_NAME" \
    bash "$SCRIPT_DIR/run_three_task_eval.sh" \
    >"$SUPERVISOR_RUNTIME/egl_gate_driver.log" 2>&1; then
    egl_gate_status=0
  else
    egl_gate_status=$?
  fi
  if ! audit_stage egl_gate "$EGL_GATE_RUN_ID"; then
    fail_terminal egl_gate_residual_processes \
      "full run locked; see $SUPERVISOR_RUNTIME/egl_gate_process_audit.json"
  fi
  if (( egl_gate_status == 0 )); then
    EGL_GATE_METRICS="$RACER_ROOT/results/$EGL_GATE_RUN_ID/$EGL_GATE_LOG_NAME/metrics.json"
    EGL_GATE_ACTOR_LOG="$RACER_ROOT/runtime/$EGL_GATE_RUN_ID/actor_eval.log"
    EGL_GATE_POINT_CLOUD="$RACER_ROOT/runtime/$EGL_GATE_RUN_ID/gate_point_cloud_evidence.json"
    if "$ACTOR_PY" "$SCRIPT_DIR/validate_single_episode.py" \
      --metrics "$EGL_GATE_METRICS" --actor-log "$EGL_GATE_ACTOR_LOG" \
      --point-cloud-evidence "$EGL_GATE_POINT_CLOUD" \
      >"$SUPERVISOR_RUNTIME/egl_gate_validation.json"; then
      selected_backend='virtualgl-egl'
      selected_egl_device="$EGL_DEVICE"
      set_state egl_gate_passed \
        '1/1 successful episode; decoded four-view GIFs and four finite/nondegenerate point clouds; natural status 0; no retry'
    else
      egl_failure_detail="strict EGL gate validation failed; see $SUPERVISOR_RUNTIME/egl_gate_validation.json"
    fi
  else
    egl_failure_detail="EGL episode process exited $egl_gate_status (124 means timeout); see $SUPERVISOR_RUNTIME/egl_gate_driver.log"
  fi
else
  egl_failure_detail="EGL device mapping failed; see $SUPERVISOR_RUNTIME/egl_device_mapping_error.log"
fi

if [[ -z "$selected_backend" ]]; then
  verify_expected_checkout || \
    fail_terminal checkout_changed_before_fallback \
      "expected clean repro/racer-egl at $EXPECTED_HEAD before isolation fallback"
  claim_once spawn_isolation_fallback
  set_state egl_gate_failed_starting_isolation_consistency \
    "$egl_failure_detail; running the one allowed spawn-isolation fallback path"
  consistency_status=0
  if env "${common_environment[@]}" \
    RACER_RUN_ID="$ISOLATION_CONSISTENCY_RUN_ID" \
    RACER_TASKS=place_cups \
    RACER_START_EPISODE=0 \
    RACER_CAPTURE_TIMEOUT_SECONDS="$CAPTURE_TIMEOUT_SECONDS" \
    bash "$SCRIPT_DIR/run_spawn_isolation_consistency.sh" \
    >"$SUPERVISOR_RUNTIME/isolation_consistency_driver.log" 2>&1; then
    consistency_status=0
  else
    consistency_status=$?
  fi
  if ! audit_stage isolation_consistency "$ISOLATION_CONSISTENCY_RUN_ID"; then
    fail_terminal isolation_consistency_residual_processes \
      "full run locked; no retry; see $SUPERVISOR_RUNTIME/isolation_consistency_process_audit.json"
  fi
  if (( consistency_status != 0 )); then
    fail_terminal isolation_consistency_failed \
      "full run locked; no retry; exit=$consistency_status; see $SUPERVISOR_RUNTIME/isolation_consistency_driver.log"
  fi

  set_state isolation_consistency_passed \
    'direct and isolated initialization observations match exactly; worker closed naturally with status 0'
  verify_expected_checkout || \
    fail_terminal checkout_changed_before_isolation_gate \
      "expected clean repro/racer-egl at $EXPECTED_HEAD before isolated episode gate"
  set_state isolation_gate_running \
    "the only spawn-isolated place_cups episode 0; success=true; zero retries; timeout=${EPISODE_GATE_TIMEOUT_SECONDS}s; full run locked"
  isolation_gate_status=0
  if timeout --signal=TERM --kill-after=30s "${EPISODE_GATE_TIMEOUT_SECONDS}s" \
    env "${common_environment[@]}" \
    RACER_GL_BACKEND=spawn-isolated-software \
    RACER_RUN_ID="$ISOLATION_GATE_RUN_ID" \
    RACER_TASKS=place_cups \
    RACER_START_EPISODE=0 \
    RACER_EVAL_EPISODES=1 \
    RACER_RETRY_FOR_INVALID_ACTION_ERROR=0 \
    RACER_GATE_EVIDENCE_REQUIRED=1 \
    RACER_LOG_NAME="$ISOLATION_GATE_LOG_NAME" \
    bash "$SCRIPT_DIR/run_three_task_eval.sh" \
    >"$SUPERVISOR_RUNTIME/isolation_gate_driver.log" 2>&1; then
    isolation_gate_status=0
  else
    isolation_gate_status=$?
  fi
  if ! audit_stage isolation_gate "$ISOLATION_GATE_RUN_ID"; then
    fail_terminal isolation_gate_residual_processes \
      "full run locked; no retry; see $SUPERVISOR_RUNTIME/isolation_gate_process_audit.json"
  fi
  if (( isolation_gate_status != 0 )); then
    fail_terminal isolation_gate_failed \
      "full run locked; no retry; exit=$isolation_gate_status (124 means timeout); see $SUPERVISOR_RUNTIME/isolation_gate_driver.log"
  fi

  ISOLATION_GATE_METRICS="$RACER_ROOT/results/$ISOLATION_GATE_RUN_ID/$ISOLATION_GATE_LOG_NAME/metrics.json"
  ISOLATION_GATE_ACTOR_LOG="$RACER_ROOT/runtime/$ISOLATION_GATE_RUN_ID/actor_eval.log"
  ISOLATION_GATE_POINT_CLOUD="$RACER_ROOT/runtime/$ISOLATION_GATE_RUN_ID/gate_point_cloud_evidence.json"
  if ! "$ACTOR_PY" "$SCRIPT_DIR/validate_single_episode.py" \
    --metrics "$ISOLATION_GATE_METRICS" --actor-log "$ISOLATION_GATE_ACTOR_LOG" \
    --point-cloud-evidence "$ISOLATION_GATE_POINT_CLOUD" \
    >"$SUPERVISOR_RUNTIME/isolation_gate_validation.json"; then
    fail_terminal isolation_gate_validation_failed \
      "full run locked; no retry; see $SUPERVISOR_RUNTIME/isolation_gate_validation.json"
  fi
  selected_backend='spawn-isolated-software'
  set_state isolation_gate_passed \
    '1/1 successful isolated episode; decoded four-view GIFs and four finite/nondegenerate point clouds; natural evaluator/worker status 0; no retry'
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

verify_expected_checkout || \
  fail_terminal checkout_changed_before_full \
    "expected clean repro/racer-egl at $EXPECTED_HEAD before 3x25"
set_state full_running \
  "three tasks x 25 fixed episodes; backend=$selected_backend; official InvalidActionError retries=5"
full_status=0
if env "${common_environment[@]}" "${backend_environment[@]}" \
  RACER_RUN_ID="$FULL_RUN_ID" \
  RACER_TASKS=place_cups,place_wine_at_rack_location,sweep_to_dustpan_of_size \
  RACER_START_EPISODE=0 \
  RACER_EVAL_EPISODES=25 \
  RACER_RETRY_FOR_INVALID_ACTION_ERROR=5 \
  RACER_LOG_NAME="$FULL_LOG_NAME" \
  bash "$SCRIPT_DIR/run_three_task_eval.sh" \
  >"$SUPERVISOR_RUNTIME/full_driver.log" 2>&1; then
  full_status=0
else
  full_status=$?
fi
if ! audit_stage full "$FULL_RUN_ID"; then
  fail_terminal full_residual_processes \
    "see $SUPERVISOR_RUNTIME/full_process_audit.json"
fi
if (( full_status != 0 )); then
  fail_terminal full_failed "see $SUPERVISOR_RUNTIME/full_driver.log"
fi

FULL_OUTPUT="$RACER_ROOT/results/$FULL_RUN_ID/$FULL_LOG_NAME"
for artifact in metrics.json comparison.json comparison.csv comparison.md paper_vs_reproduction.png; do
  [[ -s "$FULL_OUTPUT/$artifact" ]] || \
    fail_terminal full_validation_failed "missing or empty $FULL_OUTPUT/$artifact"
done

terminal_state=1
set_state complete "backend=$selected_backend; gate and 3x25 completed; outputs: $FULL_OUTPUT"
