#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
status_file="${FAIL_DETECT_STATUS_FILE:-$repo_root/baselines/fail_detect/runtime/quant_pipeline/status.json}"
gpu_index="${FAIL_DETECT_GPU_INDEX:-5}"
spr_session="${SPR_TMUX_SESSION:-dynamac_spr_full_20260821}"
guardian_session="${GUARDIAN_TMUX_SESSION:-dynamac_guardian_full_20260821}"
wait_mode=false
poll_seconds="${FAIL_DETECT_GATE_POLL_SECONDS:-60}"
tmux_cmd=(tmux)
if [[ -n "${FAIL_DETECT_TMUX_SOCKET:-}" ]]; then
  tmux_cmd+=(-L "$FAIL_DETECT_TMUX_SOCKET")
fi

session_active() {
  local session="$1"
  local pane_states
  if ! "${tmux_cmd[@]}" has-session -t "$session" 2>/dev/null; then
    return 1
  fi
  if ! pane_states="$("${tmux_cmd[@]}" list-panes -t "$session" -F '#{pane_dead}' 2>/dev/null)"; then
    # A known session whose panes cannot be inspected is conservatively active.
    return 0
  fi
  grep -qx '0' <<<"$pane_states"
}

if [[ "${1:-}" == "--session-active" ]]; then
  if [[ $# -ne 2 ]]; then
    echo "usage: $0 --session-active SESSION" >&2
    exit 2
  fi
  session_active "$2"
  exit $?
elif [[ "${1:-}" == "--wait" ]]; then
  wait_mode=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--wait]" >&2
  exit 2
fi

check_gate() {
  local blockers=()
  local session
  for session in "$spr_session" "$guardian_session"; do
    if session_active "$session"; then
      blockers+=("tmux:$session")
    fi
  done

  local compute_pids
  compute_pids="$(nvidia-smi -i "$gpu_index" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "$compute_pids" ]]; then
    blockers+=("gpu${gpu_index}:compute_pids=$(tr '\n' ',' <<<"$compute_pids" | sed 's/,$//')")
  fi

  local used_mib
  used_mib="$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ ! "$used_mib" =~ ^[0-9]+$ ]]; then
    blockers+=("gpu${gpu_index}:unreadable")
  elif (( used_mib > 512 )); then
    blockers+=("gpu${gpu_index}:memory_used=${used_mib}MiB")
  fi

  if (( ${#blockers[@]} > 0 )); then
    printf '%s\n' "${blockers[*]}"
    return 1
  fi
  printf 'gate-open gpu=%s memory_used=%sMiB\n' "$gpu_index" "$used_mib"
}

while true; do
  if detail="$(check_gate)"; then
    python3 "$repo_root/baselines/fail_detect/scripts/quant_status.py" "$status_file" update \
      --state ready --stage resource_gate --detail "$detail"
    printf '%s\n' "$detail"
    exit 0
  fi
  python3 "$repo_root/baselines/fail_detect/scripts/quant_status.py" "$status_file" update \
    --state waiting --stage resource_gate --detail "$detail"
  printf 'resource gate closed: %s\n' "$detail" >&2
  if [[ "$wait_mode" != true ]]; then
    exit 3
  fi
  sleep "$poll_seconds"
done
