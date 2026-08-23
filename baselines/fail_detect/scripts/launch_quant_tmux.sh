#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
session="${FAIL_DETECT_TMUX_SESSION:-dynamac_fail_detect_quant_20260821}"
runtime="$repo_root/baselines/fail_detect/runtime/quant_pipeline"
log_file="$runtime/pipeline.log"

mkdir -p "$runtime"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 2
fi
tmux new-session -d -s "$session" \
  "cd '$repo_root' && bash baselines/fail_detect/scripts/run_quant_pipeline.sh --wait >>'$log_file' 2>&1"
echo "started $session; log=$log_file"
