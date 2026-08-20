#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
baseline_dir="$(cd "$script_dir/.." && pwd)"
runtime_dir="$baseline_dir/runtime"
results_dir="$baseline_dir/results"
aha_root="${AHA_SOURCE_ROOT:-/data/yukun/essay2608/baselines/aha/upstream/aha}"
cop_root="${COPPELIASIM_ROOT:-/data/yukun/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
cache_dir="${AHA_CACHE_DIR:-/data/yukun/.cache/dynamac-baselines/aha}"
xvfb_bin="${XVFB_BIN:-$cache_dir/xvfb-ubuntu-root/usr/bin/Xvfb}"
conda_root="${CONDA_ROOT:-/data/yukun/miniconda3}"
official_commit="0d39c0591566ddaf997be5822f3dead8e08501aa"
official_eval="$aha_root/Data_Generation/rlbench-failgen/examples/ex_data_generator_eval.sh"

case "${1-}" in
  "") mode=run ;;
  --static-check) mode=static ;;
  *) echo "Usage: $0 [--static-check]" >&2; exit 2 ;;
esac

if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
  echo "Conda activation script not found: $conda_root" >&2
  exit 2
fi
if ! git -C "$aha_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Pinned AHA checkout not found: $aha_root" >&2
  exit 2
fi
observed_commit="$(git -C "$aha_root" rev-parse HEAD)"
if [[ "$observed_commit" != "$official_commit" ]]; then
  echo "AHA commit mismatch: expected $official_commit, got $observed_commit" >&2
  exit 2
fi
if [[ ! -f "$official_eval" ]]; then
  echo "Official eval script not found: $official_eval" >&2
  exit 2
fi

source "$conda_root/etc/profile.d/conda.sh"
conda activate dynamac-aha

common_args=(
  --output-root "$results_dir/static-check-placeholder"
  --worker "$script_dir/failgen_one_episode.py"
  --official-eval-script "$official_eval"
  --xvfb-bin "$xvfb_bin"
  --cop-root "$cop_root"
  --total-timeout-seconds 7200
  --task-timeout-seconds 600
  --attempt-timeout-seconds 300
  --max-restarts 1
  --max-tries 1
)

if [[ "$mode" == static ]]; then
  python "$script_dir/failgen_eval10.py" "${common_args[@]}" --static-check
  exit 0
fi

if [[ ! -x "$xvfb_bin" ]]; then
  echo "Xvfb is not executable: $xvfb_bin" >&2
  exit 2
fi
if [[ ! -d "$cop_root" ]]; then
  echo "CoppeliaSim root not found: $cop_root" >&2
  exit 2
fi

mkdir -p "$runtime_dir" "$results_dir"
claim_dir="$runtime_dir/failgen_eval10.claim"
if ! mkdir "$claim_dir" 2>/dev/null; then
  echo "Another AHA eval claim already exists: $claim_dir" >&2
  exit 3
fi
cleanup_claim() {
  rmdir "$claim_dir" >/dev/null 2>&1 || true
}
trap cleanup_claim EXIT

echo "Waiting for all current SPR and Guardian tmux sessions to terminate."
while true; do
  active_sessions="$(python "$script_dir/tmux_gate.py")"
  if [[ -z "$active_sessions" ]]; then
    break
  fi
  echo "Still gated by: $(tr '\n' ' ' <<<"$active_sessions")"
  sleep 60
done

# Recheck immediately before launching the first simulator process.
active_sessions="$(python "$script_dir/tmux_gate.py")"
if [[ -n "$active_sessions" ]]; then
  echo "Gate changed during launch; refusing to start AHA." >&2
  exit 4
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$results_dir/failgen_eval10_$run_id"
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1
unset CUDA_DEVICE_ORDER LIBGL_DRIVERS_PATH MESA_LOADER_DRIVER_OVERRIDE GALLIUM_DRIVER

echo "Starting bounded AHA FailGen eval on CPU/llvmpipe: $output_root"
python "$script_dir/failgen_eval10.py" \
  --output-root "$output_root" \
  --worker "$script_dir/failgen_one_episode.py" \
  --official-eval-script "$official_eval" \
  --xvfb-bin "$xvfb_bin" \
  --cop-root "$cop_root" \
  --total-timeout-seconds 7200 \
  --task-timeout-seconds 600 \
  --attempt-timeout-seconds 300 \
  --max-restarts 1 \
  --max-tries 1
