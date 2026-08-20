#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
baseline="$repo_root/baselines/fail_detect"
scripts="$baseline/scripts"
runtime="$baseline/runtime/quant_pipeline"
results="$baseline/results/external_dp_logpzo_v1"
status_file="$runtime/status.json"
artifact_lock="$runtime/artifacts.lock.json"
repair_file="$runtime/compatibility_repair.json"
validation_file="$runtime/artifact_validation.json"
feature_validation_file="$runtime/feature_validation.json"
detector_validation_file="$runtime/detector_validation.json"
deadline_marker="$runtime/deadline.triggered"
provenance_file="$baseline/provenance/external_dp_logpzo_v1.json"
upstream="$baseline/upstream"
conda_bin="${FAIL_DETECT_CONDA_BIN:-/data/yukun/miniconda3/bin/conda}"
conda_env="${FAIL_DETECT_CONDA_ENV:-dynamac-fail-detect}"
gpu_index="${FAIL_DETECT_GPU_INDEX:-5}"
source_commit="b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"
checkpoint="$baseline/checkpoints/transport_ph_dp_train0_epoch2750.ckpt"
features="$upstream/data/outputs/transport_data_diffusion.pt"
detector="$upstream/UQ_baselines/logpZO/transport_diffusion.ckpt"
id_rollouts="$results/id_rollouts.json"
ood_rollouts="$results/ood_rollouts.json"

status() {
  python3 "$scripts/quant_status.py" "$status_file" update \
    --state "$1" --stage "$2" --detail "${3:-}"
}

check_resources() {
  FAIL_DETECT_STATUS_FILE="$status_file" FAIL_DETECT_GPU_INDEX="$gpu_index" \
    bash "$scripts/resource_gate.sh"
}

run_conda() {
  "$conda_bin" run --no-capture-output -n "$conda_env" "$@"
}

inspect_generated() {
  local kind="$1"
  local path="$2"
  artifact_state_rc=0
  run_conda python "$scripts/generated_artifact_state.py" \
    --repo-root "$repo_root" --kind "$kind" --path "$path" || artifact_state_rc=$?
}

quarantine_damaged() {
  local label="$1"
  local path="$2"
  local destination="$runtime/quarantine/$3"
  [[ -e "$path" ]]
  if [[ -e "$destination" ]]; then
    echo "refusing to overwrite prior quarantine artifact: $destination" >&2
    return 1
  fi
  mkdir -p "$runtime/quarantine"
  python3 "$scripts/compatibility_repair.py" "$repair_file" register \
    "quarantine and rebuild damaged $label artifact"
  mv -- "$path" "$destination"
  status running compatibility_repair "moved damaged $label to ignored quarantine before rebuild"
}

ensure_upstream() {
  if [[ ! -d "$upstream/.git" ]]; then
    git clone --no-checkout https://github.com/CXU-TRI/FAIL-Detect.git "$upstream"
    git -C "$upstream" checkout --detach "$source_commit"
  fi
  [[ "$(git -C "$upstream" rev-parse HEAD)" == "$source_commit" ]]
  git -C "$upstream" diff --quiet
  git -C "$upstream" diff --cached --quiet
  local config_link="$upstream/diffusion_policy/config"
  if [[ ! -e "$config_link" && ! -L "$config_link" ]]; then
    ln -s configs_robomimic "$config_link"
  fi
  [[ -L "$config_link" && "$(readlink "$config_link")" == "configs_robomimic" ]]
}

export_features() {
  check_resources
  status running features "extracting released global_cond/action training tensors"
  (
    cd "$upstream"
    CUDA_VISIBLE_DEVICES="$gpu_index" run_conda python save_data.py \
      --config-dir=diffusion_policy/configs_robomimic \
      --config-name=image_transport_ph_visual_diffusion_policy_cnn.yaml \
      training.seed=1103 training.device=cuda:0 \
      "hydra.run.dir=$runtime/hydra/save_data"
  )
}

ensure_features() {
  local repaired=false
  inspect_generated features "$features"
  case "$artifact_state_rc" in
    0) return ;;
    10) ;;
    12)
      quarantine_damaged features "$features" "transport_data_diffusion.partial.pt"
      repaired=true
      ;;
    *) echo "unexpected feature artifact state exit: $artifact_state_rc" >&2; return 1 ;;
  esac
  export_features
  inspect_generated features "$features"
  if [[ "$artifact_state_rc" -eq 0 ]]; then
    return
  fi
  if [[ "$artifact_state_rc" -eq 12 && "$repaired" == false ]]; then
    quarantine_damaged features "$features" "transport_data_diffusion.partial.pt"
    export_features
    inspect_generated features "$features"
  fi
  [[ "$artifact_state_rc" -eq 0 ]]
}

train_detector() {
  check_resources
  (
    cd "$upstream/UQ_baselines/logpZO"
    CUDA_VISIBLE_DEVICES="$gpu_index" run_conda python train.py \
      --policy_type=diffusion --type=transport
  )
}

ensure_detector() {
  local repaired=false
  inspect_generated detector "$detector"
  case "$artifact_state_rc" in
    0) return ;;
    10) status running logpzo_train "starting released logpZO EPOCHS=200 training" ;;
    11) status running logpzo_resume "invoking the official train.py checkpoint resume path" ;;
    12)
      quarantine_damaged detector "$detector" "transport_diffusion.partial.ckpt"
      repaired=true
      status running logpzo_train "restarting released logpZO training after damaged checkpoint"
      ;;
    *) echo "unexpected detector artifact state exit: $artifact_state_rc" >&2; return 1 ;;
  esac
  train_detector
  inspect_generated detector "$detector"
  if [[ "$artifact_state_rc" -eq 0 ]]; then
    return
  fi
  if [[ "$repaired" == false && ( "$artifact_state_rc" -eq 11 || "$artifact_state_rc" -eq 12 ) ]]; then
    quarantine_damaged detector "$detector" "transport_diffusion.partial.ckpt"
    repaired=true
    status running logpzo_train "restarting once after official resume did not yield a complete checkpoint"
    train_detector
    inspect_generated detector "$detector"
  fi
  [[ "$artifact_state_rc" -eq 0 ]]
}

evaluate() {
  local episodes="$1"
  local parallel_envs="$2"
  local modify_flag=()
  local output="$id_rollouts"
  check_resources
  if [[ "$3" == "ood" ]]; then
    modify_flag=(--modify)
    output="$ood_rollouts"
  fi
  CUDA_VISIBLE_DEVICES="$gpu_index" run_conda python "$scripts/logpzo_evaluate.py" \
    --repo-root "$repo_root" \
    --checkpoint "$checkpoint" \
    --logpzo-checkpoint "$detector" \
    --artifact-lock "$artifact_lock" \
    --output "$output" \
    --episodes "$episodes" \
    --start-seed 100000 \
    --parallel-envs "$parallel_envs" \
    --device cuda:0 \
    "${modify_flag[@]}"
}

if [[ "${1:-}" == "--wait" ]]; then
  mkdir -p "$runtime"
  status waiting resource_gate "waiting for SPR and Guardian tmux sessions to end and GPU5 to become idle"
  FAIL_DETECT_STATUS_FILE="$status_file" FAIL_DETECT_GPU_INDEX="$gpu_index" \
    bash "$scripts/resource_gate.sh" --wait
  exec "$scripts/deadline_runner.sh" \
    --status-script "$scripts/quant_status.py" \
    --status-file "$status_file" \
    --deadline-marker "$deadline_marker" \
    --duration "${FAIL_DETECT_DEADLINE:-24h}" \
    --kill-after "${FAIL_DETECT_KILL_AFTER:-5m}" \
    -- "$0" --run
elif [[ "${1:-}" != "--run" ]]; then
  echo "usage: $0 --wait|--run" >&2
  exit 2
fi

mkdir -p "$runtime" "$results"
if ! mkdir "$runtime/run.lock" 2>/dev/null; then
  status stopped lock "another run owns $runtime/run.lock"
  exit 5
fi
printf '%s\n' "$$" >"$runtime/run.lock/pid"

cleanup() {
  rm -f "$runtime/run.lock/pid"
  rmdir "$runtime/run.lock" 2>/dev/null || true
}
mark_deadline() {
  : >"$deadline_marker"
  status stopping deadline "inner pipeline received TERM; outer deadline runner owns final state"
  exit 124
}
trap cleanup EXIT
trap mark_deadline TERM INT

status running preflight "post-gate 24-hour clock started"
check_resources
python3 "$scripts/compatibility_repair.py" "$repair_file" check
[[ -x "$conda_bin" ]]
"$conda_bin" env list | grep -Fq "$conda_env"

status running source "pinning official FAIL-Detect source"
ensure_upstream
status running module_smoke "importing real CFM.net_CFM and strict-loading logpZO architecture"
CUDA_VISIBLE_DEVICES="" run_conda python "$scripts/smoke_logpzo_module.py" \
  --upstream "$upstream" --expected-commit "$source_commit"

status running artifacts "downloading and SHA-locking official external artifacts"
python3 "$scripts/prepare_official_artifacts.py" \
  --repo-root "$repo_root" \
  --manifest "$baseline/quant_artifacts.json" \
  --lock-file "$artifact_lock" \
  --gpu-index "$gpu_index"

status running validation "validating HDF5 schema and strict-loading policy checkpoint"
run_conda python "$scripts/validate_quant_artifacts.py" \
  --repo-root "$repo_root" \
  --artifact-lock "$artifact_lock" \
  --output "$validation_file"

ensure_features
status running feature_validation "validating generated feature tensor schema"
run_conda python "$scripts/validate_logpzo_inputs.py" \
  --repo-root "$repo_root" --features "$features" --output "$feature_validation_file"

ensure_detector
status running detector_validation "strict-loading released logpZO checkpoint"
run_conda python "$scripts/validate_logpzo_inputs.py" \
  --repo-root "$repo_root" --features "$features" --checkpoint "$detector" \
  --output "$detector_validation_file"

check_resources
status running gate_rollouts "running paired 10 ID + 10 OOD gate"
evaluate 10 5 id
evaluate 10 5 ood
status running gate_statistics "validating the technical gate without detector-performance cherry-picking"
run_conda python "$scripts/summarize_logpzo.py" \
  --repo-root "$repo_root" --id-rollouts "$id_rollouts" --ood-rollouts "$ood_rollouts" \
  --output-json "$results/gate_summary.json" --output-md "$results/gate_summary.md" \
  --output-provenance "$provenance_file" --artifact-lock "$artifact_lock" \
  --input-validation "$detector_validation_file" \
  --limit 10 --calibration-successes 4 --alpha 0.05 --gate \
  --minimum-id-success-rate 0.7

check_resources
status running final_rollouts "extending the same paired seeds to 50 ID + 50 OOD"
evaluate 50 10 id
evaluate 50 10 ood
status running final_statistics "computing Wilson intervals and released conformal-band metrics"
run_conda python "$scripts/summarize_logpzo.py" \
  --repo-root "$repo_root" --id-rollouts "$id_rollouts" --ood-rollouts "$ood_rollouts" \
  --output-json "$results/final_summary.json" --output-md "$results/final_summary.md" \
  --output-provenance "$provenance_file" --artifact-lock "$artifact_lock" \
  --input-validation "$detector_validation_file" \
  --limit 50 --calibration-successes 20 --alpha 0.05
