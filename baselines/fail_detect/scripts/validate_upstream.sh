#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
upstream="$repo_root/baselines/fail_detect/upstream"
conda_bin="${CONDA_BIN:-/data/yukun/miniconda3/bin/conda}"
env_name="${FAIL_DETECT_ENV:-dynamac-fail-detect}"

export PYTHONPATH="$upstream${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TMPDIR="${TMPDIR:-/data/yukun/tmp}"

"$conda_bin" run -n "$env_name" python -m pip check
"$conda_bin" run -n "$env_name" python -c \
  'import dm_control, mujoco; print("mujoco", mujoco.__version__)'
"$conda_bin" run -n "$env_name" python -c \
  'import mujoco_py, r3m, robomimic, robosuite, torch; print("policy", torch.__version__, torch.version.cuda)'

(
  cd "$upstream"
  "$conda_bin" run -n "$env_name" python train.py \
    --config-name=image_transport_ph_visual_flow_policy_cnn --help >/dev/null
  "$conda_bin" run -n "$env_name" python UQ_test/eval_together.py --help >/dev/null
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  "$conda_bin" run -n "$env_name" python -c \
    'import torch; x=torch.randn(1024,1024,device="cuda"); y=x@x; torch.cuda.synchronize(); print(torch.cuda.get_device_name(), float(y.mean()))'
fi
