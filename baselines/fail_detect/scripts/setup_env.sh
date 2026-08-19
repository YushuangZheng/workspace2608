#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
upstream="$repo_root/baselines/fail_detect/upstream"
conda_bin="${CONDA_BIN:-/data/yukun/miniconda3/bin/conda}"
env_name="${FAIL_DETECT_ENV:-dynamac-fail-detect}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/yukun/.cache/dynamac-baselines}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/data/yukun/.cache/conda-pkgs-fail-detect}"
export TMPDIR="${TMPDIR:-/data/yukun/tmp}"
unset PIP_EXTRA_INDEX_URL

if ! "$conda_bin" env list | awk '{print $1}' | grep -qx "$env_name"; then
  if ! "$conda_bin" env create -y -n "$env_name" -f "$upstream/conda_environment.yaml"; then
    # In 2026 the YAML's unbounded dm-control dependency selects a source-only
    # MuJoCo build and the pip phase exits after the conda prefix is complete.
    # Continue only for that recoverable partial-environment case.
    "$conda_bin" env list | awk '{print $1}' | grep -qx "$env_name"
  fi
fi

# The upstream prerequisites require patchelf.  Keep it in the user's isolated
# conda environment instead of changing the server's system installation.
"$conda_bin" install -y -n "$env_name" -c conda-forge patchelf

# dm-control==1.0.9 only lower-bounds MuJoCo.  In 2026 pip resolves that to a
# source-only MuJoCo release which requires an external MUJOCO_PATH.  Pinning
# its first compatible wheel preserves the 2023 environment semantics.
"$conda_bin" run -n "$env_name" python -m pip install \
  'mujoco==2.3.1.post1'

"$conda_bin" run -n "$env_name" python -m pip install \
  'ray[default,tune]==2.2.0' \
  'free-mujoco-py==2.1.6' \
  'pygame==2.1.2' \
  'pybullet-svl==3.1.6.4' \
  'robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz' \
  'robomimic==0.2.0' \
  'pytorchvideo==0.1.5' \
  'imagecodecs==2022.9.26' \
  'r3m @ https://github.com/facebookresearch/r3m/archive/b2334e726887fa0206962d7984c69c5fb09cceab.tar.gz' \
  'dm-control==1.0.9' \
  'lightkit' \
  'scikit-learn' \
  'gurobipy' \
  'huggingface_hub==0.25.0' \
  'mujoco==2.3.1.post1' \
  'multiprocess==0.70.13'

# Both entrypoints point at diffusion_policy/config, while the release only
# contains configs_robomimic.  The README's configs are exposed without
# changing any Python or YAML implementation.
if [[ ! -e "$upstream/diffusion_policy/config" ]]; then
  ln -s configs_robomimic "$upstream/diffusion_policy/config"
fi
[[ "$(readlink "$upstream/diffusion_policy/config")" == configs_robomimic ]]

MUJOCO_GL=osmesa "$conda_bin" run -n "$env_name" python -c \
  'import dm_control, mujoco; print("mujoco", mujoco.__version__)'
MUJOCO_GL=osmesa "$conda_bin" run -n "$env_name" python -c \
  'import mujoco_py, numpy, r3m, robomimic, robosuite, torch; print("policy", torch.__version__, torch.version.cuda, numpy.__version__)'
"$conda_bin" run -n "$env_name" python -m pip check
