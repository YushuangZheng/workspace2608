#!/usr/bin/env bash
set -euo pipefail

SPR_BASE=/data/yukun/essay2608/baselines/spr
SPR_UPSTREAM="$SPR_BASE/upstream"
SPR_CHECKPOINT="$SPR_BASE/checkpoints/libero_10"
SPR_RUN_ROOT="$SPR_BASE/results/released_code_libero10_task0"
SPR_ENV_PYTHON=/data/yukun/miniconda3/envs/dynamac-spr/bin/python
SPR_SELECTED_GPUS="${SPR_CUDA_VISIBLE_DEVICES:-1,2,3,4}"

IFS=',' read -r -a SPR_GPU_ARRAY <<< "$SPR_SELECTED_GPUS"
if (( ${#SPR_GPU_ARRAY[@]} < 1 || ${#SPR_GPU_ARRAY[@]} > 4 )); then
    echo "SPR_CUDA_VISIBLE_DEVICES must select between one and four GPUs" >&2
    exit 2
fi

mkdir -p "$SPR_RUN_ROOT"
ln -sfn "$SPR_CHECKPOINT" "$SPR_RUN_ROOT/libero_10"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$SPR_SELECTED_GPUS"
export HF_HOME=/data/yukun/.cache/dynamac-baselines/huggingface
export HF_HUB_CACHE=/data/yukun/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH="$SPR_BASE/runtime/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME=/data/yukun/.cache/dynamac-baselines

SPR_RUN_ID="$(date +%Y%m%d_%H%M%S)"
SPR_LOG="$SPR_RUN_ROOT/$SPR_RUN_ID.log"

cd "$SPR_RUN_ROOT"
{
    echo "run_id=$SPR_RUN_ID"
    echo "upstream_commit=$(git -C "$SPR_UPSTREAM" rev-parse HEAD)"
    echo "checkpoint_revision=b5838d84d462abd41a45c2b3e7258fa11ec0ed0f"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    "$SPR_ENV_PYTHON" --version
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    "$SPR_ENV_PYTHON" "$SPR_UPSTREAM/experiments/libero/run_libero_eval_vllm.py" \
        --task 10 \
        --task_id 0 \
        --checkpoint libero_10
} 2>&1 | tee "$SPR_LOG"
