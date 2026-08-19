#!/usr/bin/env bash
set -euo pipefail

# Frozen follow-up queue. Do not start this until GPU scheduling is confirmed.
SPR_BASE=/data/yukun/essay2608/baselines/spr
SPR_UPSTREAM="$SPR_BASE/upstream"
SPR_CHECKPOINT="$SPR_BASE/checkpoints/libero_10"
SPR_ENV_PYTHON=/data/yukun/miniconda3/envs/dynamac-spr/bin/python
SPR_SELECTED_GPUS="${SPR_CUDA_VISIBLE_DEVICES:-1,2,3,4}"
SPR_TASK_IDS=(1 2 3 4 5 6 7 8 9)

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

IFS=',' read -r -a SPR_GPU_ARRAY <<< "$SPR_SELECTED_GPUS"
if (( ${#SPR_GPU_ARRAY[@]} < 1 || ${#SPR_GPU_ARRAY[@]} > 4 )); then
    echo "SPR_CUDA_VISIBLE_DEVICES must select between one and four GPUs" >&2
    exit 2
fi

test "$(git -C "$SPR_UPSTREAM" rev-parse HEAD)" = d57e4b81ebdcacea574b68be29d61ba04cdc7051
test "$(sha256sum "$SPR_UPSTREAM/experiments/libero/run_libero_eval_vllm.py" | awk '{print $1}')" = f8785337c4711f5f40fe5961a788f06587f366117d252b89c60b6fec1c90f4fb

selected_gpu_processes() {
    local SPR_GPU_INDEX=$1
    nvidia-smi -i "$SPR_GPU_INDEX" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader
}

ensure_selected_gpus_ready() {
    local SPR_ALLOW_PREVIOUS_WORKERS=$1
    local SPR_WAIT_STARTED
    local SPR_GPU_INDEX
    local SPR_PROCESSES
    local SPR_PROCESS_PID
    local SPR_PROCESS_EXE
    local SPR_BUSY=0

    SPR_WAIT_STARTED=$(date +%s)
    while true; do
        SPR_BUSY=0
        for SPR_GPU_INDEX in "${SPR_GPU_ARRAY[@]}"; do
            if ! SPR_PROCESSES=$(selected_gpu_processes "$SPR_GPU_INDEX" 2>&1); then
                fail "cannot inspect GPU $SPR_GPU_INDEX: $SPR_PROCESSES"
            fi
            [[ -z "$SPR_PROCESSES" ]] && continue
            SPR_BUSY=1
            if (( SPR_ALLOW_PREVIOUS_WORKERS == 0 )); then
                fail "GPU $SPR_GPU_INDEX is busy before task launch: $SPR_PROCESSES"
            fi
            while IFS=',' read -r SPR_PROCESS_PID _; do
                SPR_PROCESS_PID=${SPR_PROCESS_PID//[[:space:]]/}
                [[ -r "/proc/$SPR_PROCESS_PID/exe" ]] || continue
                SPR_PROCESS_EXE=$(readlink -f "/proc/$SPR_PROCESS_PID/exe" || true)
                if [[ "$SPR_PROCESS_EXE" != /data/yukun/miniconda3/envs/dynamac-spr/bin/python* ]]; then
                    fail "GPU $SPR_GPU_INDEX has a non-SPR process after the previous task: $SPR_PROCESSES"
                fi
            done <<< "$SPR_PROCESSES"
        done
        (( SPR_BUSY == 0 )) && return 0
        if (( $(date +%s) - SPR_WAIT_STARTED >= 60 )); then
            fail "previous SPR workers did not release all selected GPUs within 60 seconds"
        fi
        sleep 5
    done
}

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

SPR_PREVIOUS_TASK_FINISHED=0
for SPR_TASK_ID in "${SPR_TASK_IDS[@]}"; do
    ensure_selected_gpus_ready "$SPR_PREVIOUS_TASK_FINISHED"
    SPR_RUN_ROOT="$SPR_BASE/results/released_code_libero10_task${SPR_TASK_ID}"
    SPR_RUN_ID="$(date +%Y%m%d_%H%M%S)"
    SPR_LOG="$SPR_RUN_ROOT/$SPR_RUN_ID.log"
    mkdir -p "$SPR_RUN_ROOT"
    ln -sfn "$SPR_CHECKPOINT" "$SPR_RUN_ROOT/libero_10"
    cd "$SPR_RUN_ROOT"
    {
        echo "run_id=$SPR_RUN_ID"
        echo "task_id=$SPR_TASK_ID"
        echo "upstream_commit=$(git -C "$SPR_UPSTREAM" rev-parse HEAD)"
        echo "evaluator_sha256=f8785337c4711f5f40fe5961a788f06587f366117d252b89c60b6fec1c90f4fb"
        echo "checkpoint_revision=b5838d84d462abd41a45c2b3e7258fa11ec0ed0f"
        echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
        "$SPR_ENV_PYTHON" --version
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
        "$SPR_ENV_PYTHON" "$SPR_UPSTREAM/experiments/libero/run_libero_eval_vllm.py" \
            --task 10 \
            --task_id "$SPR_TASK_ID" \
            --checkpoint libero_10
    } 2>&1 | tee "$SPR_LOG"
    SPR_PREVIOUS_TASK_FINISHED=1
done
