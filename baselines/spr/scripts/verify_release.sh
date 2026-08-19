#!/usr/bin/env bash
set -euo pipefail

SPR_BASE=/data/yukun/essay2608/baselines/spr
SPR_UPSTREAM="$SPR_BASE/upstream"
SPR_CHECKPOINT="$SPR_BASE/checkpoints/libero_10"
SPR_ENV_PYTHON=/data/yukun/miniconda3/envs/dynamac-spr/bin/python

export LIBERO_CONFIG_PATH="$SPR_BASE/runtime/libero_config"
export XDG_CACHE_HOME=/data/yukun/.cache/dynamac-baselines
export HF_HOME=/data/yukun/.cache/dynamac-baselines/huggingface
export HF_HUB_CACHE=/data/yukun/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export VLLM_WORKER_MULTIPROC_METHOD=spawn

test "$(git -C "$SPR_UPSTREAM" rev-parse HEAD)" = d57e4b81ebdcacea574b68be29d61ba04cdc7051

"$SPR_ENV_PYTHON" - <<'PY'
import json
from pathlib import Path

from safetensors import safe_open

checkpoint = Path("/data/yukun/essay2608/baselines/spr/checkpoints/libero_10")
index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (checkpoint / name).is_file()]
assert not missing, f"missing checkpoint shards: {missing}"

tensor_count = 0
for shard in shards:
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
        tensor_count += len(list(handle.keys()))

assert tensor_count == len(index["weight_map"]) == 614
assert index["metadata"]["total_size"] == 16_238_835_616
print(f"checkpoint: {len(shards)} shards, {tensor_count} tensors, index OK")
PY

cd "$SPR_UPSTREAM/experiments/libero"
"$SPR_ENV_PYTHON" - <<'PY'
from sprvla import SPRVLAParser

checkpoint = "/data/yukun/essay2608/baselines/spr/checkpoints/libero_10"
parser = SPRVLAParser.from_pretrained(checkpoint)
assert "libero_10_no_noops_modified" in parser.norm_stats
print("parser: pinned Qwen2 tokenizer cache and LIBERO-10 normalization stats OK")
PY

"$SPR_ENV_PYTHON" run_libero_eval_vllm.py --help
