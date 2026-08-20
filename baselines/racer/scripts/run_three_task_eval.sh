#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACER_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
UPSTREAM="$RACER_ROOT/upstream"
OPEN_LLAVA="$UPSTREAM/Open-LLaVA-NeXT"
ACTOR_PY='/data/yukun/miniconda3/envs/dynamac-racer/bin/python'
LLAVA_PY='/data/yukun/miniconda3/envs/dynamac-racer-llava/bin/python'
COPPELIASIM_ROOT='/data/yukun/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
XVFB='/data/yukun/.cache/racer/xvfb-ubuntu-root/usr/bin/Xvfb'
HF_HOME='/data/yukun/.cache/huggingface-racer/llava-runtime'

LM_GPU=${RACER_LM_GPU:-1}
VLM_GPUS=${RACER_VLM_GPUS:-2,3}
ACTOR_GPU=${RACER_ACTOR_GPU:-4}
LM_PORT=${RACER_LM_PORT:-18000}
VLM_PORT=${RACER_VLM_PORT:-21002}
DISPLAY_ID=${RACER_DISPLAY_ID:-95}
HEALTH_ONLY=${RACER_HEALTH_ONLY:-0}
RUN_ID=${RACER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
TASKS='place_cups,place_wine_at_rack_location,sweep_to_dustpan_of_size'
LOG_NAME='official_ckpt_three_task'
RUNTIME_DIR="$RACER_ROOT/runtime/$RUN_ID"
RESULT_DIR="$RACER_ROOT/results/$RUN_ID"

# The model services are local even if the interactive shell has proxy-on set.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

mkdir -p "$RUNTIME_DIR" "$RESULT_DIR"

children=()
actor_pid=''

cleanup() {
  status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$actor_pid" ]] && kill -0 "$actor_pid" 2>/dev/null; then
    kill -TERM -- "-$actor_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$actor_pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL -- "-$actor_pid" 2>/dev/null || true
    wait "$actor_pid" 2>/dev/null || true
  fi
  for ((index=${#children[@]}-1; index>=0; index--)); do
    pid=${children[$index]}
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null || fail 'nvidia-smi is unavailable.'
command -v curl >/dev/null || fail 'curl is unavailable.'
command -v xdpyinfo >/dev/null || fail 'xdpyinfo is unavailable.'
command -v glxinfo >/dev/null || fail 'glxinfo is unavailable.'
[[ -x "$ACTOR_PY" ]] || fail "actor interpreter missing: $ACTOR_PY"
[[ -x "$LLAVA_PY" ]] || fail "LLaVA interpreter missing: $LLAVA_PY"
[[ -x "$XVFB" ]] || fail "user-space Xvfb missing; run $SCRIPT_DIR/bootstrap_user_xvfb.sh"
[[ -f "$SCRIPT_DIR/racer_lm_server.py" ]] || fail 'RACER language-service wrapper is missing.'
[[ "$DISPLAY_ID" =~ ^[0-9]+$ ]] || fail 'RACER_DISPLAY_ID must be numeric.'
[[ "$LM_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || \
  fail 'RACER_LM_PORT must be an integer between 1 and 65535.'
(( 10#$LM_PORT <= 65535 )) || fail 'RACER_LM_PORT must be at most 65535.'

IFS=',' read -r VLM_GPU_A VLM_GPU_B VLM_GPU_EXTRA <<<"$VLM_GPUS"
[[ -n "$VLM_GPU_A" && -n "$VLM_GPU_B" && -z "${VLM_GPU_EXTRA:-}" ]] || \
  fail 'RACER_VLM_GPUS must contain exactly two comma-separated GPU indices.'

declare -A seen_gpu=()
for gpu in "$LM_GPU" "$VLM_GPU_A" "$VLM_GPU_B" "$ACTOR_GPU"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || fail "invalid GPU index: $gpu"
  [[ -z "${seen_gpu[$gpu]+present}" ]] || fail "GPU index is assigned twice: $gpu"
  seen_gpu[$gpu]=1
done

check_gpu_idle() {
  local gpu=$1
  local processes
  if ! processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>&1); then
    fail "cannot inspect GPU $gpu: $processes"
  fi
  [[ -z "$processes" ]] || fail "GPU $gpu is busy: $processes"
}

for gpu in "$LM_GPU" "$VLM_GPU_A" "$VLM_GPU_B" "$ACTOR_GPU"; do
  check_gpu_idle "$gpu"
done

port_is_free() {
  "$ACTOR_PY" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
PY
}

port_is_free "$LM_PORT" || fail "port $LM_PORT is occupied"
port_is_free "$VLM_PORT" || fail "port $VLM_PORT is occupied"

echo 'Running immutable source, artifact, dataset, and import preflight...'
"$ACTOR_PY" "$SCRIPT_DIR/verify_setup.py" >"$RUNTIME_DIR/preflight.json"

T5_ROOT="$RACER_ROOT/checkpoints/t5-11b"
T5_ADAPTER_DIR="$OPEN_LLAVA/<your-dir-to-store-t5-11b>"
T5_ADAPTER="$T5_ADAPTER_DIR/t5-11b"
mkdir -p "$T5_ADAPTER_DIR"
if [[ -L "$T5_ADAPTER" ]]; then
  [[ "$(readlink -f "$T5_ADAPTER")" == "$(readlink -f "$T5_ROOT")" ]] || \
    fail "T5 path adapter points somewhere unexpected: $T5_ADAPTER"
elif [[ -e "$T5_ADAPTER" ]]; then
  fail "T5 path adapter exists but is not a symlink: $T5_ADAPTER"
else
  ln -s "$T5_ROOT" "$T5_ADAPTER"
fi

[[ -f /data/yukun/.cache/clip/RN50.pt ]] || \
  fail 'official CLIP RN50 cache is missing: /data/yukun/.cache/clip/RN50.pt'

DISPLAY_SOCKET="/tmp/.X11-unix/X$DISPLAY_ID"
DISPLAY_LOCK="/tmp/.X${DISPLAY_ID}-lock"
[[ ! -e "$DISPLAY_LOCK" && ! -S "$DISPLAY_SOCKET" ]] || \
  fail "X display :$DISPLAY_ID is already in use"

echo "Starting user-space Xvfb on :$DISPLAY_ID..."
env -u LD_LIBRARY_PATH XDG_CACHE_HOME=/data/yukun/.cache/racer \
  "$XVFB" ":$DISPLAY_ID" -screen 0 1280x1024x24 -ac \
  +extension GLX +iglx +render -noreset \
  >"$RUNTIME_DIR/xvfb.log" 2>&1 &
xvfb_pid=$!
children+=("$xvfb_pid")
for _ in $(seq 1 100); do
  [[ -S "$DISPLAY_SOCKET" ]] && break
  kill -0 "$xvfb_pid" 2>/dev/null || fail "Xvfb exited; see $RUNTIME_DIR/xvfb.log"
  sleep 0.1
done
[[ -S "$DISPLAY_SOCKET" ]] || fail 'Xvfb did not become ready in 10 seconds.'
DISPLAY=":$DISPLAY_ID" xdpyinfo >"$RUNTIME_DIR/xdpyinfo.txt" 2>&1 || \
  fail "xdpyinfo failed; see $RUNTIME_DIR/xdpyinfo.txt"
grep -q 'GLX' "$RUNTIME_DIR/xdpyinfo.txt" || fail 'Xvfb does not expose GLX.'
DISPLAY=":$DISPLAY_ID" XDG_CACHE_HOME=/data/yukun/.cache/racer \
  LIBGL_ALWAYS_SOFTWARE=1 glxinfo -B >"$RUNTIME_DIR/glxinfo.txt" 2>&1 || \
  fail "GLX context creation failed; see $RUNTIME_DIR/glxinfo.txt"

wait_for_get() {
  local name=$1
  local url=$2
  local pid=$3
  local timeout_seconds=$4
  local elapsed=0
  while (( elapsed < timeout_seconds )); do
    if curl --connect-timeout 2 --max-time 5 -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || fail "$name exited during startup"
    sleep 2
    elapsed=$((elapsed + 2))
  done
  fail "$name health endpoint did not become ready within ${timeout_seconds}s"
}

echo "Starting T5-11B/CLIP service on GPU $LM_GPU, port $LM_PORT..."
(
  cd "$OPEN_LLAVA"
  export CUDA_VISIBLE_DEVICES="$LM_GPU"
  export HF_HOME
  export TRANSFORMERS_CACHE="$HF_HOME/hub"
  export TOKENIZERS_PARALLELISM=false
  exec "$LLAVA_PY" -u "$SCRIPT_DIR/racer_lm_server.py" \
    --open-llava-root "$OPEN_LLAVA" \
    --host 127.0.0.1 \
    --port "$LM_PORT"
) >"$RUNTIME_DIR/lm_server.log" 2>&1 &
lm_pid=$!
children+=("$lm_pid")
wait_for_get 'language service' "http://127.0.0.1:$LM_PORT/openapi.json" "$lm_pid" 240

curl --connect-timeout 3 --max-time 30 -fsS \
  -H 'Content-Type: application/json' \
  -d '{"text":"pick up the red cup","model":"t5-11b"}' \
  "http://127.0.0.1:$LM_PORT/encode/" >"$RUNTIME_DIR/lm_health.json"
"$ACTOR_PY" - "$RUNTIME_DIR/lm_health.json" <<'PY'
import json
import math
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert int(payload["token_len"]) > 0
embedding = payload["embeddings"]
assert len(embedding) == 1 and len(embedding[0]) == int(payload["token_len"])
assert len(embedding[0][0]) == 1024
assert all(math.isfinite(float(value)) for value in embedding[0][0])
PY

LORA="$RACER_ROOT/checkpoints/racer-llava-llama3-lora-rich"
LLAVA_BASE="$RACER_ROOT/checkpoints/llama3-llava-next-8b"
echo "Starting rich LLaVA service on GPUs $VLM_GPU_A,$VLM_GPU_B, port $VLM_PORT..."
(
  cd "$OPEN_LLAVA"
  export CUDA_VISIBLE_DEVICES="$VLM_GPU_A,$VLM_GPU_B"
  export HF_HOME
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export TOKENIZERS_PARALLELISM=false
  exec "$LLAVA_PY" -u "$SCRIPT_DIR/racer_llava_server.py" \
    --open-llava-root "$OPEN_LLAVA" \
    --model-path "$LORA" \
    --model-base "$LLAVA_BASE" \
    --model-name llava_llama3_lora \
    --host 127.0.0.1 \
    --port "$VLM_PORT" \
    --max-memory-gib 20
) >"$RUNTIME_DIR/llava_server.log" 2>&1 &
vlm_pid=$!
children+=("$vlm_pid")
wait_for_get 'LLaVA service' "http://127.0.0.1:$VLM_PORT/test" "$vlm_pid" 180
curl --connect-timeout 2 --max-time 5 -fsS \
  "http://127.0.0.1:$VLM_PORT/test" >"$RUNTIME_DIR/vlm_health.json"
"$ACTOR_PY" - "$RUNTIME_DIR/vlm_health.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload == {"message": "Hello World"}
PY

echo 'All source, data, GLX, model-load, and service health checks passed.'
if [[ "$HEALTH_ONLY" == '1' ]]; then
  echo 'RACER_HEALTH_ONLY=1: stopping before episode 0.'
  exit 0
fi
[[ "$HEALTH_ONLY" == '0' ]] || fail 'RACER_HEALTH_ONLY must be 0 or 1.'

# Recheck the actor card immediately before the long process starts.
check_gpu_idle "$ACTOR_GPU"

ACTOR_LOG="$RUNTIME_DIR/actor_eval.log"
echo "Starting frozen 75-episode evaluator on GPU $ACTOR_GPU."
echo "Runtime logs: $RUNTIME_DIR"
echo "Evaluation outputs: $RESULT_DIR/$LOG_NAME"
start_epoch=$(date +%s)
(
  cd "$UPSTREAM"
  exec setsid env \
    CUDA_VISIBLE_DEVICES="$ACTOR_GPU" \
    DISPLAY=":$DISPLAY_ID" \
    XDG_CACHE_HOME=/data/yukun/.cache/racer \
    COPPELIASIM_ROOT="$COPPELIASIM_ROOT" \
    LD_LIBRARY_PATH="$COPPELIASIM_ROOT" \
    QT_PLUGIN_PATH="$COPPELIASIM_ROOT" \
    QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT/platforms" \
    QT_QPA_PLATFORM=xcb \
    QT_XCB_GL_INTEGRATION=xcb_glx \
    LIBGL_ALWAYS_SOFTWARE=1 \
    "$ACTOR_PY" -u racer/evaluation/rollout.py \
      --model-folder "$UPSTREAM/racer/runs/racer-visuomotor-policy-rich" \
      --model-name model_17.pth \
      --eval-datafolder "$UPSTREAM/racer/data/rlbench/test" \
      --tasks "$TASKS" \
      --start-episode 0 \
      --eval-episodes 25 \
      --episode-length 30 \
      --retry-for-InvalidActionError 5 \
      --log-name "$LOG_NAME" \
      --eval-log-dir "$RESULT_DIR" \
      --lm-address "http://127.0.0.1:$LM_PORT/encode/" \
      --vlm-address "http://127.0.0.1:$VLM_PORT" \
      --use-vlm
) >"$ACTOR_LOG" 2>&1 &
actor_pid=$!

last_report=0
while kill -0 "$actor_pid" 2>/dev/null; do
  for service in "$lm_pid:language service" "$vlm_pid:LLaVA service" "$xvfb_pid:Xvfb"; do
    pid=${service%%:*}
    name=${service#*:}
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name died during evaluation; stopping actor." >&2
      kill -TERM -- "-$actor_pid" 2>/dev/null || true
      wait "$actor_pid" 2>/dev/null || true
      actor_pid=''
      exit 1
    fi
  done
  now=$(date +%s)
  if (( now - last_report >= 60 )); then
    echo "Evaluator running for $((now - start_epoch)) seconds; log: $ACTOR_LOG"
    last_report=$now
  fi
  sleep 5
done

set +e
wait "$actor_pid"
actor_status=$?
set -e
actor_pid=''
if (( actor_status != 0 )); then
  echo "ERROR: evaluator exited with status $actor_status; tail follows." >&2
  tail -80 "$ACTOR_LOG" >&2 || true
  exit "$actor_status"
fi

METRICS="$RESULT_DIR/$LOG_NAME/metrics.json"
[[ -f "$METRICS" ]] || fail "evaluator exited without metrics: $METRICS"
"$ACTOR_PY" "$SCRIPT_DIR/summarize_three_task.py" "$METRICS" \
  --paper-reference "$RACER_ROOT/paper_reference.json" \
  --output-dir "$RESULT_DIR/$LOG_NAME" \
  >"$RUNTIME_DIR/comparison_summary.json"

echo "RACER three-task evaluation completed in $(($(date +%s) - start_epoch)) seconds."
echo "Comparison: $RESULT_DIR/$LOG_NAME/comparison.md"
