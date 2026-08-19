#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
baseline_dir="$(cd "$script_dir/.." && pwd)"
project_dir="$(cd "$baseline_dir/../.." && pwd)"
aha_dir="$baseline_dir/upstream/aha"
failgen_dir="$aha_dir/Data_Generation/rlbench-failgen"
rouge_script="$aha_dir/evaluation/eval_metrics/check_answer_ROGUE.py"
runtime_dir="$baseline_dir/runtime"
cop_root="${COPPELIASIM_ROOT:-$project_dir/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"

case "${1-}" in
  ""|--sim-reset|--episode) ;;
  *)
    echo "Usage: $0 [--sim-reset|--episode]" >&2
    exit 2
    ;;
esac

if [[ ! -d "$failgen_dir/failgen" ]]; then
  echo "AHA checkout not found: $aha_dir" >&2
  exit 2
fi
if [[ ! -d "$cop_root" ]]; then
  echo "CoppeliaSim 4.1 directory not found: $cop_root" >&2
  exit 2
fi

python -m compileall -q "$aha_dir"

LD_LIBRARY_PATH="$cop_root${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python - <<'PY'
import numpy
import pyrep
import rlbench
import failgen

print(f"Imported NumPy {numpy.__version__}, PyRep, RLBench, and FailGen.")
PY

LD_LIBRARY_PATH="$cop_root${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  python "$failgen_dir/examples/ex_failgen_data_collection.py" --help \
  >/dev/null
python "$rouge_script" --help >/dev/null

python - "$rouge_script" <<'PY'
import importlib.util
from pathlib import Path
import sys

path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("aha_rouge", path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
score = module.compute_rouge("failure detected", "failure detected")
value = score["rougeL"].fmeasure
if value != 1.0:
    raise RuntimeError(f"Expected exact-match ROUGE-L=1.0, got {value}")
print("Released ROUGE-L exact-match smoke: 1.0")
PY

if [[ -z "${1-}" ]]; then
  echo "Static AHA release checks passed. Simulator not launched by default."
  exit 0
fi

cache_dir="${AHA_CACHE_DIR:-/data/yukun/.cache/dynamac-baselines/aha}"
xvfb_bin="${XVFB_BIN:-$cache_dir/xvfb-ubuntu-root/usr/bin/Xvfb}"
display="${AHA_DISPLAY:-:93}"

if [[ ! -x "$xvfb_bin" ]]; then
  echo "Ubuntu Xvfb binary not found: $xvfb_bin" >&2
  exit 2
fi
if [[ ! "$display" =~ ^:[0-9]+$ ]]; then
  echo "AHA_DISPLAY must look like :93, got: $display" >&2
  exit 2
fi
if DISPLAY="$display" xdpyinfo >/dev/null 2>&1; then
  echo "Display already in use: $display; choose another AHA_DISPLAY." >&2
  exit 2
fi

mkdir -p "$runtime_dir" "$cache_dir"
display_number="${display#:}"
xvfb_log="$runtime_dir/xvfb-${display_number}.log"

# Launch Xvfb without a caller-supplied GL library path.  The simulator client
# receives the CoppeliaSim libraries only after the X server is ready.
env -u LD_LIBRARY_PATH "$xvfb_bin" "$display" \
  -screen 0 1280x1024x24 -ac +extension GLX +iglx +render -noreset \
  >"$xvfb_log" 2>&1 &
xvfb_pid=$!
cleanup() {
  kill "$xvfb_pid" >/dev/null 2>&1 || true
  wait "$xvfb_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

display_ready=0
for _ in {1..50}; do
  if DISPLAY="$display" xdpyinfo >/dev/null 2>&1; then
    display_ready=1
    break
  fi
  if ! kill -0 "$xvfb_pid" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
if ((display_ready == 0)); then
  echo "Xvfb did not become ready; see $xvfb_log" >&2
  exit 1
fi

export DISPLAY="$display"
export XDG_CACHE_HOME="$cache_dir"
export COPPELIASIM_ROOT="$cop_root"
export LD_LIBRARY_PATH="$cop_root"
export QT_PLUGIN_PATH="$cop_root"
export QT_QPA_PLATFORM_PLUGIN_PATH="$cop_root/platforms"
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=xcb_glx
export LIBGL_ALWAYS_SOFTWARE=1
export CUDA_VISIBLE_DEVICES=""
unset LIBGL_DRIVERS_PATH MESA_LOADER_DRIVER_OVERRIDE GALLIUM_DRIVER

if [[ "$1" == "--sim-reset" ]]; then
  export AHA_RESET_OUTPUT="$runtime_dir/sim_reset"
  python - <<'PY'
import os

from failgen.env_wrapper import FailGenEnvWrapper

wrapper = FailGenEnvWrapper(
    task_name="basketball_in_hoop",
    headless=True,
    record=False,
    save_data=False,
    save_path=os.environ["AHA_RESET_OUTPUT"],
    save_keyframes_only=True,
)
try:
    if str(wrapper.config.data.renderer) != "opengl3":
        raise RuntimeError(
            f"Expected official renderer opengl3, got "
            f"{wrapper.config.data.renderer!r}"
        )
    wrapper.reset()
    print("Official opengl3 FailGen environment reset passed.")
finally:
    wrapper.shutdown()
PY
  exit 0
fi

python "$script_dir/failgen_one_episode.py" \
  --output "$runtime_dir/failgen_smoke" \
  --max-tries 1
