# RACER native-system reproduction

This server-B directory contains only RACER project adaptation and reproduction
records.  Shared interfaces, task/fault configuration, manifests, audit and
recovery logic, runners, analysis, and controlled results remain server-A
owned and are not changed here.

## Validated on server B

- Upstream RACER commit: `df8cb2beec2e2061392ef0c4be93bda916dfd51e`.
- Released rich visuomotor checkpoint: `model_17.pth`, SHA-256
  `73687bb41342b724d6fff8bb8776a0419155aaa6113e455907786e50c69b33f2`.
- The checkpoint loads and completes a CUDA forward through RACER's
  PyTorch3D renderer on an NVIDIA RTX 6000D.
- The pinned RACER/RLBench fork launches and resets the official `close_jar`
  task under CoppeliaSim.
- The released LLaVA rich LoRA adapter and non-LoRA weights are downloaded and
  checksum-validated outside the project repository.

RACER's published Python 3.8 / PyTorch 1.13 / CUDA 11.6 stack cannot execute
on compute capability 12.0.  The compatibility environment uses Python 3.9.23,
PyTorch 2.7.1+cu128, and PyTorch3D 0.7.9 rebuilt from pinned source for sm_120.
TensorFlow 2.15.1 replaces the upstream 2.13.1 metadata pin because 2.13.1's
typing-extensions constraint conflicts with PyTorch 2.7.  This deliberate
`pip check` discrepancy is recorded rather than concealed.

## Recreate

Create the base environment:

```bash
conda env create -f \
  evaluations/iclr2027/native_systems/racer/reproduction/environment.gpu.yaml
```

Install the pinned local RACER, PyRep, RLBench, YARR, PerAct-colab and CLIP
checkouts, then rebuild the pinned PyTorch3D checkout with
`FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0`.  Exact commits and weight checksums
are recorded in `OFFICIAL_SOURCES.json`.

```bash
RACER_ROOT=/home/ubuntu/workspace/_external/RACER
CLIP_ROOT=/home/ubuntu/workspace/_external/CLIP
PYTORCH3D_ROOT=/home/ubuntu/workspace/_external/pytorch3d
RACER_PY=/home/ubuntu/miniforge3/envs/racer-official/bin/python

"$RACER_PY" -m pip install --no-build-isolation --no-deps \
  -e "$CLIP_ROOT" \
  -e "$RACER_ROOT/libs/PyRep" \
  -e "$RACER_ROOT/libs/RLbench" \
  -e "$RACER_ROOT/libs/YARR" \
  -e "$RACER_ROOT/libs/peract_colab" \
  -e "$RACER_ROOT"

export CUDA_HOME=/home/ubuntu/miniforge3/envs/racer-official
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=12.0
"$RACER_PY" -m pip install --no-build-isolation --no-deps \
  -e "$PYTORCH3D_ROOT"
```

Run the checkpoint and simulator smoke from the project root:

```bash
export COPPELIASIM_ROOT=/home/ubuntu/workspace/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=xcb_glx
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

xvfb-run -a -s '-screen 0 1280x1024x24 +extension GLX +render' \
  conda run -n racer-official python -m \
  evaluations.iclr2027.native_systems.racer.reproduction.official_smoke \
  --task-reset --output \
  evaluations/iclr2027/native_systems/racer/reproduction/official_smoke_result.json
```

The synthetic T5-shaped embedding validates the released visuomotor policy
without claiming semantic language-service correctness.  End-to-end RACER
service and E6 rollouts additionally require the T5-11B/LLaVA services and,
for formal paper evidence, server A's frozen Native-6 manifest and shared
monitor/recovery contract.
