# RVT native-system reproduction

This server-B directory records the pinned upstream project, the compatibility
environment, and nominal reproduction evidence for RVT.  It does not modify
server A's shared interfaces, task/fault definitions, manifests, audit logic,
recovery code, runners, analysis, or controlled results.

## Validated on server B

- Upstream RVT commit: `367995a1a2169b6352bf4e8b0ed405890462a3a0`.
- Released RVT-2 checkpoint: `model_99.pth`, SHA-256
  `25b4cdeae72ac98ecf7eac79604d6553f50d4f8bef6cb2610631875b42cf437f`.
- The released checkpoint loads and completes a CUDA forward through RVT-2's
  custom point renderer on an NVIDIA RTX 6000D.
- RLBench's official `close_jar` task launches in the pinned simulator stack,
  resets successfully, and returns RGB, point-cloud, and low-dimensional
  observations.

The upstream Python 3.8 / PyTorch 1.12.1 stack predates compute capability
12.0.  The validated compatibility environment therefore uses Python 3.9.23,
PyTorch 2.7.1+cu128, and a CUDA 12.8 build of the pinned custom point renderer.
The upstream exact `bitsandbytes==0.38.1` pin is replaced by 0.45.5 because the
old binary does not support this GPU generation.  `pip check` consequently
reports that one deliberate metadata mismatch; it is not hidden.

## Recreate

Create the base environment:

```bash
conda env create -f \
  evaluations/iclr2027/native_systems/rvt/reproduction/environment.gpu.yaml
```

Install the pinned local checkouts and build the point renderer after setting
`COPPELIASIM_ROOT`.  The exact repository/submodule commits are in
`OFFICIAL_SOURCES.json`; external sources and weights stay outside this repo.

```bash
RVT_ROOT=/home/ubuntu/workspace/_external/RVT
CLIP_ROOT=/home/ubuntu/workspace/_external/CLIP
RVT_PY=/home/ubuntu/miniforge3/envs/rvt-official/bin/python

"$RVT_PY" -m pip install --no-build-isolation --no-deps \
  -e "$CLIP_ROOT" \
  -e "$RVT_ROOT/rvt/libs/PyRep" \
  -e "$RVT_ROOT/rvt/libs/RLBench" \
  -e "$RVT_ROOT/rvt/libs/YARR" \
  -e "$RVT_ROOT/rvt/libs/peract_colab" \
  -e "$RVT_ROOT"

export CUDA_HOME=/home/ubuntu/miniforge3/envs/rvt-official
export TORCH_CUDA_ARCH_LIST=12.0
"$RVT_PY" -m pip install --no-build-isolation --no-deps \
  -e "$RVT_ROOT/rvt/libs/point-renderer"
```

Run the complete checkpoint and task-reset smoke from the project root:

```bash
export COPPELIASIM_ROOT=/home/ubuntu/workspace/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=xcb_glx
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

xvfb-run -a -s '-screen 0 1280x1024x24 +extension GLX +render' \
  conda run -n rvt-official python -m \
  evaluations.iclr2027.native_systems.rvt.reproduction.official_smoke \
  --task-reset --output \
  evaluations/iclr2027/native_systems/rvt/reproduction/official_smoke_result.json
```

The synthetic tensor forward validates the released checkpoint and its GPU
operators; it is not an RLBench success-rate measurement.  Formal Native-6/E6
evaluation begins only after server A supplies the frozen Native-6 manifest,
task mapping, observation schema, and controlled import contract.
