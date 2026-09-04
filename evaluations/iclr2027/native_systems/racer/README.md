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
- The official CLIP RN50 language encoder produces finite `[1, 77, 512]`
  token features in the isolated service environment.
- The official T5-11B FP32 checkpoint loads its 4,864,791,552-parameter encoder
  on one GPU and produces finite `[1, 6, 1024]` token features.  The measured
  peak is 18,569.445 MiB and the load-plus-forward time is 361.673 seconds.
- The released LLaVA rich adapter, Llama-3 LLaVA base, and CLIP vision tower
  complete a single-GPU FP16 image-to-text generation.  The measured peak is
  18,075.587 MiB for 8,354,760,704 parameters.

RACER's published Python 3.8 / PyTorch 1.13 / CUDA 11.6 stack cannot execute
on compute capability 12.0.  The compatibility environment uses Python 3.9.23,
PyTorch 2.7.1+cu128, and PyTorch3D 0.7.9 rebuilt from pinned source for sm_120.
TensorFlow 2.15.1 replaces the upstream 2.13.1 metadata pin because 2.13.1's
typing-extensions constraint conflicts with PyTorch 2.7.  This deliberate
`pip check` discrepancy is recorded rather than concealed.

The separate `racer-services` environment preserves the official
Transformers 4.37.2/tokenizers 0.15.1 service stack.  PyTorch 2.1.2 is replaced
by 2.7.1+cu128 because the released binary cannot execute on compute capability
12.0.  FlashAttention is not used: the published server leaves its optional
`use_flash_attn` path disabled.

PyTorch 2.7 emits legacy `low_cpu_mem_usage` meta-parameter copy warnings while
the old loader reconstructs the vision tower.  The recorded smoke proceeds
through vision preprocessing and text generation, so the warning is retained
for audit rather than treated as proof of success by itself.

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

Create the isolated language/VLM service environment and install the pinned
editable projects:

```bash
conda env create -f \
  evaluations/iclr2027/native_systems/racer/reproduction/environment.services.gpu.yaml

SERVICE_PY=/home/ubuntu/miniforge3/envs/racer-services/bin/python
"$SERVICE_PY" -m pip install --no-build-isolation --no-deps \
  -e /home/ubuntu/workspace/_external/CLIP \
  -e /home/ubuntu/workspace/_external/Open-LLaVA-NeXT
```

Each service is measured in a fresh process so peak allocation is attributable
to that component.  Restrict the process to one physical GPU when collecting a
single-GPU baseline:

```bash
CUDA_VISIBLE_DEVICES=1 "$SERVICE_PY" -m \
  evaluations.iclr2027.native_systems.racer.reproduction.service_smoke clip \
  --output evaluations/iclr2027/native_systems/racer/reproduction/service_smoke_clip_result.json

CUDA_VISIBLE_DEVICES=2 "$SERVICE_PY" -m \
  evaluations.iclr2027.native_systems.racer.reproduction.service_smoke t5 \
  --output evaluations/iclr2027/native_systems/racer/reproduction/service_smoke_t5_result.json

CUDA_VISIBLE_DEVICES=3 "$SERVICE_PY" -m \
  evaluations.iclr2027.native_systems.racer.reproduction.service_smoke llava \
  --output evaluations/iclr2027/native_systems/racer/reproduction/service_smoke_llava_result.json
```

## Adapter requirements for Native-6

- Preserve the four official cameras (`front`, `left_shoulder`,
  `right_shoulder`, `wrist`), RGB, point clouds, camera intrinsics and the
  policy low-dimensional state.  RACER observes 512-pixel images and applies
  its own checkpoint-defined downsampling.
- Preserve the native 9-value action: XYZ waypoint, XYZW quaternion, discrete
  gripper command and ignore-collision flag.  Execution uses
  `EndEffectorPoseViaPlanning` followed by the discrete gripper action.
- The language endpoint is `POST /encode/` with `text` and `model`; it returns
  token embeddings plus `token_len`.  The VLM endpoint is
  `POST /worker_generate_stream` and returns null-delimited generated text.
- Server A must provide the frozen Native-6 task/variation mapping, physical
  fault boundary and success/audit contract.  The adapter may observe public
  RGB/point-cloud/action traffic but must not read simulator-private task
  state.  These requirements are recorded here without changing A's interface.

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

The synthetic T5-shaped embedding in the policy-only smoke isolates and
validates the released visuomotor checkpoint.  Semantic service correctness is
covered separately by the actual CLIP, T5-11B, and LLaVA GPU smokes above.
End-to-end E6 rollouts and formal paper evidence still require server A's
frozen Native-6 manifest and shared monitor/recovery contract.

## Seeded live nominal gates

`reproduction/live_nominal.py` runs the official checkpoint and RACER RLBench
environment from deterministic released RNG states.  It supports both the
task-goal-only path and the complete rich path in which LLaVA emits the current
instruction and T5-11B encodes the policy input.  The local language server
retains the released `/encode/` response contract while making the external
checkpoint path configurable and caching identical text requests.

The resume-safe supervisor in
`../complete_live_reproduction.py` waits for the 8-GPU FAIL-Detect job to
release the machine, then runs RVT, starts T5 while RVT is executing, runs the
RACER task-goal gate while LLaVA loads, and finally runs the rich RACER gate.
Large logs, videos and model artifacts remain under `_runs`, outside Git.

```bash
python -m evaluations.iclr2027.native_systems.complete_live_reproduction \
  --eval-episodes 25
```

These live gates prove real perception-action execution and service wiring on
server B.  They are explicitly not labelled as formal E6 evidence until A
delivers the frozen Native-6 manifest and shared auditor contract.
