# M3 FAIL-Detect adapter and reproduction

This method-owned directory contains server B's M3 adapter, preprocessing,
inference code, dependency records, and official reproduction evidence. It
does not define or modify A-owned interfaces, tasks, faults, recovery, runners,
manifests, calibration data, sealed tests, or controlled results.

## Current boundary

- The official source is `CXU-TRI/FAIL-Detect` at commit
  `b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`; the checkout and its conda
  environment live outside this repository.
- The exact upstream conda file is currently unsatisfiable because its fixed
  `av=10.0.0` dependency has no FFmpeg build compatible with the remaining
  pins in current channel metadata. The reproducible
  `faildetect-logpzo-b758e55f` specification records Python 3.9, PyTorch 1.12.1,
  CUDA Toolkit 11.6, NumPy 1.23.3, SciPy 1.9.1, and Einops 0.4.1 for an
  algorithm-level smoke.  It runs on CPU because that PyTorch build predates
  server B's compute capability 12.0 GPU. This old CPU-only environment was
  removed after validation to keep server B's environment list lean; it can be
  rebuilt from `reproduction/environment.official_cpu.yaml`. MKL is pinned to
  2023.2.0 because a current solve installs an incompatible MKL 2026 build.
- The Square Proficient-Human image dataset was downloaded from Robomimic's
  authoritative v0.1 URL and converted with the pinned repository's official
  conversion script.  The validated `image_abs.hdf5` contains 200 demos and
  30,154 transitions; its provenance and checksums are in
  `reproduction/square_dataset.json`.
- The pinned robosuite 1.2.0 setup declares `numba<=0.53.1`, which conflicts
  with the upstream FAIL-Detect conda file's `numba=0.56.4`.  The working
  conversion environment records the dependency-compatible 0.53.1 choice
  instead of hiding the discrepancy.  Building its legacy MuJoCo extension
  on this host also requires GCC/G++ 12 and the OSMesa/GLFW development
  libraries listed in `OFFICIAL_SOURCES.json`.
- The public source ships no policy dataset, policy checkpoint, extracted
  feature tensor, or logpZO checkpoint.  The public Square dataset and one
  official policy optimizer step are now reproduced, but a result-level run
  still requires full policy training, feature export, and logpZO training.
- M3 is observation-only, matching the paper's logpZO definition.  An
  A-provided frozen feature encoder will later map the shared causal schema to
  the vector consumed here.  Audit labels, fault metadata, and future frames
  are not accepted as monitor inputs.
- Server B can reproduce the official calibration algorithm on public or
  development data, but formal thresholds and persistence settings are built
  only on server A's private normal-calibration split.

## Public-source discrepancies retained for audit

The RSS/arXiv v3 appendix describes 500 logpZO epochs in simulation and
`alpha=0.05` throughout.  The pinned public code currently trains the score
network for 200 epochs and its plotting entrypoint chooses task/policy-specific
alpha values.  Neither value is silently selected by this adapter: training
configuration and `alpha` are explicit artifacts.

The pinned `train.py` also declares `diffusion_policy/config` as Hydra's
primary config directory even though only `diffusion_policy/configs_robomimic`
exists; the README's `--config-dir` command does not bypass that missing
primary path with Hydra 1.2.  The entrypoint smoke used a temporary symlink and
removed it afterward, leaving the official checkout clean.  Its workspace
also imports pandas although pandas is absent from the upstream environment.

`TimeVaryingConformalBand.fit` independently implements the public code's
upper one-sided T-function band.  `TorchLogpZOScorer` implements both the
public code's `adjust_xshape` padding/reshape and the paper's one-step score
`||O + f(O, 0)||^2`; it receives the trained velocity model and explicit task
action dimension from the isolated method environment.

Recreate the optional parity environment and run the pinned synthetic-data
smoke from the repository root with:

```bash
conda env create -f \
  evaluations/iclr2027/methods/fail_detect/reproduction/environment.official_cpu.yaml

conda run -n faildetect-logpzo-b758e55f \
  python -m evaluations.iclr2027.methods.fail_detect.reproduction.official_smoke
```

On server B, the old PyTorch package also needs its obsolete executable-stack
flag cleared once after environment creation:

```bash
conda run -n faildetect-logpzo-b758e55f bash -c \
  'patchelf --clear-execstack "$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so"'
```

Passing this smoke is not an official-task result.  The latter remains blocked
until a policy checkpoint, exported features, and trained logpZO checkpoint
are reproduced or obtained from their authoritative source.

## Server-B GPU compatibility environment

Server B actually exposes eight NVIDIA RTX 6000D devices with compute
capability 12.0.  The upstream PyTorch 1.12.1 / PyTorch3D 0.7.0 combination
cannot execute on them, so `faildetect-gpu-b758e55f` uses the earliest tested
PyTorch CUDA 12.8 build here (`2.7.1+cu128`) and PyTorch3D v0.7.9 built from
official source. This compatibility environment is independent from the
optional CPU environment used for old-version algorithm parity.

Recreate it after the external sources and CUDA wheel cache are present:

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MUJOCO_GL=osmesa \
conda env create -f \
  evaluations/iclr2027/methods/fail_detect/reproduction/environment.gpu.yaml
```

Run the GPU algorithm check and the real-data Square optimizer-step check:

```bash
conda run -n faildetect-gpu-b758e55f \
  python -m evaluations.iclr2027.methods.fail_detect.reproduction.official_smoke \
  --device cuda:0

CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MUJOCO_GL=osmesa \
conda run -n faildetect-gpu-b758e55f \
  python -m evaluations.iclr2027.methods.fail_detect.reproduction.square_policy_smoke \
  --device cuda:0
```

Both pass on server B.  Their machine-readable outputs are stored beside the
scripts.  The real-data check uses the official 278,006,344-parameter Square
flow-policy configuration, two causal image observations, and one optimizer
step; it does not create a checkpoint or claim a converged policy.

The official training entrypoint was additionally exercised with seed 1103,
one train batch, one validation batch, and one eight-step MuJoCo rollout.  All
EGL probes failed on this host, so that smoke intentionally used OSMesa.  The
run completed offline without a checkpoint; exact overrides and logs are in
`/home/ubuntu/workspace/_runs/fail_detect/square_flow_seed1103_smoke_20260904`,
and the concise result is in
`reproduction/square_entrypoint_smoke_result.json`.  PyAV 12.3.0
is the recorded compatibility substitute for the unsatisfiable upstream
PyAV 10 / FFmpeg combination.

## Complete Square reproduction pipeline

The pinned policy workspace is single-process.  The method-owned distributed
launcher preserves its model, flow-matching loss, AdamW optimizer, EMA,
train/validation split, FP32 precision, scheduler horizon, and official global
batch size of 64, while dividing each batch over eight GPUs.  Simulator
rollouts are moved after training so seven ranks are not idle during every
evaluation interval.  Checkpoints retain the upstream workspace format and
are written atomically every 50 epochs.

```bash
cd /home/ubuntu/workspace/essay2608
RUN=/home/ubuntu/workspace/_runs/fail_detect/square_flow_seed1103_full

conda run -n faildetect-gpu-b758e55f \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m evaluations.iclr2027.methods.fail_detect.reproduction.distributed_policy_train \
  --output-dir "$RUN" --epochs 800 --seed 1103 --global-batch-size 64
```

`complete_square_reproduction` is a resume-safe continuation driver.  It
waits for the 800-epoch checkpoint, exports the official embedded observation
features and action trajectories, trains logpZO for the public code's 200
epochs with global batch size 128 across eight GPUs, runs 2,000 official
Square ID and 2,000 modified-environment rollouts in eight disjoint shards,
then generates the time-varying functional CP band and per-episode alarms.

```bash
conda run -n faildetect-gpu-b758e55f \
  python -m evaluations.iclr2027.methods.fail_detect.reproduction.complete_square_reproduction \
  --run-dir "$RUN"
```

The first atomic checkpoint is independently audited on CPU by
`validate_policy_checkpoint`.  The validator reconstructs the pinned upstream
workspace from the serialized config and loads the model, EMA model,
optimizer, scheduler, epoch and global step through the same payload path used
for resume.  It also checks floating tensors and records the checkpoint hash,
so a running multi-GPU job need not be interrupted merely to prove recovery.

The targeted rollout driver disables unrelated baselines and unconditional
PNG diagnostics in the public all-baselines runner.  It retains the pinned
policy, simulator, modification operation, success condition, and logpZO
implementation.  The resulting official-task artifact validates the public
score/calibration/alarm chain only; it is not server A's formal Main-10
calibration artifact or a paper result.
