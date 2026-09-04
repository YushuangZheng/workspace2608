# ICLR 2027 runtime monitors

This directory contains the benchmark-neutral online-monitor contract and the
server-B implementation of M3 (FAIL-Detect).  It does not reuse or modify the
frozen stage-six evaluator.

## Current boundary

- The official source is `CXU-TRI/FAIL-Detect` at commit
  `b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`; the checkout and its conda
  environment live outside this repository.
- The exact upstream conda file is currently unsatisfiable because its fixed
  `av=10.0.0` dependency has no FFmpeg build compatible with the remaining
  pins in current channel metadata.  The isolated
  `faildetect-logpzo-b758e55f` environment retains Python 3.9, PyTorch 1.12.1,
  CUDA Toolkit 11.6, NumPy 1.23.3, SciPy 1.9.1, and Einops 0.4.1 for an
  algorithm-level smoke.  It runs on CPU because that PyTorch build predates
  server B's compute capability 12.0 GPU.  MKL is pinned to 2023.2.0 because
  a current unconstrained solve installs an ABI-incompatible MKL 2026 build.
- The Square Proficient-Human image dataset was downloaded from Robomimic's
  authoritative v0.1 URL and converted with the pinned repository's official
  conversion script.  The validated `image_abs.hdf5` contains 200 demos and
  30,154 transitions; its provenance and checksums are in
  `reproduction/fail_detect_square_dataset.json`.
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

Run the pinned synthetic-data parity smoke from the repository root with:

```bash
conda run -n faildetect-logpzo-b758e55f \
  python -m evaluations.iclr2027.reproduction.fail_detect_official_smoke
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
official source.  This compatibility environment is separate from the CPU
environment retained for old-version algorithm parity.

Recreate it after the external sources and CUDA wheel cache are present:

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MUJOCO_GL=osmesa \
conda env create -f \
  evaluations/iclr2027/reproduction/environment.fail_detect_gpu.yaml
```

Run the GPU algorithm check and the real-data Square optimizer-step check:

```bash
conda run -n faildetect-gpu-b758e55f \
  python -m evaluations.iclr2027.reproduction.fail_detect_official_smoke \
  --device cuda:0

CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MUJOCO_GL=osmesa \
conda run -n faildetect-gpu-b758e55f \
  python -m evaluations.iclr2027.reproduction.fail_detect_square_policy_smoke \
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
`reproduction/fail_detect_square_entrypoint_smoke_result.json`.  PyAV 12.3.0
is the recorded compatibility substitute for the unsatisfiable upstream
PyAV 10 / FFmpeg combination.

## M4 supervised monitor boundary

`FailureSupervisedMonitor` uses the same frozen one-dimensional causal feature
vector as the FAIL-Detect adapter.  `training/causal_gru.py` provides a small
unidirectional GRU and a padding-aware cycle loss; the runtime wrapper retains
only past hidden state and emits the current violation probability.  Fault
family, severity, trigger time, audit metadata, and future labels are not model
inputs.

The network is ready for synthetic/API checks, but no formal checkpoint is
trained until server A freezes and transfers the failure-train bundle.  Server
B will train weights from that bundle; server A alone will derive the formal
normal-calibration threshold and persistence rule.
