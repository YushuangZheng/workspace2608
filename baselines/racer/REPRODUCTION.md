# RACER released-checkpoint reproduction

## Scope and current status

This reproduction keeps all RACER model and protocol logic at the pinned
official revisions in `manifest.json`. The first formal target is the official
rich-instruction checkpoint on 25 fixed test episodes for each of:

- `place_cups`
- `place_wine_at_rack_location`
- `sweep_to_dustpan_of_size`

That is 75 episodes in total. The full 18-task run is deferred until this
bounded run is sound. No training is needed: the actor, rich LLaVA LoRA, LLaVA
base, T5-11B weights, CLIP weights, and fixed RLBench episodes are local.

Component validation completed successfully:

- all source and Hugging Face revisions match the manifest;
- all model shard headers and the LLaVA weight index are readable;
- the official actor checkpoint constructs and loads;
- the rich LoRA merges into the official LLaVA base on two GPUs, has no meta
  vision parameters, and completes a multimodal generation;
- the unchanged language service loads T5-11B and CLIP, and its real
  `POST /encode/` response is finite with shape `1 x 6 x 1024`;
- the rich LLaVA service passes its real health/generation request;
- all three test archives pass ZIP integrity checks and episode 0 of every task
  loads RGB, depth, and point clouds;
- CoppeliaSim launches, the RACER `RenderMode.OPENGL` observation path resets
  fixed `place_cups` episode 0, renders four 512x512 RGB/point-cloud views, and
  shuts down under the user-space GLX display described below.

These are component smokes, not a completed RACER evaluation. The formal retry2
run `after_spr_task0_restart_20260820_070434_retry2` passed the immutable
preflight, both live service checks, actor construction/checkpoint loading, and
`Agent Reset`. Its first RLBench reset then received SIGSEGV (status 139) in
`libcoppeliaSim.so.1 -> /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so`, before an
episode-0 result was produced. The formal runtime is
`baselines/racer/runtime/after_spr_task0_restart_20260820_070434_retry2/`; the
result directory contains only evaluator configuration/log scaffolding.

The measured result is therefore **0 of 75 episodes completed**, not a 0%
success rate. There is no local RACER success-rate estimate, `metrics.json`, or
valid paper-comparison figure from this attempt.

## Environments

| Role | Prefix | Key validated versions |
|---|---|---|
| Actor, RLBench, simulator | `/data/yukun/miniconda3/envs/dynamac-racer` | Python 3.8.18, PyTorch 1.13.0+cu116, torchvision 0.14.0, PyTorch3D 0.7.5 build `py38_cu116_pyt1130`, PyRep 4.1.0.3, RLBench 1.2.0 |
| LLaVA and language services | `/data/yukun/miniconda3/envs/dynamac-racer-llava` | Python 3.10, PyTorch 2.1.2+cu121, torchvision 0.16.2, transformers 4.37.2, accelerate 0.21.0, PEFT 0.10.0 |

The official actor `setup.py` dependency metadata is not jointly satisfiable:
TensorFlow 2.13.1 asks for `typing-extensions<4.6`, while FastAPI 0.111.0 asks
for `typing-extensions>=4.8` and current transitive packages require newer
versions. TensorFlow 2.13.1 was therefore installed without dependency
resolution, all of its actual runtime dependencies were installed, and
`typing-extensions==4.13.2` was retained. TensorFlow imports successfully; the
single expected `pip check` warning is this metadata conflict. No model code
was changed.

The Open-LLaVA commit leaves PEFT unpinned. PEFT 0.20.0 is incompatible with
its pinned Accelerate 0.21.0, so PEFT 0.10.0 is frozen as the commit-era
runtime-compatible release. Optional training extras and FlashAttention are
not installed because checkpoint evaluation invokes the official builder with
`use_flash_attn=False`.

## Four-GPU orchestration

The frozen defaults in `scripts/run_three_task_eval.sh` are:

| GPU | Process | Port | Measured or documented memory |
|---:|---|---:|---:|
| 1 | T5-11B plus CLIP language service | 18000 | 18.8 GiB measured |
| 2-3 | Rich LLaVA service | 21002 | 8.3 and 9.8 GiB after merge; two GPUs avoid the documented 31.7 GiB merge peak |
| 4 | RACER visuomotor actor and RLBench | none | 15.5 GiB peak reported by the release; checkpoint construction smoke was 141 MiB |

GPU indices can be changed only through `RACER_LM_GPU`,
`RACER_VLM_GPUS` (exactly two comma-separated indices), and
`RACER_ACTOR_GPU`. The launcher rejects duplicate or busy GPUs. It also rejects
occupied ports, missing assets, invalid revisions, missing GLX, failed service
health checks, or a service that exits during evaluation.

The language-service port defaults to 18000 on this workstation because port
8000 is occupied by the resident XPolicy Eval service. Any other available TCP
port can be selected with `RACER_LM_PORT`; the actor receives that same port in
its unchanged `--lm-address` request URL.

Warm startup is approximately 1.5-2 minutes for T5/CLIP and 20-40 seconds for
LLaVA. The official README reports about five hours for 450 episodes. The
75-episode subset should be budgeted at roughly 50-90 minutes; user-space
software rendering and VLM latency can move it outside a simple linear
estimate.

## User-space GLX display

The machine has no system VNC/Xvfb service available to this user. CoppeliaSim
can launch with Qt's offscreen plugin, but a vision-sensor render then fails
because that plugin cannot create the required OpenGL context. The tested conda
Xvfb builds also did not advertise a usable GLX visual.

`scripts/bootstrap_user_xvfb.sh` downloads the Ubuntu-matched `xvfb` package,
verifies its exact SHA-256, and extracts it under `/data/yukun/.cache/racer`
without sudo or system-library changes. The evaluator starts this Xvfb before
adding CoppeliaSim to `LD_LIBRARY_PATH`, then uses:

```bash
DISPLAY=:95
XDG_CACHE_HOME=/data/yukun/.cache/racer
COPPELIASIM_ROOT=/data/yukun/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
LD_LIBRARY_PATH=$COPPELIASIM_ROOT
QT_PLUGIN_PATH=$COPPELIASIM_ROOT
QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT/platforms
QT_QPA_PLATFORM=xcb
QT_XCB_GL_INTEGRATION=xcb_glx
LIBGL_ALWAYS_SOFTWARE=1
```

No `LIBGL_DRIVERS_PATH`, `MESA_LOADER_DRIVER_OVERRIDE`, or `GALLIUM_DRIVER`
override is used. The launcher requires `xdpyinfo` to list GLX and `glxinfo -B`
to succeed before starting CoppeliaSim. Those probes and a simulator-only reset
smoke pass, but they do not make this software-rendering stack safe after the
RACER policy is loaded; the formal failure and diagnostics below supersede the
earlier component-level GLX result.

## Infrastructure-only adapters

These adapters do not change model or evaluation semantics:

1. The official actor data and checkpoint locations are symlinks from
   `upstream/racer/data/rlbench/test` and
   `upstream/racer/runs/racer-visuomotor-policy-rich` to the local artifact
   directories.
2. The official `deploy/lm_server.py` contains the literal placeholder
   `<your-dir-to-store-t5-11b>`. A matching local directory/symlink is created
   instead of editing that source file.
3. `scripts/racer_lm_server.py` imports `app` from the pinned official
   `deploy/lm_server.py` and passes it to Uvicorn with the requested host and
   port. This bypasses only the official `__main__` block's hard-coded port
   8000; model loading, routes, request schema, and inference are unchanged.
4. `scripts/racer_llava_server.py` delegates to the official deployment entry
   point and injects only a two-GPU `max_memory` map. This prevents the LoRA
   merge peak from exceeding one 24-GiB card.
5. Author-machine IP addresses in `scripts/eval_racer.sh` are replaced at the
   command line with local `127.0.0.1` service addresses.
6. The simulator uses a user-space Xvfb/GLX process. RACER's own camera config
   remains its released `RenderMode.OPENGL` implementation.

## Preflight and launch

From `/data/yukun/essay2608`:

```bash
bash baselines/racer/scripts/bootstrap_user_xvfb.sh
/data/yukun/miniconda3/envs/dynamac-racer/bin/python \
  baselines/racer/scripts/verify_setup.py
```

The frozen command remains:

```bash
RACER_LM_PORT=18000 bash baselines/racer/scripts/run_three_task_eval.sh
```

Do not rerun the 75-episode target with the current Xvfb/software-Mesa backend:
retry2 already passed preflight and failed before episode 0. Reuse this command
only after replacing the display backend and passing the acceptance checks in
the diagnostic conclusion below.

To validate the full orchestration and health checks but stop before episode 0:

```bash
RACER_LM_PORT=18000 RACER_HEALTH_ONLY=1 \
  bash baselines/racer/scripts/run_three_task_eval.sh
```

The launcher starts Xvfb, T5/CLIP, and LLaVA itself and removes only those
processes on success, failure, or interruption. The model services are ready
only after these semantic checks pass:

- `POST /encode/` returns a finite nonempty T5 embedding and token length;
- `GET /test` returns the official LLaVA `Hello World` response;
- the fixed simulator episode completes one reset and four-view render during
  the component validation recorded above.

If either model service dies during the actor run, the monitor terminates the
actor process group and the run fails rather than allowing the client's retry
loop to wait indefinitely.

## A-K2 renderer diagnostics and stop decision

The ignored `baselines/racer/runtime/glx_diag_*` runs isolate the failure
without changing pinned RACER, RLBench, PyRep, or CoppeliaSim code:

| Cases | Isolated condition | Result |
|---|---|---|
| A | RLBenchSim reset/render/close without the policy | natural status 0 |
| B-F | policy/actor loaded before the first real render; thread limits and simulator-launch reorder included | first reset still exits 139 in `swrast_dri.so` |
| G, I1 | perform a real task reset/render before importing/loading the policy, then create a second simulator | resets complete; I1 exits 0 but changes 3 of 4 RGB observations substantially |
| H1-H2 | preload `swrast` or isolate local-policy imports | preload reaches reset but exits 134; local-policy isolation still exits 139 |
| K1 | create and destroy a standards-based tiny GLX context before the official evaluator | random and fixed episode-0 resets pass, then `free(): invalid pointer` and exit 134; two X11 library closures are present |
| K2 | repeat K1 with one Conda X11 closure | resets and observation checks pass, then the same post-`atexit` invalid free and exit 134 |

I1 is not a semantics-neutral workaround. Between its first clean simulator and
the second post-policy simulator, low-dimensional state and all four point
clouds are byte-identical, but front/right/wrist RGB means change from about
105.5/151.4/147.9 to 155.1/208.3/215.1. Restoring Python, NumPy, Torch, or task
seeds cannot undo process-global native GL state, so this warmup must not be
inserted into the official rollout.

K1 and K2 are also rejected despite reaching both reset paths: an adapter must
preserve observations **and** let the official process terminate naturally with
status 0. No A-K2 workaround meets reset stability, observation fidelity, and
clean shutdown together. The local Xvfb/Mesa path is therefore closed rather
than used to manufacture a nominal result.

## Queued VirtualGL EGL retry

The user-space retry uses VirtualGL 3.1.5's EGL backend rather than another
Mesa load-order workaround. The server already permits this account to open
`/dev/nvidia*`, and NVIDIA EGL 1.5 exposes the required surfaceless and
no-config-context extensions. Xvfb remains the 2D X server; VirtualGL emulates
the application's GLX path on the NVIDIA EGL device. No NVIDIA Xorg server,
`vglserver_config`, root access, reboot, simulator patch, or policy patch is
used.

The graphics probe is stronger than checking `nvidia-smi`: the launcher runs
`glxinfo -B` through the selected VirtualGL EGL device, records it in the run
directory, requires `OpenGL vendor string: NVIDIA Corporation`, and rejects
`llvmpipe`, `softpipe`, or `swrast` before starting either model service.

`scripts/bootstrap_user_virtualgl.sh` pins the official amd64 Debian artifact
and SHA-256, then extracts it under `/data/yukun/.cache/racer`. The physical
actor GPU is mapped to an explicit VirtualGL `eglN` identifier through
NVIDIA's `EGL_CUDA_DEVICE_NV` attribute, checked against the physical
`nvidia-smi` index; the script never assumes CUDA and EGL numbering are
identical.

`scripts/run_virtualgl_egl_after_baselines.sh` enforces this order:

1. wait until both pre-existing SPR and Guardian tmux sessions terminate;
2. require GPUs 1-4 to be idle for three consecutive checks;
3. run exactly `place_cups` fixed episode 0 through the official evaluator,
   with zero evaluator retries;
4. require natural status 0, `success: true`, one valid metrics record, one
   success marker, four Pillow-verified 256x346 camera GIFs with readable frames
   and nondegenerate pixels in the top 256x256 camera region (excluding the
   lower 90px text overlay), and four raw float32 `(3, 512, 512)` point clouds that
   are finite and nondegenerate, with no native-renderer failure signature;
5. only then run the three tasks x 25 fixed episodes and generate the paper
   comparison.

Runtime state is written atomically to
`runtime/<supervisor-id>/status.tsv`. If EGL device mapping, execution, or
strict validation fails, the supervisor may enter one fallback path, with no
loop or automatic retry:

1. a direct clean process resets fixed `place_cups` episode 0 and records
   stable dtype/shape/byte-hash fingerprints for every numeric observation;
2. a parent that has imported the official RACER rollout/policy stack performs
   the same reset through a clean spawned simulator worker;
3. both captures and the worker must exit naturally with status 0, all values
   must be finite/supported, and both complete snapshots must match exactly;
4. only then may one spawn-isolated episode 0 run, with zero evaluator retries;
5. only `success: true`, natural evaluator and worker status 0, and the strict
   single-episode artifact validator unlock 3 x 25 under that same isolated
   backend.

Any initialization mismatch, unsupported value, timeout, abnormal exit, or
episode-gate failure records a terminal isolation state and leaves the
75-episode target locked. The adapter replaces only the rollout module's
`RLBenchSim` symbol; the official policy, evaluator loop, simulator, RLBench,
PyRep, and CoppeliaSim sources remain unchanged. The known A-K2 Mesa
load-order/context workarounds remain permanently excluded.

The EGL and isolation gates set `retry-for-InvalidActionError=0`, so each gate
has exactly one episode attempt. The subsequent 3 x 25 run sets it to 5. This
matches the released evaluator's default in `racer/evaluation/rollout.py`; the
official `scripts/eval_racer.sh` invokes that evaluator without overriding the
default. Keeping five only for the full evaluation therefore preserves the
released evaluation protocol rather than weakening the one-shot unlock gate.

Point-cloud evidence is captured by a thin simulator subclass at the return of
the first `RLBenchSim.reset`, before `ModelRVTAgent` downsamples or copies the
observation. It writes an ignored `gate_point_clouds.npz` plus JSON containing
the task/episode, shapes, dtypes, axis spans, and SHA-256 values. The validator
independently reloads the raw NPZ with pickle disabled and recomputes all
conditions and hashes. The adapter returns the original observation objects
unchanged, so this is observation-only instrumentation rather than a policy or
simulator change.

The bounded execution and provenance contract is:

- freeze the exact 40-character Git HEAD at supervisor startup, record it in
  `expected_head.txt`/`frozen_contract.tsv`, and require the same clean branch
  after the dependency wait, before the episode gates, before fallback, and
  before 3 x 25;
- cap each direct/isolated initialization capture at 600 seconds and each
  episode-0 evaluator gate at 3600 seconds, with TERM followed by KILL after a
  30-second cleanup window;
- tag every child with its exact run/owner token, then scan `/proc` after each
  capture and evaluation stage; record the result and reject any remaining
  CoppeliaSim, simulator-worker, model-service, evaluator, Xvfb, or other owned
  process;
- record isolated-worker PID, return code, and whether explicit close produced
  a natural status 0.

These limits apply to gates, not the official 3 x 25 runtime. The full run
still fails if its launcher cleanup or supervisor audit finds an owned
residual process.

Key provenance:

- formal actor log:
  `baselines/racer/runtime/after_spr_task0_restart_20260820_070434_retry2/actor_eval.log`
  (SHA-256 `c4b19c453debe4b257363d231a1ad40dda656ec53a2c7078fec8564fb94c63e3`);
- I1: `baselines/racer/runtime/glx_diag_I1_two_lifecycles_20260820_213700/`;
- K1: `baselines/racer/runtime/glx_diag_K1_supported_glx_context_20260820_214306/`;
- K2: `baselines/racer/runtime/glx_diag_K2_conda_x11_context_20260820_214708/`.

## Expected outputs and paper-comparison boundary

Runtime logs go under ignored `baselines/racer/runtime/<run-id>/`. Episode
statistics, camera GIFs, and `metrics.json` go under ignored
`baselines/racer/results/<run-id>/official_ckpt_three_task/`.

No episode statistics, `metrics.json`, summary tables, or comparison figure
exist for retry2. On a future successful completion, the launcher would call
`scripts/summarize_three_task.py`,
which creates:

- `comparison.csv`
- `comparison.json`
- `comparison.md`
- `paper_vs_reproduction.png`

The paper's Table I reports five-seed mean and standard deviation: Place Cups
`6.4 +/- 4.1`, Place Wine `98.4 +/- 2.0`, and Sweep to Dustpan `84.0 +/- 0.0`
percent. The local release provides one actor checkpoint, so the local
25-episode rates are a single-checkpoint comparison, not a reconstruction of
the five-seed error bars. Local task bars use a binomial Wilson 95% confidence
interval over 25 episodes; paper bars retain the reported five-seed standard
deviation. The comparison files are valid only after all expected episode
records exist; their presence in the implementation must not be presented as a
paper-result reproduction.
