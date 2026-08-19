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

Validated before the formal run:

- all source and Hugging Face revisions match the manifest;
- all model shard headers and the LLaVA weight index are readable;
- the official actor checkpoint constructs and loads;
- the rich LoRA merges into the official LLaVA base on two GPUs, has no meta
  vision parameters, and completes a multimodal generation;
- the unchanged language service loads T5-11B and CLIP and encodes text;
- all three test archives pass ZIP integrity checks and episode 0 of every task
  loads RGB, depth, and point clouds;
- CoppeliaSim launches, the RACER `RenderMode.OPENGL` observation path resets
  fixed `place_cups` episode 0, renders four 512x512 RGB/point-cloud views, and
  shuts down under the user-space GLX display described below.

The formal evaluator has deliberately not been started while its four GPU
slots are occupied by other work.

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
| 1 | T5-11B plus CLIP language service | 8000 | 18.8 GiB measured |
| 2-3 | Rich LLaVA service | 21002 | 8.3 and 9.8 GiB after merge; two GPUs avoid the documented 31.7 GiB merge peak |
| 4 | RACER visuomotor actor and RLBench | none | 15.5 GiB peak reported by the release; checkpoint construction smoke was 141 MiB |

GPU indices can be changed only through `RACER_LM_GPU`,
`RACER_VLM_GPUS` (exactly two comma-separated indices), and
`RACER_ACTOR_GPU`. The launcher rejects duplicate or busy GPUs. It also rejects
occupied ports, missing assets, invalid revisions, missing GLX, failed service
health checks, or a service that exits during evaluation.

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
to succeed before starting CoppeliaSim.

## Infrastructure-only adapters

These adapters do not change model or evaluation semantics:

1. The official actor data and checkpoint locations are symlinks from
   `upstream/racer/data/rlbench/test` and
   `upstream/racer/runs/racer-visuomotor-policy-rich` to the local artifact
   directories.
2. The official `deploy/lm_server.py` contains the literal placeholder
   `<your-dir-to-store-t5-11b>`. A matching local directory/symlink is created
   instead of editing that source file.
3. `scripts/racer_llava_server.py` delegates to the official deployment entry
   point and injects only a two-GPU `max_memory` map. This prevents the LoRA
   merge peak from exceeding one 24-GiB card.
4. Author-machine IP addresses in `scripts/eval_racer.sh` are replaced at the
   command line with local `127.0.0.1` service addresses.
5. The simulator uses a user-space Xvfb/GLX process. RACER's own camera config
   remains its released `RenderMode.OPENGL` implementation.

## Preflight and launch

From `/data/yukun/essay2608`:

```bash
bash baselines/racer/scripts/bootstrap_user_xvfb.sh
/data/yukun/miniconda3/envs/dynamac-racer/bin/python \
  baselines/racer/scripts/verify_setup.py
```

After confirming GPUs 1-4 are idle, start the frozen run:

```bash
bash baselines/racer/scripts/run_three_task_eval.sh
```

To validate the full orchestration and health checks but stop before episode 0:

```bash
RACER_HEALTH_ONLY=1 bash baselines/racer/scripts/run_three_task_eval.sh
```

The launcher starts Xvfb, T5/CLIP, and LLaVA itself and removes only those
processes on success, failure, or interruption. The model services are ready
only after these semantic checks pass:

- `POST /encode/` returns a finite nonempty T5 embedding and token length;
- `GET /test` returns the official LLaVA `Hello World` response;
- the fixed simulator episode completes one reset and vision render during the
  preflight validation already recorded above.

If either model service dies during the actor run, the monitor terminates the
actor process group and the run fails rather than allowing the client's retry
loop to wait indefinitely.

## Outputs and paper comparison

Runtime logs go under ignored `baselines/racer/runtime/<run-id>/`. Episode
statistics, camera GIFs, and `metrics.json` go under ignored
`baselines/racer/results/<run-id>/official_ckpt_three_task/`.

On successful completion the launcher calls `scripts/summarize_three_task.py`,
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
deviation. CSV, JSON, Markdown, and the figure label these two uncertainty
definitions separately.
