# Server-B external method adapter requirements

This checklist records the three external-system boundaries without copying or
modifying server A's task, fault, manifest, auditor, recovery, runner, or
formal-result definitions.

## M3: FAIL-Detect

- Inject an A-owned encoder that maps only the frozen causal observation
  schema to one finite one-dimensional feature vector of checkpoint-fixed
  width.
- Do not consume fault family, severity, trigger time, auditor labels, target
  pose, repair direction, future observations, or sealed-result metadata.
- Preserve the shared `reset`, `observe`, `score`, and `alarm` monitor calls.
- Emit the score index, `logpzo`, A-frozen time-varying threshold, margin,
  persistence count, alarm state, and first alarm index.
- Load the official velocity-model checkpoint separately from A's formal
  conformal artifact.  The public-task CP result is reproduction evidence only.

## RVT

- Preserve `front`, `left_shoulder`, `right_shoulder`, and `wrist` RGB and
  point clouds at the checkpoint's native 128-pixel resolution, plus camera
  intrinsics/extrinsics and the checkpoint-defined low-dimensional state.
- Preserve the official language-goal token stream and do not add monitor or
  auditor features to the policy input.
- Execute the native nine-value action: XYZ waypoint, XYZW quaternion,
  discrete gripper, and ignore-collision flag, using the pinned planning action
  mode.
- Accept A's frozen Native-6 task/variation and initialization mapping at the
  environment boundary; emit raw per-episode actions, success, length, timing,
  and identity metadata for A-side audit.

## RACER

- Preserve the same four cameras at RACER's native 512-pixel observation size;
  let the released policy perform its checkpoint-defined downsampling.
- Keep task goal, optional LLaVA current instruction, and T5 token embeddings
  distinct in logs.  Do not silently replace or quantize any backbone.
- Preserve `POST /encode/` (`text`, `model`) and
  `POST /worker_generate_stream` service contracts and record the service/GPU
  placement used for every run.
- Execute the same native nine-value planning action and task-specific released
  postprocessing.
- Accept A's frozen Native-6 task/variation and initialization mapping at the
  environment boundary; emit raw per-episode actions, language inputs,
  success, length, latency, memory, and identity metadata.

## Required frozen inputs from server A

- Evaluation interface bundle and development-only examples for M3/M4 adapter
  integration.
- Immutable `main10_failure_train.jsonl` plus trajectory bundle for all formal
  M4 budget, seed, and LOFO training.
- Native-6 nominal/perturbed manifests, public task mapping, initialization
  contract, physical-fault boundary, and success/audit contract for formal E6.

Until those inputs arrive, server B may run official/public or seeded live
development reproductions, but must not label them as Main-10 or Native-6
formal evidence.
