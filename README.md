# DynaMAC Reproduction

This repository contains an independent implementation of DynaMAC and
MiDiGaP, together with an RLBench reproduction suite for simulator Tables
I–III. It is based on the papers, pinned public TAPAS/RLBench code, and written
implementation clarifications; it does not contain unreleased author code.

The RLBench integration and its default artifact paths target the frozen local
`v3` protocol. Completed `v1` and `v2` checkpoints and results remain immutable
historical provenance. V3 uses a sealed, reusable fixed evaluation set; its
formal 200-episode matrix is complete, while dynamic results remain explicitly
non-comparable diagnostics wherever the paper protocol is unpublished.

## What is implemented

- DynaMAC with single-mode DiGaP trajectory models on the product manifold
  `R³ × S³`, including task-frame transforms, Riemannian Gaussian fitting,
  kinematic-link analysis, task-frame selection, and product-of-experts
  inference.
- MiDiGaP clustering and multimodal policy support as a separate core module.
- Environment-independent TAPAS-style skill segmentation from end-effector
  velocity, gripper-state changes, and task-frame distance signals.
- Single-arm and dual-arm RLBench adapters, training entry points, evaluators,
  result authentication, comparison reports, and failure-replay tooling.
- Offline regression tests for core mathematics, segmentation, model identity,
  controller transactions, scene launch, dynamic interventions, and reporting.

## Main mechanisms in the current reproduction

### Skill segmentation

The generic implementation lives in
[`source/policy/tapas_segmentation.py`](source/policy/tapas_segmentation.py) and
accepts normalized NumPy trajectories rather than RLBench objects. It produces
candidate boundaries from end-effector stops, binary gripper transitions, and
optional task-frame distance events, then merges and aligns corresponding
stages across demonstrations.

For dual-arm demonstrations, `independent` segmentation lets each arm retain
its own boundaries and schedule. `shared_union` takes the union of both arms'
candidate events and assigns the resulting shared boundaries to both policies.
The choice and task-specific thresholds are dataset protocol, not a universal
DynaMAC equation.

RLBench observation extraction, current-observation signed gripper encoding, task
profiles, configuration paths, and debug plots remain in
[`integrations/rlbench/rlbench_dynamac/`](integrations/rlbench/rlbench_dynamac/).

### Task frames and model selection

- Equation (5) detects kinematic links from position covariance using
  `tau_M=0.005`. A strict skill majority decides only whether to enable the raw
  per-time-step mask; availability itself remains time-indexed.
- Equation (6) uses `tau_omega=0.5`. The current `v3` configuration evaluates
  it in the same Equation (5)-weighted subspace: 3D position under position and
  rotation weights `1/0`. At each time step its denominator includes only
  available frames, and final participation is
  `Eq6Selected(frame) AND Eq5Available(frame,t)`.
- If strict Equation (6) thresholding selects no frame, the executable local
  protocol keeps the numerical argmax. This is an explicit local completion,
  not a confirmed author-side rule.
- Each skill has a frozen virtual frame captured at its first sample. Earlier
  virtual frames remain available to later skills.
- Current RLBench models use aligned time-state observations, TAPAS-style
  index subsampling, diagonal empirical covariance plus a `1e-6` ridge, and a
  single mode per skill.

The exact Equation (6) covariance subspace, empty-selection behavior, and some
task-specific segmentation settings remain author-side reproduction questions.
They are serialized in every checkpoint; authenticated evaluator and report
validation reject mixed `v1`/`v2`/`v3` model and result identities.

### Dual-arm coordination

The bimanual wrapper runs two DynaMAC policies concurrently. Each arm can use
the other end effector as a candidate dynamic task frame, and both predictions
are computed from the same pre-action simulator snapshot. The arms retain
their own skill clocks; an arm that finishes first holds its final command
while the other completes. RLBench-specific action layout, IK, gripper control,
and simulator stepping are kept outside the core policy.

### Evaluation protocol

- Policies command absolute world-frame end-effector poses. Jacobian IK is
  attempted before sampling IK.
- Grippers actuate at `0.04`, matching the pinned demonstration generator
  rather than the vendor evaluation default `0.2`.
- A policy tick is transactional. An RLBench `InvalidAction` aborts the
  tentative target and recomputes the same tick from a fresh observation, with
  `max_primary_action_attempts=3`.
- This retry budget handles controller execution failures only. It is not a
  DynaMAC grasp-retry mechanism: an action that executes but misses contact is
  committed, and the fixed skill schedule is not extended.
- Static, smooth, and teleport conditions reuse the same sealed per-episode
  source state; smooth and teleport also share the same preregistered goal B.
  Disposable offline generations certify A and B, while formal rollout binds
  A and never samples, restores, or selects a scene from policy outcomes.
- Dynamic onset uses preregistered task/skill ticks on the committed policy
  clock. Every boundary-root command preserves structural and semantic
  invariants. Its exact robot-contact delta is authenticated as a diagnostic,
  not used to censor a fixed episode based on one policy's evolved pose.
- Policy task-frame inputs are simulator-state ground-truth poses, not outputs
  from a visual pose detector.

These evaluator rules are task-independent local reproduction protocols. The
paper does not fully specify controller failure handling or the exact dynamic
environment intervention implementation.

## Repository layout

- [`source/policy/`](source/policy/): DynaMAC/DiGaP policy logic, MiDiGaP, and
  generic skill segmentation.
- [`source/data/`](source/data/): core demonstration schema and validation.
- [`configs/`](configs/): explicit core and smoke configurations.
- [`scripts/run.py`](scripts/run.py): compact fit, verify, and inspect commands.
- [`tests/`](tests/): core mathematical, persistence, integration, and oracle
  tests.
- [`integrations/rlbench/`](integrations/rlbench/): pinned-source metadata,
  RLBench adapters, task profiles, training/evaluation commands, and tests.
- [`baselines/`](baselines/): isolated, upstream-pinned reproductions of
  FAIL-Detect, RACER, SPR, AgentChord, Guardian/FailCoT, and AHA. Only
  provenance, environment manifests, wrapper scripts, and comparison metadata
  are tracked; upstream sources and generated artifacts remain local.
- [`papers/`](papers/): local reference-paper library. Its tracked index records
  canonical sources and checksums, while PDF files remain excluded from Git.

## Local releases and artifacts

The working reproduction workspace currently uses:

- 45 five-demonstration `low_dim_obs.pkl` episodes for Tables I–III;
- immutable historical `models/v1`, `models/v2`, `results/v1`, and `results/v2`
  release directories;
- `models/v3` and `results/v3` as the current training/evaluation defaults;
- a sealed V3 fixed evaluation set containing no outcomes or model data;
- 33 confirmed-failure replay videos from archived `v1` evaluations and the
  canonical paper comparison.

Demonstrations, checkpoints, result JSON, videos, reference-paper copies,
RoboTwin, RoboDojo, RLBench, TAPAS, PyRep, and CoppeliaSim are intentionally
excluded from Git. The repository publishes the independent implementation,
frozen protocols, tests, dependency pins, patches, and commands needed to
rebuild the external environment and regenerate experiment artifacts. Raw RGB,
depth, and mask streams are not used for policy fitting and are not retained in
the compact local dataset.

## Installation and verification

The core package requires Python 3.10 or newer:

```bash
python -m pip install -e '.[test,midigap]' ruff
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider
ruff check --no-cache .
```

The optional core smoke command takes an explicitly supplied local
demonstration bundle:

```bash
python scripts/run.py verify --data /path/to/dynamac_demos.npz
```

No demonstration bundle is tracked in Git. The smoke bundle uses precomputed
coarse skill labels and is separate from the RLBench/TAPAS segmentation
pipeline.

## RLBench reproduction

See [`integrations/rlbench/README.md`](integrations/rlbench/README.md) for
dependency setup, the low-dimensional demonstration layout, training and
evaluation commands, release directories, and report generation.

The complete frozen V3 mechanism and trigger table are in
[integrations/rlbench/V3_PROTOCOL.md](integrations/rlbench/V3_PROTOCOL.md).

Pinned external revisions and licenses are recorded in
[`integrations/rlbench/THIRD_PARTY.md`](integrations/rlbench/THIRD_PARTY.md).
Open details required for a stricter author-side match are maintained in
[`integrations/rlbench/OPEN_QUESTIONS.md`](integrations/rlbench/OPEN_QUESTIONS.md).
