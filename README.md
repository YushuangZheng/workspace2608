# DynaMAC Reproduction

This repository contains an independent implementation of DynaMAC and
MiDiGaP, together with an RLBench reproduction suite for simulator Tables
I–III. It is based on the papers, pinned public TAPAS/RLBench code, and written
implementation clarifications; it does not contain unreleased author code.

The RLBench integration and its default artifact paths target the local `v2`
protocol. The completed `v1` checkpoints and results remain immutable
historical provenance. The `v2` evaluation matrix is still being regenerated
and must not yet be treated as a final paper comparison.

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

RLBench observation extraction, next-observation gripper encoding, task
profiles, configuration paths, and debug plots remain in
[`integrations/rlbench/rlbench_dynamac/`](integrations/rlbench/rlbench_dynamac/).

### Task frames and model selection

- Equation (5) detects kinematic links from position covariance using
  `tau_M=0.005`; the pointwise mask is promoted to a skill-level mask by a
  strict majority.
- Equation (6) uses `tau_omega=0.5`. The current `v2` configuration evaluates
  it in the same Equation (5)-weighted subspace: 3D position under position and
  rotation weights `1/0`.
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
validation rejects mixed `v1`/`v2` model and result identities.

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
- Dynamic diagnostics move the existing episode's `boundary_root()` without
  calling `task.init_episode()`. Goal sampling restores only the task
  configuration tree, leaves the live robot untouched, rejects newly
  introduced robot–environment collision pairs, and records preservation and
  actual-motion evidence in every applied intervention.
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

## Local releases and artifacts

The working reproduction workspace currently uses:

- 45 five-demonstration `low_dim_obs.pkl` episodes for Tables I–III;
- historical `models/v1/` checkpoints and the completed `results/v1/` set;
- retrained `models/v2/` checkpoints for the current Equation (6)
  configuration;
- a partially regenerated `results/v2/` evaluation matrix;
- 33 confirmed-failure replay videos from archived `v1` evaluations.

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

Pinned external revisions and licenses are recorded in
[`integrations/rlbench/THIRD_PARTY.md`](integrations/rlbench/THIRD_PARTY.md).
Open details required for a stricter author-side match are maintained in
[`integrations/rlbench/OPEN_QUESTIONS.md`](integrations/rlbench/OPEN_QUESTIONS.md).
