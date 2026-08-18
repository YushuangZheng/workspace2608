# DynaMAC Reproduction

This repository contains an independent implementation of DynaMAC and MiDiGaP, together with an RLBench reproduction suite for simulator Tables I–III. It is based on the papers, pinned public TAPAS/RLBench code, and written implementation clarifications; it does not contain unreleased author code.

## Repository layout

- `source/policy/`: DynaMAC, DiGaP, MiDiGaP, and environment-independent TAPAS skill segmentation.
- `configs/dynamac.json`: the fully explicit, fail-closed core configuration.
- `configs/dynamac_smoke.json`: a lightweight configuration for local smoke data.
- `scripts/run.py`: compact fit, verify, and inspect commands using an explicit local dataset.
- `tests/`: mathematical, persistence, integration, and TAPAS-oracle tests.
- `integrations/rlbench/`: pinned-source metadata, task adapters, evaluators, protocols, and tests.

The local RLBench workspace uses the following compact artifact set:

- 45 five-demonstration `low_dim_obs.pkl` episodes for Tables I–III;
- immutable historical `models/v1`, `models/v2`, `results/v1`, and `results/v2`
  release directories;
- `models/v3` and `results/v3` as the current training/evaluation defaults;
- nine confirmed-failure replay videos and the canonical paper comparison.

Raw RGB, depth, and mask observations are not used by policy fitting and are not retained.
Demonstrations, checkpoints, evaluation JSON, videos, and local copies of the reference papers
are intentionally excluded from Git. The repository publishes the implementation, frozen
protocols, tests, dependency pins, and commands needed to regenerate the experiment artifacts.

## Verification

```bash
python -m pip install -e '.[test,midigap]'
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
ruff check --no-cache source scripts tests integrations/rlbench
```

The optional core smoke command takes an explicitly supplied local demonstration bundle:

```bash
python scripts/run.py verify --data /path/to/dynamac_demos.npz
```

No demonstration bundle is tracked in this repository. The smoke bundle uses precomputed coarse
skill labels and is separate from the RLBench/TAPAS segmentation pipeline.

The generic skill segmenter is [source/policy/tapas_segmentation.py](source/policy/tapas_segmentation.py). It accepts only normalized NumPy pose, task-frame, and gripper-state trajectories. RLBench-specific observation extraction, current-observation signed gripper encoding, task profiles, default config paths, and debug-plot handling remain in [integrations/rlbench/rlbench_dynamac/](integrations/rlbench/rlbench_dynamac/).

In the RLBench evaluator, `max_primary_action_attempts=3` is only a local
controller-execution tolerance for `InvalidAction` (IK or low-level execution
failure), not a DynaMAC grasp-retry mechanism. The rejected target is aborted
and the same policy tick is recomputed from a fresh simulator observation; an
action that executes but misses a grasp is still committed normally. This does
not add skill samples, extend the configured skill schedule, or trigger a
contact-conditioned re-grasp. DynaMAC follows a moving task frame only while
the current skill has selected that frame and its majority-gated per-time-step
Equation (5) availability is true. Dynamic onset uses preregistered task/skill
ticks on the committed policy clock. This reproduction uses simulator-state
ground-truth poses, not a visual pose detector.

## RLBench

See [integrations/rlbench/README.md](integrations/rlbench/README.md) for setup,
artifact layout, training, evaluation, and report-generation commands. The
complete frozen V3 mechanism and trigger table are in
[integrations/rlbench/V3_PROTOCOL.md](integrations/rlbench/V3_PROTOCOL.md).

Pinned external revisions and licenses are recorded in [integrations/rlbench/THIRD_PARTY.md](integrations/rlbench/THIRD_PARTY.md). Third-party source trees and CoppeliaSim are installed separately.
