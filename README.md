# DynaMAC Reproduction

This repository contains an independent implementation of DynaMAC and MiDiGaP, together with an RLBench reproduction suite for simulator Tables I–III. It is based on the papers, pinned public TAPAS/RLBench code, and written implementation clarifications; it does not contain unreleased author code.

## Repository layout

- `source/policy/`: DynaMAC, DiGaP, MiDiGaP, and environment-independent TAPAS skill segmentation.
- `configs/dynamac.json`: the fully explicit, fail-closed core configuration.
- `configs/dynamac_smoke.json`: the runnable configuration for the bundled smoke data.
- `scripts/run.py`: compact fit, verify, and inspect commands.
- `tests/`: mathematical, persistence, integration, and TAPAS-oracle tests.
- `integrations/rlbench/`: pinned-source metadata, task adapters, low-dimensional demonstrations, authenticated checkpoints, evaluators, and results.

The local RLBench workspace uses the following compact artifact set:

- 45 five-demonstration `low_dim_obs.pkl` episodes for Tables I–III;
- `integrations/rlbench/models/v1/` as the only checkpoint set;
- `integrations/rlbench/results/v1/` as the only numerical result set;
- nine confirmed-failure replay videos and the canonical paper comparison.

Raw RGB, depth, and mask observations are not used by policy fitting and are not retained.
Demonstrations, checkpoints, evaluation JSON, and videos are local experiment artifacts and
are intentionally excluded from Git. The repository publishes the implementation, frozen
protocols, tests, dependency pins, and commands needed to regenerate them.

## Verification

```bash
python -m pip install -e '.[test,midigap]'
python scripts/run.py verify
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
ruff check --no-cache source scripts tests integrations/rlbench
```

The small `data/dynamac_demos.npz` file is a self-contained core training smoke test, not an RLBench benchmark dataset. Its coarse skill labels are precomputed and do not come from the RLBench/TAPAS segmentation pipeline.

The generic skill segmenter is [source/policy/tapas_segmentation.py](source/policy/tapas_segmentation.py). It accepts only normalized NumPy pose, task-frame, and gripper-state trajectories. RLBench-specific observation extraction, next-observation gripper encoding, task profiles, default config paths, and debug-plot handling remain in [integrations/rlbench/rlbench_dynamac/](integrations/rlbench/rlbench_dynamac/).

## RLBench

See [integrations/rlbench/README.md](integrations/rlbench/README.md) for setup, artifact layout,
training, evaluation, and report-generation commands.

Pinned external revisions and licenses are recorded in [integrations/rlbench/THIRD_PARTY.md](integrations/rlbench/THIRD_PARTY.md). Third-party source trees and CoppeliaSim are installed separately.
