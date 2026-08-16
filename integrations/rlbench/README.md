# DynaMAC on RLBench

This integration contains the maintained local reproduction for:

- Table I: StackWine, PlaceCups, OpenMicrowave, and WipeDesk;
- Table II: StoreBottle, HandOver, SweepDust, and LiftTray;
- Table III: local environment-motion and arm-coordination diagnostics.

Static cells use independent five-demonstration cohorts. Dynamic cells are explicitly non-comparable diagnostics because the paper does not publish the exact DynaBench movement and perturbation protocol.

## Pinned dependencies

| Project | Revision |
|---|---|
| `vonHartz/RLBench` | `tapas@a51b4e609dc5c3e1a8c06046bd87a9da24723da4` |
| `robot-learning-freiburg/TAPAS` | `52e35214b9baa7b190b87196c36b9e98f4006149` |
| `vonHartz/PyRep` | `b8bd1d7a3182adcd570d001649c0849047ebf197` |
| CoppeliaSim Edu | 4.1 |

See [PINNED_SOURCES.json](PINNED_SOURCES.json), [THIRD_PARTY.md](THIRD_PARTY.md), and [patches/](patches/) for exact provenance.

## Runtime setup

Policy fitting and serving use Python 3.10. RLBench, PyRep, and CoppeliaSim use Python 3.8.

```bash
export DYNAMAC_POLICY_PYTHON=/path/to/policy-python3.10
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONPATH=/path/to/RLBench:/path/to/essay2608
```

Install the two environments from [requirements/](requirements/). Demonstrations use the standard layout:

```text
<data-root>/<task>/all_variations/episodes/episode0/low_dim_obs.pkl
```

## Configuration

- [dynamac_table_ii.json](configs/dynamac_table_ii.json): strict paper interpretation; exposes empty Eq. (6) selections in the local cohort.
- [dynamac_rlbench_local.json](configs/dynamac_rlbench_local.json): executable local protocol used by `v1`.
- [tapas_segmentation.json](configs/tapas_segmentation.json): segmentation profiles.
- [tasks.json](configs/tasks.json): task frames and bimanual coordination.
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md): implementation boundary.
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md): details still required for an exact author-side match.

## Code boundary

The numerical skill-segmentation implementation is shared core code in
[`source/policy/tapas_segmentation.py`](../../source/policy/tapas_segmentation.py).
It operates only on normalized NumPy trajectories and contains the velocity,
gripper-change, distance, candidate-merging, alignment, and independent/shared-union
algorithms. This integration keeps the RLBench protocol in
[`rlbench_dynamac/tapas_segmentation.py`](rlbench_dynamac/tapas_segmentation.py): the
default config path, next-observation action timing, signed gripper encoding, debug
plots, and compatibility exports. Task-specific profile values and arm-coordination
choices remain in `configs/` and are applied by `demo_adapter.py`.
The core result object retains the historical v1 claim text as inert provenance so
existing training manifests remain exactly verifiable; it does not select any
RLBench task or protocol.

## Artifacts

- Demonstrations: `data/` (locally, 45 low-dimensional episodes; no unused image streams).
- Authenticated checkpoints: `models/v1/`.
- Audited 200-episode outputs: `results/v1/`.
- Failure replays: `results/failure_videos/v1/`.
- Generated comparison: `results/paper_comparison.md`, with CSV and JSON beside it.

These experiment artifacts are intentionally excluded from Git. They contain generated or
upstream-derived data and should be transferred or published separately only after checking
the applicable data and asset licenses. The tracked `data/README.md` defines the required
layout; the commands below regenerate models, evaluations, and reports.

Release directories follow the compact `vN` convention: `models/v1`, `results/v1`, and `results/failure_videos/v1`. Future `v2` and `v3` artifacts should use parallel directories rather than overwrite an earlier release. Select a report release explicitly with `paper_comparison --release vN`.

## Training

Existing `v1` models are immutable. Use a new output directory for a retrain.

```bash
# Table II bimanual tasks
python3.10 -m integrations.rlbench.rlbench_dynamac.direct_policy train \
  --task all \
  --data-root integrations/rlbench/data/dynamac_table_ii_g5_a51b4e_128x128_seed0_20260811/stage_5_demos \
  --models-dir integrations/rlbench/models/retrained

# Table I unimanual tasks
python3.10 -m integrations.rlbench.rlbench_dynamac.direct_policy train \
  --task all-unimanual \
  --data-root integrations/rlbench/data/dynamac_table_i_live_g5_seed0 \
  --models-dir integrations/rlbench/models/retrained
```

## Evaluation

The evaluators load `models/v1` by default. They use absolute world-frame end-effector control, Jacobian IK followed by sampling IK for the same target, and a one-step no-op only when both fail.

```bash
# Table II static example
python3.8 -m integrations.rlbench.rlbench_dynamac.direct_evaluate \
  --task bimanual_handover_item --episodes 1 --seed 0 --horizon 1000 \
  --output /tmp/handover_smoke.json --headless

# Table I static example
python3.8 -m integrations.rlbench.rlbench_dynamac.unimanual_evaluate \
  --task stack_wine --scenario static --episodes 1 --seed 0 --horizon 1000 \
  --output /tmp/stack_wine_smoke.json --headless
```

Regenerate the consolidated report without rerunning experiments:

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.paper_comparison
```

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider integrations/rlbench/tests
ruff check --no-cache integrations/rlbench
```
