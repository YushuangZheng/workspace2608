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
- [dynamac_rlbench_local.json](configs/dynamac_rlbench_local.json): current executable `v2` model configuration. Equation (6) uses the same weighted subspace as Equation (5), which is 3D position under the frozen `1/0` position/rotation weights, for every task and arm.
- [dynamac_rlbench_v1.json](configs/dynamac_rlbench_v1.json): immutable copy of the executable `v1` model configuration.
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
- Historical authenticated checkpoints: `models/v1/`.
- Historical audited 200-episode outputs: `results/v1/`.
- Current retrained checkpoints: `models/v2/`. The `results/v2/` evaluation
  matrix is being regenerated; the report validator rejects dynamic outputs
  that do not use the current preserve-instance motion protocol.
- Failure replays: `results/failure_videos/v1/`.
- Generated `v2` comparison: `results/v2/paper_comparison.md`, with CSV and JSON beside it.

These experiment artifacts are intentionally excluded from Git. They contain generated or
upstream-derived data and should be transferred or published separately only after checking
the applicable data and asset licenses. The tracked `data/README.md` defines the required
layout; the commands below regenerate models, evaluations, and reports.

Release directories follow the compact `vN` convention. The current defaults are `models/v2` and `results/v2`; `models/v1` and `results/v1` remain immutable provenance. Later releases must use parallel directories rather than overwrite an earlier release. Select a report release explicitly with `paper_comparison --release vN`; the report JSON records both that release and the exact expected model/configuration identity.

## Training

Existing `v1` models are immutable. The commands below retrain all tasks into `v2`; this is required because changing the Equation (6) covariance subspace can change frame selection for any skill, not only HandOver.

```bash
# Table II bimanual tasks
python3.10 -m integrations.rlbench.rlbench_dynamac.direct_policy train \
  --task all \
  --data-root integrations/rlbench/data/dynamac_table_ii_g5_a51b4e_128x128_seed0_20260811/stage_5_demos \
  --models-dir integrations/rlbench/models/v2

# Table I unimanual tasks
python3.10 -m integrations.rlbench.rlbench_dynamac.direct_policy train \
  --task all-unimanual \
  --data-root integrations/rlbench/data/dynamac_table_i_live_g5_seed0 \
  --models-dir integrations/rlbench/models/v2
```

## Evaluation

The Table I and Table II evaluators load `models/v2` and write below `results/v2` by default. They use absolute world-frame end-effector control and Jacobian IK followed by sampling IK. Grippers actuate at `0.04`, matching the pinned demonstration generator rather than the vendor evaluation default `0.2`. Dynamic task motion samples and moves `boundary_root()` without calling `task.init_episode()`, so the initialized objects and success conditions remain the same episode instance. These are evaluator-wide rules, not task-specific corrections.

`max_primary_action_attempts=3` is a local controller-execution tolerance, not
a retry count from the DynaMAC paper. It is consumed only when RLBench raises
`InvalidAction` for an IK or low-level execution failure: the tentative target
is aborted, one current-state no-op obtains a fresh observation, and the same
policy time index is recomputed. A command that executes but fails to establish
a grasp is committed normally; the evaluator does not add policy samples,
extend the configured skill schedule, or initiate a semantic/contact-based
re-grasp. The authors' exact failed-action clock semantics and tolerance remain
unconfirmed.

DynaMAC's dynamic following is separate from that tolerance. A moving task
frame changes the recomputed target only while the current skill has not ended
and that skill selected the relevant frame; after the fixed skill transition,
or when the frame was not selected, there is no such tracking. Policy inputs in
this reproduction come from RLBench simulator-state ground-truth poses
(`gripper_pose` and `task_low_dim_state`), not from a visual pose detector.

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
