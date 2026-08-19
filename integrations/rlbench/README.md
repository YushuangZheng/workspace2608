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
- [dynamac_rlbench_v1.json](configs/dynamac_rlbench_v1.json): immutable copy of the executable `v1` model configuration.
- [dynamac_rlbench_v2.json](configs/dynamac_rlbench_v2.json): immutable `v2` identity; Equation (6) uses the Equation (5)-weighted 3D position subspace uniformly for every task and arm, with position/rotation weights `1/0`, while Equation (5) is promoted by skill majority.
- [dynamac_rlbench_v3.json](configs/dynamac_rlbench_v3.json): current default; it retains the V2 Equation (6) subspace and uses a strict skill-majority gate to decide whether the raw Equation (5) mask remains active per time step.
- [dynamac_rlbench_local.json](configs/dynamac_rlbench_local.json): compatibility alias for the historical V2 config, not a V3 identity source.
- [tapas_segmentation.json](configs/tapas_segmentation.json): segmentation profiles.
- [tasks.json](configs/tasks.json): task frames and bimanual coordination.
- [v3_interventions.json](configs/v3_interventions.json): preregistered task/skill trigger ticks and protocol constants.
- [v3_motion_sources.json](configs/v3_motion_sources.json): spatial roots and deterministic offline source/goal generation budgets.
- [evaluation_set_spec.json](configs/v4/evaluation_set_spec.json): V4 input-only
  specification for `rlbench_eval_v2`. Unchanged tasks remain authenticated,
  zero-copy references to `rlbench_fixed_v1`; StoreBottle and LiftTray require
  newly generated task-scoped plan-batch envelopes before sealing.
- [V3_PROTOCOL.md](V3_PROTOCOL.md): frozen V3 mechanism, trigger evidence, staging, clock, settling, and accounting contract.
- [V4_PROTOCOL.md](V4_PROTOCOL.md): V4 release identity, six-cell first-run
  scope, intervention changes, formal video retention, and diagnostic boundary.
- [V4_STORE_BOTTLE.md](V4_STORE_BOTTLE.md): StoreBottle-only V4 semantic, collection, training, serving, and model-release boundary.
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md): implementation boundary.
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md): details still required for an exact author-side match.

## Code boundary

The numerical skill-segmentation implementation is shared core code in
[`source/policy/tapas_segmentation.py`](../../source/policy/tapas_segmentation.py).
It operates only on normalized NumPy trajectories and contains the velocity,
gripper-change, distance, candidate-merging, alignment, and independent/shared-union
algorithms. This integration keeps the RLBench protocol in
[`rlbench_dynamac/tapas_segmentation.py`](rlbench_dynamac/tapas_segmentation.py): the
default config path, current-observation gripper timing, signed gripper encoding, debug
plots, and compatibility exports. Task-specific profile values and arm-coordination
choices remain in `configs/` and are applied by `demo_adapter.py`.
The core result object retains the historical v1 claim text as inert provenance so
existing training manifests remain exactly verifiable; it does not select any
RLBench task or protocol.

## Artifacts

- Demonstrations: `data/` (locally, 45 low-dimensional episodes; no unused image streams).
- Historical authenticated checkpoints: `models/v1/`.
- Historical audited 200-episode outputs: `results/v1/`.
- Immutable second-release checkpoints and outputs: `models/v2/` and `results/v2/`.
- Current checkpoints and outputs: `models/v3/` and `results/v3/`.
- Tracked, outcome-free fixed evaluation inputs: `evaluation_sets/rlbench_fixed_v1/`.
- V4 evaluation inputs (after generation): `evaluation_sets/rlbench_eval_v2/`.
  `NOT_RUN` and other result state are deliberately excluded from this input
  schema and belong only in V4 result reports.
- Failure replays: `results/failure_videos/v1/`.
- Generated V3 comparison: `results/v3/paper_comparison.md`, with CSV and JSON beside it.

Demonstrations, models, result JSON, and videos are intentionally excluded from
Git. They contain generated or upstream-derived data and should be transferred
or published separately only after checking the applicable data and asset
licenses. The sealed fixed evaluation inputs are tracked because they contain
no outcomes or model data and define the reusable benchmark scenes. The tracked
`data/README.md` defines the demonstration layout; the commands below regenerate
models, evaluations, and reports.

Release directories follow the compact `vN` convention. The current defaults
are `models/v3` and `results/v3`; V1/V2 remain immutable provenance. A release
uses its explicitly named config and parallel artifact directories rather than
overwriting an earlier release. Select a report release with
`paper_comparison --release vN`; the report authenticates the release config,
checkpoint semantics, manifest, adapter, evaluator, and V3 protocol evidence.

## Training

V1/V2 models are immutable. V3 changes both per-time-step Equation (5)
participation and gripper training timing, so all eight main policies and the
separate coordination HandOver policy must be retrained.

```bash
export TABLE_II_DATA_ROOT=/path/to/table_ii/stage_5_demos
export TABLE_I_DATA_ROOT=/path/to/table_i/demos
export TABLE_III_DATA_ROOT=/path/to/table_iii/demos

# Table II bimanual tasks
python3.10 -m integrations.rlbench.rlbench_dynamac.direct_policy train \
  --task all \
  --data-root "$TABLE_II_DATA_ROOT" \
  --models-dir integrations/rlbench/models/v3 \
  --config integrations/rlbench/configs/dynamac_rlbench_v3.json

# Table I unimanual tasks
python3.10 -m integrations.rlbench.rlbench_dynamac.direct_policy train \
  --task all-unimanual \
  --data-root "$TABLE_I_DATA_ROOT" \
  --models-dir integrations/rlbench/models/v3 \
  --config integrations/rlbench/configs/dynamac_rlbench_v3.json

# Table III coordination cohort
python3.10 -m integrations.rlbench.rlbench_dynamac.table_iii_coordination train \
  --data-root "$TABLE_III_DATA_ROOT" \
  --models-dir integrations/rlbench/models/v3/table_iii \
  --config integrations/rlbench/configs/dynamac_rlbench_v3.json
```

## Evaluation

The evaluators load `models/v3` and write below `results/v3` by default. They
use absolute world-frame end-effector control and Jacobian IK followed by
sampling IK. Grippers actuate at `0.04`, matching the pinned demonstration
generator rather than the vendor evaluation default `0.2`.

The Table III coordination diagnostic freezes the dynamic HandOver task at
five variations and assigns episode `i` to variation `i % 5`. Both the complete
schedule and each row's variation are authenticated by the V3 report.

For each dynamic episode, a disposable independent staging environment creates
and waypoint-validates A and B before formal rollout. Only numeric poses,
semantic fingerprints, and validation provenance cross into the formal scene;
the formal scene never samples or restores B. Smooth and teleport share the
same per-episode plan fingerprint and use the preregistered task/skill trigger
on the committed policy clock. Smooth advances one of ten fractions per new
committed tick. See [V3_PROTOCOL.md](V3_PROTOCOL.md) for the fail-closed
contract.

The frozen staging budgets are 20 deterministic candidates for source A and
100 candidates for goal B. These are independent spatial-generation limits;
they do not change the temporal intervention profiles or permit result-based
scene selection.

Staging task-tree evidence is frame-aware. Before every candidate and formal
episode after the first, the preceding task is unloaded while physics is still
running so runtime objects can be cleaned up safely. The evaluator then stops
physics, loads a fresh task environment, seeds, sets the variation, and performs
exactly one `reset(False)`. A disposable proof generation explicitly validates
the selected source seed and is discarded; source replay, every B retry, and
formal A binding then reconstruct that same seed with strict `1e-6` audits.
During
the commanded A-to-B move, descendants of `boundary_root` are compared relative
to that root while ancestors and other objects remain fixed in world
coordinates. Root pose, task pose chunks, scalar state, and joint positions
use strict `1e-6` reconstruction limits. Object topology, parent handles,
boundary-root-subtree membership, semantics, descriptions, grasp state, robot
numeric state, and collisions remain exact. Finite task-object
velocity summaries are diagnostic only. The V3.4 protocol,
plan, batch, and validation schemas reject caches created before this contract;
task-tree snapshots use the dual-frame V1 state schema. Only the task model is
reloaded; the base scene, explicit vision-sensor handling, robot/action mode,
and policy inputs remain unchanged. Typed semantic signatures preserve all
condition structure and parameters while excluding only declared execution
progress fields.

Each applied formal teleport or smooth fraction is also audited from its
current policy-evolved pre-command state to the immediate post-command state.
The same strict task-tree limits, semantic/registry/grasp invariants, and
exact robot-contact before/after/new delta are recorded. Task-tree,
semantic/registry, and grasp invariants remain hard failures. Contact deltas are
diagnostic rather than an admission gate because the trigger-time robot pose is
policy-evolved; controller progress therefore does not censor a frozen A/B
episode according to one policy's trajectory.

`max_primary_action_attempts=3` is a local controller-execution tolerance, not
a retry count from the DynaMAC paper. It is consumed only when RLBench raises
`InvalidAction` for an IK or low-level execution failure: the tentative target
is aborted, one current-state no-op obtains a fresh observation, and the same
policy time index is recomputed. A command that executes but fails to establish
a grasp is committed normally; the evaluator does not add policy samples,
extend the configured skill schedule, or initiate a semantic/contact-based
re-grasp. The authors' exact failed-action clock semantics and tolerance remain
unconfirmed. The local logical rollback does not undo physical simulator state.

DynaMAC's dynamic following is separate from that tolerance. A moving task
frame changes the recomputed target only while Equation (6) selected it and the
frozen Equation (5) mask is available at the current frame. Both arms keep
independent schedules and have no mid-skill resynchronization. After normal
two-arm policy completion, the evaluator allows up to ten raw hold/settle steps
for every task, stopping at the first success or explicit termination; it does
not settle after retry, explicit failure, or horizon termination. Policy inputs
come from RLBench simulator-state ground-truth poses
(`gripper_pose` and `task_low_dim_state`), not a visual pose detector.

```bash
# Table II static example
python3.8 -m integrations.rlbench.rlbench_dynamac.direct_evaluate \
  --task bimanual_handover_item --episodes 200 --seed 2608000000 \
  --eval-set-id rlbench_fixed_v1 --horizon 1000 \
  --output integrations/rlbench/results/v3/table_ii/handover_fixed.json --headless

# Table I static example
python3.8 -m integrations.rlbench.rlbench_dynamac.unimanual_evaluate \
  --task stack_wine --scenario static --episodes 200 --seed 2608000000 \
  --eval-set-id rlbench_fixed_v1 --horizon 1000 \
  --output integrations/rlbench/results/v3/table_i/stack_wine_fixed.json --headless
```

Regenerate the consolidated report without rerunning experiments:

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.paper_comparison --release v3
```

The main dynamic success rate always uses all 200 planned episodes. The report
also gives trigger reach, complete-to-B count, incomplete count, and success
conditioned on complete intervention. A complete-only diagnostic cohort cannot
replace the planned denominator.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider integrations/rlbench/tests
ruff check --no-cache integrations/rlbench
```
