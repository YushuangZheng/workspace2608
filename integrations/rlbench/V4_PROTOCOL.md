# V4 RLBench protocol

V4 is selected explicitly with `release=v4`. It uses the sealed evaluation
input set `rlbench_eval_v2`, the held-out seeds `2608000000..2608000199`, and
200 episodes per formal cell. Evaluation inputs contain scenes and plan
identities only: outcomes, `NOT_RUN`, reports, and videos remain under
`results/v4/` and must never be written back into the evaluation set.

`rlbench_eval_v2` preserves unchanged `rlbench_fixed_v1` artifacts as
authenticated, read-only references. StoreBottle and LiftTray instead use
regenerated task-scoped plan envelopes that bind task semantics, motion source,
and intervention identities. Candidate B selection is based only on scene
validity; policy outcomes are not available to generation or sealing.

## Model and controller boundary

Only `bimanual_put_bottle_in_fridge` is retrained in V4. Its five static
demonstrations, corrected task frames, and model authentication are documented
in [V4_STORE_BOTTLE.md](V4_STORE_BOTTLE.md). All other model groups are
inherited from V3 without retraining.

V4 retains the V3 limit of three primary execution attempts for a policy tick.
The policy clock advances only after a successful primary action commit. The
controller still tries Jacobian IK first; its sampling fallback now uses
collision checking (`ignore_collisions=False`), rejects malformed, non-finite,
or out-of-limit candidates, and selects the valid solution nearest the current
joint configuration. This changes execution robustness, not the learned
policy or the three-attempt budget.

## Task-scoped interventions

| Task | V4 formal intervention |
|---|---|
| StoreBottle | The bottle and physical fridge are independent semantic roots. Episodes cycle through `bottle_only`, `fridge_only`, and `both`. Before the corresponding skill-0 policy action, the fridge moves at committed tick 45 and the bottle at tick 60; moved entities are applied once and independently. Formal scenarios are static and teleport only. |
| LiftTray | The trigger is committed skill-0 tick 35, before requesting that policy action, while both grippers are still open and approaching the tray. Teleport B is source-relative: a world-XY radial translation of 3–8 cm, unchanged Z, absolute yaw change at most 0.10 rad, and unchanged roll/pitch. Formal scenarios are static and teleport only. |
| HandOver coordination | At committed global tick 235, the selected arm starts from its measured current end-effector pose and receives a one-shot smooth Cartesian world `+Z` push of 3 cm over 10 substeps. Orientation is fixed; the other arm and both grippers hold their measured states. Substeps do not request policy actions or advance the policy clock. The refreshed observation then resumes the unmodified policy, with no persistent target offset. |

All three intervention families use integer committed-policy ticks fixed before
formal evaluation. Formal result files record their protocol IDs and the
`rlbench_eval_v2` manifest and selected-batch hashes.

## First formal execution scope

The first V4 formal run is deliberately limited to six cells:

| Cell | Paper reference used by video retention |
|---|---:|
| StoreBottle static | 0.82 |
| StoreBottle teleport | 0.82 |
| LiftTray static | 1.00 |
| LiftTray teleport | 1.00 |
| HandOver coordination hand-left | 0.97 |
| HandOver coordination hand-right | 0.97 |

This list is an execution scope, not a claim that a cell has run or achieved a
particular result. Every other formal cell is `NOT_RUN` in the partial result
report. That status is report state only; the complete input catalog remains
available and outcome-free in `rlbench_eval_v2`.

Generate or refresh the six-cell report without launching the simulator:

```bash
python -m integrations.rlbench.rlbench_dynamac.v4_partial_report
```

The report validates any present result before publishing its statistics. A
missing result remains `NOT_RUN`; the report does not synthesize a full Table
I–III or a cross-cell average. V4-versus-V3 deltas are descriptive
multi-factor comparisons and are not attributed to one code change.

## Formal video evidence

Formal V4 commands must enable the V4 release gate and lightweight episode
capture:

```text
--release v4 --record-v4-evaluation-videos
```

The evaluator streams front-camera RGB from returned high-level observations
across every actual episode's full trajectory. It first records the complete
200-episode cell, then uses a fixed-seed, outcome-stratified quota:

| Observed success-rate tier | Retained successes | Retained failures |
|---|---:|---:|
| at least 0.80 and within strictly less than 2 percentage points of the paper reference | 0 | 0 |
| at least 0.80 otherwise | 3 | 3 |
| 0.50 to below 0.80 | 5 | 10 |
| below 0.50 | 5 | 20 |

If an outcome class contains fewer episodes than its quota, all available
episodes in that class are retained and unused quota is not transferred. The
selection manifest records the complete pre-selection inventory, retained
paths, hashes, and quota. Selection and pruning must finish before the formal
result JSON is committed; an interrupted cell keeps its unselected recordings
but has no selection manifest or formal result.

## Diagnostic boundary

The OpenMicrowave gripper-timing A/B run is a separate development diagnostic.
It compares the current gripper command with a single close command at
committed tick 113 on an independent paired development cohort. It is marked
`PROVISIONAL / NON_COMPARABLE`, reads neither sealed evaluation set, and is
ineligible for formal Tables I–III. It must not be counted among the six V4
formal cells.

Canonical machine-readable definitions live in
[`configs/v4/`](configs/v4/): `evaluation_set_spec.json`,
`store_bottle_intervention.json`, `store_bottle_motion_source.json`,
`lift_tray_intervention.json`, `lift_tray_motion_source.json`, and
`coordination_intervention.json`.
