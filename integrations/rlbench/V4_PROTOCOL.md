# V4 RLBench protocol

This document is the normative contract for the current RLBench release. V4
uses the sealed `rlbench_eval_v2` data, 200 held-out episodes per formal cell,
seeds `2608000000..2608000199`, horizon `1000`, and the authenticated model
inventory in `models/v4/release_manifest.json`.

The 22 formal cells are:

- Table I: StackWine, PlaceCups, OpenMicrowave, and WipeDesk under
  static, smooth, and teleport conditions (12 cells);
- Table II: static StoreBottle, HandOver, SweepDust, and LiftTray (4 cells);
- Table III: teleport StoreBottle, HandOver, SweepDust, and LiftTray, plus
  HandOver coordination hand-left and hand-right (6 cells).

Dynamic results are local diagnostics rather than exact paper reproductions
because the paper does not publish its complete DynaBench sampler and temporal
intervention protocol.

## Policy and training semantics

Training and inference consume RLBench low-dimensional simulator state, not a
visual detector. Every pose returned by a task's public `get_low_dim_state()`
is a candidate task frame in source order.

The shared TAPAS NumPy implementation segments demonstrations from
end-effector velocity and gripper transitions, merges and aligns candidates,
and resamples corresponding skills to their mean duration. StoreBottle uses
independent arm segmentation; HandOver, LiftTray, and SweepDust use their
configured shared-union profiles. The two arms then execute independently:
each keeps its own skill index, local clock, and duration; a finished arm holds
its final command and there is no mid-skill barrier or resynchronization.

Pose and signed gripper training labels use the same observation `obs[t]`.
RLBench runtime applies the same global skill-boundary gripper lookahead to all
tasks: when the next skill changes the gripper state, that command is issued at
the preceding boundary tick. This is an execution adapter only; it does not
alter segmentation or the saved training labels.

The current DynaMAC selection semantics are:

- Equation (5) computes a raw linked mask per time step. Strict skill majority
  decides whether that time-varying mask is enabled; exactly 50% does not enable
  it.
- Equation (6) uses the Equation (5)-weighted positional subspace (position
  weight `1`, rotation weight `0`) and normalizes over frames available at that
  time.
- Final product-of-experts participation is the Equation (6) skill selection
  intersected with Equation (5) availability at the current time.
- `tau_M=0.005`, `tau_omega=0.5`, diagonal empirical covariance plus `1e-6 I`,
  local `keep_argmax` empty-selection completion, unimodal fitting, and disabled
  temporal-variance filtering are fixed release choices.
- A virtual frame is captured from the current observation at the first sample
  of each skill; earlier virtual frames remain available to later skills.

## Current training cohorts and models

`data/training/manifest.json` authenticates 45 demonstrations and 125 episode
files across nine five-demo cohorts. Eight policy tasks live under
`data/training/main/`; the dynamic HandOver coordination cohort lives under
`data/training/coordination/` because it shares the normal HandOver policy-task
alias.

Two V4 policies are retrained:

- StoreBottle is trained from five successful static demonstrations at
  variation `0`, seeds `4104000000..4104000004`. Its task frames are `bottle`
  and physical object `fridge_base`, exposed to the policy as `fridge`. Bottle
  and fridge remain independently movable semantic groups.
- SweepDust is trained from the final strict five-demo task-frame cohort. The
  five scenes cover center, positive/negative X, and positive/negative Y
  placements while retaining one common task-relative skill template and
  timing. Its augmentation manifest and input hashes are stored with the task.

The remaining six main policies and the separate coordination HandOver policy
are byte-identical inherited checkpoints. `models/v4/release_manifest.json`
binds every model artifact, the StoreBottle semantic identity, and the
SweepDust five-input identity. Moving byte-identical training data changes only
its storage path, not an already released checkpoint identity.

## Sealed evaluation data

The only current evaluation set is materialized under `data/evaluation/`:

- `spec.json` defines tasks, seed namespace, consumers, and isolation rules;
- `manifest.json` seals the selected batches and records the current training
  manifest as provenance;
- `environment/` contains eight A/B batches;
- `coordination/` contains the HandOver A-only initialization batch.

Evaluation artifacts contain no policy result, report, success label, or video.
Candidate B selection uses task/scene validity and only the task-scoped safety
checks declared by its plan protocol; it never reads rollout outcomes. Static
and dynamic consumers of one task/episode bind the same source A; smooth and
teleport consumers bind the same A/B plan where both exist.

Integrity is cell-scoped. Every result records the evaluation-set ID, selected
batch SHA-256/fingerprint, model identity, task semantics, and runtime protocol.
Manifest/spec SHA values are retained as provenance, but changing one batch
invalidates only consumers that actually read that batch.

## Scene staging and interventions

Dynamic plans are prepared independently from formal policy rollout. A
disposable scene deterministically selects and certifies source A, reconstructs
that source for each B attempt, moves the declared task root with one SE(3)
transform, and accepts B only from task and scene validity. The formal scene
then reconstructs and binds A; it never samples B and never performs a
temporary B-to-A restore.

Task-tree audits preserve object topology, parents, joints, semantic
registries, grasp membership, and unmoved world-frame objects. Descendants of a
moved root are checked in that root's relative frame. Each versioned audit
enforces its declared translation, rotation, scalar, and joint tolerances;
quaternion comparison is sign-invariant. Runtime contact changes are recorded
for audit. The OpenMicrowave plan protocol additionally certifies that applying
B at its authenticated trigger introduces no new robot-appliance collision
pairs; other tasks do not inherit this task-specific geometric constraint.

Teleport applies B once. Smooth applies fractions `1/10` through `10/10` on ten
successive committed policy ticks. The policy clock advances normally through
the complete smooth window for every task, including Coordination.

The integer tick, not an approximate rollout fraction, is authoritative. For
profiles inherited from V3, a displayed phase is `local_tick / (duration - 1)`;
StoreBottle and LiftTray use the explicit V4 overrides shown here.

| Dynamic task | Anchor / moved evidence | Skill-0 duration | Trigger tick |
|---|---|---:|---:|
| StackWine | single / `wine_bottle` | 72 | 58 |
| PlaceCups | single / `mug0` | 59 | 45 |
| OpenMicrowave | single / `microwave_door` | 101 | 85 |
| WipeDesk | single / `sponge` | 74 | 52 |
| StoreBottle | left / `bottle` | 115 | 60 |
| HandOver | left / `item0` | 74 | 50 |
| SweepDust | left / `dustpan` | 64 | 53 |
| LiftTray | both / `tray` | 63 | 35 |

Coordination instead starts at global committed tick `235` (left skill 5,
local tick 15). Its five task variations follow `episode_index % 5`, and that
schedule is authenticated in both the result payload and episode rows.

Current task-specific rules include:

- StoreBottle formal dynamics are `bottle_only`: the bottle moves at its
  authenticated trigger while the fridge remains at A. `fridge_only` and
  `both` are diagnostic-only modes.
- OpenMicrowave consumes the current collision-safe sealed A/B batch; plans
  that displaced the appliance into the robot are not part of the evaluation
  set.
- Coordination offsets the selected arm's predicted world-frame target along
  `+Z` over committed ticks `235..244`, from 3 mm through 30 mm. The other arm,
  both orientations, and both grippers remain policy outputs; the 30 mm offset
  persists afterward.

Other task roots and motion bounds are fixed by the current sealed data,
`configs/v3_interventions.json`, and the explicit `configs/v4/` overrides.
Runtime policy code has no per-episode result-based scene selection or
task-specific IK relaxation.

## Controller and committed clock

Each committed tick requests one policy action. For every absolute world-frame
end-effector target, the controller uses this global order:

1. current-seeded CoppeliaSim pseudo-inverse IK;
2. bounded TRAC-IK Distance;
3. collision-aware sampling IK (`100` trials, at most `5` candidates,
   `10 ms`, `ignore_collisions=False`);
4. collision-aware path planning only when the preceding methods all fail and
   end-effector translation is strictly greater than `0.10 m`.

Both arms are prepared before shared physics execution. If the complete chain
cannot produce an action, exactly one raw current-joint hold commits that policy
transaction. The clock and horizon advance once; the episode does not terminate
as an abnormal no-solution case, and it does not skip multiple policy targets
inside one commit.

Grippers actuate at velocity `0.04`, matching the demonstration generator.
After normal policy completion, every task receives up to ten raw settling
steps, stopping at the first success or explicit termination. Settling is not
used after explicit failure or horizon exhaustion.

## Formal execution and reporting

The formal launcher authenticates the training/evaluation manifests, model
release, pinned TRAC-IK build, simulator and policy interpreters, and any
existing result before launching work. It uses at most eight reusable
GPU/Xvfb lanes and starts only cells that are not already
`COMPLETED_VALIDATED`. Result replacement is staged and atomic.

The current consolidated report is
`results/v4/reports/full_22_cell.md`. It contains all 22 formal cells; diagnostic
consumers outside those cells do not enter formal tables or cross-cell totals.

The primary success rate always uses the 200 planned episodes as denominator.
Dynamic results also retain trigger reach, complete-intervention count,
pre-trigger or partial termination, and success conditional on a complete
intervention. A success before the trigger remains a success in the primary
denominator and is marked as an unexercised dynamic condition; it is never
replaced by an extra rollout.

## Post-evaluation replay evidence

Formal evaluation does not record videos. After all 22 result files validate,
the separate replay launcher selects deterministic success/failure episodes by
ascending SHA-256 rank within each outcome class and records only the declared
quota:

| Observed success-rate tier | Successes | Failures |
|---|---:|---:|
| at least 80% and strictly within 2 percentage points of paper | 0 | 0 |
| at least 80%, otherwise | 3 | 3 |
| 50% to below 80% | 5 | 10 |
| below 50% | 5 | 20 |

Available episodes cap each outcome quota; unused quota is never transferred.
Each replay is rerun from the sealed episode identity and publishes atomically
only after its requested outcome is reproduced. Videos combine front and
overhead views. The front camera is unchanged; replay-only overhead perspective
is 70 degrees to keep the task visible at a useful scale.

Replay manifests bind the source result SHA, selected episodes, output hashes,
camera contract, and quota. Replay artifacts remain below
`results/v4/replay_video/` and are not formal evaluation inputs.

## Reproduction boundary

This repository fixes every local ambiguity needed for execution, but does not
claim that unpublished DynaBench sampling, controller failure handling, or
task-specific temporal settings match the paper authors' private experiment
configuration. Those differences affect comparability, not the internal
validity of the sealed V4 protocol.
