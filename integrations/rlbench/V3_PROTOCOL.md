# V3 frozen protocol

This document is the preregistered local protocol for `models/v3` and
`results/v3`. Model semantics were frozen for V3 training; the task-specific
temporal triggers were then manually corrected and checkpoint-authenticated
before any formal V3 evaluation. The trigger corrections did not retrain or
change the model arrays. V1 and V2 artifacts remain immutable and are not valid
inputs to a V3 report.

The staged-motion subprotocol is revision V3.4. A disposable proof generation
uses `reset(verify_instance=False)` and exactly one explicit `Task.validate()`
to certify a deterministic source seed, then is discarded. The exported A,
every B retry, and formal rollout each reconstruct that same selected source
seed with a fresh `reset(False)`; neither B sampling nor formal rollout
revalidates A. All reconstructions use strict `1e-6` state audits, while root
motion uses the physically correct world/root-relative task-tree split.
The protocol ID ends in `boundary-root-v3.4`; plan, batch, and validation
schemas end in `-v3.4`. Task-tree snapshots use
`rlbench-task-tree-dual-frame-state-v1`, deterministic reset evidence uses
`dynamac-rlbench-deterministic-source-reset-evidence-v1`, and evaluator IDs
include `staged34-deterministic-source-reset1`. Older plan caches fail closed.

V3 is an independent reproduction protocol, not unreleased author code. Static
RLBench cells are local reproductions from an independent five-demonstration
cohort. Dynamic-environment and arm-perturbation cells remain non-comparable
diagnostics because the paper does not publish the exact DynaBench protocol.

## Model and demonstration semantics

- Equation (5) uses position weight `1`, rotation weight `0`, `d=3`, and
  `tau_M=0.005`.
- Equation (6) uses the same Equation (5)-weighted 3D position subspace,
  `tau_omega=0.5`, and the existing `keep_argmax` empty-selection rule.
- Equation (5) first produces a raw per-frame, per-time-step linked mask. A
  strict skill majority only gates whether that raw mask is enabled: when
  `mean(raw_linked) > 0.5`, availability is `NOT raw_linked[t]`; otherwise the
  raw mask is disabled and availability is true for the whole skill. Exactly
  50% does not enable the mask. The majority never turns V3 into a constant
  skill mask.
- At each time step, Equation (6)'s relative-precision denominator contains
  only frames available at that same time step after this gate. Equation (6)
  still selects a frame over the skill as a whole; product-of-experts
  participation is
  `Eq6Selected(frame) AND Eq5Available(frame, time)`.
- Temporal-variance filtering is disabled (`link_filter=none`).
- Both end-effector pose and signed gripper state are learned from the same
  current observation `obs[t]`. A new gripper state belongs to the first frame
  on which it is observed; V3 does not shift the gripper target to `obs[t+1]`.
- The serialized model schema remains `13` because its structure is compatible,
  but V3 has a distinct selection-semantics identity and training-manifest
  schema. Exact config, manifest, adapter timing, and checkpoint fingerprints
  are authenticated by the report.

## Skill segmentation and two-arm timing

The environment-independent TAPAS NumPy port supplies velocity/gripper-change
boundaries, distance filtering, merging, alignment, and independent or
shared-union segmentation. RLBench code only extracts observations, applies
the frozen task profiles, encodes gripper state, and writes audit evidence.

V3 executes the author-confirmed independent-arm interpretation: each arm has
its own boundaries, mean skill durations, skill index, and local time. One
successfully committed combined control tick advances each unfinished arm once.
There is no intermediate resynchronization, phase barrier, or wait operation;
an arm that finishes first holds its final command while the other continues.

## Preregistered dynamic triggers

Dynamic onset is task-profile data, not a common one-third-of-rollout rule. The
integer local tick is authoritative. Its reported phase is always

```text
phase = local_tick / (skill_duration - 1)
```

and never `local_tick / skill_duration`. These values were selected from the
V2 model manifests and frame-selection windows before V3 results existed. They
must not be tuned from V3 success rates. After V3 training, each anchor must be
verified fail-closed against the new manifest before evaluation starts.

| Task | Anchor arm | Skill | Evidence frame | Duration | Trigger local tick | Phase | V2 raw diagnostic used to preregister the window |
|---|---|---:|---|---:|---:|---:|---|
| StackWine | single | 0 | `wine_bottle` | 72 | 58 | 58/71 = 0.816901 | chosen 58–67; raw linked begins 68; close around 83 |
| PlaceCups | single | 0 | `mug0` | 59 | 45 | 45/58 = 0.775862 | chosen 45–54; raw linked begins 55; close around 75 |
| OpenMicrowave | single | 0 | `microwave_door` | 101 | 85 | 85/100 = 0.850000 | chosen 85–94; raw linked begins 95; close around 114 |
| WipeDesk | single | 0 | `sponge` | 74 | 52 | 52/73 = 0.712329 | chosen 52–61; raw linked begins 62; close around 74 |
| StoreBottle | right | 0 | `fridge_root` | 126 | 110 | 110/125 = 0.880000 | chosen 110–119; raw linked begins 120; close around 126 |
| HandOver | left | 0 | `item0` | 74 | 50 | 50/73 = 0.684932 | chosen 50–59; open gripper is still moving toward the pose above item0 |
| LiftTray | left | 0 | `tray` | 63 | 49 | 49/62 = 0.790323 | chosen 49–58; raw linked begins 59; closes around R79/L84 |
| SweepDust | left | 0 | `dustpan` | 64 | 53 | 53/63 = 0.841270 | chosen 53–62; raw linked begins 63; closes around L78/R86 |

The last column is selection provenance, not an assertion that V3 availability
must switch off immediately after the window. The V3 majority gate can disable
a sparse raw mask and make the full skill available. Retrained checkpoints are
accepted only when every tick in the preregistered intervention window is
Equation (6)-selected, Equation (5)-available, and active in the final PoE.

Each dynamic profile also records a manual, result-independent semantic audit:
the interacting arm and object, the pre-interaction event, and the expected
current-state gripper value. For HandOver, the five training demonstrations map
policy tick 50 to original frames 50/51/49/50/51. At those frames the left
gripper is open and still converging on the pose above `item0`; this replaces
tick 64, which had already reached that pose in two demonstrations. Updating
this evaluator trigger does not alter or retrain the policy arrays.

The coordination diagnostic is also preregistered. Both arm-perturbation
conditions start in the same physical handover stage: left-arm skill 5 at local
tick 15 (global committed tick 235), phase `15/18 = 0.833333`, using `right_ee`
as the authenticated evidence frame. At this point the left giver is closed on
`item0`, while the right receiver is open and has reached the held object; the
following skill performs the gripper transfer. The two conditions differ only
in which arm receives the persistent action offset. Across the five dedicated
coordination demonstrations, this tick maps to original frames
232/240/210/251/256; right-gripper-to-item distances are
0.75/0.59/0.72/0.69/1.32 cm, both gripper states are consistently
left-closed/right-open, and item motion is negligible. Integer ticks are
authoritative here as well.

The coordination task has exactly five registered variations. Formal episode
`i` uses variation `i % 5`; the result records `variation_count=5`, the complete
variation schedule, and the variation on every episode row. Report generation
rejects a missing, reordered, or inconsistent schedule.

Smooth and teleport conditions for the same task and episode must consume the
same staged A/B plan fingerprint. The deterministic staging seed domain is
therefore independent of motion scenario.

## Independent staging and formal rollout

Every dynamic episode preregisters source pose A and goal pose B in an
independent disposable RLBench scene before formal rollout:

1. deterministically try up to 20 source seeds. Each proof generation performs
   one fresh `reset(False)`, checks placement/collision, explicitly calls
   `Task.validate()` once, audits its side effects, and is then discarded;
2. reconstruct the selected source seed with a fresh `reset(False)` for every
   B retry, move only `task.boundary_root()`, audit workspace/collision and the
   rigid move, and validate B with exactly one `Task.validate()` call;
3. export only numeric A/B root poses, a semantic source fingerprint, plan
   fingerprint, seed provenance, and validation evidence;
4. initialize every formal episode, static or dynamic, from the selected source
   seed with `reset(False)` and bind it fail-closed to the complete certified A.

Only the task model is reloaded. The base scene, the unimanual explicit-sensor
handling patch, robot/action-mode objects, and policy input contract are not
rebuilt or modified.

Source A certification and candidate B sampling have independent frozen
maximums of 20 and 100 attempts per episode, respectively
(`source_selection_max_attempts=20` and `goal_sampling_max_attempts=100`). A
different budget is a different protocol
and is rejected before evaluation. Increasing the A budget does not alter its
deterministic seed order or select scenes using rollout results.

Every rejected B generation is discarded. Candidate attempt `k` freshly
reconstructs the same selected source seed A before its strict A-to-B rigid
motion and B-pre/post-validation checks. The accepted plan exports that
certified A, B, source fingerprint, and attempt number.

The formal rollout never samples B, never calls `restore_state()` for goal
sampling, and never performs a temporary move-to-B then restore-to-A. This also
avoids the previously identified OpenMicrowave lower-limit canonicalization
hazard because no staging restore round trip occurs. Any reusable staging
implementation would instead have to canonicalize a restore round trip and
compare against that fixed point; V3 uses the disposable-scene route.

Every applied formal root command has a separate same-instance audit
(`rlbench-formal-boundary-root-same-instance-state-audit-v2`). Its reference is
the policy-evolved formal state immediately before that command, never staged
A. The immediate post-command state must preserve task-tree topology, parents,
joints, and root-relative subtree state at strict `1e-6 m` / `1e-6 rad`;
external task objects remain fixed in world coordinates. Semantic registries,
condition/grasp identities, and gripper membership/parentage remain hard
requirements. Robot collision pairs immediately before and after the command,
their exact set difference, and the corresponding empty-difference flag are
authenticated diagnostics. They are not an episode-admission gate because the
robot pose at the trigger is a policy outcome; filtering a frozen A/B plan on
that pose would make the evaluation set policy-dependent. Smooth motion performs
the structural audit and records the collision delta for every committed
fraction. Controller progress is committed after the structural audit passes.

Cross-process CoppeliaSim handles and Python object identities are not compared.
Task semantic signatures use explicit schemas for built-in RLBench conditions:
structural conditions and parameters remain authenticated, while only declared
execution progress (for example `OrConditions._current_condition_index`) is
excluded. Custom conditions with no reset state serialize every field exactly;
an unmodeled custom reset implementation fails closed.
The task-tree state records every object below the task model base with its
world pose, pose relative to `boundary_root`, parent, joint value, and explicit
membership in the moved root subtree. Within one attempt, A-to-B sampling uses
strict `1e-6 m` / `1e-6 rad` limits: root-subtree objects are compared relative
to `boundary_root`, while all objects outside that subtree stay fixed in world
coordinates. The separate B-before-validation to B-after-validation audit uses
the same strict limits in world coordinates for every object.

Across independent reconstructions of the selected source seed, root pose,
task pose chunks, scalar state, and joint position are all strict at `1e-6` in
their respective units. Quaternion rotation uses the sign-invariant normalized
chord formula `4*asin(shorter_chord/2)`. Object set, parentage, root-subtree
membership, descriptions, semantic registries, grasp state, robot numeric
state, and collision pairs remain exact. Formal binding applies the same audit.
Task-object velocity summaries must be finite but remain diagnostic only.

Plan authentication recomputes the source task-tree and semantic fingerprints,
selected-source fingerprint, object and low-dimensional-state counts, stable
name-based collision records, deterministic reconstruction evidence, and the
nonzero A-to-B root motion. Full task-tree comparison rows are bound to the
source `(name, type)` identities. Re-signing a self-inconsistent cache therefore
does not make it admissible.

The selected source reconstruction, every B retry, and formal A binding all use
the same strict task-independent `1e-6` audit. This does not relax the A-to-B
or immediate formal root-command audits. Runtime logic has no task-specific
branch.

This frame-aware split is necessary because an RLBench task model base can be
an unmoved ancestor of `boundary_root`; requiring that ancestor to retain a
constant pose *relative* to its moved child is physically incorrect. The V3.4
staging protocol and plan schemas invalidate caches produced under the older
contract. Every lifecycle record is fingerprinted and binds generation index,
seed, variation, task, stop/load/start order, and its single verified reset.
The result must record that A and B passed waypoint validation, that
both B tree audits passed, that formal source binding passed, and that formal
rollout performed no sampling or restore.

Waypoint validation is an RLBench benchmark-validity check of expert waypoint
IK/path reachability. It is not part of the DynaMAC policy algorithm.

## Intervention, policy clock, and local rollback

Teleport applies B once at the trigger. Smooth motion applies ten fractions
`1/10, ..., 10/10`, with linear position interpolation and quaternion SLERP.
Each fraction is submitted on a new successfully committed policy tick. A
failed `InvalidAction`, retry, or no-op neither triggers twice nor advances the
next smooth fraction.

The policy worker keeps a logical transaction around each prediction. Up to
three primary attempts are allowed only for RLBench `InvalidAction` caused by
IK or low-level execution. An aborted attempt restores policy runtime state and
recomputes the same tick from a fresh observation; it does not restore the
physical simulator and does not implement grasp retry. A successfully executed
command that grasps nothing still commits. This rollback/tolerance is a local
evaluation safeguard, not a claimed DynaMAC mechanism.

## Generic final settling

Only normal completion of both policy arms starts final settling. The evaluator
stops requesting policy actions, holds the last joint targets and gripper
states, and executes up to ten raw physics/task steps while checking success
and explicit termination after every step. It stops on the first terminal
outcome. These steps do not advance the policy clock or dynamic interpolation.
Retry exhaustion, an explicit failure before normal policy completion, and
horizon exhaustion do not enter settling. Results authenticate the maximum
budget, actual steps executed, and first terminal outcome. The mechanism is
identical for all tasks and contains no task-name branch.

## Dynamic accounting

The primary rate is always `successes / 200 planned episodes`. Every dynamic
result additionally reports:

- episodes that reached the trigger;
- episodes that completed the full A-to-B intervention;
- episodes ending before the trigger (including successes) or during a partial
  intervention;
- successes among complete interventions and the corresponding conditional
  rate.

`intervention_complete` is `true` after a valid teleport or after all ten smooth
steps reach B, `false` for a valid smooth prefix, and `null` before a trigger or
for static evaluation. A success before the trigger stays in the planned main
denominator and is marked as an unexercised dynamic condition, with both
effective and complete intervention fields `null`. A separately collected
cohort of 200 complete interventions, if ever run, is a diagnostic and must
never replace the planned 200-episode denominator without an explicit separate
label.
