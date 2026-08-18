# Implementation notes

The current default is the frozen local V3 protocol. The normative specification
is [V3_PROTOCOL.md](V3_PROTOCOL.md); this file summarizes the boundary between
the reproduced policy, RLBench adaptation, and local evaluation safeguards.

## Policy and training

- Every pose returned by a task's public `get_low_dim_state()` is a candidate
  task frame in source order.
- The shared core TAPAS NumPy port segments demonstrations from end-effector
  velocity and gripper-state transitions, merges and aligns candidates, and
  resamples each corresponding skill to its mean duration.
- StoreBottle uses independent arm segmentation. HandOver uses a shared union
  with six boundaries; LiftTray and SweepDust use local shared-union profiles
  with two and four boundaries. These task-specific segmentation choices remain
  pending author confirmation, while the resulting arm policies execute on
  independent clocks.
- Both learned pose and signed gripper state use the same current sample
  `obs[t]`. The first observation showing a new gripper state belongs to the new
  state; V3 does not shift gripper targets to `obs[t+1]`.
- The author-confirmed bimanual interpretation is independent execution. Each
  arm keeps its own skill boundaries, durations, skill index, and local time.
  One committed combined tick advances each unfinished arm once; there is no
  intermediate barrier or resynchronization, and a finished arm holds its last
  command.
- V3 Equation (5) first computes a raw linked mask at every time step. Strict
  `mean(raw_linked) > 0.5` only decides whether that raw mask is enabled. If
  enabled, availability is `NOT raw_linked[t]`; otherwise availability is true
  for the whole skill. Exactly 50% does not enable it.
- Equation (6) uses the same weighted subspace as Equation (5): position weight
  `1`, rotation weight `0`, therefore three positional dimensions. At each
  time, its normalization denominator contains only frames available at that
  time. Equation (6) still selects over the whole skill, and final PoE
  participation is `Eq6Selected(frame) AND Eq5Available(frame,t)`.
- `tau_M=0.005`, `tau_omega=0.5`, the local `keep_argmax` empty-selection rule,
  diagonal empirical covariance plus `1e-6 I`, and unimodal fitting remain
  frozen. Temporal-variance filtering remains disabled.
- A virtual frame is captured from the current observation at the first sample
  of each skill. Earlier virtual frames remain available to later skills.

## RLBench protocol

- Training and inference consume low-dimensional simulator ground truth, not a
  visual detector.
- Evaluation uses absolute world-frame end-effector targets, Jacobian IK with a
  sampling-IK fallback, and gripper actuation speed `0.04`, matching the pinned
  demonstration generator.
- Dynamic onset is selected from the preregistered task, anchor arm, skill, and
  integer local tick in `configs/v3_interventions.json`; there is no common
  one-third trigger. The reported phase is `local_tick/(duration-1)`.
- A disposable staging simulator unloads any preceding task while physics is
  still running, then stops physics and loads a fresh task environment for
  every candidate. It then seeds, sets variation, and performs exactly one
  reset before waypoint-validating A and B.
  The formal rollout receives numeric poses and semantic fingerprints, binds
  its initialized source A fail-closed, and never samples B or restores a
  temporary B-to-A move. Smooth and teleport use the same plan fingerprint for
  each task/episode.
- A disposable proof generation certifies a deterministic source seed and is
  discarded. Every candidate retry and formal episode freshly reconstructs
  that same selected seed as its strict A-to-B source.
- Task-tree evidence records world and boundary-root-relative poses plus root
  subtree membership. Within an attempt, B-pre/post-validation checks use
  strict `1e-6` world-pose limits for every object; A-to-B uses the same strict
  limits with root-relative poses only for moved descendants and world poses
  for ancestors and other objects. This avoids treating an unmoved task base
  ancestor as though it should move relative to `boundary_root`.
- Selected-source reconstruction uses strict `1e-6` limits for root/task pose,
  scalar, and joint state. Topology, parents, subtree membership, semantics,
  descriptions, grasp/robot state, and collision pairs remain exact. The
  rotation metric uses a sign-invariant quaternion chord/`asin` formula.
  Finite object-velocity summaries are diagnostic, not identity.
- Every formal static, dynamic, and coordination episode uses the same
  pre-stop task unload (when present), stop/reload/start/seed/variation/
  single-reset lifecycle. Runtime task objects are therefore cleaned up while
  their simulator handles remain valid. The base scene, explicit sensor
  handling, action mode, and policy inputs are unchanged.
- Built-in condition semantics use typed structural schemas; only declared
  execution counters are excluded, and custom reset state fails closed.
- The staging protocol plus plan, batch, and validation schemas are revision
  V3.4; task-tree snapshots use `rlbench-task-tree-dual-frame-state-v1`.
- Each formal root command (including every smooth fraction) snapshots the
  current policy-evolved task immediately before `set_pose`, then strictly
  checks the post-command task tree at `1e-6`, semantic/registry identities,
  and grasp membership/parentage. Robot collision pairs are captured before
  and after the command with an authenticated exact delta, but remain
  diagnostic: using the policy-evolved robot pose as an A/B validity gate would
  make fixed-set episode admission policy-dependent. Controller progress is
  committed only after the structural and semantic audit succeeds.
- Independent staging also avoids the OpenMicrowave lower-limit joint
  canonicalization hazard because the formal rollout never restores a
  temporary B-to-A task state. No task-specific tolerance or global relaxation
  is used. V3.4 staged-plan schemas and deterministic-source reset V1 evidence
  contract reject caches from earlier lifecycles.
- Teleport moves to B once. Smooth applies fractions `1/10` through `10/10`, one
  per unique committed policy tick. Invalid actions and no-ops neither retrigger
  nor advance interpolation.
- Normal policy completion enables a task-independent hold for up to ten raw
  physics/task steps. Success or explicit termination stops it early. Retry
  exhaustion, a prior explicit failure, and horizon exhaustion do not settle.

## Local evaluation safeguards

`max_primary_action_attempts=3` is a controller-execution tolerance, not a
DynaMAC grasp-retry mechanism. Only RLBench `InvalidAction` from IK or low-level
execution consumes it. The worker aborts its tentative policy transaction,
restores logical policy runtime, obtains a fresh post-failure observation, and
recomputes the same tick. It does not restore physical robot/object/contact
state. An action that executes but misses a grasp commits normally.

This transactional rollback, the three-attempt allowance, independent staging,
and final settling are explicitly local evaluator mechanisms. They affect the
measured protocol but are not claimed as mechanisms described by the DynaMAC
paper.

## Release identity

- `dynamac_rlbench_v1.json`: full-pose Equation (6), skill-majority constant
  mask, next-observation gripper target, and the local `keep_argmax` completion
  when strict Equation (6) selection is empty.
- `dynamac_rlbench_v2.json`: Equation (5)-weighted positional Equation (6),
  uniformly across every task and arm with no HandOver-specific override;
  otherwise V1 mask/timing semantics.
- `dynamac_rlbench_v3.json`: majority-gated raw per-time-step Equation (5),
  current-observation gripper target, and the V3 protocol in this document.
- `dynamac_rlbench_local.json` is a V2 compatibility alias and is never the V3
  report identity.

V1/V2 models and results remain immutable. A V3 result is accepted only when
its model schema, distinct selection-semantics ID, explicit V3 config, training
manifest, aligned adapter timing, checkpoint fingerprints, trigger-mask
evidence, evaluator protocol, staging evidence, committed clock, settling, and
dynamic accounting all authenticate. Version directories are never mixed.
Table III coordination results additionally authenticate the fixed five-way
variation cycle (`episode % 5`) at both payload and per-episode row level.

## Remaining author-side unknowns

The covariance subspace and `keep_argmax` behavior remain local
interpretations. The exact paper DynaBench A/B sampler, per-task onset,
interpolation, perturbation protocol, seeds, horizon, controller fallback,
invalid-action clock, and any contact-conditioned re-grasp are unpublished.
Consequently dynamic environment and coordination results remain explicitly
non-comparable diagnostics. See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).
