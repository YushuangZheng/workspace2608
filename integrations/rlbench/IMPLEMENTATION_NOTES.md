# Implementation Notes

The current configuration follows the DynaMAC paper and the available implementation details:

- All poses returned by `get_low_dim_state()` are candidate task frames, in source order.
- Demonstrations are segmented independently and each corresponding skill is resampled to its mean duration.
- End-effector velocity and gripper transitions are both used as boundary signals.
- StoreBottle uses independent arm segmentation. HandOver uses the union of both arms' boundaries. LiftTray and SweepDust currently use the same shared-union strategy.
- The Table II tasks are fitted as unimodal DiGaP policies without modal partitioning.
- Equation (5), Equation (6), and policy fitting use the aligned time-state stream with the current end-effector pose.
- The Equation (5) mask is promoted to a skill-level mask when it is active for more than half of that skill.
- Position is weighted by `1`, rotation by `0`, `d=3`, `tau_M=0.005`, and `tau_omega=0.5`.
- A virtual frame is captured at the first sample of each skill and remains fixed. Earlier virtual frames remain available to later skills.
- RLBench evaluation uses absolute end-effector control. In the local `v2`
  execution protocol, a policy tick is transactional and commits only after
  the primary combined action succeeds. Only an `InvalidAction` raised by IK
  or low-level execution aborts the tentative target; it is followed by a
  current-state no-op and recomputation from a fresh observation at the same
  policy time index, with at most `max_primary_action_attempts=3` attempts.
- Dynamic following is a policy property, not a consequence of that retry
  budget: the target responds to a moving task frame only while the current
  skill remains active and its selected-frame mask includes that frame.
- Evaluation policy observations use simulator-state ground-truth end-effector
  and task-frame poses; this reproduction does not run a visual pose detector.

The following details are not uniquely specified and remain configurable:

- covariance regularization (`1e-5` or `1e-6`) and the exact covariance form;
- whether the authors' reported Equation (6) uses the same position-only subspace as Equation (5); the local `v2` protocol freezes that interpretation explicitly;
- the fallback when no frame passes `tau_omega`;
- the exact LiftTray and SweepDust boundary counts and arm-coordination choices;
- task-specific velocity thresholds, boundary merging, and temporal-consensus rules;
- whether gripper targets use the current or next observation;
- the exact Table II seeds and episode horizon;
- the precise switch between Jacobian IK, sampling IK, and path planning;
- whether the authors advance the policy clock after an invalid controller
  action, how many execution attempts they tolerate, and whether an action
  that executes but misses contact causes any contact-conditioned phase hold
  or re-grasp.

The local limit of three attempts is therefore controller fault handling, not
a paper-level DynaMAC grasp-retry mechanism. It neither adds samples to a skill
nor extends the configured skill schedule. An action accepted by RLBench is
committed even if it did not establish the intended grasp; no semantic re-grasp
is triggered. The authors have not yet confirmed the corresponding failure
semantics.

The `v2` evaluator also applies two environment-wide parity fixes. First,
single- and dual-arm discrete grippers both actuate at `0.04`, the velocity
used while executing the pinned RLBench demonstrations, instead of the
vendor evaluation class's `0.2`. Second, dynamic diagnostics move the existing
episode's `boundary_root()` directly and never call `Scene.kidnap()`,
`Scene.move_task_smoothly()`, or `task.init_episode()`; goal sampling is
transactionally rolled back before the intervention, and the task state and
condition/grasp registries are checked for instance preservation. Smooth
motion uses fractions `1/N` through `N/N`, including the exact endpoint. Both
rules apply uniformly to every task.

The literal configuration in `configs/dynamac_table_ii.json` adds a `1e-6` diagonal covariance ridge, uses position-only statistics for Equation (5) and the full pose covariance for Equation (6), and fails when strict thresholding leaves no selected frame. This exposes an unresolved case in StackWine, HandOver, and SweepDust with the current five demonstrations.

The archived `v1` executable configuration in `configs/dynamac_rlbench_v1.json` differs from the literal configuration only in that case: it retains all numerically tied highest-scoring frames when no frame exceeds `tau_omega`.

The current `v2` executable configuration in `configs/dynamac_rlbench_local.json` also sets `eq6_covariance_scope=eq5_weighted_subspace`. With the frozen Equation (5) weights (`position=1`, `rotation=0`), both selection equations therefore operate on the same 3D position subspace. This is a single configuration-level rule for every RLBench task, arm, skill, and candidate frame; there is no HandOver-specific frame override. Both inferred choices are serialized in every checkpoint and remain explicitly local pending author confirmation. Because Equation (6) can select a different frame in any skill, every Table I/II task must be retrained for `v2`; mixing a `v1` checkpoint with a `v2` result is rejected by the report's exact training-configuration identity check.

The segmentation profiles use two boundaries for StoreBottle, six for HandOver, two for LiftTray, and four for SweepDust.
