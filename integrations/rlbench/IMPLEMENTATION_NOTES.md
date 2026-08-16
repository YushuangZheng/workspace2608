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
- RLBench evaluation uses one continuous policy clock and absolute end-effector control.

The following details are not uniquely specified and remain configurable:

- covariance regularization (`1e-5` or `1e-6`) and the exact covariance form;
- whether Equation (6) uses the same position-only subspace as Equation (5);
- the fallback when no frame passes `tau_omega`;
- the exact LiftTray and SweepDust boundary counts and arm-coordination choices;
- task-specific velocity thresholds, boundary merging, and temporal-consensus rules;
- whether gripper targets use the current or next observation;
- the exact Table II seeds and episode horizon;
- the precise switch between Jacobian IK, sampling IK, and path planning;
- whether the policy clock advances after a failed action.

The literal configuration in `configs/dynamac_table_ii.json` adds a `1e-6` diagonal covariance ridge, uses position-only statistics for Equation (5) and the full pose covariance for Equation (6), and fails when strict thresholding leaves no selected frame. This exposes an unresolved case in StackWine, HandOver, and SweepDust with the current five demonstrations.

The executable local configuration in `configs/dynamac_rlbench_local.json` differs only in that case: it retains all numerically tied highest-scoring frames when no frame exceeds `tau_omega`. This local completion is serialized in every checkpoint and is not treated as an author-confirmed rule. It allows the full experiment matrix to run; low or zero success rates are reported without task-specific retuning.

The segmentation profiles use two boundaries for StoreBottle, six for HandOver, two for LiftTray, and four for SweepDust.
