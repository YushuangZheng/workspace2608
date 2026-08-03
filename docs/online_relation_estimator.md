# Bidirectional online relation estimator

## Scope and provenance

`OnlineRelationEstimator` is a new essay2608 mechanism, not the link analysis
from the DynaMAC paper. The paper identifies links from within-skill
demonstration precision before fitting a policy. This estimator consumes runtime
robot/object state, can both establish and revoke a relation, and never reads
the current phase. `KinematicConnectionDetector` remains unchanged as the
legacy, translation-only, open-command-reset detector.

`RelationDynaMACPolicy` is a separate policy label, `relation_dynamac`. It uses
the new relation decision to mask the object stream and captures a virtual
end-effector frame on the transition to `CONNECTED`. Its current frame
activation still uses phase 4 for the virtual-frame stream, so only the relation
logic—not the complete controller—is phase independent.

## Inputs and state machine

Each update consumes:

- actual summed Franka finger-joint opening and its actual velocity;
- object-in-end-effector relative position and orientation;
- instantaneous relative linear and angular velocity;
- windowed relative-position RMS variation and orientation span;
- windowed object/end-effector velocity correlation and minimum co-motion speed;
- optional contact evidence when a contact sensor exists.

The current custom task has no object contact sensor, so contact is recorded as
unavailable and is not required. The estimator does not use
`world_model.mean_gripper` as its gate.

| State | Forward transition | Cancellation/recovery |
|---|---|---|
| `DISCONNECTED` | connection score ≥ 0.65 → `CANDIDATE_CONNECTED` | remain disconnected |
| `CANDIDATE_CONNECTED` | three qualifying samples → `CONNECTED` | score < 0.40 → `DISCONNECTED` |
| `CONNECTED` | loss score ≥ 0.70 → `CANDIDATE_LOST` | remain connected |
| `CANDIDATE_LOST` | three qualifying loss samples → `DISCONNECTED` | loss score ≤ 0.35 → `CONNECTED` |

`CANDIDATE_LOST` retains the connected control decision until loss is confirmed,
so a one-step relative-pose spike does not switch frames. Establishment requires
co-motion and correlation; retention does not require continued motion. Opening
the gripper, an empty fully closed gripper, or sustained relative translation,
rotation, position dispersion, or orientation dispersion can cause loss.
Confidence is a continuous [0, 1] exponential moving average with coefficient
0.30. It follows the connection evidence while disconnected and one minus loss
evidence while connected.

## Frozen-data calibration

Calibration uses all five frozen demonstrations and no simulator test seed. The
manual states 4–6 identify positive connected windows only during calibration;
states 0–2 and 8–9 provide the actual open-gripper distribution. Phase labels
are also used to score the offline replay, but are never passed to `update()`.

The accepted calibration has 263 complete ten-step positive windows. The 99th
percentile plus a 1.25 margin determines connection-side motion thresholds;
position/orientation floors prevent sub-resolution thresholds. Loss thresholds
are deliberately wider than establishment thresholds.

| Quantity | Connection side | Loss side |
|---|---:|---:|
| occupied opening band | 0.02257–0.06267 m | open ≥ 0.07129 m or near-empty close |
| actual gripper speed | ≤ 0.02045 m/s | not a sole loss trigger |
| relative linear speed | ≤ 0.02197 m/s | ≥ 0.075 m/s |
| relative angular speed | ≤ 0.02775 rad/s | ≥ 0.105 rad/s |
| relative-position RMS std | ≤ 0.00050 m | ≥ 0.002 m |
| relative-orientation span | ≤ 0.005 rad | ≥ 0.030 rad |
| co-motion speed | ≥ 0.06166 m/s | not required for retention |
| velocity correlation | ≥ 0.79942 | not required for retention |

The opening lower bound is half the first percentile of occupied-gripper
openings. This rejects a fully closed miss: the successful grasps stop at an
approximately 45.3 mm summed finger opening because the cube occupies the
gripper, whereas an empty gripper can close near zero.

## Offline replay

Replaying actual joint and pose arrays from the five demonstrations gives a mean
onset offset of -8 ms relative to the scripted start of state 4 (range -220 to
+80 ms) and a 60 ms release delay relative to state 7. Mean false-positive and
false-negative fractions against states 4–6 are 0.01845 and 0.00681. The early
onset on one demonstration occurs late in the grasp dwell when actual occupied
fingers and rigid co-motion already provide evidence; the scripted phase label
is therefore a comparison convention, not direct physical ground truth.

## Mechanism counterexamples and simulator smoke

Deterministic unit tests exercise all four required mechanisms:

1. a fully closed gripper that misses a stationary object never leaves
   `DISCONNECTED`;
2. occupied fingers plus rigid correlated transport pass through
   `CANDIDATE_CONNECTED` and reach `CONNECTED`;
3. forced relative object motion with the gripper still closed passes through
   `CANDIDATE_LOST` and returns to `DISCONNECTED`;
4. an externally moving ungrasped object never creates a relation.

The first Isaac Lab smoke run uses seed 6200 only as a held-out mechanism check,
not for threshold tuning. `relation_dynamac` succeeds in all six original
conditions, with 120 ms connection onset delay, 60 ms normal release delay, and
about 0.009 connected false-positive fraction. Smooth and sudden external object
motion before grasp do not create a persistent false connection.

Two new instantaneous perturbations expose the missing recovery layer:

- `drop_after_grasp` teleports the object 18 cm away onto the support during
  transport without opening the gripper. The estimator revokes the relation in
  40 ms, but the phase-clock policy does not regrasp and final placement fails.
- `close_without_grasp` moves the object 18 cm immediately before closure. No
  connection is ever declared (maximum confidence 0.067), but the policy
  continues its fixed phase sequence and final placement fails.

These are correct detector outcomes and failed task outcomes. Bidirectional
relation estimation is necessary for recovery, but not sufficient: recovery
also needs a replanning/regrasp transition policy.

## Reproduction

```bash
conda run -n env_isaaclab python scripts/analyze_relation_estimator.py \
  --data_dir data/pick_place_static/v1 \
  --output_dir outputs/single_arm_scientific/relation_calibration_v1_clean

conda run -n env_isaaclab python scripts/eval_single_arm.py --headless \
  --methods relation_dynamac \
  --conditions static smooth_object sudden_object smooth_target sudden_target \
  arm_offset drop_after_grasp close_without_grasp \
  --seeds 6200 \
  --output_dir outputs/single_arm_scientific/relation_smoke_v1_clean
```

The accepted runs are regenerated after the implementation commit so their
source commit and content fingerprints identify clean code.
