# Expanded single-arm DynaMAC evaluation

## Protocol and integrity

All methods use the same five frozen `pick_place_static/v1` demonstrations,
dataset SHA-256
`8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8`.
The ten simulator seeds 6300–6309 are evaluation instances, not independent
training sets, and were reserved after method and detector thresholds were
frozen. None was used for tuning.

The matrix contains six geometric methods, eight conditions, and ten seeds:
480 isolated Isaac Lab worker processes. The two counterexamples augment the
original six conditions: `drop_after_grasp` moves the object 18 cm away during
transport while the gripper remains closed, and `close_without_grasp` moves it
18 cm immediately before closure.

Mechanical acceptance passed:

- 480 JSON and 480 NPZ trial files, 480 unique method/condition/seed tuples,
  and 480 unique experiment fingerprints;
- ten seeds with 48 trials each, six methods with 80 trials each, and eight
  conditions with 60 trials each;
- zero missing metrics; one schema version (5), Git commit
  `3673dd2e48115b553c53d28cab30ddf2a38ea68b`, source hash
  `a7123de8c07d5dc4d2d4e725e642e60937e3d4f4a0d72b377b981cd432f2a0c6`,
  and frozen dataset hash;
- maximum residual between the sum of destination-phase paths and total path:
  `6.66e-16 m`.

The accepted result is
`outputs/single_arm_scientific/v1/summary.json`. No prior result directory was
overwritten.

## Methods and claim boundary

- World Gaussian is a single world-frame trajectory baseline.
- Static Multi-stream is the project's translational Gaussian product of
  object and target experts.
- SkillDynaMAC is the paper-faithful simplified, fixed-skill baseline; its 6-D
  link and frame selections are learned offline from the five demonstrations.
- Mask-only and Full are the legacy online engineering ablations. Their detector
  latches while the demonstrated gripper command is closed.
- RelationDynaMAC uses the new four-state bidirectional relation estimator with
  actual finger feedback and phase-independent relation updates.

None is a full TAPAS/MiDiGaP/Riemannian reproduction of the supplied paper.

## Success and recovery

The table reports semantic stable-place success. Each condition has ten trials.
For reference, Wilson 95% intervals are: 10/10 `[0.722, 1.000]`, 9/10
`[0.596, 0.982]`, 8/10 `[0.490, 0.943]`, 2/10 `[0.057, 0.510]`, 1/10
`[0.018, 0.404]`, and 0/10 `[0.000, 0.278]`.

| Method | Static | Smooth obj. | Sudden obj. | Smooth target | Sudden target | Arm offset | Drop | Miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| World | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Static Multi-stream | 10% | 10% | 10% | 20% | 20% | 20% | 0% | 0% |
| SkillDynaMAC | 100% | 90% | 90% | 0% | 0% | 100% | 0% | 0% |
| Mask-only | 90% | 90% | 80% | 80% | 80% | 90% | 0% | 0% |
| Legacy Full | 90% | 90% | 80% | 80% | 80% | 90% | 0% | 0% |
| RelationDynaMAC | 90% | 90% | 80% | 80% | 80% | 90% | 0% | 0% |

Across the six original conditions, descriptive success is 0/60 World, 9/60
Static, 38/60 SkillDynaMAC, and 51/60 for each online method. The three online
methods therefore have equal task success on this matrix; the relation estimator
does not create a success gain under ordinary perturbations.

Including both deliberately unrecoverable counterexamples, the condition-
balanced seed bootstrap estimates are:

| Method | Mean success | Seed-bootstrap 95% CI | Mean recovery | Mean XY error | Mean path |
|---|---:|---:|---:|---:|---:|
| World | 0.0% | [0.0, 0.0]% | 0.0% | 183.52 mm | 1.127 m |
| Static Multi-stream | 11.25% | [2.5, 21.25]% | 11.43% | 94.00 mm | 1.194 m |
| SkillDynaMAC | 47.5% | [42.5, 50.0]% | 40.0% | 80.54 mm | 1.115 m |
| Mask-only | 63.75% | [52.5, 72.5]% | 60.0% | 60.66 mm | 1.266 m |
| Legacy Full | 63.75% | [52.5, 72.5]% | 60.0% | 74.20 mm | 1.172 m |
| RelationDynaMAC | 63.75% | [52.5, 72.5]% | 60.0% | 70.22 mm | 1.166 m |

These all-condition error means include the 20 drop/miss failures per method and
should not be read as placement accuracy conditional on success.

## Bidirectional relation mechanism

The counterexamples separate detector behavior from task recovery:

| Online method | Drop: detected connection | Drop: loss delay | Miss: ever connected | Drop task | Miss task |
|---|---:|---:|---:|---:|---:|
| Mask-only | 10/10 | 0.910 s | 10/10 | 0/10 | 0/10 |
| Legacy Full | 10/10 | 0.880 s | 10/10 | 0/10 | 0/10 |
| RelationDynaMAC | 10/10 | 0.040 s | 0/10 | 0/10 | 0/10 |

The legacy methods clear only when their later gripper command opens, not when
the closed-gripper object is lost. The new estimator revokes every forced-drop
relation after exactly two 20 ms control intervals and rejects every empty
closed gripper. This is the strongest evidence for an essay2608 contribution
beyond the paper-faithful baseline.

Task recovery remains zero for every method. Once the object is dropped or
moved before closure, the phase clock continues rather than returning to an
approach/regrasp state. The result supports the detector mechanism and directly
falsifies any claim that bidirectional detection alone provides recovery.

On the six regular conditions, RelationDynaMAC's observed onset is on average
50–86 ms before the scripted start of phase 4 depending on condition; its normal
release delay is 60 ms. Early onset occurs during late grasp when actual fingers
and rigid co-motion already indicate attachment. Scripted phase 4 is a
comparison convention, not exact physical contact ground truth. For
SkillDynaMAC, the reported negative multi-second “onset” is not meaningful:
its `connected` field is a fixed offline skill label rather than runtime onset.

## Motion, path attribution, and compute

Over the six original conditions:

| Method | Mean XY error | Mean path | Mean policy compute |
|---|---:|---:|---:|
| World | 164.91 mm | 1.129 m | 0.035 ms |
| Static Multi-stream | 34.28 mm | 1.164 m | 0.219 ms |
| SkillDynaMAC | 25.04 mm | 1.113 m | 0.296 ms |
| Mask-only | 4.76 mm | 1.222 m | 0.244 ms |
| Legacy Full | 4.74 mm | 1.084 m | 0.249 ms |
| RelationDynaMAC | 5.04 mm | 1.117 m | 0.899 ms |

The online relation features cost about 3.6 times the legacy Full compute, but
remain below 1 ms on this workstation. All methods cap the rate-limited policy
position jump at 20 mm. Mean maximum physical end-effector speeds range from
0.721 to 0.765 m/s. No method except World has a forced phase transition; World
has eight across 80 trials.

The destination-phase partition localizes the regular-condition path gap:

| Method | Lift phase 4 | Move phase 5 | Total regular path |
|---|---:|---:|---:|
| Mask-only | 0.2010 m | 0.2433 m | 1.2223 m |
| Legacy Full | 0.1253 m | 0.1827 m | 1.0836 m |
| RelationDynaMAC | 0.1442 m | 0.1776 m | 1.1166 m |

RelationDynaMAC retains most of the legacy Full path benefit while paying about
19 mm more in phase 3 and 19 mm more in phase 4 because capture is driven by
observed relation onset rather than a hardcoded phase-4 boundary. This remains a
coupled frame/timing effect, not proof of globally optimal virtual frames.

## Failure taxonomy

Across all 80 trials per method:

- World: 80 placement-XY failures.
- Static: 9 successes, 68 placement-XY failures, 3 environment terminations.
- SkillDynaMAC: 38 successes and 42 placement-XY failures.
- Mask-only and Legacy Full: 51 successes and 29 placement-XY failures each.
- RelationDynaMAC: 51 successes, 27 placement-XY failures, and 2 environment
  terminations in `close_without_grasp`.

The semantic threshold was never relaxed. The stable support, released gripper,
post-release stability, legacy 3-D error, XY sensitivities, raw/rate-limited
jumps, phase paths, maximum speed, relation states, and inference time remain in
each trial for alternative analysis.

## What the result supports and limits

Supported within this custom task:

- static object/target product-of-experts can be causally harmful after grasp;
- masking and a virtual frame improve ordinary dynamic performance and shorten
  the dominant lift/transport path;
- an offline Eq. (5–6)-style skill baseline is reproducible but fails target
  dynamics because simplified frame selection over-selects static references;
- actual-gripper, hysteretic 6-D relation estimation reliably distinguishes
  grasp, forced loss, empty closure, and external ungrasped motion.

Not supported:

- equivalence to TAPAS, MiDiGaP, the paper's Riemannian policy, or DynaBench;
- superiority of RelationDynaMAC in ordinary task success—it ties both legacy
  online methods at 51/60;
- recovery after drop or miss; a phase/replanning layer is still absent;
- broad generalization from ten seeds in one custom task, or independence of the
  eight conditions within a seed.

The next single-arm research step should couple `CANDIDATE_LOST`/`DISCONNECTED`
to an explicit recovery graph, then evaluate whether mechanism improvements
translate into task recovery without altering detector thresholds.

## Reproduction

```bash
conda run -n env_isaaclab python scripts/eval_single_arm.py --headless \
  --methods world_gaussian static_multistream skill_dynamac mask_only \
  full_dynamac relation_dynamac \
  --conditions static smooth_object sudden_object smooth_target sudden_target \
  arm_offset drop_after_grasp close_without_grasp \
  --seeds 6300 6301 6302 6303 6304 6305 6306 6307 6308 6309 \
  --output_dir outputs/single_arm_scientific/v1
```
