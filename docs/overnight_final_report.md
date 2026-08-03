# Overnight DynaMAC research final report

## Outcome

The requested six-phase loop is complete on branch
`codex/overnight-dynamac-audit`. It converts the existing single-arm prototype
into an auditable scientific baseline, separates paper-derived and project-new
mechanisms, evaluates six methods in 480 isolated simulator trials, and closes
the minimal bimanual handover environment/data chain without claiming a full
bimanual DynaMAC result.

The strongest new result is mechanistic rather than end-task superiority:
RelationDynaMAC revokes every forced-drop relation in 40 ms and rejects every
empty closure, while both legacy online variants retain the relation until
scheduled opening and falsely connect on every miss. All three nevertheless tie
at 51/60 on the six ordinary conditions and fail all explicit drop/miss recovery
trials. Online relation estimation is therefore validated as a detector, not as
a recovery policy.

## Integrity and protected assets

- `DynaMAC.pdf` was not modified; SHA-256 remains
  `3fdf0a6ac46bced00885ea01e2a21d918ce12f4fd832a3e0b2d97ed34af10431`.
- The single-arm frozen dataset was never rewritten. All methods train on the
  same five `data/pick_place_static/v1` demonstrations, dataset SHA-256
  `8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8`.
- Test seeds 6300–6309 were reserved after implementations and thresholds were
  frozen; they are simulator instances, not independent training datasets.
- Old outputs were retained. New scientific outputs use new directories and
  carry source/config/data fingerprints.
- Frozen `data/handover_static/v1` was retained unchanged. The corrected
  four-label handover schema is a separate frozen v2.

## Phase results

### 1. Single-arm metric and result audit

The semantic success definition now requires XY placement, support height,
released gripper, and post-release stability. The legacy 3-D error remains as a
diagnostic, not the success gate. The audit reconciled all 72 pre-existing
trials and exposed a 59 mm fixed Z offset between the target reference and
support geometry that made the legacy 3-D threshold misleading.

Paired Full-versus-Mask analysis finds a mean 104.6 mm shorter path for Full
over 18 matched trials, concentrated in lift (77.8 mm) and transport (25.0 mm),
but only 10/18 individual pairs are shorter. Path and duration differences have
correlation 0.86. This supports a coupled frame/timing effect, not a claim of
global path optimality.

Primary report: `docs/single_arm_scientific_audit.md`.

### 2. Paper-faithful simplified SkillDynaMAC

`SkillDynaMACPolicy` implements fixed manual skills, 6-D pose covariance,
simplified Eq. (5) link selection, simplified Eq. (6) frame selection, and
skill-local virtual frames fitted only from the five demonstrations. It is
explicitly separate from the legacy online prototype.

Across 90 clean trials, SkillDynaMAC succeeds in 10/18 static/arm-offset trials
and recovers in 8/15 eligible trials, but scores 0/18 on target-motion
conditions. The simplified covariance rules over-select static references; this
is a negative but useful faithful-baseline result.

Primary report: `docs/method_provenance.md`; output:
`outputs/single_arm_scientific/skill_baseline_v1`.

### 3. Velocity segmentation diagnostic

The data-derived diagnostic recovers five repeatable velocity segments in every
frozen demonstration and four boundary clusters supported by all five demos.
Aligned boundary-time standard deviation averages 39 ms, but nearest manual
edge deviation averages 211 ms. The detected groups are coarse approach/grasp,
transport/place, and retreat structure; they are not semantic TAPAS skills.

Primary report: `docs/segmentation_analysis.md`; output:
`outputs/single_arm_scientific/segmentation_v1_clean/analysis.json`.

### 4. Bidirectional online relation estimator

The new phase-independent estimator uses four hysteretic states
(`DISCONNECTED`, `CANDIDATE_CONNECTED`, `CONNECTED`, `CANDIDATE_LOST`), actual
finger opening/velocity, relative 6-D motion, windowed stability, co-motion
correlation, optional contact, asymmetric thresholds, and continuous confidence.
All thresholds were calibrated from frozen demonstrations, never test seeds.

Deterministic tests cover empty closure, successful transport, closed-gripper
drop, and external ungrasped object motion. In the clean simulator smoke, all
six ordinary conditions succeed; forced drop disconnects in 40 ms, and empty
closure never connects. Neither counterexample is recovered because no regrasp
or phase-replanning graph exists.

Primary report: `docs/online_relation_estimator.md`; outputs:
`outputs/single_arm_scientific/relation_calibration_v1_clean/analysis.json` and
`outputs/single_arm_scientific/relation_smoke_v1_clean`.

### 5. Expanded held-out single-arm evaluation

The accepted matrix contains six methods, eight conditions, and ten seeds: 480
independent Isaac workers, 480 JSON/NPZ pairs, 480 unique tuples/fingerprints,
one schema/source/data identity, zero missing metrics, and a maximum phase-path
partition residual of `6.67e-16 m`.

Ordinary six-condition stable-place success:

| Method | Success | Mean XY error | Mean path | Mean compute |
|---|---:|---:|---:|---:|
| World Gaussian | 0/60 | 164.91 mm | 1.129 m | 0.035 ms |
| Static Multi-stream | 9/60 | 34.28 mm | 1.164 m | 0.219 ms |
| SkillDynaMAC | 38/60 | 25.04 mm | 1.113 m | 0.296 ms |
| Mask-only | 51/60 | 4.76 mm | 1.222 m | 0.244 ms |
| Legacy Full | 51/60 | 4.74 mm | 1.084 m | 0.249 ms |
| RelationDynaMAC | 51/60 | 5.04 mm | 1.117 m | 0.899 ms |

Counterexample mechanism results:

| Method | Drop loss delay | Empty closure false connection | Drop recovery | Miss recovery |
|---|---:|---:|---:|---:|
| Mask-only | 0.910 s | 10/10 | 0/10 | 0/10 |
| Legacy Full | 0.880 s | 10/10 | 0/10 | 0/10 |
| RelationDynaMAC | 0.040 s | 0/10 | 0/10 | 0/10 |

Wilson intervals, seed-balanced bootstrap intervals, recovery, onset/release,
phase paths, jumps, maximum speed, failure taxonomy, and exact reproduction are
in `docs/single_arm_final_report.md`. Accepted output:
`outputs/single_arm_scientific/v1/summary.json`.

### 6. Minimal bimanual handover chain

`Essay2608-Bimanual-Handover-v0` now exposes two independent absolute-pose
Franka IK actions, two independent grippers, and explicit left/right EE, object,
target, and measured two-finger observations. The 13-state expert records the
complete `none → left_only → both → right_only → none` supervision sequence.

One separate seed-7300 smoke episode succeeds in 575 steps with 10.62 mm final
error. Formal v2 collection accepts five complete episodes from eight isolated
worker attempts. All five contain a 15-step/0.30 s co-hold interval. The frozen
v2 digest is
`91706df18abfea606c9e6836f1864e675610633ce5cb0c3c23846a1ea4f5fe18`;
maximum final error is 11.04 mm, maximum Cartesian step is 29.21 mm, and minimum
initial-object separation is 13.91 mm. Refreeze and recollection attempts both
fail closed and leave the manifest unchanged.

Primary report: `docs/bimanual_handover_setup.md`; frozen data:
`data/handover_static/v2`.

## Commits

| Commit | Deliverable |
|---|---|
| `4d0b806` | Audit single-arm success metrics and experiment validity |
| `c979a94` | Add paper-faithful skill-level DynaMAC baseline |
| `48346b7` | Record SkillDynaMAC baseline evaluation |
| `7bfdfc4` | Add velocity-based skill segmentation diagnostics |
| `0620b7b` | Record clean segmentation analysis fingerprint |
| `1143f17` | Prototype bidirectional online relation estimation |
| `7362714` | Record clean online relation evaluation |
| `3673dd2` | Prepare expanded single-arm evaluation metrics |
| `e8df22c` | Run expanded single-arm DynaMAC evaluation |
| `5c4921e` | Prepare audited bimanual handover dataset v2 |
| `7489495` | Add bimanual handover environment and scripted demonstrations |

## Validation and reproduction commands

Final pure test and syntax checks:

```bash
conda run -n env_isaaclab python -m pytest -q
conda run -n env_isaaclab python -m compileall -q \
  source/essay2608/essay2608 scripts tests
git diff --check
```

Result: 20 tests pass. The environment does not currently contain the optional
`ruff` executable, so it was not used as an acceptance signal.

Key integration commands:

```bash
conda run -n env_isaaclab python scripts/eval_single_arm.py --headless \
  --methods world_gaussian static_multistream skill_dynamac mask_only \
  full_dynamac relation_dynamac \
  --conditions static smooth_object sudden_object smooth_target sudden_target \
  arm_offset drop_after_grasp close_without_grasp \
  --seeds 6300 6301 6302 6303 6304 6305 6306 6307 6308 6309 \
  --output_dir outputs/single_arm_scientific/v1

conda run -n env_isaaclab python scripts/collect_handover.py --headless \
  --num_demos 5 --max_attempts 10 --seed 7400 \
  --output_dir data/handover_static/v2
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir data/handover_static/v2
```

## Resolved and unresolved questions

Resolved:

- success is now semantic and stable rather than a loose 3-D position check;
- all compared single-arm methods use one immutable five-demo training set;
- paper-faithful simplified SkillDynaMAC is distinct from online project code;
- velocity segmentation repeatability and semantic mismatch are quantified;
- online relation establishment and loss have counterexample coverage;
- detector correctness is separated from end-task recovery;
- path changes are attributable by destination phase;
- the handover task has an explicit observation/label/data contract and an
  immutable audited v2 dataset.

Unresolved:

- no policy returns to approach/regrasp after `CANDIDATE_LOST` or a miss;
- relation detection does not yet improve ordinary-condition task success;
- SkillDynaMAC's simplified Eq. (5–6) selection is brittle under tiny
  orientation covariance and target dynamics;
- the velocity segments are not learned semantic skills;
- the single-arm result is one custom task with ten correlated condition slices
  per seed, not broad benchmark evidence;
- the handover object is gravity-disabled and kinematically attached to one
  carrier; scripted `both` is not observed contact truth;
- the existing older bimanual policy pilots remain outside this accepted
  skeleton and should not be cited as a full DynaMAC comparison.

## Paper claims and project direction

The evidence supports a narrow paper narrative: static task-parameter products
can become causally harmful when relations change; relation-aware masking and
virtual frames improve dynamic single-arm behavior; and a bidirectional,
actual-state relation estimator fixes forced-loss and empty-closure detector
failures that a phase/gripper-command latch cannot detect.

It does not support equivalence to TAPAS, MiDiGaP, Riemannian DynaMAC, or
DynaBench; ordinary-task superiority of RelationDynaMAC; recovery after loss;
contact-rich bimanual transfer; or broad generalization.

The clearest project-owned research direction is to couple the validated
four-state relation lifecycle to a recovery graph, then replace scripted
bimanual relation labels with observed two-arm contact/relative-motion evidence.
Only after those mechanisms pass causal counterexamples should the repository
fit and compare concurrent bimanual DynaMAC policies.

## Next-day top three checks

1. Inspect ten representative videos/traces: one success and one failure for
   SkillDynaMAC, Full, and RelationDynaMAC, plus drop and miss counterexamples;
   confirm the numeric taxonomy matches physical behavior.
2. Design a frozen recovery-state protocol (`LOST → retreat → re-approach →
   regrasp`) and pre-register success/recovery criteria before running new test
   seeds.
3. Decide whether to invest next in contact-rich bimanual physics or improve the
   paper-faithful frame/link estimator; do not claim a bimanual learning result
   from the current geometric skeleton.
