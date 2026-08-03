# Overnight DynaMAC progress

Started: 2026-08-04 (Asia/Shanghai)

## Baseline and constraints

- Branch: `codex/overnight-dynamac-audit`
- Starting commit/checkpoint: `57e01c4fef0cacda9a37e7313afcff40c0b1b496`
  (`single-arm-v1`)
- The starting worktree was clean, so no synthetic checkpoint commit was created.
- Python: 3.10.19
- Isaac Lab: 0.54.0
- Isaac Sim: 4.5.0.0
- PyTorch: 2.7.0+cu128
- NumPy: 1.26.4
- GPU: NVIDIA GeForce RTX 4090; driver 550.90.07
- Frozen single-arm dataset: `pick_place_static/v1`, five demonstrations,
  SHA-256 `8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8`

The frozen dataset, its manifest, `FROZEN`, existing output directories, and the
paper PDF are read-only inputs for this work. Test seeds are evaluation-only and
will not be used for threshold tuning.

## Execution plan

1. Re-audit the completed strict single-arm correction and add missing phase-level
   causal diagnostics without overwriting prior summaries.
2. Separate method provenance and add a simplified paper-faithful skill-level
   baseline while preserving compatibility names.
3. Add velocity-based skill segmentation as a diagnostic-only tool.
4. Add a bidirectional online relation estimator and synthetic plus simulator
   counterexample tests using thresholds derived from training/calibration data.
5. Run the stable methods on at least ten held-out evaluation seeds and preserve
   per-trial evidence under a new fingerprinted output directory.
6. Audit the already-present bimanual handover environment, expert, frozen data,
   and smoke path; only fill gaps rather than rebuilding it.

## Phase log

### Phase 1 — scientific audit

Status: complete.

Existing work at the checkpoint already corrected the fixed 59 mm vertical
residual, semantic placement success, threshold sensitivity, action-rate limiting,
post-step terminal reads, and experiment fingerprints. The remaining Phase 1 work
is to quantify Mask-only versus Full behavior by phase and determine which phases,
durations, frame switches, or forced transitions explain the path difference.

Completed work:

- Added explicit legacy 3-D, XY, and composite stable-place indicators.
- Added an additive phase-attribution module and a thin analysis entry point.
- Reconciled all 72 saved strict traces without rerunning or overwriting them.
- Verified that every phase partition reproduces the saved path within
  `2.3e-16 m`.
- Found that Full is shorter in only 10/18 pairs. Mean Full-minus-Mask path is
  -77.80 mm in lift and -24.99 mm in move-above-target; all other phases account
  for about -1.81 mm.
- Found a strong duration/path association (`r = 0.86`) and zero forced
  transitions. This limits the claim to a coupled virtual-frame/timing effect.
- Added `docs/single_arm_scientific_audit.md` with the target/support geometry,
  metric reconciliation, causal limits, and reproduction command.

Validation:

- `python -m compileall -q source/essay2608/essay2608 scripts tests`
- `python -m pytest -q` — 5 passed
- `python scripts/analyze_phase_diagnostics.py ...` — 36 Mask/Full traces,
  18 exact pairs, no partition failure
- `git diff --check`

### Phase 2 — method provenance and skill-level baseline

Status: complete.

Implementation completed before the full evaluation matrix:

- Renamed the runtime engineering controller to `OnlineDynaMACPrototype` while
  preserving `DynaMACPolicy` as a compatibility alias and `full_dynamac` as its
  result label.
- Added `SkillDynaMACPolicy`, which follows Algorithm 1 and Eqs. (5–6) using
  the project's phase labels, Gaussian fitting, and translational product of
  experts.
- Added a 6-D position/rotation-vector covariance while retaining the original
  3-D covariance used for action fusion.
- Added all ten skill-start virtual frames, fixed per-skill frame selection, and
  serializable training diagnostics.
- Added explicit policy and perturbation configurations to evaluation
  fingerprints and advanced the schema version.
- Documented exact provenance and omissions in `docs/method_provenance.md`.

Training-only diagnostics show that the unweighted 6-D determinant is dominated
by tiny rotational variance on this constrained dataset: it calls the object
linked in phase 0 and phases 2–9. Eq. (6) also over-selects historical virtual
frames in phase 5. These are recorded as limitations rather than hidden by
evaluation-seed tuning.

Pre-commit validation so far:

- `python -m pytest -q` — 8 passed
- `python -m compileall -q source/essay2608/essay2608 scripts tests`
- Static Isaac Lab smoke, seed 6200 — stable-place success, 4.65 mm XY error,
  59.18 mm 3-D error, 322 steps, 1.037 m path, no forced transition
- Six-condition Isaac Lab smoke, seed 6200 — 4/6 stable-place and recovery
  successes. Both 10 cm target shifts failed at 67.83 mm XY error, exposing the
  cost of over-selected static virtual streams; the result is retained rather
  than used to tune thresholds.

Clean evaluation at implementation commit `c979a94`:

- Command: `conda run -n env_isaaclab python scripts/eval_single_arm.py
  --headless --methods world_gaussian static_multistream skill_dynamac
  mask_only full_dynamac --conditions static smooth_object sudden_object
  smooth_target sudden_target arm_offset --seeds 6200 6201 6202 --output_dir
  outputs/single_arm_scientific/skill_baseline_v1`
- Integrity: 90/90 unique combinations, 90 JSON and 90 NPZ files, all metrics
  present, schema 3, one source hash, one commit, and the frozen dataset hash.
- World Gaussian and Static Multi-stream: 0/18 stable-place successes each.
- SkillDynaMAC: 10/18 successes and 8/15 recoveries. All object shifts
  succeeded; all six target shifts failed; seed 6201 also failed static and arm
  offset. Condition-balanced mean XY error is 25.95 mm and path is 1.152 m.
- SkillDynaMAC's offline fixed link labels have 0.621 mean false-positive
  fraction against scripted physical-link phases, zero false-negative fraction,
  0.259 m mean raw frame-switch jump limited to 0.020 m, and zero forced phase
  transitions.
- Mask-only and the online project prototype: 18/18 successes and 15/15
  recoveries each. These stronger engineering results do not make them
  paper-faithful methods.

Implementation commit: `c979a94 Add paper-faithful skill-level DynaMAC baseline`.

### Phase 3 — automatic skill segmentation diagnostic

Status: complete.

- Added a source-level velocity diagnostic using both end-effector linear and
  angular speeds, shared training-only quantile calibration, persistent
  low-speed intervals, short-run removal, close-run merging, endpoint-aware
  candidate extraction, and reference-free cross-demo alignment.
- Added a thin analysis/visualization entry point with dataset/source/config
  fingerprinting and two plots.
- All five frozen demonstrations produce five automatic segments and four
  boundary clusters with 5/5 support. The aligned boundary time standard
  deviation averages 39 ms.
- Candidate times differ from the nearest manual controller transition by
  211 ms on average because grasp/release dwell centers are events while manual
  states label the dwell edges.
- The result supports coarse approach-and-grasp, transport-and-place, and
  retreat macros, but velocity alone cannot isolate gripper semantics and does
  not reproduce TAPAS.

Validation:

- `conda run -n env_isaaclab python -m pytest -q` — 11 passed
- `conda run -n env_isaaclab python scripts/analyze_segmentation.py ...` — five
  demos, segment counts `[5, 5, 5, 5, 5]`, four fully supported clusters
- Both generated figures inspected for complete traces, thresholds, manual
  transitions, candidates, and alignment
- `python -m compileall -q source/essay2608/essay2608 scripts tests`
- `git diff --check`

Clean reproduction: `outputs/single_arm_scientific/segmentation_v1_clean` at
commit `7bfdfc4`, source hash `30cded39e7941bf39070079771db321fcbb8effb311094fec530fa6b38d348c4`,
analysis fingerprint
`867c512ca7a7ecee6a6905cd71303d9ed749534cd207a7afd7949c7d983dd3eb`.

### Phase 4 — bidirectional online relation estimation

Status: complete.

- Preserved `KinematicConnectionDetector` as the legacy implementation and
  added a phase-independent four-state `OnlineRelationEstimator`.
- Added actual finger opening/velocity, 6-D relative motion, windowed stability,
  object/EE velocity correlation, optional contact, asymmetric connection/loss
  thresholds, temporal hysteresis, and continuous confidence.
- Calibrated every threshold from all five frozen demonstrations; no simulator
  test seed contributes to calibration.
- Added the independent `relation_dynamac` policy, source/config fingerprints,
  per-step relation state/confidence/gripper traces, onset/release/loss delays,
  and two new perturbations.
- Frozen-demo replay: mean onset offset -8 ms, release delay 60 ms, false
  positive 0.01845, false negative 0.00681 against scripted states 4–6.
- Four deterministic mechanism tests cover miss, successful transport, closed
  gripper drop, and external object motion.
- Dirty-run Isaac smoke, seed 6200: 6/6 original conditions succeeded; onset
  delay 120 ms and release delay 60 ms. Forced drop revoked in 40 ms; the miss
  never connected. Both counterexample tasks failed because regrasp/replanning
  is not implemented.

Validation so far:

- `conda run -n env_isaaclab python -m pytest -q` — 16 passed
- `conda run -n env_isaaclab python scripts/analyze_relation_estimator.py ...`
  — five complete replays and a visually inspected confidence/state plot
- `conda run -n env_isaaclab python scripts/eval_single_arm.py ...` — eight
  isolated workers, complete JSON/NPZ trials
- `python -m compileall -q source/essay2608/essay2608 scripts tests`
- `git diff --check`

Clean reproduction at implementation commit `1143f17`:

- `outputs/single_arm_scientific/relation_calibration_v1_clean`: source hash
  `23056a2b48bdca97620f545ba5c73a47e22545d62dc227e769093fcf44786a11`,
  analysis fingerprint
  `d74669c3ece5682d3c4d76ff276899867a87774f1976cb81a2359c245ca195cb`.
- `outputs/single_arm_scientific/relation_smoke_v1_clean`: 8/8 complete unique
  JSON/NPZ pairs, schema 4, common source hash
  `66fd9063d7032306e1d0ba8c5187e6248b546a2fc749567b6103772f9f6454ca`.
- The six regular task outcomes remain 6/6; only drop is marked as expecting
  relation loss and reports 40 ms post-event loss. Miss never connects.

Implementation commit: `1143f17 Prototype bidirectional online relation estimation`.

### Phase 5 — expanded single-arm evaluation

Status: complete.

- Reserved ten new held-out simulator seeds: 6300–6309. These were not used in
  implementation smoke tests or threshold calibration.
- Stable method set: World Gaussian, Static Multi-stream, SkillDynaMAC,
  Mask-only, legacy online prototype, and bidirectional relation prototype.
- Condition set: the six existing perturbations plus `drop_after_grasp` and
  `close_without_grasp`.
- Added an exact destination-phase path partition to every trial and aggregate;
  schema 5 also retains onset/release/loss delay, action jump, maximum speed,
  inference time, recovery, and failure taxonomy.
- Planned matrix: 6 methods × 8 conditions × 10 seeds = 480 isolated Isaac Lab
  processes under a new output directory. No result will be used to alter
  thresholds or success criteria.

Accepted matrix at commit `3673dd2`:

- 480/480 unique complete trials, 480 JSON/NPZ pairs, ten 48-trial seed slices,
  one source/data hash, schema 5, and phase-path residual ≤ `6.67e-16 m`.
- Regular six-condition success: World 0/60, Static 9/60, SkillDynaMAC 38/60,
  Mask-only 51/60, legacy Full 51/60, RelationDynaMAC 51/60.
- All methods fail all drop and miss task trials. RelationDynaMAC nevertheless
  revokes every drop in 40 ms and rejects every empty closure; legacy online
  methods revoke only after 0.88–0.91 s and falsely connect in every miss.
- Regular-condition mean path: Mask 1.222 m, legacy Full 1.084 m, Relation 1.117
  m. Relation compute remains below 1 ms but is 3.6× legacy Full.
- Added `docs/single_arm_final_report.md` with Wilson intervals, seed-balanced
  bootstrap results, phase paths, action/speed/compute, failure taxonomy, and
  explicit claim limits.

Validation:

- `conda run -n env_isaaclab python -m pytest -q` — 17 passed before freeze
- 480 independent Isaac workers completed; transient Isaac plugin-exit warnings
  produced no missing artifact or metric
- full matrix identity, count, fingerprint, path-partition, and hash audit passed

### Phase 6 — bimanual handover skeleton

Status: complete.

Audit findings and minimum plan:

- The existing `Essay2608-Bimanual-Handover-v0` already has two Frankas,
  independent absolute Cartesian IK actions, independent grippers, a complete
  13-state scripted expert, isolated-process collection, and five frozen v1
  demonstrations.
- The task observation manager exposes only joint state and prior action, so it
  does not yet satisfy the explicit geometric observation contract.
- Frozen v1 records a single kinematic `carrier` and cannot represent the short
  `both` interval. It must remain immutable; the corrected schema will be
  collected as `data/handover_static/v2`.
- Add explicit left/right EE, object, target, and measured finger observations;
  add state-aligned `none → left_only → both → right_only → none` supervision;
  keep the physical carrier field for backward compatibility.
- Unit-test the simulator-independent schema and v1 compatibility, run one
  isolated headless smoke episode, then collect and independently audit five v2
  episodes before freezing them.

Accepted results:

- Pure tests pass without starting Isaac; the handover schema exposes the exact
  `none → left_only → both → right_only → none` sequence and rejects a corrupted
  transfer label. Frozen v1 remains loadable through `legacy_carrier_only`.
- Headless seed 7300 completed in 575 steps at 10.62 mm final error. Runtime
  inspection confirmed the six required observation shapes and 16-D action.
- Formal v2 accepted five successes from eight isolated attempts at seeds 7400,
  7403, 7404, 7406, and 7407. Rejected workers contributed no data.
- All five trajectories contain states 0–12, all four labels, exactly 15 `both`
  steps, measured 0–40 mm gripper motion, continuous 20 ms timestamps, and no
  reset-like jump.
- Frozen hash:
  `91706df18abfea606c9e6836f1864e675610633ce5cb0c3c23846a1ea4f5fe18`.
  Maximum final error is 11.04 mm; minimum start separation is 13.91 mm.
- Collection and audit both refuse to overwrite v2 after `FROZEN`; the manifest
  remained unchanged. `DynaMAC.pdf`, single-arm v1, and handover v1 were not
  modified.
- `docs/bimanual_handover_setup.md` records the contract, commands, audit, and
  scientific limitations. The cube remains a gravity-disabled geometric
  attachment, so this is infrastructure rather than contact-rich evidence.

Validation:

- `conda run -n env_isaaclab python -m pytest -q` — 20 passed
- one independent headless smoke worker — success
- eight formal independent workers — five accepted, three rejected by the
  unchanged success gate
- pre-freeze and post-freeze full dataset audits — pass with identical digest
- frozen overwrite/refreeze counterexamples — both refused, manifest unchanged
