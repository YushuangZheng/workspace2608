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

Status: in progress.

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
