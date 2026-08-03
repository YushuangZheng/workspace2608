# Method provenance and implementation boundary

This document separates claims supported by the supplied DynaMAC paper from
paper-inspired simplifications and new project mechanisms. The local source of
record is the unmodified `DynaMAC.pdf`, titled *One Hand Watches The Other:
Dynamic Multi-Agent Cooperation for Sample-Efficient Bimanual Manipulation in
Dynamic Environments* by von Hartz, Valada, and Boedecker.

## Provenance map

| Component | Paper method | This repository | Classification |
|---|---|---|---|
| Local streams | Dense 6-D end-effector poses on `R3 x S3`, transformed into task-parameter frames; Riemannian Gaussian marginals (Sec. III-A, Eq. 1–2) | Phase-normalized Euclidean Gaussian positions in world, object, and target frames; orientation is taken from the world model | Paper-inspired simplification |
| Fusion | Conditional-independent Gaussian product of experts (Eq. 3) | Translational Gaussian product of experts with rotated 3-D covariance | Paper-inspired simplification |
| Base learner | MiDiGaP paired with TAPAS | Small phase-aligned Gaussian learner written in this repository | Project implementation, not a MiDiGaP reproduction |
| Skill segmentation | TAPAS automatically segments the demonstrations (Algorithm 1, line 1) | Scripted expert phases 0–9 are provisional fixed skill labels | Placeholder simplification |
| Link statistic | Within-skill 6-D precision, `M = |det Lambda|^(-1/(2d))`, `d=6` (Eq. 5) | The same determinant scale is computed from a regularized Euclidean position-plus-rotation-vector covariance | Paper-faithful equation with a simplified pose geometry |
| Link filtering | Remove linked dynamic frames; short-lived precision peaks may be ignored or temporally filtered (Sec. III-B and footnote 1) | An object is removed for a whole skill when at least half its normalized bins have `M < 0.001` | Paper-inspired fixed-skill temporal rule |
| Virtual frames | Capture a static end-effector frame at every skill boundary and retain its history (Algorithm 1, line 5) | `virtual_skill_k` is captured when phase `k` starts; all frames up to the current skill are candidates | Direct structural reproduction with the simplified learner |
| Task-parameter selection | Normalize precision determinants across candidates at each time, take the time maximum, and threshold `omega` (Eq. 6) | Same normalization/max structure over 6-D covariances; fixed threshold 0.2 | Paper-faithful equation; threshold is a predeclared project choice because the paper does not specify it |
| Execution | Fit and sequence one multi-stream policy per skill (Algorithm 1, lines 7–8) | One phase-clock controller uses a fixed selected frame set for each expert-labelled phase | Paper-inspired simplification |
| Online mask | The paper derives links from demonstration skill distributions before fitting policies | `OnlineDynaMACPrototype` uses a runtime sliding window over object/EE translation and gripper state | New project prototype, not the paper algorithm |
| Bidirectional relation | The paper permits per-skill links to start/end but does not define this runtime state machine | `OnlineRelationEstimator` uses four hysteretic states, actual finger feedback, 6-D relative motion, co-motion correlation, and continuous confidence | New essay2608 mechanism |
| Gripper gating and latching | Not specified as the primary link test; links are inferred per skill from precision | The online prototype clears on an open demonstrated gripper command and otherwise latches; SkillDynaMAC has no runtime gripper gate | New/incomplete project logic |
| Frame activation | Selected task parameters are fixed for each learned skill (Algorithm 1) | SkillDynaMAC uses fixed training-only selections; the online prototype switches frames from runtime connection plus hardcoded phase 4 | First is paper-inspired; second is project-specific |
| Release handling | Per-skill analysis permits links to end after placement (Fig. 3) | Current online prototype latches until an open-gripper command; `SkillDynaMACPolicy` changes only at phase boundaries | Incomplete project behavior |
| Bimanual reduction | Fit two concurrent DynaMAC instances and add the opposite end effector as a candidate task parameter (Sec. III-C) | Existing bimanual controllers use handcrafted coordination and transfer logic | Paper-inspired engineering prototype, not a reproduction |
| Perturbations | DynaBench changes valid task configurations and tests smooth/abrupt dynamics (Sec. IV) | Deterministic 8 cm object shifts, 10 cm target shifts, and a temporary 6 cm action offset in a custom Isaac Lab task | Project evaluation mechanism inspired by the paper, not DynaBench |
| Metrics | The paper reports task success over 200 episodes and dynamic/static task groups | Composite stable placement, XY sensitivity, legacy 3-D error, recovery, path/action diagnostics, and connection decisions against scripted phases | Project metrics; no numerical equivalence to paper results |

## Three names with intentionally different meanings

`SkillDynaMACPolicy` is the paper-faithful simplified baseline. It implements
the ordering in Algorithm 1—label skills, identify linked object frames, add one
virtual end-effector frame at each boundary, select frames with an Eq. (6)-style
score, and execute fixed per-skill policies. It does not contain an online
detector.

`OnlineDynaMACPrototype` is the earlier project mechanism. It estimates a
translation-only connection online, masks the object stream after detection,
and creates one virtual frame at phase 4. The compatibility alias
`DynaMACPolicy` still resolves to this class so old scripts and saved method
labels continue to work. Results named `full_dynamac` therefore refer to this
prototype, not to a faithful reproduction of the paper.

`MaskOnlyPolicy` isolates the runtime masking component and deliberately omits
the virtual frame. It is an ablation of the online project prototype.

## Exact simplified SkillDynaMAC procedure

The procedure consumes only the five frozen training demonstrations. It does
not inspect evaluation seeds or outcomes.

1. Use each scripted expert phase as one provisional skill and resample it to
   25 normalized bins.
2. Fit world, object, target, and ten virtual-skill Gaussian streams. Each model
   stores the original 3-D position covariance and a 6-D covariance formed from
   position plus a quaternion tangent residual represented as a rotation vector.
3. Apply Eq. (5) to the object stream with `tau_M = 0.001`. Mark the object as
   linked when at least 50% of skill bins fall below the threshold. The target
   is declared exogenous by the task ontology and is not a link candidate.
4. Remove a linked object, add all virtual frames through the current skill,
   compute the Eq. (6)-style normalized precision scores, and retain candidates
   whose maximum score exceeds 0.2. If numerical competition removes every
   frame, retain the single highest-scoring candidate.
5. During execution, capture the current end-effector pose on every phase
   transition and fuse the selected streams with the repository's translational
   product of experts. Use the world stream only for orientation and gripper.

The fitted artifact manifest records every candidate, Eq. (5) statistic, Eq.
(6) score, selection, and threshold. Evaluation fingerprints additionally
record policy, detector, phase-clock, perturbation, success, data, source, and
checkpoint configuration.

## Training-only diagnostic and known failure modes

With the frozen five-demonstration dataset, Eq. (5) marks the object linked in
phase 0 and phases 2–9, but not phase 1. This is not plausible physical ground
truth. Object orientation is almost constant, so the regularization-scale
rotational variances shrink the 6-D determinant even when translational relative
motion is not rigid. The paper explicitly permits a weighted precision matrix,
but this baseline leaves it unweighted to expose the mismatch instead of tuning
weights against test performance.

Eq. (6) also selects many historical virtual frames in phase 5. Its maximum over
time admits a frame that is dominant in only one normalized bin, and several
near-degenerate covariances compete sharply. These selections are deterministic
and inspectable, but they should not be interpreted as learned semantic frame
relevance.

The six-condition simulator smoke test (seed 6200) succeeded for static, smooth
object, sudden object, and temporary arm-offset conditions, but failed both
10 cm target-shift conditions with 67.83 mm final XY error. The static trial had
4.65 mm final XY error, 59.18 mm legacy 3-D error, 322 steps, and 1.037 m
end-effector path. The target failure is consistent with the target stream being
diluted by several selected static virtual streams. These are engineering smoke
results only. They neither validate the detected link labels nor support
paper-level performance claims.

The clean three-seed matrix at commit `c979a94` contains 90 unique trials (five
methods, six conditions, three seeds), each with paired JSON/NPZ evidence and
one common source/data fingerprint. SkillDynaMAC succeeded in 10/18 trials and
recovered in 8/15 perturbed trials. It succeeded in all object-shift trials,
failed all six target-shift trials, and had one additional static plus one arm
offset placement failure on seed 6201. Its condition-balanced mean XY error was
25.95 mm and mean path length 1.152 m. Comparing its fixed per-skill link labels
against the repository's scripted physical-link phases yields a 0.621 mean false
positive fraction and zero false-negative fraction; this comparison is an
offline-label diagnostic, not an online detector rate. Mask-only and the online
prototype succeeded in all 18 trials, whereas World and Static Multi-stream
succeeded in none. With only three held-out seeds, these results are diagnostic
and not a paper-ready estimate.

## Not implemented from the paper

- TAPAS segmentation and its learned visual task parameters;
- MiDiGaP or the paper's full Riemannian pose distributions and parallel
  covariance transport;
- perception from RGB-D/DINO features, clutter rejection, or occlusion tests;
- the paper's RLBench/DynaBench tasks, 200-episode protocol, or reported
  baselines;
- a validated position/rotation weighting matrix, calibration procedure, or
  confidence interval for link detection;
- two concurrent paper-faithful bimanual DynaMAC policies with opposite-arm
  candidate streams.

Claims in this repository must therefore use “paper-faithful simplified
baseline” for `SkillDynaMACPolicy`, “online project prototype” for
`OnlineDynaMACPrototype`, and avoid calling either a reproduction of the paper.
