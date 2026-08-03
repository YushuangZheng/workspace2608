# Velocity-based skill segmentation analysis

## Scope

This is a diagnostic-only segmentation of the five frozen
`pick_place_static/v1` demonstrations. It does not replace the expert's 0–9
states in policy fitting, and it is not a TAPAS reproduction. Its purpose is to
test whether end-effector velocity alone exposes repeatable coarse temporal
structure before adding semantic, gripper, contact, or visual signals.

The diagnostic computes end-effector linear speed and quaternion angular speed,
smooths both over 0.10 s, and calibrates one shared pair of thresholds as the
40th percentiles over all five frozen training demonstrations. A sample is low
speed only when both signals are below threshold. Low-speed runs shorter than
0.12 s are removed and gaps of at most 0.08 s are merged. Interior low-speed
runs yield their center as a transition candidate; endpoint runs yield their
inner edge. Candidate times are aligned across demonstrations by reference-free
clustering within 0.05 normalized trajectory time.

All durations and the quantile are predeclared diagnostic choices. They were
not adjusted against held-out simulator seeds or task success.

## Results

The shared thresholds are 0.021813 m/s linear speed and 0.017696 rad/s angular
speed. Every demonstration produces four persistent low-speed regions and four
candidate boundaries, hence five automatic segments rather than the ten expert
controller states.

| Candidate | demo 000 | demo 001 | demo 002 | demo 003 | demo 004 | aligned mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| initial-rest exit | 0.48 s | 0.48 s | 0.48 s | 0.48 s | 0.48 s | 0.480 ± 0.000 s |
| grasp dwell center | 2.22 s | 2.24 s | 2.26 s | 2.14 s | 2.18 s | 2.208 ± 0.043 s |
| release dwell center | 4.32 s | 4.34 s | 4.18 s | 4.26 s | 4.20 s | 4.260 ± 0.063 s |
| final-rest entry | 5.20 s | 5.22 s | 5.10 s | 5.16 s | 5.10 s | 5.156 ± 0.050 s |

All four normalized-time clusters contain all five demonstrations. Automatic
segment count has mean 5.0 and standard deviation 0.0; the mean time standard
deviation across aligned candidates is 39 ms. The persistent interior intervals
fall entirely in manual state 3 (grasp) and state 7 (release), while endpoint
intervals fall in states 0 and 9.

The nearest manual state-transition deviation is 211 ms on average and 300 ms
at worst. This is not a simple timing error: manual boundaries label entry and
exit from the scripted dwell states, whereas this detector intentionally places
one event boundary at each dwell center. The visual comparison therefore keeps
both sets of lines instead of treating the ten state transitions as ground-truth
velocity boundaries.

## Interpretation

Velocity alone consistently reduces ten low-level controller states to five
temporal regions. Ignoring initial and final idle regions, these support three
coarse action macros: approach-and-grasp, transport-and-place, and retreat. The
grasp and release dwell regions are highly repeatable transition events, but
they are not recovered as separate semantic skills because zero velocity cannot
tell closing from opening or intentional holding from rest.

The result supports using these candidates as initialization or a prior for a
future segmenter. It does not justify replacing the phase labels yet. A useful
next segmentation model should add actual gripper position/velocity and contact
or object-motion changes, then test whether grasp and place emerge separately.

## Difference from TAPAS

The supplied DynaMAC paper delegates both automatic skill segmentation and
task-parameter selection to TAPAS. This diagnostic has no learned visual
representation, keypoints, object relevance, task-parameter precision, or model
selection. It applies one global velocity rule to already-clean robot poses and
does not fit a policy per discovered segment. Agreement across five scripted
demonstrations establishes repeatability only for this dataset, not equivalence
to TAPAS or transfer to a new task.

## Reproduction

```bash
conda run -n env_isaaclab python scripts/analyze_segmentation.py \
  --data_dir data/pick_place_static/v1 \
  --output_dir outputs/single_arm_scientific/segmentation_v1
```

The output contains `analysis.json`, `velocity_boundaries.png`, and
`boundary_alignment.png`. The JSON stores the immutable dataset hash, every
parameter, per-demo interval and boundary, alignment membership, source content
hash, Git revision, and an analysis fingerprint.
