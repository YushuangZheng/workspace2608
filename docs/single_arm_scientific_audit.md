# Single-arm scientific audit

This audit separates observations from claims. It uses the frozen five-demo
dataset and the 72 rollout traces at commit `57e01c4`; it does not modify either
input. Phase-level results are written to the new directory
`outputs/single_arm_scientific/audit_v1/phase_diagnostics`.

## Target and support geometry

The task inherits Isaac Lab's cube-lift command, whose target is a desired
**object pose**. The project fixes that object-center command at `z = 0.08 m`.
The scaled DexCube settles on the table with its measured center at about
`z = 0.021 m`. Therefore a stable table placement at the commanded XY location
has an unavoidable vertical residual of about:

```text
0.080 m - 0.021 m = 0.059 m
```

The legacy 3-D error was consequently measuring the mismatch between a desired
object-center height and the stable support surface, rather than placement
quality alone. It should remain available for historical comparison but must not
be the paper-facing success definition.

## Parallel success indicators

Future rollouts explicitly store three related but distinct indicators:

1. `final_error_3d_m` and `legacy_success_3d` at the historical 60 mm radius;
2. `final_xy_error_m`, with 5/10/20 mm sensitivity;
3. `stable_place_success`, requiring the primary XY threshold, demonstrated
   support height, an open gripper, policy completion, and low final displacement
   and speed over 25 control steps.

Offline reconciliation of the existing strict traces gives:

| Method | Legacy 3-D success | Stable-place success | Mean 3-D error | Mean XY error |
|---|---:|---:|---:|---:|
| World Gaussian | 0/18 | 0/18 | 213.85 mm | 200.48 mm |
| Static Multi-stream | 0/18 | 0/18 | 73.64 mm | 42.46 mm |
| Mask-only | 18/18 | 18/18 | 59.09 mm | 3.01 mm |
| Full online prototype | 18/18 | 18/18 | 59.08 mm | 2.98 mm |

All 72 trials reached the demonstrated support height, released the object, and
were stable at the end. In this matrix, failure is therefore explained by XY
placement rather than the fixed height residual. Mask-only and Full remain 17/18
at 5 mm and 18/18 at 10/20 mm, so their strict successes are not 60 mm edge
artifacts.

## Additive phase attribution

Every inter-step EE displacement is assigned to its destination phase. The sum
over phases reproduces every saved rollout path within
`2.3e-16 m`, so the attribution is numerically complete.

Across 18 paired Mask-only/Full trials, Full minus Mask is:

| Phase | Mean path difference | Mean step difference |
|---|---:|---:|
| Lift object | -77.80 mm | -6.39 |
| Move above target | -24.99 mm | -1.11 |
| Lower to target | -1.75 mm | 0.00 |
| All other phases combined | -0.06 mm | 0.00 |
| Total | -104.60 mm | -7.50 |

The ratio of aggregate mean paths is an 8.5% reduction; averaging each paired
percentage gives 8.0%. Full is shorter in only 10/18 paired trials. It is longer
in all six seed-6202 conditions, so virtual-frame efficiency is not a stable
per-seed conclusion.

Total step difference and total path difference have correlation `r = 0.86`.
There are no forced phase transitions in either method. Pre-grasp phases are
identical, and essentially all change starts in phase 4, the only phase where the
online prototype activates the virtual EE frame. The most defensible explanation
is therefore:

- the virtual-frame command changes phase-4 tracking and reach timing;
- that changes both phase-4 path/duration and the phase-5 starting state;
- phase 5 can either save path or add path, producing the seed-6202 reversal.

This is a coupled observational attribution, not proof that the frame alone is
causal. A paper-faithful skill baseline and a timing-controlled ablation are still
needed before claiming a general virtual-frame efficiency benefit.

## Action discontinuity

For Full/seed 6202, the largest raw phase-4 to phase-5 desired-position change is
about 406 mm. The shared limiter bounds the policy command change to 20 mm. The
largest post-perturbation jump is about 80 mm in `arm_offset`, where a 60 mm test
offset is deliberately injected after policy limiting; maximum measured EE speed
is about 1.01 m/s. This fixes the command discontinuity but is not a hardware
safety certification, because orientation rate, acceleration, force, and robot
limits are not certified here.

## Cache and claim boundaries

Trial reuse requires an exact fingerprint over the Git commit, relevant source
contents, frozen dataset SHA, method, condition, seed, task, rollout length,
success criteria, action-rate limit, perturbation source, and generative
checkpoint when used. A mismatch invalidates the cache.

The three evaluation seeds share one frozen training set. They measure test-time
robustness for that dataset, not the variance of independently sampled five-demo
training sets and not arbitrary 5-shot sample efficiency.

Reproduce the offline audit with:

```bash
python scripts/analyze_phase_diagnostics.py \
  --result_dir outputs/single_arm_strict/v2 \
  --output_dir outputs/single_arm_scientific/audit_v1/phase_diagnostics
```
