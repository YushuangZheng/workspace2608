# Single-arm DynaMAC minimal loop

This stage is deliberately limited to the existing Franka pick-and-place task. It validates the dataset, geometric
phenomenon, Gaussian baselines, dynamic masking, virtual EE frame, and controlled test-time perturbations before any
bimanual environment is introduced.

## Frozen dataset

`data/pick_place_static/v1` is immutable input data. Its manifest contains per-file SHA-256 hashes, seeds, initial
poses, state coverage, step counts, final errors, reset-jump diagnostics, connection stability, and the source Git
commit. `scripts/collect_demos.py` refuses to use a directory containing the `FROZEN` marker as an output directory.

Verify it with:

```bash
python scripts/audit_dataset.py --data_dir data/pick_place_static/v1
```

## Reproduce the loop

```bash
python scripts/analyze_relative_frames.py
python scripts/train_single_arm.py
python scripts/eval_single_arm.py --headless --seeds 6200 6201 6202 \
  --output_dir outputs/single_arm_strict/v2
```

Every evaluation rollout uses a fresh Isaac Lab process. The study includes:

- World Gaussian;
- static object/target Gaussian Product-of-Experts;
- mask-only multi-stream policy;
- Full DynaMAC with connection detection and a virtual EE frame;
- static, smooth/sudden object shift, smooth/sudden target shift, and arm-command offset conditions.

Resume is content-addressed. A cached trial is reused only when its fingerprint matches the Git commit, frozen
dataset, relevant source files, method, perturbation, seed, rollout settings, success criteria, and generative
checkpoint.

## Semantic success and command continuity

The old 60 mm 3-D radius is retained as `legacy_success_3d` for comparison only. It was invalid as a primary
success criterion because the target
command uses `z = 0.08 m`, while a released cube rests with its center near `z = 0.021 m`; every correct placement
therefore carried an almost fixed 59 mm vertical residual. The primary success definition now requires:

- XY object-to-target error below 10 mm;
- object height within 10 mm of the support height measured from the frozen demonstrations;
- an open gripper;
- less than 5 mm displacement and 0.05 m/s maximum speed over the final 25 control steps.

The evaluator also reports composite success at 5, 10, and 20 mm XY thresholds. A common Cartesian rate limiter
bounds every policy command to 20 mm per control step. Raw policy intent, limited policy command, command after the
controlled perturbation, and frame-switch diagnostics are stored separately.

The additive phase audit in `docs/single_arm_scientific_audit.md` shows that Full's aggregate path reduction is
coupled to shorter phase-4/5 execution. Full is shorter in only 10/18 paired trials and longer in every seed-6202
condition, so the virtual frame does not support a seed-independent efficiency claim.

## Three-seed controlled result

Seeds `6200`, `6201`, and `6202` cover 72 isolated simulator rollouts (four methods by six conditions by three
seeds). The table averages each condition within a seed before computing the reported across-seed statistics.

| Method | Mean success | Mean recovery | XY error, mean [95% bootstrap CI] | Path length | Policy computation |
|---|---:|---:|---:|---:|---:|
| World Gaussian | 0.0% | 0.0% | 200.48 [53.11, 340.75] mm | 1.128 m | 0.039 ms |
| Static Multi-stream | 0.0% | 0.0% | 42.46 [29.96, 59.25] mm | 1.219 m | 0.233 ms |
| Mask-only | 100.0% | 100.0% | 3.01 [1.91, 3.57] mm | 1.226 m | 0.259 ms |
| Full DynaMAC | 100.0% | 100.0% | 2.98 [1.99, 3.73] mm | 1.122 m | 0.262 ms |

The static PoE assigns the object stream about 79% of the positional precision during transport even though the
object is already an effect of the robot motion. Dynamic masking removes that endogenous stream. Full DynaMAC
preserves the successes of mask-only and reduces mean path length by about 8.5% in the three-seed aggregate. This
path effect is seed-dependent: it reverses by about 3.6% for seed `6202`.

Mask-only and Full pass 17/18 trials at the stricter 5 mm XY threshold and 18/18 at both 10 and 20 mm. Thus their
primary successes are not threshold-edge artifacts. For Full on seed `6202`, the largest raw frame-transition request
remains about 406 mm, but the limited policy jump is 20 mm and maximum measured EE speed across all its trials is
about 1.01 m/s. The largest post-perturbation jump is about 80 mm because `arm_offset` intentionally injects a 60 mm
test disturbance after policy limiting.

Three seeds are controlled pilot evidence, not a publication-scale sample. They vary evaluation initialization while
all policies are fitted to the same frozen five-demo dataset. The experiment therefore measures test-time robustness
for that dataset, not variance across independently sampled five-demo training sets and not 5-shot sample efficiency.
The summary JSON additionally contains per-condition Wilson intervals, deterministic bootstrap intervals, semantic
failure causes, and individual rollout records.

Generate the seed-6202 transition audit and plots with:

```bash
python scripts/analyze_action_transitions.py \
  --result_dir outputs/single_arm_strict/v2
```

## Conditional diffusion baseline

The later-stage generative comparison is a compact low-dimensional conditional DDPM with an eight-action chunk,
32 denoising steps, and 177,984 parameters. It is intentionally not presented as a reproduction of the official
image-based Diffusion Policy architecture.

```bash
python scripts/train_diffusion.py
python scripts/eval_single_arm.py --headless \
  --methods diffusion_policy --seeds 6200 \
  --output_dir outputs/diffusion/v2/strict_eval
```

In the strict one-seed pilot it completed four conditions but had 42.4--58.9 mm XY error, and failed to complete the
two moving-target conditions. Overall success and recovery were 0/6. This fixed-dataset negative result supports only
the narrow observation that this compact DDPM did not match the explicit geometric methods from these five
demonstrations; it is not a sample-efficiency estimate or evidence against a full visual Diffusion Policy.
