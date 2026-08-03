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
python scripts/eval_single_arm.py --headless --seeds 6200 6201 6202
```

Every evaluation rollout uses a fresh Isaac Lab process. The study includes:

- World Gaussian;
- static object/target Gaussian Product-of-Experts;
- mask-only multi-stream policy;
- Full DynaMAC with connection detection and a virtual EE frame;
- static, smooth/sudden object shift, smooth/sudden target shift, and arm-command offset conditions.

## Three-seed controlled result

Seeds `6200`, `6201`, and `6202` cover 72 isolated simulator rollouts (four methods by six conditions by three
seeds). The table averages each condition within a seed before computing the reported across-seed statistics.

| Method | Mean success | Mean recovery | Final error, mean [95% bootstrap CI] | Path length | Inference |
|---|---:|---:|---:|---:|---:|
| World Gaussian | 0.0% | 0.0% | 213.85 [79.59, 345.90] mm | 1.118 m | 0.025 ms |
| Static Multi-stream | 16.7% | 20.0% | 78.92 [61.58, 107.32] mm | 1.328 m | 0.210 ms |
| Mask-only | 100.0% | 100.0% | 59.12 [59.04, 59.24] mm | 1.233 m | 0.235 ms |
| Full DynaMAC | 100.0% | 100.0% | 59.12 [59.04, 59.23] mm | 1.129 m | 0.239 ms |

The static PoE assigns the object stream about 79% of the positional precision during transport even though the
object is already an effect of the robot motion. Dynamic masking removes that endogenous stream. Full DynaMAC
preserves the successes of mask-only and reduces mean path length by about 8.5% in the three-seed aggregate.

Three seeds are controlled pilot evidence, not a publication-scale sample. The summary JSON additionally contains
per-condition Wilson intervals, deterministic bootstrap intervals, failure causes, and individual rollout records.
Keep the frozen dataset unchanged when adding more seeds.

## Conditional diffusion baseline

The later-stage generative comparison is a compact low-dimensional conditional DDPM with an eight-action chunk,
32 denoising steps, and 177,984 parameters. It is intentionally not presented as a reproduction of the official
image-based Diffusion Policy architecture.

```bash
python scripts/train_diffusion.py
python scripts/eval_single_arm.py --headless \
  --methods diffusion_policy --seeds 6200 \
  --output_dir outputs/diffusion/v1/eval
```

In the one-seed pilot it completed four conditions but missed the 60 mm threshold (72.6--83.3 mm final error),
and failed to complete the two moving-target conditions. Overall success and recovery were 0/6, with 0.49 ms mean
inference. This negative result supports only the narrow observation that this compact DDPM did not match the
explicit geometric methods from five demonstrations; it is not evidence against a full visual Diffusion Policy.
