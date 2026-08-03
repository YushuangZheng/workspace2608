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

## Minimal one-seed debug result

The first end-to-end engineering run used seed `6200`. It is a debugging result, not a paper-level statistical claim.

| Method | Successes / 6 | Mean final error | Mean path length | Mean inference |
|---|---:|---:|---:|---:|
| World Gaussian | 0 / 6 | 79.59 mm | 1.118 m | 0.024 ms |
| Static Multi-stream | 0 / 6 | 107.32 mm | 1.548 m | 0.210 ms |
| Mask-only | 6 / 6 | 59.24 mm | 1.333 m | 0.234 ms |
| Full DynaMAC | 6 / 6 | 59.23 mm | 1.039 m | 0.236 ms |

The static PoE assigns the object stream about 79% of the positional precision during transport even though the
object is already an effect of the robot motion. Dynamic masking removes that endogenous stream. The Full DynaMAC
virtual frame preserves the six successes of mask-only while reducing path length by 18%--25% across the six test
conditions in this seed.

Before using these numbers in a paper, run at least three evaluation seeds, report confidence intervals, inspect all
failure traces, and keep the frozen dataset unchanged.
