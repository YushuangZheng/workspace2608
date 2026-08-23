# Failure-detection and recovery baselines

This directory isolates six official upstream projects from the DynaMAC
implementation. DynaMAC source, models, data, and existing evaluations are not
modified by baseline setup or execution.

Each method directory contains tracked provenance and reproduction notes. Its
official checkout lives in ignored `upstream/`; datasets, checkpoints, runtime
state, and generated results are also ignored. Conda environments live under
`/data/yukun/miniconda3/envs/` rather than inside this repository.

Status snapshot: 2026-08-23 (Asia/Shanghai). A component smoke is not counted
as a reproduced paper result; incomplete evaluations report no success rate.

| Priority | Method | First reproducible target | Status | Boundary |
|---:|---|---|---|---|
| 1 | Guardian/FailCoT | official thinking checkpoint, five Table II OOD cells | complete: all 18 available thinking/vanilla checkpoint cells, 7,640 predictions | released-checkpoint evaluation; no retraining |
| 2 | SPR | official `libero_10` checkpoint, one task then the full suite | paused: tasks 0-7 complete at 323/400 (80.75%); task 8 hit CUDA OOM before episode 0; task 9 not started | no ten-task aggregate or direct comparison with the paper's 82.8% aggregate |
| 3 | RACER | official checkpoint, three related RLBench tasks then all 18 | paused before the new EGL single-episode gate; 0/75 quantitative episodes | NVIDIA EGL and isolated-process gates are merged; no local success rate yet |
| 4 | FAIL-Detect | environment/data smoke, then Transport + flow matching | paused at the resource gate before artifact download, training, or evaluation | bounded external-DP-checkpoint protocol is merged; no local metric yet |
| 5 | AgentChord | simulated nominal and recovery on/off rollouts | 29 source tests passed; simulation blocked on endpoint and secure DexSim runtime | requires an OpenAI-compatible GPT-5 endpoint; no paper batch protocol |
| 6 | AHA | FailGen smoke only | one earlier smoke complete; new 10-task x 1-episode queue paused before task 0 | FailGen engineering coverage only; final model/data/inference path unavailable |

Paper values and local values are compared only within the same method,
benchmark, split, metric, and released protocol. Results from different robot
benchmarks are never presented as a direct ranking against DynaMAC.

The exact pause state, completed metrics, raw artifact locations, and safe
resume boundary are recorded in [PAUSE_CHECKPOINT_20260823.md](PAUSE_CHECKPOINT_20260823.md).

## Local layout

```text
baselines/<method>/
├── README.md       # scope and known reproduction boundary
├── manifest.json   # immutable upstream and paper provenance
├── upstream/       # ignored official checkout
├── datasets/       # ignored downloaded data
├── checkpoints/    # ignored released weights
└── results/        # ignored raw and derived results
```
