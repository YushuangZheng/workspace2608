# Failure-detection and recovery baselines

This directory isolates six official upstream projects from the DynaMAC
implementation. DynaMAC source, models, data, and existing evaluations are not
modified by baseline setup or execution.

Each method directory contains tracked provenance and reproduction notes. Its
official checkout lives in ignored `upstream/`; datasets, checkpoints, runtime
state, and generated results are also ignored. Conda environments live under
`/data/yukun/miniconda3/envs/` rather than inside this repository.

Status snapshot: 2026-08-20 (Asia/Shanghai). Update the two queued rows after
their formal evaluations finish.

| Priority | Method | First reproducible target | Status | Boundary |
|---:|---|---|---|---|
| 1 | Guardian/FailCoT | official thinking checkpoint, five Table II OOD cells | complete: 820-sample Table II checkpoint evaluation | checkpoint evaluation; no retraining first |
| 2 | SPR | official `libero_10` checkpoint, one task then the full suite | task 0 running; tasks 1-9 queued after RACER | released-code evaluator; full training data/code are unavailable |
| 3 | RACER | official checkpoint, three related RLBench tasks then all 18 | queued after SPR task 0 | one released actor cannot reproduce five-seed standard deviations |
| 4 | FAIL-Detect | environment/data smoke, then Transport + flow matching | environment and model smoke complete; training/evaluation blocked on unreleased data/checkpoints | no checkpoint; released code and paper protocol differ |
| 5 | AgentChord | simulated nominal and recovery on/off rollouts | 29 source tests passed; simulation blocked on endpoint and secure DexSim runtime | requires an OpenAI-compatible GPT-5 endpoint; no paper batch protocol |
| 6 | AHA | FailGen smoke only | simulator reset and one FailGen episode complete; paper model evaluation blocked | final model/data/inference path unavailable; full training exceeds budget |

Paper values and local values are compared only within the same method,
benchmark, split, metric, and released protocol. Results from different robot
benchmarks are never presented as a direct ranking against DynaMAC.

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
