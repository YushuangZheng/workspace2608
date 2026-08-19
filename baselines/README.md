# Failure-detection and recovery baselines

This directory isolates six official upstream projects from the DynaMAC
implementation. DynaMAC source, models, data, and existing evaluations are not
modified by baseline setup or execution.

Each method directory contains tracked provenance and reproduction notes. Its
official checkout lives in ignored `upstream/`; datasets, checkpoints, runtime
state, and generated results are also ignored. Conda environments live under
`/data/yukun/miniconda3/envs/` rather than inside this repository.

| Priority | Method | First reproducible target | Boundary |
|---:|---|---|---|
| 1 | Guardian/FailCoT | official thinking checkpoint, five Table II OOD cells | checkpoint evaluation; no retraining first |
| 2 | SPR | official `libero_10` checkpoint, one task then the full suite | released-code evaluator; full training data/code are unavailable |
| 3 | RACER | official checkpoint, three related RLBench tasks then all 18 | one released actor cannot reproduce five-seed standard deviations |
| 4 | FAIL-Detect | environment/data smoke, then Transport + flow matching | no checkpoint; released code and paper protocol differ |
| 5 | AgentChord | simulated nominal and recovery on/off rollouts | requires an OpenAI-compatible GPT-5 endpoint; no paper batch protocol |
| 6 | AHA | FailGen smoke only | final model/data/inference path unavailable; full training exceeds budget |

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
