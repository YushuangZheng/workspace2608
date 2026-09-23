# M4 failure-supervised monitor

This server-B method directory contains the causal classifier, training-only
views over A-frozen manifests, training loop, checkpoint I/O, and the runtime
adapter. It deliberately does not define task, fault, audit, recovery, runner,
or formal calibration behavior.

The frozen A2B interface and 2,000 training episodes were received and SHA-256
verified on 2026-09-05. `train_cli.py` uses A's canonical reader, including the
pre-step feature / previous-step audit label alignment. A-owned files are
read-only and are not duplicated or edited by this method.

The production entry point is `runtime.CanonicalFailureSupervisedMonitor`.
The older `adapter.FailureSupervisedMonitor` is a generic, array-only test API,
not the final frozen-feature entry point. The new runtime implements A's
`RuntimeMonitor` and accepts `EpisodeContext` with checked task/config/checkpoint
identities. `observe_record(feature)` consumes a complete public feature record;
the three-argument ABI accepts observation fields, an action mapping containing
`action` and `action_timestamp`, and the public `policy_state`. It never alters
actions or invokes simulator/recovery code. Missing calibration means scores
only, threshold null and no alarms. A supplies `threshold(cycle)` and owns formal
normal-calibration and sealed evaluation. Consecutive exceedance uses strict `>`.

M3 and M4 share `methods/fail_detect/preprocessing.py`. It encodes arm
pose/gripper, raw task state and action (preserving A's action order), policy
step/reference-state fields with presence indicators, stream counts and weight
summaries, and previous-action resolution. It does not encode string identities,
absolute episode/cycle IDs, audit fields, family, severity, trigger, labels, or
optional stream Gaussian parameters. Per-task geometry fixes vector dimensions.

The method config freezes a 64-wide single-layer causal GRU, AdamW, 100 epochs,
batch 16 full episodes with right-padding masks, gradient clipping, three seeds
1103/2207/3301, and persistence 3 before seeing evaluation outcomes. This is a
fixed training schedule, not an accuracy-selected checkpoint. Normalization and
class weights are fitted only to the selected budget/LOFO subset. Each seed uses
the identical manifest-selected episodes. No validation outcome chooses weights.
LOFO `training_budget=200` denotes the parent pool; `actual_training_episodes`
records the smaller retained pool. No resampling fills the removed family.

From repository root, in the locked PyTorch environment:

```bash
python -m evaluations.iclr2027.methods.failure_supervised.train_cli train --task close_jar --budget 200 --seed 1103
python -m evaluations.iclr2027.methods.failure_supervised.queue --gpus 0,1,2,3,4,5,6,7
python -m pytest -q evaluations/iclr2027/methods/failure_supervised/tests
```

The queue prepares each task once (derived train-only tensors under
`artifacts/training/m4/encoded/`, not a second raw dataset). It finishes all 30
Main-10 models before the 36 small-budget and 48 LOFO models. Queue progress is
`artifacts/training/m4/queue_20260909_current_executor/status.json`. Incomplete existing runs are
not silently overwritten or resumed with changed code/config. Each checkpoint
is CPU-reloaded and must reproduce the pre-save CPU parameters and predictions
exactly. GPU/CPU differences are reported separately (cuDNN and CPU float32
arithmetic are not bit-identical). Calibration, golden replay and formal
inference are frozen to CPU float32 to keep the working point on one backend;
the small GRU still trains on GPU. Checkpoints, selection hashes,
source hashes, exact config, logs, package versions, and manifests are retained.

Golden outputs use only A's development fixture. Its 18 records cover two tasks
and contain gaps, so the golden replay explicitly resets before each discontinuous
segment. These are **not** 20 full development rollouts or formal acceptance.
Tasks without fixture records are marked unavailable, not fabricated. A must
still perform its full development dry run and formal per-checkpoint calibration.

Method-scoped tests cover prefix causality, right-padding, deterministic budget
views, causal labels, leakage rejection, reset/gap behavior, identities, strict
persistence, training, and checkpoint round trips. Older copies under the shared
tests directory are left untouched; they are not part of B's new delivery.

## Frozen-checkpoint inference audit

`tests/verify_frozen_checkpoints.py` independently reselects every budget/LOFO
view using A's reader, recomputes its exact normalization, and replays the first
and last selected full episode through both batched and online CPU inference.
It checks episode resets and score-only behavior without fitting weights or
thresholds. The 2026-09-05 audit passed all 114 models / 228 full training
sequences; maximum probability difference was 1.61e-6. This is numerical
consistency validation, not a held-out performance measurement.

```bash
python -m evaluations.iclr2027.methods.failure_supervised.tests.verify_frozen_checkpoints
```

The separate report is `artifacts/training/m4/INFERENCE_VALIDATION.json`.
Per user instruction, no artifacts have been sent back to A.
