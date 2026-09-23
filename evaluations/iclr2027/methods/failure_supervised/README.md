# Failure-supervised monitor

This package implements the causal GRU monitor used as a failure-supervised
comparison method. It contains the model, deterministic training-data views,
training and checkpoint code, normal-rollout calibration interface, and the
runtime adapter used by the shared evaluator.

The monitor consumes the same public, pre-action feature record as the other
controlled monitoring methods. Input features include arm pose and gripper
state, task state, action, policy reference state, stream summaries, and the
previous action result. Evaluator-only fields such as fault family, severity,
trigger time, audit labels, episode identity, and future observations are
excluded. Labels are aligned causally to the physical-event audit.

The default configuration uses a single-layer causal GRU with width 64,
AdamW, 100 epochs, full-episode batches of 16, right-padding masks, and
gradient clipping. The three configured random seeds share the same selected
episode identities. Normalization and class weights are fitted only from the
selected training split; alarm thresholds are calibrated separately from
nominal rollouts.

The production runtime entry point is
`runtime.CanonicalFailureSupervisedMonitor`. It implements the common
`RuntimeMonitor` interface, verifies task/config/checkpoint identities, and
has no authority to modify actions or invoke recovery. The shared evaluator
decides how an alarm is connected to an execution response.

From the repository root:

```bash
python -m evaluations.iclr2027.methods.failure_supervised.train_cli train \
  --task close_jar --budget 200 --seed 1103
python -m evaluations.iclr2027.methods.failure_supervised.queue --gpus 0,1,2,3
python -m pytest -q evaluations/iclr2027/methods/failure_supervised/tests
```

Generated tensors, checkpoints, calibration files, logs, and result records
are stored under the ignored `evaluations/iclr2027/artifacts/` and
`evaluations/iclr2027/results/` trees.
