# ICLR 2027 formal evaluation workspace

This directory contains the paper-evaluation layer built after the phase-six
method implementation was frozen.  It is intentionally separate from
`evaluations/development/`, whose artifacts are development evidence rather
than paper results.

## Ownership and data boundary

- `interfaces/`, `configs/shared/`, `manifests/`, `runners/`, `audit/`,
  `recovery/`, and `analysis/` are owned by server A.
- Server B owns only its method adapters/model directories and native-system
  adapters.  It must not overwrite the shared runner or schema.
- `datasets/normal_calibration*` and every sealed manifest/result remain on A.
- B receives only the canonical interface/development files and audited
  `failure_train` files listed by the A2 SHA256 handoff indexes. No persistent
  delivery copy is kept under `artifacts/`. Formal thresholds are generated on A.
- `interfaces/failure_train.py` is the sole reference for M4 manifest selection
  and causal target alignment; B may wrap it for batching but must not redefine
  its labels or nested/LOFO subsets.

## Causal record separation

Every cycle file contains three sibling records: monitor-visible `feature`,
evaluator-only `audit`, and execution bookkeeping.  A `RuntimeMonitor` receives
only the feature record.  Fault family, trigger schedule, physical-event labels,
and future observations are never monitor inputs.

## Main A2 commands

```bash
python -m evaluations.iclr2027.manifests.build
python -m evaluations.iclr2027.runners.launch --manifest evaluations/iclr2027/manifests/main10_development.jsonl --output-root evaluations/iclr2027/results/development --workers 48
```

One manifest episode is one queue job.  Each worker has an isolated DISPLAY,
temporary directory, simulator process, result shard, and CPU affinity.  The
queue is global and dynamic: a free worker immediately takes the next episode.

## A3 methods and calibration

- `methods/restart/`, `methods/trajectory_likelihood/`, and
  `methods/ours_monitor/` implement the A-owned M1, M2, and M6 monitors.
- `recovery/skill_retry.py` is the shared bounded retry executor used by
  M1/M2/M6 and the generic-retry ablation.
- `configs/methods/` is the authoritative registry for M0/M1/M2/M5/M6 and the
  three principal ablations.
- `calibration/boundary.py` builds normal-demonstration-only runtime boundary
  parameters. Its generated artifacts remain under `artifacts/calibration/`.
- `manifests/horizon3_{per_stage,single_event}.jsonl` contain the two frozen
  Horizon-3 protocols.

A3 acceptance is recorded in `results/a3_acceptance/`. The A2 handoff to B is
still marked not sent; before first delivery, any verified source correction
must be followed by rebuilding and re-auditing both SHA256 handoff indexes.
