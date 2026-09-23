# FAIL-Detect baseline

This package adapts the published FAIL-Detect logpZO monitor to the common
causal monitoring interface used by the evaluation suite. It contains feature
preprocessing, score-model loading, functional conformal calibration, runtime
alarm logic, and an optional reproduction of the public Square-task pipeline.

## Runtime interface

`runtime.CanonicalFailDetectMonitor` consumes one public pre-action feature
record at a time. The encoding includes robot pose and gripper state, task
state, action, policy reference state, stream summaries, and previous-action
resolution. It excludes evaluator-only fault metadata, audit labels, episode
outcomes, and future observations.

The runtime binds each scorer to:

- the task and public feature schema;
- the exact preprocessing layout and normalizer;
- scorer, checkpoint, and configuration digests; and
- a normal-rollout conformal threshold schedule.

Alarm persistence is applied after scoring. The monitor cannot modify actions
or invoke recovery; the shared evaluator connects its alarms to the configured
execution response.

## Controlled-task scorer

`demo_features.py` exports causal vectors from the successful demonstrations
used by the motion policy. `training.py` fits one task-specific velocity model
without reading failure trajectories, fault identities, or retained outcomes.
Normal-rollout calibration is stored separately under the ignored runtime
artifact tree.

The controlled-task checkpoints use the published logpZO score and
time-varying conformal-alarm construction while matching the low-dimensional
observation interface shared by the controlled methods. They are distinct from
the visual-policy checkpoint used by the public Square example.

## Optional upstream reproduction

The `reproduction/` package preserves an independent check of the upstream
Square-task implementation at commit
`b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`. It covers dataset conversion,
policy and score-model training, feature export, conformal scoring, and
checkpoint parity. Source and environment identities are recorded in
`OFFICIAL_SOURCES.json` and the YAML files in `reproduction/`.

The pinned upstream environment contains dependencies that are no longer
jointly resolvable from current package channels. The supplied compatibility
environment records the tested substitutions; it is for reproduction only and
is not required for the controlled evaluation runtime.

## Verification

Run the method tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  pytest -q -p no:cacheprovider \
  evaluations/iclr2027/methods/fail_detect/tests
```

Validate the controlled-task scorer and the adapter contract with:

```bash
python -m evaluations.iclr2027.methods.fail_detect.tools.validate_main10
python -m evaluations.iclr2027.methods.fail_detect.tools.validate_adapter
```

Generated checkpoints, calibration artifacts, logs, and golden outputs are
stored under the ignored `evaluations/iclr2027/artifacts/` tree.
