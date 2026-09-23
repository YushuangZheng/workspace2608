# Task-State Feedback for Robust Multi-Stage Manipulation

This repository contains the reference implementation of **Task-State Feedback
(TSF)**, a framework for robust object-centric, multi-stage robot manipulation.
TSF constructs a structured model of valid task evolution from successful
demonstrations and uses online relation and progress inference to regulate
execution, repair interaction failures, and re-enter the task at a supported
state.

The method core is independent of a particular simulator. The included RLBench
integration supplies observations, robot execution, task assets, and evaluation
adapters around a reproduced DynaMAC motion backbone.

## Highlights

- Builds task-state support from successful demonstrations only.
- Couples object-relative motion, interaction relations, scene state, and task
  progress in one online inference loop.
- Supports guarded transitions, active relation verification, relation repair,
  and legal task re-entry.
- Uses the same abstractions for single-arm and bimanual manipulation.
- Separates the benchmark-independent method from simulator-specific execution.

## Repository layout

```text
configs/                         TSF and motion-policy configuration
data/                            small synthetic fixtures for smoke tests
source/
  data/                          demonstration data structures and validation
  policy/dynamac.py              object-centric multi-stream motion policy
  policy/tsf/                    benchmark-independent TSF implementation
integrations/rlbench/
  rlbench_dynamac/               RLBench motion-policy and controller adapters
  rlbench_tsf/                   RLBench adapter for TSF
  configs/                       simulator and task configuration
  requirements/                  offline and simulator dependency sets
evaluations/                     benchmark protocols and analysis code
tests/tsf/                       TSF core unit and integration tests
scripts/run.py                   lightweight command-line entry point
```

For a code tour, start at `source/policy/tsf/policy.py`, then follow the
`model/`, `inference/`, `control/`, and `recovery/` packages. Simulator process
boundaries and observation conversion live under
`integrations/rlbench/rlbench_tsf/`.

## Installation

### Core package

The benchmark-independent package requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,midigap]'
```

Run the self-contained structural example:

```bash
python scripts/run.py --config configs/dynamac_smoke.json \
  verify --data data/dynamac_demos.npz
```

This command fits the bundled synthetic single-arm and bimanual examples and
prints model summaries. It is a software smoke test and does not require
RLBench.

### RLBench integration

The simulator runtime uses the pinned RLBench, PyRep, TAPAS, CoppeliaSim, and
TRAC-IK versions listed in
[`integrations/rlbench/THIRD_PARTY.md`](integrations/rlbench/THIRD_PARTY.md).
Policy fitting uses Python 3.10; the pinned simulator stack uses Python 3.8.
Follow the setup instructions in
[`integrations/rlbench/README.md`](integrations/rlbench/README.md).

The bounded TRAC-IK fallback can be built with:

```bash
bash integrations/rlbench/build_pytracik_bounded.sh
```

Build a TSF sidecar from five demonstrations without modifying the motion
checkpoint:

```bash
python3.10 -m integrations.rlbench.rlbench_tsf.build_models \
  --task place_cups \
  --data-root integrations/rlbench/data/training/main \
  --base-models integrations/rlbench/models/dynamac_backbone \
  --output integrations/rlbench/models/tsf
```

Model builders refuse to overwrite an existing output directory.

## Evaluation

Benchmark runners, physical-event auditing, paired manifests, and deterministic
analysis live under `evaluations/`. See
[`evaluations/README.md`](evaluations/README.md) for the entry points and data
contracts. Raw rollouts are not required to import or test the core package.

## Tests

Run the benchmark-independent suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  pytest -q -p no:cacheprovider tests/tsf -k 'not rlbench_replay'
```

Run the RLBench adapter tests after installing the simulator dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  pytest -q -p no:cacheprovider integrations/rlbench/tests
```

Static checks use Ruff:

```bash
ruff check --no-cache source tests evaluations integrations
```

## Assets

Large demonstrations, model weights, simulator binaries, and rollout videos
are not part of the source checkout. Provide only the assets required by the
workflow being run and verify the applicable third-party licenses before
redistribution. The expected RLBench runtime layout is documented in
[`integrations/rlbench/README.md`](integrations/rlbench/README.md).

## Citation

The archival citation will be added when the paper record is public. Until
then, please cite the accompanying anonymous manuscript.

## License

A project license has not yet been selected. Add the intended code license
before publishing the repository. Third-party components and assets remain
subject to their respective upstream licenses.
