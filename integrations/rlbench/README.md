# RLBench Integration

This package connects the benchmark-independent DynaMAC and Task-State
Feedback (TSF) policies to RLBench/CoppeliaSim. It provides observation
conversion, task specifications, low-level robot execution, model builders,
fault injection, and rollout interfaces.

## Package layout

```text
rlbench_dynamac/
  core/       task specifications, controllers, IK, and runtime records
  data/       demonstration conversion and motion-policy fitting
  eval/       motion-policy rollout interfaces
rlbench_tsf/  TSF observation, policy-server, and sidecar adapters
configs/      simulator, controller, and task configuration
requirements/ offline and simulator dependency sets
tests/        adapter and protocol regression tests
```

Task-state inference and recovery are implemented in `essay2608.policy.tsf`;
the integration converts RLBench observations and execution feedback to that
package's public interfaces.

## Supported software

Pinned revisions and license boundaries are listed in
[`THIRD_PARTY.md`](THIRD_PARTY.md).

| Component | Role |
|---|---|
| RLBench | tasks, demonstrations, observations, and success conditions |
| PyRep | Python interface to CoppeliaSim |
| CoppeliaSim Edu 4.1 | physics and rendering |
| TAPAS | trajectory segmentation and alignment |
| bounded TRAC-IK | endpoint IK fallback used by the shared executor |

Third-party source trees and simulator binaries are not vendored into the TSF
package.

## Environment setup

Offline model fitting uses Python 3.10. The pinned simulator stack uses Python
3.8.

```bash
python3.10 -m pip install -r integrations/rlbench/requirements/offline.txt
python3.8 -m pip install -r integrations/rlbench/requirements/simulator-py38.txt
```

Set the runtime paths for your installation:

```bash
export DYNAMAC_POLICY_PYTHON=/path/to/python3.10
export DYNAMAC_SIM_PYTHON=/path/to/python3.8
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONPATH=/path/to/RLBench:/path/to/task-state-feedback
```

Build the bounded TRAC-IK dependency once:

```bash
bash integrations/rlbench/build_pytracik_bounded.sh
```

## Runtime assets

Use a separate local directory for large runtime assets. The minimal layout is:

```text
integrations/rlbench/
├── data/
│   └── training/main/<task>/all_variations/episodes/episode{0..4}/
├── models/
│   ├── dynamac_backbone/<task>/
│   └── tsf/<task>/
└── results/
```

The source package does not require full training datasets, retained rollout
archives, or videos. A runnable distribution should include only the model and
example assets needed by its documented commands. Any redistributed RLBench,
CoppeliaSim, TAPAS, or model asset remains subject to its original license.

## Fitting a motion policy

Fit a DynaMAC policy from five demonstrations:

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.data.direct_policy train \
  --task place_cups \
  --data-root integrations/rlbench/data/training/main \
  --models-dir integrations/rlbench/models/dynamac_backbone \
  --demonstrations 5
```

The data loader rejects evaluation and result directories as training inputs.
Implementation details for the reproduced motion backbone are documented in
[`rlbench_dynamac/README.md`](rlbench_dynamac/README.md).

## Building TSF sidecars

TSF stores its task model beside, rather than inside, the frozen motion
checkpoint:

```bash
python3.10 -m integrations.rlbench.rlbench_tsf.build_models \
  --task place_cups \
  --data-root integrations/rlbench/data/training/main \
  --base-models integrations/rlbench/models/dynamac_backbone \
  --output integrations/rlbench/models/tsf \
  --demonstrations 5
```

The builder records the demonstrations, base checkpoint, task configuration,
and TSF configuration used for each sidecar and refuses to overwrite an
existing output directory.

## Policy server

The Python 3.8 simulator process communicates with the Python 3.10 TSF policy
through a transactional JSON-line server:

```bash
python3.10 -m integrations.rlbench.rlbench_tsf.policy_server serve \
  --task place_cups \
  --models-dir integrations/rlbench/models/tsf \
  --base-models-dir integrations/rlbench/models/dynamac_backbone
```

Each tentative action can be committed or aborted after physical execution, so
failed simulator actions do not silently advance the policy state. The shared
executor reports structured `reached`, `progressed`, and `stopped` outcomes to
the adapter.

## Verification

Run adapter tests after the simulator environment is configured:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  pytest -q -p no:cacheprovider integrations/rlbench/tests
```

Run static checks without creating repository caches:

```bash
ruff check --no-cache integrations/rlbench source
```

Do not commit local simulator paths, model weights, raw episode traces, replay
videos, native extensions, or credentials.
