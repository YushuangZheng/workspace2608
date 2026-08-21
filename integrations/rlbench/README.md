# DynaMAC on RLBench

This directory contains the current V4 RLBench reproduction. The formal release
evaluates 22 cells over 200 held-out episodes each. Static results are local
reproductions; dynamic-environment and arm-coordination results remain local
diagnostics because the paper does not publish the exact DynaBench protocol.

The normative execution and identity contract is [V4_PROTOCOL.md](V4_PROTOCOL.md).
Pinned external revisions and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Current artifact layout

| Artifact | Location | Contents |
|---|---|---|
| Training data | `data/training/` | 45 demonstrations in nine five-demo cohorts, plus the current SHA-256 manifest |
| Evaluation data | `data/evaluation/` | sealed `rlbench_eval_v2` spec, manifest, eight environment batches, and one coordination initialization |
| Models | `models/v4/` | eight main policies, the separate coordination HandOver policy, and `release_manifest.json` |
| Formal results | `results/v4/` | the current 22-cell JSON results and reports |
| Replay videos | `results/v4/replay_video/` | post-evaluation, outcome-stratified front/overhead replays |
| Consolidated report | `results/v4/reports/full_22_cell.md` | validated status and metrics for all 22 cells |

Training and evaluation data are disjoint. Evaluation artifacts contain no
policy outcomes, reports, or videos. Result and replay files never modify the
sealed evaluation set.

The integration code under `rlbench_dynamac/` is grouped by responsibility:

| Package | Responsibility |
|---|---|
| `core/` | shared runtime, controller, IK, task schemas, and atomic records |
| `data/` | demonstration adapters, collection, training, policy serving, and model release |
| `protocols/` | inherited and task-scoped intervention/semantic contracts |
| `eval/` | evaluation-set loading, evaluators, and the formal launcher |
| `report/` | the 22-cell matrix, reports, video selection, and post-evaluation replay |

`store_bottle_live_v4.py` intentionally remains at the package root because
that exact import path is already part of the authenticated StoreBottle task
identity. It is not an unclassified leftover.

The JSON files in `configs/` are executable protocol inputs, not generated
results. `tasks.json` defines low-dimensional task frames,
`tapas_segmentation.json` defines demonstration segmentation, and
`dynamac_rlbench_v3.json` is the policy configuration authenticated by the
current training manifest and checkpoints. The filename retains `v3` because
most V4 checkpoints inherit that exact fit identity. `v3_interventions.json`
and `v3_motion_sources.json` are likewise the frozen baseline still consumed
by current Table-I, HandOver, and SweepDust cells; task-scoped files under
`configs/v4/` override that baseline for StoreBottle, LiftTray, and
Coordination.

## Runtime setup

Policy fitting and serving use Python 3.10. RLBench, PyRep, CoppeliaSim, and
formal rollout use Python 3.8.

```bash
export DYNAMAC_POLICY_PYTHON=/path/to/python3.10
export DYNAMAC_SIM_PYTHON=/path/to/python3.8
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONPATH=/path/to/RLBench:/path/to/essay2608
```

Install the two environments from [requirements/](requirements/). Build the
pinned bounded TRAC-IK dependency once with:

```bash
bash integrations/rlbench/build_pytracik_bounded.sh
```

Formal preflight rejects an unpinned, unbounded, wrong-ABI, or ad-hoc build.

## Training data and model release

The normal training root contains all eight policy tasks:

```text
data/training/main/<task>/all_variations/episodes/episode{0..4}/
```

The separate dynamic HandOver cohort is stored at:

```text
data/training/coordination/bimanual_handover_item/all_variations/episodes/episode{0..4}/
```

`data/training/manifest.json` binds all 45 demonstrations and 125 episode files.
Training consumes `low_dim_obs.pkl`; variation files and the collection
manifests remain as provenance.

The current V4 release retrains two policies:

- StoreBottle uses five successful static demonstrations with seeds
  `4104000000..4104000004`. Its online and training frames are `bottle` and the
  physical `fridge_base` pose exposed as `fridge`.
- SweepDust uses the strict five-demo task-frame cohort in
  `data/training/main/bimanual_sweep_to_dustpan`; its augmentation manifest and
  five input hashes are stored beside the episodes.

The remaining six main policies and the separate coordination policy are the
authenticated byte-identical inherited checkpoints recorded by the V4 release
manifest. Validate the complete model inventory without training or copying:

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.data.store_bottle_v4 \
  release-manifest --dry-run --require-complete
```

The StoreBottle collector and trainer use the canonical training path by
default:

```bash
python3.8 -m integrations.rlbench.rlbench_dynamac.data.store_bottle_v4 collect --headless
python3.10 -m integrations.rlbench.rlbench_dynamac.data.store_bottle_v4 train
```

Fit any ordinary main-task policy by naming it explicitly; for example, the
current SweepDust cohort writes its checkpoint directly into the V4 model
directory with:

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.data.direct_policy train \
  --task bimanual_sweep_to_dustpan \
  --data-root integrations/rlbench/data/training/main \
  --models-dir integrations/rlbench/models/v4 \
  --config integrations/rlbench/configs/dynamac_rlbench_v3.json \
  --demonstrations 5
```

The separate coordination HandOver cohort is trained with:

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.eval.table_iii_coordination train \
  --data-root integrations/rlbench/data/training/coordination \
  --models-dir integrations/rlbench/models/v4/table_iii \
  --config integrations/rlbench/configs/dynamac_rlbench_v3.json \
  --demonstrations 5
```

Collection and training reject paths below `data/evaluation/` and `results/`.
Any checkpoint change requires regenerating and validating
`models/v4/release_manifest.json` before formal evaluation.

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.data.store_bottle_v4 \
  release-manifest --require-complete \
  --output integrations/rlbench/models/v4/release_manifest.json
```

## Formal 22-cell evaluation

Every cell uses evaluation set `rlbench_eval_v2`, seeds
`2608000000..2608000199`, horizon `1000`, and the V4 model release.

| Group | Cells | Count |
|---|---|---:|
| Table I | StackWine, PlaceCups, OpenMicrowave, WipeDesk × static/smooth/teleport | 12 |
| Table II | StoreBottle, HandOver, SweepDust, LiftTray × static | 4 |
| Table III | StoreBottle, HandOver, SweepDust, LiftTray × teleport; coordination hand-left/hand-right | 6 |
| Total |  | 22 |

The launcher defaults to eight reusable GPU/Xvfb lanes. `plan` is read-only,
`preflight` validates dependencies, data, models, and existing results, and
`execute` starts only cells that are not already valid.

```bash
python3.8 -m integrations.rlbench.rlbench_dynamac.eval.v4_formal_launch plan \
  --gpus 0,1,2,3,4,5,6,7

python3.8 -m integrations.rlbench.rlbench_dynamac.eval.v4_formal_launch preflight \
  --gpus 0,1,2,3,4,5,6,7

python3.8 -m integrations.rlbench.rlbench_dynamac.eval.v4_formal_launch execute \
  --gpus 0,1,2,3,4,5,6,7
```

Launcher state is written below
`results/v4/_launch/formal_22_cells/<run-id>/launch_summary.json`. Formal
evaluation writes result JSON only; it does not record process videos.

Result admission is cell-scoped. A cell must match its selected evaluation
batch, model, task semantics, controller, and intervention identity. Global
manifest/spec hashes remain provenance, but an unrelated batch change does not
invalidate a cell that did not read that batch.

## Post-evaluation replay videos

Replays are generated only after all 22 result files validate. The replay
launcher derives deterministic success/failure candidates from each completed
result, then records only the quota required by its observed success-rate tier.

| Observed success rate | Retained successes | Retained failures |
|---|---:|---:|
| at least 80%, and strictly within 2 percentage points of the paper reference | 0 | 0 |
| at least 80%, otherwise | 3 | 3 |
| 50% to below 80% | 5 | 10 |
| below 50% | 5 | 20 |

If a class contains fewer episodes than requested, all available episodes in
that class are used and quota is not transferred. Replays use front and
overhead views; the overhead view uses a 70-degree perspective while the front
view remains unchanged.

```bash
python3.8 -m integrations.rlbench.rlbench_dynamac.report.v4_post_evaluation_replays plan \
  --gpus 0,1,2,3,4,5,6,7

python3.8 -m integrations.rlbench.rlbench_dynamac.report.v4_post_evaluation_replays execute \
  --gpus 0,1,2,3,4,5,6,7 --overwrite
```

Replay state is written below
`results/v4/replay_video/_launch/post_evaluation/<run-id>/launch_summary.json`.

## Verification

The current report is
[`results/v4/reports/full_22_cell.md`](results/v4/reports/full_22_cell.md).
Run repository checks without creating cache files:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider integrations/rlbench/tests
ruff check --no-cache integrations/rlbench source
```
