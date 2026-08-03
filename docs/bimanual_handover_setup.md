# Bimanual handover environment and dataset skeleton

## Scope

This phase validates only the environment, scripted expert, collection,
dataset, audit, and headless smoke chain for
`Essay2608-Bimanual-Handover-v0`. It does not claim a contact-rich handover or
a complete bimanual DynaMAC implementation. The older bimanual policy pilots in
the repository predate this audit and were not used to accept the v2 dataset.

The implementation source used for v2 collection is commit `5c4921e`. The
previous `data/handover_static/v1` remains byte-for-byte unchanged and continues
to load through the backward-compatible legacy carrier schema.

## Environment contract

The scene contains two independently rooted Franka Pandas, one transfer cube,
a table, and a fixed placement target. The 16-D action is:

| Slice | Meaning | Controller |
|---|---|---|
| 0:7 | left tool pose, `xyz + wxyz` | absolute DLS differential IK |
| 7 | left gripper | independent binary position command |
| 8:15 | right tool pose, `xyz + wxyz` | absolute DLS differential IK |
| 15 | right gripper | independent binary position command |

The policy observation group now exposes the required geometric state directly:

| Observation | Shape | Source |
|---|---:|---|
| `left_ee_pose` | 7 | measured left tool pose in local environment coordinates |
| `right_ee_pose` | 7 | measured right tool pose in local environment coordinates |
| `object_pose` | 7 | measured rigid-object root pose |
| `target_pose` | 7 | fixed placement target |
| `left_gripper_state` | 2 | measured left finger joint positions |
| `right_gripper_state` | 2 | measured right finger joint positions |
| `actions` | 16 | previous action for diagnostics |

A headless construction at seed 7300 confirmed these term names and shapes, all
four action terms, and one complete 575-step episode with 10.62 mm final error.

## Expert and relation supervision

The expert executes all 13 ordered states:

```text
REST → LEFT_APPROACH → LEFT_GRASP → LEFT_LIFT → LEFT_TO_HANDOVER
→ RIGHT_APPROACH → RIGHT_GRASP → TRANSFER → LEFT_RELEASE
→ RIGHT_TO_TARGET → RIGHT_RELEASE → RETREAT → COMPLETE
```

Every recorded step has a state-aligned `relation_label`:

| Label | Expert states | Meaning |
|---|---|---|
| `none` | 0–1 and 10–12 | no arm is treated as attached |
| `left_only` | 2–6 | left carries while right approaches and closes |
| `both` | 7 | confirmed short co-hold before left release |
| `right_only` | 8–9 | right carries to the target |

At 20 ms control time, `TRANSFER` lasts 15 recorded steps, or 0.30 s, in every
accepted v2 demonstration. The legacy integer `carrier` is retained separately
because it selects the one end effector used by the geometric attachment
emulator. It must not be interpreted as four-value relation ground truth.

## Isolated collection and frozen v2

`scripts/collect_handover.py` is the canonical thin entry point. Its controller
starts a fresh simulator process for every attempt, accepts only a complete
episode below the fixed 60 mm success threshold, and moves only successful NPZ
files into the requested output directory.

The one-demo smoke used seed 7300 and a separate output directory. Formal v2
collection began at seed 7400. Five successes were accepted from eight attempts:
7400, 7403, 7404, 7406, and 7407. Failed workers were rejected and contributed
no trajectory to the dataset.

Frozen dataset:

- path: `data/handover_static/v2`;
- demonstrations: 5, with 582–604 steps each;
- dataset SHA-256:
  `91706df18abfea606c9e6836f1864e675610633ce5cb0c3c23846a1ea4f5fe18`;
- maximum final error: 11.04 mm;
- minimum pairwise initial-object distance: 13.91 mm;
- maximum per-step Cartesian jump: 29.21 mm;
- maximum left/right object-connection position RMS standard deviation:
  2.01/3.38 mm;
- relation schema: `four_value_state_aligned_v2` in all five files;
- both finger measurements cover approximately 0–40 mm in all files.

The audit checks required arrays, finite values, 16-D actions, 7-D poses,
continuous timestamps, complete state and relation sequences, label/state
agreement, measured open/closed grippers, reset-like jumps, distinct starts,
final error, per-file SHA-256, and connection stability. Both collection and
freezing refuse a directory containing `FROZEN`; the refusal test returned
nonzero twice and left the manifest hash unchanged.

## Reproduction

One isolated smoke demonstration:

```bash
conda run -n env_isaaclab python scripts/collect_handover.py --headless \
  --num_demos 1 --max_attempts 3 --seed 7300 \
  --output_dir outputs/handover_scientific/smoke_v2
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir outputs/handover_scientific/smoke_v2
```

Collect a new, unfrozen five-demo version. Never reuse `v1` or `v2` as the
output directory:

```bash
conda run -n env_isaaclab python scripts/collect_handover.py --headless \
  --num_demos 5 --max_attempts 10 --seed 7500 \
  --output_dir data/handover_static/v3
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir data/handover_static/v3
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir data/handover_static/v3 --freeze \
  --dataset_version handover_static_v3
```

## Scientific limits

The cube has gravity disabled and is written to a single carrier tool pose; the
short `both` interval is scripted supervision rather than a force/contact
measurement. Success is geometric completion and final position error, not a
validated physical grasp-stability metric. The five demonstrations establish a
reproducible data and supervision interface for later work, but provide no
evidence that an existing bimanual learned policy generalizes or outperforms a
baseline.

Before policy research, the next environment upgrade should add contact sensing
and a non-kinematic held object, then compare scripted labels against observed
two-arm contact/relative-motion evidence.
