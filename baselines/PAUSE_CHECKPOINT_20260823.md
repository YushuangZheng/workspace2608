# Baseline reproduction pause checkpoint: 2026-08-23

This is the authoritative hand-off for the deliberate pause requested on
2026-08-23 (Asia/Shanghai). No reproduction job should be resumed merely by
checking out this revision.

## Git and runtime state

- The three implementation PRs were merged into `yukun`, never `master`:
  [#11](https://github.com/YushuangZheng/workspace2608/pull/11) at
  `bf31dabc8932ac49403c05f7bd1e2371e8e7e2de`,
  [#12](https://github.com/YushuangZheng/workspace2608/pull/12) at
  `6c0b5d042ee83c583234464fc7b9d10190506ef6`, and
  [#13](https://github.com/YushuangZheng/workspace2608/pull/13) at
  `55119035ca1116eb3ab121eed86dcdef246779ce`.
- The remote feature branches were deleted after merging. The intended remote
  heads are only `master`, `dev/yushuang`, and `yukun`.
- `master` and `dev/yushuang` were left untouched at
  `6ddd92ff0a09133e50c31f6ec301f703c67621e2`.
- All owned tmux sessions and reproduction processes were stopped. GPUs 0-5
  were idle at the stop audit. By the final verification, later root-owned
  Spark-0/Maniverse jobs had occupied GPUs 0-7; they are unrelated to this
  project and were not touched.
- The recovery tag is `yukun-reproduction-pause-20260823`.

The feature worktrees are intentionally retained because their ignored logs
and runtime state are not stored by Git:

- AHA: `/data/yukun/essay2608-wt-aha-failgen`
- FAIL-Detect: `/data/yukun/essay2608-wt-fail-detect-quant`
- RACER: `/data/yukun/worktrees/essay2608-racer-egl`

## Completed results

### Guardian / FailCoT

The released-checkpoint matrix is complete: 18 cells and 7,640 predictions.
The five original thinking OOD results live under ignored
`baselines/guardian_failcot/results/table_ii/guardian-thinking/`; the other 13
results live under ignored
`baselines/guardian_failcot/results/full_checkpoint_eval_20260821/`.

| Checkpoint | Split | Mode | Correct | Total | Accuracy |
|---|---|---|---:|---:|---:|
| thinking | RoboFail | execution | 129 | 153 | 84.31% |
| thinking | RoboFail | planning | 20 | 30 | 66.67% |
| thinking | UR5-Fail | execution | 108 | 140 | 77.14% |
| thinking | UR5-Fail | planning | 123 | 140 | 87.86% |
| thinking | RoboVQA | execution | 291 | 357 | 81.51% |
| thinking | BDV2-Fail | execution | 847 | 1,000 | 84.70% |
| thinking | RLBench-Fail | execution | 834 | 1,000 | 83.40% |
| thinking | BDV2-Fail | planning | 455 | 500 | 91.00% |
| thinking | RLBench-Fail | planning | 431 | 500 | 86.20% |
| vanilla | RoboFail | execution | 128 | 153 | 83.66% |
| vanilla | RoboFail | planning | 18 | 30 | 60.00% |
| vanilla | UR5-Fail | execution | 105 | 140 | 75.00% |
| vanilla | UR5-Fail | planning | 109 | 140 | 77.86% |
| vanilla | RoboVQA | execution | 276 | 357 | 77.31% |
| vanilla | BDV2-Fail | execution | 826 | 1,000 | 82.60% |
| vanilla | RLBench-Fail | execution | 836 | 1,000 | 83.60% |
| vanilla | BDV2-Fail | planning | 448 | 500 | 89.60% |
| vanilla | RLBench-Fail | planning | 403 | 500 | 80.60% |

The full pipeline status is `complete` at
`baselines/guardian_failcot/runtime/full_checkpoint_eval_20260821/status.tsv`.

### SPR

Tasks 0-7 each passed the 50-outcome/50-annotation/50-raw-video/
50-annotated-video consistency checks.

| Task | Correct | Total | Success rate |
|---:|---:|---:|---:|
| 0 | 41 | 50 | 82.0% |
| 1 | 47 | 50 | 94.0% |
| 2 | 38 | 50 | 76.0% |
| 3 | 44 | 50 | 88.0% |
| 4 | 33 | 50 | 66.0% |
| 5 | 42 | 50 | 84.0% |
| 6 | 38 | 50 | 76.0% |
| 7 | 40 | 50 | 80.0% |
| **Available aggregate** | **323** | **400** | **80.75%** |

Raw logs and videos are under ignored
`baselines/spr/results/released_code_libero10_task{0..7}/`. Task 8 failed in
vLLM CUDA-graph warm-up before episode 0 with a 20 MiB allocation OOM while
GPU 1 had 14.31 MiB free; its log is
`baselines/spr/results/released_code_libero10_task8/20260821_194332.log`.
Task 9 was not started. Consequently, 80.75% is an incomplete eight-task
aggregate and must not be presented as a reproduction of the paper's 82.8%
ten-task aggregate.

## Paused before quantitative execution

- **RACER:** the NVIDIA EGL and isolated-process fallback code is merged, but
  the new single-episode gate never started. There are 0/75 quantitative
  episodes. The latest ignored state is
  `baselines/racer/runtime/virtualgl_egl_20260821_0244_overlayfix1/` in the
  retained RACER worktree.
- **FAIL-Detect:** the bounded external-Diffusion-Policy protocol is merged,
  but artifact download, training, and ID/OOD evaluation never started. Its
  ignored status remained `resource_gate/waiting`; there is no quantitative
  metric.
- **AHA:** the earlier one-episode FailGen smoke remains valid. The new
  10-task x 1-episode queue never entered task 0; no new metric was produced.
- **AgentChord:** remains externally blocked on a secure official
  `dexsim_engine==0.3.11` runtime, container support, and official GPT-5
  credentials. No substitute protocol was run.

The AHA, RACER, and FAIL-Detect status files may still say `waiting` because
their resource-gate processes were deliberately terminated. This checkpoint
supersedes those live-state labels and records the intentional pause.

## Safe recovery

To inspect the exact tracked state without moving the active branch:

```bash
git fetch --tags origin
git worktree add ../essay2608-pause yukun-reproduction-pause-20260823
```

Before any future resume, confirm explicit user approval and re-audit tmux,
process ownership, GPU allocation, artifact integrity, and the current
`origin/yukun` head. Do not rerun completed SPR tasks 0-7. Resolve task 8's
vLLM memory failure first, then run only tasks 8 and 9 and regenerate the
ten-task aggregate. The AHA, RACER, and FAIL-Detect launchers are preserved in
their merged method directories, but none should be started automatically.
