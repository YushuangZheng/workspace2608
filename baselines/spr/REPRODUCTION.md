# SPR released-code reproduction

## Reproduction boundary

This reproduction evaluates the official `SPRVLA/libero_10` checkpoint with
the released `run_libero_eval_vllm.py` evaluator. It does not claim to
reproduce training: the paper reports 32 A100-80GB GPUs, while the public
release does not include the training data or a complete training recipe.

The result must be labelled **released-code evaluator reproduction**. The
released evaluator implements the progress-count trigger but does not include
the paper-described eight-step trajectory-stagnation trigger.

## First quantitative target

- Suite: LIBERO-Long (`--task 10`)
- Task: ID 0, `put both the alphabet soup and the tomato sauce in the basket`
- Initial states: the 50 bundled fixed LIBERO states
- Episodes: 50 (the evaluator's fixed official value)
- Seed: 7 (hard-coded upstream)
- Warm-up: 10 simulator steps (hard-coded upstream)
- Inference: BF16 vLLM 0.8.5, tensor parallelism over at most four RTX 4090s
- Paper reference for the full ten-task LIBERO-Long suite: 82.8% for SPR and
  85.4% for the joint `Ours*` checkpoint. A single-task result is not directly
  comparable with either suite average.

The paper does not publish task-level LIBERO-Long values. Table 2 reports only
the ten-task subset aggregate, and the evaluation section states that this is
50 episodes per task across ten tasks. Therefore a paper-matched quantitative
comparison requires task IDs 0-9, not task 0 alone.

## Commands

Verify source, checkpoint structure, and the official CLI:

```bash
bash baselines/spr/scripts/verify_release.sh
```

Run the official 50-episode task-0 evaluation after confirming the selected
GPUs are idle:

```bash
SPR_CUDA_VISIBLE_DEVICES=1,2,3,4 \
  bash baselines/spr/scripts/run_libero10_task0.sh
```

Logs, rollouts, annotations, and videos are written below
`baselines/spr/results/released_code_libero10_task0/` and remain Git-ignored.

Summarize a completed or in-progress log without modifying upstream output:

```bash
baselines/spr/scripts/summarize_libero10_task0.py \
  --full-verify-checkpoint \
  --output baselines/spr/results/released_code_libero10_task0/summary.json
```

The summary cross-checks the official log counts against independently parsed
annotation JSON files and both video streams. It also verifies the clean
upstream commit, evaluator/parser/launcher hashes, checkpoint index, and - with
`--full-verify-checkpoint` - every 16 GB checkpoint shard hash.

After task 0 completes and GPU scheduling is explicitly confirmed, the frozen
follow-up queue for task IDs 1-9 is:

```bash
SPR_CUDA_VISIBLE_DEVICES=1,2,3,4 \
  bash baselines/spr/scripts/run_libero10_tasks1to9.sh
```

Each queued task invokes the unmodified official CLI for its own fixed 50
episodes. It intentionally reloads the model between task IDs instead of
introducing a custom in-process evaluator.

Before every queued task, the launcher inspects each selected GPU separately.
After a completed task it allows only residual `dynamac-spr` workers up to 60
seconds to exit naturally. An external process, an inspection error, or an SPR
worker that remains after 60 seconds fails closed; the launcher never kills or
waits on an unrelated workload.

After all ten tasks finish, validate every run, compute the 500-episode
aggregate and Wilson 95% confidence interval, then render the frozen chart:

```bash
baselines/spr/scripts/aggregate_libero10.py
baselines/spr/scripts/plot_libero10.py
```

The aggregator refuses partial runs and requires exact agreement among each
official log, 50 annotation JSON files, and both 50-video streams. It then
rehashes the evaluator, parser, and all four checkpoint shards before writing
the ignored aggregate result. The plotter likewise refuses unverified input.

One- or two-GPU execution is not selected for the paper-matched run. The 7B
BF16 checkpoint may fit on fewer 24 GB cards, but the paper reports 4x RTX 4090
inference and the official evaluator derives tensor parallelism from visible
GPU count. Fewer cards are expected to reduce compute throughput and would
change the published hardware configuration without changing episode-level
parallelism; they are not a safe acceleration path.

## Verified preflight

- Upstream commit: `d57e4b81ebdcacea574b68be29d61ba04cdc7051`
- Checkpoint revision: `b5838d84d462abd41a45c2b3e7258fa11ec0ed0f`
- Parser tokenizer: `Qwen/Qwen2-7B` at revision
  `453ed1575b739b5b03ce3758b23befdb0967f40e` (hard-coded upstream)
- Checkpoint: four readable safetensors shards, 614 indexed tensors, no missing
  shards
- Official evaluator CLI: import and argument parsing pass
- LIBERO task 0: all 50 fixed initial states load
- Headless simulator: EGL reset, 256x256 agent/wrist rendering, and one action
  step pass
