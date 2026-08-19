# Guardian / FailCoT checkpoint reproduction

Run date: 2026-08-19.

This reproduces the five real-robot OOD cells reported for the released
`guardian-thinking` checkpoint in paper Table II. The model, annotations,
media, and evaluator are all official release artifacts. Only the SLURM
launcher was replaced: the repository's `evaluate.sh` requests an H100 job and
rejects RTX 4090, so the same unmodified `eval.py` was launched with
`torchrun`.

## Fixed provenance

- Code: `paulpacaud/Guardian-FailCot` at
  `4a37cee9cf186fa7bdca2155ce7c25d9e1e08e38`.
- Checkpoint: `paulpacaud/guardian-thinking` at
  `ebad73bb2171d98be883bed614bde01f9386a8a9`.
- OOD dataset: `paulpacaud/Guardian-FailCot-OOD-datasets` at
  `bd2b24224e4da833a8d42ca694d839622e54bb48`.
- Hugging Face commit history on the run date lists each of the checkpoint and
  OOD dataset revisions above as its repository's sole initial release, so the
  run did not accidentally select a later artifact update.
- Checkpoint integrity: 685 indexed tensors; all seven referenced safetensor
  shards present; no extra model shards; 31.78 GB in total.
- Dataset integrity: all JSONL rows and every referenced image exist. Counts
  are 153/30 RoboFail execution/planning, 140/140 UR5-Fail
  execution/planning, and 357 RoboVQA execution.

## Runtime

- Conda environment: `dynamac-guardian` under `/data/yukun/miniconda3/envs/`.
- Python 3.10.20; PyTorch 2.3.0, TorchVision 0.18.0, and TorchAudio 2.3.0
  with CUDA 12.1;
  Transformers 4.41.2; Accelerate 0.30.1; FlashAttention 2.5.9.post1;
  NumPy 1.26.4; timm 1.0.9; einops 0.8.0; Pillow 10.4.0;
  sentencepiece 0.1.99; peft 0.10.0; decord 0.6.0.
- Seven NVIDIA RTX 4090 replicas in BF16 data parallelism (GPUs 1-7). GPU 0
  was occupied by an unrelated process and was not used or modified.

`pip check` reports only `decord 0.6.0 is not supported on this platform`, a
known metadata issue in that wheel. The import succeeds, and all evaluated OOD
samples use images rather than video decoding.

This is the official checkpoint-evaluation runtime, not a full Guardian
training/simulation environment. Dependencies used only by data generation,
RLBench simulation, or training were deliberately not installed because they
are outside this Table II reproduction.

## Commands

From `baselines/guardian_failcot/upstream`:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTHONPATH="$PWD"
export XDG_CACHE_HOME=/data/yukun/.cache/dynamac-baselines
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

/data/yukun/miniconda3/envs/dynamac-guardian/bin/torchrun \
  --standalone --nnodes=1 --nproc_per_node=7 \
  InternVL/internvl_chat/eval/failure_dataset/eval.py \
  --checkpoint data/failure_forge/models/guardian-thinking \
  --config_name planning_thinking \
  --save_output_dir ../results/table_ii \
  --batch_size 1 --num_workers 1 \
  --datasets robofail,ur5fail_test

/data/yukun/miniconda3/envs/dynamac-guardian/bin/torchrun \
  --standalone --nnodes=1 --nproc_per_node=7 \
  InternVL/internvl_chat/eval/failure_dataset/eval.py \
  --checkpoint data/failure_forge/models/guardian-thinking \
  --config_name execution_thinking \
  --save_output_dir ../results/table_ii \
  --batch_size 1 --num_workers 1 \
  --datasets robofail,ur5fail_test,robovqa
```

## Results

Paper accuracies are the two-decimal values published in Table II. Local
accuracies and correct counts are computed from the saved per-sample JSONL.

| Benchmark | Mode | n | Paper | Reproduced | Correct | Delta (pp) |
|---|---|---:|---:|---:|---:|---:|
| RoboFail | execution | 153 | 0.86 | 0.8431 | 129 | -1.69 |
| RoboFail | planning | 30 | 0.70 | 0.6667 | 20 | -3.33 |
| UR5-Fail | execution | 140 | 0.77 | 0.7714 | 108 | +0.14 |
| UR5-Fail | planning | 140 | 0.89 | 0.8786 | 123 | -1.14 |
| RoboVQA | execution | 357 | 0.85 | 0.8151 | 291 | -3.49 |

All 820 predictions contain valid `<think>`, `<answer>`, and `<category>`
fields; there are no truncated or unparsable answers. A one-GPU RoboFail
planning smoke produced exactly the same 20/30 result as the seven-GPU run, so
data-parallel world size does not explain the small gaps.

No source, checkpoint, dataset, prompt, generation-parameter, or media-path
mismatch was found. The remaining differences from the rounded paper values
are therefore release-level or hardware/numerics-level differences, not a
known local pipeline error. Confirming the exact paper-run artifact revisions
and H100/A100 software stack with the authors would distinguish those causes.

The evaluator's printed `think_accuracy` is always zero because the upstream
parser never creates the `think_correct` key that `compute_metrics` reads.
This upstream auxiliary-metric bug does not affect the binary accuracies above.

Raw outputs are intentionally ignored by Git under:

```
baselines/guardian_failcot/results/table_ii/guardian-thinking/
```

First rebuild the ignored chart input directly from the five official JSONL
outputs. The summarizer fails closed on missing rows, non-binary outcomes, or
duplicate sample IDs:

```bash
/data/yukun/miniconda3/envs/dynamac-guardian/bin/python \
  baselines/guardian_failcot/scripts/summarize_table_ii.py
```

Then rebuild the matched paper/local chart (including 95% Wilson intervals for
the finite local samples):

```bash
/data/yukun/miniconda3/bin/conda run -n dynamac-baseline-analysis \
  python baselines/guardian_failcot/scripts/plot_comparison.py \
  --input baselines/guardian_failcot/results/comparison/comparison.csv \
  --output baselines/guardian_failcot/results/comparison/guardian_table_ii.png
```

The generated chart SHA-256 is
`2c480e975876a7ff01c7b1909b4135cd6b588cd88aa371c6c9e3e70c8aaccaa7`.

SHA-256 anchors for those exact JSONL outputs:

| Output (relative to `guardian-thinking/`) | SHA-256 |
|---|---|
| `execution_thinking/DATASET_robofail.jsonl` | `1232f8e8dae6c77e4ca510e18b6772d31fbf967b67c7e48c25b22ad18c8e0f11` |
| `execution_thinking/DATASET_ur5fail_test.jsonl` | `02d39269a663102c759753e0b6fad048850852b951c93ad8c1e117c8f92ad99a` |
| `execution_thinking/DATASET_robovqa.jsonl` | `649d1e098924bea2c1e3e0f32f84f82ce870cf1e7966cd9ec7523118b31efba0` |
| `planning_thinking/DATASET_robofail.jsonl` | `f3880e059b8e350358a9f53259d7dfde39c45530a3ff5043c14305db520d24b3` |
| `planning_thinking/DATASET_ur5fail_test.jsonl` | `597da5f81bc75d69797a85b9689246c7071897cf1e642ceee42c53a45d9e4db6` |
