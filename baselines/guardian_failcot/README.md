# Guardian / FailCoT

The official release is fixed at commit `4a37cee9cf186fa7bdca2155ce7c25d9e1e08e38`.
The first target is the official `guardian-thinking` checkpoint on the five OOD
cells in paper Table II: RoboFail execution/planning, UR5-Fail
execution/planning, and RoboVQA execution.

The repository's `evaluate.sh` is tied to SLURM and rejects RTX 4090 GPUs. The
local wrapper calls the same `eval.py` directly with `torchrun`; this changes
only cluster launch plumbing. Seven RTX 4090 GPUs ran seven identical BF16 model
replicas for data-parallel evaluation; GPU 0 remained assigned to an unrelated
server job.

The isolated environment is `dynamac-guardian` (Python 3.10, PyTorch 2.3.0,
CUDA 12.1, Transformers 4.41.2, FlashAttention 2.5.9.post1).

The later full released-checkpoint pass completed at
`2026-08-21T02:28:15+08:00`. Together with the canonical five Table II outputs,
it covers all 18 available checkpoint/dataset/config cells: nine for
`guardian-thinking` and nine for `guardian-vanilla`, totalling 7,640 validated
predictions. The additional 13 JSONL files are under ignored
`results/full_checkpoint_eval_20260821/`; the original five thinking OOD files
remain under `results/table_ii/guardian-thinking/`.

See [REPRODUCTION.md](REPRODUCTION.md) for pinned artifact revisions, exact
commands, integrity checks, and Table II results. The ignored local file
`results/comparison/comparison.csv` is rebuilt from the five official JSONL
outputs by `scripts/summarize_table_ii.py`. The plotting script generates the matched comparison at
`results/comparison/guardian_table_ii.png` without mixing benchmarks.
