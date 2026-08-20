# AHA

The official release is fixed at commit `0d39c0591566ddaf997be5822f3dead8e08501aa`.
AHA is retained primarily as a failure-generation and evaluation-protocol
reference. Its bounded one-task FailGen smoke test has been completed
independently of the checkpoint-based baseline evaluations.

An exact paper reproduction is currently blocked upstream: the final AHA model
checkpoint, AHA train/test dataset download, and inference program that creates
the evaluated answer file are not released. The documented full training run
requires roughly 8 A100-80GB GPUs for 40 hours, which exceeds the allowed 8x4090
budget. Paper numbers remain reference-only and receive no fabricated local
counterpart.

The released FailGen stack is installed and its static/import, official
`opengl3` simulator reset, and bounded one-episode smoke paths have been checked
independently of model training. A bounded runner now covers the release's
official ten-task eval list with one episode per task, `max_tries=1`, at most one
CoppeliaSim restart per task, and a two-hour wall-clock budget. It runs
sequentially on CPU/llvmpipe, waits for the existing SPR and Guardian tmux jobs
to end, and records task-level outcomes plus keyframe-image integrity. See
[REPRODUCTION.md](REPRODUCTION.md) for the exact boundary and commands. These
checks do not remove the missing-model and missing-data blockers for paper-level
metrics.
