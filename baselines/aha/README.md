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

The released FailGen stack is now installed and its static/import, official
`opengl3` simulator reset, and bounded one-episode smoke paths have been checked
independently of model training. The episode succeeded on its first allowed
attempt and saved 12 keyframe PNGs. See [REPRODUCTION.md](REPRODUCTION.md) for
the exact boundary and commands. These checks do not remove the missing-model
and missing-data blockers for paper-level metrics.
