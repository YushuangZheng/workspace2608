# DynaMAC RLBench Adapter

This package contains the RLBench-facing implementation used to fit and execute
the reproduced DynaMAC motion backbone.

```text
core/       task specifications, low-level execution, IK, and record schemas
data/       demonstration adapters, fitting, and checkpoint validation
eval/       rollout entry points
protocols/  task-independent execution protocols
report/     compact rollout summaries and replay helpers
```

The reproduction is an independently maintained implementation based on the
published DynaMAC method. It is not an official upstream release. Historical
workspace paths containing `v4` identify the fourth internal reproduction
revision used while validating the backbone; they do not denote a fourth
version published by the original authors. New user-facing assets should use a
functional name such as `dynamac_backbone` instead.

For environment setup, fitting commands, and the TSF sidecar workflow, see the
parent [`README.md`](../README.md).
