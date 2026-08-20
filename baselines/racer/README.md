# RACER

This directory keeps the reproducible, repository-owned shell around the
official RACER release. The large source checkout, checkpoints, RLBench test
episodes, runtime logs, and results stay local and are ignored by Git.

The released-checkpoint reproduction is frozen to three RLBench tasks first:
`place_cups`, `place_wine_at_rack_location`, and
`sweep_to_dustpan_of_size`. It uses the official actor, rich-instruction LoRA,
LLaVA base, T5-11B encoder, and fixed PerAct test episodes. Training is not
required for this stage.

See [manifest.json](manifest.json) for immutable revisions, artifact identities,
and run provenance. See [REPRODUCTION.md](REPRODUCTION.md) for the environments,
four-GPU layout, validated component checks, formal failure, and diagnostic
boundary.

## Current result

The released-checkpoint reproduction is **blocked before episode 0** on this
workstation. Source, weights, fixed episodes, both model services, actor
construction/checkpoint loading, and a simulator-only four-camera reset/render
smoke all pass. In the formal retry2 run, the actor then exits with status 139
inside Mesa `swrast_dri.so` during its first RLBench reset, after `Agent Reset.
Model loaded.` and before any episode result is written.

Consequently the local result is **0/75 completed episodes**: there is no local
success rate and no valid paper-comparison chart. The A-K2 diagnostics found no
Xvfb/software-Mesa workaround that both preserves observations and exits
cleanly, so the 75-episode run is intentionally stopped under this renderer.

Port 8000 belongs to a resident external XPolicy Eval service and was left
untouched. The tracked infrastructure-only wrapper imports the same official
language-service app and binds it to port 18000; it does not change model or
inference logic. A future retry should first use an NVIDIA-backed Xorg or
VirtualGL display and pass one exact end-to-end episode with a natural status 0.
