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
four-GPU layout, validated component checks, formal failure, diagnostic
boundary, and the queued NVIDIA EGL retry.

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
language-service app and binds it to a selectable local port; it does not
change model or inference logic.

The next bounded retry uses VirtualGL 3.1.5's EGL backend. Xvfb remains only the
2D X server, while VirtualGL redirects GLX rendering to the NVIDIA EGL device
that maps to the actor GPU. The retry is fail-closed: one fixed `place_cups`
episode 0 must complete with a natural status 0 and four nonempty camera GIFs
before the three-task 75-episode evaluation can start. It also waits for the
existing SPR and Guardian tmux jobs to terminate and for GPUs 1-4 to remain
idle. See `scripts/run_virtualgl_egl_after_baselines.sh`.

VirtualGL is downloaded from its official release, checksum-verified, and
extracted under `/data/yukun/.cache/racer` without root or system changes:

```bash
bash baselines/racer/scripts/bootstrap_user_virtualgl.sh
```

The official RACER, PyRep, RLBench, CoppeliaSim, policy, and evaluation code is
not patched by this graphics-transport retry.
