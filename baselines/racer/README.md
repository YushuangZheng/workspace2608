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
and must report `success: true` before the three-task 75-episode evaluation can
start. It also waits for the
existing SPR and Guardian tmux jobs to terminate and for GPUs 1-4 to remain
idle. See `scripts/run_virtualgl_egl_after_baselines.sh`.

VirtualGL is downloaded from its official release, checksum-verified, and
extracted under `/data/yukun/.cache/racer` without root or system changes:

```bash
bash baselines/racer/scripts/bootstrap_user_virtualgl.sh
```

The official RACER, PyRep, RLBench, CoppeliaSim, policy, and evaluation code is
not patched by this graphics-transport retry. The launcher-owned isolation
adapter changes only where `RLBenchSim` executes: policy/PyTorch3D remains in
the evaluator parent and the official simulator runs in a clean spawned Python
worker.

If the EGL episode gate fails, the supervisor has exactly one fail-closed
fallback path. It first compares complete numeric fingerprints from a direct
fixed reset and a policy-loaded-parent/spawn-isolated fixed reset, and requires
both capture processes plus the simulator worker to exit naturally with status
0. Only an exact initialization-observation match permits one isolated
`place_cups` episode 0 attempt. That attempt has zero evaluator retries and
must report `success: true` and pass the same
metrics/marker/four-GIF/native-failure validator before it can unlock 3 x 25
under the same isolated backend. Any mismatch, abnormal exit, or unsuccessful
episode stops without retry and leaves the full run locked.

Both one-episode gates override InvalidActionError retries to zero. After a
successful gate, the 3 x 25 evaluation explicitly uses five retries, matching
the released `rollout.py` default; the official `scripts/eval_racer.sh` leaves
that default unchanged. Thus the fidelity run preserves the released protocol,
while the unlock decision cannot be satisfied by a retried gate episode.

The gate validator opens and decodes every GIF, requires the official
256x346 rendered-frame size, at least one readable frame, and nondegenerate
pixels specifically within the top 256x256 camera region; the lower 90px text
overlay cannot satisfy this check. It also requires an exact success marker. A
thin rollout adapter saves the four raw reset
point clouds before policy preprocessing. The validator reloads that NPZ and
requires exactly four float32 arrays of shape `(3, 512, 512)`, all finite and
nondegenerate, with matching array and archive hashes. This records evidence
only; it does not alter observations passed to the released policy.

The supervisor freezes the exact Git HEAD at startup and checks the same clean
checkout again after the dependency wait and before later stages. Direct and
isolated reset captures each have a 600-second wall-clock limit; either
episode-0 gate has a 3600-second limit. Every capture/evaluation stage records
an exact environment-owned `/proc` audit after cleanup, including CoppeliaSim,
worker, model-service, evaluator, and Xvfb descendants. Any timeout, checkout
change, or residual owned process fails closed.
