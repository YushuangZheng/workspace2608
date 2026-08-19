# RACER

This directory keeps the reproducible, repository-owned shell around the
official RACER release. The large source checkout, checkpoints, RLBench test
episodes, runtime logs, and results stay local and are ignored by Git.

The released-checkpoint reproduction is frozen to three RLBench tasks first:
`place_cups`, `place_wine_at_rack_location`, and
`sweep_to_dustpan_of_size`. It uses the official actor, rich-instruction LoRA,
LLaVA base, T5-11B encoder, and fixed PerAct test episodes. Training is not
required for this stage.

See `manifest.json` for immutable revisions and artifact identities, and
`REPRODUCTION.md` for the validated environments, four-GPU layout, display
adapter, health checks, launch command, output format, and comparison scope.

The formal 75-episode evaluator is generated but must not be started until its
four requested GPUs are idle:

```bash
bash baselines/racer/scripts/run_three_task_eval.sh
```

It fails before evaluation if an asset, GPU, port, service, or GLX simulator
check is not valid, and it tears down only the processes it started.
