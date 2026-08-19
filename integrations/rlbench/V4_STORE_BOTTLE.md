# StoreBottle V4 training boundary

V4 retrains only `bimanual_put_bottle_in_fridge`. Its training and online
policy inputs are the same two poses, in this order:

1. `bottle` from scene object `bottle`;
2. `fridge` from the physical scene object `fridge_base`.

The legacy V1–V3 task class, TTM, data, and checkpoints remain unchanged.
Other V4 model groups are inherited byte-for-byte from V3; the manifest API
only inventories them and never copies or trains checkpoints.

## Preflight and one-demo smoke

These commands do not write artifacts or launch CoppeliaSim:

```bash
python3 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 collect --dry-run
python3 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 train --dry-run
python3 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 release-manifest --dry-run
```

After preflight acceptance, validate the live task with one successful demo in
a disposable, non-release directory. A smoke manifest cannot be consumed by
the official training loader.

```bash
python3.8 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 smoke-collect \
  --output-dir /tmp/store_bottle_v4_smoke --headless
```

## Official five-demo collection and training

The collection protocol is fixed at variation `0`, no environment
intervention, and seeds `4104000000..4104000004`. RLBench only returns a live
demo after `task.success()` is true. The collector validates all observations
as two finite task poses, writes each file hash, and atomically publishes the
five-demo directory.

```bash
python3.8 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 collect --headless

python3.10 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 train
```

Artifacts are isolated at:

- demonstrations: `integrations/rlbench/data/v4/store_bottle`;
- StoreBottle model: `integrations/rlbench/models/v4/bimanual_put_bottle_in_fridge`.

Collection and training reject paths below `evaluation_sets/` and `results/`.
The training manifest binds the five-demo manifest hash, policy-config hash,
semantic version/fingerprint, frame objects, pose chunks, and segmentation
contract. `PolicyServer` reads that authenticated V4 identity and selects the
corrected StoreBottle spec automatically; a V3 manifest continues to select
the legacy `bottle/fridge_root` spec.

Serve the trained StoreBottle policy with:

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 serve
```

After the Store model is trained and the other model groups have been copied
by an explicit release operation, authenticate their byte identities without
performing either operation:

```bash
python3 -m integrations.rlbench.rlbench_dynamac.store_bottle_v4 release-manifest \
  --require-complete \
  --output integrations/rlbench/models/v4/release_manifest.json
```

## Rebind the archived evaluation plans after retraining

When retraining changes the authenticated intervention evidence, the archived
StoreBottle scene plans are not resampled. The identity-only rebind command
hash-pins the archived outer envelope, updates the intervention identity and
dependent fingerprints, and refuses publication unless every non-identity
runtime field has an exactly equal projection. It does not start RLBench or
read evaluation results.

```bash
python3.10 -m integrations.rlbench.rlbench_dynamac.store_bottle_plan_rebind
```

The defaults bind the archived envelope with SHA-256
`4767f199f3af2b1464b47194bb7a8de8e9c0932482c2ff8ea227fd89f6310a81`
and atomically reserve the canonical
`evaluation_sets/rlbench_eval_v2/plans/environment/` destination. An existing
destination is never overwritten.
