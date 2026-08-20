# FAIL-Detect reproduction status

Originally verified on 2026-08-19 and bounded quantitative path added on
2026-08-21 against upstream commit
`b758e55f7c0c988188f2e4876ffc03ae8a3c30ed` and the RSS 2025 paper
*Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure
Detection for Imitation Learning Policies*.

## Current result and boundary

The environment/model-construction smoke test is complete. A resource-gated
bounded quantitative run is now defined, but it is not paper evaluation:

- the isolated Python 3.9 / CUDA 11.6 environment imports the official policy,
  simulator, and uncertainty dependencies;
- `pip check`, the official entry-point help paths, and an RTX 4090 CUDA tensor
  test pass;
- the released Transport policy Hydra configuration composes;
- its 308,292,372-parameter policy constructs on an RTX 4090 and allocates
  1,239,120,896 CUDA bytes before batch activations.

No FAIL-Detect success-rate, detection-accuracy, or detection-time number has
been produced locally. The release contains neither an official policy/detector
checkpoint nor the required Transport dataset, and its README does not identify
a source for:

```text
data/robomimic/datasets/transport/ph/image_abs.hdf5
```

For the bounded run only, the exact file is extracted from the official
Diffusion Policy image archive and paired with its official Transport-PH
checkpoint. These external artifacts are never represented as author-confirmed
FAIL-Detect artifacts.

## Fixed provenance

- Code: `CXU-TRI/FAIL-Detect` at
  `b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`.
- Paper: RSS 2025 proceedings paper `p073`; the local indexed copy is
  `papers/FAIL-Detect.pdf`.
- Bounded target: Robomimic `Transport` with the diffusion-policy checkpoint,
  because it is the released task closest to bimanual manipulation with an
  official external checkpoint path.
- Official checkpoint: none released.
- Official dataset artifact for the configured `image_abs.hdf5`: none
  identified by the release.

## Bounded quantitative protocol

Start the detached pipeline from the FAIL-Detect reproduction branch/worktree:

```bash
bash baselines/fail_detect/scripts/launch_quant_tmux.sh
```

It waits without downloading or claiming a GPU until both
`dynamac_spr_full_20260821` and `dynamac_guardian_full_20260821` have ended and
GPU 5 has no compute process and at most 512 MiB allocated. After that resource
gate opens, the whole download/train/evaluate sequence is limited to 24 hours.

The fixed stages are:

1. Pin the official FAIL-Detect source and route only its documented missing
   config symlink.
2. Download, metadata-check, SHA-lock, and extract Transport `image_abs.hdf5`
   from the official Diffusion Policy archive. Use its official Transport-PH
   diffusion checkpoint, not an invented FAIL-Detect checkpoint.
3. Validate HDF5 magic, demonstrations, four 84x84 RGB streams, raw action
   dimension 14, released absolute-action dimension 20, and strict model/EMA
   loading.
4. Import the real released `UQ_baselines/CFM/net_CFM.py` from the pinned
   checkout, construct `get_unet(20)`, and strict-load its state into a second
   instance before any large download. Then run released `save_data.py`,
   validate `(N,548)` condition and `(N,320)`
   action tensors, train only logpZO with released `EPOCHS = 200`, and
   strict-load the detector. A valid checkpoint below epoch 200 follows the
   official resume path. A corrupt/non-resumable checkpoint or partial feature
   file is moved to ignored quarantine before rebuilding, consuming the single
   permitted reactive compatibility repair; file size alone is never accepted
   as completeness.
5. Evaluate paired seeds for 10 ID + 10 OOD rollouts. Continue only if the
   trajectories are finite/equal-length, at least four successful ID rollouts
   are available for calibration, both outcome classes exist after calibration,
   and ID policy success is at least 7/10. Detector performance is deliberately
   not a continuation threshold.
6. Extend the same outputs and seeds to a total 50 ID + 50 OOD rollouts. Use
   the first 20 successful ID trajectories as 6 mean / 14 band trajectories for
   the released upper `FunctionalPredictor(Tfunc, Mean)` band at alpha 0.05.

The report records ID/OOD success, TPR, and TNR with Wilson 95% intervals,
TP/TN/FP/FN, balanced accuracy, and mean true-positive detection step with
standard error. Balanced accuracy is a mean of two class-conditional
proportions, so it is not assigned a single Wilson interval. Calibration
successes are excluded from detection testing. OOD preserves
the released Transport threshold `t >= 50` with delta 0.1 (the first 8-step
decision boundary reached is step 56). Only the
released logpZO-on-`global_cond` score is evaluated; other detectors and STAC's
costly resampling are outside this thin run.

Ignored status and full outputs are under
`baselines/fail_detect/runtime/quant_pipeline/` and
`baselines/fail_detect/results/external_dp_logpzo_v1/`. The summarizer writes a
small reviewable hash/protocol record to
`baselines/fail_detect/provenance/external_dp_logpzo_v1.json`; large artifacts
remain ignored. The outer deadline runner is the sole owner of final status,
and a TERM marker plus GNU timeout exit 124 can only yield `stopped/deadline`,
never `complete`. Stop rather than broaden the claim if the 24-hour deadline
fires, a strict/schema gate fails, resources are reclaimed, or a second
reactive compatibility repair would be required.

## Isolated environment and verified commands

The local Conda environment is
`/data/yukun/miniconda3/envs/dynamac-fail-detect`. The complete resolved version
record and smoke measurements are in [environment.json](environment.json).

From `/data/yukun/essay2608`:

```bash
bash baselines/fail_detect/scripts/setup_env.sh

CUDA_VISIBLE_DEVICES=5 \
  bash baselines/fail_detect/scripts/validate_upstream.sh
```

GPU 5 was the idle card used for the recorded CUDA/model smoke. A rerun may use
another explicitly verified idle GPU. These scripts do not download the
missing demonstration dataset and do not start long training.

Three release-time compatibility repairs are applied without changing detector
or policy logic:

1. MuJoCo is pinned to `2.3.1.post1`, the first compatible wheel for
   `dm-control==1.0.9`, instead of accepting a current source-only resolution
   that requires an external `MUJOCO_PATH`.
2. `patchelf` is installed inside the user-owned Conda environment.
3. The missing `diffusion_policy/config` path referenced by the entry points is
   a symlink to the released `diffusion_policy/configs_robomimic` directory.

## Released pipeline after data becomes available

The commands below document the upstream-release path; they are not evidence
that the currently missing data or checkpoints were reconstructed. From
`/data/yukun/essay2608/baselines/fail_detect/upstream`, after activating
`dynamac-fail-detect`:

```bash
python train.py \
  --config-dir=diffusion_policy/configs_robomimic \
  --config-name=image_transport_ph_visual_flow_policy_cnn.yaml \
  training.seed=1103 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${name}_${task_name}'

python save_data.py \
  --config-dir=diffusion_policy/configs_robomimic \
  --config-name=image_transport_ph_visual_flow_policy_cnn.yaml \
  training.seed=1103 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${name}_${task_name}'

cd UQ_baselines/logpZO
python train.py --policy_type=flow --type transport
cd ../../UQ_test

python eval_together.py \
  --policy_type=flow --task_name=transport --device=0 --modify=false --num=2000
python eval_together.py \
  --policy_type=flow --task_name=transport --device=0 --modify=true --num=2000

python plot_with_CP_band.py
python barplot.py
```

The released files are expected under these ignored upstream locations:

- policy workspace and checkpoint:
  `upstream/data/outputs/train_flow_unet_visual_transport_image/`;
- ID/OOD rollouts and logs:
  `upstream/data/outputs/train_flow_unet_visual_transport_image/final_eval/`;
- logpZO checkpoint:
  `upstream/UQ_baselines/logpZO/transport_flow.ckpt`;
- detector metric cache and figures:
  `upstream/UQ_test/logging/` and `upstream/UQ_test/full_results/`.

## Released-code versus paper protocol

The public code is not silently presented as the exact paper protocol:

| Item | Released evidence | Reproduction handling |
|---|---|---|
| logpZO training | `UQ_baselines/logpZO/train.py` fixes `EPOCHS = 200`; the paper audit records 500 epochs | preserve the released value only under label `upstream_release`; a 500-epoch run must be labelled `paper_aligned` |
| conformal alpha | `UQ_test/plot_with_CP_band.py::get_alpha()` selects task/split-dependent values | do not claim a paper match until the exact paper settings are resolved |
| evaluation and detector splits | the upstream README requests 2,000 evaluation rollouts, while plotting defaults are 300 train / 700 calibration / 1,000 test samples | record explicit counts with every future result; never mix defaults silently |
| SPARC | discussed in the paper comparison but absent from the released implementation | report it as unavailable rather than assigning a zero or fabricated result |
| randomization | released training/evaluation commands use seed `1103`; no complete multi-seed paper batch protocol is published | a single released-code run cannot reproduce paper uncertainty |

Accordingly, quantitative artifacts must carry `upstream_release`,
`upstream_release_external_dp_checkpoint`, or `paper_aligned`, and record the
exact dataset identity, policy checkpoint, split counts, alpha settings, epochs,
and seeds. The external-checkpoint result can validate the released mechanism,
but it cannot reproduce or refute the paper's reported number.
