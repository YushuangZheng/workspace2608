# FAIL-Detect reproduction status

Verified on 2026-08-19 against upstream commit
`b758e55f7c0c988188f2e4876ffc03ae8a3c30ed` and the RSS 2025 paper
*Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure
Detection for Imitation Learning Policies*.

## Current result and boundary

The local reproduction currently reaches an **environment and model-construction
smoke test**, not policy training or paper evaluation:

- the isolated Python 3.9 / CUDA 11.6 environment imports the official policy,
  simulator, and uncertainty dependencies;
- `pip check`, the official entry-point help paths, and an RTX 4090 CUDA tensor
  test pass;
- the released Transport flow-policy Hydra configuration composes;
- its 308,292,372-parameter policy constructs on an RTX 4090 and allocates
  1,239,120,896 CUDA bytes before batch activations.

No FAIL-Detect success-rate, detection-accuracy, or detection-time number has
been produced locally. The release contains neither an official policy/detector
checkpoint nor the required Transport dataset, and its README does not identify
a source for:

```text
data/robomimic/datasets/transport/ph/image_abs.hdf5
```

The approximately 85 GB Diffusion Policy image bundle is a plausible data
source, but it is not treated as author-confirmed FAIL-Detect data. Training or
evaluation must remain blocked until the dataset provenance is resolved.

## Fixed provenance

- Code: `CXU-TRI/FAIL-Detect` at
  `b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`.
- Paper: RSS 2025 proceedings paper `p073`; the local indexed copy is
  `papers/FAIL-Detect.pdf`.
- First target: Robomimic `Transport`, flow-matching policy, because it is the
  released task closest to bimanual manipulation.
- Official checkpoint: none released.
- Official dataset artifact for the configured `image_abs.hdf5`: none
  identified by the release.

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

Accordingly, future quantitative artifacts must carry either the
`upstream_release` or `paper_aligned` protocol label and record the exact
dataset identity, policy checkpoint, split counts, alpha settings, epochs, and
seed. Until the missing author-side artifacts are identified, the verified
environment/model smoke is the complete reproducible result.
