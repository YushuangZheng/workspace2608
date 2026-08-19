# FAIL-Detect

The official release is fixed at commit `b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`.
The first target is the Robomimic `Transport` task with the flow-matching policy,
because it is the released simulation task closest to bimanual manipulation.

There is no official policy or detector checkpoint, and the upstream README
does not identify a source for the required `image_abs.hdf5` dataset. The
released code also differs from the paper in conformal alpha, logpZO training
epochs, diffusion-policy sample counts, and the absence of SPARC. Consequently,
any run is labelled either `upstream_release` or `paper_aligned`; neither label
is silently substituted for the other.

The local isolated environment is `dynamac-fail-detect` (Python 3.9, PyTorch
1.12.1, CUDA 11.6); the upstream YAML calls its environment `faildetect`. A
24 GB RTX 4090 memory smoke test precedes any long training.

## Local setup status

`scripts/setup_env.sh` creates the environment under the yukun Conda prefix and
keeps the official package versions. It records three release-time compatibility
repairs without changing the detector or policy implementation:

- pin MuJoCo to the first wheel satisfying `dm-control==1.0.9`, because the
  unbounded dependency now resolves to a source build requiring an unpublished
  `MUJOCO_PATH`;
- install the documented `patchelf` prerequisite inside the user environment;
- expose the released `diffusion_policy/configs_robomimic/` directory at the
  `diffusion_policy/config/` path hard-coded by `train.py` and `save_data.py`.

The environment passes `pip check`, imports Robomimic/Robosuite/MuJoCo/R3M, and
runs a CUDA tensor smoke test on an RTX 4090. The official Transport flow policy
Hydra configuration composes and its 308,292,372-parameter policy instantiates
on a 4090, occupying about 1.24 GB before batch activations. Training remains
blocked at the data boundary: the release neither contains nor documents the
required `data/robomimic/datasets/transport/ph/image_abs.hdf5`, and no official
policy or detector checkpoint is available. The roughly 85 GB Diffusion Policy
image bundle is a plausible upstream source, but it is not silently treated as
an author-confirmed FAIL-Detect artifact.

See [REPRODUCTION.md](REPRODUCTION.md) for the exact verified commands, the
released pipeline and output paths, compatibility repairs, and the boundary
between an `upstream_release` run and a future paper-aligned experiment.
