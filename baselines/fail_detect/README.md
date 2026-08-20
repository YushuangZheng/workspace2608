# FAIL-Detect

The official release is fixed at commit `b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`.
The bounded quantitative target is Robomimic `Transport` with the diffusion
policy, because it is the released simulation task closest to bimanual
manipulation and an official external Diffusion Policy checkpoint exists.

FAIL-Detect itself still publishes no policy or detector checkpoint. The
bounded run therefore uses the official Diffusion Policy `robomimic_image.zip`
and official Transport-PH diffusion checkpoint, with fixed remote metadata and
locally recorded SHA-256 hashes. It is labelled
`upstream_release_external_dp_checkpoint` and is not presented as the paper's
300-epoch policy or full evaluation protocol.

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
runs a CUDA tensor smoke test on an RTX 4090. The official Transport policy
configuration composes and its 308,292,372-parameter model instantiates on a
4090, occupying about 1.24 GB before batch activations.

The prior artifact blocker now has an explicit bounded path: the official
Diffusion Policy archive supplies the exact Transport `image_abs.hdf5`, and its
official Transport checkpoint must pass a strict-load gate. Run
`scripts/launch_quant_tmux.sh` to start a pipeline that waits for the existing
SPR and Guardian jobs and GPU 5, then trains only released logpZO with
`EPOCHS = 200`, runs a 10+10 technical gate, and extends the same seeds to a
total 50+50 evaluation. It has a 24-hour post-gate deadline and permits at most
one explicitly recorded reactive compatibility repair.

See [REPRODUCTION.md](REPRODUCTION.md) for the exact verified commands, the
released pipeline and output paths, compatibility repairs, and the boundary
between an `upstream_release` run and a future paper-aligned experiment.
