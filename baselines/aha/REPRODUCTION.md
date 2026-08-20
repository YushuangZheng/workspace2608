# AHA reproduction status

Verified on 2026-08-19 against AHA commit
`0d39c0591566ddaf997be5822f3dead8e08501aa`.

## Current status

The released data-generation and metric utilities are locally runnable:

- PyRep, the patched RLBench fork, and FailGen import in the isolated
  `dynamac-aha` environment;
- the official FailGen CLI and ROUGE-L CLI parse successfully;
- an exact-match call through the released ROUGE function returns `1.0`;
- CoppeliaSim 4.1 launches, loads `basketball_in_hoop`, and completes an
  official `opengl3` task reset under software GLX;
- one bounded `basketball_in_hoop` / `grasp` FailGen episode completed on its
  first allowed attempt and saved 12 keyframe PNGs (1.1 MB). Its runtime output
  is under ignored `runtime/`, never in Git.

This validates the released FailGen path only. It is not an AHA model result and
must not be compared with a paper metric as if it were one.

## Pinned source components

The AHA README does not pin its two simulator dependencies. The commits resolved
and tested for this reproduction are:

- PyRep: `8f420be8064b1970aae18a9cfbc978dfb15747ef`;
- MohitShridhar/RLBench `peract` branch:
  `ad991951bc53e4f3b73b803a75cf4b7d55295cf7`.

The official `python update.py` step was then applied. It replaces exactly six
files in the ignored RLBench checkout with the versions bundled by AHA:
`observation.py`, `scene.py`, `waypoints.py`, `demo.py`,
`observation_config.py`, and `task_environment.py`.

## Isolated environment

The environment lives at `/data/yukun/miniconda3/envs/dynamac-aha` and uses
Python 3.10, PyRep 4.1.0.3, RLBench 1.2.0, and FailGen 0.0.1.

From `/data/yukun/essay2608/baselines/aha`:

```bash
source /data/yukun/miniconda3/etc/profile.d/conda.sh
conda create -n dynamac-aha python=3.10 pip -y
conda activate dynamac-aha
unset PIP_EXTRA_INDEX_URL
export PIP_CONFIG_FILE=/dev/null

export COPPELIASIM_ROOT=/data/yukun/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

python -m pip install --index-url https://pypi.org/simple \
  -r upstream/PyRep/requirements.txt
python -m pip install --index-url https://pypi.org/simple upstream/PyRep
(cd upstream && python update.py)
python -m pip install --index-url https://pypi.org/simple \
  -r upstream/RLBench/requirements.txt
python -m pip install --no-build-isolation -e upstream/RLBench
python -m pip install --index-url https://pypi.org/simple \
  -r upstream/aha/Data_Generation/rlbench-failgen/requirements.txt
python -m pip install --no-build-isolation -e \
  upstream/aha/Data_Generation/rlbench-failgen
python -m pip install --index-url https://pypi.org/simple rouge-score

bash scripts/smoke_release.sh
```

After installation, `python -m pip check` reports no broken requirements in
this environment. The smoke script additionally verifies the released FailGen
collection CLI parser and ROUGE evaluation entry point before any simulator is
launched.

For headless rendering, use the user-space Ubuntu Xvfb binary configured by
`scripts/smoke_release.sh`; do not install a Conda Xvfb/libGL stack into this
environment. Mixing Conda's libGL with CoppeliaSim 4.1 caused a reproducible
`swrast_dri.so` crash during vision reset. With the isolated Ubuntu Xvfb and the
official `opengl3` configuration, reset succeeds without changing AHA code or
renderer settings.

Run the simulator checks explicitly:

```bash
bash scripts/smoke_release.sh --sim-reset
bash scripts/smoke_release.sh --episode
```

Both are CPU/software-rendering checks and do not reserve a training GPU.
The episode helper uses the released FailGen wrapper and failure classes, keeps
the official `opengl3` renderer, and limits output to keyframes so that a smoke
test does not create a large demonstration dataset. It produced:

```text
Saved FailGen smoke episode 1 / 1
Renderer: opengl3
Waypoint index: 1
Keyframe PNGs: 12
```

That local output-path/keyframe limit is a smoke-test boundary, not a paper
protocol change and not evidence of AHA model quality.

## Bounded official-list FailGen run

The released `examples/ex_data_generator_eval.sh` names ten eval tasks. The
bounded runner preserves that task order while deliberately limiting the work
to one generated episode per task and one `get_failure()` call (`max_tries=1`).
For each task it selects the first supported failure in the official
`FAILURES_LIST` order and the first configured waypoint. This is a coverage run
of the released failure generator, not the upstream 100-episode-per-task data
collection protocol and not an AHA model evaluation.

Each task runs in its own process and X display. A failed attempt may start the
task once more, so the maximum is one CoppeliaSim restart per task. Attempts are
limited to five minutes, each task to ten minutes, and the complete ten-task
run to two hours. Tasks remain sequential. The launcher waits until all tmux
sessions whose names begin with `dynamac_spr` or `dynamac_guardian` have ended,
then sets an empty `CUDA_VISIBLE_DEVICES` and forces software GL rendering.

Run the source/protocol check without launching CoppeliaSim:

```bash
bash scripts/run_failgen_eval10.sh --static-check
```

Start the gated run (it waits if SPR or Guardian is still active):

```bash
bash scripts/run_failgen_eval10.sh
```

Large outputs stay under ignored `results/failgen_eval10_<UTC timestamp>/`.
After every task, `events.jsonl` receives and stdout yields a complete task
record. The final `summary.json` and `summary.csv` include the selected failure
type, waypoint, attempts, restart count, failure class, PNG count, and counts
for the `front`, `overhead`, and `wrist` streams. Image acceptance requires all
three streams, equal nonzero and contiguous frame counts, valid decodable PNGs,
positive dimensions, and a recorded SHA-256 digest for every file.

## Paper-level blockers

1. The final AHA checkpoint is not released.
2. The generated AHA train/test data and model answer files used by the metric
   scripts are not published as a ready-to-download artifact.
3. The repository provides RoboPoint fine-tuning instructions but no matching
   AHA inference entry point that creates the evaluated answer JSONL.
4. The documented full fine-tuning run takes about 40 hours on eight A100-80GB
   GPUs, beyond the allowed eight-RTX4090 budget.
5. `LLM_fuzzy.py` additionally instantiates an Anthropic client and therefore
   needs an external Anthropic API credential; none is configured.

Accordingly, no AHA paper metric is reported locally. The successful FailGen
checks establish only that the released data-generation front end is usable.
