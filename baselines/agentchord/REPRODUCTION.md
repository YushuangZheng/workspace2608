# AgentChord reproduction status

Verified on 2026-08-19 against upstream commit
`74fe7dd756964b737952d26e9e65d4d5e2e29ec3`.

## Current status

The released recovery-graph code is testable without the private simulator:

- all 25 JSON configurations parse;
- `embodichain` imports as version `0.1.3`;
- the focused upstream suites
  `tests/sim/agent/test_graph_spec.py` and
  `tests/sim/agent/test_agent_graph.py` pass: **29 passed**;
- the repository LLM health check correctly stops with
  `Missing required environment variable(s): OPENAI_API_KEY, LLM_URL`.

These are source and orchestration checks, not robot rollouts and not Table I
measurements. No AgentChord success-rate number has been produced locally.

## Isolated audit environment

The environment lives under the yukun Conda installation:
`/data/yukun/miniconda3/envs/dynamac-agentchord`.

```bash
source /data/yukun/miniconda3/etc/profile.d/conda.sh
conda create -n dynamac-agentchord python=3.10 pip -y
conda activate dynamac-agentchord
unset PIP_EXTRA_INDEX_URL
export PIP_CONFIG_FILE=/dev/null

python -m pip install --index-url https://pypi.org/simple \
  'pytest>=8,<9' 'langchain-openai>=0.3,<1' numpy
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0
python -m pip install --no-deps --no-build-isolation -e upstream

bash scripts/smoke_release.sh
```

The CPU PyTorch wheel is sufficient for the focused graph tests. It does not
make the GPU simulator runnable. The editable install intentionally uses
`--no-deps`: installing the full metadata would request the unavailable
`dexsim_engine` package and cannot make the simulator usable without its secure
distribution. Consequently, `pip check` is expected to list simulator-side
requirements as absent in this source-audit environment.

## Complete rollout blockers

1. The simulation task, recovery, and compile agents are hard-wired to model
   `gpt-5`. The server has neither `OPENAI_API_KEY` nor `LLM_URL`, so graph
   generation cannot start.
2. The package requires `dexsim_engine==0.3.11`. It is not available from the
   public HTTPS PyPI index. The only installation command published upstream
   adds `http://pyp.open3dv.site:2345/simple/` as a trusted host. That plaintext
   private index was deliberately **not contacted**.
3. The documented alternative is the vendor Docker image, but Docker, Podman,
   Singularity, and Apptainer are absent on this server.
4. The paper's random-drop schedules, fixed seeds, and batch evaluator are not
   released. Even after runtime access is restored, a single interactive
   rollout would be a functional released-code check rather than an exact
   Table I reproduction.

AgentChord is therefore fully blocked for simulation on external prerequisites,
not by a discovered DynaMAC or local project-code defect. When a GPT-5-compatible
endpoint and a secure DexSim distribution are provided, run
`bash scripts/smoke_release.sh --llm` first, then the upstream `SinglePourWater`
command from its README.
