# AgentChord

The official release is fixed at commit `74fe7dd756964b737952d26e9e65d4d5e2e29ec3`.
The first target is a nominal `SinglePourWater` simulation followed by the same
manual failure with recovery disabled/enabled; `DualPourWater` is next because
it is the most relevant bimanual case.

The released code requires an OpenAI-compatible GPT-5 endpoint. No endpoint or
API key is configured on the server, and the repository does not publish the
paper's random-drop schedules, fixed seeds, or batch evaluator. An exact GPT-5
API endpoint can satisfy the model requirement if the user explicitly supplies
credentials and authorizes its cost, but a different model would change the
paper protocol. Until the remaining prerequisites are available, this method
is limited to a released-code functional reproduction and is not represented
as an exact Table I reproduction.

The source-only recovery-graph audit is complete: 29 focused upstream tests
pass in the isolated `dynamac-agentchord` environment. A full simulation rollout
is still blocked before execution. The required `dexsim_engine==0.3.11` has no
verified public HTTPS distribution, and the published plaintext package-index
command is outside the security boundary and was not contacted. See
[REPRODUCTION.md](REPRODUCTION.md) for the verified commands, environment
boundary, and upstream-owned blockers.
