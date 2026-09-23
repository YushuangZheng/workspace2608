# Native-6 Event-Grounded Evaluation

Native-6 compares RVT, RACER, and TSF on six common single-arm RLBench tasks
while retaining each system's native observations, policy, supervision, and
execution stack. The protocol matches task assignments, success conditions,
and physical perturbation semantics rather than forcing the systems through a
shared action interface.

## Protocol

- six tasks;
- 100 nominal and 100 perturbed episodes per task and system;
- actuation delay, missed interaction, and relation loss balanced in the
  perturbed manifest;
- event triggers based on simulator time and physical contact or attachment
  evidence; and
- intention-to-treat success over every scheduled episode.

The fault label, intended mismatch, repair target, and future outcome are not
available to a policy. Eligibility, injection, physical effect, and final task
success are recorded separately by the common result contract.

## Implementation

```text
native6_v3/
├── physical_clock.py  simulator-step and simulation-time recording
├── events.py          event-trigger state machines
├── gate.py            development and result validation
└── result.py          shared result schema validation
```

The source checkout includes the simulator-facing physical adapter used by the
TSF runner. Native RVT and RACER execution stacks, machine identities, and
checkpoints are supplied separately. Compatibility adapters may implement the
common physical event definition, but may not alter a native system's model,
action targets, observation inputs, or nominal controller semantics.

## Results

The accepted comparison contains 3,600 episodes across the three systems. Its
deterministic aggregate and manuscript inputs are:

```text
results/native_v3/derived/E6_ANALYSIS.json
results/native_v3/derived/table2_native_systems.csv
results/native_v3/derived/appendix_native_complete.csv
```

The top-level integrity check validates endpoint counts, result identities, and
derived-file hashes:

```bash
python -m evaluations.iclr2027.analysis.a6_acceptance status
```
