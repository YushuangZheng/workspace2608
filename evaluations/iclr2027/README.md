# TSF Evaluation Suite

This directory contains the controlled evaluation code for Task-State Feedback
(TSF). Runtime monitor inputs, evaluator-only physical audit data, and execution
bookkeeping are stored separately so that a method cannot observe the injected
fault label or future episode outcome.

## Current experiment set

| Suite | Evaluation question | Retained outputs |
|---|---|---|
| E1 | Does TSF improve task success under perturbations while preserving nominal performance on the ten-task shared-backbone suite? | per-task nominal/perturbed success, paired tests, Table 1 |
| E2 | How do the monitors trade detection quality, delay, and false interventions, and how much does state-aware recovery add beyond retry? | shadow monitoring, fault-family metrics, recovery funnel, Figure 3 |
| E3 | How does monitoring generalize with limited or held-out failure supervision, and how does closed-loop execution respond to stronger or composed events? | failure-budget and LOFO shadow scores, severity/composition success, oracle-timing feasibility, Figure 4 |
| E4 | Does robustness persist as tasks contain more sequential interactions and more disturbance opportunities? | nested per-stage evaluation, completion after recovery, Figure 5 |
| E5 | Which state representation, alignment, recovery, and execution components account for the result? | core and fine-grained ablations, threshold sensitivity, control-equivalence diagnostics, Table 3 and appendix |
| E6 | How does the complete TSF system compare with native RVT and RACER systems on their common six-task subset? | nominal/perturbed success and per-task retention, Table 2 |

Three additional checks strengthen the interpretation of these suites:

- **calibration-matched monitoring:** monitor recall and delay at independently
  selected false-intervention budgets for Main-10 and leave-one-family-out
  evaluation;
- **paired recovery paths:** explicit stage denominators, common-trigger
  intersections, and path-by-outcome counts for retry, same-state continuation,
  and legal re-entry; and
- **nearest-state inference:** a paired four-task ablation that retains the TSF
  repair and re-entry interface while replacing coupled relation--progress
  inference with a one-hot nearest-demonstration state.

Their machine-readable results are under
`results/reviewer_decisive/{equal_fi,recovery_paths,simple_state}/derived/`.

## Controlled methods

| ID | System |
|---|---|
| M0 | frozen DynaMAC motion policy |
| M1 | DynaMAC with no-progress skill restart |
| M2 | trajectory-likelihood monitor with shared skill retry |
| M3 | FAIL-Detect with shared skill retry |
| M4 | supervised GRU failure monitor with shared skill retry |
| M5 | complete TSF with relation repair and legal re-entry |
| M6 | TSF monitor with shared skill retry |

M4 is evaluated over three training seeds where the result represents a learned
monitor. Shadow evaluation passes actions through unchanged; closed-loop
evaluation uses the recovery executor specified by the method configuration.

## Directory layout

```text
analysis/        deterministic aggregation and statistical analysis
audit/           independent physical-event and horizon audit
calibration/     normal-only monitor and boundary calibration
configs/         method, monitor, fault, and shared protocol configuration
interfaces/      stable schemas shared by runners and native systems
manifests/       paired episode populations and immutable indexes
methods/         controlled monitoring baselines and adapters
native6_v3/      common event-grounded native-system protocol
recovery/        shared generic-retry baseline
runners/         episode, shadow, horizon, native, and replay launchers
tests/           protocol and interface regression tests
```

`datasets/`, `artifacts/`, and `results/` are runtime inputs and outputs rather
than Python packages. A minimal source or reviewer-demo distribution does not
need to contain the full rollout archive.

## Data contract

Each cycle record contains three sibling payloads:

- `feature`: causal information available to a runtime method;
- `audit`: evaluator-only physical state and event labels; and
- execution bookkeeping used to validate pairing and implementation identity.

Episode manifests bind task, variation, seed, condition, perturbation
assignment, horizon, and pair identifier independently of method. A
perturbation counts as physically triggered only after its eligibility and
effect predicates are satisfied. Every scheduled episode remains in the
intention-to-treat denominator, including infrastructure failures.

## Running a development evaluation

Generate and validate the controlled manifests:

```bash
python -m evaluations.iclr2027.manifests.build
pytest -q evaluations/iclr2027/tests
```

The generic launcher executes one isolated simulator process per manifest row:

```bash
python -m evaluations.iclr2027.runners.launch \
  --manifest evaluations/iclr2027/manifests/main10_development.jsonl \
  --output-root /path/to/development-results \
  --workers 8
```

Use development manifests before retained evaluation. Native-system manifests
have separate contracts because RVT and RACER preserve their own observations,
action policies, and execution stacks. Their machine-specific adapters and
runtime identities are supplied separately from this source checkout.

## Analysis and integrity checks

The retained E1--E6 analyses are complete when this read-only command reports
`ready_for_acceptance: true`:

```bash
python -m evaluations.iclr2027.analysis.a6_acceptance status
```

The three supplemental checks expose their own read-only status commands:

```bash
python -m evaluations.iclr2027.analysis.decisive_equal_fi status
python -m evaluations.iclr2027.analysis.decisive_equal_fi_lofo status
python -m evaluations.iclr2027.analysis.decisive_recovery_paths status
python -m evaluations.iclr2027.analysis.decisive_simple_state status
```

Derived tables and plot inputs are generated from episode records rather than
edited by hand. The primary aggregate files are:

```text
results/controlled/e1_e2/derived/
results/controlled/e3/derived/
results/controlled/e4/derived/
results/controlled/e5/derived/
results/native_v3/derived/
results/reviewer_decisive/*/derived/
```

## Adding a method

1. Implement the monitor or adapter under `methods/`.
2. Register it in `methods/registry.py` and add a configuration under
   `configs/methods/`.
3. Consume only the causal `feature` payload.
4. Select an existing recovery executor or define a new one explicitly.
5. Add interface, pairing, and audit tests before running a development
   manifest.

Do not place model weights, raw cycle traces, videos, local simulator paths, or
credentials in the tracked source tree.
