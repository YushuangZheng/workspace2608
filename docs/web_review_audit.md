# Audit of the external GPT review

The review was checked against the current source, frozen data, and saved rollout traces. Its observations were not
treated as facts until reproduced locally.

## Confirmed and corrected

1. **The old 60 mm 3-D success radius was scientifically misleading.** The target command has `z = 0.08 m`, while
   a released cube rests with its center at about `z = 0.021 m`. Successful placements therefore had a nearly fixed
   59 mm vertical residual even when XY error was only 1.5--6.1 mm. Success now requires XY error below 10 mm, the
   object on the demonstrated support height, an open gripper, and low displacement/velocity over the final 25
   control steps. Results also report 5/10/20 mm XY sensitivity and retain 3-D error only as a diagnostic.
2. **Resume cache validation was incomplete.** Each trial now stores and validates a SHA-256 fingerprint covering
   the Git commit, frozen dataset, relevant source files, method, condition, seed, perturbation implementation, all
   success thresholds, rollout length, action-rate limit, and the diffusion checkpoint when applicable. Old result
   files without this fingerprint are invalidated automatically.
3. **The phase-4 to phase-5 command switch was discontinuous.** In the reproduced seed-6202 Full DynaMAC static
   trial, the raw desired-position jump was about 406 mm. A common Cartesian command-rate limiter now bounds every
   policy to 20 mm per control step. Raw policy intent, limited policy command, post-perturbation command, EE speed,
   and frame-switch diagnostics are stored separately. `scripts/analyze_action_transitions.py` produces the JSON
   audit and plots.
4. **A terminated rollout used a pre-step final error.** The worker now reads the scene after the last executed step
   for every rollout. The normal evaluation horizon remains shorter than the environment time limit, preventing a
   routine timeout autoreset from contaminating that read.
5. **The old documentation overstated a one-seed path result.** The result is now reported as a
   three-evaluation-seed mean, with an explicit warning that the path benefit is seed-dependent. The strict rerun
   still reverses slightly for seed 6202 after command-rate limiting.

## Checked and not reproduced as defects

- The virtual frame is defined independently for every training demonstration at the first phase-4 sample, exactly
  as the review recommends. At runtime it is reset per episode and captured again on the phase-4 transition.
- Evaluation performs exactly one `env.step()` per policy action. Trace rows intentionally contain the pre-action
  state and the action about to execute; the final post-action scene state is read separately.
- The reported inference metric times only `policy.act()`. Documentation calls it policy computation time rather
  than end-to-end control latency.

## Remaining claim boundaries

- The connection detector is deliberately a 3-D positional coupling proxy with gripper gating, not the paper's full
  6-D precision-matrix detector. It showed no false positives in this task, but robustness to rotational slip remains
  future work.
- The three seeds vary evaluation initialization while all models use the same frozen five-demo training set. They
  measure test-time robustness for that dataset, not variance across independently sampled five-demo training sets.
  No 5-shot sample-efficiency claim should be made without multiple frozen training datasets.
- Bimanual results are still one-seed pilots using geometric virtual attachment, not contact-rich grasp validation.

## Corrected regression result

The corrected matrix contains 72/72 complete rollouts over seeds `6200--6202`. Under the 10 mm semantic criterion,
World Gaussian and Static Multi-stream score 0/18, while Mask-only and Full DynaMAC score 18/18 with 3.01 and
2.98 mm mean XY error. Both dynamic methods score 17/18 at 5 mm and 18/18 at 10/20 mm. Full retains an 8.5% lower
mean path length than Mask-only, with the documented seed-6202 reversal. Across Full trials, the largest raw desired
jump is about 406 mm, the largest rate-limited policy jump is 20 mm, and maximum measured EE speed is about 1.01 m/s.
