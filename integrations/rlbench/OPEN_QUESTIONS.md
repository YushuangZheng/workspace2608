# Open implementation questions

The following details remain necessary for an exact reproduction. Items already
resolved by the author response, the papers, or the pinned TAPAS implementation
are intentionally omitted.

## Highest impact

1. **Exact DynaBench protocol for Tables I and III**

   Could you share the exact DynaBench configuration or patch used for Tables I
   and III? For each task, which root or objects were moved, how the two valid
   configurations were sampled, the intervention onset and number of
   transformations, the smooth interpolation duration, teleportation semantics,
   and validation arguments would be especially helpful. For HandOver
   coordination, did you use the public `BimanualHandoverItemDynamic` variant,
   and was the perturbation applied to the commanded target or to the
   executed/measured arm pose? We would also need its coordinate frame, axis,
   magnitude, onset, duration, and whether it persisted.

2. **Final alignment and frame-selection artifacts**

   Could you share the final boundary indices or alignment plots and selected
   frame sets for SweepDust and LiftTray and, if available, WipeDesk? We
   especially need the task-specific velocity/gripper thresholds, stop
   clustering, endpoint exclusion and event-merging settings, and whether
   LiftTray and SweepDust used independent segmentation or the shared union. Our
   fixed V3 HandOver cohort reaches 171/200 static, 170/200 under the left-arm
   coordination perturbation, 10/200 under the right-arm perturbation, and
   167/200 under the environment teleport. SweepDust reaches 199/200 static but
   39/200 under the environment teleport. Because the reported perturbation
   protocol is unpublished, these dynamic values are non-comparable
   diagnostics; final selected-frame masks and compact configurations would
   help distinguish protocol/cohort differences from implementation effects.
   For WipeDesk, was the complete back-and-forth wipe retained as one skill, or
   were direction reversals separate boundaries? An aligned SweepDust gripper
   trace would also help verify the contact-stage alignment.

## Important numerical details

3. **Covariance regularization and frame-selection edge cases**

   For the reported runs, was the covariance ridge `1e-6` or `1e-5`, and was it
   constant across tasks and arms? Did Equation (6) use the same weighted 3D
   position subspace as Equation (5), or the full 6D pose covariance? Was
   `tau_omega = 0.5` used for every task? Did
   any skill produce no frame under the strict condition
   `omega(f) > tau_omega`; if so, what did the implementation do? Did the
   reported models use temporal-variance filtering? After the raw per-time-step
   Equation (5) mask was computed, did the reported policy use a constant
   skill-majority mask, use a strict majority only to enable the raw mask, or
   read the raw mask directly at every time step? Local V3 freezes the second
   interpretation and keeps temporal-variance filtering disabled.

## Exact-reproduction details

4. **Controller and failed-action configuration**

   Which exact absolute-EE action-mode constructor, collision settings, IK
   fallback parameters, and gripper actuation velocity were used?
   In particular, did evaluation use the demonstration generator's `0.04`
   gripper velocity or the public discrete action mode's `0.2`? When a primary
   command raised `InvalidAction`, did the reported evaluator terminate,
   advance the policy clock, or abort it and recompute the same tick from a
   fresh observation, and how many such controller attempts were allowed?
   Separately, when a command executed but the intended grasp contact was not
   established, did the fixed skill schedule continue, or was there any
   contact-conditioned phase hold or semantic re-grasp? Our local limit of
   three attempts applies only to `InvalidAction`; it is not assumed to be a
   DynaMAC grasp-retry mechanism. Did the reported evaluator include any final
   physics settling after both policy arms completed, and if so, for how many
   steps and with what terminal checks?

5. **Experiment manifest**

   If convenient, could you share the five demonstration IDs/seeds, task
   variations, evaluation seeds and horizon, or a redacted configuration dump
   for the reported simulator tables?
