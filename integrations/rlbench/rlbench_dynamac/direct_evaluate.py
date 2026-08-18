"""Run direct RLBench evaluation against the current DynaMAC policy worker.

The module is Python 3.8 compatible and imports RLBench only after argument
parsing, so ``--help`` also works without launching CoppeliaSim.  The simulator
uses absolute world-frame end-effector IK for both Panda arms.  Policy math is
kept in the Python 3.10 worker started by this process.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import select
import subprocess
from pathlib import Path

import numpy as np

from .eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
    fixed_environment_plans,
    validate_formal_artifact_paths,
)
from .records import atomic_json, reserve_output
from .runtime import (
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
    DiscreteGripperProtocol,
    PrimaryActionRetryBudget,
    ScenarioController,
    execute_joint_target_control,
    final_settling_metadata,
    initialize_fresh_task_generation,
    run_final_settling,
    stage_scenario_motion_plan,
    staged_motion_plan_batch,
)
from .v3_protocol import (
    load_v3_intervention_protocol,
    load_v3_motion_source_protocol,
    resolve_authenticated_v3_trigger,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3"
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v3"
DEFAULT_POLICY_PYTHON = Path(
    os.environ.get(
        "DYNAMAC_POLICY_PYTHON",
        "python3.10",
    )
)
TASKS = {
    "bimanual_put_bottle_in_fridge": (
        "rlbench.bimanual_tasks.bimanual_put_bottle_in_fridge",
        "BimanualPutBottleInFridge",
    ),
    "bimanual_handover_item": (
        "rlbench.bimanual_tasks.bimanual_handover_item",
        "BimanualHandoverItem",
    ),
    "bimanual_lift_tray": (
        "rlbench.bimanual_tasks.bimanual_lift_tray",
        "BimanualLiftTray",
    ),
    "bimanual_sweep_to_dustpan": (
        "rlbench.bimanual_tasks.bimanual_sweep_to_dustpan",
        "BimanualSweepToDustpan",
    ),
}

SCENARIO_KINDS = {
    "static": "static",
    "smooth": "smooth_task_motion",
    "teleport": "teleport_task",
}
DYNAMIC_EPISODE_ACCOUNTING_SCHEMA = (
    "planned-denominator-trigger-completion-conditional-success-v3"
)

POLICY_CLOCK_SEMANTICS_ID = (
    "policy-tick-transaction-commit-on-primary-action-success-v1"
)
GRIPPER_PROTOCOL = DiscreteGripperProtocol(bimanual=True)


def evaluation_protocol_id(max_primary_action_attempts):
    attempts = PrimaryActionRetryBudget(max_primary_action_attempts).max_attempts
    return GRIPPER_PROTOCOL.extend_evaluation_protocol_id(
        "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-"
        "noop-retry-same-policy-tick-fresh-observation-"
        f"primary-attempt{attempts}-committed-dynamic-clock-"
        "final-settle-up-to-raw10-staged34-deterministic-source-reset1-"
        "formal-root-state-audit2-contact-delta-diagnostic-v3"
    )


EVALUATION_PROTOCOL_ID = evaluation_protocol_id(
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
)


def _validate_v3_protocol_budgets(args):
    """Fail before staging if any frozen V3 budget was overridden."""

    protocol = load_v3_intervention_protocol()
    motion_protocol = load_v3_motion_source_protocol()
    if args.scenario_steps != protocol["smooth_steps"]:
        raise RuntimeError(
            "V3 smooth-step budget differs from the frozen intervention protocol"
        )
    if args.scenario_max_attempts != motion_protocol["goal_sampling_max_attempts"]:
        raise RuntimeError(
            "V3 goal-sampling budget differs from the frozen motion-source protocol"
        )
    if args.final_settling_steps != protocol["final_settling_physics_steps"]:
        raise RuntimeError(
            "V3 final-settling budget differs from the frozen intervention protocol"
        )
    return protocol


def _authenticated_v3_dynamic_trigger(args, worker):
    """Resolve the preregistered tick from authenticated checkpoint evidence."""

    protocol = _validate_v3_protocol_budgets(args)
    authentication = resolve_authenticated_v3_trigger(
        worker.model_identity,
        task=args.task,
    )
    authenticated_step = authentication["trigger_step"]
    requested_step = getattr(args, "scenario_trigger_step", None)
    if requested_step is not None and requested_step != authenticated_step:
        raise RuntimeError(
            "command-line trigger step differs from authenticated V3 preregistration"
        )
    if authenticated_step >= worker.policy_steps:
        raise RuntimeError("authenticated V3 trigger lies outside the loaded policy clock")
    requested_reference = getattr(args, "scenario_reference_steps", None)
    if requested_reference is not None and requested_reference != worker.policy_steps:
        raise RuntimeError(
            "V3 trigger reference must equal the authenticated loaded policy clock"
        )
    return protocol, authentication


def _learned_policy_steps(models_dir, task):
    """Read the fitted bimanual clock length used to schedule interventions."""

    path = Path(models_dir) / task / "training.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    durations = []
    for arm in ("left", "right"):
        values = payload.get(arm, {}).get("durations")
        if not isinstance(values, list) or not values:
            raise ValueError(f"missing {arm} skill durations in {path}")
        durations.append(sum(int(value) for value in values))
    return max(durations)


def _make_action_mode():
    """Construct the fork's missing bimanual absolute-EE IK action mode."""

    from pyrep.errors import ConfigurationError, IKError
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        ArmActionMode,
        assert_action_shape,
        assert_unit_quaternion,
    )
    from rlbench.backend.exceptions import InvalidActionError

    class BimanualAbsoluteEndEffectorIK(ArmActionMode):
        def __init__(self):
            self._execution_diagnostics = {
                "jacobian_failures": 0,
                "sampling_fallback_successes": 0,
                "sampling_fallback_failures": 0,
                "joint_target_reached": 0,
                "joint_target_stopped": 0,
                "joint_target_timeouts": 0,
            }

        def diagnostics(self):
            return dict(self._execution_diagnostics)

        def _solve(self, arm, target):
            """Use local Jacobian IK first, then PyRep's far-target IK search.

            The author reports using IK for RLBench and reserving a heavier
            fallback for goals that are too far from the current pose.  PyRep's
            Jacobian solver is deliberately local; a single failure must not
            turn every later clock tick into a no-op.  Sampling solves the same
            absolute target without changing the policy command or clock.
            """

            try:
                return arm.solve_ik_via_jacobian(
                    target[:3], quaternion=target[3:], relative_to=None
                )
            except IKError:
                self._execution_diagnostics["jacobian_failures"] += 1
                try:
                    result = arm.solve_ik_via_sampling(
                        target[:3],
                        quaternion=target[3:],
                        ignore_collisions=True,
                        trials=100,
                        max_configs=5,
                        max_time_ms=10,
                        relative_to=None,
                    )[0]
                    self._execution_diagnostics["sampling_fallback_successes"] += 1
                    return result
                except ConfigurationError as exc:
                    self._execution_diagnostics["sampling_fallback_failures"] += 1
                    raise InvalidActionError(
                        "bimanual absolute end-effector IK failed"
                    ) from exc

        def action(self, scene, action, ignore_collisions=True):
            del ignore_collisions  # the public IK controller has no collision option
            assert_action_shape(action, (14,))
            right_action = np.asarray(action[:7], dtype=np.float64)
            left_action = np.asarray(action[7:], dtype=np.float64)
            assert_unit_quaternion(right_action[3:])
            assert_unit_quaternion(left_action[3:])
            # Solve both sides before moving either side, so a target that is
            # invalid even after the far-target fallback cannot create a
            # half-applied bimanual command.
            right_joints = self._solve(scene.robot.right_arm, right_action)
            left_joints = self._solve(scene.robot.left_arm, left_action)

            scene.robot.right_arm.set_joint_target_positions(right_joints)
            scene.robot.left_arm.set_joint_target_positions(left_joints)
            try:
                status = execute_joint_target_control(
                    scene,
                    (
                        (scene.robot.right_arm, right_joints),
                        (scene.robot.left_arm, left_joints),
                    ),
                    invalid_action_error=InvalidActionError,
                    error_message=(
                        "bimanual absolute end-effector IK did not converge within "
                        "200 simulation steps"
                    ),
                )
            except InvalidActionError:
                self._execution_diagnostics["joint_target_timeouts"] += 1
                raise
            self._execution_diagnostics[f"joint_target_{status}"] += 1

        def action_shape(self, scene):
            del scene
            return (14,)

        def unimanual_action_shape(self, scene):
            del scene
            return (7,)

    return BimanualMoveArmThenGripper(
        BimanualAbsoluteEndEffectorIK(),
        GRIPPER_PROTOCOL.make_action_mode(),
    )


def _observation_payload(observation):
    state = observation.task_low_dim_state
    if isinstance(state, tuple) and len(state) == 1:
        state = state[0]
    return {
        "left": {
            "gripper_pose": np.asarray(observation.left.gripper_pose).tolist(),
        },
        "right": {
            "gripper_pose": np.asarray(observation.right.gripper_pose).tolist(),
        },
        "task_low_dim_state": np.asarray(state).reshape(-1).tolist(),
    }


def _noop_action(observation):
    """Current right/left EE poses and gripper states in the fork's 18D layout."""

    right = np.concatenate(
        (
            np.asarray(observation.right.gripper_pose, dtype=np.float64),
            [float(observation.right.gripper_open >= 0.5), 0.0],
        )
    )
    left = np.concatenate(
        (
            np.asarray(observation.left.gripper_pose, dtype=np.float64),
            [float(observation.left.gripper_open >= 0.5), 0.0],
        )
    )
    return np.concatenate((right, left))


class PolicyProcess:
    def __init__(self, python, task, models_dir, timeout=120.0):
        self.timeout = float(timeout)
        if self.timeout <= 0.0:
            raise ValueError("policy timeout must be positive")
        command = [
            str(python),
            "-m",
            "integrations.rlbench.rlbench_dynamac.direct_policy",
            "serve",
            "--task",
            task,
            "--models-dir",
            str(Path(models_dir).resolve()),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            response = self.request("ping")
            if not response.get("ready"):
                raise RuntimeError("policy worker did not report ready")
            if not response.get("bimanual") or response.get("task") != task:
                raise RuntimeError("policy worker identity does not match the requested task")
            self.policy_steps = int(response.get("policy_steps", 0))
            if self.policy_steps < 1:
                raise RuntimeError("policy worker reported an empty trajectory")
            self.model_identity = response.get("model_identity")
            if not isinstance(self.model_identity, dict):
                raise RuntimeError("policy worker did not report model identity")
            self.policy_clock_semantics_id = response.get(
                "policy_clock_semantics_id"
            )
            if self.policy_clock_semantics_id != POLICY_CLOCK_SEMANTICS_ID:
                raise RuntimeError("policy worker clock semantics do not match evaluator")
        except Exception:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            raise

    def request(self, command, observation=None, **fields):
        request = {"command": command}
        request.update(fields)
        if observation is not None:
            request["observation"] = _observation_payload(observation)
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            self.process.terminate()
            raise TimeoutError(
                f"policy worker did not respond within {self.timeout:g} seconds"
            )
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"policy worker exited without a response (code={code!r})")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"policy worker error: {response.get('error')}")
        return response

    def close(self):
        if self.process.poll() is not None:
            return
        try:
            self.request("close")
            self.process.wait(timeout=5)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _apply_scenario(controller, task_environment, observation, *, step, horizon):
    """Apply one intervention tick and return a fresh observation if it moved.

    ``ScenarioController`` records the before/after task state for diagnostics.
    The additional observation fetch here is deliberate: the policy must see
    the relocated task in the same control tick, before it predicts an action.
    """

    event = controller.apply(task_environment, step=step, horizon=horizon)
    if event["applied"]:
        observation = task_environment.get_observation()
        event["policy_observation_refreshed"] = True
    else:
        event["policy_observation_refreshed"] = False
    return observation, event


def _trigger_control_step(reference_steps, trigger_fraction):
    if reference_steps < 1:
        raise ValueError("intervention reference steps must be positive")
    return min(
        reference_steps - 1,
        int(round(trigger_fraction * (reference_steps - 1))),
    )


def _finalize_episode_intervention_status(
    row,
    *,
    scenario,
    trigger_step,
    trigger_reached,
    smooth_steps,
):
    """Attach authenticated trigger eligibility and smooth-prefix progress."""

    events = row.get("scenario_events")
    if not isinstance(events, list):
        raise RuntimeError("episode scenario_events must be a list")
    dynamic = scenario != "static"
    if not dynamic:
        if any(
            isinstance(event, dict) and event.get("applied") is True
            for event in events
        ):
            raise RuntimeError("static episode unexpectedly applied an intervention")
        row.update(
            {
                "trigger_step": None,
                "intervention_eligible": False,
                "intervention_reached": False,
                "pre_intervention_terminal": False,
                "pre_intervention_terminal_outcome": None,
                "dynamic_condition_exercised": False,
                "dynamic_condition_unexercised": None,
                "intervention_effective": None,
                "intervention_complete": None,
            }
        )
        return row

    if not isinstance(trigger_step, int) or isinstance(trigger_step, bool):
        raise RuntimeError("dynamic episode trigger step is invalid")
    applied = [
        event
        for event in events
        if isinstance(event, dict) and event.get("applied") is True
    ]
    if any(not isinstance(event, dict) for event in events):
        raise RuntimeError("dynamic episode contains a malformed intervention event")

    preterminal = not trigger_reached
    if preterminal:
        if applied:
            raise RuntimeError("pre-trigger episode cannot contain an intervention")
        row.update(
            {
                "trigger_step": trigger_step,
                "intervention_eligible": False,
                "intervention_reached": False,
                "pre_intervention_terminal": True,
                "pre_intervention_terminal_outcome": (
                    "success" if row.get("success") is True else "failure"
                ),
                "dynamic_condition_exercised": False,
                "dynamic_condition_unexercised": True,
                "intervention_effective": None,
                "intervention_complete": None,
            }
        )
        return row

    if not applied or any(
        event.get("protocol_effective") is not True for event in applied
    ):
        raise RuntimeError(
            "dynamic episode reached its trigger without an effective intervention"
        )
    if any(
        event.get("trigger_step") != trigger_step for event in applied
    ):
        raise RuntimeError("dynamic intervention trigger evidence is inconsistent")

    complete = True
    if scenario == "teleport":
        valid = (
            len(applied) == 1
            and applied[0].get("kind") == "teleport_task"
            and applied[0].get("step") == trigger_step
        )
        if not valid:
            raise RuntimeError("teleport episode must contain one trigger-step event")
    elif scenario == "smooth":
        count = len(applied)
        if not 1 <= count <= smooth_steps:
            raise RuntimeError("smooth intervention event count is invalid")
        for index, event in enumerate(applied, start=1):
            endpoint = index == smooth_steps
            if (
                event.get("kind") != "smooth_task_motion"
                or event.get("step") != trigger_step + index - 1
                or event.get("smooth_call") != index
                or event.get("complete") is not endpoint
                or event.get("endpoint_applied") is not endpoint
            ):
                raise RuntimeError("smooth intervention is not a strict prefix")
        complete = count == smooth_steps
        if not complete:
            final_event_step = trigger_step + count - 1
            clock_at_terminal = row.get("committed_policy_steps", row.get("steps"))
            if clock_at_terminal not in {
                final_event_step,
                final_event_step + 1,
            }:
                raise RuntimeError(
                    "smooth intervention stopped despite reaching its next motion tick"
                )
    else:
        raise RuntimeError(f"unsupported dynamic scenario: {scenario}")

    row.update(
        {
            "trigger_step": trigger_step,
            "intervention_eligible": True,
            "intervention_reached": True,
            "pre_intervention_terminal": False,
            "pre_intervention_terminal_outcome": None,
            "dynamic_condition_exercised": True,
            "dynamic_condition_unexercised": False,
            "intervention_effective": True,
            "intervention_complete": complete,
        }
    )
    return row


def _run_episode(
    task_environment,
    worker,
    episode,
    seed,
    horizon,
    *,
    scenario="static",
    scenario_trigger_fraction=1.0 / 3.0,
    scenario_trigger_step=None,
    scenario_reference_steps=None,
    scenario_steps=10,
    scenario_max_attempts=100,
    max_primary_action_attempts=DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    motion_plan=None,
    final_settling_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    descriptions=None,
    observation=None,
    fresh_task_generation=None,
):
    if descriptions is None or observation is None or not isinstance(
        fresh_task_generation,
        dict,
    ):
        raise RuntimeError("formal episode requires fresh task-generation input")
    invalid_actions = 0
    retry_budget = PrimaryActionRetryBudget(max_primary_action_attempts)
    scenario_events = []
    controller = ScenarioController(
        kind=SCENARIO_KINDS[scenario],
        trigger_fraction=scenario_trigger_fraction,
        trigger_step=scenario_trigger_step,
        total_steps=scenario_steps,
        max_attempts=scenario_max_attempts,
        motion_plan=motion_plan,
    )
    if scenario_reference_steps is None:
        scenario_reference_steps = horizon
    trigger_step = (
        None
        if scenario == "static"
        else controller.resolved_trigger_step(scenario_reference_steps)
    )
    trigger_reached = False
    committed_policy_steps = 0
    last_scenario_policy_step = None
    source_binding = controller.bind_staged_source(
        task_environment,
        episode_seed=seed + episode,
        variation=episode % task_environment.variation_count(),
        descriptions=descriptions,
    )
    if motion_plan is not None:
        observation = task_environment.get_observation()
        source_binding["formal_observation_refreshed_after_binding"] = True
    else:
        source_binding["formal_observation_refreshed_after_binding"] = False
    # Authenticate formal A before the policy worker receives the episode.
    # The worker is out-of-process and cannot mutate the scene, but this order
    # makes the fail-closed boundary explicit.
    worker.request("reset", observation)

    def finish(row):
        row.setdefault(
            "final_settling",
            {
                **final_settling_metadata(final_settling_steps),
                "attempted": False,
                "available": True,
                "steps_executed": 0,
                "first_terminal_step": None,
                "stop_reason": "not_entered",
                "success": False,
                "terminate": False,
            },
        )
        row.setdefault("committed_policy_steps", committed_policy_steps)
        row.setdefault("dynamic_clock_steps", committed_policy_steps)
        row.setdefault("staged_source_binding", source_binding)
        row.setdefault("fresh_task_generation", fresh_task_generation)
        row.setdefault(
            "motion_plan_fingerprint",
            motion_plan.fingerprint() if motion_plan is not None else None,
        )
        row.setdefault(
            "motion_plan_protocol_id",
            (
                motion_plan.metadata()["protocol_id"]
                if motion_plan is not None
                else None
            ),
        )
        row.setdefault(
            "motion_plan_evidence",
            (
                {
                    "plan_fingerprint": motion_plan.fingerprint(),
                    "source_waypoint_validated": motion_plan.validation.get(
                        "source_waypoint_validated"
                    ),
                    "goal_waypoint_validated": motion_plan.validation.get(
                        "goal_waypoint_validated"
                    ),
                    "formal_rollout_sample_or_restore": motion_plan.validation.get(
                        "formal_rollout_sample_or_restore"
                    ),
                    "formal_source_bound": source_binding.get(
                        "formal_source_bound"
                    ),
                    "formal_task_name_bound": source_binding.get("task_name"),
                    "formal_task_semantics_matched": source_binding.get(
                        "task_semantics_matched"
                    ),
                    "formal_task_tree_matched": source_binding.get(
                        "task_tree_matched"
                    ),
                    "formal_deterministic_source_reconstruction_passed": (
                        source_binding.get(
                            "deterministic_source_reconstruction", {}
                        ).get("passed")
                    ),
                    "formal_task_validate_calls": source_binding.get(
                        "formal_task_validate_calls"
                    ),
                    "formal_observation_refreshed_after_binding": source_binding.get(
                        "formal_observation_refreshed_after_binding"
                    ),
                    "formal_robot_external_collision_pairs_matched": (
                        source_binding.get(
                            "robot_external_collision_pairs_matched"
                        )
                    ),
                    "selected_source_fingerprint": source_binding.get(
                        "selected_source_fingerprint"
                    ),
                    "formal_source_fingerprint": source_binding.get(
                        "formal_source_fingerprint"
                    ),
                }
                if motion_plan is not None
                else None
            ),
        )
        return _finalize_episode_intervention_status(
            row,
            scenario=scenario,
            trigger_step=trigger_step,
            trigger_reached=trigger_reached,
            smooth_steps=scenario_steps,
        )

    from rlbench.backend.exceptions import InvalidActionError

    for control_step in range(horizon):
        if last_scenario_policy_step != committed_policy_steps:
            if trigger_step is not None and committed_policy_steps >= trigger_step:
                trigger_reached = True
            observation, event = _apply_scenario(
                controller,
                task_environment,
                observation,
                step=committed_policy_steps,
                horizon=scenario_reference_steps,
            )
            last_scenario_policy_step = committed_policy_steps
            if event.get("trigger_step") != trigger_step and scenario != "static":
                raise RuntimeError("scenario controller returned an inconsistent trigger")
            if event.get("applied") and (
                trigger_step is None or committed_policy_steps < trigger_step
            ):
                raise RuntimeError("scenario controller applied before the trigger")
            if event["applied"]:
                scenario_events.append(event)
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            settling = run_final_settling(
                task_environment,
                physics_steps=final_settling_steps,
            )
            if settling["success"]:
                reason = "success_after_final_settling"
            elif settling["terminate"]:
                reason = "terminate_during_final_settling"
            elif settling["available"]:
                reason = "policy_complete_after_final_settling"
            else:
                reason = "policy_complete"
            return finish({
                "episode": episode,
                "success": bool(settling["success"]),
                "steps": control_step,
                "control_attempts": control_step,
                "reason": reason,
                "invalid_actions": invalid_actions,
                "scenario_events": scenario_events,
                "final_settling": settling,
            })
        transaction_id = response.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy worker did not return an action transaction id")
        primary_action_succeeded = False
        retry_exhausted = False
        try:
            observation, reward, terminate = task_environment.step(
                np.asarray(action, dtype=np.float64)
            )
            primary_action_succeeded = True
        except InvalidActionError:
            invalid_actions += 1
            retry_exhausted = retry_budget.record_failure()
            worker.request("abort", transaction_id=transaction_id)
            try:
                observation, reward, terminate = task_environment.step(
                    _noop_action(observation)
                )
            except InvalidActionError:
                return finish({
                    "episode": episode,
                    "success": False,
                    "steps": control_step + 1,
                    "control_attempts": control_step + 1,
                    "reason": "noop_failed",
                    "invalid_actions": invalid_actions,
                    "scenario_events": scenario_events,
                })
        except Exception:
            worker.request("abort", transaction_id=transaction_id)
            raise
        if primary_action_succeeded:
            commit = worker.request("commit", transaction_id=transaction_id)
            retry_budget.record_success()
            policy_complete = bool(commit.get("complete"))
            committed_policy_steps += 1
        else:
            policy_complete = False
        settling = None
        if reward > 0.0:
            reason = "success"
        elif terminate:
            reason = "terminate"
        elif retry_exhausted:
            reason = "primary_action_retry_exhausted"
        elif policy_complete:
            settling = run_final_settling(
                task_environment,
                physics_steps=final_settling_steps,
            )
            if settling["success"]:
                reason = "success_after_final_settling"
            elif settling["terminate"]:
                reason = "terminate_during_final_settling"
            elif settling["available"]:
                reason = "policy_complete_after_final_settling"
            else:
                reason = "policy_complete"
        else:
            continue
        return finish({
            "episode": episode,
            "success": bool(reward > 0.0 or (settling or {}).get("success")),
            "steps": control_step + 1,
            "control_attempts": control_step + 1,
            "reason": reason,
            "invalid_actions": invalid_actions,
            **(
                {"primary_action_attempts": retry_budget.attempts}
                if reason == "primary_action_retry_exhausted"
                else {}
            ),
            "scenario_events": scenario_events,
            **({"final_settling": settling} if settling is not None else {}),
        })
    return finish({
        "episode": episode,
        "success": False,
        "steps": horizon,
        "control_attempts": horizon,
        "reason": "horizon",
        "invalid_actions": invalid_actions,
        "scenario_events": scenario_events,
    })


def _stage_motion_plan_batch(args, task_class):
    """Generate every A/B plan in this process's disposable Environment."""

    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    environment = Environment(
        action_mode=_make_action_mode(),
        obs_config=observation_config,
        headless=args.headless,
        robot_setup="dual_panda",
    )
    launched = False
    plans = []
    variations = []
    try:
        environment.launch()
        launched = True
        variation_count = task_class(
            environment._pyrep,
            environment._robot,
        ).variation_count()
        for episode in range(args.episodes):
            variation = episode % variation_count
            plans.append(
                stage_scenario_motion_plan(
                    environment,
                    task_class,
                    task_name=args.task,
                    episode_seed=args.seed + episode,
                    variation=variation,
                    max_attempts=args.scenario_max_attempts,
                )
            )
            variations.append(variation)
            print(
                f"staged {args.task} A/B {episode + 1}/{args.episodes}",
                flush=True,
            )
    finally:
        if launched:
            environment.shutdown()
    payload = staged_motion_plan_batch(
        task_name=args.task,
        base_seed=args.seed,
        variations=variations,
        plans=plans,
    )
    atomic_json(args.stage_motion_plans_output, payload)
    return payload


def _motion_plan_cache_path(args):
    if getattr(args, "motion_plans", None) is not None:
        return Path(args.motion_plans)
    return DEFAULT_RESULTS_DIR / "motion_plans" / (
        f"{args.task}_seed{args.seed}_n{args.episodes}_v34.json"
    )


def _load_fixed_motion_plans(args):
    """Read the preregistered evaluation set; formal runs never generate it."""

    if args.eval_set_id is None:
        raise RuntimeError("formal evaluation requires --eval-set-id")
    if args.motion_plans is not None:
        raise RuntimeError("--motion-plans is not allowed for fixed formal evaluation")
    if args.seed != GLOBAL_EVAL_SEED_START or args.episodes != FIXED_EVAL_EPISODES:
        raise RuntimeError(
            "formal evaluation seed/episode count differs from the fixed eval set"
        )
    manifest, selected = fixed_environment_plans(args.eval_set_id, args.task)
    plans = selected["plans"]
    if any(
        plan.validation.get("goal_sampling_max_attempts")
        != args.scenario_max_attempts
        for plan in plans
    ):
        raise RuntimeError("fixed eval-set goal-sampling budget is inconsistent")
    return manifest, selected


def evaluate(args):
    validate_formal_artifact_paths(output=args.output, models_dir=args.models_dir)
    with reserve_output(args.output):
        return _evaluate_reserved(args)


def _evaluate_reserved(args):
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    # Do this before spawning the staging process so an invalid CLI override
    # can never populate the canonical V3 plan cache.
    _validate_v3_protocol_budgets(args)
    module_name, class_name = TASKS[args.task]
    task_class = getattr(importlib.import_module(module_name), class_name)
    eval_set, selected_batch = _load_fixed_motion_plans(args)
    motion_plan_payload = selected_batch["payload"]
    motion_plans = selected_batch["plans"]
    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    action_mode = _make_action_mode()
    environment = Environment(
        action_mode=action_mode,
        obs_config=observation_config,
        headless=args.headless,
        robot_setup="dual_panda",
    )
    worker = PolicyProcess(
        args.policy_python,
        args.task,
        args.models_dir,
        timeout=args.policy_timeout,
    )
    results = []
    launched = False
    try:
        intervention_registry, trigger_authentication = (
            _authenticated_v3_dynamic_trigger(args, worker)
        )
        scenario_reference_steps = worker.policy_steps
        scenario_reference_source = (
            "AUTHENTICATED_LOADED_CHECKPOINT_POLICY_STEPS"
            if args.scenario_reference_steps is None
            else "COMMAND_LINE_MATCHED_AUTHENTICATED_LOADED_CHECKPOINT_POLICY_STEPS"
        )
        resolved_trigger_step = (
            None
            if args.scenario == "static"
            else trigger_authentication["trigger_step"]
        )
        environment.launch()
        launched = True
        variation_count = task_class(
            environment._pyrep,
            environment._robot,
        ).variation_count()
        for episode in range(args.episodes):
            variation = episode % variation_count
            episode_motion_plan = (
                motion_plans[episode]
            )
            reset_seed = (
                episode_motion_plan.validation["source_seed"]
                if episode_motion_plan is not None
                else args.seed + episode
            )
            (
                task_environment,
                descriptions,
                observation,
                fresh_task_generation,
            ) = initialize_fresh_task_generation(
                environment,
                task_class,
                episode_seed=reset_seed,
                variation=variation,
                verify_instance=False,
            )
            result = _run_episode(
                task_environment,
                worker,
                episode,
                args.seed,
                args.horizon,
                scenario=args.scenario,
                scenario_trigger_fraction=args.scenario_trigger_fraction,
                scenario_trigger_step=resolved_trigger_step,
                scenario_reference_steps=scenario_reference_steps,
                scenario_steps=args.scenario_steps,
                scenario_max_attempts=args.scenario_max_attempts,
                max_primary_action_attempts=args.max_primary_action_attempts,
                motion_plan=episode_motion_plan,
                final_settling_steps=args.final_settling_steps,
                descriptions=descriptions,
                observation=observation,
                fresh_task_generation=fresh_task_generation,
            )
            results.append(result)
            successes = sum(item["success"] for item in results)
            success_rate = 100.0 * successes / len(results)
            print(
                f"{args.task} episode {episode + 1}/{args.episodes}: "
                f"{result['reason']} (success {success_rate:.1f}%)",
                flush=True,
            )
    finally:
        worker.close()
        if launched:
            environment.shutdown()

    applied_by_episode = [
        any(event["applied"] for event in item["scenario_events"]) for item in results
    ]
    effective_by_episode = [item["intervention_effective"] is True for item in results]
    eligible_by_episode = [item["intervention_eligible"] is True for item in results]
    preterminal_by_episode = [
        item["pre_intervention_terminal"] is True for item in results
    ]
    complete_by_episode = [
        item["intervention_complete"] is True for item in results
    ]
    complete_results = [
        item for item in results if item["intervention_complete"] is True
    ]
    eligible_effective = [
        item["intervention_effective"] is True
        for item in results
        if item["intervention_eligible"] is True
    ]
    covered_by_episode = [
        item["pre_intervention_terminal"] is True
        or (
            item["intervention_eligible"] is True
            and item["intervention_effective"] is True
        )
        for item in results
    ]
    motion_protocol = ScenarioController(
        SCENARIO_KINDS[args.scenario],
        trigger_fraction=args.scenario_trigger_fraction,
        trigger_step=resolved_trigger_step,
        total_steps=args.scenario_steps,
        max_attempts=args.scenario_max_attempts,
        motion_plan=(motion_plans[0] if motion_plans else None),
    ).protocol_metadata()
    summary = {
        "task": args.task,
        "scenario": args.scenario,
        "scenario_protocol": {
            "status": (
                "STATIC_REFERENCE"
                if args.scenario == "static"
                else "V3_PREREGISTERED_CHECKPOINT_AUTHENTICATED"
            ),
            "motion_kind": SCENARIO_KINDS[args.scenario],
            "motion_protocol": motion_protocol,
            "legacy_trigger_fraction_ignored": args.scenario_trigger_fraction,
            "trigger_reference_steps": scenario_reference_steps,
            "trigger_reference_source": scenario_reference_source,
            "trigger_reference_domain": (
                "successfully_committed_policy_ticks"
            ),
            "trigger_policy_step": resolved_trigger_step,
            "trigger_authentication": trigger_authentication,
            "intervention_registry_schema": intervention_registry["schema"],
            "intervention_registry_fingerprint": intervention_registry["fingerprint"],
            "smooth_interpolation_calls": (
                args.scenario_steps if args.scenario == "smooth" else None
            ),
            "max_sampling_attempts": args.scenario_max_attempts,
            "observation_refreshed_before_policy_action": True,
            "dynamic_episode_accounting_schema": (
                DYNAMIC_EPISODE_ACCOUNTING_SCHEMA
            ),
            "pre_intervention_failure_policy": (
                "retain_failure_with_null_intervention_effectiveness"
            ),
            "pre_intervention_success_policy": (
                "retain_success_in_planned_denominator_with_unexercised_condition"
            ),
            "smooth_terminal_progress_policy": (
                "strict_effective_prefix_until_episode_terminal"
            ),
            "episodes_intervention_eligible": sum(eligible_by_episode),
            "episodes_pre_intervention_terminal": sum(preterminal_by_episode),
            "episodes_dynamic_condition_unexercised": sum(
                item.get("dynamic_condition_unexercised") is True
                for item in results
            ),
            "pre_trigger_successes": sum(
                item.get("pre_intervention_terminal") is True
                and item.get("success") is True
                for item in results
            ),
            "planned_episode_denominator": args.episodes,
            "completed_episode_count": len(results),
            "episodes_with_intervention": sum(applied_by_episode),
            "episodes_with_effective_intervention": sum(effective_by_episode),
            "episodes_with_complete_intervention": sum(complete_by_episode),
            "successes_in_complete_intervention_subset": sum(
                int(item["success"]) for item in complete_results
            ),
            "success_rate_in_complete_intervention_subset": (
                sum(int(item["success"]) for item in complete_results)
                / float(len(complete_results))
                if complete_results
                else None
            ),
            "all_episodes_intervened": all(applied_by_episode),
            "all_interventions_effective": (
                all(
                    item["intervention_effective"] is True
                    for item, applied in zip(results, applied_by_episode)
                    if applied
                )
                if args.scenario != "static"
                else None
            ),
            "all_eligible_interventions_effective": (
                all(eligible_effective) if args.scenario != "static" else None
            ),
            "protocol_valid": (
                True
                if args.scenario == "static"
                else all(covered_by_episode)
            ),
            "paper_comparable": args.scenario == "static",
            "staged_motion_plan_cache": (
                {
                    "schema": motion_plan_payload["schema"],
                    "protocol_id": motion_plan_payload["protocol_id"],
                    "batch_fingerprint": motion_plan_payload["batch_fingerprint"],
                    "plan_fingerprints": [plan.fingerprint() for plan in motion_plans],
                    "scenario_independent": True,
                    "seed_domain": motion_plan_payload["seed_domain"],
                    "goal_sampling_max_attempts": args.scenario_max_attempts,
                    "source_selection_max_attempts": 20,
                    "motion_source_protocol_schema": motion_plans[0].validation[
                        "motion_source_protocol_schema"
                    ],
                    "motion_source_protocol_fingerprint": motion_plans[0].validation[
                        "motion_source_protocol_fingerprint"
                    ],
                    "formal_dynamic_reset_verify_instance": False,
                    "cache_key": {
                        "task": args.task,
                        "base_seed": args.seed,
                        "episodes": args.episodes,
                        "variation_schedule": motion_plan_payload[
                            "variation_schedule"
                        ],
                    },
                    "formal_access": "canonical_eval_set_read_only",
                    "staging_shutdown_before_formal_launch": True,
                    "fresh_task_generation_per_formal_episode": True,
                }
                if motion_plan_payload is not None
                else None
            ),
        },
        "episodes": args.episodes,
        "episodes_requested": args.episodes,
        "episodes_completed": len(results),
        "seed": args.seed,
        "horizon": args.horizon,
        "variation_count": variation_count,
        "variation_schedule": [
            episode % variation_count for episode in range(args.episodes)
        ],
        "evaluation_protocol_id": evaluation_protocol_id(
            args.max_primary_action_attempts
        ),
        "fixed_eval_set": {
            "evaluation_set_id": eval_set["payload"]["evaluation_set_id"],
            "manifest_sha256": eval_set["manifest_sha256"],
            "spec_sha256": eval_set["payload"]["spec"]["sha256"],
            "selected_batch_sha256": eval_set["payload"][
                "environment_plan_batches"
            ][args.task]["sha256"],
            "selected_batch_fingerprint": motion_plan_payload[
                "batch_fingerprint"
            ],
            "formal_access": "canonical_id_read_only_no_generation",
        },
        "controller": {
            "command": "absolute_world_end_effector_pose",
            "primary_ik": "jacobian",
            "fallback_ik": "sampling",
            "sampling_trials": 100,
            "sampling_max_configs": 5,
            "sampling_max_time_ms": 10,
            "sampling_ignore_collisions": True,
            "joint_target_max_steps": 200,
            "failed_action": "abort_policy_target_then_current_pose_current_gripper_noop",
            "failed_action_next_tick": "retry_same_policy_tick_from_fresh_observation",
            "primary_action_retry": PrimaryActionRetryBudget(
                args.max_primary_action_attempts
            ).metadata(),
            "policy_clock_rollback": True,
            "policy_clock_semantics_id": worker.policy_clock_semantics_id,
            "dynamic_clock_semantics": "advance_only_after_policy_commit",
            "formal_episode_initialization": DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
            "final_settling": final_settling_metadata(args.final_settling_steps),
            "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "successes": sum(item["success"] for item in results),
        "success_rate": sum(item["success"] for item in results) / float(args.episodes),
        "episode_accounting": {
            "schema": DYNAMIC_EPISODE_ACCOUNTING_SCHEMA,
            "planned_episode_denominator": args.episodes,
            "completed_episode_count": len(results),
            "successes_in_planned_denominator": sum(
                int(item["success"]) for item in results
            ),
            "success_rate_all_planned_episodes": (
                sum(int(item["success"]) for item in results) / float(args.episodes)
            ),
            "trigger_reached_count": sum(
                item["intervention_reached"] is True for item in results
            ),
            "intervention_complete_count": sum(complete_by_episode),
            "dynamic_condition_unexercised_count": sum(
                item.get("dynamic_condition_unexercised") is True
                for item in results
            ),
            "pre_trigger_success_count": sum(
                item.get("pre_intervention_terminal") is True
                and item.get("success") is True
                for item in results
            ),
            "complete_intervention_subset_count": len(complete_results),
            "successes_in_complete_intervention_subset": sum(
                int(item["success"]) for item in complete_results
            ),
            "success_rate_in_complete_intervention_subset": (
                sum(int(item["success"]) for item in complete_results)
                / float(len(complete_results))
                if complete_results
                else None
            ),
        },
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
        "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
        "final_settling_protocol": final_settling_metadata(
            args.final_settling_steps
        ),
        "motion_plan_batch_fingerprint": (
            motion_plan_payload["batch_fingerprint"]
            if motion_plan_payload is not None
            else None
        ),
        "results": results,
        "fresh_task_generation": {
            "required_per_formal_episode": True,
            "all_episodes_recorded": all(
                isinstance(item.get("fresh_task_generation"), dict)
                for item in results
            ),
            "evidence": [
                item["fresh_task_generation"] for item in results
            ],
        },
    }
    atomic_json(args.output, summary)
    print(f"wrote {args.output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=GLOBAL_EVAL_SEED_START)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_KINDS),
        default="static",
        help="Evaluation condition; dynamic modes are local approximations of Table III.",
    )
    parser.add_argument(
        "--scenario-trigger-fraction",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Legacy audit value only; V3 uses the authenticated task-specific "
            "absolute trigger tick."
        ),
    )
    parser.add_argument(
        "--scenario-trigger-step",
        type=int,
        default=None,
        help="Explicit committed-policy trigger tick; V3 task profiles set this.",
    )
    parser.add_argument(
        "--scenario-reference-steps",
        type=int,
        default=None,
        help=(
            "Optional V3 assertion; when supplied it must equal the loaded "
            "checkpoint policy length."
        ),
    )
    parser.add_argument("--scenario-steps", type=int, default=10)
    parser.add_argument("--scenario-max-attempts", type=int, default=100)
    parser.add_argument(
        "--final-settling-steps",
        type=int,
        default=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    )
    parser.add_argument(
        "--motion-plans",
        type=Path,
        default=None,
        help="Scenario-independent staged A/B plan cache shared by smooth/teleport.",
    )
    parser.add_argument(
        "--eval-set-id",
        default=None,
        help="Canonical immutable evaluation-set ID (required for formal runs).",
    )
    parser.add_argument(
        "--motion-plan-wait-timeout",
        type=float,
        default=86_400.0,
        help="Seconds to wait when another evaluator is staging the shared cache.",
    )
    parser.add_argument(
        "--stage-motion-plans-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-primary-action-attempts",
        type=int,
        default=DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
        help="Maximum primary InvalidAction attempts for one policy clock tick.",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage_motion_plans_output is not None:
        _validate_v3_protocol_budgets(args)
        if args.episodes < 1 or args.seed < 0 or args.scenario_max_attempts < 1:
            raise ValueError("staging episodes/attempts must be positive")
        module_name, class_name = TASKS[args.task]
        task_class = getattr(importlib.import_module(module_name), class_name)
        with reserve_output(args.stage_motion_plans_output):
            _stage_motion_plan_batch(args, task_class)
        return 0
    if args.episodes < 1 or args.horizon < 1:
        raise ValueError("episodes and horizon must be positive")
    if args.motion_plan_wait_timeout <= 0.0:
        raise ValueError("motion-plan wait timeout must be positive")
    if not 0.0 <= args.scenario_trigger_fraction <= 1.0:
        raise ValueError("scenario trigger fraction must lie in [0, 1]")
    if (
        args.scenario_steps < 1
        or args.scenario_max_attempts < 1
        or args.max_primary_action_attempts < 1
        or args.final_settling_steps < 0
    ):
        raise ValueError("scenario steps and max attempts must be positive")
    if args.scenario_reference_steps is not None and args.scenario_reference_steps < 1:
        raise ValueError("scenario reference steps must be positive")
    if args.output is None:
        family = "table_ii" if args.scenario == "static" else "table_iii_environment"
        args.output = (
            DEFAULT_RESULTS_DIR
            / family
            / f"{args.task}_{args.scenario}_seed{args.seed}_n{args.episodes}_h{args.horizon}.json"
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
