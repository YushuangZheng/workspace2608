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
import random
import select
import subprocess
from pathlib import Path

import numpy as np

from .records import atomic_json, reserve_output
from .runtime import (
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    DiscreteGripperProtocol,
    PrimaryActionRetryBudget,
    ScenarioController,
    execute_joint_target_control,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v2"
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v2"
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
DYNAMIC_EPISODE_ACCOUNTING_SCHEMA = "trigger-eligibility-smooth-prefix-v1"

POLICY_CLOCK_SEMANTICS_ID = (
    "policy-tick-transaction-commit-on-primary-action-success-v1"
)
GRIPPER_PROTOCOL = DiscreteGripperProtocol(bimanual=True)


def evaluation_protocol_id(max_primary_action_attempts):
    attempts = PrimaryActionRetryBudget(max_primary_action_attempts).max_attempts
    return GRIPPER_PROTOCOL.extend_evaluation_protocol_id(
        "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-"
        "noop-retry-same-policy-tick-fresh-observation-"
        f"primary-attempt{attempts}-v4"
    )


EVALUATION_PROTOCOL_ID = evaluation_protocol_id(
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
)


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
        if row.get("success") is True:
            raise RuntimeError(
                "dynamic episode succeeded before its intervention trigger"
            )
        row.update(
            {
                "trigger_step": trigger_step,
                "intervention_eligible": False,
                "intervention_reached": False,
                "pre_intervention_terminal": True,
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
            if row.get("steps") not in {final_event_step, final_event_step + 1}:
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
    scenario_reference_steps=None,
    scenario_steps=10,
    scenario_max_attempts=20,
    max_primary_action_attempts=DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
):
    random.seed(seed + episode)
    np.random.seed(seed + episode)
    task_environment.set_variation(episode % task_environment.variation_count())
    _, observation = task_environment.reset()
    worker.request("reset", observation)
    invalid_actions = 0
    retry_budget = PrimaryActionRetryBudget(max_primary_action_attempts)
    scenario_events = []
    controller = ScenarioController(
        kind=SCENARIO_KINDS[scenario],
        trigger_fraction=scenario_trigger_fraction,
        total_steps=scenario_steps,
        max_attempts=scenario_max_attempts,
    )
    if scenario_reference_steps is None:
        scenario_reference_steps = horizon
    trigger_step = (
        None
        if scenario == "static"
        else _trigger_control_step(
            scenario_reference_steps,
            scenario_trigger_fraction,
        )
    )
    trigger_reached = False

    def finish(row):
        return _finalize_episode_intervention_status(
            row,
            scenario=scenario,
            trigger_step=trigger_step,
            trigger_reached=trigger_reached,
            smooth_steps=scenario_steps,
        )

    from rlbench.backend.exceptions import InvalidActionError

    for step in range(horizon):
        if trigger_step is not None and step >= trigger_step:
            trigger_reached = True
        observation, event = _apply_scenario(
            controller,
            task_environment,
            observation,
            step=step,
            horizon=scenario_reference_steps,
        )
        expected_controller_trigger = _trigger_control_step(
            scenario_reference_steps,
            scenario_trigger_fraction,
        )
        if event.get("trigger_step") != expected_controller_trigger:
            raise RuntimeError("scenario controller returned an inconsistent trigger")
        if event.get("applied") and (trigger_step is None or step < trigger_step):
            raise RuntimeError("scenario controller applied before the trigger")
        if event["applied"] or step == event["trigger_step"]:
            scenario_events.append(event)
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            return finish({
                "episode": episode,
                "success": False,
                "steps": step,
                "reason": "policy_complete",
                "invalid_actions": invalid_actions,
                "scenario_events": scenario_events,
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
                    "steps": step + 1,
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
        else:
            policy_complete = False
        if reward > 0.0:
            reason = "success"
        elif terminate:
            reason = "terminate"
        elif retry_exhausted:
            reason = "primary_action_retry_exhausted"
        elif policy_complete:
            reason = "policy_complete"
        else:
            continue
        return finish({
            "episode": episode,
            "success": bool(reward > 0.0),
            "steps": step + 1,
            "reason": reason,
            "invalid_actions": invalid_actions,
            **(
                {"primary_action_attempts": retry_budget.attempts}
                if reason == "primary_action_retry_exhausted"
                else {}
            ),
            "scenario_events": scenario_events,
        })
    return finish({
        "episode": episode,
        "success": False,
        "steps": horizon,
        "reason": "horizon",
        "invalid_actions": invalid_actions,
        "scenario_events": scenario_events,
    })


def evaluate(args):
    with reserve_output(args.output):
        return _evaluate_reserved(args)


def _evaluate_reserved(args):
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    module_name, class_name = TASKS[args.task]
    task_class = getattr(importlib.import_module(module_name), class_name)
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
    if args.scenario_reference_steps is None:
        scenario_reference_steps = worker.policy_steps
        scenario_reference_source = "LOADED_CHECKPOINT_POLICY_STEPS"
    else:
        scenario_reference_steps = args.scenario_reference_steps
        scenario_reference_source = "COMMAND_LINE"
    results = []
    launched = False
    try:
        environment.launch()
        launched = True
        task_environment = environment.get_task(task_class)
        for episode in range(args.episodes):
            result = _run_episode(
                task_environment,
                worker,
                episode,
                args.seed,
                args.horizon,
                scenario=args.scenario,
                scenario_trigger_fraction=args.scenario_trigger_fraction,
                scenario_reference_steps=scenario_reference_steps,
                scenario_steps=args.scenario_steps,
                scenario_max_attempts=args.scenario_max_attempts,
                max_primary_action_attempts=args.max_primary_action_attempts,
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
        total_steps=args.scenario_steps,
        max_attempts=args.scenario_max_attempts,
    ).protocol_metadata()
    summary = {
        "task": args.task,
        "scenario": args.scenario,
        "scenario_protocol": {
            "status": (
                "STATIC_REFERENCE"
                if args.scenario == "static"
                else "LOCAL_APPROXIMATION_PAPER_MOTION_TYPE_UNSPECIFIED"
            ),
            "motion_kind": SCENARIO_KINDS[args.scenario],
            "motion_protocol": motion_protocol,
            "trigger_fraction_of_nominal_policy_length": (
                args.scenario_trigger_fraction
            ),
            "trigger_reference_steps": scenario_reference_steps,
            "trigger_reference_source": scenario_reference_source,
            "trigger_reference_domain": (
                "evaluator_control_ticks_including_failed_primary_actions"
            ),
            "trigger_control_step": min(
                scenario_reference_steps - 1,
                int(
                    round(
                        args.scenario_trigger_fraction
                        * (scenario_reference_steps - 1)
                    )
                ),
            ),
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
                "fail_closed_unexercised_dynamic_condition"
            ),
            "smooth_terminal_progress_policy": (
                "strict_effective_prefix_until_episode_terminal"
            ),
            "episodes_intervention_eligible": sum(eligible_by_episode),
            "episodes_pre_intervention_terminal": sum(preterminal_by_episode),
            "episodes_with_intervention": sum(applied_by_episode),
            "episodes_with_effective_intervention": sum(effective_by_episode),
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
        },
        "episodes": args.episodes,
        "seed": args.seed,
        "horizon": args.horizon,
        "evaluation_protocol_id": evaluation_protocol_id(
            args.max_primary_action_attempts
        ),
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
            "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "successes": sum(item["success"] for item in results),
        "success_rate": sum(item["success"] for item in results) / float(args.episodes),
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
        "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
        "results": results,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_KINDS),
        default="static",
        help="Evaluation condition; dynamic modes are local approximations of Table III.",
    )
    parser.add_argument("--scenario-trigger-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--scenario-reference-steps",
        type=int,
        default=None,
        help="Clock length used for the trigger fraction; defaults to fitted policy length.",
    )
    parser.add_argument("--scenario-steps", type=int, default=10)
    parser.add_argument("--scenario-max-attempts", type=int, default=20)
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
    if args.episodes < 1 or args.horizon < 1:
        raise ValueError("episodes and horizon must be positive")
    if not 0.0 <= args.scenario_trigger_fraction <= 1.0:
        raise ValueError("scenario trigger fraction must lie in [0, 1]")
    if (
        args.scenario_steps < 1
        or args.scenario_max_attempts < 1
        or args.max_primary_action_attempts < 1
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
