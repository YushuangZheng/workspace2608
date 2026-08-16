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
from .runtime import ScenarioController, execute_joint_target_control

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v1"
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

EVALUATION_PROTOCOL_ID = (
    "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-noop-clock-v2"
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
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
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
        BimanualDiscrete(),
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
        except Exception:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            raise

    def request(self, command, observation=None):
        request = {"command": command}
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
):
    random.seed(seed + episode)
    np.random.seed(seed + episode)
    task_environment.set_variation(episode % task_environment.variation_count())
    _, observation = task_environment.reset()
    worker.request("reset", observation)
    invalid_actions = 0
    scenario_events = []
    controller = ScenarioController(
        kind=SCENARIO_KINDS[scenario],
        trigger_fraction=scenario_trigger_fraction,
        total_steps=scenario_steps,
        max_attempts=scenario_max_attempts,
    )
    if scenario_reference_steps is None:
        scenario_reference_steps = horizon
    from rlbench.backend.exceptions import InvalidActionError

    for step in range(horizon):
        observation, event = _apply_scenario(
            controller,
            task_environment,
            observation,
            step=step,
            horizon=scenario_reference_steps,
        )
        if event["applied"] or step == event["trigger_step"]:
            scenario_events.append(event)
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            return {
                "episode": episode,
                "success": False,
                "steps": step,
                "reason": "policy_complete",
                "invalid_actions": invalid_actions,
                "scenario_events": scenario_events,
            }
        try:
            observation, reward, terminate = task_environment.step(
                np.asarray(action, dtype=np.float64)
            )
        except InvalidActionError:
            invalid_actions += 1
            try:
                observation, reward, terminate = task_environment.step(
                    _noop_action(observation)
                )
            except InvalidActionError:
                return {
                    "episode": episode,
                    "success": False,
                    "steps": step + 1,
                    "reason": "noop_failed",
                    "invalid_actions": invalid_actions,
                    "scenario_events": scenario_events,
                }
        if reward > 0.0:
            reason = "success"
        elif terminate:
            reason = "terminate"
        elif response.get("complete"):
            reason = "policy_complete"
        else:
            continue
        return {
            "episode": episode,
            "success": bool(reward > 0.0),
            "steps": step + 1,
            "reason": reason,
            "invalid_actions": invalid_actions,
            "scenario_events": scenario_events,
        }
    return {
        "episode": episode,
        "success": False,
        "steps": horizon,
        "reason": "horizon",
        "invalid_actions": invalid_actions,
        "scenario_events": scenario_events,
    }


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
    effective_by_episode = [
        any(
            event.get("protocol_effective", False)
            for event in item["scenario_events"]
            if event["applied"]
        )
        for item in results
    ]
    summary = {
        "task": args.task,
        "scenario": args.scenario,
        "scenario_protocol": {
            "status": (
                "STATIC_REFERENCE"
                if args.scenario == "static"
                else "LOCAL_APPROXIMATION_PAPER_MOTION_TYPE_UNSPECIFIED"
            ),
            "public_scene_method": SCENARIO_KINDS[args.scenario],
            "trigger_fraction": args.scenario_trigger_fraction,
            "trigger_reference_steps": scenario_reference_steps,
            "trigger_reference_source": scenario_reference_source,
            "trigger_step": min(
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
            "episodes_with_intervention": sum(applied_by_episode),
            "episodes_with_effective_intervention": sum(effective_by_episode),
            "all_episodes_intervened": all(applied_by_episode),
            "all_interventions_effective": (
                all(effective_by_episode) if args.scenario != "static" else None
            ),
            "protocol_valid": (
                True
                if args.scenario == "static"
                else all(applied_by_episode) and all(effective_by_episode)
            ),
            "paper_comparable": args.scenario == "static",
        },
        "episodes": args.episodes,
        "seed": args.seed,
        "horizon": args.horizon,
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "controller": {
            "command": "absolute_world_end_effector_pose",
            "primary_ik": "jacobian",
            "fallback_ik": "sampling",
            "sampling_trials": 100,
            "sampling_max_configs": 5,
            "sampling_max_time_ms": 10,
            "sampling_ignore_collisions": True,
            "joint_target_max_steps": 200,
            "failed_action": "one_current_pose_current_gripper_noop",
            "policy_clock_rollback": False,
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "successes": sum(item["success"] for item in results),
        "success_rate": sum(item["success"] for item in results) / float(args.episodes),
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
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
    if args.scenario_steps < 1 or args.scenario_max_attempts < 1:
        raise ValueError("scenario steps and max attempts must be positive")
    if args.scenario_reference_steps is not None and args.scenario_reference_steps < 1:
        raise ValueError("scenario reference steps must be positive")
    if args.output is None:
        family = "table_ii" if args.scenario == "static" else "table_iii_environment"
        args.output = (
            INTEGRATION_ROOT
            / "results"
            / "v1"
            / family
            / f"{args.task}_{args.scenario}_seed{args.seed}_n{args.episodes}_h{args.horizon}.json"
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
