"""Evaluate DynaMAC on the static and dynamic Table I RLBench tasks.

The simulator process is Python 3.8 compatible and delegates policy math to
the current Python 3.10 worker.  Dynamic movement uses the public pinned
``Scene.kidnap`` and ``Scene.move_task_smoothly`` methods.  Triggering at one
third of the fitted policy duration and ten smooth-motion calls are explicit
local defaults because the paper does not publish those task-wise values.
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
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v1"
DEFAULT_POLICY_PYTHON = Path(os.environ.get("DYNAMAC_POLICY_PYTHON", "python3.10"))
TASKS = {
    "stack_wine": ("rlbench.tasks.stack_wine", "StackWine"),
    "place_cups": ("rlbench.tasks.place_cups", "PlaceCups"),
    "open_microwave": ("rlbench.tasks.open_microwave", "OpenMicrowave"),
    "wipe_desk": ("rlbench.tasks.wipe_desk", "WipeDesk"),
}
SCENARIOS = {
    "static": "static",
    "smooth": "smooth_task_motion",
    "teleport": "teleport_task",
}
PROTOCOL_LABEL = "local_table_i_v1"
EVALUATION_PROTOCOL_ID = (
    "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-noop-clock-v2"
)


def _make_action_mode():
    from pyrep.errors import ConfigurationError, IKError
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        ArmActionMode,
        assert_action_shape,
        assert_unit_quaternion,
    )
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.exceptions import InvalidActionError

    class AbsoluteEndEffectorIK(ArmActionMode):
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
                        "unimanual absolute end-effector IK failed"
                    ) from exc

        def action(self, scene, action, ignore_collisions=True):
            del ignore_collisions
            assert_action_shape(action, (7,))
            target = np.asarray(action, dtype=np.float64)
            assert_unit_quaternion(target[3:])
            joints = self._solve(scene.robot.arm, target)
            scene.robot.arm.set_joint_target_positions(joints)
            try:
                status = execute_joint_target_control(
                    scene,
                    ((scene.robot.arm, joints),),
                    invalid_action_error=InvalidActionError,
                    error_message=(
                        "unimanual absolute end-effector IK did not converge within "
                        "200 simulation steps"
                    ),
                )
            except InvalidActionError:
                self._execution_diagnostics["joint_target_timeouts"] += 1
                raise
            self._execution_diagnostics[f"joint_target_{status}"] += 1

        def action_shape(self, scene):
            del scene
            return (7,)

    class PoseGripperIgnore(MoveArmThenGripper):
        def action_shape(self, scene):
            return super().action_shape(scene) + 1

    return PoseGripperIgnore(AbsoluteEndEffectorIK(), Discrete())


def _observation_payload(observation):
    state = observation.task_low_dim_state
    if isinstance(state, tuple) and len(state) == 1:
        state = state[0]
    return {
        "gripper_pose": np.asarray(observation.gripper_pose).tolist(),
        "task_low_dim_state": np.asarray(state).reshape(-1).tolist(),
    }


def _noop_action(observation):
    return np.concatenate(
        (
            np.asarray(observation.gripper_pose, dtype=np.float64),
            [float(observation.gripper_open >= 0.5), 0.0],
        )
    )


class PolicyProcess:
    def __init__(self, python, task, models_dir, timeout=120.0):
        self.timeout = float(timeout)
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
            if not response.get("ready") or response.get("bimanual"):
                raise RuntimeError("policy worker did not report a ready unimanual model")
            if response.get("task") != task:
                raise RuntimeError("policy worker identity does not match the requested task")
            self.policy_steps = int(response["policy_steps"])
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
            raise TimeoutError("policy worker response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("policy worker exited without a response")
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


def _run_episode(task_environment, worker, args, episode):
    random.seed(args.seed + episode)
    np.random.seed(args.seed + episode)
    task_environment.set_variation(args.variation)
    _, observation = task_environment.reset()
    worker.request("reset", observation)
    controller = ScenarioController(
        SCENARIOS[args.scenario],
        trigger_fraction=args.trigger_fraction,
        total_steps=args.smooth_steps,
        max_attempts=args.intervention_attempts,
        verify_instance=True,
    )
    invalid_actions = 0
    events = []
    from rlbench.backend.exceptions import InvalidActionError

    for step in range(args.horizon):
        event = controller.apply(
            task_environment,
            step=step,
            horizon=worker.policy_steps,
        )
        if event.get("applied"):
            events.append(event)
            observation = task_environment.get_observation()
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            return {
                "episode": episode,
                "success": False,
                "steps": step,
                "reason": "policy_complete",
                "invalid_actions": invalid_actions,
                "interventions": events,
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
                    "interventions": events,
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
            "interventions": events,
        }
    return {
        "episode": episode,
        "success": False,
        "steps": args.horizon,
        "reason": "horizon",
        "invalid_actions": invalid_actions,
        "interventions": events,
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
        environment.launch()
        launched = True
        task_environment = environment.get_task(task_class)
        if args.variation >= task_environment.variation_count():
            raise ValueError("variation is outside task variation count")
        for episode in range(args.episodes):
            result = _run_episode(task_environment, worker, args, episode)
            if args.scenario == "static":
                result["intervention_effective"] = None
            else:
                result["intervention_effective"] = any(
                    event.get("protocol_effective") is True
                    for event in result["interventions"]
                )
                if not result["intervention_effective"]:
                    raise RuntimeError(
                        f"{args.task} {args.scenario} episode {episode} had no "
                        "verified task-frame movement"
                    )
            results.append(result)
            successes = sum(item["success"] for item in results)
            print(
                f"{args.task} {args.scenario} episode {episode + 1}/{args.episodes}: "
                f"{result['reason']} (success "
                f"{100.0 * successes / len(results):.1f}%)",
                flush=True,
            )
    finally:
        worker.close()
        if launched:
            environment.shutdown()

    successes = sum(item["success"] for item in results)
    summary = {
        "schema": "dynamac-table-i-evaluation-v1",
        "protocol_label": PROTOCOL_LABEL,
        "paper_comparable": False,
        "task": args.task,
        "scenario": args.scenario,
        "episodes": args.episodes,
        "seed": args.seed,
        "variation": args.variation,
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
        "successes": successes,
        "success_rate": successes / float(args.episodes),
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
        "protocol": {
            "label": PROTOCOL_LABEL,
            "paper_comparable": False,
            "comparison_scope": "diagnostic_against_paper_targets_not_protocol_exact",
            "dynamic_method": SCENARIOS[args.scenario],
            "trigger_fraction_of_fitted_policy": args.trigger_fraction,
            "fitted_policy_steps": worker.policy_steps,
            "smooth_motion_calls": args.smooth_steps,
            "intervention_max_attempts": args.intervention_attempts,
            "source": "public pinned RLBench Scene dynamic methods",
            "claim_boundary": (
                "The live five-demonstration cohort, fixed variation, segmentation, "
                "movement timing, and dynamic magnitude are explicit local defaults. "
                "The paper's exact Table I datasets and task-wise DynaBench protocol "
                "are unavailable, so these results are not paper-comparable."
            ),
        },
        "results": results,
    }
    atomic_json(args.output, summary)
    print(f"wrote {args.output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--trigger-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--smooth-steps", type=int, default=10)
    parser.add_argument("--intervention-attempts", type=int, default=20)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.horizon < 1 or args.smooth_steps < 1:
        raise ValueError("episodes, horizon, and smooth steps must be positive")
    if args.seed < 0 or args.variation < 0:
        raise ValueError("seed and variation must be non-negative")
    if not 0.0 <= args.trigger_fraction <= 1.0:
        raise ValueError("trigger fraction must lie in [0, 1]")
    if args.output is None:
        family = "table_i" if args.scenario == "static" else "table_i_dynamic"
        args.output = DEFAULT_RESULTS_DIR / family / (
            f"{args.task}_{args.scenario}_variation{args.variation}_"
            f"seed{args.seed}_n{args.episodes}_h{args.horizon}.json"
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
