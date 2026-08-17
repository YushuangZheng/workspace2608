"""Evaluate DynaMAC on the static and dynamic Table I RLBench tasks.

The simulator process is Python 3.8 compatible and delegates policy math to
the current Python 3.10 worker. Dynamic movement samples and moves only the
current task's ``boundary_root`` while preserving the initialized episode;
it never reinitializes task objects or success conditions. Triggering at one
third of the fitted policy duration and ten smooth-motion calls are explicit
local defaults because the paper does not publish those task-wise values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import select
import subprocess
import tempfile
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
PROTOCOL_LABEL = "local_table_i_v2"
DYNAMIC_EPISODE_ACCOUNTING_SCHEMA = "trigger-eligibility-smooth-prefix-v1"
POLICY_CLOCK_SEMANTICS_ID = (
    "policy-tick-transaction-commit-on-primary-action-success-v1"
)
GRIPPER_PROTOCOL = DiscreteGripperProtocol(bimanual=False)
LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID = (
    "rlbench-prestep-explicit-base-vision-sensors-v1"
)
EXPECTED_UNIMANUAL_BASE_SCENE_SHA256 = (
    "66e1cfa0a6ee5a5e635917d23ce1b5f8ba7159ee1a5326588d798030b972306a"
)
EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT = 10


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


def _prepare_low_dim_headless_scene(environment_module, *, enabled):
    """Prepare a render-free copy of the pinned RLBench base scene.

    The pinned fork constructs ``ObservationConfig.camera_configs`` as an empty
    mapping for a low-dimensional policy.  Consequently ``Scene`` never marks
    the ten base-scene ``cam_*`` vision sensors as explicitly handled, and the
    very first ``PyRep.launch`` step renders them even though no camera data is
    requested.  On a genuinely headless host this can fail before the
    evaluator has constructed the task.

    Load the base scene without stepping it, mark every base-scene vision
    sensor explicit, save a temporary scene, and let RLBench launch that copy
    normally.  No task model has been loaded at this point, so task sensors,
    physics, success conditions, and policy observations are untouched.
    """

    original_scene = environment_module.TTT_FILE
    source = Path(original_scene)
    if not source.is_absolute():
        source = Path(environment_module.DIR_PATH) / source
    source = source.resolve()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "protocol_id": LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
        "applied": False,
        "scope": "all_base_scene_vision_sensors",
        "camera_observations_requested": False,
        "task_model_loaded_during_rewrite": False,
        "populated_scene_steps_before_patch": 0,
        "physics_modified": False,
        "task_modified": False,
        "policy_input_modified": False,
        "source_scene_sha256": source_sha256,
        "vision_sensors": [],
        "vision_sensor_handling": [],
        "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM"),
        "qt_qpa_platform_defaulted": False,
    }
    if not enabled:
        metadata["reason"] = "windowed_evaluation_uses_the_native_base_scene"
        return (lambda: None), metadata

    if source_sha256 != EXPECTED_UNIMANUAL_BASE_SCENE_SHA256:
        raise RuntimeError(
            "unimanual base scene does not match the pinned RLBench asset: "
            f"{source_sha256}"
        )

    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        metadata["qt_qpa_platform_defaulted"] = True
    metadata["qt_qpa_platform"] = os.environ["QT_QPA_PLATFORM"]

    from pyrep import PyRep
    from pyrep.backend import sim
    from pyrep.const import ObjectType

    temporary = tempfile.TemporaryDirectory(prefix="dynamac-lowdim-scene-")
    target = Path(temporary.name) / source.name
    simulator = PyRep()
    launched = False
    try:
        simulator.launch("", headless=True)
        launched = True
        sim.simLoadScene(str(source))
        vision_sensors = simulator.get_objects_in_tree(
            object_type=ObjectType.VISION_SENSOR
        )
        if not vision_sensors:
            raise RuntimeError(
                f"no base-scene vision sensors found in {source}"
            )
        if len(vision_sensors) != EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT:
            raise RuntimeError(
                "unexpected base-scene vision-sensor count: "
                f"{len(vision_sensors)}"
            )
        handling = []
        for sensor in vision_sensors:
            before = int(sensor.get_explicit_handling())
            sensor.set_explicit_handling(1)
            after = int(sensor.get_explicit_handling())
            if after != 1:
                raise RuntimeError(
                    f"failed to set explicit handling for {sensor.get_name()}"
                )
            handling.append(
                {"name": sensor.get_name(), "before": before, "after": after}
            )
        vision_sensor_names = sorted(sensor.get_name() for sensor in vision_sensors)
        handling.sort(key=lambda item: item["name"])
        simulator.export_scene(str(target))
    except Exception:
        temporary.cleanup()
        raise
    finally:
        if launched:
            try:
                simulator.shutdown()
            except Exception:
                temporary.cleanup()
                raise

    environment_module.TTT_FILE = str(target)
    metadata.update(
        {
            "applied": True,
            "vision_sensors": vision_sensor_names,
            "vision_sensor_handling": handling,
            "vision_sensor_count": len(vision_sensor_names),
            "derived_scene_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    )
    restored = [False]

    def restore():
        if restored[0]:
            return
        restored[0] = True
        environment_module.TTT_FILE = original_scene
        temporary.cleanup()

    return restore, metadata


def _make_action_mode():
    from pyrep.errors import ConfigurationError, IKError
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        ArmActionMode,
        assert_action_shape,
        assert_unit_quaternion,
    )
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

    return PoseGripperIgnore(
        AbsoluteEndEffectorIK(), GRIPPER_PROTOCOL.make_action_mode()
    )


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
    """Attach fail-closed dynamic eligibility and progress evidence.

    An episode that fails before the scheduled trigger is a real policy
    failure, but no dynamic condition was exercised.  Such a row is retained
    with a null effectiveness value.  Once the trigger is reached, teleport
    must apply exactly once and smooth motion must be a strict effective
    prefix; an episode that lives through the whole window must contain the
    endpoint.
    """

    events = row.get("interventions")
    if not isinstance(events, list):
        raise RuntimeError("episode interventions must be a list")
    dynamic = scenario != "static"
    if not dynamic:
        if events:
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
    if len(applied) != len(events):
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
    retry_budget = PrimaryActionRetryBudget(
        getattr(
            args,
            "max_primary_action_attempts",
            DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
        )
    )
    events = []
    trigger_step = (
        None
        if args.scenario == "static"
        else _trigger_control_step(worker.policy_steps, args.trigger_fraction)
    )
    trigger_reached = False

    def finish(row):
        return _finalize_episode_intervention_status(
            row,
            scenario=args.scenario,
            trigger_step=trigger_step,
            trigger_reached=trigger_reached,
            smooth_steps=args.smooth_steps,
        )

    from rlbench.backend.exceptions import InvalidActionError

    for step in range(args.horizon):
        if trigger_step is not None and step >= trigger_step:
            trigger_reached = True
        event = controller.apply(
            task_environment,
            step=step,
            horizon=worker.policy_steps,
        )
        expected_controller_trigger = _trigger_control_step(
            worker.policy_steps,
            args.trigger_fraction,
        )
        if event.get("trigger_step") != expected_controller_trigger:
            raise RuntimeError("scenario controller returned an inconsistent trigger")
        if event.get("applied"):
            if trigger_step is None or step < trigger_step:
                raise RuntimeError("scenario controller applied before the trigger")
            observation = task_environment.get_observation()
            event["policy_observation_refreshed"] = True
            events.append(event)
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            return finish({
                "episode": episode,
                "success": False,
                "steps": step,
                "reason": "policy_complete",
                "invalid_actions": invalid_actions,
                "interventions": events,
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
                    "interventions": events,
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
            "interventions": events,
        })
    return finish({
        "episode": episode,
        "success": False,
        "steps": args.horizon,
        "reason": "horizon",
        "invalid_actions": invalid_actions,
        "interventions": events,
    })


def evaluate(args):
    with reserve_output(args.output):
        return _evaluate_reserved(args)


def _evaluate_reserved(args):
    import rlbench.environment as environment_module
    from rlbench.observation_config import ObservationConfig

    module_name, class_name = TASKS[args.task]
    task_class = getattr(importlib.import_module(module_name), class_name)
    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    action_mode = _make_action_mode()
    environment = environment_module.Environment(
        action_mode=action_mode,
        obs_config=observation_config,
        headless=args.headless,
    )
    restore_scene, scene_launch = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=args.headless,
    )
    worker = None
    results = []
    launched = False
    try:
        worker = PolicyProcess(
            args.policy_python,
            args.task,
            args.models_dir,
            timeout=args.policy_timeout,
        )
        environment.launch()
        launched = True
        task_environment = environment.get_task(task_class)
        if args.variation >= task_environment.variation_count():
            raise ValueError("variation is outside task variation count")
        for episode in range(args.episodes):
            result = _run_episode(task_environment, worker, args, episode)
            results.append(result)
            successes = sum(item["success"] for item in results)
            print(
                f"{args.task} {args.scenario} episode {episode + 1}/{args.episodes}: "
                f"{result['reason']} (success "
                f"{100.0 * successes / len(results):.1f}%)",
                flush=True,
            )
    finally:
        try:
            if worker is not None:
                worker.close()
            if launched:
                environment.shutdown()
        finally:
            restore_scene()

    successes = sum(item["success"] for item in results)
    applied_by_episode = [
        any(event["applied"] for event in item["interventions"])
        for item in results
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
        SCENARIOS[args.scenario],
        trigger_fraction=args.trigger_fraction,
        total_steps=args.smooth_steps,
        max_attempts=args.intervention_attempts,
    ).protocol_metadata()
    summary = {
        "schema": "dynamac-table-i-evaluation-v2",
        "protocol_label": PROTOCOL_LABEL,
        "paper_comparable": False,
        "task": args.task,
        "scenario": args.scenario,
        "episodes": args.episodes,
        "seed": args.seed,
        "variation": args.variation,
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
            "scene_launch": scene_launch,
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "successes": successes,
        "success_rate": successes / float(args.episodes),
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
        "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
        "protocol": {
            "label": PROTOCOL_LABEL,
            "paper_comparable": False,
            "comparison_scope": "diagnostic_against_paper_targets_not_protocol_exact",
            "dynamic_method": SCENARIOS[args.scenario],
            "motion_protocol": motion_protocol,
            "trigger_fraction_of_nominal_policy_length": args.trigger_fraction,
            "trigger_reference_domain": (
                "evaluator_control_ticks_including_failed_primary_actions"
            ),
            "fitted_policy_steps": worker.policy_steps,
            "trigger_control_step": _trigger_control_step(
                worker.policy_steps,
                args.trigger_fraction,
            ),
            "smooth_motion_calls": args.smooth_steps,
            "intervention_max_attempts": args.intervention_attempts,
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
            "source": "project preserve-initialized-episode boundary-root motion",
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
    parser.add_argument(
        "--max-primary-action-attempts",
        type=int,
        default=DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
        help="Maximum primary InvalidAction attempts for one policy clock tick.",
    )
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        args.episodes < 1
        or args.horizon < 1
        or args.smooth_steps < 1
        or args.max_primary_action_attempts < 1
    ):
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
