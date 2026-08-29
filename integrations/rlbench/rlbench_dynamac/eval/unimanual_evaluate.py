"""Evaluate DynaMAC on the static and dynamic Table I RLBench tasks.

The simulator process is Python 3.8 compatible and delegates policy math to
the current Python 3.10 worker. V3 dynamic movement is preregistered per task,
samples and waypoint-validates its A/B poses in a disposable staging process,
and applies the immutable B pose in the formal rollout on the committed-policy
clock. The formal rollout never samples or restores a task configuration tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import select
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    _commit_formal_result_with_optional_v4_videos,
    _finalize_v4_video_cell,
    _prepare_v4_video_cell,
    _run_episode_with_optional_v4_video,
    _v4_video_capture_enabled,
    _v4_video_cell,
)
from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
    fixed_environment_plans,
    validate_formal_artifact_paths,
)
from integrations.rlbench.rlbench_dynamac.report.evaluation_videos import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_V4_EVALUATION_VIDEO_ROOT,
    LightweightCaptureConfig,
)
from integrations.rlbench.rlbench_dynamac.core.gripper_timing import (
    GLOBAL_GRIPPER_TIMING_PROTOCOL_ID,
    global_gripper_timing_metadata,
)
from integrations.rlbench.rlbench_dynamac.core.records import (
    atomic_json,
    reserve_output,
)
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
    DiscreteGripperProtocol,
    FORMAL_POLICY_CLOCK_SEMANTICS_ID,
    FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE,
    FROZEN_V4_CONTROLLER_PROFILE,
    GLOBAL_IK_CONTROLLER_PROFILE,
    STAGE6_IK_CONTROLLER_PROFILE,
    GlobalIKControllerConfig,
    Stage6IKControllerConfig,
    PrimaryActionRetryBudget,
    ScenarioController,
    commit_joint_hold_after_primary_failure,
    execute_global_ik_ee_control,
    execute_stage6_ik_ee_control,
    execute_joint_target_control,
    final_settling_metadata,
    global_ik_controller_metadata,
    initialize_fresh_task_generation,
    initialize_global_ik_controller_diagnostics,
    initialize_ik_solver_diagnostics,
    run_final_settling,
    solve_absolute_ee_ik_with_sampling_fallback,
    step_current_joint_hold_noop,
    stage_scenario_motion_plan,
    staged_motion_plan_batch,
    validate_primary_retry_exhaustion_mode,
)
from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
    load_v3_intervention_protocol,
    load_v3_motion_source_protocol,
    resolve_authenticated_v3_trigger,
)

from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT

DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3"
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v3"
CLOSED_LOOP_MODELS_DIR = INTEGRATION_ROOT / "models" / "closed_loop_v1"
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
PROTOCOL_LABEL = "local_table_i_v3"
DYNAMIC_EPISODE_ACCOUNTING_SCHEMA = (
    "planned-denominator-trigger-completion-conditional-success-v3"
)
POLICY_CLOCK_SEMANTICS_ID = FORMAL_POLICY_CLOCK_SEMANTICS_ID
GRIPPER_PROTOCOL = DiscreteGripperProtocol(bimanual=False)
LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID = "rlbench-prestep-explicit-base-vision-sensors-v1"
EXPECTED_UNIMANUAL_BASE_SCENE_SHA256 = (
    "66e1cfa0a6ee5a5e635917d23ce1b5f8ba7159ee1a5326588d798030b972306a"
)
EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT = 10
V4_VIDEO_PAPER_TARGETS = {
    ("stack_wine", "static"): 1.00,
    ("stack_wine", "smooth"): 1.00,
    ("stack_wine", "teleport"): 1.00,
    ("place_cups", "static"): 0.99,
    ("place_cups", "smooth"): 0.97,
    ("place_cups", "teleport"): 0.99,
    ("open_microwave", "static"): 0.99,
    ("open_microwave", "smooth"): 0.99,
    ("open_microwave", "teleport"): 0.97,
    ("wipe_desk", "static"): 1.00,
    ("wipe_desk", "smooth"): 0.66,
    ("wipe_desk", "teleport"): 0.69,
}


def legacy_v3_evaluation_protocol_id(max_primary_action_attempts):
    attempts = PrimaryActionRetryBudget(max_primary_action_attempts).max_attempts
    return GRIPPER_PROTOCOL.extend_evaluation_protocol_id(
        "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-"
        "noop-retry-same-policy-tick-fresh-observation-"
        f"primary-attempt{attempts}-committed-dynamic-clock-"
        "final-settle-up-to-raw10-staged34-deterministic-source-reset1-"
        "formal-root-state-audit2-contact-delta-diagnostic-v3"
    )


def evaluation_protocol_id(
    max_primary_action_attempts,
    controller_profile=GLOBAL_IK_CONTROLLER_PROFILE,
):
    attempts = PrimaryActionRetryBudget(max_primary_action_attempts).max_attempts
    if attempts != 1:
        raise ValueError(
            "formal global controller requires one request per policy tick"
        )
    if controller_profile == STAGE6_IK_CONTROLLER_PROFILE:
        controller = (
            "rlbench-stage6-current-seeded-pseudo6-then-bounded-trac-distance-"
            "then-cartesian-continuation-collision-aware-or-relaxed-sampling-path-"
            "converged-joint-target-cartesian-feedback-"
        )
    elif controller_profile == GLOBAL_IK_CONTROLLER_PROFILE:
        controller = (
            "rlbench-absolute-ee-pseudo6-then-bounded-trac-distance-"
            "then-collision-aware-sampling5-nearest-current-q-"
            "then-far10cm-collision-aware-path-timeout200-"
        )
    else:
        raise ValueError("unsupported evaluation controller profile")
    return GRIPPER_PROTOCOL.extend_evaluation_protocol_id(
        controller + f"{GLOBAL_GRIPPER_TIMING_PROTOCOL_ID}-"
        "single-primary-request-raw-joint-hold-same-transaction-"
        f"primary-attempt{attempts}-committed-shared-clock-"
        "final-settle-up-to-raw10-staged34-deterministic-source-reset1-"
        "formal-root-state-audit2-contact-delta-diagnostic-v5"
    )


LEGACY_V3_EVALUATION_PROTOCOL_ID = legacy_v3_evaluation_protocol_id(3)
EVALUATION_PROTOCOL_ID = evaluation_protocol_id(DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS)


def _validate_formal_execution_args(args):
    if (
        getattr(
            args,
            "max_primary_action_attempts",
            DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
        )
        != 1
    ):
        raise RuntimeError("formal evaluation requires one primary request per tick")
    profile = _resolved_controller_profile(args)
    return global_ik_controller_metadata(_controller_config(profile))


def _resolved_controller_profile(args):
    requested = getattr(args, "controller_profile", "auto")
    if requested == "auto":
        return (
            STAGE6_IK_CONTROLLER_PROFILE
            if getattr(args, "policy_type", "dynamac") == "closed_loop_multistream"
            else GLOBAL_IK_CONTROLLER_PROFILE
        )
    return requested


def _controller_config(controller_profile):
    if controller_profile == STAGE6_IK_CONTROLLER_PROFILE:
        return Stage6IKControllerConfig()
    if controller_profile == GLOBAL_IK_CONTROLLER_PROFILE:
        return GlobalIKControllerConfig()
    if controller_profile == FROZEN_V4_CONTROLLER_PROFILE:
        return None
    raise ValueError("unsupported unimanual controller profile")


def _validate_v3_protocol_budgets(args):
    """Fail before staging if any frozen V3 budget was overridden."""

    protocol = load_v3_intervention_protocol()
    motion_protocol = load_v3_motion_source_protocol()
    if args.smooth_steps != protocol["smooth_steps"]:
        raise RuntimeError(
            "V3 smooth-step budget differs from the frozen intervention protocol"
        )
    if args.intervention_attempts != motion_protocol["goal_sampling_max_attempts"]:
        raise RuntimeError(
            "V3 goal-sampling budget differs from the frozen motion-source protocol"
        )
    if args.final_settling_steps != protocol["final_settling_physics_steps"]:
        raise RuntimeError(
            "V3 final-settling budget differs from the frozen intervention protocol"
        )
    return protocol


def _authenticated_v3_dynamic_trigger(args, worker):
    """Resolve the frozen per-task trigger from checkpoint-backed evidence."""

    protocol = _validate_v3_protocol_budgets(args)
    authentication = resolve_authenticated_v3_trigger(
        worker.model_identity,
        task=args.task,
    )
    authenticated_step = authentication["trigger_step"]
    requested_step = getattr(args, "trigger_step", None)
    if requested_step is not None and requested_step != authenticated_step:
        raise RuntimeError(
            "command-line trigger step differs from authenticated V3 preregistration"
        )
    if authenticated_step >= worker.policy_steps:
        raise RuntimeError(
            "authenticated V3 trigger lies outside the loaded policy clock"
        )
    return protocol, authentication


def _prepare_low_dim_headless_scene(
    environment_module,
    *,
    enabled,
    camera_observations_requested=False,
):
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
        "camera_observations_requested": bool(camera_observations_requested),
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
            raise RuntimeError(f"no base-scene vision sensors found in {source}")
        if len(vision_sensors) != EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT:
            raise RuntimeError(
                "unexpected base-scene vision-sensor count: " f"{len(vision_sensors)}"
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


def _make_action_mode(
    controller_profile=GLOBAL_IK_CONTROLLER_PROFILE,
    controller_config=None,
    external_solver_factory=None,
):
    if controller_profile not in {
        GLOBAL_IK_CONTROLLER_PROFILE,
        STAGE6_IK_CONTROLLER_PROFILE,
        FROZEN_V4_CONTROLLER_PROFILE,
    }:
        raise ValueError("unsupported unimanual controller profile")
    if controller_profile in {
        GLOBAL_IK_CONTROLLER_PROFILE,
        STAGE6_IK_CONTROLLER_PROFILE,
    }:
        if controller_config is None:
            controller_config = _controller_config(controller_profile)
        if not isinstance(controller_config, GlobalIKControllerConfig):
            raise ValueError("global IK controller requires its formal config")
        if external_solver_factory is not None and not callable(
            external_solver_factory
        ):
            raise TypeError("external_solver_factory must be callable")
    elif controller_config is not None:
        raise ValueError("frozen V4 controller does not accept dev configuration")
    elif external_solver_factory is not None:
        raise ValueError("frozen V4 controller does not accept an external solver")

    from pyrep.const import ConfigurationPathAlgorithms
    from pyrep.errors import ConfigurationError, ConfigurationPathError, IKError
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        ArmActionMode,
        assert_action_shape,
        assert_unit_quaternion,
    )
    from rlbench.backend.exceptions import InvalidActionError

    class AbsoluteEndEffectorIK(ArmActionMode):
        def __init__(self):
            self._controller_profile = controller_profile
            self._controller_config = controller_config
            self._external_solver_factory = external_solver_factory
            self._external_solvers = {}
            if controller_profile in {
                GLOBAL_IK_CONTROLLER_PROFILE,
                STAGE6_IK_CONTROLLER_PROFILE,
            }:
                self._execution_diagnostics = (
                    initialize_global_ik_controller_diagnostics()
                )
            else:
                self._execution_diagnostics = initialize_ik_solver_diagnostics()
            self._execution_diagnostics.update(
                {
                    "joint_target_reached": 0,
                    "joint_target_progressed": 0,
                    "joint_target_stopped": 0,
                    "joint_target_timeouts": 0,
                }
            )
            if controller_profile in {
                GLOBAL_IK_CONTROLLER_PROFILE,
                STAGE6_IK_CONTROLLER_PROFILE,
            }:
                self._execution_diagnostics.update(
                    {
                        "controller_profile": controller_profile,
                        "controller_config": controller_config.metadata(),
                    }
                )

        def diagnostics(self):
            return dict(self._execution_diagnostics)

        def _external_solver_for_arm(self, arm):
            key = id(arm)
            if key not in self._external_solvers:
                factory = self._external_solver_factory
                if factory is None:
                    from integrations.rlbench.rlbench_dynamac.core.trac_ik import (
                        AlignedTracIKDistanceSolver,
                        TracIKDistanceConfig,
                    )

                    solver_config = TracIKDistanceConfig(
                        timeout_s=self._controller_config.trac_ik_timeout_s,
                        epsilon=self._controller_config.trac_ik_epsilon,
                        translation_tolerance_m=(
                            self._controller_config.trac_ik_translation_tolerance_m
                        ),
                        rotation_tolerance_rad=(
                            self._controller_config.trac_ik_rotation_tolerance_rad
                        ),
                        joint_window_rad=self._controller_config.trac_ik_joint_window_rad,
                        joint_delta_abs_max_rad=(
                            self._controller_config.max_joint_delta_abs_rad
                        ),
                        joint_delta_l2_max_rad=(
                            self._controller_config.max_joint_delta_l2_rad
                        ),
                        fk_translation_max_m=(
                            self._controller_config.trac_ik_fk_translation_max_m
                        ),
                        fk_rotation_max_rad=(
                            self._controller_config.trac_ik_fk_rotation_max_rad
                        ),
                    )
                    solver = AlignedTracIKDistanceSolver(arm, config=solver_config)
                else:
                    solver = factory(arm)
                if not callable(getattr(solver, "solve", None)):
                    raise TypeError("TRAC-IK Distance solver must provide solve()")
                self._external_solvers[key] = solver
            return self._external_solvers[key]

        def _solve(self, arm, target):
            return solve_absolute_ee_ik_with_sampling_fallback(
                arm,
                target,
                diagnostics=self._execution_diagnostics,
                ik_error=IKError,
                configuration_error=ConfigurationError,
                invalid_action_error=InvalidActionError,
                error_message="unimanual absolute end-effector IK failed",
            )

        def action(self, scene, action, ignore_collisions=True):
            del ignore_collisions  # V4 collision policy is frozen in the IK helper
            assert_action_shape(action, (7,))
            target = np.asarray(action, dtype=np.float64)
            assert_unit_quaternion(target[3:])
            if self._controller_profile in {
                GLOBAL_IK_CONTROLLER_PROFILE,
                STAGE6_IK_CONTROLLER_PROFILE,
            }:
                try:
                    executor = (
                        execute_stage6_ik_ee_control
                        if self._controller_profile == STAGE6_IK_CONTROLLER_PROFILE
                        else execute_global_ik_ee_control
                    )
                    status = executor(
                        scene,
                        ((scene.robot.arm, target),),
                        config=self._controller_config,
                        diagnostics=self._execution_diagnostics,
                        external_solver_factory=self._external_solver_for_arm,
                        ik_error=IKError,
                        configuration_error=ConfigurationError,
                        configuration_path_error=ConfigurationPathError,
                        invalid_action_error=InvalidActionError,
                        path_algorithm=ConfigurationPathAlgorithms.RRTConnect,
                        error_message=(
                            "unimanual formal pseudo/TRAC-IK/sampling/path EE "
                            "control failed"
                        ),
                    )
                except InvalidActionError:
                    self._execution_diagnostics[
                        "trac_ik_distance_controller_invalid_actions"
                    ] += 1
                    raise
                self._execution_diagnostics[f"joint_target_{status}"] += 1
                return
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
        "gripper_open": float(observation.gripper_open),
        "task_low_dim_state": np.asarray(state).reshape(-1).tolist(),
    }


def _noop_action(observation):
    """Frozen-V4 current-EE no-op in the fork's 9D action layout."""

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
    if len(applied) != len(events):
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
    if any(event.get("trigger_step") != trigger_step for event in applied):
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


class PolicyProcess:
    def __init__(
        self,
        python,
        task,
        models_dir,
        timeout=120.0,
        *,
        policy_type="dynamac",
        closed_loop_models_dir=CLOSED_LOOP_MODELS_DIR,
        diagnostics_dir=None,
    ):
        self.timeout = float(timeout)
        if policy_type not in {"dynamac", "closed_loop_multistream"}:
            raise ValueError("policy_type must be dynamac or closed_loop_multistream")
        self.policy_type = policy_type
        if policy_type == "dynamac":
            command = [
                str(python),
                "-m",
                "integrations.rlbench.rlbench_dynamac.data.direct_policy",
                "serve",
                "--task",
                task,
                "--models-dir",
                str(Path(models_dir).resolve()),
            ]
        else:
            command = [
                str(python),
                "-m",
                "integrations.rlbench.rlbench_closed_loop.policy_server",
                "serve",
                "--task",
                task,
                "--models-dir",
                str(Path(closed_loop_models_dir).resolve()),
                "--base-models-dir",
                str(Path(models_dir).resolve()),
            ]
            if diagnostics_dir is not None:
                command.extend(
                    ["--diagnostics-dir", str(Path(diagnostics_dir).resolve())]
                )
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
                raise RuntimeError(
                    "policy worker did not report a ready unimanual model"
                )
            if response.get("task") != task:
                raise RuntimeError(
                    "policy worker identity does not match the requested task"
                )
            self.policy_steps = int(response["policy_steps"])
            if self.policy_steps < 1:
                raise RuntimeError("policy worker reported an empty trajectory")
            self.model_identity = response.get("model_identity")
            if not isinstance(self.model_identity, dict):
                raise RuntimeError("policy worker did not report model identity")
            self.policy_clock_semantics_id = response.get("policy_clock_semantics_id")
            if self.policy_clock_semantics_id != POLICY_CLOCK_SEMANTICS_ID:
                raise RuntimeError(
                    "policy worker clock semantics do not match evaluator"
                )
            self.gripper_timing = response.get("gripper_timing")
            if self.policy_type == "dynamac":
                if self.gripper_timing != global_gripper_timing_metadata():
                    raise RuntimeError(
                        "policy worker gripper timing does not match evaluator"
                    )
            elif (
                not isinstance(self.gripper_timing, dict)
                or self.gripper_timing.get("protocol_id")
                != GLOBAL_GRIPPER_TIMING_PROTOCOL_ID
                or self.gripper_timing.get("rule")
                != "committed_reference_state_command"
                or response.get("policy_type") != self.policy_type
            ):
                raise RuntimeError("closed-loop policy worker timing identity mismatch")
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


def _run_episode(
    task_environment,
    worker,
    args,
    episode,
    motion_plan=None,
    *,
    descriptions=None,
    observation=None,
    fresh_task_generation=None,
    retry_exhaustion_mode=FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE,
):
    retry_exhaustion_mode = validate_primary_retry_exhaustion_mode(
        retry_exhaustion_mode
    )
    if retry_exhaustion_mode != FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE:
        raise ValueError("formal evaluation requires the global joint-hold clock")
    if (
        getattr(
            args,
            "max_primary_action_attempts",
            DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
        )
        != 1
    ):
        raise ValueError("formal evaluation requests each policy tick exactly once")
    if (
        descriptions is None
        or observation is None
        or not isinstance(
            fresh_task_generation,
            dict,
        )
    ):
        raise RuntimeError("formal episode requires fresh task-generation input")
    controller = ScenarioController(
        SCENARIOS[args.scenario],
        trigger_fraction=args.trigger_fraction,
        trigger_step=getattr(args, "trigger_step", None),
        total_steps=args.smooth_steps,
        max_attempts=args.intervention_attempts,
        verify_instance=True,
        motion_plan=motion_plan,
    )
    invalid_actions = 0
    joint_hold_commits = 0
    events = []
    trigger_step = (
        None
        if args.scenario == "static"
        else (
            controller.resolved_trigger_step(worker.policy_steps)
            if callable(getattr(controller, "resolved_trigger_step", None))
            else _trigger_control_step(worker.policy_steps, args.trigger_fraction)
        )
    )
    trigger_reached = False
    committed_policy_steps = 0
    last_scenario_policy_step = None
    bind_source = getattr(controller, "bind_staged_source", None)
    source_binding = (
        bind_source(
            task_environment,
            episode_seed=args.seed + episode,
            variation=args.variation,
            descriptions=descriptions,
        )
        if callable(bind_source)
        else {"required": False, "matched": None}
    )
    if motion_plan is not None:
        observation = task_environment.get_observation()
        source_binding["formal_observation_refreshed_after_binding"] = True
    else:
        source_binding["formal_observation_refreshed_after_binding"] = False
    # Bind the formal source before handing its first observation to policy.
    worker.request("reset", observation)
    final_settling_steps = getattr(
        args,
        "final_settling_steps",
        DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    )

    def finish(row):
        row.setdefault("primary_failure_joint_hold_commits", joint_hold_commits)
        row.setdefault("policy_clock_semantics_id", FORMAL_POLICY_CLOCK_SEMANTICS_ID)
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
            motion_plan.metadata()["protocol_id"] if motion_plan is not None else None,
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
                    "formal_source_bound": source_binding.get("formal_source_bound"),
                    "formal_task_name_bound": source_binding.get("task_name"),
                    "formal_task_semantics_matched": source_binding.get(
                        "task_semantics_matched"
                    ),
                    "formal_task_tree_matched": source_binding.get("task_tree_matched"),
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
                        source_binding.get("robot_external_collision_pairs_matched")
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
            scenario=args.scenario,
            trigger_step=trigger_step,
            trigger_reached=trigger_reached,
            smooth_steps=args.smooth_steps,
        )

    from rlbench.backend.exceptions import InvalidActionError

    control_attempts = 0
    while committed_policy_steps < args.horizon:
        if last_scenario_policy_step != committed_policy_steps:
            if trigger_step is not None and committed_policy_steps >= trigger_step:
                trigger_reached = True
            event = controller.apply(
                task_environment,
                step=committed_policy_steps,
                horizon=worker.policy_steps,
            )
            last_scenario_policy_step = committed_policy_steps
            if event.get("trigger_step") != trigger_step and args.scenario != "static":
                raise RuntimeError(
                    "scenario controller returned an inconsistent trigger"
                )
            if event.get("applied"):
                if trigger_step is None or committed_policy_steps < trigger_step:
                    raise RuntimeError("scenario controller applied before the trigger")
                observation = task_environment.get_observation()
                event["policy_observation_refreshed"] = True
                events.append(event)
        response = worker.request("act", observation)
        action = response.get("action")
        if response.get("policy_failed") is True:
            transaction_id = response.get("transaction_id")
            if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
                raise RuntimeError(
                    "failed closed-loop policy cycle did not return a transaction id"
                )
            worker.request(
                "commit",
                transaction_id=transaction_id,
                primary_action_succeeded=False,
            )
            committed_policy_steps += 1
            return finish(
                {
                    "episode": episode,
                    "success": False,
                    "steps": control_attempts,
                    "control_attempts": control_attempts,
                    "reason": "policy_structured_failure",
                    "policy_failure_reasons": response.get("failure_reasons", {}),
                    "invalid_actions": invalid_actions,
                    "interventions": events,
                }
            )
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
            return finish(
                {
                    "episode": episode,
                    "success": bool(settling["success"]),
                    "steps": control_attempts,
                    "control_attempts": control_attempts,
                    "reason": reason,
                    "invalid_actions": invalid_actions,
                    "interventions": events,
                    "final_settling": settling,
                }
            )
        transaction_id = response.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy worker did not return an action transaction id")
        control_attempts += 1
        primary_action_succeeded = False
        try:
            observation, reward, terminate = task_environment.step(
                np.asarray(action, dtype=np.float64)
            )
            primary_action_succeeded = True
        except InvalidActionError:
            invalid_actions += 1
            try:
                observation, reward, terminate, policy_complete = (
                    commit_joint_hold_after_primary_failure(
                        task_environment,
                        worker,
                        transaction_id=transaction_id,
                    )
                )
            except InvalidActionError:
                return finish(
                    {
                        "episode": episode,
                        "success": False,
                        "steps": control_attempts,
                        "control_attempts": control_attempts,
                        "reason": "joint_hold_failed",
                        "invalid_actions": invalid_actions,
                        "interventions": events,
                    }
                )
            joint_hold_commits += 1
            committed_policy_steps += 1
        except Exception:
            worker.request("abort", transaction_id=transaction_id)
            raise
        if primary_action_succeeded:
            commit = worker.request("commit", transaction_id=transaction_id)
            policy_complete = bool(commit.get("complete"))
            committed_policy_steps += 1
        settling = None
        if reward > 0.0:
            reason = "success"
        elif terminate:
            reason = "terminate"
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
        return finish(
            {
                "episode": episode,
                "success": bool(reward > 0.0 or (settling or {}).get("success")),
                "steps": control_attempts,
                "control_attempts": control_attempts,
                "reason": reason,
                "invalid_actions": invalid_actions,
                "interventions": events,
                **({"final_settling": settling} if settling is not None else {}),
            }
        )
    return finish(
        {
            "episode": episode,
            "success": False,
            "steps": control_attempts,
            "control_attempts": control_attempts,
            "reason": "horizon",
            "invalid_actions": invalid_actions,
            "interventions": events,
        }
    )


def _stage_motion_plan_batch(args, task_class):
    """Generate A/B plans in one disposable, persistent staging process."""

    import rlbench.environment as environment_module
    from rlbench.observation_config import ObservationConfig

    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    environment = environment_module.Environment(
        action_mode=_make_action_mode(),
        obs_config=observation_config,
        headless=args.headless,
    )
    restore_scene, _ = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=args.headless,
    )
    launched = False
    plans = []
    try:
        environment.launch()
        launched = True
        variation_count = task_class(
            environment._pyrep,
            environment._robot,
        ).variation_count()
        if args.variation >= variation_count:
            raise ValueError("variation is outside task variation count")
        for episode in range(args.episodes):
            plans.append(
                stage_scenario_motion_plan(
                    environment,
                    task_class,
                    task_name=args.task,
                    episode_seed=args.seed + episode,
                    variation=args.variation,
                    max_attempts=args.intervention_attempts,
                )
            )
            print(
                f"staged {args.task} A/B {episode + 1}/{args.episodes}",
                flush=True,
            )
    finally:
        try:
            if launched:
                environment.shutdown()
        finally:
            restore_scene()
    payload = staged_motion_plan_batch(
        task_name=args.task,
        base_seed=args.seed,
        variations=[args.variation] * args.episodes,
        plans=plans,
    )
    atomic_json(args.stage_motion_plans_output, payload)
    return payload


def _motion_plan_cache_path(args):
    if getattr(args, "motion_plans", None) is not None:
        return Path(args.motion_plans)
    return (
        DEFAULT_RESULTS_DIR
        / "motion_plans"
        / (
            f"{args.task}_variation{args.variation}_seed{args.seed}_n{args.episodes}_v34.json"
        )
    )


def _load_fixed_motion_plans(args):
    """Read one preregistered batch; formal runs never generate or repair it."""

    if args.eval_set_id is None:
        raise RuntimeError("formal evaluation requires --eval-set-id")
    if args.motion_plans is not None:
        raise RuntimeError("--motion-plans is not allowed for fixed formal evaluation")
    if args.seed != GLOBAL_EVAL_SEED_START or args.episodes != FIXED_EVAL_EPISODES:
        raise RuntimeError(
            "formal evaluation seed/episode count differs from the fixed eval set"
        )
    manifest, selected = fixed_environment_plans(args.eval_set_id, args.task)
    payload = selected["payload"]
    plans = selected["plans"]
    if payload.get("variation_schedule") != [args.variation] * args.episodes:
        raise RuntimeError("fixed eval-set variation schedule does not match")
    if any(
        plan.validation.get("goal_sampling_max_attempts") != args.intervention_attempts
        for plan in plans
    ):
        raise RuntimeError("fixed eval-set goal-sampling budget is inconsistent")
    return manifest, selected


def evaluate(args):
    validate_formal_artifact_paths(output=args.output, models_dir=args.models_dir)
    with reserve_output(args.output):
        return _evaluate_reserved(args)


def _evaluate_reserved(args):
    _validate_formal_execution_args(args)
    video_capture_enabled = _v4_video_capture_enabled(args)
    import rlbench.environment as environment_module
    from rlbench.observation_config import ObservationConfig

    # Reject noncanonical budgets before a staging child can create a cache.
    _validate_v3_protocol_budgets(args)
    module_name, class_name = TASKS[args.task]
    task_class = getattr(importlib.import_module(module_name), class_name)
    eval_set, selected_batch = _load_fixed_motion_plans(args)
    selected_motion_plan_payload = selected_batch["payload"]
    motion_plan_payload = (
        selected_motion_plan_payload["runtime_batch"]
        if isinstance(selected_motion_plan_payload.get("runtime_batch"), dict)
        else selected_motion_plan_payload
    )
    motion_plans = selected_batch["plans"]
    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    video_capture_config = LightweightCaptureConfig() if video_capture_enabled else None
    video_cell_dir = None
    video_cell_key = None
    episode_videos = []
    if video_capture_enabled:
        from rlbench.observation_config import CameraConfig

        observation_config.camera_configs = {
            video_capture_config.camera: CameraConfig(
                rgb=True,
                depth=False,
                point_cloud=False,
                mask=False,
            )
        }
        video_cell_dir, video_cell_key = _v4_video_cell(
            getattr(
                args,
                "v4_evaluation_video_root",
                DEFAULT_V4_EVALUATION_VIDEO_ROOT,
            ),
            args.task,
            args.scenario,
        )
        video_cell_dir = _prepare_v4_video_cell(video_cell_dir)
    controller_profile = _resolved_controller_profile(args)
    controller_config = _controller_config(controller_profile)
    action_mode = _make_action_mode(controller_profile, controller_config)
    environment = environment_module.Environment(
        action_mode=action_mode,
        obs_config=observation_config,
        headless=args.headless,
    )
    restore_scene, scene_launch = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=args.headless,
        camera_observations_requested=video_capture_enabled,
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
            policy_type=getattr(args, "policy_type", "dynamac"),
            closed_loop_models_dir=getattr(
                args, "closed_loop_models_dir", CLOSED_LOOP_MODELS_DIR
            ),
            diagnostics_dir=getattr(args, "policy_diagnostics_dir", None),
        )
        intervention_registry, trigger_authentication = (
            _authenticated_v3_dynamic_trigger(args, worker)
        )
        args.trigger_step = (
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
        if args.variation >= variation_count:
            raise ValueError("variation is outside task variation count")
        for episode in range(args.episodes):
            episode_motion_plan = motion_plans[episode]
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
                variation=args.variation,
                verify_instance=False,
            )

            def run_formal_episode(formal_task_environment):
                return _run_episode(
                    formal_task_environment,
                    worker,
                    args,
                    episode,
                    motion_plan=episode_motion_plan,
                    descriptions=descriptions,
                    observation=observation,
                    fresh_task_generation=fresh_task_generation,
                )

            result, episode_video = _run_episode_with_optional_v4_video(
                task_environment,
                observation,
                enabled=video_capture_enabled,
                cell_dir=video_cell_dir,
                cell_key=video_cell_key,
                episode=episode,
                episode_seed=args.seed + episode,
                run_episode=run_formal_episode,
                capture_config=video_capture_config,
            )
            results.append(result)
            if episode_video is not None:
                episode_videos.append(episode_video)
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
        any(event["applied"] for event in item["interventions"]) for item in results
    ]
    effective_by_episode = [item["intervention_effective"] is True for item in results]
    eligible_by_episode = [item["intervention_eligible"] is True for item in results]
    preterminal_by_episode = [
        item["pre_intervention_terminal"] is True for item in results
    ]
    complete_by_episode = [item["intervention_complete"] is True for item in results]
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
        SCENARIOS[args.scenario],
        trigger_fraction=args.trigger_fraction,
        trigger_step=getattr(args, "trigger_step", None),
        total_steps=args.smooth_steps,
        max_attempts=args.intervention_attempts,
        motion_plan=(motion_plans[0] if motion_plans else None),
    ).protocol_metadata()
    release = getattr(args, "release", "v3")
    summary = {
        "schema": (
            "dynamac-table-i-evaluation-v4"
            if release == "v4"
            else "dynamac-table-i-evaluation-v3"
        ),
        **({"release": "v4"} if release == "v4" else {}),
        "policy_type": getattr(args, "policy_type", "dynamac"),
        "protocol_label": PROTOCOL_LABEL,
        "paper_comparable": False,
        "task": args.task,
        "scenario": args.scenario,
        "episodes": args.episodes,
        "episodes_requested": args.episodes,
        "episodes_completed": len(results),
        "seed": args.seed,
        "variation": args.variation,
        "variation_count": variation_count,
        "variation_schedule": [args.variation] * args.episodes,
        "horizon": args.horizon,
        "evaluation_protocol_id": evaluation_protocol_id(
            args.max_primary_action_attempts,
            controller_profile,
        ),
        "fixed_eval_set": {
            "evaluation_set_id": eval_set["payload"]["evaluation_set_id"],
            "manifest_sha256": eval_set["manifest_sha256"],
            "spec_sha256": eval_set["payload"]["spec"]["sha256"],
            "selected_batch_sha256": eval_set["payload"]["environment_plan_batches"][
                args.task
            ]["sha256"],
            "selected_batch_fingerprint": selected_motion_plan_payload[
                "batch_fingerprint"
            ],
            "formal_access": "canonical_id_read_only_no_generation",
        },
        "controller": {
            **global_ik_controller_metadata(controller_config),
            "worker_clock_handshake_id": worker.policy_clock_semantics_id,
            "worker_gripper_timing_handshake": worker.gripper_timing,
            "formal_episode_initialization": DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
            "final_settling": final_settling_metadata(args.final_settling_steps),
            "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
            "gripper_timing": worker.gripper_timing,
            "scene_launch": scene_launch,
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "successes": successes,
        "success_rate": successes / float(args.episodes),
        "episode_accounting": {
            "schema": DYNAMIC_EPISODE_ACCOUNTING_SCHEMA,
            "planned_episode_denominator": args.episodes,
            "completed_episode_count": len(results),
            "successes_in_planned_denominator": successes,
            "success_rate_all_planned_episodes": successes / float(args.episodes),
            "trigger_reached_count": sum(
                item["intervention_reached"] is True for item in results
            ),
            "intervention_complete_count": sum(complete_by_episode),
            "dynamic_condition_unexercised_count": sum(
                item.get("dynamic_condition_unexercised") is True for item in results
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
        "gripper_timing": global_gripper_timing_metadata(),
        "protocol": {
            "label": PROTOCOL_LABEL,
            "paper_comparable": False,
            "comparison_scope": "diagnostic_against_paper_targets_not_protocol_exact",
            "dynamic_method": SCENARIOS[args.scenario],
            "motion_protocol": motion_protocol,
            "legacy_trigger_fraction_ignored": args.trigger_fraction,
            "trigger_reference_domain": ("successfully_committed_policy_ticks"),
            "fitted_policy_steps": worker.policy_steps,
            "trigger_policy_step": args.trigger_step,
            "trigger_authentication": trigger_authentication,
            "intervention_registry_schema": intervention_registry["schema"],
            "intervention_registry_fingerprint": intervention_registry["fingerprint"],
            "smooth_motion_calls": args.smooth_steps,
            "intervention_max_attempts": args.intervention_attempts,
            "dynamic_episode_accounting_schema": (DYNAMIC_EPISODE_ACCOUNTING_SCHEMA),
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
                item.get("dynamic_condition_unexercised") is True for item in results
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
                True if args.scenario == "static" else all(covered_by_episode)
            ),
            "source": (
                "frozen V3 task-specific trigger plus independently staged "
                "waypoint-validated boundary-root motion"
            ),
            "staged_motion_plan_cache": (
                {
                    "schema": motion_plan_payload["schema"],
                    "protocol_id": motion_plan_payload["protocol_id"],
                    "batch_fingerprint": motion_plan_payload["batch_fingerprint"],
                    "plan_fingerprints": [plan.fingerprint() for plan in motion_plans],
                    "scenario_independent": True,
                    "seed_domain": motion_plan_payload["seed_domain"],
                    "goal_sampling_max_attempts": args.intervention_attempts,
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
                        "variation_schedule": motion_plan_payload["variation_schedule"],
                    },
                    "formal_access": "canonical_eval_set_read_only",
                    "staging_shutdown_before_formal_launch": True,
                    "fresh_task_generation_per_formal_episode": True,
                }
                if motion_plan_payload is not None
                else None
            ),
            "claim_boundary": (
                "The live five-demonstration cohort, fixed variation, segmentation, "
                "and dynamic displacement distribution remain explicit local choices. "
                "V3 trigger ticks are preregistered and checkpoint-authenticated, but "
                "the paper's exact Table I datasets and full DynaBench protocol remain "
                "unavailable, so these results are not paper-comparable."
            ),
        },
        "results": results,
        "fresh_task_generation": {
            "required_per_formal_episode": True,
            "all_episodes_recorded": all(
                isinstance(item.get("fresh_task_generation"), dict) for item in results
            ),
            "evidence": [item["fresh_task_generation"] for item in results],
        },
        "final_settling_protocol": final_settling_metadata(args.final_settling_steps),
        "motion_plan_batch_fingerprint": (
            selected_motion_plan_payload["batch_fingerprint"]
            if selected_motion_plan_payload is not None
            else None
        ),
    }
    if video_capture_enabled:
        video_capture_metadata = {
            "release": "v4",
            "cell_key": video_cell_key,
            "cell_dir": str(video_cell_dir),
            "episodes_recorded": len(episode_videos),
            "capture_config": dict(video_capture_config.audit()),
            "paper_success_rate": V4_VIDEO_PAPER_TARGETS[(args.task, args.scenario)],
        }

        def finalize_videos():
            return _finalize_v4_video_cell(
                video_cell_dir,
                episode_videos,
                cell_key=video_cell_key,
                successes=successes,
                episodes=args.episodes,
                paper_success_rate=V4_VIDEO_PAPER_TARGETS[(args.task, args.scenario)],
                cell_metadata={
                    "evaluator": "unimanual_evaluate",
                    "task": args.task,
                    "scenario": args.scenario,
                    "variation": args.variation,
                    "formal_result": str(args.output),
                },
            )

    else:
        video_capture_metadata = None
        finalize_videos = None
    _commit_formal_result_with_optional_v4_videos(
        args.output,
        summary,
        enabled=video_capture_enabled,
        capture_metadata=video_capture_metadata,
        finalize_videos=finalize_videos,
    )
    print(f"wrote {args.output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--policy-type",
        choices=("dynamac", "closed_loop_multistream"),
        default="dynamac",
    )
    parser.add_argument(
        "--closed-loop-models-dir",
        type=Path,
        default=CLOSED_LOOP_MODELS_DIR,
    )
    parser.add_argument("--policy-diagnostics-dir", type=Path, default=None)
    parser.add_argument(
        "--controller-profile",
        choices=(
            "auto",
            GLOBAL_IK_CONTROLLER_PROFILE,
            STAGE6_IK_CONTROLLER_PROFILE,
        ),
        default="auto",
        help=(
            "IK executor profile; auto preserves the V4 controller for DynaMAC "
            "and enables Cartesian-verified execution for the closed-loop policy."
        ),
    )
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=GLOBAL_EVAL_SEED_START)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--trigger-fraction",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Legacy audit value only; V3 uses the authenticated task-specific "
            "absolute trigger tick."
        ),
    )
    parser.add_argument(
        "--trigger-step",
        type=int,
        default=None,
        help="Explicit committed-policy trigger tick; V3 task profiles set this.",
    )
    parser.add_argument("--smooth-steps", type=int, default=10)
    parser.add_argument("--intervention-attempts", type=int, default=100)
    parser.add_argument(
        "--final-settling-steps",
        type=int,
        default=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    )
    parser.add_argument("--motion-plans", type=Path, default=None)
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
    parser.add_argument(
        "--release",
        choices=("v3", "v4"),
        default="v3",
        help="Evaluation release gate; V4 requires formal episode video capture.",
    )
    parser.add_argument(
        "--record-v4-evaluation-videos",
        action="store_true",
        help=(
            "Stream front RGB from the actual formal V4 episodes and retain "
            "the fixed outcome-stratified sample; disabled by default for V3."
        ),
    )
    parser.add_argument(
        "--v4-evaluation-video-root",
        type=Path,
        default=DEFAULT_V4_EVALUATION_VIDEO_ROOT,
        help="Root for V4 formal evaluation video cells.",
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
    if args.stage_motion_plans_output is not None:
        _validate_v3_protocol_budgets(args)
        if (
            args.episodes < 1
            or args.seed < 0
            or args.variation < 0
            or args.intervention_attempts < 1
        ):
            raise ValueError("staging episode parameters are invalid")
        module_name, class_name = TASKS[args.task]
        task_class = getattr(importlib.import_module(module_name), class_name)
        with reserve_output(args.stage_motion_plans_output):
            _stage_motion_plan_batch(args, task_class)
        return 0
    if (
        args.episodes < 1
        or args.horizon < 1
        or args.smooth_steps < 1
        or args.max_primary_action_attempts < 1
        or args.intervention_attempts < 1
        or args.final_settling_steps < 0
    ):
        raise ValueError("episodes, horizon, and smooth steps must be positive")
    if args.seed < 0 or args.variation < 0:
        raise ValueError("seed and variation must be non-negative")
    if args.motion_plan_wait_timeout <= 0.0:
        raise ValueError("motion-plan wait timeout must be positive")
    if not 0.0 <= args.trigger_fraction <= 1.0:
        raise ValueError("trigger fraction must lie in [0, 1]")
    if args.output is None:
        family = "table_i" if args.scenario == "static" else "table_i_dynamic"
        args.output = (
            DEFAULT_RESULTS_DIR
            / family
            / (
                f"{args.task}_{args.scenario}_variation{args.variation}_"
                f"seed{args.seed}_n{args.episodes}_h{args.horizon}.json"
            )
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
