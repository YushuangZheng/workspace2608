"""Run direct RLBench evaluation against the current DynaMAC policy worker.

The module is Python 3.8 compatible and imports RLBench only after argument
parsing, so ``--help`` also works without launching CoppeliaSim.  The simulator
uses absolute world-frame end-effector IK for both Panda arms.  Policy math is
kept in the Python 3.10 worker started by this process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import select
import subprocess
from pathlib import Path

import numpy as np

from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
    fixed_environment_plans,
    validate_formal_artifact_paths,
)
from integrations.rlbench.rlbench_dynamac.report.evaluation_videos import (
    DEFAULT_MANIFEST_NAME as V4_VIDEO_SELECTION_MANIFEST,
    DEFAULT_OUTPUT_ROOT as DEFAULT_V4_EVALUATION_VIDEO_ROOT,
    EpisodeVideo,
    LightweightCaptureConfig,
    ObservationVideoRecorder,
    RecordingTaskEnvironment,
    finalize_cell_videos,
)
from integrations.rlbench.rlbench_dynamac.core.gripper_timing import (
    GLOBAL_GRIPPER_TIMING_PROTOCOL_ID,
    global_gripper_timing_metadata,
)
from integrations.rlbench.rlbench_closed_loop.protocol import (
    closed_loop_gripper_timing_metadata,
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
    apply_gripper_for_policy_target,
    commit_joint_hold_after_primary_failure,
    execute_global_ik_ee_control,
    execute_stage6_ik_ee_control,
    execute_joint_target_control,
    final_settling_metadata,
    global_ik_controller_metadata,
    initialize_fresh_task_generation,
    initialize_global_ik_controller_diagnostics,
    initialize_ik_solver_diagnostics,
    policy_action_execution_status,
    policy_action_execution_statuses,
    set_policy_gripper_authorization,
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
from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import (
    V4_LIFT_MOTION_PROTOCOL_ID,
    V4_LIFT_RUNTIME_LOADER_ID,
    V4_LIFT_TASK,
    V4_LIFT_TRIGGER_STEP,
    build_v4_lift_task_scoped_plan_batch,
    load_v4_lift_intervention_protocol,
    load_v4_lift_motion_plan_batch,
    load_v4_lift_motion_source_protocol,
    stage_v4_lift_motion_plan,
    v4_lift_task_identity_components,
    v4_lift_trigger_authentication,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_eval_v4 import (
    V4_STORE_MOTION_PROTOCOL_ID,
    V4_STORE_RUNTIME_LOADER_ID,
    V4_STORE_TRIGGER_STEPS,
    StoreBottleMultiEntityController,
    StoreBottleMultiEntityPlan,
    build_v4_store_task_scoped_plan_batch,
    load_v4_store_intervention_protocol,
    load_v4_store_motion_plan_batch,
    load_v4_store_motion_source_protocol,
    stage_v4_store_motion_plan,
    v4_store_task_class,
    v4_store_task_identity_components,
    v4_store_trigger_authentication,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    STORE_BOTTLE_TASK_NAME,
)

from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT

DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3"
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v3"
V4_MODELS_DIR = INTEGRATION_ROOT / "models" / "v4"
V4_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v4"
CLOSED_LOOP_MODELS_DIR = INTEGRATION_ROOT / "models" / "closed_loop_phase6_v1"
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

POLICY_CLOCK_SEMANTICS_ID = FORMAL_POLICY_CLOCK_SEMANTICS_ID
GRIPPER_PROTOCOL = DiscreteGripperProtocol(bimanual=True)
V4_VIDEO_SIDECAR_SCHEMA = "dynamac-v4-formal-evaluation-video-sidecar-v1"
V4_VIDEO_PAPER_TARGETS = {
    ("bimanual_put_bottle_in_fridge", "static"): 0.82,
    ("bimanual_put_bottle_in_fridge", "teleport"): 0.82,
    ("bimanual_handover_item", "static"): 0.97,
    ("bimanual_handover_item", "teleport"): 0.97,
    ("bimanual_sweep_to_dustpan", "static"): 1.0,
    ("bimanual_sweep_to_dustpan", "teleport"): 1.0,
    ("bimanual_lift_tray", "static"): 1.0,
    ("bimanual_lift_tray", "teleport"): 1.0,
}


def _v4_video_capture_enabled(args):
    release = getattr(args, "release", "v3")
    enabled = bool(getattr(args, "record_v4_evaluation_videos", False))
    if release not in {"v3", "v4"}:
        raise ValueError("evaluation release must be v3 or v4")
    if release != "v4" and enabled:
        raise ValueError("V4 episode video capture requires --release v4")
    return enabled


def _v4_video_cell(root, task, scenario):
    cell_key = f"{task}/{scenario}"
    return Path(root) / task / scenario, cell_key


def _prepare_v4_video_cell(cell_dir):
    cell_dir = Path(cell_dir)
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix a V4 video cell with existing artifacts: {cell_dir}"
        )
    cell_dir.mkdir(parents=True, exist_ok=True)
    return cell_dir


def _v4_video_selection_audit(cell_dir, manifest):
    manifest_path = Path(cell_dir) / V4_VIDEO_SELECTION_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("V4 video finalizer did not write its selection manifest")
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "schema": manifest["schema"],
        "selection": manifest["selection"],
        "selected_episodes": [row["episode"] for row in manifest["selected"]],
        "all_episodes_recorded_before_selection": manifest[
            "all_episodes_recorded_before_selection"
        ],
    }


def _finalize_v4_video_cell(
    cell_dir,
    recordings,
    *,
    cell_key,
    successes,
    episodes,
    paper_success_rate,
    cell_metadata,
):
    manifest = finalize_cell_videos(
        cell_dir,
        recordings,
        cell_key=cell_key,
        successes=successes,
        episodes=episodes,
        paper_success_rate=paper_success_rate,
        cell_metadata=cell_metadata,
    )
    return _v4_video_selection_audit(cell_dir, manifest)


def _commit_formal_result_with_optional_v4_videos(
    output,
    payload,
    *,
    enabled,
    capture_metadata=None,
    finalize_videos=None,
):
    """Finalize V4 video selection before publishing the formal result."""

    if enabled:
        if not isinstance(capture_metadata, dict) or not callable(finalize_videos):
            raise ValueError("V4 result commit requires capture metadata and finalizer")
        selection = finalize_videos()
        payload["evaluation_video_capture"] = {
            **capture_metadata,
            "selection_manifest": selection,
            "formal_result_committed_after_video_selection": True,
        }
    atomic_json(output, payload)
    return payload


def _run_episode_with_optional_v4_video(
    task_environment,
    initial_observation,
    *,
    enabled,
    cell_dir,
    cell_key,
    episode,
    episode_seed,
    run_episode,
    capture_config=None,
    recorder_factory=None,
    proxy_factory=None,
):
    """Run one formal episode and optionally retain its streamed V4 recording."""

    if not enabled:
        return run_episode(task_environment), None
    if cell_dir is None or not cell_key:
        raise ValueError("V4 video capture requires a cell directory and key")
    capture_config = capture_config or LightweightCaptureConfig()
    recorder_factory = recorder_factory or ObservationVideoRecorder
    proxy_factory = proxy_factory or RecordingTaskEnvironment
    stem = f"episode_{episode:03d}_seed_{episode_seed}"
    video_path = Path(cell_dir) / f"{stem}.mp4"
    sidecar_path = Path(cell_dir) / f"{stem}.json"
    recorder = recorder_factory(video_path, config=capture_config)
    try:
        recorder.capture(initial_observation)
    except BaseException:
        recorder.abort()
        raise
    recording_environment = proxy_factory(task_environment, recorder)
    try:
        result = run_episode(recording_environment)
    except BaseException as exc:
        try:
            capture = recorder.close()
            atomic_json(
                sidecar_path,
                {
                    "schema": V4_VIDEO_SIDECAR_SCHEMA,
                    "cell_key": cell_key,
                    "episode": episode,
                    "episode_seed": episode_seed,
                    "formal_episode_completed": False,
                    "error_type": type(exc).__name__,
                    "capture": capture,
                },
            )
        except BaseException:
            recorder.abort()
            raise
        raise

    try:
        capture = recorder.close()
        success = bool(result["success"])
        atomic_json(
            sidecar_path,
            {
                "schema": V4_VIDEO_SIDECAR_SCHEMA,
                "cell_key": cell_key,
                "episode": episode,
                "episode_seed": episode_seed,
                "formal_episode_completed": True,
                "success": success,
                "reason": result.get("reason"),
                "steps": result.get("steps"),
                "invalid_actions": result.get("invalid_actions"),
                "capture": capture,
            },
        )
    except BaseException:
        recorder.abort()
        raise
    return result, EpisodeVideo(
        episode=episode,
        episode_seed=episode_seed,
        success=success,
        video=video_path,
        companions=(sidecar_path,),
    )


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
    if getattr(args, "max_primary_action_attempts", None) != 1:
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
    raise ValueError("unsupported bimanual controller profile")


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
        raise RuntimeError(
            "authenticated V3 trigger lies outside the loaded policy clock"
        )
    requested_reference = getattr(args, "scenario_reference_steps", None)
    if requested_reference is not None and requested_reference != worker.policy_steps:
        raise RuntimeError(
            "V3 trigger reference must equal the authenticated loaded policy clock"
        )
    return protocol, authentication


def _is_v4_lift(args):
    return getattr(args, "release", "v3") == "v4" and args.task == V4_LIFT_TASK


def _is_v4_store(args):
    return (
        getattr(args, "release", "v3") == "v4" and args.task == STORE_BOTTLE_TASK_NAME
    )


def _is_v4_task_scoped(args):
    return _is_v4_lift(args) or _is_v4_store(args)


def _task_model_content_identity(models_root, task_name):
    """Fingerprint one task model independently of its containing directory."""

    task_root = Path(models_root).resolve() / task_name
    if not task_root.is_dir():
        return None
    files = sorted(path for path in task_root.rglob("*") if path.is_file())
    if not files:
        return None
    return tuple(
        (
            path.relative_to(task_root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in files
    )


def _validate_v4_store_protocol_args(args):
    """Fail before simulator launch on any StoreBottle V4 protocol drift."""

    intervention = load_v4_store_intervention_protocol()
    motion = load_v4_store_motion_source_protocol()
    if args.scenario not in intervention["formal_scenarios"]:
        raise ValueError(
            "StoreBottle V4 formal evaluation supports only static/teleport"
        )
    if args.scenario_max_attempts != motion["goal_sampling_max_attempts"]:
        raise RuntimeError("StoreBottle V4 goal-sampling budget differs from protocol")
    if args.final_settling_steps != intervention["final_settling_physics_steps"]:
        raise RuntimeError("StoreBottle V4 final-settling budget differs from protocol")
    if getattr(args, "scenario_trigger_step", None) is not None:
        raise RuntimeError(
            "StoreBottle V4 has two entity triggers; --scenario-trigger-step is invalid"
        )
    candidate_identity = _task_model_content_identity(
        args.models_dir,
        STORE_BOTTLE_TASK_NAME,
    )
    frozen_identity = _task_model_content_identity(
        V4_MODELS_DIR,
        STORE_BOTTLE_TASK_NAME,
    )
    if candidate_identity is None or candidate_identity != frozen_identity:
        raise RuntimeError(
            "StoreBottle V4 formal evaluation requires model content identical "
            "to the frozen models/v4 snapshot"
        )
    return intervention, motion


def _authenticated_v4_store_triggers(args, worker):
    intervention, _motion = _validate_v4_store_protocol_args(args)
    authentication = v4_store_trigger_authentication(worker.policy_steps)
    requested_reference = getattr(args, "scenario_reference_steps", None)
    if requested_reference is not None and requested_reference != worker.policy_steps:
        raise RuntimeError(
            "StoreBottle V4 trigger reference must equal the loaded policy clock"
        )
    return intervention, authentication


def _validate_v4_lift_protocol_args(args):
    """Fail before simulator launch on any V4 LiftTray protocol drift."""

    intervention = load_v4_lift_intervention_protocol()
    motion = load_v4_lift_motion_source_protocol()
    if args.scenario not in intervention["formal_scenarios"]:
        raise ValueError("V4 LiftTray formal evaluation supports only static/teleport")
    if args.scenario_max_attempts != motion["goal_sampling_max_attempts"]:
        raise RuntimeError("V4 LiftTray goal-sampling budget differs from protocol")
    if args.final_settling_steps != intervention["final_settling_physics_steps"]:
        raise RuntimeError("V4 LiftTray final-settling budget differs from protocol")
    requested = getattr(args, "scenario_trigger_step", None)
    if requested is not None and requested != V4_LIFT_TRIGGER_STEP:
        raise RuntimeError("V4 LiftTray trigger must be committed policy tick 35")
    return intervention, motion


def _authenticated_v4_lift_trigger(args, worker):
    intervention, _motion = _validate_v4_lift_protocol_args(args)
    authentication = v4_lift_trigger_authentication(worker.policy_steps)
    requested_reference = getattr(args, "scenario_reference_steps", None)
    if requested_reference is not None and requested_reference != worker.policy_steps:
        raise RuntimeError(
            "V4 LiftTray trigger reference must equal the loaded policy clock"
        )
    return intervention, authentication


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


def _make_action_mode(
    controller_profile=GLOBAL_IK_CONTROLLER_PROFILE,
    controller_config=None,
    external_solver_factory=None,
):
    """Construct the fork's missing bimanual absolute-EE IK action mode."""

    if controller_profile not in {
        GLOBAL_IK_CONTROLLER_PROFILE,
        STAGE6_IK_CONTROLLER_PROFILE,
        FROZEN_V4_CONTROLLER_PROFILE,
    }:
        raise ValueError("unsupported bimanual controller profile")
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
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        ArmActionMode,
        assert_action_shape,
        assert_unit_quaternion,
    )
    from rlbench.backend.exceptions import InvalidActionError

    class BimanualAbsoluteEndEffectorIK(ArmActionMode):
        def __init__(self):
            self._controller_profile = controller_profile
            self._controller_config = controller_config
            self._external_solver_factory = external_solver_factory
            self._external_solvers = {}
            self._last_policy_action_status = "reached"
            self._last_policy_action_statuses = {
                "left": "reached",
                "right": "reached",
            }
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

        def policy_action_status(self):
            return (
                self._last_policy_action_status
                if self._controller_profile == STAGE6_IK_CONTROLLER_PROFILE
                else "reached"
            )

        def policy_action_statuses(self):
            return (
                dict(self._last_policy_action_statuses)
                if self._controller_profile == STAGE6_IK_CONTROLLER_PROFILE
                else {"left": "reached", "right": "reached"}
            )

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
            """Use local Jacobian IK, then validated collision-aware sampling.

            The author reports using IK for RLBench and reserving a heavier
            fallback for goals that are too far from the current pose.  PyRep's
            Jacobian solver is deliberately local; a single failure must not
            turn every later clock tick into a no-op.  Sampling solves the same
            absolute target without changing the policy command or clock.
            """

            return solve_absolute_ee_ik_with_sampling_fallback(
                arm,
                target,
                diagnostics=self._execution_diagnostics,
                ik_error=IKError,
                configuration_error=ConfigurationError,
                invalid_action_error=InvalidActionError,
                error_message="bimanual absolute end-effector IK failed",
            )

        def action(self, scene, action, ignore_collisions=True):
            del ignore_collisions  # V4 collision policy is frozen in the IK helper
            assert_action_shape(action, (14,))
            right_action = np.asarray(action[:7], dtype=np.float64)
            left_action = np.asarray(action[7:], dtype=np.float64)
            assert_unit_quaternion(right_action[3:])
            assert_unit_quaternion(left_action[3:])
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
                    per_arm_status = {}
                    executor_kwargs = {}
                    if executor is execute_stage6_ik_ee_control:
                        executor_kwargs["per_arm_status_out"] = per_arm_status
                    status = executor(
                        scene,
                        (
                            (scene.robot.right_arm, right_action),
                            (scene.robot.left_arm, left_action),
                        ),
                        config=self._controller_config,
                        diagnostics=self._execution_diagnostics,
                        external_solver_factory=self._external_solver_for_arm,
                        ik_error=IKError,
                        configuration_error=ConfigurationError,
                        configuration_path_error=ConfigurationPathError,
                        invalid_action_error=InvalidActionError,
                        path_algorithm=ConfigurationPathAlgorithms.RRTConnect,
                        error_message=(
                            "bimanual formal pseudo/TRAC-IK/sampling/path EE "
                            "control failed"
                        ),
                        **executor_kwargs,
                    )
                except InvalidActionError:
                    self._execution_diagnostics[
                        "trac_ik_distance_controller_invalid_actions"
                    ] += 1
                    raise
                self._execution_diagnostics[f"joint_target_{status}"] += 1
                self._last_policy_action_status = status
                self._last_policy_action_statuses = {
                    "right": per_arm_status.get(id(scene.robot.right_arm), status),
                    "left": per_arm_status.get(id(scene.robot.left_arm), status),
                }
                return status
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
            self._last_policy_action_status = "reached"
            self._last_policy_action_statuses = {
                "left": "reached",
                "right": "reached",
            }
            return "reached"

        def action_shape(self, scene):
            del scene
            return (14,)

        def unimanual_action_shape(self, scene):
            del scene
            return (7,)

    class BoundedBimanualMoveArmThenGripper(BimanualMoveArmThenGripper):
        def __init__(self, arm_action_mode, gripper_action_mode):
            super().__init__(arm_action_mode, gripper_action_mode)
            self._policy_gripper_authorization = None

        def set_policy_gripper_authorization(self, authorization):
            if authorization is None:
                self._policy_gripper_authorization = None
                return
            if set(authorization) != {"left", "right"}:
                raise ValueError(
                    "bimanual gripper authorization must cover left and right"
                )
            normalized = {}
            for arm, value in authorization.items():
                if value is not None and not isinstance(value, bool):
                    raise TypeError("bimanual gripper authorization must be Boolean")
                normalized[arm] = value
            self._policy_gripper_authorization = normalized

        def action(self, scene, action):
            if len(action) != 18:
                raise ValueError("bimanual absolute EE action must contain 18 values")
            arm_size = int(np.prod(self.arm_action_mode.unimanual_action_shape(scene)))
            gripper_size = int(
                np.prod(self.gripper_action_mode.unimanual_action_shape(scene))
            )
            lane_size = arm_size + gripper_size + 1
            if lane_size != 9:
                raise RuntimeError("unexpected bimanual absolute EE lane shape")
            right = np.asarray(action[:lane_size])
            left = np.asarray(action[lane_size:])
            arm_action = np.concatenate((right[:arm_size], left[:arm_size]))
            gripper_action = np.concatenate(
                (
                    right[arm_size : arm_size + gripper_size],
                    left[arm_size : arm_size + gripper_size],
                )
            )
            ignore_collisions = [bool(right[-1]), bool(left[-1])]
            status = self.arm_action_mode.action(
                scene,
                arm_action,
                ignore_collisions,
            )
            authorization = self._policy_gripper_authorization
            self._policy_gripper_authorization = None
            if authorization is None:
                apply_gripper_for_policy_target(
                    self.gripper_action_mode,
                    scene,
                    gripper_action,
                    arm_status=status,
                )
                return
            statuses = self.arm_action_mode.policy_action_statuses()
            ready = {
                arm: (
                    statuses[arm] == "reached"
                    if authorization[arm] is None
                    else bool(authorization[arm])
                )
                for arm in ("right", "left")
            }
            if not any(ready.values()):
                return
            current = np.asarray(
                [
                    float(
                        all(
                            value > 0.9
                            for value in scene.robot.right_gripper.get_open_amount()
                        )
                    ),
                    float(
                        all(
                            value > 0.9
                            for value in scene.robot.left_gripper.get_open_amount()
                        )
                    ),
                ],
                dtype=np.float64,
            )
            masked = gripper_action.copy()
            for index, arm in enumerate(("right", "left")):
                if not ready[arm]:
                    masked[index] = current[index]
            self.gripper_action_mode.action(scene, masked)

    return BoundedBimanualMoveArmThenGripper(
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
            "gripper_open": float(observation.left.gripper_open),
        },
        "right": {
            "gripper_pose": np.asarray(observation.right.gripper_pose).tolist(),
            "gripper_open": float(observation.right.gripper_open),
        },
        "task_low_dim_state": np.asarray(state).reshape(-1).tolist(),
    }


def _noop_action(observation):
    """Frozen-V4 current-EE no-op in the fork's 18D action layout."""

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
        closed_loop_feature_profile="full",
    ):
        self.timeout = float(timeout)
        if self.timeout <= 0.0:
            raise ValueError("policy timeout must be positive")
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
                "--feature-profile",
                str(closed_loop_feature_profile),
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
            if not response.get("ready"):
                raise RuntimeError("policy worker did not report ready")
            if not response.get("bimanual") or response.get("task") != task:
                raise RuntimeError(
                    "policy worker identity does not match the requested task"
                )
            self.policy_steps = int(response.get("policy_steps", 0))
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
                self.gripper_timing != closed_loop_gripper_timing_metadata()
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
            raise TimeoutError(
                f"policy worker did not respond within {self.timeout:g} seconds"
            )
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(
                f"policy worker exited without a response (code={code!r})"
            )
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
            isinstance(event, dict) and event.get("applied") is True for event in events
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


def _run_v4_store_episode(
    task_environment,
    worker,
    episode,
    horizon,
    *,
    scenario,
    max_primary_action_attempts,
    retry_exhaustion_mode=FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE,
    motion_plan,
    final_settling_steps,
    descriptions,
    observation,
    fresh_task_generation,
):
    """Run one StoreBottle episode with two independently scheduled roots."""

    retry_exhaustion_mode = validate_primary_retry_exhaustion_mode(
        retry_exhaustion_mode
    )
    if retry_exhaustion_mode != FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE:
        raise ValueError("formal StoreBottle requires the global joint-hold clock")
    if max_primary_action_attempts != 1:
        raise ValueError("formal StoreBottle requests each policy tick exactly once")
    if not isinstance(motion_plan, StoreBottleMultiEntityPlan):
        raise TypeError("StoreBottle V4 requires a multi-entity motion plan")
    if (
        descriptions is None
        or observation is None
        or not isinstance(
            fresh_task_generation,
            dict,
        )
    ):
        raise RuntimeError("formal StoreBottle episode requires fresh task input")
    controller = StoreBottleMultiEntityController(plan=motion_plan, scenario=scenario)
    source_binding = controller.bind_source(
        task_environment,
        descriptions=descriptions,
        fresh_task_generation=fresh_task_generation,
    )
    observation = task_environment.get_observation()
    source_binding["formal_observation_refreshed_after_binding"] = True
    worker.request("reset", observation)
    invalid_actions = 0
    joint_hold_commits = 0
    committed_policy_steps = 0
    last_scenario_policy_step = None
    scenario_events = []
    reached_entities = set()

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
        row.setdefault("motion_plan_fingerprint", motion_plan.fingerprint())
        row.setdefault("motion_plan_protocol_id", V4_STORE_MOTION_PROTOCOL_ID)
        row.setdefault(
            "motion_plan_evidence",
            {
                "plan_fingerprint": motion_plan.fingerprint(),
                "source_waypoint_validated": motion_plan.validation[
                    "source_waypoint_validated"
                ],
                "goal_waypoint_validated": motion_plan.validation[
                    "goal_waypoint_validated"
                ],
                "scene_safety": motion_plan.validation.get("scene_safety"),
                "formal_source_bound": source_binding["formal_source_bound"],
                "formal_task_semantics_matched": source_binding[
                    "task_semantics_matched"
                ],
                "formal_task_tree_matched": source_binding["task_tree_matched"],
                "formal_deterministic_source_reconstruction_passed": (
                    source_binding["deterministic_source_reconstruction"]["passed"]
                ),
                "policy_result_fields_read": False,
            },
        )
        required = set(controller.required_entities)
        applied = {
            event["entity"] for event in scenario_events if event.get("applied") is True
        }
        per_entity = {}
        for name in ("bottle", "fridge"):
            is_required = name in required
            is_reached = name in reached_entities
            entity_applied = name in applied
            per_entity[name] = {
                "required": is_required,
                "trigger_step": (V4_STORE_TRIGGER_STEPS[name] if is_required else None),
                "eligible": is_reached if is_required else False,
                "reached": is_reached if is_required else False,
                "applied": entity_applied,
                "effective": entity_applied if is_required else None,
                "complete": entity_applied if is_required else None,
            }
        dynamic = scenario == "teleport"
        all_reached = bool(dynamic and required and required <= reached_entities)
        all_applied = bool(dynamic and required and required <= applied)
        any_applied = bool(applied)
        if not dynamic:
            status = {
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
        else:
            status = {
                "trigger_step": {
                    name: V4_STORE_TRIGGER_STEPS[name] for name in sorted(required)
                },
                "intervention_eligible": all_reached,
                "intervention_reached": all_reached,
                "pre_intervention_terminal": not all_reached,
                "pre_intervention_terminal_outcome": (
                    ("success" if row.get("success") is True else "failure")
                    if not all_reached
                    else None
                ),
                "dynamic_condition_exercised": any_applied,
                "dynamic_condition_unexercised": not any_applied,
                "intervention_effective": all_applied if all_reached else None,
                "intervention_complete": all_applied if any_applied else None,
            }
        row.update(status)
        row["store_mode"] = motion_plan.mode
        row["store_required_entities"] = sorted(required)
        row["store_entity_interventions"] = per_entity
        row["scenario_events"] = scenario_events
        return row

    from rlbench.backend.exceptions import InvalidActionError

    control_attempts = 0
    while committed_policy_steps < horizon:
        if last_scenario_policy_step != committed_policy_steps:
            for entity in controller.required_entities:
                if committed_policy_steps == V4_STORE_TRIGGER_STEPS[entity]:
                    reached_entities.add(entity)
            observation, events = controller.apply(
                task_environment,
                observation,
                policy_step=committed_policy_steps,
            )
            scenario_events.extend(events)
            last_scenario_policy_step = committed_policy_steps
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
                primary_action_status="stopped",
                primary_action_applied=False,
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
                    "final_settling": settling,
                }
            )
        transaction_id = response.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy worker did not return an action transaction id")
        control_attempts += 1
        set_policy_gripper_authorization(
            task_environment,
            response.get("gripper_authorization"),
        )
        primary_action_applied = False
        try:
            observation, reward, terminate = task_environment.step(
                np.asarray(action, dtype=np.float64)
            )
            primary_action_applied = True
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
                    }
                )
            joint_hold_commits += 1
            committed_policy_steps += 1
        except Exception:
            worker.request("abort", transaction_id=transaction_id)
            raise
        if primary_action_applied:
            commit = worker.request(
                "commit",
                transaction_id=transaction_id,
                primary_action_status=policy_action_execution_status(task_environment),
                primary_action_statuses=policy_action_execution_statuses(
                    task_environment
                ),
            )
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
        }
    )


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
    retry_exhaustion_mode=FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE,
    motion_plan=None,
    final_settling_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    descriptions=None,
    observation=None,
    fresh_task_generation=None,
    episode_variation=None,
    post_success_policy_steps=0,
):
    retry_exhaustion_mode = validate_primary_retry_exhaustion_mode(
        retry_exhaustion_mode
    )
    if retry_exhaustion_mode != FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE:
        raise ValueError("formal evaluation requires the global joint-hold clock")
    if max_primary_action_attempts != 1:
        raise ValueError("formal evaluation requests each policy tick exactly once")
    if (
        isinstance(post_success_policy_steps, bool)
        or not isinstance(post_success_policy_steps, int)
        or post_success_policy_steps < 0
    ):
        raise ValueError(
            "post-success policy continuation must be a non-negative integer"
        )
    if isinstance(motion_plan, StoreBottleMultiEntityPlan):
        if post_success_policy_steps:
            raise ValueError(
                "post-success policy continuation is unavailable for StoreBottle"
            )
        return _run_v4_store_episode(
            task_environment,
            worker,
            episode,
            horizon,
            scenario=scenario,
            max_primary_action_attempts=max_primary_action_attempts,
            retry_exhaustion_mode=retry_exhaustion_mode,
            motion_plan=motion_plan,
            final_settling_steps=final_settling_steps,
            descriptions=descriptions,
            observation=observation,
            fresh_task_generation=fresh_task_generation,
        )
    if (
        descriptions is None
        or observation is None
        or not isinstance(
            fresh_task_generation,
            dict,
        )
    ):
        raise RuntimeError("formal episode requires fresh task-generation input")
    invalid_actions = 0
    joint_hold_commits = 0
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
    success_latched_policy_step = None
    post_success_policy_complete = False
    last_scenario_policy_step = None
    resolved_variation = (
        episode % task_environment.variation_count()
        if episode_variation is None
        else int(episode_variation)
    )
    source_binding = controller.bind_staged_source(
        task_environment,
        episode_seed=seed + episode,
        variation=resolved_variation,
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
        executed_after_success = (
            0
            if success_latched_policy_step is None
            else max(0, committed_policy_steps - success_latched_policy_step)
        )
        row.setdefault(
            "post_success_policy_continuation",
            {
                "diagnostic_only": bool(post_success_policy_steps),
                "requested_steps": int(post_success_policy_steps),
                "success_latched_policy_step": success_latched_policy_step,
                "executed_steps": executed_after_success,
                "policy_complete": bool(post_success_policy_complete),
            },
        )
        row.setdefault("staged_source_binding", source_binding)
        row.setdefault("fresh_task_generation", fresh_task_generation)
        row.setdefault(
            "motion_plan_fingerprint",
            motion_plan.fingerprint() if motion_plan is not None else None,
        )
        row.setdefault(
            "motion_plan_protocol_id",
            (
                (
                    V4_LIFT_MOTION_PROTOCOL_ID
                    if motion_plan.validation.get("v4_lift_tray") is not None
                    else motion_plan.metadata()["protocol_id"]
                )
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
                    **(
                        {"v4_lift_tray": motion_plan.validation["v4_lift_tray"]}
                        if motion_plan.validation.get("v4_lift_tray") is not None
                        else {}
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

    control_attempts = 0
    while committed_policy_steps < horizon:
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
                raise RuntimeError(
                    "scenario controller returned an inconsistent trigger"
                )
            if event.get("applied") and (
                trigger_step is None or committed_policy_steps < trigger_step
            ):
                raise RuntimeError("scenario controller applied before the trigger")
            if event["applied"]:
                scenario_events.append(event)
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
                primary_action_status="stopped",
                primary_action_applied=False,
            )
            committed_policy_steps += 1
            return finish(
                {
                    "episode": episode,
                    "success": success_latched_policy_step is not None,
                    "steps": control_attempts,
                    "control_attempts": control_attempts,
                    "reason": (
                        "success_latched_then_policy_structured_failure"
                        if success_latched_policy_step is not None
                        else "policy_structured_failure"
                    ),
                    "policy_failure_reasons": response.get("failure_reasons", {}),
                    "invalid_actions": invalid_actions,
                    "scenario_events": scenario_events,
                }
            )
        if action is None:
            if success_latched_policy_step is not None:
                post_success_policy_complete = True
                return finish(
                    {
                        "episode": episode,
                        "success": True,
                        "steps": control_attempts,
                        "control_attempts": control_attempts,
                        "reason": "success_then_policy_complete",
                        "invalid_actions": invalid_actions,
                        "scenario_events": scenario_events,
                    }
                )
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
                    "scenario_events": scenario_events,
                    "final_settling": settling,
                }
            )
        transaction_id = response.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy worker did not return an action transaction id")
        control_attempts += 1
        set_policy_gripper_authorization(
            task_environment,
            response.get("gripper_authorization"),
        )
        primary_action_applied = False
        try:
            observation, reward, terminate = task_environment.step(
                np.asarray(action, dtype=np.float64)
            )
            primary_action_applied = True
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
                        "scenario_events": scenario_events,
                    }
                )
            joint_hold_commits += 1
            committed_policy_steps += 1
        except Exception:
            worker.request("abort", transaction_id=transaction_id)
            raise
        if primary_action_applied:
            commit = worker.request(
                "commit",
                transaction_id=transaction_id,
                primary_action_status=policy_action_execution_status(task_environment),
                primary_action_statuses=policy_action_execution_statuses(
                    task_environment
                ),
            )
            policy_complete = bool(commit.get("complete"))
            committed_policy_steps += 1
        settling = None
        if reward > 0.0 and success_latched_policy_step is None:
            success_latched_policy_step = committed_policy_steps
        if success_latched_policy_step is not None:
            post_success_policy_complete = bool(policy_complete)
            continued_steps = committed_policy_steps - success_latched_policy_step
            if policy_complete:
                reason = "success_then_policy_complete"
            elif continued_steps >= post_success_policy_steps:
                reason = (
                    "success"
                    if post_success_policy_steps == 0
                    else "success_after_post_success_policy_continuation"
                )
            else:
                continue
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
                "success": bool(
                    success_latched_policy_step is not None
                    or (settling or {}).get("success")
                ),
                "steps": control_attempts,
                "control_attempts": control_attempts,
                "reason": reason,
                "invalid_actions": invalid_actions,
                "scenario_events": scenario_events,
                **({"final_settling": settling} if settling is not None else {}),
            }
        )
    return finish(
        {
            "episode": episode,
            "success": success_latched_policy_step is not None,
            "steps": control_attempts,
            "control_attempts": control_attempts,
            "reason": (
                "success_latched_post_success_horizon"
                if success_latched_policy_step is not None
                else "horizon"
            ),
            "invalid_actions": invalid_actions,
            "scenario_events": scenario_events,
        }
    )


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
            if _is_v4_store(args):
                plan = stage_v4_store_motion_plan(
                    environment,
                    task_class,
                    episode_index=episode,
                    episode_seed=args.seed + episode,
                    variation=variation,
                    max_attempts=args.scenario_max_attempts,
                )
            elif _is_v4_lift(args):
                plan = stage_v4_lift_motion_plan(
                    environment,
                    task_class,
                    episode_seed=args.seed + episode,
                    variation=variation,
                    max_attempts=args.scenario_max_attempts,
                )
            else:
                plan = stage_scenario_motion_plan(
                    environment,
                    task_class,
                    task_name=args.task,
                    episode_seed=args.seed + episode,
                    variation=variation,
                    max_attempts=args.scenario_max_attempts,
                )
            plans.append(plan)
            variations.append(variation)
            print(
                f"staged {args.task} A/B {episode + 1}/{args.episodes}",
                flush=True,
            )
    finally:
        if launched:
            environment.shutdown()
    if _is_v4_store(args):
        payload = build_v4_store_task_scoped_plan_batch(
            base_seed=args.seed,
            variations=variations,
            plans=plans,
        )
    elif _is_v4_lift(args):
        payload = build_v4_lift_task_scoped_plan_batch(
            base_seed=args.seed,
            variations=variations,
            plans=plans,
        )
    else:
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
    return (
        DEFAULT_RESULTS_DIR
        / "motion_plans"
        / (f"{args.task}_seed{args.seed}_n{args.episodes}_v34.json")
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
    if _is_v4_store(args):
        runtime_loaders = {V4_STORE_RUNTIME_LOADER_ID: load_v4_store_motion_plan_batch}
    elif _is_v4_lift(args):
        runtime_loaders = {V4_LIFT_RUNTIME_LOADER_ID: load_v4_lift_motion_plan_batch}
    else:
        runtime_loaders = None
    if runtime_loaders is None:
        manifest, selected = fixed_environment_plans(args.eval_set_id, args.task)
    else:
        manifest, selected = fixed_environment_plans(
            args.eval_set_id,
            args.task,
            runtime_loaders=runtime_loaders,
        )
    plans = selected["plans"]
    if any(
        plan.validation.get("goal_sampling_max_attempts") != args.scenario_max_attempts
        for plan in plans
    ):
        raise RuntimeError("fixed eval-set goal-sampling budget is inconsistent")
    if _is_v4_store(args):
        payload = selected.get("payload")
        if (
            not isinstance(payload, dict)
            or payload.get("runtime_loader") != V4_STORE_RUNTIME_LOADER_ID
            or payload.get("task_identity", {}).get("components")
            != v4_store_task_identity_components()
        ):
            raise RuntimeError("fixed StoreBottle V4 task-scoped identity is invalid")
    elif _is_v4_lift(args):
        payload = selected.get("payload")
        if (
            not isinstance(payload, dict)
            or payload.get("runtime_loader") != V4_LIFT_RUNTIME_LOADER_ID
            or payload.get("task_identity", {}).get("components")
            != v4_lift_task_identity_components()
        ):
            raise RuntimeError("fixed V4 LiftTray task-scoped identity is invalid")
    return manifest, selected


def evaluate(args):
    validate_formal_artifact_paths(output=args.output, models_dir=args.models_dir)
    with reserve_output(args.output):
        return _evaluate_reserved(args)


def _evaluate_reserved(args):
    _validate_formal_execution_args(args)
    video_capture_enabled = _v4_video_capture_enabled(args)
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    # Validate the release-scoped protocol before loading a formal plan batch.
    if _is_v4_store(args):
        _validate_v4_store_protocol_args(args)
    elif _is_v4_lift(args):
        _validate_v4_lift_protocol_args(args)
    else:
        _validate_v3_protocol_budgets(args)
    if _is_v4_store(args):
        task_class = v4_store_task_class()
    else:
        module_name, class_name = TASKS[args.task]
        task_class = getattr(importlib.import_module(module_name), class_name)
    eval_set, selected_batch = _load_fixed_motion_plans(args)
    selected_motion_plan_payload = selected_batch["payload"]
    motion_plan_payload = (
        selected_motion_plan_payload["runtime_batch"]
        if _is_v4_task_scoped(args)
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
        policy_type=getattr(args, "policy_type", "dynamac"),
        closed_loop_models_dir=getattr(
            args, "closed_loop_models_dir", CLOSED_LOOP_MODELS_DIR
        ),
        diagnostics_dir=getattr(args, "policy_diagnostics_dir", None),
        closed_loop_feature_profile=getattr(
            args, "closed_loop_feature_profile", "full"
        ),
    )
    results = []
    launched = False
    try:
        if _is_v4_store(args):
            intervention_registry, trigger_authentication = (
                _authenticated_v4_store_triggers(args, worker)
            )
        elif _is_v4_lift(args):
            intervention_registry, trigger_authentication = (
                _authenticated_v4_lift_trigger(args, worker)
            )
        else:
            intervention_registry, trigger_authentication = (
                _authenticated_v3_dynamic_trigger(args, worker)
            )
        scenario_reference_steps = worker.policy_steps
        scenario_reference_source = (
            (
                (
                    "V4_STORE_INDEPENDENT_SKILL0_ENTITY_TICKS_WITH_LOADED_POLICY_HORIZON"
                    if _is_v4_store(args)
                    else "V4_LIFT_FIXED_SKILL0_GLOBAL_TICK_WITH_LOADED_POLICY_HORIZON"
                )
                if _is_v4_task_scoped(args)
                else "AUTHENTICATED_LOADED_CHECKPOINT_POLICY_STEPS"
            )
            if args.scenario_reference_steps is None
            else (
                "COMMAND_LINE_MATCHED_V4_LOADED_POLICY_HORIZON"
                if _is_v4_task_scoped(args)
                else "COMMAND_LINE_MATCHED_AUTHENTICATED_LOADED_CHECKPOINT_POLICY_STEPS"
            )
        )
        if args.scenario == "static":
            resolved_trigger_step = None
        elif _is_v4_store(args):
            resolved_trigger_step = {
                name: value["trigger_step"]
                for name, value in trigger_authentication["triggers"].items()
            }
        else:
            resolved_trigger_step = trigger_authentication["trigger_step"]
        environment.launch()
        launched = True
        variation_count = task_class(
            environment._pyrep,
            environment._robot,
        ).variation_count()
        episode_variation_offset = int(getattr(args, "episode_variation_offset", 0))
        for episode in range(args.episodes):
            variation = (episode_variation_offset + episode) % variation_count
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
                variation=variation,
                verify_instance=False,
            )

            def run_formal_episode(formal_task_environment):
                return _run_episode(
                    formal_task_environment,
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
                    episode_variation=variation,
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
    if _is_v4_store(args):
        motion_protocol = StoreBottleMultiEntityController(
            plan=motion_plans[0],
            scenario=args.scenario,
        ).protocol_metadata()
    else:
        motion_protocol = ScenarioController(
            SCENARIO_KINDS[args.scenario],
            trigger_fraction=args.scenario_trigger_fraction,
            trigger_step=resolved_trigger_step,
            total_steps=args.scenario_steps,
            max_attempts=args.scenario_max_attempts,
            motion_plan=(motion_plans[0] if motion_plans else None),
        ).protocol_metadata()
    store_mode_subgroups = None
    if _is_v4_store(args):
        store_mode_subgroups = {}
        for mode in ("bottle_only", "fridge_only", "both"):
            members = [item for item in results if item.get("store_mode") == mode]
            successes = sum(int(item["success"]) for item in members)
            store_mode_subgroups[mode] = {
                "planned": sum(plan.mode == mode for plan in motion_plans),
                "completed": len(members),
                "successes": successes,
                "success_rate": (successes / float(len(members)) if members else None),
                "intervention_complete": sum(
                    item.get("intervention_complete") is True for item in members
                ),
                "dynamic_condition_unexercised": sum(
                    item.get("dynamic_condition_unexercised") is True
                    for item in members
                ),
            }
    summary = {
        **({"release": "v4"} if getattr(args, "release", "v3") == "v4" else {}),
        "policy_type": getattr(args, "policy_type", "dynamac"),
        "closed_loop_feature_profile": (
            getattr(args, "closed_loop_feature_profile", "full")
            if getattr(args, "policy_type", "dynamac") == "closed_loop_multistream"
            else None
        ),
        "task": args.task,
        "scenario": args.scenario,
        "scenario_protocol": {
            "status": (
                "STATIC_REFERENCE"
                if args.scenario == "static"
                else (
                    (
                        "V4_STORE_INDEPENDENT_ENTITY_TRIGGERS_TASK_SCOPED"
                        if _is_v4_store(args)
                        else "V4_LIFT_SKILL0_TICK35_TASK_SCOPED"
                    )
                    if _is_v4_task_scoped(args)
                    else "V3_PREREGISTERED_CHECKPOINT_AUTHENTICATED"
                )
            ),
            "motion_kind": SCENARIO_KINDS[args.scenario],
            "motion_protocol": motion_protocol,
            "legacy_trigger_fraction_ignored": args.scenario_trigger_fraction,
            "trigger_reference_steps": scenario_reference_steps,
            "trigger_reference_source": scenario_reference_source,
            "trigger_reference_domain": ("successfully_committed_policy_ticks"),
            "trigger_policy_step": resolved_trigger_step,
            "trigger_authentication": trigger_authentication,
            "intervention_registry_schema": intervention_registry["schema"],
            "intervention_registry_fingerprint": intervention_registry["fingerprint"],
            "smooth_interpolation_calls": (
                args.scenario_steps if args.scenario == "smooth" else None
            ),
            "max_sampling_attempts": args.scenario_max_attempts,
            "observation_refreshed_before_policy_action": True,
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
                    "motion_source_protocol_schema": (
                        (
                            motion_plans[0].validation["motion_source_schema"]
                            if _is_v4_store(args)
                            else motion_plans[0].validation["v4_lift_tray"][
                                "motion_source_schema"
                            ]
                        )
                        if _is_v4_task_scoped(args)
                        else motion_plans[0].validation["motion_source_protocol_schema"]
                    ),
                    "motion_source_protocol_fingerprint": (
                        (
                            motion_plans[0].validation["motion_source_fingerprint"]
                            if _is_v4_store(args)
                            else motion_plans[0].validation["v4_lift_tray"][
                                "motion_source_fingerprint"
                            ]
                        )
                        if _is_v4_task_scoped(args)
                        else motion_plans[0].validation[
                            "motion_source_protocol_fingerprint"
                        ]
                    ),
                    **(
                        {
                            "runtime_protocol_id": (
                                V4_STORE_MOTION_PROTOCOL_ID
                                if _is_v4_store(args)
                                else V4_LIFT_MOTION_PROTOCOL_ID
                            ),
                            "task_scoped_envelope": {
                                "schema": selected_motion_plan_payload["schema"],
                                "protocol_id": selected_motion_plan_payload[
                                    "protocol_id"
                                ],
                                "runtime_loader": selected_motion_plan_payload[
                                    "runtime_loader"
                                ],
                                "task_identity_fingerprint": selected_motion_plan_payload[
                                    "task_identity"
                                ][
                                    "fingerprint"
                                ],
                                "batch_fingerprint": selected_motion_plan_payload[
                                    "batch_fingerprint"
                                ],
                            },
                        }
                        if _is_v4_task_scoped(args)
                        else {}
                    ),
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
        },
        "episodes": args.episodes,
        "episodes_requested": args.episodes,
        "episodes_completed": len(results),
        "seed": args.seed,
        "horizon": args.horizon,
        "variation_count": variation_count,
        "variation_schedule": [
            (int(getattr(args, "episode_variation_offset", 0)) + episode)
            % variation_count
            for episode in range(args.episodes)
        ],
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
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        **(
            {"store_mode_subgroups": store_mode_subgroups}
            if store_mode_subgroups is not None
            else {}
        ),
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
        "final_settling_protocol": final_settling_metadata(args.final_settling_steps),
        "motion_plan_batch_fingerprint": (
            selected_motion_plan_payload["batch_fingerprint"]
            if selected_motion_plan_payload is not None
            else None
        ),
        "results": results,
        "fresh_task_generation": {
            "required_per_formal_episode": True,
            "all_episodes_recorded": all(
                isinstance(item.get("fresh_task_generation"), dict) for item in results
            ),
            "evidence": [item["fresh_task_generation"] for item in results],
        },
    }
    if video_capture_enabled:
        video_capture_metadata = {
            "release": "v4",
            "cell_key": video_cell_key,
            "cell_dir": str(video_cell_dir),
            "episodes_recorded": len(episode_videos),
            "capture_config": dict(video_capture_config.audit()),
            "paper_success_rate": V4_VIDEO_PAPER_TARGETS.get(
                (args.task, args.scenario)
            ),
        }

        def finalize_videos():
            return _finalize_v4_video_cell(
                video_cell_dir,
                episode_videos,
                cell_key=video_cell_key,
                successes=sum(int(item["success"]) for item in results),
                episodes=args.episodes,
                paper_success_rate=V4_VIDEO_PAPER_TARGETS.get(
                    (args.task, args.scenario)
                ),
                cell_metadata={
                    "evaluator": "direct_evaluate",
                    "task": args.task,
                    "scenario": args.scenario,
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
    parser.add_argument(
        "--closed-loop-feature-profile",
        choices=("progress_only", "progress_dynamic_roles", "full"),
        default="full",
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
    parser.add_argument(
        "--episode-variation-offset",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument(
        "--release",
        choices=("v3", "v4"),
        default="v3",
        help="Evaluation release gate; formal V4 videos are replayed afterward.",
    )
    parser.add_argument(
        "--record-v4-evaluation-videos",
        action="store_true",
        help=(
            "Legacy diagnostic option to stream RGB during a V4 evaluation. "
            "The formal launcher leaves it disabled and generates replays afterward."
        ),
    )
    parser.add_argument(
        "--v4-evaluation-video-root",
        type=Path,
        default=DEFAULT_V4_EVALUATION_VIDEO_ROOT,
        help="Root for V4 formal evaluation video cells.",
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
    if _is_v4_store(args) and Path(args.models_dir) == DEFAULT_MODELS_DIR:
        args.models_dir = V4_MODELS_DIR
    if args.stage_motion_plans_output is not None:
        if args.episode_variation_offset != 0:
            raise ValueError(
                "episode variation offset is only valid for read-only evaluation"
            )
        if _is_v4_store(args):
            _validate_v4_store_protocol_args(args)
        elif _is_v4_lift(args):
            _validate_v4_lift_protocol_args(args)
        else:
            _validate_v3_protocol_budgets(args)
        if args.episodes < 1 or args.seed < 0 or args.scenario_max_attempts < 1:
            raise ValueError("staging episodes/attempts must be positive")
        if _is_v4_store(args):
            task_class = v4_store_task_class()
        else:
            module_name, class_name = TASKS[args.task]
            task_class = getattr(importlib.import_module(module_name), class_name)
        with reserve_output(args.stage_motion_plans_output):
            _stage_motion_plan_batch(args, task_class)
        return 0
    if args.episodes < 1 or args.horizon < 1:
        raise ValueError("episodes and horizon must be positive")
    if args.episode_variation_offset < 0:
        raise ValueError("episode variation offset must be non-negative")
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
        results_root = V4_RESULTS_DIR if _is_v4_store(args) else DEFAULT_RESULTS_DIR
        args.output = (
            results_root
            / family
            / f"{args.task}_{args.scenario}_seed{args.seed}_n{args.episodes}_h{args.horizon}.json"
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
