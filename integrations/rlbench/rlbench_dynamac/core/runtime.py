"""Pure conversion and intervention helpers for the optional RLBench runtime.

No RLBench/PyRep import happens at module import time.  This lets conversion
and action-layout tests run on machines without CoppeliaSim.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

import numpy as np

from integrations.rlbench.rlbench_dynamac.core.pytracik_dependency import (
    pytracik_dependency_identity,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    TaskSpec,
    get_task_spec,
    unwrap_task_low_dim_state,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    wxyz_to_xyzw as _wxyz_to_xyzw,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    xyzw_to_wxyz as _xyzw_to_wxyz,
)

Array = np.ndarray

PRESERVE_INSTANCE_MOTION_PROTOCOL_ID = (
    "rlbench-boundary-root-preserve-initialized-episode-v4"
)
STAGED_VALIDATED_MOTION_PROTOCOL_ID = (
    "rlbench-deterministic-source-staging-waypoint-validated-boundary-root-v3.4"
)
STAGED_MOTION_PLAN_SCHEMA = "dynamac-rlbench-staged-motion-plan-v3.4"
STAGED_MOTION_PLAN_BATCH_SCHEMA = "dynamac-rlbench-staged-motion-plan-batch-v3.4"
STAGED_MOTION_PLAN_VALIDATION_SCHEMA = (
    "dynamac-rlbench-staged-motion-plan-validation-v3.4"
)
STAGED_SOURCE_PLAN_SCHEMA = "dynamac-rlbench-staged-source-plan-v1"
STAGED_SOURCE_PLAN_BATCH_SCHEMA = "dynamac-rlbench-staged-source-plan-batch-v1"
STAGED_SOURCE_VALIDATION_SCHEMA = "dynamac-rlbench-staged-source-validation-v1"
STAGED_SOURCE_PROTOCOL_ID = "rlbench-deterministic-source-a-only-v1"
FRESH_TASK_GENERATION_PROTOCOL_ID = (
    "rlbench-prestop-unload-if-present-stop-reload-start-seed-variation-"
    "single-reset-v2"
)
FRESH_TASK_GENERATION_EVIDENCE_SCHEMA = (
    "dynamac-rlbench-fresh-task-generation-evidence-v2"
)
DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID = (
    "rlbench-prestop-unload-if-present-stop-reload-start-source-seed-variation-"
    "single-reset-verify-false-v1"
)
DETERMINISTIC_SOURCE_RESET_EVIDENCE_SCHEMA = (
    "dynamac-rlbench-deterministic-source-reset-evidence-v1"
)
SOURCE_SEED_SELECTION_SCHEMA = "dynamac-rlbench-source-seed-selection-v1"
SOURCE_CERTIFICATION_SCHEMA = "dynamac-rlbench-offline-source-certification-v1"
GOAL_CERTIFICATION_SCHEMA = "dynamac-rlbench-offline-goal-certification-v1"
SOURCE_RECONSTRUCTION_SCHEMA = "dynamac-rlbench-source-reconstruction-audit-v1"
DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS = 20
SOURCE_RECONSTRUCTION_TRANSLATION_TOLERANCE_M = 1.0e-6
SOURCE_RECONSTRUCTION_ROTATION_TOLERANCE_RAD = 1.0e-6
SOURCE_RECONSTRUCTION_SCALAR_TOLERANCE = 1.0e-6
SOURCE_RECONSTRUCTION_JOINT_TOLERANCE = 1.0e-6
TASK_SEMANTIC_SIGNATURE_SCHEMA = "dynamac-rlbench-task-semantic-signature-v2"
TASK_TREE_STATE_SCHEMA = "rlbench-task-tree-dual-frame-state-v1"
CROSS_INITIALIZATION_REPRODUCIBILITY_SCHEMA = (
    "rlbench-cross-initialization-reproducibility-audit-v1"
)
FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA = (
    "rlbench-formal-boundary-root-same-instance-state-audit-v2"
)
FORMAL_INTERVENTION_COLLISION_PAIR_POLICY = (
    "record_exact_post_command_pair_delta_diagnostic_only"
)
CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M = 2.0e-5
CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD = 2.0e-4
CROSS_INITIALIZATION_SCALAR_TOLERANCE = 1.0e-4
CROSS_INITIALIZATION_JOINT_TOLERANCE = 1.0e-4
QUATERNION_ROTATION_METRIC = "sign_invariant_normalized_chord_4asin_half_chord"
FINAL_SETTLING_PROTOCOL_ID = "rlbench-hold-final-command-up-to-10-raw-physics-steps-v3"
DEFAULT_FINAL_SETTLING_PHYSICS_STEPS = 10
LOW_DIM_STATE_ROUNDTRIP_ATOL = 1.0e-6
LOW_DIM_POSE_TRANSLATION_TOLERANCE_M = 1.0e-6
LOW_DIM_POSE_ROTATION_TOLERANCE_RAD = 1.0e-6
LOW_DIM_POSE_QUATERNION_NORM_ATOL = 1.0e-3
ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M = 1.0e-9
ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD = 1.0e-9
ROOT_COMMAND_TRANSLATION_TOLERANCE_M = 1.0e-6
ROOT_COMMAND_ROTATION_TOLERANCE_RAD = 1.0e-6


def initialize_fresh_task_generation(
    environment: Any,
    task_class: Any,
    *,
    episode_seed: int,
    variation: int,
    verify_instance: bool = True,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load and reset one task generation with no prior physics history.

    The order is deliberately fixed.  If a preceding task exists, unload it
    while physics is still running so task-owned runtime objects remain valid
    during ``cleanup``.  Then stop physics, load a new task environment (whose
    constructor starts physics), seed both RNGs immediately before
    initialization, set the variation, and call ``reset`` exactly once.
    Keeping the lifecycle in one helper avoids accidental extra resets in
    staging candidates and formal episodes.
    """

    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise TypeError("fresh task-generation seed must be an integer")
    if episode_seed < 0:
        raise ValueError("fresh task-generation seed must be non-negative")
    if isinstance(variation, bool) or not isinstance(variation, int):
        raise TypeError("fresh task-generation variation must be an integer")
    if variation < 0:
        raise ValueError("fresh task-generation variation must be non-negative")
    pyrep = getattr(environment, "_pyrep", None)
    stop = getattr(pyrep, "stop", None)
    get_task = getattr(environment, "get_task", None)
    if pyrep is None or not callable(stop) or not callable(get_task):
        raise RuntimeError("RLBench fresh task-generation lifecycle API is unavailable")

    running_before_stop = getattr(pyrep, "running", None)
    if not isinstance(running_before_stop, bool):
        raise RuntimeError("PyRep running-state audit is unavailable")
    scene = getattr(environment, "_scene", None)
    unload = getattr(scene, "unload", None)
    if scene is None or not callable(unload):
        raise RuntimeError("RLBench scene unload lifecycle API is unavailable")
    previous_task = getattr(scene, "task", None)
    previous_task_present = previous_task is not None
    if running_before_stop is not previous_task_present:
        raise RuntimeError(
            "fresh task-generation physics/task lifecycle state is inconsistent"
        )
    if previous_task_present:
        if running_before_stop is not True:
            raise RuntimeError(
                "preceding task must be unloaded before physics is stopped"
            )
        unload()
        if getattr(scene, "task", None) is not None:
            raise RuntimeError("preceding task did not unload before physics stop")
        if getattr(pyrep, "running", None) is not True:
            raise RuntimeError("preceding task unload unexpectedly stopped physics")
    stop()
    if getattr(pyrep, "running", None) is not False:
        raise RuntimeError("physics did not stop before fresh task reload")

    task_environment = get_task(task_class)
    if getattr(pyrep, "running", None) is not True:
        raise RuntimeError("fresh TaskEnvironment did not restart physics")
    set_variation = getattr(task_environment, "set_variation", None)
    reset = getattr(task_environment, "reset", None)
    if not callable(set_variation) or not callable(reset):
        raise RuntimeError("RLBench fresh TaskEnvironment reset API is unavailable")

    # Reload may consume RNG state in third-party task constructors.  Install
    # the episode seed immediately before the only initialization reset.
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    set_variation(variation)
    if not isinstance(verify_instance, bool):
        raise TypeError("fresh task-generation verification flag must be boolean")
    reset_collision_checks: list[bool] = []
    reset_scene = getattr(task_environment, "_scene", None)
    reset_robot = getattr(reset_scene, "robot", None)
    original_collision_check = getattr(reset_robot, "is_in_collision", None)
    robot_dict = getattr(reset_robot, "__dict__", {})
    had_instance_collision_check = "is_in_collision" in robot_dict
    prior_instance_collision_check = robot_dict.get("is_in_collision")
    if not verify_instance:
        if not callable(original_collision_check):
            raise RuntimeError("RLBench reset robot collision check is unavailable")

        def audited_collision_check(*args: Any, **kwargs: Any) -> bool:
            result = bool(original_collision_check(*args, **kwargs))
            reset_collision_checks.append(result)
            return result

        setattr(reset_robot, "is_in_collision", audited_collision_check)
    try:
        reset_result = reset(verify_instance=verify_instance)
    finally:
        if not verify_instance:
            if had_instance_collision_check:
                setattr(reset_robot, "is_in_collision", prior_instance_collision_check)
            else:
                delattr(reset_robot, "is_in_collision")
    if not isinstance(reset_result, tuple) or len(reset_result) < 2:
        raise RuntimeError("fresh TaskEnvironment.reset returned an invalid result")
    descriptions, observation = reset_result[0], reset_result[1]

    scene = getattr(task_environment, "_scene", None)
    task = getattr(scene, "task", None)
    if task is None or task is previous_task:
        raise RuntimeError("fresh task reload did not create a new task instance")
    get_name = getattr(task, "get_name", None)
    task_name = str(get_name()) if callable(get_name) else type(task).__name__
    generation_index = getattr(environment, "_dynamac_generation_index", 0) + 1
    setattr(environment, "_dynamac_generation_index", generation_index)
    body = {
        "schema": (
            FRESH_TASK_GENERATION_EVIDENCE_SCHEMA
            if verify_instance
            else DETERMINISTIC_SOURCE_RESET_EVIDENCE_SCHEMA
        ),
        "protocol_id": (
            FRESH_TASK_GENERATION_PROTOCOL_ID
            if verify_instance
            else DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
        ),
        "generation_index": int(generation_index),
        "episode_seed": int(episode_seed),
        "variation": int(variation),
        "task_name": task_name,
        "physics_running_before_stop": running_before_stop,
        "physics_stopped_before_task_reload": True,
        "previous_task_present": previous_task_present,
        "previous_task_unloaded_before_stop": previous_task_present,
        "previous_task_unloaded_while_physics_running": previous_task_present,
        "scene_task_absent_before_stop": True,
        "task_model_loaded_fresh": True,
        "fresh_task_python_instance_created": True,
        "task_model_only_reloaded": True,
        "base_scene_reloaded": False,
        "physics_started_by_task_environment": True,
        "rng_seeded_after_reload_immediately_before_reset": True,
        "variation_set_after_seed_before_reset": True,
        "task_environment_reset_calls": 1,
        "reset_verify_instance": verify_instance,
    }
    if not verify_instance:
        static_positions = bool(getattr(task_environment, "_static_positions", False))
        is_static_workspace = bool(task.is_static_workspace())
        body.update(
            {
                "reset_random_placement_expected": bool(
                    not static_positions and not is_static_workspace
                ),
                "reset_robot_collision_check_count": len(reset_collision_checks),
                "reset_robot_collision_check_results": reset_collision_checks,
            }
        )
    evidence = {
        **body,
        "fingerprint": _canonical_json_fingerprint(body),
    }
    setattr(task_environment, "_dynamac_fresh_generation_evidence", evidence)
    return task_environment, descriptions, observation, evidence


def _canonical_json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_fresh_task_generation_evidence(
    evidence: Any,
    *,
    episode_seed: int,
    variation: int,
    task_name: str,
    verify_instance: bool = True,
) -> dict[str, Any]:
    """Authenticate one lifecycle record and return a defensive copy."""

    expected_fields = {
        "schema",
        "protocol_id",
        "generation_index",
        "episode_seed",
        "variation",
        "task_name",
        "physics_running_before_stop",
        "physics_stopped_before_task_reload",
        "previous_task_present",
        "previous_task_unloaded_before_stop",
        "previous_task_unloaded_while_physics_running",
        "scene_task_absent_before_stop",
        "task_model_loaded_fresh",
        "fresh_task_python_instance_created",
        "task_model_only_reloaded",
        "base_scene_reloaded",
        "physics_started_by_task_environment",
        "rng_seeded_after_reload_immediately_before_reset",
        "variation_set_after_seed_before_reset",
        "task_environment_reset_calls",
        "reset_verify_instance",
        "fingerprint",
    }
    if not verify_instance:
        expected_fields.update(
            {
                "reset_robot_collision_check_count",
                "reset_robot_collision_check_results",
                "reset_random_placement_expected",
            }
        )
    if not isinstance(evidence, dict) or set(evidence) != expected_fields:
        raise RuntimeError("fresh task-generation evidence fields are invalid")
    body = {key: value for key, value in evidence.items() if key != "fingerprint"}
    if (
        evidence.get("schema")
        != (
            FRESH_TASK_GENERATION_EVIDENCE_SCHEMA
            if verify_instance
            else DETERMINISTIC_SOURCE_RESET_EVIDENCE_SCHEMA
        )
        or evidence.get("protocol_id")
        != (
            FRESH_TASK_GENERATION_PROTOCOL_ID
            if verify_instance
            else DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
        )
        or isinstance(evidence.get("generation_index"), bool)
        or not isinstance(evidence.get("generation_index"), int)
        or evidence["generation_index"] < 1
        or isinstance(evidence.get("episode_seed"), bool)
        or not isinstance(evidence.get("episode_seed"), int)
        or isinstance(evidence.get("variation"), bool)
        or not isinstance(evidence.get("variation"), int)
        or evidence.get("episode_seed") != episode_seed
        or evidence.get("variation") != variation
        or evidence["episode_seed"] < 0
        or evidence["variation"] < 0
        or evidence.get("task_name") != task_name
        or not isinstance(evidence.get("physics_running_before_stop"), bool)
        or evidence.get("physics_running_before_stop")
        is not evidence.get("previous_task_present")
        or evidence.get("physics_stopped_before_task_reload") is not True
        or not isinstance(evidence.get("previous_task_present"), bool)
        or evidence.get("previous_task_unloaded_before_stop")
        is not evidence.get("previous_task_present")
        or evidence.get("previous_task_unloaded_while_physics_running")
        is not evidence.get("previous_task_present")
        or evidence.get("scene_task_absent_before_stop") is not True
        or evidence.get("task_model_loaded_fresh") is not True
        or evidence.get("fresh_task_python_instance_created") is not True
        or evidence.get("task_model_only_reloaded") is not True
        or evidence.get("base_scene_reloaded") is not False
        or evidence.get("physics_started_by_task_environment") is not True
        or evidence.get("rng_seeded_after_reload_immediately_before_reset") is not True
        or evidence.get("variation_set_after_seed_before_reset") is not True
        or isinstance(evidence.get("task_environment_reset_calls"), bool)
        or not isinstance(evidence.get("task_environment_reset_calls"), int)
        or evidence.get("task_environment_reset_calls") != 1
        or evidence.get("reset_verify_instance") is not verify_instance
        or (
            not verify_instance
            and (
                isinstance(evidence.get("reset_robot_collision_check_count"), bool)
                or not isinstance(
                    evidence.get("reset_robot_collision_check_count"), int
                )
                or evidence.get("reset_robot_collision_check_count") < 0
                or not isinstance(evidence.get("reset_random_placement_expected"), bool)
                or not isinstance(
                    evidence.get("reset_robot_collision_check_results"), list
                )
                or any(
                    not isinstance(value, bool)
                    for value in evidence["reset_robot_collision_check_results"]
                )
                or evidence["reset_robot_collision_check_count"]
                != len(evidence["reset_robot_collision_check_results"])
                or evidence["reset_robot_collision_check_results"]
                != ([False] if evidence["reset_random_placement_expected"] else [])
            )
        )
        or evidence.get("fingerprint") != _canonical_json_fingerprint(body)
    ):
        raise RuntimeError("fresh task-generation evidence is invalid")
    return dict(evidence)


# RLBench's waypoint demonstration generator actuates every gripper at 0.04
# (see ``Scene._handle_extensions_strings`` in the pinned fork).  Evaluation
# must use the same physical command speed: the upstream Discrete modes use
# 0.2, which changes contact dynamics even when the policy trajectory is
# identical to the demonstration.
DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY = 0.04
DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS = 1
FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE = "commit_joint_hold_noop_formal"
FORMAL_POLICY_CLOCK_SEMANTICS_ID = (
    "policy-tick-single-primary-request-then-raw-joint-hold-commit-v1"
)
# Explicit legacy-only mode remains available for replaying already-published
# V4 artifacts. New formal evaluation never selects it.
FROZEN_PRIMARY_RETRY_EXHAUSTION_MODE = "terminate_episode_legacy_v4"
IK_SAMPLING_TRIALS = 100
IK_SAMPLING_MAX_CONFIGS = 5
IK_SAMPLING_MAX_TIME_MS = 10
IK_JOINT_LIMIT_ATOL = 1.0e-9
GLOBAL_IK_CONTROLLER_PROFILE = "global_pseudo_trac_sampling_path_formal_v1"
STAGE6_IK_CONTROLLER_PROFILE = "stage6_hybrid_cartesian_executor_v19"
FROZEN_V4_CONTROLLER_PROFILE = "v4_frozen_legacy_replay"
FROZEN_V4_IK_RESOLUTION_METHOD = "pseudo_inverse"
FROZEN_V4_IK_MAX_ITERATIONS = 6
FROZEN_V4_IK_DAMPING = 0.1


class TracIKDistanceSolver(Protocol):
    """Pure external IK interface bound to one simulator arm.

    Implementations receive an absolute world-frame EE pose in RLBench's
    ``[x, y, z, qx, qy, qz, qw]`` convention.  A live adapter is responsible
    for transforming that pose into its calibrated kinematic-chain base.  It
    must return one joint vector without changing simulator state.
    """

    chain_source: str

    def solve(self, target_pose: Array) -> Any: ...


TracIKDistanceSolverFactory = Callable[[Any], TracIKDistanceSolver]


@dataclass(frozen=True)
class GlobalIKControllerConfig:
    """Parameters for the global formal absolute-EE controller."""

    trac_ik_timeout_s: float = 0.03
    trac_ik_epsilon: float = 1.0e-5
    trac_ik_translation_tolerance_m: float = 0.001
    trac_ik_rotation_tolerance_rad: float = math.radians(1.0)
    trac_ik_joint_window_rad: float = 0.35
    trac_ik_fk_translation_max_m: float = 0.002
    trac_ik_fk_rotation_max_rad: float = math.radians(2.0)
    far_translation_threshold_m: float = 0.10
    max_joint_delta_abs_rad: float = 0.35
    max_joint_delta_l2_rad: float = 0.50
    planner_trials: int = 200
    planner_max_configs: int = 10
    planner_max_time_ms: int = 20
    planner_trials_per_goal: int = 1

    def __post_init__(self) -> None:
        for name in (
            "trac_ik_timeout_s",
            "trac_ik_epsilon",
            "trac_ik_translation_tolerance_m",
            "trac_ik_rotation_tolerance_rad",
            "trac_ik_joint_window_rad",
            "trac_ik_fk_translation_max_m",
            "trac_ik_fk_rotation_max_rad",
            "far_translation_threshold_m",
            "max_joint_delta_abs_rad",
            "max_joint_delta_l2_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "planner_trials",
            "planner_max_configs",
            "planner_max_time_ms",
            "planner_trials_per_goal",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def protocol_id(self) -> str:
        return "rlbench-global-pseudo-trac-distance-sampling-path-formal-v1"

    def metadata(self) -> dict[str, Any]:
        return {
            "profile": GLOBAL_IK_CONTROLLER_PROFILE,
            "protocol_id": self.protocol_id,
            "formal_default": True,
            "global_task_special_cases": False,
            "ik_order": (
                "current_seeded_pseudo_inverse_then_bounded_trac_ik_distance_"
                "then_frozen_v4_collision_aware_sampling_then_far_path"
            ),
            "primary_resolution_method": FROZEN_V4_IK_RESOLUTION_METHOD,
            "primary_max_iterations": FROZEN_V4_IK_MAX_ITERATIONS,
            "primary_damping": FROZEN_V4_IK_DAMPING,
            "external_fallback": "bounded_trac_ik_distance_exact_chain",
            "external_fallback_entry_condition": "pseudo_inverse_exhausted",
            "external_solver_live_exact_chain_required": True,
            "external_solver_bounded_cartesian_api_required": True,
            "trac_ik_timeout_s": self.trac_ik_timeout_s,
            "trac_ik_epsilon": self.trac_ik_epsilon,
            "trac_ik_translation_tolerance_m": (self.trac_ik_translation_tolerance_m),
            "trac_ik_rotation_tolerance_rad": (self.trac_ik_rotation_tolerance_rad),
            "trac_ik_joint_window_rad": self.trac_ik_joint_window_rad,
            "trac_ik_fk_translation_max_m": self.trac_ik_fk_translation_max_m,
            "trac_ik_fk_rotation_max_rad": self.trac_ik_fk_rotation_max_rad,
            "sampling_entry_condition": "pseudo_inverse_and_trac_ik_exhausted",
            "sampling_trials": IK_SAMPLING_TRIALS,
            "sampling_max_configs": IK_SAMPLING_MAX_CONFIGS,
            "sampling_max_time_ms": IK_SAMPLING_MAX_TIME_MS,
            "sampling_ignore_collisions": False,
            "sampling_selection": "valid_candidate_nearest_current_joint_state",
            "sampling_hard_joint_delta_rejection": False,
            "sampling_fallback": True,
            "legacy_frozen_ik_helper_used": False,
            "far_translation_threshold_m": self.far_translation_threshold_m,
            "path_entry_condition": (
                "pseudo_inverse_trac_ik_distance_and_sampling_failed_and_"
                "translation_strictly_above_threshold"
            ),
            "rotation_alone_triggers_path": False,
            "path_controller": "collision_aware_get_path_rrt_connect",
            "far_path_ignore_collisions": False,
            "prepare_all_arms_before_physics": True,
            "trac_ik_max_joint_delta_abs_rad": self.max_joint_delta_abs_rad,
            "trac_ik_max_joint_delta_l2_rad": self.max_joint_delta_l2_rad,
            "pseudo_candidate_validation": "shape_finite_and_joint_limits",
            "trac_candidate_validation": (
                "shape_finite_joint_limits_and_joint_continuity"
            ),
            "sampling_candidate_validation": "shape_finite_and_joint_limits",
            "planner_trials": self.planner_trials,
            "planner_max_configs": self.planner_max_configs,
            "planner_max_time_ms": self.planner_max_time_ms,
            "planner_trials_per_goal": self.planner_trials_per_goal,
            "ik_group_post_prepare_method": FROZEN_V4_IK_RESOLUTION_METHOD,
            "ik_group_post_prepare_max_iterations": FROZEN_V4_IK_MAX_ITERATIONS,
            "ik_group_post_prepare_damping": FROZEN_V4_IK_DAMPING,
        }


@dataclass(frozen=True)
class Stage6IKControllerConfig(GlobalIKControllerConfig):
    """RLBench integration profile with physical Cartesian feedback.

    The frozen V4 executor accepts a joint controller ``stopped`` return as a
    completed primary action.  That is adequate for reproducing its clock but
    not for a closed-loop policy whose StateId must remain at an unreached
    target.  Stage six therefore verifies physical Cartesian progress while
    keeping solver failure separate from a physically stalled but valid motor
    command.  The latter is reported to the closed-loop policy without hidden
    extra motion, so progress and recovery remain observation-driven.
    """

    # Solver/post-execution verification keeps a 0.5 mm physical envelope.
    # The outer closed-loop acceptance envelope is 1 mm / 0.1 degree, matching
    # the smallest retained DynaMAC pose-factor scale and ordinary Panda servo
    # jitter.  This prevents an already adequate endpoint from being reissued
    # until random sub-millimetre/angular noise happens to cross the solver's
    # tighter envelope, while leaving state/guard compatibility to the learned
    # probabilistic models.
    # These values are global integration settings; they do not depend on a
    # task name, StateId, or boundary identity.
    trac_ik_timeout_s: float = 0.05
    trac_ik_translation_tolerance_m: float = 0.0005
    trac_ik_rotation_tolerance_rad: float = math.radians(0.05)
    trac_ik_fk_translation_max_m: float = 0.002
    trac_ik_fk_rotation_max_rad: float = math.radians(1.0)
    physical_completion_translation_tolerance_m: float = 0.0005
    physical_completion_rotation_tolerance_rad: float = math.radians(0.05)
    control_acceptance_translation_tolerance_m: float = 0.001
    control_acceptance_rotation_tolerance_rad: float = math.radians(0.1)
    cartesian_continuation_translation_step_m: float = 0.005
    cartesian_continuation_rotation_step_rad: float = math.radians(2.0)
    cartesian_continuation_backoff_factors: tuple[float, ...] = (
        1.0,
        0.5,
        0.25,
        0.125,
    )
    cartesian_continuation_max_segments: int = 64
    # Bound the aggregate simulator work of one policy command.  The segment
    # limit bounds solver reconstruction count, but each segment can itself
    # consume ``max_steps`` raw physics updates; without this second budget a
    # slowly converging yet non-stalled target can monopolize one closed-loop
    # observation for minutes.  Sixty-four raw updates still give a strict
    # bound.  The segment count uses the same bound because every successfully
    # executed segment consumes at least one raw update; a second, tighter
    # eight-segment limit previously returned before the raw budget and let a
    # moving trajectory cursor outrun the physical arm.
    cartesian_continuation_max_raw_physics_steps: int = 64
    linear_path_steps: int = 50
    allow_collision_relaxed_linear_path: bool = True
    allow_collision_relaxed_nonlinear_path: bool = True
    allow_collision_relaxed_sampling: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "physical_completion_translation_tolerance_m",
            "physical_completion_rotation_tolerance_rad",
            "control_acceptance_translation_tolerance_m",
            "control_acceptance_rotation_tolerance_rad",
            "cartesian_continuation_translation_step_m",
            "cartesian_continuation_rotation_step_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        factors = tuple(
            float(value) for value in self.cartesian_continuation_backoff_factors
        )
        if (
            not factors
            or any(
                not math.isfinite(value) or not 0.0 < value <= 1.0 for value in factors
            )
            or any(left <= right for left, right in zip(factors, factors[1:]))
        ):
            raise ValueError(
                "cartesian continuation backoff factors must be finite, "
                "positive, at most one, and strictly descending"
            )
        object.__setattr__(self, "cartesian_continuation_backoff_factors", factors)
        if (
            isinstance(self.cartesian_continuation_max_segments, bool)
            or not isinstance(self.cartesian_continuation_max_segments, int)
            or self.cartesian_continuation_max_segments < 1
        ):
            raise ValueError("cartesian_continuation_max_segments must be positive")
        if (
            isinstance(self.cartesian_continuation_max_raw_physics_steps, bool)
            or not isinstance(self.cartesian_continuation_max_raw_physics_steps, int)
            or self.cartesian_continuation_max_raw_physics_steps < 1
        ):
            raise ValueError(
                "cartesian_continuation_max_raw_physics_steps must be positive"
            )
        if (
            isinstance(self.linear_path_steps, bool)
            or not isinstance(self.linear_path_steps, int)
            or self.linear_path_steps < 2
        ):
            raise ValueError("linear_path_steps must be an integer of at least two")
        if not isinstance(self.allow_collision_relaxed_linear_path, bool):
            raise TypeError("allow_collision_relaxed_linear_path must be boolean")
        if not isinstance(self.allow_collision_relaxed_nonlinear_path, bool):
            raise TypeError("allow_collision_relaxed_nonlinear_path must be boolean")
        if not isinstance(self.allow_collision_relaxed_sampling, bool):
            raise TypeError("allow_collision_relaxed_sampling must be boolean")

    @property
    def protocol_id(self) -> str:
        return "rlbench-stage6-hybrid-cartesian-continuation-v23"

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update(
            {
                "profile": STAGE6_IK_CONTROLLER_PROFILE,
                "protocol_id": self.protocol_id,
                "formal_default": False,
                "ik_order": (
                    "current_seeded_continuous_pseudo_inverse_then_"
                    "bounded_trac_ik_distance_"
                    "then_bounded_cartesian_continuation_"
                    "then_collision_aware_or_relaxed_sampling_"
                    "then_collision_aware_linear_then_bounded_nonlinear_path_"
                    "then_collision_relaxed_linear_then_collision_relaxed_"
                    "nonlinear_path"
                ),
                "primary_resolution_method": (
                    "current_seeded_coppeliasim_pseudo_inverse"
                ),
                "external_fallback_entry_condition": ("pseudo_inverse_exhausted"),
                "pseudo_inverse_continuity_gate": True,
                "post_execution_cartesian_verification": True,
                "cartesian_translation_tolerance_m": (
                    self.physical_completion_translation_tolerance_m
                ),
                "cartesian_rotation_tolerance_rad": (
                    self.physical_completion_rotation_tolerance_rad
                ),
                "control_acceptance_translation_tolerance_m": (
                    self.control_acceptance_translation_tolerance_m
                ),
                "control_acceptance_rotation_tolerance_rad": (
                    self.control_acceptance_rotation_tolerance_rad
                ),
                "cartesian_continuation": {
                    "entry_condition": "full_target_local_ik_exhausted",
                    "translation_step_m": (
                        self.cartesian_continuation_translation_step_m
                    ),
                    "rotation_step_rad": (
                        self.cartesian_continuation_rotation_step_rad
                    ),
                    "backoff_factors": list(
                        self.cartesian_continuation_backoff_factors
                    ),
                    "max_segments_per_policy_action": (
                        self.cartesian_continuation_max_segments
                    ),
                    "max_raw_physics_steps_per_policy_action": (
                        self.cartesian_continuation_max_raw_physics_steps
                    ),
                    "interpolation": "linear_translation_shortest_xyzw_slerp",
                    "progress_feedback": ("physical_pose_reobserved_between_segments"),
                    "commit_semantics": (
                        "reach_or_report_bounded_progress_for_closed_loop_reobservation"
                    ),
                    "task_specific": False,
                },
                "path_fallback": {
                    "order": (
                        "collision_aware_linear_then_collision_aware_rrt_connect_"
                        "after_measured_stall_or_for_far_targets_then_"
                        "collision_relaxed_linear_then_collision_relaxed_"
                        "rrt_connect"
                    ),
                    "linear_steps": self.linear_path_steps,
                    "collision_relaxed_linear_enabled": (
                        self.allow_collision_relaxed_linear_path
                    ),
                    "collision_relaxed_nonlinear_enabled": (
                        self.allow_collision_relaxed_nonlinear_path
                    ),
                    "nonlinear_minimum_translation_m": (
                        self.far_translation_threshold_m
                    ),
                    "near_nonlinear_entry_condition": (
                        "same_target_measured_stall_exhausted_local_solver_tiers"
                    ),
                    "motion_handle_lifetime": (
                        "released_when_complete_or_before_bounded_controller_return"
                    ),
                    "task_specific": False,
                },
                "sampling_collision_policy": (
                    "collision_aware_then_collision_relaxed"
                    if self.allow_collision_relaxed_sampling
                    else "collision_aware_only"
                ),
                "stopped_resolution": "report_observed_stall_to_closed_loop",
                "physical_stall_resolution": (
                    "same_target_bounded_solver_escalation_then_report_stall"
                ),
                "fallback_collision_policy": (
                    "collision_aware_first_then_bounded_relaxation"
                ),
                "same_target_alternate_solver_after_primary_stall": True,
                "same_target_solver_tier_persistence": (
                    "per_arm_exact_target_across_closed_loop_cycles"
                ),
                "physical_stall_solver_order": (
                    "pseudo_inverse_then_trac_ik_distance_then_"
                    "cartesian_continuation_then_sampling_then_"
                    "collision_aware_linear_path_then_collision_aware_"
                    "nonlinear_path_then_collision_relaxed_linear_path"
                ),
                "physical_stall_sampling_candidate_validation": (
                    "shape_finite_joint_limits_and_joint_continuity"
                ),
                "unreported_hidden_motion_after_physical_stall": False,
                "joint_target_execution": (
                    "shared_clock_joint_target_until_reached_or_stopped"
                ),
                "same_target_ik_solution_cache": True,
                "same_target_ik_solution_cache_semantics": (
                    "reuse_only_after_measured_cartesian_progress_and_"
                    "invalidate_before_bounded_stall_escalation"
                ),
                "task_specific_controller_branches": False,
            }
        )
        return value


FORMAL_CONTROLLER_METADATA_SCHEMA = "dynamac-global-absolute-ee-controller-v1"


def global_ik_controller_metadata(
    config: GlobalIKControllerConfig | None = None,
) -> dict[str, Any]:
    """Return the exact controller/clock identity shared by every evaluator."""

    config = config or GlobalIKControllerConfig()
    metadata = {
        "schema": FORMAL_CONTROLLER_METADATA_SCHEMA,
        "profile": (
            STAGE6_IK_CONTROLLER_PROFILE
            if isinstance(config, Stage6IKControllerConfig)
            else GLOBAL_IK_CONTROLLER_PROFILE
        ),
        "protocol_id": config.protocol_id,
        "command": "absolute_world_end_effector_pose",
        "ik_order": config.metadata()["ik_order"],
        "primary_ik": config.metadata()["primary_resolution_method"],
        "primary_ik_parameters": {
            "max_iterations": FROZEN_V4_IK_MAX_ITERATIONS,
            "damping": FROZEN_V4_IK_DAMPING,
        },
        "bounded_trac_ik": {
            "solve_type": "Distance",
            "timeout_s": config.trac_ik_timeout_s,
            "epsilon": config.trac_ik_epsilon,
            "cartesian_translation_tolerance_m": (
                config.trac_ik_translation_tolerance_m
            ),
            "cartesian_rotation_tolerance_rad": (config.trac_ik_rotation_tolerance_rad),
            "joint_window_rad": config.trac_ik_joint_window_rad,
            "fk_translation_residual_max_m": (config.trac_ik_fk_translation_max_m),
            "fk_rotation_residual_max_rad": config.trac_ik_fk_rotation_max_rad,
            "exact_live_coppeliasim_chain_required": True,
            "exact_live_chain_schema": "coppeliasim-moving-frame-panda-chain-v1",
            "exact_live_chain_source": "live_coppeliasim_moving_frame_segments",
            "bounded_cartesian_api_required": True,
            "joint_delta_abs_max_rad": config.max_joint_delta_abs_rad,
            "joint_delta_l2_max_rad": config.max_joint_delta_l2_rad,
            "dependency": pytracik_dependency_identity(),
        },
        "sampling_ik": {
            "trials": IK_SAMPLING_TRIALS,
            "max_configs": IK_SAMPLING_MAX_CONFIGS,
            "max_time_ms": IK_SAMPLING_MAX_TIME_MS,
            "ignore_collisions": False,
            "selection": "nearest_current_joint_l2",
            "candidate_validation": "shape_finite_noncyclic_joint_limits",
            "hard_joint_delta_rejection": False,
        },
        "far_path": {
            "translation_threshold_m": config.far_translation_threshold_m,
            "entry_condition": "all_three_ik_methods_failed_and_strictly_far",
            "ignore_collisions": False,
            "trials": config.planner_trials,
            "max_configs": config.planner_max_configs,
            "max_time_ms": config.planner_max_time_ms,
            "trials_per_goal": config.planner_trials_per_goal,
        },
        "prepare_all_arms_before_physics": True,
        "joint_target_max_steps": 200,
        "primary_action_requests_per_policy_tick": 1,
        "primary_failure_resolution": (
            "one_raw_current_joint_hold_then_commit_same_transaction"
        ),
        "policy_clock_semantics_id": FORMAL_POLICY_CLOCK_SEMANTICS_ID,
        "formal_horizon_domain": "committed_policy_ticks",
        "task_specific_controller_branches": False,
        "legacy_v4_frozen_profile_used": False,
    }
    if isinstance(config, Stage6IKControllerConfig):
        metadata.update(
            {
                "post_execution_cartesian_verification": True,
                "cartesian_translation_tolerance_m": (
                    config.physical_completion_translation_tolerance_m
                ),
                "cartesian_rotation_tolerance_rad": (
                    config.physical_completion_rotation_tolerance_rad
                ),
                "control_acceptance_translation_tolerance_m": (
                    config.control_acceptance_translation_tolerance_m
                ),
                "control_acceptance_rotation_tolerance_rad": (
                    config.control_acceptance_rotation_tolerance_rad
                ),
                "stopped_resolution": "report_observed_stall_to_closed_loop",
                "fallback_collision_policy": (
                    "collision_aware_first_then_bounded_relaxation"
                ),
                "cartesian_continuation": config.metadata()["cartesian_continuation"],
                "path_fallback": config.metadata()["path_fallback"],
                "sampling_collision_policy": config.metadata()[
                    "sampling_collision_policy"
                ],
                "same_target_alternate_solver_after_primary_stall": True,
                "same_target_solver_tier_persistence": config.metadata()[
                    "same_target_solver_tier_persistence"
                ],
                "physical_stall_resolution": (
                    "same_target_bounded_solver_escalation_then_report_stall"
                ),
                "physical_stall_solver_order": (
                    "pseudo_inverse_then_trac_ik_distance_then_"
                    "cartesian_continuation_then_sampling_then_"
                    "collision_aware_linear_path_then_collision_aware_"
                    "nonlinear_path_then_collision_relaxed_linear_path"
                ),
                "physical_stall_sampling_candidate_validation": (
                    "shape_finite_joint_limits_and_joint_continuity"
                ),
                "unreported_hidden_motion_after_physical_stall": False,
                "joint_target_execution": (
                    "shared_clock_joint_target_until_reached_or_stopped"
                ),
                "same_target_ik_solution_cache": True,
                "same_target_ik_solution_cache_semantics": config.metadata()[
                    "same_target_ik_solution_cache_semantics"
                ],
            }
        )
    return metadata


@dataclass(frozen=True)
class PreparedEECommand:
    arm: Any
    target_pose: Array
    mode: Literal["joint_target", "planned_path"]
    target_joints: Array | None = None
    path: Any = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.target_pose, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("prepared EE target pose must be finite shape (7,)")
        object.__setattr__(self, "target_pose", pose.copy())
        if self.mode == "joint_target":
            joints = np.asarray(self.target_joints, dtype=np.float64)
            if joints.ndim != 1 or not np.all(np.isfinite(joints)):
                raise ValueError("prepared joint target must be a finite vector")
            if self.path is not None:
                raise ValueError("joint-target command cannot contain a path")
            object.__setattr__(self, "target_joints", joints.copy())
        elif self.mode == "planned_path":
            if self.path is None or self.target_joints is not None:
                raise ValueError("planned-path command must contain only a path")
            if not callable(getattr(self.path, "step", None)):
                raise ValueError("planned-path command requires a step API")
        else:
            raise ValueError("unsupported prepared EE command mode")


def _release_configuration_path_motion_handle(
    path: Any,
    *,
    remover: Callable[[int], Any] | None = None,
) -> bool:
    """Release one unfinished PyRep configuration-path motion handle.

    ``ArmConfigurationPath.step`` removes its Reflexxes handle only when the
    complete path reaches its end.  Stage six deliberately returns to the
    observer after a bounded number of raw physics steps, so an unfinished
    path is discarded by design and must release that simulator resource at
    the same ownership boundary.  Completed paths have already removed their
    handle inside PyRep and are left untouched.
    """

    handle = getattr(path, "_rml_handle", None)
    if handle is None or bool(getattr(path, "_path_done", False)):
        return False
    if remover is None:
        from pyrep.backend import sim

        remover = sim.simRMLRemove
    remover(int(handle))
    # This path object is discarded after the bounded controller return.  Mark
    # it terminal as well as clearing the stale simulator handle so accidental
    # reuse cannot recreate or step a prefix that no longer owns its motion.
    setattr(path, "_rml_handle", None)
    setattr(path, "_path_done", True)
    return True


_STAGE6_JOINT_CACHE_ATTRIBUTE = "_dynamac_stage6_joint_target_cache_v1"
_STAGE6_SOLVER_TIER_ATTRIBUTE = "_dynamac_stage6_solver_tier_v1"
_STAGE6_SOLVER_TIER_COUNT = 8


def _same_stage6_cartesian_target(left: Array, right: Array) -> bool:
    translation, rotation = end_effector_pose_distance(left, right)
    return bool(translation <= 1.0e-10 and rotation <= 1.0e-10)


def _cached_stage6_joint_command(
    arm: Any,
    target: Array,
    *,
    current: Array,
    limits: tuple[tuple[bool, float, float], ...],
    config: Stage6IKControllerConfig,
    diagnostics: dict[str, Any],
) -> PreparedEECommand | None:
    cached = getattr(arm, _STAGE6_JOINT_CACHE_ATTRIBUTE, None)
    if not isinstance(cached, dict):
        _increment_ik_diagnostic(diagnostics, "same_target_joint_cache_misses")
        return None
    cached_pose = np.asarray(cached.get("target_pose"), dtype=np.float64)
    if (
        cached_pose.shape != (7,)
        or not np.all(np.isfinite(cached_pose))
        or not _same_stage6_cartesian_target(cached_pose, target)
    ):
        _increment_ik_diagnostic(diagnostics, "same_target_joint_cache_misses")
        return None
    candidate, rejection = _validated_continuous_ik_candidate(
        cached.get("target_joints"),
        current_joints=current,
        limits=limits,
        config=config,
    )
    if candidate is None:
        _increment_ik_diagnostic(diagnostics, "same_target_joint_cache_rejections")
        _record_ik_candidate_rejection(diagnostics, str(rejection))
        try:
            delattr(arm, _STAGE6_JOINT_CACHE_ATTRIBUTE)
        except AttributeError:
            pass
        return None
    _increment_ik_diagnostic(diagnostics, "same_target_joint_cache_hits")
    _record_selected_joint_delta(diagnostics, current, candidate)
    return PreparedEECommand(
        arm=arm,
        target_pose=target,
        mode="joint_target",
        target_joints=candidate,
    )


def _store_stage6_joint_command(command: PreparedEECommand, target: Array) -> None:
    if (
        command.mode != "joint_target"
        or command.target_joints is None
        or not _same_stage6_cartesian_target(command.target_pose, target)
    ):
        return
    setattr(
        command.arm,
        _STAGE6_JOINT_CACHE_ATTRIBUTE,
        {
            "target_pose": np.asarray(target, dtype=np.float64).copy(),
            "target_joints": command.target_joints.copy(),
        },
    )


def _invalidate_stage6_joint_command(arm: Any, target: Array) -> bool:
    cached = getattr(arm, _STAGE6_JOINT_CACHE_ATTRIBUTE, None)
    if not isinstance(cached, dict):
        return False
    cached_pose = np.asarray(cached.get("target_pose"), dtype=np.float64)
    if (
        cached_pose.shape != (7,)
        or not np.all(np.isfinite(cached_pose))
        or not _same_stage6_cartesian_target(cached_pose, target)
    ):
        return False
    delattr(arm, _STAGE6_JOINT_CACHE_ATTRIBUTE)
    return True


def _stage6_same_target_solver_tier(
    arm: Any,
    target: Array,
    *,
    diagnostics: dict[str, Any],
) -> int:
    """Restore the bounded solver tier for one unchanged Cartesian target.

    A Stage-6 policy target may span several observation cycles.  The raw
    physics budget intentionally ends each cycle early, so a physically
    stalled solver family must remember that result; otherwise every new
    cycle restarts at pseudo-inverse and the documented fallback chain is
    unreachable.  A changed target starts a fresh chain at tier zero.
    """

    state = getattr(arm, _STAGE6_SOLVER_TIER_ATTRIBUTE, None)
    if isinstance(state, dict):
        cached_pose = np.asarray(state.get("target_pose"), dtype=np.float64)
        tier = state.get("minimum_solver_tier")
        if (
            cached_pose.shape == (7,)
            and np.all(np.isfinite(cached_pose))
            and _same_stage6_cartesian_target(cached_pose, target)
            and isinstance(tier, int)
            and not isinstance(tier, bool)
            and tier in range(_STAGE6_SOLVER_TIER_COUNT)
        ):
            _increment_ik_diagnostic(diagnostics, "same_target_solver_tier_cache_hits")
            if tier > 0:
                _increment_ik_diagnostic(
                    diagnostics, "same_target_cross_cycle_solver_resumes"
                )
            return int(tier)
    _increment_ik_diagnostic(diagnostics, "same_target_solver_tier_cache_misses")
    setattr(
        arm,
        _STAGE6_SOLVER_TIER_ATTRIBUTE,
        {
            "target_pose": np.asarray(target, dtype=np.float64).copy(),
            "minimum_solver_tier": 0,
        },
    )
    return 0


def _store_stage6_same_target_solver_tier(
    arm: Any,
    target: Array,
    tier: int,
) -> None:
    if isinstance(tier, bool) or tier not in range(_STAGE6_SOLVER_TIER_COUNT):
        raise ValueError("stage-six solver tier must be an integer in [0, 7]")
    setattr(
        arm,
        _STAGE6_SOLVER_TIER_ATTRIBUTE,
        {
            "target_pose": np.asarray(target, dtype=np.float64).copy(),
            "minimum_solver_tier": int(tier),
        },
    )


def _clear_stage6_same_target_solver_tier(arm: Any) -> None:
    try:
        delattr(arm, _STAGE6_SOLVER_TIER_ATTRIBUTE)
    except AttributeError:
        pass


@dataclass
class PrimaryActionRetryBudget:
    """Bound consecutive InvalidAction retries for one policy clock tick.

    Formal evaluation makes one primary attempt. A failure is resolved by a
    raw current-joint hold that commits the same policy transaction.
    """

    max_attempts: int = DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
    attempts: int = 0
    peak_attempts: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TypeError("max primary action attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max primary action attempts must be positive")

    def record_failure(self) -> bool:
        """Record one aborted primary attempt and report budget exhaustion."""

        self.attempts += 1
        self.peak_attempts = max(self.peak_attempts, self.attempts)
        return self.attempts >= self.max_attempts

    def record_success(self) -> None:
        """Reset only after the primary command is successfully committed."""

        self.attempts = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "max_primary_action_attempts_per_policy_tick": self.max_attempts,
            "exhaustion_reason": "primary_action_retry_exhausted",
            "counter_reset": "after_successful_primary_action_commit",
        }


def validate_primary_retry_exhaustion_mode(mode: str) -> str:
    """Validate formal or explicitly isolated legacy exhaustion behavior."""

    if mode not in {
        FORMAL_PRIMARY_RETRY_EXHAUSTION_MODE,
        FROZEN_PRIMARY_RETRY_EXHAUSTION_MODE,
    }:
        raise ValueError("unsupported primary-action retry exhaustion mode")
    return mode


def step_current_joint_hold_noop(
    task_environment: Any,
) -> tuple[Any, Any, Any]:
    """Advance one raw physics step while holding every robot joint.

    This deliberately bypasses ``TaskEnvironment.step`` and the configured
    action mode.  In particular, no absolute-EE command, IK solve, or discrete
    gripper action is issued.  Arm and gripper attachments are left untouched.
    """

    if getattr(task_environment, "_reset_called", True) is False:
        raise RuntimeError("Call 'reset' before advancing a joint-hold no-op.")
    scene = getattr(task_environment, "_scene", None)
    robot = getattr(scene, "robot", None)
    task = getattr(scene, "task", None)
    step_scene = getattr(scene, "step", None)
    success_fn = getattr(task, "success", None)
    get_observation = getattr(task_environment, "get_observation", None)
    if (
        robot is None
        or not callable(step_scene)
        or not callable(success_fn)
        or not callable(get_observation)
    ):
        raise RuntimeError("joint-hold no-op requires a live RLBench task scene")

    if bool(getattr(robot, "is_bimanual", False)):
        components = (
            getattr(robot, "right_arm", None),
            getattr(robot, "right_gripper", None),
            getattr(robot, "left_arm", None),
            getattr(robot, "left_gripper", None),
        )
    else:
        components = (
            getattr(robot, "arm", None),
            getattr(robot, "gripper", None),
        )

    snapshots: list[tuple[Any, Array]] = []
    for component in components:
        get_positions = getattr(component, "get_joint_positions", None)
        set_targets = getattr(component, "set_joint_target_positions", None)
        if not callable(get_positions) or not callable(set_targets):
            raise RuntimeError("robot component lacks joint-hold APIs")
        positions = np.asarray(get_positions(), dtype=np.float64)
        if (
            positions.ndim != 1
            or positions.size == 0
            or not np.all(np.isfinite(positions))
        ):
            raise RuntimeError("robot component returned invalid joint positions")
        snapshots.append((component, positions.copy()))

    # Snapshot every component before changing any target, then arm all holds
    # before the single shared physics step.
    for component, positions in snapshots:
        component.set_joint_target_positions(positions.tolist())
    step_scene()

    success, terminate = success_fn()
    reward: Any = float(success)
    if bool(getattr(task_environment, "_shaped_rewards", False)):
        reward_fn = getattr(task, "reward", None)
        if not callable(reward_fn):
            raise RuntimeError(
                "shaped rewards requested but task.reward is unavailable"
            )
        reward = reward_fn()
        if reward is None:
            raise RuntimeError(
                "User requested shaped rewards, but the task returned no reward."
            )
    observation = get_observation()
    return observation, reward, terminate


def policy_action_execution_status(task_environment: Any) -> str:
    """Return whether the latest absolute-EE policy target was completed.

    RLBench's public ``TaskEnvironment.step`` API does not return the status of
    its arm action mode.  Stage 6 deliberately allows one absolute Cartesian
    target to consume several bounded observation cycles, so the evaluator
    must distinguish a fully reached target from useful bounded progress.  A
    non-Stage-6 action mode retains the historical one-step ``reached``
    semantics.
    """

    action_mode = getattr(task_environment, "_action_mode", None)
    arm_action_mode = getattr(action_mode, "arm_action_mode", None)
    status_fn = getattr(arm_action_mode, "policy_action_status", None)
    status = "reached" if not callable(status_fn) else status_fn()
    if status not in {"reached", "progressed", "stopped"}:
        raise RuntimeError(f"unsupported policy action execution status: {status!r}")
    return str(status)


def policy_action_execution_statuses(task_environment: Any) -> dict[str, str]:
    """Return per-arm physical response status for the latest policy target.

    The aggregate status remains the authority for atomic gripper submission.
    These per-arm values are executor diagnostics only; the closed-loop task
    progress prior consumes the applied command and next observation, not this
    low-level classification.
    """

    action_mode = getattr(task_environment, "_action_mode", None)
    arm_action_mode = getattr(action_mode, "arm_action_mode", None)
    statuses_fn = getattr(arm_action_mode, "policy_action_statuses", None)
    if callable(statuses_fn):
        statuses = statuses_fn()
        if not isinstance(statuses, dict) or not statuses:
            raise RuntimeError("policy arm action statuses must be a non-empty dict")
    else:
        statuses = {"single": policy_action_execution_status(task_environment)}
    result = {}
    for arm, status in statuses.items():
        if not isinstance(arm, str) or not arm:
            raise RuntimeError("policy arm action status keys must be arm names")
        if status not in {"reached", "progressed", "stopped"}:
            raise RuntimeError(f"unsupported per-arm policy action status: {status!r}")
        result[arm] = str(status)
    return result


def apply_gripper_for_policy_target(
    gripper_action_mode: Any,
    scene: Any,
    gripper_action: Array,
    *,
    arm_status: str,
    gripper_authorized: bool | None = None,
) -> bool:
    """Apply a gripper command under task authorization or legacy sequencing.

    Closed-loop TASK commands pass an explicit authorization derived from the
    task posterior and boundary transaction.  Frozen DynaMAC passes an
    explicit authorization from its own fixed-clock gripper command.  ``None``
    retains pose-completion sequencing only for auxiliary commands that have
    no task-level authorization source.
    """

    if arm_status not in {"reached", "progressed", "stopped"}:
        raise ValueError(f"unsupported arm execution status: {arm_status!r}")
    if gripper_authorized is not None and not isinstance(
        gripper_authorized, (bool, np.bool_)
    ):
        raise TypeError("gripper_authorized must be Boolean or None")
    apply = (
        arm_status == "reached"
        if gripper_authorized is None
        else bool(gripper_authorized)
    )
    if not apply:
        return False
    gripper_action_mode.action(scene, gripper_action)
    return True


def set_policy_gripper_authorization(
    task_environment: Any,
    authorization: Mapping[str, bool | None] | None,
) -> None:
    """Attach one cycle's closed-loop gripper authorization to its action."""

    action_mode = getattr(task_environment, "_action_mode", None)
    setter = getattr(action_mode, "set_policy_gripper_authorization", None)
    if callable(setter):
        setter(authorization)
        return
    if authorization is not None and any(
        value is not None for value in authorization.values()
    ):
        raise RuntimeError("action mode cannot consume policy gripper authorization")


def commit_joint_hold_after_primary_failure(
    task_environment: Any,
    worker: Any,
    *,
    transaction_id: int,
) -> tuple[Any, Any, Any, bool]:
    """Resolve one failed primary transaction with one raw joint-hold step.

    The transaction deliberately remains pending while the current-state
    hold is executed. If the hold fails, it is aborted; only a successful hold
    consumes the tentative shared policy tick.
    """

    try:
        observation, reward, terminate = step_current_joint_hold_noop(task_environment)
    except Exception:
        worker.request("abort", transaction_id=transaction_id)
        raise
    try:
        commit = worker.request(
            "commit",
            transaction_id=transaction_id,
            primary_action_status="stopped",
            primary_action_applied=False,
        )
    except Exception:
        # The raw physics step cannot be rolled back. Best-effort resolution
        # prevents a locally known pending transaction from being reused.
        try:
            worker.request("abort", transaction_id=transaction_id)
        except Exception:
            pass
        raise
    return observation, reward, terminate, bool(commit.get("complete"))


def initialize_ik_solver_diagnostics() -> dict[str, Any]:
    """Return compact counters shared by the uni- and bimanual IK modes."""

    return {
        "jacobian_failures": 0,
        "jacobian_candidate_rejections": 0,
        "sampling_fallback_successes": 0,
        "sampling_fallback_failures": 0,
        "sampling_candidates_evaluated": 0,
        "candidate_rejections_nonfinite": 0,
        "candidate_rejections_joint_limits": 0,
        "candidate_rejections_shape": 0,
        "selected_via_jacobian": 0,
        "selected_via_sampling": 0,
        "selected_joint_delta_l2_max": 0.0,
    }


def initialize_global_ik_controller_diagnostics() -> dict[str, Any]:
    """Counters for the global pseudo/TRAC/sampling/path controller."""

    diagnostics = initialize_ik_solver_diagnostics()
    diagnostics.update(
        {
            "selected_joint_delta_abs_max": 0.0,
            "target_pose_count": 0,
            "target_translation_m_max": 0.0,
            "target_rotation_rad_max": 0.0,
            "pseudo_inverse_ik_attempts": 0,
            "pseudo_inverse_ik_successes": 0,
            "pseudo_inverse_ik_failures": 0,
            "pseudo_inverse_ik_candidate_rejections": 0,
            "selected_via_pseudo_inverse": 0,
            "trac_ik_distance_attempts": 0,
            "trac_ik_distance_successes": 0,
            "trac_ik_distance_failures": 0,
            "trac_ik_distance_factory_failures": 0,
            "trac_ik_distance_candidate_rejections": 0,
            "trac_ik_distance_solve_time_ms_total": 0.0,
            "trac_ik_distance_solve_time_ms_max": 0.0,
            "trac_ik_distance_reported_solve_time_ms_total": 0.0,
            "trac_ik_distance_reported_solve_time_ms_max": 0.0,
            "trac_ik_distance_chain_sources": [],
            "trac_ik_distance_chain_schemas": [],
            "trac_ik_distance_chain_source_missing": 0,
            "trac_ik_distance_chain_schema_missing": 0,
            "trac_ik_distance_bounded_cartesian_api_uses": 0,
            "trac_ik_distance_unbounded_cartesian_api_uses": 0,
            "trac_ik_distance_result_metadata_missing": 0,
            "trac_ik_distance_fk_translation_error_m_max": 0.0,
            "trac_ik_distance_fk_rotation_error_rad_max": 0.0,
            "trac_ik_distance_exhaustions": 0,
            "sampling_after_trac_attempts": 0,
            "sampling_after_trac_successes": 0,
            "sampling_after_trac_failures": 0,
            "sampling_collision_relaxed_attempts": 0,
            "sampling_collision_relaxed_successes": 0,
            "sampling_collision_relaxed_failures": 0,
            "selected_via_collision_relaxed_sampling": 0,
            "candidate_rejections_joint_delta_abs": 0,
            "candidate_rejections_joint_delta_l2": 0,
            "selected_via_trac_ik_distance": 0,
            "all_ik_exhaustions": 0,
            "path_after_all_ik_exhaustion": 0,
            "far_target_planner_attempts": 0,
            "far_target_planner_successes": 0,
            "far_target_planner_failures": 0,
            "nonlinear_path_attempts": 0,
            "nonlinear_path_successes": 0,
            "nonlinear_path_failures": 0,
            "near_target_nonlinear_planner_attempts": 0,
            "linear_path_attempts": 0,
            "linear_path_collision_aware_successes": 0,
            "linear_path_collision_aware_failures": 0,
            "linear_path_collision_relaxed_attempts": 0,
            "linear_path_collision_relaxed_successes": 0,
            "linear_path_collision_relaxed_failures": 0,
            "nonlinear_path_collision_relaxed_attempts": 0,
            "nonlinear_path_collision_relaxed_successes": 0,
            "nonlinear_path_collision_relaxed_failures": 0,
            "near_target_nonlinear_planner_skips": 0,
            "commands_prepared_before_physics": 0,
            "ik_group_baseline_restore_attempts": 0,
            "ik_group_baseline_restore_failures": 0,
            "trac_ik_distance_controller_invalid_actions": 0,
            "trac_ik_distance_controller_actions": 0,
            "trac_ik_distance_controller_raw_physics_steps": 0,
            "trac_ik_distance_controller_raw_physics_steps_max": 0,
            "cartesian_verification_checks": 0,
            "cartesian_verification_failures": 0,
            "cartesian_post_translation_error_m_max": 0.0,
            "cartesian_post_rotation_error_rad_max": 0.0,
            "stopped_with_cartesian_residual": 0,
            "reached_joint_target_with_cartesian_residual": 0,
            "controller_raw_physics_budget_exhaustions": 0,
            "cartesian_direct_goal_attempts": 0,
            "cartesian_direct_goal_reaches": 0,
            "cartesian_direct_goal_control_accepts": 0,
            "cartesian_direct_goal_progress_accepts": 0,
            "cartesian_direct_goal_stalls": 0,
            "physical_stall_solver_escalations": 0,
            "physical_stall_solver_escalation_exhaustions": 0,
            "physical_stall_solver_tier_max": 0,
            "same_target_solver_tier_cache_hits": 0,
            "same_target_solver_tier_cache_misses": 0,
            "same_target_cross_cycle_solver_resumes": 0,
            "same_target_cross_cycle_solver_escalations": 0,
            "same_target_solver_tier_resets_after_progress": 0,
            "same_target_joint_cache_hits": 0,
            "same_target_joint_cache_misses": 0,
            "same_target_joint_cache_rejections": 0,
            "same_target_joint_cache_stores": 0,
            "same_target_joint_cache_invalidations": 0,
            "cartesian_goal_directed_progress_accepts": 0,
            "cartesian_partial_arm_progress_accepts": 0,
            "cartesian_continuation_attempts": 0,
            "cartesian_continuation_successes": 0,
            "cartesian_continuation_failures": 0,
            "selected_via_cartesian_continuation": 0,
            "cartesian_continuation_fraction_min": 1.0,
            "cartesian_multi_pass_followup_passes": 0,
            "cartesian_multi_pass_goals_completed": 0,
            "cartesian_multi_pass_limit_exhaustions": 0,
            "cartesian_multi_pass_solver_exhaustions_after_progress": 0,
            "planned_path_motion_handle_releases": 0,
            "planned_path_motion_handle_release_failures": 0,
        }
    )
    return diagnostics


def _ensure_global_ik_controller_diagnostics(
    diagnostics: dict[str, Any],
) -> None:
    for key, value in initialize_global_ik_controller_diagnostics().items():
        diagnostics.setdefault(key, value)


def _increment_ik_diagnostic(
    diagnostics: dict[str, Any], key: str, amount: int = 1
) -> None:
    diagnostics[key] = int(diagnostics.get(key, 0)) + amount


def _record_solver_duration_ms(
    diagnostics: dict[str, Any], prefix: str, started_at: float
) -> None:
    duration_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    total_key = f"{prefix}_solve_time_ms_total"
    max_key = f"{prefix}_solve_time_ms_max"
    diagnostics[total_key] = float(diagnostics.get(total_key, 0.0)) + duration_ms
    diagnostics[max_key] = max(float(diagnostics.get(max_key, 0.0)), duration_ms)


def _record_ik_candidate_rejection(diagnostics: dict[str, Any], reason: str) -> None:
    _increment_ik_diagnostic(diagnostics, f"candidate_rejections_{reason}")


def end_effector_pose_distance(
    current_pose: Any, target_pose: Any
) -> tuple[float, float]:
    """Return translation and sign-invariant quaternion angular distance."""

    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    if (
        current.shape != (7,)
        or target.shape != (7,)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("end-effector poses must be finite vectors of shape (7,)")
    current_norm = float(np.linalg.norm(current[3:]))
    target_norm = float(np.linalg.norm(target[3:]))
    if current_norm <= 1.0e-12 or target_norm <= 1.0e-12:
        raise ValueError("end-effector quaternion norm must be positive")
    dot = abs(float(np.dot(current[3:], target[3:]) / (current_norm * target_norm)))
    rotation = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    return float(np.linalg.norm(target[:3] - current[:3])), rotation


def _record_target_pose_diagnostic(
    diagnostics: dict[str, Any], current_pose: Any, target_pose: Any
) -> tuple[float, float]:
    translation, rotation = end_effector_pose_distance(current_pose, target_pose)
    _increment_ik_diagnostic(diagnostics, "target_pose_count")
    diagnostics["target_translation_m_max"] = max(
        float(diagnostics.get("target_translation_m_max", 0.0)), translation
    )
    diagnostics["target_rotation_rad_max"] = max(
        float(diagnostics.get("target_rotation_rad_max", 0.0)), rotation
    )
    return translation, rotation


def _joint_limit_specification(
    arm: Any, current_joints: Array
) -> tuple[tuple[bool, float, float], ...]:
    cyclics, intervals = arm.get_joint_intervals()
    if len(cyclics) != current_joints.size or len(intervals) != current_joints.size:
        raise ValueError("joint interval count does not match current joint state")

    limits = []
    for cyclic, interval in zip(cyclics, intervals):
        if bool(cyclic):
            limits.append((True, -math.inf, math.inf))
            continue
        values = np.asarray(interval, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("non-cyclic joint interval must be two finite values")
        lower = float(values[0])
        span = float(values[1])
        upper = lower + span
        if span < 0.0 or not math.isfinite(upper):
            raise ValueError("non-cyclic joint interval range must be non-negative")
        limits.append((False, lower, upper))
    return tuple(limits)


def _validated_ik_candidate(
    candidate: Any,
    *,
    current_joints: Array,
    limits: tuple[tuple[bool, float, float], ...],
) -> tuple[Array | None, str | None]:
    try:
        joints = np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "shape"
    if joints.shape != current_joints.shape:
        return None, "shape"
    if not np.all(np.isfinite(joints)):
        return None, "nonfinite"
    for value, (cyclic, lower, upper) in zip(joints, limits):
        if cyclic:
            continue
        if value < lower - IK_JOINT_LIMIT_ATOL or value > upper + IK_JOINT_LIMIT_ATOL:
            return None, "joint_limits"
    return joints.copy(), None


def solve_absolute_ee_ik_with_sampling_fallback(
    arm: Any,
    target: Array,
    *,
    diagnostics: dict[str, Any],
    ik_error: type[Exception],
    configuration_error: type[Exception],
    invalid_action_error: type[Exception],
    error_message: str,
) -> Array:
    """Solve one absolute EE target with a validated, collision-aware fallback.

    Jacobian IK remains the local first choice and has no PyRep collision-check
    option.  If it fails or returns an invalid joint vector, sampling explicitly
    enables collision checking.  All returned configurations are checked for
    shape, finite values, and non-cyclic joint limits before the valid solution
    nearest to the current joint state is selected.
    """

    target = np.asarray(target, dtype=np.float64)
    current_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    if current_joints.ndim != 1 or not np.all(np.isfinite(current_joints)):
        raise invalid_action_error(error_message)
    try:
        limits = _joint_limit_specification(arm, current_joints)
    except (TypeError, ValueError) as exc:
        raise invalid_action_error(error_message) from exc

    try:
        jacobian_result = arm.solve_ik_via_jacobian(
            target[:3], quaternion=target[3:], relative_to=None
        )
    except ik_error:
        _increment_ik_diagnostic(diagnostics, "jacobian_failures")
    else:
        candidate, rejection = _validated_ik_candidate(
            jacobian_result,
            current_joints=current_joints,
            limits=limits,
        )
        if candidate is not None:
            _increment_ik_diagnostic(diagnostics, "selected_via_jacobian")
            delta = float(np.linalg.norm(candidate - current_joints))
            diagnostics["selected_joint_delta_l2_max"] = max(
                float(diagnostics.get("selected_joint_delta_l2_max", 0.0)),
                delta,
            )
            return candidate
        _increment_ik_diagnostic(diagnostics, "jacobian_candidate_rejections")
        _record_ik_candidate_rejection(diagnostics, str(rejection))

    try:
        sampling_result = arm.solve_ik_via_sampling(
            target[:3],
            quaternion=target[3:],
            ignore_collisions=False,
            trials=IK_SAMPLING_TRIALS,
            max_configs=IK_SAMPLING_MAX_CONFIGS,
            max_time_ms=IK_SAMPLING_MAX_TIME_MS,
            relative_to=None,
        )
    except configuration_error as exc:
        _increment_ik_diagnostic(diagnostics, "sampling_fallback_failures")
        raise invalid_action_error(error_message) from exc

    if isinstance(sampling_result, np.ndarray):
        if sampling_result.ndim == 2:
            candidates = tuple(sampling_result)
        elif sampling_result.ndim == 1 and sampling_result.size:
            candidates = (sampling_result,)
        elif sampling_result.size == 0:
            candidates = ()
        else:
            candidates = (sampling_result,)
    else:
        try:
            values = tuple(sampling_result)
        except TypeError:
            candidates = (sampling_result,)
        else:
            if values and all(np.asarray(value).ndim == 0 for value in values):
                candidates = (values,)
            else:
                candidates = values

    valid_candidates = []
    for raw_candidate in candidates:
        _increment_ik_diagnostic(diagnostics, "sampling_candidates_evaluated")
        candidate, rejection = _validated_ik_candidate(
            raw_candidate,
            current_joints=current_joints,
            limits=limits,
        )
        if candidate is None:
            _record_ik_candidate_rejection(diagnostics, str(rejection))
            continue
        valid_candidates.append(candidate)

    if not valid_candidates:
        _increment_ik_diagnostic(diagnostics, "sampling_fallback_failures")
        raise invalid_action_error(error_message)

    selected = min(
        valid_candidates,
        key=lambda candidate: float(np.linalg.norm(candidate - current_joints)),
    )
    _increment_ik_diagnostic(diagnostics, "sampling_fallback_successes")
    _increment_ik_diagnostic(diagnostics, "selected_via_sampling")
    delta = float(np.linalg.norm(selected - current_joints))
    diagnostics["selected_joint_delta_l2_max"] = max(
        float(diagnostics.get("selected_joint_delta_l2_max", 0.0)),
        delta,
    )
    return selected.copy()


def final_settling_metadata(
    physics_steps: int = DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
) -> dict[str, Any]:
    """Describe the task-independent terminal settling protocol."""

    if isinstance(physics_steps, bool) or not isinstance(physics_steps, int):
        raise TypeError("final settling physics steps must be an integer")
    if physics_steps < 0:
        raise ValueError("final settling physics steps must be non-negative")
    return {
        "protocol_id": (
            FINAL_SETTLING_PROTOCOL_ID
            if physics_steps == DEFAULT_FINAL_SETTLING_PHYSICS_STEPS
            else (
                "rlbench-hold-final-command-up-to-"
                f"{physics_steps}-raw-physics-steps-v3"
            )
        ),
        # ``physics_steps`` is retained as the backwards-compatible budget
        # field.  It is a maximum: success or an explicit task termination
        # stops settling immediately.
        "physics_steps": physics_steps,
        "maximum_physics_steps": physics_steps,
        "entry_condition": "normal_policy_complete_without_terminal_outcome",
        "command_semantics": "hold_existing_joint_targets_and_gripper_state",
        "step_api": "Scene.step",
        "success_and_failure_check": "Task.success_after_each_raw_physics_step",
        "early_stop": "first_success_or_explicit_terminate",
        "policy_clock_advanced": False,
        "dynamic_clock_advanced": False,
        "task_specific_adaptation": False,
    }


def run_final_settling(
    task_environment: Any,
    *,
    physics_steps: int = DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
) -> dict[str, Any]:
    """Hold final commands and advance only raw simulator physics.

    No policy action is requested and no high-level ``TaskEnvironment.step`` is
    executed.  Existing joint targets and the gripper state therefore remain
    active while ``Scene.step`` advances physics and the task callback once.
    RLBench success/failure conditions are checked after every raw step.
    """

    metadata = final_settling_metadata(physics_steps)
    result = {
        **metadata,
        "attempted": physics_steps > 0,
        "available": True,
        "steps_executed": 0,
        "first_terminal_step": None,
        "stop_reason": "budget_zero" if physics_steps == 0 else None,
        "success": False,
        "terminate": False,
    }
    if physics_steps == 0:
        return result
    scene = getattr(task_environment, "_scene", None)
    step_scene = getattr(scene, "step", None)
    task = getattr(scene, "task", None)
    success_fn = getattr(task, "success", None)
    if not callable(step_scene) or not callable(success_fn):
        # Lightweight evaluator unit-test doubles intentionally omit private
        # RLBench scene internals. Real V3 evaluation always has both APIs.
        result.update(
            {
                "available": False,
                "attempted": False,
                "stop_reason": "raw_scene_or_task_success_api_unavailable",
            }
        )
        return result
    for index in range(1, physics_steps + 1):
        step_scene()
        success, terminate = success_fn()
        result["steps_executed"] = index
        result["success"] = bool(success)
        result["terminate"] = bool(terminate)
        if success or terminate:
            result["first_terminal_step"] = index
            result["stop_reason"] = "success" if success else "explicit_terminate"
            break
    if result["stop_reason"] is None:
        result["stop_reason"] = "maximum_physics_steps_reached"
    return result


def _protocol_float_token(value: float) -> str:
    """Return a compact, deterministic float token suitable for protocol IDs."""

    return format(value, ".12g").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class DiscreteGripperProtocol:
    """Task-independent discrete-gripper evaluation protocol.

    The pinned RLBench fork hard-codes velocity ``0.2`` in both ``Discrete``
    action modes, while its demonstration generator hard-codes ``0.04``.
    This project-owned protocol changes only that actuation velocity and
    inherits every other vendor behavior, including grasp attachment,
    release settling, and the bimanual handover logic.

    Keeping construction, metadata, and the protocol-ID fragment on one
    immutable object prevents an evaluator from recording a velocity other
    than the one it actually executes.
    """

    bimanual: bool
    actuation_velocity: float = DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY
    attach_grasped_objects: bool = True
    detach_before_open: bool = True
    ownership_transfer_requires_receiver_detection: bool = True

    def __post_init__(self) -> None:
        for name in (
            "bimanual",
            "attach_grasped_objects",
            "detach_before_open",
            "ownership_transfer_requires_receiver_detection",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        velocity = float(self.actuation_velocity)
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError("gripper actuation velocity must be finite and positive")
        object.__setattr__(self, "actuation_velocity", velocity)

    @property
    def protocol_id(self) -> str:
        layout = "bimanual" if self.bimanual else "unimanual"
        velocity = _protocol_float_token(self.actuation_velocity)
        attach = int(self.attach_grasped_objects)
        detach = int(self.detach_before_open)
        return (
            f"rlbench-discrete-gripper-{layout}-velocity{velocity}"
            f"-attach{attach}-detach-before-open{detach}"
            f"-transfer-detect{int(self.ownership_transfer_requires_receiver_detection)}"
            "-retry-pending-close1-v3"
        )

    def extend_evaluation_protocol_id(self, base_protocol_id: str) -> str:
        """Include this physical gripper protocol in a stable evaluator ID."""

        base = str(base_protocol_id).strip()
        if not base:
            raise ValueError("base evaluation protocol ID must be non-empty")
        suffix = f"+{self.protocol_id}"
        return base if base.endswith(suffix) else f"{base}{suffix}"

    def metadata(self) -> dict[str, Any]:
        """Return JSON-stable metadata for an evaluation result."""

        return {
            "protocol_id": self.protocol_id,
            "action_mode": "BimanualDiscrete" if self.bimanual else "Discrete",
            "arm_layout": "bimanual" if self.bimanual else "unimanual",
            "actuation_velocity": self.actuation_velocity,
            "demonstration_actuation_velocity": (
                DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY
            ),
            "velocity_aligned_with_demonstrations": bool(
                math.isclose(
                    self.actuation_velocity,
                    DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            "attach_grasped_objects": self.attach_grasped_objects,
            "detach_before_open": self.detach_before_open,
            "ownership_transfer_requires_receiver_detection": (
                self.ownership_transfer_requires_receiver_detection
            ),
            "delayed_attachment_retry": True,
            "delayed_attachment_retry_scope": (
                "unresolved_open_to_closed_command_until_attach_or_reopen"
            ),
            "implementation": (
                "project_subclass_with_pending_close_attachment_retry_and_"
                "receiver_proximity_checked_bimanual_transfer"
                if self.bimanual and self.ownership_transfer_requires_receiver_detection
                else "project_subclass_with_pending_close_attachment_retry"
            ),
        }

    def make_action_mode(self) -> Any:
        """Create the configured RLBench mode without importing RLBench eagerly."""

        return _make_discrete_gripper_action_mode(self)


def make_discrete_gripper_action_mode(
    *,
    bimanual: bool,
    actuation_velocity: float = DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY,
    attach_grasped_objects: bool = True,
    detach_before_open: bool = True,
) -> Any:
    """Convenience factory for the aligned single- or dual-arm action mode."""

    return DiscreteGripperProtocol(
        bimanual=bimanual,
        actuation_velocity=actuation_velocity,
        attach_grasped_objects=attach_grasped_objects,
        detach_before_open=detach_before_open,
    ).make_action_mode()


def _make_discrete_gripper_action_mode(protocol: DiscreteGripperProtocol) -> Any:
    """Build a minimal velocity override around the pinned vendor classes."""

    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, Discrete

    if protocol.bimanual:

        class ProtocolBimanualDiscrete(BimanualDiscrete):
            dynamac_protocol = protocol

            @staticmethod
            def _receiver_detects(receiver: Any, obj: Any) -> bool:
                sensor = getattr(receiver, "_proximity_sensor", None)
                detects = getattr(sensor, "is_detected", None)
                if not callable(detects):
                    raise RuntimeError(
                        "RLBench 双臂附着转移要求夹爪 proximity 检测接口"
                    )
                return bool(detects(obj))

            def _grasp_with_ownership(
                self,
                receiver: Any,
                donor: Any,
                obj: Any,
            ) -> bool:
                donor_objects = donor.get_grasped_objects()
                if obj in donor_objects:
                    if (
                        protocol.ownership_transfer_requires_receiver_detection
                        and not self._receiver_detects(receiver, obj)
                    ):
                        return False
                    donor.release()
                return bool(receiver.grasp(obj))

            def _pending_close(self) -> dict[str, bool]:
                pending = getattr(self, "_dynamac_pending_close_attachment", None)
                if pending is None:
                    pending = {"right": False, "left": False}
                    self._dynamac_pending_close_attachment = pending
                return pending

            def _attempt_pending_close(
                self,
                scene: Any,
                *,
                arm_id: str,
            ) -> None:
                pending = self._pending_close()
                suppressed = set(
                    getattr(self, "_dynamac_attachment_suppressed_arms", ())
                )
                if (
                    not pending[arm_id]
                    or not self._attach_grasped_objects
                    or arm_id in suppressed
                ):
                    return
                if arm_id == "right":
                    receiver = scene.robot.right_gripper
                    donor = scene.robot.left_gripper
                else:
                    receiver = scene.robot.left_gripper
                    donor = scene.robot.right_gripper
                if receiver.get_grasped_objects():
                    pending[arm_id] = False
                    return
                attached = any(
                    self._grasp_with_ownership(receiver, donor, obj)
                    for obj in scene.task.get_graspable_objects()
                )
                if attached:
                    pending[arm_id] = False

            def action(self, scene: Any, action: Array) -> None:
                """Vendor-compatible discrete actuation with safe ownership transfer.

                The pinned vendor mode releases an object from the opposite arm
                before checking whether the closing receiver can detect it.  An
                asynchronous close can therefore drop a remotely held object.
                Keep every other vendor timing/attachment rule, but require the
                receiver's existing attachment proximity sensor to confirm an
                actual handover before releasing the donor.
                """

                expected_shape = self.action_shape(scene.robot)
                if np.shape(action) != expected_shape:
                    raise ValueError(
                        f"双臂夹爪动作维度应为 {expected_shape}，实际为 {np.shape(action)}"
                    )
                values = np.asarray(action, dtype=np.float64)
                if (
                    not np.all(np.isfinite(values))
                    or np.any(values < 0.0)
                    or np.any(values > 1.0)
                ):
                    raise ValueError(
                        "Gripper action expected to be finite within 0 and 1."
                    )

                right_current = float(
                    all(
                        value > 0.9
                        for value in scene.robot.right_gripper.get_open_amount()
                    )
                )
                left_current = float(
                    all(
                        value > 0.9
                        for value in scene.robot.left_gripper.get_open_amount()
                    )
                )
                right_action = float(values[0] > 0.5)
                left_action = float(values[1] > 0.5)
                changed = right_current != right_action or left_current != left_action
                pending = self._pending_close()
                if right_current != right_action:
                    pending["right"] = right_action == 0.0
                if left_current != left_action:
                    pending["left"] = left_action == 0.0
                if changed and not self._detach_before_open:
                    self._actuate(scene, values)

                if right_current != right_action:
                    if right_action == 1.0:
                        scene.robot.right_gripper.release()
                if left_current != left_action:
                    if left_action == 1.0:
                        scene.robot.left_gripper.release()

                # A Cartesian controller may close just before the arm enters
                # the attachment proximity volume.  Preserve that unresolved
                # close event and retry it on later closed-command cycles.
                # Only the arm whose close event is pending may claim an
                # object, preventing an already-closed donor from stealing it
                # back during the same bimanual call.
                self._attempt_pending_close(scene, arm_id="right")
                self._attempt_pending_close(scene, arm_id="left")

                if changed:
                    if self._detach_before_open:
                        self._actuate(scene, values)
                        self._attempt_pending_close(scene, arm_id="right")
                        self._attempt_pending_close(scene, arm_id="left")
                    if right_action == 1.0 or left_action == 1.0:
                        for _ in range(10):
                            scene.pyrep.step()
                            scene.task.step()

            def _actuate(self, scene: Any, action: Array) -> None:
                right_action = action[0]
                left_action = action[1]
                right_done = False
                left_done = False
                while not (right_done and left_done):
                    if not right_done:
                        right_done = scene.robot.right_gripper.actuate(
                            right_action,
                            velocity=protocol.actuation_velocity,
                        )
                    if not left_done:
                        left_done = scene.robot.left_gripper.actuate(
                            left_action,
                            velocity=protocol.actuation_velocity,
                        )
                    scene.pyrep.step()
                    scene.task.step()

        mode_class = ProtocolBimanualDiscrete
    else:

        class ProtocolDiscrete(Discrete):
            dynamac_protocol = protocol

            def _pending_close(self) -> bool:
                return bool(getattr(self, "_dynamac_pending_close_attachment", False))

            def _attempt_pending_close(self, scene: Any) -> None:
                suppressed = set(
                    getattr(self, "_dynamac_attachment_suppressed_arms", ())
                )
                if (
                    not self._pending_close()
                    or not self._attach_grasped_objects
                    or "single" in suppressed
                ):
                    return
                gripper = scene.robot.gripper
                if gripper.get_grasped_objects():
                    self._dynamac_pending_close_attachment = False
                    return
                attached = any(
                    bool(gripper.grasp(obj))
                    for obj in scene.task.get_graspable_objects()
                )
                if attached:
                    self._dynamac_pending_close_attachment = False

            def action(self, scene: Any, action: Array) -> None:
                if np.shape(action) != (1,):
                    raise ValueError(
                        f"单臂夹爪动作维度应为 (1,)，实际为 {np.shape(action)}"
                    )
                values = np.asarray(action, dtype=np.float64)
                if (
                    not np.all(np.isfinite(values))
                    or np.any(values < 0.0)
                    or np.any(values > 1.0)
                ):
                    raise ValueError(
                        "Gripper action expected to be finite within 0 and 1."
                    )
                gripper = scene.robot.gripper
                current = float(all(value > 0.9 for value in gripper.get_open_amount()))
                requested = float(values[0] > 0.5)
                changed = current != requested
                if changed:
                    self._dynamac_pending_close_attachment = requested == 0.0
                    if not self._detach_before_open:
                        self._actuate(scene, requested)
                    if requested == 1.0:
                        gripper.release()

                self._attempt_pending_close(scene)
                if changed:
                    if self._detach_before_open:
                        self._actuate(scene, requested)
                        self._attempt_pending_close(scene)
                    if requested == 1.0:
                        for _ in range(10):
                            scene.pyrep.step()
                            scene.task.step()

            def _actuate(self, scene: Any, action: Array | float) -> None:
                done = False
                while not done:
                    done = scene.robot.gripper.actuate(
                        action,
                        velocity=protocol.actuation_velocity,
                    )
                    scene.pyrep.step()
                    scene.task.step()

        mode_class = ProtocolDiscrete

    return mode_class(
        attach_grasped_objects=protocol.attach_grasped_objects,
        detach_before_open=protocol.detach_before_open,
    )


def execute_joint_target_control(
    scene: Any,
    arm_targets: tuple[tuple[Any, Array], ...],
    *,
    max_steps: int = 200,
    reached_atol: float = 0.01,
    stopped_atol: float = 0.001,
    invalid_action_error: type[Exception] = RuntimeError,
    error_message: str = "absolute end-effector IK execution timed out",
) -> Literal["reached", "stopped"]:
    """Drive one synchronized joint-target command to a terminal arm state.

    RLBench's public IK controller treats either reaching the target or ceasing
    to move (for example after contact) as the end of one high-level arm
    command.  The local controllers add a finite safety bound; exhausting that
    bound is an invalid action and must enter the evaluator's no-op fallback,
    rather than silently continuing with the accompanying gripper command.

    Task success is deliberately not inspected here.  ``MoveArmThenGripper``
    owns one combined action and would otherwise still execute its gripper
    command after this arm helper returned.  Episode termination is therefore
    evaluated once by ``TaskEnvironment.step`` after the combined action.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not arm_targets:
        raise ValueError("at least one arm target is required")
    if reached_atol <= 0.0 or stopped_atol <= 0.0:
        raise ValueError("joint tolerances must be positive")

    normalized = tuple(
        (arm, np.asarray(target, dtype=np.float64).copy())
        for arm, target in arm_targets
    )
    previous: tuple[Array, ...] | None = None
    for _ in range(max_steps):
        scene.step()
        current = tuple(
            np.asarray(arm.get_joint_positions(), dtype=np.float64)
            for arm, _ in normalized
        )
        if all(
            np.allclose(value, target, atol=reached_atol)
            for value, (_, target) in zip(current, normalized)
        ):
            return "reached"
        if previous is not None and all(
            np.allclose(value, prior, atol=stopped_atol)
            for value, prior in zip(current, previous)
        ):
            return "stopped"
        previous = current

    raise invalid_action_error(error_message)


def _arm_tip_pose(arm: Any) -> Array:
    get_tip = getattr(arm, "get_tip", None)
    if not callable(get_tip):
        raise RuntimeError("arm tip API is unavailable")
    get_pose = getattr(get_tip(), "get_pose", None)
    if not callable(get_pose):
        raise RuntimeError("arm tip pose API is unavailable")
    pose = np.asarray(get_pose(), dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise RuntimeError("arm tip returned an invalid pose")
    return pose


def _normalized_xyzw_quaternion(value: Any) -> Array:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    return quaternion / norm


def _shortest_xyzw_slerp(left: Any, right: Any, fraction: float) -> Array:
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("SLERP fraction must lie in [0, 1]")
    source = _normalized_xyzw_quaternion(left)
    target = _normalized_xyzw_quaternion(right)
    dot = float(np.dot(source, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalized_xyzw_quaternion(source + fraction * (target - source))
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return _normalized_xyzw_quaternion(
        math.sin((1.0 - fraction) * angle) / denominator * source
        + math.sin(fraction * angle) / denominator * target
    )


def _cartesian_continuation_subgoal(
    current_pose: Any,
    target_pose: Any,
    *,
    translation_step_m: float,
    rotation_step_rad: float,
) -> tuple[Array, float] | None:
    """Return one bounded SE(3) step when the full target is not local."""

    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    translation, rotation = end_effector_pose_distance(current, target)
    fractions = [1.0]
    if translation > translation_step_m:
        fractions.append(translation_step_m / translation)
    if rotation > rotation_step_rad:
        fractions.append(rotation_step_rad / rotation)
    fraction = float(min(fractions))
    if fraction >= 1.0 - 1.0e-12:
        return None
    subgoal = np.empty(7, dtype=np.float64)
    subgoal[:3] = current[:3] + fraction * (target[:3] - current[:3])
    subgoal[3:] = _shortest_xyzw_slerp(current[3:], target[3:], fraction)
    return subgoal, fraction


def _execute_prepared_ee_commands(
    scene: Any,
    commands: tuple[PreparedEECommand, ...],
    *,
    invalid_action_error: type[Exception],
    error_message: str,
    max_steps: int = 200,
    reached_atol: float = 0.01,
    stopped_atol: float = 0.001,
    step_counter: dict[str, int] | None = None,
    budget_exhaustion_is_stopped: bool = False,
) -> Literal["reached", "stopped"]:
    """Advance fully prepared commands on one shared physics clock."""

    if not commands:
        raise ValueError("at least one prepared EE command is required")
    if len({id(command.arm) for command in commands}) != len(commands):
        raise ValueError("prepared EE commands must target distinct arms")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    path_done = [command.mode != "planned_path" for command in commands]
    path_targets: list[Array | None] = [None] * len(commands)
    previous: tuple[Array, ...] | None = None
    for command in commands:
        if command.mode == "joint_target":
            command.arm.set_joint_target_positions(command.target_joints)

    for _ in range(max_steps):
        for index, command in enumerate(commands):
            if command.mode != "planned_path" or path_done[index]:
                continue
            try:
                path_done[index] = bool(command.path.step())
                get_action = getattr(
                    command.path, "get_executed_joint_position_action", None
                )
                executed = get_action() if callable(get_action) else None
            except Exception as exc:
                raise invalid_action_error(error_message) from exc
            if executed is not None:
                value = np.asarray(executed, dtype=np.float64)
                if value.ndim != 1 or not np.all(np.isfinite(value)):
                    raise invalid_action_error(error_message)
                path_targets[index] = value.copy()

        scene.step()
        if step_counter is not None:
            step_counter["steps"] = int(step_counter.get("steps", 0)) + 1
        current = tuple(
            np.asarray(command.arm.get_joint_positions(), dtype=np.float64)
            for command in commands
        )
        if any(value.ndim != 1 or not np.all(np.isfinite(value)) for value in current):
            raise invalid_action_error(error_message)
        targets = tuple(
            (
                command.target_joints
                if command.mode == "joint_target"
                else path_targets[index]
            )
            for index, command in enumerate(commands)
        )
        if all(path_done) and any(target is None for target in targets):
            raise invalid_action_error(error_message)
        if all(path_done) and all(
            np.allclose(value, target, atol=reached_atol)
            for value, target in zip(current, targets)
        ):
            return "reached"
        # Two nearly identical joint samples terminate one high-level action
        # as ``stopped``.  The caller classifies the measured Cartesian result
        # separately, so a valid but physically blocked command is observable
        # instead of being mistaken for an IK failure.
        if (
            all(path_done)
            and previous is not None
            and all(
                np.allclose(value, prior, atol=stopped_atol)
                for value, prior in zip(current, previous)
            )
        ):
            return "stopped"
        previous = current
    if budget_exhaustion_is_stopped:
        if step_counter is not None:
            step_counter["budget_exhausted"] = 1
        return "stopped"
    raise invalid_action_error(error_message)


def _set_ik_group_properties(
    arm: Any,
    *,
    resolution_method: str,
    max_iterations: int,
    damping: float,
    invalid_action_error: type[Exception],
    error_message: str,
) -> None:
    setter = getattr(arm, "set_ik_group_properties", None)
    if not callable(setter):
        raise invalid_action_error(error_message)
    try:
        setter(
            resolution_method=resolution_method,
            max_iterations=max_iterations,
            dls_damping=damping,
        )
    except Exception as exc:
        raise invalid_action_error(error_message) from exc


def _validated_continuous_ik_candidate(
    candidate: Any,
    *,
    current_joints: Array,
    limits: tuple[tuple[bool, float, float], ...],
    config: GlobalIKControllerConfig,
) -> tuple[Array | None, str | None]:
    joints, rejection = _validated_ik_candidate(
        candidate,
        current_joints=current_joints,
        limits=limits,
    )
    if joints is None:
        return None, rejection
    delta = joints - current_joints
    if float(np.max(np.abs(delta), initial=0.0)) > config.max_joint_delta_abs_rad:
        return None, "joint_delta_abs"
    if float(np.linalg.norm(delta)) > config.max_joint_delta_l2_rad:
        return None, "joint_delta_l2"
    return joints, None


def _record_selected_joint_delta(
    diagnostics: dict[str, Any], current_joints: Array, selected_joints: Array
) -> None:
    delta = selected_joints - current_joints
    diagnostics["selected_joint_delta_abs_max"] = max(
        float(diagnostics.get("selected_joint_delta_abs_max", 0.0)),
        float(np.max(np.abs(delta), initial=0.0)),
    )
    diagnostics["selected_joint_delta_l2_max"] = max(
        float(diagnostics.get("selected_joint_delta_l2_max", 0.0)),
        float(np.linalg.norm(delta)),
    )


def _prepare_trac_ik_distance_command(
    arm: Any,
    target: Array,
    *,
    current: Array,
    limits: tuple[tuple[bool, float, float], ...],
    config: GlobalIKControllerConfig,
    diagnostics: dict[str, Any],
    external_solver_factory: TracIKDistanceSolverFactory,
) -> PreparedEECommand | None:
    """Try one bounded TRAC-IK Distance candidate without simulator mutation."""

    _increment_ik_diagnostic(diagnostics, "trac_ik_distance_attempts")
    try:
        solver = external_solver_factory(arm)
    except Exception:
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_factory_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_exhaustions")
        return None
    if not callable(getattr(solver, "solve", None)):
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_factory_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_exhaustions")
        return None
    chain_source = getattr(solver, "chain_source", None)
    chain_schema = getattr(solver, "chain_schema", None)
    if isinstance(chain_source, str) and chain_source:
        sources = diagnostics.setdefault("trac_ik_distance_chain_sources", [])
        if chain_source not in sources:
            sources.append(chain_source)
    else:
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_chain_source_missing")
    if isinstance(chain_schema, str) and chain_schema:
        schemas = diagnostics.setdefault("trac_ik_distance_chain_schemas", [])
        if chain_schema not in schemas:
            schemas.append(chain_schema)
    else:
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_chain_schema_missing")

    started_at = time.perf_counter()
    try:
        result = solver.solve(target.copy())
    except Exception:
        _record_solver_duration_ms(diagnostics, "trac_ik_distance", started_at)
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_exhaustions")
        return None
    _record_solver_duration_ms(diagnostics, "trac_ik_distance", started_at)
    if result is None:
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_exhaustions")
        return None
    bounded_api_used = getattr(result, "bounded_cartesian_api_used", None)
    fk_translation_error = getattr(result, "fk_translation_error_m", None)
    fk_rotation_error = getattr(result, "fk_rotation_error_rad", None)
    try:
        fk_translation_error = float(fk_translation_error)
        fk_rotation_error = float(fk_rotation_error)
    except (TypeError, ValueError):
        fk_translation_error = math.nan
        fk_rotation_error = math.nan
    if bounded_api_used is True:
        _increment_ik_diagnostic(
            diagnostics, "trac_ik_distance_bounded_cartesian_api_uses"
        )
    elif bounded_api_used is False:
        _increment_ik_diagnostic(
            diagnostics, "trac_ik_distance_unbounded_cartesian_api_uses"
        )
    if (
        bounded_api_used is not True
        or not math.isfinite(fk_translation_error)
        or fk_translation_error < 0.0
        or not math.isfinite(fk_rotation_error)
        or fk_rotation_error < 0.0
    ):
        _increment_ik_diagnostic(
            diagnostics, "trac_ik_distance_result_metadata_missing"
        )
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_exhaustions")
        return None
    diagnostics["trac_ik_distance_fk_translation_error_m_max"] = max(
        float(diagnostics.get("trac_ik_distance_fk_translation_error_m_max", 0.0)),
        fk_translation_error,
    )
    diagnostics["trac_ik_distance_fk_rotation_error_rad_max"] = max(
        float(diagnostics.get("trac_ik_distance_fk_rotation_error_rad_max", 0.0)),
        fk_rotation_error,
    )
    reported_elapsed_ms = getattr(result, "elapsed_ms", None)
    if reported_elapsed_ms is not None:
        try:
            reported_elapsed_ms = float(reported_elapsed_ms)
        except (TypeError, ValueError):
            reported_elapsed_ms = None
        if reported_elapsed_ms is not None and math.isfinite(reported_elapsed_ms):
            diagnostics["trac_ik_distance_reported_solve_time_ms_total"] = float(
                diagnostics.get("trac_ik_distance_reported_solve_time_ms_total", 0.0)
            ) + max(0.0, reported_elapsed_ms)
            diagnostics["trac_ik_distance_reported_solve_time_ms_max"] = max(
                float(
                    diagnostics.get("trac_ik_distance_reported_solve_time_ms_max", 0.0)
                ),
                max(0.0, reported_elapsed_ms),
            )
    raw_candidate = getattr(result, "joints", result)
    candidate, rejection = _validated_continuous_ik_candidate(
        raw_candidate,
        current_joints=current,
        limits=limits,
        config=config,
    )
    if candidate is None:
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_candidate_rejections")
        _record_ik_candidate_rejection(diagnostics, str(rejection))
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_failures")
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_exhaustions")
        return None

    _increment_ik_diagnostic(diagnostics, "trac_ik_distance_successes")
    _increment_ik_diagnostic(diagnostics, "selected_via_trac_ik_distance")
    _record_selected_joint_delta(diagnostics, current, candidate)
    return PreparedEECommand(
        arm=arm,
        target_pose=target,
        mode="joint_target",
        target_joints=candidate,
    )


def _prepare_sampling_after_trac_command(
    arm: Any,
    target: Array,
    *,
    current: Array,
    limits: tuple[tuple[bool, float, float], ...],
    diagnostics: dict[str, Any],
    configuration_error: type[Exception],
    allow_collision_relaxed: bool = False,
    continuity_config: GlobalIKControllerConfig | None = None,
) -> PreparedEECommand | None:
    """Sample alternate IK branches, optionally relaxing collision filtering."""

    _increment_ik_diagnostic(diagnostics, "sampling_after_trac_attempts")
    for ignore_collisions in (False, True) if allow_collision_relaxed else (False,):
        if ignore_collisions:
            _increment_ik_diagnostic(diagnostics, "sampling_collision_relaxed_attempts")
        try:
            sampling_result = arm.solve_ik_via_sampling(
                target[:3],
                quaternion=target[3:],
                ignore_collisions=ignore_collisions,
                trials=IK_SAMPLING_TRIALS,
                max_configs=IK_SAMPLING_MAX_CONFIGS,
                max_time_ms=IK_SAMPLING_MAX_TIME_MS,
                relative_to=None,
            )
        except configuration_error:
            if ignore_collisions:
                _increment_ik_diagnostic(
                    diagnostics, "sampling_collision_relaxed_failures"
                )
            continue

        if isinstance(sampling_result, np.ndarray):
            if sampling_result.ndim == 2:
                candidates = tuple(sampling_result)
            elif sampling_result.ndim == 1 and sampling_result.size:
                candidates = (sampling_result,)
            elif sampling_result.size == 0:
                candidates = ()
            else:
                candidates = (sampling_result,)
        else:
            try:
                values = tuple(sampling_result)
            except TypeError:
                candidates = (sampling_result,)
            else:
                if values and all(np.asarray(value).ndim == 0 for value in values):
                    candidates = (values,)
                else:
                    candidates = values

        valid_candidates = []
        for raw_candidate in candidates:
            _increment_ik_diagnostic(diagnostics, "sampling_candidates_evaluated")
            if continuity_config is None:
                candidate, rejection = _validated_ik_candidate(
                    raw_candidate,
                    current_joints=current,
                    limits=limits,
                )
            else:
                candidate, rejection = _validated_continuous_ik_candidate(
                    raw_candidate,
                    current_joints=current,
                    limits=limits,
                    config=continuity_config,
                )
            if candidate is None:
                _record_ik_candidate_rejection(diagnostics, str(rejection))
                continue
            valid_candidates.append(candidate)

        if valid_candidates:
            selected = min(
                valid_candidates,
                key=lambda candidate: float(np.linalg.norm(candidate - current)),
            )
            _increment_ik_diagnostic(diagnostics, "sampling_fallback_successes")
            _increment_ik_diagnostic(diagnostics, "sampling_after_trac_successes")
            _increment_ik_diagnostic(diagnostics, "selected_via_sampling")
            if ignore_collisions:
                _increment_ik_diagnostic(
                    diagnostics, "sampling_collision_relaxed_successes"
                )
                _increment_ik_diagnostic(
                    diagnostics, "selected_via_collision_relaxed_sampling"
                )
            _record_selected_joint_delta(diagnostics, current, selected)
            return PreparedEECommand(
                arm=arm,
                target_pose=target,
                mode="joint_target",
                target_joints=selected,
            )
        if ignore_collisions:
            _increment_ik_diagnostic(diagnostics, "sampling_collision_relaxed_failures")

    _increment_ik_diagnostic(diagnostics, "sampling_fallback_failures")
    _increment_ik_diagnostic(diagnostics, "sampling_after_trac_failures")
    _increment_ik_diagnostic(diagnostics, "all_ik_exhaustions")
    return None


def _prepare_pseudo_trac_sampling_command(
    arm: Any,
    target: Array,
    *,
    config: GlobalIKControllerConfig,
    diagnostics: dict[str, Any],
    external_solver_factory: TracIKDistanceSolverFactory,
    ik_error: type[Exception],
    configuration_error: type[Exception],
    invalid_action_error: type[Exception],
    error_message: str,
) -> PreparedEECommand | None:
    """Try current-seeded pseudo-inverse, bounded TRAC, then V4 sampling."""

    current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    if current.ndim != 1 or not np.all(np.isfinite(current)):
        raise invalid_action_error(error_message)
    try:
        limits = _joint_limit_specification(arm, current)
    except (TypeError, ValueError) as exc:
        raise invalid_action_error(error_message) from exc

    pseudo_command = _prepare_pseudo_inverse_command(
        arm,
        target,
        current=current,
        limits=limits,
        diagnostics=diagnostics,
        ik_error=ik_error,
        invalid_action_error=invalid_action_error,
        error_message=error_message,
        continuity_config=(
            config if isinstance(config, Stage6IKControllerConfig) else None
        ),
    )
    if pseudo_command is not None:
        return pseudo_command

    trac_command = _prepare_trac_ik_distance_command(
        arm,
        target,
        current=current,
        limits=limits,
        config=config,
        diagnostics=diagnostics,
        external_solver_factory=external_solver_factory,
    )
    if trac_command is not None:
        return trac_command
    return _prepare_sampling_after_trac_command(
        arm,
        target,
        current=current,
        limits=limits,
        diagnostics=diagnostics,
        configuration_error=configuration_error,
        allow_collision_relaxed=(
            isinstance(config, Stage6IKControllerConfig)
            and config.allow_collision_relaxed_sampling
        ),
    )


def _prepare_pseudo_inverse_command(
    arm: Any,
    target: Array,
    *,
    current: Array,
    limits: tuple[tuple[bool, float, float], ...],
    diagnostics: dict[str, Any],
    ik_error: type[Exception],
    invalid_action_error: type[Exception],
    error_message: str,
    continuity_config: GlobalIKControllerConfig | None = None,
) -> PreparedEECommand | None:
    """Try the legacy current-seeded local pseudo-inverse once."""

    _set_ik_group_properties(
        arm,
        resolution_method=FROZEN_V4_IK_RESOLUTION_METHOD,
        max_iterations=FROZEN_V4_IK_MAX_ITERATIONS,
        damping=FROZEN_V4_IK_DAMPING,
        invalid_action_error=invalid_action_error,
        error_message=error_message,
    )
    _increment_ik_diagnostic(diagnostics, "pseudo_inverse_ik_attempts")
    try:
        raw_pseudo_candidate = arm.solve_ik_via_jacobian(
            target[:3], quaternion=target[3:], relative_to=None
        )
    except ik_error:
        _increment_ik_diagnostic(diagnostics, "pseudo_inverse_ik_failures")
        _increment_ik_diagnostic(diagnostics, "jacobian_failures")
    else:
        pseudo_candidate, rejection = (
            _validated_ik_candidate(
                raw_pseudo_candidate,
                current_joints=current,
                limits=limits,
            )
            if continuity_config is None
            else _validated_continuous_ik_candidate(
                raw_pseudo_candidate,
                current_joints=current,
                limits=limits,
                config=continuity_config,
            )
        )
        if pseudo_candidate is not None:
            _increment_ik_diagnostic(diagnostics, "pseudo_inverse_ik_successes")
            _increment_ik_diagnostic(diagnostics, "selected_via_pseudo_inverse")
            _increment_ik_diagnostic(diagnostics, "selected_via_jacobian")
            _record_selected_joint_delta(diagnostics, current, pseudo_candidate)
            return PreparedEECommand(
                arm=arm,
                target_pose=target,
                mode="joint_target",
                target_joints=pseudo_candidate,
            )
        _increment_ik_diagnostic(diagnostics, "pseudo_inverse_ik_candidate_rejections")
        _increment_ik_diagnostic(diagnostics, "jacobian_candidate_rejections")
        _record_ik_candidate_rejection(diagnostics, str(rejection))
    return None


def _prepare_stage6_hybrid_command(
    arm: Any,
    target: Array,
    *,
    config: Stage6IKControllerConfig,
    diagnostics: dict[str, Any],
    external_solver_factory: TracIKDistanceSolverFactory,
    ik_error: type[Exception],
    configuration_error: type[Exception],
    invalid_action_error: type[Exception],
    error_message: str,
    minimum_solver_tier: int = 0,
) -> PreparedEECommand | None:
    """Use bounded solver diversity at or above one physical-feedback tier.

    The tier is raised only after a mathematically valid command has failed to
    reduce the measured Cartesian residual for the same policy target.  A
    continuation subgoal may still use local pseudo/TRAC IK because it is a
    different, re-observed Cartesian command rather than the ineffective full
    target solution.
    """

    if isinstance(minimum_solver_tier, bool) or minimum_solver_tier not in range(
        _STAGE6_SOLVER_TIER_COUNT
    ):
        raise ValueError("minimum_solver_tier must be an integer in [0, 7]")

    current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    if current.ndim != 1 or not np.all(np.isfinite(current)):
        raise invalid_action_error(error_message)
    try:
        limits = _joint_limit_specification(arm, current)
    except (TypeError, ValueError) as exc:
        raise invalid_action_error(error_message) from exc

    if minimum_solver_tier <= 0:
        cached_command = _cached_stage6_joint_command(
            arm,
            target,
            current=current,
            limits=limits,
            config=config,
            diagnostics=diagnostics,
        )
        if cached_command is not None:
            return cached_command
        pseudo_command = _prepare_pseudo_inverse_command(
            arm,
            target,
            current=current,
            limits=limits,
            diagnostics=diagnostics,
            ik_error=ik_error,
            invalid_action_error=invalid_action_error,
            error_message=error_message,
            continuity_config=config,
        )
        if pseudo_command is not None:
            return pseudo_command
    if minimum_solver_tier <= 1:
        trac_command = _prepare_trac_ik_distance_command(
            arm,
            target,
            current=current,
            limits=limits,
            config=config,
            diagnostics=diagnostics,
            external_solver_factory=external_solver_factory,
        )
        if trac_command is not None:
            return trac_command
    if minimum_solver_tier <= 2:
        current_pose = _arm_tip_pose(arm)
        continuation_available = False
        for backoff in config.cartesian_continuation_backoff_factors:
            continuation = _cartesian_continuation_subgoal(
                current_pose,
                target,
                translation_step_m=(
                    backoff * config.cartesian_continuation_translation_step_m
                ),
                rotation_step_rad=(
                    backoff * config.cartesian_continuation_rotation_step_rad
                ),
            )
            if continuation is None:
                continue
            continuation_available = True
            subgoal, fraction = continuation
            _increment_ik_diagnostic(diagnostics, "cartesian_continuation_attempts")
            diagnostics["cartesian_continuation_fraction_min"] = min(
                float(diagnostics.get("cartesian_continuation_fraction_min", 1.0)),
                fraction,
            )
            continuation_command = _prepare_pseudo_inverse_command(
                arm,
                subgoal,
                current=current,
                limits=limits,
                diagnostics=diagnostics,
                ik_error=ik_error,
                invalid_action_error=invalid_action_error,
                error_message=error_message,
                continuity_config=config,
            )
            if continuation_command is None:
                continuation_command = _prepare_trac_ik_distance_command(
                    arm,
                    subgoal,
                    current=current,
                    limits=limits,
                    config=config,
                    diagnostics=diagnostics,
                    external_solver_factory=external_solver_factory,
                )
            if continuation_command is not None:
                _increment_ik_diagnostic(
                    diagnostics, "cartesian_continuation_successes"
                )
                _increment_ik_diagnostic(
                    diagnostics, "selected_via_cartesian_continuation"
                )
                return continuation_command
        if continuation_available:
            _increment_ik_diagnostic(diagnostics, "cartesian_continuation_failures")
    if minimum_solver_tier <= 3:
        return _prepare_sampling_after_trac_command(
            arm,
            target,
            current=current,
            limits=limits,
            diagnostics=diagnostics,
            configuration_error=configuration_error,
            allow_collision_relaxed=config.allow_collision_relaxed_sampling,
            # A global sampling branch may be necessary when every local IK
            # family is initially infeasible.  After a valid command has
            # merely stalled, however, an abrupt elbow/configuration switch
            # can sweep through task objects while leaving the EE target
            # nearly unchanged.  Physical-stall escalation therefore keeps
            # the same continuity gate as bounded TRAC; continuous path
            # fallbacks remain available after a rejection.
            continuity_config=config if minimum_solver_tier > 0 else None,
        )
    return None


def _prepare_collision_aware_path_command(
    arm: Any,
    target: Array,
    *,
    config: GlobalIKControllerConfig,
    diagnostics: dict[str, Any],
    configuration_path_error: type[Exception],
    invalid_action_error: type[Exception],
    path_algorithm: Any,
    error_message: str,
    allow_near_nonlinear: bool = False,
    minimum_path_tier: int = 0,
) -> PreparedEECommand | None:
    if isinstance(minimum_path_tier, bool) or minimum_path_tier not in range(4):
        raise ValueError("minimum_path_tier must be an integer in [0, 3]")
    _set_ik_group_properties(
        arm,
        resolution_method=FROZEN_V4_IK_RESOLUTION_METHOD,
        max_iterations=FROZEN_V4_IK_MAX_ITERATIONS,
        damping=FROZEN_V4_IK_DAMPING,
        invalid_action_error=invalid_action_error,
        error_message=error_message,
    )
    if isinstance(config, Stage6IKControllerConfig):
        linear_path = getattr(arm, "get_linear_path", None)
        if callable(linear_path) and minimum_path_tier <= 0:
            _increment_ik_diagnostic(diagnostics, "linear_path_attempts")
            try:
                path = linear_path(
                    target[:3],
                    quaternion=target[3:],
                    steps=config.linear_path_steps,
                    ignore_collisions=False,
                    relative_to=None,
                )
            except configuration_path_error:
                _increment_ik_diagnostic(
                    diagnostics, "linear_path_collision_aware_failures"
                )
            else:
                _increment_ik_diagnostic(
                    diagnostics, "linear_path_collision_aware_successes"
                )
                return PreparedEECommand(
                    arm=arm,
                    target_pose=target,
                    mode="planned_path",
                    path=path,
                )

        translation, _rotation = end_effector_pose_distance(_arm_tip_pose(arm), target)
        nonlinear_allowed = bool(
            allow_near_nonlinear or translation > config.far_translation_threshold_m
        )
        if nonlinear_allowed and minimum_path_tier <= 1:
            _increment_ik_diagnostic(diagnostics, "nonlinear_path_attempts")
            if translation > config.far_translation_threshold_m:
                _increment_ik_diagnostic(diagnostics, "far_target_planner_attempts")
            else:
                _increment_ik_diagnostic(
                    diagnostics, "near_target_nonlinear_planner_attempts"
                )
            try:
                path = arm.get_path(
                    target[:3],
                    quaternion=target[3:],
                    ignore_collisions=False,
                    trials=config.planner_trials,
                    max_configs=config.planner_max_configs,
                    max_time_ms=config.planner_max_time_ms,
                    trials_per_goal=config.planner_trials_per_goal,
                    algorithm=path_algorithm,
                    relative_to=None,
                )
            except configuration_path_error:
                _increment_ik_diagnostic(diagnostics, "nonlinear_path_failures")
                if translation > config.far_translation_threshold_m:
                    _increment_ik_diagnostic(diagnostics, "far_target_planner_failures")
            else:
                _increment_ik_diagnostic(diagnostics, "nonlinear_path_successes")
                if translation > config.far_translation_threshold_m:
                    _increment_ik_diagnostic(
                        diagnostics, "far_target_planner_successes"
                    )
                return PreparedEECommand(
                    arm=arm,
                    target_pose=target,
                    mode="planned_path",
                    path=path,
                )
        elif minimum_path_tier <= 1:
            _increment_ik_diagnostic(diagnostics, "near_target_nonlinear_planner_skips")

        # Collision relaxation remains a last resort for contact goals whose
        # desired endpoint itself touches a task object.  It must not pre-empt
        # a collision-aware detour after the same target has physically
        # stalled.
        if (
            callable(linear_path)
            and config.allow_collision_relaxed_linear_path
            and minimum_path_tier <= 2
        ):
            _increment_ik_diagnostic(
                diagnostics, "linear_path_collision_relaxed_attempts"
            )
            try:
                path = linear_path(
                    target[:3],
                    quaternion=target[3:],
                    steps=config.linear_path_steps,
                    ignore_collisions=True,
                    relative_to=None,
                )
            except configuration_path_error:
                _increment_ik_diagnostic(
                    diagnostics, "linear_path_collision_relaxed_failures"
                )
            else:
                _increment_ik_diagnostic(
                    diagnostics, "linear_path_collision_relaxed_successes"
                )
                return PreparedEECommand(
                    arm=arm,
                    target_pose=target,
                    mode="planned_path",
                    path=path,
                )

        # A contact-rich target can make every straight Cartesian path
        # infeasible even when a valid joint-space route still exists.  Keep
        # collision-relaxed nonlinear planning as the final, bounded solver
        # family after measured stall has exhausted all collision-aware and
        # linear alternatives.  This is a global executor fallback: it is not
        # keyed by task, state, relation, or failure label.
        if (
            nonlinear_allowed
            and config.allow_collision_relaxed_nonlinear_path
            and minimum_path_tier <= 3
        ):
            _increment_ik_diagnostic(
                diagnostics, "nonlinear_path_collision_relaxed_attempts"
            )
            try:
                path = arm.get_path(
                    target[:3],
                    quaternion=target[3:],
                    ignore_collisions=True,
                    trials=config.planner_trials,
                    max_configs=config.planner_max_configs,
                    max_time_ms=config.planner_max_time_ms,
                    trials_per_goal=config.planner_trials_per_goal,
                    algorithm=path_algorithm,
                    relative_to=None,
                )
            except configuration_path_error:
                _increment_ik_diagnostic(
                    diagnostics, "nonlinear_path_collision_relaxed_failures"
                )
            else:
                _increment_ik_diagnostic(
                    diagnostics, "nonlinear_path_collision_relaxed_successes"
                )
                return PreparedEECommand(
                    arm=arm,
                    target_pose=target,
                    mode="planned_path",
                    path=path,
                )
        return None

    if isinstance(config, Stage6IKControllerConfig):
        raise AssertionError("unreachable stage-six path branch")

    _increment_ik_diagnostic(diagnostics, "far_target_planner_attempts")
    try:
        path = arm.get_path(
            target[:3],
            quaternion=target[3:],
            ignore_collisions=False,
            trials=config.planner_trials,
            max_configs=config.planner_max_configs,
            max_time_ms=config.planner_max_time_ms,
            trials_per_goal=config.planner_trials_per_goal,
            algorithm=path_algorithm,
            relative_to=None,
        )
    except configuration_path_error as exc:
        _increment_ik_diagnostic(diagnostics, "far_target_planner_failures")
        raise invalid_action_error(error_message) from exc
    _increment_ik_diagnostic(diagnostics, "far_target_planner_successes")
    return PreparedEECommand(
        arm=arm,
        target_pose=target,
        mode="planned_path",
        path=path,
    )


def execute_global_ik_ee_control(
    scene: Any,
    arm_targets: tuple[tuple[Any, Array], ...],
    *,
    config: GlobalIKControllerConfig,
    diagnostics: dict[str, Any],
    external_solver_factory: TracIKDistanceSolverFactory,
    ik_error: type[Exception],
    configuration_error: type[Exception],
    configuration_path_error: type[Exception],
    invalid_action_error: type[Exception],
    path_algorithm: Any,
    error_message: str,
    max_steps: int = 200,
    budget_exhaustion_is_stopped: bool = False,
    use_stage6_hybrid_solver_order: bool = False,
    stage6_minimum_solver_tier: int = 0,
    prepared_commands_out: dict[int, PreparedEECommand] | None = None,
) -> Literal["reached", "stopped"]:
    """Run the global formal pseudo/TRAC/sampling/path controller."""

    if not arm_targets:
        raise ValueError("at least one arm target is required")
    if not callable(external_solver_factory):
        raise TypeError("external_solver_factory must be callable")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if stage6_minimum_solver_tier != 0 and not use_stage6_hybrid_solver_order:
        raise ValueError(
            "stage6_minimum_solver_tier requires the stage-six solver order"
        )
    if use_stage6_hybrid_solver_order and not isinstance(
        config, Stage6IKControllerConfig
    ):
        raise TypeError("the stage-six solver order requires its stage-six config")
    _ensure_global_ik_controller_diagnostics(diagnostics)
    normalized = tuple(
        (arm, np.asarray(target, dtype=np.float64).copy())
        for arm, target in arm_targets
    )
    translations = []
    for arm, target in normalized:
        translation, _rotation = _record_target_pose_diagnostic(
            diagnostics, _arm_tip_pose(arm), target
        )
        translations.append(translation)

    prepared: list[PreparedEECommand] = []
    try:
        for (arm, target), translation in zip(normalized, translations):
            if use_stage6_hybrid_solver_order:
                assert isinstance(config, Stage6IKControllerConfig)
                command = _prepare_stage6_hybrid_command(
                    arm,
                    target,
                    config=config,
                    diagnostics=diagnostics,
                    external_solver_factory=external_solver_factory,
                    ik_error=ik_error,
                    configuration_error=configuration_error,
                    invalid_action_error=invalid_action_error,
                    error_message=error_message,
                    minimum_solver_tier=stage6_minimum_solver_tier,
                )
            else:
                command = _prepare_pseudo_trac_sampling_command(
                    arm,
                    target,
                    config=config,
                    diagnostics=diagnostics,
                    external_solver_factory=external_solver_factory,
                    ik_error=ik_error,
                    configuration_error=configuration_error,
                    invalid_action_error=invalid_action_error,
                    error_message=error_message,
                )
            if command is None and (
                isinstance(config, Stage6IKControllerConfig)
                or translation > config.far_translation_threshold_m
            ):
                _increment_ik_diagnostic(diagnostics, "path_after_all_ik_exhaustion")
                command = _prepare_collision_aware_path_command(
                    arm,
                    target,
                    config=config,
                    diagnostics=diagnostics,
                    configuration_path_error=configuration_path_error,
                    invalid_action_error=invalid_action_error,
                    path_algorithm=path_algorithm,
                    error_message=error_message,
                    allow_near_nonlinear=bool(
                        use_stage6_hybrid_solver_order
                        and stage6_minimum_solver_tier >= 4
                    ),
                    minimum_path_tier=(
                        max(0, stage6_minimum_solver_tier - 4)
                        if use_stage6_hybrid_solver_order
                        else 0
                    ),
                )
            if command is None:
                raise invalid_action_error(error_message)
            prepared.append(command)
        _increment_ik_diagnostic(
            diagnostics, "commands_prepared_before_physics", len(prepared)
        )
        if prepared_commands_out is not None:
            prepared_commands_out.clear()
            prepared_commands_out.update(
                (id(command.arm), command) for command in prepared
            )
    finally:
        restore_error: Exception | None = None
        for arm, _target in normalized:
            _increment_ik_diagnostic(diagnostics, "ik_group_baseline_restore_attempts")
            try:
                _set_ik_group_properties(
                    arm,
                    resolution_method=FROZEN_V4_IK_RESOLUTION_METHOD,
                    max_iterations=FROZEN_V4_IK_MAX_ITERATIONS,
                    damping=FROZEN_V4_IK_DAMPING,
                    invalid_action_error=invalid_action_error,
                    error_message=error_message,
                )
            except invalid_action_error as exc:
                _increment_ik_diagnostic(
                    diagnostics, "ik_group_baseline_restore_failures"
                )
                restore_error = exc
        if restore_error is not None:
            raise invalid_action_error(error_message) from restore_error

    counter = {"steps": 0}
    try:
        return _execute_prepared_ee_commands(
            scene,
            tuple(prepared),
            invalid_action_error=invalid_action_error,
            error_message=error_message,
            max_steps=max_steps,
            step_counter=counter,
            budget_exhaustion_is_stopped=budget_exhaustion_is_stopped,
        )
    finally:
        release_error: Exception | None = None
        for command in prepared:
            if command.mode != "planned_path":
                continue
            try:
                if _release_configuration_path_motion_handle(command.path):
                    _increment_ik_diagnostic(
                        diagnostics, "planned_path_motion_handle_releases"
                    )
            except Exception as exc:
                _increment_ik_diagnostic(
                    diagnostics, "planned_path_motion_handle_release_failures"
                )
                release_error = exc
        steps = int(counter["steps"])
        if int(counter.get("budget_exhausted", 0)):
            _increment_ik_diagnostic(
                diagnostics, "controller_raw_physics_budget_exhaustions"
            )
        _increment_ik_diagnostic(diagnostics, "trac_ik_distance_controller_actions")
        diagnostics["trac_ik_distance_controller_raw_physics_steps"] = (
            int(diagnostics.get("trac_ik_distance_controller_raw_physics_steps", 0))
            + steps
        )
        diagnostics["trac_ik_distance_controller_raw_physics_steps_max"] = max(
            int(
                diagnostics.get("trac_ik_distance_controller_raw_physics_steps_max", 0)
            ),
            steps,
        )
        if release_error is not None:
            raise invalid_action_error(error_message) from release_error


def _cartesian_unresolved_targets(
    arm_targets: tuple[tuple[Any, Array], ...],
    *,
    config: Stage6IKControllerConfig,
    diagnostics: dict[str, Any],
) -> tuple[tuple[Any, Array], ...]:
    """Return arms whose physical tips still miss their commanded poses."""

    unresolved = []
    for arm, target in arm_targets:
        translation, rotation = end_effector_pose_distance(_arm_tip_pose(arm), target)
        _increment_ik_diagnostic(diagnostics, "cartesian_verification_checks")
        diagnostics["cartesian_post_translation_error_m_max"] = max(
            float(diagnostics.get("cartesian_post_translation_error_m_max", 0.0)),
            translation,
        )
        diagnostics["cartesian_post_rotation_error_rad_max"] = max(
            float(diagnostics.get("cartesian_post_rotation_error_rad_max", 0.0)),
            rotation,
        )
        if (
            translation > config.physical_completion_translation_tolerance_m
            or rotation > config.physical_completion_rotation_tolerance_rad
        ):
            _increment_ik_diagnostic(diagnostics, "cartesian_verification_failures")
            unresolved.append((arm, target))
    return tuple(unresolved)


def _cartesian_residual_score(
    arm: Any,
    target: Array,
    *,
    config: Stage6IKControllerConfig,
) -> float:
    translation, rotation = end_effector_pose_distance(_arm_tip_pose(arm), target)
    return math.hypot(
        translation / config.physical_completion_translation_tolerance_m,
        rotation / config.physical_completion_rotation_tolerance_rad,
    )


def _cartesian_targets_within_tolerance(
    arm_targets: tuple[tuple[Any, Array], ...],
    *,
    translation_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> bool:
    return all(
        translation <= translation_tolerance_m and rotation <= rotation_tolerance_rad
        for translation, rotation in (
            end_effector_pose_distance(_arm_tip_pose(arm), target)
            for arm, target in arm_targets
        )
    )


def _cartesian_vector_made_progress(
    before_scores: dict[int, float],
    after_scores: dict[int, float],
) -> tuple[bool, bool]:
    """Apply a non-compensatory progress test to a multi-arm command.

    A bimanual servo step is useful when at least one unresolved arm moves
    closer and no unresolved arm moves farther away.  Requiring every arm to
    improve on every raw step incorrectly rejects valid asynchronous motor
    responses; summing both residuals would instead allow one arm's large
    improvement to hide regression of the other.  This Pareto test permits a
    stationary arm but never compensates one arm's regression with another's
    progress.
    """

    if before_scores.keys() != after_scores.keys():
        raise ValueError("Cartesian progress scores must describe the same arms")
    numerical_tolerance = 1.0e-6
    no_arm_regressed = all(
        after_scores[key] <= before_scores[key] + numerical_tolerance
        for key in before_scores
    )
    improved = [
        key
        for key in before_scores
        if after_scores[key] < before_scores[key] - numerical_tolerance
    ]
    return no_arm_regressed and bool(improved), len(improved) < len(before_scores)


def execute_stage6_ik_ee_control(
    scene: Any,
    arm_targets: tuple[tuple[Any, Array], ...],
    *,
    config: Stage6IKControllerConfig,
    diagnostics: dict[str, Any],
    external_solver_factory: TracIKDistanceSolverFactory,
    ik_error: type[Exception],
    configuration_error: type[Exception],
    configuration_path_error: type[Exception],
    invalid_action_error: type[Exception],
    path_algorithm: Any,
    error_message: str,
    max_steps: int = 200,
    per_arm_status_out: (
        dict[int, Literal["reached", "progressed", "stopped"]] | None
    ) = None,
) -> Literal["reached", "progressed", "stopped"]:
    """Reach one policy target with bounded Cartesian continuation.

    A bounded continuation command is an executor-internal decomposition of
    the *same* absolute policy target, not a completed policy action.  The
    physical tip is therefore re-observed and the local IK chain is rebuilt
    after every segment until the original target is reached, motion stalls,
    or the fixed segment budget is exhausted.  This keeps the policy commit
    aligned with the pose that was actually requested.  Collision-aware
    sampling/path search is attempted first; a bounded collision-relaxed
    fallback is available for contact-rich RLBench poses.  Nonlinear
    RRTConnect is used immediately for far targets and, for near targets, only
    after measured physical stall has exhausted the local solver families.
    """

    if not arm_targets:
        raise ValueError("at least one arm target is required")
    normalized = tuple(
        (arm, np.asarray(target, dtype=np.float64).copy())
        for arm, target in arm_targets
    )
    original_scores = {
        id(arm): _cartesian_residual_score(arm, target, config=config)
        for arm, target in normalized
    }

    def finish(
        overall_status: Literal["reached", "progressed", "stopped"],
    ) -> Literal["reached", "progressed", "stopped"]:
        if per_arm_status_out is not None:
            per_arm_status_out.clear()
            for arm, target in normalized:
                translation, rotation = end_effector_pose_distance(
                    _arm_tip_pose(arm), target
                )
                if (
                    translation <= config.control_acceptance_translation_tolerance_m
                    and rotation <= config.control_acceptance_rotation_tolerance_rad
                ):
                    status: Literal["reached", "progressed", "stopped"] = "reached"
                elif (
                    _cartesian_residual_score(arm, target, config=config)
                    < original_scores[id(arm)] - 1.0e-6
                ):
                    status = "progressed"
                else:
                    status = "stopped"
                per_arm_status_out[id(arm)] = status
        return overall_status

    # The previous bounded prefix may already have placed every tip inside the
    # accepted Cartesian envelope.  Accept that observation before rebuilding
    # or stepping any IK controller; issuing a fresh joint command here can
    # kick a physically completed target back outside a sub-millimetre
    # tolerance and create an arbitrary reached/progressed oscillation.
    if _cartesian_targets_within_tolerance(
        normalized,
        translation_tolerance_m=config.control_acceptance_translation_tolerance_m,
        rotation_tolerance_rad=config.control_acceptance_rotation_tolerance_rad,
    ):
        for arm, _target in normalized:
            _clear_stage6_same_target_solver_tier(arm)
        _increment_ik_diagnostic(
            diagnostics,
            "cartesian_pre_execution_control_accepts",
        )
        return finish("reached")
    before_scores = dict(original_scores)
    made_any_progress = False
    # Restore the lowest tier shared by the unchanged multi-arm transaction.
    # If any arm has a new target its tier is zero, so the whole synchronized
    # command safely restarts before later common stalls advance every lane.
    minimum_solver_tier = min(
        _stage6_same_target_solver_tier(
            arm,
            target,
            diagnostics=diagnostics,
        )
        for arm, target in normalized
    )
    raw_steps_at_entry = int(
        diagnostics.get("trac_ik_distance_controller_raw_physics_steps", 0)
    )
    for segment_index in range(config.cartesian_continuation_max_segments):
        raw_steps_before_segment = int(
            diagnostics.get("trac_ik_distance_controller_raw_physics_steps", 0)
        )
        remaining_raw_steps = config.cartesian_continuation_max_raw_physics_steps - (
            raw_steps_before_segment - raw_steps_at_entry
        )
        if remaining_raw_steps <= 0:
            _increment_ik_diagnostic(
                diagnostics,
                "cartesian_policy_action_physics_budget_exhaustions",
            )
            return finish("progressed" if made_any_progress else "stopped")
        _increment_ik_diagnostic(diagnostics, "cartesian_direct_goal_attempts")
        prepared_commands: dict[int, PreparedEECommand] = {}
        try:
            direct_status = execute_global_ik_ee_control(
                scene,
                normalized,
                config=config,
                diagnostics=diagnostics,
                external_solver_factory=external_solver_factory,
                ik_error=ik_error,
                configuration_error=configuration_error,
                configuration_path_error=configuration_path_error,
                invalid_action_error=invalid_action_error,
                path_algorithm=path_algorithm,
                error_message=error_message,
                max_steps=min(max_steps, remaining_raw_steps),
                # A Stage-6 policy action owns only the remaining raw-physics
                # budget.  Reaching that budget is therefore a normal bounded
                # return to the closed-loop observer, not an IK failure.
                budget_exhaustion_is_stopped=True,
                use_stage6_hybrid_solver_order=True,
                stage6_minimum_solver_tier=minimum_solver_tier,
                prepared_commands_out=prepared_commands,
            )
        except invalid_action_error:
            if minimum_solver_tier == 0 and not made_any_progress:
                raise
            _increment_ik_diagnostic(
                diagnostics, "physical_stall_solver_escalation_exhaustions"
            )
            _increment_ik_diagnostic(
                diagnostics,
                "cartesian_multi_pass_solver_exhaustions_after_progress",
            )
            return finish("progressed" if made_any_progress else "stopped")

        if segment_index > 0:
            _increment_ik_diagnostic(
                diagnostics, "cartesian_multi_pass_followup_passes"
            )
        unresolved_after_direct = _cartesian_unresolved_targets(
            normalized,
            config=config,
            diagnostics=diagnostics,
        )
        if not unresolved_after_direct:
            for arm, _target in normalized:
                _clear_stage6_same_target_solver_tier(arm)
            for arm, target in normalized:
                command = prepared_commands.get(id(arm))
                if command is not None:
                    _store_stage6_joint_command(command, target)
                    if (
                        command.mode == "joint_target"
                        and _same_stage6_cartesian_target(command.target_pose, target)
                    ):
                        _increment_ik_diagnostic(
                            diagnostics, "same_target_joint_cache_stores"
                        )
            _increment_ik_diagnostic(diagnostics, "cartesian_direct_goal_reaches")
            if segment_index > 0:
                _increment_ik_diagnostic(
                    diagnostics, "cartesian_multi_pass_goals_completed"
                )
            return finish("reached")
        if _cartesian_targets_within_tolerance(
            normalized,
            translation_tolerance_m=(config.control_acceptance_translation_tolerance_m),
            rotation_tolerance_rad=config.control_acceptance_rotation_tolerance_rad,
        ):
            for arm, _target in normalized:
                _clear_stage6_same_target_solver_tier(arm)
            for arm, target in normalized:
                command = prepared_commands.get(id(arm))
                if command is not None:
                    _store_stage6_joint_command(command, target)
                    if (
                        command.mode == "joint_target"
                        and _same_stage6_cartesian_target(command.target_pose, target)
                    ):
                        _increment_ik_diagnostic(
                            diagnostics, "same_target_joint_cache_stores"
                        )
            _increment_ik_diagnostic(
                diagnostics, "cartesian_direct_goal_control_accepts"
            )
            if segment_index > 0:
                _increment_ik_diagnostic(
                    diagnostics, "cartesian_multi_pass_goals_completed"
                )
            return finish("reached")

        direct_scores = {
            id(arm): _cartesian_residual_score(arm, target, config=config)
            for arm, target in normalized
        }
        direct_progressed, direct_partial = _cartesian_vector_made_progress(
            before_scores,
            direct_scores,
        )
        raw_steps_used = (
            int(diagnostics.get("trac_ik_distance_controller_raw_physics_steps", 0))
            - raw_steps_at_entry
        )
        raw_physics_budget_exhausted = bool(
            raw_steps_used >= config.cartesian_continuation_max_raw_physics_steps
        )
        if direct_progressed:
            made_any_progress = True
            for arm, target in normalized:
                key = id(arm)
                if direct_scores[key] >= before_scores[key] - 1.0e-6:
                    continue
                command = prepared_commands.get(key)
                if command is None:
                    continue
                _store_stage6_joint_command(command, target)
                if command.mode == "joint_target" and _same_stage6_cartesian_target(
                    command.target_pose, target
                ):
                    _increment_ik_diagnostic(
                        diagnostics, "same_target_joint_cache_stores"
                    )
            _increment_ik_diagnostic(
                diagnostics, "cartesian_direct_goal_progress_accepts"
            )
            _increment_ik_diagnostic(
                diagnostics, "cartesian_goal_directed_progress_accepts"
            )
            if direct_partial:
                _increment_ik_diagnostic(
                    diagnostics, "cartesian_partial_arm_progress_accepts"
                )
            # A fallback tier that produced real motion has changed the local
            # kinematic/contact state.  Re-observe it on the next policy cycle
            # and give the current-seeded local controller a fresh chance;
            # tier persistence is only evidence about a *stalled* command,
            # not a permanent ban after the world has improved.
            if minimum_solver_tier > 0:
                for arm, target in normalized:
                    _store_stage6_same_target_solver_tier(arm, target, 0)
                _increment_ik_diagnostic(
                    diagnostics, "same_target_solver_tier_resets_after_progress"
                )
            # Once an alternate family has produced useful motion, return the
            # measured progress to the policy.  The next policy cycle then
            # re-observes the world before deciding whether the same target is
            # still required; this avoids chaining unobserved solver changes.
            if raw_physics_budget_exhausted:
                _increment_ik_diagnostic(
                    diagnostics,
                    "cartesian_policy_action_physics_budget_exhaustions",
                )
                diagnostics["cartesian_policy_action_raw_physics_steps_max"] = max(
                    int(
                        diagnostics.get(
                            "cartesian_policy_action_raw_physics_steps_max", 0
                        )
                    ),
                    raw_steps_used,
                )
                return finish("progressed")
            if minimum_solver_tier > 0:
                return finish("progressed")
            before_scores = direct_scores
            continue

        _increment_ik_diagnostic(diagnostics, "cartesian_direct_goal_stalls")
        for arm, target in normalized:
            if _invalidate_stage6_joint_command(arm, target):
                _increment_ik_diagnostic(
                    diagnostics, "same_target_joint_cache_invalidations"
                )
        _increment_ik_diagnostic(
            diagnostics,
            (
                "stopped_with_cartesian_residual"
                if direct_status == "stopped"
                else "reached_joint_target_with_cartesian_residual"
            ),
        )
        if raw_physics_budget_exhausted:
            if (
                not direct_progressed
                and minimum_solver_tier < _STAGE6_SOLVER_TIER_COUNT - 1
            ):
                minimum_solver_tier += 1
                for arm, target in normalized:
                    _store_stage6_same_target_solver_tier(
                        arm, target, minimum_solver_tier
                    )
                _increment_ik_diagnostic(
                    diagnostics, "physical_stall_solver_escalations"
                )
                _increment_ik_diagnostic(
                    diagnostics, "same_target_cross_cycle_solver_escalations"
                )
                diagnostics["physical_stall_solver_tier_max"] = max(
                    int(diagnostics.get("physical_stall_solver_tier_max", 0)),
                    minimum_solver_tier,
                )
            _increment_ik_diagnostic(
                diagnostics,
                "cartesian_policy_action_physics_budget_exhaustions",
            )
            diagnostics["cartesian_policy_action_raw_physics_steps_max"] = max(
                int(
                    diagnostics.get("cartesian_policy_action_raw_physics_steps_max", 0)
                ),
                raw_steps_used,
            )
            return finish("progressed" if made_any_progress else "stopped")
        if made_any_progress:
            return finish("progressed")
        if minimum_solver_tier < _STAGE6_SOLVER_TIER_COUNT - 1:
            minimum_solver_tier += 1
            for arm, target in normalized:
                _store_stage6_same_target_solver_tier(arm, target, minimum_solver_tier)
            _increment_ik_diagnostic(diagnostics, "physical_stall_solver_escalations")
            diagnostics["physical_stall_solver_tier_max"] = max(
                int(diagnostics.get("physical_stall_solver_tier_max", 0)),
                minimum_solver_tier,
            )
            continue
        _increment_ik_diagnostic(
            diagnostics, "physical_stall_solver_escalation_exhaustions"
        )
        return finish("progressed" if made_any_progress else "stopped")

    _increment_ik_diagnostic(diagnostics, "cartesian_multi_pass_limit_exhaustions")
    final_scores = {
        id(arm): _cartesian_residual_score(arm, target, config=config)
        for arm, target in normalized
    }
    overall_progressed, _partial = _cartesian_vector_made_progress(
        original_scores,
        final_scores,
    )
    return finish("progressed" if overall_progressed else "stopped")


def xyzw_to_wxyz(pose: Array) -> Array:
    """Compatibility export for the canonical, audited convention crossing."""

    return _xyzw_to_wxyz(pose)


def wxyz_to_xyzw(pose: Array) -> Array:
    """Compatibility export for the canonical, audited convention crossing."""

    return _wxyz_to_xyzw(pose)


def unimanual_observation_from_rlbench(observation: Any, task: str | TaskSpec) -> Any:
    """Build a core observation without importing RLBench or PyRep."""

    from essay2608.policy import DynaMACObservation

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if spec.bimanual:
        raise ValueError(f"{spec.task_name} is bimanual")
    return DynaMACObservation(
        ee_pose=xyzw_to_wxyz(observation.gripper_pose),
        frames=spec.extract_pose_chunks(observation.task_low_dim_state),
    )


def bimanual_observations_from_rlbench(
    observation: Any,
    task: str | TaskSpec,
) -> tuple[Any, Any]:
    """Build synchronized left/right core observations from one simulator snapshot."""

    from essay2608.policy import DynaMACObservation

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if not spec.bimanual:
        raise ValueError(f"{spec.task_name} is unimanual")
    frames = spec.extract_pose_chunks(observation.task_low_dim_state)
    return (
        DynaMACObservation(
            ee_pose=xyzw_to_wxyz(observation.left.gripper_pose),
            frames={name: value.copy() for name, value in frames.items()},
        ),
        DynaMACObservation(
            ee_pose=xyzw_to_wxyz(observation.right.gripper_pose),
            frames={name: value.copy() for name, value in frames.items()},
        ),
    )


def _gripper_to_rlbench(value: Array | float) -> float:
    scalar = float(np.asarray(value, dtype=np.float64).reshape(-1).mean())
    if not np.isfinite(scalar):
        raise ValueError("gripper prediction must be finite")
    # TAPAS stores 2 * gripper_open - 1.  Zero is the deterministic midpoint.
    return float(scalar > 0.0)


def unimanual_action_to_rlbench(
    action: Any, *, ignore_collisions: bool = False
) -> Array:
    """Return the fork's 9D ``pose, gripper, ignore`` action."""

    pose = wxyz_to_xyzw(np.asarray(action.pose, dtype=np.float64))
    return np.concatenate(
        (pose, [_gripper_to_rlbench(action.gripper), float(ignore_collisions)])
    )


@dataclass(frozen=True)
class ArmActionOffset:
    """Explicitly inferred diagnostic intervention, never an author default."""

    arm: Literal["left", "right"]
    translation: tuple[float, float, float]

    def apply(self, left_pose: Array, right_pose: Array) -> tuple[Array, Array]:
        left = np.asarray(left_pose, dtype=np.float64).copy()
        right = np.asarray(right_pose, dtype=np.float64).copy()
        target = left if self.arm == "left" else right
        target[:3] += np.asarray(self.translation, dtype=np.float64)
        return left, right


def bimanual_action_to_rlbench(
    action: Any,
    *,
    left_ignore_collisions: bool = False,
    right_ignore_collisions: bool = False,
    offset: ArmActionOffset | None = None,
) -> Array:
    """Return the author's right-first 18D bimanual action layout.

    The core action object is left/right named, while the RLBench fork expects
    ``[right pose7, right grip, right ignore, left pose7, left grip, left ignore]``.
    """

    left_pose = np.asarray(action.left.pose, dtype=np.float64)
    right_pose = np.asarray(action.right.pose, dtype=np.float64)
    if offset is not None:
        left_pose, right_pose = offset.apply(left_pose, right_pose)
    right = np.concatenate(
        (
            wxyz_to_xyzw(right_pose),
            [
                _gripper_to_rlbench(action.right.gripper),
                float(right_ignore_collisions),
            ],
        )
    )
    left = np.concatenate(
        (
            wxyz_to_xyzw(left_pose),
            [
                _gripper_to_rlbench(action.left.gripper),
                float(left_ignore_collisions),
            ],
        )
    )
    result = np.concatenate((right, left))
    if result.shape != (18,):
        raise AssertionError(f"invalid RLBench bimanual action shape: {result.shape}")
    return result


def pose_execution_error(command_wxyz: Array, observed_xyzw: Array) -> dict[str, float]:
    command = np.asarray(command_wxyz, dtype=np.float64)
    observed = xyzw_to_wxyz(np.asarray(observed_xyzw, dtype=np.float64))
    position = float(np.linalg.norm(command[:3] - observed[:3]))
    q_command = command[3:7] / np.linalg.norm(command[3:7])
    q_observed = observed[3:7] / np.linalg.norm(observed[3:7])
    dot = float(np.clip(abs(np.dot(q_command, q_observed)), 0.0, 1.0))
    rotation = float(2.0 * math.acos(dot))
    return {"position_m": position, "rotation_rad": rotation}


def _task_low_dim_state(task: Any) -> Array:
    """Read task state directly, without triggering observation recorders."""

    get_state = getattr(task, "get_low_dim_state", None)
    if not callable(get_state):
        raise RuntimeError("RLBench task.get_low_dim_state() is unavailable")
    return unwrap_task_low_dim_state(get_state())


def _instance_reference_snapshot(task: Any) -> dict[str, tuple[Any, ...]]:
    """Capture registries that ``init_episode`` is allowed to replace.

    Comparing object identity (rather than equality) catches a silent rebuild
    of success conditions even when the replacement conditions happen to have
    equal values.
    """

    attributes = (
        "_success_conditions",
        "_fail_conditions",
        "_graspable_objects",
    )
    return {name: tuple(getattr(task, name, ())) for name in attributes}


def _same_instance_references(
    before: dict[str, tuple[Any, ...]],
    after: dict[str, tuple[Any, ...]],
) -> bool:
    if before.keys() != after.keys():
        return False
    return all(
        len(before[name]) == len(after[name])
        and all(left is right for left, right in zip(before[name], after[name]))
        for name in before
    )


def _source_seed(episode_seed: int, variation: int, attempt: int) -> int:
    """Return the frozen, scenario-independent RNG seed for one source try."""

    if attempt < 1:
        raise ValueError("source seed attempt must be positive")
    if attempt == 1:
        return int(episode_seed)
    return int(
        (episode_seed * 1_000_003 + variation * 9_176 + attempt * 104_729) % (2**32 - 1)
    )


def _source_selection_failure_diagnostic(attempts: list[dict[str, Any]]) -> str:
    """Summarize rejected A candidates without persisting a partial plan."""

    counts: dict[str, int] = {}
    for row in attempts:
        rejection_type = str(row.get("rejection_type") or "unknown")
        counts[rejection_type] = counts.get(rejection_type, 0) + 1
    last_reason = (
        str(attempts[-1].get("rejection_reason") or "unknown")
        if attempts
        else "no candidate evidence"
    )
    return f"rejection_counts={counts}; last_reason={last_reason}"


def _goal_sampling_failure_diagnostic(attempts: list[dict[str, Any]]) -> str:
    """Summarize rejected B candidates without persisting a partial plan."""

    counts: dict[str, int] = {}
    for row in attempts:
        outcome = str(row.get("outcome") or "unknown")
        counts[outcome] = counts.get(outcome, 0) + 1
    last_reason = (
        str(attempts[-1].get("reason") or "unknown")
        if attempts
        else "no candidate evidence"
    )
    return f"rejection_counts={counts}; last_reason={last_reason}"


def _robot_numeric_state(robot: Any) -> dict[str, Any]:
    """Capture a stable numeric robot state without configuration-tree access."""

    component_names = (
        ("right_arm", "left_arm", "right_gripper", "left_gripper")
        if bool(getattr(robot, "is_bimanual", False))
        else ("arm", "gripper")
    )
    rows = []
    for name in component_names:
        component = getattr(robot, name, None)
        if component is None:
            raise RuntimeError(f"RLBench robot component {name!r} is unavailable")
        getters = {
            "joint_positions": "get_joint_positions",
            "joint_target_positions": "get_joint_target_positions",
            "joint_velocities": "get_joint_velocities",
        }
        row = {"name": name}
        for field, getter_name in getters.items():
            getter = getattr(component, getter_name, None)
            if not callable(getter):
                raise RuntimeError(
                    f"RLBench robot component {name!r} cannot report {field}"
                )
            values = np.asarray(getter(), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"RLBench robot {name!r} {field} is not finite")
            row[field] = values.tolist()
        get_tip = getattr(component, "get_tip", None)
        if callable(get_tip):
            tip = get_tip()
            get_pose = getattr(tip, "get_pose", None)
            if not callable(get_pose):
                raise RuntimeError(f"RLBench robot {name!r} tip pose is unavailable")
            pose = np.asarray(get_pose(), dtype=np.float64)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise RuntimeError(f"RLBench robot {name!r} tip pose is invalid")
            row["tip_pose"] = pose.tolist()
        rows.append(row)
    return {"components": rows}


def _robot_numeric_state_audit(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compare robot state around offline waypoint validation."""

    before_rows = {row["name"]: row for row in before.get("components", [])}
    after_rows = {row["name"]: row for row in after.get("components", [])}
    names_matched = before_rows.keys() == after_rows.keys()
    max_joint_position = 0.0
    max_joint_target = 0.0
    max_joint_velocity = 0.0
    max_tip_translation = 0.0
    max_tip_rotation = 0.0
    structure_matched = names_matched
    for name in sorted(before_rows.keys() | after_rows.keys()):
        left = before_rows.get(name)
        right = after_rows.get(name)
        if left is None or right is None or left.keys() != right.keys():
            structure_matched = False
            continue
        for field, target in (
            ("joint_positions", "position"),
            ("joint_target_positions", "target"),
            ("joint_velocities", "velocity"),
        ):
            a = np.asarray(left[field], dtype=np.float64)
            b = np.asarray(right[field], dtype=np.float64)
            if a.shape != b.shape:
                structure_matched = False
                continue
            error = float(np.max(np.abs(a - b))) if a.size else 0.0
            if target == "position":
                max_joint_position = max(max_joint_position, error)
            elif target == "target":
                max_joint_target = max(max_joint_target, error)
            else:
                max_joint_velocity = max(max_joint_velocity, error)
        if "tip_pose" in left:
            metrics = _root_motion_metrics(
                np.asarray(left["tip_pose"], dtype=np.float64),
                np.asarray(right["tip_pose"], dtype=np.float64),
            )
            max_tip_translation = max(
                max_tip_translation,
                float(metrics["planned_root_translation_m"]),
            )
            max_tip_rotation = max(
                max_tip_rotation,
                float(metrics["planned_root_rotation_rad"]),
            )
    passed = bool(
        structure_matched
        and max_joint_position <= SOURCE_RECONSTRUCTION_JOINT_TOLERANCE
        and max_joint_target <= SOURCE_RECONSTRUCTION_JOINT_TOLERANCE
        and max_joint_velocity <= SOURCE_RECONSTRUCTION_SCALAR_TOLERANCE
        and max_tip_translation <= SOURCE_RECONSTRUCTION_TRANSLATION_TOLERANCE_M
        and max_tip_rotation <= SOURCE_RECONSTRUCTION_ROTATION_TOLERANCE_RAD
    )
    return {
        "structure_matched": structure_matched,
        "max_joint_position_error": max_joint_position,
        "max_joint_target_position_error": max_joint_target,
        "max_joint_velocity_error": max_joint_velocity,
        "max_tip_translation_error_m": max_tip_translation,
        "max_tip_rotation_error_rad": max_tip_rotation,
        "joint_tolerance": SOURCE_RECONSTRUCTION_JOINT_TOLERANCE,
        "velocity_tolerance": SOURCE_RECONSTRUCTION_SCALAR_TOLERANCE,
        "tip_translation_tolerance_m": SOURCE_RECONSTRUCTION_TRANSLATION_TOLERANCE_M,
        "tip_rotation_tolerance_rad": SOURCE_RECONSTRUCTION_ROTATION_TOLERANCE_RAD,
        "passed": passed,
    }


_CONDITION_STRUCTURAL_FIELDS = {
    "rlbench.backend.conditions.ColorCondition": (
        ("shape", "success_rgb"),
        (),
    ),
    "rlbench.backend.conditions.JointCondition": (
        ("_joint", "_original_pos", "_pos"),
        (),
    ),
    "rlbench.backend.conditions.DetectedCondition": (
        ("_obj", "_detector", "_negated"),
        (),
    ),
    "rlbench.backend.conditions.NothingGrasped": (("_gripper",), ()),
    "rlbench.backend.conditions.GraspedCondition": (
        ("_gripper", "_object_handle"),
        (),
    ),
    "rlbench.backend.conditions.DetectedSeveralCondition": (
        ("_objects", "_detector", "_number_needed"),
        (),
    ),
    "rlbench.backend.conditions.EmptyCondition": (("_container",), ()),
    "rlbench.backend.conditions.FollowCondition": (
        (
            "_obj",
            "_ponts",
            "_relative_to",
            "_delta_limit",
            "_start_after_first",
        ),
        ("_index", "_strikes"),
    ),
    "rlbench.backend.conditions.ConditionSet": (
        ("_conditions", "_order_matters", "_simultaneously_met"),
        ("_current_condition_index",),
    ),
    "rlbench.backend.conditions.OrConditions": (
        ("_conditions",),
        ("_current_condition_index",),
    ),
}


def _semantic_structural_field_value(name: str, item: Any, *, depth: int) -> Any:
    """Serialize a structural field with stable cross-process object names."""

    if name.endswith("handle") and isinstance(item, int):
        try:
            from pyrep.objects.object import Object

            return {"object_name": str(Object.get_object_name(item))}
        except Exception as error:
            raise RuntimeError(
                "could not resolve condition object handle to a stable name"
            ) from error
    return _semantic_reference_value(item, depth=depth + 1)


def _condition_structural_value(value: Any, *, depth: int) -> Any | None:
    """Serialize known Condition structure while excluding execution progress.

    RLBench's ``OrConditions.reset`` creates an otherwise unused
    ``_current_condition_index`` field, while ``ConditionSet`` and
    ``FollowCondition`` maintain real runtime progress.  Those counters are
    not task semantics.  Every structural field is explicit and any unknown
    Condition subclass or unexpected field fails closed.
    """

    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    schema = _CONDITION_STRUCTURAL_FIELDS.get(value_type)
    condition_base = any(
        cls.__module__ == "rlbench.backend.conditions"
        and cls.__qualname__ == "Condition"
        for cls in type(value).__mro__
    )
    if schema is None:
        if condition_base:
            condition_base_class = next(
                cls
                for cls in type(value).__mro__
                if cls.__module__ == "rlbench.backend.conditions"
                and cls.__qualname__ == "Condition"
            )
            if getattr(type(value), "reset", None) is not getattr(
                condition_base_class,
                "reset",
                None,
            ):
                raise RuntimeError(
                    f"custom condition {value_type} has unmodeled runtime state"
                )
            attributes = getattr(value, "__dict__", None)
            if not isinstance(attributes, dict):
                raise RuntimeError(
                    f"custom condition {value_type} has no inspectable fields"
                )
            if any(callable(item) for item in attributes.values()):
                raise RuntimeError(
                    f"custom condition {value_type} contains callable state"
                )
            return {
                "type": value_type,
                "structural_fields": {
                    str(name): _semantic_structural_field_value(
                        str(name),
                        item,
                        depth=depth,
                    )
                    for name, item in sorted(attributes.items())
                },
                "excluded_runtime_progress_fields": [],
            }
        return None
    required_fields, runtime_fields = schema
    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, dict):
        raise RuntimeError(f"condition {value_type} has no inspectable fields")
    present = set(attributes)
    required = set(required_fields)
    allowed = required | set(runtime_fields)
    if not required.issubset(present) or not present.issubset(allowed):
        raise RuntimeError(f"condition {value_type} fields do not match its schema")
    return {
        "type": value_type,
        "structural_fields": {
            name: _semantic_structural_field_value(
                name,
                attributes[name],
                depth=depth,
            )
            for name in required_fields
        },
        # Record the schema, not which counters happen to have been materialized
        # yet; OrConditions.reset() lazily creates its unused progress field.
        "excluded_runtime_progress_fields": list(runtime_fields),
    }


def _semantic_reference_value(value: Any, *, depth: int = 0) -> Any:
    """Return a deterministic, cross-scene description of task semantics."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 16:
        raise RuntimeError("task semantic structure exceeds the supported depth")
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {
            str(key): _semantic_reference_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_reference_value(item, depth=depth + 1) for item in value]
    get_handle = getattr(value, "get_handle", None)
    get_name = getattr(value, "get_name", None)
    if callable(get_handle):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": str(get_name()) if callable(get_name) else None,
        }
    condition_value = _condition_structural_value(value, depth=depth)
    if condition_value is not None:
        return condition_value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                str(name): _semantic_structural_field_value(
                    str(name),
                    item,
                    depth=depth,
                )
                for name, item in sorted(attributes.items())
                if not callable(item) and name not in {"pyrep", "robot", "_robot"}
            },
        }
    raise RuntimeError(
        "unsupported task semantic value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _task_semantic_signature(task: Any) -> dict[str, Any]:
    """Describe success/failure/grasp semantics without Python identities."""

    return {
        "schema": TASK_SEMANTIC_SIGNATURE_SCHEMA,
        "task_class": f"{type(task).__module__}.{type(task).__qualname__}",
        "success_conditions": [
            _semantic_reference_value(value)
            for value in getattr(task, "_success_conditions", ())
        ],
        "fail_conditions": [
            _semantic_reference_value(value)
            for value in getattr(task, "_fail_conditions", ())
        ],
        "graspable_objects": [
            _semantic_reference_value(value)
            for value in getattr(task, "_graspable_objects", ())
        ],
    }


def _task_tree_relative_state(task: Any) -> list[dict[str, Any]]:
    """Capture the complete task tree in both world and boundary-root frames.

    A task's model base is commonly an ancestor of ``boundary_root`` rather
    than part of the subtree moved by RLBench's workspace sampler.  Persisting
    both frames plus explicit subtree membership lets source replays compare
    every object in world coordinates, while a planned boundary-root move can
    compare only moved descendants in boundary-root coordinates.  Simulator
    handles are used neither as persisted identities nor across launches.
    """

    base = getattr(task, "get_base", lambda: None)()
    root = task.boundary_root()
    get_objects = getattr(base, "get_objects_in_tree", None)
    get_root_objects = getattr(root, "get_objects_in_tree", None)
    if not callable(get_objects) or not callable(get_root_objects):
        raise RuntimeError("RLBench task tree enumeration API is unavailable")
    objects = list(get_objects(exclude_base=False))
    root_objects = list(get_root_objects(exclude_base=False))

    def stable_key(value: Any) -> tuple[str, str]:
        get_name = getattr(value, "get_name", None)
        get_type = getattr(value, "get_type", None)
        if not callable(get_name) or not callable(get_type):
            raise RuntimeError("RLBench task object identity API is unavailable")
        return str(get_name()), str(get_type())

    root_subtree_keys = {stable_key(value) for value in root_objects}
    rows = []
    for value in objects:
        get_name = getattr(value, "get_name", None)
        get_type = getattr(value, "get_type", None)
        get_pose = getattr(value, "get_pose", None)
        get_parent = getattr(value, "get_parent", None)
        if not callable(get_name) or not callable(get_type) or not callable(get_pose):
            raise RuntimeError("RLBench task object introspection API is unavailable")
        parent = get_parent() if callable(get_parent) else None
        parent_name_fn = getattr(parent, "get_name", None)
        row = {
            "name": str(get_name()),
            "type": str(get_type()),
            "parent": (str(parent_name_fn()) if callable(parent_name_fn) else None),
            "in_boundary_root_subtree": stable_key(value) in root_subtree_keys,
            "world_pose": np.asarray(
                get_pose(),
                dtype=np.float64,
            ).tolist(),
            "pose_relative_to_boundary_root": np.asarray(
                get_pose(relative_to=root),
                dtype=np.float64,
            ).tolist(),
        }
        get_joint_position = getattr(value, "get_joint_position", None)
        if callable(get_joint_position):
            row["joint_position"] = float(get_joint_position())
        rows.append(row)
    rows.sort(key=lambda row: (row["name"], row["type"], row["parent"] or ""))
    if len({(row["name"], row["type"]) for row in rows}) != len(rows):
        raise RuntimeError("task tree contains duplicate stable name/type identities")
    return rows


def _task_tree_velocity_summary(task: Any) -> dict[str, Any]:
    """Record finite task-object velocities as diagnostics, never identity."""

    base = getattr(task, "get_base", lambda: None)()
    get_objects = getattr(base, "get_objects_in_tree", None)
    if not callable(get_objects):
        raise RuntimeError("RLBench task tree enumeration API is unavailable")
    rows = []
    for value in get_objects(exclude_base=False):
        get_velocity = getattr(value, "get_velocity", None)
        get_name = getattr(value, "get_name", None)
        get_type = getattr(value, "get_type", None)
        if (
            not callable(get_velocity)
            or not callable(get_name)
            or not callable(get_type)
        ):
            raise RuntimeError("RLBench task object velocity API is unavailable")
        linear, angular = get_velocity()
        linear = np.asarray(linear, dtype=np.float64)
        angular = np.asarray(angular, dtype=np.float64)
        finite = bool(
            linear.shape == (3,)
            and angular.shape == (3,)
            and np.all(np.isfinite(linear))
            and np.all(np.isfinite(angular))
        )
        rows.append(
            {
                "name": str(get_name()),
                "type": str(get_type()),
                "finite": finite,
                "linear_speed_m_s": (float(np.linalg.norm(linear)) if finite else None),
                "angular_speed_rad_s": (
                    float(np.linalg.norm(angular)) if finite else None
                ),
            }
        )
    rows.sort(key=lambda row: (row["name"], row["type"]))
    all_finite = bool(rows) and all(row["finite"] for row in rows)
    return {
        "schema": "rlbench-task-tree-velocity-summary-v1",
        "compared_for_identity": False,
        "diagnostic_only": True,
        "object_count": len(rows),
        "all_finite": all_finite,
        "max_linear_speed_m_s": max(
            (
                float(row["linear_speed_m_s"])
                for row in rows
                if row["linear_speed_m_s"] is not None
            ),
            default=0.0,
        ),
        "max_angular_speed_rad_s": max(
            (
                float(row["angular_speed_rad_s"])
                for row in rows
                if row["angular_speed_rad_s"] is not None
            ),
            default=0.0,
        ),
    }


def _compare_task_tree_relative_state(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    boundary_root_may_move: bool = False,
    translation_tolerance_m: float = LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    rotation_tolerance_rad: float = LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
    joint_tolerance: float = LOW_DIM_STATE_ROUNDTRIP_ATOL,
) -> dict[str, Any]:
    """Compare task topology and state in the physically correct frame.

    Source replay (A-to-A) uses world poses for every object.  Planned A-to-B
    root motion uses root-relative poses for boundary-root descendants and
    world poses for objects outside that subtree.  Parent relationships,
    subtree membership, and joint values must remain unchanged in both modes.
    """

    expected_map = {(row["name"], row["type"]): row for row in expected}
    actual_map = {(row["name"], row["type"]): row for row in actual}
    topology_matched = expected_map.keys() == actual_map.keys()
    rows = []
    all_matched = topology_matched
    for key in sorted(expected_map.keys() | actual_map.keys()):
        left = expected_map.get(key)
        right = actual_map.get(key)
        if left is None or right is None:
            rows.append(
                {
                    "name": key[0],
                    "type": key[1],
                    "matched": False,
                    "missing_from_expected": left is None,
                    "missing_from_actual": right is None,
                }
            )
            all_matched = False
            continue
        parent_matched = left.get("parent") == right.get("parent")
        subtree_membership_matched = left.get("in_boundary_root_subtree") == right.get(
            "in_boundary_root_subtree"
        )
        world_pose_metrics = _root_motion_metrics(
            np.asarray(left["world_pose"], dtype=np.float64),
            np.asarray(right["world_pose"], dtype=np.float64),
        )
        relative_pose_metrics = _root_motion_metrics(
            np.asarray(left["pose_relative_to_boundary_root"], dtype=np.float64),
            np.asarray(right["pose_relative_to_boundary_root"], dtype=np.float64),
        )
        compare_relative = bool(
            boundary_root_may_move and left.get("in_boundary_root_subtree") is True
        )
        pose_metrics = relative_pose_metrics if compare_relative else world_pose_metrics
        pose_matched = bool(
            pose_metrics["planned_root_translation_m"] <= translation_tolerance_m
            and pose_metrics["planned_root_rotation_rad"] <= rotation_tolerance_rad
        )
        joint_left = left.get("joint_position")
        joint_right = right.get("joint_position")
        joint_matched = (
            joint_left is None
            and joint_right is None
            or joint_left is not None
            and joint_right is not None
            and abs(float(joint_left) - float(joint_right)) <= joint_tolerance
        )
        matched = (
            parent_matched
            and subtree_membership_matched
            and pose_matched
            and joint_matched
        )
        all_matched = all_matched and matched
        rows.append(
            {
                "name": key[0],
                "type": key[1],
                "matched": matched,
                "parent_matched": parent_matched,
                "subtree_membership_matched": subtree_membership_matched,
                "in_boundary_root_subtree": left.get("in_boundary_root_subtree"),
                "pose_comparison_frame": (
                    "boundary_root" if compare_relative else "world"
                ),
                "translation_error_m": pose_metrics["planned_root_translation_m"],
                "rotation_error_rad": pose_metrics["planned_root_rotation_rad"],
                "world_translation_error_m": world_pose_metrics[
                    "planned_root_translation_m"
                ],
                "world_rotation_error_rad": world_pose_metrics[
                    "planned_root_rotation_rad"
                ],
                "boundary_root_relative_translation_error_m": (
                    relative_pose_metrics["planned_root_translation_m"]
                ),
                "boundary_root_relative_rotation_error_rad": (
                    relative_pose_metrics["planned_root_rotation_rad"]
                ),
                "joint_position_error": (
                    None
                    if joint_left is None or joint_right is None
                    else abs(float(joint_left) - float(joint_right))
                ),
            }
        )
    return {
        "matched": bool(all_matched),
        "topology_matched": bool(topology_matched),
        "comparison_mode": (
            "boundary_root_subtree_relative_else_world"
            if boundary_root_may_move
            else "all_objects_world"
        ),
        "expected_object_count": len(expected),
        "actual_object_count": len(actual),
        "translation_tolerance_m": float(translation_tolerance_m),
        "rotation_tolerance_rad": float(rotation_tolerance_rad),
        "joint_tolerance": float(joint_tolerance),
        "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
        "objects": rows,
    }


def _compact_task_tree_comparison(
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Keep cross-initialization evidence bounded across retry attempts."""

    rows = comparison.get("objects", [])
    comparable = [row for row in rows if "translation_error_m" in row]
    joint_errors = [
        float(row["joint_position_error"])
        for row in comparable
        if row.get("joint_position_error") is not None
    ]
    return {
        "matched": comparison.get("matched"),
        "topology_matched": comparison.get("topology_matched"),
        "comparison_mode": comparison.get("comparison_mode"),
        "expected_object_count": comparison.get("expected_object_count"),
        "actual_object_count": comparison.get("actual_object_count"),
        "translation_tolerance_m": comparison.get("translation_tolerance_m"),
        "rotation_tolerance_rad": comparison.get("rotation_tolerance_rad"),
        "joint_tolerance": comparison.get("joint_tolerance"),
        "quaternion_rotation_metric": comparison.get("quaternion_rotation_metric"),
        "all_parents_matched": all(
            row.get("parent_matched") is True for row in comparable
        ),
        "all_subtree_memberships_matched": all(
            row.get("subtree_membership_matched") is True for row in comparable
        ),
        "max_translation_error_m": max(
            (float(row["translation_error_m"]) for row in comparable),
            default=0.0,
        ),
        "max_rotation_error_rad": max(
            (float(row["rotation_error_rad"]) for row in comparable),
            default=0.0,
        ),
        "max_joint_position_error": max(joint_errors, default=0.0),
    }


def _reference_key(value: Any) -> tuple[str, int] | tuple[str, None]:
    """Return a stable identity key for a PyRep object wrapper or ``None``."""

    if value is None:
        return ("none", None)
    get_handle = getattr(value, "get_handle", None)
    if callable(get_handle):
        return ("handle", int(get_handle()))
    return ("python_id", id(value))


def _robot_collision_arms(robot: Any) -> tuple[tuple[str, Any], ...]:
    """Return named arm collision collections for either robot layout."""

    names = (
        ("right_arm", "left_arm")
        if bool(getattr(robot, "is_bimanual", False))
        else ("arm",)
    )
    arms = []
    for name in names:
        arm = getattr(robot, name, None)
        if arm is None or not callable(getattr(arm, "check_arm_collision", None)):
            raise RuntimeError(f"RLBench robot arm {name!r} cannot check collisions")
        arms.append((name, arm))
    return tuple(arms)


def _arm_collision_collection_member_handles(arm: Any) -> frozenset[int]:
    """Read the current members of a pinned PyRep arm collision collection.

    ``Arm.check_arm_collision(None)`` checks its collection against *all other*
    collidable scene objects. PyRep exposes that check but not the collection's
    current members, so use the already-bound CoppeliaSim API to reproduce the
    exclusion exactly. Current membership matters because a grasped tool can
    become a descendant of the arm and therefore a collection member.
    """

    collection = getattr(arm, "_collision_collection", None)
    if collection is None:
        raise RuntimeError("PyRep arm collision collection handle is unavailable")

    from pyrep.backend import sim

    count = sim.ffi.new("int *")
    values = sim.lib.simGetCollectionObjects(int(collection), count)
    if values == sim.ffi.NULL:
        raise RuntimeError("CoppeliaSim could not enumerate arm collision collection")
    try:
        member_count = int(count[0])
        if member_count < 0:
            raise RuntimeError("CoppeliaSim returned an invalid collection size")
        return frozenset(int(values[index]) for index in range(member_count))
    finally:
        sim.simReleaseBuffer(sim.ffi.cast("char *", values))


def _pyrep_shape_object_type() -> Any:
    """Load the optional PyRep shape enum only when a simulator is active."""

    from pyrep.const import ObjectType

    return ObjectType.SHAPE


def _robot_external_collision_pairs(
    scene: Any,
    robot: Any,
) -> tuple[tuple[str, int, str], ...]:
    """Return concrete arm-to-external-object collision pairs.

    The pair granularity is the named arm collision collection and one current
    external collidable scene shape. Enumerating shapes and invoking
    ``Arm.check_arm_collision(object)`` preserves the vendor collision geometry
    while exposing which pair caused the aggregate boolean. Objects inside an
    arm's current collection are excluded to match CoppeliaSim's
    ``sim_handle_all`` ("all other") semantics; this also handles grasped tools
    without task-specific exclusions.
    """

    pyrep = getattr(scene, "pyrep", None)
    get_objects = getattr(pyrep, "get_objects_in_tree", None)
    if not callable(get_objects):
        raise RuntimeError("PyRep scene object enumeration API is unavailable")

    collidable_objects = []
    for value in get_objects(object_type=_pyrep_shape_object_type()):
        get_handle = getattr(value, "get_handle", None)
        get_name = getattr(value, "get_name", None)
        is_collidable = getattr(value, "is_collidable", None)
        if not all(callable(item) for item in (get_handle, get_name, is_collidable)):
            raise RuntimeError("PyRep scene object collision API is unavailable")
        if bool(is_collidable()):
            collidable_objects.append((int(get_handle()), str(get_name()), value))
    collidable_objects.sort(key=lambda item: item[:2])

    pairs = []
    for arm_name, arm in _robot_collision_arms(robot):
        members = _arm_collision_collection_member_handles(arm)
        arm_pairs = []
        for handle, object_name, value in collidable_objects:
            if handle in members:
                continue
            if bool(arm.check_arm_collision(value)):
                arm_pairs.append((arm_name, handle, object_name))

        # Fail closed if object-level enumeration ever misses a collision type
        # represented by the pinned aggregate PyRep API.
        aggregate_collision = bool(arm.check_arm_collision())
        if aggregate_collision != bool(arm_pairs):
            raise RuntimeError(
                f"could not resolve {arm_name} aggregate collision into "
                "external collidable object pairs"
            )
        pairs.extend(arm_pairs)
    return tuple(sorted(pairs))


def _collision_pair_records(
    pairs: tuple[tuple[str, int, str], ...],
) -> list[dict[str, Any]]:
    """Convert collision-pair identities into JSON-stable evidence rows."""

    return [
        {
            "arm": arm_name,
            "external_object_handle": object_handle,
            "external_object_name": object_name,
        }
        for arm_name, object_handle, object_name in pairs
    ]


def _stable_collision_pair_records(
    pairs: tuple[tuple[str, int, str], ...],
) -> list[dict[str, Any]]:
    """Cross-launch collision evidence using names, never simulator handles."""

    return [
        {"arm": arm_name, "external_object_name": object_name}
        for arm_name, object_name in sorted(
            {(arm_name, object_name) for arm_name, _handle, object_name in pairs}
        )
    ]


def _grasp_state_snapshot(task: Any, robot: Any) -> dict[str, Any]:
    """Capture gripper membership and object parents without changing them."""

    grippers = (
        (
            ("right_gripper", robot.right_gripper),
            ("left_gripper", robot.left_gripper),
        )
        if bool(getattr(robot, "is_bimanual", False))
        else (("gripper", robot.gripper),)
    )
    tracked_objects = list(getattr(task, "_graspable_objects", ()))
    gripper_rows = []
    for name, gripper in grippers:
        get_grasped = getattr(gripper, "get_grasped_objects", None)
        if not callable(get_grasped):
            raise RuntimeError(f"RLBench {name} cannot report grasped objects")
        grasped = tuple(get_grasped())
        tracked_objects.extend(grasped)
        gripper_rows.append(
            {
                "name": name,
                "gripper": gripper,
                "grasped": grasped,
                "old_parent_keys": tuple(
                    _reference_key(parent)
                    for parent in getattr(gripper, "_old_parents", ())
                ),
            }
        )

    unique_objects = []
    seen = set()
    for value in tracked_objects:
        key = _reference_key(value)
        if key in seen:
            continue
        seen.add(key)
        unique_objects.append(value)
    parent_rows = []
    for value in unique_objects:
        get_parent = getattr(value, "get_parent", None)
        if callable(get_parent):
            parent_rows.append((value, _reference_key(get_parent())))
    return {
        "grippers": tuple(gripper_rows),
        "parents": tuple(parent_rows),
    }


def _same_grasp_state(before: dict[str, Any], task: Any, robot: Any) -> bool:
    """Compare grasp lists and object parents after a sampling transaction."""

    after = _grasp_state_snapshot(task, robot)
    before_grippers = before["grippers"]
    after_grippers = after["grippers"]
    if len(before_grippers) != len(after_grippers):
        return False
    for left, right in zip(before_grippers, after_grippers):
        if left["name"] != right["name"] or left["gripper"] is not right["gripper"]:
            return False
        if len(left["grasped"]) != len(right["grasped"]):
            return False
        if any(
            prior is not current
            for prior, current in zip(left["grasped"], right["grasped"])
        ):
            return False
        if left["old_parent_keys"] != right["old_parent_keys"]:
            return False
    if len(before["parents"]) != len(after["parents"]):
        return False
    return all(
        prior_object is current_object and prior_parent == current_parent
        for (prior_object, prior_parent), (current_object, current_parent) in zip(
            before["parents"], after["parents"]
        )
    )


def _stable_grasp_state(task: Any, robot: Any) -> dict[str, Any]:
    """Serialize grasp membership and parentage without process-local handles."""

    grippers = (
        (("right_gripper", robot.right_gripper), ("left_gripper", robot.left_gripper))
        if bool(getattr(robot, "is_bimanual", False))
        else (("gripper", robot.gripper),)
    )
    rows = []
    tracked = list(getattr(task, "_graspable_objects", ()))
    for name, gripper in grippers:
        grasped = list(gripper.get_grasped_objects())
        tracked.extend(grasped)
        rows.append(
            {
                "gripper": name,
                "grasped_objects": sorted(str(value.get_name()) for value in grasped),
            }
        )
    parents = []
    seen = set()
    for value in tracked:
        name = str(value.get_name())
        if name in seen:
            continue
        seen.add(name)
        parent = value.get_parent()
        parent_name = (
            str(parent.get_name())
            if parent is not None and callable(getattr(parent, "get_name", None))
            else None
        )
        parents.append({"object": name, "parent": parent_name})
    return {
        "grippers": rows,
        "tracked_object_parents": sorted(parents, key=lambda row: row["object"]),
    }


def _workspace_boundary_contains_root(scene: Any, root: Any) -> bool:
    """Return whether SpawnBoundary recorded a successful root placement."""

    workspace = getattr(scene, "_workspace_boundary", None)
    boundaries = getattr(workspace, "_boundaries", None)
    if not isinstance(boundaries, list):
        raise RuntimeError("RLBench workspace placement audit is unavailable")
    get_root_handle = getattr(root, "get_handle", None)
    if not callable(get_root_handle):
        raise RuntimeError("RLBench boundary-root handle is unavailable")
    root_handle = int(get_root_handle())
    return any(
        any(
            callable(getattr(value, "get_handle", None))
            and int(value.get_handle()) == root_handle
            for value in getattr(boundary, "_contained_objects", ())
        )
        for boundary in boundaries
    )


def _workspace_boundary_accepts_current_root(scene: Any, root: Any) -> bool:
    """Check an explicitly commanded root pose against the RLBench workspace.

    ``SpawnBoundary.sample`` chooses both the pose and placement.  V4 LiftTray
    instead preregisters B as a source-relative transform, so staging needs the
    same bounding-box admission check without asking the sampler to replace B.
    This mirrors the pinned fork's ``BoundaryObject.add`` geometry and does not
    mutate its private contained-object ledger.
    """

    workspace = getattr(scene, "_workspace_boundary", None)
    boundaries = getattr(workspace, "_boundaries", None)
    if not isinstance(boundaries, list) or not boundaries:
        raise RuntimeError("RLBench workspace placement audit is unavailable")
    is_model = getattr(root, "is_model", None)
    if not callable(is_model):
        raise RuntimeError("RLBench boundary-root model query is unavailable")
    bounds_getter = (
        getattr(root, "get_model_bounding_box", None)
        if bool(is_model())
        else getattr(root, "get_bounding_box", None)
    )
    if not callable(bounds_getter):
        raise RuntimeError("RLBench boundary-root bounding box is unavailable")
    bounds = np.asarray(bounds_getter(), dtype=np.float64)
    if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
        raise RuntimeError("RLBench boundary-root bounding box is invalid")
    points = np.asarray(
        [
            [x, y, z]
            for x in (bounds[0], bounds[1])
            for y in (bounds[2], bounds[3])
            for z in (bounds[4], bounds[5])
        ],
        dtype=np.float64,
    )
    for entry in boundaries:
        boundary_object = getattr(entry, "_boundary", None)
        boundary_box = getattr(entry, "_boundary_bbox", None)
        if boundary_object is None or boundary_box is None:
            raise RuntimeError("RLBench workspace boundary internals are unavailable")
        orientation = np.asarray(
            root.get_orientation(boundary_object),
            dtype=np.float64,
        )
        position = np.asarray(
            root.get_position(boundary_object),
            dtype=np.float64,
        )
        if (
            orientation.shape != (3,)
            or position.shape != (3,)
            or not np.all(np.isfinite(orientation))
            or not np.all(np.isfinite(position))
        ):
            raise RuntimeError("RLBench boundary-relative root pose is invalid")
        rx, ry, rz = orientation
        rotation_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(rx), -math.sin(rx)],
                [0.0, math.sin(rx), math.cos(rx)],
            ]
        )
        rotation_y = np.asarray(
            [
                [math.cos(ry), 0.0, math.sin(ry)],
                [0.0, 1.0, 0.0],
                [-math.sin(ry), 0.0, math.cos(ry)],
            ]
        )
        rotation_z = np.asarray(
            [
                [math.cos(rz), -math.sin(rz), 0.0],
                [math.sin(rz), math.cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Match the row-vector convention used by pinned SpawnBoundary.
        rotated = np.dot(points, np.dot(rotation_z, np.dot(rotation_y, rotation_x)))
        minimum = np.amin(rotated, axis=0) + position
        maximum = np.amax(rotated, axis=0) + position
        within_xy = bool(
            minimum[0] > float(boundary_box.min_x)
            and maximum[0] < float(boundary_box.max_x)
            and minimum[1] > float(boundary_box.min_y)
            and maximum[1] < float(boundary_box.max_y)
        )
        within_z = bool(
            getattr(entry, "_is_plane", False)
            or (
                minimum[2] > float(boundary_box.min_z)
                and maximum[2] < float(boundary_box.max_z)
            )
        )
        if within_xy and within_z:
            return True
    return False


def _workspace_source_placement_succeeded(scene: Any, task: Any) -> bool:
    """Detect a BoundaryError swallowed by reset(verify_instance=False)."""

    is_static = getattr(task, "is_static_workspace", None)
    if not callable(is_static):
        raise RuntimeError("RLBench task static-workspace API is unavailable")
    if bool(is_static()):
        return True
    return _workspace_boundary_contains_root(scene, task.boundary_root())


def _authenticate_motion_source_root(
    task: Any, profile: dict[str, Any]
) -> dict[str, str]:
    root = task.boundary_root()
    get_name = getattr(root, "get_name", None)
    get_type = getattr(root, "get_type", None)
    if not callable(get_name) or not callable(get_type):
        raise RuntimeError("RLBench boundary-root identity API is unavailable")
    object_type = get_type()
    type_name = getattr(object_type, "name", None)
    if not isinstance(type_name, str) or not type_name:
        raise RuntimeError("RLBench boundary-root type name is unavailable")
    actual = {"name": str(get_name()), "type": str(type_name)}
    expected = {
        "name": profile.get("spatial_root_name"),
        "type": profile.get("spatial_root_type"),
    }
    if actual != expected:
        raise RuntimeError("task boundary_root does not match frozen spatial profile")
    return actual


def _is_expected_placement_error(error: Exception) -> bool:
    """Recognize RLBench placement failures without importing RLBench eagerly."""

    expected_names = {"BoundaryError", "WaypointError"}
    return any(base.__name__ in expected_names for base in type(error).__mro__)


def _quaternion_angle_xyzw(left: Array, right: Array) -> float:
    """Return a sign-invariant physical angle with stable small-angle math."""

    q_left = np.asarray(left, dtype=np.float64)
    q_right = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(q_left))
    right_norm = float(np.linalg.norm(q_right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("pose quaternion must be non-zero")
    q_left = q_left / left_norm
    q_right = q_right / right_norm
    # For unit quaternions, the shorter sign-invariant chord is
    # c = 2 sin(theta / 4).  ``4 asin(c / 2)`` is substantially more stable
    # than ``2 acos(abs(dot))`` when independently initialized scenes differ
    # by only a few microradians.
    chord = min(
        float(np.linalg.norm(q_left - q_right)),
        float(np.linalg.norm(q_left + q_right)),
    )
    return float(4.0 * math.asin(float(np.clip(chord * 0.5, 0.0, 1.0))))


def _valid_low_dim_pose_chunks(value: Array) -> Array | None:
    """Return normalized ``[N, 7]`` pose chunks or ``None`` for scalar data.

    Every DynaMAC RLBench task schema is pose based, but keeping a deliberate
    scalar fallback makes this preservation guard safe for diagnostic or
    future tasks whose low-dimensional state is not entirely composed of
    world-frame poses.  Requiring approximately unit quaternions prevents an
    arbitrary seven-scalar vector from being misclassified as a pose.
    """

    state = np.asarray(value, dtype=np.float64)
    if state.ndim != 1 or state.size == 0 or state.size % 7 != 0:
        return None
    chunks = state.reshape(-1, 7)
    if not np.all(np.isfinite(chunks)):
        return None
    quaternion_norms = np.linalg.norm(chunks[:, 3:7], axis=1)
    if np.any(quaternion_norms <= 0.0) or not np.allclose(
        quaternion_norms,
        1.0,
        rtol=0.0,
        atol=LOW_DIM_POSE_QUATERNION_NORM_ATOL,
    ):
        return None
    return chunks


def _low_dim_roundtrip_metrics(
    before: Array,
    restored: Array,
    *,
    translation_tolerance_m: float = LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    rotation_tolerance_rad: float = LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
    scalar_tolerance: float = LOW_DIM_STATE_ROUNDTRIP_ATOL,
) -> dict[str, Any]:
    """Compare a task-state round trip without quaternion-gauge false alarms.

    Complete valid seven-value pose chunks use Euclidean translation and the
    sign-invariant physical quaternion angle.  Other arrays use a clearly
    labelled scalar maximum-absolute-error fallback.  Raw L2/max metrics are
    retained in both modes for forensic reporting, but do not override the
    physical pose decision.
    """

    source = np.asarray(before, dtype=np.float64)
    result = np.asarray(restored, dtype=np.float64)
    if source.shape != result.shape:
        raise ValueError("task low-dimensional state schema changed")
    delta = result - source
    raw_finite = bool(np.all(np.isfinite(source)) and np.all(np.isfinite(result)))
    raw_l2 = float(np.linalg.norm(delta)) if raw_finite else math.inf
    raw_max_abs = (
        float(np.max(np.abs(delta)))
        if raw_finite and delta.size
        else (0.0 if raw_finite else math.inf)
    )
    source_chunks = _valid_low_dim_pose_chunks(source)
    result_chunks = _valid_low_dim_pose_chunks(result)
    if source_chunks is not None and result_chunks is not None:
        translations = np.linalg.norm(
            result_chunks[:, :3] - source_chunks[:, :3],
            axis=1,
        )
        rotations = np.asarray(
            [
                _quaternion_angle_xyzw(left[3:7], right[3:7])
                for left, right in zip(source_chunks, result_chunks)
            ],
            dtype=np.float64,
        )
        max_translation = float(np.max(translations))
        max_rotation = float(np.max(rotations))
        preserved = bool(
            raw_finite
            and max_translation <= translation_tolerance_m
            and max_rotation <= rotation_tolerance_rad
        )
        comparison_mode = "pose_chunks_sign_invariant"
        chunk_count = int(source_chunks.shape[0])
    else:
        max_translation = None
        max_rotation = None
        preserved = bool(raw_finite and raw_max_abs <= scalar_tolerance)
        comparison_mode = "scalar_max_abs"
        chunk_count = 0
    return {
        "preserved": preserved,
        "comparison_mode": comparison_mode,
        "chunk_count": chunk_count,
        "raw_l2": raw_l2,
        "raw_max_abs": raw_max_abs,
        "max_translation_m": max_translation,
        "max_rotation_rad": max_rotation,
        "translation_tolerance_m": float(translation_tolerance_m),
        "rotation_tolerance_rad": float(rotation_tolerance_rad),
        "scalar_tolerance": float(scalar_tolerance),
        "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
    }


def _root_motion_metrics(source: Array, goal: Array) -> dict[str, Any]:
    source_pose = np.asarray(source, dtype=np.float64)
    goal_pose = np.asarray(goal, dtype=np.float64)
    if source_pose.shape != (7,) or goal_pose.shape != (7,):
        raise ValueError("RLBench boundary-root poses must contain seven values")
    translation = float(np.linalg.norm(goal_pose[:3] - source_pose[:3]))
    rotation = _quaternion_angle_xyzw(source_pose[3:7], goal_pose[3:7])
    return {
        "planned_root_translation_m": translation,
        "planned_root_rotation_rad": rotation,
        "planned_root_motion": bool(
            translation > ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
            or rotation > ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
        ),
    }


def _pose_reproducibility_metrics(
    reference: Array,
    current: Array,
    *,
    translation_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> dict[str, Any]:
    """Audit two independently initialized poses under explicit hard caps."""

    delta = _root_motion_metrics(reference, current)
    translation = float(delta["planned_root_translation_m"])
    rotation = float(delta["planned_root_rotation_rad"])
    return {
        "preserved": bool(
            translation <= translation_tolerance_m
            and rotation <= rotation_tolerance_rad
        ),
        "translation_error_m": translation,
        "rotation_error_rad": rotation,
        "translation_tolerance_m": float(translation_tolerance_m),
        "rotation_tolerance_rad": float(rotation_tolerance_rad),
        "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
    }


def _staging_source_fingerprint(
    *,
    task_name: str,
    variation: int,
    root_pose: Array,
    low_dim_state: Array,
    task_tree_state: list[dict[str, Any]],
    semantic_signature: dict[str, Any],
    descriptions: list[str],
    collision_pair_records: list[dict[str, Any]],
) -> str:
    """Fingerprint source identity while excluding diagnostic velocities."""

    return _canonical_json_fingerprint(
        {
            "task_name": task_name,
            "variation": int(variation),
            "root_pose": np.asarray(root_pose, dtype=np.float64).tolist(),
            "low_dim_state": np.asarray(low_dim_state, dtype=np.float64).tolist(),
            "task_tree_state": task_tree_state,
            "semantic_signature": semantic_signature,
            "descriptions": list(descriptions),
            "collision_pairs": collision_pair_records,
        }
    )


def _source_state_snapshot(
    scene: Any,
    descriptions: Any,
) -> dict[str, Any]:
    """Capture every portable field used to certify/reconstruct source A."""

    task = getattr(scene, "task", None)
    robot = getattr(scene, "robot", None)
    if task is None or robot is None:
        raise RuntimeError("RLBench source task or robot is unavailable")
    root_pose = np.asarray(task.boundary_root().get_pose(), dtype=np.float64)
    low_dim = _task_low_dim_state(task)
    tree = _task_tree_relative_state(task)
    semantics = _task_semantic_signature(task)
    collisions = _stable_collision_pair_records(
        _robot_external_collision_pairs(scene, robot)
    )
    description_list = list(descriptions) if descriptions is not None else None
    if description_list is None or not all(
        isinstance(value, str) for value in description_list
    ):
        raise RuntimeError("RLBench source descriptions are invalid")
    get_name = getattr(task, "get_name", None)
    task_name = str(get_name()) if callable(get_name) else type(task).__name__
    return {
        "task_name": task_name,
        "root_pose": root_pose,
        "low_dim_state": low_dim,
        "task_tree": tree,
        "task_semantics": semantics,
        "descriptions": description_list,
        "robot_numeric_state": _robot_numeric_state(robot),
        "stable_grasp_state": _stable_grasp_state(task, robot),
        "robot_external_collision_pairs": collisions,
        "velocity_summary": _task_tree_velocity_summary(task),
        "instance_references": _instance_reference_snapshot(task),
        "grasp_identity_state": _grasp_state_snapshot(task, robot),
    }


def _source_reconstruction_audit(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    """Strictly bind a fresh reset(false) source to its certified A."""

    root = _pose_reproducibility_metrics(
        expected["root_pose"],
        actual["root_pose"],
        translation_tolerance_m=SOURCE_RECONSTRUCTION_TRANSLATION_TOLERANCE_M,
        rotation_tolerance_rad=SOURCE_RECONSTRUCTION_ROTATION_TOLERANCE_RAD,
    )
    low_dim = _low_dim_roundtrip_metrics(
        expected["low_dim_state"],
        actual["low_dim_state"],
        translation_tolerance_m=SOURCE_RECONSTRUCTION_TRANSLATION_TOLERANCE_M,
        rotation_tolerance_rad=SOURCE_RECONSTRUCTION_ROTATION_TOLERANCE_RAD,
        scalar_tolerance=SOURCE_RECONSTRUCTION_SCALAR_TOLERANCE,
    )
    tree = _compare_task_tree_relative_state(
        expected["task_tree"],
        actual["task_tree"],
        boundary_root_may_move=False,
        translation_tolerance_m=SOURCE_RECONSTRUCTION_TRANSLATION_TOLERANCE_M,
        rotation_tolerance_rad=SOURCE_RECONSTRUCTION_ROTATION_TOLERANCE_RAD,
        joint_tolerance=SOURCE_RECONSTRUCTION_JOINT_TOLERANCE,
    )
    robot = _robot_numeric_state_audit(
        expected["robot_numeric_state"],
        actual["robot_numeric_state"],
    )
    exact = {
        "task_name": expected["task_name"] == actual["task_name"],
        "task_semantics": expected["task_semantics"] == actual["task_semantics"],
        "descriptions": expected["descriptions"] == actual["descriptions"],
        "stable_grasp_state": (
            expected["stable_grasp_state"] == actual["stable_grasp_state"]
        ),
        "robot_external_collision_pairs": (
            expected["robot_external_collision_pairs"]
            == actual["robot_external_collision_pairs"]
        ),
    }
    velocities_finite = bool(
        expected["velocity_summary"].get("all_finite") is True
        and actual["velocity_summary"].get("all_finite") is True
    )
    passed = bool(
        root["preserved"]
        and low_dim["preserved"]
        and tree["matched"]
        and robot["passed"]
        and all(exact.values())
        and velocities_finite
    )
    return {
        "schema": SOURCE_RECONSTRUCTION_SCHEMA,
        "root": root,
        "low_dim_state": low_dim,
        "task_tree": _compact_task_tree_comparison(tree),
        "robot_numeric_state": robot,
        "exact_matches": exact,
        "task_object_velocities_finite": velocities_finite,
        "passed": passed,
    }


def _waypoint_cache_evidence(task: Any) -> dict[str, Any]:
    value = getattr(task, "_waypoints", None)
    if value is None:
        return {"state": "none", "waypoints": []}
    if not isinstance(value, list):
        raise RuntimeError("RLBench waypoint cache has an invalid type")
    rows = []
    for waypoint in value:
        name = getattr(waypoint, "name", None)
        if name is None:
            get_name = getattr(waypoint, "get_name", None)
            name = get_name() if callable(get_name) else None
        if not isinstance(name, str) or not name:
            raise RuntimeError("RLBench waypoint cache identity is unavailable")
        rows.append({"name": name, "type": type(waypoint).__name__})
    return {"state": "materialized", "waypoints": rows}


def _certify_source_a(
    scene: Any,
    descriptions: Any,
    generation_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Explicitly validate A offline and prove validation only caches waypoints."""

    task = scene.task
    robot = scene.robot
    random_placement_expected = generation_evidence.get(
        "reset_random_placement_expected"
    )
    reset_collision_checks = generation_evidence.get(
        "reset_robot_collision_check_results"
    )
    expected_reset_collision_checks = [False] if random_placement_expected else []
    if reset_collision_checks != expected_reset_collision_checks:
        raise RuntimeError("reset(false) source placement collision audit failed")
    if not _workspace_source_placement_succeeded(scene, task):
        raise RuntimeError("reset(false) source workspace placement did not succeed")
    is_in_collision = getattr(robot, "is_in_collision", None)
    if not callable(is_in_collision):
        raise RuntimeError("RLBench robot collision aggregate is unavailable")
    if bool(is_in_collision()):
        raise RuntimeError("reset(false) source robot is in collision")
    before = _source_state_snapshot(scene, descriptions)
    before_references = before["instance_references"]
    before_grasp = before["grasp_identity_state"]
    waypoint_before = _waypoint_cache_evidence(task)
    if waypoint_before["state"] != "none":
        raise RuntimeError("fresh source waypoint cache is already materialized")
    validate = getattr(task, "validate", None)
    if not callable(validate):
        raise RuntimeError("RLBench Task.validate() is unavailable in staging")
    validate()
    after = _source_state_snapshot(scene, descriptions)
    waypoint_after = _waypoint_cache_evidence(task)
    if waypoint_after["state"] != "materialized" or not waypoint_after["waypoints"]:
        raise RuntimeError("source validation did not materialize a waypoint cache")
    state_audit = _source_reconstruction_audit(before, after)
    registry_preserved = _same_instance_references(
        before_references,
        _instance_reference_snapshot(task),
    )
    grasp_preserved = _same_grasp_state(before_grasp, task, robot)
    passed = bool(state_audit["passed"] and registry_preserved and grasp_preserved)
    certification = {
        "schema": SOURCE_CERTIFICATION_SCHEMA,
        "reset_verify_instance": False,
        "workspace_placement_succeeded": True,
        "reset_placement_collision_check_results": reset_collision_checks,
        "source_robot_collision_free": True,
        "task_validate_calls": 1,
        "task_validate_api": "Task.validate",
        "waypoint_cache_before": waypoint_before,
        "waypoint_cache_after": waypoint_after,
        "allowed_validation_mutation": "waypoint_cache_none_to_materialized_only",
        "state_audit": state_audit,
        "condition_fail_grasp_registry_identity_preserved": registry_preserved,
        "grasp_membership_and_parentage_preserved": grasp_preserved,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError("explicit source validation mutated source A")
    return before, certification


def _certify_goal_b(
    scene: Any,
    descriptions: Any,
) -> dict[str, Any]:
    """Validate one sampled B and reject every side effect except its cache."""

    task = scene.task
    robot = scene.robot
    before = _source_state_snapshot(scene, descriptions)
    before_references = before["instance_references"]
    before_grasp = before["grasp_identity_state"]
    waypoint_before = _waypoint_cache_evidence(task)
    if waypoint_before["state"] != "none":
        raise RuntimeError("fresh goal waypoint cache is already materialized")
    validate = getattr(task, "validate", None)
    if not callable(validate):
        raise RuntimeError("RLBench Task.validate() is unavailable in staging")
    validate()
    after = _source_state_snapshot(scene, descriptions)
    waypoint_after = _waypoint_cache_evidence(task)
    if waypoint_after["state"] != "materialized" or not waypoint_after["waypoints"]:
        raise RuntimeError("goal validation did not materialize a waypoint cache")
    state_audit = _source_reconstruction_audit(before, after)
    registry_preserved = _same_instance_references(
        before_references,
        _instance_reference_snapshot(task),
    )
    grasp_preserved = _same_grasp_state(before_grasp, task, robot)
    passed = bool(state_audit["passed"] and registry_preserved and grasp_preserved)
    certification = {
        "schema": GOAL_CERTIFICATION_SCHEMA,
        "task_validate_calls": 1,
        "task_validate_api": "Task.validate",
        "waypoint_cache_before": waypoint_before,
        "waypoint_cache_after": waypoint_after,
        "allowed_validation_mutation": "waypoint_cache_none_to_materialized_only",
        "state_audit": state_audit,
        "condition_fail_grasp_registry_identity_preserved": registry_preserved,
        "grasp_membership_and_parentage_preserved": grasp_preserved,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError("explicit goal validation mutated sampled B")
    return certification


def _formal_intervention_state_snapshot(scene: Any) -> dict[str, Any]:
    """Capture the current formal state immediately around one root command."""

    task = getattr(scene, "task", None)
    robot = getattr(scene, "robot", None)
    if task is None or robot is None:
        raise RuntimeError("formal RLBench task or robot is unavailable")
    collision_pairs = _robot_external_collision_pairs(scene, robot)
    return {
        "task_tree": _task_tree_relative_state(task),
        "task_semantics": _task_semantic_signature(task),
        "instance_references": _instance_reference_snapshot(task),
        "grasp_state": _grasp_state_snapshot(task, robot),
        "robot_external_collision_pairs": collision_pairs,
    }


def _formal_intervention_state_audit(
    scene: Any,
    before: dict[str, Any],
) -> dict[str, Any]:
    """Verify that one formal root command has only its intended rigid effect.

    The reference is the policy-evolved state captured immediately before this
    command, never staged A.  This distinction matters for every smooth
    fraction: policy and physics may legitimately evolve the formal episode
    between committed ticks, while ``set_pose`` itself must still preserve the
    current task topology, joints, semantics, and root-relative subtree state.
    Robot contact deltas are recorded exactly but are not an integrity gate:
    the robot pose at intervention time is a policy outcome, so rejecting a
    fixed A/B plan on that contact would make episode admission policy-dependent.
    """

    after = _formal_intervention_state_snapshot(scene)
    tree = _compare_task_tree_relative_state(
        before["task_tree"],
        after["task_tree"],
        boundary_root_may_move=True,
    )
    semantics_matched = before["task_semantics"] == after["task_semantics"]
    registry_identity_preserved = _same_instance_references(
        before["instance_references"],
        _instance_reference_snapshot(scene.task),
    )
    grasp_membership_and_parentage_preserved = _same_grasp_state(
        before["grasp_state"],
        scene.task,
        scene.robot,
    )
    before_pairs = tuple(before["robot_external_collision_pairs"])
    after_pairs = tuple(after["robot_external_collision_pairs"])
    new_pairs = tuple(sorted(frozenset(after_pairs) - frozenset(before_pairs)))
    no_new_collision_pairs = not new_pairs
    passed = bool(
        tree["matched"]
        and semantics_matched
        and registry_identity_preserved
        and grasp_membership_and_parentage_preserved
    )
    audit = {
        "schema": FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA,
        "comparison_class": (
            "same_formal_initialized_task_instance_immediate_pre_to_post_"
            "boundary_root_command"
        ),
        "reference_state": "current_policy_evolved_formal_state",
        "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
        "task_tree": tree,
        "task_semantics_matched": semantics_matched,
        "condition_and_grasp_registry_identity_preserved": (
            registry_identity_preserved
        ),
        "gripper_grasp_membership_and_parentage_preserved": (
            grasp_membership_and_parentage_preserved
        ),
        "robot_external_collision_pair_policy": (
            FORMAL_INTERVENTION_COLLISION_PAIR_POLICY
        ),
        "before_robot_external_collision_pairs": (
            _stable_collision_pair_records(before_pairs)
        ),
        "after_robot_external_collision_pairs": (
            _stable_collision_pair_records(after_pairs)
        ),
        "new_robot_external_collision_pairs": (
            _stable_collision_pair_records(new_pairs)
        ),
        "no_new_robot_external_collision_pairs": no_new_collision_pairs,
        "passed": passed,
    }
    if not tree["matched"]:
        raise RuntimeError(
            "formal boundary-root command changed task-tree topology or state "
            "outside its strict same-instance rigid motion"
        )
    if not semantics_matched:
        raise RuntimeError("formal boundary-root command changed task semantics")
    if not registry_identity_preserved:
        raise RuntimeError(
            "formal boundary-root command replaced task condition/grasp registries"
        )
    if not grasp_membership_and_parentage_preserved:
        raise RuntimeError(
            "formal boundary-root command changed gripper membership or parentage"
        )
    return audit


def _root_application_metrics(
    before: Array,
    commanded: Array,
    applied: Array,
) -> dict[str, Any]:
    """Measure actual root motion and residual to the commanded pose."""

    actual = _root_motion_metrics(before, applied)
    residual = _root_motion_metrics(commanded, applied)
    actual_translation = float(actual["planned_root_translation_m"])
    actual_rotation = float(actual["planned_root_rotation_rad"])
    translation_residual = float(residual["planned_root_translation_m"])
    rotation_residual = float(residual["planned_root_rotation_rad"])
    command_reached = bool(
        translation_residual <= ROOT_COMMAND_TRANSLATION_TOLERANCE_M
        and rotation_residual <= ROOT_COMMAND_ROTATION_TOLERANCE_RAD
    )
    return {
        "actual_root_translation_m": actual_translation,
        "actual_root_rotation_rad": actual_rotation,
        "actual_root_motion": bool(
            actual_translation > ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
            or actual_rotation > ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
        ),
        "commanded_root_translation_residual_m": translation_residual,
        "commanded_root_rotation_residual_rad": rotation_residual,
        "commanded_root_pose_reached": command_reached,
    }


def _root_goal_reached_metrics(goal: Array, applied: Array) -> dict[str, Any]:
    """Measure an applied pose against the final sampled motion goal."""

    residual = _root_motion_metrics(goal, applied)
    translation = float(residual["planned_root_translation_m"])
    rotation = float(residual["planned_root_rotation_rad"])
    return {
        "goal_root_translation_residual_m": translation,
        "goal_root_rotation_residual_rad": rotation,
        "goal_root_pose_reached": bool(
            translation <= ROOT_COMMAND_TRANSLATION_TOLERANCE_M
            and rotation <= ROOT_COMMAND_ROTATION_TOLERANCE_RAD
        ),
    }


def _interpolate_rlbench_pose(source: Array, goal: Array, fraction: float) -> Array:
    """Interpolate an RLBench ``[xyz, qxyzw]`` pose with shortest-path SLERP."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("pose interpolation fraction must lie in [0, 1]")
    source_pose = np.asarray(source, dtype=np.float64)
    goal_pose = np.asarray(goal, dtype=np.float64)
    if source_pose.shape != (7,) or goal_pose.shape != (7,):
        raise ValueError("RLBench poses must contain seven values")

    position = (1.0 - fraction) * source_pose[:3] + fraction * goal_pose[:3]
    q_source = source_pose[3:7] / np.linalg.norm(source_pose[3:7])
    q_goal = goal_pose[3:7] / np.linalg.norm(goal_pose[3:7])
    dot = float(np.dot(q_source, q_goal))
    if dot < 0.0:
        q_goal = -q_goal
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quaternion = q_source + fraction * (q_goal - q_source)
        quaternion /= np.linalg.norm(quaternion)
    else:
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * fraction
        scale_source = math.sin(theta_0 - theta) / sin_theta_0
        scale_goal = math.sin(theta) / sin_theta_0
        quaternion = scale_source * q_source + scale_goal * q_goal
    return np.concatenate((position, quaternion))


def _quaternion_multiply_xyzw(left: Array, right: Array) -> Array:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quaternion_rotate_xyzw(quaternion: Array, vector: Array) -> Array:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    pure = np.concatenate((np.asarray(vector, dtype=np.float64), [0.0]))
    conjugate = np.asarray([-q[0], -q[1], -q[2], q[3]])
    return _quaternion_multiply_xyzw(
        _quaternion_multiply_xyzw(q, pure),
        conjugate,
    )[:3]


def _task_frame_rigid_motion_evidence(
    task_name: str,
    source_root: Array,
    goal_root: Array,
    source_low_dim: Array,
    goal_low_dim: Array,
) -> dict[str, Any]:
    """Verify every public task frame follows the boundary-root rigid delta."""

    spec = get_task_spec(task_name)
    source_frames = spec.extract_pose_chunks(
        source_low_dim,
        convention="rlbench_xyzw",
    )
    goal_frames = spec.extract_pose_chunks(
        goal_low_dim,
        convention="rlbench_xyzw",
    )
    source_root = np.asarray(source_root, dtype=np.float64)
    goal_root = np.asarray(goal_root, dtype=np.float64)
    q_source = source_root[3:7] / np.linalg.norm(source_root[3:7])
    q_goal = goal_root[3:7] / np.linalg.norm(goal_root[3:7])
    q_source_inverse = np.asarray(
        [-q_source[0], -q_source[1], -q_source[2], q_source[3]],
    )
    delta_q = _quaternion_multiply_xyzw(q_goal, q_source_inverse)
    rows = []
    all_preserved = True
    for frame_name in spec.frame_names:
        source = source_frames[frame_name]
        goal = goal_frames[frame_name]
        expected_position = goal_root[:3] + _quaternion_rotate_xyzw(
            delta_q,
            source[:3] - source_root[:3],
        )
        expected_quaternion = _quaternion_multiply_xyzw(delta_q, source[3:7])
        translation_error = float(np.linalg.norm(goal[:3] - expected_position))
        rotation_error = _quaternion_angle_xyzw(goal[3:7], expected_quaternion)
        preserved = bool(
            translation_error <= LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
            and rotation_error <= LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
        )
        all_preserved = all_preserved and preserved
        rows.append(
            {
                "frame": frame_name,
                "translation_error_m": translation_error,
                "rotation_error_rad": rotation_error,
                "preserved": preserved,
            }
        )
    return {
        "task_spec": task_name,
        "source_expression": spec.source_expression,
        "checked_frames": list(spec.frame_names),
        "all_pose_chunks_follow_boundary_root_rigid_transform": all_preserved,
        "translation_tolerance_m": LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
        "rotation_tolerance_rad": LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
        "frames": rows,
    }


def _validate_staged_motion_plan_validation(
    *,
    task_name: str,
    episode_seed: int,
    variation: int,
    source_pose: Array,
    source_low_dim_state: Array,
    validation: dict[str, Any],
) -> None:
    """Fail closed on a re-signed but internally inconsistent V3.4 plan."""

    from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
        load_v3_motion_source_protocol,
        motion_source_profile,
    )

    motion_protocol = load_v3_motion_source_protocol()
    profile = motion_source_profile(task_name, motion_protocol)
    source_seed = validation.get("source_seed")
    selection = validation.get("source_seed_selection")
    certification = validation.get("source_certification")
    selected_reconstruction = validation.get("selected_source_reconstruction")
    source_attempts = selection.get("attempts") if isinstance(selection, dict) else None
    goal_attempts = validation.get("goal_candidate_attempts")
    goal_certification = validation.get("goal_certification")
    sampling_attempts = validation.get("sampling_attempts")
    if (
        validation.get("schema") != STAGED_MOTION_PLAN_VALIDATION_SCHEMA
        or validation.get("fresh_task_generation_protocol_id")
        != DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
        or validation.get("motion_source_protocol_schema") != motion_protocol["schema"]
        or validation.get("motion_source_protocol_fingerprint")
        != motion_protocol["fingerprint"]
        or validation.get("motion_source_profile") != profile
        or validation.get("resolved_spatial_root")
        != {"name": profile["spatial_root_name"], "type": profile["spatial_root_type"]}
        or validation.get("formal_rollout_sample_or_restore") is not False
        or validation.get("formal_source_binding_required") is not True
        or validation.get("source_waypoint_validated") is not True
        or validation.get("goal_waypoint_validated") is not True
        or not isinstance(goal_certification, dict)
        or goal_certification.get("schema") != GOAL_CERTIFICATION_SCHEMA
        or goal_certification.get("task_validate_calls") != 1
        or goal_certification.get("waypoint_cache_before")
        != {"state": "none", "waypoints": []}
        or not isinstance(goal_certification.get("waypoint_cache_after"), dict)
        or goal_certification["waypoint_cache_after"].get("state") != "materialized"
        or not goal_certification["waypoint_cache_after"].get("waypoints")
        or goal_certification.get("state_audit", {}).get("passed") is not True
        or goal_certification.get("condition_fail_grasp_registry_identity_preserved")
        is not True
        or goal_certification.get("grasp_membership_and_parentage_preserved")
        is not True
        or goal_certification.get("passed") is not True
        or validation.get(
            "source_certification_and_exported_source_different_generations"
        )
        is not True
        or validation.get("accepted_candidate_reconstructed_A_and_B_same_instance")
        is not True
        or validation.get("task_get_state_called_directly_by_candidate_sampler")
        is not False
        or validation.get("task_restore_state_called_directly_by_candidate_sampler")
        is not False
        or validation.get("goal_sampling_max_attempts")
        != motion_protocol["goal_sampling_max_attempts"]
        or isinstance(source_seed, bool)
        or not isinstance(source_seed, int)
        or source_seed < 0
        or isinstance(sampling_attempts, bool)
        or not isinstance(sampling_attempts, int)
        or not 1 <= sampling_attempts <= validation.get("goal_sampling_max_attempts", 0)
    ):
        raise ValueError("staged deterministic-source validation header is invalid")
    if (
        not isinstance(selection, dict)
        or selection.get("schema") != SOURCE_SEED_SELECTION_SCHEMA
        or selection.get("logical_episode_seed") != episode_seed
        or selection.get("max_attempts")
        != motion_protocol["source_selection_max_attempts"]
        or selection.get("selected_source_seed") != source_seed
        or not isinstance(source_attempts, list)
        or not source_attempts
        or selection.get("attempts_used") != len(source_attempts)
        or len(source_attempts) > selection["max_attempts"]
        or [row.get("attempt") for row in source_attempts]
        != list(range(1, len(source_attempts) + 1))
    ):
        raise ValueError("staged source-seed selection evidence is invalid")
    for index, row in enumerate(source_attempts, 1):
        expected_seed = _source_seed(episode_seed, variation, index)
        accepted = index == len(source_attempts)
        if (
            not isinstance(row, dict)
            or row.get("source_seed") != expected_seed
            or row.get("resolved_spatial_root") != validation["resolved_spatial_root"]
            or row.get("accepted") is not accepted
            or (row.get("certification_ref") is not None) is not accepted
            or (row.get("rejection_type") is None) is not accepted
            or (row.get("rejection_reason") is None) is not accepted
        ):
            raise ValueError("staged source attempt ledger is invalid")
        try:
            _validate_fresh_task_generation_evidence(
                row.get("fresh_task_generation"),
                episode_seed=expected_seed,
                variation=variation,
                task_name=task_name,
                verify_instance=False,
            )
        except RuntimeError as error:
            raise ValueError("staged source reset evidence is invalid") from error
    if (
        source_seed != source_attempts[-1]["source_seed"]
        or validation.get("source_certification_fresh_task_generation")
        != source_attempts[-1]["fresh_task_generation"]
        or source_attempts[-1].get("certification_ref")
        != "validation.source_certification"
        or not isinstance(certification, dict)
        or certification.get("schema") != SOURCE_CERTIFICATION_SCHEMA
        or certification.get("reset_verify_instance") is not False
        or certification.get("task_validate_calls") != 1
        or certification.get("workspace_placement_succeeded") is not True
        or certification.get("source_robot_collision_free") is not True
        or certification.get("condition_fail_grasp_registry_identity_preserved")
        is not True
        or certification.get("grasp_membership_and_parentage_preserved") is not True
        or certification.get("waypoint_cache_before")
        != {"state": "none", "waypoints": []}
        or not isinstance(certification.get("waypoint_cache_after"), dict)
        or certification["waypoint_cache_after"].get("state") != "materialized"
        or not certification["waypoint_cache_after"].get("waypoints")
        or certification.get("state_audit", {}).get("passed") is not True
        or certification.get("passed") is not True
    ):
        raise ValueError("staged source certification proof is invalid")
    try:
        _validate_fresh_task_generation_evidence(
            validation.get("selected_source_fresh_task_generation"),
            episode_seed=source_seed,
            variation=variation,
            task_name=task_name,
            verify_instance=False,
        )
    except RuntimeError as error:
        raise ValueError("selected source reset evidence is invalid") from error
    if (
        not isinstance(selected_reconstruction, dict)
        or selected_reconstruction.get("schema") != SOURCE_RECONSTRUCTION_SCHEMA
        or selected_reconstruction.get("resolved_spatial_root")
        != validation["resolved_spatial_root"]
        or selected_reconstruction.get("passed") is not True
        or not isinstance(goal_attempts, list)
        or len(goal_attempts) != sampling_attempts
        or [row.get("attempt") for row in goal_attempts]
        != list(range(1, sampling_attempts + 1))
    ):
        raise ValueError("selected or candidate source reconstruction is invalid")
    for index, row in enumerate(goal_attempts, 1):
        expected_candidate_seed = int(
            (episode_seed * 1_000_003 + variation * 9_176 + index * 7_919) % (2**32 - 1)
        )
        accepted = index == sampling_attempts
        reconstruction = row.get("source_reconstruction")
        if (
            row.get("candidate_seed") != expected_candidate_seed
            or (row.get("outcome") == "accepted") is not accepted
            or row.get("outcome")
            not in {
                "accepted",
                "placement_rejected",
                "waypoint_rejected",
                "collision_rejected",
            }
            or (row.get("reason") is None) is not accepted
            or not isinstance(reconstruction, dict)
            or reconstruction.get("schema") != SOURCE_RECONSTRUCTION_SCHEMA
            or reconstruction.get("resolved_spatial_root")
            != validation["resolved_spatial_root"]
            or reconstruction.get("source_waypoint_cache_state") != "none"
            or reconstruction.get("passed") is not True
            or (
                row.get("outcome") != "placement_rejected"
                and row.get("workspace_placement_succeeded") is not True
            )
            or (
                accepted
                and (
                    row.get("goal_certification_ref") != "validation.goal_certification"
                    or not isinstance(row.get("goal_waypoint_cache"), dict)
                    or row["goal_waypoint_cache"].get("state") != "materialized"
                    or not row["goal_waypoint_cache"].get("waypoints")
                )
            )
        ):
            raise ValueError("goal-candidate reconstruction ledger is invalid")
        try:
            _validate_fresh_task_generation_evidence(
                row.get("fresh_task_generation"),
                episode_seed=source_seed,
                variation=variation,
                task_name=task_name,
                verify_instance=False,
            )
        except RuntimeError as error:
            raise ValueError("goal-candidate reset evidence is invalid") from error
    if (
        validation.get("selected_candidate_seed") != goal_attempts[-1]["candidate_seed"]
        or validation.get("placement_rejections")
        != sum(row["outcome"] == "placement_rejected" for row in goal_attempts)
        or validation.get("waypoint_rejections")
        != sum(row["outcome"] == "waypoint_rejected" for row in goal_attempts)
        or validation.get("new_collision_pair_rejections")
        != sum(row["outcome"] == "collision_rejected" for row in goal_attempts)
    ):
        raise ValueError("staged goal rejection accounting is invalid")
    generation_evidence = (
        [row["fresh_task_generation"] for row in source_attempts]
        + [validation["selected_source_fresh_task_generation"]]
        + [row["fresh_task_generation"] for row in goal_attempts]
    )
    if any(
        left["generation_index"] + 1 != right["generation_index"]
        for left, right in zip(generation_evidence, generation_evidence[1:])
    ):
        raise ValueError("staged fresh-generation ledger is invalid")
    if (
        validation.get("source_task_tree_object_count")
        != len(validation.get("source_task_tree_relative_state", []))
        or validation.get("source_task_tree_fingerprint")
        != _canonical_json_fingerprint(validation["source_task_tree_relative_state"])
        or validation.get("task_semantic_fingerprint")
        != _canonical_json_fingerprint(validation.get("task_semantic_signature"))
        or validation.get("source_low_dim_size")
        != int(np.asarray(source_low_dim_state).size)
    ):
        raise ValueError("staged source structural fingerprints are invalid")
    selected_source_fingerprint = _staging_source_fingerprint(
        task_name=task_name,
        variation=variation,
        root_pose=source_pose,
        low_dim_state=source_low_dim_state,
        task_tree_state=validation["source_task_tree_relative_state"],
        semantic_signature=validation["task_semantic_signature"],
        descriptions=validation["task_descriptions"],
        collision_pair_records=validation["source_robot_external_collision_pairs"],
    )
    if validation.get("selected_source_fingerprint") != selected_source_fingerprint:
        raise ValueError("staged selected-source fingerprint is invalid")


@dataclass(frozen=True)
class StagedMotionPlan:
    """A/B root poses validated before the formal rollout is launched."""

    task_name: str
    source_pose: tuple[float, ...]
    goal_pose: tuple[float, ...]
    source_low_dim_state: tuple[float, ...]
    episode_seed: int
    variation: int
    validation: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.task_name, str) or not self.task_name.strip():
            raise ValueError("staged task name must be non-empty")
        source = np.asarray(self.source_pose, dtype=np.float64)
        goal = np.asarray(self.goal_pose, dtype=np.float64)
        low_dim = np.asarray(self.source_low_dim_state, dtype=np.float64).reshape(-1)
        if source.shape != (7,) or goal.shape != (7,):
            raise ValueError("staged source and goal poses must contain seven values")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(goal)):
            raise ValueError("staged motion poses must be finite")
        if not np.all(np.isfinite(low_dim)):
            raise ValueError("staged source low-dimensional state must be finite")
        if isinstance(self.episode_seed, bool) or not isinstance(
            self.episode_seed, int
        ):
            raise TypeError("staged episode seed must be an integer")
        if self.episode_seed < 0:
            raise ValueError("staged episode seed must be non-negative")
        if isinstance(self.variation, bool) or not isinstance(self.variation, int):
            raise TypeError("staged variation must be an integer")
        if self.variation < 0:
            raise ValueError("staged variation must be non-negative")
        if not isinstance(self.validation, dict):
            raise TypeError("staged validation evidence must be a dictionary")
        if self.validation.get("schema") != STAGED_MOTION_PLAN_VALIDATION_SCHEMA:
            raise ValueError("staged motion-plan validation schema is invalid")
        if self.validation.get("source_waypoint_validated") is not True:
            raise ValueError("staged source configuration was not waypoint validated")
        if self.validation.get("goal_waypoint_validated") is not True:
            raise ValueError("staged goal configuration was not waypoint validated")
        staging_max_attempts = self.validation.get("goal_sampling_max_attempts")
        sampling_attempts = self.validation.get("sampling_attempts")
        if (
            isinstance(staging_max_attempts, bool)
            or not isinstance(staging_max_attempts, int)
            or staging_max_attempts < 1
        ):
            raise ValueError("staged motion plan has no valid maximum attempt budget")
        if (
            isinstance(sampling_attempts, bool)
            or not isinstance(sampling_attempts, int)
            or not 1 <= sampling_attempts <= staging_max_attempts
        ):
            raise ValueError("staged motion plan sampling-attempt evidence is invalid")
        _validate_staged_motion_plan_validation(
            task_name=self.task_name,
            episode_seed=self.episode_seed,
            variation=self.variation,
            source_pose=source,
            source_low_dim_state=low_dim,
            validation=self.validation,
        )
        object.__setattr__(self, "source_pose", tuple(float(v) for v in source))
        object.__setattr__(self, "goal_pose", tuple(float(v) for v in goal))
        object.__setattr__(
            self,
            "source_low_dim_state",
            tuple(float(v) for v in low_dim),
        )
        object.__setattr__(self, "validation", dict(self.validation))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": STAGED_MOTION_PLAN_SCHEMA,
            "protocol_id": STAGED_VALIDATED_MOTION_PROTOCOL_ID,
            "task_name": self.task_name,
            "episode_seed": self.episode_seed,
            "variation": self.variation,
            "source_pose": list(self.source_pose),
            "goal_pose": list(self.goal_pose),
            "source_low_dim_state": list(self.source_low_dim_state),
            "validation": dict(self.validation),
        }

    def fingerprint(self) -> str:
        return _canonical_json_fingerprint(self.metadata())

    def to_json(self) -> dict[str, Any]:
        return {**self.metadata(), "fingerprint": self.fingerprint()}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> StagedMotionPlan:
        if not isinstance(payload, dict):
            raise ValueError("staged motion plan payload must be a dictionary")
        validation = payload.get("validation")
        plan = cls(
            task_name=payload.get("task_name", ""),
            source_pose=tuple(payload.get("source_pose", ())),
            goal_pose=tuple(payload.get("goal_pose", ())),
            source_low_dim_state=tuple(payload.get("source_low_dim_state", ())),
            episode_seed=payload.get("episode_seed"),
            variation=payload.get("variation"),
            validation=validation if isinstance(validation, dict) else {},
        )
        if payload.get("protocol_id") != STAGED_VALIDATED_MOTION_PROTOCOL_ID:
            raise ValueError("staged motion plan protocol ID is invalid")
        if payload.get("schema") != STAGED_MOTION_PLAN_SCHEMA:
            raise ValueError("staged motion plan schema is invalid")
        if payload.get("fingerprint") != plan.fingerprint():
            raise ValueError("staged motion plan fingerprint is invalid")
        return plan


def staged_motion_plan_batch(
    *,
    task_name: str,
    base_seed: int,
    variations: list[int],
    plans: list[StagedMotionPlan],
) -> dict[str, Any]:
    """Build a scenario-independent, fingerprinted plan-cache payload."""

    if len(variations) != len(plans):
        raise ValueError("staged plan variation schedule length mismatch")
    for episode, (variation, plan) in enumerate(zip(variations, plans)):
        if plan.task_name != task_name:
            raise ValueError("staged plan task does not match batch task")
        if plan.episode_seed != base_seed + episode:
            raise ValueError("staged plan episode seed does not match batch schedule")
        if plan.variation != variation:
            raise ValueError("staged plan variation does not match batch schedule")
    body = {
        "schema": STAGED_MOTION_PLAN_BATCH_SCHEMA,
        "protocol_id": STAGED_VALIDATED_MOTION_PROTOCOL_ID,
        "task_name": task_name,
        "base_seed": int(base_seed),
        "episodes": len(plans),
        "variation_schedule": [int(value) for value in variations],
        "scenario_independent": True,
        "seed_domain": (
            "logical_episode=base_seed+episode;"
            "A=certified_selected_source_seed_rebuilt_with_reset_false;"
            "B=deterministic_post_source_reconstruction_candidate_seed;"
            "scenario_excluded"
        ),
        "plans": [plan.to_json() for plan in plans],
    }
    return {**body, "batch_fingerprint": _canonical_json_fingerprint(body)}


def load_staged_motion_plan_batch(payload: dict[str, Any]) -> list[StagedMotionPlan]:
    """Authenticate and deserialize one scenario-independent plan batch."""

    if not isinstance(payload, dict):
        raise ValueError("staged motion plan batch must be a dictionary")
    if payload.get("schema") != STAGED_MOTION_PLAN_BATCH_SCHEMA:
        raise ValueError("staged motion plan batch schema is invalid")
    if payload.get("protocol_id") != STAGED_VALIDATED_MOTION_PROTOCOL_ID:
        raise ValueError("staged motion plan batch protocol is invalid")
    if payload.get("scenario_independent") is not True:
        raise ValueError("staged motion plan batch is scenario-dependent")
    fingerprint = payload.get("batch_fingerprint")
    body = {key: value for key, value in payload.items() if key != "batch_fingerprint"}
    expected = _canonical_json_fingerprint(body)
    if fingerprint != expected:
        raise ValueError("staged motion plan batch fingerprint is invalid")
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, list):
        raise ValueError("staged motion plan batch plans are invalid")
    plans = [StagedMotionPlan.from_json(item) for item in raw_plans]
    generation_evidence = [
        evidence
        for plan in plans
        for evidence in (
            [
                row["fresh_task_generation"]
                for row in plan.validation["source_seed_selection"]["attempts"]
            ]
            + [plan.validation["selected_source_fresh_task_generation"]]
            + [
                row["fresh_task_generation"]
                for row in plan.validation["goal_candidate_attempts"]
            ]
        )
    ]
    if [row["generation_index"] for row in generation_evidence] != list(
        range(1, len(generation_evidence) + 1)
    ) or any(
        row.get("previous_task_present") is not (index > 0)
        or row.get("physics_running_before_stop") is not (index > 0)
        for index, row in enumerate(generation_evidence)
    ):
        raise ValueError("staged plan batch fresh-generation ledger is invalid")
    expected_payload = staged_motion_plan_batch(
        task_name=payload.get("task_name"),
        base_seed=payload.get("base_seed"),
        variations=payload.get("variation_schedule"),
        plans=plans,
    )
    if expected_payload != payload:
        raise ValueError("staged motion plan batch fields are inconsistent")
    return plans


@dataclass(frozen=True)
class StagedSourcePlan:
    """One offline-certified deterministic A for coordination evaluation."""

    task_name: str
    source_pose: tuple[float, ...]
    source_low_dim_state: tuple[float, ...]
    episode_seed: int
    variation: int
    validation: dict[str, Any]

    def __post_init__(self) -> None:
        source = np.asarray(self.source_pose, dtype=np.float64)
        low_dim = np.asarray(self.source_low_dim_state, dtype=np.float64).reshape(-1)
        if (
            not isinstance(self.task_name, str)
            or not self.task_name
            or source.shape != (7,)
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(low_dim))
            or isinstance(self.episode_seed, bool)
            or not isinstance(self.episode_seed, int)
            or self.episode_seed < 0
            or isinstance(self.variation, bool)
            or not isinstance(self.variation, int)
            or self.variation < 0
        ):
            raise ValueError("staged source plan identity or numeric state is invalid")
        validation = self.validation
        selection = (
            validation.get("source_seed_selection")
            if isinstance(validation, dict)
            else None
        )
        reconstruction = (
            validation.get("selected_source_reconstruction")
            if isinstance(validation, dict)
            else None
        )
        certification = (
            validation.get("source_certification")
            if isinstance(validation, dict)
            else None
        )
        if (
            validation.get("schema") != STAGED_SOURCE_VALIDATION_SCHEMA
            or validation.get("fresh_task_generation_protocol_id")
            != DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
            or validation.get("formal_task_validate_calls") != 0
            or validation.get("source_seed") != selection.get("selected_source_seed")
            or selection.get("logical_episode_seed") != self.episode_seed
            or selection.get("max_attempts") != DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS
            or not isinstance(selection.get("attempts"), list)
            or selection.get("attempts_used") != len(selection["attempts"])
            or not isinstance(certification, dict)
            or certification.get("schema") != SOURCE_CERTIFICATION_SCHEMA
            or certification.get("reset_verify_instance") is not False
            or certification.get("workspace_placement_succeeded") is not True
            or certification.get("source_robot_collision_free") is not True
            or certification.get("task_validate_calls") != 1
            or certification.get("waypoint_cache_before")
            != {"state": "none", "waypoints": []}
            or not isinstance(certification.get("waypoint_cache_after"), dict)
            or certification["waypoint_cache_after"].get("state") != "materialized"
            or not certification["waypoint_cache_after"].get("waypoints")
            or certification.get("state_audit", {}).get("passed") is not True
            or certification.get("condition_fail_grasp_registry_identity_preserved")
            is not True
            or certification.get("grasp_membership_and_parentage_preserved") is not True
            or certification.get("passed") is not True
            or not isinstance(reconstruction, dict)
            or reconstruction.get("schema") != SOURCE_RECONSTRUCTION_SCHEMA
            or reconstruction.get("passed") is not True
            or validation.get("selected_source_fingerprint")
            != _staging_source_fingerprint(
                task_name=self.task_name,
                variation=self.variation,
                root_pose=source,
                low_dim_state=low_dim,
                task_tree_state=validation.get("source_task_tree_relative_state"),
                semantic_signature=validation.get("task_semantic_signature"),
                descriptions=validation.get("task_descriptions"),
                collision_pair_records=validation.get(
                    "source_robot_external_collision_pairs"
                ),
            )
        ):
            raise ValueError("staged source validation evidence is invalid")
        for index, row in enumerate(selection["attempts"], 1):
            accepted = index == selection["attempts_used"]
            expected_seed = _source_seed(self.episode_seed, self.variation, index)
            if (
                row.get("attempt") != index
                or row.get("source_seed") != expected_seed
                or row.get("accepted") is not accepted
            ):
                raise ValueError("staged source selection ledger is invalid")
            try:
                _validate_fresh_task_generation_evidence(
                    row.get("fresh_task_generation"),
                    episode_seed=expected_seed,
                    variation=self.variation,
                    task_name=self.task_name,
                    verify_instance=False,
                )
            except RuntimeError as error:
                raise ValueError("staged source proof reset is invalid") from error
        proof = validation.get("source_certification_fresh_task_generation")
        selected = validation.get("selected_source_fresh_task_generation")
        try:
            _validate_fresh_task_generation_evidence(
                selected,
                episode_seed=validation["source_seed"],
                variation=self.variation,
                task_name=self.task_name,
                verify_instance=False,
            )
        except RuntimeError as error:
            raise ValueError("selected staged source reset is invalid") from error
        if (
            not isinstance(proof, dict)
            or not isinstance(selected, dict)
            or proof != selection["attempts"][-1]["fresh_task_generation"]
            or proof.get("generation_index") >= selected.get("generation_index")
            or validation.get(
                "source_certification_and_exported_source_different_generations"
            )
            is not True
        ):
            raise ValueError("staged source proof/export isolation is invalid")

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": STAGED_SOURCE_PLAN_SCHEMA,
            "protocol_id": STAGED_SOURCE_PROTOCOL_ID,
            "task_name": self.task_name,
            "source_pose": list(self.source_pose),
            "source_low_dim_state": list(self.source_low_dim_state),
            "episode_seed": self.episode_seed,
            "variation": self.variation,
            "validation": self.validation,
        }

    def fingerprint(self) -> str:
        return _canonical_json_fingerprint(self.metadata())

    def to_json(self) -> dict[str, Any]:
        return {**self.metadata(), "fingerprint": self.fingerprint()}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> StagedSourcePlan:
        if not isinstance(payload, dict):
            raise ValueError("staged source payload must be a dictionary")
        plan = cls(
            task_name=payload.get("task_name"),
            source_pose=tuple(payload.get("source_pose", ())),
            source_low_dim_state=tuple(payload.get("source_low_dim_state", ())),
            episode_seed=payload.get("episode_seed"),
            variation=payload.get("variation"),
            validation=payload.get("validation"),
        )
        if (
            payload.get("schema") != STAGED_SOURCE_PLAN_SCHEMA
            or payload.get("protocol_id") != STAGED_SOURCE_PROTOCOL_ID
            or payload.get("fingerprint") != plan.fingerprint()
        ):
            raise ValueError("staged source payload authentication failed")
        return plan


def stage_source_plan(
    environment: Any,
    task_class: Any,
    *,
    task_name: str,
    episode_seed: int,
    variation: int,
) -> StagedSourcePlan:
    """Certify A offline, discard the proof generation, and rebuild clean A."""

    attempts = []
    certified = certification = proof_evidence = None
    selected_seed = None
    for attempt in range(1, DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS + 1):
        source_seed = _source_seed(episode_seed, variation, attempt)
        task_environment, descriptions, _observation, evidence = (
            initialize_fresh_task_generation(
                environment,
                task_class,
                episode_seed=source_seed,
                variation=variation,
                verify_instance=False,
            )
        )
        scene = task_environment._scene
        actual_name = str(scene.task.get_name())
        if actual_name != task_name:
            raise RuntimeError("A-only task identity does not match preregistration")
        try:
            snapshot, candidate_certification = _certify_source_a(
                scene,
                descriptions,
                evidence,
            )
        except Exception as error:
            expected = _is_expected_placement_error(error) or str(error) in {
                "reset(false) source workspace placement did not succeed",
                "reset(false) source robot is in collision",
                "reset(false) source placement collision audit failed",
            }
            if not expected:
                raise
            attempts.append(
                {
                    "attempt": attempt,
                    "source_seed": source_seed,
                    "accepted": False,
                    "fresh_task_generation": evidence,
                    "rejection_type": type(error).__name__,
                    "rejection_reason": str(error) or type(error).__name__,
                }
            )
            continue
        attempts.append(
            {
                "attempt": attempt,
                "source_seed": source_seed,
                "accepted": True,
                "fresh_task_generation": evidence,
                "rejection_type": None,
                "rejection_reason": None,
            }
        )
        certified = snapshot
        certification = candidate_certification
        proof_evidence = evidence
        selected_seed = source_seed
        break
    if certified is None or selected_seed is None:
        raise RuntimeError(
            "could not certify deterministic coordination source A after "
            f"{DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS} attempts; "
            f"{_source_selection_failure_diagnostic(attempts)}"
        )
    task_environment, descriptions, _observation, selected_evidence = (
        initialize_fresh_task_generation(
            environment,
            task_class,
            episode_seed=selected_seed,
            variation=variation,
            verify_instance=False,
        )
    )
    selected = _source_state_snapshot(task_environment._scene, descriptions)
    reconstruction = _source_reconstruction_audit(certified, selected)
    if (
        not reconstruction["passed"]
        or _waypoint_cache_evidence(task_environment._scene.task)["state"] != "none"
    ):
        raise RuntimeError("selected coordination A did not reconstruct cleanly")
    selection = {
        "schema": SOURCE_SEED_SELECTION_SCHEMA,
        "logical_episode_seed": episode_seed,
        "seed_derivation": "attempt1=episode_seed;fallback=(episode_seed*1000003+variation*9176+attempt*104729)%4294967295",
        "max_attempts": DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS,
        "attempts_used": len(attempts),
        "selected_source_seed": selected_seed,
        "attempts": attempts,
    }
    validation = {
        "schema": STAGED_SOURCE_VALIDATION_SCHEMA,
        "fresh_task_generation_protocol_id": DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
        "source_seed": selected_seed,
        "source_seed_selection": selection,
        "source_certification": certification,
        "source_certification_fresh_task_generation": proof_evidence,
        "selected_source_fresh_task_generation": selected_evidence,
        "selected_source_reconstruction": reconstruction,
        "source_certification_and_exported_source_different_generations": True,
        "formal_task_validate_calls": 0,
        "task_descriptions": selected["descriptions"],
        "task_semantic_signature": selected["task_semantics"],
        "source_task_tree_relative_state": selected["task_tree"],
        "selected_source_robot_numeric_state": selected["robot_numeric_state"],
        "selected_source_stable_grasp_state": selected["stable_grasp_state"],
        "source_robot_external_collision_pairs": selected[
            "robot_external_collision_pairs"
        ],
        "selected_source_fingerprint": _staging_source_fingerprint(
            task_name=task_name,
            variation=variation,
            root_pose=selected["root_pose"],
            low_dim_state=selected["low_dim_state"],
            task_tree_state=selected["task_tree"],
            semantic_signature=selected["task_semantics"],
            descriptions=selected["descriptions"],
            collision_pair_records=selected["robot_external_collision_pairs"],
        ),
    }
    return StagedSourcePlan(
        task_name=task_name,
        source_pose=tuple(selected["root_pose"]),
        source_low_dim_state=tuple(selected["low_dim_state"]),
        episode_seed=episode_seed,
        variation=variation,
        validation=validation,
    )


def staged_source_plan_batch(
    *,
    task_name: str,
    task_module: str,
    task_class: str,
    base_seed: int,
    variations: list[int],
    plans: list[StagedSourcePlan],
) -> dict[str, Any]:
    if len(variations) != len(plans):
        raise ValueError("staged source variation schedule length mismatch")
    if any(
        plan.task_name != task_name
        or plan.episode_seed != base_seed + episode
        or plan.variation != variation
        for episode, (variation, plan) in enumerate(zip(variations, plans))
    ):
        raise ValueError("staged source batch schedule is inconsistent")
    body = {
        "schema": STAGED_SOURCE_PLAN_BATCH_SCHEMA,
        "protocol_id": STAGED_SOURCE_PROTOCOL_ID,
        "task_name": task_name,
        "task_module": task_module,
        "task_class": task_class,
        "base_seed": base_seed,
        "episodes": len(plans),
        "variation_schedule": variations,
        "plans": [plan.to_json() for plan in plans],
    }
    return {**body, "batch_fingerprint": _canonical_json_fingerprint(body)}


def load_staged_source_plan_batch(payload: dict[str, Any]) -> list[StagedSourcePlan]:
    if not isinstance(payload, dict):
        raise ValueError("staged source batch must be a dictionary")
    fingerprint = payload.get("batch_fingerprint")
    body = {key: value for key, value in payload.items() if key != "batch_fingerprint"}
    if (
        payload.get("schema") != STAGED_SOURCE_PLAN_BATCH_SCHEMA
        or payload.get("protocol_id") != STAGED_SOURCE_PROTOCOL_ID
        or fingerprint != _canonical_json_fingerprint(body)
        or not isinstance(payload.get("plans"), list)
    ):
        raise ValueError("staged source batch authentication failed")
    plans = [StagedSourcePlan.from_json(row) for row in payload["plans"]]
    expected = staged_source_plan_batch(
        task_name=payload.get("task_name"),
        task_module=payload.get("task_module"),
        task_class=payload.get("task_class"),
        base_seed=payload.get("base_seed"),
        variations=payload.get("variation_schedule"),
        plans=plans,
    )
    if expected != payload:
        raise ValueError("staged source batch fields are inconsistent")
    return plans


def bind_staged_source_plan(
    task_environment: Any,
    plan: StagedSourcePlan,
    *,
    descriptions: Any,
    fresh_task_generation: dict[str, Any],
) -> dict[str, Any]:
    """Strictly bind a formal reset(false) to one sealed coordination A."""

    try:
        _validate_fresh_task_generation_evidence(
            fresh_task_generation,
            episode_seed=plan.validation["source_seed"],
            variation=plan.variation,
            task_name=plan.task_name,
            verify_instance=False,
        )
    except RuntimeError as error:
        raise RuntimeError("formal coordination reset evidence is invalid") from error

    selected = {
        "task_name": plan.task_name,
        "root_pose": np.asarray(plan.source_pose, dtype=np.float64),
        "low_dim_state": np.asarray(plan.source_low_dim_state, dtype=np.float64),
        "task_tree": plan.validation["source_task_tree_relative_state"],
        "task_semantics": plan.validation["task_semantic_signature"],
        "descriptions": plan.validation["task_descriptions"],
        "robot_numeric_state": plan.validation["selected_source_robot_numeric_state"],
        "stable_grasp_state": plan.validation["selected_source_stable_grasp_state"],
        "robot_external_collision_pairs": plan.validation[
            "source_robot_external_collision_pairs"
        ],
        "velocity_summary": {"all_finite": True},
    }
    actual = _source_state_snapshot(task_environment._scene, descriptions)
    # Velocity magnitudes are diagnostic, not source identity.
    selected["velocity_summary"] = actual["velocity_summary"]
    audit = _source_reconstruction_audit(selected, actual)
    if (
        not audit["passed"]
        or _waypoint_cache_evidence(task_environment._scene.task)["state"] != "none"
    ):
        raise RuntimeError("formal coordination source A binding failed")
    return {
        "schema": "dynamac-rlbench-formal-source-a-binding-v1",
        "required": True,
        "matched": True,
        "source_seed": plan.validation["source_seed"],
        "fresh_task_generation": fresh_task_generation,
        "task_validate_calls": 0,
        "source_reconstruction": audit,
        "plan_fingerprint": plan.fingerprint(),
    }


def stage_scenario_motion_plan(
    environment: Any,
    task_class: Any,
    *,
    episode_seed: int,
    variation: int,
    task_name: str | None = None,
    max_attempts: int = 100,
    source_max_attempts: int = DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS,
    goal_candidate_sampler: Any = None,
) -> StagedMotionPlan:
    """Certify deterministic A offline, then sample B from fresh A rebuilds."""

    if max_attempts < 1:
        raise ValueError("staging max attempts must be positive")
    if source_max_attempts < 1:
        raise ValueError("source selection max attempts must be positive")
    if source_max_attempts != DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS:
        raise ValueError("source selection max attempts differ from frozen protocol")
    if goal_candidate_sampler is not None and not callable(goal_candidate_sampler):
        raise TypeError("goal candidate sampler must be callable or None")
    from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
        load_v3_motion_source_protocol,
        motion_source_profile,
    )

    motion_source_protocol = load_v3_motion_source_protocol()
    if (
        motion_source_protocol["source_selection_max_attempts"] != source_max_attempts
        or motion_source_protocol["goal_sampling_max_attempts"] != max_attempts
    ):
        raise RuntimeError("motion-source staging budgets are inconsistent")
    if task_name is None:
        raise ValueError("staging requires a frozen motion-source task name")
    spatial_profile = motion_source_profile(task_name, motion_source_protocol)

    resolved_task_name = task_name
    source_attempt_rows: list[dict[str, Any]] = []
    certified_source = None
    source_certification = None
    certification_generation_evidence = None
    selected_source_seed = None
    selected_source_attempt = None

    for source_attempt in range(1, source_max_attempts + 1):
        candidate_source_seed = _source_seed(
            episode_seed,
            variation,
            source_attempt,
        )
        (
            task_environment,
            descriptions,
            _observation,
            generation_evidence,
        ) = initialize_fresh_task_generation(
            environment,
            task_class,
            episode_seed=candidate_source_seed,
            variation=variation,
            verify_instance=False,
        )
        scene = getattr(task_environment, "_scene", None)
        task = getattr(scene, "task", None)
        if task is None:
            raise RuntimeError("RLBench source-certification task is unavailable")
        get_name = getattr(task, "get_name", None)
        actual_task_name = (
            str(get_name()) if callable(get_name) else type(task).__name__
        )
        if resolved_task_name is None:
            resolved_task_name = actual_task_name
        elif resolved_task_name != actual_task_name:
            raise RuntimeError("staging task identity does not match requested task")
        resolved_source_root = _authenticate_motion_source_root(task, spatial_profile)
        try:
            source_snapshot, certification = _certify_source_a(
                scene,
                descriptions,
                generation_evidence,
            )
        except Exception as error:
            expected = _is_expected_placement_error(error) or str(error) in {
                "reset(false) source workspace placement did not succeed",
                "reset(false) source robot is in collision",
                "reset(false) source placement collision audit failed",
            }
            if not expected:
                raise
            source_attempt_rows.append(
                {
                    "attempt": source_attempt,
                    "source_seed": candidate_source_seed,
                    "resolved_spatial_root": resolved_source_root,
                    "fresh_task_generation": generation_evidence,
                    "accepted": False,
                    "rejection_type": type(error).__name__,
                    "rejection_reason": str(error) or type(error).__name__,
                    "certification_ref": None,
                }
            )
            continue
        source_attempt_rows.append(
            {
                "attempt": source_attempt,
                "source_seed": candidate_source_seed,
                "resolved_spatial_root": resolved_source_root,
                "fresh_task_generation": generation_evidence,
                "accepted": True,
                "rejection_type": None,
                "rejection_reason": None,
                "certification_ref": "validation.source_certification",
            }
        )
        certified_source = source_snapshot
        source_certification = certification
        certification_generation_evidence = generation_evidence
        selected_source_seed = candidate_source_seed
        selected_source_attempt = source_attempt
        break
    if certified_source is None or selected_source_seed is None:
        raise RuntimeError(
            "could not certify a deterministic source A after "
            f"{source_max_attempts} attempts; "
            f"{_source_selection_failure_diagnostic(source_attempt_rows)}"
        )

    # Discard the OMPL-certified generation.  The exported A is a fresh
    # reset(false) reconstruction, matching B retries and formal rollout.
    (
        task_environment,
        selected_descriptions,
        _observation,
        selected_source_generation_evidence,
    ) = initialize_fresh_task_generation(
        environment,
        task_class,
        episode_seed=selected_source_seed,
        variation=variation,
        verify_instance=False,
    )
    selected_scene = task_environment._scene
    selected_resolved_root = _authenticate_motion_source_root(
        selected_scene.task,
        spatial_profile,
    )
    selected_source = _source_state_snapshot(selected_scene, selected_descriptions)
    selected_reconstruction = _source_reconstruction_audit(
        certified_source,
        selected_source,
    )
    if not selected_reconstruction["passed"]:
        raise RuntimeError("selected source seed did not reconstruct certified A")
    selected_reconstruction["resolved_spatial_root"] = selected_resolved_root

    selected_source_fingerprint = _staging_source_fingerprint(
        task_name=resolved_task_name,
        variation=variation,
        root_pose=selected_source["root_pose"],
        low_dim_state=selected_source["low_dim_state"],
        task_tree_state=selected_source["task_tree"],
        semantic_signature=selected_source["task_semantics"],
        descriptions=selected_source["descriptions"],
        collision_pair_records=selected_source["robot_external_collision_pairs"],
    )

    goal_pose = None
    goal_collision_pairs = None
    accepted_source_collision_pairs = None
    goal_pre_validation_task_tree = None
    goal_waypoint_validation_task_tree = None
    task_tree_motion = None
    rigid_motion = None
    goal_certification = None
    goal_attempt_rows: list[dict[str, Any]] = []
    attempts_used = 0
    placement_rejections = 0
    waypoint_rejections = 0
    collision_rejections = 0
    last_error = None

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        (
            task_environment,
            descriptions,
            _observation,
            generation_evidence,
        ) = initialize_fresh_task_generation(
            environment,
            task_class,
            episode_seed=selected_source_seed,
            variation=variation,
            verify_instance=False,
        )
        scene = getattr(task_environment, "_scene", None)
        task = getattr(scene, "task", None)
        workspace_boundary = getattr(scene, "_workspace_boundary", None)
        robot = getattr(scene, "robot", None)
        if task is None or workspace_boundary is None or robot is None:
            raise RuntimeError(
                "RLBench staging scene placement internals are unavailable"
            )
        resolved_candidate_root = _authenticate_motion_source_root(
            task,
            spatial_profile,
        )
        root = task.boundary_root()
        get_name = getattr(task, "get_name", None)
        actual_task_name = (
            str(get_name()) if callable(get_name) else type(task).__name__
        )
        if resolved_task_name is None:
            resolved_task_name = actual_task_name
        elif actual_task_name != resolved_task_name:
            raise RuntimeError("staging task identity does not match requested task")
        current_source = _source_state_snapshot(scene, descriptions)
        reconstruction = _source_reconstruction_audit(selected_source, current_source)
        reconstruction["resolved_spatial_root"] = resolved_candidate_root
        if not reconstruction["passed"]:
            raise RuntimeError("B retry source did not reconstruct selected A")
        source_waypoint_cache = _waypoint_cache_evidence(task)
        if source_waypoint_cache["state"] != "none":
            raise RuntimeError("B retry source unexpectedly materialized waypoints")
        reconstruction["source_waypoint_cache_state"] = "none"
        current_descriptions = current_source["descriptions"]
        current_semantic_signature = current_source["task_semantics"]
        current_task_tree_state = current_source["task_tree"]
        current_source_pose = current_source["root_pose"]
        current_source_low_dim = current_source["low_dim_state"]
        validate = getattr(task, "validate", None)
        if not callable(validate):
            raise RuntimeError("RLBench Task.validate() is unavailable in staging")
        current_source_collisions = _robot_external_collision_pairs(scene, robot)
        initial_orientation = getattr(scene, "_initial_task_pose", None)
        if initial_orientation is None:
            raise RuntimeError("RLBench Scene._initial_task_pose is unavailable")
        min_rotation, max_rotation = task.base_rotation_bounds()
        # B randomness is separate from A randomness, making retries both
        # deterministic and independent of how much RNG reset() consumed.
        candidate_seed = int(
            (episode_seed * 1_000_003 + variation * 9_176 + attempt * 7_919)
            % (2**32 - 1)
        )
        random.seed(candidate_seed)
        np.random.seed(candidate_seed)
        workspace_boundary.clear()
        try:
            if goal_candidate_sampler is None:
                root.set_orientation(initial_orientation)
                workspace_boundary.sample(
                    root,
                    min_rotation=min_rotation,
                    max_rotation=max_rotation,
                )
                placement_succeeded = _workspace_boundary_contains_root(
                    scene,
                    root,
                )
                candidate = np.asarray(root.get_pose(), dtype=np.float64).copy()
            else:
                candidate = np.asarray(
                    goal_candidate_sampler(
                        np.asarray(current_source_pose, dtype=np.float64).copy(),
                        candidate_seed,
                    ),
                    dtype=np.float64,
                )
                if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                    raise RuntimeError(
                        "explicit goal candidate sampler returned an invalid root pose"
                    )
                root.set_pose(candidate)
                candidate = np.asarray(root.get_pose(), dtype=np.float64).copy()
                placement_succeeded = _workspace_boundary_accepts_current_root(
                    scene,
                    root,
                )
            if not placement_succeeded:
                placement_rejections += 1
                last_error = (
                    "workspace sampler did not record a successful goal"
                    if goal_candidate_sampler is None
                    else "goal candidate does not fit the workspace boundary"
                )
                goal_attempt_rows.append(
                    {
                        "attempt": attempt,
                        "candidate_seed": candidate_seed,
                        "source_reconstruction": reconstruction,
                        "workspace_placement_succeeded": False,
                        "outcome": "placement_rejected",
                        "reason": last_error,
                        "fresh_task_generation": generation_evidence,
                    }
                )
                continue
            if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                raise RuntimeError("workspace sampler returned an invalid root pose")
            if not _root_motion_metrics(
                current_source_pose,
                candidate,
            )["planned_root_motion"]:
                placement_rejections += 1
                last_error = "sampled root pose equals its source pose"
                goal_attempt_rows.append(
                    {
                        "attempt": attempt,
                        "candidate_seed": candidate_seed,
                        "source_reconstruction": reconstruction,
                        "workspace_placement_succeeded": True,
                        "outcome": "placement_rejected",
                        "reason": last_error,
                        "fresh_task_generation": generation_evidence,
                    }
                )
                continue
            candidate_collisions = _robot_external_collision_pairs(scene, robot)
            if frozenset(candidate_collisions) - frozenset(current_source_collisions):
                collision_rejections += 1
                last_error = "sampled goal introduces new robot collision pairs"
                goal_attempt_rows.append(
                    {
                        "attempt": attempt,
                        "candidate_seed": candidate_seed,
                        "source_reconstruction": reconstruction,
                        "workspace_placement_succeeded": True,
                        "outcome": "collision_rejected",
                        "reason": last_error,
                        "fresh_task_generation": generation_evidence,
                    }
                )
                continue
            candidate_tree_pre_validation = _task_tree_relative_state(task)
            goal_pre_validation_task_tree = _compare_task_tree_relative_state(
                current_task_tree_state,
                candidate_tree_pre_validation,
                boundary_root_may_move=True,
            )
            if not goal_pre_validation_task_tree["matched"]:
                raise RuntimeError(
                    "staged B sampling changed task-tree topology or state "
                    "outside the commanded boundary-root rigid motion"
                )
            if _task_semantic_signature(task) != current_semantic_signature:
                raise RuntimeError("staging B sampling changed task semantics")
            try:
                candidate_goal_certification = _certify_goal_b(
                    scene,
                    current_descriptions,
                )
            except Exception as error:
                if not _is_expected_placement_error(error):
                    raise
                waypoint_rejections += 1
                last_error = str(error) or type(error).__name__
                goal_attempt_rows.append(
                    {
                        "attempt": attempt,
                        "candidate_seed": candidate_seed,
                        "source_reconstruction": reconstruction,
                        "workspace_placement_succeeded": True,
                        "goal_waypoint_cache": _waypoint_cache_evidence(task),
                        "outcome": "waypoint_rejected",
                        "reason": last_error,
                        "fresh_task_generation": generation_evidence,
                    }
                )
                continue
            candidate_collisions = _robot_external_collision_pairs(scene, robot)
            if frozenset(candidate_collisions) - frozenset(current_source_collisions):
                collision_rejections += 1
                last_error = (
                    "waypoint validation leaves new robot collision pairs at goal"
                )
                goal_attempt_rows.append(
                    {
                        "attempt": attempt,
                        "candidate_seed": candidate_seed,
                        "source_reconstruction": reconstruction,
                        "workspace_placement_succeeded": True,
                        "outcome": "collision_rejected",
                        "reason": last_error,
                        "fresh_task_generation": generation_evidence,
                    }
                )
                continue
            candidate_tree_post_validation = _task_tree_relative_state(task)
            goal_waypoint_cache = _waypoint_cache_evidence(task)
            if goal_waypoint_cache["state"] != "materialized":
                raise RuntimeError("B validation did not materialize waypoints")
            goal_waypoint_validation_task_tree = _compare_task_tree_relative_state(
                candidate_tree_pre_validation,
                candidate_tree_post_validation,
                boundary_root_may_move=False,
            )
            if not goal_waypoint_validation_task_tree["matched"]:
                raise RuntimeError(
                    "staged B waypoint validation changed task-tree topology or state"
                )
            if _task_semantic_signature(task) != current_semantic_signature:
                raise RuntimeError("staging B validation changed task semantics")
            task_tree_motion = _compare_task_tree_relative_state(
                current_task_tree_state,
                candidate_tree_post_validation,
                boundary_root_may_move=True,
            )
            if not task_tree_motion["matched"]:
                raise RuntimeError(
                    "staged B changed task-tree topology or state outside the "
                    "commanded boundary-root rigid motion"
                )
            candidate_low_dim = _task_low_dim_state(task)
            rigid_motion = _task_frame_rigid_motion_evidence(
                resolved_task_name,
                current_source_pose,
                candidate,
                current_source_low_dim,
                candidate_low_dim,
            )
            if not rigid_motion["all_pose_chunks_follow_boundary_root_rigid_transform"]:
                raise RuntimeError(
                    "staged boundary-root motion does not rigidly move every task frame"
                )
            goal_pose = candidate
            goal_certification = candidate_goal_certification
            goal_collision_pairs = candidate_collisions
            accepted_source_collision_pairs = current_source_collisions
            goal_attempt_rows.append(
                {
                    "attempt": attempt,
                    "candidate_seed": candidate_seed,
                    "source_reconstruction": reconstruction,
                    "workspace_placement_succeeded": True,
                    "goal_waypoint_cache": goal_waypoint_cache,
                    "goal_certification_ref": "validation.goal_certification",
                    "outcome": "accepted",
                    "reason": None,
                    "fresh_task_generation": generation_evidence,
                }
            )
            break
        except Exception as error:
            if not _is_expected_placement_error(error):
                raise
            placement_rejections += 1
            last_error = str(error) or type(error).__name__
            goal_attempt_rows.append(
                {
                    "attempt": attempt,
                    "candidate_seed": candidate_seed,
                    "source_reconstruction": reconstruction,
                    "workspace_placement_succeeded": False,
                    "outcome": "placement_rejected",
                    "reason": last_error,
                    "fresh_task_generation": generation_evidence,
                }
            )

    if (
        goal_pose is None
        or goal_collision_pairs is None
        or accepted_source_collision_pairs is None
    ):
        raise RuntimeError(
            "could not sample a waypoint-valid staging goal after "
            f"{max_attempts} attempts; "
            f"{_goal_sampling_failure_diagnostic(goal_attempt_rows)}"
        )
    selected_source_collision_set = frozenset(accepted_source_collision_pairs)
    new_pairs = tuple(
        sorted(frozenset(goal_collision_pairs) - selected_source_collision_set)
    )
    if new_pairs:
        raise RuntimeError("staged goal contains an unvalidated collision pair")
    source_seed_selection = {
        "schema": SOURCE_SEED_SELECTION_SCHEMA,
        "logical_episode_seed": int(episode_seed),
        "seed_derivation": "attempt1=episode_seed;fallback=(episode_seed*1000003+variation*9176+attempt*104729)%4294967295",
        "max_attempts": int(source_max_attempts),
        "attempts_used": int(selected_source_attempt),
        "selected_source_seed": int(selected_source_seed),
        "attempts": source_attempt_rows,
    }
    validation = {
        "schema": STAGED_MOTION_PLAN_VALIDATION_SCHEMA,
        "environment_role": "independent_disposable_staging",
        "fresh_task_generation_protocol_id": DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
        "selected_source_fresh_task_generation": (selected_source_generation_evidence),
        "source_certification_fresh_task_generation": certification_generation_evidence,
        "source_seed": int(selected_source_seed),
        "motion_source_protocol_schema": motion_source_protocol["schema"],
        "motion_source_protocol_fingerprint": motion_source_protocol["fingerprint"],
        "motion_source_profile": spatial_profile,
        "resolved_spatial_root": selected_resolved_root,
        "source_seed_selection": source_seed_selection,
        "source_certification": source_certification,
        "selected_source_reconstruction": selected_reconstruction,
        "goal_candidate_attempts": goal_attempt_rows,
        "formal_rollout_sample_or_restore": False,
        "formal_source_binding_required": True,
        "source_waypoint_validated": True,
        "goal_waypoint_validated": True,
        "goal_certification": goal_certification,
        "waypoint_validation_api": "Task.validate",
        "source_validation": "offline_exactly_one_Task.validate_after_reset_verify_instance_false",
        "task_init_episode_called_by_candidate_sampler": False,
        "task_init_episode_called_by_staging_reset": True,
        "source_certification_and_exported_source_different_generations": True,
        "accepted_candidate_reconstructed_A_and_B_same_instance": True,
        "task_descriptions": selected_source["descriptions"],
        "task_semantic_signature": selected_source["task_semantics"],
        "task_semantic_fingerprint": _canonical_json_fingerprint(
            selected_source["task_semantics"]
        ),
        "selected_source_fingerprint": selected_source_fingerprint,
        "selected_source_task_object_velocity_summary": (
            selected_source["velocity_summary"]
        ),
        "selected_source_robot_numeric_state": selected_source["robot_numeric_state"],
        "selected_source_stable_grasp_state": selected_source["stable_grasp_state"],
        "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
        "source_task_tree_relative_state": selected_source["task_tree"],
        "source_task_tree_object_count": len(selected_source["task_tree"]),
        "source_task_tree_fingerprint": _canonical_json_fingerprint(
            selected_source["task_tree"]
        ),
        "goal_task_tree_relative_state_preserved": task_tree_motion,
        "goal_pre_validation_task_tree_state_preserved": (
            goal_pre_validation_task_tree
        ),
        "goal_waypoint_validation_task_tree_state_preserved": (
            goal_waypoint_validation_task_tree
        ),
        "source_low_dim_size": int(np.asarray(selected_source["low_dim_state"]).size),
        "task_frame_rigid_motion": rigid_motion,
        "candidate_isolation": "fresh_reset_false_same_selected_source_seed_no_restore",
        "task_get_state_called_directly_by_candidate_sampler": False,
        "task_restore_state_called_directly_by_candidate_sampler": False,
        "open_microwave_limit_normalization_false_guard_possible": False,
        "source_robot_external_collision_pairs": _stable_collision_pair_records(
            selected_source_collision_set
        ),
        "goal_robot_external_collision_pairs": _stable_collision_pair_records(
            goal_collision_pairs
        ),
        "goal_new_robot_external_collision_pairs": _stable_collision_pair_records(
            new_pairs
        ),
        "sampling_attempts": attempts_used,
        "goal_sampling_max_attempts": int(max_attempts),
        "selected_candidate_seed": candidate_seed,
        "placement_rejections": placement_rejections,
        "waypoint_rejections": waypoint_rejections,
        "new_collision_pair_rejections": collision_rejections,
    }
    return StagedMotionPlan(
        task_name=resolved_task_name,
        source_pose=tuple(selected_source["root_pose"]),
        goal_pose=tuple(goal_pose),
        source_low_dim_state=tuple(selected_source["low_dim_state"]),
        episode_seed=int(episode_seed),
        variation=int(variation),
        validation=validation,
    )


def _restore_sampling_configuration(
    *,
    task: Any,
    task_state: Any,
    workspace_boundary: Any,
) -> None:
    """Restore only the temporarily moved task hierarchy, then clear sampling.

    Live force-controlled robot trees are deliberately never read or restored.
    CoppeliaSim configuration-tree restoration can snap a loaded joint from its
    instantaneous physical position to its control target, moving any grasped
    object even when the sampler never moved the robot.
    """

    errors = []
    try:
        task.restore_state(task_state)
    except Exception as error:  # pragma: no cover - defensive aggregation
        errors.append(("task", error))
    try:
        workspace_boundary.clear()
    except Exception as error:  # pragma: no cover - defensive aggregation
        errors.append(("workspace boundary", error))
    if errors:
        scope, error = errors[0]
        raise RuntimeError(
            f"failed to restore {scope} configuration tree after goal sampling"
        ) from error


def _sample_preserving_instance_goal(
    scene: Any,
    *,
    max_attempts: int,
) -> tuple[Array, Array, dict[str, Any]]:
    """Sample a valid root goal without reinitializing the current episode.

    The workspace sampler needs to move the real root in order to evaluate its
    bounding box and collisions. Each attempt is therefore transactional, but
    only the task configuration tree is restored. The live robot is queried for
    source/candidate external collision pairs and grasp auditing, never moved,
    validated through task waypoints, or configuration-tree restored. The
    task's low-dimensional state, condition/grasp registries, grasp membership,
    relevant parent handles, and waypoint-cache identity are checked at the
    restored source pose. A failed preservation check is fatal; silently
    continuing would mix two different episode instances.
    """

    task = getattr(scene, "task", None)
    workspace_boundary = getattr(scene, "_workspace_boundary", None)
    robot = getattr(scene, "robot", None)
    if task is None or workspace_boundary is None or robot is None:
        raise RuntimeError("RLBench scene placement internals are unavailable")

    root = task.boundary_root()
    source_pose = np.asarray(root.get_pose(), dtype=np.float64).copy()
    if source_pose.shape != (7,) or not np.all(np.isfinite(source_pose)):
        raise RuntimeError("boundary root returned an invalid source pose")
    initial_orientation = getattr(scene, "_initial_task_pose", None)
    if initial_orientation is None:
        raise RuntimeError("RLBench Scene._initial_task_pose is unavailable")

    before_state = _task_low_dim_state(task)
    before_references = _instance_reference_snapshot(task)
    get_task_state = getattr(task, "get_state", None)
    restore_task_state = getattr(task, "restore_state", None)
    if not callable(get_task_state) or not callable(restore_task_state):
        raise RuntimeError("RLBench task configuration-tree API is unavailable")
    task_state = get_task_state()
    before_grasp_state = _grasp_state_snapshot(task, robot)
    source_collision_pairs = _robot_external_collision_pairs(scene, robot)
    source_collision_pair_set = frozenset(source_collision_pairs)
    waypoint_sentinel = object()
    before_waypoints = getattr(task, "_waypoints", waypoint_sentinel)
    min_rotation, max_rotation = task.base_rotation_bounds()
    goal_pose = None
    attempts_used = 0
    last_placement_error = None
    goal_collision_pairs = None
    new_collision_pair_rejections = 0

    try:
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            workspace_boundary.clear()
            try:
                # This is the placement part of Scene._place_task, deliberately
                # separated from kidnap()/init_episode().
                root.set_orientation(initial_orientation)
                workspace_boundary.sample(
                    root,
                    min_rotation=min_rotation,
                    max_rotation=max_rotation,
                )
                candidate = np.asarray(root.get_pose(), dtype=np.float64).copy()
                if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                    raise RuntimeError(
                        "workspace sampler returned an invalid root pose"
                    )
                if not _root_motion_metrics(source_pose, candidate)[
                    "planned_root_motion"
                ]:
                    last_placement_error = "sampled root pose equals its source pose"
                    continue
                candidate_collision_pairs = _robot_external_collision_pairs(
                    scene,
                    robot,
                )
                new_collision_pairs = tuple(
                    sorted(
                        frozenset(candidate_collision_pairs) - source_collision_pair_set
                    )
                )
                if new_collision_pairs:
                    new_collision_pair_rejections += 1
                    last_placement_error = (
                        "sampled task root introduces "
                        f"{len(new_collision_pairs)} new robot collision pair(s)"
                    )
                    continue
                goal_pose = candidate
                goal_collision_pairs = candidate_collision_pairs
                break
            except Exception as error:
                if not _is_expected_placement_error(error):
                    raise
                last_placement_error = str(error) or type(error).__name__
            finally:
                # A 7D root round trip is insufficient: moving the root resets
                # dynamic descendants and can introduce float32 drift. Restore
                # the complete task tree, but never touch live robot trees.
                _restore_sampling_configuration(
                    task=task,
                    task_state=task_state,
                    workspace_boundary=workspace_boundary,
                )
    finally:
        _restore_sampling_configuration(
            task=task,
            task_state=task_state,
            workspace_boundary=workspace_boundary,
        )

    restored_state = _task_low_dim_state(task)
    if before_state.shape != restored_state.shape:
        raise RuntimeError("goal sampling changed task low-dimensional state schema")
    roundtrip = _low_dim_roundtrip_metrics(before_state, restored_state)
    state_preserved = bool(roundtrip["preserved"])
    references_preserved = _same_instance_references(
        before_references,
        _instance_reference_snapshot(task),
    )
    grasp_state_preserved = _same_grasp_state(before_grasp_state, task, robot)
    after_waypoints = getattr(task, "_waypoints", waypoint_sentinel)
    waypoint_cache_preserved = after_waypoints is before_waypoints
    if not state_preserved:
        raise RuntimeError(
            "goal sampling changed the initialized task instance's low-dimensional "
            f"state beyond {roundtrip['comparison_mode']} tolerance "
            f"(raw max {roundtrip['raw_max_abs']:.9g}, "
            f"translation {roundtrip['max_translation_m']}, "
            f"rotation {roundtrip['max_rotation_rad']})"
        )
    if not references_preserved:
        raise RuntimeError(
            "goal sampling replaced task success/failure/grasp registry objects"
        )
    if not grasp_state_preserved:
        raise RuntimeError(
            "goal sampling changed gripper grasp membership or object parents"
        )
    if not waypoint_cache_preserved:
        raise RuntimeError("goal sampling changed the task waypoint cache")
    if goal_pose is None:
        detail = last_placement_error or "no valid root goal was sampled"
        raise RuntimeError(
            "could not sample a preserve-instance task-root goal after "
            f"{max_attempts} attempts: {detail}"
        )
    if goal_collision_pairs is None:  # pragma: no cover - guarded by goal_pose
        raise RuntimeError("goal collision-pair evidence is unavailable")

    selected_new_collision_pairs = tuple(
        sorted(frozenset(goal_collision_pairs) - source_collision_pair_set)
    )
    if selected_new_collision_pairs:  # pragma: no cover - loop rejects these
        raise RuntimeError("selected goal contains an unvalidated collision pair")

    preservation = {
        "initialized_episode_preserved": True,
        "task_init_episode_called": False,
        "task_validate_called": False,
        "low_dim_state_roundtrip_preserved": state_preserved,
        "low_dim_state_roundtrip_comparison_mode": roundtrip["comparison_mode"],
        "low_dim_state_roundtrip_chunk_count": roundtrip["chunk_count"],
        "low_dim_state_roundtrip_l2": roundtrip["raw_l2"],
        "low_dim_state_roundtrip_max_abs": roundtrip["raw_max_abs"],
        "low_dim_state_roundtrip_max_translation_m": roundtrip["max_translation_m"],
        "low_dim_state_roundtrip_max_rotation_rad": roundtrip["max_rotation_rad"],
        "condition_and_grasp_registry_identity_preserved": references_preserved,
        "gripper_grasp_membership_and_parentage_preserved": (grasp_state_preserved),
        "configuration_tree_rollback": "task_only_after_each_attempt_and_outer_finally",
        "task_configuration_tree_restored": True,
        "live_robot_state_untouched": True,
        "live_robot_configuration_trees_accessed": False,
        "robot_collision_pair_policy": (
            "reject_candidate_external_pairs_absent_at_source"
        ),
        "robot_collision_pair_granularity": (
            "named_arm_collection_x_external_collidable_scene_shape"
        ),
        "source_robot_external_collision_pairs": _collision_pair_records(
            source_collision_pairs
        ),
        "goal_robot_external_collision_pairs": _collision_pair_records(
            goal_collision_pairs
        ),
        "goal_new_robot_external_collision_pairs": _collision_pair_records(
            selected_new_collision_pairs
        ),
        "sampling_attempts_rejected_for_new_robot_collision_pairs": (
            new_collision_pair_rejections
        ),
        "sampling_attempts": attempts_used,
        "waypoint_cache_identity_preserved": waypoint_cache_preserved,
    }
    return source_pose, goal_pose, preservation


@dataclass
class ScenarioController:
    """Move a task root while preserving the already initialized episode.

    In the pinned fork, both ``Scene.kidnap`` and
    ``Scene.move_task_smoothly`` call ``task.init_episode`` while choosing a
    destination.  That changes task-internal random objects and success
    conditions, so neither method is a valid motion intervention on one
    episode instance. This controller samples a workspace-fitting
    ``boundary_root`` pose, rejects external robot collision pairs absent at
    the source, restores only the task hierarchy transactionally, and then
    applies teleportation or interpolation itself. It never runs waypoint
    validation against, or restores configuration trees for, the live robot.
    """

    kind: Literal["static", "teleport_task", "smooth_task_motion"]
    trigger_fraction: float = 1.0 / 3.0
    trigger_step: int | None = None
    total_steps: int = 10
    max_attempts: int = 100
    verify_instance: bool = True
    motion_plan: StagedMotionPlan | None = None
    _teleported: bool = False
    _smooth_calls: int = 0
    _smooth_complete: bool = False
    _motion_source_pose: Any = None
    _motion_goal_pose: Any = None
    _instance_preservation: Any = None
    _staged_source_bound: bool = False
    _last_motion_policy_step: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must lie in [0, 1]")
        if self.trigger_step is not None and (
            isinstance(self.trigger_step, bool)
            or not isinstance(self.trigger_step, int)
            or self.trigger_step < 0
        ):
            raise ValueError("trigger_step must be a non-negative integer or None")
        if self.total_steps < 1 or self.max_attempts < 1:
            raise ValueError("total_steps and max_attempts must be positive")
        if self.verify_instance is not True:
            raise ValueError("preserve-instance auditing cannot be disabled")
        if self.motion_plan is not None:
            if not isinstance(self.motion_plan, StagedMotionPlan):
                raise TypeError("motion_plan must be a StagedMotionPlan")
            if self.motion_plan.validation.get("v4_lift_tray") is not None:
                from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import (
                    V4_LIFT_TRIGGER_STEP,
                    validate_v4_lift_motion_plan,
                )

                validate_v4_lift_motion_plan(self.motion_plan)
                if self.kind != "static" and self.trigger_step != V4_LIFT_TRIGGER_STEP:
                    raise ValueError("V4 LiftTray dynamic trigger must be tick 35")
            self._motion_source_pose = np.asarray(
                self.motion_plan.source_pose,
                dtype=np.float64,
            )
            self._motion_goal_pose = np.asarray(
                self.motion_plan.goal_pose,
                dtype=np.float64,
            )
            self._instance_preservation = {
                "motion_plan_fingerprint": self.motion_plan.fingerprint(),
                "validation_fingerprint": _canonical_json_fingerprint(
                    self.motion_plan.validation
                ),
            }

    def protocol_metadata(self) -> dict[str, Any]:
        """Return JSON-stable semantics shared by both evaluator frontends."""

        if self.motion_plan is not None:
            plan_validation = getattr(self.motion_plan, "validation", {})
            v4_lift = (
                plan_validation.get("v4_lift_tray")
                if isinstance(plan_validation, dict)
                else None
            )
            if v4_lift is not None:
                from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import (
                    V4_LIFT_MOTION_PROTOCOL_ID,
                )

                return {
                    "protocol_id": V4_LIFT_MOTION_PROTOCOL_ID,
                    "release": "v4",
                    "task": "bimanual_lift_tray",
                    "episode_instance_semantics": (
                        "offline_certified_source_seed_strictly_reconstructed_by_reset_false"
                    ),
                    "goal_object": "task.boundary_root()",
                    "goal_sampling": (
                        "source_relative_world_xy_radial_candidate_scene_validity_only"
                    ),
                    "translation_radius_m": [0.03, 0.08],
                    "z_delta_m": 0.0,
                    "yaw_delta_abs_max_rad": 0.10,
                    "roll_pitch": "unchanged_from_source_A",
                    "formal_scenarios": ["static", "teleport", "smooth"],
                    "trigger_clock": "committed_policy_ticks",
                    "trigger_step": 35,
                    "formal_intervention_sampling": False,
                    "formal_intervention_task_get_state": False,
                    "formal_intervention_task_restore_state": False,
                    "result_based_candidate_selection": False,
                    "stage6_smooth_background_extension": (
                        self.kind == "smooth_task_motion"
                    ),
                    "task_scoped_plan_evidence": dict(v4_lift),
                }
            return {
                "protocol_id": STAGED_VALIDATED_MOTION_PROTOCOL_ID,
                "episode_instance_semantics": (
                    "offline_certified_source_seed_strictly_reconstructed_by_reset_false"
                ),
                "goal_object": "task.boundary_root()",
                "goal_sampling": "independent_disposable_staging_environment",
                "goal_sampling_max_attempts": self.max_attempts,
                "source_selection_max_attempts": DEFAULT_SOURCE_SELECTION_MAX_ATTEMPTS,
                "fresh_task_generation_protocol_id": (
                    DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
                ),
                "candidate_isolation": (
                    "prestop_unload_if_present_stop_physics_fresh_task_"
                    "reload_single_reset_verify_instance_false"
                ),
                "candidate_source_policy": (
                    "every_candidate_reconstructs_selected_source_seed_A"
                ),
                "source_reconstruction_schema": SOURCE_RECONSTRUCTION_SCHEMA,
                "source_reconstruction_tolerances": {
                    "translation_m": SOURCE_RECONSTRUCTION_TRANSLATION_TOLERANCE_M,
                    "rotation_rad": SOURCE_RECONSTRUCTION_ROTATION_TOLERANCE_RAD,
                    "scalar_state": SOURCE_RECONSTRUCTION_SCALAR_TOLERANCE,
                    "joint_position": SOURCE_RECONSTRUCTION_JOINT_TOLERANCE,
                },
                "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
                "task_object_velocity_role": (
                    "finite_diagnostic_not_identity_comparison"
                ),
                "staging_waypoint_validation": (
                    "A_seed_offline_via_exactly_one_explicit_Task.validate_then_discard;"
                    "B_via_fresh_source_reconstruction_and_exactly_one_Task.validate"
                ),
                "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
                "source_replay_task_tree_comparison": "all_objects_world",
                "root_motion_task_tree_comparison": (
                    "boundary_root_subtree_relative_else_world"
                ),
                "goal_validation_task_tree_comparison": "all_objects_world",
                "formal_episode_initialization": (
                    "prestop_unload_if_present_stop_physics_fresh_task_"
                    "reload_then_selected_source_seed_variation_single_reset_verify_instance_false"
                ),
                "calls_task_validate_after_formal_reset": False,
                "formal_intervention_sampling": False,
                "formal_intervention_task_get_state": False,
                "formal_intervention_task_restore_state": False,
                "formal_rollout_receives": "immutable_numeric_A_B_poses_and_fingerprint",
                "formal_intervention_state_audit_schema": (
                    FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA
                ),
                "formal_intervention_state_reference": (
                    "current_policy_evolved_formal_state_immediately_before_"
                    "each_root_command"
                ),
                "formal_intervention_task_tree_comparison": (
                    "strict_1e-6_boundary_root_subtree_relative_else_world"
                ),
                "formal_intervention_semantic_guard": "exact",
                "formal_intervention_registry_and_grasp_guard": (
                    "same_python_identities_membership_and_parentage"
                ),
                "formal_intervention_collision_audit": (
                    FORMAL_INTERVENTION_COLLISION_PAIR_POLICY
                ),
                "robot_collision_validation": (
                    "reject_candidate_external_pairs_absent_at_source"
                ),
                "trigger_clock": "committed_policy_ticks",
                "duplicate_policy_tick_motion": False,
                "smooth_schedule": "one_fraction_per_unique_committed_policy_tick",
                "smooth_fractions": "1_over_n_through_n_over_n",
                "smooth_endpoint_validation": "final_goal_pose_reached",
                "calls_task_init_episode_from_sampler": False,
                "calls_scene_kidnap": False,
                "calls_scene_move_task_smoothly": False,
            }

        return {
            "protocol_id": PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
            "episode_instance_semantics": "preserve_initialized_episode",
            "goal_object": "task.boundary_root()",
            "goal_sampling": "scene_workspace_boundary_without_task_reinitialization",
            "sampling_rollback": "task_configuration_tree_only_live_robot_untouched",
            "sampling_rollback_frequency": "after_each_attempt_and_outer_finally",
            "task_configuration_tree_restore_api": "Task.get_state/restore_state",
            "task_tree_object_count_guard": True,
            "live_robot_state_during_goal_sampling": "untouched",
            "live_robot_configuration_tree_access": "none",
            "online_task_waypoint_validation": "disabled_to_preserve_live_robot_state",
            "calls_task_validate": False,
            "grasp_membership_and_parentage_audited": True,
            "robot_collision_validation": (
                "reject_candidate_external_pairs_absent_at_source"
            ),
            "robot_collision_pair_granularity": (
                "named_arm_collection_x_external_collidable_scene_shape"
            ),
            "source_robot_contacts_allowed": True,
            "grasped_tool_collision_semantics": (
                "current_arm_collection_membership_without_task_filters"
            ),
            "self_collision_semantics": (
                "current_arm_collection_members_excluded_matching_all_other"
            ),
            "low_dim_state_roundtrip_comparison": (
                "valid_pose_chunks_sign_invariant_else_scalar_max_abs"
            ),
            "low_dim_state_roundtrip_scalar_tolerance": (LOW_DIM_STATE_ROUNDTRIP_ATOL),
            "low_dim_state_roundtrip_pose_translation_tolerance_m": (
                LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
            ),
            "low_dim_state_roundtrip_pose_rotation_tolerance_rad": (
                LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
            ),
            "root_application_validation": (
                "planned_motion_and_actual_motion_and_commanded_pose_reached"
            ),
            "root_actual_motion_translation_tolerance_m": (
                ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
            ),
            "root_actual_motion_rotation_tolerance_rad": (
                ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
            ),
            "root_command_translation_tolerance_m": (
                ROOT_COMMAND_TRANSLATION_TOLERANCE_M
            ),
            "root_command_rotation_tolerance_rad": (
                ROOT_COMMAND_ROTATION_TOLERANCE_RAD
            ),
            "dynamic_state_note": (
                "the task configuration tree restores task poses and joints and "
                "resets task dynamics; live robot trees remain untouched; the "
                "subsequent root-motion intervention resets moved task dynamics"
            ),
            "goal_validation": ("workspace_fit_no_new_robot_external_collision_pairs"),
            "calls_task_init_episode": False,
            "calls_scene_kidnap": False,
            "calls_scene_move_task_smoothly": False,
            "smooth_schedule": "fractions_1_over_n_through_n_over_n",
            "smooth_endpoint_validation": "final_goal_pose_reached",
            "smooth_endpoint_guaranteed": True,
        }

    def resolved_trigger_step(self, horizon: int) -> int:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if self.trigger_step is not None:
            if self.trigger_step >= horizon:
                raise ValueError("explicit trigger_step lies outside policy horizon")
            return self.trigger_step
        return min(
            horizon - 1,
            int(round(self.trigger_fraction * (horizon - 1))),
        )

    def bind_staged_source(
        self,
        task_environment: Any,
        *,
        episode_seed: int,
        variation: int,
        descriptions: Any = None,
    ) -> dict[str, Any]:
        """Bind formal reset(false) A to the independently certified source."""

        if self.motion_plan is None:
            return {"required": False, "matched": None}
        if episode_seed != self.motion_plan.episode_seed:
            raise RuntimeError("formal episode seed does not match staged motion plan")
        if variation != self.motion_plan.variation:
            raise RuntimeError("formal variation does not match staged motion plan")
        scene = getattr(task_environment, "_scene", None)
        task = getattr(scene, "task", None)
        if task is None:
            raise RuntimeError("formal RLBench scene task is unavailable")
        get_name = getattr(task, "get_name", None)
        formal_task_name = (
            str(get_name()) if callable(get_name) else type(task).__name__
        )
        if formal_task_name != self.motion_plan.task_name:
            raise RuntimeError("formal task identity does not match staged motion plan")
        validation = self.motion_plan.validation
        if validation.get("v4_lift_tray") is not None:
            from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import (
                validate_v4_lift_motion_plan,
            )

            validate_v4_lift_motion_plan(self.motion_plan)
        from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
            load_v3_motion_source_protocol,
            motion_source_profile,
        )

        motion_source_protocol = load_v3_motion_source_protocol()
        spatial_profile = motion_source_profile(
            self.motion_plan.task_name,
            motion_source_protocol,
        )
        if (
            validation.get("motion_source_protocol_schema")
            != motion_source_protocol["schema"]
            or validation.get("motion_source_protocol_fingerprint")
            != motion_source_protocol["fingerprint"]
            or validation.get("motion_source_profile") != spatial_profile
            or validation.get("resolved_spatial_root")
            != _authenticate_motion_source_root(task, spatial_profile)
        ):
            raise RuntimeError("staged spatial motion-source profile is invalid")
        if (
            validation.get("fresh_task_generation_protocol_id")
            != DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
        ):
            raise RuntimeError("staged fresh task-generation protocol is invalid")
        source_seed = validation.get("source_seed")
        if isinstance(source_seed, bool) or not isinstance(source_seed, int):
            raise RuntimeError("staged source seed is invalid")
        selected_generation_evidence = _validate_fresh_task_generation_evidence(
            validation.get("selected_source_fresh_task_generation"),
            episode_seed=source_seed,
            variation=variation,
            task_name=formal_task_name,
            verify_instance=False,
        )
        formal_generation_evidence = _validate_fresh_task_generation_evidence(
            getattr(
                task_environment,
                "_dynamac_fresh_generation_evidence",
                None,
            ),
            episode_seed=source_seed,
            variation=variation,
            task_name=formal_task_name,
            verify_instance=False,
        )
        certification = validation.get("source_certification")
        selected_reconstruction = validation.get("selected_source_reconstruction")
        if (
            not isinstance(certification, dict)
            or certification.get("schema") != SOURCE_CERTIFICATION_SCHEMA
            or certification.get("passed") is not True
            or not isinstance(selected_reconstruction, dict)
            or selected_reconstruction.get("schema") != SOURCE_RECONSTRUCTION_SCHEMA
            or selected_reconstruction.get("passed") is not True
        ):
            raise RuntimeError("staged source certification proof is invalid")
        expected_descriptions = validation.get("task_descriptions")
        if descriptions is None or list(descriptions) != expected_descriptions:
            raise RuntimeError(
                "formal task descriptions do not match staged motion plan"
            )
        if validation.get("task_tree_state_schema") != TASK_TREE_STATE_SCHEMA:
            raise RuntimeError("staged task-tree state schema is invalid")
        if _waypoint_cache_evidence(task)["state"] != "none":
            raise RuntimeError(
                "formal reset(false) unexpectedly materialized waypoints"
            )
        expected_source = {
            "task_name": self.motion_plan.task_name,
            "root_pose": np.asarray(self.motion_plan.source_pose, dtype=np.float64),
            "low_dim_state": np.asarray(
                self.motion_plan.source_low_dim_state,
                dtype=np.float64,
            ),
            "task_tree": validation.get("source_task_tree_relative_state"),
            "task_semantics": validation.get("task_semantic_signature"),
            "descriptions": expected_descriptions,
            "robot_numeric_state": validation.get(
                "selected_source_robot_numeric_state"
            ),
            "stable_grasp_state": validation.get("selected_source_stable_grasp_state"),
            "robot_external_collision_pairs": validation.get(
                "source_robot_external_collision_pairs"
            ),
            "velocity_summary": validation.get(
                "selected_source_task_object_velocity_summary"
            ),
        }
        formal_source = _source_state_snapshot(scene, descriptions)
        formal_reconstruction = _source_reconstruction_audit(
            expected_source,
            formal_source,
        )
        if not formal_reconstruction["passed"]:
            raise RuntimeError("formal source A did not reconstruct certified A")
        selected_source_fingerprint = _staging_source_fingerprint(
            task_name=self.motion_plan.task_name,
            variation=self.motion_plan.variation,
            root_pose=np.asarray(self.motion_plan.source_pose, dtype=np.float64),
            low_dim_state=np.asarray(
                self.motion_plan.source_low_dim_state,
                dtype=np.float64,
            ),
            task_tree_state=expected_source["task_tree"],
            semantic_signature=expected_source["task_semantics"],
            descriptions=expected_descriptions,
            collision_pair_records=expected_source["robot_external_collision_pairs"],
        )
        if selected_source_fingerprint != validation.get("selected_source_fingerprint"):
            raise RuntimeError("staged selected-source fingerprint is invalid")
        formal_source_fingerprint = _staging_source_fingerprint(
            task_name=formal_task_name,
            variation=variation,
            root_pose=formal_source["root_pose"],
            low_dim_state=formal_source["low_dim_state"],
            task_tree_state=formal_source["task_tree"],
            semantic_signature=formal_source["task_semantics"],
            descriptions=list(descriptions),
            collision_pair_records=formal_source["robot_external_collision_pairs"],
        )
        self._staged_source_bound = True
        return {
            "required": True,
            "matched": True,
            "formal_source_bound": True,
            "source_seed": source_seed,
            "root": formal_reconstruction["root"],
            "low_dim_state": formal_reconstruction["low_dim_state"],
            "deterministic_source_reconstruction": formal_reconstruction,
            "formal_sampling_or_restore": False,
            "formal_task_validate_calls": 0,
            "formal_waypoint_cache_state": "none",
            "staging_source_certification_reused": True,
            "fresh_task_generation_protocol_id": (
                DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
            ),
            "selected_source_fresh_task_generation": (selected_generation_evidence),
            "formal_source_fresh_task_generation": formal_generation_evidence,
            "task_name": formal_task_name,
            "task_semantics_matched": formal_reconstruction["exact_matches"][
                "task_semantics"
            ],
            "task_tree_matched": formal_reconstruction["task_tree"]["matched"],
            "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
            "task_tree_match": formal_reconstruction["task_tree"],
            "task_descriptions_matched": formal_reconstruction["exact_matches"][
                "descriptions"
            ],
            "robot_external_collision_pairs_matched": (
                formal_reconstruction["exact_matches"]["robot_external_collision_pairs"]
            ),
            "selected_source_fingerprint": selected_source_fingerprint,
            "formal_source_fingerprint": formal_source_fingerprint,
            "motion_plan_fingerprint": self.motion_plan.fingerprint(),
        }

    def _ensure_motion_plan(self, scene: Any) -> None:
        if self._motion_goal_pose is not None:
            return
        source, goal, preservation = _sample_preserving_instance_goal(
            scene,
            max_attempts=self.max_attempts,
        )
        self._motion_source_pose = source
        self._motion_goal_pose = goal
        self._instance_preservation = preservation

    def apply(
        self, task_environment: Any, *, step: int, horizon: int
    ) -> dict[str, Any]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("committed policy step must be a non-negative integer")
        trigger = self.resolved_trigger_step(horizon)
        protocol = self.protocol_metadata()
        event_protocol = (
            {
                "protocol_id": protocol["protocol_id"],
                "metadata_fingerprint": _canonical_json_fingerprint(protocol),
            }
            if self.motion_plan is not None
            else protocol
        )
        event: dict[str, Any] = {
            "kind": self.kind,
            "step": step,
            "trigger_step": trigger,
            "applied": False,
            "clock_domain": "committed_policy_ticks",
            "motion_protocol": event_protocol,
        }
        if self.kind == "static" or step < trigger:
            return event
        if self.motion_plan is not None and not self._staged_source_bound:
            raise RuntimeError(
                "staged motion plan must be bound immediately after formal reset"
            )
        if self.kind == "teleport_task" and self._teleported:
            return event
        if self.kind == "smooth_task_motion" and self._smooth_complete:
            return event
        if self._last_motion_policy_step == step:
            event["duplicate_policy_tick_suppressed"] = True
            return event
        if (
            self._last_motion_policy_step is not None
            and step < self._last_motion_policy_step
        ):
            raise RuntimeError("committed policy clock moved backwards")
        scene = getattr(task_environment, "_scene", None)
        if scene is None:
            raise RuntimeError("author RLBench TaskEnvironment._scene is unavailable")
        task = scene.task
        root = task.boundary_root()
        before_state = _task_low_dim_state(task)
        before_root = np.asarray(root.get_pose(), dtype=np.float64)
        formal_state_before = (
            _formal_intervention_state_snapshot(scene)
            if self.motion_plan is not None
            else None
        )
        if self.motion_plan is not None and self._last_motion_policy_step is None:
            source_match = _pose_reproducibility_metrics(
                self._motion_source_pose,
                before_root,
                translation_tolerance_m=ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
                rotation_tolerance_rad=ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
            )
            if not source_match["preserved"]:
                raise RuntimeError(
                    "formal boundary root changed before staged intervention"
                )
        if self.kind == "teleport_task":
            self._ensure_motion_plan(scene)
            commanded_pose = np.asarray(self._motion_goal_pose, dtype=np.float64)
            root.set_pose(commanded_pose)
            if formal_state_before is not None:
                event["formal_intervention_state_audit"] = (
                    _formal_intervention_state_audit(scene, formal_state_before)
                )
            self._last_motion_policy_step = step
            self._teleported = True
            event["applied"] = True
            after_state = _task_low_dim_state(task)
            after_root = np.asarray(root.get_pose(), dtype=np.float64)
            event.update(
                _intervention_change(before_state, after_state, before_root, after_root)
            )
            event.update(
                _root_motion_metrics(self._motion_source_pose, self._motion_goal_pose)
            )
            event.update(
                _root_application_metrics(before_root, commanded_pose, after_root)
            )
            event.update(_root_goal_reached_metrics(commanded_pose, after_root))
            event["protocol_effective"] = bool(
                event["planned_root_motion"]
                and event["actual_root_motion"]
                and event["commanded_root_pose_reached"]
            )
            if self.motion_plan is not None:
                event["motion_plan_reference"] = dict(self._instance_preservation)
            else:
                event["instance_preservation"] = dict(self._instance_preservation)
            return event
        if self.kind == "smooth_task_motion":
            self._ensure_motion_plan(scene)
            next_smooth_calls = self._smooth_calls + 1
            next_smooth_complete = next_smooth_calls >= self.total_steps
            fraction = min(next_smooth_calls / float(self.total_steps), 1.0)
            if next_smooth_complete:
                # Use the sampled value verbatim at the endpoint; interpolation
                # roundoff must not leave the task fractionally short of goal.
                next_pose = np.asarray(self._motion_goal_pose, dtype=np.float64)
            else:
                next_pose = _interpolate_rlbench_pose(
                    self._motion_source_pose,
                    self._motion_goal_pose,
                    fraction,
                )
            root.set_pose(next_pose)
            if formal_state_before is not None:
                event["formal_intervention_state_audit"] = (
                    _formal_intervention_state_audit(scene, formal_state_before)
                )
            self._last_motion_policy_step = step
            self._smooth_calls = next_smooth_calls
            self._smooth_complete = next_smooth_complete
            after_state = _task_low_dim_state(task)
            after_root = np.asarray(root.get_pose(), dtype=np.float64)
            application = _root_application_metrics(
                before_root,
                next_pose,
                after_root,
            )
            goal_reached = _root_goal_reached_metrics(
                self._motion_goal_pose,
                after_root,
            )
            event.update(
                {
                    "applied": True,
                    "smooth_call": self._smooth_calls,
                    "complete": self._smooth_complete,
                    "endpoint_applied": bool(
                        self._smooth_complete and goal_reached["goal_root_pose_reached"]
                    ),
                    "endpoint_fraction": fraction,
                }
            )
            if self.motion_plan is not None:
                event["motion_plan_reference"] = dict(self._instance_preservation)
            else:
                event["instance_preservation"] = dict(self._instance_preservation)
            event.update(
                _intervention_change(
                    before_state,
                    after_state,
                    before_root,
                    after_root,
                )
            )
            event.update(
                _root_motion_metrics(self._motion_source_pose, self._motion_goal_pose)
            )
            event.update(application)
            event.update(goal_reached)
            event["protocol_effective"] = bool(
                event["planned_root_motion"]
                and event["actual_root_motion"]
                and event["commanded_root_pose_reached"]
            )
            return event
        raise ValueError(f"unsupported scenario kind: {self.kind}")


def _intervention_change(
    before_state: Array,
    after_state: Array,
    before_root: Array,
    after_root: Array,
) -> dict[str, Any]:
    if before_state.shape != after_state.shape:
        raise RuntimeError("dynamic intervention changed task-state schema")
    state_l2 = float(np.linalg.norm(after_state - before_state))
    root_l2 = float(np.linalg.norm(after_root - before_root))
    return {
        "task_state_l2": state_l2,
        "task_state_changed": bool(state_l2 > 1.0e-9),
        "root_pose_l2": root_l2,
        "root_pose_changed": bool(root_l2 > 1.0e-9),
    }
