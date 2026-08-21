"""Task-scoped V4 LiftTray and coordination intervention protocols.

This module intentionally has no RLBench/PyRep imports at module load time so
the protocol and plan-envelope checks can run in the policy Python process.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
V4_LIFT_INTERVENTION_CONFIG = (
    INTEGRATION_ROOT / "configs" / "v4" / "lift_tray_intervention.json"
)
V4_LIFT_MOTION_SOURCE_CONFIG = (
    INTEGRATION_ROOT / "configs" / "v4" / "lift_tray_motion_source.json"
)
V4_COORDINATION_INTERVENTION_CONFIG = (
    INTEGRATION_ROOT / "configs" / "v4" / "coordination_intervention.json"
)

V4_LIFT_TASK = "bimanual_lift_tray"
V4_LIFT_INTERVENTION_SCHEMA = "rlbench-dynamac-lift-tray-intervention-v4"
V4_LIFT_MOTION_SOURCE_SCHEMA = "rlbench-dynamac-lift-tray-motion-source-v4"
V4_LIFT_PLAN_EVIDENCE_SCHEMA = "dynamac-lift-tray-plan-evidence-v4"
V4_LIFT_RUNTIME_LOADER_ID = "lift-tray-source-relative-motion-plan-batch-v4"
V4_LIFT_MOTION_PROTOCOL_ID = (
    "lift-tray-source-relative-radial-teleport-fixed-trigger-v4"
)
V4_COORDINATION_INTERVENTION_SCHEMA = (
    "rlbench-dynamac-coordination-intervention-v4"
)
V4_COORDINATION_PROTOCOL_ID = (
    "handover-policy-clocked-world-z-smooth-persistent-offset-global-ik-v6"
)

V4_LIFT_TRIGGER_STEP = 35
V4_COORDINATION_TRIGGER_STEP = 235
V4_COORDINATION_SMOOTH_POLICY_TICKS = 10
V4_COORDINATION_TRANSLATION_METERS = (0.0, 0.0, 0.03)
V4_PLAN_POSE_ATOL = 5.0e-6
V4_PLAN_ROTATION_ATOL_RAD = 5.0e-5


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_config(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"unsupported V4 protocol schema in {path}")
    return {
        **payload,
        "fingerprint": canonical_fingerprint(payload),
    }


def load_v4_lift_intervention_protocol(
    path: Path = V4_LIFT_INTERVENTION_CONFIG,
) -> dict[str, Any]:
    payload = _load_config(path, V4_LIFT_INTERVENTION_SCHEMA)
    if set(payload) != {
        "schema",
        "release",
        "task",
        "formal_scenarios",
        "trigger",
        "teleport",
        "final_settling_physics_steps",
        "provenance",
        "fingerprint",
    }:
        raise ValueError("V4 LiftTray intervention fields are invalid")
    trigger = payload.get("trigger")
    grippers = trigger.get("expected_gripper_states") if isinstance(trigger, dict) else None
    provenance = payload.get("provenance")
    if (
        payload.get("release") != "v4"
        or payload.get("task") != V4_LIFT_TASK
        or payload.get("formal_scenarios") != ["static", "teleport"]
        or not isinstance(trigger, dict)
        or trigger.get("clock") != "successfully_committed_policy_ticks"
        or trigger.get("skill_label") != 0
        or trigger.get("local_tick") != V4_LIFT_TRIGGER_STEP
        or trigger.get("global_tick") != V4_LIFT_TRIGGER_STEP
        or trigger.get("application_timing")
        != "before_requesting_policy_action_at_global_tick"
        or grippers != {"left": "open", "right": "open"}
        or payload.get("teleport")
        != {
            "applications_per_episode": 1,
            "observation_refreshed_before_policy_action": True,
        }
        or payload.get("final_settling_physics_steps") != 10
        or not isinstance(provenance, dict)
        or provenance.get("selection_authority")
        != "manual_pre_interaction_skill_semantics"
        or provenance.get("integer_ticks_authoritative") is not True
        or provenance.get("result_based_retuning_forbidden") is not True
        or provenance.get("legacy_v3_profile_unchanged") is not True
    ):
        raise ValueError("V4 LiftTray intervention protocol is invalid")
    return payload


def load_v4_lift_motion_source_protocol(
    path: Path = V4_LIFT_MOTION_SOURCE_CONFIG,
) -> dict[str, Any]:
    payload = _load_config(path, V4_LIFT_MOTION_SOURCE_SCHEMA)
    if set(payload) != {
        "schema",
        "release",
        "task",
        "task_semantics",
        "source_selection_max_attempts",
        "goal_sampling_max_attempts",
        "translation",
        "rotation",
        "candidate_generation",
        "runtime_loader",
        "fingerprint",
    }:
        raise ValueError("V4 LiftTray motion-source fields are invalid")
    translation = payload.get("translation")
    rotation = payload.get("rotation")
    generation = payload.get("candidate_generation")
    semantics = payload.get("task_semantics")
    if (
        payload.get("release") != "v4"
        or payload.get("task") != V4_LIFT_TASK
        or payload.get("runtime_loader") != V4_LIFT_RUNTIME_LOADER_ID
        or payload.get("source_selection_max_attempts") != 20
        or payload.get("goal_sampling_max_attempts") != 100
        or not isinstance(semantics, dict)
        or semantics.get("schema")
        != "rlbench-public-bimanual-lift-tray-task-semantics-v1"
        or semantics.get("spatial_root_name") != V4_LIFT_TASK
        or semantics.get("spatial_root_type") != "DUMMY"
        or translation
        != {
            "reference": "source_A_boundary_root",
            "frame": "world_xy",
            "radial_min_m": 0.03,
            "radial_max_m": 0.08,
            "z_delta_m": 0.0,
        }
        or rotation
        != {
            "composition": "world_z_yaw_left_multiply_source_quaternion",
            "yaw_delta_abs_max_rad": 0.1,
            "roll_pitch": "unchanged_from_source_A",
        }
        or not isinstance(generation, dict)
        or generation.get("protocol_id")
        != "lift-tray-source-relative-radial-candidate-v4"
        or generation.get("selection_authority") != "scene_validity_only"
        or generation.get("policy_result_fields_read") is not False
        or generation.get("result_based_candidate_selection_forbidden") is not True
    ):
        raise ValueError("V4 LiftTray motion-source protocol is invalid")
    return payload


def load_v4_coordination_intervention_protocol(
    path: Path = V4_COORDINATION_INTERVENTION_CONFIG,
) -> dict[str, Any]:
    payload = _load_config(path, V4_COORDINATION_INTERVENTION_SCHEMA)
    if set(payload) != {
        "schema",
        "release",
        "task",
        "scenarios",
        "trigger",
        "motion",
        "clock_semantics",
        "invalid_action",
        "provenance",
        "fingerprint",
    }:
        raise ValueError("V4 coordination intervention fields are invalid")
    trigger = payload.get("trigger")
    motion = payload.get("motion")
    clock = payload.get("clock_semantics")
    invalid = payload.get("invalid_action")
    provenance = payload.get("provenance")
    if (
        payload.get("release") != "v4"
        or payload.get("task") != "bimanual_handover_item_dynamic"
        or payload.get("scenarios")
        != ["coordination_hand_left", "coordination_hand_right"]
        or not isinstance(trigger, dict)
        or trigger.get("clock") != "successfully_committed_policy_ticks"
        or trigger.get("global_tick") != V4_COORDINATION_TRIGGER_STEP
        or trigger.get("application_timing")
        != "on_policy_action_at_global_tick_before_environment_step"
        or not isinstance(motion, dict)
        or motion.get("target_source")
        != "predicted_absolute_end_effector_action"
        or motion.get("translation_frame") != "world"
        or motion.get("translation_m")
        != list(V4_COORDINATION_TRANSLATION_METERS)
        or motion.get("smooth_policy_ticks")
        != V4_COORDINATION_SMOOTH_POLICY_TICKS
        or motion.get("smooth_fractions")
        != "1_over_n_through_n_over_n"
        or motion.get("orientation") != "unmodified_policy_target"
        or motion.get("other_arm") != "unmodified_policy_target"
        or motion.get("both_grippers") != "unmodified_policy_target"
        or motion.get("application")
        != "fractional_offset_during_smooth_window_then_hold_final_offset"
        or motion.get("persistent_policy_target_offset") is not True
        or clock
        != {
            "policy_requests_during_smooth_window": (
                V4_COORDINATION_SMOOTH_POLICY_TICKS
            ),
            "policy_clock_advances_during_smooth_window": True,
            "policy_action_and_intervention_share_one_transaction": True,
            "observation_refreshed_by_each_committed_policy_step": True,
        }
        or not isinstance(invalid, dict)
        or invalid.get("max_primary_action_attempts_per_policy_tick") != 1
        or invalid.get("solver")
        != "global_pseudo_trac_sampling_path_formal_v1"
        or invalid.get("failure_behavior")
        != "commit_joint_hold_and_advance_same_policy_tick"
        or not isinstance(provenance, dict)
        or provenance.get("result_based_retuning_forbidden") is not True
        or provenance.get("legacy_v3_profile_unchanged") is not True
    ):
        raise ValueError("V4 coordination intervention protocol is invalid")
    return payload


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _quaternion_angle_xyzw(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    lhs /= np.linalg.norm(lhs)
    rhs /= np.linalg.norm(rhs)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(lhs, rhs)), 0.0, 1.0))))


def sample_v4_lift_goal_pose(
    source_pose: Any,
    candidate_seed: int,
) -> np.ndarray:
    """Derive B from only A and the preregistered candidate seed."""

    source = np.asarray(source_pose, dtype=np.float64)
    if source.shape != (7,) or not np.all(np.isfinite(source)):
        raise ValueError("V4 LiftTray source pose must be finite 7D")
    if (
        isinstance(candidate_seed, bool)
        or not isinstance(candidate_seed, int)
        or candidate_seed < 0
    ):
        raise ValueError("V4 LiftTray candidate seed must be non-negative")
    quaternion = source[3:7]
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("V4 LiftTray source quaternion is invalid")
    quaternion = quaternion / norm
    generator = random.Random(candidate_seed)
    radius = generator.uniform(0.03, 0.08)
    angle = generator.uniform(-math.pi, math.pi)
    yaw_delta = generator.uniform(-0.10, 0.10)
    position = source[:3].copy()
    position[0] += radius * math.cos(angle)
    position[1] += radius * math.sin(angle)
    # World-z left multiplication preserves source roll/pitch in ZYX form.
    half = 0.5 * yaw_delta
    yaw_quaternion = np.asarray([0.0, 0.0, math.sin(half), math.cos(half)])
    goal_quaternion = _quaternion_multiply_xyzw(yaw_quaternion, quaternion)
    goal_quaternion /= np.linalg.norm(goal_quaternion)
    return np.concatenate((position, goal_quaternion))


def v4_lift_plan_geometry(source_pose: Any, goal_pose: Any) -> dict[str, Any]:
    source = np.asarray(source_pose, dtype=np.float64)
    goal = np.asarray(goal_pose, dtype=np.float64)
    if (
        source.shape != (7,)
        or goal.shape != (7,)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(goal))
    ):
        raise ValueError("V4 LiftTray plan poses must be finite 7D")
    delta = goal[:3] - source[:3]
    source_q = source[3:7] / np.linalg.norm(source[3:7])
    goal_q = goal[3:7] / np.linalg.norm(goal[3:7])
    inverse = np.asarray([-source_q[0], -source_q[1], -source_q[2], source_q[3]])
    relative = _quaternion_multiply_xyzw(goal_q, inverse)
    if relative[3] < 0.0:
        relative = -relative
    relative /= np.linalg.norm(relative)
    yaw_delta = 2.0 * math.atan2(float(relative[2]), float(relative[3]))
    return {
        "xy_radius_m": float(np.linalg.norm(delta[:2])),
        "z_delta_m": float(delta[2]),
        "yaw_delta_rad": float(yaw_delta),
        "relative_rotation_xy_norm": float(np.linalg.norm(relative[:2])),
    }


def _v4_lift_plan_evidence(plan: Any) -> dict[str, Any]:
    motion = load_v4_lift_motion_source_protocol()
    intervention = load_v4_lift_intervention_protocol()
    candidate_seed = plan.validation.get("selected_candidate_seed")
    expected = sample_v4_lift_goal_pose(plan.source_pose, candidate_seed)
    geometry = v4_lift_plan_geometry(plan.source_pose, plan.goal_pose)
    return {
        "schema": V4_LIFT_PLAN_EVIDENCE_SCHEMA,
        "runtime_protocol_id": V4_LIFT_MOTION_PROTOCOL_ID,
        "motion_source_schema": motion["schema"],
        "motion_source_fingerprint": motion["fingerprint"],
        "intervention_schema": intervention["schema"],
        "intervention_fingerprint": intervention["fingerprint"],
        "candidate_generator": motion["candidate_generation"]["protocol_id"],
        "selection_authority": "scene_validity_only",
        "policy_result_fields_read": False,
        "accepted_candidate_geometry": geometry,
        "expected_goal_pose": expected.tolist(),
    }


def attach_v4_lift_plan_evidence(plan: Any) -> Any:
    """Return a signed legacy-runtime plan carrying V4 task-scoped evidence."""

    from integrations.rlbench.rlbench_dynamac.core.runtime import StagedMotionPlan

    if plan.task_name != V4_LIFT_TASK:
        raise ValueError("V4 LiftTray evidence cannot be attached to another task")
    validation = dict(plan.validation)
    validation["v4_lift_tray"] = _v4_lift_plan_evidence(plan)
    return StagedMotionPlan(
        task_name=plan.task_name,
        source_pose=plan.source_pose,
        goal_pose=plan.goal_pose,
        source_low_dim_state=plan.source_low_dim_state,
        episode_seed=plan.episode_seed,
        variation=plan.variation,
        validation=validation,
    )


def validate_v4_lift_motion_plan(plan: Any) -> dict[str, Any]:
    if plan.task_name != V4_LIFT_TASK:
        raise ValueError("V4 LiftTray plan has the wrong task")
    actual = plan.validation.get("v4_lift_tray")
    expected = _v4_lift_plan_evidence(plan)
    if not isinstance(actual, dict):
        raise ValueError("V4 LiftTray task-scoped plan evidence is invalid")
    actual_geometry = actual.get("accepted_candidate_geometry")
    expected_geometry = expected["accepted_candidate_geometry"]
    actual_expected_goal = actual.get("expected_goal_pose")
    expected_expected_goal = expected["expected_goal_pose"]
    exact_actual = {
        key: value
        for key, value in actual.items()
        if key not in {"accepted_candidate_geometry", "expected_goal_pose"}
    }
    exact_expected = {
        key: value
        for key, value in expected.items()
        if key not in {"accepted_candidate_geometry", "expected_goal_pose"}
    }
    geometry_matches = (
        isinstance(actual_geometry, dict)
        and set(actual_geometry) == set(expected_geometry)
        and all(
            isinstance(actual_geometry[key], (int, float))
            and not isinstance(actual_geometry[key], bool)
            and math.isfinite(float(actual_geometry[key]))
            and math.isclose(
                float(actual_geometry[key]),
                float(expected_geometry[key]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            for key in expected_geometry
        )
    )
    expected_goal_matches = (
        isinstance(actual_expected_goal, (list, tuple))
        and len(actual_expected_goal) == 7
        and all(
            isinstance(actual_value, (int, float))
            and not isinstance(actual_value, bool)
            and math.isfinite(float(actual_value))
            and math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            for actual_value, expected_value in zip(
                actual_expected_goal,
                expected_expected_goal,
            )
        )
    )
    if (
        exact_actual != exact_expected
        or not geometry_matches
        or not expected_goal_matches
    ):
        raise ValueError("V4 LiftTray task-scoped plan evidence is invalid")
    expected_goal = np.asarray(expected["expected_goal_pose"], dtype=np.float64)
    actual_goal = np.asarray(plan.goal_pose, dtype=np.float64)
    position_error = float(np.linalg.norm(expected_goal[:3] - actual_goal[:3]))
    rotation_error = _quaternion_angle_xyzw(expected_goal[3:7], actual_goal[3:7])
    geometry = expected["accepted_candidate_geometry"]
    if (
        position_error > V4_PLAN_POSE_ATOL
        or rotation_error > V4_PLAN_ROTATION_ATOL_RAD
        or geometry["xy_radius_m"] < 0.03 - V4_PLAN_POSE_ATOL
        or geometry["xy_radius_m"] > 0.08 + V4_PLAN_POSE_ATOL
        or abs(geometry["z_delta_m"]) > V4_PLAN_POSE_ATOL
        or abs(geometry["yaw_delta_rad"]) > 0.10 + V4_PLAN_ROTATION_ATOL_RAD
        or geometry["relative_rotation_xy_norm"] > V4_PLAN_ROTATION_ATOL_RAD
    ):
        raise ValueError("V4 LiftTray A-to-B geometry is outside its protocol")
    return {
        **geometry,
        "expected_goal_position_error_m": position_error,
        "expected_goal_rotation_error_rad": rotation_error,
        "validated": True,
    }


def stage_v4_lift_motion_plan(
    environment: Any,
    task_class: Any,
    *,
    episode_seed: int,
    variation: int,
    max_attempts: int = 100,
    source_max_attempts: int = 20,
) -> Any:
    """Stage one V4 LiftTray A/B without observing a policy outcome."""

    motion = load_v4_lift_motion_source_protocol()
    if (
        max_attempts != motion["goal_sampling_max_attempts"]
        or source_max_attempts != motion["source_selection_max_attempts"]
    ):
        raise ValueError("V4 LiftTray staging budgets differ from the protocol")
    from integrations.rlbench.rlbench_dynamac.core.runtime import stage_scenario_motion_plan

    plan = stage_scenario_motion_plan(
        environment,
        task_class,
        episode_seed=episode_seed,
        variation=variation,
        task_name=V4_LIFT_TASK,
        max_attempts=max_attempts,
        source_max_attempts=source_max_attempts,
        goal_candidate_sampler=sample_v4_lift_goal_pose,
    )
    return attach_v4_lift_plan_evidence(plan)


def load_v4_lift_motion_plan_batch(payload: dict[str, Any]) -> list[Any]:
    """Runtime loader registered by the ``rlbench_eval_v2`` envelope."""

    from integrations.rlbench.rlbench_dynamac.core.runtime import load_staged_motion_plan_batch

    plans = load_staged_motion_plan_batch(payload)
    if not plans:
        raise ValueError("V4 LiftTray plan batch is empty")
    for plan in plans:
        validate_v4_lift_motion_plan(plan)
    return plans


def v4_runtime_loaders() -> dict[str, Any]:
    """Builtin registry hook for eval-set seal/preflight and formal loaders."""

    return {V4_LIFT_RUNTIME_LOADER_ID: load_v4_lift_motion_plan_batch}


def v4_lift_task_identity_components() -> dict[str, dict[str, str]]:
    motion = load_v4_lift_motion_source_protocol()
    intervention = load_v4_lift_intervention_protocol()
    semantics = motion["task_semantics"]
    return {
        "task_semantics": {
            "schema": semantics["schema"],
            "fingerprint": canonical_fingerprint(semantics),
        },
        "motion_source": {
            "schema": motion["schema"],
            "fingerprint": motion["fingerprint"],
        },
        "intervention": {
            "schema": intervention["schema"],
            "fingerprint": intervention["fingerprint"],
        },
    }


def build_v4_lift_task_scoped_plan_batch(
    *,
    base_seed: int,
    variations: list[int],
    plans: list[Any],
) -> dict[str, Any]:
    """Wrap the runtime batch in the regenerated ``rlbench_eval_v2`` envelope."""

    from integrations.rlbench.rlbench_dynamac.eval.eval_set import build_task_scoped_identity, build_task_scoped_plan_batch
    from integrations.rlbench.rlbench_dynamac.core.runtime import staged_motion_plan_batch

    for plan in plans:
        validate_v4_lift_motion_plan(plan)
    runtime_batch = staged_motion_plan_batch(
        task_name=V4_LIFT_TASK,
        base_seed=base_seed,
        variations=variations,
        plans=plans,
    )
    identity = build_task_scoped_identity(
        task_name=V4_LIFT_TASK,
        components=v4_lift_task_identity_components(),
    )
    return build_task_scoped_plan_batch(
        task_name=V4_LIFT_TASK,
        task_identity=identity,
        runtime_loader=V4_LIFT_RUNTIME_LOADER_ID,
        runtime_batch=runtime_batch,
    )


def v4_lift_trigger_authentication(policy_steps: int) -> dict[str, Any]:
    protocol = load_v4_lift_intervention_protocol()
    if (
        isinstance(policy_steps, bool)
        or not isinstance(policy_steps, int)
        or policy_steps <= V4_LIFT_TRIGGER_STEP
    ):
        raise ValueError("V4 LiftTray trigger lies outside the policy clock")
    return {
        "schema": "dynamac-lift-tray-trigger-authentication-v4",
        "protocol_schema": protocol["schema"],
        "protocol_fingerprint": protocol["fingerprint"],
        "skill_label": 0,
        "local_tick": V4_LIFT_TRIGGER_STEP,
        "global_tick": V4_LIFT_TRIGGER_STEP,
        "trigger_step": V4_LIFT_TRIGGER_STEP,
        "expected_gripper_states": {"left": "open", "right": "open"},
        "validated_against_policy_horizon": policy_steps,
        "result_based_retuning": False,
    }


def v4_coordination_trigger_authentication(
    *,
    arm: str,
    policy_steps: int,
) -> dict[str, Any]:
    protocol = load_v4_coordination_intervention_protocol()
    if arm not in {"left", "right"}:
        raise ValueError("V4 coordination trigger requires left or right arm")
    if (
        isinstance(policy_steps, bool)
        or not isinstance(policy_steps, int)
        or policy_steps
        < V4_COORDINATION_TRIGGER_STEP + V4_COORDINATION_SMOOTH_POLICY_TICKS
    ):
        raise ValueError(
            "V4 coordination smooth window lies outside the policy clock"
        )
    return {
        "schema": "dynamac-coordination-trigger-authentication-v4",
        "protocol_id": V4_COORDINATION_PROTOCOL_ID,
        "protocol_schema": protocol["schema"],
        "protocol_fingerprint": protocol["fingerprint"],
        "perturbed_arm": arm,
        "trigger_step": V4_COORDINATION_TRIGGER_STEP,
        "smooth_policy_ticks": V4_COORDINATION_SMOOTH_POLICY_TICKS,
        "final_smooth_tick": (
            V4_COORDINATION_TRIGGER_STEP
            + V4_COORDINATION_SMOOTH_POLICY_TICKS
            - 1
        ),
        "validated_against_policy_horizon": policy_steps,
        "result_based_retuning": False,
    }
