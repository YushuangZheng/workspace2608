"""Pure conversion and intervention helpers for the optional RLBench runtime.

No RLBench/PyRep import happens at module import time.  This lets conversion
and action-layout tests run on machines without CoppeliaSim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .task_specs import TaskSpec, get_task_spec, unwrap_task_low_dim_state
from .task_specs import wxyz_to_xyzw as _wxyz_to_xyzw
from .task_specs import xyzw_to_wxyz as _xyzw_to_wxyz

Array = np.ndarray


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


def unimanual_action_to_rlbench(action: Any, *, ignore_collisions: bool = False) -> Array:
    """Return the fork's 9D ``pose, gripper, ignore`` action."""

    pose = wxyz_to_xyzw(np.asarray(action.pose, dtype=np.float64))
    return np.concatenate((pose, [_gripper_to_rlbench(action.gripper), float(ignore_collisions)]))


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


@dataclass
class ScenarioController:
    """Call only the two dynamic-scene mechanisms present in public code."""

    kind: Literal["static", "teleport_task", "smooth_task_motion"]
    trigger_fraction: float = 1.0 / 3.0
    total_steps: int = 10
    max_attempts: int = 20
    verify_instance: bool = True
    _teleported: bool = False
    _smooth_calls: int = 0
    _smooth_complete: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must lie in [0, 1]")
        if self.total_steps < 1 or self.max_attempts < 1:
            raise ValueError("total_steps and max_attempts must be positive")

    def apply(self, task_environment: Any, *, step: int, horizon: int) -> dict[str, Any]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        trigger = min(horizon - 1, int(round(self.trigger_fraction * (horizon - 1))))
        event: dict[str, Any] = {
            "kind": self.kind,
            "step": step,
            "trigger_step": trigger,
            "applied": False,
        }
        if self.kind == "static" or step < trigger:
            return event
        scene = getattr(task_environment, "_scene", None)
        if scene is None:
            raise RuntimeError("author RLBench TaskEnvironment._scene is unavailable")
        before_observation = task_environment.get_observation()
        before_state = unwrap_task_low_dim_state(before_observation.task_low_dim_state)
        root = scene.task.boundary_root()
        before_root = np.asarray(root.get_pose(), dtype=np.float64)
        if self.kind == "teleport_task":
            if not self._teleported:
                scene.kidnap(
                    max_attempts=self.max_attempts,
                    verify_instance=self.verify_instance,
                )
                self._teleported = True
                event["applied"] = True
                after_observation = task_environment.get_observation()
                after_state = unwrap_task_low_dim_state(after_observation.task_low_dim_state)
                after_root = np.asarray(root.get_pose(), dtype=np.float64)
                event.update(
                    _intervention_change(before_state, after_state, before_root, after_root)
                )
                event["protocol_effective"] = bool(event["task_state_changed"])
            return event
        if self.kind == "smooth_task_motion":
            if self._smooth_complete:
                return event
            if self._smooth_calls == 0:
                sentinel = object()
                prior_state = getattr(scene, "_move_task_smoothly_state", sentinel)
                # The pinned fork deletes this attribute when a smooth move
                # completes; init_episode restores it to None. Establish one
                # controller-local state machine defensively so repeated calls
                # before the next init_episode cannot observe a missing value.
                scene._move_task_smoothly_state = None
                event["upstream_smooth_state_reinitialized"] = bool(
                    prior_state is sentinel or prior_state is not None
                )
            state_before_call = getattr(scene, "_move_task_smoothly_state", None)
            self._smooth_calls += 1
            self._smooth_complete = bool(
                scene.move_task_smoothly(
                    total_steps=self.total_steps,
                    max_attempts=self.max_attempts,
                    verify_instance=self.verify_instance,
                )
            )
            # The pinned fork declares completion after evaluating interpolation
            # fractions 0/N ... (N-1)/N.  Apply the already sampled and validated
            # endpoint explicitly on the final call instead of reporting a full
            # move while leaving the root at only (N-1)/N of the path.
            state = getattr(scene, "_move_task_smoothly_state", None)
            endpoint_state = state if isinstance(state, dict) else state_before_call
            endpoint_applied = False
            if self._smooth_complete and isinstance(endpoint_state, dict):
                goal_pose = endpoint_state.get("goal_pose")
                if goal_pose is not None:
                    root.set_pose(np.asarray(goal_pose, dtype=np.float64))
                    endpoint_applied = True
            after_observation = task_environment.get_observation()
            after_state = unwrap_task_low_dim_state(after_observation.task_low_dim_state)
            after_root = np.asarray(root.get_pose(), dtype=np.float64)
            event.update(
                {
                    "applied": True,
                    "smooth_call": self._smooth_calls,
                    "complete": self._smooth_complete,
                    "endpoint_applied": endpoint_applied,
                }
            )
            event.update(_intervention_change(before_state, after_state, before_root, after_root))
            if (
                isinstance(endpoint_state, dict)
                and endpoint_state.get("source_pose") is not None
            ):
                source = np.asarray(endpoint_state["source_pose"], dtype=np.float64)
                goal = np.asarray(endpoint_state["goal_pose"], dtype=np.float64)
                event["planned_root_translation_m"] = float(np.linalg.norm(goal[:3] - source[:3]))
                event["planned_root_motion"] = bool(
                    not np.allclose(source, goal, rtol=0.0, atol=1.0e-9)
                )
                event["protocol_effective"] = event["planned_root_motion"]
                event["internal_state_changed_without_root_goal"] = bool(
                    event["task_state_changed"] and not event["planned_root_motion"]
                )
                event["endpoint_fraction"] = 1.0 if endpoint_applied else min(
                    self._smooth_calls / self.total_steps,
                    1.0,
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
