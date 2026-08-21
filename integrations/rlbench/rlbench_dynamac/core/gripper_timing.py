"""Global task-independent gripper timing for RLBench execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


GLOBAL_GRIPPER_TIMING_PROTOCOL_ID = (
    "dynamac-global-skill-boundary-transition-lookahead-v1"
)
GLOBAL_GRIPPER_TIMING_RULE = "skill_boundary_transition_lookahead"
ACTION_GRIPPER_INDICES = {9: (7,), 18: (7, 16)}


def global_gripper_timing_metadata() -> dict[str, Any]:
    """Return the task-agnostic execution identity used by every evaluator."""

    return {
        "protocol_id": GLOBAL_GRIPPER_TIMING_PROTOCOL_ID,
        "rule": GLOBAL_GRIPPER_TIMING_RULE,
        "policy_pose_tick": "t",
        "default_gripper_tick": "t",
        "boundary_transition_gripper_tick": "t_plus_1",
        "lookahead_scope": "discrete_transition_at_learned_skill_boundary_only",
        "lookahead_source": "read_only_policy_core_without_pose_prediction",
        "terminal_behavior": "repeat_final_gripper_command",
        "bimanual_boundary_semantics": "independent_per_arm",
        "pose_predictions_per_policy_tick": 1,
        "task_specific_branches": False,
        "task_name_or_tick_special_cases": False,
        "training_labels_modified": False,
        "checkpoint_refit_required": False,
    }


def action_gripper_indices(action_dimension: int) -> tuple[int, ...]:
    if isinstance(action_dimension, bool) or not isinstance(action_dimension, int):
        raise TypeError("action dimension must be an integer")
    try:
        return ACTION_GRIPPER_INDICES[action_dimension]
    except KeyError as exc:
        raise ValueError("gripper timing supports only 9D or 18D RLBench actions") from exc


def native_gripper_to_wire(value: Any) -> float:
    """Map the policy's signed gripper factor to RLBench's discrete scalar."""

    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("gripper preview must be a non-empty finite vector")
    return float(float(np.mean(array)) > 0.0)


def _exact_lane_map(
    values: Mapping[int, Any],
    *,
    indices: tuple[int, ...],
    name: str,
) -> dict[int, Any]:
    if not isinstance(values, Mapping) or set(values) != set(indices):
        raise ValueError(f"{name} must contain exactly the action gripper lanes")
    return {index: values[index] for index in indices}


def apply_global_gripper_timing(
    action: Any,
    *,
    next_wire_gripper_by_index: Mapping[int, Any],
    crosses_skill_boundary_by_index: Mapping[int, Any],
) -> np.ndarray:
    """Advance only a discrete gripper transition at a skill boundary.

    The input is a single unimanual (9D) or bimanual (18D) RLBench wire action.
    Every pose and ignore-collision scalar is copied exactly.  Each bimanual arm
    uses its own boundary flag, so asynchronous skill clocks remain independent.
    """

    original = np.asarray(action, dtype=np.float64)
    if original.ndim != 1 or not np.all(np.isfinite(original)):
        raise ValueError("action must be one finite 9D or 18D vector")
    indices = action_gripper_indices(len(original))
    next_commands = _exact_lane_map(
        next_wire_gripper_by_index,
        indices=indices,
        name="next_wire_gripper_by_index",
    )
    boundaries = _exact_lane_map(
        crosses_skill_boundary_by_index,
        indices=indices,
        name="crosses_skill_boundary_by_index",
    )
    emitted = original.copy()
    for index in indices:
        command = next_commands[index]
        if isinstance(command, bool):
            command = float(command)
        elif isinstance(command, (int, float, np.integer, np.floating)):
            command = float(command)
        else:
            raise TypeError("next wire gripper commands must be numeric scalars")
        if not np.isfinite(command) or command not in {0.0, 1.0}:
            raise ValueError("next wire gripper commands must be exactly 0 or 1")
        boundary = boundaries[index]
        if not isinstance(boundary, (bool, np.bool_)):
            raise TypeError("skill-boundary flags must be booleans")
        if bool(boundary) and (original[index] > 0.5) != (command > 0.5):
            emitted[index] = command

    changed = set(np.flatnonzero(emitted != original).tolist())
    if not changed.issubset(indices):
        raise AssertionError("gripper timing changed a non-gripper action scalar")
    return emitted


__all__ = [
    "GLOBAL_GRIPPER_TIMING_PROTOCOL_ID",
    "GLOBAL_GRIPPER_TIMING_RULE",
    "action_gripper_indices",
    "apply_global_gripper_timing",
    "global_gripper_timing_metadata",
    "native_gripper_to_wire",
]
