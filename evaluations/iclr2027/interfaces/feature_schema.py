"""Frozen causal feature and result schemas for common-backbone experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

FEATURE_SCHEMA = "essay2608.iclr2027.causal-features.v1"
AUDIT_SCHEMA = "essay2608.iclr2027.physical-event-audit.v1"
EPISODE_SCHEMA = "essay2608.iclr2027.episode-result.v1"
ARM_NAMES = frozenset({"single", "left", "right"})
FORBIDDEN_MONITOR_KEYS = frozenset(
    {
        "fault_family",
        "fault_severity",
        "trigger_stage",
        "injector_event",
        "injector",
        "eligible",
        "physically_triggered",
        "violation_onset",
        "violation_onset_cycle",
        "audit_label",
        "future_observation",
    }
)


def _finite_vector(value: Any, *, name: str, size: Optional[int] = None) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array.tolist()


def _forbidden_nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_MONITOR_KEYS:
                found.add(str(key))
            found.update(_forbidden_nested_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_forbidden_nested_keys(nested))
    return found


@dataclass(frozen=True)
class FeatureRecord:
    """One monitor-visible record; evaluator-only fields are intentionally absent."""

    episode_id: str
    cycle: int
    observation_timestamp: int
    action_timestamp: int
    arms: Mapping[str, Mapping[str, Any]]
    task_state: tuple[float, ...]
    action: tuple[float, ...]
    policy_state: Mapping[str, Any]
    action_resolution: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": FEATURE_SCHEMA,
            "episode_id": self.episode_id,
            "cycle": self.cycle,
            "observation_timestamp": self.observation_timestamp,
            "action_timestamp": self.action_timestamp,
            "arms": {name: dict(fields) for name, fields in self.arms.items()},
            "task_state": list(self.task_state),
            "action": list(self.action),
            "policy_state": dict(self.policy_state),
            "action_resolution": dict(self.action_resolution),
        }
        return validate_feature_record(value)


def validate_feature_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-ready copy of one strictly causal record."""

    if value.get("schema") != FEATURE_SCHEMA:
        raise ValueError("unsupported causal feature schema")
    episode_id = value.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("feature record requires a non-empty episode_id")
    integers = {}
    for name in ("cycle", "observation_timestamp", "action_timestamp"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        integers[name] = item
    if integers["observation_timestamp"] != integers["cycle"]:
        raise ValueError("observation timestamp must equal the current cycle")
    if integers["action_timestamp"] != integers["cycle"]:
        raise ValueError("action timestamp must equal the current cycle")

    arms = value.get("arms")
    if not isinstance(arms, Mapping) or set(arms) not in (
        {"single"},
        {"left", "right"},
    ):
        raise ValueError("arms must cover single or both left/right arms")
    clean_arms: dict[str, dict[str, Any]] = {}
    for arm, fields in arms.items():
        if arm not in ARM_NAMES or not isinstance(fields, Mapping):
            raise ValueError("invalid arm feature record")
        gripper = float(fields.get("gripper_open"))
        if not np.isfinite(gripper):
            raise ValueError("gripper_open must be finite")
        clean_arms[arm] = {
            "ee_pose_xyzw": _finite_vector(
                fields.get("ee_pose_xyzw"), name=f"{arm}.ee_pose_xyzw", size=7
            ),
            "gripper_open": gripper,
        }
    policy_state = value.get("policy_state")
    resolution = value.get("action_resolution")
    if not isinstance(policy_state, Mapping) or not isinstance(resolution, Mapping):
        raise ValueError("policy_state and action_resolution must be mappings")
    leaked = _forbidden_nested_keys(policy_state) | _forbidden_nested_keys(resolution)
    if leaked:
        raise ValueError(f"evaluator-only fields leaked into monitor features: {sorted(leaked)}")
    return {
        "schema": FEATURE_SCHEMA,
        "episode_id": episode_id,
        **integers,
        "arms": clean_arms,
        "task_state": _finite_vector(value.get("task_state"), name="task_state"),
        "action": _finite_vector(value.get("action"), name="action"),
        "policy_state": dict(policy_state),
        "action_resolution": dict(resolution),
    }


__all__ = [
    "AUDIT_SCHEMA",
    "ARM_NAMES",
    "EPISODE_SCHEMA",
    "FEATURE_SCHEMA",
    "FeatureRecord",
    "validate_feature_record",
]
