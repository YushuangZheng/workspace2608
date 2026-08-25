"""Unified per-tick observation used by all phase-two belief modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..dynamac import DynaMACObservation, normalize_quaternion

Array = np.ndarray


def _pose(value: Array, name: str) -> Array:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} 必须为有限 [7] 位姿")
    result = pose.copy()
    result[3:7] = normalize_quaternion(result[3:7])
    return result


@dataclass(frozen=True)
class RuntimeObservation:
    """One shared online observation without any task-clock side effects.

    ``entity_configurations`` is the minimal extension required to evaluate the
    node factors learned in phase one.  Platforms that expose only rigid-body
    poses may leave it empty; edge factors continue to use ``frame_poses``.
    """

    tick: int
    ee_pose: Array
    frame_poses: dict[str, Array]
    gripper_state: Array
    previous_command_pose: Array | None
    previous_ee_pose: Array | None
    tracking_reliability: dict[str, float]
    frame_visibility: dict[str, bool]
    entity_configurations: dict[str, dict[str, Array]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.tick, (bool, np.bool_)) or not isinstance(
            self.tick, (int, np.integer)
        ):
            raise TypeError("RuntimeObservation.tick 必须为整数")
        if self.tick < 0:
            raise ValueError("RuntimeObservation.tick 必须非负")

        frames: dict[str, Array] = {}
        for name, frame_pose in self.frame_poses.items():
            if not isinstance(name, str) or not name:
                raise ValueError("运行观测参考系名称必须为非空字符串")
            frames[name] = _pose(frame_pose, f"frame_poses[{name}]")

        gripper = np.asarray(self.gripper_state, dtype=np.float64)
        if gripper.ndim == 0:
            gripper = gripper.reshape(1)
        if gripper.ndim != 1 or len(gripper) == 0 or not np.all(np.isfinite(gripper)):
            raise ValueError("gripper_state 必须为非空有限一维数组")

        reliability: dict[str, float] = {}
        for name, reliability_value in self.tracking_reliability.items():
            if name not in frames:
                raise ValueError(f"跟踪可靠性引用未知参考系 {name}")
            number = float(reliability_value)
            if not np.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError("跟踪可靠性必须位于 [0,1]")
            reliability[name] = number

        visibility: dict[str, bool] = {}
        for name, visibility_value in self.frame_visibility.items():
            if name not in frames:
                raise ValueError(f"可见性引用未知参考系 {name}")
            if not isinstance(visibility_value, (bool, np.bool_)):
                raise TypeError("参考系可见性必须为布尔值")
            visibility[name] = bool(visibility_value)

        configurations: dict[str, dict[str, Array]] = {}
        for entity, raw_fields in self.entity_configurations.items():
            if not isinstance(entity, str) or not entity:
                raise ValueError("实体构型名称必须为非空字符串")
            fields: dict[str, Array] = {}
            for feature, raw_value in raw_fields.items():
                if not isinstance(feature, str) or not feature:
                    raise ValueError("实体构型字段名必须为非空字符串")
                value = np.asarray(raw_value, dtype=np.float64)
                if value.ndim == 0:
                    value = value.reshape(1)
                if value.ndim != 1 or len(value) == 0 or not np.all(np.isfinite(value)):
                    raise ValueError("实体构型观测必须为非空有限一维数组")
                fields[feature] = value.copy()
            configurations[entity] = fields

        object.__setattr__(self, "tick", int(self.tick))
        object.__setattr__(self, "ee_pose", _pose(self.ee_pose, "ee_pose"))
        object.__setattr__(self, "frame_poses", frames)
        object.__setattr__(self, "gripper_state", gripper.copy())
        object.__setattr__(
            self,
            "previous_command_pose",
            (
                None
                if self.previous_command_pose is None
                else _pose(self.previous_command_pose, "previous_command_pose")
            ),
        )
        object.__setattr__(
            self,
            "previous_ee_pose",
            (
                None
                if self.previous_ee_pose is None
                else _pose(self.previous_ee_pose, "previous_ee_pose")
            ),
        )
        object.__setattr__(self, "tracking_reliability", reliability)
        object.__setattr__(self, "frame_visibility", visibility)
        object.__setattr__(self, "entity_configurations", configurations)

    def visibility(self, frame: str) -> bool:
        """Return explicit visibility, defaulting available simulator poses to true."""

        return self.frame_visibility.get(frame, frame in self.frame_poses)

    def reliability(self, frame: str) -> float:
        """Return explicit tracking reliability, defaulting simulator truth to one."""

        return self.tracking_reliability.get(
            frame, 1.0 if frame in self.frame_poses else 0.0
        )

    @classmethod
    def from_dynamac(
        cls,
        observation: DynaMACObservation,
        *,
        tick: int,
        gripper_state: Array,
        previous_command_pose: Array | None = None,
        previous_ee_pose: Array | None = None,
        tracking_reliability: Mapping[str, float] | None = None,
        frame_visibility: Mapping[str, bool] | None = None,
        additional_frame_poses: Mapping[str, Array] | None = None,
        entity_configurations: Mapping[str, Mapping[str, Array]] | None = None,
    ) -> RuntimeObservation:
        frames = {name: value.copy() for name, value in observation.frames.items()}
        if additional_frame_poses is not None:
            frames.update(additional_frame_poses)
        return cls(
            tick=tick,
            ee_pose=observation.ee_pose,
            frame_poses=frames,
            gripper_state=gripper_state,
            previous_command_pose=previous_command_pose,
            previous_ee_pose=previous_ee_pose,
            tracking_reliability=dict(tracking_reliability or {}),
            frame_visibility=dict(frame_visibility or {}),
            entity_configurations={
                entity: dict(fields)
                for entity, fields in (entity_configurations or {}).items()
            },
        )


__all__ = ["RuntimeObservation"]
