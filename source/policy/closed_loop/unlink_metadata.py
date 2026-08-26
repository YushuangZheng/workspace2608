"""Selection and instantiation of event-level UNLINK metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamac import pose_compose
from .relation_events import UnlinkEventMetadata
from .state_index import StateId
from .task_model import ClosedLoopTaskModel

Array = np.ndarray


@dataclass(frozen=True)
class InstantiatedUnlinkTarget:
    metadata: UnlinkEventMetadata
    pose: Array
    used_learned_target: bool

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("UNLINK 脱离目标必须为有限 [7] 位姿")
        object.__setattr__(self, "pose", pose.copy())


class UnlinkMetadataRepository:
    """Resolve the normal release event nearest the current expected state."""

    def __init__(self, task_model: ClosedLoopTaskModel) -> None:
        self.task_model = task_model
        self._states = tuple(sorted(task_model.states))
        self._global_index = {state: index for index, state in enumerate(self._states)}

    def resolve(
        self,
        frame_id: str,
        state_id: StateId,
        mode: int,
    ) -> UnlinkEventMetadata:
        if state_id not in self._global_index:
            raise KeyError(f"UNLINK 查询状态不存在：{state_id}")
        current = self._global_index[state_id]
        candidates = [
            event
            for event_id, event in self.task_model.unlink_events.items()
            if event_id.arm_id == self.task_model.arm_id
            and event_id.frame_id == frame_id
            and event_id.mode == mode
        ]
        if not candidates:
            raise KeyError(
                f"关系 {self.task_model.arm_id}/{frame_id} 在 mode {mode} "
                "没有 UNLINK 元数据"
            )

        preceding = [
            event
            for event in candidates
            if self._global_index[event.release_state] <= current
        ]
        if preceding:
            return max(
                preceding,
                key=lambda event: self._global_index[event.release_state],
            )
        return min(
            candidates,
            key=lambda event: abs(self._global_index[event.release_state] - current),
        )

    @staticmethod
    def instantiate(
        metadata: UnlinkEventMetadata,
        frame_pose: Array,
        current_ee_pose: Array,
        *,
        fallback_distance: float,
    ) -> InstantiatedUnlinkTarget:
        frame = np.asarray(frame_pose, dtype=np.float64)
        current = np.asarray(current_ee_pose, dtype=np.float64)
        if frame.shape != (7,) or current.shape != (7,):
            raise ValueError("UNLINK 实例化需要 [7] 参考系与末端位姿")
        if fallback_distance <= 0.0 or not np.isfinite(fallback_distance):
            raise ValueError("UNLINK 通用脱离距离必须为有限正数")
        if metadata.local_detachment_target is not None:
            return InstantiatedUnlinkTarget(
                metadata=metadata,
                pose=pose_compose(frame, metadata.local_detachment_target),
                used_learned_target=True,
            )

        direction = current[:3] - frame[:3]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction /= norm
        target = current.copy()
        target[:3] += fallback_distance * direction
        return InstantiatedUnlinkTarget(
            metadata=metadata,
            pose=target,
            used_learned_target=False,
        )


__all__ = ["InstantiatedUnlinkTarget", "UnlinkMetadataRepository"]
