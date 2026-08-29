"""Episode-scoped LINK anchor resolution and world-frame instantiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..dynamac import transform_marginal
from .relation_events import (
    LinkPendingCandidate,
    LinkRecoveryAnchor,
    RelationEventId,
    RelationStateKey,
)
from .state_index import StateId
from .task_model import ClosedLoopTaskModel

Array = np.ndarray
AnchorSource = Literal["offline_link", "pending_recovery", "verified_pending"]


@dataclass(frozen=True)
class RuntimeLinkAnchor:
    """One immutable recovery template available in the current episode."""

    origin_event_id: RelationEventId
    arm_id: str
    frame_id: str
    context_state: StateId
    local_means: Array
    local_covariances: Array
    gripper_commands: Array
    legal_reentry_states: tuple[StateId, ...]
    source: AnchorSource

    def __post_init__(self) -> None:
        means = np.asarray(self.local_means, dtype=np.float64)
        covariances = np.asarray(self.local_covariances, dtype=np.float64)
        gripper = np.asarray(self.gripper_commands, dtype=np.float64)
        if means.ndim != 2 or means.shape[1] != 7 or not len(means):
            raise ValueError("运行时 LINK 锚点必须包含非空 [H,7] 局部轨迹")
        if covariances.shape != (len(means), 6, 6):
            raise ValueError("运行时 LINK 锚点协方差必须为 [H,6,6]")
        if gripper.ndim != 2 or gripper.shape[0] != len(means):
            raise ValueError("运行时 LINK 锚点夹爪命令必须为 [H,G]")
        if not self.legal_reentry_states:
            raise ValueError("运行时 LINK 锚点必须提供至少一个合法重入状态")
        if self.source not in {
            "offline_link",
            "pending_recovery",
            "verified_pending",
        }:
            raise ValueError("未知 LINK 锚点来源")
        object.__setattr__(self, "local_means", means.copy())
        object.__setattr__(self, "local_covariances", covariances.copy())
        object.__setattr__(self, "gripper_commands", gripper.copy())

    @classmethod
    def from_offline(cls, anchor: LinkRecoveryAnchor) -> RuntimeLinkAnchor:
        if not anchor.linked_entry_states:
            raise ValueError("正式 LINK 锚点缺少合法 linked 状态")
        return cls(
            origin_event_id=anchor.event_id,
            arm_id=anchor.arm_id,
            frame_id=anchor.frame_id,
            context_state=anchor.context_state,
            local_means=anchor.local_means,
            local_covariances=anchor.local_covariances,
            gripper_commands=anchor.gripper_commands,
            legal_reentry_states=anchor.linked_entry_states,
            source="offline_link",
        )

    @classmethod
    def from_pending(
        cls,
        candidate: LinkPendingCandidate,
        *,
        verified: bool,
    ) -> RuntimeLinkAnchor:
        return cls(
            origin_event_id=candidate.event_id,
            arm_id=candidate.arm_id,
            frame_id=candidate.frame_id,
            context_state=candidate.context_state,
            local_means=candidate.local_means,
            local_covariances=candidate.local_covariances,
            gripper_commands=candidate.gripper_commands,
            legal_reentry_states=(candidate.candidate_state,),
            source="verified_pending" if verified else "pending_recovery",
        )


@dataclass(frozen=True)
class InstantiatedLinkWaypoint:
    sequence_index: int
    pose: Array
    covariance: Array
    gripper_command: Array

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        gripper = np.asarray(self.gripper_command, dtype=np.float64)
        if self.sequence_index < 0:
            raise ValueError("LINK 恢复路点索引必须非负")
        if pose.shape != (7,) or covariance.shape != (6, 6):
            raise ValueError("LINK 恢复路点必须使用 [7] 位姿和 [6,6] 协方差")
        if gripper.ndim != 1 or not len(gripper):
            raise ValueError("LINK 恢复路点夹爪命令必须为非空一维数组")
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(covariance)):
            raise ValueError("LINK 恢复路点包含非有限值")
        object.__setattr__(self, "pose", pose.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "gripper_command", gripper.copy())


class EpisodeLinkAnchorRegistry:
    """Resolve formal origins and activate verified Pending templates per episode."""

    def __init__(self, task_model: ClosedLoopTaskModel) -> None:
        self.task_model = task_model
        self.reset()

    def reset(self) -> None:
        self._active_pending: dict[str, RuntimeLinkAnchor] = {}

    @property
    def active_pending(self) -> dict[str, RuntimeLinkAnchor]:
        return dict(self._active_pending)

    def activate_pending(self, event_id: RelationEventId) -> RuntimeLinkAnchor:
        try:
            candidate = self.task_model.link_pending_events[event_id]
        except KeyError as exc:
            raise KeyError(f"不存在 LINK_PENDING 候选 {event_id.token}") from exc
        anchor = RuntimeLinkAnchor.from_pending(candidate, verified=True)
        self._active_pending[candidate.frame_id] = anchor
        return anchor

    def _pending_recovery_candidate(
        self,
        frame_id: str,
        state_id: StateId,
        mode: int,
    ) -> LinkPendingCandidate | None:
        """Return the latest unresolved Pending occurrence already reached.

        A Pending trajectory may be executed to repair the same failed LINK
        occurrence, but it is not installed in ``active_pending`` and therefore
        does not claim that a relation or ``link_origin`` already exists.  A
        later UNLINK invalidates older Pending candidates for recovery.
        """

        latest_unlink = max(
            (
                metadata.release_state
                for event_id, metadata in self.task_model.unlink_events.items()
                if event_id.arm_id == self.task_model.arm_id
                and event_id.frame_id == frame_id
                and event_id.mode == mode
                and metadata.release_state <= state_id
            ),
            default=None,
        )
        candidates = [
            candidate
            for event_id, candidate in self.task_model.link_pending_events.items()
            if event_id.arm_id == self.task_model.arm_id
            and event_id.frame_id == frame_id
            and event_id.mode == mode
            and candidate.candidate_state <= state_id
            and (latest_unlink is None or candidate.candidate_state > latest_unlink)
        ]
        if not candidates:
            return None
        latest_state = max(candidate.candidate_state for candidate in candidates)
        latest = [
            candidate
            for candidate in candidates
            if candidate.candidate_state == latest_state
        ]
        if len(latest) != 1:
            raise RuntimeError(
                "同一 arm-frame-mode 状态匹配到多个 LINK_PENDING occurrence"
            )
        return latest[0]

    def has_pending_recovery_candidate(
        self,
        frame_id: str,
        state_id: StateId,
        mode: int,
    ) -> bool:
        return self._pending_recovery_candidate(frame_id, state_id, mode) is not None

    def resolve_for_recovery(
        self,
        frame_id: str,
        state_id: StateId,
        mode: int,
    ) -> RuntimeLinkAnchor:
        """Resolve a confirmed origin or a non-activated Pending repair template."""

        try:
            return self.resolve(frame_id, state_id, mode)
        except KeyError as original_error:
            candidate = self._pending_recovery_candidate(frame_id, state_id, mode)
            if candidate is None:
                raise original_error
            return RuntimeLinkAnchor.from_pending(candidate, verified=False)

    def release(self, frame_id: str) -> None:
        """Clear only an intentionally released episode-scoped relation origin."""

        self._active_pending.pop(frame_id, None)

    def resolve(
        self,
        frame_id: str,
        state_id: StateId,
        mode: int,
    ) -> RuntimeLinkAnchor:
        key = RelationStateKey(self.task_model.arm_id, frame_id, state_id, mode)
        formal_event = self.task_model.link_origins.get(key)
        if formal_event is not None:
            return RuntimeLinkAnchor.from_offline(
                self.task_model.link_anchors[formal_event]
            )

        runtime = self._active_pending.get(frame_id)
        if runtime is not None:
            return runtime

        # The origin table is the primary lookup.  Retain a strict fallback for
        # sidecars produced before an origin was copied to every linked state.
        candidates = [
            anchor
            for event_id, anchor in self.task_model.link_anchors.items()
            if event_id.arm_id == self.task_model.arm_id
            and event_id.frame_id == frame_id
            and event_id.mode == mode
            and state_id in anchor.linked_entry_states
        ]
        if len(candidates) == 1:
            return RuntimeLinkAnchor.from_offline(candidates[0])
        if not candidates:
            raise KeyError(
                f"状态 {state_id} 的关系 {self.task_model.arm_id}/{frame_id} "
                "没有事件级 LINK 来源"
            )
        raise RuntimeError("同一关系状态匹配到多个 LINK 事件锚点")

    @staticmethod
    def instantiate(
        anchor: RuntimeLinkAnchor,
        frame_pose: Array,
        covariance_inflation: float,
    ) -> tuple[InstantiatedLinkWaypoint, ...]:
        if covariance_inflation < 0.0 or not np.isfinite(covariance_inflation):
            raise ValueError("恢复协方差放宽量必须为有限非负数")
        result = []
        for index, (mean, covariance, gripper) in enumerate(
            zip(
                anchor.local_means,
                anchor.local_covariances,
                anchor.gripper_commands,
                strict=True,
            )
        ):
            widened = covariance + np.eye(6, dtype=np.float64) * covariance_inflation
            marginal = transform_marginal(
                anchor.frame_id,
                frame_pose,
                mean,
                widened,
            )
            result.append(
                InstantiatedLinkWaypoint(
                    sequence_index=index,
                    pose=marginal.mean,
                    covariance=marginal.covariance,
                    gripper_command=gripper,
                )
            )
        return tuple(result)


__all__ = [
    "EpisodeLinkAnchorRegistry",
    "InstantiatedLinkWaypoint",
    "RuntimeLinkAnchor",
]
