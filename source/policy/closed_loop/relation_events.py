"""Event-level LINK anchors and UNLINK reentry metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_index import StateId

Array = np.ndarray


@dataclass(frozen=True, order=True)
class RelationEventId:
    arm_id: str
    frame_id: str
    skill_index: int
    mode: int
    occurrence: int
    transition: str

    def __post_init__(self) -> None:
        if not self.arm_id or not self.frame_id:
            raise ValueError("关系事件需要非空机械臂和参考系标识")
        if self.skill_index < 0 or self.mode < 0 or self.occurrence < 0:
            raise ValueError("关系事件索引必须为非负整数")
        if self.transition not in {"link", "link_pending", "unlink"}:
            raise ValueError("关系事件 transition 必须为 link、link_pending 或 unlink")

    @property
    def token(self) -> str:
        return (
            f"{self.arm_id}:{self.frame_id}:k{self.skill_index}:m{self.mode}:"
            f"{self.transition}:{self.occurrence}"
        )


@dataclass(frozen=True, order=True)
class RelationStateKey:
    """One mode component whose expected linked state has an origin event."""

    arm_id: str
    frame_id: str
    state_id: StateId
    mode: int

    def __post_init__(self) -> None:
        if not self.arm_id or not self.frame_id:
            raise ValueError("关系状态键需要非空机械臂和参考系标识")
        if self.mode < 0:
            raise ValueError("关系状态键 mode 必须为非负整数")

    @property
    def token(self) -> str:
        return (
            f"{self.arm_id}:{self.frame_id}:k{self.state_id.skill_index}:"
            f"t{self.state_id.local_index}:m{self.mode}"
        )


@dataclass(frozen=True)
class LinkRecoveryAnchor:
    event_id: RelationEventId
    arm_id: str
    frame_id: str
    context_state: StateId
    local_means: Array
    local_covariances: Array
    gripper_commands: Array
    linked_entry_states: tuple[StateId, ...]
    support_fraction: float = 1.0
    # LODO folds that reproduced the all-demo event; anchor fitting still uses
    # every normal demonstration in the event mode.
    demonstration_indices: tuple[int, ...] = ()
    event_local_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        means = np.asarray(self.local_means, dtype=np.float64)
        covariance = np.asarray(self.local_covariances, dtype=np.float64)
        gripper = np.asarray(self.gripper_commands, dtype=np.float64)
        if means.ndim != 2 or means.shape[1] != 7:
            raise ValueError("LINK 锚点均值必须为 [H,7]")
        if covariance.shape != (len(means), 6, 6):
            raise ValueError("LINK 锚点协方差必须为 [H,6,6]")
        if gripper.ndim != 2 or gripper.shape[0] != len(means):
            raise ValueError("LINK 锚点夹爪命令必须为 [H,G]")
        if not 0.0 <= self.support_fraction <= 1.0:
            raise ValueError("LINK 事件示范支持率必须位于 [0,1]")
        if len(self.demonstration_indices) != len(self.event_local_indices):
            raise ValueError("LINK 事件示范索引与对齐位置必须一一对应")
        object.__setattr__(self, "local_means", means.copy())
        object.__setattr__(self, "local_covariances", covariance.copy())
        object.__setattr__(self, "gripper_commands", gripper.copy())


@dataclass(frozen=True)
class LinkPendingCandidate:
    """A repeatable LINK hypothesis whose demonstrations lack motion evidence."""

    event_id: RelationEventId
    arm_id: str
    frame_id: str
    candidate_state: StateId
    context_state: StateId
    local_means: Array
    local_covariances: Array
    gripper_commands: Array
    support_fraction: float = 1.0
    demonstration_indices: tuple[int, ...] = ()
    event_local_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.event_id.transition != "link_pending":
            raise ValueError("LINK_PENDING 候选必须使用 link_pending 事件标识")
        if (
            self.event_id.arm_id != self.arm_id
            or self.event_id.frame_id != self.frame_id
            or self.event_id.skill_index != self.candidate_state.skill_index
        ):
            raise ValueError("LINK_PENDING 候选的机械臂、参考系或技能不一致")
        if not 0.0 <= self.support_fraction <= 1.0:
            raise ValueError("LINK_PENDING 候选示范支持率必须位于 [0,1]")
        if len(self.demonstration_indices) != len(self.event_local_indices):
            raise ValueError("LINK_PENDING 示范索引与对齐位置必须一一对应")
        means = np.asarray(self.local_means, dtype=np.float64)
        covariance = np.asarray(self.local_covariances, dtype=np.float64)
        gripper = np.asarray(self.gripper_commands, dtype=np.float64)
        if means.ndim != 2 or means.shape[1] != 7 or not len(means):
            raise ValueError("LINK_PENDING 局部轨迹均值必须为非空 [H,7]")
        if covariance.shape != (len(means), 6, 6):
            raise ValueError("LINK_PENDING 局部轨迹协方差必须为 [H,6,6]")
        if gripper.ndim != 2 or gripper.shape[0] != len(means):
            raise ValueError("LINK_PENDING 夹爪命令必须为 [H,G]")
        object.__setattr__(self, "local_means", means.copy())
        object.__setattr__(self, "local_covariances", covariance.copy())
        object.__setattr__(self, "gripper_commands", gripper.copy())


@dataclass(frozen=True)
class UnlinkEventMetadata:
    event_id: RelationEventId
    arm_id: str
    frame_id: str
    release_state: StateId
    legal_reentry_states: tuple[StateId, ...]
    local_detachment_target: Array | None = None
    support_fraction: float = 1.0
    # LODO folds that reproduced the all-demo event; target fitting still uses
    # every normal demonstration in the event mode.
    demonstration_indices: tuple[int, ...] = ()
    event_local_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.support_fraction <= 1.0:
            raise ValueError("UNLINK 事件示范支持率必须位于 [0,1]")
        if len(self.demonstration_indices) != len(self.event_local_indices):
            raise ValueError("UNLINK 事件示范索引与对齐位置必须一一对应")
        if self.local_detachment_target is not None:
            target = np.asarray(self.local_detachment_target, dtype=np.float64)
            if target.shape != (7,) or not np.all(np.isfinite(target)):
                raise ValueError("UNLINK 局部脱离目标必须为有限 [7] 位姿")
            object.__setattr__(self, "local_detachment_target", target.copy())


__all__ = [
    "LinkPendingCandidate",
    "LinkRecoveryAnchor",
    "RelationEventId",
    "RelationStateKey",
    "UnlinkEventMetadata",
]
