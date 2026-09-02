"""Relation recovery goals and gripper-resource ordering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .frame_roles import RelationRecoveryIntent
from .link_anchors import EpisodeLinkAnchorRegistry, RuntimeLinkAnchor
from .relation_events import UnlinkEventMetadata
from .relation_filter import RelationDecision
from .state_index import StateId
from .unlink_metadata import UnlinkMetadataRepository


class RelationGoalKind(str, Enum):
    LINK = "link"
    UNLINK = "unlink"


@dataclass(frozen=True)
class RelationGoal:
    arm_id: str
    frame_id: str
    kind: RelationGoalKind
    expected_relation: RelationDecision
    actual_relation: RelationDecision
    source_state: StateId
    mode: int
    link_anchor: RuntimeLinkAnchor | None = None
    unlink_metadata: UnlinkEventMetadata | None = None

    def __post_init__(self) -> None:
        if not self.arm_id or not self.frame_id or self.mode < 0:
            raise ValueError("关系恢复目标的机械臂、参考系或 mode 无效")
        if RelationDecision.UNKNOWN in {
            self.expected_relation,
            self.actual_relation,
        }:
            raise ValueError("关系恢复目标必须来自可靠二元失配")
        expected_kind = (
            RelationGoalKind.LINK
            if self.expected_relation == RelationDecision.LINKED
            else RelationGoalKind.UNLINK
        )
        if self.kind != expected_kind:
            raise ValueError("关系恢复目标方向与期望关系不一致")
        if self.kind == RelationGoalKind.LINK:
            if self.actual_relation != RelationDecision.EXTERNAL:
                raise ValueError("LINK 目标要求当前关系为 external")
            if self.link_anchor is None or self.unlink_metadata is not None:
                raise ValueError("LINK 目标必须且只能携带 LINK 锚点")
        else:
            if self.actual_relation != RelationDecision.LINKED:
                raise ValueError("UNLINK 目标要求当前关系为 linked")
            if self.unlink_metadata is None or self.link_anchor is not None:
                raise ValueError("UNLINK 目标必须且只能携带 UNLINK 元数据")

    @property
    def legal_reentry_states(self) -> tuple[StateId, ...]:
        if self.kind == RelationGoalKind.LINK:
            assert self.link_anchor is not None
            return self.link_anchor.legal_reentry_states
        assert self.unlink_metadata is not None
        return self.unlink_metadata.legal_reentry_states


class RelationGoalPlanner:
    """Translate reliable mismatches and release gripper resources first."""

    def __init__(
        self,
        anchor_registry: EpisodeLinkAnchorRegistry,
        unlink_repository: UnlinkMetadataRepository,
    ) -> None:
        if anchor_registry.task_model is not unlink_repository.task_model:
            raise ValueError("关系目标规划器的 LINK/UNLINK 存储必须属于同一任务模型")
        self.anchor_registry = anchor_registry
        self.unlink_repository = unlink_repository
        self.task_model = anchor_registry.task_model

    def plan(
        self,
        intents: Sequence[RelationRecoveryIntent],
        *,
        source_state: StateId,
        mode: int,
    ) -> tuple[RelationGoal, ...]:
        by_frame: dict[str, RelationRecoveryIntent] = {}
        for intent in intents:
            if intent.arm_id != self.task_model.arm_id:
                raise ValueError("恢复意图不属于当前机械臂")
            previous = by_frame.get(intent.frame_id)
            if previous is not None and previous != intent:
                raise ValueError("同一参考系收到相互冲突的关系恢复意图")
            by_frame[intent.frame_id] = intent

        goals = []
        for frame, intent in sorted(by_frame.items()):
            if intent.expected_relation == RelationDecision.LINKED:
                goals.append(
                    RelationGoal(
                        arm_id=intent.arm_id,
                        frame_id=frame,
                        kind=RelationGoalKind.LINK,
                        expected_relation=intent.expected_relation,
                        actual_relation=intent.actual_relation,
                        source_state=source_state,
                        mode=mode,
                        link_anchor=(
                            self.anchor_registry.resolve_for_recovery(
                                frame, source_state, mode
                            )
                            if intent.origin_event_id is None
                            else self.anchor_registry.resolve_event_for_recovery(
                                intent.origin_event_id
                            )
                        ),
                    )
                )
            else:
                goals.append(
                    RelationGoal(
                        arm_id=intent.arm_id,
                        frame_id=frame,
                        kind=RelationGoalKind.UNLINK,
                        expected_relation=intent.expected_relation,
                        actual_relation=intent.actual_relation,
                        source_state=source_state,
                        mode=mode,
                        unlink_metadata=self.unlink_repository.resolve(
                            frame, source_state, mode
                        ),
                    )
                )

        # One gripper cannot establish a new relation until every relation that
        # currently occupies it has been intentionally released.
        return tuple(
            sorted(
                goals,
                key=lambda goal: (
                    0 if goal.kind == RelationGoalKind.UNLINK else 1,
                    goal.frame_id,
                ),
            )
        )

    def plan_available(
        self,
        intents: Sequence[RelationRecoveryIntent],
        *,
        source_state: StateId,
        mode: int,
    ) -> tuple[tuple[RelationGoal, ...], tuple[RelationRecoveryIntent, ...]]:
        """Plan only recovery goals grounded by learned event metadata.

        A reliable online mismatch does not imply that successful
        demonstrations contain the inverse relation-changing event.  In
        particular, an unexpectedly early ``linked`` decision can occur for a
        relation that is normally linked once and never released.  Such an
        observation must keep normal advancement blocked, but it cannot be
        converted into an invented UNLINK trajectory.

        The strict :meth:`plan` API remains unchanged for callers that require
        every intent to be resolvable.  Runtime orchestration uses this method
        to keep unsupported intents diagnostic-only while continuing to plan
        every learned recovery primitive that is available.
        """

        by_frame: dict[str, RelationRecoveryIntent] = {}
        for intent in intents:
            if intent.arm_id != self.task_model.arm_id:
                raise ValueError("恢复意图不属于当前机械臂")
            previous = by_frame.get(intent.frame_id)
            if previous is not None and previous != intent:
                raise ValueError("同一参考系收到相互冲突的关系恢复意图")
            by_frame[intent.frame_id] = intent

        goals: list[RelationGoal] = []
        unavailable: list[RelationRecoveryIntent] = []
        for intent in (by_frame[frame] for frame in sorted(by_frame)):
            try:
                goals.extend(
                    self.plan(
                        (intent,),
                        source_state=source_state,
                        mode=mode,
                    )
                )
            except KeyError:
                unavailable.append(intent)
        return (
            tuple(
                sorted(
                    goals,
                    key=lambda goal: (
                        0 if goal.kind == RelationGoalKind.UNLINK else 1,
                        goal.frame_id,
                    ),
                )
            ),
            tuple(unavailable),
        )


__all__ = ["RelationGoal", "RelationGoalKind", "RelationGoalPlanner"]
