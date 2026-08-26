"""Dynamic roles for selected experts and event-confirmed relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from .belief_updater import ClosedLoopBelief
from .relation_events import RelationEventId, RelationStateKey
from .relation_filter import RelationDecision
from .state_index import StateId
from .task_model import ClosedLoopTaskModel

Array = np.ndarray


class FrameRole(str, Enum):
    EXECUTE = "execute"
    MONITOR = "monitor"
    RECOVER = "recover"
    DEFER = "defer"


@dataclass(frozen=True)
class RelationRecoveryIntent:
    arm_id: str
    frame_id: str
    expected_relation: RelationDecision
    actual_relation: RelationDecision

    def __post_init__(self) -> None:
        if not self.arm_id or not self.frame_id:
            raise ValueError("关系恢复意图需要非空机械臂和参考系")
        if RelationDecision.UNKNOWN in {
            self.expected_relation,
            self.actual_relation,
        }:
            raise ValueError("可靠关系恢复意图不能以 Unknown 为端点")
        if self.expected_relation == self.actual_relation:
            raise ValueError("关系恢复意图必须包含实际失配")


@dataclass(frozen=True)
class RelationVerificationRequest:
    arm_id: str
    frame_id: str
    relation: str
    pending_event_id: RelationEventId

    def __post_init__(self) -> None:
        if not self.arm_id or not self.frame_id:
            raise ValueError("主动关系验证请求需要非空机械臂和参考系")
        if self.relation != "linked":
            raise ValueError("阶段三只生成 linked 主动验证请求")
        if (
            self.pending_event_id.transition != "link_pending"
            or self.pending_event_id.arm_id != self.arm_id
            or self.pending_event_id.frame_id != self.frame_id
        ):
            raise ValueError("主动关系验证请求与 LINK_PENDING 事件不一致")


@dataclass(frozen=True)
class FrameRoleDecision:
    frame_id: str
    role: FrameRole
    selected_offline: bool
    expected_distribution: Array | None
    expected_relation: RelationDecision | None
    actual_relation: RelationDecision | None
    relation_compatibility: float
    execution_weight: float
    monitor: bool
    blocks_advance: bool
    recovery_intent: RelationRecoveryIntent | None = None

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("流角色需要非空参考系")
        if self.expected_distribution is not None:
            distribution = np.asarray(self.expected_distribution, dtype=np.float64)
            if (
                distribution.shape != (2,)
                or np.any(distribution < 0.0)
                or not np.isclose(np.sum(distribution), 1.0)
            ):
                raise ValueError("期望关系必须是归一化 external/linked 向量")
            object.__setattr__(self, "expected_distribution", distribution.copy())
        if not 0.0 <= self.relation_compatibility <= 1.0:
            raise ValueError("关系兼容度必须位于 [0,1]")
        if not np.isfinite(self.execution_weight) or self.execution_weight < 0.0:
            raise ValueError("执行精度权重必须为有限非负数")
        if self.role == FrameRole.MONITOR and not self.monitor:
            raise ValueError("MONITOR 角色必须启用关系监测标志")
        if self.role in {FrameRole.MONITOR, FrameRole.RECOVER} and (
            self.execution_weight != 0.0
        ):
            raise ValueError("MONITOR/RECOVER 不能参与正常 PoE")


@dataclass(frozen=True)
class FrameRoleSnapshot:
    state_id: StateId
    decisions: dict[str, FrameRoleDecision]
    recovery_intents: tuple[RelationRecoveryIntent, ...] = ()
    verification_requests: tuple[RelationVerificationRequest, ...] = ()

    @property
    def execution_weights(self) -> dict[str, float]:
        return {
            frame: decision.execution_weight
            for frame, decision in self.decisions.items()
            if decision.selected_offline
        }

    @property
    def blocks_advance(self) -> bool:
        return any(decision.blocks_advance for decision in self.decisions.values())


@dataclass(frozen=True)
class FrameRoleConfig:
    expected_relation_probability: float = 0.70
    minimum_tracking_reliability: float = 0.25
    minimum_information_weight: float = 0.10

    def __post_init__(self) -> None:
        if not 0.5 < self.expected_relation_probability <= 1.0:
            raise ValueError("期望关系判定阈值必须位于 (0.5,1]")
        for value in (
            self.minimum_tracking_reliability,
            self.minimum_information_weight,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("流角色可靠性阈值必须位于 [0,1]")


class FrameRoleRouter:
    """Route retained experts while monitoring confirmed inactive relations."""

    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: FrameRoleConfig = FrameRoleConfig(),
    ) -> None:
        self.task_model = task_model
        self.config = config
        self._states = tuple(sorted(task_model.states))
        self._global_index = {state: index for index, state in enumerate(self._states)}
        # A stable MONITOR or RECOVER decision stores zero.  Therefore a later
        # one-frame Unknown cannot accidentally revive a formerly linked flow.
        self._last_trusted_weights: dict[str, float] = {}

    def reset(self) -> None:
        self._last_trusted_weights.clear()

    def commit(
        self,
        snapshot: FrameRoleSnapshot,
        belief: ClosedLoopBelief | None = None,
    ) -> None:
        """Commit only stable role weights after the controller validates a cycle."""

        # Relation filtering observes every modeled physical frame even when
        # that frame is not selected by the current skill.  Cache its latest
        # stable relation-derived weight without assigning it an active role;
        # this lets a newly selected DEFER stream reuse genuine prior evidence.
        if belief is not None:
            features = belief.runtime_features
            for frame, estimate in belief.relation_estimates.items():
                if frame not in self.task_model.relation_frames:
                    continue
                reliability = features.tracking_reliability.get(frame, 0.0)
                visible = features.frame_visibility.get(frame, False)
                if (
                    not visible
                    or reliability < self.config.minimum_tracking_reliability
                ):
                    continue
                if estimate.decision_state == RelationDecision.EXTERNAL:
                    self._last_trusted_weights[frame] = reliability * estimate.external
                elif estimate.decision_state == RelationDecision.LINKED:
                    self._last_trusted_weights[frame] = 0.0
        for frame, decision in snapshot.decisions.items():
            if decision.role != FrameRole.DEFER:
                self._last_trusted_weights[frame] = decision.execution_weight

    @staticmethod
    def _mode_for_state(
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> int | None:
        return (
            None if mode_by_skill is None else mode_by_skill.get(state_id.skill_index)
        )

    def _selected_frames(
        self,
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[str, ...]:
        node = self.task_model.state(state_id)
        mode = self._mode_for_state(state_id, mode_by_skill)
        if mode is None:
            return node.selected_frames
        if mode < 0 or mode >= len(node.mode_selected_frames):
            raise IndexError("mode_index 超出流角色模型的模态范围")
        return node.mode_selected_frames[mode]

    def _expected_distribution(
        self,
        frame: str,
        posterior: Mapping[StateId, float],
        mode_by_skill: Mapping[int, int] | None,
    ) -> Array:
        result = np.zeros(2, dtype=np.float64)
        total = 0.0
        for state_id, probability in posterior.items():
            if probability <= 0.0:
                continue
            node = self.task_model.state(state_id)
            priors = node.demo_relation_priors.get(frame)
            if priors is None:
                prior = np.asarray([0.5, 0.5], dtype=np.float64)
            else:
                mode = self._mode_for_state(state_id, mode_by_skill)
                if mode is None:
                    prior = np.sum(node.mode_priors[:, None] * priors, axis=0)
                else:
                    if mode < 0 or mode >= len(priors):
                        raise IndexError("mode_index 超出期望关系模型的模态范围")
                    prior = priors[mode]
            result += float(probability) * prior
            total += float(probability)
        if total <= 0.0:
            raise ValueError("进度后验必须具有正概率质量")
        result /= total
        result /= np.sum(result)
        return result

    def _expected_decision(self, distribution: Array) -> RelationDecision:
        maximum = float(np.max(distribution))
        if maximum < self.config.expected_relation_probability:
            return RelationDecision.UNKNOWN
        return (
            RelationDecision.LINKED
            if distribution[1] > distribution[0]
            else RelationDecision.EXTERNAL
        )

    def _confirmed_monitor_frames(
        self,
        posterior: Mapping[StateId, float],
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[str, ...]:
        """Return event-confirmed linked relations, including inactive experts."""

        frames = set()
        for state_id, probability in posterior.items():
            if probability <= 0.0:
                continue
            node = self.task_model.state(state_id)
            mode = self._mode_for_state(state_id, mode_by_skill)
            modes = range(len(node.mode_priors)) if mode is None else (mode,)
            for candidate_mode in modes:
                for frame in self.task_model.relation_frames:
                    key = RelationStateKey(
                        self.task_model.arm_id,
                        frame,
                        state_id,
                        candidate_mode,
                    )
                    if key in self.task_model.link_origins:
                        frames.add(frame)
        return tuple(sorted(frames))

    def _pending_relation_needed(
        self,
        event_id: RelationEventId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> bool:
        candidate = self.task_model.link_pending_events[event_id]
        start = self._global_index[candidate.candidate_state]
        for state_id in self._states[start + 1 :]:
            node = self.task_model.state(state_id)
            mode = self._mode_for_state(state_id, mode_by_skill)
            if mode is None:
                selected = candidate.frame_id in node.selected_frames
                priors = node.demo_relation_priors.get(candidate.frame_id)
                linked = (
                    0.0
                    if priors is None
                    else float(np.sum(node.mode_priors * priors[:, 1]))
                )
            else:
                if mode < 0 or mode >= len(node.mode_selected_frames):
                    raise IndexError("mode_index 超出 Pending 后续状态模态范围")
                selected = candidate.frame_id in node.mode_selected_frames[mode]
                priors = node.demo_relation_priors.get(candidate.frame_id)
                linked = 0.0 if priors is None else float(priors[mode, 1])
            if selected and linked >= self.config.expected_relation_probability:
                return True

        boundary = next(
            (
                model
                for boundary_id, model in sorted(self.task_model.boundaries.items())
                if boundary_id.source_skill == candidate.candidate_state.skill_index
            ),
            None,
        )
        if boundary is None:
            return False
        key = f"{self.task_model.arm_id}/{candidate.frame_id}"
        conditions = {
            **boundary.local_completion_model.own_relation_conditions,
            **boundary.relation_conditions,
        }
        condition = conditions.get(key)
        return condition is not None and condition.required_state == "linked"

    def _verification_requests(
        self,
        state_id: StateId,
        estimated_state: StateId,
        belief: ClosedLoopBelief,
        selected_frames: frozenset[str],
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[RelationVerificationRequest, ...]:
        requests = []
        for event_id, candidate in sorted(self.task_model.link_pending_events.items()):
            if candidate.candidate_state not in {state_id, estimated_state}:
                continue
            mode = self._mode_for_state(candidate.candidate_state, mode_by_skill)
            if mode is not None and mode != event_id.mode:
                continue
            frame = candidate.frame_id
            estimate = belief.relation_estimates.get(frame)
            features = belief.runtime_features
            if (
                frame not in selected_frames
                or estimate is None
                or estimate.decision_state != RelationDecision.UNKNOWN
                or not self._pending_relation_needed(event_id, mode_by_skill)
                or not features.frame_pair_available.get(frame, False)
                or features.paired_tracking_reliability.get(frame, 0.0)
                < self.config.minimum_tracking_reliability
                or estimate.information_weight >= self.config.minimum_information_weight
            ):
                continue
            requests.append(
                RelationVerificationRequest(
                    arm_id=self.task_model.arm_id,
                    frame_id=frame,
                    relation="linked",
                    pending_event_id=event_id,
                )
            )
        return tuple(requests)

    def route(
        self,
        state_id: StateId,
        belief: ClosedLoopBelief,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
        commit: bool = True,
    ) -> FrameRoleSnapshot:
        if state_id not in self.task_model.states:
            raise KeyError(f"流角色引用未知状态 {state_id}")
        selected_frames = self._selected_frames(state_id, mode_by_skill)
        selected = frozenset(selected_frames)
        confirmed_monitors = tuple(
            frame
            for frame in self._confirmed_monitor_frames(
                belief.progress.posterior,
                mode_by_skill,
            )
            if frame not in selected
        )
        decisions: dict[str, FrameRoleDecision] = {}
        recovery_intents = []
        features = belief.runtime_features

        for frame in (*selected_frames, *confirmed_monitors):
            selected_offline = frame in selected
            reliability = features.tracking_reliability.get(frame, 0.0)
            if frame.startswith("virtual_skill_"):
                weight = (
                    reliability if features.frame_visibility.get(frame, False) else 0.0
                )
                decisions[frame] = FrameRoleDecision(
                    frame_id=frame,
                    role=FrameRole.EXECUTE,
                    selected_offline=selected_offline,
                    expected_distribution=None,
                    expected_relation=None,
                    actual_relation=None,
                    relation_compatibility=1.0,
                    execution_weight=weight,
                    monitor=False,
                    blocks_advance=False,
                )
                continue

            distribution = self._expected_distribution(
                frame, belief.progress.posterior, mode_by_skill
            )
            expected = self._expected_decision(distribution)
            if not selected_offline and expected != RelationDecision.LINKED:
                # Event-confirmed monitoring never re-enables an inactive
                # external expert and never broadens PoE participation.
                continue
            estimate = belief.relation_estimates.get(frame)
            actual = (
                RelationDecision.UNKNOWN
                if estimate is None
                else estimate.decision_state
            )
            compatibility = (
                0.5
                if estimate is None
                else float(np.dot(estimate.posterior, distribution))
            )
            if (
                expected == RelationDecision.UNKNOWN
                or actual == RelationDecision.UNKNOWN
            ):
                role = FrameRole.DEFER
                weight = (
                    self._last_trusted_weights.get(frame, 0.0)
                    if selected_offline
                    else 0.0
                )
                monitor = False
                blocks = True
                intent = None
            elif expected == actual == RelationDecision.EXTERNAL:
                role = FrameRole.EXECUTE
                external = 0.0 if estimate is None else estimate.external
                weight = reliability * external
                monitor = False
                blocks = False
                intent = None
            elif expected == actual == RelationDecision.LINKED:
                role = FrameRole.MONITOR
                weight = 0.0
                monitor = True
                blocks = False
                intent = None
            else:
                role = FrameRole.RECOVER
                weight = 0.0
                monitor = False
                blocks = True
                assert expected != RelationDecision.UNKNOWN
                assert actual != RelationDecision.UNKNOWN
                intent = RelationRecoveryIntent(
                    arm_id=self.task_model.arm_id,
                    frame_id=frame,
                    expected_relation=expected,
                    actual_relation=actual,
                )
                recovery_intents.append(intent)
            decisions[frame] = FrameRoleDecision(
                frame_id=frame,
                role=role,
                selected_offline=selected_offline,
                expected_distribution=distribution,
                expected_relation=expected,
                actual_relation=actual,
                relation_compatibility=compatibility,
                execution_weight=weight,
                monitor=monitor,
                blocks_advance=blocks,
                recovery_intent=intent,
            )

        requests = self._verification_requests(
            state_id,
            belief.progress.estimated_state,
            belief,
            selected,
            mode_by_skill,
        )
        snapshot = FrameRoleSnapshot(
            state_id=state_id,
            decisions=decisions,
            recovery_intents=tuple(recovery_intents),
            verification_requests=requests,
        )
        if commit:
            self.commit(snapshot, belief)
        return snapshot


__all__ = [
    "FrameRole",
    "FrameRoleConfig",
    "FrameRoleDecision",
    "FrameRoleRouter",
    "FrameRoleSnapshot",
    "RelationRecoveryIntent",
    "RelationVerificationRequest",
]
