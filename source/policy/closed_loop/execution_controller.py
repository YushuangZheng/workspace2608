"""Unified phase-three owner of HOLD/REALIGN/ADVANCE cursor commits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..dynamac import DynaMACObservation
from .belief_updater import ClosedLoopBelief
from .execution_cursor import ClosedLoopCursor, ExecutionDecision
from .frame_roles import (
    FrameRoleConfig,
    FrameRoleRouter,
    FrameRoleSnapshot,
)
from .mismatch import (
    MismatchConfig,
    MismatchTracker,
    MismatchUpdate,
)
from .progress_filter import ProgressStatus
from .state_index import StateId
from .task_model import ClosedLoopTaskModel
from .weighted_poe import WeightedPoEExecutor, WeightedPoEResult


@dataclass(frozen=True)
class ClosedLoopExecutionConfig:
    frame_roles: FrameRoleConfig = field(default_factory=FrameRoleConfig)
    mismatch: MismatchConfig = field(default_factory=MismatchConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClosedLoopExecutionConfig:
        known = {"frame_roles", "mismatch"}
        unknown = set(value).difference(known)
        if unknown:
            raise ValueError(f"阶段三配置包含未知分区：{sorted(unknown)}")
        raw_roles = value.get("frame_roles", {})
        raw_mismatch = value.get("mismatch", {})
        if not isinstance(raw_roles, Mapping) or not isinstance(raw_mismatch, Mapping):
            raise TypeError("阶段三配置分区必须为对象")
        return cls(
            frame_roles=FrameRoleConfig(**dict(raw_roles)),
            mismatch=MismatchConfig(**dict(raw_mismatch)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ClosedLoopExecutionConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("阶段三配置文件根节点必须为对象")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class ExecutionCycleResult:
    tick: int
    cursor_before: ClosedLoopCursor
    cursor_after: ClosedLoopCursor
    decision: ExecutionDecision
    reasons: tuple[str, ...]
    roles: FrameRoleSnapshot
    weighted_action: WeightedPoEResult
    mismatch: MismatchUpdate


class ClosedLoopExecutionController:
    """Prepare and commit every normal-task reference-state change in one place."""

    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: ClosedLoopExecutionConfig = ClosedLoopExecutionConfig(),
    ) -> None:
        self.task_model = task_model
        self.config = config
        self.role_router = FrameRoleRouter(task_model, config.frame_roles)
        self.weighted_poe = WeightedPoEExecutor(task_model)
        self.mismatch_tracker = MismatchTracker(config.mismatch)
        self.reset()

    def reset(self, initial_state: StateId | None = None) -> None:
        if initial_state is None:
            initial_state = min(self.task_model.states)
        if initial_state not in self.task_model.states:
            raise KeyError(f"闭环执行器初始状态不存在：{initial_state}")
        self._cursor = ClosedLoopCursor(
            nominal_state=initial_state,
            estimated_state=initial_state,
            reference_state=initial_state,
        )
        self.role_router.reset()
        self.mismatch_tracker.reset()
        self._last_tick: int | None = None

    @property
    def cursor(self) -> ClosedLoopCursor:
        return self._cursor

    def _decision(
        self,
        belief: ClosedLoopBelief,
        roles: FrameRoleSnapshot,
        successor_ready: bool,
    ) -> tuple[ExecutionDecision, StateId, tuple[str, ...]]:
        current = self._cursor.reference_state
        nominal = belief.progress.nominal_state
        estimated = belief.progress.estimated_state
        reasons = []
        if belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE:
            reasons.append("no_plausible_state")
        elif belief.progress.status == ProgressStatus.LOW_CONFIDENCE:
            reasons.append("low_progress_confidence")
        if roles.blocks_advance:
            if roles.recovery_intents:
                reasons.append("reliable_relation_mismatch")
            else:
                reasons.append("critical_relation_unknown")
        if not successor_ready:
            reasons.append("successor_not_ready")
        if estimated.skill_index != current.skill_index:
            reasons.append("skill_boundary_requires_guard")
        if reasons:
            return ExecutionDecision.HOLD, current, tuple(dict.fromkeys(reasons))

        direct_successors = self.task_model.state(current).topology.successors
        same_skill_successors = tuple(
            state
            for state in direct_successors
            if state.skill_index == current.skill_index
        )

        # The action-after nominal state is the state expected to have just
        # been reached.  When both nominal and estimated equal the old cursor's
        # direct successor, commit exactly that successor rather than applying
        # succ(estimated) a second time.
        if estimated == nominal and estimated in same_skill_successors:
            return ExecutionDecision.ADVANCE, estimated, ("normal_successor_reached",)

        # If observation still explains the old target while the action-after
        # prior expected its successor, the target transition is incomplete.
        if estimated == current and nominal != current:
            return ExecutionDecision.HOLD, current, ("current_target_not_completed",)

        if estimated != nominal:
            if estimated == current:
                return (
                    ExecutionDecision.HOLD,
                    current,
                    ("current_target_not_completed",),
                )
            return (
                ExecutionDecision.REALIGN,
                estimated,
                ("trusted_progress_realignment",),
            )

        if estimated != current:
            # A trusted jump that is not a normal direct successor is an
            # inference correction, not repeated normal ADVANCE operations.
            return (
                ExecutionDecision.REALIGN,
                estimated,
                ("trusted_progress_realignment",),
            )

        return ExecutionDecision.HOLD, current, ("no_confirmed_successor",)

    @staticmethod
    def _mode_index(
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> int | None:
        return (
            None if mode_by_skill is None else mode_by_skill.get(state_id.skill_index)
        )

    def update(
        self,
        belief: ClosedLoopBelief,
        observation: DynaMACObservation,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
        successor_ready: bool = True,
    ) -> ExecutionCycleResult:
        if self._last_tick is not None and belief.tick <= self._last_tick:
            raise ValueError("闭环执行控制器每个递增控制周期只能提交一次")
        cursor_before = self._cursor
        if belief.progress.nominal_state not in self.task_model.states:
            raise KeyError("nominal_state 不在任务模型中")
        if belief.progress.estimated_state not in self.task_model.states:
            raise KeyError("estimated_state 不在任务模型中")

        roles = self.role_router.route(
            cursor_before.reference_state,
            belief,
            mode_by_skill=mode_by_skill,
            commit=False,
        )
        decision, proposed_reference, reasons = self._decision(
            belief, roles, successor_ready
        )
        if proposed_reference != cursor_before.reference_state:
            proposed_roles = self.role_router.route(
                proposed_reference,
                belief,
                mode_by_skill=mode_by_skill,
                commit=False,
            )
            if proposed_roles.blocks_advance:
                decision = ExecutionDecision.HOLD
                proposed_reference = cursor_before.reference_state
                reasons = tuple(
                    dict.fromkeys((*reasons, "proposed_reference_blocked_by_role"))
                )
            else:
                roles = proposed_roles

        weighted_action = self.weighted_poe.query(
            observation,
            proposed_reference,
            roles,
            mode_index=self._mode_index(proposed_reference, mode_by_skill),
        )
        if (
            not weighted_action.available
            and proposed_reference != cursor_before.reference_state
        ):
            decision = ExecutionDecision.HOLD
            proposed_reference = cursor_before.reference_state
            reasons = tuple(dict.fromkeys((*reasons, "no_positive_execution_stream")))
            roles = self.role_router.route(
                proposed_reference,
                belief,
                mode_by_skill=mode_by_skill,
                commit=False,
            )
            weighted_action = self.weighted_poe.query(
                observation,
                proposed_reference,
                roles,
                mode_index=self._mode_index(proposed_reference, mode_by_skill),
            )
        elif not weighted_action.available:
            decision = ExecutionDecision.HOLD
            reasons = tuple(dict.fromkeys((*reasons, "no_positive_execution_stream")))

        cursor_after = ClosedLoopCursor(
            nominal_state=belief.progress.nominal_state,
            estimated_state=belief.progress.estimated_state,
            reference_state=proposed_reference,
        )
        cursor_after.validate(self.task_model)
        mismatch = self.mismatch_tracker.update(
            belief,
            cursor_after,
            decision,
            roles,
        )

        self.role_router.commit(roles, belief)
        self._cursor = cursor_after
        self._last_tick = belief.tick
        return ExecutionCycleResult(
            tick=belief.tick,
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            decision=decision,
            reasons=reasons,
            roles=roles,
            weighted_action=weighted_action,
            mismatch=mismatch,
        )


__all__ = [
    "ClosedLoopExecutionConfig",
    "ClosedLoopExecutionController",
    "ExecutionCycleResult",
]
