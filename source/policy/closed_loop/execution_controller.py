"""Unified phase-three owner of HOLD/REALIGN/ADVANCE cursor commits."""

from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..dynamac import DynaMACAction, DynaMACObservation, pose_log_nearest
from .belief_updater import ClosedLoopBelief
from .boundary_runtime import TransitionRequest
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
from .progress_filter import ProgressFilterConfig, ProgressStatus
from .state_index import StateId
from .task_model import ClosedLoopTaskModel
from .weighted_poe import WeightedPoEExecutor, WeightedPoEResult


@dataclass(frozen=True)
class ClosedLoopExecutionConfig:
    frame_roles: FrameRoleConfig = field(default_factory=FrameRoleConfig)
    mismatch: MismatchConfig = field(default_factory=MismatchConfig)
    minimum_action_equivalence_compatibility: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_action_equivalence_compatibility <= 1.0:
            raise ValueError("动作等价兼容度阈值必须位于 (0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClosedLoopExecutionConfig:
        known = {
            "frame_roles",
            "mismatch",
            "minimum_action_equivalence_compatibility",
        }
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
            minimum_action_equivalence_compatibility=float(
                value.get("minimum_action_equivalence_compatibility", 0.8)
            ),
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
    progress_anchor_state: StateId
    control_equivalence: ControlEquivalenceAssessment


@dataclass(frozen=True)
class ControlEquivalenceAssessment:
    """Audit whether a local discrete progress difference changes control.

    The raw phase-two posterior and status are deliberately left untouched.
    This record only states whether the posterior mass around its MAP state can
    be collapsed because every state in that class produces the same control
    semantics under the current observation.  Besides LOW_CONFIDENCE, the same
    audit is used for a high-confidence one-state BACKWARD_REALIGNMENT so a
    statistically preferred but control-equivalent predecessor cannot freeze
    an already reached command indefinitely.
    """

    evaluated: bool
    accepted: bool
    anchor_state: StateId
    equivalent_states: tuple[StateId, ...]
    aggregated_confidence: float
    normalized_class_entropy: float
    class_count: int
    minimum_action_compatibility: float
    reason: str


class ClosedLoopExecutionController:
    """Prepare and commit every normal-task reference-state change in one place."""

    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: ClosedLoopExecutionConfig = ClosedLoopExecutionConfig(),
        progress_filter_config: ProgressFilterConfig = ProgressFilterConfig(),
        *,
        dynamic_frame_roles: bool = True,
    ) -> None:
        self.task_model = task_model
        self.config = config
        self.progress_filter_config = progress_filter_config
        self.dynamic_frame_roles = bool(dynamic_frame_roles)
        self.role_router = FrameRoleRouter(task_model, config.frame_roles)
        self.weighted_poe = WeightedPoEExecutor(task_model)
        self.mismatch_tracker = MismatchTracker(config.mismatch)
        self.reset()

    def _route_roles(
        self,
        state_id: StateId,
        belief: ClosedLoopBelief,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
        captured_virtual_frames: frozenset[str] = frozenset(),
        commit: bool = True,
    ) -> FrameRoleSnapshot:
        route = (
            self.role_router.route
            if self.dynamic_frame_roles
            else self.role_router.route_fixed
        )
        return route(
            state_id,
            belief,
            mode_by_skill=mode_by_skill,
            captured_virtual_frames=captured_virtual_frames,
            commit=commit,
        )

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
        self._last_boundary_transition_tick: int | None = None

    @staticmethod
    def _action_compatibility(left: DynaMACAction, right: DynaMACAction) -> float:
        """Return dimension-normalized symmetric pose-target compatibility."""

        residual = pose_log_nearest(left.pose, right.pose)
        covariance = (
            np.asarray(left.covariance, dtype=np.float64)
            + np.asarray(right.covariance, dtype=np.float64)
            + np.eye(6, dtype=np.float64) * 1.0e-12
        )
        try:
            solved = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(covariance) @ residual
        mahalanobis = max(0.0, float(residual @ solved))
        return float(np.exp(max(-750.0, -0.5 * mahalanobis / 6.0)))

    @staticmethod
    def _roles_equivalent(
        left: FrameRoleSnapshot,
        right: FrameRoleSnapshot,
    ) -> bool:
        """Compare every control-relevant dynamic-role decision."""

        if set(left.decisions) != set(right.decisions):
            return False
        if (
            left.recovery_intents != right.recovery_intents
            or left.verification_requests != right.verification_requests
            or left.confirmed_link_events != right.confirmed_link_events
            or left.rejected_link_events != right.rejected_link_events
        ):
            return False
        for frame in sorted(left.decisions):
            lhs = left.decisions[frame]
            rhs = right.decisions[frame]
            if (
                lhs.role != rhs.role
                or lhs.selected_offline != rhs.selected_offline
                or lhs.expected_relation != rhs.expected_relation
                or lhs.actual_relation != rhs.actual_relation
                or lhs.monitor != rhs.monitor
                or lhs.blocks_advance != rhs.blocks_advance
                or lhs.recovery_intent != rhs.recovery_intent
                or lhs.formal_link_confirmation_pending
                != rhs.formal_link_confirmation_pending
                or not np.isclose(
                    lhs.execution_weight,
                    rhs.execution_weight,
                    rtol=1.0e-9,
                    atol=1.0e-12,
                )
            ):
                return False
            if (lhs.expected_distribution is None) != (
                rhs.expected_distribution is None
            ):
                return False
            if lhs.expected_distribution is not None:
                assert rhs.expected_distribution is not None
                if not np.allclose(
                    lhs.expected_distribution,
                    rhs.expected_distribution,
                    rtol=1.0e-9,
                    atol=1.0e-12,
                ):
                    return False
        return True

    def _state_controls(
        self,
        state_id: StateId,
        belief: ClosedLoopBelief,
        observation: DynaMACObservation,
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[FrameRoleSnapshot, WeightedPoEResult]:
        roles = self._route_roles(
            state_id,
            belief,
            mode_by_skill=mode_by_skill,
            commit=False,
        )
        action = self.weighted_poe.query(
            observation,
            state_id,
            roles,
            mode_index=self._mode_index(state_id, mode_by_skill),
        )
        return roles, action

    def _states_control_equivalent(
        self,
        left: StateId,
        right: StateId,
        controls: Mapping[StateId, tuple[FrameRoleSnapshot, WeightedPoEResult]],
    ) -> tuple[bool, float]:
        if left.skill_index != right.skill_index:
            return False, 0.0
        left_roles, left_weighted = controls[left]
        right_roles, right_weighted = controls[right]
        if not self._roles_equivalent(left_roles, right_roles):
            return False, 0.0
        left_action = left_weighted.action
        right_action = right_weighted.action
        if left_action is None or right_action is None:
            return False, 0.0
        if not np.allclose(
            np.atleast_1d(left_action.gripper),
            np.atleast_1d(right_action.gripper),
            rtol=0.0,
            atol=1.0e-8,
        ):
            return False, 0.0
        compatibility = self._action_compatibility(left_action, right_action)
        return (
            compatibility >= self.config.minimum_action_equivalence_compatibility,
            compatibility,
        )

    def _control_equivalence(
        self,
        belief: ClosedLoopBelief,
        observation: DynaMACObservation,
        mode_by_skill: Mapping[int, int] | None,
    ) -> ControlEquivalenceAssessment:
        """Collapse only contiguous, same-skill, pairwise-equivalent controls."""

        anchor = belief.progress.estimated_state
        eligible_statuses = {
            ProgressStatus.LOW_CONFIDENCE,
            ProgressStatus.BACKWARD_REALIGNMENT,
        }
        if belief.progress.status not in eligible_statuses:
            return ControlEquivalenceAssessment(
                evaluated=False,
                accepted=False,
                anchor_state=anchor,
                equivalent_states=(anchor,),
                aggregated_confidence=belief.progress.confidence,
                normalized_class_entropy=0.0,
                class_count=len(belief.progress.posterior),
                minimum_action_compatibility=1.0,
                reason="progress_status_not_control_equivalence_candidate",
            )

        states = tuple(
            sorted(
                state
                for state, probability in belief.progress.posterior.items()
                if probability > 0.0
            )
        )
        if not states or anchor not in states:
            raise ValueError("控制等价评估要求进度后验包含正概率 MAP 状态")
        controls = {
            state: self._state_controls(
                state,
                belief,
                observation,
                mode_by_skill,
            )
            for state in states
        }

        classes: list[list[StateId]] = []
        pair_compatibilities: dict[frozenset[StateId], float] = {}
        for state in states:
            if not classes:
                classes.append([state])
                continue
            current_class = classes[-1]
            previous = current_class[-1]
            contiguous = bool(
                state.skill_index == previous.skill_index
                and state.local_index == previous.local_index + 1
            )
            equivalent_to_all = contiguous
            for member in current_class if contiguous else ():
                equivalent, compatibility = self._states_control_equivalent(
                    member,
                    state,
                    controls,
                )
                pair_compatibilities[frozenset((member, state))] = compatibility
                if not equivalent:
                    equivalent_to_all = False
            if equivalent_to_all:
                current_class.append(state)
            else:
                classes.append([state])

        anchor_class = next(group for group in classes if anchor in group)
        class_masses = np.asarray(
            [
                sum(belief.progress.posterior[state] for state in group)
                for group in classes
            ],
            dtype=np.float64,
        )
        class_masses /= float(np.sum(class_masses))
        class_entropy = -float(
            np.sum(
                class_masses
                * np.log(
                    np.maximum(
                        class_masses,
                        self.progress_filter_config.probability_floor,
                    )
                )
            )
        )
        normalized_entropy = (
            0.0 if len(classes) == 1 else class_entropy / math.log(float(len(classes)))
        )
        confidence = float(
            sum(belief.progress.posterior[state] for state in anchor_class)
        )
        accepted = bool(
            confidence >= self.progress_filter_config.minimum_confidence
            and normalized_entropy
            <= self.progress_filter_config.maximum_normalized_entropy
        )
        if belief.progress.status == ProgressStatus.BACKWARD_REALIGNMENT:
            # A confident backward estimate is not statistical ambiguity.  It
            # may only be treated as a harmless discretization lag when the
            # commanded reference is its immediate same-skill successor and
            # both controls are already in the accepted MAP equivalence class.
            current = self._cursor.reference_state
            accepted = bool(
                accepted
                and current.skill_index == anchor.skill_index
                and current.local_index == anchor.local_index + 1
                and current in anchor_class
            )
        relevant_compatibilities = [
            pair_compatibilities[frozenset((left, right))]
            for index, left in enumerate(anchor_class)
            for right in anchor_class[index + 1 :]
        ]
        minimum_compatibility = (
            min(relevant_compatibilities) if relevant_compatibilities else 1.0
        )
        accepted_reason = (
            "control_equivalent_progress_uncertainty"
            if belief.progress.status == ProgressStatus.LOW_CONFIDENCE
            else "control_equivalent_backward_realignment"
        )
        rejected_reason = (
            "control_relevant_progress_uncertainty"
            if belief.progress.status == ProgressStatus.LOW_CONFIDENCE
            else "control_relevant_backward_realignment"
        )
        return ControlEquivalenceAssessment(
            evaluated=True,
            accepted=accepted,
            anchor_state=anchor,
            equivalent_states=tuple(anchor_class),
            aggregated_confidence=confidence,
            normalized_class_entropy=normalized_entropy,
            class_count=len(classes),
            minimum_action_compatibility=float(minimum_compatibility),
            reason=accepted_reason if accepted else rejected_reason,
        )

    @property
    def cursor(self) -> ClosedLoopCursor:
        return self._cursor

    def query_frozen_reference(
        self,
        belief: ClosedLoopBelief,
        observation: DynaMACObservation,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> WeightedPoEResult:
        """Query the current reference without advancing any TASK state.

        Recovery freezes the discrete task cursor, not the physical pose at
        which an estimator failure happened.  This read-only query therefore
        keeps servoing the frozen DynaMAC target while progress, role history,
        mismatch counters, and the task clock remain unchanged.
        """

        state_id = self._cursor.reference_state
        roles = self._route_roles(
            state_id,
            belief,
            mode_by_skill=mode_by_skill,
            commit=False,
        )
        return self.weighted_poe.query(
            observation,
            state_id,
            roles,
            mode_index=self._mode_index(state_id, mode_by_skill),
        )

    def query_reentry_alignment(
        self,
        state_id: StateId,
        belief: ClosedLoopBelief,
        observation: DynaMACObservation,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> WeightedPoEResult:
        """Query one legal recovery-alignment target without TASK commits.

        This is a read-only recovery action query: it does not change beta,
        the execution cursor, role history, mismatch counters, or the task
        clock.  A critical relation role still blocks the query exactly as it
        would block normal execution.
        """

        if state_id not in self.task_model.states:
            raise KeyError(f"恢复重入对齐引用未知状态 {state_id}")
        # Reentry selection has already checked this candidate's robot,
        # scene, and relation conditions.  Route the *alignment action* with
        # that same candidate-conditioned relation expectation instead of the
        # frozen fault state's beta-weighted expectation.  Otherwise a valid
        # earlier state can be selected for alignment and then be blocked by
        # the relation semantics of the state from which recovery started.
        #
        # This is a read-only routing view.  The committed belief, beta,
        # cursor, role history, and task clock stay frozen until the ordinary
        # full-state reentry decision succeeds below the existing thresholds.
        conditioned_progress = replace(
            belief.progress,
            prior={state_id: 1.0},
            posterior={state_id: 1.0},
            nominal_state=state_id,
            estimated_state=state_id,
        )
        conditioned_belief = replace(belief, progress=conditioned_progress)
        roles = self._route_roles(
            state_id,
            conditioned_belief,
            mode_by_skill=mode_by_skill,
            commit=False,
        )
        if roles.blocks_advance:
            return WeightedPoEResult(
                state_id=state_id,
                stream_weights=roles.execution_weights,
                participating_frames=(),
                action=None,
            )
        return self.weighted_poe.query(
            observation,
            state_id,
            roles,
            mode_index=self._mode_index(state_id, mode_by_skill),
        )

    def validate_boundary_transition(self, request: TransitionRequest) -> None:
        """Validate a staged cross-skill request without changing the cursor."""

        if request.arm_id != self.task_model.arm_id:
            raise ValueError("跨界请求不属于当前执行控制器")
        if not request.permitted:
            raise ValueError("未放行的跨界请求不能提交")
        if request.boundary_id not in self.task_model.boundaries:
            raise KeyError(f"任务模型不存在边界 {request.boundary_id.token}")
        if self._last_tick != request.tick:
            raise ValueError("跨界请求必须提交到产生它的同一控制周期")
        if self._last_boundary_transition_tick == request.tick:
            raise ValueError("同一机械臂每个周期最多提交一个跨界")
        if self._cursor.reference_state != request.source_state:
            raise ValueError("跨界请求源状态与执行游标不一致")
        boundary = self.task_model.boundaries[request.boundary_id]
        expected_target = self.task_model.skill_states[boundary.target_skill][0]
        if request.target_state != expected_target:
            raise ValueError("跨界目标必须是下一技能的首状态")

    def commit_boundary_transition(self, request: TransitionRequest) -> None:
        """Commit one already batch-validated boundary request."""

        self.validate_boundary_transition(request)
        self._cursor = ClosedLoopCursor(
            nominal_state=self._cursor.nominal_state,
            estimated_state=self._cursor.estimated_state,
            reference_state=request.target_state,
        )
        self._cursor.validate(self.task_model)
        self._last_boundary_transition_tick = request.tick

    def _decision(
        self,
        belief: ClosedLoopBelief,
        roles: FrameRoleSnapshot,
        control_equivalence: ControlEquivalenceAssessment,
        current_discrete_action_complete: bool,
        action_executed: bool,
    ) -> tuple[ExecutionDecision, StateId, tuple[str, ...]]:
        current = self._cursor.reference_state
        nominal = belief.progress.nominal_state
        estimated = belief.progress.estimated_state
        reasons = []
        if belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE:
            reasons.append("no_plausible_state")
        elif (
            belief.progress.status == ProgressStatus.LOW_CONFIDENCE
            and not control_equivalence.accepted
        ):
            reasons.append("low_progress_confidence")
        if roles.blocks_advance:
            if roles.recovery_intents:
                reasons.append("reliable_relation_mismatch")
            else:
                reasons.append("critical_relation_unknown")
        if not current_discrete_action_complete:
            # The progress posterior explains the continuous robot state, but
            # a discrete command stored on that same StateNode (currently the
            # gripper value) must not be skipped by a smooth cursor advance.
            # This is task-action sequencing, not an executor completion test:
            # reached/progressed/stopped never enter this decision.
            reasons.append("current_discrete_action_pending")
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

        # A trusted estimate behind the currently commanded reference means
        # that the target has not yet been reached.  Re-querying the earlier
        # state would physically replay the normal trajectory backward and can
        # create a two-state ADVANCE/REALIGN oscillation under ordinary plant
        # lag.  HOLD keeps servoing the existing target as required by the
        # current-target-incomplete rule.  Forward estimates remain eligible
        # for REALIGN below.
        if (
            estimated.skill_index == current.skill_index
            and estimated.local_index < current.local_index
        ):
            if (
                control_equivalence.accepted
                and estimated in control_equivalence.equivalent_states
                and current in control_equivalence.equivalent_states
            ):
                if same_skill_successors:
                    return (
                        ExecutionDecision.ADVANCE,
                        same_skill_successors[0],
                        ("control_equivalent_current_reference_reached",),
                    )
                return (
                    ExecutionDecision.HOLD,
                    current,
                    ("skill_boundary_requires_guard",),
                )
            return (
                ExecutionDecision.HOLD,
                current,
                ("current_target_incomplete",),
            )

        # If the observation is already explained by the old cursor's direct
        # successor, commit exactly that successor.  This is an early/fast
        # completion, but it must still advance at most one topology edge from
        # the previous reference in this control cycle.
        if estimated == nominal and estimated in same_skill_successors:
            return ExecutionDecision.ADVANCE, estimated, ("early_successor_reached",)

        # State distributions describe reached robot configurations.  Hence a
        # trusted estimate equal to the current reference means that target has
        # completed; the next command must servo its direct successor.  The
        # reset cycle remains a HOLD because its nominal state is also current
        # (there is no preceding executed action yet).
        if estimated == current and (nominal != current or action_executed):
            if same_skill_successors:
                return (
                    ExecutionDecision.ADVANCE,
                    same_skill_successors[0],
                    ("current_reference_reached",),
                )
            return ExecutionDecision.HOLD, current, ("skill_boundary_requires_guard",)

        if estimated != nominal:
            if estimated == current:
                return (
                    ExecutionDecision.HOLD,
                    current,
                    ("no_confirmed_successor",),
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
        current_discrete_action_complete: bool = True,
        action_executed: bool = False,
    ) -> ExecutionCycleResult:
        if self._last_tick is not None and belief.tick <= self._last_tick:
            raise ValueError("闭环执行控制器每个递增控制周期只能提交一次")
        cursor_before = self._cursor
        if belief.progress.nominal_state not in self.task_model.states:
            raise KeyError("nominal_state 不在任务模型中")
        if belief.progress.estimated_state not in self.task_model.states:
            raise KeyError("estimated_state 不在任务模型中")

        roles = self._route_roles(
            cursor_before.reference_state,
            belief,
            mode_by_skill=mode_by_skill,
            commit=False,
        )
        mismatch_roles = roles
        control_equivalence = self._control_equivalence(
            belief,
            observation,
            mode_by_skill,
        )
        decision, proposed_reference, reasons = self._decision(
            belief,
            roles,
            control_equivalence,
            current_discrete_action_complete,
            action_executed,
        )
        if control_equivalence.accepted:
            reasons = tuple(dict.fromkeys((control_equivalence.reason, *reasons)))
        if proposed_reference != cursor_before.reference_state:
            proposed_roles = self._route_roles(
                proposed_reference,
                belief,
                mode_by_skill=mode_by_skill,
                commit=False,
            )
            if proposed_roles.blocks_advance:
                decision = ExecutionDecision.HOLD
                proposed_reference = cursor_before.reference_state
                # The current state can be a bounded transition/grace state
                # whose own roles remain executable while the next state is
                # already blocked by a reliable relation mismatch.  Preserve
                # that proposed-state evidence for the persistent mismatch
                # tracker; otherwise the cursor can hold forever at the edge
                # of the grace window without ever creating a recovery intent.
                mismatch_roles = proposed_roles
                reasons = tuple(
                    dict.fromkeys((*reasons, "proposed_reference_blocked_by_role"))
                )
            else:
                roles = proposed_roles
                mismatch_roles = roles

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
            roles = self._route_roles(
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
            mismatch_roles,
            action_executed=action_executed,
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
            progress_anchor_state=weighted_action.state_id,
            control_equivalence=control_equivalence,
        )

    def query_after_boundary_transition(
        self,
        previous: ExecutionCycleResult,
        belief: ClosedLoopBelief,
        observation: DynaMACObservation,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> ExecutionCycleResult:
        """Build the guarded DynaMAC boundary-transition command.

        Stage four commits a guarded cross-skill transition after the normal
        phase-three update has already evaluated the source state.  Preserve
        the frozen DynaMAC hybrid boundary convention: execute the source
        skill's final continuous target while applying the committed entry
        state's gripper command.  The cursor and next-cycle progress anchor are
        nevertheless the committed entry state.  This prevents a terminal
        grasp/release waypoint from being skipped when the learned completion
        window permits a transition slightly before the final discrete index.

        Relation/progress inference, mismatch counters and the cursor
        transaction are deliberately not repeated.
        """

        if previous.tick != belief.tick or self._last_tick != belief.tick:
            raise ValueError("边界后动作重查询必须使用本周期信念与执行结果")
        if self._last_boundary_transition_tick != belief.tick:
            raise RuntimeError("只有本周期已提交的边界事务可以重查询入口动作")
        target = self._cursor.reference_state
        source_skill = previous.cursor_after.reference_state.skill_index
        if target.skill_index == source_skill:
            raise RuntimeError("边界后动作重查询要求 reference_state 已跨技能")
        source_terminal = self.task_model.skill_states[source_skill][-1]
        source_roles = self._route_roles(
            source_terminal,
            belief,
            mode_by_skill=mode_by_skill,
            captured_virtual_frames=frozenset(
                frame
                for frame in observation.frames
                if frame.startswith("virtual_skill_")
            ),
            commit=False,
        )
        source_weighted = self.weighted_poe.query(
            observation,
            source_terminal,
            source_roles,
            mode_index=self._mode_index(source_terminal, mode_by_skill),
        )
        if source_weighted.action is None:
            weighted_action = source_weighted
        else:
            entry_mode = self._mode_index(target, mode_by_skill)
            if entry_mode is None:
                entry_mode = int(
                    self.task_model.base_policy.selected_mode_path[target.skill_index]
                )
            entry_node = self.task_model.state(target)
            if entry_mode < 0 or entry_mode >= len(entry_node.gripper_commands):
                raise IndexError("边界入口模态超出 StateNode 夹爪命令范围")
            bridge_action = DynaMACAction(
                pose=source_weighted.action.pose.copy(),
                covariance=source_weighted.action.covariance.copy(),
                gripper=entry_node.gripper_commands[entry_mode].copy(),
                diagnostics={
                    **source_weighted.action.diagnostics,
                    "boundary_transition": {
                        "source_terminal_state": (
                            f"k{source_terminal.skill_index}:t{source_terminal.local_index}"
                        ),
                        "committed_entry_state": (
                            f"k{target.skill_index}:t{target.local_index}"
                        ),
                        "continuous_target_source": "source_terminal_state",
                        "gripper_target_source": "committed_entry_state",
                        "virtual_frame_capture": "next_post_action_observation",
                    },
                },
            )
            weighted_action = WeightedPoEResult(
                state_id=source_terminal,
                stream_weights=source_weighted.stream_weights,
                participating_frames=source_weighted.participating_frames,
                action=bridge_action,
            )
        self.role_router.commit(source_roles, belief)
        reasons: tuple[str, ...] = (
            "entry_guard_transaction_committed",
            "dynamac_boundary_terminal_bridge",
        )
        if not source_weighted.available:
            reasons = (*reasons, "no_positive_execution_stream")
        return ExecutionCycleResult(
            tick=belief.tick,
            cursor_before=previous.cursor_before,
            cursor_after=self._cursor,
            decision=ExecutionDecision.ADVANCE,
            reasons=reasons,
            roles=source_roles,
            weighted_action=weighted_action,
            mismatch=previous.mismatch,
            progress_anchor_state=target,
            control_equivalence=previous.control_equivalence,
        )


__all__ = [
    "ClosedLoopExecutionConfig",
    "ClosedLoopExecutionController",
    "ControlEquivalenceAssessment",
    "ExecutionCycleResult",
]
