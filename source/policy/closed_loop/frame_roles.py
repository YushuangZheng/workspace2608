"""Dynamic roles for selected experts and event-confirmed relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from .belief_updater import ClosedLoopBelief
from .relation_events import RelationEventId, RelationStateKey
from .relation_filter import RelationDecision, RelationEstimate
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
    formal_link_confirmation_pending: bool = False

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
        if self.formal_link_confirmation_pending and (
            self.role != FrameRole.DEFER
            or self.execution_weight != 0.0
            or self.blocks_advance
            or self.recovery_intent is not None
        ):
            raise ValueError("正式 LINK 确认窗口必须是零权重、非阻断 DEFER")


@dataclass(frozen=True)
class FrameRoleSnapshot:
    state_id: StateId
    decisions: dict[str, FrameRoleDecision]
    recovery_intents: tuple[RelationRecoveryIntent, ...] = ()
    verification_requests: tuple[RelationVerificationRequest, ...] = ()
    confirmed_link_events: tuple[RelationEventId, ...] = ()
    rejected_link_events: tuple[RelationEventId, ...] = ()
    pending_link_confirmation_events: tuple[RelationEventId, ...] = ()

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
        # A demonstrated LINK is an expected event, not proof that this
        # episode has established the physical relation.  Remember which
        # occurrence has actually been confirmed by the online filter so a
        # later drop can never reuse the short post-LINK confirmation window.
        self._confirmed_link_events: set[RelationEventId] = set()
        # Once informative relative motion contradicts a formal LINK
        # occurrence, later low-excitation cycles must not repeatedly restore
        # that occurrence's confirmation grace.
        self._rejected_link_events: set[RelationEventId] = set()
        # A formal LINK confirmation window is only activated by causally
        # traversing its immediate predecessor.  Merely resetting or reentering
        # at a state whose offline prior is linked must not grant fresh grace.
        self._pending_link_confirmation_events: set[RelationEventId] = set()
        self._last_committed_state: StateId | None = None

    def reset(self) -> None:
        self._last_trusted_weights.clear()
        self._confirmed_link_events.clear()
        self._rejected_link_events.clear()
        self._pending_link_confirmation_events.clear()
        self._last_committed_state = None

    def commit(
        self,
        snapshot: FrameRoleSnapshot,
        belief: ClosedLoopBelief | None = None,
    ) -> None:
        """Commit only stable role weights after the controller validates a cycle."""

        self._confirmed_link_events.update(snapshot.confirmed_link_events)
        self._rejected_link_events.update(snapshot.rejected_link_events)
        self._pending_link_confirmation_events = set(
            snapshot.pending_link_confirmation_events
        )
        self._last_committed_state = snapshot.state_id

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
        # A soft progress-weighted demonstration distribution can be
        # temporarily ambiguous around a normal relation transition.  Keep
        # that uncertainty explicit: it is neither an actual-q Unknown nor a
        # reliable expected/actual mismatch.
        if float(np.max(distribution)) < self.config.expected_relation_probability:
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

    def _link_origins_for_state(
        self,
        frame: str,
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[RelationEventId, ...]:
        node = self.task_model.state(state_id)
        mode = self._mode_for_state(state_id, mode_by_skill)
        modes = range(len(node.mode_priors)) if mode is None else (mode,)
        return tuple(
            sorted(
                {
                    event_id
                    for candidate_mode in modes
                    for event_id in (
                        self.task_model.link_origins.get(
                            RelationStateKey(
                                self.task_model.arm_id,
                                frame,
                                state_id,
                                candidate_mode,
                            )
                        ),
                    )
                    if event_id is not None
                }
            )
        )

    def _formal_link_confirmation_events(
        self,
        frame: str,
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[RelationEventId, ...]:
        """Return unresolved formal LINKs inside their learned linked interval."""

        events = []
        for event_id in self._link_origins_for_state(
            frame,
            state_id,
            mode_by_skill,
        ):
            if event_id in self._confirmed_link_events:
                continue
            if event_id in self._rejected_link_events:
                continue
            anchor = self.task_model.link_anchors[event_id]
            if state_id in anchor.linked_entry_states:
                events.append(event_id)
        return tuple(events)

    def _formal_link_confirmation_pending(
        self,
        frame: str,
        state_id: StateId,
        estimate: RelationEstimate | None,
        belief: ClosedLoopBelief,
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[RelationEventId, ...]:
        """Admit learned linked-interval motion while a formal LINK catches up.

        The binary posterior carries legitimate pre-event persistence, and the
        first samples in the learned LINK-entry interval include the causal
        grasp/close command itself.  They therefore cannot be used to reject
        the relation whose establishment they are still executing.  The
        bounded offline entry interval supplies natural carrying motion; after
        it ends, ordinary informative external evidence rejects the event.
        Tracking must remain available and reliable throughout this window.
        """

        if estimate is None:
            return ()
        events = self._formal_link_confirmation_events(
            frame,
            state_id,
            mode_by_skill,
        )
        if not events:
            return ()
        causally_entered = set(self._pending_link_confirmation_events)
        if self._last_committed_state is not None:
            previous = self.task_model.state(self._last_committed_state)
            if state_id in previous.topology.successors:
                causally_entered.update(
                    event_id
                    for event_id in events
                    if self.task_model.link_anchors[event_id].linked_entry_states[0]
                    == state_id
                )
        events = tuple(event for event in events if event in causally_entered)
        # The final learned entry state remains available while beta still
        # reports physical lag.  Once beta has reached it, re-evaluating that
        # same state means the complete offline interval has been consumed;
        # keeping the event pending would turn a bounded confirmation interval
        # into an unbounded terminal-state grace period.
        events = tuple(
            event
            for event in events
            if not (
                self._last_committed_state == state_id
                and belief.progress.estimated_state == state_id
                and self.task_model.link_anchors[event].linked_entry_states[-1]
                == state_id
            )
        )
        if not events:
            return ()
        features = belief.runtime_features
        if (
            not features.frame_pair_available.get(frame, False)
            or features.paired_tracking_reliability.get(frame, 0.0)
            < self.config.minimum_tracking_reliability
        ):
            return ()
        if estimate.decision_state == RelationDecision.LINKED:
            return ()
        return events

    def _planned_unlink_is_direct_successor(
        self,
        frame: str,
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> bool:
        """Return whether the next normal command causes a learned UNLINK.

        Relation uncertainty must not block the gripper-opening command that
        is itself required to make a demonstrated detachment observable.  The
        exception is deliberately narrow: a cross-demonstration confirmed
        UNLINK must be the direct successor in every active trajectory mode.
        It cannot skip progress or admit an arbitrary future release.
        """

        successors = set(self.task_model.state(state_id).topology.successors)
        mode = self._mode_for_state(state_id, mode_by_skill)
        modes = (
            range(len(self.task_model.state(state_id).mode_priors))
            if mode is None
            else (mode,)
        )
        for candidate_mode in modes:
            if not any(
                event_id.arm_id == self.task_model.arm_id
                and event_id.frame_id == frame
                and event_id.mode == candidate_mode
                and metadata.release_state in successors
                for event_id, metadata in self.task_model.unlink_events.items()
            ):
                return False
        return True

    def _planned_link_is_direct_successor(
        self,
        frame: str,
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None,
    ) -> bool:
        """Return whether the next normal command enters a learned LINK.

        A relation-changing command cannot be gated by the relation that the
        command itself is expected to establish.  In particular, the online
        filter can still carry a valid pre-grasp ``external`` decision at the
        predecessor of a demonstrated LINK.  Blocking that predecessor would
        prevent the controller from ever entering the learned linked interval
        whose natural motion supplies the confirming evidence.

        The allowance is event-local and mode-specific: every active mode must
        have a cross-demonstration LINK whose first legal linked state is an
        immediate topology successor.  Earlier mismatches and post-event drops
        therefore keep the ordinary RECOVER semantics.
        """

        successors = set(self.task_model.state(state_id).topology.successors)
        mode = self._mode_for_state(state_id, mode_by_skill)
        modes = (
            range(len(self.task_model.state(state_id).mode_priors))
            if mode is None
            else (mode,)
        )
        for candidate_mode in modes:
            if not any(
                event_id.arm_id == self.task_model.arm_id
                and event_id.frame_id == frame
                and event_id.mode == candidate_mode
                and bool(anchor.linked_entry_states)
                and anchor.linked_entry_states[0] in successors
                for event_id, anchor in self.task_model.link_anchors.items()
            ):
                return False
        return True

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
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[RelationVerificationRequest, ...]:
        requests = []
        context_state = max(
            (state_id, estimated_state),
            key=self._global_index.__getitem__,
        )
        for event_id, candidate in sorted(self.task_model.link_pending_events.items()):
            active_candidate = self.task_model.active_link_pending_candidate(
                candidate.frame_id,
                context_state,
                mode_by_skill,
            )
            if active_candidate is None or active_candidate.event_id != event_id:
                continue
            mode = self._mode_for_state(candidate.candidate_state, mode_by_skill)
            if mode is not None and mode != event_id.mode:
                continue
            frame = candidate.frame_id
            estimate = belief.relation_estimates.get(frame)
            features = belief.runtime_features
            if (
                estimate is None
                or estimate.decision_state == RelationDecision.LINKED
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
        captured_virtual_frames: frozenset[str] = frozenset(),
        commit: bool = True,
    ) -> FrameRoleSnapshot:
        if state_id not in self.task_model.states:
            raise KeyError(f"流角色引用未知状态 {state_id}")
        selected_frames = self._selected_frames(state_id, mode_by_skill)
        selected = frozenset(selected_frames)
        monitor_frames = set(
            self._confirmed_monitor_frames(
                belief.progress.posterior,
                mode_by_skill,
            )
        )
        # A newly queried LINK-entry state can lead the posterior by one
        # control cycle.  Its physical frame must nevertheless be routed now,
        # even when it is not an Eq.6 action stream; otherwise the causal
        # confirmation window cannot activate until after the controller has
        # already advanced past its first state.
        monitor_frames.update(
            frame
            for frame in self.task_model.relation_frames
            if self._formal_link_confirmation_events(
                frame,
                state_id,
                mode_by_skill,
            )
        )
        confirmed_monitors = tuple(sorted(monitor_frames.difference(selected)))
        decisions: dict[str, FrameRoleDecision] = {}
        recovery_intents = []
        confirmed_link_events: set[RelationEventId] = set()
        rejected_link_events: set[RelationEventId] = set()
        pending_link_confirmation_events: set[RelationEventId] = set()
        features = belief.runtime_features

        for frame in (*selected_frames, *confirmed_monitors):
            selected_offline = frame in selected
            reliability = (
                features.tracking_reliability.get(frame, 0.0)
                if features.frame_visibility.get(frame, False)
                else 0.0
            )
            if frame.startswith("virtual_skill_"):
                # A boundary transaction may capture the next skill's fixed
                # virtual frame after this cycle's belief was already built.
                # It is an internal reference, not a delayed external tracker,
                # so the same-tick entry query may use the explicitly supplied
                # capture at full reliability without rerunning inference.
                weight = (
                    1.0
                    if frame in captured_virtual_frames
                    else (
                        reliability
                        if features.frame_visibility.get(frame, False)
                        else 0.0
                    )
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
            estimate = belief.relation_estimates.get(frame)
            actual = (
                RelationDecision.UNKNOWN
                if estimate is None
                else estimate.decision_state
            )
            expected = self._expected_decision(distribution)
            current_link_events = self._link_origins_for_state(
                frame,
                state_id,
                mode_by_skill,
            )
            if actual == RelationDecision.LINKED:
                confirmed_link_events.update(current_link_events)
            confirmation_events = self._formal_link_confirmation_pending(
                frame,
                state_id,
                estimate,
                belief,
                mode_by_skill,
            )
            pending_link_confirmation_events.update(confirmation_events)
            if (
                not confirmation_events
                and expected == RelationDecision.LINKED
                and actual == RelationDecision.EXTERNAL
                and estimate is not None
                and estimate.informative
                and estimate.observation_likelihood[0]
                >= estimate.observation_likelihood[1]
            ):
                rejected_link_events.update(current_link_events)
            planned_link = self._planned_link_is_direct_successor(
                frame,
                state_id,
                mode_by_skill,
            )
            planned_unlink = self._planned_unlink_is_direct_successor(
                frame,
                state_id,
                mode_by_skill,
            )
            if (
                not selected_offline
                and not confirmation_events
                and expected != RelationDecision.LINKED
                and not (
                    expected == RelationDecision.UNKNOWN
                    and actual == RelationDecision.LINKED
                )
            ):
                # Event-confirmed monitoring never re-enables an inactive
                # external expert and never broadens PoE participation.
                continue
            compatibility = (
                0.5
                if estimate is None
                else float(np.dot(estimate.posterior, distribution))
            )
            if confirmation_events:
                # The physical object stream must stay out of the normal PoE
                # while its new linked relation is unresolved.  Other selected
                # streams execute the demonstrated post-grasp motion, which is
                # itself the non-tactile relation probe.  This is distinct from
                # LINK_PENDING/VERIFY_LINK: the offline event is already a
                # formal LINK and no auxiliary execution mode is entered.
                role = FrameRole.DEFER
                weight = 0.0
                monitor = True
                blocks = False
                intent = None
            elif (
                planned_link
                and expected == RelationDecision.LINKED
                and actual in {RelationDecision.UNKNOWN, RelationDecision.EXTERNAL}
            ):
                # The immediate successor is the learned relation-changing
                # event.  Let that command run with the physical stream
                # deferred; the event's linked interval then owns natural
                # confirmation and any fresh external contradiction.
                role = FrameRole.DEFER
                weight = 0.0
                monitor = True
                blocks = False
                intent = None
            elif (
                planned_unlink
                and expected == RelationDecision.LINKED
                and actual in {RelationDecision.UNKNOWN, RelationDecision.EXTERNAL}
            ):
                # The current relation may already be ambiguous or detached
                # after a coordinated transfer.  Do not request a spurious
                # LINK repair and thereby suppress the learned opening action.
                # The direct successor remains the ordinary task state; this
                # branch only removes the causal deadlock at its predecessor.
                role = FrameRole.DEFER
                weight = 0.0
                monitor = False
                blocks = False
                intent = None
            elif actual == RelationDecision.UNKNOWN:
                role = FrameRole.DEFER
                safe_weight = (
                    0.0 if estimate is None else reliability * estimate.external
                )
                trusted = frame in self._last_trusted_weights
                weight = (
                    self._last_trusted_weights.get(frame, safe_weight)
                    if selected_offline
                    else 0.0
                )
                monitor = False
                # At startup there may be no previously confirmed relation.
                # If both the demonstrated expectation and the binary
                # posterior point in the external direction for a
                # visible/reliable relation, a conservatively weighted action
                # is needed to create the motion evidence that can resolve
                # Unknown.  Do not require the normal stable-decision
                # probability here: one persistence transition turns a soft
                # [0.7, 0.3] demonstration prior into [0.692, 0.308], making a
                # 0.7 bootstrap gate unreachable without the very motion it
                # blocks.  This remains an Unknown decision, not external.
                # Once a trusted relation exists, any later Unknown is
                # blocking because persistence already failed its continuity
                # checks.
                safe_external_bootstrap = bool(
                    selected_offline
                    and not trusted
                    and expected == RelationDecision.EXTERNAL
                    and estimate is not None
                    and estimate.external > estimate.linked
                    and reliability >= self.config.minimum_tracking_reliability
                    and weight > 0.0
                )
                blocks = not safe_external_bootstrap
                intent = None
            elif expected == RelationDecision.UNKNOWN:
                # The online relation is stable, but beta-weighted normal
                # expectations straddle a demonstrated relation transition.
                # Preserve a safe role without inventing a mismatch.  This
                # ambiguity alone must not deadlock the progress transition;
                # a genuine actual-q Unknown remains blocking above.
                role = FrameRole.DEFER
                if actual == RelationDecision.LINKED:
                    weight = 0.0
                    monitor = True
                else:
                    assert estimate is not None
                    safe_weight = reliability * estimate.external
                    weight = self._last_trusted_weights.get(frame, safe_weight)
                    monitor = False
                blocks = False
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
                formal_link_confirmation_pending=bool(confirmation_events),
            )

        requests = self._verification_requests(
            state_id,
            belief.progress.estimated_state,
            belief,
            mode_by_skill,
        )
        snapshot = FrameRoleSnapshot(
            state_id=state_id,
            decisions=decisions,
            recovery_intents=tuple(recovery_intents),
            verification_requests=requests,
            confirmed_link_events=tuple(sorted(confirmed_link_events)),
            rejected_link_events=tuple(sorted(rejected_link_events)),
            pending_link_confirmation_events=tuple(
                sorted(
                    pending_link_confirmation_events.difference(
                        confirmed_link_events, rejected_link_events
                    )
                )
            ),
        )
        if commit:
            self.commit(snapshot, belief)
        return snapshot

    def route_fixed(
        self,
        state_id: StateId,
        belief: ClosedLoopBelief,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
        captured_virtual_frames: frozenset[str] = frozenset(),
        commit: bool = True,
    ) -> FrameRoleSnapshot:
        """Use the frozen offline stream mask for the progress-only ablation.

        Relation posteriors remain available to the phase-two progress model
        and diagnostics, but they cannot change stream participation, block a
        cursor commit, or create verification/recovery intents here.
        """

        if state_id not in self.task_model.states:
            raise KeyError(f"流角色引用未知状态 {state_id}")
        selected_frames = self._selected_frames(state_id, mode_by_skill)
        decisions = {}
        for frame in selected_frames:
            virtual = frame.startswith("virtual_skill_")
            distribution = (
                None
                if virtual
                else self._expected_distribution(
                    frame, belief.progress.posterior, mode_by_skill
                )
            )
            estimate = belief.relation_estimates.get(frame)
            actual = None if estimate is None else estimate.decision_state
            expected = (
                None if distribution is None else self._expected_decision(distribution)
            )
            compatibility = (
                1.0
                if distribution is None
                else (
                    0.5
                    if estimate is None
                    else float(np.dot(estimate.posterior, distribution))
                )
            )
            # A virtual frame captured by a boundary transaction is always an
            # internally available execution reference.  Ordinary selected
            # frames preserve the original fixed DynaMAC participation mask.
            weight = 1.0
            decisions[frame] = FrameRoleDecision(
                frame_id=frame,
                role=FrameRole.EXECUTE,
                selected_offline=True,
                expected_distribution=distribution,
                expected_relation=expected,
                actual_relation=actual,
                relation_compatibility=compatibility,
                execution_weight=weight,
                monitor=False,
                blocks_advance=False,
            )
        snapshot = FrameRoleSnapshot(state_id=state_id, decisions=decisions)
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
