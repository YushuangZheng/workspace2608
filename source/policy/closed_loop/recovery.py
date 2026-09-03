"""Top-level execution modes and bounded LINK/UNLINK recovery control."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .belief_updater import BeliefUpdater, ClosedLoopBelief
from .boundary_model import BoundaryId
from .boundary_runtime import ConditionKind, TransitionRequest
from .execution_controller import ClosedLoopExecutionController
from .frame_roles import RelationRecoveryIntent, RelationVerificationRequest
from .link_anchors import EpisodeLinkAnchorRegistry
from .mismatch import MismatchKind, MismatchUpdate
from .reentry import ReentryConfig, ReentryDecision, ReentryEvaluation, ReentrySelector
from .relation_events import RelationEventId
from .relation_filter import RelationDecision, RelationEstimate
from .relation_goals import RelationGoal, RelationGoalKind, RelationGoalPlanner
from .relation_verification import (
    AuxiliaryAction,
    ProbeExitReason,
    RelationVerificationConfig,
    RelationVerificationController,
    RelationVerificationStep,
    SafetyConstraintStatus,
    VerificationPhase,
)
from .runtime_observation import RuntimeObservation
from .state_index import StateId
from .task_model import ClosedLoopTaskModel
from .unlink_metadata import UnlinkMetadataRepository

Array = np.ndarray


class ExecutionMode(str, Enum):
    TASK = "task"
    VERIFY_LINK = "verify_link"
    RECOVERY = "recovery"


class RecoveryPhase(str, Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    REENTRY = "reentry"
    COMPLETE = "complete"
    FAILED = "failed"


class RelationGoalPhase(str, Enum):
    OPEN = "open"
    TRACK = "track"
    VERIFY = "verify"


@dataclass(frozen=True)
class RecoveryConfig:
    covariance_inflation: float = 1.0e-4
    pose_position_tolerance: float = 0.004
    maximum_waypoint_cycles: int = 40
    maximum_relation_verify_cycles: int = 30
    maximum_attempts_per_goal: int = 2
    maximum_total_cycles: int = 400
    maximum_reentry_cycles: int = 40
    minimum_relation_confirmation_cycles: int = 2
    minimum_information_weight: float = 0.10
    unlink_open_cycles: int = 2
    open_gripper_command: float = 1.0
    unlink_fallback_distance: float = 0.05
    action_position_variance: float = 2.0e-4
    action_rotation_variance: float = 5.0e-4
    boundary_relation_mismatch_cycles: int = 3
    link_frame_stability_confirmation_cycles: int = 3
    link_frame_stability_translation_tolerance: float = 0.001
    link_frame_stability_rotation_tolerance: float = math.radians(1.0)
    link_frame_restart_translation: float = 0.01
    link_frame_restart_rotation: float = math.radians(10.0)

    def __post_init__(self) -> None:
        nonnegative = (self.covariance_inflation,)
        if any(value < 0.0 or not np.isfinite(value) for value in nonnegative):
            raise ValueError("恢复协方差放宽量必须为有限非负数")
        positive = (
            self.pose_position_tolerance,
            self.unlink_fallback_distance,
            self.action_position_variance,
            self.action_rotation_variance,
            self.link_frame_stability_translation_tolerance,
            self.link_frame_stability_rotation_tolerance,
            self.link_frame_restart_translation,
            self.link_frame_restart_rotation,
        )
        if any(value <= 0.0 or not np.isfinite(value) for value in positive):
            raise ValueError("恢复运动尺度参数必须为有限正数")
        counts = (
            self.maximum_waypoint_cycles,
            self.maximum_relation_verify_cycles,
            self.maximum_attempts_per_goal,
            self.maximum_total_cycles,
            self.maximum_reentry_cycles,
            self.minimum_relation_confirmation_cycles,
            self.unlink_open_cycles,
            self.boundary_relation_mismatch_cycles,
            self.link_frame_stability_confirmation_cycles,
        )
        if any(value < 1 for value in counts):
            raise ValueError("恢复周期与尝试上限必须为正整数")
        if not 0.0 <= self.minimum_information_weight <= 1.0:
            raise ValueError("恢复关系信息权重阈值必须位于 [0,1]")

    @property
    def action_covariance(self) -> Array:
        return np.diag(
            [
                *([self.action_position_variance] * 3),
                *([self.action_rotation_variance] * 3),
            ]
        )


@dataclass(frozen=True)
class ClosedLoopRecoveryConfig:
    verification: RelationVerificationConfig = field(
        default_factory=RelationVerificationConfig
    )
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    reentry: ReentryConfig = field(default_factory=ReentryConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClosedLoopRecoveryConfig:
        known = {"verification", "recovery", "reentry"}
        unknown = set(value).difference(known)
        if unknown:
            raise ValueError(f"阶段五配置包含未知分区：{sorted(unknown)}")
        sections = {}
        for name, section_type in (
            ("verification", RelationVerificationConfig),
            ("recovery", RecoveryConfig),
            ("reentry", ReentryConfig),
        ):
            raw = value.get(name, {})
            if not isinstance(raw, Mapping):
                raise TypeError(f"阶段五配置分区 {name} 必须为对象")
            sections[name] = section_type(**dict(raw))
        return cls(**sections)

    @classmethod
    def from_json(cls, path: str | Path) -> ClosedLoopRecoveryConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("阶段五配置文件根节点必须为对象")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class RecoverySafetyStatus:
    action_safe: bool = True
    reason: str = ""


@dataclass(frozen=True)
class RecoveryFailure:
    reason: str
    goal: RelationGoal | None
    attempts: int
    total_cycles: int


@dataclass(frozen=True)
class RecoveryCycleResult:
    phase: RecoveryPhase
    action: AuxiliaryAction | None
    active_goal: RelationGoal | None
    goal_phase: RelationGoalPhase | None
    completed_goals: tuple[RelationGoal, ...]
    legal_reentry_states: tuple[StateId, ...]
    failure: RecoveryFailure | None = None
    link_probe_phase: str | None = None
    link_probe_exit_reason: ProbeExitReason | None = None
    link_probe_cycles: int = 0
    link_return_cycles: int = 0
    link_probe_motion: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.phase in {RecoveryPhase.COMPLETE, RecoveryPhase.FAILED}


@dataclass(frozen=True)
class RecoveryTriggerDecision:
    triggered: bool
    reasons: tuple[str, ...]
    intents: tuple[RelationRecoveryIntent, ...]


class RecoveryTriggerTracker:
    """Combine persistent estimator events with persistent boundary mismatches."""

    def __init__(
        self,
        task_models: Mapping[str, ClosedLoopTaskModel],
        config: RecoveryConfig = RecoveryConfig(),
    ) -> None:
        self.task_models = dict(task_models)
        self.config = config
        self._boundary_counts: dict[tuple[str, str, str], int] = {}
        self._boundary_progress_reentry_attempted: set[tuple[str, str]] = set()

    @staticmethod
    def _required_decision(value: str) -> RelationDecision:
        return (
            RelationDecision.LINKED if value == "linked" else RelationDecision.EXTERNAL
        )

    def update(
        self,
        mismatch_updates: Mapping[str, MismatchUpdate],
        *,
        transition_requests: Mapping[str, TransitionRequest] | None = None,
        beliefs: Mapping[str, ClosedLoopBelief] | None = None,
        mode_by_arm_skill: Mapping[str, Mapping[int, int]] | None = None,
    ) -> RecoveryTriggerDecision:
        transition_requests = transition_requests or {}
        beliefs = beliefs or {}
        reasons = []
        intents: dict[tuple[str, str], RelationRecoveryIntent] = {}
        for arm, update in mismatch_updates.items():
            for event in update.events:
                if event.kind == MismatchKind.NO_PLAUSIBLE_STATE:
                    request = transition_requests.get(arm)
                    if request is not None:
                        # Permit one local state-realignment attempt for a
                        # genuinely displaced terminal pose, then preserve the
                        # source terminal target for the rest of this boundary
                        # context.  Repeated broad reentry can otherwise replay
                        # the manipulation backwards in a terminal->interior
                        # loop and disturb the peer condition being awaited.
                        # A new skill boundary (or episode reset) has a new
                        # context key and may make its own attempt.
                        key = (arm, request.boundary_id.token)
                        if key in self._boundary_progress_reentry_attempted:
                            continue
                        self._boundary_progress_reentry_attempted.add(key)
                    reasons.append(f"{arm}:no_plausible_state")
                elif event.kind == MismatchKind.RELATION_MISMATCH:
                    reasons.append(f"{arm}:relation_mismatch")
                    for intent in event.recovery_intents:
                        intents[(intent.arm_id, intent.frame_id)] = intent

        active_boundary_keys = set()
        boundary_link_events: dict[tuple[str, str], RelationEventId] = {}
        for prepared_request in transition_requests.values():
            preparation = prepared_request.preparation
            event_ids = [] if preparation is None else list(preparation.event_ids)
            # After TransitionPreparation has physically closed the gripper,
            # the preparation field is consumed but the boundary retains the
            # exact Pending occurrence in its verification request.  Preserve
            # that identity so a verified external result is routed to the
            # matching temporary recovery template instead of guessed from
            # the current StateId.
            event_ids.extend(
                request.event_id for request in prepared_request.verification_requests
            )
            for event_id in event_ids:
                if event_id.transition not in {"link", "link_pending"}:
                    continue
                key = (event_id.arm_id, event_id.frame_id)
                previous = boundary_link_events.get(key)
                if previous is not None and previous != event_id:
                    raise RuntimeError("同一边界周期的 arm-frame 存在多个关系建立事件")
                boundary_link_events[key] = event_id
        for arm, request in transition_requests.items():
            if arm not in self.task_models:
                raise KeyError(f"恢复触发器缺少机械臂模型 {arm}")
            model = self.task_models[arm]
            boundary = model.boundaries[request.boundary_id]
            for condition_id, result in request.condition_results.items():
                if condition_id.kind not in {
                    ConditionKind.OWN_RELATION,
                    ConditionKind.GUARD_RELATION,
                }:
                    continue
                if not result.observed or not result.stable or result.raw_satisfied:
                    continue
                if (
                    condition_id.kind == ConditionKind.GUARD_RELATION
                    and not request.local_done
                ):
                    continue
                relation_arm, frame = condition_id.token.split("/", 1)
                belief = beliefs.get(relation_arm)
                if belief is None:
                    continue
                estimate = belief.relation_estimates.get(frame)
                if (
                    estimate is None
                    or estimate.decision_state == RelationDecision.UNKNOWN
                ):
                    continue
                if condition_id.kind == ConditionKind.OWN_RELATION:
                    condition = boundary.local_completion_model.own_relation_conditions[
                        condition_id.token
                    ]
                else:
                    condition = boundary.relation_conditions[condition_id.token]
                expected = self._required_decision(condition.required_state)
                if expected == estimate.decision_state:
                    continue
                boundary_event = boundary_link_events.get((relation_arm, frame))
                if (
                    condition_id.kind == ConditionKind.GUARD_RELATION
                    and relation_arm != arm
                ):
                    # A peer guard is normally only a readiness wait: the other
                    # arm may not have reached the relation-changing event yet.
                    # It becomes a repair request only when that arm has already
                    # reached a matching LINK_PENDING occurrence and reliable
                    # online evidence resolves the required LINK as external.
                    # This routes the repair to the relation-owning arm without
                    # turning every directional dependency into recovery.
                    if (
                        expected != RelationDecision.LINKED
                        or estimate.decision_state != RelationDecision.EXTERNAL
                    ):
                        continue
                    relation_state = belief.progress.estimated_state
                    configured_mode = None
                    if mode_by_arm_skill is not None:
                        configured_mode = mode_by_arm_skill.get(relation_arm, {}).get(
                            relation_state.skill_index
                        )
                    candidate_modes = (
                        (configured_mode,)
                        if configured_mode is not None
                        else tuple(
                            sorted(
                                {
                                    event_id.mode
                                    for event_id in self.task_models[
                                        relation_arm
                                    ].link_pending_events
                                    if event_id.frame_id == frame
                                }
                            )
                        )
                    )
                    registry = EpisodeLinkAnchorRegistry(self.task_models[relation_arm])
                    if boundary_event is None and not any(
                        registry.has_pending_recovery_candidate(
                            frame,
                            relation_state,
                            candidate_mode,
                        )
                        for candidate_mode in candidate_modes
                    ):
                        continue
                key = (arm, request.boundary_id.token, condition_id.token)
                active_boundary_keys.add(key)
                count = self._boundary_counts.get(key, 0) + 1
                self._boundary_counts[key] = count
                if count >= self.config.boundary_relation_mismatch_cycles:
                    intent = RelationRecoveryIntent(
                        arm_id=relation_arm,
                        frame_id=frame,
                        expected_relation=expected,
                        actual_relation=estimate.decision_state,
                        origin_event_id=(
                            boundary_event
                            if expected == RelationDecision.LINKED
                            and estimate.decision_state == RelationDecision.EXTERNAL
                            else None
                        ),
                    )
                    intents[(relation_arm, frame)] = intent
                    reasons.append(f"{arm}:boundary_relation_mismatch")
        for key in tuple(self._boundary_counts):
            if key not in active_boundary_keys:
                self._boundary_counts.pop(key, None)
        return RecoveryTriggerDecision(
            triggered=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
            intents=tuple(intents[key] for key in sorted(intents)),
        )


class RelationRecoveryController:
    """Execute ordered relation goals without querying the normal task PoE."""

    def __init__(
        self,
        anchor_registry: EpisodeLinkAnchorRegistry,
        config: RecoveryConfig = RecoveryConfig(),
        verification_config: RelationVerificationConfig = RelationVerificationConfig(),
    ) -> None:
        self.anchor_registry = anchor_registry
        self.config = config
        self.verification_config = verification_config
        self.reset()

    def reset(self) -> None:
        self.phase = RecoveryPhase.IDLE
        self._goals: tuple[RelationGoal, ...] = ()
        self._goal_index = 0
        self._goal_phase: RelationGoalPhase | None = None
        self._waypoint_index = 0
        self._waypoint_cycles = 0
        self._verify_cycles = 0
        self._open_cycles = 0
        self._attempts = 0
        self._total_cycles = 0
        self._goal_had_expected_information = False
        self._goal_confirmation_cycles = 0
        self._link_probe_phase: str | None = None
        self._link_probe_entry_pose: Array | None = None
        self._link_probe_direction: Array | None = None
        self._link_probe_path: list[Array] = []
        self._link_return_path: list[Array] = []
        self._link_probe_cycles = 0
        self._link_return_cycles = 0
        self._link_probe_motion = 0.0
        self._link_probe_exit_reason: ProbeExitReason | None = None
        self._link_probe_decision = RelationDecision.UNKNOWN
        self._link_probe_candidate = RelationDecision.UNKNOWN
        self._link_probe_candidate_cycles = 0
        self._completed: list[RelationGoal] = []
        self._reentry_states: list[StateId] = []
        self._reentry_cycles = 0
        self._failure: RecoveryFailure | None = None
        self._unlink_target: Array | None = None
        self._link_frame_previous_pose: Array | None = None
        self._link_frame_stability_cycles = 0
        self._link_anchor_frame_pose: Array | None = None

    def start(
        self,
        goals: Sequence[RelationGoal],
        *,
        fallback_reentry_states: Sequence[StateId] = (),
    ) -> None:
        if self.phase not in {
            RecoveryPhase.IDLE,
            RecoveryPhase.COMPLETE,
            RecoveryPhase.FAILED,
        }:
            raise RuntimeError("已有关系恢复尚未结束")
        self.reset()
        self._goals = tuple(goals)
        self._reentry_states = list(dict.fromkeys(fallback_reentry_states))
        self.phase = RecoveryPhase.EXECUTING if self._goals else RecoveryPhase.REENTRY
        if self._goals:
            self._start_goal()

    @property
    def active_goal(self) -> RelationGoal | None:
        return (
            self._goals[self._goal_index]
            if self.phase == RecoveryPhase.EXECUTING
            and self._goal_index < len(self._goals)
            else None
        )

    @property
    def legal_reentry_states(self) -> tuple[StateId, ...]:
        return tuple(self._reentry_states)

    @property
    def completed_goals(self) -> tuple[RelationGoal, ...]:
        """Relation repairs that must remain true through task reentry."""

        return tuple(self._completed)

    @property
    def has_relation_goals(self) -> bool:
        """Whether this recovery attempt ever contained a relation repair."""

        return bool(self._goals)

    @staticmethod
    def _pose(value: Array, name: str) -> Array:
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{name} 必须为有限 [7] 位姿")
        return pose.copy()

    def _start_goal(self) -> None:
        goal = self.active_goal
        if goal is None:
            self.phase = RecoveryPhase.REENTRY
            return
        self._goal_phase = (
            RelationGoalPhase.TRACK
            if goal.kind == RelationGoalKind.LINK
            else RelationGoalPhase.OPEN
        )
        self._waypoint_index = 0
        self._waypoint_cycles = 0
        self._verify_cycles = 0
        self._open_cycles = 0
        self._attempts = max(1, self._attempts + 1)
        self._goal_had_expected_information = False
        self._goal_confirmation_cycles = 0
        self._link_probe_phase = None
        self._link_probe_entry_pose = None
        self._link_probe_direction = None
        self._link_probe_path = []
        self._link_return_path = []
        self._link_probe_cycles = 0
        self._link_return_cycles = 0
        self._link_probe_motion = 0.0
        self._link_probe_exit_reason = None
        self._link_probe_decision = RelationDecision.UNKNOWN
        self._link_probe_candidate = RelationDecision.UNKNOWN
        self._link_probe_candidate_cycles = 0
        self._unlink_target = None
        self._link_frame_previous_pose = None
        self._link_frame_stability_cycles = 0
        self._link_anchor_frame_pose = None

    def _action(
        self,
        pose: Array,
        gripper: Array,
        source: str,
        covariance: Array | None = None,
    ) -> AuxiliaryAction:
        return AuxiliaryAction(
            pose=pose,
            covariance=(
                self.config.action_covariance if covariance is None else covariance
            ),
            gripper_command=gripper,
            source=source,
        )

    @staticmethod
    def _frame_pose_distance(left: Array, right: Array) -> tuple[float, float]:
        translation = float(np.linalg.norm(left[:3] - right[:3]))
        left_quaternion = left[3:] / np.linalg.norm(left[3:])
        right_quaternion = right[3:] / np.linalg.norm(right[3:])
        dot = float(np.clip(abs(np.dot(left_quaternion, right_quaternion)), 0.0, 1.0))
        return translation, float(2.0 * math.acos(dot))

    def _wait_for_stable_link_frame(
        self,
        *,
        frame_pose: Array,
        current_pose: Array,
        current_gripper: Array,
    ) -> AuxiliaryAction | None:
        """Lock one stationary object pose before replaying a LINK anchor."""

        if self._link_frame_previous_pose is None:
            self._link_frame_stability_cycles = 1
        else:
            translation, rotation = self._frame_pose_distance(
                self._link_frame_previous_pose,
                frame_pose,
            )
            if (
                translation <= self.config.link_frame_stability_translation_tolerance
                and rotation <= self.config.link_frame_stability_rotation_tolerance
            ):
                self._link_frame_stability_cycles += 1
            else:
                self._link_frame_stability_cycles = 1
        self._link_frame_previous_pose = frame_pose.copy()
        if (
            self._link_frame_stability_cycles
            >= self.config.link_frame_stability_confirmation_cycles
        ):
            self._link_anchor_frame_pose = frame_pose.copy()
            return None
        return self._action(
            current_pose,
            np.full_like(current_gripper, self.config.open_gripper_command),
            "recovery_link_settle",
        )

    def _restart_link_frame_settling(self, frame_pose: Array) -> None:
        self._link_anchor_frame_pose = None
        self._link_frame_previous_pose = frame_pose.copy()
        self._link_frame_stability_cycles = 1
        self._waypoint_index = 0
        self._waypoint_cycles = 0

    def _fail(self, reason: str) -> None:
        self._failure = RecoveryFailure(
            reason=reason,
            goal=self.active_goal,
            attempts=self._attempts,
            total_cycles=self._total_cycles,
        )
        self.phase = RecoveryPhase.FAILED

    def _retry_or_fail(self, reason: str) -> None:
        if self._attempts < self.config.maximum_attempts_per_goal:
            self._start_goal()
        else:
            self._fail(reason)

    def _complete_goal(self) -> None:
        goal = self.active_goal
        assert goal is not None
        self._completed.append(goal)
        for state in goal.legal_reentry_states:
            if state not in self._reentry_states:
                self._reentry_states.append(state)
        if (
            goal.kind == RelationGoalKind.LINK
            and goal.link_anchor is not None
            and goal.link_anchor.source == "pending_recovery"
        ):
            # The Pending trajectory was only a repair template.  Promote it
            # to this episode's relation origin after recovery has obtained
            # informative linked evidence, never before.
            self.anchor_registry.activate_pending(goal.link_anchor.origin_event_id)
        elif goal.kind == RelationGoalKind.UNLINK:
            self.anchor_registry.release(goal.frame_id)
        self._goal_index += 1
        self._attempts = 0
        if self._goal_index >= len(self._goals):
            self.phase = RecoveryPhase.REENTRY
            self._goal_phase = None
        else:
            self._start_goal()

    def _snapshot(self, action: AuxiliaryAction | None) -> RecoveryCycleResult:
        return RecoveryCycleResult(
            phase=self.phase,
            action=action,
            active_goal=self.active_goal,
            goal_phase=self._goal_phase,
            completed_goals=tuple(self._completed),
            legal_reentry_states=tuple(self._reentry_states),
            failure=self._failure,
            link_probe_phase=self._link_probe_phase,
            link_probe_exit_reason=self._link_probe_exit_reason,
            link_probe_cycles=self._link_probe_cycles,
            link_return_cycles=self._link_return_cycles,
            link_probe_motion=self._link_probe_motion,
        )

    def _begin_link_probe(
        self,
        current: Array,
        waypoint_poses: Sequence[Array],
    ) -> None:
        history = [self._pose(value, "LINK 恢复锚点") for value in waypoint_poses]
        if len(history) < 2:
            self._fail("link_probe_direction_unavailable")
            return
        displacement = history[-1][:3] - history[0][:3]
        norm = float(np.linalg.norm(displacement))
        if norm < self.verification_config.minimum_approach_displacement:
            self._fail("link_probe_direction_unavailable")
            return
        self._link_probe_phase = "probe"
        self._link_probe_entry_pose = current.copy()
        self._link_probe_direction = displacement / norm
        self._link_probe_path = [current.copy()]

    def _link_probe_stable(self, estimate: RelationEstimate) -> RelationDecision:
        candidate = RelationDecision.UNKNOWN
        if (
            self._link_probe_motion >= self.verification_config.minimum_probe_motion
            and estimate.decision_state != RelationDecision.UNKNOWN
            and estimate.informative
            and estimate.information_weight
            >= self.verification_config.minimum_information_weight
        ):
            candidate = estimate.decision_state
        if candidate == RelationDecision.UNKNOWN:
            self._link_probe_candidate = RelationDecision.UNKNOWN
            self._link_probe_candidate_cycles = 0
            return RelationDecision.UNKNOWN
        if candidate == self._link_probe_candidate:
            self._link_probe_candidate_cycles += 1
        else:
            self._link_probe_candidate = candidate
            self._link_probe_candidate_cycles = 1
        return (
            candidate
            if self._link_probe_candidate_cycles
            >= self.config.minimum_relation_confirmation_cycles
            else RelationDecision.UNKNOWN
        )

    def _begin_link_probe_return(self, reason: ProbeExitReason) -> None:
        assert self._link_probe_entry_pose is not None
        self._link_probe_exit_reason = reason
        path = [value.copy() for value in reversed(self._link_probe_path[:-1])]
        if not path or not np.allclose(path[-1], self._link_probe_entry_pose):
            path.append(self._link_probe_entry_pose.copy())
        self._link_return_path = path
        self._link_probe_phase = "return"

    def _update_link_probe(
        self,
        *,
        current: Array,
        estimate: RelationEstimate,
        final_gripper: Array,
        final_covariance: Array,
    ) -> RecoveryCycleResult:
        if self._link_probe_phase == "probe":
            if not np.allclose(current, self._link_probe_path[-1]):
                self._link_probe_motion += float(
                    np.linalg.norm(current[:3] - self._link_probe_path[-1][:3])
                )
                self._link_probe_path.append(current.copy())
            stable = self._link_probe_stable(estimate)
            if stable != RelationDecision.UNKNOWN:
                self._link_probe_decision = stable
                self._begin_link_probe_return(ProbeExitReason.STABLE_RELATION)
            elif (
                self._link_probe_cycles >= self.verification_config.maximum_probe_cycles
            ):
                self._begin_link_probe_return(ProbeExitReason.TIMEOUT)
            else:
                assert self._link_probe_entry_pose is not None
                assert self._link_probe_direction is not None
                self._link_probe_cycles += 1
                distance = (
                    self.verification_config.probe_speed
                    * self.verification_config.control_period_seconds
                    * self._link_probe_cycles
                )
                target = self._link_probe_entry_pose.copy()
                target[:3] -= distance * self._link_probe_direction
                return self._snapshot(
                    self._action(
                        target,
                        final_gripper,
                        "recovery_link_probe",
                        final_covariance,
                    )
                )

        if self._link_probe_phase == "return":
            stable = self._link_probe_stable(estimate)
            if stable != RelationDecision.UNKNOWN:
                self._link_probe_decision = stable
            self._link_return_cycles += 1
            if (
                self._link_return_cycles
                > self.verification_config.maximum_return_cycles
            ):
                self._fail("link_probe_return_timeout")
                return self._snapshot(None)
            while (
                self._link_return_path
                and np.linalg.norm(current[:3] - self._link_return_path[0][:3])
                <= self.verification_config.return_position_tolerance
            ):
                self._link_return_path.pop(0)
            if self._link_return_path:
                return self._snapshot(
                    self._action(
                        self._link_return_path[0],
                        final_gripper,
                        "recovery_link_return",
                        final_covariance,
                    )
                )
            if (
                self._link_probe_decision == RelationDecision.LINKED
                and self._goal_had_expected_information
                and estimate.decision_state == RelationDecision.LINKED
            ):
                self._complete_goal()
            else:
                self._retry_or_fail("link_relation_not_recovered")
            return self._snapshot(None)

        raise RuntimeError("LINK 恢复主动验证阶段无效")

    def update(
        self,
        *,
        current_pose: Array,
        current_gripper: Array,
        frame_poses: Mapping[str, Array],
        relation_estimates: Mapping[str, RelationEstimate],
        safety: RecoverySafetyStatus = RecoverySafetyStatus(),
    ) -> RecoveryCycleResult:
        if self.phase == RecoveryPhase.IDLE:
            raise RuntimeError("RECOVERY 尚未启动")
        if self.phase != RecoveryPhase.EXECUTING:
            return self._snapshot(None)
        self._total_cycles += 1
        if self._total_cycles > self.config.maximum_total_cycles:
            self._fail("maximum_total_cycles")
            return self._snapshot(None)
        if not safety.action_safe:
            self._fail(safety.reason or "safety_constraint")
            return self._snapshot(None)

        goal = self.active_goal
        assert goal is not None
        if goal.frame_id not in frame_poses or goal.frame_id not in relation_estimates:
            self._fail("relation_observation_unavailable")
            return self._snapshot(None)
        current = self._pose(current_pose, "RECOVERY 当前末端位姿")
        gripper = np.asarray(current_gripper, dtype=np.float64)
        if gripper.ndim == 0:
            gripper = gripper.reshape(1)
        estimate = relation_estimates[goal.frame_id]
        goal_relation_satisfied = bool(
            estimate.decision_state == goal.expected_relation
            and estimate.informative
            and estimate.information_weight >= self.config.minimum_information_weight
        )
        if goal_relation_satisfied:
            self._goal_had_expected_information = True
            self._goal_confirmation_cycles += 1
        else:
            self._goal_confirmation_cycles = 0

        # A relation goal is complete when the requested physical relation is
        # already stably observed.  LINK anchors and UNLINK detachment paths
        # are means of restoring that relation, not additional task goals that
        # must be followed after recovery has succeeded.  The sole exception
        # is an active LINK probe: it deliberately displaced the arm and must
        # still execute its recorded return path before leaving recovery.
        if (
            self._goal_confirmation_cycles
            >= self.config.minimum_relation_confirmation_cycles
            and not (
                goal.kind == RelationGoalKind.LINK
                and self._link_probe_phase is not None
            )
        ):
            self._complete_goal()
            return self._snapshot(None)

        if goal.kind == RelationGoalKind.LINK:
            assert goal.link_anchor is not None
            frame_pose = self._pose(
                frame_poses[goal.frame_id],
                "LINK 恢复参考实体位姿",
            )
            if self._link_anchor_frame_pose is None:
                settling_action = self._wait_for_stable_link_frame(
                    frame_pose=frame_pose,
                    current_pose=current,
                    current_gripper=gripper,
                )
                if settling_action is not None:
                    return self._snapshot(settling_action)
            assert self._link_anchor_frame_pose is not None
            waypoints = self.anchor_registry.instantiate(
                goal.link_anchor,
                self._link_anchor_frame_pose,
                self.config.covariance_inflation,
            )
            first_closing_waypoint = next(
                (
                    index
                    for index, waypoint in enumerate(waypoints)
                    if np.any(waypoint.gripper_command <= 0.5)
                ),
                len(waypoints),
            )
            if self._goal_phase == RelationGoalPhase.TRACK:
                if self._waypoint_index < first_closing_waypoint:
                    translation, rotation = self._frame_pose_distance(
                        self._link_anchor_frame_pose,
                        frame_pose,
                    )
                    if (
                        translation > self.config.link_frame_restart_translation
                        or rotation > self.config.link_frame_restart_rotation
                    ):
                        self._restart_link_frame_settling(frame_pose)
                        return self._snapshot(
                            self._action(
                                current,
                                np.full_like(
                                    gripper,
                                    self.config.open_gripper_command,
                                ),
                                "recovery_link_settle",
                            )
                        )
                waypoint = waypoints[self._waypoint_index]
                if (
                    np.linalg.norm(current[:3] - waypoint.pose[:3])
                    <= self.config.pose_position_tolerance
                ):
                    self._waypoint_index += 1
                    self._waypoint_cycles = 0
                    if self._waypoint_index >= len(waypoints):
                        self._goal_phase = RelationGoalPhase.VERIFY
                    else:
                        waypoint = waypoints[self._waypoint_index]
                else:
                    self._waypoint_cycles += 1
                    if self._waypoint_cycles > self.config.maximum_waypoint_cycles:
                        self._retry_or_fail("link_waypoint_timeout")
                        return self._snapshot(None)
                if self._goal_phase == RelationGoalPhase.TRACK:
                    return self._snapshot(
                        self._action(
                            waypoint.pose,
                            waypoint.gripper_command,
                            "recovery_link_anchor",
                            waypoint.covariance,
                        )
                    )

            if self._goal_phase == RelationGoalPhase.VERIFY:
                if (
                    self._link_probe_phase is None
                    and estimate.decision_state == RelationDecision.LINKED
                    and self._goal_confirmation_cycles
                    >= self.config.minimum_relation_confirmation_cycles
                ):
                    self._complete_goal()
                    return self._snapshot(None)
                final = waypoints[-1]
                if self._link_probe_phase is None:
                    # The anchor deliberately contains post-close holding
                    # states so recovery can reproduce the demonstrated
                    # interaction.  Those repeated terminal poses must not
                    # replace the actual pre-close approach when deriving the
                    # reverse probe direction.  Use the learned anchor only
                    # through its first close command.
                    approach_end = min(len(waypoints), first_closing_waypoint + 1)
                    self._begin_link_probe(
                        current,
                        tuple(waypoint.pose for waypoint in waypoints[:approach_end]),
                    )
                    if self.phase == RecoveryPhase.FAILED:
                        return self._snapshot(None)
                return self._update_link_probe(
                    current=current,
                    estimate=estimate,
                    final_gripper=final.gripper_command,
                    final_covariance=final.covariance,
                )

        else:
            assert goal.unlink_metadata is not None
            open_command = np.full_like(gripper, self.config.open_gripper_command)
            if self._goal_phase == RelationGoalPhase.OPEN:
                self._open_cycles += 1
                if self._open_cycles >= self.config.unlink_open_cycles:
                    self._goal_phase = RelationGoalPhase.TRACK
                return self._snapshot(
                    self._action(current, open_command, "recovery_unlink_open")
                )
            if self._unlink_target is None:
                self._unlink_target = UnlinkMetadataRepository.instantiate(
                    goal.unlink_metadata,
                    frame_poses[goal.frame_id],
                    current,
                    fallback_distance=self.config.unlink_fallback_distance,
                ).pose
            if self._goal_phase == RelationGoalPhase.TRACK:
                if (
                    np.linalg.norm(current[:3] - self._unlink_target[:3])
                    <= self.config.pose_position_tolerance
                ):
                    self._goal_phase = RelationGoalPhase.VERIFY
                    self._waypoint_cycles = 0
                else:
                    self._waypoint_cycles += 1
                    if self._waypoint_cycles > self.config.maximum_waypoint_cycles:
                        self._retry_or_fail("unlink_detachment_timeout")
                        return self._snapshot(None)
                if self._goal_phase == RelationGoalPhase.TRACK:
                    return self._snapshot(
                        self._action(
                            self._unlink_target,
                            open_command,
                            "recovery_unlink_detach",
                        )
                    )
            if self._goal_phase == RelationGoalPhase.VERIFY:
                if (
                    estimate.decision_state == RelationDecision.EXTERNAL
                    and self._goal_confirmation_cycles
                    >= self.config.minimum_relation_confirmation_cycles
                ):
                    self._complete_goal()
                    return self._snapshot(None)
                self._verify_cycles += 1
                if self._verify_cycles > self.config.maximum_relation_verify_cycles:
                    self._retry_or_fail("unlink_relation_not_recovered")
                    return self._snapshot(None)
                return self._snapshot(
                    self._action(
                        self._unlink_target,
                        open_command,
                        "recovery_unlink_verify",
                    )
                )
        return self._snapshot(None)

    def update_reentry(self, evaluation: ReentryEvaluation) -> RecoveryCycleResult:
        if self.phase != RecoveryPhase.REENTRY:
            raise RuntimeError("当前恢复流程尚未进入任务重入阶段")
        self._reentry_cycles += 1
        if evaluation.decision is not None:
            self.phase = RecoveryPhase.COMPLETE
            return self._snapshot(None)
        if self._reentry_cycles >= self.config.maximum_reentry_cycles:
            self._fail("no_legal_reentry_state")
        return self._snapshot(None)


@dataclass(frozen=True)
class RecoveryManagerResult:
    mode: ExecutionMode
    verification: RelationVerificationStep | None = None
    recovery: RecoveryCycleResult | None = None
    reentry: ReentryDecision | None = None
    reentry_evaluation: ReentryEvaluation | None = None


class ClosedLoopRecoveryManager:
    """Own TASK/VERIFY_LINK/RECOVERY mode transitions without a task clock."""

    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: ClosedLoopRecoveryConfig = ClosedLoopRecoveryConfig(),
    ) -> None:
        self.task_model = task_model
        self.config = config
        self.anchor_registry = EpisodeLinkAnchorRegistry(task_model)
        self.unlink_repository = UnlinkMetadataRepository(task_model)
        self.goal_planner = RelationGoalPlanner(
            self.anchor_registry, self.unlink_repository
        )
        self._unavailable_recovery_intents: tuple[RelationRecoveryIntent, ...] = ()
        self.verification = RelationVerificationController(config.verification)
        self.recovery = RelationRecoveryController(
            self.anchor_registry,
            config.recovery,
            config.verification,
        )
        self.reentry = ReentrySelector(
            task_model,
            config.reentry,
            robot_covariance_inflation=config.recovery.covariance_inflation,
        )
        self.reset()

    def reset(self) -> None:
        self.mode = ExecutionMode.TASK
        self._unavailable_recovery_intents = ()
        self.anchor_registry.reset()
        self.verification.reset_runtime()
        self.verification.attempts.reset()
        self.recovery.reset()
        self._task_pose_history: list[Array] = []
        self._frozen_reference: StateId | None = None

    @property
    def frozen_reference(self) -> StateId | None:
        return self._frozen_reference

    def record_task_pose(self, pose: Array) -> None:
        if self.mode != ExecutionMode.TASK:
            return
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("TASK 实际末端历史必须为有限 [7] 位姿")
        self._task_pose_history.append(value.copy())
        maximum = self.verification.config.task_history_length
        self._task_pose_history = self._task_pose_history[-maximum:]

    def can_begin_verification(
        self,
        request: RelationVerificationRequest,
        belief: ClosedLoopBelief,
        *,
        task_state: StateId,
        grasp_event: Hashable,
    ) -> bool:
        """Return whether an unresolved LINK request is eligible and re-armed now.

        Boundary and role inference may keep emitting the same request while
        its guard remains unsatisfied.  A completed attempt is therefore a
        normal no-op until the relation decision, task state, or grasp-event
        identity changes; it is not an exceptional policy failure.
        """

        event_exists = bool(
            request.event_id in self.task_model.link_pending_events
            or request.event_id in self.task_model.link_anchors
        )
        estimate = belief.relation_estimates.get(request.frame_id)
        features = belief.runtime_features
        return bool(
            self.mode == ExecutionMode.TASK
            and event_exists
            and estimate is not None
            and estimate.decision_state != RelationDecision.LINKED
            and features.frame_pair_available.get(request.frame_id, False)
            and features.paired_tracking_reliability.get(request.frame_id, 0.0)
            >= self.verification.config.minimum_tracking_reliability
            and estimate.information_weight
            < self.verification.config.minimum_information_weight
            and self.verification.approach_direction_available(self._task_pose_history)
            and self.verification.can_attempt(
                request.event_id,
                task_state=task_state,
                relation_state=estimate.decision_state,
                grasp_event=grasp_event,
            )
        )

    def begin_verification(
        self,
        request: RelationVerificationRequest,
        belief: ClosedLoopBelief,
        *,
        task_state: StateId,
        grasp_event: Hashable,
        current_pose: Array,
        current_gripper: Array,
    ) -> None:
        if self.mode != ExecutionMode.TASK:
            raise RuntimeError("只有 TASK 模式可以进入 VERIFY_LINK")
        if (
            request.event_id not in self.task_model.link_pending_events
            and request.event_id not in self.task_model.link_anchors
        ):
            raise KeyError("主动验证请求没有对应 LINK 事件")
        estimate = belief.relation_estimates.get(request.frame_id)
        features = belief.runtime_features
        if (
            estimate is None
            or estimate.decision_state == RelationDecision.LINKED
            or not features.frame_pair_available.get(request.frame_id, False)
            or features.paired_tracking_reliability.get(request.frame_id, 0.0)
            < self.verification.config.minimum_tracking_reliability
            or estimate.information_weight
            >= self.verification.config.minimum_information_weight
        ):
            raise ValueError("主动验证请求不满足未决、可见可靠且动作激励不足条件")
        self.verification.start(
            request,
            task_state=task_state,
            relation_state=estimate.decision_state,
            grasp_event=grasp_event,
            entry_pose=current_pose,
            gripper_command=current_gripper,
            recent_task_poses=self._task_pose_history,
        )
        self._frozen_reference = task_state
        self.mode = ExecutionMode.VERIFY_LINK

    def update_verification(
        self,
        belief: ClosedLoopBelief,
        *,
        current_pose: Array,
        safety: SafetyConstraintStatus = SafetyConstraintStatus(),
    ) -> RecoveryManagerResult:
        if self.mode != ExecutionMode.VERIFY_LINK:
            raise RuntimeError("当前不在 VERIFY_LINK 模式")
        assert self.verification.request is not None
        estimate = belief.relation_estimates[self.verification.request.frame_id]
        step = self.verification.update(
            current_pose=current_pose,
            features=belief.runtime_features,
            estimate=estimate,
            safety=safety,
        )
        if step.phase == VerificationPhase.COMPLETE:
            if (
                step.decision == RelationDecision.LINKED
                and self.verification.request.event_id.transition == "link_pending"
            ):
                self.anchor_registry.activate_pending(
                    self.verification.request.event_id
                )
            self.mode = ExecutionMode.TASK
            self._frozen_reference = None
        elif step.phase == VerificationPhase.FAILED:
            # The structured failure is returned to the caller.  Safety owns
            # the physical stop; no automatic verification loop is allowed.
            self.mode = ExecutionMode.TASK
            self._frozen_reference = None
        return RecoveryManagerResult(mode=self.mode, verification=step)

    def begin_recovery(
        self,
        trigger: RecoveryTriggerDecision,
        *,
        source_state: StateId,
        mode: int,
    ) -> bool:
        if self.mode != ExecutionMode.TASK:
            raise RuntimeError("只有 TASK 模式可以进入 RECOVERY")
        if not trigger.triggered:
            raise ValueError("没有恢复触发证据")
        if trigger.intents:
            goals, unavailable = self.goal_planner.plan_available(
                trigger.intents,
                source_state=source_state,
                mode=mode,
            )
        else:
            goals, unavailable = (), ()
        self._unavailable_recovery_intents = unavailable
        if trigger.intents and not goals:
            # The dynamic-role layer still blocks normal advancement while the
            # reliable relation mismatch is present.  Remaining in TASK keeps
            # beta/q updates alive so a progress-relation disagreement can
            # realign on the next observation.  Entering RECOVERY here would
            # either invent a relation-changing primitive absent from the
            # successful demonstrations or misroute to unconstrained reentry.
            return False
        self.recovery.start(
            goals,
            fallback_reentry_states=(
                () if goals else tuple(sorted(self.task_model.states))
            ),
        )
        self._frozen_reference = source_state
        self.mode = ExecutionMode.RECOVERY
        return True

    @property
    def unavailable_recovery_intents(self) -> tuple[RelationRecoveryIntent, ...]:
        """Latest reliable intents lacking a learned recovery primitive."""

        return self._unavailable_recovery_intents

    def update_recovery(
        self,
        *,
        current_pose: Array,
        current_gripper: Array,
        frame_poses: Mapping[str, Array],
        relation_estimates: Mapping[str, RelationEstimate],
        safety: RecoverySafetyStatus = RecoverySafetyStatus(),
    ) -> RecoveryManagerResult:
        if self.mode != ExecutionMode.RECOVERY:
            raise RuntimeError("当前不在 RECOVERY 模式")
        result = self.recovery.update(
            current_pose=current_pose,
            current_gripper=current_gripper,
            frame_poses=frame_poses,
            relation_estimates=relation_estimates,
            safety=safety,
        )
        return RecoveryManagerResult(mode=self.mode, recovery=result)

    def evaluate_reentry(
        self,
        belief: ClosedLoopBelief,
        *,
        observation: RuntimeObservation,
        belief_updater: BeliefUpdater,
        execution_controller: ClosedLoopExecutionController,
        permitted_boundaries: frozenset[BoundaryId] = frozenset(),
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> RecoveryManagerResult:
        evaluation = self.select_reentry(
            belief,
            permitted_boundaries=permitted_boundaries,
            mode_by_skill=mode_by_skill,
        )
        return self.commit_reentry(
            evaluation,
            belief=belief,
            observation=observation,
            belief_updater=belief_updater,
            execution_controller=execution_controller,
        )

    def select_reentry(
        self,
        belief: ClosedLoopBelief,
        *,
        permitted_boundaries: frozenset[BoundaryId] = frozenset(),
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> ReentryEvaluation:
        """Validate a reentry candidate without mutating recovery or cursors."""

        if self.mode != ExecutionMode.RECOVERY or self._frozen_reference is None:
            raise RuntimeError("当前没有可执行的恢复重入")
        if self.recovery.phase != RecoveryPhase.REENTRY:
            raise RuntimeError("关系恢复尚未进入重入阶段")
        return self.reentry.select(
            self.recovery.legal_reentry_states,
            belief,
            current_reference=self._frozen_reference,
            permitted_boundaries=permitted_boundaries,
            mode_by_skill=mode_by_skill,
            required_relations={
                goal.frame_id: (
                    RelationDecision.LINKED
                    if goal.kind == RelationGoalKind.LINK
                    else RelationDecision.EXTERNAL
                )
                for goal in self.recovery.completed_goals
            },
        )

    def commit_reentry(
        self,
        evaluation: ReentryEvaluation,
        *,
        belief: ClosedLoopBelief,
        observation: RuntimeObservation,
        belief_updater: BeliefUpdater,
        execution_controller: ClosedLoopExecutionController,
    ) -> RecoveryManagerResult:
        """Atomically apply one already-selected state and finish this attempt."""

        if self.mode != ExecutionMode.RECOVERY or self._frozen_reference is None:
            raise RuntimeError("当前没有可提交的恢复重入")
        if self.recovery.phase != RecoveryPhase.REENTRY:
            raise RuntimeError("关系恢复尚未进入可提交的重入阶段")
        decision = evaluation.decision
        if decision is not None:
            ReentrySelector.apply(
                decision,
                belief=belief,
                observation=observation,
                belief_updater=belief_updater,
                execution_controller=execution_controller,
            )
        result = self.recovery.update_reentry(evaluation)
        if result.phase == RecoveryPhase.COMPLETE:
            self.mode = ExecutionMode.TASK
            self._frozen_reference = None
        return RecoveryManagerResult(
            mode=self.mode,
            recovery=result,
            reentry=decision,
            reentry_evaluation=evaluation,
        )


__all__ = [
    "ClosedLoopRecoveryManager",
    "ClosedLoopRecoveryConfig",
    "ExecutionMode",
    "RecoveryConfig",
    "RecoveryCycleResult",
    "RecoveryFailure",
    "RecoveryManagerResult",
    "RecoveryPhase",
    "RecoverySafetyStatus",
    "RecoveryTriggerDecision",
    "RecoveryTriggerTracker",
    "RelationGoalPhase",
    "RelationRecoveryController",
]
