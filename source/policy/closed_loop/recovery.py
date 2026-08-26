"""Top-level execution modes and bounded LINK/UNLINK recovery control."""

from __future__ import annotations

import json
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
from .relation_filter import RelationDecision, RelationEstimate
from .relation_goals import RelationGoal, RelationGoalKind, RelationGoalPlanner
from .relation_verification import (
    AuxiliaryAction,
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
    minimum_information_weight: float = 0.10
    unlink_open_cycles: int = 2
    open_gripper_command: float = 1.0
    unlink_fallback_distance: float = 0.05
    action_position_variance: float = 2.0e-4
    action_rotation_variance: float = 5.0e-4
    boundary_relation_mismatch_cycles: int = 3

    def __post_init__(self) -> None:
        nonnegative = (self.covariance_inflation,)
        if any(value < 0.0 or not np.isfinite(value) for value in nonnegative):
            raise ValueError("恢复协方差放宽量必须为有限非负数")
        positive = (
            self.pose_position_tolerance,
            self.unlink_fallback_distance,
            self.action_position_variance,
            self.action_rotation_variance,
        )
        if any(value <= 0.0 or not np.isfinite(value) for value in positive):
            raise ValueError("恢复运动尺度参数必须为有限正数")
        counts = (
            self.maximum_waypoint_cycles,
            self.maximum_relation_verify_cycles,
            self.maximum_attempts_per_goal,
            self.maximum_total_cycles,
            self.maximum_reentry_cycles,
            self.unlink_open_cycles,
            self.boundary_relation_mismatch_cycles,
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
    ) -> RecoveryTriggerDecision:
        transition_requests = transition_requests or {}
        beliefs = beliefs or {}
        reasons = []
        intents: dict[tuple[str, str], RelationRecoveryIntent] = {}
        for arm, update in mismatch_updates.items():
            for event in update.events:
                if event.kind == MismatchKind.NO_PLAUSIBLE_STATE:
                    reasons.append(f"{arm}:no_plausible_state")
                elif event.kind == MismatchKind.RELATION_MISMATCH:
                    reasons.append(f"{arm}:relation_mismatch")
                    for intent in event.recovery_intents:
                        intents[(intent.arm_id, intent.frame_id)] = intent

        active_boundary_keys = set()
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
    ) -> None:
        self.anchor_registry = anchor_registry
        self.config = config
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
        self._completed: list[RelationGoal] = []
        self._reentry_states: list[StateId] = []
        self._reentry_cycles = 0
        self._failure: RecoveryFailure | None = None
        self._unlink_target: Array | None = None

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
        self._unlink_target = None

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
        if goal.kind == RelationGoalKind.UNLINK:
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
        )

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
        if (
            estimate.decision_state == goal.expected_relation
            and estimate.informative
            and estimate.information_weight >= self.config.minimum_information_weight
        ):
            self._goal_had_expected_information = True

        if goal.kind == RelationGoalKind.LINK:
            assert goal.link_anchor is not None
            waypoints = self.anchor_registry.instantiate(
                goal.link_anchor,
                frame_poses[goal.frame_id],
                self.config.covariance_inflation,
            )
            if self._goal_phase == RelationGoalPhase.TRACK:
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
                    estimate.decision_state == RelationDecision.LINKED
                    and self._goal_had_expected_information
                ):
                    self._complete_goal()
                    return self._snapshot(None)
                self._verify_cycles += 1
                if self._verify_cycles > self.config.maximum_relation_verify_cycles:
                    self._retry_or_fail("link_relation_not_recovered")
                    return self._snapshot(None)
                final = waypoints[-1]
                return self._snapshot(
                    self._action(
                        final.pose,
                        final.gripper_command,
                        "recovery_link_verify",
                        final.covariance,
                    )
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
                    and self._goal_had_expected_information
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
        self.verification = RelationVerificationController(config.verification)
        self.recovery = RelationRecoveryController(
            self.anchor_registry, config.recovery
        )
        self.reentry = ReentrySelector(task_model, config.reentry)
        self.reset()

    def reset(self) -> None:
        self.mode = ExecutionMode.TASK
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
        try:
            candidate = self.task_model.link_pending_events[request.pending_event_id]
        except KeyError as exc:
            raise KeyError("主动验证请求没有对应 Pending 候选") from exc
        estimate = belief.relation_estimates.get(request.frame_id)
        features = belief.runtime_features
        if (
            estimate is None
            or estimate.decision_state != RelationDecision.UNKNOWN
            or not features.frame_pair_available.get(request.frame_id, False)
            or features.paired_tracking_reliability.get(request.frame_id, 0.0)
            < self.verification.config.minimum_tracking_reliability
            or estimate.information_weight
            >= self.verification.config.minimum_information_weight
        ):
            raise ValueError("主动验证请求不满足可见可靠但动作激励不足条件")
        self.verification.start(
            request,
            candidate,
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
            if step.decision == RelationDecision.LINKED:
                self.anchor_registry.activate_pending(
                    self.verification.request.pending_event_id
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
    ) -> None:
        if self.mode != ExecutionMode.TASK:
            raise RuntimeError("只有 TASK 模式可以进入 RECOVERY")
        if not trigger.triggered:
            raise ValueError("没有恢复触发证据")
        goals = self.goal_planner.plan(
            trigger.intents,
            source_state=source_state,
            mode=mode,
        )
        self.recovery.start(
            goals,
            fallback_reentry_states=(
                () if goals else tuple(sorted(self.task_model.states))
            ),
        )
        self._frozen_reference = source_state
        self.mode = ExecutionMode.RECOVERY

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
        if self.mode != ExecutionMode.RECOVERY or self._frozen_reference is None:
            raise RuntimeError("当前没有可执行的恢复重入")
        evaluation = self.reentry.select(
            self.recovery.legal_reentry_states,
            belief,
            current_reference=self._frozen_reference,
            permitted_boundaries=permitted_boundaries,
            mode_by_skill=mode_by_skill,
        )
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
