"""Bounded active verification for unresolved LINK_PENDING relations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Sequence

import numpy as np

from .frame_roles import RelationVerificationRequest
from .relation_events import LinkPendingCandidate, RelationEventId
from .relation_filter import RelationDecision, RelationEstimate
from .runtime_features import RuntimeFeatures
from .state_index import StateId

Array = np.ndarray


class VerificationPhase(str, Enum):
    IDLE = "idle"
    PROBE = "probe"
    RETURN = "return"
    COMPLETE = "complete"
    FAILED = "failed"


class ProbeExitReason(str, Enum):
    STABLE_RELATION = "stable_relation"
    TIMEOUT = "timeout"
    SAFETY = "safety"


@dataclass(frozen=True)
class AuxiliaryAction:
    pose: Array
    covariance: Array
    gripper_command: Array
    source: str

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        gripper = np.asarray(self.gripper_command, dtype=np.float64)
        if pose.shape != (7,) or covariance.shape != (6, 6):
            raise ValueError("辅助动作必须使用 [7] 位姿和 [6,6] 协方差")
        if gripper.ndim != 1 or not len(gripper):
            raise ValueError("辅助动作夹爪命令必须为非空一维数组")
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(covariance)):
            raise ValueError("辅助动作包含非有限值")
        if not self.source:
            raise ValueError("辅助动作必须标识来源")
        object.__setattr__(self, "pose", pose.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "gripper_command", gripper.copy())


@dataclass(frozen=True)
class SafetyConstraintStatus:
    probe_safe: bool = True
    return_safe: bool = True
    reason: str = ""


@dataclass(frozen=True)
class RelationVerificationConfig:
    control_period_seconds: float = 0.05
    probe_speed: float = 0.02
    maximum_probe_seconds: float = 1.0
    minimum_probe_motion: float = 0.002
    minimum_response_samples: int = 3
    maximum_response_residual_ratio: float = 0.85
    minimum_information_weight: float = 0.10
    minimum_tracking_reliability: float = 0.25
    minimum_approach_displacement: float = 1.0e-4
    task_history_length: int = 12
    return_position_tolerance: float = 0.002
    maximum_return_cycles: int = 80
    action_position_variance: float = 1.0e-5
    action_rotation_variance: float = 1.0e-4

    def __post_init__(self) -> None:
        positive = (
            self.control_period_seconds,
            self.probe_speed,
            self.maximum_probe_seconds,
            self.minimum_probe_motion,
            self.minimum_approach_displacement,
            self.return_position_tolerance,
            self.action_position_variance,
            self.action_rotation_variance,
        )
        if any(value <= 0.0 or not np.isfinite(value) for value in positive):
            raise ValueError("主动关系验证的运动和时间参数必须为有限正数")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                self.minimum_information_weight,
                self.minimum_tracking_reliability,
            )
        ):
            raise ValueError("主动关系验证信息与跟踪阈值必须位于 [0,1]")
        if self.task_history_length < 2 or self.maximum_return_cycles < 1:
            raise ValueError("主动关系验证轨迹历史和返回周期上限无效")
        if self.minimum_response_samples < 2:
            raise ValueError("主动关系验证至少需要两个动作响应样本")
        if not 0.0 < self.maximum_response_residual_ratio < 1.0:
            raise ValueError("主动关系验证共动残差比必须位于 (0,1)")

    @property
    def maximum_probe_cycles(self) -> int:
        return max(
            1, math.ceil(self.maximum_probe_seconds / self.control_period_seconds)
        )

    @property
    def action_covariance(self) -> Array:
        return np.diag(
            [
                *([self.action_position_variance] * 3),
                *([self.action_rotation_variance] * 3),
            ]
        )


@dataclass(frozen=True)
class VerificationAttemptSignature:
    relation_state: RelationDecision
    task_state: StateId
    grasp_event: Hashable


class VerificationAttemptRegistry:
    """Prevent immediate TASK/VERIFY_LINK loops for one Pending occurrence."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._attempted: dict[RelationEventId, VerificationAttemptSignature] = {}

    def can_attempt(
        self,
        event_id: RelationEventId,
        signature: VerificationAttemptSignature,
    ) -> bool:
        return self._attempted.get(event_id) != signature

    def record(
        self,
        event_id: RelationEventId,
        signature: VerificationAttemptSignature,
    ) -> None:
        self._attempted[event_id] = signature


@dataclass(frozen=True)
class RelationVerificationStep:
    phase: VerificationPhase
    action: AuxiliaryAction | None
    decision: RelationDecision
    probe_exit_reason: ProbeExitReason | None
    probe_cycles: int
    return_cycles: int
    accumulated_probe_motion: float
    response_samples: int
    response_residual_ratio: float | None
    response_decision: RelationDecision
    failure_reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.phase in {VerificationPhase.COMPLETE, VerificationPhase.FAILED}


class RelationVerificationController:
    """Probe opposite the actual approach, then safely replay the path backward."""

    def __init__(
        self,
        config: RelationVerificationConfig = RelationVerificationConfig(),
        attempts: VerificationAttemptRegistry | None = None,
    ) -> None:
        self.config = config
        self.attempts = attempts or VerificationAttemptRegistry()
        self.reset_runtime()

    def reset_runtime(self) -> None:
        self.phase = VerificationPhase.IDLE
        self.request: RelationVerificationRequest | None = None
        self.candidate: LinkPendingCandidate | None = None
        self._entry_pose: Array | None = None
        self._gripper_command: Array | None = None
        self._approach_direction: Array | None = None
        self._probe_path: list[Array] = []
        self._return_path: list[Array] = []
        self._probe_cycles = 0
        self._return_cycles = 0
        self._accumulated_probe_motion = 0.0
        self._response_pending = False
        self._response_samples = 0
        self._response_translation_motion = 0.0
        self._response_ee_squared = 0.0
        self._response_residual_squared = 0.0
        self._probe_exit_reason: ProbeExitReason | None = None
        self._verified_decision = RelationDecision.UNKNOWN
        self._failure_reason: str | None = None

    @staticmethod
    def _pose(value: Array, name: str) -> Array:
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{name} 必须为有限 [7] 位姿")
        return pose.copy()

    def _approach_from_history(self, poses: Sequence[Array]) -> Array:
        history = [self._pose(value, "TASK 末端历史") for value in poses]
        if len(history) < 2:
            raise ValueError("VERIFY_LINK 至少需要两个 TASK 实际末端历史点")
        history = history[-self.config.task_history_length :]
        displacement = history[-1][:3] - history[0][:3]
        norm = float(np.linalg.norm(displacement))
        if norm < self.config.minimum_approach_displacement:
            raise ValueError("最近 TASK 实际轨迹不足以确定抓取接近方向")
        return displacement / norm

    def start(
        self,
        request: RelationVerificationRequest,
        candidate: LinkPendingCandidate,
        *,
        task_state: StateId,
        relation_state: RelationDecision,
        grasp_event: Hashable,
        entry_pose: Array,
        gripper_command: Array,
        recent_task_poses: Sequence[Array],
    ) -> None:
        if self.phase not in {
            VerificationPhase.IDLE,
            VerificationPhase.COMPLETE,
            VerificationPhase.FAILED,
        }:
            raise RuntimeError("已有主动关系验证尚未结束")
        if request.pending_event_id != candidate.event_id:
            raise ValueError("主动验证请求与 Pending 候选不一致")
        if relation_state == RelationDecision.LINKED:
            raise ValueError("VERIFY_LINK 不能从已经确认的 linked 关系开始")
        signature = self.attempt_signature(
            task_state=task_state,
            relation_state=relation_state,
            grasp_event=grasp_event,
        )
        if not self.attempts.can_attempt(candidate.event_id, signature):
            raise RuntimeError("同一 Pending 关系事件在当前上下文中已经验证过")

        self.reset_runtime()
        self.request = request
        self.candidate = candidate
        self._entry_pose = self._pose(entry_pose, "VERIFY_LINK 入口位姿")
        gripper = np.asarray(gripper_command, dtype=np.float64)
        if gripper.ndim == 0:
            gripper = gripper.reshape(1)
        if gripper.ndim != 1 or not len(gripper) or not np.all(np.isfinite(gripper)):
            raise ValueError("VERIFY_LINK 夹爪命令必须为非空有限一维数组")
        self._gripper_command = gripper.copy()
        self._approach_direction = self._approach_from_history(recent_task_poses)
        self._probe_path = [self._entry_pose.copy()]
        self.phase = VerificationPhase.PROBE
        self.attempts.record(candidate.event_id, signature)

    @staticmethod
    def attempt_signature(
        *,
        task_state: StateId,
        relation_state: RelationDecision,
        grasp_event: Hashable,
    ) -> VerificationAttemptSignature:
        """Return the context whose change is required before retrying."""

        return VerificationAttemptSignature(
            relation_state=relation_state,
            task_state=task_state,
            grasp_event=grasp_event,
        )

    def can_attempt(
        self,
        candidate: LinkPendingCandidate,
        *,
        task_state: StateId,
        relation_state: RelationDecision,
        grasp_event: Hashable,
    ) -> bool:
        """Whether this Pending occurrence is re-armed in the current context."""

        signature = self.attempt_signature(
            task_state=task_state,
            relation_state=relation_state,
            grasp_event=grasp_event,
        )
        return self.attempts.can_attempt(candidate.event_id, signature)

    def _action(self, pose: Array, source: str) -> AuxiliaryAction:
        assert self._gripper_command is not None
        # The observation received on the *next* update is the response to
        # this auxiliary action.  The update that enters VERIFY_LINK still
        # contains the preceding TASK motion and must never be counted as
        # probe evidence.
        self._response_pending = True
        return AuxiliaryAction(
            pose=pose,
            covariance=self.config.action_covariance,
            gripper_command=self._gripper_command,
            source=source,
        )

    @property
    def _response_residual_ratio(self) -> float | None:
        if self._response_ee_squared <= np.finfo(np.float64).eps:
            return None
        return float(
            np.sqrt(self._response_residual_squared / self._response_ee_squared)
        )

    def _record_action_response(self, features: RuntimeFeatures) -> None:
        if not self._response_pending:
            return
        self._response_pending = False
        assert self.request is not None
        frame = self.request.frame_id
        if (
            not features.frame_pair_available.get(frame, False)
            or features.paired_tracking_reliability.get(frame, 0.0)
            < self.config.minimum_tracking_reliability
        ):
            return
        ee_translation = np.asarray(features.actual_ee_motion[:3], dtype=np.float64)
        frame_motion = features.frame_world_motion.get(frame)
        if frame_motion is None:
            return
        frame_translation = np.asarray(frame_motion[:3], dtype=np.float64)
        ee_norm = float(np.linalg.norm(ee_translation))
        if ee_norm <= np.finfo(np.float64).eps:
            return
        residual = frame_translation - ee_translation
        self._response_samples += 1
        self._response_translation_motion += ee_norm
        self._response_ee_squared += float(ee_translation @ ee_translation)
        self._response_residual_squared += float(residual @ residual)

    def _response_decision(self) -> RelationDecision:
        ratio = self._response_residual_ratio
        if (
            ratio is None
            or self._response_samples < self.config.minimum_response_samples
            or self._response_translation_motion < self.config.minimum_probe_motion
        ):
            return RelationDecision.UNKNOWN
        return (
            RelationDecision.LINKED
            if ratio <= self.config.maximum_response_residual_ratio
            else RelationDecision.EXTERNAL
        )

    def _stable_decision(
        self,
        estimate: RelationEstimate,
    ) -> RelationDecision:
        response = self._response_decision()
        if (
            estimate.decision_state != RelationDecision.UNKNOWN
            and estimate.informative
            and estimate.information_weight >= self.config.minimum_information_weight
            and self._accumulated_probe_motion >= self.config.minimum_probe_motion
            and estimate.decision_state == response
        ):
            return estimate.decision_state
        return RelationDecision.UNKNOWN

    def _begin_return(self, reason: ProbeExitReason) -> None:
        assert self._entry_pose is not None
        self._probe_exit_reason = reason
        reversed_path = [value.copy() for value in reversed(self._probe_path[:-1])]
        if not reversed_path or not np.allclose(reversed_path[-1], self._entry_pose):
            reversed_path.append(self._entry_pose.copy())
        self._return_path = reversed_path
        self.phase = VerificationPhase.RETURN

    def _snapshot(self, action: AuxiliaryAction | None) -> RelationVerificationStep:
        return RelationVerificationStep(
            phase=self.phase,
            action=action,
            decision=self._verified_decision,
            probe_exit_reason=self._probe_exit_reason,
            probe_cycles=self._probe_cycles,
            return_cycles=self._return_cycles,
            accumulated_probe_motion=self._accumulated_probe_motion,
            response_samples=self._response_samples,
            response_residual_ratio=self._response_residual_ratio,
            response_decision=self._response_decision(),
            failure_reason=self._failure_reason,
        )

    def update(
        self,
        *,
        current_pose: Array,
        features: RuntimeFeatures,
        estimate: RelationEstimate,
        safety: SafetyConstraintStatus = SafetyConstraintStatus(),
    ) -> RelationVerificationStep:
        if self.phase == VerificationPhase.IDLE:
            raise RuntimeError("VERIFY_LINK 尚未启动")
        if self.request is None or estimate.frame_id != self.request.frame_id:
            raise ValueError("VERIFY_LINK 关系估计与当前请求不一致")
        current = self._pose(current_pose, "VERIFY_LINK 当前位姿")
        self._record_action_response(features)

        if self.phase == VerificationPhase.PROBE:
            # Count motion only after a VERIFY_LINK command has actually been
            # issued.  Entry-cycle TASK motion is deliberately excluded by
            # ``_record_action_response`` and ``_response_pending``.
            if self._response_samples:
                self._accumulated_probe_motion = self._response_translation_motion
            if not np.allclose(current, self._probe_path[-1]):
                self._probe_path.append(current.copy())
            stable = self._stable_decision(estimate)
            if stable != RelationDecision.UNKNOWN:
                self._verified_decision = stable
                self._begin_return(ProbeExitReason.STABLE_RELATION)
            elif not safety.probe_safe:
                self._begin_return(ProbeExitReason.SAFETY)
            elif self._probe_cycles >= self.config.maximum_probe_cycles:
                self._begin_return(ProbeExitReason.TIMEOUT)
            else:
                assert self._entry_pose is not None
                assert self._approach_direction is not None
                self._probe_cycles += 1
                distance = (
                    self.config.probe_speed
                    * self.config.control_period_seconds
                    * self._probe_cycles
                )
                target = self._entry_pose.copy()
                target[:3] -= distance * self._approach_direction
                return self._snapshot(self._action(target, "verify_link_probe"))

        if self.phase == VerificationPhase.RETURN:
            stable = self._stable_decision(estimate)
            if stable != RelationDecision.UNKNOWN:
                self._verified_decision = stable
            if not safety.return_safe:
                self.phase = VerificationPhase.FAILED
                self._failure_reason = safety.reason or "unsafe_return"
                return self._snapshot(None)
            self._return_cycles += 1
            if self._return_cycles > self.config.maximum_return_cycles:
                self.phase = VerificationPhase.FAILED
                self._failure_reason = "return_timeout"
                return self._snapshot(None)
            while (
                self._return_path
                and np.linalg.norm(current[:3] - self._return_path[0][:3])
                <= self.config.return_position_tolerance
            ):
                self._return_path.pop(0)
            if not self._return_path:
                self.phase = VerificationPhase.COMPLETE
                return self._snapshot(None)
            return self._snapshot(
                self._action(self._return_path[0], "verify_link_return")
            )

        return self._snapshot(None)


__all__ = [
    "AuxiliaryAction",
    "ProbeExitReason",
    "RelationVerificationConfig",
    "RelationVerificationController",
    "RelationVerificationStep",
    "SafetyConstraintStatus",
    "VerificationAttemptRegistry",
    "VerificationAttemptSignature",
    "VerificationPhase",
]
