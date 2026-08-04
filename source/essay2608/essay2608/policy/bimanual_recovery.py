"""Relation-gated receiver recovery for contact-rich physical handover."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from essay2608.data.handover_schema import HandoverState
from essay2608.data.transforms import rotate_vector
from essay2608.policy.relation import RelationState


class BimanualRecoveryState(str, Enum):
    """Independent supervisor lifecycle for the receiving arm."""

    NORMAL = "NORMAL"
    RECEIVER_MISS_DETECTED = "RECEIVER_MISS_DETECTED"
    RECEIVER_LOSS_DETECTED = "RECEIVER_LOSS_DETECTED"
    SAFE_RETREAT = "SAFE_RETREAT"
    REAPPROACH = "REAPPROACH"
    REGRASP = "REGRASP"
    VERIFY_BOTH = "VERIFY_BOTH"
    VERIFY_RECEIVER = "VERIFY_RECEIVER"
    RESUME_TASK = "RESUME_TASK"
    RECOVERY_FAILED = "RECOVERY_FAILED"


class BimanualRecoveryTrigger(str, Enum):
    """Causal relation event that started receiver recovery."""

    NONE = "NONE"
    RECEIVER_MISS = "RECEIVER_MISS"
    RECEIVER_LOSS = "RECEIVER_LOSS"


@dataclass(frozen=True)
class BimanualRecoveryConfig:
    """Bounded geometry and timing for receiver-side recovery."""

    enable_recovery: bool = True
    miss_verification_steps: int = 25
    loss_confirmation_steps: int = 15
    minimum_verified_both_steps: int = 5
    maximum_regrasp_attempts: int = 2
    maximum_recovery_steps: int = 600
    maximum_state_steps: int = 220
    right_grasp_offset_xyz_m: tuple[float, float, float] = (0.100, 0.0, -0.002)
    right_pregrasp_clearance_xyz_m: tuple[float, float, float] = (0.050, 0.0, 0.080)
    right_retreat_delta_xyz_m: tuple[float, float, float] = (0.050, 0.0, 0.080)
    position_tolerance_m: float = 0.022
    grasp_position_tolerance_m: float = 0.018
    open_gripper_threshold_m: float = 0.0738
    maximum_arm_target_step_m: float = 0.050

    def __post_init__(self) -> None:
        integer_fields = (
            self.miss_verification_steps,
            self.loss_confirmation_steps,
            self.minimum_verified_both_steps,
            self.maximum_regrasp_attempts,
            self.maximum_recovery_steps,
            self.maximum_state_steps,
        )
        if any(int(value) <= 0 for value in integer_fields):
            raise ValueError("双臂恢复的计数和时限必须为正数")
        for offset in (
            self.right_grasp_offset_xyz_m,
            self.right_pregrasp_clearance_xyz_m,
            self.right_retreat_delta_xyz_m,
        ):
            if len(offset) != 3 or not np.all(np.isfinite(offset)):
                raise ValueError("双臂恢复位移必须是有限三维向量")
        if (
            self.position_tolerance_m <= 0.0
            or self.grasp_position_tolerance_m <= 0.0
            or self.open_gripper_threshold_m <= 0.0
            or self.maximum_arm_target_step_m <= 0.0
        ):
            raise ValueError("双臂恢复的几何和夹爪阈值必须为正数")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BimanualRecoveryDecision:
    """One 16-D action override and task-clock decision."""

    action: np.ndarray
    state: BimanualRecoveryState
    trigger: BimanualRecoveryTrigger
    pause_task_clock: bool
    action_overridden: bool
    transfer_gate_active: bool
    regrasp_attempts: int
    state_steps: int
    total_recovery_steps: int
    verified_both_steps: int
    requires_giver_connection: bool
    transition: str | None = None


class BimanualRelationRecoveryController:
    """Prevent unsafe release and recover the receiver without contact truth."""

    def __init__(self, config: BimanualRecoveryConfig | None = None) -> None:
        self.config = config or BimanualRecoveryConfig()
        self.reset()

    def reset(self) -> None:
        self.state = BimanualRecoveryState.NORMAL
        self.trigger = BimanualRecoveryTrigger.NONE
        self.state_steps = 0
        self.total_recovery_steps = 0
        self.regrasp_attempts = 0
        self.miss_evidence_steps = 0
        self.verified_both_steps = 0
        self.had_receiver_connection = False
        self.requires_giver_connection = False
        self.regrasp_closing = False
        self.retreat_target: np.ndarray | None = None
        self.last_left_target: np.ndarray | None = None
        self.last_right_target: np.ndarray | None = None
        self.slew_limiter_active = False

    @property
    def active(self) -> bool:
        return self.state != BimanualRecoveryState.NORMAL

    @property
    def failed(self) -> bool:
        return self.state == BimanualRecoveryState.RECOVERY_FAILED

    def _transition(self, state: BimanualRecoveryState) -> str:
        before = self.state
        self.state = state
        self.state_steps = 0
        return f"{before.value}->{state.value}"

    @staticmethod
    def _at(right_pose: np.ndarray, target: np.ndarray, tolerance_m: float) -> bool:
        return bool(np.linalg.norm(right_pose[:3] - target) <= tolerance_m)

    @staticmethod
    def _connected(state: RelationState) -> bool:
        return state == RelationState.CONNECTED

    def _object_site(self, object_pose: np.ndarray, offset_xyz_m: tuple[float, float, float]) -> np.ndarray:
        offset = np.asarray(offset_xyz_m, dtype=np.float64)
        return object_pose[:3] + rotate_vector(object_pose[3:7], offset)

    def _recovery_action(
        self,
        normal_action: np.ndarray,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        right_target: np.ndarray,
        right_gripper: float,
    ) -> np.ndarray:
        action = np.asarray(normal_action, dtype=np.float64).copy()
        action[:7] = np.asarray(left_pose, dtype=np.float64)
        action[7] = -1.0
        action[8:15] = np.asarray(right_pose, dtype=np.float64)
        desired = np.asarray(right_target, dtype=np.float64)
        displacement = desired - right_pose[:3]
        distance = float(np.linalg.norm(displacement))
        if distance > self.config.maximum_arm_target_step_m:
            desired = right_pose[:3] + displacement * (
                self.config.maximum_arm_target_step_m / distance
            )
        action[8:11] = desired
        action[15] = float(right_gripper)
        return action

    def _hold_action(
        self,
        normal_action: np.ndarray,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        *,
        right_gripper: float,
    ) -> np.ndarray:
        return self._recovery_action(
            normal_action,
            left_pose,
            right_pose,
            right_pose[:3],
            right_gripper,
        )

    def _decision(
        self,
        action: np.ndarray,
        *,
        pause: bool,
        overridden: bool,
        gate: bool,
        transition: str | None = None,
    ) -> BimanualRecoveryDecision:
        action = np.asarray(action, dtype=np.float64).copy()
        if overridden:
            self.slew_limiter_active = True
        clamped = False
        if self.slew_limiter_active:
            for target_slice, previous in (
                (slice(0, 3), self.last_left_target),
                (slice(8, 11), self.last_right_target),
            ):
                if previous is None:
                    continue
                displacement = action[target_slice] - previous
                distance = float(np.linalg.norm(displacement))
                if distance > self.config.maximum_arm_target_step_m:
                    action[target_slice] = previous + displacement * (
                        self.config.maximum_arm_target_step_m / distance
                    )
                    clamped = True
        self.last_left_target = action[:3].copy()
        self.last_right_target = action[8:11].copy()
        if (
            self.state == BimanualRecoveryState.NORMAL
            and not overridden
            and not clamped
        ):
            self.slew_limiter_active = False
        return BimanualRecoveryDecision(
            action=action,
            state=self.state,
            trigger=self.trigger,
            pause_task_clock=pause,
            action_overridden=overridden or clamped,
            transfer_gate_active=gate,
            regrasp_attempts=self.regrasp_attempts,
            state_steps=self.state_steps,
            total_recovery_steps=self.total_recovery_steps,
            verified_both_steps=self.verified_both_steps,
            requires_giver_connection=self.requires_giver_connection,
            transition=transition,
        )

    def update(
        self,
        *,
        task_state: HandoverState,
        normal_action: np.ndarray,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        object_pose: np.ndarray,
        right_gripper_opening_m: float,
        left_relation_state: RelationState,
        right_relation_state: RelationState,
    ) -> BimanualRecoveryDecision:
        """Advance one causal step using current geometry and inferred edges only."""

        normal_action = np.asarray(normal_action, dtype=np.float64)
        left_pose = np.asarray(left_pose, dtype=np.float64)
        right_pose = np.asarray(right_pose, dtype=np.float64)
        object_pose = np.asarray(object_pose, dtype=np.float64)
        right_connected = self._connected(right_relation_state)
        left_connected = self._connected(left_relation_state)
        if right_connected:
            self.had_receiver_connection = True

        transition = None
        if self.state == BimanualRecoveryState.NORMAL:
            self.state_steps = 0
            self.total_recovery_steps = 0
            self.regrasp_attempts = 0
            self.trigger = BimanualRecoveryTrigger.NONE
            self.requires_giver_connection = False
            close_commanded = bool(normal_action[15] < 0.0)

            if task_state == HandoverState.RIGHT_GRASP and close_commanded:
                if right_connected:
                    self.miss_evidence_steps = 0
                    self.verified_both_steps += 1
                else:
                    self.miss_evidence_steps += 1
                    self.verified_both_steps = 0
                if self.miss_evidence_steps >= self.config.miss_verification_steps:
                    if not self.config.enable_recovery:
                        return self._decision(
                            normal_action,
                            pause=True,
                            overridden=False,
                            gate=True,
                        )
                    self.trigger = BimanualRecoveryTrigger.RECEIVER_MISS
                    self.requires_giver_connection = True
                    transition = self._transition(
                        BimanualRecoveryState.RECEIVER_MISS_DETECTED
                    )
                    action = self._hold_action(
                        normal_action,
                        left_pose,
                        right_pose,
                        right_gripper=1.0,
                    )
                    return self._decision(
                        action,
                        pause=True,
                        overridden=True,
                        gate=True,
                        transition=transition,
                    )
                gate = self.verified_both_steps < self.config.minimum_verified_both_steps
                return self._decision(
                    normal_action,
                    pause=gate,
                    overridden=False,
                    gate=gate,
                )

            self.miss_evidence_steps = 0
            if task_state in {
                HandoverState.TRANSFER,
                HandoverState.LEFT_RELEASE,
                HandoverState.RIGHT_TO_TARGET,
            }:
                if right_relation_state in {
                    RelationState.CANDIDATE_LOST,
                    RelationState.DISCONNECTED,
                }:
                    if not self.config.enable_recovery:
                        action = self._hold_action(
                            normal_action,
                            left_pose,
                            right_pose,
                            right_gripper=-1.0,
                        )
                        return self._decision(
                            action,
                            pause=True,
                            overridden=True,
                            gate=True,
                        )
                    self.trigger = BimanualRecoveryTrigger.RECEIVER_LOSS
                    self.requires_giver_connection = left_connected
                    transition = self._transition(
                        BimanualRecoveryState.RECEIVER_LOSS_DETECTED
                    )
                    action = self._hold_action(
                        normal_action,
                        left_pose,
                        right_pose,
                        right_gripper=-1.0,
                    )
                    return self._decision(
                        action,
                        pause=True,
                        overridden=True,
                        gate=True,
                        transition=transition,
                    )
                gate = not right_connected
                action = (
                    self._hold_action(
                        normal_action,
                        left_pose,
                        right_pose,
                        right_gripper=-1.0,
                    )
                    if gate
                    else normal_action
                )
                return self._decision(
                    action,
                    pause=gate,
                    overridden=gate,
                    gate=gate,
                )

            self.verified_both_steps = 0
            return self._decision(
                normal_action,
                pause=False,
                overridden=False,
                gate=False,
            )

        self.state_steps += 1
        self.total_recovery_steps += 1
        if self.total_recovery_steps > self.config.maximum_recovery_steps:
            transition = self._transition(BimanualRecoveryState.RECOVERY_FAILED)

        if self.state == BimanualRecoveryState.RECEIVER_LOSS_DETECTED:
            action = self._hold_action(
                normal_action,
                left_pose,
                right_pose,
                right_gripper=-1.0,
            )
            if right_connected:
                transition = self._transition(BimanualRecoveryState.NORMAL)
                self.trigger = BimanualRecoveryTrigger.NONE
                return self._decision(
                    normal_action,
                    pause=False,
                    overridden=False,
                    gate=False,
                    transition=transition,
                )
            if self.state_steps >= self.config.loss_confirmation_steps:
                self.retreat_target = right_pose[:3] + np.asarray(
                    self.config.right_retreat_delta_xyz_m,
                    dtype=np.float64,
                )
                transition = self._transition(BimanualRecoveryState.SAFE_RETREAT)
                action = self._recovery_action(
                    normal_action,
                    left_pose,
                    right_pose,
                    self.retreat_target,
                    1.0,
                )
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.RECEIVER_MISS_DETECTED:
            self.retreat_target = right_pose[:3] + np.asarray(
                self.config.right_retreat_delta_xyz_m,
                dtype=np.float64,
            )
            transition = self._transition(BimanualRecoveryState.SAFE_RETREAT)
            action = self._recovery_action(
                normal_action,
                left_pose,
                right_pose,
                self.retreat_target,
                1.0,
            )
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.SAFE_RETREAT:
            if self.retreat_target is None:
                raise RuntimeError("SAFE_RETREAT 缺少冻结的撤离目标")
            if self._at(right_pose, self.retreat_target, self.config.position_tolerance_m) and (
                right_gripper_opening_m >= self.config.open_gripper_threshold_m
            ):
                transition = self._transition(BimanualRecoveryState.REAPPROACH)
            elif self.state_steps >= self.config.maximum_state_steps:
                transition = self._transition(BimanualRecoveryState.RECOVERY_FAILED)
            action = self._recovery_action(
                normal_action,
                left_pose,
                right_pose,
                self.retreat_target,
                1.0,
            )
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.REAPPROACH:
            pregrasp = self._object_site(
                object_pose,
                tuple(
                    np.asarray(self.config.right_grasp_offset_xyz_m)
                    + np.asarray(self.config.right_pregrasp_clearance_xyz_m)
                ),
            )
            if self._at(right_pose, pregrasp, self.config.position_tolerance_m):
                if self.regrasp_attempts >= self.config.maximum_regrasp_attempts:
                    transition = self._transition(BimanualRecoveryState.RECOVERY_FAILED)
                else:
                    self.regrasp_attempts += 1
                    self.regrasp_closing = False
                    transition = self._transition(BimanualRecoveryState.REGRASP)
            elif self.state_steps >= self.config.maximum_state_steps:
                transition = self._transition(BimanualRecoveryState.RECOVERY_FAILED)
            action = self._recovery_action(
                normal_action,
                left_pose,
                right_pose,
                pregrasp,
                1.0,
            )
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.REGRASP:
            grasp = self._object_site(object_pose, self.config.right_grasp_offset_xyz_m)
            if not self.regrasp_closing and self._at(
                right_pose,
                grasp,
                self.config.grasp_position_tolerance_m,
            ):
                self.regrasp_closing = True
            if self.regrasp_closing and right_connected:
                self.verified_both_steps = 0
                verification_state = (
                    BimanualRecoveryState.VERIFY_BOTH
                    if self.requires_giver_connection
                    else BimanualRecoveryState.VERIFY_RECEIVER
                )
                transition = self._transition(verification_state)
            elif self.state_steps >= self.config.maximum_state_steps:
                if self.regrasp_attempts >= self.config.maximum_regrasp_attempts:
                    transition = self._transition(BimanualRecoveryState.RECOVERY_FAILED)
                else:
                    self.retreat_target = right_pose[:3] + np.asarray(
                        self.config.right_retreat_delta_xyz_m,
                        dtype=np.float64,
                    )
                    self.regrasp_closing = False
                    transition = self._transition(BimanualRecoveryState.SAFE_RETREAT)
            action = self._recovery_action(
                normal_action,
                left_pose,
                right_pose,
                grasp,
                -1.0 if self.regrasp_closing else 1.0,
            )
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.VERIFY_BOTH:
            if left_connected and right_connected:
                self.verified_both_steps += 1
            else:
                self.verified_both_steps = 0
            action = self._hold_action(
                normal_action,
                left_pose,
                right_pose,
                right_gripper=-1.0,
            )
            if self.verified_both_steps >= self.config.minimum_verified_both_steps:
                transition = self._transition(BimanualRecoveryState.RESUME_TASK)
            elif self.state_steps >= self.config.maximum_state_steps:
                if self.regrasp_attempts >= self.config.maximum_regrasp_attempts:
                    transition = self._transition(BimanualRecoveryState.RECOVERY_FAILED)
                else:
                    self.retreat_target = right_pose[:3] + np.asarray(
                        self.config.right_retreat_delta_xyz_m,
                        dtype=np.float64,
                    )
                    transition = self._transition(BimanualRecoveryState.SAFE_RETREAT)
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.VERIFY_RECEIVER:
            if right_connected:
                self.verified_both_steps += 1
            else:
                self.verified_both_steps = 0
            action = self._hold_action(
                normal_action,
                left_pose,
                right_pose,
                right_gripper=-1.0,
            )
            if self.verified_both_steps >= self.config.minimum_verified_both_steps:
                transition = self._transition(BimanualRecoveryState.RESUME_TASK)
            elif self.state_steps >= self.config.maximum_state_steps:
                if self.regrasp_attempts >= self.config.maximum_regrasp_attempts:
                    transition = self._transition(BimanualRecoveryState.RECOVERY_FAILED)
                else:
                    self.retreat_target = right_pose[:3] + np.asarray(
                        self.config.right_retreat_delta_xyz_m,
                        dtype=np.float64,
                    )
                    transition = self._transition(BimanualRecoveryState.SAFE_RETREAT)
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.RESUME_TASK:
            transition = self._transition(BimanualRecoveryState.NORMAL)
            self.trigger = BimanualRecoveryTrigger.NONE
            self.miss_evidence_steps = 0
            return self._decision(
                normal_action,
                pause=False,
                overridden=False,
                gate=False,
                transition=transition,
            )

        if self.state == BimanualRecoveryState.RECOVERY_FAILED:
            safe_target = (
                self.retreat_target
                if self.retreat_target is not None
                else right_pose[:3]
                + np.asarray(self.config.right_retreat_delta_xyz_m, dtype=np.float64)
            )
            action = self._recovery_action(
                normal_action,
                left_pose,
                right_pose,
                safe_target,
                1.0,
            )
            return self._decision(
                action,
                pause=True,
                overridden=True,
                gate=True,
                transition=transition,
            )

        raise RuntimeError(f"未处理的双臂恢复状态：{self.state}")
