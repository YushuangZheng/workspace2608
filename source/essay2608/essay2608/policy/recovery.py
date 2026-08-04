"""Relation-triggered recovery supervisor for single-arm pick-and-place."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum

import numpy as np

from essay2608.data.dataset import Demonstration

from .base import PolicyObservation
from .relation import RelationEstimate, RelationState


class RecoveryState(str, Enum):
    """Lifecycle of the supervisor that temporarily overrides the task policy."""

    NORMAL = "NORMAL"
    MISS_DETECTED = "MISS_DETECTED"
    LOSS_DETECTED = "LOSS_DETECTED"
    SAFE_RETREAT = "SAFE_RETREAT"
    RELOCALIZE = "RELOCALIZE"
    REAPPROACH = "REAPPROACH"
    REGRASP = "REGRASP"
    VERIFY_GRASP = "VERIFY_GRASP"
    RESUME_TASK = "RESUME_TASK"
    RECOVERY_FAILED = "RECOVERY_FAILED"


class RecoveryTrigger(str, Enum):
    """Causal event that started the current recovery episode."""

    NONE = "NONE"
    MISS = "MISS"
    LOSS = "LOSS"


@dataclass(frozen=True)
class RecoveryConfig:
    """Geometry and bounded-state defaults fixed before recovery experiments."""

    miss_verification_steps: int = 12
    loss_confirmation_steps: int = 4
    maximum_regrasp_attempts: int = 2
    maximum_recovery_steps: int = 450
    maximum_state_steps: int = 120
    verify_grasp_steps: int = 80
    retreat_height_m: float = 0.12
    approach_height_m: float = 0.065
    grasp_offset_xyz_m: tuple[float, float, float] = (-0.0028, -0.0005, 0.0105)
    grasp_position_tolerance_m: float = 0.006
    verification_lift_m: float = 0.08
    position_tolerance_m: float = 0.018
    open_gripper_threshold_m: float = 0.068
    occupied_gripper_max_m: float = 0.063
    resume_phase: int = 4
    loss_sensitive_phases: tuple[int, ...] = (4, 5, 6)

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate_recovery_config(
    demonstrations: list[Demonstration],
    base: RecoveryConfig | None = None,
) -> tuple[RecoveryConfig, dict]:
    """Calibrate regrasp alignment from late-grasp samples in frozen demonstrations."""

    if not demonstrations:
        raise ValueError("At least one demonstration is required for recovery calibration.")
    offsets = []
    per_demo_samples = []
    for demonstration in demonstrations:
        indices = demonstration.phase_indices(3)
        selected = indices[len(indices) // 2 :]
        if not len(selected):
            raise ValueError("Every recovery calibration demonstration must contain grasp phase 3.")
        values = demonstration.ee_pose[selected, :3] - demonstration.object_pose[selected, :3]
        offsets.append(values)
        per_demo_samples.append(len(values))
    offset = np.median(np.concatenate(offsets, axis=0), axis=0)
    config = replace(base or RecoveryConfig(), grasp_offset_xyz_m=tuple(float(value) for value in offset))
    diagnostics = {
        "source": "frozen_demonstrations_late_phase_3",
        "num_demonstrations": len(demonstrations),
        "samples_per_demonstration": per_demo_samples,
        "grasp_offset_xyz_m": list(config.grasp_offset_xyz_m),
        "test_seeds_used": False,
    }
    return config, diagnostics


@dataclass(frozen=True)
class RecoveryDecision:
    """One supervisor output for action override and scientific tracing."""

    action: np.ndarray
    state: RecoveryState
    trigger: RecoveryTrigger
    pause_task_clock: bool
    action_overridden: bool
    regrasp_attempts: int
    state_steps: int
    total_recovery_steps: int
    transition: str | None = None
    resume_phase: int | None = None


class RelationRecoveryController:
    """Pure recovery graph driven only by current observations and relation state."""

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        self.reset()

    def reset(self) -> None:
        self.state = RecoveryState.NORMAL
        self.trigger = RecoveryTrigger.NONE
        self.state_steps = 0
        self.total_recovery_steps = 0
        self.regrasp_attempts = 0
        self.had_connection = False
        self.miss_evidence_steps = 0
        self.validation_target: np.ndarray | None = None
        self.regrasp_closing = False

    @property
    def failed(self) -> bool:
        return self.state == RecoveryState.RECOVERY_FAILED

    @property
    def active(self) -> bool:
        return self.state != RecoveryState.NORMAL

    def _transition(self, state: RecoveryState) -> str:
        before = self.state
        self.state = state
        self.state_steps = 0
        return f"{before.value}->{state.value}"

    @staticmethod
    def _action(normal_action: np.ndarray, position: np.ndarray, gripper: float) -> np.ndarray:
        action = np.asarray(normal_action, dtype=np.float64).copy()
        action[:3] = np.asarray(position, dtype=np.float64)
        action[7] = float(gripper)
        return action

    @staticmethod
    def _at(observation: PolicyObservation, position: np.ndarray, tolerance: float) -> bool:
        return bool(np.linalg.norm(observation.ee_pose[:3] - position) <= tolerance)

    def _retreat_target(self, observation: PolicyObservation) -> np.ndarray:
        target = observation.ee_pose[:3].astype(np.float64, copy=True)
        target[2] = max(float(target[2]), float(observation.object_pose[2]) + self.config.retreat_height_m)
        return target

    def _object_target(self, observation: PolicyObservation, height: float) -> np.ndarray:
        target = observation.object_pose[:3].astype(np.float64, copy=True)
        target[2] += height
        return target

    def _grasp_target(self, observation: PolicyObservation) -> np.ndarray:
        return observation.object_pose[:3].astype(np.float64, copy=True) + np.asarray(
            self.config.grasp_offset_xyz_m,
            dtype=np.float64,
        )

    def _decision(
        self,
        action: np.ndarray,
        *,
        overridden: bool,
        transition: str | None = None,
        resume_phase: int | None = None,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            action=action,
            state=self.state,
            trigger=self.trigger,
            pause_task_clock=self.state != RecoveryState.NORMAL,
            action_overridden=overridden,
            regrasp_attempts=self.regrasp_attempts,
            state_steps=self.state_steps,
            total_recovery_steps=self.total_recovery_steps,
            transition=transition,
            resume_phase=resume_phase,
        )

    def update(
        self,
        observation: PolicyObservation,
        relation: RelationEstimate,
        task_phase: int,
        normal_action: np.ndarray,
    ) -> RecoveryDecision:
        """Advance the graph once without reading future state or simulator truth."""

        normal_action = np.asarray(normal_action, dtype=np.float64)
        relation_state = relation.state
        if relation_state == RelationState.CONNECTED:
            self.had_connection = True

        transition = None
        if self.state == RecoveryState.NORMAL:
            self.state_steps = 0
            self.total_recovery_steps = 0
            self.regrasp_attempts = 0
            self.trigger = RecoveryTrigger.NONE
            if (
                self.had_connection
                and task_phase in self.config.loss_sensitive_phases
                and relation_state == RelationState.CANDIDATE_LOST
            ):
                self.trigger = RecoveryTrigger.LOSS
                transition = self._transition(RecoveryState.LOSS_DETECTED)
                freeze = self._action(normal_action, observation.ee_pose[:3], -1.0)
                return self._decision(freeze, overridden=True, transition=transition)

            closed = bool(
                observation.gripper_opening_m is not None
                and observation.gripper_opening_m <= self.config.occupied_gripper_max_m
            )
            if task_phase == self.config.resume_phase and not self.had_connection and closed:
                self.miss_evidence_steps += 1
            elif relation_state == RelationState.CONNECTED or task_phase < self.config.resume_phase:
                self.miss_evidence_steps = 0
            if self.miss_evidence_steps >= self.config.miss_verification_steps:
                self.trigger = RecoveryTrigger.MISS
                transition = self._transition(RecoveryState.MISS_DETECTED)
                freeze = self._action(normal_action, observation.ee_pose[:3], -1.0)
                return self._decision(freeze, overridden=True, transition=transition)
            return self._decision(normal_action, overridden=False)

        self.state_steps += 1
        self.total_recovery_steps += 1
        if (
            self.total_recovery_steps > self.config.maximum_recovery_steps
            or self.state_steps > self.config.maximum_state_steps
        ):
            transition = self._transition(RecoveryState.RECOVERY_FAILED)

        if self.state == RecoveryState.LOSS_DETECTED:
            freeze = self._action(normal_action, observation.ee_pose[:3], -1.0)
            if relation_state == RelationState.CONNECTED:
                transition = self._transition(RecoveryState.NORMAL)
                self.trigger = RecoveryTrigger.NONE
                return self._decision(normal_action, overridden=False, transition=transition)
            if (
                relation_state == RelationState.DISCONNECTED
                or self.state_steps >= self.config.loss_confirmation_steps
            ):
                transition = self._transition(RecoveryState.SAFE_RETREAT)
            return self._decision(freeze, overridden=True, transition=transition)

        if self.state == RecoveryState.MISS_DETECTED:
            transition = self._transition(RecoveryState.SAFE_RETREAT)
            retreat = self._retreat_target(observation)
            return self._decision(
                self._action(normal_action, retreat, 1.0),
                overridden=True,
                transition=transition,
            )

        if self.state == RecoveryState.SAFE_RETREAT:
            retreat = self._retreat_target(observation)
            if (
                self._at(observation, retreat, self.config.position_tolerance_m)
                and observation.gripper_opening_m is not None
                and observation.gripper_opening_m >= self.config.open_gripper_threshold_m
            ):
                transition = self._transition(RecoveryState.RELOCALIZE)
            return self._decision(
                self._action(normal_action, retreat, 1.0),
                overridden=True,
                transition=transition,
            )

        if self.state == RecoveryState.RELOCALIZE:
            relocalize = self._object_target(observation, self.config.retreat_height_m)
            if self._at(observation, relocalize, self.config.position_tolerance_m):
                transition = self._transition(RecoveryState.REAPPROACH)
            return self._decision(
                self._action(normal_action, relocalize, 1.0),
                overridden=True,
                transition=transition,
            )

        if self.state == RecoveryState.REAPPROACH:
            approach = self._object_target(observation, self.config.approach_height_m)
            if self._at(observation, approach, self.config.position_tolerance_m):
                if self.regrasp_attempts >= self.config.maximum_regrasp_attempts:
                    transition = self._transition(RecoveryState.RECOVERY_FAILED)
                else:
                    self.regrasp_attempts += 1
                    self.regrasp_closing = False
                    transition = self._transition(RecoveryState.REGRASP)
            return self._decision(
                self._action(normal_action, approach, 1.0),
                overridden=True,
                transition=transition,
            )

        if self.state == RecoveryState.REGRASP:
            grasp = self._grasp_target(observation)
            if not self.regrasp_closing and self._at(
                observation,
                grasp,
                self.config.grasp_position_tolerance_m,
            ):
                self.regrasp_closing = True
            if (
                self.regrasp_closing
                and self._at(observation, grasp, self.config.position_tolerance_m)
                and observation.gripper_opening_m is not None
                and observation.gripper_opening_m <= self.config.occupied_gripper_max_m
            ):
                self.validation_target = self._object_target(observation, self.config.verification_lift_m)
                transition = self._transition(RecoveryState.VERIFY_GRASP)
            return self._decision(
                self._action(normal_action, grasp, -1.0 if self.regrasp_closing else 1.0),
                overridden=True,
                transition=transition,
            )

        if self.state == RecoveryState.VERIFY_GRASP:
            if self.validation_target is None:
                raise RuntimeError("VERIFY_GRASP requires a captured validation target.")
            verify = self._action(normal_action, self.validation_target, -1.0)
            if relation_state == RelationState.CONNECTED:
                transition = self._transition(RecoveryState.RESUME_TASK)
                return self._decision(
                    verify,
                    overridden=True,
                    transition=transition,
                    resume_phase=self.config.resume_phase,
                )
            if self.state_steps >= self.config.verify_grasp_steps:
                if self.regrasp_attempts >= self.config.maximum_regrasp_attempts:
                    transition = self._transition(RecoveryState.RECOVERY_FAILED)
                else:
                    transition = self._transition(RecoveryState.SAFE_RETREAT)
            return self._decision(verify, overridden=True, transition=transition)

        if self.state == RecoveryState.RESUME_TASK:
            transition = self._transition(RecoveryState.NORMAL)
            self.trigger = RecoveryTrigger.NONE
            self.miss_evidence_steps = 0
            return self._decision(normal_action, overridden=False, transition=transition)

        if self.state == RecoveryState.RECOVERY_FAILED:
            safe = self._retreat_target(observation)
            return self._decision(self._action(normal_action, safe, 1.0), overridden=True, transition=transition)

        raise RuntimeError(f"Unhandled recovery state: {self.state}")
