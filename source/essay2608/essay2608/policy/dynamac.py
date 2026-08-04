"""Dynamic reference masking and virtual end-effector frame."""

from __future__ import annotations

from collections import deque

import numpy as np

from essay2608.data.dataset import Demonstration
from essay2608.data.transforms import relative_pose

from .base import PolicyObservation, PolicyStep
from .gaussian import fit_frame_gaussian
from .multistream import StaticMultiStreamPolicy
from .relation import (
    OnlineRelationEstimator,
    RelationEstimate,
    RelationSample,
    RelationState,
    calibrate_relation_estimator,
)
from .recovery import RecoveryConfig, RelationRecoveryController, calibrate_recovery_config


class KinematicConnectionDetector:
    """Detect object/EE coupling from low relative variance plus object motion."""

    def __init__(
        self,
        window: int = 10,
        relative_std_threshold: float = 0.0015,
        object_motion_threshold: float = 0.004,
    ) -> None:
        self.window = int(window)
        self.relative_std_threshold = float(relative_std_threshold)
        self.object_motion_threshold = float(object_motion_threshold)
        self.relative_positions: deque[np.ndarray] = deque(maxlen=self.window)
        self.object_positions: deque[np.ndarray] = deque(maxlen=self.window)
        self.connected = False

    def reset(self) -> None:
        self.relative_positions.clear()
        self.object_positions.clear()
        self.connected = False

    def update(self, observation: PolicyObservation, gripper: float) -> bool:
        """Update and return the current connection decision."""

        if gripper > 0.0:
            self.reset()
            return False
        self.relative_positions.append(relative_pose(observation.object_pose, observation.ee_pose)[:3])
        self.object_positions.append(observation.object_pose[:3].copy())
        if len(self.relative_positions) < self.window:
            return self.connected

        relative = np.stack(self.relative_positions)
        object_position = np.stack(self.object_positions)
        relative_rms_std = float(np.sqrt(np.mean(np.square(np.std(relative, axis=0)))))
        object_motion = float(np.linalg.norm(object_position[-1] - object_position[0]))
        if relative_rms_std < self.relative_std_threshold and object_motion > self.object_motion_threshold:
            self.connected = True
        return self.connected


class MaskOnlyPolicy(StaticMultiStreamPolicy):
    """Multi-stream policy that removes the endogenous object stream."""

    name = "mask_only"

    def __init__(self, bins: int = 25) -> None:
        super().__init__(bins=bins)
        self.detector = KinematicConnectionDetector()

    def _on_reset(self, observation: PolicyObservation) -> None:
        super()._on_reset(observation)
        self.detector.reset()

    def _update_online_state(self, observation: PolicyObservation, index: int) -> None:
        world_model = self.models["world"]
        gripper = world_model.mean_gripper[self.phase, index]
        self.detector.update(observation, gripper)

    def _active_frames(self, observation: PolicyObservation) -> list[str]:
        del observation
        if self.detector.connected:
            return ["target"]
        return ["object", "target"]

    def _connection_state(self) -> bool:
        return self.detector.connected


class OnlineDynaMACPrototype(MaskOnlyPolicy):
    """Project prototype with online relation masking and one virtual EE frame."""

    name = "full_dynamac"
    fitted_frames = ("world", "object", "target", "virtual_ee")

    def __init__(self, bins: int = 25) -> None:
        super().__init__(bins=bins)
        self.virtual_frame_pose: np.ndarray | None = None

    def fit(self, demonstrations: list[Demonstration]) -> None:
        self._fit_phase_durations(demonstrations)
        self.models = {
            frame_name: fit_frame_gaussian(demonstrations, frame_name, bins=self.bins)
            for frame_name in self.fitted_frames
        }

    def _on_reset(self, observation: PolicyObservation) -> None:
        super()._on_reset(observation)
        self.virtual_frame_pose = None

    def _on_transition(self, new_phase: int, observation: PolicyObservation) -> None:
        if new_phase == 4:
            self.virtual_frame_pose = observation.ee_pose.copy()

    def _frame_pose(self, frame_name: str, observation: PolicyObservation) -> np.ndarray:
        if frame_name == "virtual_ee":
            if self.virtual_frame_pose is None:
                raise RuntimeError("Virtual EE frame has not been captured.")
            return self.virtual_frame_pose
        return super()._frame_pose(frame_name, observation)

    def _active_frames(self, observation: PolicyObservation) -> list[str]:
        del observation
        if self.detector.connected and self.virtual_frame_pose is not None:
            if self.phase == 4:
                return ["target", "virtual_ee"]
            return ["target"]
        return ["object", "target"]


# Backward-compatible name for saved configs and older scripts. New code should
# use OnlineDynaMACPrototype so it is not confused with the paper-level method.
DynaMACPolicy = OnlineDynaMACPrototype


class RelationDynaMACPolicy(StaticMultiStreamPolicy):
    """Project policy driven by bidirectional, phase-independent relation state."""

    name = "relation_dynamac"
    fitted_frames = ("world", "object", "target", "virtual_ee")

    def __init__(self, bins: int = 25, control_dt: float = 0.02) -> None:
        super().__init__(bins=bins)
        self.control_dt = float(control_dt)
        self.estimator: OnlineRelationEstimator | None = None
        self.calibration_diagnostics: dict = {}
        self.virtual_frame_pose: np.ndarray | None = None
        self.relation_estimate: RelationEstimate | None = None

    def fit(self, demonstrations: list[Demonstration]) -> None:
        self._fit_phase_durations(demonstrations)
        self.models = {
            frame_name: fit_frame_gaussian(demonstrations, frame_name, bins=self.bins)
            for frame_name in self.fitted_frames
        }
        config, self.calibration_diagnostics = calibrate_relation_estimator(
            demonstrations
        )
        self.estimator = OnlineRelationEstimator(config)

    def _on_reset(self, observation: PolicyObservation) -> None:
        super()._on_reset(observation)
        if self.estimator is None:
            raise RuntimeError("Policy must be fitted before reset.")
        self.estimator.reset()
        self.virtual_frame_pose = None
        self.relation_estimate = None

    def _frame_pose(self, frame_name: str, observation: PolicyObservation) -> np.ndarray:
        if frame_name == "virtual_ee":
            if self.virtual_frame_pose is None:
                raise RuntimeError("Virtual EE frame has not been captured.")
            return self.virtual_frame_pose
        return super()._frame_pose(frame_name, observation)

    def _update_online_state(self, observation: PolicyObservation, index: int) -> None:
        del index
        if self.estimator is None:
            raise RuntimeError("Policy must be fitted before use.")
        if observation.gripper_opening_m is None or observation.gripper_velocity_m_s is None:
            raise ValueError("RelationDynaMACPolicy requires actual gripper joint observations.")
        estimate = self.estimator.update(
            RelationSample(
                ee_pose=observation.ee_pose,
                object_pose=observation.object_pose,
                gripper_opening_m=observation.gripper_opening_m,
                gripper_velocity_m_s=observation.gripper_velocity_m_s,
                control_dt_s=self.control_dt,
                contact=observation.object_contact,
            )
        )
        if estimate.transitioned and estimate.state == RelationState.CONNECTED:
            self.virtual_frame_pose = observation.ee_pose.copy()
        self.relation_estimate = estimate

    def _active_frames(self, observation: PolicyObservation) -> list[str]:
        del observation
        if self.estimator is not None and self.estimator.connected:
            if self.virtual_frame_pose is not None and self.phase == 4:
                return ["target", "virtual_ee"]
            return ["target"]
        return ["object", "target"]

    def _connection_state(self) -> bool:
        return bool(self.estimator is not None and self.estimator.connected)

    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        step = super()._compute_action(observation)
        estimate = self.relation_estimate
        diagnostics = {
            **step.diagnostics,
            "policy_family": "bidirectional_online_project_prototype",
            "relation_state": estimate.state.value if estimate else RelationState.DISCONNECTED.value,
            "relation_confidence": estimate.confidence if estimate else 0.0,
            "relation_connection_score": estimate.connection_score if estimate else 0.0,
            "relation_loss_score": estimate.loss_score if estimate else 0.0,
            "relation_features": estimate.features if estimate else {},
        }
        return PolicyStep(action=step.action, diagnostics=diagnostics)


class RelationDynaMACRecoveryPolicy(RelationDynaMACPolicy):
    """RelationDynaMAC plus an independent relation-triggered recovery supervisor."""

    name = "relation_dynamac_recovery"

    def __init__(
        self,
        bins: int = 25,
        control_dt: float = 0.02,
        recovery_config: RecoveryConfig | None = None,
    ) -> None:
        super().__init__(bins=bins, control_dt=control_dt)
        self._explicit_recovery_config = recovery_config
        self.recovery = RelationRecoveryController(recovery_config)
        self.recovery_calibration_diagnostics: dict = {}

    def fit(self, demonstrations: list[Demonstration]) -> None:
        super().fit(demonstrations)
        config, self.recovery_calibration_diagnostics = calibrate_recovery_config(
            demonstrations,
            base=self._explicit_recovery_config,
        )
        self.recovery = RelationRecoveryController(config)

    def _on_reset(self, observation: PolicyObservation) -> None:
        super()._on_reset(observation)
        self.recovery.reset()

    @property
    def recovery_failed(self) -> bool:
        return self.recovery.failed

    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        normal_step = super()._compute_action(observation)
        if self.relation_estimate is None:
            raise RuntimeError("Relation estimate must be updated before recovery supervision.")
        task_phase_before = self.phase
        decision = self.recovery.update(
            observation=observation,
            relation=self.relation_estimate,
            task_phase=task_phase_before,
            normal_action=normal_step.action,
        )
        if decision.resume_phase is not None:
            self.phase = int(decision.resume_phase)
            self.phase_step = 0
            self._complete = False
        diagnostics = {
            **normal_step.diagnostics,
            "method": self.name,
            "policy_family": "bidirectional_relation_with_recovery_graph",
            "task_phase_before_recovery": task_phase_before,
            "phase": self.phase,
            "phase_name": self.phase_name,
            "recovery_state": decision.state.value,
            "recovery_trigger": decision.trigger.value,
            "recovery_transition": decision.transition,
            "recovery_state_steps": decision.state_steps,
            "total_recovery_steps": decision.total_recovery_steps,
            "regrasp_attempts": decision.regrasp_attempts,
            "pause_task_clock": decision.pause_task_clock,
            "recovery_action_override": decision.action_overridden,
        }
        return PolicyStep(action=decision.action, diagnostics=diagnostics)


class OracleRelationRecoveryPolicy(RelationDynaMACRecoveryPolicy):
    """Recovery ablation whose relation input is current privileged contact truth."""

    name = "oracle_relation_recovery"

    def _update_online_state(self, observation: PolicyObservation, index: int) -> None:
        del index
        if observation.object_contact is None:
            raise ValueError("OracleRelationRecoveryPolicy requires current simulator contact truth.")
        connected = bool(observation.object_contact)
        state = RelationState.CONNECTED if connected else RelationState.DISCONNECTED
        transitioned = self.relation_estimate is not None and self.relation_estimate.state != state
        self.relation_estimate = RelationEstimate(
            state=state,
            connected=connected,
            confidence=float(connected),
            connection_score=float(connected),
            loss_score=float(not connected),
            features={"privileged_instantaneous_grasp_predicate": connected},
            transitioned=transitioned,
        )
        if transitioned and connected:
            self.virtual_frame_pose = observation.ee_pose.copy()

    def _active_frames(self, observation: PolicyObservation) -> list[str]:
        del observation
        connected = bool(self.relation_estimate is not None and self.relation_estimate.connected)
        if connected:
            if self.virtual_frame_pose is not None and self.phase == 4:
                return ["target", "virtual_ee"]
            return ["target"]
        return ["object", "target"]

    def _connection_state(self) -> bool:
        return bool(self.relation_estimate is not None and self.relation_estimate.connected)

    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        step = super()._compute_action(observation)
        diagnostics = {
            **step.diagnostics,
            "method": self.name,
            "policy_family": "privileged_contact_relation_with_recovery_graph",
            "oracle_relation": True,
            "oracle_information": "current_privileged_geometry_and_gripper_occupancy_only",
        }
        return PolicyStep(action=step.action, diagnostics=diagnostics)
