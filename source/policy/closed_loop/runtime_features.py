"""Shared kinematic and observation-quality features for one control tick."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamac import pose_log_nearest, relative_pose
from .runtime_observation import RuntimeObservation

Array = np.ndarray


@dataclass(frozen=True)
class RuntimeFeatureConfig:
    rotation_length_scale: float = 0.10
    full_excitation_motion: float = 0.01

    def __post_init__(self) -> None:
        if self.rotation_length_scale < 0.0:
            raise ValueError("rotation_length_scale 必须非负")
        if self.full_excitation_motion <= 0.0:
            raise ValueError("full_excitation_motion 必须为正数")


@dataclass(frozen=True)
class RuntimeFeatures:
    tick: int
    ee_pose: Array
    actual_ee_motion: Array
    commanded_ee_motion: Array
    frame_poses: dict[str, Array]
    frame_world_motion: dict[str, Array]
    relative_poses: dict[str, Array]
    relative_motion_residuals: dict[str, Array]
    gripper_state: Array
    gripper_change: Array
    frame_visibility: dict[str, bool]
    tracking_reliability: dict[str, float]
    frame_pair_available: dict[str, bool]
    paired_tracking_reliability: dict[str, float]
    actual_motion_magnitude: float
    command_motion_magnitude: float
    command_response_consistency: float
    command_tracking_available: bool
    command_tracking_compatibility: float
    command_tracking_mahalanobis_squared: float
    action_excitation: float
    relation_information_weight: dict[str, float]
    entity_configurations: dict[str, dict[str, Array]]


class RuntimeFeatureBuilder:
    """Compute phase-two features once and share the immutable result."""

    def __init__(self, config: RuntimeFeatureConfig = RuntimeFeatureConfig()) -> None:
        self.config = config

    def motion_component_magnitudes(self, tangent: Array) -> Array:
        """Return translation and rotation-equivalent action magnitudes.

        Keeping the two components available is important for the
        action-conditioned relation likelihood: a translational action does
        not, by itself, identify a relation through an uncommanded rotational
        contact wobble (and vice versa).  ``motion_magnitude`` remains the
        Euclidean norm of these two physically scaled components.
        """

        value = np.asarray(tangent, dtype=np.float64)
        if value.shape != (6,):
            raise ValueError("运动切向量必须为 [6]")
        translation = float(np.linalg.norm(value[:3]))
        # The baseline S3 logarithm is half-angle; convert it to physical angle.
        rotation = 2.0 * float(np.linalg.norm(value[3:]))
        return np.asarray(
            [translation, self.config.rotation_length_scale * rotation],
            dtype=np.float64,
        )

    def motion_magnitude(self, tangent: Array) -> float:
        return float(np.linalg.norm(self.motion_component_magnitudes(tangent)))

    def build(
        self,
        observation: RuntimeObservation,
        previous_observation: RuntimeObservation | None = None,
    ) -> RuntimeFeatures:
        previous_ee = observation.previous_ee_pose
        if previous_ee is None and previous_observation is not None:
            previous_ee = previous_observation.ee_pose

        actual_motion = (
            np.zeros(6, dtype=np.float64)
            if previous_ee is None
            else pose_log_nearest(previous_ee, observation.ee_pose)
        )
        command_motion = (
            np.zeros(6, dtype=np.float64)
            if previous_ee is None or observation.previous_command_pose is None
            else pose_log_nearest(previous_ee, observation.previous_command_pose)
        )
        actual_magnitude = self.motion_magnitude(actual_motion)
        command_magnitude = self.motion_magnitude(command_motion)
        raw_excitation = float(
            np.clip(
                actual_magnitude / self.config.full_excitation_motion,
                0.0,
                1.0,
            )
        )
        command_response = (
            1.0
            if observation.previous_command_pose is None
            or command_magnitude <= np.finfo(np.float64).eps
            else float(np.clip(actual_magnitude / command_magnitude, 0.0, 1.0))
        )
        excitation = raw_excitation * command_response
        tracking_available = bool(
            observation.previous_command_pose is not None
            and observation.previous_command_covariance is not None
        )
        if tracking_available:
            assert observation.previous_command_pose is not None
            assert observation.previous_command_covariance is not None
            tracking_residual = pose_log_nearest(
                observation.previous_command_pose,
                observation.ee_pose,
            )
            covariance = (
                observation.previous_command_covariance
                + np.eye(6, dtype=np.float64) * 1.0e-12
            )
            try:
                solved = np.linalg.solve(covariance, tracking_residual)
            except np.linalg.LinAlgError:
                solved = np.linalg.pinv(covariance) @ tracking_residual
            tracking_mahalanobis = max(0.0, float(tracking_residual @ solved))
            tracking_compatibility = float(
                np.exp(max(-750.0, -0.5 * tracking_mahalanobis / 6.0))
            )
        else:
            tracking_mahalanobis = 0.0
            tracking_compatibility = 1.0

        frame_motion: dict[str, Array] = {}
        relative_poses: dict[str, Array] = {}
        residuals: dict[str, Array] = {}
        visibility: dict[str, bool] = {}
        reliability: dict[str, float] = {}
        pair_available: dict[str, bool] = {}
        paired_tracking_reliability: dict[str, float] = {}
        information: dict[str, float] = {}
        for name, current_frame in observation.frame_poses.items():
            current_relative = relative_pose(current_frame, observation.ee_pose)
            relative_poses[name] = current_relative
            previous_frame = (
                None
                if previous_observation is None
                else previous_observation.frame_poses.get(name)
            )
            if previous_frame is None:
                frame_motion[name] = np.zeros(6, dtype=np.float64)
                residuals[name] = np.zeros(6, dtype=np.float64)
            else:
                assert previous_observation is not None
                frame_motion[name] = pose_log_nearest(previous_frame, current_frame)
                prior_ee = (
                    previous_observation.ee_pose
                    if observation.previous_ee_pose is None
                    else observation.previous_ee_pose
                )
                previous_relative = relative_pose(previous_frame, prior_ee)
                residuals[name] = pose_log_nearest(previous_relative, current_relative)
            visibility[name] = observation.visibility(name)
            reliability[name] = observation.reliability(name)
            prior_visible = bool(
                previous_observation is not None
                and previous_frame is not None
                and previous_observation.visibility(name)
            )
            paired_reliability = (
                min(reliability[name], previous_observation.reliability(name))
                if prior_visible and previous_observation is not None
                else 0.0
            )
            pair_available[name] = bool(visibility[name] and prior_visible)
            paired_tracking_reliability[name] = paired_reliability
            # Relation evidence is action-conditioned: it is fully informative
            # when robot motion dominates the frame comparison (including the
            # canonical disconnect case where the frame stays still), but is
            # discounted when an independently moving frame overwhelms the
            # robot's own motion.  The latter commonly occurs during transient
            # multi-body contact and cannot by itself identify this arm-frame
            # relation.
            frame_motion_magnitude = self.motion_magnitude(frame_motion[name])
            numerical_epsilon = float(np.finfo(np.float64).eps)
            action_dominance = float(
                np.clip(
                    actual_magnitude / max(frame_motion_magnitude, numerical_epsilon),
                    0.0,
                    1.0,
                )
            )
            information[name] = (
                excitation * paired_reliability * action_dominance
                if pair_available[name]
                else 0.0
            )

        if previous_observation is None:
            gripper_change = np.zeros_like(observation.gripper_state)
        elif (
            previous_observation.gripper_state.shape == observation.gripper_state.shape
        ):
            gripper_change = (
                observation.gripper_state - previous_observation.gripper_state
            )
        else:
            gripper_change = np.zeros_like(observation.gripper_state)

        return RuntimeFeatures(
            tick=observation.tick,
            ee_pose=observation.ee_pose.copy(),
            actual_ee_motion=actual_motion.copy(),
            commanded_ee_motion=command_motion.copy(),
            frame_poses={
                name: value.copy() for name, value in observation.frame_poses.items()
            },
            frame_world_motion={
                name: value.copy() for name, value in frame_motion.items()
            },
            relative_poses={
                name: value.copy() for name, value in relative_poses.items()
            },
            relative_motion_residuals={
                name: value.copy() for name, value in residuals.items()
            },
            gripper_state=observation.gripper_state.copy(),
            gripper_change=gripper_change.copy(),
            frame_visibility=visibility,
            tracking_reliability=reliability,
            frame_pair_available=pair_available,
            paired_tracking_reliability=paired_tracking_reliability,
            actual_motion_magnitude=actual_magnitude,
            command_motion_magnitude=command_magnitude,
            command_response_consistency=command_response,
            command_tracking_available=tracking_available,
            command_tracking_compatibility=tracking_compatibility,
            command_tracking_mahalanobis_squared=tracking_mahalanobis,
            action_excitation=excitation,
            relation_information_weight=information,
            entity_configurations={
                entity: {feature: value.copy() for feature, value in fields.items()}
                for entity, fields in observation.entity_configurations.items()
            },
        )


__all__ = [
    "RuntimeFeatureBuilder",
    "RuntimeFeatureConfig",
    "RuntimeFeatures",
]
