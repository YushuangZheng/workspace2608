"""Static Gaussian multi-stream Product-of-Experts policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from essay2608.data.dataset import Demonstration
from essay2608.data.transforms import pose_multiply, quaternion_to_matrix

from .base import PhaseClockPolicy, PolicyObservation, PolicyStep
from .gaussian import FrameGaussianModel, fit_frame_gaussian


IDENTITY_POSE = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class StaticMultiStreamPolicy(PhaseClockPolicy):
    """MiDiGaP-style world/object/target Gaussian PoE baseline."""

    name = "static_multistream"
    fitted_frames = ("world", "object", "target")

    def __init__(self, bins: int = 25) -> None:
        super().__init__(bins=bins)
        self.models: dict[str, FrameGaussianModel] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def fit(self, demonstrations: list[Demonstration]) -> None:
        self._fit_phase_durations(demonstrations)
        self.models = {
            frame_name: fit_frame_gaussian(demonstrations, frame_name, bins=self.bins)
            for frame_name in self.fitted_frames
        }

    def _on_reset(self, observation: PolicyObservation) -> None:
        del observation
        self.last_diagnostics = {}

    def _frame_pose(self, frame_name: str, observation: PolicyObservation) -> np.ndarray:
        if frame_name == "world":
            return IDENTITY_POSE
        if frame_name == "object":
            return observation.object_pose
        if frame_name == "target":
            return observation.target_pose
        raise ValueError(frame_name)

    def _active_frames(self, observation: PolicyObservation) -> list[str]:
        del observation
        return ["object", "target"]

    def _update_online_state(self, observation: PolicyObservation, index: int) -> None:
        del observation, index

    def _connection_state(self) -> bool:
        return False

    def _stream_world_distribution(
        self,
        frame_name: str,
        observation: PolicyObservation,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = self.models[frame_name]
        frame_pose = self._frame_pose(frame_name, observation)
        world_pose = pose_multiply(frame_pose, model.mean_pose[self.phase, index])
        rotation = quaternion_to_matrix(frame_pose[3:7])
        covariance = rotation @ model.position_covariance[self.phase, index] @ rotation.T
        return world_pose, covariance

    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        if not self.models:
            raise RuntimeError("Policy must be fitted before use.")
        index = self.profile_index()
        self._update_online_state(observation, index)
        active_frames = self._active_frames(observation)

        stream_means: dict[str, np.ndarray] = {}
        stream_covariances: dict[str, np.ndarray] = {}
        precisions: dict[str, np.ndarray] = {}
        precision_scores: dict[str, float] = {}
        for frame_name in active_frames:
            mean_pose, covariance = self._stream_world_distribution(frame_name, observation, index)
            precision = np.linalg.inv(covariance)
            stream_means[frame_name] = mean_pose
            stream_covariances[frame_name] = covariance
            precisions[frame_name] = precision
            precision_scores[frame_name] = float(np.trace(precision) / 3.0)

        total_precision = np.sum(list(precisions.values()), axis=0)
        fused_covariance = np.linalg.inv(total_precision)
        information = np.sum(
            [precisions[name] @ stream_means[name][:3] for name in active_frames],
            axis=0,
        )
        fused_position = fused_covariance @ information

        world_model = self.models["world"]
        orientation = world_model.mean_pose[self.phase, index, 3:7]
        gripper = -1.0 if world_model.mean_gripper[self.phase, index] < 0.0 else 1.0
        action = np.concatenate((fused_position, orientation, [gripper]))

        score_sum = sum(precision_scores.values())
        diagnostics = {
            "method": self.name,
            "phase": self.phase,
            "phase_name": self.phase_name,
            "profile_index": index,
            "active_frames": active_frames,
            "stream_uncertainty_m": {
                name: float(np.linalg.det(stream_covariances[name]) ** (1.0 / 6.0))
                for name in active_frames
            },
            "stream_precision": precision_scores,
            "stream_weights": {name: score / score_sum for name, score in precision_scores.items()},
            "fused_uncertainty_m": float(np.linalg.det(fused_covariance) ** (1.0 / 6.0)),
            "connected": self._connection_state(),
        }
        self.last_diagnostics = diagnostics
        return PolicyStep(action=action, diagnostics=diagnostics)
