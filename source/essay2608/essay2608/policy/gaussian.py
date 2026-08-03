"""Phase-aligned Gaussian trajectory models and world-frame baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from essay2608.data.dataset import Demonstration
from essay2608.data.transforms import (
    interpolate_poses,
    interpolate_rows,
    quaternion_mean,
    quaternion_residual_vector,
    relative_pose,
)

from .base import PHASE_NAMES, PhaseClockPolicy, PolicyObservation, PolicyStep


@dataclass(frozen=True)
class FrameGaussianModel:
    """Per-phase, per-progress Gaussian pose trajectories in one frame."""

    frame_name: str
    mean_pose: np.ndarray
    position_covariance: np.ndarray
    pose_covariance: np.ndarray
    mean_gripper: np.ndarray

    def uncertainty_scale(self, phase: int, index: int) -> float:
        covariance = self.position_covariance[phase, index]
        return float(np.linalg.det(covariance) ** (1.0 / 6.0))


def _virtual_frame_pose(demonstration: Demonstration) -> np.ndarray:
    indices = demonstration.phase_indices(4)
    return demonstration.ee_pose[indices[0]]


def _skill_virtual_frame_pose(demonstration: Demonstration, phase: int) -> np.ndarray:
    indices = demonstration.phase_indices(phase)
    return demonstration.ee_pose[indices[0]]


def _frame_poses(demonstration: Demonstration, frame_name: str, indices: np.ndarray) -> np.ndarray | None:
    if frame_name == "world":
        return None
    if frame_name == "object":
        return demonstration.object_pose[indices]
    if frame_name == "target":
        return demonstration.target_pose[indices]
    if frame_name == "virtual_ee":
        pose = _virtual_frame_pose(demonstration)
        return np.repeat(pose[None], len(indices), axis=0)
    if frame_name.startswith("virtual_skill_"):
        phase = int(frame_name.removeprefix("virtual_skill_"))
        pose = _skill_virtual_frame_pose(demonstration, phase)
        return np.repeat(pose[None], len(indices), axis=0)
    raise ValueError(frame_name)


def fit_frame_gaussian(
    demonstrations: list[Demonstration],
    frame_name: str,
    bins: int = 25,
    variance_floor: float = 1.0e-6,
    pose_variance_floor: float = 1.0e-8,
) -> FrameGaussianModel:
    """Fit a time-aligned local trajectory Gaussian for every phase."""

    mean_pose = np.zeros((len(PHASE_NAMES), bins, 7), dtype=np.float64)
    covariance = np.zeros((len(PHASE_NAMES), bins, 3, 3), dtype=np.float64)
    pose_covariance = np.zeros((len(PHASE_NAMES), bins, 6, 6), dtype=np.float64)
    mean_gripper = np.zeros((len(PHASE_NAMES), bins), dtype=np.float64)

    for phase in range(len(PHASE_NAMES)):
        if frame_name == "virtual_ee" and phase < 4:
            covariance[phase] = np.eye(3) * 1.0e6
            pose_covariance[phase] = np.eye(6) * 1.0e6
            mean_pose[phase, :, 3] = 1.0
            continue

        command_trajectories = []
        executed_trajectories = []
        gripper_trajectories = []
        for demonstration in demonstrations:
            indices = demonstration.phase_indices(phase)
            frame_pose = _frame_poses(demonstration, frame_name, indices)
            command_pose = demonstration.action[indices, :7]
            executed_pose = demonstration.ee_pose[indices]
            local_command = command_pose if frame_pose is None else relative_pose(frame_pose, command_pose)
            local_executed = executed_pose if frame_pose is None else relative_pose(frame_pose, executed_pose)
            command_trajectories.append(interpolate_poses(local_command, bins))
            executed_trajectories.append(interpolate_poses(local_executed, bins))
            gripper_trajectories.append(interpolate_rows(demonstration.action[indices, 7:8], bins)[:, 0])

        commands = np.stack(command_trajectories)
        executed = np.stack(executed_trajectories)
        grippers = np.stack(gripper_trajectories)
        mean_pose[phase, :, :3] = np.mean(commands[..., :3], axis=0)
        mean_gripper[phase] = np.mean(grippers, axis=0)
        for index in range(bins):
            mean_pose[phase, index, 3:7] = quaternion_mean(commands[:, index, 3:7])
            executed_mean = np.mean(executed[:, index, :3], axis=0)
            centered = executed[:, index, :3] - executed_mean
            covariance[phase, index] = centered.T @ centered / max(len(demonstrations) - 1, 1)
            covariance[phase, index] += np.eye(3) * variance_floor
            executed_quaternion_mean = quaternion_mean(executed[:, index, 3:7])
            rotation_residual = quaternion_residual_vector(
                executed_quaternion_mean,
                executed[:, index, 3:7],
            )
            pose_residual = np.concatenate((centered, rotation_residual), axis=-1)
            pose_covariance[phase, index] = (
                pose_residual.T @ pose_residual / max(len(demonstrations) - 1, 1)
            )
            pose_covariance[phase, index] += np.eye(6) * pose_variance_floor

    return FrameGaussianModel(
        frame_name=frame_name,
        mean_pose=mean_pose,
        position_covariance=covariance,
        pose_covariance=pose_covariance,
        mean_gripper=mean_gripper,
    )


class WorldGaussianPolicy(PhaseClockPolicy):
    """Absolute world-coordinate Gaussian trajectory baseline."""

    name = "world_gaussian"

    def __init__(self, bins: int = 25) -> None:
        super().__init__(bins=bins)
        self.model: FrameGaussianModel | None = None

    def fit(self, demonstrations: list[Demonstration]) -> None:
        self._fit_phase_durations(demonstrations)
        self.model = fit_frame_gaussian(demonstrations, "world", bins=self.bins)

    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        del observation
        if self.model is None:
            raise RuntimeError("Policy must be fitted before use.")
        index = self.profile_index()
        pose = self.model.mean_pose[self.phase, index].copy()
        gripper = -1.0 if self.model.mean_gripper[self.phase, index] < 0.0 else 1.0
        action = np.concatenate((pose, [gripper]))
        diagnostics = {
            "method": self.name,
            "phase": self.phase,
            "phase_name": self.phase_name,
            "profile_index": index,
            "active_frames": ["world"],
            "stream_uncertainty_m": {
                "world": self.model.uncertainty_scale(self.phase, index),
            },
            "stream_weights": {"world": 1.0},
            "connected": False,
        }
        return PolicyStep(action=action, diagnostics=diagnostics)
