"""Dynamic reference masking and virtual end-effector frame."""

from __future__ import annotations

from collections import deque

import numpy as np

from essay2608.data.dataset import Demonstration
from essay2608.data.transforms import relative_pose

from .base import PolicyObservation
from .gaussian import fit_frame_gaussian
from .multistream import StaticMultiStreamPolicy


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


class DynaMACPolicy(MaskOnlyPolicy):
    """Mask invalid object frame and substitute a captured virtual EE frame."""

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
