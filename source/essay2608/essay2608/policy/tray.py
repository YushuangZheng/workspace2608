"""Gaussian ablations for simultaneous bilateral tray transport."""

from __future__ import annotations

from typing import Any

import numpy as np

from essay2608.data.dataset import BimanualDemonstration
from essay2608.data.transforms import interpolate_rows, pose_multiply, quaternion_to_matrix, relative_pose

from .bimanual import BimanualPolicyObservation, BimanualPolicyStep, IDENTITY_POSE, _PoseModel, _fit_pose_model


TRAY_PHASE_NAMES = (
    "rest",
    "approach",
    "grasp",
    "lift",
    "transport",
    "lower",
    "release",
    "retreat",
    "complete",
)
TRAY_MOVEMENT_PHASES = {1, 3, 4, 5, 7}


class TrayGaussianPolicy:
    """Independent, static-object, and DynaMAC bilateral tray policies."""

    valid_modes = ("independent_arms", "static_shared_object", "full_dynamac")

    def __init__(self, mode: str, bins: int = 25) -> None:
        if mode not in self.valid_modes:
            raise ValueError(mode)
        self.mode = mode
        self.bins = int(bins)
        self.models: dict[str, dict[str, list[_PoseModel]]] = {}
        self.phase_durations = np.ones(len(TRAY_PHASE_NAMES), dtype=np.int64)
        self.grippers = np.ones((len(TRAY_PHASE_NAMES), bins, 2), dtype=np.float64)
        self.reset_state()

    def reset_state(self) -> None:
        self.phase = 0
        self.phase_step = 0
        self.forced_transitions = 0
        self._complete = False
        self.virtual_left = None
        self.virtual_right = None

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def connected(self) -> bool:
        return 2 <= self.phase <= 6

    def fit(self, demonstrations: list[BimanualDemonstration]) -> None:
        self.phase_durations = np.asarray(
            [
                round(np.median([len(demo.phase_indices(phase)) for demo in demonstrations]))
                for phase in range(len(TRAY_PHASE_NAMES))
            ],
            dtype=np.int64,
        )
        frames = ("world", "object", "opposite_ee", "virtual_ee")
        self.models = {arm: {frame: [] for frame in frames} for arm in ("left", "right")}
        for phase in range(len(TRAY_PHASE_NAMES)):
            samples = {arm: {frame: [] for frame in frames} for arm in ("left", "right")}
            grippers = []
            for demo in demonstrations:
                indices = demo.phase_indices(phase)
                capture_left = demo.left_ee_pose[demo.phase_indices(2)[0]]
                capture_right = demo.right_ee_pose[demo.phase_indices(2)[0]]
                for arm, action, opposite, capture in (
                    ("left", demo.action[indices, :7], demo.right_ee_pose[indices], capture_left),
                    ("right", demo.action[indices, 8:15], demo.left_ee_pose[indices], capture_right),
                ):
                    samples[arm]["world"].append(action)
                    samples[arm]["object"].append(relative_pose(demo.object_pose[indices], action))
                    samples[arm]["opposite_ee"].append(relative_pose(opposite, action))
                    frame = np.repeat(capture[None], len(indices), axis=0)
                    samples[arm]["virtual_ee"].append(relative_pose(frame, action))
                grippers.append(demo.action[indices][:, [7, 15]])
            for arm in ("left", "right"):
                for frame in frames:
                    self.models[arm][frame].append(_fit_pose_model(samples[arm][frame], self.bins))
            self.grippers[phase] = np.mean(
                np.stack([interpolate_rows(value, self.bins) for value in grippers]), axis=0
            )

    def reset(self, observation: BimanualPolicyObservation) -> None:
        del observation
        self.reset_state()

    def _index(self) -> int:
        duration = max(int(self.phase_durations[self.phase]), 1)
        progress = min(self.phase_step, duration - 1) / max(duration - 1, 1)
        return min(round(progress * (self.bins - 1)), self.bins - 1)

    def _active_frames(self) -> list[str]:
        if self.mode == "independent_arms":
            return ["world"]
        if self.mode == "static_shared_object":
            return ["object"] if 1 <= self.phase <= 6 else ["world"]
        if self.phase in {1, 2}:
            return ["object"]
        if 3 <= self.phase <= 6:
            return ["virtual_ee", "opposite_ee"]
        return ["world"]

    def _frame(self, arm: str, frame: str, observation: BimanualPolicyObservation) -> np.ndarray:
        if frame == "world":
            return IDENTITY_POSE
        if frame == "object":
            return observation.object_pose
        if frame == "opposite_ee":
            return observation.right_ee_pose if arm == "left" else observation.left_ee_pose
        if frame == "virtual_ee":
            value = self.virtual_left if arm == "left" else self.virtual_right
            return IDENTITY_POSE if value is None else value
        raise ValueError(frame)

    def _fuse(
        self, arm: str, observation: BimanualPolicyObservation, index: int
    ) -> tuple[np.ndarray, dict[str, float]]:
        frames = self._active_frames()
        means = {}
        precisions = {}
        scores = {}
        for frame in frames:
            model = self.models[arm][frame][self.phase]
            frame_pose = self._frame(arm, frame, observation)
            means[frame] = pose_multiply(frame_pose, model.mean[index])
            rotation = quaternion_to_matrix(frame_pose[3:7])
            covariance = rotation @ model.covariance[index] @ rotation.T
            precisions[frame] = np.linalg.inv(covariance)
            scores[frame] = float(np.trace(precisions[frame]))
        total = np.sum(list(precisions.values()), axis=0)
        position = np.linalg.solve(
            total, np.sum([precisions[name] @ means[name][:3] for name in frames], axis=0)
        )
        orientation = self.models[arm]["world"][self.phase].mean[index, 3:7]
        denominator = sum(scores.values())
        return np.concatenate((position, orientation)), {
            name: score / denominator for name, score in scores.items()
        }

    def act(self, observation: BimanualPolicyObservation) -> BimanualPolicyStep:
        index = self._index()
        left, left_weights = self._fuse("left", observation, index)
        right, right_weights = self._fuse("right", observation, index)
        grippers = np.where(self.grippers[self.phase, index] < 0.0, -1.0, 1.0)
        action = np.concatenate((left, [grippers[0]], right, [grippers[1]]))
        phase_before = self.phase
        active_before = self._active_frames()
        self.phase_step += 1
        if self.phase_step >= max(int(self.phase_durations[self.phase]), 1):
            reached = max(
                np.linalg.norm(observation.left_ee_pose[:3] - left[:3]),
                np.linalg.norm(observation.right_ee_pose[:3] - right[:3]),
            ) < 0.055
            advance = self.phase not in TRAY_MOVEMENT_PHASES or reached
            if self.phase_step >= int(self.phase_durations[self.phase]) + 180:
                advance = True
                self.forced_transitions += 1
            if advance:
                if self.phase == len(TRAY_PHASE_NAMES) - 1:
                    self._complete = True
                else:
                    self.phase += 1
                    self.phase_step = 0
                    if self.phase == 2:
                        self.virtual_left = observation.left_ee_pose.copy()
                        self.virtual_right = observation.right_ee_pose.copy()
        diagnostics: dict[str, Any] = {
            "method": self.mode,
            "phase": phase_before,
            "phase_name": TRAY_PHASE_NAMES[phase_before],
            "active_frames": active_before,
            "left_stream_weights": left_weights,
            "right_stream_weights": right_weights,
            "connected": self.connected,
        }
        return BimanualPolicyStep(action, diagnostics)
