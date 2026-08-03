"""Data-derived bimanual Gaussian and dynamic cross-arm policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from essay2608.data.dataset import BimanualDemonstration
from essay2608.data.transforms import (
    interpolate_poses,
    interpolate_rows,
    pose_multiply,
    quaternion_mean,
    quaternion_to_matrix,
    relative_pose,
    rotate_vector,
)


IDENTITY_POSE = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
PHASE_NAMES = (
    "rest",
    "left_approach",
    "left_grasp",
    "left_lift",
    "left_to_handover",
    "right_approach",
    "right_grasp",
    "transfer",
    "left_release",
    "right_to_target",
    "right_release",
    "retreat",
    "complete",
)
MOVEMENT_PHASES = {1, 3, 4, 5, 6, 9, 11}


@dataclass(frozen=True)
class BimanualPolicyObservation:
    """Geometric state used by all bimanual policies."""

    left_ee_pose: np.ndarray
    right_ee_pose: np.ndarray
    object_pose: np.ndarray
    target_pose: np.ndarray


@dataclass(frozen=True)
class BimanualPolicyStep:
    """One 16-D Cartesian action and interpretable frame diagnostics."""

    action: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _PoseModel:
    mean: np.ndarray
    covariance: np.ndarray


def _mean_pose(samples: np.ndarray) -> np.ndarray:
    return np.concatenate((np.mean(samples[:, :3], axis=0), quaternion_mean(samples[:, 3:7])))


def _fit_pose_model(samples: list[np.ndarray], bins: int) -> _PoseModel:
    aligned = np.stack([interpolate_poses(value, bins) for value in samples])
    mean = np.stack([_mean_pose(aligned[:, index]) for index in range(bins)])
    covariance = np.empty((bins, 3, 3), dtype=np.float64)
    for index in range(bins):
        positions = aligned[:, index, :3]
        covariance[index] = np.cov(positions, rowvar=False, ddof=1) + np.eye(3) * 2.5e-5
    return _PoseModel(mean=mean, covariance=covariance)


class BimanualGaussianPolicy:
    """Four-mode bimanual baseline sharing one learned phase representation.

    ``static_cross_arm`` deliberately keeps the opposite arm in its PoE after
    transfer.  ``full_dynamac`` activates cross-arm/object frames only while they
    are exogenous and captures virtual gripper frames at each connection event.
    """

    valid_modes = ("independent_arms", "fixed_handover", "static_cross_arm", "full_dynamac")

    def __init__(self, mode: str, bins: int = 25) -> None:
        if mode not in self.valid_modes:
            raise ValueError(f"Unknown bimanual mode: {mode}")
        self.mode = mode
        self.name = mode
        self.bins = int(bins)
        self.models: dict[str, dict[str, list[_PoseModel]]] = {}
        self.phase_durations = np.ones(len(PHASE_NAMES), dtype=np.int64)
        self.grippers = np.ones((len(PHASE_NAMES), self.bins, 2), dtype=np.float64)
        self.reset_state()

    def reset_state(self) -> None:
        self.phase = 0
        self.phase_step = 0
        self.total_step = 0
        self.forced_transitions = 0
        self._complete = False
        self.virtual_left_pose: np.ndarray | None = None
        self.virtual_right_pose: np.ndarray | None = None
        self.right_attachment_offset: np.ndarray | None = None

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def carrier(self) -> str | None:
        if 2 <= self.phase <= 6:
            return "left"
        if 7 <= self.phase <= 10:
            return "right"
        return None

    def fit(self, demonstrations: list[BimanualDemonstration]) -> None:
        if not demonstrations:
            raise ValueError("At least one bimanual demonstration is required.")
        self.phase_durations = np.asarray(
            [
                round(np.median([len(demonstration.phase_indices(phase)) for demonstration in demonstrations]))
                for phase in range(len(PHASE_NAMES))
            ],
            dtype=np.int64,
        )
        frames = ("world", "object", "target", "opposite_ee", "virtual_ee")
        self.models = {arm: {frame: [] for frame in frames} for arm in ("left", "right")}
        gripper_samples: list[list[np.ndarray]] = [[] for _ in PHASE_NAMES]

        for phase in range(len(PHASE_NAMES)):
            phase_by_arm_frame = {
                arm: {frame: [] for frame in frames} for arm in ("left", "right")
            }
            for demonstration in demonstrations:
                indices = demonstration.phase_indices(phase)
                left_action = demonstration.action[indices, :7]
                right_action = demonstration.action[indices, 8:15]
                object_pose = demonstration.object_pose[indices]
                target_pose = demonstration.target_pose[indices]
                left_ee = demonstration.left_ee_pose[indices]
                right_ee = demonstration.right_ee_pose[indices]
                left_capture = demonstration.left_ee_pose[demonstration.phase_indices(2)[0]]
                right_capture = demonstration.right_ee_pose[demonstration.phase_indices(7)[0]]
                for arm, action, opposite, capture in (
                    ("left", left_action, right_ee, left_capture),
                    ("right", right_action, left_ee, right_capture),
                ):
                    phase_by_arm_frame[arm]["world"].append(action)
                    phase_by_arm_frame[arm]["object"].append(relative_pose(object_pose, action))
                    phase_by_arm_frame[arm]["target"].append(relative_pose(target_pose, action))
                    phase_by_arm_frame[arm]["opposite_ee"].append(relative_pose(opposite, action))
                    virtual = np.repeat(capture[None], len(indices), axis=0)
                    phase_by_arm_frame[arm]["virtual_ee"].append(relative_pose(virtual, action))
                gripper_samples[phase].append(demonstration.action[indices][:, [7, 15]])

            for arm in ("left", "right"):
                for frame in frames:
                    self.models[arm][frame].append(
                        _fit_pose_model(phase_by_arm_frame[arm][frame], self.bins)
                    )
            aligned_grippers = np.stack(
                [interpolate_rows(values, self.bins) for values in gripper_samples[phase]]
            )
            self.grippers[phase] = np.mean(aligned_grippers, axis=0)

    def reset(self, observation: BimanualPolicyObservation) -> None:
        del observation
        self.reset_state()

    def set_right_attachment_offset(self, local_position: np.ndarray) -> None:
        """Provide the observed right-hand-to-object translation at transfer."""

        self.right_attachment_offset = np.asarray(local_position, dtype=np.float64).copy()

    def _profile_index(self) -> int:
        duration = max(int(self.phase_durations[self.phase]), 1)
        progress = min(self.phase_step, duration - 1) / max(duration - 1, 1)
        return min(round(progress * (self.bins - 1)), self.bins - 1)

    def _frame_pose(self, arm: str, frame: str, observation: BimanualPolicyObservation) -> np.ndarray:
        if frame == "world":
            return IDENTITY_POSE
        if frame == "object":
            return observation.object_pose
        if frame == "target":
            return observation.target_pose
        if frame == "opposite_ee":
            return observation.right_ee_pose if arm == "left" else observation.left_ee_pose
        if frame == "virtual_ee":
            captured = self.virtual_left_pose if arm == "left" else self.virtual_right_pose
            if captured is None:
                return IDENTITY_POSE
            return captured
        raise ValueError(frame)

    def _active_frames(self, arm: str) -> list[str]:
        phase = self.phase
        if self.mode == "independent_arms":
            return ["world"]
        if self.mode == "fixed_handover":
            if arm == "left" and phase in {1, 2, 3}:
                return ["object"]
            if arm == "right" and phase in {9, 10, 11}:
                return ["target"]
            return ["world"]
        if self.mode == "static_cross_arm":
            # Fixed, demonstration-derived schedule: unlike DynaMAC this cannot
            # change validity online, but it is still a competent static-task
            # baseline instead of multiplying every candidate indiscriminately.
            if arm == "left" and phase in {1, 2, 3}:
                return ["object"]
            if arm == "right" and phase in {5, 6, 7, 8}:
                return ["opposite_ee"]
            if arm == "right" and phase in {9, 10, 11}:
                return ["target"]
            return ["world"]

        # Dynamic reference validity: object is exogenous before connection;
        # the opposite arm is useful only around rendezvous; target is exogenous
        # after transfer.  Virtual frames preserve the captured local skill.
        if arm == "left":
            if phase in {1, 2}:
                return ["object"]
            if phase in {3, 4} and self.virtual_left_pose is not None:
                return ["virtual_ee", "world"]
            return ["world"]
        if phase in {5, 6, 7, 8}:
            return ["object", "opposite_ee"]
        if phase in {9, 10, 11}:
            return ["target"]
        return ["world"]

    def _fuse(
        self,
        arm: str,
        observation: BimanualPolicyObservation,
        index: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        active = self._active_frames(arm)
        means: dict[str, np.ndarray] = {}
        precisions: dict[str, np.ndarray] = {}
        scores: dict[str, float] = {}
        for frame in active:
            model = self.models[arm][frame][self.phase]
            frame_pose = self._frame_pose(arm, frame, observation)
            means[frame] = pose_multiply(frame_pose, model.mean[index])
            rotation = quaternion_to_matrix(frame_pose[3:7])
            covariance = rotation @ model.covariance[index] @ rotation.T
            precisions[frame] = np.linalg.inv(covariance)
            scores[frame] = float(np.trace(precisions[frame]) / 3.0)
        total_precision = np.sum(list(precisions.values()), axis=0)
        position = np.linalg.solve(
            total_precision,
            np.sum([precisions[frame] @ means[frame][:3] for frame in active], axis=0),
        )
        orientation = self.models[arm]["world"][self.phase].mean[index, 3:7]
        pose = np.concatenate((position, orientation))
        score_sum = sum(scores.values())
        return pose, {frame: score / score_sum for frame, score in scores.items()}

    def act(self, observation: BimanualPolicyObservation) -> BimanualPolicyStep:
        if not self.models:
            raise RuntimeError("Policy must be fitted before use.")
        if self._complete:
            raise RuntimeError("Cannot act after completion.")
        index = self._profile_index()
        left_active_frames = self._active_frames("left")
        right_active_frames = self._active_frames("right")
        left_pose, left_weights = self._fuse("left", observation, index)
        right_pose, right_weights = self._fuse("right", observation, index)
        if (
            self.mode == "full_dynamac"
            and self.phase in {9, 10, 11}
            and self.right_attachment_offset is not None
        ):
            # Solve p_object = p_hand + R_hand p_attachment for the hand
            # position.  Static baselines retain the demonstration offset.
            right_pose[:3] = observation.target_pose[:3] - rotate_vector(
                right_pose[3:7], self.right_attachment_offset
            )
        grippers = np.where(self.grippers[self.phase, index] < 0.0, -1.0, 1.0)
        action = np.concatenate((left_pose, [grippers[0]], right_pose, [grippers[1]]))
        phase_before = self.phase
        self.phase_step += 1
        self.total_step += 1
        if self.phase_step >= max(int(self.phase_durations[self.phase]), 1):
            relevant = observation.left_ee_pose if self.phase <= 4 else observation.right_ee_pose
            desired = left_pose if self.phase <= 4 else right_pose
            reached = np.linalg.norm(relevant[:3] - desired[:3]) < 0.05
            advance = self.phase not in MOVEMENT_PHASES or reached
            if self.phase_step >= int(self.phase_durations[self.phase]) + 150:
                advance = True
                self.forced_transitions += 1
            if advance:
                if self.phase == len(PHASE_NAMES) - 1:
                    self._complete = True
                else:
                    self.phase += 1
                    self.phase_step = 0
                    if self.phase == 2:
                        self.virtual_left_pose = observation.left_ee_pose.copy()
                    if self.phase == 7:
                        self.virtual_right_pose = observation.right_ee_pose.copy()
        diagnostics = {
            "method": self.mode,
            "phase": phase_before,
            "phase_name": PHASE_NAMES[phase_before],
            "profile_index": index,
            "left_active_frames": left_active_frames,
            "right_active_frames": right_active_frames,
            "left_stream_weights": left_weights,
            "right_stream_weights": right_weights,
            "carrier": self.carrier,
        }
        return BimanualPolicyStep(action=action, diagnostics=diagnostics)
