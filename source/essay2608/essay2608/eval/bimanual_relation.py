"""Phase-independent physical truth for the two arm-object relation edges."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmRelationEvidence:
    """Contact and motion evidence for one arm-object edge."""

    connected: bool
    both_fingers_contact: bool
    near_object: bool
    relative_motion_consistent: bool
    force_norms_n: tuple[float, float]
    relative_position_rms_m: float | None
    confidence: float


@dataclass(frozen=True)
class BimanualRelationTruth:
    """Two independent physical edges and their four-value lifecycle label."""

    left: ArmRelationEvidence
    right: ArmRelationEvidence
    label: str


class _ArmPhysicalRelation:
    def __init__(
        self,
        *,
        force_threshold_n: float,
        proximity_threshold_m: float,
        motion_window_steps: int,
        relative_rms_threshold_m: float,
        confirmation_steps: int,
        release_steps: int,
    ) -> None:
        self.force_threshold_n = float(force_threshold_n)
        self.proximity_threshold_m = float(proximity_threshold_m)
        self.relative_rms_threshold_m = float(relative_rms_threshold_m)
        self.confirmation_steps = int(confirmation_steps)
        self.release_steps = int(release_steps)
        self.relative_positions: deque[np.ndarray] = deque(maxlen=int(motion_window_steps))
        self.connected = False
        self.positive_steps = 0
        self.negative_steps = 0

    def update(
        self,
        ee_position: np.ndarray,
        object_position: np.ndarray,
        finger_forces: np.ndarray,
    ) -> ArmRelationEvidence:
        force_norms = np.linalg.norm(np.asarray(finger_forces, dtype=np.float64), axis=-1)
        if force_norms.shape != (2,):
            raise ValueError(f"每臂必须提供两个指尖接触力，实际形状为 {force_norms.shape}")
        relative = np.asarray(object_position, dtype=np.float64) - np.asarray(
            ee_position, dtype=np.float64
        )
        self.relative_positions.append(relative)
        near = bool(np.linalg.norm(relative) <= self.proximity_threshold_m)
        both_contact = bool(np.all(force_norms >= self.force_threshold_n))
        rms = None
        motion_consistent = False
        if len(self.relative_positions) == self.relative_positions.maxlen:
            values = np.stack(tuple(self.relative_positions))
            rms = float(np.sqrt(np.mean(np.sum((values - values.mean(axis=0)) ** 2, axis=-1))))
            motion_consistent = rms <= self.relative_rms_threshold_m

        acquisition_evidence = both_contact and near and motion_consistent
        retention_evidence = both_contact and near
        positive = retention_evidence if self.connected else acquisition_evidence
        if positive:
            self.positive_steps += 1
            self.negative_steps = 0
        else:
            self.negative_steps += 1
            self.positive_steps = 0
        if not self.connected and self.positive_steps >= self.confirmation_steps:
            self.connected = True
        elif self.connected and self.negative_steps >= self.release_steps:
            self.connected = False

        force_score = min(float(np.min(force_norms)) / max(self.force_threshold_n, 1e-9), 1.0)
        distance_score = max(
            0.0,
            1.0 - float(np.linalg.norm(relative)) / max(self.proximity_threshold_m, 1e-9),
        )
        motion_score = (
            max(0.0, 1.0 - rms / max(self.relative_rms_threshold_m, 1e-9))
            if rms is not None
            else 0.0
        )
        confidence = float(force_score * np.sqrt(distance_score * motion_score))
        return ArmRelationEvidence(
            connected=self.connected,
            both_fingers_contact=both_contact,
            near_object=near,
            relative_motion_consistent=motion_consistent,
            force_norms_n=(float(force_norms[0]), float(force_norms[1])),
            relative_position_rms_m=rms,
            confidence=confidence,
        )


class PhysicalRelationTracker:
    """Infer privileged hand-object edges without consulting expert phase."""

    def __init__(
        self,
        *,
        force_threshold_n: float = 0.15,
        # The receiver grasps a 240 mm baton away from its center.  This gate
        # rejects remote contacts while admitting the known endpoint geometry.
        proximity_threshold_m: float = 0.160,
        motion_window_steps: int = 6,
        relative_rms_threshold_m: float = 0.006,
        confirmation_steps: int = 3,
        release_steps: int = 3,
    ) -> None:
        settings = {
            "force_threshold_n": force_threshold_n,
            "proximity_threshold_m": proximity_threshold_m,
            "motion_window_steps": motion_window_steps,
            "relative_rms_threshold_m": relative_rms_threshold_m,
            "confirmation_steps": confirmation_steps,
            "release_steps": release_steps,
        }
        self.left = _ArmPhysicalRelation(**settings)
        self.right = _ArmPhysicalRelation(**settings)

    def update(
        self,
        *,
        left_ee_position: np.ndarray,
        right_ee_position: np.ndarray,
        object_position: np.ndarray,
        left_finger_forces: np.ndarray,
        right_finger_forces: np.ndarray,
    ) -> BimanualRelationTruth:
        left = self.left.update(left_ee_position, object_position, left_finger_forces)
        right = self.right.update(right_ee_position, object_position, right_finger_forces)
        if left.connected and right.connected:
            label = "both"
        elif left.connected:
            label = "left_only"
        elif right.connected:
            label = "right_only"
        else:
            label = "none"
        return BimanualRelationTruth(left=left, right=right, label=label)
