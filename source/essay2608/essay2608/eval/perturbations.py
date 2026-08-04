"""Deterministic single-arm dynamic evaluation perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONDITIONS = (
    "static",
    "smooth_object",
    "sudden_object",
    "smooth_target",
    "sudden_target",
    "arm_offset",
)


@dataclass(frozen=True)
class PerturbationStatus:
    """One-step perturbation state for trace logging."""

    active: bool
    kind: str
    offset: np.ndarray


class PerturbationController:
    """Apply one controlled test-time shift to an Isaac Lab episode."""

    object_shift = np.asarray([0.0, 0.08, 0.0], dtype=np.float64)
    target_shift = np.asarray([0.0, -0.10, 0.0], dtype=np.float64)
    arm_shift = np.asarray([0.0, 0.06, 0.0], dtype=np.float64)

    def __init__(self, condition: str, env) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        self.condition = condition
        self.env = env
        self.object_asset = env.unwrapped.scene["object"]
        self.env_origin = env.unwrapped.scene.env_origins[0].detach().clone()
        self.initial_object_pose_w = self.object_asset.data.root_pose_w[0].detach().clone()
        self.object_shift_origin_w = None
        self.target_term = env.unwrapped.command_manager.get_term("object_pose")
        self.initial_target_pose = self.target_term.pose_command_b[0].detach().clone()
        self.maximum_fraction = 0.0
        self.event_started = False
        self.event_finished = False

    @staticmethod
    def _smooth_fraction(phase: int, phase_step: int, trigger_phase: int) -> float:
        if phase < trigger_phase:
            return 0.0
        if phase > trigger_phase:
            return 1.0
        return float(np.clip((phase_step - 4) / 20.0, 0.0, 1.0))

    @staticmethod
    def _sudden_fraction(phase: int, phase_step: int, trigger_phase: int) -> float:
        return float(phase > trigger_phase or (phase == trigger_phase and phase_step >= 10))

    def _write_object_shift(self, fraction: float) -> None:
        if self.object_shift_origin_w is None:
            self.object_shift_origin_w = self.object_asset.data.root_pose_w[0].detach().clone()
        pose = self.object_shift_origin_w.clone()
        shift = pose.new_tensor(self.object_shift * fraction)
        pose[:3] += shift
        self.object_asset.write_root_pose_to_sim(pose.unsqueeze(0))

    def _write_target_shift(self, fraction: float) -> None:
        pose = self.initial_target_pose.clone()
        shift = pose.new_tensor(self.target_shift * fraction)
        pose[:3] += shift
        self.target_term.pose_command_b[0] = pose

    def update_scene(self, phase: int, phase_step: int) -> PerturbationStatus:
        """Update object/target perturbations before policy observation."""

        fraction = 0.0
        shift = np.zeros(3, dtype=np.float64)
        if self.condition == "smooth_object":
            fraction = self._smooth_fraction(phase, phase_step, trigger_phase=1)
            shift = self.object_shift
            if phase == 1:
                self._write_object_shift(fraction)
        elif self.condition == "sudden_object":
            fraction = self._sudden_fraction(phase, phase_step, trigger_phase=1)
            shift = self.object_shift
            if phase == 1:
                self._write_object_shift(fraction)
        elif self.condition == "smooth_target":
            fraction = self._smooth_fraction(phase, phase_step, trigger_phase=4)
            shift = self.target_shift
            self._write_target_shift(fraction)
        elif self.condition == "sudden_target":
            fraction = self._sudden_fraction(phase, phase_step, trigger_phase=4)
            shift = self.target_shift
            self._write_target_shift(fraction)

        self.maximum_fraction = max(self.maximum_fraction, fraction)
        self.event_started |= fraction > 0.0
        self.event_finished |= fraction >= 1.0
        return PerturbationStatus(
            active=0.0 < fraction < 1.0,
            kind=self.condition,
            offset=shift * fraction,
        )

    def update_action(self, action: np.ndarray, phase: int, phase_step: int) -> tuple[np.ndarray, PerturbationStatus]:
        """Apply a temporary command offset during transport."""

        result = action.copy()
        active = self.condition == "arm_offset" and phase == 5 and 5 <= phase_step < 25
        if active:
            result[:3] += self.arm_shift
            self.event_started = True
        if self.condition == "arm_offset" and (phase > 5 or (phase == 5 and phase_step >= 25)):
            self.event_finished = True
        offset = self.arm_shift.copy() if active else np.zeros(3, dtype=np.float64)
        return result, PerturbationStatus(active=active, kind=self.condition, offset=offset)
