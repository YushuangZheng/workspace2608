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
    "drop_after_grasp",
    "close_without_grasp",
    "drop_lift_early",
    "drop_transport_middle",
    "drop_before_lower",
    "miss_small_shift",
    "miss_large_shift",
    "edge_grasp",
    "normal_no_failure",
)

RECOVERY_CONDITIONS = (
    "drop_lift_early",
    "drop_transport_middle",
    "drop_before_lower",
    "miss_small_shift",
    "miss_large_shift",
    "edge_grasp",
    "normal_no_failure",
)


def recovery_condition_parameters(condition: str, seed: int) -> dict:
    """Return preregistered seed-deterministic parameters for one recovery condition."""

    if condition not in RECOVERY_CONDITIONS:
        raise ValueError(f"Not a recovery-protocol condition: {condition}")
    directions = (
        ("front", np.asarray([1.0, 0.0, 0.0])),
        ("back", np.asarray([-1.0, 0.0, 0.0])),
        ("left", np.asarray([0.0, 1.0, 0.0])),
        ("right", np.asarray([0.0, -1.0, 0.0])),
    )
    direction_name, direction = directions[(int(seed) // 4) % len(directions)]
    common = {
        "condition": condition,
        "seed": int(seed),
        "direction": direction_name,
        "place_on_support": True,
    }
    drop_triggers = {
        "drop_lift_early": (4, 8),
        "drop_transport_middle": (5, 12),
        "drop_before_lower": (5, 22),
    }
    if condition in drop_triggers:
        distance = (0.05, 0.10, 0.15, 0.20)[int(seed) % 4]
        phase, phase_step = drop_triggers[condition]
        return {
            **common,
            "kind": "drop",
            "trigger_phase": phase,
            "trigger_phase_step": phase_step,
            "distance_m": distance,
            "shift_m": (direction * distance).tolist(),
            "force_open_steps": 3 if int(seed) % 2 else 0,
        }
    if condition == "normal_no_failure":
        return {
            "condition": condition,
            "seed": int(seed),
            "kind": "none",
        }
    distance = {
        "miss_small_shift": 0.030,
        "miss_large_shift": 0.100,
        "edge_grasp": 0.018,
    }[condition]
    return {
        **common,
        "kind": "miss",
        "trigger_phase": 3,
        "trigger_phase_step": 0,
        "distance_m": distance,
        "shift_m": (direction * distance).tolist(),
        "edge_case": condition == "edge_grasp",
    }


def perturbation_parameters(condition: str, seed: int) -> dict:
    """Expose every old and preregistered perturbation parameter for fingerprinting."""

    if condition in RECOVERY_CONDITIONS:
        return recovery_condition_parameters(condition, seed)
    configurations = {
        "static": {"kind": "none"},
        "smooth_object": {
            "shift_m": [0.0, 0.08, 0.0],
            "trigger_phase": 1,
            "ramp_phase_steps": [4, 24],
        },
        "sudden_object": {
            "shift_m": [0.0, 0.08, 0.0],
            "trigger_phase": 1,
            "trigger_phase_step": 10,
        },
        "smooth_target": {
            "shift_m": [0.0, -0.10, 0.0],
            "trigger_phase": 4,
            "ramp_phase_steps": [4, 24],
        },
        "sudden_target": {
            "shift_m": [0.0, -0.10, 0.0],
            "trigger_phase": 4,
            "trigger_phase_step": 10,
        },
        "arm_offset": {
            "shift_m": [0.0, 0.06, 0.0],
            "active_phase": 5,
            "active_phase_steps": [5, 25],
        },
        "drop_after_grasp": {
            "shift_m": [0.0, 0.18, 0.0],
            "trigger_phase": 5,
            "trigger_phase_step": 10,
            "place_on_support": True,
            "gripper_command_unchanged": True,
        },
        "close_without_grasp": {
            "shift_m": [0.0, -0.18, 0.0],
            "trigger_phase": 3,
            "place_on_support": True,
            "gripper_command_unchanged": True,
        },
    }
    return configurations[condition]


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
    drop_shift = np.asarray([0.0, 0.18, 0.0], dtype=np.float64)
    miss_shift = np.asarray([0.0, -0.18, 0.0], dtype=np.float64)

    def __init__(self, condition: str, env, seed: int = 0) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        self.condition = condition
        self.seed = int(seed)
        self.parameters = perturbation_parameters(condition, seed)
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
        self.instantaneous_event_applied = False
        self.force_open_steps_remaining = 0

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

    def _teleport_object(self, shift: np.ndarray, place_on_support: bool) -> None:
        pose = self.object_asset.data.root_pose_w[0].detach().clone()
        pose[:3] += pose.new_tensor(shift)
        if place_on_support:
            pose[2] = self.env_origin[2] + 0.021
        self.object_asset.write_root_pose_to_sim(pose.unsqueeze(0))

    def update_scene(self, phase: int, phase_step: int) -> PerturbationStatus:
        """Update object/target perturbations before policy observation."""

        fraction = 0.0
        shift = np.zeros(3, dtype=np.float64)
        if self.condition in RECOVERY_CONDITIONS and self.parameters["kind"] in {"drop", "miss"}:
            trigger_phase = int(self.parameters["trigger_phase"])
            trigger_step = int(self.parameters["trigger_phase_step"])
            should_trigger = phase == trigger_phase and phase_step >= trigger_step
            if should_trigger and not self.instantaneous_event_applied:
                shift = np.asarray(self.parameters["shift_m"], dtype=np.float64)
                self._teleport_object(shift, place_on_support=bool(self.parameters["place_on_support"]))
                self.force_open_steps_remaining = int(self.parameters.get("force_open_steps", 0))
                self.instantaneous_event_applied = True
                self.event_started = True
                self.event_finished = True
                fraction = 1.0
        elif self.condition == "smooth_object":
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
        elif (
            self.condition == "drop_after_grasp"
            and phase == 5
            and phase_step >= 10
            and not self.instantaneous_event_applied
        ):
            self._teleport_object(self.drop_shift, place_on_support=True)
            self.instantaneous_event_applied = True
            self.event_started = True
            self.event_finished = True
            fraction = 1.0
            shift = self.drop_shift
        elif (
            self.condition == "close_without_grasp"
            and phase == 3
            and not self.instantaneous_event_applied
        ):
            self._teleport_object(self.miss_shift, place_on_support=True)
            self.instantaneous_event_applied = True
            self.event_started = True
            self.event_finished = True
            fraction = 1.0
            shift = self.miss_shift

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
        if self.force_open_steps_remaining > 0:
            result[7] = 1.0
            self.force_open_steps_remaining -= 1
        active = self.condition == "arm_offset" and phase == 5 and 5 <= phase_step < 25
        if active:
            result[:3] += self.arm_shift
            self.event_started = True
        if self.condition == "arm_offset" and (phase > 5 or (phase == 5 and phase_step >= 25)):
            self.event_finished = True
        offset = self.arm_shift.copy() if active else np.zeros(3, dtype=np.float64)
        return result, PerturbationStatus(active=active, kind=self.condition, offset=offset)
