"""Metrics for the single-arm DynaMAC ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeTrace:
    """Compact rollout trace used to compute paper-facing metrics."""

    control_dt: float
    ee_positions: list[np.ndarray] = field(default_factory=list)
    object_positions: list[np.ndarray] = field(default_factory=list)
    target_positions: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    connected: list[bool] = field(default_factory=list)
    perturbation_active: list[bool] = field(default_factory=list)
    active_frames: list[list[str]] = field(default_factory=list)
    stream_weights: list[dict[str, float]] = field(default_factory=list)

    def append(
        self,
        observation,
        action: np.ndarray,
        diagnostics: dict[str, Any],
        inference_ms: float,
        perturbation_active: bool,
    ) -> None:
        self.ee_positions.append(observation.ee_pose[:3].copy())
        self.object_positions.append(observation.object_pose[:3].copy())
        self.target_positions.append(observation.target_pose[:3].copy())
        self.actions.append(action.copy())
        self.phases.append(int(diagnostics["phase"]))
        self.inference_ms.append(float(inference_ms))
        self.connected.append(bool(diagnostics.get("connected", False)))
        self.perturbation_active.append(bool(perturbation_active))
        self.active_frames.append(list(diagnostics.get("active_frames", [])))
        self.stream_weights.append(dict(diagnostics.get("stream_weights", {})))

    def summary(
        self,
        final_error: float,
        success_threshold: float,
        policy_complete: bool,
        environment_done: bool,
        forced_transitions: int,
        perturbation_started: bool,
    ) -> dict[str, Any]:
        ee = np.asarray(self.ee_positions)
        actions = np.asarray(self.actions)
        phases = np.asarray(self.phases)
        connected = np.asarray(self.connected, dtype=bool)
        expected_connected = np.isin(phases, [4, 5, 6])

        path_length = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=-1))) if len(ee) > 1 else 0.0
        speed = np.linalg.norm(np.diff(ee, axis=0), axis=-1) / self.control_dt if len(ee) > 1 else np.zeros(1)
        action_jump = (
            np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1) if len(actions) > 1 else np.zeros(1)
        )
        false_positive = float(np.mean(connected & ~expected_connected)) if len(connected) else 0.0
        false_negative = float(np.mean(~connected & expected_connected)) if len(connected) else 0.0

        frame_switch_steps = [
            index
            for index in range(1, len(self.active_frames))
            if self.active_frames[index] != self.active_frames[index - 1]
        ]
        object_weights = [weights.get("object", 0.0) for weights in self.stream_weights]
        success = bool(policy_complete and not environment_done and final_error < success_threshold)
        return {
            "success": success,
            "recovery_success": success if perturbation_started else None,
            "policy_complete": bool(policy_complete),
            "environment_done": bool(environment_done),
            "steps": len(self.actions),
            "final_error_m": float(final_error),
            "path_length_m": path_length,
            "max_ee_speed_m_s": float(np.max(speed)),
            "max_action_position_jump_m": float(np.max(action_jump)),
            "mean_inference_ms": float(np.mean(self.inference_ms)),
            "p95_inference_ms": float(np.percentile(self.inference_ms, 95)),
            "frame_switch_steps": frame_switch_steps,
            "connection_detected": bool(np.any(connected)),
            "mask_false_positive_rate": false_positive,
            "mask_false_negative_rate": false_negative,
            "max_object_stream_weight": float(max(object_weights, default=0.0)),
            "forced_phase_transitions": int(forced_transitions),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """Return numeric arrays suitable for NPZ persistence."""

        return {
            "ee_position": np.asarray(self.ee_positions, dtype=np.float32),
            "object_position": np.asarray(self.object_positions, dtype=np.float32),
            "target_position": np.asarray(self.target_positions, dtype=np.float32),
            "action": np.asarray(self.actions, dtype=np.float32),
            "phase": np.asarray(self.phases, dtype=np.int64),
            "inference_ms": np.asarray(self.inference_ms, dtype=np.float32),
            "connected": np.asarray(self.connected, dtype=np.bool_),
            "perturbation_active": np.asarray(self.perturbation_active, dtype=np.bool_),
        }
