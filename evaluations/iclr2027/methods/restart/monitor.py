"""Causal fixed no-progress rule for DynaMAC + Restart."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from evaluations.iclr2027.interfaces.runtime_monitor import (
    EpisodeContext,
    RuntimeMonitor,
)


class NoProgressMonitor(RuntimeMonitor):
    def __init__(
        self,
        consecutive_stopped_cycles: int,
        *,
        minimum_command_distance_m: float = 0.005,
        maximum_realized_motion_m: float = 0.001,
    ) -> None:
        if consecutive_stopped_cycles < 1:
            raise ValueError("no-progress persistence must be positive")
        if minimum_command_distance_m <= 0.0 or maximum_realized_motion_m < 0.0:
            raise ValueError("no-progress distances must be non-negative")
        self.required = int(consecutive_stopped_cycles)
        self.minimum_command_distance_m = float(minimum_command_distance_m)
        self.maximum_realized_motion_m = float(maximum_realized_motion_m)
        self.reset(None)

    def reset(self, episode_context: EpisodeContext | None) -> None:
        self._streak = 0
        self._alarm = False
        self._previous_poses: dict[str, np.ndarray] | None = None
        self._previous_targets: dict[str, np.ndarray] | None = None
        self._demanded_arms = 0
        self._stalled_arms = 0

    @staticmethod
    def _targets(action: Mapping[str, Any], arms: tuple[str, ...]) -> dict[str, np.ndarray]:
        value = np.asarray(action.get("action", ()), dtype=np.float64).reshape(-1)
        if arms == ("single",) and value.size == 9:
            return {"single": value[:3].copy()}
        if set(arms) == {"left", "right"} and value.size == 18:
            return {
                "right": value[:3].copy(),
                "left": value[9:12].copy(),
            }
        raise ValueError("no-progress monitor received an invalid action layout")

    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        raw_arms = observation.get("arms", {})
        if not isinstance(raw_arms, Mapping):
            raise ValueError("no-progress monitor requires arm observations")
        arms = tuple(sorted(str(arm) for arm in raw_arms))
        poses = {
            arm: np.asarray(raw_arms[arm]["ee_pose_xyzw"], dtype=np.float64)[:3]
            for arm in arms
        }
        targets = self._targets(action, arms)
        demanded = []
        stalled = []
        if self._previous_poses is not None and self._previous_targets is not None:
            for arm in arms:
                command_distance = float(
                    np.linalg.norm(
                        self._previous_targets[arm] - self._previous_poses[arm]
                    )
                )
                realized_motion = float(
                    np.linalg.norm(poses[arm] - self._previous_poses[arm])
                )
                is_demanded = command_distance >= self.minimum_command_distance_m
                demanded.append(is_demanded)
                stalled.append(
                    is_demanded
                    and realized_motion <= self.maximum_realized_motion_m
                )
        self._demanded_arms = sum(demanded)
        self._stalled_arms = sum(stalled)
        no_progress = bool(any(stalled))
        self._streak = self._streak + 1 if no_progress else 0
        self._alarm = self._streak >= self.required
        self._previous_poses = {arm: pose.copy() for arm, pose in poses.items()}
        self._previous_targets = targets

    def score(self) -> Mapping[str, float]:
        return {
            "no_progress_streak": float(self._streak),
            "demanded_arms": float(self._demanded_arms),
            "stalled_arms": float(self._stalled_arms),
        }

    def alarm(self) -> bool:
        return self._alarm


__all__ = ["NoProgressMonitor"]
