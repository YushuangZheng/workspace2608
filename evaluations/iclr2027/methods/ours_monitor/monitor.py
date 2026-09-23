"""Expose the complete method's persisted task-state alarm without recovery."""

from __future__ import annotations

from typing import Any, Mapping

from evaluations.iclr2027.interfaces.runtime_monitor import (
    EpisodeContext,
    RuntimeMonitor,
)


class OursTaskStateMonitor(RuntimeMonitor):
    def __init__(self) -> None:
        self.reset(None)

    def reset(self, episode_context: EpisodeContext | None) -> None:
        self._alarm = False
        self._reason_count = 0
        self._continuous_score = 0.0

    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        monitor = policy_state.get("monitor", {})
        if not isinstance(monitor, Mapping):
            raise ValueError("TSF monitor state is missing from the policy output")
        reasons = monitor.get("reasons", ())
        self._reason_count = len(reasons) if isinstance(reasons, (list, tuple)) else 0
        self._alarm = bool(monitor.get("alarm", False))
        self._continuous_score = float(
            monitor.get("continuous_score", float(self._alarm))
        )

    def score(self) -> Mapping[str, float]:
        return {
            "task_state_mismatch": self._continuous_score,
            "trigger_reasons": float(self._reason_count),
        }

    def alarm(self) -> bool:
        return self._alarm

    @property
    def threshold(self) -> float:
        return 1.0


__all__ = ["OursTaskStateMonitor"]
