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

    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        monitor = policy_state.get("monitor", {})
        if not isinstance(monitor, Mapping):
            raise ValueError("M6 requires the closed-loop monitor state")
        reasons = monitor.get("reasons", ())
        self._reason_count = len(reasons) if isinstance(reasons, (list, tuple)) else 0
        self._alarm = bool(monitor.get("alarm", False))

    def score(self) -> Mapping[str, float]:
        return {
            "task_state_mismatch": float(self._alarm),
            "trigger_reasons": float(self._reason_count),
        }

    def alarm(self) -> bool:
        return self._alarm


__all__ = ["OursTaskStateMonitor"]
