"""Environment-neutral interface shared by online failure monitors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RuntimeMonitor(Protocol):
    """Causal monitor contract used by shadow and intervention runners."""

    def reset(self, episode_context: Mapping[str, Any]) -> None:
        """Reset all episode-local state without changing the environment."""

    def observe(self, observation: Any, action: Any, policy_state: Any) -> None:
        """Consume one synchronized pre-action sample and its policy output."""

    def score(self) -> dict[str, float]:
        """Return the latest scalar scores and threshold diagnostics."""

    def alarm(self) -> bool:
        """Return the latest causal alarm decision."""


__all__ = ["RuntimeMonitor"]
