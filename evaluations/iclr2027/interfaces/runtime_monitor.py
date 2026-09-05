"""Causal interface shared by every execution-time monitor.

The interface deliberately receives no fault name, injection schedule, simulator
audit label, or future observation.  Those values live in a separate evaluation
record and are joined only after inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    task_id: str
    method_id: str
    bimanual: bool
    horizon: int
    feature_schema: str
    method_config_hash: str
    checkpoint_hash: Optional[str] = None


@dataclass(frozen=True)
class MonitorOutput:
    cycle: int
    scores: Mapping[str, float]
    alarm: bool
    threshold: Optional[float]
    persistence_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RuntimeMonitor(ABC):
    """Strictly causal runtime monitor contract."""

    @abstractmethod
    def reset(self, episode_context: EpisodeContext) -> None:
        """Reset all temporal state for exactly one episode."""

    @abstractmethod
    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        """Consume one aligned causal record.

        ``observation`` contains arm state, flat task state, the current
        observation timestamp, and the previous command's physical resolution;
        ``action`` contains the command selected for the current cycle.  No
        evaluator-only fault or audit field is present.
        """

    @abstractmethod
    def score(self) -> Mapping[str, float]:
        """Return current scalar scores without changing monitor state."""

    @abstractmethod
    def alarm(self) -> bool:
        """Return the current persisted alarm decision."""


__all__ = ["EpisodeContext", "MonitorOutput", "RuntimeMonitor"]
