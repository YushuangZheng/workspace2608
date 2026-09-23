"""Read-only adapter exposing arbitrary-state queries from the base policy."""

from __future__ import annotations

from dataclasses import dataclass

from ...dynamac import DynaMACAction, DynaMACObservation
from ..model.state_index import StateId
from ..model.task_model import TSFTaskModel


@dataclass(frozen=True)
class StateQueryAdapter:
    task_model: TSFTaskModel

    def query_state(
        self,
        observation: DynaMACObservation,
        state_id: StateId,
        stream_weights: dict[str, float] | None = None,
        *,
        mode_index: int | None = None,
    ) -> DynaMACAction:
        return self.task_model.query_state(
            observation,
            state_id,
            stream_weights,
            mode_index=mode_index,
        )


__all__ = ["StateQueryAdapter"]
