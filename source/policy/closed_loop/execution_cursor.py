"""Closed-loop separation of nominal, estimated, and commanded progress."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state_index import StateId
from .task_model import ClosedLoopTaskModel


class ExecutionDecision(str, Enum):
    HOLD = "hold"
    REALIGN = "realign"
    ADVANCE = "advance"


@dataclass(frozen=True)
class ClosedLoopCursor:
    nominal_state: StateId
    estimated_state: StateId
    reference_state: StateId

    def validate(self, task_model: ClosedLoopTaskModel) -> None:
        unknown = {
            self.nominal_state,
            self.estimated_state,
            self.reference_state,
        }.difference(task_model.states)
        if unknown:
            raise KeyError(f"闭环游标包含未知状态：{sorted(unknown)}")


__all__ = ["ClosedLoopCursor", "ExecutionDecision"]
