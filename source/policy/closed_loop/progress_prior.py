"""Action-after nominal progress prior for the phase-two update chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .boundary_model import BoundaryId
from .state_index import StateId
from .task_model import ClosedLoopTaskModel


def _normalize(values: Mapping[StateId, float]) -> dict[StateId, float]:
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("进度概率必须为有限非负数")
    total = float(sum(values.values()))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("进度概率必须具有正的有限总和")
    return {state: float(value / total) for state, value in values.items()}


@dataclass(frozen=True)
class ProgressPriorConfig:
    incomplete_weight: float = 0.20
    normal_successor_weight: float = 0.65
    early_completion_weight: float = 0.15
    local_backward_radius: int = 1
    local_forward_radius: int = 2

    def __post_init__(self) -> None:
        weights = np.asarray(
            [
                self.incomplete_weight,
                self.normal_successor_weight,
                self.early_completion_weight,
            ],
            dtype=np.float64,
        )
        if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("名义进度转移权重必须非负且归一化")
        if self.local_backward_radius < 0 or self.local_forward_radius < 0:
            raise ValueError("局部候选半径必须非负")


@dataclass(frozen=True)
class ProgressPrior:
    probabilities: dict[StateId, float]
    nominal_state: StateId
    candidates: tuple[StateId, ...]


class ProgressPriorBuilder:
    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: ProgressPriorConfig = ProgressPriorConfig(),
    ) -> None:
        self.task_model = task_model
        self.config = config
        self._states = tuple(sorted(task_model.states))
        self._global_index = {state: index for index, state in enumerate(self._states)}

    def _boundary_allowed(
        self,
        source: StateId,
        target: StateId,
        permitted_boundaries: frozenset[BoundaryId],
    ) -> bool:
        if source.skill_index == target.skill_index:
            return True
        boundary_id = BoundaryId(
            self.task_model.arm_id, source.skill_index, target.skill_index
        )
        return boundary_id in permitted_boundaries

    def _advance(
        self,
        state: StateId,
        steps: int,
        permitted_boundaries: frozenset[BoundaryId],
    ) -> StateId:
        current = state
        for _ in range(steps):
            successors = self.task_model.states[current].topology.successors
            if not successors:
                break
            successor = successors[0]
            if not self._boundary_allowed(current, successor, permitted_boundaries):
                break
            crossed = successor.skill_index != current.skill_index
            current = successor
            # One boundary permission admits only the next skill's entry state.
            if crossed:
                break
        return current

    def build(
        self,
        previous_posterior: Mapping[StateId, float],
        *,
        executed_reference_state: StateId | None = None,
        permitted_boundaries: frozenset[BoundaryId] = frozenset(),
    ) -> ProgressPrior:
        unknown = set(previous_posterior).difference(self.task_model.states)
        if unknown:
            raise KeyError(f"进度后验包含未知状态：{sorted(unknown)}")
        posterior = _normalize(previous_posterior)
        if (
            executed_reference_state is not None
            and executed_reference_state not in self.task_model.states
        ):
            raise KeyError(f"动作引用状态不存在：{executed_reference_state}")

        raw: dict[StateId, float] = {}
        if executed_reference_state is None:
            # Phase-two can still be exercised as a standalone sidecar.  In
            # that case the posterior itself supplies the only available
            # action-state approximation.
            transitions = (
                (0, self.config.incomplete_weight),
                (1, self.config.normal_successor_weight),
                (2, self.config.early_completion_weight),
            )
            for source, probability in posterior.items():
                for steps, weight in transitions:
                    target = self._advance(source, steps, permitted_boundaries)
                    raw[target] = raw.get(target, 0.0) + probability * weight
        else:
            # In the integrated loop, ``reference_state`` identifies the
            # policy action actually queried at t-1.  It anchors the normal
            # and slightly-early action outcomes, while beta_{t-1} is still
            # retained for the incomplete-action branch.  This implements
            # P_prog(s | s', a_{t-1}) beta_{t-1}(s') without either ignoring
            # the action or replacing the posterior by a one-hot cursor.
            for source, probability in posterior.items():
                raw[source] = raw.get(source, 0.0) + (
                    probability * self.config.incomplete_weight
                )
            for steps, weight in (
                (1, self.config.normal_successor_weight),
                (2, self.config.early_completion_weight),
            ):
                target = self._advance(
                    executed_reference_state,
                    steps,
                    permitted_boundaries,
                )
                raw[target] = raw.get(target, 0.0) + weight

        raw = _normalize(raw)
        nominal = max(raw, key=lambda state: (raw[state], -self._global_index[state]))
        center = self._global_index[nominal]
        lower = max(0, center - self.config.local_backward_radius)
        upper = min(len(self._states), center + self.config.local_forward_radius + 1)
        window = set(self._states[lower:upper])
        restricted = {
            state: probability for state, probability in raw.items() if state in window
        }
        restricted = _normalize(restricted)
        candidates = tuple(sorted(restricted, key=self._global_index.__getitem__))
        return ProgressPrior(restricted, nominal, candidates)


__all__ = [
    "ProgressPrior",
    "ProgressPriorBuilder",
    "ProgressPriorConfig",
]
