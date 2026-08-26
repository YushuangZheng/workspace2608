"""Runtime precision weighting without changing Gaussian expert means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamac import (
    DynaMACAction,
    DynaMACObservation,
    GaussianMarginal,
    product_of_experts,
)
from .frame_roles import FrameRoleSnapshot
from .state_index import StateId
from .task_model import ClosedLoopTaskModel

Array = np.ndarray


def weighted_product_of_experts(
    marginals: tuple[GaussianMarginal, ...] | list[GaussianMarginal],
    precision_weights: tuple[float, ...] | list[float],
) -> tuple[Array, Array, dict[str, float]]:
    """Delegate to the baseline PoE's already verified precision scaling path."""

    return product_of_experts(
        marginals,
        precision_weights=precision_weights,
    )


@dataclass(frozen=True)
class WeightedPoEResult:
    state_id: StateId
    stream_weights: dict[str, float]
    participating_frames: tuple[str, ...]
    action: DynaMACAction | None

    @property
    def available(self) -> bool:
        return self.action is not None


class WeightedPoEExecutor:
    """Query one reference state using only role-approved precision weights."""

    def __init__(self, task_model: ClosedLoopTaskModel) -> None:
        self.task_model = task_model

    def query(
        self,
        observation: DynaMACObservation,
        state_id: StateId,
        roles: FrameRoleSnapshot,
        *,
        mode_index: int | None = None,
    ) -> WeightedPoEResult:
        if roles.state_id != state_id:
            raise ValueError("流角色状态与动作查询 reference_state 不一致")
        weights = roles.execution_weights
        participating = tuple(
            frame for frame, weight in weights.items() if weight > 0.0
        )
        if not participating:
            return WeightedPoEResult(
                state_id=state_id,
                stream_weights=weights,
                participating_frames=(),
                action=None,
            )
        action = self.task_model.query_state(
            observation,
            state_id,
            weights,
            mode_index=mode_index,
        )
        return WeightedPoEResult(
            state_id=state_id,
            stream_weights=weights,
            participating_frames=participating,
            action=action,
        )


__all__ = [
    "WeightedPoEExecutor",
    "WeightedPoEResult",
    "weighted_product_of_experts",
]
