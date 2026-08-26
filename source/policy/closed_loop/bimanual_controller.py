"""Shared-snapshot boundary evaluation for one or multiple robot arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .belief_updater import ClosedLoopBelief
from .boundary_runtime import (
    BoundaryRuntimeConfig,
    LocalCompletionResult,
    TransitionRequest,
)
from .entry_guard import EntryGuard
from .execution_controller import ClosedLoopExecutionController
from .task_model import ClosedLoopTaskModel
from .transition_transaction import (
    TransitionCommitResult,
    TransitionTransactionCoordinator,
)


@dataclass(frozen=True)
class BoundaryCycleResult:
    tick: int
    requests: dict[str, TransitionRequest]
    local_completion: dict[str, LocalCompletionResult]
    transaction: TransitionCommitResult | None


class MultiArmBoundaryController:
    """Evaluate all current boundaries, then commit one validated batch."""

    def __init__(
        self,
        task_models: Mapping[str, ClosedLoopTaskModel],
        execution_controllers: Mapping[str, ClosedLoopExecutionController],
        config: BoundaryRuntimeConfig,
    ) -> None:
        if not task_models:
            raise ValueError("阶段四至少需要一只机械臂")
        self.task_models = dict(task_models)
        self.execution_controllers = dict(execution_controllers)
        self.config = config
        self.guards = {
            arm: EntryGuard(self.task_models, arm, config) for arm in self.task_models
        }
        self.transactions = TransitionTransactionCoordinator(
            self.task_models, self.execution_controllers
        )

    def reset(self) -> None:
        for guard in self.guards.values():
            guard.reset()
        self.transactions = TransitionTransactionCoordinator(
            self.task_models, self.execution_controllers
        )

    def update(
        self,
        beliefs: Mapping[str, ClosedLoopBelief],
        *,
        mode_by_arm_skill: Mapping[str, Mapping[int, int]] | None = None,
    ) -> BoundaryCycleResult:
        if set(beliefs) != set(self.task_models):
            raise ValueError("阶段四必须一次提供所有机械臂的 pre-action 信念")
        ticks = {belief.tick for belief in beliefs.values()}
        if len(ticks) != 1:
            raise ValueError("多臂边界评估必须共享同一 pre-action tick")
        tick = ticks.pop()
        requests = {}
        local_results = {}
        for arm, model in self.task_models.items():
            source_state = self.execution_controllers[arm].cursor.reference_state
            boundary = next(
                (
                    candidate
                    for boundary_id, candidate in sorted(model.boundaries.items())
                    if boundary_id.source_skill == source_state.skill_index
                ),
                None,
            )
            if boundary is None:
                continue
            request, local = self.guards[arm].evaluate(
                boundary.boundary_id,
                beliefs,
                source_state,
                mode_by_arm_skill=mode_by_arm_skill,
            )
            requests[arm] = request
            local_results[arm] = local

        transaction = (
            None if not requests else self.transactions.commit(tuple(requests.values()))
        )
        return BoundaryCycleResult(
            tick=tick,
            requests=requests,
            local_completion=local_results,
            transaction=transaction,
        )


BimanualBoundaryController = MultiArmBoundaryController


__all__ = [
    "BimanualBoundaryController",
    "BoundaryCycleResult",
    "MultiArmBoundaryController",
]
