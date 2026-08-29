"""Shared-snapshot boundary evaluation for one or multiple robot arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .belief_updater import BeliefUpdater, ClosedLoopBelief
from .boundary_runtime import (
    BoundaryRuntimeConfig,
    LocalCompletionResult,
    TransitionRequest,
)
from .entry_guard import EntryGuard
from .execution_controller import ClosedLoopExecutionController
from .state_index import StateId
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
        *,
        belief_updaters: Mapping[str, BeliefUpdater],
    ) -> None:
        if not task_models:
            raise ValueError("阶段四至少需要一只机械臂")
        self.task_models = dict(task_models)
        self.execution_controllers = dict(execution_controllers)
        self.belief_updaters = dict(belief_updaters)
        self.config = config
        self.guards = {
            arm: EntryGuard(self.task_models, arm, config) for arm in self.task_models
        }
        self.transactions = TransitionTransactionCoordinator(
            self.task_models,
            self.execution_controllers,
            self.belief_updaters,
        )

    def reset(self) -> None:
        for guard in self.guards.values():
            guard.reset()
        self.transactions = TransitionTransactionCoordinator(
            self.task_models,
            self.execution_controllers,
            self.belief_updaters,
        )

    def evaluate(
        self,
        beliefs: Mapping[str, ClosedLoopBelief],
        *,
        arms: frozenset[str] | None = None,
        source_states: Mapping[str, StateId] | None = None,
        mode_by_arm_skill: Mapping[str, Mapping[int, int]] | None = None,
    ) -> BoundaryCycleResult:
        if set(beliefs) != set(self.task_models):
            raise ValueError("阶段四必须一次提供所有机械臂的 pre-action 信念")
        ticks = {belief.tick for belief in beliefs.values()}
        if len(ticks) != 1:
            raise ValueError("多臂边界评估必须共享同一 pre-action tick")
        tick = ticks.pop()
        selected_arms = set(self.task_models) if arms is None else set(arms)
        unknown_arms = selected_arms.difference(self.task_models)
        if unknown_arms:
            raise KeyError(f"边界评估包含未知机械臂：{sorted(unknown_arms)}")
        requests = {}
        local_results = {}
        for arm, model in self.task_models.items():
            if arm not in selected_arms:
                continue
            source_state = (
                self.execution_controllers[arm].cursor.reference_state
                if source_states is None or arm not in source_states
                else source_states[arm]
            )
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

        return BoundaryCycleResult(
            tick=tick,
            requests=requests,
            local_completion=local_results,
            transaction=None,
        )

    def commit_requests(
        self,
        evaluation: BoundaryCycleResult,
        *,
        requests: tuple[TransitionRequest, ...] | None = None,
        externally_committed_arms: frozenset[str] = frozenset(),
    ) -> BoundaryCycleResult:
        selected = tuple(evaluation.requests.values()) if requests is None else requests
        transaction = (
            None
            if not selected
            else self.transactions.commit(
                selected,
                externally_committed_arms=externally_committed_arms,
            )
        )
        return BoundaryCycleResult(
            tick=evaluation.tick,
            requests=evaluation.requests,
            local_completion=evaluation.local_completion,
            transaction=transaction,
        )

    def update(
        self,
        beliefs: Mapping[str, ClosedLoopBelief],
        *,
        arms: frozenset[str] | None = None,
        mode_by_arm_skill: Mapping[str, Mapping[int, int]] | None = None,
    ) -> BoundaryCycleResult:
        evaluation = self.evaluate(
            beliefs,
            arms=arms,
            mode_by_arm_skill=mode_by_arm_skill,
        )
        return self.commit_requests(evaluation)


BimanualBoundaryController = MultiArmBoundaryController


__all__ = [
    "BimanualBoundaryController",
    "BoundaryCycleResult",
    "MultiArmBoundaryController",
]
