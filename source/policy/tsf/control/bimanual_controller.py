"""Shared-snapshot boundary evaluation for one or multiple robot arms."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from ..inference.belief_updater import BeliefUpdater, TSFBelief
from .boundary_runtime import (
    BoundaryRuntimeConfig,
    ConditionKind,
    LocalCompletionResult,
    TransitionPreparation,
    TransitionRequest,
)
from .entry_guard import EntryGuard
from .execution_controller import TSFExecutionController
from ..model.state_index import StateId
from ..model.task_model import TSFTaskModel
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
    preparations: dict[str, TransitionPreparation] = field(default_factory=dict)


class MultiArmBoundaryController:
    """Evaluate all current boundaries, then commit one validated batch."""

    def __init__(
        self,
        task_models: Mapping[str, TSFTaskModel],
        execution_controllers: Mapping[str, TSFExecutionController],
        config: BoundaryRuntimeConfig,
        *,
        belief_updaters: Mapping[str, BeliefUpdater],
        relation_scene_guards: bool = True,
        boundary_gated_advancement: bool = True,
        robot_runtime_variance: float = 0.0,
    ) -> None:
        if not task_models:
            raise ValueError("多臂边界控制器至少需要一只机械臂")
        self.task_models = dict(task_models)
        self.execution_controllers = dict(execution_controllers)
        self.belief_updaters = dict(belief_updaters)
        self.config = config
        self.boundary_gated_advancement = bool(boundary_gated_advancement)
        self.guards = {
            arm: EntryGuard(
                self.task_models,
                arm,
                config,
                relation_scene_guards=relation_scene_guards,
                robot_runtime_variance=robot_runtime_variance,
            )
            for arm in self.task_models
        }
        self.transactions = TransitionTransactionCoordinator(
            self.task_models,
            self.execution_controllers,
            self.belief_updaters,
        )
        self._preparations: dict[str, TransitionPreparation] = {}

    def reset(self) -> None:
        for guard in self.guards.values():
            guard.reset()
        self.transactions = TransitionTransactionCoordinator(
            self.task_models,
            self.execution_controllers,
            self.belief_updaters,
        )
        self._preparations.clear()

    @property
    def preparations(self) -> dict[str, TransitionPreparation]:
        return dict(self._preparations)

    def _required_link_relations(
        self,
        requests: Mapping[str, TransitionRequest],
    ) -> frozenset[tuple[str, str]]:
        """Return LINK relations explicitly required by active entry guards.

        A learned LINK occurrence only says that a relation changes near an
        edge.  It does not by itself make the target-entry close an admissible
        pre-commit action.  Preparation is authorized only when an entry guard
        evaluated in this shared snapshot explicitly requires that exact
        relation.  Reading the condition results also keeps progress-only
        ablations from inheriting relation-boundary behavior.
        """

        required: set[tuple[str, str]] = set()
        for request in requests.values():
            boundary = self.task_models[request.arm_id].boundaries[
                request.boundary_id
            ]
            local = getattr(boundary, "local_completion_model", None)
            relation_conditions = {
                **({} if local is None else local.own_relation_conditions),
                **boundary.relation_conditions,
            }
            for condition_id in request.condition_results:
                if condition_id.kind is not ConditionKind.GUARD_RELATION:
                    continue
                condition = relation_conditions[condition_id.token]
                if condition.required_state != "linked":
                    continue
                relation_arm, frame = condition_id.token.split("/", 1)
                if relation_arm != condition_id.arm_id:
                    raise ValueError("边界关系条件标识与关系键机械臂不一致")
                required.add((relation_arm, frame))
        return frozenset(required)

    @staticmethod
    def _preparation_serves_required_link(
        preparation: TransitionPreparation,
        required: frozenset[tuple[str, str]],
    ) -> bool:
        return any(
            (event.arm_id, event.frame_id) in required
            for event in preparation.event_ids
        )

    def _reconcile_preparations(
        self,
        requests: Mapping[str, TransitionRequest],
        source_by_arm: Mapping[str, StateId],
    ) -> None:
        required_links = self._required_link_relations(requests)
        for arm, request in requests.items():
            boundary = self.task_models[arm].boundaries[request.boundary_id]
            source_state = source_by_arm[arm]
            active = self._preparations.get(arm)
            if active is not None and (
                active.boundary_id != request.boundary_id
                or source_state not in boundary.terminal_window
                or not self._preparation_serves_required_link(
                    active, required_links
                )
            ):
                self._preparations.pop(arm, None)
                active = None
            if (
                active is None
                and request.preparation is not None
                and self._preparation_serves_required_link(
                    request.preparation, required_links
                )
            ):
                self._preparations[arm] = request.preparation

    def evaluate(
        self,
        beliefs: Mapping[str, TSFBelief],
        *,
        arms: frozenset[str] | None = None,
        source_states: Mapping[str, StateId] | None = None,
        mode_by_arm_skill: Mapping[str, Mapping[int, int]] | None = None,
    ) -> BoundaryCycleResult:
        if set(beliefs) != set(self.task_models):
            raise ValueError("多臂边界控制器必须一次提供所有机械臂的动作前信念")
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
        source_by_arm: dict[str, StateId] = {}
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
            source_by_arm[arm] = source_state
            request, local = self.guards[arm].evaluate(
                boundary.boundary_id,
                beliefs,
                source_state,
                mode_by_arm_skill=mode_by_arm_skill,
            )
            if not self.boundary_gated_advancement:
                terminal = model.skill_states[source_state.skill_index][-1]
                ready = source_state == terminal
                local = LocalCompletionResult(
                    boundary_id=boundary.boundary_id,
                    end_probability=float(ready),
                    score=float(ready),
                    threshold=0.0,
                    raw_satisfied=ready,
                    consecutive_cycles=1 if ready else 0,
                    required_cycles=1,
                    done=ready,
                    evidence_available=True,
                )
                request = replace(
                    request,
                    permitted=ready,
                    local_done=ready,
                    condition_results={},
                    verification_requests=(),
                    preparation=None,
                )
            requests[arm] = request
            local_results[arm] = local

        self._reconcile_preparations(requests, source_by_arm)

        for arm in selected_arms.difference(requests):
            self._preparations.pop(arm, None)

        return BoundaryCycleResult(
            tick=tick,
            requests=requests,
            local_completion=local_results,
            transaction=None,
            preparations=self.preparations,
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
        if transaction is not None:
            for request in transaction.committed:
                self._preparations.pop(request.arm_id, None)
        return BoundaryCycleResult(
            tick=evaluation.tick,
            requests=evaluation.requests,
            local_completion=evaluation.local_completion,
            transaction=transaction,
            preparations=self.preparations,
        )

    def update(
        self,
        beliefs: Mapping[str, TSFBelief],
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
