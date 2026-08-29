"""Batch validation and atomic phase-four skill-boundary commits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from .boundary_model import BoundaryId
from .boundary_runtime import TransitionRequest
from .execution_controller import ClosedLoopExecutionController
from .task_model import ClosedLoopTaskModel

if TYPE_CHECKING:
    from .belief_updater import BeliefUpdater


@dataclass(frozen=True)
class TransitionCommitResult:
    tick: int
    requests: tuple[TransitionRequest, ...]
    committed: tuple[TransitionRequest, ...]
    held_transaction_groups: tuple[str, ...]

    @property
    def permitted_boundaries(self) -> dict[str, frozenset[BoundaryId]]:
        result: dict[str, set[BoundaryId]] = {}
        for request in self.committed:
            result.setdefault(request.arm_id, set()).add(request.boundary_id)
        return {arm: frozenset(values) for arm, values in result.items()}


class TransitionTransactionCoordinator:
    """Commit independent requests asynchronously and transaction groups jointly."""

    def __init__(
        self,
        task_models: Mapping[str, ClosedLoopTaskModel],
        controllers: Mapping[str, ClosedLoopExecutionController],
        belief_updaters: Mapping[str, BeliefUpdater],
    ) -> None:
        if not set(task_models) == set(controllers) == set(belief_updaters):
            raise ValueError("事务协调器的模型、执行器和信念更新器必须覆盖同一组机械臂")
        if any(model.arm_id != arm for arm, model in task_models.items()):
            raise ValueError("事务协调器模型 arm_id 与字典键不一致")
        self.task_models = dict(task_models)
        self.controllers = dict(controllers)
        self.belief_updaters = dict(belief_updaters)
        groups: dict[str, set[BoundaryId]] = {}
        for model in self.task_models.values():
            for boundary_id, boundary in model.boundaries.items():
                if boundary.transaction_group is not None:
                    groups.setdefault(boundary.transaction_group, set()).add(
                        boundary_id
                    )
        self.transaction_groups = {
            group: frozenset(boundaries) for group, boundaries in groups.items()
        }
        self._last_tick: int | None = None

    def _resolve(
        self,
        requests: Sequence[TransitionRequest],
    ) -> tuple[int, list[TransitionRequest], list[str]]:
        if not requests:
            raise ValueError("事务提交至少需要一个转换请求")
        ticks = {request.tick for request in requests}
        if len(ticks) != 1:
            raise ValueError("所有转换请求必须来自同一 pre-action tick")
        tick = ticks.pop()
        if len({request.arm_id for request in requests}) != len(requests):
            raise ValueError("同一机械臂在一个周期不能生成多个跨界请求")
        unknown_arms = {request.arm_id for request in requests}.difference(
            self.controllers
        )
        if unknown_arms:
            raise KeyError(f"事务请求包含未知机械臂：{sorted(unknown_arms)}")

        by_group: dict[str, list[TransitionRequest]] = {}
        independent = []
        for request in requests:
            if request.transaction_group is None:
                independent.append(request)
            else:
                by_group.setdefault(request.transaction_group, []).append(request)

        candidates = [request for request in independent if request.permitted]
        held_groups = []
        for group, members in sorted(by_group.items()):
            expected = self.transaction_groups.get(group)
            if expected is None:
                raise KeyError(f"转换请求引用未知事务组 {group}")
            received = {request.boundary_id for request in members}
            if received == expected and all(request.permitted for request in members):
                candidates.extend(members)
            else:
                held_groups.append(group)

        return tick, candidates, held_groups

    def commit(
        self,
        requests: Sequence[TransitionRequest],
        *,
        externally_committed_arms: frozenset[str] = frozenset(),
    ) -> TransitionCommitResult:
        """Commit one resolved batch, allowing recovery to own selected cursors.

        An arm in recovery reentry resets its belief and execution cursor via
        ``ReentrySelector.apply``.  The transaction coordinator still decides
        whether its boundary is jointly permitted, but must not also perform
        the normal entry-state cursor write for that arm.
        """

        tick, candidates, held_groups = self._resolve(requests)
        if self._last_tick is not None and tick <= self._last_tick:
            raise ValueError("事务协调器每个递增控制周期只能提交一次")
        unknown_external = externally_committed_arms.difference(
            request.arm_id for request in candidates
        )
        if unknown_external:
            raise ValueError("外部提交机械臂不在本次放行事务中")

        # Validate the complete normal subset before changing any cursor.  The
        # external subset has already passed reentry candidate validation.
        for request in candidates:
            if request.arm_id not in externally_committed_arms:
                self.controllers[request.arm_id].validate_boundary_transition(request)
                self.belief_updaters[request.arm_id].validate_boundary_transition(
                    request.boundary_id,
                    request.target_state,
                )
        for request in candidates:
            if request.arm_id not in externally_committed_arms:
                self.controllers[request.arm_id].commit_boundary_transition(request)
                self.belief_updaters[request.arm_id].commit_boundary_transition(
                    request.boundary_id,
                    request.target_state,
                )
        self._last_tick = tick
        return TransitionCommitResult(
            tick=tick,
            requests=tuple(requests),
            committed=tuple(candidates),
            held_transaction_groups=tuple(held_groups),
        )

    def preview(
        self,
        requests: Sequence[TransitionRequest],
    ) -> TransitionCommitResult:
        """Resolve independent/joint permissions without changing any cursor."""

        tick, candidates, held_groups = self._resolve(requests)
        return TransitionCommitResult(
            tick=tick,
            requests=tuple(requests),
            committed=tuple(candidates),
            held_transaction_groups=tuple(held_groups),
        )


__all__ = ["TransitionCommitResult", "TransitionTransactionCoordinator"]
