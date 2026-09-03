"""Offline skill-boundary models shared by entry guards and recovery reentry."""

from __future__ import annotations

from dataclasses import dataclass, field

from .scene_factors import FactorDistribution, FactorId
from .state_index import StateId


@dataclass(frozen=True, order=True)
class BoundaryId:
    arm_id: str
    source_skill: int
    target_skill: int

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("BoundaryId 需要非空 arm_id")
        if self.source_skill < 0 or self.target_skill != self.source_skill + 1:
            raise ValueError("BoundaryId 必须连接相邻技能")

    @property
    def token(self) -> str:
        return f"{self.arm_id}:{self.source_skill}->{self.target_skill}"


@dataclass(frozen=True)
class LocalCompletionModel:
    terminal_states: tuple[StateId, ...]
    goal_distributions: dict[str, FactorDistribution]
    minimum_goal_log_likelihood: dict[str, float]
    own_relation_conditions: dict[str, RelationGuardDistribution] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RelationGuardDistribution:
    external: float
    linked: float
    required_state: str

    def __post_init__(self) -> None:
        if self.required_state not in {"external", "linked"}:
            raise ValueError("边界关系目标必须为 external 或 linked")
        if self.external < 0.0 or self.linked < 0.0:
            raise ValueError("边界关系支持概率不能为负")
        if abs(self.external + self.linked - 1.0) > 1.0e-9:
            raise ValueError("边界关系支持概率必须归一化")


@dataclass(frozen=True)
class ReliabilityStatistics:
    observed_fraction: float
    stable_fraction: float


@dataclass(frozen=True)
class BoundaryModel:
    boundary_id: BoundaryId
    source_skill: int
    target_skill: int
    terminal_window: tuple[StateId, ...]
    local_completion_model: LocalCompletionModel
    relation_conditions: dict[str, RelationGuardDistribution]
    scene_conditions: dict[FactorId, FactorDistribution]
    scene_condition_thresholds: dict[FactorId, float]
    condition_reliability: dict[str, ReliabilityStatistics]
    affected_arms: tuple[str, ...]
    transaction_group: str | None = None

    def __post_init__(self) -> None:
        if self.source_skill != self.boundary_id.source_skill:
            raise ValueError("边界 source_skill 与 BoundaryId 不一致")
        if self.target_skill != self.boundary_id.target_skill:
            raise ValueError("边界 target_skill 与 BoundaryId 不一致")
        if not self.terminal_window:
            raise ValueError("边界终止窗口不能为空")
        if not self.affected_arms:
            raise ValueError("边界至少影响一只机械臂")
        if set(self.scene_condition_thresholds) != set(self.scene_conditions):
            raise ValueError("边界场景条件及其兼容度阈值必须一一对应")
        overlap = set(self.local_completion_model.own_relation_conditions).intersection(
            self.relation_conditions
        )
        if overlap:
            raise ValueError(
                f"本地完成关系不能重复保存为边界关系条件：{sorted(overlap)}"
            )
        expected_reliability = (
            set(self.local_completion_model.own_relation_conditions)
            .union(self.relation_conditions)
            .union(factor_id.token for factor_id in self.scene_conditions)
        )
        if set(self.condition_reliability) != expected_reliability:
            raise ValueError("边界每个已保存条件必须且只能有一项可靠性统计")
        if any(
            not 0.0 <= value <= 1.0
            for value in self.scene_condition_thresholds.values()
        ):
            raise ValueError("边界场景兼容度阈值必须位于 [0,1]")


__all__ = [
    "BoundaryId",
    "BoundaryModel",
    "LocalCompletionModel",
    "RelationGuardDistribution",
    "ReliabilityStatistics",
]
