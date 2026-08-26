"""Runtime records and calibrated parameters for phase-four boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .boundary_model import BoundaryId
from .frame_roles import RelationVerificationRequest
from .state_index import StateId


class ConditionKind(str, Enum):
    LOCAL_PROGRESS = "local_progress"
    LOCAL_GOAL = "local_goal"
    OWN_RELATION = "own_relation"
    GUARD_RELATION = "guard_relation"
    GUARD_SCENE = "guard_scene"


@dataclass(frozen=True, order=True)
class ConditionId:
    kind: ConditionKind
    arm_id: str
    token: str

    def __post_init__(self) -> None:
        if not self.arm_id or not self.token:
            raise ValueError("边界条件标识需要非空机械臂和 token")


@dataclass(frozen=True)
class ConditionResult:
    condition_id: ConditionId
    compatibility: float
    reliability: float
    threshold: float | None
    observed: bool
    stable: bool
    raw_satisfied: bool
    consecutive_cycles: int
    required_cycles: int
    satisfied: bool
    reason: str

    def __post_init__(self) -> None:
        for name in ("compatibility", "reliability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("条件阈值必须位于 [0,1]")
        if self.consecutive_cycles < 0 or self.required_cycles < 1:
            raise ValueError("连续周期计数无效")
        if self.satisfied != (
            self.raw_satisfied and self.consecutive_cycles >= self.required_cycles
        ):
            raise ValueError("条件稳定满足状态与连续周期不一致")


@dataclass(frozen=True)
class LocalCompletionResult:
    boundary_id: BoundaryId
    end_probability: float
    goal_compatibility: float
    own_relation_compatibility: float
    score: float
    threshold: float
    raw_satisfied: bool
    consecutive_cycles: int
    required_cycles: int
    done: bool
    evidence_available: bool

    def __post_init__(self) -> None:
        for name in (
            "end_probability",
            "goal_compatibility",
            "own_relation_compatibility",
            "score",
            "threshold",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]")
        if self.consecutive_cycles < 0 or self.required_cycles < 1:
            raise ValueError("本地完成连续周期无效")
        if self.done != (
            self.raw_satisfied and self.consecutive_cycles >= self.required_cycles
        ):
            raise ValueError("本地完成状态与连续周期不一致")


@dataclass(frozen=True)
class TransitionRequest:
    tick: int
    arm_id: str
    boundary_id: BoundaryId
    permitted: bool
    source_state: StateId
    target_state: StateId
    local_done: bool
    condition_results: dict[ConditionId, ConditionResult]
    verification_requests: tuple[RelationVerificationRequest, ...] = ()
    transaction_group: str | None = None

    def __post_init__(self) -> None:
        if self.tick < 0 or not self.arm_id:
            raise ValueError("转换请求的 tick 和 arm_id 无效")
        if self.arm_id != self.boundary_id.arm_id:
            raise ValueError("转换请求机械臂与边界不一致")
        if self.source_state.skill_index != self.boundary_id.source_skill:
            raise ValueError("转换源状态不在边界源技能")
        if self.target_state.skill_index != self.boundary_id.target_skill:
            raise ValueError("转换目标状态不在边界目标技能")
        if self.permitted and not self.local_done:
            raise ValueError("本地技能未完成时不能生成跨界许可")
        if self.permitted and any(
            not result.satisfied
            for condition_id, result in self.condition_results.items()
            if condition_id.kind
            in {ConditionKind.GUARD_RELATION, ConditionKind.GUARD_SCENE}
        ):
            raise ValueError("守卫必要条件未满足时不能生成跨界许可")


@dataclass(frozen=True)
class BoundaryCalibration:
    local_score_threshold: float
    confirmation_cycles: int
    relation_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.local_score_threshold <= 1.0:
            raise ValueError("本地完成阈值必须位于 [0,1]")
        if self.confirmation_cycles < 1:
            raise ValueError("连续确认周期必须为正整数")
        if any(not 0.5 < value <= 1.0 for value in self.relation_thresholds.values()):
            raise ValueError("关系守卫阈值必须位于 (0.5,1]")


@dataclass(frozen=True)
class BoundaryRuntimeConfig:
    calibrations: dict[str, BoundaryCalibration]
    default_relation_probability: float = 0.70
    minimum_tracking_reliability: float = 0.25
    minimum_scene_reliability: float = 0.25
    minimum_information_weight: float = 0.10

    def __post_init__(self) -> None:
        if not 0.5 < self.default_relation_probability <= 1.0:
            raise ValueError("默认关系阈值必须位于 (0.5,1]")
        for value in (
            self.minimum_tracking_reliability,
            self.minimum_scene_reliability,
            self.minimum_information_weight,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("阶段四可靠性参数必须位于 [0,1]")

    def calibration_for(self, boundary_id: BoundaryId) -> BoundaryCalibration:
        try:
            return self.calibrations[boundary_id.token]
        except KeyError as exc:
            raise KeyError(f"边界 {boundary_id.token} 尚未完成正常数据标定") from exc

    def relation_threshold(self, boundary_id: BoundaryId, condition: str) -> float:
        calibration = self.calibration_for(boundary_id)
        return float(
            calibration.relation_thresholds.get(
                condition, self.default_relation_probability
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_relation_probability": self.default_relation_probability,
            "minimum_tracking_reliability": self.minimum_tracking_reliability,
            "minimum_scene_reliability": self.minimum_scene_reliability,
            "minimum_information_weight": self.minimum_information_weight,
            "calibrations": {
                token: {
                    "local_score_threshold": value.local_score_threshold,
                    "confirmation_cycles": value.confirmation_cycles,
                    "relation_thresholds": dict(value.relation_thresholds),
                }
                for token, value in sorted(self.calibrations.items())
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BoundaryRuntimeConfig:
        known = {
            "default_relation_probability",
            "minimum_tracking_reliability",
            "minimum_scene_reliability",
            "minimum_information_weight",
            "calibrations",
        }
        unknown = set(value).difference(known)
        if unknown:
            raise ValueError(f"阶段四配置包含未知字段：{sorted(unknown)}")
        raw_calibrations = value.get("calibrations", {})
        if not isinstance(raw_calibrations, Mapping):
            raise TypeError("阶段四 calibrations 必须为对象")
        calibrations = {}
        for token, raw in raw_calibrations.items():
            if not isinstance(token, str) or not isinstance(raw, Mapping):
                raise TypeError("阶段四边界标定项格式无效")
            calibrations[token] = BoundaryCalibration(
                local_score_threshold=float(raw["local_score_threshold"]),
                confirmation_cycles=int(raw["confirmation_cycles"]),
                relation_thresholds={
                    str(name): float(threshold)
                    for name, threshold in dict(
                        raw.get("relation_thresholds", {})
                    ).items()
                },
            )
        return cls(
            calibrations=calibrations,
            default_relation_probability=float(
                value.get("default_relation_probability", 0.70)
            ),
            minimum_tracking_reliability=float(
                value.get("minimum_tracking_reliability", 0.25)
            ),
            minimum_scene_reliability=float(
                value.get("minimum_scene_reliability", 0.25)
            ),
            minimum_information_weight=float(
                value.get("minimum_information_weight", 0.10)
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> BoundaryRuntimeConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("阶段四配置文件根节点必须为对象")
        return cls.from_mapping(value)


__all__ = [
    "BoundaryCalibration",
    "BoundaryRuntimeConfig",
    "ConditionId",
    "ConditionKind",
    "ConditionResult",
    "LocalCompletionResult",
    "TransitionRequest",
]
