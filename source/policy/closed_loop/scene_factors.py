"""Sparse entity-configuration distributions for progress and boundary models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..dynamac import _fit_pose_sequence, pose_log_nearest

Array = np.ndarray
FactorKind = Literal["node", "edge"]
FactorSpace = Literal["se3", "euclidean"]


@dataclass(frozen=True, order=True)
class FactorId:
    """One internal entity field or one directed relative-pose edge."""

    kind: FactorKind
    source: str
    target: str | None = None
    feature: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"node", "edge"}:
            raise ValueError("场景因子类型必须为 node 或 edge")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("场景因子 source 必须为非空字符串")
        if self.kind == "node":
            if self.target is not None:
                raise ValueError("node 因子不能包含 target")
            if not isinstance(self.feature, str) or not self.feature:
                raise ValueError("node 因子必须标识实体内部构型字段")
        elif (
            not isinstance(self.target, str)
            or not self.target
            or self.target == self.source
            or self.feature is not None
        ):
            raise ValueError("edge 因子必须连接两个不同实体且不含 feature")

    @property
    def token(self) -> str:
        return (
            f"node:{self.source}#{self.feature}"
            if self.kind == "node"
            else f"edge:{self.source}->{self.target}"
        )

    @property
    def entities(self) -> tuple[str, ...]:
        return (self.source,) if self.target is None else (self.source, self.target)

    @classmethod
    def from_token(cls, value: str) -> FactorId:
        if value.startswith("node:") and "#" in value:
            source, feature = value.removeprefix("node:").rsplit("#", 1)
            return cls("node", source, feature=feature)
        if value.startswith("edge:") and "->" in value:
            source, target = value.removeprefix("edge:").split("->", 1)
            return cls("edge", source, target)
        raise ValueError(f"未知场景因子标识：{value}")


@dataclass(frozen=True)
class FactorDistribution:
    """A pose/Euclidean Gaussian plus cross-demonstration audit statistics."""

    mean: Array
    covariance: Array
    sample_count: int
    space: FactorSpace = "se3"
    observability: float = 1.0
    stable_fraction: float = 1.0
    loo_gain: float = 0.0
    loo_accuracy: float = 0.0
    neighborhood_radius: int = 0

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.space == "se3":
            if mean.shape != (7,) or covariance.shape != (6, 6):
                raise ValueError("SE(3) 场景因子必须使用 [7] 均值和 [6,6] 协方差")
        elif self.space == "euclidean":
            if (
                mean.ndim != 1
                or len(mean) == 0
                or covariance.shape != (len(mean), len(mean))
            ):
                raise ValueError("欧氏场景因子必须使用 [D] 均值和 [D,D] 协方差")
        else:
            raise ValueError("场景因子空间必须为 se3 或 euclidean")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("场景因子分布包含非有限值")
        if self.sample_count < 1:
            raise ValueError("场景因子分布至少需要一个样本")
        if not 0.0 <= self.observability <= 1.0:
            raise ValueError("场景因子可观测率必须位于 [0,1]")
        if not 0.0 <= self.stable_fraction <= 1.0:
            raise ValueError("场景因子稳定支持率必须位于 [0,1]")
        if not 0.0 <= self.loo_accuracy <= 1.0:
            raise ValueError("场景因子留一正增益率必须位于 [0,1]")
        if self.neighborhood_radius not in {0, 1, 2}:
            raise ValueError("场景因子邻域半径必须为 0、1 或 2")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "covariance", covariance.copy())

    def log_likelihood(self, value: Array) -> float:
        current = np.asarray(value, dtype=np.float64)
        residual = (
            pose_log_nearest(self.mean, current)
            if self.space == "se3"
            else current - self.mean
        )
        if residual.shape != (self.covariance.shape[0],):
            raise ValueError("场景因子观测维数与分布不一致")
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0.0:
            raise RuntimeError("场景因子协方差不是正定矩阵")
        try:
            precision_residual = np.linalg.solve(self.covariance, residual)
        except np.linalg.LinAlgError:
            precision_residual = np.linalg.solve(
                self.covariance + np.eye(len(residual)) * 1.0e-8,
                residual,
            )
        dimension = len(residual)
        return float(
            -0.5
            * (
                residual @ precision_residual
                + logdet
                + dimension * math.log(2.0 * math.pi)
            )
        )

    def mahalanobis_squared(
        self,
        value: Array,
        observation_covariance: Array | None = None,
    ) -> float:
        """Return the tangent/Euclidean squared distance to this distribution."""

        current = np.asarray(value, dtype=np.float64)
        residual = (
            pose_log_nearest(self.mean, current)
            if self.space == "se3"
            else current - self.mean
        )
        covariance = self.covariance.copy()
        if observation_covariance is not None:
            observed = np.asarray(observation_covariance, dtype=np.float64)
            if observed.shape != covariance.shape:
                raise ValueError("场景因子观测协方差维数不一致")
            covariance += observed
        covariance += np.eye(len(residual), dtype=np.float64) * 1.0e-12
        try:
            solved = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(covariance) @ residual
        return max(0.0, float(residual @ solved))

    def compatibility(
        self,
        value: Array,
        observation_covariance: Array | None = None,
    ) -> float:
        """Map Mahalanobis support to the common ``(0, 1]`` guard scale."""

        return float(
            math.exp(
                -0.5
                * min(
                    self.mahalanobis_squared(value, observation_covariance),
                    1500.0,
                )
            )
        )


def fit_factor_distribution(
    samples: Array,
    *,
    position_variance_floor: float,
    rotation_variance_floor: float,
    covariance_estimation_method: str,
    space: FactorSpace = "se3",
    observability: float = 1.0,
    stable_fraction: float = 1.0,
    loo_gain: float = 0.0,
    loo_accuracy: float = 0.0,
    neighborhood_radius: int = 0,
) -> FactorDistribution:
    """Fit a factor while reusing DynaMAC's manifold fit for SE(3) values."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("场景因子样本必须为非空 [N,D] 数组")
    if space == "se3":
        if values.shape[1] != 7:
            raise ValueError("SE(3) 场景因子样本必须为 [N,7]")
        mean, covariance = _fit_pose_sequence(
            values[:, None, :],
            position_variance_floor,
            rotation_variance_floor,
            covariance_estimation_method=covariance_estimation_method,
        )
        fitted_mean = mean[0]
        fitted_covariance = covariance[0]
    elif space == "euclidean":
        if values.shape[1] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("欧氏场景因子样本必须为有限 [N,D] 数组")
        fitted_mean = np.mean(values, axis=0)
        residual = values - fitted_mean
        denominator = max(1, len(values) - 1)
        fitted_covariance = residual.T @ residual / denominator
        fitted_covariance += np.eye(values.shape[1]) * position_variance_floor
    else:
        raise ValueError("场景因子空间必须为 se3 或 euclidean")
    return FactorDistribution(
        mean=fitted_mean,
        covariance=fitted_covariance,
        sample_count=len(values),
        space=space,
        observability=float(observability),
        stable_fraction=float(stable_fraction),
        loo_gain=float(loo_gain),
        loo_accuracy=float(loo_accuracy),
        neighborhood_radius=int(neighborhood_radius),
    )


__all__ = [
    "FactorDistribution",
    "FactorId",
    "FactorKind",
    "FactorSpace",
    "fit_factor_distribution",
]
