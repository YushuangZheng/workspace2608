"""Active-stream trajectory deviation with DynaMAC PoE aggregation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from evaluations.iclr2027.interfaces.runtime_monitor import (
    EpisodeContext,
    RuntimeMonitor,
)
from integrations.rlbench.rlbench_dynamac.core.runtime import xyzw_to_wxyz


def _normalized_quaternion(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        raise ValueError("quaternion norm must be nonzero")
    return value / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _pose_log_nearest(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Dependency-free copy of the frozen TAPAS R3 x S3 residual convention."""

    reference = np.asarray(base, dtype=np.float64)
    aligned = np.asarray(point, dtype=np.float64).copy()
    if reference.shape != (7,) or aligned.shape != (7,):
        raise ValueError("pose residual requires two [7] poses")
    base_q = _normalized_quaternion(reference[3:7])
    point_q = _normalized_quaternion(aligned[3:7])
    if float(np.dot(base_q, point_q)) < 0.0:
        point_q *= -1.0
    conjugate = base_q.copy()
    conjugate[1:] *= -1.0
    relative = _normalized_quaternion(_quaternion_multiply(conjugate, point_q))
    vector_norm = float(np.linalg.norm(relative[1:]))
    if abs(float(relative[0]) - 1.0) <= 1.0e-6 or vector_norm <= np.finfo(float).eps:
        rotation = np.zeros(3, dtype=np.float64)
    else:
        rotation = relative[1:] * (
            math.acos(float(np.clip(relative[0], -1.0, 1.0))) / vector_norm
        )
    return np.concatenate((aligned[:3] - reference[:3], rotation))


class TrajectoryLikelihoodMonitor(RuntimeMonitor):
    """Score the current EE against the emitted active-stream marginals.

    The decision statistic is the PoE-weighted, dimension-normalized squared
    Mahalanobis deviation, ``0.5 d_M^2 / 6``.  The complete Gaussian NLL and
    log determinant remain diagnostic fields, so narrow covariances cannot
    silently change the alarm scale through a density-normalizer offset.
    """

    def __init__(
        self,
        *,
        threshold: float | None,
        persistence_cycles: int,
        covariance_regularization: float = 1.0e-12,
    ) -> None:
        if threshold is not None and not math.isfinite(float(threshold)):
            raise ValueError("trajectory threshold must be finite or null")
        if persistence_cycles < 1 or covariance_regularization <= 0.0:
            raise ValueError("invalid trajectory monitor configuration")
        self.threshold = None if threshold is None else float(threshold)
        self.persistence_cycles = int(persistence_cycles)
        self.regularization = float(covariance_regularization)
        self.reset(None)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrajectoryLikelihoodMonitor":
        return cls(
            threshold=value.get("threshold"),
            persistence_cycles=int(value["persistence_cycles"]),
            covariance_regularization=float(
                value.get("covariance_regularization", 1.0e-12)
            ),
        )

    def reset(self, episode_context: EpisodeContext | None) -> None:
        self._scores = {
            "standardized_nll": 0.0,
            "full_nll": 0.0,
            "available_streams": 0.0,
        }
        self._streak = 0
        self._alarm = False

    @staticmethod
    def _arm_metadata(policy_state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        metadata = policy_state.get("stream_metadata", {})
        if not isinstance(metadata, Mapping):
            return {}
        if "active_streams" in metadata:
            return {"single": metadata}
        return {
            str(arm): value
            for arm, value in metadata.items()
            if isinstance(value, Mapping)
        }

    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        arms = observation.get("arms", {})
        weighted_standardized = 0.0
        weighted_full = 0.0
        total_weight = 0.0
        count = 0
        for arm, metadata in self._arm_metadata(policy_state).items():
            if arm not in arms:
                continue
            current = xyzw_to_wxyz(
                np.asarray(arms[arm]["ee_pose_xyzw"], dtype=np.float64)
            )
            active = tuple(metadata.get("active_streams", ()))
            means = metadata.get("marginal_means", {})
            covariances = metadata.get("marginal_covariances", {})
            weights = metadata.get("poe_weights", {})
            for frame in active:
                if frame not in means or frame not in covariances:
                    continue
                weight = float(weights.get(frame, 0.0))
                if not math.isfinite(weight) or weight <= 0.0:
                    continue
                mean = np.asarray(means[frame], dtype=np.float64)
                covariance = np.asarray(covariances[frame], dtype=np.float64)
                if mean.shape != (7,) or covariance.shape != (6, 6):
                    raise ValueError("active-stream marginal has invalid dimensions")
                regularized = covariance + np.eye(6) * self.regularization
                residual = _pose_log_nearest(mean, current)
                try:
                    solved = np.linalg.solve(regularized, residual)
                except np.linalg.LinAlgError:
                    solved = np.linalg.pinv(regularized) @ residual
                mahalanobis = max(0.0, float(residual @ solved))
                sign, logdet = np.linalg.slogdet(regularized)
                if sign <= 0.0:
                    raise ValueError("active-stream covariance must be positive definite")
                standardized = 0.5 * mahalanobis / 6.0
                full = 0.5 * (mahalanobis + logdet + 6.0 * math.log(2.0 * math.pi))
                weighted_standardized += weight * standardized
                weighted_full += weight * full
                total_weight += weight
                count += 1
        available = total_weight > 0.0
        score = weighted_standardized / total_weight if available else 0.0
        full_score = weighted_full / total_weight if available else 0.0
        self._scores = {
            "standardized_nll": float(score),
            "full_nll": float(full_score),
            "available_streams": float(count),
        }
        exceeded = bool(
            available
            and self.threshold is not None
            and score > self.threshold
        )
        self._streak = self._streak + 1 if exceeded else 0
        self._alarm = self._streak >= self.persistence_cycles

    def score(self) -> Mapping[str, float]:
        return dict(self._scores)

    def alarm(self) -> bool:
        return self._alarm

    @property
    def persistence_count(self) -> int:
        return self._streak


__all__ = ["TrajectoryLikelihoodMonitor"]
