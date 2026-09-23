"""Official-style one-sided functional conformal bands for FAIL-Detect."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

BAND_SCHEMA = "fail-detect-functional-cp-v1"


def _score_matrix(value: Any, *, name: str) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float64)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError(f"{name} must have shape [episodes, score_steps]")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} contains non-finite scores")
    return scores


@dataclass(frozen=True)
class TimeVaryingConformalBand:
    """Upper prediction band used by FAIL-Detect.

    ``fit`` implements the public FAIL-Detect repository's upper-band path:
    the mean and T-function modulation use one successful-rollout split, and
    a disjoint split determines the band-width quantile.  Splits are explicit
    so server B never needs access to server A's formal calibration episodes.
    """

    alpha: float
    mean: np.ndarray
    modulation: np.ndarray
    band_width: float
    upper: np.ndarray
    mean_episode_count: int
    width_episode_count: int
    modulation_kind: str = "tfunc"
    quantile_method: str = "numpy_linear"
    schema: str = BAND_SCHEMA

    def __post_init__(self) -> None:
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        mean = np.asarray(self.mean, dtype=np.float64)
        modulation = np.asarray(self.modulation, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if mean.ndim != 1 or not len(mean):
            raise ValueError("mean must be a non-empty score trajectory")
        if modulation.shape != mean.shape or upper.shape != mean.shape:
            raise ValueError("mean, modulation, and upper must have equal shape")
        if not (
            np.all(np.isfinite(mean))
            and np.all(np.isfinite(modulation))
            and np.all(np.isfinite(upper))
            and np.isfinite(self.band_width)
        ):
            raise ValueError("conformal band contains non-finite values")
        if np.any(modulation <= 0.0):
            raise ValueError("conformal modulation must be strictly positive")
        if self.mean_episode_count <= 0 or self.width_episode_count <= 0:
            raise ValueError("conformal calibration splits must be non-empty")
        if self.modulation_kind not in {"constant", "tfunc"}:
            raise ValueError("unsupported modulation kind")
        if self.quantile_method != "numpy_linear":
            raise ValueError("unsupported quantile method")
        if self.schema != BAND_SCHEMA:
            raise ValueError("unsupported conformal band schema")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "modulation", modulation.copy())
        object.__setattr__(self, "band_width", float(self.band_width))
        object.__setattr__(self, "upper", upper.copy())

    @property
    def horizon(self) -> int:
        return int(len(self.upper))

    @classmethod
    def fit(
        cls,
        mean_scores: Any,
        width_scores: Any,
        *,
        alpha: float,
        modulation_kind: str = "tfunc",
        epsilon: float = 1.0e-8,
    ) -> TimeVaryingConformalBand:
        """Fit an upper band from two disjoint successful-rollout splits."""

        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if epsilon <= 0.0 or not np.isfinite(epsilon):
            raise ValueError("epsilon must be finite and positive")
        if modulation_kind not in {"constant", "tfunc"}:
            raise ValueError("modulation_kind must be constant or tfunc")

        training = _score_matrix(mean_scores, name="mean_scores")
        calibration = _score_matrix(width_scores, name="width_scores")
        if training.shape[1] != calibration.shape[1]:
            raise ValueError("conformal score splits must share one horizon")

        mean = np.mean(training, axis=0)
        if modulation_kind == "constant":
            modulation = np.full_like(mean, 1.0 / float(training.shape[1]))
        else:
            absolute_residual = np.abs(training - mean)
            maximum_residual = np.max(absolute_residual, axis=1)
            rank = int(np.ceil((training.shape[0] + 1) * (1.0 - alpha)))
            if rank > training.shape[0]:
                retained = absolute_residual
            else:
                gamma = np.sort(maximum_residual)[rank - 1]
                retained = absolute_residual[maximum_residual <= gamma]
            modulation = np.max(retained, axis=0) + float(epsilon)

        nonconformity = np.max((calibration - mean) / modulation, axis=1)
        band_width = float(np.quantile(nonconformity, 1.0 - alpha))
        upper = mean + band_width * modulation
        return cls(
            alpha=float(alpha),
            mean=mean,
            modulation=modulation,
            band_width=band_width,
            upper=upper,
            mean_episode_count=int(training.shape[0]),
            width_episode_count=int(calibration.shape[0]),
            modulation_kind=modulation_kind,
        )

    def threshold(self, score_index: int) -> float:
        if isinstance(score_index, bool) or not isinstance(score_index, (int, np.integer)):
            raise TypeError("score_index must be an integer")
        index = int(score_index)
        if not 0 <= index < self.horizon:
            raise IndexError("score_index is outside the calibrated horizon")
        return float(self.upper[index])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alpha": self.alpha,
            "modulation_kind": self.modulation_kind,
            "quantile_method": self.quantile_method,
            "mean_episode_count": self.mean_episode_count,
            "width_episode_count": self.width_episode_count,
            "mean": self.mean.tolist(),
            "modulation": self.modulation.tolist(),
            "band_width": self.band_width,
            "upper": self.upper.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TimeVaryingConformalBand:
        expected = {
            "schema",
            "alpha",
            "modulation_kind",
            "quantile_method",
            "mean_episode_count",
            "width_episode_count",
            "mean",
            "modulation",
            "band_width",
            "upper",
        }
        unknown = set(payload).difference(expected)
        missing = expected.difference(payload)
        if unknown or missing:
            raise ValueError(
                f"invalid conformal artifact fields: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**dict(payload))


__all__ = ["BAND_SCHEMA", "TimeVaryingConformalBand"]
