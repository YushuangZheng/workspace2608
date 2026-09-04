"""Causal runtime adapter for the M4 failure-supervised monitor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .conformal import TimeVaryingConformalBand
from .fail_detect import ArrayObservationEncoder, FeatureEncoder


class StatefulProbabilityModel(Protocol):
    def reset(self) -> None: ...

    def __call__(self, features: np.ndarray) -> float: ...


@dataclass(frozen=True)
class FailureSupervisedMonitorConfig:
    persistence: int = 1
    score_name: str = "violation_probability"

    def __post_init__(self) -> None:
        if (
            isinstance(self.persistence, bool)
            or not isinstance(self.persistence, int)
            or self.persistence <= 0
        ):
            raise ValueError("persistence must be a positive integer")
        if not self.score_name:
            raise ValueError("score_name must be non-empty")


class FailureSupervisedMonitor:
    """Apply a causal stateful classifier and an A-frozen threshold band."""

    def __init__(
        self,
        probability_model: StatefulProbabilityModel,
        conformal_band: TimeVaryingConformalBand,
        *,
        feature_encoder: FeatureEncoder | None = None,
        config: FailureSupervisedMonitorConfig = FailureSupervisedMonitorConfig(),
    ) -> None:
        self.probability_model = probability_model
        self.conformal_band = conformal_band
        self.feature_encoder = feature_encoder or ArrayObservationEncoder()
        self.config = config
        self._feature_shape: tuple[int, ...] | None = None
        self._score_index = -1
        self._last_probability: float | None = None
        self._last_threshold: float | None = None
        self._consecutive_exceedances = 0
        self._alarm = False
        self._first_alarm_index: int | None = None

    @property
    def ready(self) -> bool:
        return self._last_probability is not None

    def reset(self, episode_context: Mapping[str, Any]) -> None:
        if not isinstance(episode_context, Mapping):
            raise TypeError("episode_context must be a mapping")
        self.probability_model.reset()
        self._feature_shape = None
        self._score_index = -1
        self._last_probability = None
        self._last_threshold = None
        self._consecutive_exceedances = 0
        self._alarm = False
        self._first_alarm_index = None

    def observe(self, observation: Any, action: Any, policy_state: Any) -> None:
        features = np.asarray(
            self.feature_encoder(observation, action, policy_state),
            dtype=np.float64,
        )
        if features.ndim != 1 or not len(features) or not np.all(np.isfinite(features)):
            raise ValueError("feature encoder must return one finite vector")
        if self._feature_shape is None:
            self._feature_shape = features.shape
        elif features.shape != self._feature_shape:
            raise ValueError("supervised-monitor feature shape changed within an episode")

        next_score_index = self._score_index + 1
        threshold = self.conformal_band.threshold(next_score_index)
        probability = float(self.probability_model(features.copy()))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("supervised model must return a probability in [0, 1]")
        exceeds = probability > threshold
        self._score_index = next_score_index
        self._last_probability = probability
        self._last_threshold = threshold
        self._consecutive_exceedances = self._consecutive_exceedances + 1 if exceeds else 0
        self._alarm = self._consecutive_exceedances >= self.config.persistence
        if self._alarm and self._first_alarm_index is None:
            self._first_alarm_index = self._score_index

    def score(self) -> dict[str, float]:
        base = {
            "ready": float(self.ready),
            "score_index": float(self._score_index),
            "consecutive_exceedances": float(self._consecutive_exceedances),
            "alarm": float(self._alarm),
            "first_alarm_index": float(
                -1 if self._first_alarm_index is None else self._first_alarm_index
            ),
        }
        if not self.ready:
            return base
        assert self._last_probability is not None
        assert self._last_threshold is not None
        return {
            **base,
            self.config.score_name: self._last_probability,
            "threshold": self._last_threshold,
            "margin": self._last_probability - self._last_threshold,
        }

    def alarm(self) -> bool:
        return bool(self._alarm)


class TorchGRUProbabilityScorer:
    """Stateful one-step wrapper for a unidirectional PyTorch GRU classifier."""

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchGRUProbabilityScorer requires PyTorch") from exc
        self._torch = torch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self._hidden: Any = None

    def reset(self) -> None:
        self._hidden = None

    def __call__(self, features: np.ndarray) -> float:
        torch = self._torch
        vector = np.asarray(features, dtype=np.float32)
        if vector.ndim != 1 or not len(vector) or not np.all(np.isfinite(vector)):
            raise ValueError("GRU input must be one finite feature vector")
        sequence = torch.as_tensor(vector[None, None, :], device=self.device)
        with torch.no_grad():
            logits, hidden = self.model(sequence, self._hidden)
            if logits.shape != (1, 1):
                raise ValueError("GRU classifier must return logits with shape [B, T]")
            self._hidden = hidden.detach()
            probability = torch.sigmoid(logits[0, 0])
        return float(probability.detach().cpu().item())


__all__ = [
    "FailureSupervisedMonitor",
    "FailureSupervisedMonitorConfig",
    "StatefulProbabilityModel",
    "TorchGRUProbabilityScorer",
]
