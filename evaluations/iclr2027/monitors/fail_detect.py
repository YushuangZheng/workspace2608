"""FAIL-Detect logpZO runtime adapter with no benchmark dependency."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .conformal import TimeVaryingConformalBand


class ScalarScoreModel(Protocol):
    def __call__(self, observation_window: np.ndarray) -> float: ...


class FeatureEncoder(Protocol):
    def __call__(self, observation: Any, action: Any, policy_state: Any) -> np.ndarray: ...


def prepare_logpzo_input(
    observation_window: np.ndarray,
    *,
    input_dim: int,
) -> np.ndarray:
    """Match FAIL-Detect's public ``adjust_xshape`` transformation.

    The exported policy feature is flattened, zero-padded to a multiple of the
    task action dimension, and then padded to a temporal length divisible by
    four for the official 1-D UNet.
    """

    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")
    observation = np.asarray(observation_window, dtype=np.float32)
    if observation.ndim < 2 or not observation.size or not np.all(np.isfinite(observation)):
        raise ValueError("logpZO input must be a finite observation window")

    flattened = observation.reshape(-1)
    remainder = len(flattened) % input_dim
    if remainder:
        flattened = np.pad(flattened, (0, input_dim - remainder))
    temporal_length = len(flattened) // input_dim
    if temporal_length % 4:
        extra_steps = 4 - temporal_length % 4
        flattened = np.pad(flattened, (0, extra_steps * input_dim))
    return flattened.reshape(-1, input_dim)


class ArrayObservationEncoder:
    """Accept an already frozen observation feature vector.

    FAIL-Detect logpZO is observation-only.  The action and policy-state
    arguments remain in the shared monitor contract but are intentionally not
    consumed here.  A later A-provided schema adapter should produce this
    vector and must exclude audit labels and future information.
    """

    def __call__(self, observation: Any, action: Any, policy_state: Any) -> np.ndarray:
        del action, policy_state
        vector = np.asarray(observation, dtype=np.float64)
        if vector.ndim != 1 or not len(vector) or not np.all(np.isfinite(vector)):
            raise ValueError("FAIL-Detect observation features must be a finite vector")
        return vector.copy()


@dataclass(frozen=True)
class FailDetectMonitorConfig:
    observation_window: int = 1
    persistence: int = 1
    score_name: str = "logpzo"

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_window, bool)
            or not isinstance(self.observation_window, int)
            or self.observation_window <= 0
        ):
            raise ValueError("observation_window must be a positive integer")
        if (
            isinstance(self.persistence, bool)
            or not isinstance(self.persistence, int)
            or self.persistence <= 0
        ):
            raise ValueError("persistence must be a positive integer")
        if not self.score_name:
            raise ValueError("score_name must be non-empty")


class FailDetectMonitor:
    """Causal logpZO scorer followed by a frozen conformal upper band."""

    def __init__(
        self,
        score_model: ScalarScoreModel,
        conformal_band: TimeVaryingConformalBand,
        *,
        feature_encoder: FeatureEncoder | None = None,
        config: FailDetectMonitorConfig = FailDetectMonitorConfig(),
    ) -> None:
        self.score_model = score_model
        self.conformal_band = conformal_band
        self.feature_encoder = feature_encoder or ArrayObservationEncoder()
        self.config = config
        self._features: deque[np.ndarray] = deque(maxlen=config.observation_window)
        self._feature_shape: tuple[int, ...] | None = None
        self._episode_context: dict[str, Any] = {}
        self._score_index = -1
        self._last_score: float | None = None
        self._last_threshold: float | None = None
        self._consecutive_exceedances = 0
        self._alarm = False
        self._first_alarm_index: int | None = None

    @property
    def ready(self) -> bool:
        return self._last_score is not None

    def reset(self, episode_context: Mapping[str, Any]) -> None:
        if not isinstance(episode_context, Mapping):
            raise TypeError("episode_context must be a mapping")
        self._features.clear()
        self._feature_shape = None
        self._episode_context = dict(episode_context)
        self._score_index = -1
        self._last_score = None
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
            raise ValueError("FAIL-Detect feature shape changed within an episode")
        self._features.append(features.copy())
        if len(self._features) < self.config.observation_window:
            return

        next_score_index = self._score_index + 1
        threshold = self.conformal_band.threshold(next_score_index)
        score = float(self.score_model(np.stack(tuple(self._features), axis=0)))
        if not np.isfinite(score):
            raise ValueError("FAIL-Detect score model returned a non-finite value")
        exceeds = score > threshold
        self._score_index = next_score_index
        self._last_score = score
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
        assert self._last_score is not None
        assert self._last_threshold is not None
        return {
            **base,
            self.config.score_name: self._last_score,
            "threshold": self._last_threshold,
            "margin": self._last_score - self._last_threshold,
        }

    def alarm(self) -> bool:
        return bool(self._alarm)


class TorchLogpZOScorer:
    """Evaluate the official one-step latent-noise squared-norm score.

    The caller supplies a trained velocity model with signature
    ``model(observation_batch, zero_timestep_batch)``.  PyTorch is imported
    lazily so the repository's NumPy-only core environment remains unchanged.
    """

    def __init__(self, model: Any, *, input_dim: int, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchLogpZOScorer requires PyTorch") from exc
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self._torch = torch
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def __call__(self, observation_window: np.ndarray) -> float:
        torch = self._torch
        observation = prepare_logpzo_input(
            observation_window,
            input_dim=self.input_dim,
        )
        batch = torch.as_tensor(observation[None, ...], device=self.device)
        timestep = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            velocity = self.model(batch, timestep)
            if velocity.shape != batch.shape:
                raise ValueError("logpZO velocity model output shape does not match input")
            latent_noise = batch + velocity
            score = latent_noise.reshape(1, -1).square().sum(dim=1)[0]
        return float(score.detach().cpu().item())


__all__ = [
    "ArrayObservationEncoder",
    "FailDetectMonitor",
    "FailDetectMonitorConfig",
    "FeatureEncoder",
    "ScalarScoreModel",
    "TorchLogpZOScorer",
    "prepare_logpzo_input",
]
