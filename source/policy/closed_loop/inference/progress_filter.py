"""Progress posterior and interpretable alignment status."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from .state_evaluator import CandidateScore
from ..model.state_index import StateId
from ..model.task_model import ClosedLoopTaskModel


class ProgressStatus(str, Enum):
    ALIGNED = "aligned"
    FORWARD_REALIGNMENT = "forward_realignment"
    BACKWARD_REALIGNMENT = "backward_realignment"
    LOW_CONFIDENCE = "low_confidence"
    NO_PLAUSIBLE_STATE = "no_plausible_state"


@dataclass(frozen=True)
class ProgressEstimate:
    prior: dict[StateId, float]
    posterior: dict[StateId, float]
    nominal_state: StateId
    estimated_state: StateId
    confidence: float
    entropy: float
    best_explanation_score: float
    status: ProgressStatus


@dataclass(frozen=True)
class ProgressFilterConfig:
    minimum_confidence: float = 0.55
    maximum_normalized_entropy: float = 0.80
    # Calibrated on held-out, normal RLBench closed-loop control rather than
    # on the five demonstrations used to fit the state models.  The lower
    # value accounts for normal contact/servo offsets while the score remains
    # the same joint trajectory/scene/relation explanation from phase two.
    minimum_explanation_score: float = 1.0e-5
    probability_floor: float = 1.0e-300

    def __post_init__(self) -> None:
        for value in (
            self.minimum_confidence,
            self.maximum_normalized_entropy,
            self.minimum_explanation_score,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("进度状态阈值必须位于 [0,1]")
        if self.probability_floor <= 0.0:
            raise ValueError("进度概率下限必须为正数")


class ProgressFilter:
    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: ProgressFilterConfig = ProgressFilterConfig(),
    ) -> None:
        self.task_model = task_model
        self.config = config
        self._global_index = {
            state: index for index, state in enumerate(sorted(task_model.states))
        }

    def update(
        self,
        prior: Mapping[StateId, float],
        scores: Mapping[StateId, CandidateScore],
        nominal_state: StateId,
    ) -> ProgressEstimate:
        if not prior or set(prior) != set(scores):
            raise ValueError("进度先验与候选评分必须覆盖同一非空状态集合")
        total = float(sum(prior.values()))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("进度先验必须具有正的有限总和")
        normalized_prior = {
            state: float(value / total) for state, value in prior.items()
        }

        states = tuple(sorted(prior, key=self._global_index.__getitem__))
        best_explanation = max(
            score.normalized_explanation_score for score in scores.values()
        )
        if best_explanation < self.config.minimum_explanation_score:
            posterior = dict(normalized_prior)
            estimated = max(
                states,
                key=lambda state: (posterior[state], -self._global_index[state]),
            )
            probabilities = np.asarray([posterior[state] for state in states])
            confidence = posterior[estimated]
            entropy = -float(
                np.sum(
                    probabilities
                    * np.log(np.maximum(probabilities, self.config.probability_floor))
                )
            )
            return ProgressEstimate(
                prior=normalized_prior,
                posterior=posterior,
                nominal_state=nominal_state,
                estimated_state=estimated,
                confidence=confidence,
                entropy=entropy,
                best_explanation_score=best_explanation,
                status=ProgressStatus.NO_PLAUSIBLE_STATE,
            )

        log_values = np.asarray(
            [
                math.log(max(normalized_prior[state], self.config.probability_floor))
                + scores[state].explanation_log_score
                for state in states
            ],
            dtype=np.float64,
        )
        maximum = float(np.max(log_values))
        probabilities = np.exp(log_values - maximum)
        probabilities /= np.sum(probabilities)
        posterior = {
            state: float(probability)
            for state, probability in zip(states, probabilities, strict=True)
        }
        estimated = max(
            states,
            key=lambda state: (posterior[state], -self._global_index[state]),
        )
        confidence = posterior[estimated]
        entropy = -float(
            np.sum(
                probabilities
                * np.log(np.maximum(probabilities, self.config.probability_floor))
            )
        )
        normalized_entropy = (
            0.0 if len(states) == 1 else entropy / math.log(float(len(states)))
        )
        if (
            confidence < self.config.minimum_confidence
            or normalized_entropy > self.config.maximum_normalized_entropy
        ):
            status = ProgressStatus.LOW_CONFIDENCE
        elif self._global_index[estimated] > self._global_index[nominal_state]:
            status = ProgressStatus.FORWARD_REALIGNMENT
        elif self._global_index[estimated] < self._global_index[nominal_state]:
            status = ProgressStatus.BACKWARD_REALIGNMENT
        else:
            status = ProgressStatus.ALIGNED
        return ProgressEstimate(
            prior=normalized_prior,
            posterior=posterior,
            nominal_state=nominal_state,
            estimated_state=estimated,
            confidence=confidence,
            entropy=entropy,
            best_explanation_score=best_explanation,
            status=status,
        )


__all__ = [
    "ProgressEstimate",
    "ProgressFilter",
    "ProgressFilterConfig",
    "ProgressStatus",
]
