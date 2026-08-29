"""Action-conditioned online external/linked relation filter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from .runtime_features import RuntimeFeatureBuilder, RuntimeFeatures
from .state_index import StateId
from .task_model import ClosedLoopTaskModel, StateNode

Array = np.ndarray


class RelationDecision(str, Enum):
    EXTERNAL = "external"
    LINKED = "linked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RelationEstimate:
    frame_id: str
    posterior: Array
    predicted: Array
    demonstration_prior: Array
    observation_likelihood: Array
    information_weight: float
    entropy: float
    informative: bool
    decision_state: RelationDecision
    informative_evidence_direction: RelationDecision = RelationDecision.UNKNOWN

    def __post_init__(self) -> None:
        for name in ("posterior", "predicted", "demonstration_prior"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                value.shape != (2,)
                or np.any(value < 0.0)
                or not np.isclose(np.sum(value), 1.0)
            ):
                raise ValueError(f"{name} 必须为 external/linked 归一化向量")
            object.__setattr__(self, name, value.copy())
        likelihood = np.asarray(self.observation_likelihood, dtype=np.float64)
        if likelihood.shape != (2,) or np.any(likelihood <= 0.0):
            raise ValueError("关系观测似然必须为正的 [2] 向量")
        object.__setattr__(self, "observation_likelihood", likelihood.copy())

    @property
    def external(self) -> float:
        return float(self.posterior[0])

    @property
    def linked(self) -> float:
        return float(self.posterior[1])


@dataclass(frozen=True)
class RelationChange:
    frame_id: str
    previous: RelationDecision
    current: RelationDecision

    def __post_init__(self) -> None:
        if RelationDecision.UNKNOWN in {self.previous, self.current}:
            raise ValueError("可靠关系变化的两端都必须是稳定二元判定")
        if self.previous == self.current:
            raise ValueError("RelationChange 必须包含实际状态变化")


@dataclass(frozen=True)
class RelationFilterConfig:
    persistence_probability: float = 0.98
    demonstration_prior_strength: float = 0.35
    minimum_information_weight: float = 0.10
    minimum_tracking_reliability: float = 0.25
    maximum_decision_entropy: float = 0.62
    decision_probability: float = 0.70
    residual_ratio_scale: float = 0.25
    residual_motion_floor: float = 5.0e-4
    observation_outlier_probability: float = 0.002
    gripper_log_bias: float = 0.15
    probability_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        probabilities = (
            self.persistence_probability,
            self.minimum_information_weight,
            self.minimum_tracking_reliability,
            self.decision_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("关系滤波概率参数必须位于 [0,1]")
        if self.persistence_probability <= 0.5:
            raise ValueError("关系转移先验必须偏向保持当前状态")
        if self.demonstration_prior_strength < 0.0:
            raise ValueError("示范关系先验强度必须非负")
        if not 0.0 <= self.maximum_decision_entropy <= math.log(2.0):
            raise ValueError("二元关系熵阈值必须位于 [0,log(2)]")
        if self.decision_probability <= 0.5:
            raise ValueError("稳定关系判定阈值必须大于 0.5")
        if self.residual_ratio_scale <= 0.0 or self.residual_motion_floor <= 0.0:
            raise ValueError("关系残差尺度必须为正数")
        if not 0.0 <= self.observation_outlier_probability < 1.0:
            raise ValueError("关系观测离群概率必须位于 [0,1)")
        if self.gripper_log_bias < 0.0 or self.probability_floor <= 0.0:
            raise ValueError("夹爪偏置必须非负且概率下限必须为正数")


def _normalize(values: Array, floor: float) -> Array:
    probabilities = np.maximum(np.asarray(values, dtype=np.float64), floor)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        return np.asarray([0.5, 0.5], dtype=np.float64)
    return probabilities / total


class RelationFilter:
    """Maintain two-state relation probabilities; Unknown is a decision only."""

    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: RelationFilterConfig = RelationFilterConfig(),
        *,
        feature_builder: RuntimeFeatureBuilder | None = None,
    ) -> None:
        self.task_model = task_model
        self.config = config
        self.feature_builder = feature_builder or RuntimeFeatureBuilder()
        stay = config.persistence_probability
        self.transition = np.asarray([[stay, 1.0 - stay], [1.0 - stay, stay]])

    @staticmethod
    def _node_prior(
        node: StateNode,
        frame: str,
        mode_index: int | None,
    ) -> Array:
        priors = node.demo_relation_priors.get(frame)
        if priors is None:
            return np.asarray([0.5, 0.5], dtype=np.float64)
        if mode_index is not None:
            if mode_index < 0 or mode_index >= len(priors):
                raise IndexError("mode_index 超出关系先验的模态范围")
            return priors[mode_index]
        return np.sum(node.mode_priors[:, None] * priors, axis=0)

    def demonstration_prior(
        self,
        progress_prior: Mapping[StateId, float],
        frame: str,
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> Array:
        value = np.zeros(2, dtype=np.float64)
        for state, probability in progress_prior.items():
            node = self.task_model.state(state)
            mode = (
                None if mode_by_skill is None else mode_by_skill.get(state.skill_index)
            )
            value += probability * self._node_prior(node, frame, mode)
        return _normalize(value, self.config.probability_floor)

    def _observation_likelihood(
        self,
        frame: str,
        features: RuntimeFeatures,
    ) -> Array:
        residual = features.relative_motion_residuals.get(frame)
        if residual is None:
            return np.ones(2, dtype=np.float64)
        action_components = self.feature_builder.motion_component_magnitudes(
            features.actual_ee_motion
        )
        residual_components = self.feature_builder.motion_component_magnitudes(residual)

        # Translation and rotation only identify the relation along components
        # actually excited by this action.  Collapsing both into one scalar
        # ratio lets uncommanded rotational contact wobble overwhelm clear
        # translational co-motion (or the converse).  Build one finite
        # likelihood per component and combine them geometrically in proportion
        # to the observed action magnitude.  Conflicting components therefore
        # become bounded, ambiguous evidence instead of an artificial decisive
        # disconnect observation.
        total_action = float(np.sum(action_components))
        if total_action <= np.finfo(np.float64).eps:
            likelihood = np.ones(2, dtype=np.float64)
        else:
            component_weights = action_components / total_action
            component_log_likelihood = np.zeros(2, dtype=np.float64)
            for action_magnitude, residual_magnitude, weight in zip(
                action_components,
                residual_components,
                component_weights,
            ):
                if weight <= np.finfo(np.float64).eps:
                    continue
                denominator = max(
                    float(action_magnitude),
                    self.config.residual_motion_floor,
                )
                ratio = float(residual_magnitude) / denominator
                linked_support = math.exp(
                    -0.5 * (ratio / self.config.residual_ratio_scale) ** 2
                )
                # A single relative-pose component can be corrupted by
                # tracking jitter or transient multi-body contact.  Mix it with
                # a state-independent outlier component so its evidence remains
                # finite and the temporal relation prior stays meaningful.
                outlier = self.config.observation_outlier_probability
                component_likelihood = (1.0 - outlier) * np.asarray(
                    [1.0 - linked_support, linked_support], dtype=np.float64
                ) + outlier * np.asarray([0.5, 0.5], dtype=np.float64)
                component_log_likelihood += float(weight) * np.log(
                    np.maximum(component_likelihood, self.config.probability_floor)
                )
            likelihood = np.exp(component_log_likelihood)

        # The gripper is weak context only.  It cannot create evidence without
        # motion because the whole likelihood is exponentiated by X below.
        gripper_mean = float(np.mean(features.gripper_state))
        closed_score = float(np.clip((1.0 - gripper_mean) * 0.5, 0.0, 1.0))
        bias = self.config.gripper_log_bias * (closed_score - 0.5)
        likelihood[0] *= math.exp(-bias)
        likelihood[1] *= math.exp(bias)
        return np.maximum(likelihood, self.config.probability_floor)

    def update(
        self,
        progress_prior: Mapping[StateId, float],
        features: RuntimeFeatures,
        previous_posteriors: Mapping[str, Array] | None = None,
        *,
        previous_decisions: Mapping[str, RelationDecision] | None = None,
        previous_evidence_decisions: Mapping[str, RelationDecision] | None = None,
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> dict[str, RelationEstimate]:
        previous_posteriors = previous_posteriors or {}
        previous_decisions = previous_decisions or {}
        previous_evidence_decisions = previous_evidence_decisions or {}
        estimates: dict[str, RelationEstimate] = {}
        for frame in self.task_model.relation_frames:
            demo_prior = self.demonstration_prior(progress_prior, frame, mode_by_skill)
            previous = previous_posteriors.get(frame, demo_prior)
            previous = _normalize(previous, self.config.probability_floor)
            predicted = _normalize(
                self.transition.T @ previous, self.config.probability_floor
            )
            likelihood = self._observation_likelihood(frame, features)

            visibility = features.frame_visibility.get(frame, False)
            reliability = features.tracking_reliability.get(frame, 0.0)
            information = features.relation_information_weight.get(frame, 0.0)
            # The state-conditioned demonstration prior is weak context for
            # interpreting the *current action response*, not a new physical
            # observation by itself.  Temper it by the same observation
            # information X as the action-conditioned likelihood.  Otherwise
            # repeatedly applying a linked demo prior while the controller is
            # deliberately holding can erase a previously observed external
            # relation even though no relinking evidence occurred.  At X=0
            # the recursive posterior therefore reduces to the persistent
            # prediction; visibility/reliability still control the separate
            # Unknown decision below.
            observation_log_update = information * (
                self.config.demonstration_prior_strength
                * np.log(np.maximum(demo_prior, self.config.probability_floor))
                + np.log(np.maximum(likelihood, self.config.probability_floor))
            )
            log_posterior = (
                np.log(np.maximum(predicted, self.config.probability_floor))
                + observation_log_update
            )
            log_posterior -= float(np.max(log_posterior))
            posterior = _normalize(np.exp(log_posterior), self.config.probability_floor)
            entropy = -float(
                np.sum(
                    posterior
                    * np.log(np.maximum(posterior, self.config.probability_floor))
                )
            )
            informative = bool(
                visibility
                and reliability >= self.config.minimum_tracking_reliability
                and information >= self.config.minimum_information_weight
            )
            informative_evidence_direction = RelationDecision.UNKNOWN
            if informative and not np.isclose(likelihood[0], likelihood[1]):
                informative_evidence_direction = (
                    RelationDecision.LINKED
                    if likelihood[1] > likelihood[0]
                    else RelationDecision.EXTERNAL
                )
            posterior_decision = RelationDecision.UNKNOWN
            if (
                entropy <= self.config.maximum_decision_entropy
                and float(np.max(posterior)) >= self.config.decision_probability
            ):
                posterior_decision = (
                    RelationDecision.LINKED
                    if posterior[1] > posterior[0]
                    else RelationDecision.EXTERNAL
                )

            if informative:
                decision = posterior_decision
            else:
                previous_decision = previous_decisions.get(frame)
                evidence_decision = previous_evidence_decisions.get(frame)
                may_persist = bool(
                    visibility
                    and reliability >= self.config.minimum_tracking_reliability
                    and information < self.config.minimum_information_weight
                    and previous_decision
                    in {
                        RelationDecision.EXTERNAL,
                        RelationDecision.LINKED,
                    }
                    and posterior_decision == previous_decision
                )
                evidence_memory_available = bool(
                    visibility
                    and reliability >= self.config.minimum_tracking_reliability
                    and evidence_decision
                    in {RelationDecision.EXTERNAL, RelationDecision.LINKED}
                )
                if may_persist and previous_decision is not None:
                    decision = previous_decision
                elif evidence_memory_available:
                    # The Markov prediction deliberately makes the *soft*
                    # posterior less certain during a long interval without
                    # motion excitation.  That mathematical diffusion is not
                    # physical evidence of a detach/attach event.  Preserve
                    # the last relation actually confirmed by informative
                    # motion while the frame remains visible and reliable;
                    # the next informative observation still uses the soft
                    # posterior and can confirm the opposite state normally.
                    assert evidence_decision is not None
                    decision = evidence_decision
                else:
                    decision = RelationDecision.UNKNOWN
            estimates[frame] = RelationEstimate(
                frame_id=frame,
                posterior=posterior,
                predicted=predicted,
                demonstration_prior=demo_prior,
                observation_likelihood=likelihood,
                information_weight=information,
                entropy=entropy,
                informative=informative,
                decision_state=decision,
                informative_evidence_direction=informative_evidence_direction,
            )
        return estimates


__all__ = [
    "RelationChange",
    "RelationDecision",
    "RelationEstimate",
    "RelationFilter",
    "RelationFilterConfig",
]
