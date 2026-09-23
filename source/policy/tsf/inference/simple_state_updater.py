"""Simple state inference for the equal-capability reviewer baseline.

The baseline keeps TSF's observations, demonstrations, action policy, stage
guards, repair executor, legal re-entry search, and budgets.  It replaces only
the coupled relation/progress posterior with direct relation thresholding and
a one-hot nearest-demonstration state inside the currently committed skill.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from ..model.boundary_model import BoundaryId
from ..model.state_index import StateId
from .belief_updater import BeliefUpdater, TSFBelief
from .progress_filter import ProgressEstimate, ProgressStatus
from .relation_filter import RelationDecision, RelationEstimate
from .runtime_features import RuntimeFeatures
from .runtime_observation import RuntimeObservation


class NearestDemoStateUpdater(BeliefUpdater):
    """Direct relation decisions and stage-guarded nearest-state matching."""

    @staticmethod
    def _normalize(values: np.ndarray, floor: float) -> np.ndarray:
        result = np.maximum(np.asarray(values, dtype=np.float64), floor)
        return result / float(np.sum(result))

    def reset(
        self,
        *,
        initial_progress: Mapping[StateId, float] | None = None,
        initial_relations: Mapping[str, np.ndarray] | None = None,
        initial_relation_decisions: Mapping[str, RelationDecision] | None = None,
        initial_relation_evidence_decisions: Mapping[
            str, RelationDecision
        ] | None = None,
        previous_observation: RuntimeObservation | None = None,
    ) -> None:
        inferred = dict(initial_relation_decisions or {})
        threshold = self.config.relation_filter.decision_probability
        for frame, raw in (initial_relations or {}).items():
            value = self._normalize(
                np.asarray(raw, dtype=np.float64),
                self.config.relation_filter.probability_floor,
            )
            if frame not in inferred and float(np.max(value)) >= threshold:
                inferred[frame] = (
                    RelationDecision.LINKED
                    if value[1] > value[0]
                    else RelationDecision.EXTERNAL
                )
        super().reset(
            initial_progress=initial_progress,
            initial_relations=initial_relations,
            initial_relation_decisions=inferred,
            initial_relation_evidence_decisions=(
                inferred
                if initial_relation_evidence_decisions is None
                else initial_relation_evidence_decisions
            ),
            previous_observation=previous_observation,
        )

    def _direct_relations(
        self,
        features: RuntimeFeatures,
        reference_state: StateId,
        mode_by_skill: Mapping[int, int] | None,
        *,
        demonstration_state_by_frame: Mapping[str, StateId] | None = None,
    ) -> dict[str, RelationEstimate]:
        estimates = {}
        threshold = self.config.relation_filter.decision_probability
        floor = self.config.relation_filter.probability_floor
        contexts = demonstration_state_by_frame or {}
        for frame in self.task_model.relation_frames:
            context = contexts.get(frame, reference_state)
            demo_prior = self.relation_filter.demonstration_prior(
                {context: 1.0}, frame, mode_by_skill
            )
            # Reuse the exact action-conditioned likelihood function but do
            # not feed it through the Markov transition or demonstration
            # prior. This is a direct per-cycle threshold by construction.
            likelihood = self.relation_filter._observation_likelihood(
                frame, features
            )
            direct = self._normalize(likelihood, floor)
            visibility = features.frame_visibility.get(frame, False)
            reliability = features.tracking_reliability.get(frame, 0.0)
            information = features.relation_information_weight.get(frame, 0.0)
            tracked = bool(
                visibility
                and reliability
                >= self.config.relation_filter.minimum_tracking_reliability
            )
            informative = bool(
                tracked
                and information
                >= self.config.relation_filter.minimum_information_weight
            )
            decision = RelationDecision.UNKNOWN
            direction = RelationDecision.UNKNOWN
            if informative:
                direction = (
                    RelationDecision.LINKED
                    if direct[1] > direct[0]
                    else RelationDecision.EXTERNAL
                )
                if float(np.max(direct)) >= threshold:
                    decision = direction
            elif tracked:
                # Retain only the last discrete decision during a quiet but
                # still tracked interval. No probability is accumulated.
                decision = self._stable_decisions.get(
                    frame, RelationDecision.UNKNOWN
                )
                if decision != RelationDecision.UNKNOWN:
                    direct = np.asarray(
                        [1.0 - threshold, threshold]
                        if decision == RelationDecision.LINKED
                        else [threshold, 1.0 - threshold],
                        dtype=np.float64,
                    )
            entropy = -float(np.sum(direct * np.log(np.maximum(direct, floor))))
            estimates[frame] = RelationEstimate(
                frame_id=frame,
                posterior=direct,
                predicted=direct,
                demonstration_prior=demo_prior,
                observation_likelihood=likelihood,
                information_weight=information,
                entropy=entropy,
                informative=informative,
                decision_state=decision,
                informative_evidence_direction=direction,
            )
        return estimates

    def _stage_states(self, nominal: StateId) -> tuple[StateId, ...]:
        return tuple(self.task_model.skill_states[nominal.skill_index])

    def _nearest_progress(
        self,
        nominal: StateId,
        scores: Mapping[StateId, Any],
    ) -> ProgressEstimate:
        def compatibility(state: StateId) -> float:
            score = scores[state]
            return (
                float(score.robot_peak_normalized_compatibility)
                if score.robot_evidence_available
                else 0.0
            )

        estimated = max(
            scores,
            key=lambda state: (
                compatibility(state),
                -abs(self._global_index[state] - self._global_index[nominal]),
                -self._global_index[state],
            ),
        )
        best = compatibility(estimated)
        posterior = {state: float(state == estimated) for state in scores}
        if best < self.config.progress_filter.minimum_explanation_score:
            status = ProgressStatus.NO_PLAUSIBLE_STATE
        elif self._global_index[estimated] > self._global_index[nominal]:
            status = ProgressStatus.FORWARD_REALIGNMENT
        elif self._global_index[estimated] < self._global_index[nominal]:
            status = ProgressStatus.BACKWARD_REALIGNMENT
        else:
            status = ProgressStatus.ALIGNED
        return ProgressEstimate(
            prior=dict(posterior),
            posterior=dict(posterior),
            nominal_state=nominal,
            estimated_state=estimated,
            confidence=1.0,
            entropy=0.0,
            best_explanation_score=best,
            status=status,
        )

    def update(
        self,
        observation: RuntimeObservation,
        *,
        executed_reference_state: StateId | None = None,
        action_executed: bool = True,
        permitted_boundaries: frozenset[BoundaryId] = frozenset(),
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> TSFBelief:
        if self._last_tick is not None and observation.tick <= self._last_tick:
            raise ValueError("NearestDemoStateUpdater requires increasing ticks")
        features = self.feature_builder.build(observation, self._previous_observation)
        prior = self.progress_prior_builder.build(
            self._progress_posterior,
            executed_reference_state=executed_reference_state,
            action_executed=action_executed,
            permitted_boundaries=permitted_boundaries,
        )
        nominal = prior.nominal_state
        relations = self._direct_relations(features, nominal, mode_by_skill)
        self._commit_informative_evidence(features, relations)
        changes = self._relation_changes(features, relations)
        candidates = self._stage_states(nominal)
        scores = self.state_evaluator.evaluate_many(
            candidates,
            features,
            relations,
            mode_by_skill=mode_by_skill,
        )
        progress = self._nearest_progress(nominal, scores)
        self._progress_posterior = dict(progress.posterior)
        self._relation_posteriors = {
            frame: estimate.posterior.copy()
            for frame, estimate in relations.items()
        }
        self._previous_observation = observation
        self._last_tick = observation.tick
        self._last_progress = progress
        return TSFBelief(
            tick=observation.tick,
            runtime_features=features,
            relation_estimates=relations,
            progress=progress,
            candidate_scores=scores,
            relation_changes=changes,
            local_candidates=candidates,
            expanded_candidates=(),
            update_sequence=(
                "direct_relation_threshold",
                "nearest_demo_state",
                "stage_guard",
            ),
        )

    def update_frozen(
        self,
        observation: RuntimeObservation,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
        demonstration_state_by_frame: Mapping[str, StateId] | None = None,
        uninformative_relation_frames: frozenset[str] = frozenset(),
    ) -> TSFBelief:
        if self._last_tick is not None and observation.tick <= self._last_tick:
            raise ValueError("NearestDemoStateUpdater requires increasing ticks")
        features = self.feature_builder.build(observation, self._previous_observation)
        if uninformative_relation_frames:
            information = dict(features.relation_information_weight)
            for frame in uninformative_relation_frames:
                if frame not in self.task_model.relation_frames:
                    raise KeyError(f"unknown relation frame: {frame}")
                information[frame] = 0.0
            features = replace(features, relation_information_weight=information)
        reference = self._last_progress.estimated_state
        relations = self._direct_relations(
            features,
            reference,
            mode_by_skill,
            demonstration_state_by_frame=demonstration_state_by_frame,
        )
        self._commit_informative_evidence(features, relations)
        changes = self._relation_changes(features, relations)
        candidates = tuple(self._progress_posterior)
        scores = self.state_evaluator.evaluate_many(
            candidates,
            features,
            relations,
            mode_by_skill=mode_by_skill,
        )
        best = max(
            (
                score.robot_peak_normalized_compatibility
                for score in scores.values()
                if score.robot_evidence_available
            ),
            default=self._last_progress.best_explanation_score,
        )
        progress = replace(
            self._last_progress,
            best_explanation_score=float(best),
        )
        self._relation_posteriors = {
            frame: estimate.posterior.copy()
            for frame, estimate in relations.items()
        }
        self._previous_observation = observation
        self._last_tick = observation.tick
        self._last_progress = progress
        return TSFBelief(
            tick=observation.tick,
            runtime_features=features,
            relation_estimates=relations,
            progress=progress,
            candidate_scores=scores,
            relation_changes=changes,
            local_candidates=candidates,
            expanded_candidates=(),
            update_sequence=(
                "frozen_progress",
                "direct_relation_threshold",
                "stage_guard",
            ),
        )


__all__ = ["NearestDemoStateUpdater"]
