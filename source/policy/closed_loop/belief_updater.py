"""Fixed single-pass relation-progress belief update for phase two."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .boundary_model import BoundaryId
from .progress_filter import (
    ProgressEstimate,
    ProgressFilter,
    ProgressFilterConfig,
    ProgressStatus,
)
from .progress_prior import ProgressPrior, ProgressPriorBuilder, ProgressPriorConfig
from .relation_filter import (
    RelationChange,
    RelationDecision,
    RelationEstimate,
    RelationFilter,
    RelationFilterConfig,
)
from .runtime_features import (
    RuntimeFeatureBuilder,
    RuntimeFeatureConfig,
    RuntimeFeatures,
)
from .runtime_observation import RuntimeObservation
from .state_evaluator import CandidateScore, StateEvaluator, StateEvaluatorConfig
from .state_index import StateId
from .task_model import ClosedLoopTaskModel


@dataclass(frozen=True)
class CandidateExpansionConfig:
    extension_prior_mass: float = 0.08
    minimum_relation_compatibility: float = 0.60
    minimum_robot_compatibility: float = 0.01
    minimum_scene_compatibility: float = 0.01

    def __post_init__(self) -> None:
        for value in (
            self.extension_prior_mass,
            self.minimum_relation_compatibility,
            self.minimum_robot_compatibility,
            self.minimum_scene_compatibility,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("候选扩展阈值必须位于 [0,1]")
        if self.extension_prior_mass <= 0.0:
            raise ValueError("候选扩展质量必须为正数")


@dataclass(frozen=True)
class BeliefUpdaterConfig:
    runtime_features: RuntimeFeatureConfig = field(default_factory=RuntimeFeatureConfig)
    progress_prior: ProgressPriorConfig = field(default_factory=ProgressPriorConfig)
    relation_filter: RelationFilterConfig = field(default_factory=RelationFilterConfig)
    state_evaluator: StateEvaluatorConfig = field(default_factory=StateEvaluatorConfig)
    progress_filter: ProgressFilterConfig = field(default_factory=ProgressFilterConfig)
    candidate_expansion: CandidateExpansionConfig = field(
        default_factory=CandidateExpansionConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BeliefUpdaterConfig:
        sections: tuple[tuple[str, type[Any]], ...] = (
            ("runtime_features", RuntimeFeatureConfig),
            ("progress_prior", ProgressPriorConfig),
            ("relation_filter", RelationFilterConfig),
            ("state_evaluator", StateEvaluatorConfig),
            ("progress_filter", ProgressFilterConfig),
            ("candidate_expansion", CandidateExpansionConfig),
        )
        unknown = set(value).difference(name for name, _ in sections)
        if unknown:
            raise ValueError(f"阶段二配置包含未知分区：{sorted(unknown)}")
        arguments = {}
        for name, section_type in sections:
            raw = value.get(name, {})
            if not isinstance(raw, Mapping):
                raise TypeError(f"阶段二配置分区 {name} 必须为对象")
            arguments[name] = section_type(**dict(raw))
        return cls(**arguments)

    @classmethod
    def from_json(cls, path: str | Path) -> BeliefUpdaterConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("阶段二配置文件根节点必须为对象")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class ClosedLoopBelief:
    tick: int
    runtime_features: RuntimeFeatures
    relation_estimates: dict[str, RelationEstimate]
    progress: ProgressEstimate
    candidate_scores: dict[StateId, CandidateScore]
    relation_changes: tuple[RelationChange, ...]
    local_candidates: tuple[StateId, ...]
    expanded_candidates: tuple[StateId, ...]
    update_sequence: tuple[str, ...] = (
        "progress_prior",
        "relation_posterior",
        "progress_posterior",
    )

    @property
    def relation_posteriors(self) -> dict[str, np.ndarray]:
        return {
            frame: estimate.posterior.copy()
            for frame, estimate in self.relation_estimates.items()
        }


class BeliefUpdater:
    """Execute ``progress prior -> relation posterior -> progress posterior`` once."""

    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: BeliefUpdaterConfig = BeliefUpdaterConfig(),
    ) -> None:
        self.task_model = task_model
        self.config = config
        self.feature_builder = RuntimeFeatureBuilder(config.runtime_features)
        self.progress_prior_builder = ProgressPriorBuilder(
            task_model, config.progress_prior
        )
        self.relation_filter = RelationFilter(
            task_model,
            config.relation_filter,
            feature_builder=self.feature_builder,
        )
        self.state_evaluator = StateEvaluator(task_model, config.state_evaluator)
        self.progress_filter = ProgressFilter(task_model, config.progress_filter)
        self.expansion_config = config.candidate_expansion
        self._states = tuple(sorted(task_model.states))
        self._global_index = {state: index for index, state in enumerate(self._states)}
        self._last_progress: ProgressEstimate
        self.reset()

    @property
    def progress_posterior(self) -> dict[StateId, float]:
        """Return the committed progress posterior without exposing mutation."""

        return dict(self._progress_posterior)

    def validate_boundary_transition(
        self,
        boundary_id: BoundaryId,
        target_state: StateId,
    ) -> None:
        if boundary_id.arm_id != self.task_model.arm_id:
            raise ValueError("进度投影的边界机械臂与任务模型不一致")
        if boundary_id not in self.task_model.boundaries:
            raise KeyError(f"进度投影引用未知边界：{boundary_id.token}")
        expected = self.task_model.skill_states[boundary_id.target_skill][0]
        if target_state != expected:
            raise ValueError("边界进度投影目标必须是下一技能入口状态")
        if any(
            state.skill_index > boundary_id.source_skill
            for state, probability in self._progress_posterior.items()
            if probability > 0.0
        ):
            raise RuntimeError("边界提交前进度后验已经越过待提交边界")

    def commit_boundary_transition(
        self,
        boundary_id: BoundaryId,
        target_state: StateId,
    ) -> None:
        """Project all pre-boundary progress mass onto a committed entry.

        The current cycle's returned belief remains an audit of the source
        state.  Only the internally committed posterior for the next control
        cycle is changed.  Relation posteriors and observations are untouched.
        """

        self.validate_boundary_transition(boundary_id, target_state)
        projected = {target_state: 1.0}
        self._progress_posterior = projected
        self._last_progress = replace(
            self._last_progress,
            prior=dict(projected),
            posterior=dict(projected),
            nominal_state=target_state,
            estimated_state=target_state,
            confidence=1.0,
            entropy=0.0,
            status=ProgressStatus.ALIGNED,
        )

    def reset(
        self,
        *,
        initial_progress: Mapping[StateId, float] | None = None,
        initial_relations: Mapping[str, np.ndarray] | None = None,
        initial_relation_decisions: Mapping[str, RelationDecision] | None = None,
        previous_observation: RuntimeObservation | None = None,
    ) -> None:
        if initial_progress is None:
            initial_progress = {self._states[0]: 1.0}
        total = float(sum(initial_progress.values()))
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("初始进度信念必须具有正的有限总和")
        if any(
            not np.isfinite(value) or value < 0.0 for value in initial_progress.values()
        ):
            raise ValueError("初始进度信念必须为有限非负数")
        if set(initial_progress).difference(self.task_model.states):
            raise KeyError("初始进度信念包含未知状态")
        self._progress_posterior = {
            state: float(value / total) for state, value in initial_progress.items()
        }
        initial_state = max(
            self._progress_posterior,
            key=lambda state: (
                self._progress_posterior[state],
                -self._global_index[state],
            ),
        )
        probabilities = np.asarray(tuple(self._progress_posterior.values()))
        entropy = -float(
            np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-300)))
        )
        self._last_progress = ProgressEstimate(
            prior=dict(self._progress_posterior),
            posterior=dict(self._progress_posterior),
            nominal_state=initial_state,
            estimated_state=initial_state,
            confidence=float(self._progress_posterior[initial_state]),
            entropy=entropy,
            best_explanation_score=1.0,
            status=ProgressStatus.ALIGNED,
        )
        self._relation_posteriors = {
            frame: np.asarray(value, dtype=np.float64).copy()
            for frame, value in (initial_relations or {}).items()
        }
        self._previous_observation = previous_observation
        self._last_tick = (
            None if previous_observation is None else previous_observation.tick
        )
        decisions = dict(initial_relation_decisions or {})
        unknown_frames = set(decisions).difference(self.task_model.relation_frames)
        if unknown_frames:
            raise KeyError(f"初始关系判定包含未知参考系：{sorted(unknown_frames)}")
        if any(decision == RelationDecision.UNKNOWN for decision in decisions.values()):
            raise ValueError("初始稳定关系判定不能为 Unknown")
        self._stable_decisions = decisions
        self._informative_evidence_decisions: dict[str, RelationDecision] = {}

    def _commit_informative_evidence(
        self,
        features: RuntimeFeatures,
        estimates: Mapping[str, RelationEstimate],
    ) -> None:
        """Keep the last relation *confirmed* by informative motion.

        An instantaneous likelihood direction is not itself a confirmed
        relation.  In particular, transient contact can favour ``external``
        for one or more samples while the persistent posterior still supports
        a previously confirmed ``linked`` relation.  Saving that raw direction
        would silently replace the confirmation before the filter has actually
        accepted a state change, and later quiet cycles could no longer retain
        the valid decision.

        Only an informative, stable posterior decision that agrees with the
        current evidence direction is therefore committed.  Visibility or
        reliability gaps still invalidate the confirmation exactly as before.
        """

        for frame, estimate in estimates.items():
            visible = features.frame_visibility.get(frame, False)
            reliable = (
                features.tracking_reliability.get(frame, 0.0)
                >= self.config.relation_filter.minimum_tracking_reliability
            )
            if not visible or not reliable:
                # A visibility/reliability gap invalidates continuity.  The
                # relation must acquire fresh motion evidence after reappearing.
                self._informative_evidence_decisions.pop(frame, None)
                continue
            direction = estimate.informative_evidence_direction
            if (
                estimate.informative
                and direction in {RelationDecision.EXTERNAL, RelationDecision.LINKED}
                and estimate.decision_state == direction
            ):
                self._informative_evidence_decisions[frame] = estimate.decision_state

    def _relation_changes(
        self,
        features: RuntimeFeatures,
        estimates: Mapping[str, RelationEstimate],
    ) -> tuple[RelationChange, ...]:
        changes = []
        for frame, estimate in estimates.items():
            current = estimate.decision_state
            if current == RelationDecision.UNKNOWN:
                visible = features.frame_visibility.get(frame, False)
                reliable = (
                    features.tracking_reliability.get(frame, 0.0)
                    >= self.config.relation_filter.minimum_tracking_reliability
                )
                if not visible or not reliable:
                    # An actual observation-quality gap breaks continuity: the
                    # relation must acquire fresh motion evidence after it
                    # reappears.  A visible, reliable posterior-conflict or
                    # low-excitation Unknown is only a temporary lack of a
                    # discrete decision and must not erase the last confirmed
                    # physical state.
                    self._stable_decisions.pop(frame, None)
                continue
            previous = self._stable_decisions.get(frame)
            if previous is not None and previous != current:
                changes.append(RelationChange(frame, previous, current))
            self._stable_decisions[frame] = current
        return tuple(changes)

    def _crossing_allowed(
        self,
        source: StateId,
        target: StateId,
        permitted_boundaries: frozenset[BoundaryId],
    ) -> bool:
        if source.skill_index == target.skill_index:
            return True
        return (
            BoundaryId(self.task_model.arm_id, source.skill_index, target.skill_index)
            in permitted_boundaries
        )

    def _future_states(
        self,
        after: StateId,
        permitted_boundaries: frozenset[BoundaryId],
    ) -> tuple[StateId, ...]:
        result = []
        previous = after
        for state in self._states[self._global_index[after] + 1 :]:
            if not self._crossing_allowed(previous, state, permitted_boundaries):
                break
            result.append(state)
            previous = state
        return tuple(result)

    def _relation_segments(
        self,
        prior: ProgressPrior,
        features: RuntimeFeatures,
        relations: Mapping[str, RelationEstimate],
        permitted_boundaries: frozenset[BoundaryId],
        mode_by_skill: Mapping[int, int] | None,
    ) -> tuple[tuple[StateId, ...], ...]:
        after = prior.candidates[-1]
        future = self._future_states(after, permitted_boundaries)
        segments: list[tuple[StateId, ...]] = []
        segment: list[StateId] = []
        for state in future:
            score = self.state_evaluator.evaluate(
                state,
                features,
                relations,
                mode_by_skill=mode_by_skill,
            )
            compatible = (
                bool(score.relation_frame_weights)
                and score.relation_compatibility
                >= self.expansion_config.minimum_relation_compatibility
            )
            if compatible:
                segment.append(state)
            elif segment:
                segments.append(tuple(segment))
                segment = []
        if segment:
            segments.append(tuple(segment))
        return tuple(segments)

    def _eligible_expanded_scores(
        self,
        candidates: tuple[StateId, ...],
        features: RuntimeFeatures,
        relations: Mapping[str, RelationEstimate],
        mode_by_skill: Mapping[int, int] | None,
    ) -> dict[StateId, CandidateScore]:
        scores = self.state_evaluator.evaluate_many(
            candidates,
            features,
            relations,
            mode_by_skill=mode_by_skill,
        )
        return {
            state: score
            for state, score in scores.items()
            if score.robot_evidence_available
            and score.robot_compatibility
            >= self.expansion_config.minimum_robot_compatibility
            and score.relation_compatibility
            >= self.expansion_config.minimum_relation_compatibility
            and (
                not score.scene_evidence_expected
                or (
                    score.scene_evidence_available
                    and score.state_compatibility
                    >= self.expansion_config.minimum_scene_compatibility
                )
            )
        }

    def _extend_prior(
        self,
        local: Mapping[StateId, float],
        expanded: tuple[StateId, ...],
    ) -> dict[StateId, float]:
        if not expanded:
            return dict(local)
        mass = self.expansion_config.extension_prior_mass
        result = {state: (1.0 - mass) * value for state, value in local.items()}
        each = mass / len(expanded)
        result.update({state: each for state in expanded})
        return result

    def update(
        self,
        observation: RuntimeObservation,
        *,
        executed_reference_state: StateId | None = None,
        action_executed: bool = True,
        permitted_boundaries: frozenset[BoundaryId] = frozenset(),
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> ClosedLoopBelief:
        if self._last_tick is not None and observation.tick <= self._last_tick:
            raise ValueError("BeliefUpdater 每个递增控制周期只能更新一次")
        features = self.feature_builder.build(observation, self._previous_observation)
        progress_prior = self.progress_prior_builder.build(
            self._progress_posterior,
            executed_reference_state=executed_reference_state,
            action_executed=action_executed,
            permitted_boundaries=permitted_boundaries,
        )
        relation_estimates = self.relation_filter.update(
            progress_prior.probabilities,
            features,
            self._relation_posteriors,
            previous_decisions=self._stable_decisions,
            previous_evidence_decisions=self._informative_evidence_decisions,
            mode_by_skill=mode_by_skill,
        )
        self._commit_informative_evidence(features, relation_estimates)
        changes = self._relation_changes(features, relation_estimates)
        local_scores = self.state_evaluator.evaluate_many(
            progress_prior.candidates,
            features,
            relation_estimates,
            mode_by_skill=mode_by_skill,
        )
        local_plausible = (
            max(score.normalized_explanation_score for score in local_scores.values())
            >= self.progress_filter.config.minimum_explanation_score
        )

        expanded_scores: dict[StateId, CandidateScore] = {}
        if changes and not local_plausible:
            relation_segments = self._relation_segments(
                progress_prior,
                features,
                relation_estimates,
                permitted_boundaries,
                mode_by_skill,
            )
            for relation_segment in relation_segments:
                expanded_scores = self._eligible_expanded_scores(
                    relation_segment,
                    features,
                    relation_estimates,
                    mode_by_skill,
                )
                if expanded_scores:
                    break
        expanded = tuple(sorted(expanded_scores, key=self._global_index.__getitem__))
        combined_scores = dict(local_scores)
        combined_scores.update(expanded_scores)
        combined_prior = self._extend_prior(progress_prior.probabilities, expanded)
        progress = self.progress_filter.update(
            combined_prior,
            combined_scores,
            progress_prior.nominal_state,
        )

        self._progress_posterior = dict(progress.posterior)
        self._relation_posteriors = {
            frame: estimate.posterior.copy()
            for frame, estimate in relation_estimates.items()
        }
        self._previous_observation = observation
        self._last_tick = observation.tick
        self._last_progress = progress
        return ClosedLoopBelief(
            tick=observation.tick,
            runtime_features=features,
            relation_estimates=dict(relation_estimates),
            progress=progress,
            candidate_scores=combined_scores,
            relation_changes=changes,
            local_candidates=progress_prior.candidates,
            expanded_candidates=expanded,
        )

    def update_frozen(
        self,
        observation: RuntimeObservation,
        *,
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> ClosedLoopBelief:
        """Update relation evidence while freezing the normal progress posterior.

        ``VERIFY_LINK`` and ``RECOVERY`` actions are not normal task actions, so
        their observations must not advance or realign the task clock.  The
        same feature builder and relation filter still consume the real motion.
        Candidate scores are diagnostic only; they do not alter beta until an
        explicit reentry decision resets it.
        """

        if self._last_tick is not None and observation.tick <= self._last_tick:
            raise ValueError("BeliefUpdater 每个递增控制周期只能更新一次")
        features = self.feature_builder.build(observation, self._previous_observation)
        frozen_progress = dict(self._progress_posterior)
        relation_estimates = self.relation_filter.update(
            frozen_progress,
            features,
            self._relation_posteriors,
            previous_decisions=self._stable_decisions,
            previous_evidence_decisions=self._informative_evidence_decisions,
            mode_by_skill=mode_by_skill,
        )
        self._commit_informative_evidence(features, relation_estimates)
        changes = self._relation_changes(features, relation_estimates)
        candidates = tuple(sorted(frozen_progress, key=self._global_index.__getitem__))
        scores = self.state_evaluator.evaluate_many(
            candidates,
            features,
            relation_estimates,
            mode_by_skill=mode_by_skill,
        )
        best_explanation = max(
            (score.normalized_explanation_score for score in scores.values()),
            default=self._last_progress.best_explanation_score,
        )
        progress = replace(
            self._last_progress,
            prior=frozen_progress,
            posterior=frozen_progress,
            best_explanation_score=float(best_explanation),
        )
        self._relation_posteriors = {
            frame: estimate.posterior.copy()
            for frame, estimate in relation_estimates.items()
        }
        self._previous_observation = observation
        self._last_tick = observation.tick
        self._last_progress = progress
        return ClosedLoopBelief(
            tick=observation.tick,
            runtime_features=features,
            relation_estimates=dict(relation_estimates),
            progress=progress,
            candidate_scores=scores,
            relation_changes=changes,
            local_candidates=candidates,
            expanded_candidates=(),
            update_sequence=("frozen_progress", "relation_posterior"),
        )


__all__ = [
    "BeliefUpdater",
    "BeliefUpdaterConfig",
    "CandidateExpansionConfig",
    "ClosedLoopBelief",
]
