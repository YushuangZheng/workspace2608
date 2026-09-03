"""Full-state recovery reentry selection and explicit posterior reset."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ..inference.belief_updater import BeliefUpdater, ClosedLoopBelief
from ..model.boundary_model import BoundaryId
from ..control.execution_controller import ClosedLoopExecutionController
from ..inference.relation_filter import RelationDecision
from ..inference.runtime_observation import RuntimeObservation
from ..inference.state_evaluator import CandidateScore, StateEvaluator, StateEvaluatorConfig
from ..model.state_index import StateId
from ..model.task_model import ClosedLoopTaskModel


@dataclass(frozen=True)
class ReentryConfig:
    minimum_explanation_score: float = 0.001
    minimum_robot_peak_normalized_compatibility: float = 0.001
    minimum_scene_compatibility: float = 0.01
    minimum_relation_compatibility: float = 0.60
    require_relation_evidence_when_available: bool = True

    def __post_init__(self) -> None:
        for value in (
            self.minimum_explanation_score,
            self.minimum_robot_peak_normalized_compatibility,
            self.minimum_scene_compatibility,
            self.minimum_relation_compatibility,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("任务重入兼容度阈值必须位于 [0,1]")


@dataclass(frozen=True)
class ReentryDecision:
    state_id: StateId
    score: CandidateScore
    crossed_boundary: BoundaryId | None

    @property
    def reset_progress(self) -> dict[StateId, float]:
        return {self.state_id: 1.0}


@dataclass(frozen=True)
class ReentryEvaluation:
    decision: ReentryDecision | None
    scores: dict[StateId, CandidateScore]
    rejection_reasons: dict[StateId, tuple[str, ...]]
    alignment_state: StateId | None = None


class ReentrySelector:
    """Choose the best legal state from current robot, scene, and relation data."""

    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: ReentryConfig = ReentryConfig(),
        evaluator_config: StateEvaluatorConfig = StateEvaluatorConfig(),
        *,
        robot_covariance_inflation: float = 0.0,
    ) -> None:
        inflation = float(robot_covariance_inflation)
        if not math.isfinite(inflation) or inflation < 0.0:
            raise ValueError("重入机器人协方差放宽量必须为有限非负数")
        self.task_model = task_model
        self.config = config
        self.evaluator = StateEvaluator(task_model, evaluator_config)
        self.robot_covariance_inflation = inflation
        self._global_index = {
            state: index for index, state in enumerate(sorted(task_model.states))
        }

    def _boundary(
        self,
        current_reference: StateId,
        candidate: StateId,
    ) -> tuple[BoundaryId | None, bool]:
        if current_reference.skill_index == candidate.skill_index:
            return None, True
        if candidate.skill_index == current_reference.skill_index + 1:
            return (
                BoundaryId(
                    self.task_model.arm_id,
                    current_reference.skill_index,
                    candidate.skill_index,
                ),
                True,
            )
        return None, False

    def select(
        self,
        candidates: Sequence[StateId],
        belief: ClosedLoopBelief,
        *,
        current_reference: StateId,
        permitted_boundaries: frozenset[BoundaryId] = frozenset(),
        mode_by_skill: Mapping[int, int] | None = None,
        required_relations: Mapping[str, RelationDecision] | None = None,
    ) -> ReentryEvaluation:
        unique = tuple(dict.fromkeys(candidates))
        if not unique:
            return ReentryEvaluation(None, {}, {})
        unknown = set(unique).difference(self.task_model.states)
        if unknown:
            raise KeyError(f"任务重入候选包含未知状态：{sorted(unknown)}")
        scores = self.evaluator.evaluate_many(
            unique,
            belief.runtime_features,
            belief.relation_estimates,
            mode_by_skill=mode_by_skill,
            robot_covariance_inflation=self.robot_covariance_inflation,
        )
        rejections: dict[StateId, tuple[str, ...]] = {}
        accepted = []
        alignment_candidates = []
        relation_observable = any(
            estimate.decision_state != RelationDecision.UNKNOWN
            for estimate in belief.relation_estimates.values()
        )
        required = {} if required_relations is None else dict(required_relations)
        for state in unique:
            score = scores[state]
            reasons = []
            boundary, legal_skill_transition = self._boundary(current_reference, state)
            if not legal_skill_transition:
                reasons.append("nonadjacent_or_backward_skill_reentry")
            elif boundary is not None and boundary not in permitted_boundaries:
                reasons.append("cross_skill_guard_not_permitted")
            if not score.robot_evidence_available:
                reasons.append("robot_evidence_unavailable")
            elif (
                score.robot_peak_normalized_compatibility
                < self.config.minimum_robot_peak_normalized_compatibility
            ):
                reasons.append("robot_incompatible")
            if score.scene_evidence_expected:
                if not score.scene_evidence_available:
                    reasons.append("scene_evidence_unavailable")
                elif (
                    score.state_compatibility < self.config.minimum_scene_compatibility
                ):
                    reasons.append("scene_incompatible")
            if (
                self.config.require_relation_evidence_when_available
                and self.task_model.relation_frames
                and relation_observable
                and not score.relation_frame_weights
            ):
                reasons.append("relation_evidence_unavailable")
            elif (
                score.relation_state_compatibility
                < self.config.minimum_relation_compatibility
            ):
                reasons.append("relation_incompatible")
            for frame, expected in required.items():
                estimate = belief.relation_estimates.get(frame)
                if estimate is None or estimate.decision_state != expected:
                    reasons.append("recovered_relation_not_preserved")
                    break
            if (
                score.normalized_explanation_score
                < self.config.minimum_explanation_score
            ):
                reasons.append("insufficient_joint_explanation")
            if reasons:
                rejections[state] = tuple(reasons)
            else:
                accepted.append((state, score, boundary))

            # A relation repair can finish at a geometrically displaced grasp
            # pose.  Such a state is not yet legal for reentry, but it can be a
            # safe recovery-alignment target when every non-robot condition is
            # already satisfied.  Robot compatibility and the joint score are
            # deliberately omitted only for choosing the next recovery action;
            # the ordinary full-state thresholds above remain mandatory before
            # the task posterior and cursor are reset.
            non_robot_reasons = tuple(
                reason
                for reason in reasons
                if reason
                not in {
                    "robot_incompatible",
                    "insufficient_joint_explanation",
                }
            )
            if score.robot_evidence_available and not non_robot_reasons:
                alignment_candidates.append((state, score))

        if not accepted:
            alignment_state = (
                None
                if not alignment_candidates
                else max(
                    alignment_candidates,
                    key=lambda item: (
                        item[1].robot_peak_normalized_compatibility,
                        -self._global_index[item[0]],
                    ),
                )[0]
            )
            return ReentryEvaluation(
                None,
                scores,
                rejections,
                alignment_state=alignment_state,
            )
        state, score, boundary = max(
            accepted,
            key=lambda item: (
                item[1].normalized_explanation_score,
                -self._global_index[item[0]],
            ),
        )
        return ReentryEvaluation(
            ReentryDecision(state, score, boundary),
            scores,
            rejections,
            alignment_state=None,
        )

    @staticmethod
    def apply(
        decision: ReentryDecision,
        *,
        belief: ClosedLoopBelief,
        observation: RuntimeObservation,
        belief_updater: BeliefUpdater,
        execution_controller: ClosedLoopExecutionController,
    ) -> None:
        stable = {
            frame: estimate.decision_state
            for frame, estimate in belief.relation_estimates.items()
            if estimate.decision_state != RelationDecision.UNKNOWN
        }
        belief_updater.reset(
            initial_progress=decision.reset_progress,
            initial_relations=belief.relation_posteriors,
            initial_relation_decisions=stable,
            previous_observation=observation,
        )
        execution_controller.reset(decision.state_id)


__all__ = [
    "ReentryConfig",
    "ReentryDecision",
    "ReentryEvaluation",
    "ReentrySelector",
]
