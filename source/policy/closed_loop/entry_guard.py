"""Object-centric phase-four skill-entry guard evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..dynamac import relative_pose
from .belief_updater import ClosedLoopBelief
from .boundary_model import BoundaryId, BoundaryModel, RelationGuardDistribution
from .boundary_runtime import (
    BoundaryRuntimeConfig,
    ConditionId,
    ConditionKind,
    ConditionResult,
    LocalCompletionResult,
    TransitionRequest,
)
from .frame_roles import RelationVerificationRequest
from .progress_filter import ProgressStatus
from .relation_filter import RelationDecision, RelationEstimate
from .runtime_features import RuntimeFeatures
from .scene_factors import FactorId
from .state_index import StateId
from .task_model import ClosedLoopTaskModel


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    floor = np.finfo(np.float64).tiny
    return float(math.exp(np.mean(np.log(np.maximum(values, floor)))))


def _factor_observation(
    factor: FactorId,
    features_by_arm: Mapping[str, RuntimeFeatures],
    preferred_arm: str,
) -> tuple[np.ndarray | None, float]:
    ordered = [preferred_arm, *(arm for arm in features_by_arm if arm != preferred_arm)]
    for arm in ordered:
        features = features_by_arm[arm]
        if factor.kind == "node":
            value = features.entity_configurations.get(factor.source, {}).get(
                factor.feature
            )
            if value is None:
                continue
            if factor.source not in features.frame_visibility:
                return value, 1.0
            if not features.frame_visibility[factor.source]:
                return None, 0.0
            return value, float(features.tracking_reliability.get(factor.source, 0.0))

        assert factor.target is not None
        if (
            factor.source not in features.frame_poses
            or factor.target not in features.frame_poses
        ):
            continue
        if not features.frame_visibility.get(
            factor.source, False
        ) or not features.frame_visibility.get(factor.target, False):
            return None, 0.0
        reliability = min(
            features.tracking_reliability.get(factor.source, 0.0),
            features.tracking_reliability.get(factor.target, 0.0),
        )
        value = relative_pose(
            features.frame_poses[factor.target],
            features.frame_poses[factor.source],
        )
        return value, float(reliability)
    return None, 0.0


class EntryGuard:
    """Evaluate one arm's boundary from a shared pre-action belief snapshot."""

    def __init__(
        self,
        task_models: Mapping[str, ClosedLoopTaskModel],
        arm_id: str,
        config: BoundaryRuntimeConfig,
    ) -> None:
        if arm_id not in task_models:
            raise KeyError(f"入口守卫缺少机械臂 {arm_id} 的任务模型")
        if any(model.arm_id != key for key, model in task_models.items()):
            raise ValueError("任务模型字典键必须与 arm_id 一致")
        self.task_models = dict(task_models)
        self.arm_id = arm_id
        self.task_model = self.task_models[arm_id]
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._local_streaks: dict[BoundaryId, int] = {}
        self._condition_streaks: dict[tuple[BoundaryId, ConditionId], int] = {}
        self._last_ticks: dict[BoundaryId, int] = {}

    def _begin_tick(self, boundary_id: BoundaryId, tick: int) -> None:
        previous = self._last_ticks.get(boundary_id)
        if previous is not None and tick <= previous:
            raise ValueError("同一边界每个递增控制周期只能评估一次")
        if previous is not None and tick != previous + 1:
            self._local_streaks[boundary_id] = 0
            for key in tuple(self._condition_streaks):
                if key[0] == boundary_id:
                    self._condition_streaks[key] = 0
        self._last_ticks[boundary_id] = tick

    def _advance_condition(
        self,
        boundary_id: BoundaryId,
        condition_id: ConditionId,
        raw_satisfied: bool,
    ) -> int:
        key = (boundary_id, condition_id)
        streak = self._condition_streaks.get(key, 0) + 1 if raw_satisfied else 0
        self._condition_streaks[key] = streak
        return streak

    @staticmethod
    def _mode_index(
        boundary: BoundaryModel,
        model: ClosedLoopTaskModel,
        mode_by_skill: Mapping[int, int] | None,
    ) -> int:
        if mode_by_skill is not None and boundary.source_skill in mode_by_skill:
            mode = int(mode_by_skill[boundary.source_skill])
        else:
            final = boundary.local_completion_model.terminal_states[-1]
            mode = int(np.argmax(model.state(final).mode_priors))
        modes = len(model.state(boundary.terminal_window[-1]).mode_priors)
        if mode < 0 or mode >= modes:
            raise IndexError("入口守卫 mode_index 超出源技能范围")
        return mode

    def _goal_results(
        self,
        boundary: BoundaryModel,
        belief: ClosedLoopBelief,
        mode: int,
        required_cycles: int,
    ) -> tuple[list[ConditionResult], float, bool]:
        prefix = f"m{mode}:"
        selected = {
            key: distribution
            for key, distribution in boundary.local_completion_model.goal_distributions.items()
            if key.startswith(prefix)
        }
        results = []
        compatibilities = []
        all_available = bool(selected)
        features = belief.runtime_features
        for key, distribution in sorted(selected.items()):
            frame = key.split(":", 1)[1]
            value = features.relative_poses.get(frame)
            visible = features.frame_visibility.get(frame, False)
            reliability = (
                features.tracking_reliability.get(frame, 0.0) if visible else 0.0
            )
            observed = bool(
                value is not None
                and reliability >= self.config.minimum_tracking_reliability
            )
            compatibility = 0.0 if value is None else distribution.compatibility(value)
            if observed:
                compatibilities.append(compatibility)
            else:
                all_available = False
            condition_id = ConditionId(ConditionKind.LOCAL_GOAL, self.arm_id, key)
            streak = self._advance_condition(
                boundary.boundary_id, condition_id, observed
            )
            results.append(
                ConditionResult(
                    condition_id=condition_id,
                    compatibility=compatibility,
                    reliability=reliability,
                    threshold=None,
                    observed=observed,
                    stable=observed,
                    raw_satisfied=observed,
                    consecutive_cycles=streak,
                    required_cycles=required_cycles,
                    satisfied=observed and streak >= required_cycles,
                    reason="observed" if observed else "goal_unobservable",
                )
            )
        return (
            results,
            _geometric_mean(compatibilities) if all_available else 0.0,
            all_available,
        )

    def _relation_result(
        self,
        *,
        boundary: BoundaryModel,
        condition_id: ConditionId,
        condition: RelationGuardDistribution,
        belief: ClosedLoopBelief,
        required_cycles: int,
        guard: bool,
    ) -> tuple[ConditionResult, RelationEstimate | None]:
        _, frame = condition_id.token.split("/", 1)
        estimate = belief.relation_estimates.get(frame)
        features = belief.runtime_features
        visible = features.frame_visibility.get(frame, False)
        reliability = features.tracking_reliability.get(frame, 0.0) if visible else 0.0
        observed = bool(
            estimate is not None
            and reliability >= self.config.minimum_tracking_reliability
        )
        stable = bool(
            observed
            and estimate is not None
            and estimate.decision_state != RelationDecision.UNKNOWN
        )
        target_index = 1 if condition.required_state == "linked" else 0
        target_probability = (
            0.0
            if estimate is None
            else float(np.clip(estimate.posterior[target_index], 0.0, 1.0))
        )
        overlap = (
            0.0
            if estimate is None
            else float(
                np.clip(
                    np.dot(
                        estimate.posterior,
                        np.asarray([condition.external, condition.linked]),
                    ),
                    0.0,
                    1.0,
                )
            )
        )
        threshold = (
            self.config.relation_threshold(boundary.boundary_id, condition_id.token)
            if guard
            else None
        )
        raw_satisfied = bool(
            observed
            and stable
            and (threshold is None or target_probability > threshold)
        )
        streak = self._advance_condition(
            boundary.boundary_id, condition_id, raw_satisfied
        )
        if not observed:
            reason = "relation_unobservable"
        elif not stable:
            reason = "relation_unknown"
        elif threshold is not None and target_probability <= threshold:
            reason = "relation_below_threshold"
        else:
            reason = "satisfied"
        return (
            ConditionResult(
                condition_id=condition_id,
                compatibility=target_probability if guard else overlap,
                reliability=reliability,
                threshold=threshold,
                observed=observed,
                stable=stable,
                raw_satisfied=raw_satisfied,
                consecutive_cycles=streak,
                required_cycles=required_cycles,
                satisfied=raw_satisfied and streak >= required_cycles,
                reason=reason,
            ),
            estimate,
        )

    def _pending_request(
        self,
        *,
        relation_arm: str,
        frame: str,
        estimate: RelationEstimate | None,
        beliefs: Mapping[str, ClosedLoopBelief],
        mode_by_arm_skill: Mapping[str, Mapping[int, int]] | None,
    ) -> RelationVerificationRequest | None:
        if estimate is None or estimate.decision_state != RelationDecision.UNKNOWN:
            return None
        belief = beliefs[relation_arm]
        features = belief.runtime_features
        if (
            not features.frame_pair_available.get(frame, False)
            or features.paired_tracking_reliability.get(frame, 0.0)
            < self.config.minimum_tracking_reliability
            or estimate.information_weight >= self.config.minimum_information_weight
        ):
            return None
        model = self.task_models[relation_arm]
        mode_by_skill = (
            None if mode_by_arm_skill is None else mode_by_arm_skill.get(relation_arm)
        )
        for event_id, candidate in sorted(model.link_pending_events.items()):
            if candidate.frame_id != frame:
                continue
            if candidate.candidate_state != belief.progress.estimated_state:
                continue
            if mode_by_skill is not None:
                mode = mode_by_skill.get(candidate.candidate_state.skill_index)
                if mode is not None and mode != event_id.mode:
                    continue
            return RelationVerificationRequest(
                arm_id=relation_arm,
                frame_id=frame,
                relation="linked",
                pending_event_id=event_id,
            )
        return None

    def evaluate(
        self,
        boundary_id: BoundaryId,
        beliefs: Mapping[str, ClosedLoopBelief],
        source_state: StateId,
        *,
        mode_by_arm_skill: Mapping[str, Mapping[int, int]] | None = None,
    ) -> tuple[TransitionRequest, LocalCompletionResult]:
        try:
            boundary = self.task_model.boundaries[boundary_id]
        except KeyError as exc:
            raise KeyError(f"入口守卫不存在边界 {boundary_id.token}") from exc
        if source_state.skill_index != boundary.source_skill:
            raise ValueError("当前引用状态不在边界源技能")
        missing = set(boundary.affected_arms).difference(beliefs)
        if missing:
            raise KeyError(f"入口守卫缺少 affected arms 信念：{sorted(missing)}")
        if self.arm_id not in beliefs:
            raise KeyError("入口守卫缺少本臂信念")
        ticks = {belief.tick for belief in beliefs.values()}
        if len(ticks) != 1:
            raise ValueError("所有机械臂必须基于同一 pre-action tick 评估")
        tick = ticks.pop()
        self._begin_tick(boundary_id, tick)
        calibration = self.config.calibration_for(boundary_id)
        required_cycles = calibration.confirmation_cycles
        own_belief = beliefs[self.arm_id]
        own_mode = self._mode_index(
            boundary,
            self.task_model,
            None if mode_by_arm_skill is None else mode_by_arm_skill.get(self.arm_id),
        )

        condition_results: dict[ConditionId, ConditionResult] = {}
        end_probability = float(
            np.clip(
                sum(
                    own_belief.progress.posterior.get(state_id, 0.0)
                    for state_id in boundary.local_completion_model.terminal_states
                ),
                0.0,
                1.0,
            )
        )
        progress_supported = (
            own_belief.progress.status != ProgressStatus.NO_PLAUSIBLE_STATE
        )
        progress_id = ConditionId(
            ConditionKind.LOCAL_PROGRESS,
            self.arm_id,
            boundary_id.token,
        )
        progress_streak = self._advance_condition(
            boundary_id, progress_id, progress_supported
        )
        condition_results[progress_id] = ConditionResult(
            condition_id=progress_id,
            compatibility=end_probability,
            reliability=1.0 if progress_supported else 0.0,
            threshold=None,
            observed=progress_supported,
            stable=progress_supported,
            raw_satisfied=progress_supported,
            consecutive_cycles=progress_streak,
            required_cycles=required_cycles,
            satisfied=progress_supported and progress_streak >= required_cycles,
            reason="supported" if progress_supported else "no_plausible_state",
        )

        goal_results, goal_compatibility, goals_available = self._goal_results(
            boundary, own_belief, own_mode, required_cycles
        )
        condition_results.update(
            (result.condition_id, result) for result in goal_results
        )

        own_relation_values = []
        own_relations_available = True
        verification_requests: dict[str, RelationVerificationRequest] = {}
        for key, condition in sorted(
            boundary.local_completion_model.own_relation_conditions.items()
        ):
            relation_arm, frame = key.split("/", 1)
            if relation_arm != self.arm_id:
                raise ValueError("本臂必要关系不能引用其他机械臂")
            condition_id = ConditionId(ConditionKind.OWN_RELATION, relation_arm, key)
            result, estimate = self._relation_result(
                boundary=boundary,
                condition_id=condition_id,
                condition=condition,
                belief=own_belief,
                required_cycles=required_cycles,
                guard=False,
            )
            condition_results[condition_id] = result
            own_relation_values.append(result.compatibility)
            own_relations_available &= result.observed and result.stable
            if condition.required_state == "linked" and not result.raw_satisfied:
                request = self._pending_request(
                    relation_arm=relation_arm,
                    frame=frame,
                    estimate=estimate,
                    beliefs=beliefs,
                    mode_by_arm_skill=mode_by_arm_skill,
                )
                if request is not None:
                    verification_requests[request.pending_event_id.token] = request
        own_relation_compatibility = (
            min(own_relation_values) if own_relation_values else 1.0
        )
        evidence_available = bool(
            progress_supported and goals_available and own_relations_available
        )
        local_score = float(
            np.clip(
                end_probability * goal_compatibility * own_relation_compatibility,
                0.0,
                1.0,
            )
        )
        local_raw = bool(
            evidence_available and local_score > calibration.local_score_threshold
        )
        local_streak = self._local_streaks.get(boundary_id, 0) + 1 if local_raw else 0
        self._local_streaks[boundary_id] = local_streak
        local_done = bool(local_raw and local_streak >= required_cycles)
        local = LocalCompletionResult(
            boundary_id=boundary_id,
            end_probability=end_probability,
            goal_compatibility=goal_compatibility,
            own_relation_compatibility=own_relation_compatibility,
            score=local_score,
            threshold=calibration.local_score_threshold,
            raw_satisfied=local_raw,
            consecutive_cycles=local_streak,
            required_cycles=required_cycles,
            done=local_done,
            evidence_available=evidence_available,
        )

        guard_results = []
        for key, condition in sorted(boundary.relation_conditions.items()):
            relation_arm, frame = key.split("/", 1)
            if relation_arm not in beliefs:
                raise KeyError(f"边界关系条件缺少机械臂 {relation_arm} 信念")
            condition_id = ConditionId(ConditionKind.GUARD_RELATION, relation_arm, key)
            result, estimate = self._relation_result(
                boundary=boundary,
                condition_id=condition_id,
                condition=condition,
                belief=beliefs[relation_arm],
                required_cycles=required_cycles,
                guard=True,
            )
            condition_results[condition_id] = result
            guard_results.append(result)
            if condition.required_state == "linked" and not result.raw_satisfied:
                request = self._pending_request(
                    relation_arm=relation_arm,
                    frame=frame,
                    estimate=estimate,
                    beliefs=beliefs,
                    mode_by_arm_skill=mode_by_arm_skill,
                )
                if request is not None:
                    verification_requests[request.pending_event_id.token] = request

        features_by_arm = {
            arm: belief.runtime_features for arm, belief in beliefs.items()
        }
        for factor, distribution in sorted(boundary.scene_conditions.items()):
            condition_id = ConditionId(
                ConditionKind.GUARD_SCENE, self.arm_id, factor.token
            )
            value, reliability = _factor_observation(
                factor, features_by_arm, self.arm_id
            )
            observed = bool(
                value is not None
                and reliability > self.config.minimum_scene_reliability
            )
            compatibility = 0.0 if value is None else distribution.compatibility(value)
            threshold = float(boundary.scene_condition_thresholds[factor])
            raw_satisfied = bool(observed and compatibility > threshold)
            streak = self._advance_condition(boundary_id, condition_id, raw_satisfied)
            if not observed:
                reason = "scene_unobservable"
            elif compatibility <= threshold:
                reason = "scene_below_threshold"
            else:
                reason = "satisfied"
            result = ConditionResult(
                condition_id=condition_id,
                compatibility=compatibility,
                reliability=reliability,
                threshold=threshold,
                observed=observed,
                stable=observed,
                raw_satisfied=raw_satisfied,
                consecutive_cycles=streak,
                required_cycles=required_cycles,
                satisfied=raw_satisfied and streak >= required_cycles,
                reason=reason,
            )
            condition_results[condition_id] = result
            guard_results.append(result)

        permitted = bool(
            local_done and all(result.satisfied for result in guard_results)
        )
        target_state = self.task_model.skill_states[boundary.target_skill][0]
        return (
            TransitionRequest(
                tick=tick,
                arm_id=self.arm_id,
                boundary_id=boundary_id,
                permitted=permitted,
                source_state=source_state,
                target_state=target_state,
                local_done=local_done,
                condition_results=condition_results,
                verification_requests=tuple(verification_requests.values()),
                transaction_group=boundary.transaction_group,
            ),
            local,
        )


__all__ = ["EntryGuard"]
