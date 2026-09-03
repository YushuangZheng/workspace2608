"""Auditable robot/scene/relation scoring for candidate progress states."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ...dynamac import (
    pose_log_nearest,
    product_of_experts,
    relative_pose,
    transform_marginal,
)
from .relation_filter import RelationDecision, RelationEstimate
from .runtime_features import RuntimeFeatures
from ..model.scene_factors import FactorDistribution, FactorId
from ..model.state_index import StateId
from ..model.relation_events import RelationStateKey
from ..model.task_model import ClosedLoopTaskModel, StateNode

Array = np.ndarray


def joint_peak_normalized_pose_support(
    entries: list[tuple[str, Array, Array, Array, Array, float]],
    *,
    diagonalize: bool,
) -> tuple[float, float, float]:
    """Return one multi-flow support relative to its attainable PoE peak.

    Each entry is ``(frame, frame_pose, local_mean, local_covariance,
    observed_local_pose, weight)``.  The implementation deliberately reuses
    the baseline transform and product operators; it does not introduce a
    second SE(3) or PoE path.
    """

    if not entries:
        return 1.0, 0.0, 0.0
    marginals = []
    weights = []
    current_log_sum = 0.0
    for frame, frame_pose, mean, covariance, value, weight in entries:
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight) or numeric_weight <= 0.0:
            raise ValueError("联合位姿支持权重必须为有限正数")
        support, _, _, _ = _gaussian_pose_terms(mean, covariance, value)
        current_log_sum += numeric_weight * support
        weights.append(numeric_weight)
        marginals.append(
            transform_marginal(
                frame,
                frame_pose,
                mean,
                covariance,
                diagonalize=diagonalize,
            )
        )
    joint_peak, _, _ = product_of_experts(
        marginals,
        precision_weights=weights,
    )
    peak_log_sum = 0.0
    for entry, weight in zip(entries, weights, strict=True):
        frame, frame_pose, mean, covariance, _, _ = entry
        peak_relative = relative_pose(frame_pose, joint_peak)
        peak_support, _, _, _ = _gaussian_pose_terms(mean, covariance, peak_relative)
        peak_log_sum += weight * peak_support
    normalized_log = min(0.0, current_log_sum - peak_log_sum)
    compatibility = float(math.exp(max(-750.0, normalized_log / float(sum(weights)))))
    return compatibility, float(current_log_sum), float(peak_log_sum)


def joint_poe_pose_support(
    entries: list[tuple[str, Array, Array, Array, float]],
    observed_pose: Array,
    *,
    diagonalize: bool,
) -> tuple[float, float, float]:
    """Score one world EE pose against the joint PoE target distribution.

    Each entry is ``(frame, frame_pose, local_mean, local_covariance, weight)``.
    Unlike :func:`joint_peak_normalized_pose_support`, which removes the
    attainable-peak offset from a sum of per-stream local scores for progress
    inference, this function first constructs the actual joint world target
    used by DynaMAC's PoE and then evaluates the observed end effector against
    that single Gaussian.  This is the phase-four ``L_goal`` quantity.
    """

    if not entries:
        return 1.0, 0.0, 0.0
    marginals = []
    weights = []
    for frame, frame_pose, mean, covariance, weight in entries:
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight) or numeric_weight <= 0.0:
            raise ValueError("联合目标支持权重必须为有限正数")
        marginals.append(
            transform_marginal(
                frame,
                frame_pose,
                mean,
                covariance,
                diagonalize=diagonalize,
            )
        )
        weights.append(numeric_weight)
    joint_mean, joint_covariance, _ = product_of_experts(
        marginals,
        precision_weights=weights,
    )
    normalized_log_support, _, mahalanobis, _ = _gaussian_pose_terms(
        joint_mean,
        joint_covariance,
        observed_pose,
    )
    return (
        float(math.exp(max(-750.0, normalized_log_support))),
        float(normalized_log_support),
        float(mahalanobis),
    )


def _logsumexp(values: Array, weights: Array) -> float:
    active = weights > 0.0
    if not np.any(active):
        return 0.0
    selected = values[active]
    selected_weights = weights[active]
    maximum = float(np.max(selected))
    return maximum + math.log(
        float(np.sum(selected_weights * np.exp(selected - maximum)))
        / float(np.sum(selected_weights))
    )


@dataclass(frozen=True)
class GaussianComponentAudit:
    """One Gaussian component's complete online-score audit decomposition."""

    mode_index: int
    mode_weight: float
    dimension: int
    raw_log_likelihood: float
    normalized_log_support: float
    mahalanobis_squared: float
    covariance_log_determinant: float


def _gaussian_terms(
    residual: Array,
    covariance: Array,
) -> tuple[float, float, float, float]:
    residual = np.asarray(residual, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if residual.ndim != 1 or covariance.shape != (len(residual), len(residual)):
        raise ValueError("高斯观测残差与协方差维数不一致")
    regularized = covariance + np.eye(len(residual), dtype=np.float64) * 1.0e-12
    sign, logdet = np.linalg.slogdet(regularized)
    if sign <= 0.0:
        raise RuntimeError("在线状态评分协方差不是正定矩阵")
    try:
        solved = np.linalg.solve(regularized, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(regularized) @ residual
    mahalanobis = max(0.0, float(residual @ solved))
    normalized_log_support = -0.5 * mahalanobis
    raw_log_likelihood = normalized_log_support - 0.5 * (
        logdet + len(residual) * math.log(2.0 * math.pi)
    )
    return (
        float(normalized_log_support),
        float(raw_log_likelihood),
        mahalanobis,
        float(logdet),
    )


def _gaussian_pose_terms(
    mean: Array,
    covariance: Array,
    value: Array,
) -> tuple[float, float, float, float]:
    return _gaussian_terms(pose_log_nearest(mean, value), covariance)


@dataclass(frozen=True)
class CandidateScore:
    state_id: StateId
    # Full Gaussian densities are retained for diagnosis only.  Progress
    # inference uses the peak-normalized support fields below so covariance
    # determinants cannot reward a state merely for having a narrow model.
    robot_log_likelihood: float
    state_log_likelihood: float
    # Joint-peak-normalized robot support consumed by the progress posterior.
    robot_log_support: float
    # Sum of the individually peak-normalized stream supports before removing
    # the state-dependent, jointly attainable PoE peak.
    robot_unadjusted_log_support: float
    state_log_support: float
    relation_log_compatibility: float
    explanation_log_score: float
    normalized_explanation_score: float
    robot_compatibility: float
    state_compatibility: float
    relation_compatibility: float
    # Zero during normal progress inference.  Recovery reentry reuses the
    # already-configured recovery covariance inflation so its terminal state
    # test is not stricter than the action distribution that produced it.
    robot_covariance_inflation: float = 0.0
    # ``robot_compatibility`` is the geometric mean of the individually
    # peak-normalized stream supports and is retained for component audit.
    # Absolute decisions use the jointly attainable peak-normalized value
    # below, because one world EE pose generally cannot sit at every
    # transformed expert mean at once.
    robot_peak_normalized_compatibility: float = 1.0
    relation_peak_normalized_compatibility: float = 1.0
    # Thresholded state-transition/reentry checks need both a unit-peak scale
    # and the same external/linked physical direction.  The raw overlap above
    # remains the quantity consumed by the progress posterior.
    relation_state_compatibility: float = 1.0
    robot_attainable_peak_log_support: float = 0.0
    robot_frame_terms: dict[str, float] = field(default_factory=dict)
    robot_frame_weights: dict[str, float] = field(default_factory=dict)
    scene_factor_terms: dict[FactorId, float] = field(default_factory=dict)
    scene_factor_weights: dict[FactorId, float] = field(default_factory=dict)
    relation_frame_terms: dict[str, float] = field(default_factory=dict)
    relation_frame_peak_normalized_terms: dict[str, float] = field(default_factory=dict)
    relation_frame_weights: dict[str, float] = field(default_factory=dict)
    robot_frame_raw_log_likelihoods: dict[str, float] = field(default_factory=dict)
    scene_factor_raw_log_likelihoods: dict[FactorId, float] = field(
        default_factory=dict
    )
    robot_frame_gaussian_audits: dict[str, tuple[GaussianComponentAudit, ...]] = field(
        default_factory=dict
    )
    scene_factor_gaussian_audits: dict[FactorId, tuple[GaussianComponentAudit, ...]] = (
        field(default_factory=dict)
    )
    robot_evidence_available: bool = False
    scene_evidence_expected: bool = False
    scene_evidence_available: bool = False


@dataclass(frozen=True)
class StateEvaluatorConfig:
    scene_weight: float = 1.0
    relation_weight: float = 1.0
    probability_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.scene_weight < 0.0 or self.relation_weight < 0.0:
            raise ValueError("进度证据权重必须非负")
        if self.probability_floor <= 0.0:
            raise ValueError("概率下限必须为正数")


class StateEvaluator:
    def __init__(
        self,
        task_model: ClosedLoopTaskModel,
        config: StateEvaluatorConfig = StateEvaluatorConfig(),
    ) -> None:
        self.task_model = task_model
        self.config = config

    @staticmethod
    def _mode_weights(node: StateNode, selected_mode: int | None) -> Array:
        if selected_mode is None:
            return node.mode_priors
        if selected_mode < 0 or selected_mode >= len(node.mode_priors):
            raise IndexError("mode_index 超出状态模型的模态范围")
        weights = np.zeros_like(node.mode_priors)
        weights[selected_mode] = 1.0
        return weights

    @staticmethod
    def _frame_reliability(features: RuntimeFeatures, frame: str) -> float:
        if not features.frame_visibility.get(frame, False):
            return 0.0
        return features.tracking_reliability.get(frame, 0.0)

    def _trajectory_weight(
        self,
        frame: str,
        features: RuntimeFeatures,
        relations: Mapping[str, RelationEstimate],
    ) -> float:
        reliability = self._frame_reliability(features, frame)
        if reliability <= 0.0:
            return 0.0
        if frame.startswith("virtual_skill_"):
            return reliability
        estimate = relations.get(frame)
        if estimate is None:
            return 0.0
        # Robot-trajectory evidence consumes the continuous relation belief,
        # not the thresholded relation decision.  ``Unknown`` means that the
        # current posterior cannot yet be collapsed to external/linked; it
        # does not erase a visible, reliable frame pose.  Weighting by the
        # soft external mass keeps this scorer consistent with a DEFER stream
        # that is safely down-weighted in the action PoE, while uncertainty is
        # still represented continuously.  Missing/hidden/unreliable frames
        # remain excluded by ``reliability`` above.  Discrete relation
        # compatibility deliberately retains its stricter Unknown gate.
        return reliability * estimate.external

    def _robot_score(
        self,
        node: StateNode,
        features: RuntimeFeatures,
        relations: Mapping[str, RelationEstimate],
        selected_mode: int | None,
        covariance_inflation: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, tuple[GaussianComponentAudit, ...]],
        bool,
        float,
        float,
        float,
    ]:
        mode_weights = self._mode_weights(node, selected_mode)
        # ``terms`` are the weighted peak-normalized log supports actually
        # consumed by the progress posterior.
        terms: dict[str, float] = {}
        raw_terms: dict[str, float] = {}
        weights: dict[str, float] = {}
        audits: dict[str, tuple[GaussianComponentAudit, ...]] = {}
        compatibility_log_sum = 0.0
        compatibility_weight = 0.0
        for frame in node.selected_frames:
            if frame not in features.relative_poses:
                continue
            weight = self._trajectory_weight(frame, features, relations)
            if weight <= 0.0:
                continue
            mode_raw_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_support_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_mahalanobis = np.zeros(len(mode_weights), dtype=np.float64)
            mode_logdet = np.zeros(len(mode_weights), dtype=np.float64)
            available_modes = np.zeros(len(mode_weights), dtype=np.float64)
            for mode, mode_weight in enumerate(mode_weights):
                if mode_weight <= 0.0 or frame not in node.mode_selected_frames[mode]:
                    continue
                covariance = node.stream_covariances[frame][mode]
                if covariance_inflation > 0.0:
                    covariance = covariance + np.eye(6) * covariance_inflation
                support_log, raw_log, mahalanobis, logdet = _gaussian_pose_terms(
                    node.stream_means[frame][mode],
                    covariance,
                    features.relative_poses[frame],
                )
                mode_raw_logs[mode] = raw_log
                mode_support_logs[mode] = support_log
                mode_mahalanobis[mode] = mahalanobis
                mode_logdet[mode] = logdet
                available_modes[mode] = mode_weight
            if not np.any(available_modes > 0.0):
                continue
            raw_log_likelihood = _logsumexp(mode_raw_logs, available_modes)
            log_support = _logsumexp(mode_support_logs, available_modes)
            normalized_mode_weights = available_modes / np.sum(available_modes)
            audits[frame] = tuple(
                GaussianComponentAudit(
                    mode_index=mode,
                    mode_weight=float(normalized_mode_weights[mode]),
                    dimension=6,
                    raw_log_likelihood=float(mode_raw_logs[mode]),
                    normalized_log_support=float(mode_support_logs[mode]),
                    mahalanobis_squared=float(mode_mahalanobis[mode]),
                    covariance_log_determinant=float(mode_logdet[mode]),
                )
                for mode in range(len(mode_weights))
                if available_modes[mode] > 0.0
            )
            terms[frame] = weight * log_support
            raw_terms[frame] = raw_log_likelihood
            weights[frame] = weight
            compatibility_log_sum += weight * log_support
            compatibility_weight += weight
        support_total = float(sum(terms.values()))
        raw_total = float(sum(weights[frame] * raw_terms[frame] for frame in raw_terms))
        compatibility = (
            1.0
            if compatibility_weight <= 0.0
            else float(
                math.exp(max(-750.0, compatibility_log_sum / compatibility_weight))
            )
        )
        attainable_peak_log_support = 0.0
        peak_normalized_compatibility = compatibility
        # The integrated policy fixes one DynaMAC/MiDiGaP mode per skill.
        # In that case the weighted transformed marginals define the same PoE
        # action queried by the controller.  Its joint optimum is the
        # attainable peak of L_robot; individual stream peaks generally cannot
        # all be reached by one end-effector pose.  Correct only the absolute
        # plausibility scale.  The per-stream support sum returned above—and
        # therefore progress-state ranking—remains byte-for-byte unchanged.
        if selected_mode is not None and compatibility_weight > 0.0:
            peak_entries = []
            for frame, weight in weights.items():
                if weight <= 0.0 or frame not in features.frame_poses:
                    continue
                if frame not in node.mode_selected_frames[selected_mode]:
                    continue
                covariance = node.stream_covariances[frame][selected_mode]
                if covariance_inflation > 0.0:
                    covariance = covariance + np.eye(6) * covariance_inflation
                peak_entries.append(
                    (
                        frame,
                        features.frame_poses[frame],
                        node.stream_means[frame][selected_mode],
                        covariance,
                        features.relative_poses[frame],
                        weight,
                    )
                )
            if peak_entries:
                (
                    peak_normalized_compatibility,
                    _,
                    attainable_peak_log_support,
                ) = joint_peak_normalized_pose_support(
                    peak_entries,
                    diagonalize=(
                        self.task_model.base_policy.config.diagonalize_transformed_covariance
                    ),
                )
        joint_support_total = min(0.0, support_total - attainable_peak_log_support)
        return (
            joint_support_total,
            raw_total,
            compatibility,
            terms,
            raw_terms,
            weights,
            audits,
            compatibility_weight > 0.0,
            peak_normalized_compatibility,
            float(attainable_peak_log_support),
            support_total,
        )

    def _factor_observation(
        self,
        factor_id: FactorId,
        features: RuntimeFeatures,
    ) -> tuple[Array | None, float]:
        if factor_id.kind == "node":
            assert factor_id.feature is not None
            value = features.entity_configurations.get(factor_id.source, {}).get(
                factor_id.feature
            )
            if value is None:
                return None, 0.0
            # Internal state is usable even when a pose tracker is absent.  If
            # the same entity has an explicit tracker, its quality still gates it.
            if factor_id.source in features.frame_visibility:
                reliability = self._frame_reliability(features, factor_id.source)
            else:
                reliability = 1.0
            return value, reliability

        assert factor_id.target is not None
        if (
            factor_id.source not in features.frame_poses
            or factor_id.target not in features.frame_poses
        ):
            return None, 0.0
        reliability = min(
            self._frame_reliability(features, factor_id.source),
            self._frame_reliability(features, factor_id.target),
        )
        if reliability <= 0.0:
            return None, 0.0
        value = relative_pose(
            features.frame_poses[factor_id.target],
            features.frame_poses[factor_id.source],
        )
        return value, reliability

    @staticmethod
    def _distribution_terms(
        distribution: FactorDistribution,
        value: Array,
    ) -> tuple[float, float, float, float]:
        current = np.asarray(value, dtype=np.float64)
        residual = (
            pose_log_nearest(distribution.mean, current)
            if distribution.space == "se3"
            else current - distribution.mean
        )
        return _gaussian_terms(residual, distribution.covariance)

    def _scene_score(
        self,
        node: StateNode,
        features: RuntimeFeatures,
        selected_mode: int | None,
    ) -> tuple[
        float,
        float,
        float,
        dict[FactorId, float],
        dict[FactorId, float],
        dict[FactorId, float],
        dict[FactorId, tuple[GaussianComponentAudit, ...]],
        bool,
        bool,
    ]:
        mode_weights = self._mode_weights(node, selected_mode)
        # As for robot trajectories, scene ``terms`` contain only normalized
        # support used online; complete densities remain separate audit data.
        terms: dict[FactorId, float] = {}
        raw_terms: dict[FactorId, float] = {}
        weights: dict[FactorId, float] = {}
        audits: dict[FactorId, tuple[GaussianComponentAudit, ...]] = {}
        compatibility_log_sum = 0.0
        compatibility_weight = 0.0
        expected = False
        for factor_id, distributions in node.scene_factor_models.items():
            applicable = np.asarray(
                [
                    mode_weights[mode] if mode in distributions else 0.0
                    for mode in range(len(mode_weights))
                ],
                dtype=np.float64,
            )
            if not np.any(applicable > 0.0):
                continue
            expected = True
            value, reliability = self._factor_observation(factor_id, features)
            if value is None or reliability <= 0.0:
                continue
            mode_raw_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_support_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_mahalanobis = np.zeros(len(mode_weights), dtype=np.float64)
            mode_logdet = np.zeros(len(mode_weights), dtype=np.float64)
            mode_dimensions = np.zeros(len(mode_weights), dtype=np.int64)
            for mode, mode_weight in enumerate(applicable):
                if mode_weight <= 0.0:
                    continue
                (
                    mode_support_logs[mode],
                    mode_raw_logs[mode],
                    mode_mahalanobis[mode],
                    mode_logdet[mode],
                ) = self._distribution_terms(distributions[mode], value)
                mode_dimensions[mode] = len(distributions[mode].covariance)
            raw_log_likelihood = _logsumexp(mode_raw_logs, applicable)
            log_support = _logsumexp(mode_support_logs, applicable)
            normalized = applicable / np.sum(applicable)
            audits[factor_id] = tuple(
                GaussianComponentAudit(
                    mode_index=mode,
                    mode_weight=float(normalized[mode]),
                    dimension=int(mode_dimensions[mode]),
                    raw_log_likelihood=float(mode_raw_logs[mode]),
                    normalized_log_support=float(mode_support_logs[mode]),
                    mahalanobis_squared=float(mode_mahalanobis[mode]),
                    covariance_log_determinant=float(mode_logdet[mode]),
                )
                for mode in range(len(mode_weights))
                if applicable[mode] > 0.0
            )
            terms[factor_id] = reliability * log_support
            raw_terms[factor_id] = raw_log_likelihood
            weights[factor_id] = reliability
            compatibility_log_sum += reliability * log_support
            compatibility_weight += reliability
        support_total = float(sum(terms.values()))
        raw_total = float(
            sum(weights[factor] * raw_terms[factor] for factor in raw_terms)
        )
        compatibility = (
            1.0
            if compatibility_weight <= 0.0
            else float(
                math.exp(max(-750.0, compatibility_log_sum / compatibility_weight))
            )
        )
        return (
            support_total,
            raw_total,
            compatibility,
            terms,
            raw_terms,
            weights,
            audits,
            expected,
            compatibility_weight > 0.0,
        )

    def _relation_score(
        self,
        node: StateNode,
        features: RuntimeFeatures,
        relations: Mapping[str, RelationEstimate],
        selected_mode: int | None,
    ) -> tuple[
        float,
        float,
        float,
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        mode_weights = self._mode_weights(node, selected_mode)
        terms: dict[str, float] = {}
        peak_normalized_terms: dict[str, float] = {}
        weights: dict[str, float] = {}
        direction_compatible = True
        for frame, estimate in relations.items():
            if estimate.decision_state == RelationDecision.UNKNOWN:
                continue
            reliability = self._frame_reliability(features, frame)
            if reliability <= 0.0 or frame not in node.demo_relation_priors:
                continue
            demo_prior = np.sum(
                mode_weights[:, None] * node.demo_relation_priors[frame], axis=0
            )
            compatibility = max(
                float(np.dot(estimate.posterior, demo_prior)),
                self.config.probability_floor,
            )
            # ``compatibility`` is the raw discrete overlap consumed by the
            # progress posterior.  Its maximum is ``max(demo_prior)``, not one,
            # whenever the demonstration prior is deliberately soft.  Divide
            # only the absolute plausibility support by that attainable peak so
            # all evidence families share a unit best-match scale.  This does
            # not change relation filtering or candidate-state ranking.
            attainable_peak = max(
                float(np.max(demo_prior)), self.config.probability_floor
            )
            peak_normalized = float(np.clip(compatibility / attainable_peak, 0.0, 1.0))
            # A LINK_PENDING interval is deliberately only a soft hypothesis:
            # the demonstrations contain a repeatable close but not enough
            # post-close excitation to confirm the physical relation.  Its
            # soft overlap remains useful in the progress posterior, but it
            # must not be promoted to the same absolute direction constraint
            # as a confirmed LINK origin.  If that relation is actually
            # required, the role/boundary layer requests verification and the
            # recovery reentry layer separately preserves the repaired goal.
            directional_modes = tuple(
                mode
                for mode, weight in enumerate(mode_weights)
                if weight > 0.0
                and node.demo_relation_priors[frame][mode, 1]
                > node.demo_relation_priors[frame][mode, 0]
            )
            pending_only_link_hypothesis = bool(directional_modes) and all(
                RelationStateKey(
                    self.task_model.arm_id,
                    frame,
                    node.state_id,
                    mode,
                )
                not in self.task_model.link_origins
                and (
                    (
                        candidate := self.task_model.active_link_pending_candidate(
                            frame,
                            node.state_id,
                            {node.state_id.skill_index: mode},
                        )
                    )
                    is not None
                    and candidate.event_id.mode == mode
                )
                for mode in directional_modes
            )
            if not pending_only_link_hypothesis and not np.isclose(
                demo_prior[0], demo_prior[1]
            ):
                expected = (
                    RelationDecision.LINKED
                    if demo_prior[1] > demo_prior[0]
                    else RelationDecision.EXTERNAL
                )
                if estimate.decision_state != expected:
                    direction_compatible = False
            terms[frame] = reliability * math.log(compatibility)
            peak_normalized_terms[frame] = reliability * math.log(
                max(peak_normalized, self.config.probability_floor)
            )
            weights[frame] = reliability
        total = float(sum(terms.values()))
        peak_normalized_total = float(sum(peak_normalized_terms.values()))
        total_weight = float(sum(weights.values()))
        compatibility = (
            1.0
            if total_weight <= 0.0
            else float(math.exp(max(-750.0, total / total_weight)))
        )
        peak_normalized_compatibility = (
            1.0
            if total_weight <= 0.0
            else float(math.exp(max(-750.0, peak_normalized_total / total_weight)))
        )
        state_compatibility = (
            peak_normalized_compatibility if direction_compatible else 0.0
        )
        return (
            total,
            compatibility,
            peak_normalized_compatibility,
            state_compatibility,
            terms,
            peak_normalized_terms,
            weights,
        )

    def evaluate(
        self,
        state_id: StateId,
        features: RuntimeFeatures,
        relations: Mapping[str, RelationEstimate],
        *,
        mode_by_skill: Mapping[int, int] | None = None,
        robot_covariance_inflation: float = 0.0,
    ) -> CandidateScore:
        covariance_inflation = float(robot_covariance_inflation)
        if not math.isfinite(covariance_inflation) or covariance_inflation < 0.0:
            raise ValueError("机器人轨迹评分协方差放宽量必须为有限非负数")
        node = self.task_model.state(state_id)
        selected_mode = (
            None if mode_by_skill is None else mode_by_skill.get(state_id.skill_index)
        )
        (
            robot_support_log,
            robot_raw_log,
            robot_compat,
            robot_terms,
            robot_raw_terms,
            robot_weights,
            robot_audits,
            robot_available,
            robot_peak_normalized_compat,
            robot_attainable_peak_log_support,
            robot_unadjusted_log_support,
        ) = self._robot_score(
            node,
            features,
            relations,
            selected_mode,
            covariance_inflation,
        )
        (
            state_support_log,
            state_raw_log,
            state_compat,
            scene_terms,
            scene_raw_terms,
            scene_weights,
            scene_audits,
            scene_expected,
            scene_available,
        ) = self._scene_score(node, features, selected_mode)
        (
            relation_log,
            relation_compat,
            relation_peak_normalized_compat,
            relation_state_compat,
            relation_terms,
            relation_peak_normalized_terms,
            relation_weights,
        ) = self._relation_score(node, features, relations, selected_mode)
        explanation_log = (
            robot_support_log
            + self.config.scene_weight * state_support_log
            + self.config.relation_weight * relation_log
        )
        normalized = (
            robot_peak_normalized_compat
            * state_compat**self.config.scene_weight
            * relation_peak_normalized_compat**self.config.relation_weight
        )
        return CandidateScore(
            state_id=state_id,
            robot_log_likelihood=robot_raw_log,
            state_log_likelihood=state_raw_log,
            robot_log_support=robot_support_log,
            robot_unadjusted_log_support=robot_unadjusted_log_support,
            state_log_support=state_support_log,
            relation_log_compatibility=relation_log,
            explanation_log_score=explanation_log,
            normalized_explanation_score=float(normalized),
            robot_compatibility=robot_compat,
            state_compatibility=state_compat,
            relation_compatibility=relation_compat,
            robot_covariance_inflation=covariance_inflation,
            robot_peak_normalized_compatibility=(robot_peak_normalized_compat),
            relation_peak_normalized_compatibility=(relation_peak_normalized_compat),
            relation_state_compatibility=relation_state_compat,
            robot_attainable_peak_log_support=(robot_attainable_peak_log_support),
            robot_frame_terms=robot_terms,
            robot_frame_weights=robot_weights,
            scene_factor_terms=scene_terms,
            scene_factor_weights=scene_weights,
            relation_frame_terms=relation_terms,
            relation_frame_peak_normalized_terms=(relation_peak_normalized_terms),
            relation_frame_weights=relation_weights,
            robot_frame_raw_log_likelihoods=robot_raw_terms,
            scene_factor_raw_log_likelihoods=scene_raw_terms,
            robot_frame_gaussian_audits=robot_audits,
            scene_factor_gaussian_audits=scene_audits,
            robot_evidence_available=robot_available,
            scene_evidence_expected=scene_expected,
            scene_evidence_available=scene_available,
        )

    def evaluate_many(
        self,
        states: tuple[StateId, ...] | list[StateId],
        features: RuntimeFeatures,
        relations: Mapping[str, RelationEstimate],
        *,
        mode_by_skill: Mapping[int, int] | None = None,
        robot_covariance_inflation: float = 0.0,
    ) -> dict[StateId, CandidateScore]:
        return {
            state: self.evaluate(
                state,
                features,
                relations,
                mode_by_skill=mode_by_skill,
                robot_covariance_inflation=robot_covariance_inflation,
            )
            for state in states
        }


__all__ = [
    "CandidateScore",
    "GaussianComponentAudit",
    "StateEvaluator",
    "StateEvaluatorConfig",
    "joint_peak_normalized_pose_support",
    "joint_poe_pose_support",
]
