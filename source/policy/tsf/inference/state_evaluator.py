"""Auditable robot/scene/relation scoring for candidate progress states."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ...dynamac import (
    pose_compose,
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
from ..model.task_model import TSFTaskModel, StateNode

Array = np.ndarray


def joint_peak_normalized_pose_support(
    entries: list[tuple[str, Array, Array, Array, Array, Array, float]],
    *,
    diagonalize: bool,
) -> tuple[float, float, float]:
    """Score one observed EE pose under the executable fused PoE distribution.

    Each entry is ``(frame, frame_pose, local_mean, action_covariance,
    support_covariance, observed_local_pose, weight)``.  The action
    covariances construct exactly the PoE mean and covariance used by the
    controller.  The support-covariance increment represents runtime
    observation uncertainty; it is added once to the fused covariance because
    there is one physical end effector, rather than multiplied once per
    correlated reference stream.  This keeps action generation and progress
    explanation on the same geometric state without feeding executor outcomes
    back into task progress.
    """

    if not entries:
        return 1.0, 0.0, 0.0
    marginals = []
    weights = []
    support_increments = []
    observed_world_pose = None
    for (
        frame,
        frame_pose,
        mean,
        action_covariance,
        support_covariance,
        value,
        weight,
    ) in entries:
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight) or numeric_weight <= 0.0:
            raise ValueError("联合位姿支持权重必须为有限正数")
        weights.append(numeric_weight)
        action_marginal = transform_marginal(
            frame,
            frame_pose,
            mean,
            action_covariance,
            diagonalize=diagonalize,
        )
        support_marginal = transform_marginal(
            frame,
            frame_pose,
            mean,
            support_covariance,
            diagonalize=diagonalize,
        )
        marginals.append(action_marginal)
        support_increments.append(
            support_marginal.covariance - action_marginal.covariance
        )
        if observed_world_pose is None:
            observed_world_pose = pose_compose(frame_pose, value)
    joint_mean, joint_covariance, _ = product_of_experts(
        marginals,
        precision_weights=weights,
    )
    assert observed_world_pose is not None
    total_weight = float(sum(weights))
    observation_covariance = sum(
        weight * increment
        for weight, increment in zip(weights, support_increments, strict=True)
    ) / total_weight
    effective_covariance = joint_covariance + observation_covariance
    normalized_log, _, _, _ = _gaussian_pose_terms(
        joint_mean,
        effective_covariance,
        observed_world_pose,
    )
    compatibility = float(math.exp(max(-750.0, normalized_log)))
    return compatibility, float(normalized_log), 0.0


def robot_pose_observation_covariance(
    runtime_variance: float,
    robot_pose_terms: int,
    *,
    dimension: int = 6,
) -> Array | None:
    """Propagate independent runtime robot-pose uncertainty into a factor.

    ``runtime_variance`` is one fixed, task-independent per-end-effector noise
    term of the shared low-level controller.  A relative pose that contains
    one robot pose receives one copy; an end-effector-to-end-effector edge
    receives two.  Pure environment factors receive none.
    """

    variance = float(runtime_variance)
    terms = int(robot_pose_terms)
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("机器人运行时方差必须为有限非负数")
    if terms < 0:
        raise ValueError("机器人位姿不确定性项数不能为负")
    if dimension <= 0:
        raise ValueError("观测协方差维数必须为正数")
    if variance == 0.0 or terms == 0:
        return None
    return np.eye(dimension, dtype=np.float64) * variance * terms


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
    effective_raw_log_likelihood: float | None = None
    effective_normalized_log_support: float | None = None
    effective_mahalanobis_squared: float | None = None
    effective_covariance_log_determinant: float | None = None


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
    # Sum of the per-stream supports retained for component diagnostics; it is
    # not an additional progress condition.
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
    # One task-independent execution/observation variance is convolved with
    # the demonstration distribution during runtime scoring.  The original
    # demonstration density remains available in the Gaussian audits above.
    robot_runtime_variance: float = 0.0
    # ``robot_compatibility`` is the geometric mean of the individual stream
    # diagnostics.  Progress uses the unified value below: one fused world
    # distribution for executable external streams plus relative-pose
    # monitoring evidence for linked streams.
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
    # Successful demonstrations describe task variation, while the deployed
    # controller adds ordinary servo/contact response dispersion.  Runtime
    # state matching therefore uses Sigma_demo + lambda_runtime I.  This is a
    # likelihood noise model, not an additional progress gate.
    robot_runtime_variance: float = 4.0e-5

    def __post_init__(self) -> None:
        if self.scene_weight < 0.0 or self.relation_weight < 0.0:
            raise ValueError("进度证据权重必须非负")
        if self.probability_floor <= 0.0:
            raise ValueError("概率下限必须为正数")
        if (
            not math.isfinite(self.robot_runtime_variance)
            or self.robot_runtime_variance < 0.0
        ):
            raise ValueError("机器人运行时方差必须为有限非负数")


class StateEvaluator:
    def __init__(
        self,
        task_model: TSFTaskModel,
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
        node: StateNode,
        frame: str,
        features: RuntimeFeatures,
        selected_mode: int | None,
    ) -> float:
        reliability = self._frame_reliability(features, frame)
        if reliability <= 0.0:
            return 0.0
        if frame.startswith("virtual_skill_"):
            return reliability
        return reliability

    def _trajectory_is_executable(
        self,
        node: StateNode,
        frame: str,
        selected_mode: int | None,
    ) -> bool:
        if frame.startswith("virtual_skill_"):
            return True
        priors = node.demo_relation_priors.get(frame)
        if priors is None:
            return True
        mode_weights = self._mode_weights(node, selected_mode)
        expected = np.sum(mode_weights[:, None] * priors, axis=0)
        return bool(expected[0] > expected[1])

    def _stream_runtime_covariance(self, frame: str) -> Array | None:
        # Every local stream contains this model's controlled end effector.
        # A peer ``*_ee`` frame contributes a second independently controlled
        # robot pose; ordinary object, goal, and virtual frames do not.
        peer_robot_term = int(
            frame.endswith("_ee") and frame != f"{self.task_model.arm_id}_ee"
        )
        return robot_pose_observation_covariance(
            self.config.robot_runtime_variance,
            1 + peer_robot_term,
        )

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
        executable_frames: set[str] = set()
        compatibility_log_sum = 0.0
        compatibility_weight = 0.0
        # A local EE distribution is valid continuous progress evidence only
        # when the normal-demo action-relevance audit retained that expert.
        # Candidates rejected by the base policy's task-parameter selector remain
        # available to the independent
        # relation/event and recovery models; reusing their accidental EE
        # correlation here would make progress stricter than the action model
        # that actually controls the robot.
        for frame in node.action_relevant_frames:
            if frame not in features.relative_poses:
                continue
            weight = self._trajectory_weight(
                node,
                frame,
                features,
                selected_mode,
            )
            if weight <= 0.0:
                continue
            mode_raw_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_support_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_original_support_logs = np.zeros(
                len(mode_weights), dtype=np.float64
            )
            mode_mahalanobis = np.zeros(len(mode_weights), dtype=np.float64)
            mode_logdet = np.zeros(len(mode_weights), dtype=np.float64)
            mode_effective_raw_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_effective_support_logs = np.zeros(len(mode_weights), dtype=np.float64)
            mode_effective_mahalanobis = np.zeros(
                len(mode_weights), dtype=np.float64
            )
            mode_effective_logdet = np.zeros(len(mode_weights), dtype=np.float64)
            available_modes = np.zeros(len(mode_weights), dtype=np.float64)
            for mode, mode_weight in enumerate(mode_weights):
                if (
                    mode_weight <= 0.0
                    or frame not in node.mode_action_relevant_frames[mode]
                ):
                    continue
                original_covariance = node.stream_covariances[frame][mode]
                (
                    original_support_log,
                    original_raw_log,
                    original_mahalanobis,
                    original_logdet,
                ) = _gaussian_pose_terms(
                    node.stream_means[frame][mode],
                    original_covariance,
                    features.relative_poses[frame],
                )
                runtime_covariance = self._stream_runtime_covariance(frame)
                effective_covariance = original_covariance
                if runtime_covariance is not None:
                    effective_covariance = effective_covariance + runtime_covariance
                if covariance_inflation > 0.0:
                    effective_covariance = effective_covariance + (
                        np.eye(6, dtype=np.float64) * covariance_inflation
                    )
                support_log, raw_log, mahalanobis, logdet = _gaussian_pose_terms(
                    node.stream_means[frame][mode],
                    effective_covariance,
                    features.relative_poses[frame],
                )
                mode_raw_logs[mode] = original_raw_log
                mode_support_logs[mode] = support_log
                mode_original_support_logs[mode] = original_support_log
                mode_mahalanobis[mode] = original_mahalanobis
                mode_logdet[mode] = original_logdet
                mode_effective_raw_logs[mode] = raw_log
                mode_effective_support_logs[mode] = support_log
                mode_effective_mahalanobis[mode] = mahalanobis
                mode_effective_logdet[mode] = logdet
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
                    normalized_log_support=float(mode_original_support_logs[mode]),
                    mahalanobis_squared=float(mode_mahalanobis[mode]),
                    covariance_log_determinant=float(mode_logdet[mode]),
                    effective_raw_log_likelihood=float(
                        mode_effective_raw_logs[mode]
                    ),
                    effective_normalized_log_support=float(
                        mode_effective_support_logs[mode]
                    ),
                    effective_mahalanobis_squared=float(
                        mode_effective_mahalanobis[mode]
                    ),
                    effective_covariance_log_determinant=float(
                        mode_effective_logdet[mode]
                    ),
                )
                for mode in range(len(mode_weights))
                if available_modes[mode] > 0.0
            )
            terms[frame] = weight * log_support
            raw_terms[frame] = raw_log_likelihood
            weights[frame] = weight
            if self._trajectory_is_executable(node, frame, selected_mode):
                executable_frames.add(frame)
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
        monitor_frames = set(weights).difference(executable_frames)
        monitor_weight = float(sum(weights[frame] for frame in monitor_frames))
        monitor_support = (
            0.0
            if monitor_weight <= 0.0
            else float(sum(terms[frame] for frame in monitor_frames) / monitor_weight)
        )
        monitor_compatibility = float(math.exp(max(-750.0, monitor_support)))
        joint_support_total = monitor_support
        attainable_peak_log_support = 0.0
        peak_normalized_compatibility = monitor_compatibility
        # With a fixed skill mode, score the single physical EE directly under
        # the action-relevant fused PoE.  Per-frame terms above remain audit
        # diagnostics only; they are not additional progress gates.
        if selected_mode is not None and compatibility_weight > 0.0:
            peak_entries = []
            for frame, weight in weights.items():
                if (
                    weight <= 0.0
                    or frame not in executable_frames
                    or frame not in features.frame_poses
                ):
                    continue
                if frame not in node.mode_action_relevant_frames[selected_mode]:
                    continue
                action_covariance = node.stream_covariances[frame][selected_mode]
                support_covariance = action_covariance
                runtime_covariance = self._stream_runtime_covariance(frame)
                if runtime_covariance is not None:
                    support_covariance = support_covariance + runtime_covariance
                if covariance_inflation > 0.0:
                    support_covariance = support_covariance + (
                        np.eye(6, dtype=np.float64) * covariance_inflation
                    )
                peak_entries.append(
                    (
                        frame,
                        features.frame_poses[frame],
                        node.stream_means[frame][selected_mode],
                        action_covariance,
                        support_covariance,
                        features.relative_poses[frame],
                        weight,
                    )
                )
            if peak_entries:
                (
                    executable_compatibility,
                    executable_support,
                    attainable_peak_log_support,
                ) = joint_peak_normalized_pose_support(
                    peak_entries,
                    diagonalize=(
                        self.task_model.base_policy.config.diagonalize_transformed_covariance
                    ),
                )
                joint_support_total += executable_support
                peak_normalized_compatibility *= executable_compatibility
        joint_support_total = min(0.0, joint_support_total)
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
            robot_runtime_variance=self.config.robot_runtime_variance,
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
]
