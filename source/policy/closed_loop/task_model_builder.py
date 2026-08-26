"""Build phase-one closed-loop statistics from the same successful demos."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from ..dynamac import (
    BimanualDynaMAC,
    DynaMAC,
    DynaMACDemonstration,
    _fit_pose_sequence,
    _resample_poses,
    _resample_rows,
    _resampled_skill_data,
    _skill_slice,
    _validate_demonstrations,
    geometric_mean_standard_deviation,
    pose_log_world,
    relative_pose,
    synchronized_bimanual_demonstrations,
)
from .boundary_model import (
    BoundaryId,
    BoundaryModel,
    LocalCompletionModel,
    RelationGuardDistribution,
    ReliabilityStatistics,
)
from .relation_events import (
    LinkPendingCandidate,
    LinkRecoveryAnchor,
    RelationEventId,
    RelationStateKey,
    UnlinkEventMetadata,
)
from .scene_factors import FactorDistribution, FactorId, fit_factor_distribution
from .state_index import StateId, build_state_topology
from .task_model import ClosedLoopTaskModel, StateNode

Array = np.ndarray


@dataclass(frozen=True)
class ClosedLoopTaskModelConfig:
    """Only phase-one statistical choices absent from the base checkpoint."""

    relation_alpha: float = 6.0
    relation_epsilon: float = 1.0e-12
    relation_link_threshold: float = 0.7
    relation_unlink_threshold: float = 0.3
    relation_minimum_dwell: int = 2
    relation_event_support: float = 0.8
    relation_event_alignment_tolerance: int = 2
    relation_minimum_excitation: float = 3.0e-7
    relation_comotion_residual_threshold: float = 1.0e-5
    relation_comotion_residual_ratio: float = 0.1
    link_anchor_horizon: int = 16
    reentry_horizon: int = 4
    scene_max_neighborhood: int = 2
    scene_recognition_neighborhood: int = 2
    scene_min_support: float = 0.8
    scene_min_stable_fraction: float = 0.8
    scene_support_confidence: float = 0.99
    scene_min_loo_gain: float = 1.0e-6
    scene_min_loo_accuracy: float = 0.8
    progress_scene_weight: float = 1.0
    boundary_terminal_window: int = 3
    # Boundary support is evaluated on the soft relation distribution.  Keep
    # it aligned with the stable binary relation decision threshold; cross-
    # demonstration consistency is enforced separately and remains 5/5.
    boundary_relation_support: float = 0.7
    boundary_scene_min_stable_fraction: float = 1.0
    boundary_compatibility_margin: float = 1.0e-6
    transaction_boundary_tolerance: float = 0.03

    def __post_init__(self) -> None:
        probabilities = (
            self.relation_link_threshold,
            self.relation_unlink_threshold,
            self.relation_event_support,
            self.scene_min_loo_accuracy,
            self.scene_min_support,
            self.scene_min_stable_fraction,
            self.scene_support_confidence,
            self.boundary_relation_support,
            self.boundary_scene_min_stable_fraction,
            self.transaction_boundary_tolerance,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("阶段一概率阈值必须位于 [0,1]")
        if self.relation_unlink_threshold >= self.relation_link_threshold:
            raise ValueError("unlink 阈值必须小于 link 阈值")
        if not 0.0 <= self.relation_comotion_residual_ratio <= 1.0:
            raise ValueError("共动残差比例阈值必须位于 [0,1]")
        if self.scene_support_confidence <= 0.5 or self.scene_support_confidence >= 1.0:
            raise ValueError("scene 支持域置信度必须位于 (0.5,1)")
        if self.scene_max_neighborhood not in {0, 1, 2}:
            raise ValueError("scene 最大邻域半径必须为 0、1 或 2")
        if self.scene_recognition_neighborhood < 1:
            raise ValueError("scene 局部识别邻域必须为正整数")
        if (
            self.relation_minimum_excitation < 0.0
            or self.relation_comotion_residual_threshold < 0.0
            or self.progress_scene_weight < 0.0
            or self.boundary_compatibility_margin < 0.0
        ):
            raise ValueError("评分权重和边界兼容度容差必须非负")
        if any(
            value < 1
            for value in (
                self.relation_minimum_dwell,
                self.relation_event_alignment_tolerance,
                self.link_anchor_horizon,
                self.reentry_horizon,
                self.boundary_terminal_window,
            )
        ):
            raise ValueError("阶段一窗口长度必须为正整数")


@dataclass(frozen=True)
class _AlignedSkillData:
    ee_pose: Array
    # Transient replay input only.  It is not serialized into StateNode or
    # ClosedLoopTaskModel; offline evaluators use it to reconstruct the action
    # that actually produced the next recorded observation.
    action_pose: Array
    frames: dict[str, Array]
    gripper: Array
    scene_entity_poses: dict[str, Array]
    entity_configurations: dict[str, dict[str, Array]]
    structural_bindings: dict[str, str]


def _mode_members(skill: Any, mode: int) -> Array:
    return np.asarray(skill.mode_demonstration_indices[mode], dtype=np.int64)


def _fit_pose_samples(samples: Array, policy: DynaMAC) -> tuple[Array, Array]:
    mean, covariance = _fit_pose_sequence(
        np.asarray(samples, dtype=np.float64)[:, None, :],
        policy.config.position_variance_floor,
        policy.config.rotation_variance_floor,
        covariance_estimation_method=policy.config.covariance_estimation_method,
    )
    return mean[0], covariance[0]


def _relative_batch(frame: Array, pose: Array) -> Array:
    return np.stack(
        [
            relative_pose(current_frame, current_pose)
            for current_frame, current_pose in zip(frame, pose, strict=True)
        ]
    )


def _relation_prior(
    scale: float, config: ClosedLoopTaskModelConfig, tau_m: float
) -> Array:
    logit = config.relation_alpha * (
        math.log(tau_m) - math.log(scale + config.relation_epsilon)
    )
    linked = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -60.0, 60.0))))
    return np.asarray([1.0 - linked, linked], dtype=np.float64)


def _logmeanexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.mean(np.exp(array - maximum))))


class ClosedLoopTaskModelBuilder:
    """Build one arm's unified model without refitting its base DynaMAC."""

    def __init__(
        self, config: ClosedLoopTaskModelConfig = ClosedLoopTaskModelConfig()
    ) -> None:
        self.config = config

    def _align_demonstrations(
        self,
        policy: DynaMAC,
        demonstrations: Sequence[DynaMACDemonstration],
    ) -> dict[int, _AlignedSkillData]:
        frame_names, skill_sequence = _validate_demonstrations(demonstrations)
        if frame_names != policy.frame_names or skill_sequence != policy.skill_sequence:
            raise ValueError("示范的参考系或技能序列与基础 DynaMAC 不一致")
        configuration_schema = {
            entity: {
                field_name: values.shape[1] for field_name, values in fields.items()
            }
            for entity, fields in demonstrations[0].entity_configurations.items()
        }
        scene_pose_names = set(demonstrations[0].scene_entity_poses)
        structural_bindings = demonstrations[0].structural_bindings
        for demonstration in demonstrations[1:]:
            current_schema = {
                entity: {
                    field_name: values.shape[1] for field_name, values in fields.items()
                }
                for entity, fields in demonstration.entity_configurations.items()
            }
            if current_schema != configuration_schema:
                raise ValueError("所有示范必须包含相同的实体内部构型字段")
            if set(demonstration.scene_entity_poses) != scene_pose_names:
                raise ValueError("所有示范必须包含相同的附加场景实体位姿")
            if demonstration.structural_bindings != structural_bindings:
                raise ValueError("所有示范必须包含相同的直接结构绑定")
        aligned: dict[int, _AlignedSkillData] = {}
        virtual_starts: dict[int, list[Array]] = {}
        for skill_index, (label, skill) in enumerate(
            zip(skill_sequence, policy.skills, strict=True)
        ):
            virtual_starts[label] = [
                demo.ee_pose[_skill_slice(demo, label)[0]].copy()
                for demo in demonstrations
            ]
            ee, action, frames, extra = _resampled_skill_data(
                demonstrations,
                label,
                skill.duration,
                virtual_starts,
                policy.config.resampling_method,
            )
            scene_entity_poses = {
                name: np.stack(
                    [
                        _resample_poses(
                            demonstration.scene_entity_poses[name][
                                _skill_slice(demonstration, label)
                            ],
                            skill.duration,
                            policy.config.resampling_method,
                        )
                        for demonstration in demonstrations
                    ]
                )
                for name in scene_pose_names
            }
            entity_configurations = {
                entity: {
                    field_name: np.stack(
                        [
                            _resample_rows(
                                demonstration.entity_configurations[entity][field_name][
                                    _skill_slice(demonstration, label)
                                ],
                                skill.duration,
                                policy.config.resampling_method,
                            )
                            for demonstration in demonstrations
                        ]
                    )
                    for field_name in fields
                }
                for entity, fields in configuration_schema.items()
            }
            aligned[skill_index] = _AlignedSkillData(
                ee_pose=ee,
                action_pose=action,
                frames=frames,
                gripper=extra["gripper"],
                scene_entity_poses=scene_entity_poses,
                entity_configurations=entity_configurations,
                structural_bindings=dict(structural_bindings),
            )
        return aligned

    @staticmethod
    def _is_scene_entity(name: str) -> bool:
        return not name.startswith("virtual_skill_") and name not in {
            "left_ee",
            "right_ee",
        }

    @staticmethod
    def _window_indices(duration: int, local_index: int, radius: int) -> range:
        return range(
            max(0, local_index - radius), min(duration, local_index + radius + 1)
        )

    def _active_reference_entities(
        self,
        policy: DynaMAC,
        state_id: StateId,
        mode: int,
        radius: int,
    ) -> tuple[str, ...]:
        """Frames that actually participate in this same-skill, same-mode window."""

        skill = policy.skills[state_id.skill_index]
        active = {
            name
            for index in self._window_indices(
                skill.duration, state_id.local_index, radius
            )
            for name, stream in skill.streams.items()
            if self._is_scene_entity(name) and stream.is_active(mode, index)
        }
        order = [name for name in policy.frame_names if name in active]
        return tuple(order)

    def _candidate_factors(
        self,
        policy: DynaMAC,
        data: _AlignedSkillData,
        state_id: StateId,
        mode: int,
        radius: int,
    ) -> tuple[FactorId, ...]:
        references = self._active_reference_entities(policy, state_id, mode, radius)
        return self._factors_for_references(data, references)

    def _factors_for_references(
        self,
        data: _AlignedSkillData,
        references: Sequence[str],
    ) -> tuple[FactorId, ...]:
        """Build the shared sparse library from task refs and one-hop structure."""

        reference_entities = set(references)
        direct_bindings = {
            (child, parent)
            for child, parent in data.structural_bindings.items()
            if child in reference_entities or parent in reference_entities
        }
        entities = reference_entities.union(
            entity for binding in direct_bindings for entity in binding
        )

        factors: set[FactorId] = set()
        for entity in entities:
            for feature in data.entity_configurations.get(entity, {}):
                factors.add(FactorId("node", entity, feature=feature))

        # Jointly active task references form the small local computation graph.
        for source, target in combinations(references, 2):
            factors.add(FactorId("edge", source, target))
        # The task interface may add only direct structural edges; it does not
        # induce a clique over structural neighbours.
        poses = set(data.frames).union(data.scene_entity_poses)
        for child, parent in direct_bindings:
            # An explicit internal configuration is the canonical description
            # of an articulated child.  Keeping its relative pose to the
            # structural parent as well would encode the same degree of
            # freedom twice.  Rigid children without such a field continue to
            # use the direct structural edge.
            if child not in data.entity_configurations and {child, parent} <= poses:
                factors.add(FactorId("edge", child, parent))
        return tuple(sorted(factors))

    @staticmethod
    def _factor_space(factor_id: FactorId) -> str:
        return "euclidean" if factor_id.kind == "node" else "se3"

    def _factor_values(
        self,
        data: _AlignedSkillData,
        factor_id: FactorId,
    ) -> Array:
        if factor_id.kind == "node":
            assert factor_id.feature is not None
            return data.entity_configurations[factor_id.source][factor_id.feature]
        poses = {**data.frames, **data.scene_entity_poses}
        assert factor_id.target is not None
        values = np.empty_like(poses[factor_id.source])
        for demo_index in range(len(values)):
            values[demo_index] = _relative_batch(
                poses[factor_id.target][demo_index],
                poses[factor_id.source][demo_index],
            )
        return values

    def _fit_factor(
        self,
        values: Array,
        policy: DynaMAC,
        factor_id: FactorId,
        **audit: Any,
    ) -> FactorDistribution:
        return fit_factor_distribution(
            values,
            position_variance_floor=policy.config.position_variance_floor,
            rotation_variance_floor=policy.config.rotation_variance_floor,
            covariance_estimation_method=policy.config.covariance_estimation_method,
            space=self._factor_space(factor_id),
            **audit,
        )

    def _window_samples(
        self,
        values: Array,
        members: Array,
        local_index: int,
        radius: int,
    ) -> Array:
        indices = tuple(self._window_indices(values.shape[1], local_index, radius))
        return values[np.ix_(members, indices)].reshape(-1, values.shape[-1])

    def _cross_demo_stability(
        self,
        values: Array,
        local_index: int,
        radius: int,
        members: Array,
        policy: DynaMAC,
        factor_id: FactorId,
    ) -> float:
        """LODO coverage of held-out local windows by N-1 support domains."""

        if len(members) < 3:
            return 0.0
        passed = 0
        indices = tuple(self._window_indices(values.shape[1], local_index, radius))
        for held_out in members:
            training = members[members != held_out]
            distribution = self._fit_factor(
                self._window_samples(values, training, local_index, radius),
                policy,
                factor_id,
            )
            held_score = float(
                np.mean(
                    [
                        distribution.log_likelihood(value)
                        for value in values[held_out, indices]
                    ]
                )
            )
            passed += int(held_score >= self._support_log_likelihood(distribution))
        return passed / float(len(members))

    def _support_log_likelihood(self, distribution: FactorDistribution) -> float:
        """Gaussian support-domain threshold at the configured confidence."""

        dimension = distribution.covariance.shape[0]
        z_score = NormalDist().inv_cdf(self.config.scene_support_confidence)
        # Wilson-Hilferty approximation avoids a mandatory SciPy dependency.
        chi_square = (
            dimension
            * (
                1.0
                - 2.0 / (9.0 * dimension)
                + z_score * math.sqrt(2.0 / (9.0 * dimension))
            )
            ** 3
        )
        return distribution.log_likelihood(distribution.mean) - 0.5 * chi_square

    def _support_compatibility(self, dimension: int) -> float:
        z_score = NormalDist().inv_cdf(self.config.scene_support_confidence)
        chi_square = (
            dimension
            * (
                1.0
                - 2.0 / (9.0 * dimension)
                + z_score * math.sqrt(2.0 / (9.0 * dimension))
            )
            ** 3
        )
        return float(math.exp(-0.5 * chi_square))

    def _robot_candidate_scores(
        self,
        policy: DynaMAC,
        data: _AlignedSkillData,
        state_id: StateId,
        mode: int,
        held_out: int,
        training: Array,
        candidates: Sequence[int],
    ) -> dict[int, float]:
        """Normalized robot-trajectory baseline for scene-factor LODO."""

        actual_index = state_id.local_index
        skill = policy.skills[state_id.skill_index]
        scores: dict[int, float] = {}
        for candidate in candidates:
            terms = []
            for name, stream in skill.streams.items():
                if not stream.is_selected(mode) or not stream.is_active(
                    mode, candidate
                ):
                    continue
                training_local = _relative_batch(
                    data.frames[name][training, candidate],
                    data.ee_pose[training, candidate],
                )
                mean, covariance = _fit_pose_samples(training_local, policy)
                local_value = relative_pose(
                    data.frames[name][held_out, actual_index],
                    data.ee_pose[held_out, actual_index],
                )
                distribution = FactorDistribution(
                    mean=mean,
                    covariance=covariance,
                    sample_count=len(training),
                )
                terms.append(distribution.log_likelihood(local_value))
            scores[candidate] = float(np.mean(terms)) if terms else -1.0e12
        return scores

    def _scene_candidate_scores(
        self,
        selected: dict[FactorId, int],
        factor_values: dict[FactorId, Array],
        state_id: StateId,
        held_out: int,
        training: Array,
        candidates: Sequence[int],
        policy: DynaMAC,
    ) -> dict[int, float]:
        if not selected:
            return {candidate: 0.0 for candidate in candidates}
        scores = {}
        for candidate in candidates:
            terms = []
            for factor_id, radius in selected.items():
                values = factor_values[factor_id]
                distribution = self._fit_factor(
                    self._window_samples(values, training, candidate, radius),
                    policy,
                    factor_id,
                )
                terms.append(
                    distribution.log_likelihood(values[held_out, state_id.local_index])
                )
            scores[candidate] = float(np.mean(terms))
        return scores

    def _loo_metrics(
        self,
        values: Array,
        state_id: StateId,
        mode: int,
        radius: int,
        members: Array,
        policy: DynaMAC,
        data: _AlignedSkillData,
        selected_factors: dict[FactorId, int],
        factor_values: dict[FactorId, Array],
        factor_id: FactorId,
    ) -> tuple[float, float]:
        """Gain in local progress margin after adding this one scene factor."""

        skill = policy.skills[state_id.skill_index]
        start = max(
            0, state_id.local_index - self.config.scene_recognition_neighborhood
        )
        stop = min(
            skill.duration,
            state_id.local_index + self.config.scene_recognition_neighborhood + 1,
        )
        candidates = list(range(start, stop))
        competitors = [index for index in candidates if index != state_id.local_index]
        if len(members) < 3 or not competitors:
            return 0.0, 0.0
        gains = []
        positive = 0
        for held_out in members:
            training = members[members != held_out]
            robot_scores = self._robot_candidate_scores(
                policy,
                data,
                state_id,
                mode,
                int(held_out),
                training,
                candidates,
            )
            without_scene = self._scene_candidate_scores(
                selected_factors,
                factor_values,
                state_id,
                int(held_out),
                training,
                candidates,
                policy,
            )
            with_scene = self._scene_candidate_scores(
                {**selected_factors, factor_id: radius},
                factor_values,
                state_id,
                int(held_out),
                training,
                candidates,
                policy,
            )
            without_scores = {
                candidate: robot_scores[candidate]
                + self.config.progress_scene_weight * without_scene[candidate]
                for candidate in candidates
            }
            with_scores = {
                candidate: robot_scores[candidate]
                + self.config.progress_scene_weight * with_scene[candidate]
                for candidate in candidates
            }
            baseline_margin = without_scores[state_id.local_index] - _logmeanexp(
                [without_scores[index] for index in competitors]
            )
            with_margin = with_scores[state_id.local_index] - _logmeanexp(
                [with_scores[index] for index in competitors]
            )
            gain = with_margin - baseline_margin
            gains.append(gain)
            positive += int(gain > 0.0)
        return float(np.mean(gains)), positive / float(len(members))

    def _add_scene_factors(
        self,
        policy: DynaMAC,
        states: dict[StateId, StateNode],
        aligned: dict[int, _AlignedSkillData],
    ) -> dict[int, dict[FactorId, Array]]:
        values_by_skill: dict[int, dict[FactorId, Array]] = {}
        for skill_index, skill in enumerate(policy.skills):
            data = aligned[skill_index]
            factor_values: dict[FactorId, Array] = {}
            values_by_skill[skill_index] = factor_values
            for local_index in range(skill.duration):
                state_id = StateId(skill_index, local_index)
                state = states[state_id]
                for mode in range(len(skill.mode_priors)):
                    members = _mode_members(skill, mode)
                    maximum_candidates = self._candidate_factors(
                        policy,
                        data,
                        state_id,
                        mode,
                        self.config.scene_max_neighborhood,
                    )
                    stable_candidates: dict[
                        FactorId,
                        tuple[int, float, FactorDistribution],
                    ] = {}
                    for factor_id in maximum_candidates:
                        values = factor_values.setdefault(
                            factor_id, self._factor_values(data, factor_id)
                        )
                        for radius in range(self.config.scene_max_neighborhood + 1):
                            if factor_id not in self._candidate_factors(
                                policy,
                                data,
                                state_id,
                                mode,
                                radius,
                            ):
                                continue
                            observability = 1.0
                            stable_fraction = self._cross_demo_stability(
                                values,
                                local_index,
                                radius,
                                members,
                                policy,
                                factor_id,
                            )
                            if (
                                observability < self.config.scene_min_support
                                or stable_fraction
                                < self.config.scene_min_stable_fraction
                            ):
                                continue
                            distribution = self._fit_factor(
                                self._window_samples(
                                    values, members, local_index, radius
                                ),
                                policy,
                                factor_id,
                                observability=observability,
                                stable_fraction=stable_fraction,
                                neighborhood_radius=radius,
                            )
                            stable_candidates[factor_id] = (
                                radius,
                                stable_fraction,
                                distribution,
                            )
                            break

                    selected: dict[FactorId, int] = {}
                    remaining = dict(stable_candidates)
                    while remaining:
                        qualified = []
                        for factor_id in sorted(remaining):
                            radius, stable_fraction, base = remaining[factor_id]
                            gain, accuracy = self._loo_metrics(
                                factor_values[factor_id],
                                state_id,
                                mode,
                                radius,
                                members,
                                policy,
                                data,
                                selected,
                                factor_values,
                                factor_id,
                            )
                            if (
                                gain > self.config.scene_min_loo_gain
                                and accuracy >= self.config.scene_min_loo_accuracy
                            ):
                                qualified.append(
                                    (
                                        gain,
                                        factor_id,
                                        accuracy,
                                        stable_fraction,
                                        base,
                                    )
                                )
                        if not qualified:
                            break
                        gain, factor_id, accuracy, stable_fraction, base = max(
                            qualified,
                            key=lambda item: item[0],
                        )
                        state.scene_factor_models.setdefault(factor_id, {})[mode] = (
                            FactorDistribution(
                                mean=base.mean,
                                covariance=base.covariance,
                                sample_count=base.sample_count,
                                space=base.space,
                                observability=base.observability,
                                stable_fraction=stable_fraction,
                                loo_gain=gain,
                                loo_accuracy=accuracy,
                                neighborhood_radius=base.neighborhood_radius,
                            )
                        )
                        selected[factor_id] = base.neighborhood_radius
                        del remaining[factor_id]
        return values_by_skill

    def _build_states(
        self,
        policy: DynaMAC,
        aligned: dict[int, _AlignedSkillData],
    ) -> tuple[dict[StateId, StateNode], dict[int, tuple[StateId, ...]]]:
        topology = build_state_topology(policy)
        skill_offsets = np.cumsum(
            [0, *(skill.duration for skill in policy.skills)],
            dtype=np.int64,
        )
        relation_evidence: dict[
            tuple[str, tuple[int, ...]], tuple[Array, Array, Array]
        ] = {}
        for skill in policy.skills:
            for mode in range(len(skill.mode_priors)):
                members = _mode_members(skill, mode)
                member_key = tuple(int(value) for value in members)
                for frame in policy.frame_names:
                    key = (frame, member_key)
                    if key not in relation_evidence:
                        relation_evidence[key] = self._joint_relation_evidence_sequence(
                            policy,
                            aligned,
                            frame,
                            members,
                            len(policy.skills) - 1,
                        )
        states: dict[StateId, StateNode] = {}
        skill_states: dict[int, list[StateId]] = {
            index: [] for index in range(len(policy.skills))
        }
        for skill_index, skill in enumerate(policy.skills):
            for local_index in range(skill.duration):
                state_id = StateId(skill_index, local_index)
                relation_scores = {}
                relation_priors = {}
                for frame in policy.frame_names:
                    mode_scores = []
                    mode_priors = []
                    for mode in range(len(skill.mode_priors)):
                        members = _mode_members(skill, mode)
                        member_key = tuple(int(value) for value in members)
                        scales, priors, _ = relation_evidence[(frame, member_key)]
                        global_index = int(skill_offsets[skill_index]) + local_index
                        mode_scores.append(float(scales[global_index]))
                        linked = float(priors[global_index])
                        mode_priors.append(
                            np.asarray([1.0 - linked, linked], dtype=np.float64)
                        )
                    relation_scores[frame] = np.asarray(mode_scores)
                    relation_priors[frame] = np.stack(mode_priors)
                stream_means = {
                    name: stream.mean[:, local_index]
                    for name, stream in skill.streams.items()
                }
                stream_covariances = {
                    name: stream.covariance[:, local_index]
                    for name, stream in skill.streams.items()
                }
                states[state_id] = StateNode(
                    state_id=state_id,
                    topology=topology[state_id],
                    stream_means=stream_means,
                    stream_covariances=stream_covariances,
                    demo_relation_scores=relation_scores,
                    demo_relation_priors=relation_priors,
                    mode_priors=skill.mode_priors,
                    selected_frames=tuple(skill.selected_frames),
                    mode_selected_frames=tuple(
                        tuple(
                            name
                            for name, stream in skill.streams.items()
                            if stream.is_selected(mode)
                        )
                        for mode in range(len(skill.mode_priors))
                    ),
                    gripper_commands=skill.gripper[:, local_index],
                )
                skill_states[skill_index].append(state_id)
        return states, {
            index: tuple(sorted(values)) for index, values in skill_states.items()
        }

    def _stable_relation_sequence(self, probabilities: Array) -> list[str]:
        states = []
        current = (
            "linked"
            if probabilities[0] >= self.config.relation_link_threshold
            else "external"
        )
        for probability in probabilities:
            if probability >= self.config.relation_link_threshold:
                candidate = "linked"
            elif probability <= self.config.relation_unlink_threshold:
                candidate = "external"
            else:
                candidate = current
            states.append(candidate)
            current = candidate
        result = states.copy()
        start = 0
        while start < len(result):
            stop = start + 1
            while stop < len(result) and result[stop] == result[start]:
                stop += 1
            if start > 0 and stop - start < self.config.relation_minimum_dwell:
                result[start:stop] = [result[start - 1]] * (stop - start)
            start = stop
        return result

    def _detect_relation_transitions(
        self,
        probabilities: Array,
    ) -> list[tuple[str, int]]:
        """Detect boundaries between stable hysteretic relation runs."""

        values = np.asarray(probabilities, dtype=np.float64)
        dwell = self.config.relation_minimum_dwell
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("关系事件证据必须是非空有限一维序列")
        states = self._stable_relation_sequence(values)
        runs: list[tuple[str, int, int]] = []
        start = 0
        while start < len(states):
            stop = start + 1
            while stop < len(states) and states[stop] == states[start]:
                stop += 1
            runs.append((states[start], start, stop))
            start = stop

        events = []
        for previous, current in zip(runs, runs[1:]):
            previous_state, previous_start, previous_stop = previous
            current_state, current_start, current_stop = current
            if (
                previous_stop - previous_start < dwell
                or current_stop - current_start < dwell
            ):
                continue
            if previous_state != current_state:
                transition = "link" if current_state == "linked" else "unlink"
                events.append((transition, current_start))
        return events

    def _joint_relation_prior_sequence(
        self,
        policy: DynaMAC,
        aligned: dict[int, _AlignedSkillData],
        frame: str,
        members: Array,
        final_skill: int,
    ) -> Array:
        """Fit one covariance-led relation prior from the supplied normal demos."""

        _, probabilities, _ = self._joint_relation_evidence_sequence(
            policy,
            aligned,
            frame,
            members,
            final_skill,
        )
        return probabilities

    def _joint_relation_evidence_sequence(
        self,
        policy: DynaMAC,
        aligned: dict[int, _AlignedSkillData],
        frame: str,
        members: Array,
        final_skill: int,
    ) -> tuple[Array, Array, Array]:
        """Return covariance scores, filtered priors, and event observability.

        Same-time cross-demonstration GMSD remains the primary linked evidence.
        Per-demonstration adjacent relative motion only rejects a low-GMSD false
        positive when the end effector actually moved but the frame did not
        co-move. Insufficient motion is neutral evidence for an event: it does
        not lower the saved covariance prior and cannot initiate LINK/UNLINK.
        """

        if not len(members):
            raise ValueError("关系先验至少需要一条正常示范")
        if final_skill < 0 or final_skill >= len(policy.skills):
            raise ValueError("关系先验的终止技能索引超出范围")

        ee_pose = np.concatenate(
            [aligned[index].ee_pose[members] for index in range(final_skill + 1)],
            axis=1,
        )
        frame_pose = np.concatenate(
            [aligned[index].frames[frame][members] for index in range(final_skill + 1)],
            axis=1,
        )
        local_pose = np.stack(
            [
                np.stack(
                    [
                        relative_pose(current_frame, current_ee)
                        for current_frame, current_ee in zip(
                            demonstration_frames,
                            demonstration_ee,
                            strict=True,
                        )
                    ]
                )
                for demonstration_frames, demonstration_ee in zip(
                    frame_pose,
                    ee_pose,
                    strict=True,
                )
            ]
        )
        _, covariance = _fit_pose_sequence(
            local_pose,
            policy.config.position_variance_floor,
            policy.config.rotation_variance_floor,
            covariance_estimation_method=policy.config.covariance_estimation_method,
        )
        scales = np.asarray(
            geometric_mean_standard_deviation(
                covariance,
                position_weight=policy.config.eq5_position_weight,
                rotation_weight=policy.config.eq5_rotation_weight,
            ),
            dtype=np.float64,
        )
        covariance_probabilities = np.asarray(
            [
                _relation_prior(scale, self.config, policy.config.tau_m)[1]
                for scale in scales
            ],
            dtype=np.float64,
        )

        weights = np.asarray(
            [policy.config.eq5_position_weight] * 3
            + [policy.config.eq5_rotation_weight] * 3,
            dtype=np.float64,
        )
        excitation = np.zeros((len(members), len(scales)), dtype=np.float64)
        relative_residual = np.zeros_like(excitation)
        for demonstration in range(len(members)):
            for index in range(1, len(scales)):
                ee_step = pose_log_world(
                    ee_pose[demonstration, index - 1],
                    ee_pose[demonstration, index],
                )
                local_step = pose_log_world(
                    local_pose[demonstration, index - 1],
                    local_pose[demonstration, index],
                )
                excitation[demonstration, index] = float(
                    np.sum(np.square(weights * ee_step))
                )
                relative_residual[demonstration, index] = float(
                    np.sum(np.square(weights * local_step))
                )

        excited = excitation >= self.config.relation_minimum_excitation
        temporally_stable = (
            relative_residual <= self.config.relation_comotion_residual_threshold
        ) & (
            relative_residual
            <= self.config.relation_comotion_residual_ratio
            * np.maximum(excitation, self.config.relation_epsilon)
        )
        excited_fraction = np.mean(excited, axis=0)
        stable_support = np.mean(excited & temporally_stable, axis=0)
        event_observable = excited_fraction >= self.config.relation_event_support

        # Temporal evidence is deliberately one-way: it may lower linked
        # support after informative motion, but it can never create LINK on its
        # own. With insufficient excitation the covariance prior is retained.
        probabilities = covariance_probabilities.copy()
        probabilities[event_observable] *= stable_support[event_observable]
        return scales, probabilities, event_observable

    def _joint_relation_event_probability_sequence(
        self,
        policy: DynaMAC,
        aligned: dict[int, _AlignedSkillData],
        frame: str,
        members: Array,
        final_skill: int,
    ) -> Array:
        """Return event evidence, using 0.5 while motion is uninformative."""

        _, probabilities, event_observable = self._joint_relation_evidence_sequence(
            policy,
            aligned,
            frame,
            members,
            final_skill,
        )
        return np.where(event_observable, probabilities, 0.5)

    def _joint_link_pending_evidence_sequence(
        self,
        policy: DynaMAC,
        aligned: dict[int, _AlignedSkillData],
        frame: str,
        members: Array,
        final_skill: int,
    ) -> tuple[Array, Array]:
        """Return the GMSD-led hypothesis and whether motion can confirm it."""

        _, probabilities, event_observable = self._joint_relation_evidence_sequence(
            policy,
            aligned,
            frame,
            members,
            final_skill,
        )
        return probabilities, event_observable

    def _relation_event_gripper_is_consistent(
        self,
        gripper: Array,
        members: Array,
        global_index: int,
        transition: str,
    ) -> bool:
        """Check event/anchor completeness without creating relation evidence.

        A LINK anchor must contain a closing command for every corresponding
        normal demonstration. An UNLINK must follow an opening command and may
        occur later, when relative motion finally makes detachment observable.
        Gripper changes can reject an incomplete event template, but never
        create LINK/UNLINK in the absence of kinematic relation evidence.
        """

        commands = np.mean(
            np.asarray(gripper, dtype=np.float64)[members],
            axis=2,
        )
        if commands.ndim != 2 or global_index <= 0 or global_index >= commands.shape[1]:
            return False
        tolerance = 1.0e-9
        for demonstration in commands:
            delta = np.diff(demonstration[: global_index + 1])
            changed = np.flatnonzero(np.abs(delta) > tolerance) + 1
            if not len(changed):
                return False
            latest = int(changed[-1])
            if transition == "link":
                start = max(1, global_index - self.config.link_anchor_horizon + 1)
                if latest < start or delta[latest - 1] >= -tolerance:
                    return False
            elif transition == "unlink":
                if delta[latest - 1] <= tolerance:
                    return False
            else:
                raise ValueError(f"未知关系事件类型：{transition}")
        return True

    def _build_relation_events(
        self,
        policy: DynaMAC,
        states: dict[StateId, StateNode],
        aligned: dict[int, _AlignedSkillData],
        arm_id: str,
        recoverable_frames: frozenset[str],
    ) -> tuple[
        dict[RelationEventId, LinkRecoveryAnchor],
        dict[RelationEventId, UnlinkEventMetadata],
        dict[RelationStateKey, RelationEventId],
        dict[RelationEventId, LinkPendingCandidate],
    ]:
        links: dict[RelationEventId, LinkRecoveryAnchor] = {}
        unlinks: dict[RelationEventId, UnlinkEventMetadata] = {}
        origins: dict[RelationStateKey, RelationEventId] = {}
        pending_links: dict[RelationEventId, LinkPendingCandidate] = {}
        skill_offsets = np.cumsum(
            [0, *(skill.duration for skill in policy.skills)],
            dtype=np.int64,
        )
        total_states = int(skill_offsets[-1])
        joint_gripper = np.concatenate(
            [aligned[index].gripper for index in range(len(policy.skills))],
            axis=1,
        )
        event_probability_cache: dict[tuple[str, tuple[int, ...]], Array] = {}
        pending_evidence_cache: dict[
            tuple[str, tuple[int, ...]], tuple[Array, Array]
        ] = {}

        def event_probabilities(frame: str, members: Array) -> Array:
            key = (frame, tuple(int(value) for value in members))
            if key not in event_probability_cache:
                event_probability_cache[key] = (
                    self._joint_relation_event_probability_sequence(
                        policy,
                        aligned,
                        frame,
                        members,
                        len(policy.skills) - 1,
                    )
                )
            return event_probability_cache[key].copy()

        def pending_evidence(frame: str, members: Array) -> tuple[Array, Array]:
            key = (frame, tuple(int(value) for value in members))
            if key not in pending_evidence_cache:
                pending_evidence_cache[key] = (
                    self._joint_link_pending_evidence_sequence(
                        policy,
                        aligned,
                        frame,
                        members,
                        len(policy.skills) - 1,
                    )
                )
            probabilities, observable = pending_evidence_cache[key]
            return probabilities.copy(), observable.copy()

        def unresolved_link_candidates(
            probabilities: Array,
            observable: Array,
            confirmed_probabilities: Array,
            gripper: Array,
            members: Array,
            global_start: int,
            global_stop: int,
        ) -> list[tuple[str, int]]:
            commands = np.mean(
                np.asarray(gripper, dtype=np.float64)[members],
                axis=2,
            )
            close_positions = [
                np.flatnonzero(np.diff(demonstration) < -1.0e-9) + 1
                for demonstration in commands
            ]
            if not close_positions or any(
                not len(values) for values in close_positions
            ):
                return []
            collective_closes = []
            for seed in close_positions[0]:
                aligned_positions = [int(seed)]
                for positions in close_positions[1:]:
                    nearest = int(
                        min(positions, key=lambda value: abs(int(value) - int(seed)))
                    )
                    if (
                        abs(nearest - int(seed))
                        > self.config.relation_event_alignment_tolerance
                    ):
                        break
                    aligned_positions.append(nearest)
                if len(aligned_positions) == len(close_positions):
                    collective_closes.append(max(aligned_positions))

            potential_transitions = [
                index
                for transition, index in self._detect_relation_transitions(
                    probabilities
                )
                if transition == "link"
            ]
            potential_states = self._stable_relation_sequence(probabilities)
            confirmed_states = self._stable_relation_sequence(confirmed_probabilities)
            candidates = []
            for index in sorted(set(collective_closes)):
                if not global_start <= index < global_stop:
                    continue
                if confirmed_states[index] == "linked":
                    continue
                if not any(
                    abs(position - index) <= self.config.link_anchor_horizon
                    for position in potential_transitions
                ):
                    continue
                # The closing sample may still describe the final approach
                # motion. Confirmation must come from motion after closing.
                evidence_start = index + 1
                evidence_stop = min(
                    len(observable),
                    evidence_start + self.config.relation_minimum_dwell,
                )
                evidence_window = observable[evidence_start:evidence_stop]
                if len(evidence_window) < self.config.relation_minimum_dwell:
                    continue
                # Pending means the LINK hypothesis is repeatable but cannot be
                # confirmed due to absent motion excitation. Any informative
                # post-candidate evidence must be handled by normal LINK/event
                # detection instead of being relabeled as Pending.
                if np.any(evidence_window):
                    continue
                if any(
                    state != "linked"
                    for state in potential_states[evidence_start:evidence_stop]
                ):
                    continue
                candidates.append(("link", index - global_start))
            return candidates

        def state_from_global(global_index: int) -> StateId:
            if global_index < 0 or global_index >= total_states:
                raise IndexError("全局关系事件状态索引超出范围")
            target_skill = int(
                np.searchsorted(skill_offsets[1:], global_index, side="right")
            )
            return StateId(
                target_skill,
                global_index - int(skill_offsets[target_skill]),
            )

        def fit_link_anchor_template(
            frame: str,
            skill_index: int,
            local_index: int,
            members: Array,
        ) -> tuple[StateId, Array, Array, Array]:
            """Fit one LINK template without crossing the arm's latest UNLINK."""

            event_index = int(skill_offsets[skill_index]) + local_index
            raw_start = max(
                0,
                event_index - self.config.link_anchor_horizon + 1,
            )
            previous_unlinks = [
                index
                for candidate_frame in recoverable_frames
                for transition, index in self._detect_relation_transitions(
                    event_probabilities(candidate_frame, members)
                )
                if transition == "unlink" and raw_start <= index < event_index
            ]
            anchor_start = max(
                raw_start,
                max(previous_unlinks, default=raw_start - 1) + 1,
            )
            anchor_states = [
                state_from_global(index)
                for index in range(anchor_start, event_index + 1)
            ]
            means = []
            covariances = []
            gripper_commands = []
            for state in anchor_states:
                sample_data = aligned[state.skill_index]
                local_samples = [
                    relative_pose(
                        sample_data.frames[frame][demonstration, state.local_index],
                        sample_data.ee_pose[demonstration, state.local_index],
                    )
                    for demonstration in members
                ]
                gripper_samples = [
                    sample_data.gripper[demonstration, state.local_index]
                    for demonstration in members
                ]
                mean, covariance = _fit_pose_samples(
                    np.stack(local_samples),
                    policy,
                )
                means.append(mean)
                covariances.append(covariance)
                gripper_commands.append(np.mean(gripper_samples, axis=0))
            return (
                anchor_states[0],
                np.stack(means),
                np.stack(covariances),
                np.stack(gripper_commands),
            )

        for skill_index, skill in enumerate(policy.skills):
            for mode in range(len(skill.mode_priors)):
                members = _mode_members(skill, mode)
                for frame in policy.frame_names:
                    if frame not in recoverable_frames:
                        continue
                    global_start = int(skill_offsets[skill_index])
                    global_stop = int(skill_offsets[skill_index + 1])

                    # The complete task sequence supplies both the pre-event and
                    # post-event dwell windows, including when either window
                    # crosses a skill boundary.  Events are still assigned only
                    # to transitions whose first new-relation state lies in the
                    # current skill.
                    joint_probabilities = event_probabilities(frame, members)
                    joint_relation_states = self._stable_relation_sequence(
                        joint_probabilities
                    )
                    relation_states = joint_relation_states[global_start:global_stop]
                    joint_candidates = [
                        (transition, index - global_start)
                        for transition, index in self._detect_relation_transitions(
                            joint_probabilities
                        )
                        if global_start <= index < global_stop
                    ]

                    # LODO is a stability audit of a jointly learned event, not
                    # per-demonstration event evidence.  Every fold refits the
                    # joint prior from N-1 demonstrations and checks whether the
                    # all-demo candidate reappears in the same local region.
                    fold_events: dict[int, list[tuple[str, int]]] = {}
                    if len(members) >= 2:
                        for held_out in members:
                            training = members[members != held_out]
                            fold_probabilities = event_probabilities(frame, training)
                            fold_events[int(held_out)] = [
                                (transition, index - global_start)
                                for transition, index in self._detect_relation_transitions(
                                    fold_probabilities
                                )
                                if global_start <= index < global_stop
                            ]

                    consensus_events: list[
                        tuple[str, int, list[tuple[int, int]], RelationEventId]
                    ] = []
                    occurrence = {"link": 0, "unlink": 0}
                    required = max(
                        1,
                        int(
                            math.ceil(
                                self.config.relation_event_support * len(members)
                                - 1.0e-12
                            )
                        ),
                    )
                    used_fold_events: dict[int, set[int]] = {
                        int(demonstration): set() for demonstration in members
                    }
                    for transition, local_index in sorted(
                        joint_candidates,
                        key=lambda item: (item[1], item[0]),
                    ):
                        supported = []
                        matched_fold_events = []
                        for held_out in members:
                            demonstration = int(held_out)
                            candidates = [
                                (event_index, position)
                                for event_index, (kind, position) in enumerate(
                                    fold_events.get(demonstration, ())
                                )
                                if kind == transition
                                and event_index not in used_fold_events[demonstration]
                                and abs(position - local_index)
                                <= self.config.relation_event_alignment_tolerance
                            ]
                            if not candidates:
                                continue
                            event_index, position = min(
                                candidates,
                                key=lambda item: (
                                    abs(item[1] - local_index),
                                    item[1],
                                ),
                            )
                            supported.append((demonstration, position))
                            matched_fold_events.append((demonstration, event_index))
                        if len(supported) < required:
                            continue
                        if not self._relation_event_gripper_is_consistent(
                            joint_gripper,
                            members,
                            global_start + local_index,
                            transition,
                        ):
                            continue
                        for demonstration, event_index in matched_fold_events:
                            used_fold_events[demonstration].add(event_index)
                        event_id = RelationEventId(
                            arm_id,
                            frame,
                            skill_index,
                            mode,
                            occurrence[transition],
                            transition,
                        )
                        occurrence[transition] += 1
                        consensus_events.append(
                            (transition, local_index, supported, event_id)
                        )

                    # Preserve a repeatable external->linked hypothesis when
                    # GMSD and the grasp template support it but the post-close
                    # demonstration segment has no informative motion. This is
                    # event metadata only: it does not create a LINK anchor,
                    # propagate link_origin, or claim that linked is true.
                    pending_probabilities, pending_observable = pending_evidence(
                        frame, members
                    )
                    pending_candidates = unresolved_link_candidates(
                        pending_probabilities,
                        pending_observable,
                        joint_probabilities,
                        joint_gripper,
                        members,
                        global_start,
                        global_stop,
                    )
                    pending_candidates = [
                        candidate
                        for candidate in pending_candidates
                        if not any(
                            transition == "link"
                            and abs(position - candidate[1])
                            <= self.config.relation_event_alignment_tolerance
                            for transition, position in joint_candidates
                        )
                        and not any(
                            transition == "link"
                            and abs(position - candidate[1])
                            <= self.config.link_anchor_horizon
                            for transition, position, _, _ in consensus_events
                        )
                    ]
                    fold_pending_events: dict[int, list[tuple[str, int]]] = {}
                    if len(members) >= 2:
                        for held_out in members:
                            training = members[members != held_out]
                            fold_probabilities, fold_observable = pending_evidence(
                                frame, training
                            )
                            fold_confirmed_probabilities = event_probabilities(
                                frame, training
                            )
                            fold_pending_events[int(held_out)] = (
                                unresolved_link_candidates(
                                    fold_probabilities,
                                    fold_observable,
                                    fold_confirmed_probabilities,
                                    joint_gripper,
                                    training,
                                    global_start,
                                    global_stop,
                                )
                            )

                    used_pending_fold_events: dict[int, set[int]] = {
                        int(demonstration): set() for demonstration in members
                    }
                    pending_occurrence = 0
                    for _, local_index in sorted(
                        pending_candidates,
                        key=lambda item: item[1],
                    ):
                        supported = []
                        matched_fold_events = []
                        for held_out in members:
                            demonstration = int(held_out)
                            candidates = [
                                (event_index, position)
                                for event_index, (kind, position) in enumerate(
                                    fold_pending_events.get(demonstration, ())
                                )
                                if kind == "link"
                                and event_index
                                not in used_pending_fold_events[demonstration]
                                and abs(position - local_index)
                                <= self.config.relation_event_alignment_tolerance
                            ]
                            if not candidates:
                                continue
                            event_index, position = min(
                                candidates,
                                key=lambda item: (
                                    abs(item[1] - local_index),
                                    item[1],
                                ),
                            )
                            supported.append((demonstration, position))
                            matched_fold_events.append((demonstration, event_index))
                        if len(supported) < required:
                            continue
                        if not self._relation_event_gripper_is_consistent(
                            joint_gripper,
                            members,
                            global_start + local_index,
                            "link",
                        ):
                            continue
                        for demonstration, event_index in matched_fold_events:
                            used_pending_fold_events[demonstration].add(event_index)
                        event_id = RelationEventId(
                            arm_id,
                            frame,
                            skill_index,
                            mode,
                            pending_occurrence,
                            "link_pending",
                        )
                        pending_occurrence += 1
                        (
                            context_state,
                            local_means,
                            local_covariances,
                            gripper_commands,
                        ) = fit_link_anchor_template(
                            frame,
                            skill_index,
                            local_index,
                            members,
                        )
                        pending_links[event_id] = LinkPendingCandidate(
                            event_id=event_id,
                            arm_id=arm_id,
                            frame_id=frame,
                            candidate_state=StateId(skill_index, local_index),
                            context_state=context_state,
                            local_means=local_means,
                            local_covariances=local_covariances,
                            gripper_commands=gripper_commands,
                            support_fraction=len(supported) / float(len(members)),
                            demonstration_indices=tuple(
                                demonstration for demonstration, _ in supported
                            ),
                            event_local_indices=tuple(
                                position for _, position in supported
                            ),
                        )

                    for transition, local_index, supported, event_id in sorted(
                        consensus_events,
                        key=lambda item: (item[1], item[0]),
                    ):
                        event_state = StateId(skill_index, local_index)
                        if transition == "link":
                            (
                                context_state,
                                local_means,
                                local_covariances,
                                gripper_commands,
                            ) = fit_link_anchor_template(
                                frame,
                                skill_index,
                                local_index,
                                members,
                            )
                            stop = next(
                                (
                                    index
                                    for index in range(local_index + 1, skill.duration)
                                    if relation_states[index] == "external"
                                ),
                                skill.duration,
                            )
                            linked_entry = tuple(
                                StateId(skill_index, index)
                                for index in range(local_index, stop)
                                if relation_states[index] == "linked"
                            )
                            links[event_id] = LinkRecoveryAnchor(
                                event_id=event_id,
                                arm_id=arm_id,
                                frame_id=frame,
                                context_state=context_state,
                                local_means=local_means,
                                local_covariances=local_covariances,
                                gripper_commands=gripper_commands,
                                linked_entry_states=linked_entry,
                                support_fraction=len(supported) / float(len(members)),
                                demonstration_indices=tuple(
                                    demonstration for demonstration, _ in supported
                                ),
                                event_local_indices=tuple(
                                    position for _, position in supported
                                ),
                            )
                        else:
                            global_event_index = global_start + local_index
                            legal_stop = min(
                                total_states,
                                global_event_index + self.config.reentry_horizon,
                            )
                            legal = tuple(
                                state_from_global(index)
                                for index in range(global_event_index, legal_stop)
                                if joint_relation_states[index] == "external"
                            )
                            target_global_index = min(
                                global_event_index + 1,
                                total_states - 1,
                            )
                            target_state = state_from_global(target_global_index)
                            target_data = aligned[target_state.skill_index]
                            local = _relative_batch(
                                target_data.frames[frame][
                                    members, target_state.local_index
                                ],
                                target_data.ee_pose[members, target_state.local_index],
                            )
                            target, _ = _fit_pose_samples(local, policy)
                            unlinks[event_id] = UnlinkEventMetadata(
                                event_id=event_id,
                                arm_id=arm_id,
                                frame_id=frame,
                                release_state=event_state,
                                legal_reentry_states=legal,
                                local_detachment_target=target,
                                support_fraction=len(supported) / float(len(members)),
                                demonstration_indices=tuple(
                                    demonstration for demonstration, _ in supported
                                ),
                                event_local_indices=tuple(
                                    position for _, position in supported
                                ),
                            )

        # Raw per-state covariance evidence may flicker even though a physical
        # connection cannot disappear without a confirmed UNLINK.  Reconcile
        # deployment priors and origins from the accepted event state machine;
        # rejected GMSD pulses remain in the audit scores but cannot become a
        # control-time linked expectation.
        origins = self._reconcile_confirmed_relation_paths(
            policy,
            states,
            links,
            unlinks,
            pending_links,
            arm_id,
        )

        # ``linked_entry_states`` is derived from the completed origin map so
        # it covers legal linked states across later skill and mode boundaries,
        # not only the remainder of the event's source skill.
        linked_states_by_event: dict[RelationEventId, set[StateId]] = {
            event_id: set() for event_id in links
        }
        for key, event_id in origins.items():
            if event_id in linked_states_by_event:
                linked_states_by_event[event_id].add(key.state_id)
        links = {
            event_id: replace(
                anchor,
                linked_entry_states=tuple(sorted(linked_states_by_event[event_id])),
            )
            for event_id, anchor in links.items()
        }
        return links, unlinks, origins, pending_links

    def _reconcile_confirmed_relation_paths(
        self,
        policy: DynaMAC,
        states: dict[StateId, StateNode],
        links: dict[RelationEventId, LinkRecoveryAnchor],
        unlinks: dict[RelationEventId, UnlinkEventMetadata],
        pending_links: dict[RelationEventId, LinkPendingCandidate],
        arm_id: str,
    ) -> dict[RelationStateKey, RelationEventId]:
        """Apply accepted LINK/UNLINK events as a persistent relation state.

        The covariance/GMSD values remain available in
        ``demo_relation_scores``.  Deployment priors follow the confirmed
        event state machine, so a rejected isolated covariance pulse cannot
        become a control-time linked expectation.  A Pending-only interval may
        retain its covariance-led linked hypothesis after its candidate point,
        but it never gains an origin and stops as soon as that support ends.
        """

        skill_offsets = np.cumsum(
            [0, *(skill.duration for skill in policy.skills)],
            dtype=np.int64,
        )

        def global_index(state_id: StateId) -> int:
            return int(skill_offsets[state_id.skill_index]) + state_id.local_index

        transitions: dict[
            tuple[int, str],
            list[tuple[int, str, RelationEventId | None]],
        ] = {}
        for event_id, anchor in links.items():
            if not anchor.linked_entry_states:
                raise ValueError("正式 LINK 锚点必须包含事件入口状态")
            event_index = global_index(min(anchor.linked_entry_states))
            members = _mode_members(
                policy.skills[event_id.skill_index],
                event_id.mode,
            )
            for demonstration in members:
                transitions.setdefault(
                    (int(demonstration), event_id.frame_id), []
                ).append((event_index, "link", event_id))
        for event_id, metadata in unlinks.items():
            event_index = global_index(metadata.release_state)
            members = _mode_members(
                policy.skills[event_id.skill_index],
                event_id.mode,
            )
            for demonstration in members:
                transitions.setdefault(
                    (int(demonstration), event_id.frame_id), []
                ).append((event_index, "unlink", None))
        for values in transitions.values():
            values.sort(key=lambda value: (value[0], value[1]))

        pending_starts: dict[tuple[int, str], list[int]] = {}
        for event_id, candidate in pending_links.items():
            start = global_index(candidate.candidate_state)
            members = _mode_members(
                policy.skills[event_id.skill_index],
                event_id.mode,
            )
            for demonstration in members:
                pending_starts.setdefault(
                    (int(demonstration), event_id.frame_id), []
                ).append(start)
        for values in pending_starts.values():
            values.sort()

        pending_runtime: dict[tuple[tuple[int, ...], str], tuple[int, bool]] = {}

        origins: dict[RelationStateKey, RelationEventId] = {}
        for state_id, node in sorted(states.items()):
            state_global_index = global_index(state_id)
            skill = policy.skills[state_id.skill_index]
            for mode in range(len(skill.mode_priors)):
                mode_members = tuple(int(value) for value in _mode_members(skill, mode))
                for frame in policy.frame_names:
                    raw_prior = node.demo_relation_priors[frame][mode].copy()
                    tracks = [
                        transitions.get((member, frame)) for member in mode_members
                    ]
                    if not tracks or any(track is None for track in tracks):
                        member_pending = [
                            pending_starts.get((member, frame), [])
                            for member in mode_members
                        ]
                        common_starts = (
                            sorted(
                                set.intersection(
                                    *(set(values) for values in member_pending)
                                )
                            )
                            if member_pending and all(member_pending)
                            else []
                        )
                        available_starts = [
                            start
                            for start in common_starts
                            if start <= state_global_index
                        ]
                        runtime_key = (mode_members, frame)
                        last_start, active = pending_runtime.get(
                            runtime_key, (-1, False)
                        )
                        if available_starts and available_starts[-1] > last_start:
                            last_start = available_starts[-1]
                            active = True
                        pending_hypothesis = bool(
                            active
                            and raw_prior[1] >= self.config.relation_link_threshold
                        )
                        if active and not pending_hypothesis:
                            active = False
                        pending_runtime[runtime_key] = (last_start, active)
                        if not pending_hypothesis:
                            linked_probability = self.config.relation_unlink_threshold
                            node.demo_relation_priors[frame][mode] = np.asarray(
                                [1.0 - linked_probability, linked_probability],
                                dtype=np.float64,
                            )
                        continue
                    active_origins: list[RelationEventId | None] = []
                    active_count = 0
                    for track in tracks:
                        assert track is not None
                        # A first accepted UNLINK denotes an initially linked
                        # relation for which no in-task recovery anchor exists.
                        active = bool(track and track[0][1] == "unlink")
                        origin: RelationEventId | None = None
                        for index, transition, event_id in track:
                            if index > state_global_index:
                                break
                            if transition == "link":
                                active = True
                                origin = event_id
                            else:
                                active = False
                                origin = None
                        if active:
                            active_count += 1
                            active_origins.append(origin)

                    # Accepted events determine the persistent relation state,
                    # but the resulting deployment value is still a *weak
                    # demonstration prior*.  Writing [0, 1] here would make a
                    # normal-demo expectation nearly impossible for phase two
                    # observations to overturn after a real link failure.
                    # Map unanimous external/linked event states to the same
                    # soft thresholds used by event detection; mixed modes are
                    # linearly interpolated between them.
                    active_fraction = active_count / float(len(mode_members))
                    linked_probability = (
                        self.config.relation_unlink_threshold
                        + active_fraction
                        * (
                            self.config.relation_link_threshold
                            - self.config.relation_unlink_threshold
                        )
                    )
                    node.demo_relation_priors[frame][mode] = np.asarray(
                        [1.0 - linked_probability, linked_probability],
                        dtype=np.float64,
                    )
                    non_null_origins = {
                        origin for origin in active_origins if origin is not None
                    }
                    if (
                        active_count == len(mode_members)
                        and len(non_null_origins) == 1
                        and len(active_origins) == len(mode_members)
                    ):
                        origins[RelationStateKey(arm_id, frame, state_id, mode)] = next(
                            iter(non_null_origins)
                        )
        return origins

    def _boundary_loo_coverage(
        self,
        values: Array,
        start: int,
        members: Array,
        policy: DynaMAC,
        factor_id: FactorId,
    ) -> tuple[float, ...]:
        """Return each LODO fold's worst normal-window compatibility."""

        if len(members) < 3:
            return ()
        width = values.shape[1] - start
        time_repeats = np.arange(1, width + 1, dtype=np.int64)
        fold_scores = []
        for held_out in members:
            training = members[members != held_out]
            training_window = values[training, start:]
            training_samples = np.repeat(training_window, time_repeats, axis=1).reshape(
                -1, values.shape[-1]
            )
            distribution = self._fit_factor(
                training_samples,
                policy,
                factor_id,
            )
            fold_scores.append(
                min(
                    distribution.compatibility(value)
                    for value in values[held_out, start:]
                )
            )
        return tuple(float(value) for value in fold_scores)

    def _fit_boundary_relation_guard(
        self,
        policy: DynaMAC,
        states: dict[StateId, StateNode],
        aligned: dict[int, _AlignedSkillData],
        source_skill: int,
        start: int,
        frame: str,
        *,
        expected_state: str | None = None,
    ) -> tuple[RelationGuardDistribution, ReliabilityStatistics] | None:
        """Fit one 5/5-consistent terminal relation support distribution."""

        skill = policy.skills[source_skill]
        data = aligned[source_skill]
        time_weights = np.arange(1, skill.duration - start + 1, dtype=np.float64)
        time_weights /= np.sum(time_weights)
        support = np.zeros(2, dtype=np.float64)
        for mode, mode_prior in enumerate(skill.mode_priors):
            for offset, local_index in enumerate(range(start, skill.duration)):
                support += (
                    float(mode_prior)
                    * float(time_weights[offset])
                    * states[StateId(source_skill, local_index)].demo_relation_priors[
                        frame
                    ][mode]
                )
        support /= np.sum(support)
        target_index = (
            int(np.argmax(support))
            if expected_state is None
            else {"external": 0, "linked": 1}[expected_state]
        )
        required_state = "linked" if target_index == 1 else "external"
        if (
            int(np.argmax(support)) != target_index
            or support[target_index] < self.config.boundary_relation_support
        ):
            return None

        passed = 0
        folds = 0
        for mode in range(len(skill.mode_priors)):
            members = _mode_members(skill, mode)
            if len(members) < 2:
                return None
            for held_out in members:
                training = members[members != held_out]
                fold_support = np.zeros(2, dtype=np.float64)
                for offset, local_index in enumerate(range(start, skill.duration)):
                    local = _relative_batch(
                        data.frames[frame][training, local_index],
                        data.ee_pose[training, local_index],
                    )
                    _, covariance = _fit_pose_samples(local, policy)
                    scale = geometric_mean_standard_deviation(
                        covariance,
                        position_weight=policy.config.eq5_position_weight,
                        rotation_weight=policy.config.eq5_rotation_weight,
                    )
                    fold_support += float(time_weights[offset]) * _relation_prior(
                        scale,
                        self.config,
                        policy.config.tau_m,
                    )
                fold_support /= np.sum(fold_support)
                folds += 1
                passed += int(
                    int(np.argmax(fold_support)) == target_index
                    and fold_support[target_index]
                    >= self.config.boundary_relation_support
                )
        stable_fraction = passed / float(folds) if folds else 0.0
        if stable_fraction < 1.0:
            return None
        return (
            RelationGuardDistribution(
                external=float(support[0]),
                linked=float(support[1]),
                required_state=required_state,
            ),
            ReliabilityStatistics(1.0, stable_fraction),
        )

    @staticmethod
    def _own_relation_targets(
        policy: DynaMAC,
        source_skill: int,
        link_anchors: dict[RelationEventId, LinkRecoveryAnchor],
        unlink_events: dict[RelationEventId, UnlinkEventMetadata],
    ) -> dict[str, str]:
        """Return relations directly established/released in every skill mode."""

        events: dict[tuple[str, int], list[tuple[int, str]]] = {}
        for anchor in link_anchors.values():
            event_id = anchor.event_id
            if event_id.skill_index != source_skill or not anchor.linked_entry_states:
                continue
            position = anchor.linked_entry_states[0].local_index
            events.setdefault((event_id.frame_id, event_id.mode), []).append(
                (position, "linked")
            )
        for metadata in unlink_events.values():
            event_id = metadata.event_id
            if event_id.skill_index != source_skill:
                continue
            events.setdefault((event_id.frame_id, event_id.mode), []).append(
                (metadata.release_state.local_index, "external")
            )

        targets = {}
        modes = range(len(policy.skills[source_skill].mode_priors))
        for frame in policy.frame_names:
            per_mode = []
            for mode in modes:
                mode_events = events.get((frame, mode), ())
                if not mode_events:
                    break
                per_mode.append(max(mode_events, key=lambda item: item[0])[1])
            if (
                len(per_mode) == len(policy.skills[source_skill].mode_priors)
                and len(set(per_mode)) == 1
            ):
                targets[frame] = per_mode[0]
        return targets

    def _build_boundaries(
        self,
        policy: DynaMAC,
        states: dict[StateId, StateNode],
        aligned: dict[int, _AlignedSkillData],
        factor_values: dict[int, dict[FactorId, Array]],
        link_anchors: dict[RelationEventId, LinkRecoveryAnchor],
        unlink_events: dict[RelationEventId, UnlinkEventMetadata],
        arm_id: str,
    ) -> dict[BoundaryId, BoundaryModel]:
        boundaries = {}
        for source_skill in range(len(policy.skills) - 1):
            skill = policy.skills[source_skill]
            data = aligned[source_skill]
            start = max(0, skill.duration - self.config.boundary_terminal_window)
            terminal_states = tuple(
                StateId(source_skill, local_index)
                for local_index in range(start, skill.duration)
            )
            goal_distributions: dict[str, FactorDistribution] = {}
            thresholds: dict[str, float] = {}
            for mode in range(len(skill.mode_priors)):
                members = _mode_members(skill, mode)
                final_index = skill.duration - 1
                for frame in skill.selected_frames:
                    if not skill.streams[frame].is_selected(mode):
                        continue
                    frame_samples = _relative_batch(
                        data.frames[frame][members, final_index],
                        data.ee_pose[members, final_index],
                    )
                    distribution = fit_factor_distribution(
                        frame_samples,
                        position_variance_floor=policy.config.position_variance_floor,
                        rotation_variance_floor=policy.config.rotation_variance_floor,
                        covariance_estimation_method=policy.config.covariance_estimation_method,
                    )
                    key = f"m{mode}:{frame}"
                    goal_distributions[key] = distribution
                    scores = [
                        distribution.log_likelihood(value) for value in frame_samples
                    ]
                    thresholds[key] = float(np.min(scores))

            reliability = {}
            own_relation_conditions = {}
            own_targets = self._own_relation_targets(
                policy,
                source_skill,
                link_anchors,
                unlink_events,
            )
            for frame, target in sorted(own_targets.items()):
                fitted = self._fit_boundary_relation_guard(
                    policy,
                    states,
                    aligned,
                    source_skill,
                    start,
                    frame,
                    expected_state=target,
                )
                if fitted is None:
                    continue
                condition, statistics = fitted
                condition_key = f"{arm_id}/{frame}"
                own_relation_conditions[condition_key] = condition
                reliability[condition_key] = statistics

            next_skill = policy.skills[source_skill + 1]
            relation_guard_frames = set(next_skill.selected_frames).intersection(
                policy.frame_names
            )
            relation_conditions = {}
            for frame in sorted(relation_guard_frames):
                condition_key = f"{arm_id}/{frame}"
                if condition_key in own_relation_conditions:
                    continue
                fitted = self._fit_boundary_relation_guard(
                    policy,
                    states,
                    aligned,
                    source_skill,
                    start,
                    frame,
                )
                if fitted is None:
                    continue
                condition, statistics = fitted
                relation_conditions[condition_key] = condition
                reliability[condition_key] = statistics

            guard_entities = {
                name
                for name in next_skill.selected_frames
                if self._is_scene_entity(name)
            }
            guard_entities.update(key.split("/", 1)[1] for key in relation_conditions)
            # Boundary-only references are admitted to C_b^guard here.  They do
            # not become candidates for every progress state in the source
            # skill.  The helper applies the same node/edge definitions and
            # one-hop structural rule as the per-state sparse factor library.
            sparse_library = self._factors_for_references(
                data,
                tuple(
                    name
                    for name in policy.frame_names
                    if name in guard_entities and self._is_scene_entity(name)
                ),
            )
            scene_conditions = {}
            scene_condition_thresholds = {}
            for factor_id in sparse_library:
                try:
                    values = factor_values[source_skill].setdefault(
                        factor_id, self._factor_values(data, factor_id)
                    )
                except KeyError:
                    # The factor can be sparse in the next skill but genuinely
                    # unobservable in the source boundary window.
                    continue
                samples = []
                fold_compatibilities: list[float] = []
                time_repeats = np.arange(
                    1,
                    skill.duration - start + 1,
                    dtype=np.int64,
                )
                for mode in range(len(skill.mode_priors)):
                    members = _mode_members(skill, mode)
                    window = values[members, start : skill.duration]
                    samples.append(
                        np.repeat(window, time_repeats, axis=1).reshape(
                            -1, values.shape[-1]
                        )
                    )
                    fold_compatibilities.extend(
                        self._boundary_loo_coverage(
                            values,
                            start,
                            members,
                            policy,
                            factor_id,
                        )
                    )
                support_threshold = self._support_compatibility(
                    6 if factor_id.kind == "edge" else values.shape[-1]
                )
                passed_folds = sum(
                    value >= support_threshold for value in fold_compatibilities
                )
                total_folds = len(fold_compatibilities)
                stable_fraction = (
                    passed_folds / float(total_folds) if total_folds else 0.0
                )
                if stable_fraction < self.config.boundary_scene_min_stable_fraction:
                    continue
                distribution = self._fit_factor(
                    np.concatenate(samples, axis=0),
                    policy,
                    factor_id,
                    observability=1.0,
                    stable_fraction=stable_fraction,
                    loo_accuracy=stable_fraction,
                )
                scene_conditions[factor_id] = distribution
                scene_condition_thresholds[factor_id] = max(
                    0.0,
                    min(fold_compatibilities)
                    - self.config.boundary_compatibility_margin,
                )
                reliability[factor_id.token] = ReliabilityStatistics(
                    distribution.observability,
                    stable_fraction,
                )

            boundary_id = BoundaryId(arm_id, source_skill, source_skill + 1)
            boundaries[boundary_id] = BoundaryModel(
                boundary_id=boundary_id,
                source_skill=source_skill,
                target_skill=source_skill + 1,
                terminal_window=terminal_states,
                local_completion_model=LocalCompletionModel(
                    terminal_states=terminal_states,
                    goal_distributions=goal_distributions,
                    minimum_goal_log_likelihood=thresholds,
                    own_relation_conditions=own_relation_conditions,
                ),
                relation_conditions=relation_conditions,
                scene_conditions=scene_conditions,
                scene_condition_thresholds=scene_condition_thresholds,
                condition_reliability=reliability,
                affected_arms=(arm_id,),
            )
        return boundaries

    def build(
        self,
        policy: DynaMAC,
        demonstrations: Sequence[DynaMACDemonstration],
        *,
        arm_id: str = "single",
        recoverable_frames: Sequence[str] | None = None,
    ) -> ClosedLoopTaskModel:
        if not arm_id:
            raise ValueError("arm_id 必须为非空字符串")
        recoverable = frozenset(
            policy.frame_names if recoverable_frames is None else recoverable_frames
        )
        unknown_recoverable = recoverable.difference(policy.frame_names)
        if unknown_recoverable:
            raise ValueError(
                f"可恢复关系参考系不在基础 DynaMAC 中：{sorted(unknown_recoverable)}"
            )
        aligned = self._align_demonstrations(policy, demonstrations)
        states, skill_states = self._build_states(policy, aligned)
        factor_values = self._add_scene_factors(
            policy,
            states,
            aligned,
        )
        (
            link_anchors,
            unlink_events,
            link_origins,
            link_pending_events,
        ) = self._build_relation_events(
            policy,
            states,
            aligned,
            arm_id,
            recoverable,
        )
        boundaries = self._build_boundaries(
            policy,
            states,
            aligned,
            factor_values,
            link_anchors,
            unlink_events,
            arm_id,
        )
        return ClosedLoopTaskModel(
            base_policy=policy,
            states=states,
            skill_states=skill_states,
            boundaries=boundaries,
            link_anchors=link_anchors,
            unlink_events=unlink_events,
            link_pending_events=link_pending_events,
            link_origins=link_origins,
            arm_id=arm_id,
            relation_frames=tuple(policy.frame_names),
            builder_config=asdict(self.config),
        )

    @staticmethod
    def _boundary_positions(
        demonstrations: Sequence[DynaMACDemonstration],
        skill_sequence: Sequence[int],
    ) -> dict[int, Array]:
        positions: dict[int, list[float]] = {
            source: [] for source in range(max(0, len(skill_sequence) - 1))
        }
        for demonstration in demonstrations:
            denominator = max(1, len(demonstration.skill) - 1)
            for source, target_label in enumerate(skill_sequence[1:]):
                indices = np.flatnonzero(demonstration.skill == target_label)
                if len(indices) == 0:
                    raise ValueError("双臂事务构建发现示范缺少目标技能")
                positions[source].append(float(indices[0]) / denominator)
        return {
            source: np.asarray(values, dtype=np.float64)
            for source, values in positions.items()
        }

    def _boundary_straddling_link_frames(
        self,
        model: ClosedLoopTaskModel,
        boundary: BoundaryModel,
        demonstration_count: int,
    ) -> frozenset[str]:
        """Return all-demo LINK relations established across this boundary.

        A LINK event can only be confirmed after the object has moved with the
        end effector.  For cooperative grasp-and-carry skills, the physical
        attachment can therefore be established around a skill boundary while
        the formal event is first observable a few states into the target
        skill.  Treat that event as boundary evidence only when its saved
        approach context starts in the source skill, its first confirmed
        linked state belongs to the target skill, and every normal
        demonstration supports the event.  LINK_PENDING alone is deliberately
        excluded because it is not a confirmed physical relation.
        """

        result = set()
        expected_demonstrations = set(range(demonstration_count))
        for anchor in model.link_anchors.values():
            if (
                anchor.arm_id != model.arm_id
                or anchor.event_id.transition != "link"
                or anchor.event_id.skill_index != boundary.target_skill
                or anchor.context_state.skill_index != boundary.source_skill
                or anchor.support_fraction < 1.0 - 1.0e-12
                or set(anchor.demonstration_indices) != expected_demonstrations
                or not anchor.linked_entry_states
                or anchor.linked_entry_states[0].skill_index != boundary.target_skill
            ):
                continue
            target_indices = sorted(
                {
                    state.local_index
                    for state in anchor.linked_entry_states
                    if state.skill_index == boundary.target_skill
                }
            )
            longest_run = 0
            current_run = 0
            previous = None
            for index in target_indices:
                current_run = (
                    current_run + 1
                    if previous is not None and index == previous + 1
                    else 1
                )
                longest_run = max(longest_run, current_run)
                previous = index
            if longest_run >= self.config.relation_minimum_dwell:
                result.add(anchor.frame_id)
        return frozenset(result)

    def _joint_dependency_signature(
        self,
        model: ClosedLoopTaskModel,
        boundary: BoundaryModel,
        demonstration_count: int,
    ) -> tuple[frozenset[str], frozenset[FactorId]]:
        # Only this arm's own linked relation makes it a physical participant
        # in a shared coupling.  A guard that merely observes the other arm is
        # a directional dependency and must not by itself create a transaction.
        own_prefix = f"{model.arm_id}/"
        linked = {
            key.split("/", 1)[1]
            for conditions in (
                boundary.local_completion_model.own_relation_conditions,
                boundary.relation_conditions,
            )
            for key, condition in conditions.items()
            if key.startswith(own_prefix) and condition.required_state == "linked"
        }
        linked.update(
            self._boundary_straddling_link_frames(
                model,
                boundary,
                demonstration_count,
            )
        )
        return frozenset(linked), frozenset(boundary.scene_conditions)

    def _assign_transaction_groups(
        self,
        left_model: ClosedLoopTaskModel,
        right_model: ClosedLoopTaskModel,
        left_demonstrations: Sequence[DynaMACDemonstration],
        right_demonstrations: Sequence[DynaMACDemonstration],
    ) -> None:
        """Pair only mutually shared, consistently synchronized hard boundaries."""

        left_positions = self._boundary_positions(
            left_demonstrations,
            left_model.base_policy.skill_sequence,
        )
        right_positions = self._boundary_positions(
            right_demonstrations,
            right_model.base_policy.skill_sequence,
        )
        candidates = []
        for left_id, left_boundary in left_model.boundaries.items():
            left_linked, left_scene = self._joint_dependency_signature(
                left_model,
                left_boundary,
                len(left_demonstrations),
            )
            if not left_linked and not left_scene:
                continue
            for right_id, right_boundary in right_model.boundaries.items():
                right_linked, right_scene = self._joint_dependency_signature(
                    right_model,
                    right_boundary,
                    len(right_demonstrations),
                )
                shared_linked = left_linked.intersection(right_linked)
                shared_scene = left_scene.intersection(right_scene)
                if not shared_linked and not shared_scene:
                    continue
                shared = tuple(sorted(shared_linked)) + tuple(
                    factor.token for factor in sorted(shared_scene)
                )
                differences = np.abs(
                    left_positions[left_boundary.source_skill]
                    - right_positions[right_boundary.source_skill]
                )
                if np.any(differences > self.config.transaction_boundary_tolerance):
                    continue
                candidates.append(
                    (
                        float(np.mean(differences)),
                        left_id,
                        right_id,
                        shared,
                    )
                )
        used_left = set()
        used_right = set()
        for _, left_id, right_id, shared in sorted(candidates):
            if left_id in used_left or right_id in used_right:
                continue
            group_id = (
                f"joint:left-k{left_id.source_skill}-to-{left_id.target_skill}:"
                f"right-k{right_id.source_skill}-to-{right_id.target_skill}:"
                f"{'-'.join(shared)}"
            )
            left_boundary = left_model.boundaries[left_id]
            right_boundary = right_model.boundaries[right_id]
            merged_relation_conditions = {
                **left_boundary.relation_conditions,
                **right_boundary.relation_conditions,
            }
            locally_guaranteed = set(
                left_boundary.local_completion_model.own_relation_conditions
            ).union(right_boundary.local_completion_model.own_relation_conditions)
            merged_relation_conditions = {
                key: value
                for key, value in merged_relation_conditions.items()
                if key not in locally_guaranteed
            }

            left_condition_ids = (
                set(left_boundary.local_completion_model.own_relation_conditions)
                .union(factor.token for factor in left_boundary.scene_conditions)
                .union(merged_relation_conditions)
            )
            right_condition_ids = (
                set(right_boundary.local_completion_model.own_relation_conditions)
                .union(factor.token for factor in right_boundary.scene_conditions)
                .union(merged_relation_conditions)
            )
            left_reliability = {
                key: value
                for key, value in left_boundary.condition_reliability.items()
                if key in left_condition_ids
            }
            right_reliability = {
                key: value
                for key, value in right_boundary.condition_reliability.items()
                if key in right_condition_ids
            }
            for key in merged_relation_conditions:
                if key in left_boundary.condition_reliability:
                    statistics = left_boundary.condition_reliability[key]
                else:
                    statistics = right_boundary.condition_reliability[key]
                left_reliability[key] = statistics
                right_reliability[key] = statistics
            left_model.boundaries[left_id] = replace(
                left_boundary,
                relation_conditions=dict(merged_relation_conditions),
                condition_reliability=left_reliability,
                affected_arms=("left", "right"),
                transaction_group=group_id,
            )
            right_model.boundaries[right_id] = replace(
                right_boundary,
                relation_conditions=dict(merged_relation_conditions),
                condition_reliability=right_reliability,
                affected_arms=("left", "right"),
                transaction_group=group_id,
            )
            used_left.add(left_id)
            used_right.add(right_id)

    def build_bimanual(
        self,
        policy: BimanualDynaMAC,
        left_demonstrations: Sequence[DynaMACDemonstration],
        right_demonstrations: Sequence[DynaMACDemonstration],
        *,
        recoverable_frames: Sequence[str] | None = None,
    ) -> tuple[ClosedLoopTaskModel, ClosedLoopTaskModel]:
        """Build both arm-local models from exactly the snapshots used by fit."""

        if not policy.left.fitted or not policy.right.fitted:
            raise RuntimeError("双臂基础 DynaMAC 尚未拟合")
        paired_left, paired_right = synchronized_bimanual_demonstrations(
            left_demonstrations,
            right_demonstrations,
        )
        left_model = self.build(
            policy.left,
            paired_left,
            arm_id="left",
            recoverable_frames=recoverable_frames,
        )
        right_model = self.build(
            policy.right,
            paired_right,
            arm_id="right",
            recoverable_frames=recoverable_frames,
        )
        self._assign_transaction_groups(
            left_model,
            right_model,
            paired_left,
            paired_right,
        )
        return left_model, right_model


__all__ = ["ClosedLoopTaskModelBuilder", "ClosedLoopTaskModelConfig"]
