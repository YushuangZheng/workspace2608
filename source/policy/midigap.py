"""MiDiGaP policies, constraint updates, skill transitions, and VAPOR.

Poses use ``[x, y, z, qw, qx, qy, qz]`` throughout. Covariances use the
TAPAS ``R3 x S3`` convention: a shared translation basis and a quaternion
body-tangent basis. DynaMAC applies MiDiGaP to task-parameterized streams;
this module also provides the standalone MiDiGaP interfaces from Eqs.
(14)--(24) and VAPOR.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .dynamac import (
    CovarianceEstimationMethod,
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    SkillModel,
    StreamModel,
    _fit_pose_sequence,
    _gripper_modal_factor,
    _local_trajectories,
    _partition_modes,
    _pose_mean,
    _pose_residuals,
    _resample_poses,
    _resampled_skill_data,
    _skill_slice,
    _transition_probabilities,
    _validate_demonstrations,
    pose_exp_world,
    pose_log_world,
    pose_parallel_transport,
    static_task_parameter_scores,
)

Array = np.ndarray


@dataclass(frozen=True)
class MiDiGaPConfig:
    """Clustering and numerical-stability settings for MiDiGaP."""

    # MiDiGaP Sec. VI-A uses Sigma = Sigma_hat + lambda I with lambda=1e-6.
    position_variance_floor: float = 1.0e-6
    rotation_variance_floor: float = 1.0e-6
    covariance_estimation_method: CovarianceEstimationMethod = (
        "diagonal_empirical_ridge"
    )
    resampling_method: Literal["tapas_subsample", "interpolate"] = "tapas_subsample"
    # Sec. V-B: start new tasks with DBSCAN, then run a separately labelled
    # k-means variant when DBSCAN's constant distance fails to expose modes.
    modal_partition_method: Literal["riemannian_kmeans_bic", "riemannian_gmm_bic", "dbscan"] = (
        "dbscan"
    )
    maximum_modes: int = 8
    minimum_mode_size: int = 2
    clustering_length: int = 20
    gripper_clustering_scale: float = 1.0
    dbscan_epsilon: float = 0.1
    dbscan_min_samples: int = 2
    gmm_maximum_iterations: int = 100
    default_mode_strategy: Literal["map", "sample"] = "sample"
    random_seed: int = 2608

    def __post_init__(self) -> None:
        floating_names = (
            "position_variance_floor",
            "rotation_variance_floor",
            "gripper_clustering_scale",
            "dbscan_epsilon",
        )
        floating = tuple(getattr(self, name) for name in floating_names)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in floating
        ):
            raise ValueError("MiDiGaP floating-point settings must be real numbers")
        if not all(math.isfinite(float(value)) for value in floating):
            raise ValueError("MiDiGaP floating-point settings must be finite")
        integer_names = (
            "maximum_modes",
            "minimum_mode_size",
            "clustering_length",
            "dbscan_min_samples",
            "gmm_maximum_iterations",
            "random_seed",
        )
        integer = tuple(getattr(self, name) for name in integer_names)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in integer
        ):
            raise ValueError("MiDiGaP integer settings must be integers, not booleans")
        if self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        for name, value in zip(floating_names, floating, strict=True):
            object.__setattr__(self, name, float(value))
        for name, value in zip(integer_names, integer, strict=True):
            object.__setattr__(self, name, int(value))
        if self.position_variance_floor <= 0.0 or self.rotation_variance_floor <= 0.0:
            raise ValueError("covariance floors must be positive")
        if self.covariance_estimation_method not in {
            "diagonal_empirical_ridge",
            "diagonal_empirical_spd_floor",
            "full_empirical_ridge",
        }:
            raise ValueError("unknown covariance_estimation_method")
        if self.maximum_modes < 1 or self.minimum_mode_size < 1:
            raise ValueError("mode-count settings must be positive")
        if self.gripper_clustering_scale <= 0.0:
            raise ValueError("gripper_clustering_scale must be positive")
        if self.clustering_length < 2:
            raise ValueError("clustering_length must be at least 2")
        if self.resampling_method not in {"tapas_subsample", "interpolate"}:
            raise ValueError("unknown resampling_method")
        if self.modal_partition_method not in {
            "riemannian_kmeans_bic",
            "riemannian_gmm_bic",
            "dbscan",
        }:
            raise ValueError("unknown modal_partition_method")
        if self.dbscan_epsilon <= 0.0 or self.dbscan_min_samples < 1:
            raise ValueError("invalid DBSCAN settings")
        if self.gmm_maximum_iterations < 1:
            raise ValueError("gmm_maximum_iterations must be positive")
        if self.default_mode_strategy not in {"map", "sample"}:
            raise ValueError("unknown default_mode_strategy")

    def _dynamac_config(self) -> DynaMACConfig:
        return DynaMACConfig(
            position_variance_floor=self.position_variance_floor,
            rotation_variance_floor=self.rotation_variance_floor,
            covariance_estimation_method=self.covariance_estimation_method,
            resampling_method=self.resampling_method,
            modal_partition_method=self.modal_partition_method,
            maximum_modes=self.maximum_modes,
            minimum_mode_size=self.minimum_mode_size,
            clustering_length=self.clustering_length,
            gripper_clustering_scale=self.gripper_clustering_scale,
            dbscan_epsilon=self.dbscan_epsilon,
            dbscan_min_samples=self.dbscan_min_samples,
            gmm_maximum_iterations=self.gmm_maximum_iterations,
            default_mode_strategy=self.default_mode_strategy,
            random_seed=self.random_seed,
        )


@dataclass(frozen=True)
class MiDiGaPMode:
    mean: Array  # [T, 7]
    covariance: Array  # [T, 6, 6]
    prior: float
    demonstration_indices: tuple[int, ...]


class MiDiGaP:
    """Cluster trajectories on ``M^T``, then fit a DiGaP at each time step."""

    name = "midigap"

    def __init__(self, config: MiDiGaPConfig = MiDiGaPConfig()) -> None:
        self.config = config
        self.duration = 0
        self.mode_labels = np.empty(0, dtype=np.int64)
        self.modes: tuple[MiDiGaPMode, ...] = ()
        self._rng = np.random.default_rng(config.random_seed)

    @property
    def fitted(self) -> bool:
        return bool(self.modes)

    @property
    def priors(self) -> Array:
        if not self.fitted:
            raise RuntimeError("MiDiGaP has not been fitted")
        return np.asarray([mode.prior for mode in self.modes], dtype=np.float64)

    def fit(self, trajectories: Sequence[Array]) -> MiDiGaP:
        """Fit semantically aligned, variable-length end-effector trajectories."""

        values = [np.asarray(item, dtype=np.float64) for item in trajectories]
        if not values:
            raise ValueError("at least one trajectory is required")
        if any(
            item.ndim != 2 or item.shape[1] != 7 or len(item) == 0 or not np.all(np.isfinite(item))
            for item in values
        ):
            raise ValueError("each trajectory must be a finite, nonempty [T, 7] array")
        duration = max(int(float(np.mean([len(item) for item in values]))), 1)
        resampled = np.stack(
            [_resample_poses(item, duration, self.config.resampling_method) for item in values]
        )
        mode_labels = _partition_modes(resampled, self.config._dynamac_config())
        modes = []
        for label in range(int(np.max(mode_labels)) + 1):
            members = np.flatnonzero(mode_labels == label)
            mean, covariance = _fit_pose_sequence(
                resampled[members],
                self.config.position_variance_floor,
                self.config.rotation_variance_floor,
                covariance_estimation_method=self.config.covariance_estimation_method,
            )
            modes.append(
                MiDiGaPMode(
                    mean=mean,
                    covariance=covariance,
                    prior=float(len(members) / len(values)),
                    demonstration_indices=tuple(int(index) for index in members),
                )
            )
        self.duration = duration
        self.mode_labels = mode_labels
        self.modes = tuple(modes)
        return self

    def select_mode(self, strategy: Literal["map", "sample"] | None = None) -> int:
        strategy = self.config.default_mode_strategy if strategy is None else strategy
        if strategy == "map":
            return int(np.argmax(self.priors))
        if strategy == "sample":
            return int(self._rng.choice(len(self.modes), p=self.priors))
        raise ValueError(f"unknown mode-selection strategy: {strategy}")

    def most_likely_trajectory(self) -> Array:
        return self.modes[self.select_mode("map")].mean.copy()

    def sample_trajectory(self) -> Array:
        """Sample a mean trajectory from the MiDiGaP mode prior."""

        return self.modes[self.select_mode("sample")].mean.copy()


class TaskParameterizedMiDiGaP(DynaMAC):
    """Static-frame MiDiGaP policy used as the comparison baseline.

    The policy shares DynaMAC's task-parameterized streams, MiDiGaP modes,
    and product-of-experts inference. It does not detect kinematic links or
    add virtual end-effector frames, so objects moved by the robot retain the
    baseline's exogeneity and causal-collapse limitation. The default stream
    follows the time-state convention and uses the current end-effector pose.
    Set ``policy_model="action_pose"`` explicitly for next-action targets.
    """

    name = "midigap_static_frames"

    def fit(
        self,
        demonstrations: Sequence[DynaMACDemonstration],
    ) -> TaskParameterizedMiDiGaP:
        frame_names, skill_sequence = _validate_demonstrations(demonstrations)
        fitted_skills = []
        previous_mode_labels: Array | None = None
        for label in skill_sequence:
            lengths = [len(_skill_slice(item, label)) for item in demonstrations]
            duration = max(int(float(np.mean(lengths))), 1)
            ee, actions, frames, extra = _resampled_skill_data(
                demonstrations,
                label,
                duration,
                virtual_starts={},
                resampling_method=self.config.resampling_method,
            )
            policy_pose = ee if self.config.policy_model == "time_state" else actions
            local_policy = {
                name: _local_trajectories(frame_values, policy_pose)
                for name, frame_values in frames.items()
            }
            fitted = {
                name: _fit_pose_sequence(
                    values,
                    self.config.position_variance_floor,
                    self.config.rotation_variance_floor,
                    covariance_estimation_method=self.config.covariance_estimation_method,
                )
                for name, values in local_policy.items()
            }
            static_covariances = {
                name: covariance for name, (_, covariance) in fitted.items()
            }
            scores = static_task_parameter_scores(static_covariances)
            selected = tuple(name for name in frame_names if scores[name] > self.config.tau_omega)
            if not selected:
                raise RuntimeError(
                    f"skill {label} has no static task parameter above "
                    f"tau_omega={self.config.tau_omega}"
                )

            # Cluster on the highest-scoring Eq. (6) task parameter. This avoids
            # duplicating equivalent representations in a multi-stream distance.
            clustering_frame = max(selected, key=scores.__getitem__)
            mode_labels = _partition_modes(
                local_policy[clustering_frame],
                self.config,
                _gripper_modal_factor(extra["gripper"], self.config.gripper_clustering_scale),
            )
            mode_count = int(np.max(mode_labels)) + 1
            priors = np.asarray(
                [np.mean(mode_labels == mode) for mode in range(mode_count)],
                dtype=np.float64,
            )
            mode_demonstration_indices = tuple(
                tuple(int(index) for index in np.flatnonzero(mode_labels == mode))
                for mode in range(mode_count)
            )
            transition = (
                None
                if previous_mode_labels is None
                else _transition_probabilities(previous_mode_labels, mode_labels)
            )
            streams = {}
            gripper = []
            for mode in range(mode_count):
                members = mode_labels == mode
                gripper.append(np.mean(extra["gripper"][members], axis=0))
            for name in selected:
                means = []
                covariances = []
                for mode in range(mode_count):
                    members = mode_labels == mode
                    mean, covariance = _fit_pose_sequence(
                        local_policy[name][members],
                        self.config.position_variance_floor,
                        self.config.rotation_variance_floor,
                        covariance_estimation_method=self.config.covariance_estimation_method,
                    )
                    means.append(mean)
                    covariances.append(covariance)
                streams[name] = StreamModel(name, np.stack(means), np.stack(covariances))
            fitted_skills.append(
                SkillModel(
                    label=int(label),
                    duration=duration,
                    selected_frames=selected,
                    mode_priors=priors,
                    streams=streams,
                    gripper=np.stack(gripper),
                    transition_from_previous=transition,
                    mode_demonstration_indices=mode_demonstration_indices,
                    link_diagnostics={
                        name: {"linked": False, "reason": "static_frame_baseline"}
                        for name in frame_names
                    },
                    selection_scores=scores,
                )
            )
            previous_mode_labels = mode_labels
        self.frame_names = frame_names
        self.skill_sequence = skill_sequence
        self.skills = fitted_skills
        self._invalidate_episode_after_fit()
        return self


class PoseConstraint(Protocol):
    """Feasible pose region for MiDiGaP."""

    supports_moment_matching: bool

    def contains(self, pose: Array) -> bool: ...

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool: ...


def _confidence_ellipsoid_minimum_distance(
    mean: Array,
    covariance: Array,
    point: Array,
    z: float,
) -> float:
    """Exact Euclidean distance from a point to a positional confidence ellipsoid.

    The ellipsoid is ``(x-mean)^T covariance^-1 (x-mean) <= z^2``.  Projecting
    a point onto it reduces, in the covariance eigenbasis, to a scalar secular
    equation.  This is the Eq. (22) intersection test needed for Eq. (17); an
    enclosing covariance sphere can otherwise report false intersections for
    strongly anisotropic Gaussians.
    """

    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    if mean.shape != (3,) or covariance.shape != (3, 3) or point.shape != (3,):
        raise ValueError("confidence-ellipsoid distance requires 3D mean, covariance, and point")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("confidence-ellipsoid parameters must be finite")
    if not np.all(np.isfinite(point)) or not math.isfinite(z) or z <= 0.0:
        raise ValueError("confidence-ellipsoid point must be finite and z must be positive")

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    if np.any(eigenvalues <= 0.0):
        raise ValueError("confidence-ellipsoid covariance must be positive definite")
    offset = eigenvectors.T @ (mean - point)
    if float(np.sum(np.square(offset) / eigenvalues)) <= z * z:
        return 0.0

    def secular(multiplier: float) -> float:
        return float(
            np.sum(
                np.square(offset)
                * eigenvalues
                / np.square(eigenvalues + multiplier)
            )
            - z * z
        )

    lower = 0.0
    upper = float(np.max(eigenvalues))
    while secular(upper) > 0.0:
        upper *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if secular(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    multiplier = upper
    residual = offset * multiplier / (eigenvalues + multiplier)
    return float(np.linalg.norm(residual))


@dataclass(frozen=True)
class ReachabilitySphere:
    """Spherical workspace from Eq. (17)."""

    center: Array
    radius: float
    supports_moment_matching: bool = True

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or self.radius <= 0.0:
            raise ValueError("reachability sphere requires a 3D center and positive radius")
        object.__setattr__(self, "center", center)

    def contains(self, pose: Array) -> bool:
        return bool(np.linalg.norm(np.asarray(pose)[:3] - self.center) <= self.radius)

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        distance = _confidence_ellipsoid_minimum_distance(
            np.asarray(mean)[:3],
            np.asarray(covariance)[:3, :3],
            self.center,
            z,
        )
        return distance <= self.radius


@dataclass(frozen=True)
class SelfCollisionSphere:
    """Minimum end-effector-to-base distance constraint from Eq. (20).

    The exterior of a sphere is nonconvex in 3D, so this constraint updates
    only mode weights and does not moment-match Gaussians. The confidence test
    uses a bounding-sphere upper bound. Equation (22) rejects a mode only when
    its entire pose confidence ellipsoid is proven to lie in the forbidden
    region.
    """

    center: Array
    minimum_distance: float
    supports_moment_matching: bool = False

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or not math.isfinite(self.minimum_distance)
            or self.minimum_distance <= 0.0
        ):
            raise ValueError(
                "self-collision sphere requires a finite 3D center and positive distance"
            )
        object.__setattr__(self, "center", center)

    def contains(self, pose: Array) -> bool:
        return bool(
            np.linalg.norm(np.asarray(pose)[:3] - self.center) >= self.minimum_distance
        )

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        position_covariance = np.asarray(covariance, dtype=np.float64)[:3, :3]
        eigenvalues = np.linalg.eigvalsh(0.5 * (position_covariance + position_covariance.T))
        if np.any(eigenvalues <= 0.0):
            raise ValueError("self-collision covariance must be positive definite")
        outer_radius = z * math.sqrt(float(np.max(eigenvalues)))
        maximum_possible_distance = (
            float(np.linalg.norm(np.asarray(mean)[:3] - self.center)) + outer_radius
        )
        return maximum_possible_distance >= self.minimum_distance


@dataclass(frozen=True)
class CollisionHalfSpace:
    """Collision half-space from Eqs. (18)--(19).

    The feasible side satisfies ``normal @ (x - point) >= safety_distance``.
    """

    point: Array
    normal: Array
    safety_distance: float = 0.0
    supports_moment_matching: bool = True

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if point.shape != (3,) or normal.shape != (3,) or norm < 1.0e-12:
            raise ValueError("collision half-space requires a 3D point and nonzero normal")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", normal / norm)

    def _margin(self, pose: Array) -> float:
        return float(self.normal @ (np.asarray(pose)[:3] - self.point))

    def contains(self, pose: Array) -> bool:
        return self._margin(pose) >= self.safety_distance

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        variance = float(self.normal @ covariance[:3, :3] @ self.normal)
        return self._margin(mean) + z * math.sqrt(max(variance, 0.0)) >= self.safety_distance


@dataclass(frozen=True)
class IntersectionConstraint:
    constraints: tuple[PoseConstraint, ...]

    def __init__(self, constraints: Sequence[PoseConstraint]) -> None:
        if not constraints:
            raise ValueError("intersection constraint cannot be empty")
        object.__setattr__(self, "constraints", tuple(constraints))

    @property
    def supports_moment_matching(self) -> bool:
        return all(item.supports_moment_matching for item in self.constraints)

    def contains(self, pose: Array) -> bool:
        return all(item.contains(pose) for item in self.constraints)

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        return all(item.confidence_intersects(mean, covariance, z) for item in self.constraints)


@dataclass(frozen=True)
class OccupancyConstraint:
    """Nonconvex occupancy constraint from Eqs. (21)--(22)."""

    occupancy: Callable[[Array], float]
    threshold: float
    supports_moment_matching: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("occupancy threshold must lie in [0, 1]")

    def contains(self, pose: Array) -> bool:
        return float(self.occupancy(np.asarray(pose)[:3])) < self.threshold

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        # Arbitrary occupancy fields have no reliable analytic confidence-region
        # intersection test, so feasibility is estimated by Monte Carlo sampling.
        return True


@dataclass(frozen=True)
class TruncatedGaussian:
    mean: Array
    covariance: Array
    acceptance_probability: float
    accepted_samples: int


def sample_riemannian_gaussian(
    mean: Array,
    covariance: Array,
    sample_count: int,
    rng: np.random.Generator,
) -> Array:
    """Sample in the mean tangent space and map back to ``R3 x S3`` with Exp."""

    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.shape != (7,) or covariance.shape != (6, 6) or sample_count < 1:
        raise ValueError("sampling requires a [7] mean, [6, 6] covariance, and positive count")
    symmetric = 0.5 * (covariance + covariance.T)
    tangent = rng.multivariate_normal(np.zeros(6), symmetric, size=sample_count)
    return np.stack([pose_exp_world(mean, value) for value in tangent])


def truncate_riemannian_gaussian(
    mean: Array,
    covariance: Array,
    constraint: PoseConstraint,
    *,
    sample_count: int = 1000,
    rng: np.random.Generator | None = None,
    variance_floor: float = 1.0e-10,
) -> TruncatedGaussian | None:
    """Apply Monte Carlo truncation and Frechet moment matching (Eqs. 15--16)."""

    rng = np.random.default_rng(2608) if rng is None else rng
    samples = sample_riemannian_gaussian(mean, covariance, sample_count, rng)
    accepted = samples[[constraint.contains(sample) for sample in samples]]
    probability = float(len(accepted) / sample_count)
    if len(accepted) == 0:
        return None
    updated_mean = _pose_mean(accepted)
    residuals = _pose_residuals(updated_mean, accepted)
    denominator = max(len(accepted) - 1, 1)
    diagonal = np.sum(np.square(residuals), axis=0) / denominator
    updated_covariance = np.diag(np.maximum(diagonal, variance_floor))
    return TruncatedGaussian(updated_mean, updated_covariance, probability, len(accepted))


@dataclass(frozen=True)
class ConstraintUpdate:
    mean: Array
    covariance: Array
    priors: Array
    acceptance_probability: Array  # [M, T]
    feasible_modes: Array


def constrained_midigap_update(
    mean: Array,
    covariance: Array,
    priors: Array,
    constraint: PoseConstraint,
    *,
    sample_count: int = 1000,
    q: float = 1.0,
    confidence_z: float = 1.96,
    update_gaussians: bool = True,
    random_seed: int = 2608,
) -> ConstraintUpdate:
    """Apply the MiDiGaP constraint update from Eqs. (15)--(24).

    A nonconvex ``OccupancyConstraint`` leaves each Gaussian unchanged and
    updates only its mode weight using the feasible-sample rate.
    """

    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    if mean.ndim != 3 or mean.shape[-1] != 7:
        raise ValueError("mean must have shape [M, T, 7]")
    if covariance.shape != mean.shape[:2] + (6, 6) or priors.shape != (len(mean),):
        raise ValueError("covariance or mode-prior shape does not match mean")
    if sample_count < 1 or q <= 0.0 or confidence_z <= 0.0:
        raise ValueError("sample_count, q, and confidence_z must be positive")
    if np.any(priors < 0.0) or not np.isclose(np.sum(priors), 1.0):
        raise ValueError("mode priors must be nonnegative and sum to 1")

    updated_mean = mean.copy()
    updated_covariance = covariance.copy()
    acceptance = np.zeros(mean.shape[:2], dtype=np.float64)
    hard_feasible = np.ones(mean.shape[0], dtype=bool)
    rng = np.random.default_rng(random_seed)
    for mode in range(mean.shape[0]):
        for time_index in range(mean.shape[1]):
            item_mean = mean[mode, time_index]
            item_covariance = covariance[mode, time_index]
            if not constraint.confidence_intersects(item_mean, item_covariance, confidence_z):
                # Equation (22) assigns zero likelihood to an entire trajectory
                # mode if any 95% confidence region is disjoint from the feasible set.
                hard_feasible[mode] = False
                continue
            truncated = truncate_riemannian_gaussian(
                item_mean,
                item_covariance,
                constraint,
                sample_count=sample_count,
                rng=rng,
            )
            if truncated is None:
                # Zero accepted samples from a finite Monte Carlo run do not prove
                # the confidence-region disjointness required by Eq. (22). Leave
                # this time step's empirical acceptance rate at zero; other steps
                # can still contribute to Eq. (24).
                continue
            acceptance[mode, time_index] = truncated.acceptance_probability
            if update_gaussians and constraint.supports_moment_matching:
                updated_mean[mode, time_index] = truncated.mean
                updated_covariance[mode, time_index] = truncated.covariance

    evidence = np.mean(np.power(acceptance, q), axis=1) ** (1.0 / q)
    evidence[~hard_feasible] = 0.0
    unnormalized = priors * evidence
    total = float(np.sum(unnormalized))
    if total <= 0.0:
        raise RuntimeError("constraint eliminated all MiDiGaP modes")
    return ConstraintUpdate(
        mean=updated_mean,
        covariance=updated_covariance,
        priors=unnormalized / total,
        acceptance_probability=acceptance,
        feasible_modes=hard_feasible & (evidence > 0.0),
    )


def update_incoming_transitions(transition: Array, target_evidence: Array) -> Array:
    """Propagate target-skill evidence to Eq. (12) incoming transitions."""

    transition = np.asarray(transition, dtype=np.float64)
    target_evidence = np.asarray(target_evidence, dtype=np.float64)
    if transition.ndim != 2 or target_evidence.shape != (transition.shape[1],):
        raise ValueError("transition matrix and target evidence have incompatible shapes")
    if not np.all(np.isfinite(transition)) or not np.all(np.isfinite(target_evidence)):
        raise ValueError("transition matrix and target evidence must be finite")
    if np.any(transition < 0.0) or np.any(target_evidence < 0.0):
        raise ValueError("transition probabilities and target evidence must be nonnegative")
    transition_row_sum = np.sum(transition, axis=1)
    if np.any(transition_row_sum <= 0.0) or not np.allclose(transition_row_sum, 1.0):
        raise ValueError("transition matrix rows must sum to 1")
    weighted = transition * target_evidence[None, :]
    row_sum = np.sum(weighted, axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("target evidence eliminated every outgoing edge from a source mode")
    return weighted / row_sum


def gaussian_pose_kl(
    source_mean: Array,
    source_covariance: Array,
    target_mean: Array,
    target_covariance: Array,
) -> float:
    """Compute the 6D Gaussian KL in the target-mean tangent space (Eq. 14)."""

    source_covariance = np.asarray(source_covariance, dtype=np.float64)
    target_covariance = np.asarray(target_covariance, dtype=np.float64)
    if source_covariance.shape != (6, 6) or target_covariance.shape != (6, 6):
        raise ValueError("KL covariances must have shape [6, 6]")
    target_mean = np.asarray(target_mean, dtype=np.float64).copy()
    source_mean = np.asarray(source_mean, dtype=np.float64).copy()
    if float(np.dot(source_mean[3:7], target_mean[3:7])) < 0.0:
        source_mean[3:7] *= -1.0
    transport = pose_parallel_transport(source_mean, target_mean)
    transported_source_covariance = transport @ source_covariance @ transport.T
    target_precision = np.linalg.inv(target_covariance)
    delta = pose_log_world(target_mean, source_mean)
    source_sign, source_logdet = np.linalg.slogdet(transported_source_covariance)
    target_sign, target_logdet = np.linalg.slogdet(target_covariance)
    if source_sign <= 0.0 or target_sign <= 0.0:
        raise ValueError("KL covariances must be positive definite")
    value = 0.5 * (
        np.trace(target_precision @ transported_source_covariance)
        + delta @ target_precision @ delta
        - 6.0
        + target_logdet
        - source_logdet
    )
    return float(max(value, 0.0))


def kl_transition_matrix(
    source_end_mean: Array,
    source_end_covariance: Array,
    target_start_mean: Array,
    target_start_covariance: Array,
) -> Array:
    """Build ``exp(-KL)`` mode transitions for unpaired skills (Eq. 14)."""

    source_end_mean = np.asarray(source_end_mean, dtype=np.float64)
    source_end_covariance = np.asarray(source_end_covariance, dtype=np.float64)
    target_start_mean = np.asarray(target_start_mean, dtype=np.float64)
    target_start_covariance = np.asarray(target_start_covariance, dtype=np.float64)
    if source_end_mean.ndim != 2 or target_start_mean.ndim != 2:
        raise ValueError("skill-boundary means must have shape [M, 7]")
    if source_end_covariance.shape != (len(source_end_mean), 6, 6):
        raise ValueError("source skill-boundary covariance shape does not match")
    if target_start_covariance.shape != (len(target_start_mean), 6, 6):
        raise ValueError("target skill-boundary covariance shape does not match")
    divergence = np.asarray(
        [
            [
                gaussian_pose_kl(source_mean, source_cov, target_mean, target_cov)
                for target_mean, target_cov in zip(
                    target_start_mean, target_start_covariance, strict=True
                )
            ]
            for source_mean, source_cov in zip(source_end_mean, source_end_covariance, strict=True)
        ]
    )
    # Subtract each row minimum to avoid underflow from large KL values.
    compatibility = np.exp(-(divergence - np.min(divergence, axis=1, keepdims=True)))
    return compatibility / np.sum(compatibility, axis=1, keepdims=True)


@dataclass(frozen=True)
class VAPORConfig:
    lambda_pose: float = 1.0
    lambda_joint: float = 0.1
    confidence_z: float = 1.96
    maximum_iterations: int = 300
    tolerance: float = 1.0e-7
    solver: Literal["slsqp", "augmented_lagrangian_fd"] = "slsqp"

    def __post_init__(self) -> None:
        floating_names = ("lambda_pose", "lambda_joint", "confidence_z", "tolerance")
        floating = tuple(getattr(self, name) for name in floating_names)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in floating
        ):
            raise ValueError("VAPOR floating-point settings must be real numbers")
        if not all(math.isfinite(float(value)) for value in floating):
            raise ValueError("VAPOR floating-point settings must be finite")
        if isinstance(self.maximum_iterations, (bool, np.bool_)) or not isinstance(
            self.maximum_iterations, (int, np.integer)
        ):
            raise ValueError("VAPOR maximum_iterations must be an integer")
        for name, value in zip(floating_names, floating, strict=True):
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "maximum_iterations", int(self.maximum_iterations))
        if self.lambda_pose <= 0.0 or self.lambda_joint < 0.0:
            raise ValueError("invalid VAPOR cost weights")
        if self.confidence_z <= 0.0 or self.maximum_iterations < 1 or self.tolerance <= 0.0:
            raise ValueError("invalid VAPOR constraint or solver settings")
        if self.solver not in {"slsqp", "augmented_lagrangian_fd"}:
            raise ValueError("VAPOR solver must be slsqp or augmented_lagrangian_fd")


@dataclass(frozen=True)
class VAPORResult:
    joint_trajectory: Array
    pose_trajectory: Array
    objective: float
    success: bool
    message: str
    maximum_normalized_deviation: float


def variance_aware_path_optimization(
    mean: Array,
    covariance: Array,
    initial_joint_position: Array,
    forward_kinematics: Callable[[Array], Array],
    joint_lower: Array,
    joint_upper: Array,
    config: VAPORConfig = VAPORConfig(),
) -> VAPORResult:
    """Optimize a variance-aware joint trajectory using VAPOR Eqs. (29)--(32).

    The reference method uses a Kineverse Jacobian and an augmented-Lagrangian
    solver. The default backend applies SciPy SLSQP to the same objective, joint
    limits, and component-wise 95% confidence constraints, with derivatives
    obtained by finite differences. Set
    ``VAPORConfig(solver="augmented_lagrangian_fd")`` to use a finite-difference
    augmented-Lagrangian backend for the same objective and update rule.
    """

    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("VAPOR requires the optional dependency: pip install -e '.[midigap]'") from error

    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    initial_joint_position = np.asarray(initial_joint_position, dtype=np.float64)
    joint_lower = np.asarray(joint_lower, dtype=np.float64)
    joint_upper = np.asarray(joint_upper, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != 7 or covariance.shape != (len(mean), 6, 6):
        raise ValueError("VAPOR trajectories must have shapes [T, 7] and [T, 6, 6]")
    joints = len(initial_joint_position)
    if joint_lower.shape != (joints,) or joint_upper.shape != (joints,):
        raise ValueError("joint-limit shapes do not match the initial joint position")
    if np.any(joint_lower >= joint_upper):
        raise ValueError("each joint lower bound must be below its upper bound")
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    if np.any(diagonal <= 0.0):
        raise ValueError("VAPOR covariance diagonals must be positive")
    sigma_max = float(np.max(diagonal))
    normalized_precision = np.stack([np.linalg.inv(item / sigma_max) for item in covariance])
    allowed = config.confidence_z * np.sqrt(diagonal)

    def pose_errors(flat: Array) -> Array:
        trajectory = flat.reshape(len(mean), joints)
        return np.stack(
            [pose_log_world(target, forward_kinematics(q)) for target, q in zip(mean, trajectory)]
        )

    def objective(flat: Array) -> float:
        trajectory = flat.reshape(len(mean), joints)
        errors = pose_errors(flat)
        pose_cost = sum(
            error @ precision @ error
            for error, precision in zip(errors, normalized_precision, strict=True)
        )
        smoothness = float(np.sum(np.square(np.diff(trajectory, axis=0))))
        return float(config.lambda_pose * pose_cost + config.lambda_joint * smoothness)

    # Initialize with final-target IK, then linearly interpolate from the current
    # joint position to that solution.
    final_target = mean[-1]

    def final_ik_cost(q: Array) -> float:
        error = pose_log_world(final_target, forward_kinematics(q))
        return float(error @ normalized_precision[-1] @ error)

    final_ik = minimize(
        final_ik_cost,
        np.clip(initial_joint_position, joint_lower, joint_upper),
        method="L-BFGS-B",
        bounds=list(zip(joint_lower, joint_upper, strict=True)),
        options={"maxiter": config.maximum_iterations, "ftol": config.tolerance},
    )
    fractions = np.linspace(0.0, 1.0, len(mean))[:, None]
    initial_path = (
        initial_joint_position[None]
        + fractions * (np.asarray(final_ik.x) - initial_joint_position)[None]
    )
    bounds = list(
        zip(np.tile(joint_lower, len(mean)), np.tile(joint_upper, len(mean)), strict=True)
    )

    def confidence_constraint(flat: Array) -> Array:
        # SLSQP expects g(x) >= 0. Split the absolute value into smooth upper
        # and lower linear inequalities.
        errors = pose_errors(flat)
        return np.concatenate(((allowed - errors).ravel(), (allowed + errors).ravel()))

    if config.solver == "slsqp":
        result = minimize(
            objective,
            initial_path.ravel(),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": confidence_constraint},
            options={
                "maxiter": config.maximum_iterations,
                "ftol": config.tolerance,
                "disp": False,
            },
        )
    else:
        # Apply the augmented-Lagrangian objective with finite differences to
        # the supplied forward-kinematics function.
        flat = initial_path.ravel()
        multipliers = np.zeros_like(confidence_constraint(flat))
        penalty = 10.0
        result = None
        for _ in range(config.maximum_iterations):

            def augmented_objective(value):
                constraints = confidence_constraint(value)
                violation = np.minimum(constraints, 0.0)
                return float(
                    objective(value)
                    - multipliers @ constraints
                    + 0.5 * penalty * (violation @ violation)
                )

            result = minimize(
                augmented_objective,
                flat,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": max(20, config.maximum_iterations // 4),
                    "ftol": config.tolerance,
                },
            )
            flat = np.asarray(result.x)
            constraints = confidence_constraint(flat)
            if float(np.max(np.maximum(-constraints, 0.0))) <= config.tolerance:
                break
            multipliers = np.maximum(0.0, multipliers - penalty * constraints)
            penalty *= 2.0
        assert result is not None
        final_constraints = confidence_constraint(flat)
        result.fun = objective(flat)
        result.success = bool(np.max(np.maximum(-final_constraints, 0.0)) <= config.tolerance)
        result.message = (
            "finite-difference augmented Lagrangian converged"
            if result.success
            else "finite-difference augmented Lagrangian reached its iteration limit"
        )
    joint_trajectory = np.asarray(result.x).reshape(len(mean), joints)
    pose_trajectory = np.stack([forward_kinematics(q) for q in joint_trajectory])
    deviations = np.abs(
        np.stack([pose_log_world(target, actual) for target, actual in zip(mean, pose_trajectory)])
    ) / np.sqrt(diagonal)
    return VAPORResult(
        joint_trajectory=joint_trajectory,
        pose_trajectory=pose_trajectory,
        objective=float(result.fun),
        success=bool(result.success and np.max(deviations) <= config.confidence_z + 1.0e-5),
        message=str(result.message),
        maximum_normalized_deviation=float(np.max(deviations)),
    )


__all__ = [
    "CollisionHalfSpace",
    "ConstraintUpdate",
    "IntersectionConstraint",
    "MiDiGaP",
    "MiDiGaPConfig",
    "MiDiGaPMode",
    "OccupancyConstraint",
    "PoseConstraint",
    "ReachabilitySphere",
    "SelfCollisionSphere",
    "TruncatedGaussian",
    "TaskParameterizedMiDiGaP",
    "VAPORConfig",
    "VAPORResult",
    "constrained_midigap_update",
    "gaussian_pose_kl",
    "kl_transition_matrix",
    "sample_riemannian_gaussian",
    "truncate_riemannian_gaussian",
    "update_incoming_transitions",
    "variance_aware_path_optimization",
]
