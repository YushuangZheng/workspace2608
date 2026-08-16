"""Tests for MiDiGaP constraints, skill transitions, and VAPOR."""

from __future__ import annotations

import numpy as np
import pytest
from essay2608.policy.dynamac import pose_parallel_transport
from essay2608.policy.midigap import (
    CollisionHalfSpace,
    MiDiGaP,
    MiDiGaPConfig,
    OccupancyConstraint,
    ReachabilitySphere,
    SelfCollisionSphere,
    VAPORConfig,
    constrained_midigap_update,
    gaussian_pose_kl,
    kl_transition_matrix,
    truncate_riemannian_gaussian,
    update_incoming_transitions,
    variance_aware_path_optimization,
)


def pose(position: list[float]) -> np.ndarray:
    return np.asarray([*position, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def test_standalone_midigap_fits_trajectory_modes() -> None:
    trajectories = []
    for index in range(6):
        y = -0.25 if index < 3 else 0.25
        trajectories.append(
            np.stack([pose([x, y + 0.002 * index, 0.4]) for x in np.linspace(0.2, 0.6, 8)])
        )
    model = MiDiGaP(MiDiGaPConfig(maximum_modes=2, minimum_mode_size=2, clustering_length=8)).fit(
        trajectories
    )
    assert model.fitted
    assert len(model.modes) == 2
    np.testing.assert_allclose(model.priors, [0.5, 0.5])
    assert model.most_likely_trajectory().shape == (8, 7)


@pytest.mark.parametrize(
    "config",
    [
        lambda: MiDiGaPConfig(position_variance_floor=np.nan),
        lambda: MiDiGaPConfig(dbscan_epsilon=np.inf),
        lambda: MiDiGaPConfig(gripper_clustering_scale=0.0),
        lambda: MiDiGaPConfig(covariance_estimation_method="diagonal_plus_ridge"),
        lambda: MiDiGaPConfig(maximum_modes=1.5),
        lambda: VAPORConfig(tolerance=np.nan),
        lambda: VAPORConfig(maximum_iterations=2.5),
    ],
)
def test_policy_configs_reject_nonfinite_and_noninteger_values(config) -> None:
    with pytest.raises(ValueError):
        config()


def test_policy_configs_canonicalize_numpy_scalars() -> None:
    midigap = MiDiGaPConfig(
        position_variance_floor=np.float32(1.0e-8),
        maximum_modes=np.int64(2),
        covariance_estimation_method="full_empirical_ridge",
    )
    vapor = VAPORConfig(
        lambda_pose=np.float32(1.0),
        maximum_iterations=np.int64(10),
    )
    assert type(midigap.position_variance_floor) is float
    assert type(midigap.maximum_modes) is int
    assert midigap.covariance_estimation_method == "full_empirical_ridge"
    assert midigap._dynamac_config().covariance_estimation_method == "full_empirical_ridge"
    assert type(vapor.lambda_pose) is float
    assert type(vapor.maximum_iterations) is int


def test_midigap_defaults_use_paper_covariance_regularization() -> None:
    config = MiDiGaPConfig()
    assert config.position_variance_floor == 1.0e-6
    assert config.rotation_variance_floor == 1.0e-6
    assert config.covariance_estimation_method == "diagonal_empirical_ridge"


def test_standalone_midigap_rejects_nonfinite_input_atomically() -> None:
    trajectories = [
        np.stack([pose([x, 0.01 * index, 0.4]) for x in np.linspace(0.2, 0.6, 6)])
        for index in range(3)
    ]
    model = MiDiGaP(MiDiGaPConfig(maximum_modes=1, minimum_mode_size=1)).fit(trajectories)
    previous_modes = model.modes
    previous_labels = model.mode_labels
    invalid = [item.copy() for item in trajectories]
    invalid[0][2, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        model.fit(invalid)

    assert model.modes is previous_modes
    assert model.mode_labels is previous_labels


def test_kl_transition_prefers_nearby_skill_boundary() -> None:
    source = np.stack((pose([0.0, 0.0, 0.0]), pose([1.0, 0.0, 0.0])))
    target = np.stack((pose([0.02, 0.0, 0.0]), pose([0.98, 0.0, 0.0])))
    covariance = np.repeat((np.eye(6) * 0.02)[None], 2, axis=0)
    transition = kl_transition_matrix(source, covariance, target, covariance)
    assert transition[0, 0] > transition[0, 1]
    assert transition[1, 1] > transition[1, 0]
    np.testing.assert_allclose(np.sum(transition, axis=1), 1.0)


def test_pose_kl_parallel_transports_anisotropic_source_covariance() -> None:
    source_mean = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    target_mean = np.asarray([0.2, -0.1, 0.3, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    source_covariance = np.diag([0.2, 0.4, 0.8, 0.01, 0.2, 0.9])
    target_covariance = np.diag([0.3, 0.5, 0.7, 0.6, 0.03, 0.4])
    transport = pose_parallel_transport(source_mean, target_mean)
    transported = transport @ source_covariance @ transport.T
    precision = np.linalg.inv(target_covariance)
    delta = np.asarray([-0.2, 0.1, -0.3, 0.0, 0.0, -np.pi / 4.0])
    expected = 0.5 * (
        np.trace(precision @ transported)
        + delta @ precision @ delta
        - 6.0
        + np.linalg.slogdet(target_covariance)[1]
        - np.linalg.slogdet(transported)[1]
    )
    np.testing.assert_allclose(
        gaussian_pose_kl(source_mean, source_covariance, target_mean, target_covariance),
        expected,
    )

    antipodal_target = source_mean.copy()
    antipodal_target[3:7] *= -1.0
    np.testing.assert_allclose(
        gaussian_pose_kl(source_mean, source_covariance, antipodal_target, source_covariance),
        0.0,
        atol=1.0e-12,
    )


def test_convex_truncation_and_equation_24_weight_update() -> None:
    mean = np.asarray(
        [
            [pose([-0.02, 0.0, 0.0]), pose([-0.02, 0.0, 0.0])],
            [pose([0.4, 0.0, 0.0]), pose([0.4, 0.0, 0.0])],
        ]
    )
    covariance = np.repeat((np.eye(6) * 0.01)[None, None], 4, axis=0).reshape(2, 2, 6, 6)
    constraint = CollisionHalfSpace(
        point=np.zeros(3), normal=np.asarray([1.0, 0.0, 0.0]), safety_distance=0.0
    )
    result = constrained_midigap_update(
        mean,
        covariance,
        np.asarray([0.5, 0.5]),
        constraint,
        sample_count=1500,
        random_seed=7,
    )
    assert result.priors[1] > result.priors[0]
    assert result.mean[0, 0, 0] > mean[0, 0, 0]
    np.testing.assert_allclose(np.sum(result.priors), 1.0)

    truncated = truncate_riemannian_gaussian(
        pose([0.0, 0.0, 0.0]),
        np.eye(6) * 0.01,
        ReachabilitySphere(np.zeros(3), 0.12),
        sample_count=1000,
        rng=np.random.default_rng(4),
    )
    assert truncated is not None
    assert 0.0 < truncated.acceptance_probability < 1.0


def test_reachability_uses_ellipsoid_not_anisotropic_outer_sphere() -> None:
    constraint = ReachabilitySphere(np.zeros(3), 0.1)
    covariance = np.diag([1.0e-6, 1.0e-2, 1.0e-6, 1.0e-4, 1.0e-4, 1.0e-4])

    # A covariance-enclosing sphere would reach the workspace through its large
    # y variance.  The actual confidence ellipsoid cannot move far enough in x.
    assert not constraint.confidence_intersects(pose([0.28, 0.0, 0.0]), covariance, 1.96)
    assert constraint.confidence_intersects(
        pose([0.12, 0.0, 0.0]),
        np.diag([4.0e-4, 1.0e-6, 1.0e-6, 1.0e-4, 1.0e-4, 1.0e-4]),
        1.96,
    )


def test_equation_20_self_collision_is_modal_only() -> None:
    mean = np.asarray([[pose([0.0, 0.0, 0.0])], [pose([0.5, 0.0, 0.0])]])
    covariance = np.repeat((np.eye(6) * 1.0e-6)[None, None], 2, axis=0)
    result = constrained_midigap_update(
        mean,
        covariance,
        np.asarray([0.5, 0.5]),
        SelfCollisionSphere(np.zeros(3), 0.2),
        sample_count=500,
        random_seed=19,
    )

    np.testing.assert_allclose(result.mean, mean)
    np.testing.assert_allclose(result.covariance, covariance)
    np.testing.assert_allclose(result.priors, [0.0, 1.0])
    np.testing.assert_array_equal(result.feasible_modes, [False, True])


def test_nonconvex_occupancy_only_changes_mode_weights() -> None:
    mean = np.asarray([[pose([0.0, 0.0, 0.0])], [pose([0.5, 0.0, 0.0])]])
    covariance = np.repeat((np.eye(6) * 1.0e-4)[None, None], 2, axis=0)
    constraint = OccupancyConstraint(lambda point: float(point[0] < 0.25), threshold=0.5)
    result = constrained_midigap_update(
        mean,
        covariance,
        np.asarray([0.5, 0.5]),
        constraint,
        sample_count=500,
        random_seed=9,
    )
    np.testing.assert_allclose(result.mean, mean)
    np.testing.assert_allclose(result.covariance, covariance)
    assert result.priors[1] > 0.99


def test_one_disjoint_confidence_timestep_eliminates_the_whole_mode() -> None:
    mean = np.asarray(
        [
            [pose([10.0, 0.0, 0.0]), pose([0.0, 0.0, 0.0])],
            [pose([0.0, 0.0, 0.0]), pose([0.0, 0.0, 0.0])],
        ]
    )
    covariance = np.repeat((np.eye(6) * 1.0e-6)[None, None], 4, axis=0).reshape(2, 2, 6, 6)
    result = constrained_midigap_update(
        mean,
        covariance,
        np.asarray([0.5, 0.5]),
        ReachabilitySphere(np.zeros(3), 1.0),
        sample_count=500,
        random_seed=13,
    )
    np.testing.assert_allclose(result.priors, [0.0, 1.0])
    np.testing.assert_array_equal(result.feasible_modes, [False, True])


def test_zero_occupancy_sample_acceptance_is_not_mistaken_for_disjoint_confidence() -> None:
    mean = np.asarray(
        [
            [pose([10.0, 0.0, 0.0]), pose([0.0, 0.0, 0.0])],
            [pose([0.0, 0.0, 0.0]), pose([0.0, 0.0, 0.0])],
        ]
    )
    covariance = np.repeat((np.eye(6) * 1.0e-8)[None, None], 4, axis=0).reshape(2, 2, 6, 6)
    constraint = OccupancyConstraint(lambda point: float(point[0] > 5.0), threshold=0.5)
    result = constrained_midigap_update(
        mean,
        covariance,
        np.asarray([0.5, 0.5]),
        constraint,
        sample_count=200,
        random_seed=13,
    )
    # OccupancyConstraint has no analytic confidence-region intersection test.
    # Zero hits in a finite sample only set p_R=0 at that time step; averaging
    # over time in Eq. (24) must not turn it into Eq. (22) mode elimination.
    np.testing.assert_allclose(result.priors, [1.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_array_equal(result.feasible_modes, [True, True])


def test_constraint_evidence_propagates_to_incoming_transitions() -> None:
    transition = np.asarray([[0.4, 0.6], [0.8, 0.2]])
    updated = update_incoming_transitions(transition, np.asarray([1.0, 0.1]))
    np.testing.assert_allclose(np.sum(updated, axis=1), 1.0)
    assert updated[0, 0] > transition[0, 0]
    assert updated[1, 0] > transition[1, 0]


@pytest.mark.parametrize(
    ("transition", "evidence"),
    [
        (np.asarray([[0.5, -0.5], [0.5, 0.5]]), np.ones(2)),
        (np.asarray([[0.5, 0.5], [0.5, 0.5]]), np.asarray([np.nan, 1.0])),
        (np.asarray([[0.4, 0.4], [0.5, 0.5]]), np.ones(2)),
    ],
)
def test_transition_evidence_update_rejects_invalid_probabilities(
    transition: np.ndarray,
    evidence: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        update_incoming_transitions(transition, evidence)


def test_vapor_tracks_pose_distribution_with_joint_limits() -> None:
    # Three prismatic joints form an analytic test robot with fixed orientation.
    def forward_kinematics(joints: np.ndarray) -> np.ndarray:
        return pose(joints.tolist())

    mean = np.stack([pose([x, 0.2 * x, 0.1]) for x in np.linspace(0.1, 0.5, 5)])
    covariance = np.repeat((np.eye(6) * 0.01)[None], len(mean), axis=0)
    result = variance_aware_path_optimization(
        mean,
        covariance,
        initial_joint_position=np.asarray([0.0, 0.0, 0.0]),
        forward_kinematics=forward_kinematics,
        joint_lower=np.asarray([-1.0, -1.0, -1.0]),
        joint_upper=np.asarray([1.0, 1.0, 1.0]),
        config=VAPORConfig(maximum_iterations=100),
    )
    assert result.success, result.message
    assert result.joint_trajectory.shape == (5, 3)
    assert result.maximum_normalized_deviation <= 1.96 + 1.0e-5
    assert np.all(result.joint_trajectory <= 1.0)
    assert np.all(result.joint_trajectory >= -1.0)


def test_vapor_finite_difference_augmented_lagrangian_backend() -> None:
    def forward_kinematics(joints: np.ndarray) -> np.ndarray:
        return pose(joints.tolist())

    mean = np.stack([pose([x, 0.2 * x, 0.1]) for x in np.linspace(0.1, 0.5, 5)])
    covariance = np.repeat((np.eye(6) * 0.01)[None], len(mean), axis=0)
    result = variance_aware_path_optimization(
        mean,
        covariance,
        initial_joint_position=np.zeros(3),
        forward_kinematics=forward_kinematics,
        joint_lower=np.full(3, -1.0),
        joint_upper=np.ones(3),
        config=VAPORConfig(maximum_iterations=40, solver="augmented_lagrangian_fd"),
    )
    assert result.success, result.message
