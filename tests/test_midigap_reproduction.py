"""MiDiGaP 约束更新、技能拼接与 VAPOR 复现测试。"""

from __future__ import annotations

import numpy as np
from essay2608.policy.midigap import (
    CollisionHalfSpace,
    MiDiGaP,
    MiDiGaPConfig,
    OccupancyConstraint,
    ReachabilitySphere,
    VAPORConfig,
    constrained_midigap_update,
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


def test_kl_transition_prefers_nearby_skill_boundary() -> None:
    source = np.stack((pose([0.0, 0.0, 0.0]), pose([1.0, 0.0, 0.0])))
    target = np.stack((pose([0.02, 0.0, 0.0]), pose([0.98, 0.0, 0.0])))
    covariance = np.repeat((np.eye(6) * 0.02)[None], 2, axis=0)
    transition = kl_transition_matrix(source, covariance, target, covariance)
    assert transition[0, 0] > transition[0, 1]
    assert transition[1, 1] > transition[1, 0]
    np.testing.assert_allclose(np.sum(transition, axis=1), 1.0)


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


def test_constraint_evidence_propagates_to_incoming_transitions() -> None:
    transition = np.asarray([[0.4, 0.6], [0.8, 0.2]])
    updated = update_incoming_transitions(transition, np.asarray([1.0, 0.1]))
    np.testing.assert_allclose(np.sum(updated, axis=1), 1.0)
    assert updated[0, 0] > transition[0, 0]
    assert updated[1, 0] > transition[1, 0]


def test_vapor_tracks_pose_distribution_with_joint_limits() -> None:
    # 三个移动关节构成可解析的测试机器人，姿态恒定。
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
