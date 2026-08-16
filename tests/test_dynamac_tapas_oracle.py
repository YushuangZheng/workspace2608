"""Numerical oracles for the fixed TAPAS/riepybdlib reproduction boundary."""

from __future__ import annotations

from functools import reduce

import numpy as np
from essay2608.policy.dynamac import (
    GaussianMarginal,
    _fit_pose_sequence,
    pose_exp_world,
    pose_log_world,
    pose_parallel_transport,
    product_of_experts,
    quaternion_exp,
    quaternion_log,
    quaternion_parallel_transport,
    tapas_subsample_rows,
)


def _pose(axis: tuple[float, float, float], angle: float, position=(0.0, 0.0, 0.0)):
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    return np.concatenate(
        (
            np.asarray(position, dtype=np.float64),
            np.asarray([np.cos(angle / 2.0)]),
            axis_array * np.sin(angle / 2.0),
        )
    )


def _riepy_inverse(matrix: np.ndarray) -> np.ndarray:
    """Mirror the singular-matrix fallback in riepy ``Gaussian.__mul__``."""

    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.inv(matrix + np.eye(len(matrix)) * 1.0e-8)


def test_diagonal_empirical_covariance_clips_only_under_floor_dimensions() -> None:
    trajectories = np.stack(
        [
            [_pose((1.0, 0.0, 0.0), 0.0, position=(x, 2.0 * x, 0.0))]
            for x in (-2.0, 0.0, 2.0)
        ]
    )

    _, covariance = _fit_pose_sequence(
        trajectories,
        position_variance_floor=0.5,
        rotation_variance_floor=0.25,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )

    np.testing.assert_allclose(
        covariance[0],
        np.diag([4.0, 16.0, 0.5, 0.25, 0.25, 0.25]),
        atol=1.0e-12,
    )


def test_paper_diagonal_empirical_covariance_uses_additive_ridge() -> None:
    trajectories = np.stack(
        [
            [_pose((1.0, 0.0, 0.0), 0.0, position=(x, 2.0 * x, 0.0))]
            for x in (-2.0, 0.0, 2.0)
        ]
    )

    _, covariance = _fit_pose_sequence(
        trajectories,
        position_variance_floor=0.5,
        rotation_variance_floor=0.25,
        covariance_estimation_method="diagonal_empirical_ridge",
    )

    np.testing.assert_allclose(
        covariance[0],
        np.diag([4.5, 16.5, 0.5, 0.25, 0.25, 0.25]),
        atol=1.0e-12,
    )


def test_full_empirical_covariance_uses_additive_diagonal_ridge() -> None:
    trajectories = np.stack(
        [
            [_pose((1.0, 0.0, 0.0), 0.0, position=(x, 2.0 * x, 0.0))]
            for x in (-2.0, 0.0, 2.0)
        ]
    )

    _, covariance = _fit_pose_sequence(
        trajectories,
        position_variance_floor=0.5,
        rotation_variance_floor=0.25,
        covariance_estimation_method="full_empirical_ridge",
    )

    expected = np.diag([4.5, 16.5, 0.5, 0.25, 0.25, 0.25])
    expected[0, 1] = expected[1, 0] = 8.0
    np.testing.assert_allclose(covariance[0], expected, atol=1.0e-12)


def _riepy_binary_product(
    left: GaussianMarginal,
    right: GaussianMarginal,
) -> GaussianMarginal:
    """Independent transcription of the fixed riepy binary Gaussian product."""

    left_precision = _riepy_inverse(left.covariance)
    right_precision = _riepy_inverse(right.covariance)
    mean = left.mean.copy()
    covariance = left.covariance.copy()

    # max_it=50 with ``if it > max_it`` performs at most 51 updates.
    for _ in range(51):
        left_transport = pose_parallel_transport(left.mean, mean)
        right_transport = pose_parallel_transport(right.mean, mean)
        transported_left = left_transport @ left_precision @ left_transport.T
        transported_right = right_transport @ right_precision @ right_transport.T
        covariance = _riepy_inverse(transported_left + transported_right)
        delta = covariance @ (
            transported_left @ pose_log_world(mean, left.mean)
            + transported_right @ pose_log_world(mean, right.mean)
        )
        mean = pose_exp_world(mean, delta)
        if float(delta @ delta) <= 1.0e-5:
            break

    # riepy returns the covariance computed immediately before the last mean update.
    return GaussianMarginal(f"{left.frame}*{right.frame}", mean, covariance)


def test_s3_half_angle_and_parallel_transport_match_riepy_oracle() -> None:
    identity = _pose((0.0, 0.0, 1.0), 0.0)[3:7]
    quarter_turn = _pose((0.0, 0.0, 1.0), np.pi / 2.0)[3:7]

    np.testing.assert_allclose(quaternion_log(quarter_turn), [0.0, 0.0, np.pi / 4.0])
    np.testing.assert_allclose(quaternion_exp(quaternion_log(quarter_turn)), quarter_turn)

    root_half = np.sqrt(0.5)
    expected = np.asarray(
        [
            [root_half, root_half, 0.0],
            [-root_half, root_half, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    actual = quaternion_parallel_transport(identity, quarter_turn)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.det(actual), 1.0, atol=1.0e-12)


def test_three_expert_product_uses_fixed_riepy_left_fold() -> None:
    marginals = [
        GaussianMarginal(
            "a",
            _pose((1.0, 0.0, 0.0), 0.9),
            np.diag([0.2, 0.8, 0.5, 0.01, 0.2, 0.7]),
        ),
        GaussianMarginal(
            "b",
            _pose((0.0, 1.0, 0.0), 1.2, (1.0, 2.0, 0.0)),
            np.diag([0.5, 0.1, 0.9, 0.6, 0.02, 0.3]),
        ),
        GaussianMarginal(
            "c",
            _pose((0.0, 0.0, 1.0), 1.4, (-1.0, 1.0, 2.0)),
            np.diag([0.9, 0.6, 0.1, 0.3, 0.8, 0.015]),
        ),
    ]
    oracle = reduce(_riepy_binary_product, marginals)

    mean, covariance, _ = product_of_experts(marginals)

    np.testing.assert_allclose(mean, oracle.mean, atol=1.0e-12)
    np.testing.assert_allclose(covariance, oracle.covariance, atol=1.0e-12)


def test_product_aligns_equivalent_antipodal_quaternion_representatives() -> None:
    identity = _pose((0.0, 0.0, 1.0), 0.0)
    antipode = identity.copy()
    antipode[3:7] *= -1.0
    mean, covariance, _ = product_of_experts(
        [
            GaussianMarginal("positive", identity, np.eye(6)),
            GaussianMarginal("negative", antipode, np.eye(6)),
        ]
    )
    np.testing.assert_allclose(mean, identity)
    np.testing.assert_allclose(covariance, np.eye(6) * 0.5)

    near = _pose((0.0, 0.0, 1.0), 0.02)
    near[3:7] *= -1.0
    fused, _, _ = product_of_experts(
        [
            GaussianMarginal("positive", identity, np.eye(6)),
            GaussianMarginal("near_negative", near, np.eye(6)),
        ]
    )
    assert np.linalg.norm(quaternion_log(fused[3:7])) < 0.02


def test_crossing_180_degrees_keeps_small_rotation_covariance() -> None:
    yaws = np.deg2rad(
        [
            [179.0, 181.0],
            [179.2, 181.2],
            [178.8, 180.8],
            [179.1, 181.1],
            [178.9, 180.9],
        ]
    )
    trajectories = np.stack(
        [[_pose((0.0, 0.0, 1.0), yaw) for yaw in trajectory] for trajectory in yaws]
    )
    floor = 1.0e-8

    means, covariance = _fit_pose_sequence(
        trajectories,
        position_variance_floor=floor,
        rotation_variance_floor=floor,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )

    assert float(np.dot(means[0, 3:7], means[1, 3:7])) > 0.0
    # Fixed riepy's standard-acos Log has a q0 dead zone of 1e-6. The
    # +/-0.1 degree physical offsets therefore map to zero, while +/-0.2
    # degree offsets remain active and are represented as half angles.
    active_half_angle = np.deg2rad(0.2) / 2.0
    expected_z_variance = 2.0 * active_half_angle**2 / (len(yaws) - 1)
    np.testing.assert_allclose(covariance[:, 5, 5], expected_z_variance, rtol=1.0e-10)
    np.testing.assert_allclose(covariance[:, 3, 3], floor)
    np.testing.assert_allclose(covariance[:, 4, 4], floor)


def test_demo_initial_orientations_straddling_180_degrees_share_one_s3_gauge() -> None:
    """q/-q gauge choices must not split physically adjacent demonstrations.

    The existing crossing test starts every demonstration below 180 degrees.
    This regression instead places the *first* sample on both sides of 180
    degrees, which is the failure case created by independently forcing each
    local trajectory's first quaternion to have a positive real part.
    """

    first_yaws = np.deg2rad([179.0, 181.0, 179.5, 180.5, 180.0])
    trajectories = np.stack(
        [
            [
                _pose((0.0, 0.0, 1.0), float(yaw)),
                _pose((0.0, 0.0, 1.0), float(yaw + np.deg2rad(1.0))),
            ]
            for yaw in first_yaws
        ]
    )
    floor = 1.0e-8

    means, covariance = _fit_pose_sequence(
        trajectories,
        position_variance_floor=floor,
        rotation_variance_floor=floor,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )

    expected = [
        _pose((0.0, 0.0, 1.0), np.deg2rad(180.0)),
        _pose((0.0, 0.0, 1.0), np.deg2rad(181.0)),
    ]
    for actual, target in zip(means, expected, strict=True):
        # Compare physical SO(3) orientation, not an arbitrary q/-q sign.
        assert abs(float(np.dot(actual[3:7], target[3:7]))) > 1.0 - 1.0e-10
    assert float(np.dot(means[0, 3:7], means[1, 3:7])) > 0.0

    # The demonstrations differ by at most one physical degree around their
    # mean, so their S3 half-angle variance must remain tiny.  The historical
    # bug produces an O(1 rad^2) value here.
    assert np.all(covariance[:, 5, 5] < 1.0e-3), covariance[:, 5, 5]
    np.testing.assert_allclose(covariance[:, 3, 3], floor)
    np.testing.assert_allclose(covariance[:, 4, 4], floor)


def test_tapas_demos_upsampling_repeats_along_the_whole_trajectory() -> None:
    values = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])

    upsampled = tapas_subsample_rows(values, 6)

    np.testing.assert_array_equal(
        upsampled,
        np.asarray(
            [
                [0.0, 10.0],
                [0.0, 10.0],
                [1.0, 11.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [2.0, 12.0],
            ]
        ),
    )


def test_tapas_demos_subsampling_preserves_torch_float32_tie_rounding() -> None:
    values_30 = np.arange(30, dtype=np.float64)[:, None]
    values_16 = np.arange(16, dtype=np.float64)[:, None]
    values_4 = np.arange(4, dtype=np.float64)[:, None]

    # Fixed TAPAS uses torch.linspace's default float32. These are the two
    # shortest practical cases where float64 linspace lands on the other side
    # of a half-integer and therefore selects a different demonstration row.
    assert tapas_subsample_rows(values_30, 29)[14, 0] == 14.0
    assert tapas_subsample_rows(values_16, 23)[11, 0] == 8.0
    # The torch CPU kernel computes the second half backward from ``end``;
    # ordinary NumPy float32 arithmetic returns 2 here.
    assert tapas_subsample_rows(values_4, 43)[21, 0] == 1.0
