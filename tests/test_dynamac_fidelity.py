"""Geometry and Algorithm 1 fidelity tests for DynaMAC, TAPAS, and MiDiGaP.

These tests use small deterministic synthetic inputs. They verify invariants from
the paper and pinned TAPAS implementation rather than restating implementation details.
"""

from __future__ import annotations

import numpy as np
import pytest
from essay2608.policy.dynamac import (
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
    GaussianMarginal,
    IDENTITY_POSE,
    _compose_framewise_poe_participation,
    _eq6_skill_selection,
    _filter_link_mask,
    _fit_pose_sequence,
    _gripper_modal_factor,
    _mixture_statistics,
    _partition_modes,
    _partition_product_modes,
    _prepare_pose_batch,
    _trajectory_distances,
    ensure_quaternion_continuity,
    geometric_mean_standard_deviation,
    pose_compose,
    pose_inverse,
    product_of_experts,
    quaternion_exp,
    quaternion_log,
    quaternion_to_matrix,
    static_task_parameter_score_details,
    tapas_subsample_poses,
    tapas_subsample_rows,
    task_parameter_score_details,
    transform_marginal,
)


def pose(position, tangent=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Build a pose using the TAPAS half-angle convention for ``tangent``."""

    return np.concatenate(
        (
            np.asarray(position, dtype=np.float64),
            quaternion_exp(np.asarray(tangent, dtype=np.float64)),
        )
    )


def yaw_pose(position, yaw: float) -> np.ndarray:
    return np.asarray(
        [*position, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float64,
    )


def assert_pose_equivalent(actual: np.ndarray, expected: np.ndarray, atol: float = 1.0e-10) -> None:
    np.testing.assert_allclose(actual[:3], expected[:3], atol=atol, rtol=0.0)
    # q and -q represent the same S3 orientation; compare physical orientations.
    assert abs(float(np.dot(actual[3:7], expected[3:7]))) == pytest.approx(1.0, abs=atol)


def test_tapas_quaternion_half_angle_round_trip_and_sequence_continuity() -> None:
    tangent = np.asarray([0.0, 0.0, np.pi / 4.0])
    quaternion = quaternion_exp(tangent)
    np.testing.assert_allclose(
        quaternion,
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(quaternion_log(quaternion), tangent, atol=1.0e-12)

    alternating = np.stack((quaternion, -quaternion, quaternion, -quaternion))
    continuous = ensure_quaternion_continuity(alternating)
    assert np.all(np.sum(continuous[:-1] * continuous[1:], axis=1) >= 0.0)


def test_fitted_quaternion_mean_stays_continuous_across_pi() -> None:
    """Per-step Frechet means must not flip at 180 degrees from q/-q choices."""

    angles = np.deg2rad([170.0, 175.0, 179.0, 181.0, 185.0, 190.0])
    trajectories = []
    for demo_index in range(5):
        jitter = np.deg2rad((demo_index - 2) * 0.05)
        values = []
        for time_index, angle in enumerate(angles):
            value = yaw_pose([0.02 * time_index, 0.0, 0.0], angle + jitter)
            if (demo_index + time_index) % 2:
                value[3:7] *= -1.0
            values.append(value)
        trajectory = np.stack(values)
        trajectory[:, 3:7] = ensure_quaternion_continuity(trajectory[:, 3:7])
        trajectories.append(trajectory)

    mean, _ = _fit_pose_sequence(
        np.stack(trajectories),
        position_variance_floor=1.0e-10,
        rotation_variance_floor=1.0e-10,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )
    adjacent_dots = np.sum(mean[:-1, 3:7] * mean[1:, 3:7], axis=1)
    assert np.all(adjacent_dots >= 0.0), adjacent_dots
    for actual, angle in zip(mean[:, 3:7], angles, strict=True):
        expected = yaw_pose([0.0, 0.0, 0.0], float(angle))[3:7]
        assert abs(float(np.dot(actual, expected))) == pytest.approx(1.0, abs=1.0e-6)


def test_pose_statistics_and_partition_ignore_whole_trajectory_antipodes() -> None:
    """Changing only q/-q representatives cannot change a learned policy."""

    starts = np.deg2rad([178.0, 179.0, 201.0, 202.0])
    trajectories = np.stack(
        [
            np.stack(
                [
                    yaw_pose([0.01 * time_index, 0.0, 0.0], float(yaw + 0.01 * time_index))
                    for time_index in range(4)
                ]
            )
            for yaw in starts
        ]
    )
    antipodal = trajectories.copy()
    antipodal[[0, 2], :, 3:7] *= -1.0

    original_prepared = _prepare_pose_batch(trajectories)
    antipodal_prepared = _prepare_pose_batch(antipodal)
    np.testing.assert_allclose(
        _trajectory_distances(original_prepared, original_prepared),
        _trajectory_distances(antipodal_prepared, antipodal_prepared),
        atol=1.0e-12,
    )

    original_mean, original_covariance = _fit_pose_sequence(
        trajectories,
        position_variance_floor=1.0e-8,
        rotation_variance_floor=1.0e-8,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )
    antipodal_mean, antipodal_covariance = _fit_pose_sequence(
        antipodal,
        position_variance_floor=1.0e-8,
        rotation_variance_floor=1.0e-8,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )
    np.testing.assert_allclose(original_mean[:, :3], antipodal_mean[:, :3], atol=1.0e-12)
    assert np.all(
        np.abs(np.sum(original_mean[:, 3:7] * antipodal_mean[:, 3:7], axis=1))
        > 1.0 - 1.0e-12
    )
    np.testing.assert_allclose(original_covariance, antipodal_covariance, atol=1.0e-12)

    config = DynaMACConfig(
        modal_partition_method="riemannian_kmeans_bic",
        maximum_modes=2,
        minimum_mode_size=2,
        clustering_length=4,
        clustering_restarts=16,
    )
    np.testing.assert_array_equal(
        _partition_modes(trajectories, config),
        _partition_modes(antipodal, config),
    )


def test_batch_gauge_stays_continuous_when_demo_lifts_have_different_winding() -> None:
    """SO(3) paths that reconverge must not leave mixed q/-q endpoints.

    Three demonstrations take the long continuous S3 lift from a yaw above 180
    degrees while two start near zero.  All five finish at the identity.  A gauge
    frozen at the first sample flips an entire subgroup and makes the fitted mean
    jump by roughly 2.6 radians on the final sample even though every physical
    demonstration is smooth.
    """

    samples = 24
    trajectories = np.stack(
        [
            np.stack(
                [
                    yaw_pose(
                        [0.0, 0.0, 0.0],
                        float(yaw),
                    )
                    for yaw in np.deg2rad(np.linspace(start_yaw, 0.0, samples))
                ]
            )
            for start_yaw in (0.0, 200.0, 210.0, 190.0, 10.0)
        ]
    )
    floor = 1.0e-8

    mean, covariance = _fit_pose_sequence(
        trajectories,
        position_variance_floor=floor,
        rotation_variance_floor=floor,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )
    physical_increments = np.asarray(
        [
            2.0
            * np.arccos(
                np.clip(
                    abs(float(np.dot(left[3:7], right[3:7]))),
                    -1.0,
                    1.0,
                )
            )
            for left, right in zip(mean[:-1], mean[1:], strict=True)
        ]
    )
    # The deliberately broad initial distribution is close to the SO(3) cut
    # locus and need not have a unique global mean.  Once every demonstration
    # has reconverged, however, the learned mean must remain on the same smooth
    # branch instead of jumping at the final sample.
    assert float(np.max(physical_increments[-5:])) < np.deg2rad(6.0)
    assert abs(float(np.dot(mean[-1, 3:7], IDENTITY_POSE[3:7]))) > 1.0 - 1.0e-12

    # Arbitrary q/-q representatives at individual samples remain physically
    # equivalent and must produce the same statistics.
    alternate_gauge = trajectories.copy()
    alternate_gauge[:, 1::2, 3:7] *= -1.0
    alternate_mean, alternate_covariance = _fit_pose_sequence(
        alternate_gauge,
        position_variance_floor=floor,
        rotation_variance_floor=floor,
        covariance_estimation_method="diagonal_empirical_spd_floor",
    )
    assert np.all(
        np.abs(np.sum(mean[:, 3:7] * alternate_mean[:, 3:7], axis=1))
        > 1.0 - 1.0e-12
    )
    np.testing.assert_allclose(covariance, alternate_covariance, atol=1.0e-12)


def test_tapas_subsampling_uses_rounded_indices_not_interpolation() -> None:
    rows = np.arange(6, dtype=np.float64)[:, None]
    np.testing.assert_array_equal(tapas_subsample_rows(rows, 4)[:, 0], [0.0, 2.0, 3.0, 5.0])

    orientation = yaw_pose([0.0, 0.0, 0.0], 0.8)[3:7]
    poses = np.stack(
        [
            np.concatenate(([float(index), 0.0, 0.0], orientation * (-1.0 if index % 2 else 1.0)))
            for index in range(6)
        ]
    )
    sampled = tapas_subsample_poses(poses, 4)
    np.testing.assert_array_equal(sampled[:, 0], [0.0, 2.0, 3.0, 5.0])
    assert np.all(np.sum(sampled[:-1, 3:7] * sampled[1:, 3:7], axis=1) >= 0.0)


def test_frame_transform_is_equivariant_under_global_left_action() -> None:
    frame = pose([0.4, -0.2, 0.1], [0.2, -0.1, 0.3])
    local_mean = pose([0.1, 0.2, -0.1], [-0.25, 0.15, 0.2])
    global_transform = pose([-0.3, 0.5, 0.2], [0.4, 0.1, -0.2])
    raw = np.arange(36, dtype=np.float64).reshape(6, 6) / 20.0
    local_covariance = raw @ raw.T + np.diag(np.linspace(0.2, 0.7, 6))

    original = transform_marginal("frame", frame, local_mean, local_covariance)
    transformed = transform_marginal(
        "frame", pose_compose(global_transform, frame), local_mean, local_covariance
    )
    assert_pose_equivalent(transformed.mean, pose_compose(global_transform, original.mean))

    # TAPAS uses quaternion body-tangent coordinates: a global left action rotates
    # the translational tangent while leaving the rotational body tangent unchanged.
    tangent_action = np.eye(6)
    tangent_action[:3, :3] = quaternion_to_matrix(global_transform[3:7])
    np.testing.assert_allclose(
        transformed.covariance,
        tangent_action @ original.covariance @ tangent_action.T,
        atol=2.0e-11,
        rtol=1.0e-11,
    )


def test_product_of_experts_is_equivariant_under_global_left_action() -> None:
    global_transform = pose([-0.3, 0.5, 0.2], [0.4, 0.1, -0.2])
    tangent_action = np.eye(6)
    tangent_action[:3, :3] = quaternion_to_matrix(global_transform[3:7])
    marginals = [
        GaussianMarginal(
            "a",
            pose([0.1, 0.2, 0.0], [0.1, 0.0, 0.2]),
            np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        GaussianMarginal(
            "b",
            pose([0.3, -0.1, 0.2], [-0.2, 0.2, 0.1]),
            np.diag([2.0, 1.0, 4.0, 6.0, 3.0, 5.0]),
        ),
    ]
    mean, covariance, weights = product_of_experts(marginals)
    moved_marginals = [
        GaussianMarginal(
            marginal.frame,
            pose_compose(global_transform, marginal.mean),
            tangent_action @ marginal.covariance @ tangent_action.T,
        )
        for marginal in marginals
    ]
    moved_mean, moved_covariance, moved_weights = product_of_experts(moved_marginals)

    assert_pose_equivalent(moved_mean, pose_compose(global_transform, mean))
    np.testing.assert_allclose(
        moved_covariance,
        tangent_action @ covariance @ tangent_action.T,
        atol=2.0e-10,
        rtol=1.0e-10,
    )
    assert moved_weights == pytest.approx(weights)


def two_arc_modes() -> np.ndarray:
    progress = np.linspace(0.0, 1.0, 8)
    trajectories = []
    for sign in (-1.0, 1.0):
        for demo_index in range(6):
            amplitude = 0.30 + (demo_index - 2.5) * 0.002
            trajectories.append(
                np.stack(
                    [
                        pose(
                            [
                                value,
                                sign * amplitude * np.sin(np.pi * value),
                                0.1,
                            ]
                        )
                        for value in progress
                    ]
                )
            )
    return np.stack(trajectories)


@pytest.mark.parametrize(
    "method",
    ["riemannian_kmeans_bic", "riemannian_gmm_bic", "dbscan"],
)
def test_all_midigap_partition_methods_recover_whole_trajectory_modes(method: str) -> None:
    trajectories = two_arc_modes()
    config = DynaMACConfig(
        modal_partition_method=method,
        maximum_modes=3,
        minimum_mode_size=2,
        clustering_length=8,
        dbscan_epsilon=0.08,
        dbscan_min_samples=2,
        resampling_method="interpolate",
    )
    labels = _partition_modes(trajectories, config)
    expected = np.repeat([0, 1], 6)
    # Compare co-assignment to test the partition without assigning meaning to labels.
    np.testing.assert_array_equal(labels[:, None] == labels[None, :], expected[:, None] == expected)
    np.testing.assert_array_equal(np.bincount(labels), [6, 6])
    np.testing.assert_array_equal(_partition_modes(trajectories, config), labels)

    midpoint_y = []
    for label in np.unique(labels):
        mean, _ = _fit_pose_sequence(
            trajectories[labels == label],
            position_variance_floor=1.0e-10,
            rotation_variance_floor=1.0e-10,
            covariance_estimation_method="diagonal_empirical_spd_floor",
        )
        midpoint_y.append(float(mean[len(mean) // 2, 1]))
    assert sorted(midpoint_y)[0] < -0.29
    assert sorted(midpoint_y)[1] > 0.29


@pytest.mark.parametrize(
    "method",
    ["riemannian_kmeans_bic", "riemannian_gmm_bic", "dbscan"],
)
def test_global_gripper_factor_separates_pose_identical_modes(method: str) -> None:
    """TAPAS' single global R_grip factor must participate in MiDiGaP Eq. (8)."""

    samples, duration = 5, 4
    identical_pose = np.repeat(pose([0.0, 0.0, 0.0])[None, None], duration, axis=1)
    identical_pose = np.repeat(identical_pose, samples, axis=0)
    normalized_gripper = np.concatenate(
        (
            -np.ones((2, duration, 1), dtype=np.float64),
            np.ones((3, duration, 1), dtype=np.float64),
        )
    )
    config = DynaMACConfig(
        modal_partition_method=method,
        maximum_modes=2,
        minimum_mode_size=2,
        clustering_length=duration,
        clustering_variance_floor=1.0e-4,
        dbscan_epsilon=0.5,
        dbscan_min_samples=2,
        resampling_method="interpolate",
    )
    labels = _partition_product_modes(
        {"object": identical_pose},
        config,
        {"gripper": normalized_gripper},
    )
    expected = np.asarray([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(
        labels[:, None] == labels[None, :],
        expected[:, None] == expected[None, :],
    )

    # The gripper factor is global, not copied into each task frame.  Adding a
    # second identical pose frame therefore does not change its distance term.
    one_pose_factor = identical_pose[[0, 2]]
    two_pose_factors = np.concatenate((one_pose_factor, one_pose_factor), axis=1)
    gripper_flat = normalized_gripper[[0, 2]].reshape(2, -1)
    expected_gripper_distance = duration * 4.0
    one_distance = _trajectory_distances(
        one_pose_factor,
        one_pose_factor,
        gripper_flat,
        gripper_flat,
    )[0, 1]
    two_distance = _trajectory_distances(
        two_pose_factors,
        two_pose_factors,
        gripper_flat,
        gripper_flat,
    )[0, 1]
    np.testing.assert_allclose(one_distance, expected_gripper_distance)
    np.testing.assert_allclose(two_distance, expected_gripper_distance)


@pytest.mark.parametrize(
    ("native_gripper", "scale"),
    [
        (np.asarray([-1.0, -1.0, 1.0, 1.0, 1.0]), 1.0),
        (np.asarray([0.0, 0.0, 1.0, 1.0, 1.0]), 2.0),
    ],
)
def test_public_fit_uses_explicit_gripper_metric_scale(
    native_gripper: np.ndarray,
    scale: float,
) -> None:
    """Raw R values stay generic; a dataset protocol owns their metric scale."""

    duration = 4
    identical_pose = np.repeat(pose([0.0, 0.0, 0.0])[None], duration, axis=0)
    demonstrations = [
        DynaMACDemonstration(
            ee_pose=identical_pose,
            action_pose=identical_pose,
            gripper=np.full(duration, value),
            frames={"object": identical_pose},
            skill=np.zeros(duration),
            name=f"gripper_{index}",
        )
        for index, value in enumerate(native_gripper)
    ]
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=1.0e-12,
            tau_omega=0.2,
            modal_partition_method="dbscan",
            maximum_modes=2,
            minimum_mode_size=2,
            clustering_length=duration,
            clustering_variance_floor=1.0e-4,
            gripper_clustering_scale=scale,
            resampling_method="interpolate",
        )
    ).fit(demonstrations)
    assert policy.skills[0].mode_demonstration_indices == ((0, 1), (2, 3, 4))

    if np.min(native_gripper) == 0.0:
        robo = _gripper_modal_factor(native_gripper, scale)
        tapas = 2.0 * native_gripper - 1.0
        np.testing.assert_allclose(robo[:, None] - robo[None, :], tapas[:, None] - tapas[None, :])


@pytest.mark.parametrize(
    "method",
    ["riemannian_kmeans_bic", "riemannian_gmm_bic"],
)
def test_deterministic_multistart_recovers_global_three_mode_partition(method: str) -> None:
    """Later center combinations must recover three clusters after duplicates fail."""

    trajectories = np.stack(
        [
            np.repeat(pose([position, 0.0, 0.0])[None], 2, axis=0)
            for position in (0.0, 1.0, 1.0, 2.0, 2.0)
        ]
    )
    common = dict(
        modal_partition_method=method,
        maximum_modes=3,
        minimum_mode_size=1,
        clustering_length=2,
        clustering_variance_floor=1.0e-4,
        resampling_method="interpolate",
    )

    first_initialization_only = _partition_modes(
        trajectories,
        DynaMACConfig(**common, clustering_restarts=1),
    )
    np.testing.assert_array_equal(first_initialization_only, np.zeros(5, dtype=np.int64))

    config = DynaMACConfig(**common, clustering_restarts=10)
    expected = np.asarray([0, 1, 1, 2, 2], dtype=np.int64)
    labels = _partition_modes(trajectories, config)
    np.testing.assert_array_equal(
        labels[:, None] == labels[None, :],
        expected[:, None] == expected[None, :],
    )
    np.testing.assert_array_equal(_partition_modes(trajectories, config), labels)


def test_clustering_variance_floor_is_separate_from_final_policy_covariance() -> None:
    clustering_floor = 0.125
    position_floor = 2.0e-7
    rotation_floor = 3.0e-7
    config = DynaMACConfig(
        tau_omega=0.0,
        maximum_modes=1,
        minimum_mode_size=1,
        clustering_length=2,
        clustering_variance_floor=clustering_floor,
        position_variance_floor=position_floor,
        rotation_variance_floor=rotation_floor,
        resampling_method="interpolate",
    )
    trajectory = np.stack((pose([0.0, 0.0, 0.0]), pose([0.2, 0.0, 0.0])))

    _, clustering_variance, _ = _mixture_statistics(
        trajectory[None],
        np.ones((1, 1), dtype=np.float64),
        config,
    )
    np.testing.assert_allclose(clustering_variance, clustering_floor)

    world = np.repeat(pose([0.0, 0.0, 0.0])[None], len(trajectory), axis=0)
    demonstration = DynaMACDemonstration(
        ee_pose=trajectory,
        action_pose=trajectory.copy(),
        gripper=np.zeros(len(trajectory)),
        frames={"world": world},
        skill=np.zeros(len(trajectory), dtype=np.int64),
        name="singleton",
    )
    policy = DynaMAC(config).fit([demonstration])
    # Eq. (5) removes the fully linked static world frame before Eq. (6);
    # the virtual frame remains available and preserves the covariance test.
    assert policy.skills[0].selected_frames == ("virtual_skill_0",)
    np.testing.assert_array_equal(
        policy.training_audit["skills"][0]["task_parameter_selection"][
            "eq5_availability"
        ]["world"],
        np.zeros(len(trajectory), dtype=bool),
    )
    final_covariance = policy.skills[0].streams["virtual_skill_0"].covariance
    expected_diagonal = np.asarray(
        [position_floor] * 3 + [rotation_floor] * 3,
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.diagonal(final_covariance, axis1=-2, axis2=-1),
        np.broadcast_to(expected_diagonal, final_covariance.shape[:2] + (6,)),
    )
    assert not np.any(np.isclose(final_covariance, clustering_floor))


def test_experimental_eq5_link_mask_can_remain_per_timestep() -> None:
    standard_deviation = np.asarray(
        [2.0e-3, 2.0e-4, 2.0e-3, 2.0e-3, 2.0e-4, 2.0e-4, 2.0e-4, 2.0e-4, 2.0e-3, 2.0e-3]
    )
    covariance = np.stack([np.eye(6) * value**2 for value in standard_deviation])
    np.testing.assert_allclose(
        geometric_mean_standard_deviation(covariance), standard_deviation, atol=1.0e-15
    )

    raw, filtered = _filter_link_mask(
        standard_deviation,
        DynaMACConfig(
            tau_m=1.0e-3,
            link_filter="none",
            link_mask_scope="timestep",
        ),
    )
    np.testing.assert_array_equal(
        raw,
        [False, True, False, False, True, True, True, True, False, False],
    )
    np.testing.assert_array_equal(
        filtered,
        [False, True, False, False, True, True, True, True, False, False],
    )


@pytest.mark.parametrize(
    ("linked_count", "gate_enabled"),
    ((6, True), (5, False), (4, False)),
)
def test_v3_eq5_strict_majority_only_gates_the_raw_timestep_mask(
    linked_count,
    gate_enabled,
) -> None:
    scale = np.full(10, 2.0e-3)
    scale[:linked_count] = 2.0e-4
    raw, linked = _filter_link_mask(
        scale,
        DynaMACConfig(
            tau_m=1.0e-3,
            link_filter="none",
            link_mask_scope="skill_majority_gate_timestep",
        ),
    )

    expected_raw = np.arange(10) < linked_count
    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_array_equal(
        linked,
        expected_raw if gate_enabled else np.zeros(10, dtype=bool),
    )


def test_v3_eq5_gate_has_a_distinct_artifact_identity() -> None:
    config = DynaMACConfig(link_mask_scope="skill_majority_gate_timestep")

    assert DynaMAC(config).selection_semantics_id == (
        "eq5_skill_majority_gate_timestep_availability_before_eq6_and_poe_"
        "time_state_position3d_unimodal_v1"
    )
    with pytest.raises(ValueError, match="raw Eq. \\(5\\) mask"):
        DynaMACConfig(
            link_mask_scope="skill_majority_gate_timestep",
            link_filter="temporal_variance",
        )


def test_eq5_filters_eq6_candidates_and_denominator_before_max_t() -> None:
    """Eq. (6) normalizes shares only over candidates available under Eq. (5)."""

    covariances = {
        "a": np.stack((np.eye(6), np.eye(6) * 2.0)),
        "b": np.stack((np.eye(6) * 2.0, np.eye(6))),
    }
    availability = {
        "a": np.asarray([True, True]),
        "b": np.asarray([True, False]),
    }
    selected, selected_by_eq6, details = _eq6_skill_selection(
        covariances,
        tau_omega=0.02,
        availability=availability,
        candidate_kind={"a": "dynamic", "b": "dynamic"},
    )

    # t=0: det(I)^-1 : det(2I)^-1 = 64:1.  At t=1 only a is
    # available, so a receives the entire denominator and b is exactly zero.
    expected = np.asarray([[64.0 / 65.0, 1.0], [1.0 / 65.0, 0.0]])
    np.testing.assert_allclose(details["relative_precision"], expected, atol=1.0e-15)
    assert details["scores"] == pytest.approx({"a": 1.0, "b": 1.0 / 65.0})
    assert details["argmax_time"] == {"a": 1, "b": 0}
    assert selected == ("a",)
    assert selected_by_eq6 == {"a": True, "b": False}
    assert details["empty_selection_policy"] == "error"
    assert (
        details["empty_selection_policy_source_status"]
        == "PAPER_EQ6_STRICT_THRESHOLD_FAIL_CLOSED"
    )
    assert details["normalization_scope"] == (
        "eq5_available_candidate_frames_per_timestep"
    )
    assert details["semantics_id"] == (
        "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
    )
    assert details["eq5_filters_eq6_denominator"] is True
    np.testing.assert_array_equal(details["available_candidate_count"], [2, 1])
    np.testing.assert_allclose(details["normalization_residual"], 0.0, atol=1.0e-15)
    np.testing.assert_array_equal(details["normalization_valid_mask"], [True, True])
    expected_log_denominator = np.log(np.asarray([65.0 / 64.0, 1.0 / 64.0]))
    np.testing.assert_allclose(
        details["log_precision_denominator"],
        expected_log_denominator,
        atol=1.0e-15,
    )


def test_eq6_empty_selection_can_keep_all_numerically_tied_argmax_frames() -> None:
    covariances = {
        "first": np.repeat(np.eye(6)[None], 2, axis=0),
        "second": np.repeat(np.eye(6)[None], 2, axis=0),
    }
    selected, selected_by_eq6, details = _eq6_skill_selection(
        covariances,
        tau_omega=0.5,
        availability={name: np.ones(2, dtype=bool) for name in covariances},
        candidate_kind={name: "virtual" for name in covariances},
        empty_selection="keep_argmax",
    )

    assert details["scores"] == {"first": 0.5, "second": 0.5}
    assert details["tau_omega"] == 0.5
    assert details["selected_by_threshold"] == {"first": False, "second": False}
    assert selected == ("first", "second")
    assert selected_by_eq6 == {"first": True, "second": True}
    assert details["empty_selection_policy"] == "keep_argmax"
    assert details["empty_selection_policy_source_status"] == "LOCAL_INFERENCE"
    assert details["empty_selection_fallback_applied"] is True
    assert details["fallback_selected_frames"] == ("first", "second")
    assert details["fallback_reason"] == (
        "no_frame_score_strictly_above_tau_omega_keep_argmax"
    )


def test_linked_precision_spike_cannot_contaminate_eq6_share_or_max_t() -> None:
    baseline = {
        "anchor": np.stack((np.eye(6), np.eye(6))),
        "linked": np.stack((np.eye(6) * 2.0, np.eye(6) * 2.0)),
    }
    spiked = {name: covariance.copy() for name, covariance in baseline.items()}
    spiked["linked"][1] = np.eye(6) * 1.0e-30
    availability = {
        "anchor": np.asarray([True, True]),
        "linked": np.asarray([True, False]),
    }

    baseline_details = task_parameter_score_details(
        baseline,
        availability=availability,
        candidate_kind={"anchor": "dynamic", "linked": "dynamic"},
    )
    spiked_details = task_parameter_score_details(
        spiked,
        availability=availability,
        candidate_kind={"anchor": "dynamic", "linked": "dynamic"},
    )
    np.testing.assert_array_equal(
        spiked_details["relative_precision"],
        baseline_details["relative_precision"],
    )
    assert spiked_details["relative_precision"][1, 1] == 0.0
    assert spiked_details["argmax_time"]["linked"] == 0
    assert spiked_details["scores"] == baseline_details["scores"]


def test_eq6_fails_closed_when_any_timestep_has_no_available_candidate() -> None:
    covariances = {
        "a": np.stack((np.eye(6), np.eye(6))),
        "b": np.stack((np.eye(6), np.eye(6))),
    }
    availability = {
        "a": np.asarray([True, False]),
        "b": np.asarray([False, False]),
    }
    with pytest.raises(RuntimeError, match=r"time steps \[1\].*no candidates available under Eq\. \(5\)"):
        task_parameter_score_details(
            covariances,
            availability=availability,
            candidate_kind={"a": "dynamic", "b": "dynamic"},
        )


def test_eq6_argmax_never_uses_unavailable_zero_even_when_available_share_underflows() -> None:
    covariances = {
        "anchor": np.stack((np.eye(6), np.eye(6))),
        "tiny_share": np.stack((np.eye(6), np.eye(6) * 1.0e300)),
    }
    details = task_parameter_score_details(
        covariances,
        availability={
            "anchor": np.asarray([True, True]),
            "tiny_share": np.asarray([False, True]),
        },
        candidate_kind={"anchor": "dynamic", "tiny_share": "dynamic"},
    )
    np.testing.assert_array_equal(details["relative_precision"][1], [0.0, 0.0])
    assert details["argmax_time"]["tiny_share"] == 1


def test_virtual_frame_is_always_available_and_covers_real_link_gap() -> None:
    covariances = {
        "object": np.stack((np.eye(6), np.eye(6))),
        "virtual_skill_7": np.stack((np.eye(6) * 2.0, np.eye(6) * 2.0)),
    }
    availability = {
        "object": np.asarray([False, False]),
        "virtual_skill_7": np.ones(2, dtype=bool),
    }
    candidate_kind = {"object": "dynamic", "virtual_skill_7": "virtual"}
    details = task_parameter_score_details(
        covariances,
        availability=availability,
        candidate_kind=candidate_kind,
    )
    assert details["relative_precision"][0, 0] == 0.0
    assert details["relative_precision"][1, 0] == 1.0
    assert details["scores"]["object"] == 0.0
    assert details["argmax_time"]["object"] is None
    assert details["rejection_reason"]["object"] == "eq5_never_available"
    assert details["candidate_kind"] == candidate_kind
    np.testing.assert_array_equal(
        details["availability"]["virtual_skill_7"],
        np.ones(2, dtype=bool),
    )
    availability["virtual_skill_7"][0] = False
    with pytest.raises(ValueError, match="virtual task-parameter frames must be available at every time step"):
        task_parameter_score_details(
            covariances,
            availability=availability,
            candidate_kind=candidate_kind,
        )


def test_final_poe_participation_is_availability_and_skill_selection() -> None:
    participation = _compose_framewise_poe_participation(
        {
            "selected": np.asarray([[True, False, True]]),
            "rejected": np.asarray([[True, True, False]]),
        },
        {
            "selected": np.asarray([True]),
            "rejected": np.asarray([False]),
        },
    )
    np.testing.assert_array_equal(participation["selected"], [[True, False, True]])
    np.testing.assert_array_equal(participation["rejected"], [[False, False, False]])


def test_static_eq6_matches_hand_calculated_determinant_precision_shares() -> None:
    covariances = {
        "a": np.stack((np.eye(6), np.eye(6) * 2.0)),
        "b": np.stack((np.eye(6) * 2.0, np.eye(6))),
    }
    details = static_task_parameter_score_details(covariances)

    # det(I)^-1 = 1 and det(2I)^-1 = 2^-6, hence shares 64/65 and 1/65.
    expected = np.asarray([[64.0 / 65.0, 1.0 / 65.0], [1.0 / 65.0, 64.0 / 65.0]])
    np.testing.assert_allclose(details["relative_precision"], expected, atol=1.0e-15)
    assert details["scores"] == pytest.approx({"a": 64.0 / 65.0, "b": 64.0 / 65.0})
    assert details["argmax_time"] == {"a": 0, "b": 1}
    assert details["availability_source"] == "implicit_all_candidates_static_default"
    for mask in details["availability"].values():
        np.testing.assert_array_equal(mask, np.ones(2, dtype=bool))
    np.testing.assert_allclose(
        np.sum(details["relative_precision"], axis=0),
        np.ones(2),
        atol=1.0e-15,
    )


def test_dynamic_eq6_api_requires_explicit_availability_and_candidate_kind() -> None:
    covariances = {"object": np.repeat(np.eye(6)[None], 2, axis=0)}
    with pytest.raises(TypeError, match="availability"):
        task_parameter_score_details(covariances)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="candidate_kind"):
        task_parameter_score_details(
            covariances,
            availability={"object": np.ones(2, dtype=bool)},
        )  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="availability must be provided explicitly"):
        task_parameter_score_details(
            covariances,
            availability=None,  # type: ignore[arg-type]
            candidate_kind={"object": "dynamic"},
        )


def test_framewise_poe_composition_rejects_skill_collapsed_masks() -> None:
    with pytest.raises(ValueError, match=r"\[mode,T\]"):
        _compose_framewise_poe_participation(
            {"object": np.asarray([True, False, True])},
            {"object": np.asarray([True])},
        )


def test_fit_fails_closed_when_no_eq6_candidate_exceeds_threshold() -> None:
    """Two identical virtual candidates score 0.5 and fail the strict threshold."""

    demonstrations = []
    for demo_index in range(3):
        anchor = 0.2 * demo_index
        ee = np.stack(
            (
                pose([anchor, 0.0, 0.0]),
                pose([anchor + 0.05, 0.0, 0.0]),
                pose([anchor, 0.0, 0.0]),
                pose([anchor + 0.1, 0.0, 0.0]),
            )
        )
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=ee,
                action_pose=ee.copy(),
                gripper=np.zeros(4),
                # ``attached`` stays linked to the end effector and is removed by
                # Eq. (5). The skill-1 virtual frames share an origin, so both score 0.5.
                frames={"attached": ee.copy()},
                skill=np.asarray([0, 0, 1, 1], dtype=np.int64),
                name=f"equal_virtual_{demo_index}",
            )
        )

    with pytest.raises(
        RuntimeError,
        match=r"skill 1 has no task parameter above tau_omega=0\.5",
    ):
        DynaMAC(
            DynaMACConfig(
                tau_omega=0.5,
                maximum_modes=1,
                minimum_mode_size=1,
                clustering_length=2,
                resampling_method="interpolate",
                default_mode_strategy="map",
            )
        ).fit(demonstrations)

    policy = DynaMAC(
        DynaMACConfig(
            tau_omega=0.5,
            eq6_empty_selection="keep_argmax",
            maximum_modes=1,
            minimum_mode_size=1,
            clustering_length=2,
            resampling_method="interpolate",
            default_mode_strategy="map",
        )
    ).fit(demonstrations)
    selection = policy.training_audit["skills"][1]["task_parameter_selection"]
    assert selection["scores"]["virtual_skill_0"] == pytest.approx(0.5)
    assert selection["scores"]["virtual_skill_1"] == pytest.approx(0.5)
    assert selection["selected_by_threshold"]["virtual_skill_0"] is False
    assert selection["selected_by_threshold"]["virtual_skill_1"] is False
    assert selection["empty_selection_fallback_applied"] is True
    assert selection["empty_selection_policy_source_status"] == "LOCAL_INFERENCE"
    assert selection["fallback_selected_frames"] == (
        "virtual_skill_0",
        "virtual_skill_1",
    )
    assert policy.skills[1].selected_frames == (
        "virtual_skill_0",
        "virtual_skill_1",
    )


def component_temporal_filter_demonstrations() -> list[DynaMACDemonstration]:
    """Both modes pass Eq. (5), but only mode 0 has a stable mean trajectory."""

    demonstrations = []
    duration = 5
    for demo_index in range(4):
        mode = demo_index // 2
        ee_poses = []
        object_poses = []
        for time_index in range(duration):
            if mode == 0:
                ee_pose = pose([0.1 * time_index, 0.001 * demo_index, 0.2])
                local = pose([0.05, 0.0, 0.0])
            else:
                ee_pose = pose([2.0, 0.1 * time_index + 0.001 * demo_index, 0.2])
                local = pose([0.05 + 0.2 * time_index, 0.0, 0.0])
            ee_poses.append(ee_pose)
            object_poses.append(pose_compose(ee_pose, pose_inverse(local)))
        ee = np.stack(ee_poses)
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=ee,
                action_pose=ee.copy(),
                gripper=np.zeros(duration),
                frames={"object": np.stack(object_poses)},
                skill=np.zeros(duration, dtype=np.int64),
                name=f"component_{demo_index}",
            )
        )
    return demonstrations


def test_temporal_filter_and_active_mask_remain_component_specific() -> None:
    demonstrations = component_temporal_filter_demonstrations()
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=1.0e-3,
            tau_omega=0.0,
            link_filter="temporal_variance",
            link_mask_scope="timestep",
            temporal_variance_window=3,
            temporal_variance_threshold=1.0e-5,
            position_variance_floor=1.0e-10,
            rotation_variance_floor=1.0e-10,
            preliminary_analysis="precluster_all_real_frame_product_mode_conditioned",
            modal_partition_method="riemannian_kmeans_bic",
            maximum_modes=2,
            minimum_mode_size=2,
            clustering_length=5,
            clustering_variance_floor=1.0e-4,
            clustering_restarts=16,
            resampling_method="interpolate",
            default_mode_strategy="map",
        )
    ).fit(demonstrations)
    skill = policy.skills[0]
    assert skill.mode_demonstration_indices == ((0, 1), (2, 3))

    diagnostics = skill.link_diagnostics["object"]
    selection = policy.training_audit["skills"][0]["task_parameter_selection"]
    assert selection["eq5_filters_eq6_denominator"] is True
    assert selection["normalization_scope"] == (
        "eq5_available_candidate_frames_per_timestep"
    )
    for details in selection["per_mode_details"]:
        assert details["eq5_filters_eq6_denominator"] is True
        np.testing.assert_allclose(
            np.sum(details["relative_precision"], axis=0),
            np.ones(skill.duration),
        )
    expected_raw = np.ones((2, 5), dtype=bool)
    expected_filtered = np.asarray(
        [
            [True, True, True, True, True],
            [False, False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(diagnostics["raw_link_mask"], expected_raw)
    np.testing.assert_array_equal(diagnostics["filtered_link_mask"], expected_filtered)
    np.testing.assert_array_equal(skill.streams["object"].active, ~expected_filtered)
    assert skill.streams["object"].active.shape == (2, 5)

    for mode, demo_index, object_active in ((0, 0, False), (1, 2, True)):
        demonstration = demonstrations[demo_index]
        initial = DynaMACObservation(
            demonstration.ee_pose[0],
            {"object": demonstration.frames["object"][0]},
        )
        evidence = np.zeros(2, dtype=np.float64)
        evidence[mode] = 1.0
        policy.reset(initial, mode_strategy="map", mode_evidence=[evidence])
        first_action = policy.act(initial)
        second = DynaMACObservation(
            demonstration.ee_pose[1],
            {"object": demonstration.frames["object"][1]},
        )
        second_action = policy.act(second)
        assert first_action.diagnostics["mode"] == mode
        assert ("object" in first_action.diagnostics["active_frames"]) is object_active
        assert ("object" in second_action.diagnostics["active_frames"]) is object_active


def link_transition_demonstrations(
    raw_link: np.ndarray | None = None,
) -> tuple[list[DynaMACDemonstration], np.ndarray]:
    """Build demonstrations whose Eq. (5) result follows an explicit time mask."""

    if raw_link is None:
        raw_link = np.asarray(
            [False, True, False, False, True, True, True, True, False, False]
        )
    else:
        raw_link = np.asarray(raw_link, dtype=bool)
    if raw_link.ndim != 1 or len(raw_link) < 2:
        raise ValueError("raw link fixture must be a one-dimensional sequence")
    demonstrations = []
    demo_count = 7
    for demo_index in range(demo_count):
        centered = demo_index - (demo_count - 1) / 2.0
        ee_poses = []
        object_poses = []
        for time_index, linked in enumerate(raw_link):
            ee_pose = pose(
                [
                    0.4 + 0.015 * demo_index + 0.01 * time_index,
                    -0.1 + 0.006 * demo_index,
                    0.2 + 0.004 * time_index,
                ],
                [0.01 * time_index, 0.005 * demo_index, -0.007 * time_index],
            )
            if linked:
                offset = pose([0.03, -0.02, 0.04], [0.03, -0.02, 0.01])
            else:
                offset = pose(
                    [
                        0.03 + 0.007 * centered,
                        -0.02 - 0.008 * centered,
                        0.04 + 0.009 * centered,
                    ],
                    [
                        0.03 + 0.04 * centered,
                        -0.02 - 0.035 * centered,
                        0.01 + 0.03 * centered,
                    ],
                )
            ee_poses.append(ee_pose)
            object_poses.append(pose_compose(ee_pose, pose_inverse(offset)))
        ee = np.stack(ee_poses)
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=ee,
                action_pose=ee.copy(),
                gripper=np.zeros(len(raw_link)),
                frames={"object": np.stack(object_poses)},
                skill=np.zeros(len(raw_link), dtype=np.int64),
                name=f"link_{demo_index}",
            )
        )
    return demonstrations, raw_link


def test_fit_fails_closed_when_selected_frames_leave_pointwise_poe_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject an Eq. (6) failure to exercise per-step training coverage checks."""

    demonstrations, raw_link = link_transition_demonstrations()

    def select_only_object(
        covariances,
        tau_omega,
        *,
        availability,
        candidate_kind,
        empty_selection,
        semantics_id,
    ):
        _, _, details = _eq6_skill_selection(
            covariances,
            tau_omega,
            availability=availability,
            candidate_kind=candidate_kind,
            empty_selection=empty_selection,
            semantics_id=semantics_id,
        )
        assert np.any(availability["object"])
        assert np.any(~availability["object"])
        forced_selection = {name: name == "object" for name in covariances}
        details["selected_by_eq6"] = forced_selection.copy()
        return ("object",), forced_selection, details

    monkeypatch.setattr(
        "essay2608.policy.dynamac._eq6_skill_selection",
        select_only_object,
    )
    with pytest.raises(
        RuntimeError,
        match=r"selected task parameters are all linked at \[mode, time\]",
    ):
        DynaMAC(
            DynaMACConfig(
                tau_m=1.0e-3,
                tau_omega=0.0,
                link_filter="none",
                link_mask_scope="timestep",
                position_variance_floor=1.0e-12,
                rotation_variance_floor=1.0e-12,
                maximum_modes=1,
                clustering_length=len(raw_link),
                resampling_method="interpolate",
            )
        ).fit(demonstrations)


def test_algorithm_1_applies_eq5_mask_per_timestep_and_restores_unlinked_frame() -> None:
    demonstrations, raw_link = link_transition_demonstrations()
    expected_link = raw_link
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=1.0e-3,
            tau_omega=0.0,
            link_filter="none",
            link_mask_scope="timestep",
            position_variance_floor=1.0e-12,
            rotation_variance_floor=1.0e-12,
            maximum_modes=1,
            clustering_length=len(raw_link),
            resampling_method="interpolate",
        )
    ).fit(demonstrations)
    skill = policy.skills[0]
    diagnostics = skill.link_diagnostics["object"]
    np.testing.assert_array_equal(diagnostics["raw_link_mask"], raw_link)
    np.testing.assert_array_equal(diagnostics["filtered_link_mask"], expected_link)
    assert "object" in skill.selected_frames
    np.testing.assert_array_equal(skill.streams["object"].active, ~expected_link)
    selection = policy.training_audit["skills"][0]["task_parameter_selection"]
    assert (
        selection["kinematic_link_granularity"]
        == "per_timestep_within_skill_eq5"
    )
    assert selection["task_parameter_selection_granularity"] == "per_skill_max_over_time_eq6"
    np.testing.assert_array_equal(
        selection["poe_participation_mask"]["object"][0],
        ~expected_link,
    )
    assert policy.selection_semantics_id == (
        "eq5_timestep_availability_before_eq6_and_poe_"
        "time_state_position3d_unimodal_v1"
    )

    first = demonstrations[0]
    initial = DynaMACObservation(first.ee_pose[0], {"object": first.frames["object"][0]})
    policy.reset(initial, mode_strategy="map")
    for time_index in range(len(raw_link)):
        observation = DynaMACObservation(
            first.ee_pose[time_index], {"object": first.frames["object"][time_index]}
        )
        action = policy.act(observation)
        assert ("object" in action.diagnostics["active_frames"]) is (not expected_link[time_index])
        assert action.diagnostics["frame_status"]["object"][
            "participates_in_poe_at_t"
        ] is (not expected_link[time_index])
        assert action.diagnostics["frame_status"]["object"][
            "exogenous_for_skill_by_eq5"
        ] is None
        assert set(action.diagnostics["marginal_means"]) == set(action.diagnostics["active_frames"])

    # The object is masked at t=4..7 and rejoins the PoE at t=8 within one skill.
    np.testing.assert_array_equal(
        skill.streams["object"].active[3:10],
        [True, False, False, False, False, True, True],
    )


def test_v3_majority_gate_preserves_raw_mask_in_eq6_and_final_poe() -> None:
    raw_link = np.asarray(
        [False, True, True, False, True, True, True, True, False, False]
    )
    demonstrations, measured_raw = link_transition_demonstrations(raw_link)
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=1.0e-3,
            tau_omega=0.0,
            eq6_covariance_scope="eq5_weighted_subspace",
            link_filter="none",
            link_mask_scope="skill_majority_gate_timestep",
            position_variance_floor=1.0e-12,
            rotation_variance_floor=1.0e-12,
            maximum_modes=1,
            clustering_length=len(raw_link),
            resampling_method="interpolate",
            default_mode_strategy="map",
        )
    ).fit(demonstrations)

    skill = policy.skills[0]
    diagnostics = skill.link_diagnostics["object"]
    selection = policy.training_audit["skills"][0]["task_parameter_selection"]
    np.testing.assert_array_equal(measured_raw, raw_link)
    np.testing.assert_array_equal(diagnostics["raw_link_mask"], raw_link)
    np.testing.assert_array_equal(diagnostics["filtered_link_mask"], raw_link)
    assert diagnostics["majority_gate_enabled"] is True
    assert diagnostics["majority_gate_rule"] == "strict_mean_raw_linked_gt_0.5"
    np.testing.assert_array_equal(selection["eq5_availability"]["object"], ~raw_link)
    object_index = selection["frame_names"].index("object")
    np.testing.assert_array_equal(
        selection["relative_precision"][object_index, raw_link],
        np.zeros(np.count_nonzero(raw_link)),
    )
    np.testing.assert_array_equal(
        selection["poe_participation_mask"]["object"][0],
        ~raw_link,
    )
    np.testing.assert_array_equal(skill.streams["object"].active, ~raw_link)
    assert selection["kinematic_link_granularity"] == (
        "skill_majority_gate_then_raw_per_timestep_eq5"
    )

    first = demonstrations[0]
    policy.reset(
        DynaMACObservation(first.ee_pose[0], {"object": first.frames["object"][0]}),
        mode_strategy="map",
    )
    for time_index, linked in enumerate(raw_link):
        action = policy.act(
            DynaMACObservation(
                first.ee_pose[time_index],
                {"object": first.frames["object"][time_index]},
            )
        )
        assert ("object" in action.diagnostics["active_frames"]) is (not linked)


def test_v3_exact_half_gate_keeps_frame_available_in_eq6_and_final_poe() -> None:
    demonstrations, raw_link = link_transition_demonstrations()
    assert np.mean(raw_link) == 0.5
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=1.0e-3,
            tau_omega=0.0,
            eq6_covariance_scope="eq5_weighted_subspace",
            link_filter="none",
            link_mask_scope="skill_majority_gate_timestep",
            position_variance_floor=1.0e-12,
            rotation_variance_floor=1.0e-12,
            maximum_modes=1,
            clustering_length=len(raw_link),
            resampling_method="interpolate",
            default_mode_strategy="map",
        )
    ).fit(demonstrations)

    skill = policy.skills[0]
    diagnostics = skill.link_diagnostics["object"]
    selection = policy.training_audit["skills"][0]["task_parameter_selection"]
    expected_available = np.ones(len(raw_link), dtype=bool)
    assert diagnostics["majority_gate_enabled"] is False
    np.testing.assert_array_equal(
        diagnostics["filtered_link_mask"], np.zeros(len(raw_link), dtype=bool)
    )
    np.testing.assert_array_equal(
        selection["eq5_availability"]["object"], expected_available
    )
    object_index = selection["frame_names"].index("object")
    assert np.all(selection["relative_precision"][object_index] > 0.0)
    np.testing.assert_array_equal(
        selection["poe_participation_mask"]["object"][0], expected_available
    )
    np.testing.assert_array_equal(skill.streams["object"].active, expected_available)


def test_main_and_no_ka_bind_their_own_eq5_masks_into_eq6_and_final_poe() -> None:
    demonstrations, expected_main_link = link_transition_demonstrations()
    common = {
        "tau_omega": 0.0,
        "link_filter": "none",
        "link_mask_scope": "timestep",
        "position_variance_floor": 1.0e-12,
        "rotation_variance_floor": 1.0e-12,
        "maximum_modes": 1,
        "clustering_length": len(expected_main_link),
        "resampling_method": "interpolate",
    }
    main = DynaMAC(DynaMACConfig(tau_m=1.0e-3, **common)).fit(demonstrations)
    # With these floors Eq. (5)'s GMSD lower bound is 1e-6, so 5e-7
    # deterministically realizes the experiment's no-KA configuration.
    no_ka = DynaMAC(DynaMACConfig(tau_m=5.0e-7, **common)).fit(demonstrations)

    main_selection = main.training_audit["skills"][0]["task_parameter_selection"]
    no_ka_selection = no_ka.training_audit["skills"][0]["task_parameter_selection"]
    assert main_selection["eq5_filters_eq6_denominator"] is True
    assert no_ka_selection["eq5_filters_eq6_denominator"] is True
    for name in main_selection["candidate_covariance"]:
        np.testing.assert_array_equal(
            main_selection["candidate_covariance"][name],
            no_ka_selection["candidate_covariance"][name],
        )

    np.testing.assert_array_equal(
        main_selection["eq5_availability"]["object"],
        ~expected_main_link,
    )
    np.testing.assert_array_equal(
        no_ka_selection["eq5_availability"]["object"],
        np.ones(len(expected_main_link), dtype=bool),
    )
    np.testing.assert_array_equal(
        main_selection["relative_precision"][0, expected_main_link],
        np.zeros(np.count_nonzero(expected_main_link)),
    )
    assert np.all(no_ka_selection["relative_precision"][0, expected_main_link] > 0.0)
    assert main_selection["scores"] != no_ka_selection["scores"]
    object_argmax = main_selection["argmax_time"]["object"]
    assert object_argmax is not None
    assert not expected_main_link[object_argmax]
    np.testing.assert_array_equal(
        main_selection["poe_participation_mask"]["object"][0],
        ~expected_main_link,
    )
    np.testing.assert_array_equal(
        no_ka_selection["poe_participation_mask"]["object"][0],
        np.ones(len(expected_main_link), dtype=bool),
    )


def virtual_history_demonstrations() -> list[DynaMACDemonstration]:
    demonstrations = []
    labels = (10, 20, 30)
    steps_per_skill = 5
    for demo_index in range(5):
        centered = demo_index - 2.0
        anchor = pose(
            [0.35 + 0.04 * centered, -0.15 + 0.03 * centered, 0.2 + 0.025 * centered],
            [0.05 * centered, -0.04 * centered, 0.03 * centered],
        )
        ee = []
        for skill_index, _ in enumerate(labels):
            for time_index in range(steps_per_skill):
                local = pose(
                    [
                        0.12 * skill_index + 0.015 * time_index,
                        0.04 * skill_index - 0.006 * time_index,
                        0.02 * time_index,
                    ],
                    [0.015 * skill_index, -0.01 * time_index, 0.008 * time_index],
                )
                ee.append(pose_compose(anchor, local))
        ee_array = np.stack(ee)
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=ee_array,
                action_pose=ee_array.copy(),
                gripper=np.zeros(len(ee_array)),
                frames={
                    "world": np.repeat(
                        np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                        len(ee_array),
                        axis=0,
                    )
                },
                skill=np.repeat(labels, steps_per_skill),
                name=f"history_{demo_index}",
            )
        )
    return demonstrations


def test_algorithm_1_accumulates_and_freezes_all_past_virtual_frames() -> None:
    demonstrations = virtual_history_demonstrations()
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=1.0e-3,
            tau_omega=0.0,
            link_filter="none",
            position_variance_floor=1.0e-12,
            rotation_variance_floor=1.0e-12,
            maximum_modes=1,
            clustering_length=5,
            resampling_method="interpolate",
        )
    ).fit(demonstrations)

    for skill_index, skill in enumerate(policy.skills):
        expected_virtual = {
            f"virtual_skill_{label}" for label in policy.skill_sequence[: skill_index + 1]
        }
        assert expected_virtual <= set(skill.selection_scores)
        assert expected_virtual <= set(skill.selected_frames)

        audit = policy.training_audit["skills"][skill_index]
        assert set(audit["virtual_frame_history"]) == expected_virtual
        assert expected_virtual <= set(audit["candidate_frames"])
        assert expected_virtual <= set(
            audit["task_parameter_selection"]["frame_names"]
        )
        for virtual_name in expected_virtual:
            assert audit["virtual_frame_start_poses"][virtual_name].shape == (5, 7)

    identity = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def observation(ee_pose: np.ndarray) -> DynaMACObservation:
        return DynaMACObservation(ee_pose, {"world": identity})

    captures = [
        pose([0.7, -0.1, 0.2], [0.1, 0.0, 0.0]),
        pose([0.8, 0.2, 0.3], [0.0, -0.15, 0.0]),
        pose([0.9, -0.3, 0.4], [0.0, 0.0, 0.2]),
    ]
    policy.reset(observation(captures[0]), mode_strategy="map")
    np.testing.assert_allclose(policy._virtual_frames["virtual_skill_10"], captures[0])

    for _ in range(policy.skills[0].duration):
        policy.act(observation(pose([9.0, 9.0, 9.0])))
    np.testing.assert_allclose(policy._virtual_frames["virtual_skill_10"], captures[0])

    # The next ``act`` call uses one skill-start observation both to capture the new
    # frame and to run inference for the current skill.
    policy.act(observation(captures[1]))
    assert set(policy._virtual_frames) == {"virtual_skill_10", "virtual_skill_20"}
    np.testing.assert_allclose(policy._virtual_frames["virtual_skill_10"], captures[0])
    np.testing.assert_allclose(policy._virtual_frames["virtual_skill_20"], captures[1])

    for _ in range(policy.skills[1].duration - 1):
        policy.act(observation(pose([-9.0, -9.0, -9.0])))
    policy.act(observation(captures[2]))
    assert set(policy._virtual_frames) == {
        "virtual_skill_10",
        "virtual_skill_20",
        "virtual_skill_30",
    }
    for label, expected in zip(policy.skill_sequence, captures, strict=True):
        np.testing.assert_allclose(policy._virtual_frames[f"virtual_skill_{label}"], expected)
