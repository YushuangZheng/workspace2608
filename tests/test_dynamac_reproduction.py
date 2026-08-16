"""Tests for the DynaMAC equations, Algorithm 1, and inference boundaries."""

from __future__ import annotations

import numpy as np
from essay2608.policy.dynamac import (
    MODEL_SCHEMA_VERSION,
    SELECTION_SEMANTICS_ID,
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
    GaussianMarginal,
    SkillModel,
    _filter_link_mask,
    _transition_probabilities,
    geometric_mean_standard_deviation,
    interpolate_poses,
    pose_compose,
    pose_inverse,
    product_of_experts,
    quaternion_exp,
    quaternion_log,
    relative_pose,
    task_parameter_scores,
    transform_marginal,
)
from essay2608.policy.midigap import TaskParameterizedMiDiGaP


def pose(position, yaw: float = 0.0) -> np.ndarray:
    return np.asarray(
        [*position, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float64,
    )


def synthetic_demonstrations() -> list[DynaMACDemonstration]:
    demonstrations = []
    for demo in range(5):
        steps_per_skill = 8
        skill = np.repeat([10, 20], steps_per_skill)
        ee = []
        object_frame = []
        target_frame = []
        for index in range(steps_per_skill):
            progress = index / (steps_per_skill - 1)
            object_pose = pose(
                [0.45 + 0.025 * demo, -0.12 + 0.009 * demo, 0.05 + 0.004 * demo],
                yaw=0.018 * demo,
            )
            # Before grasping, the object-relative end-effector pose varies across
            # demonstrations, so the object remains an exogenous frame.
            local = pose(
                [
                    -0.08 * (1.0 - progress) + 0.004 * demo,
                    0.003 * demo,
                    0.10 * (1.0 - progress) + 0.002 * demo,
                ],
                yaw=-0.015 * demo,
            )
            ee_pose = pose_compose(object_pose, local)
            ee.append(ee_pose)
            object_frame.append(object_pose)
            target_frame.append(
                pose([0.68 - 0.013 * demo, 0.20 + 0.017 * demo, 0.08], -0.02 * demo)
            )
        grasp_offset = pose([0.0, 0.0, -0.045])
        start = ee[-1]
        for index in range(steps_per_skill):
            progress = index / (steps_per_skill - 1)
            ee_pose = pose_compose(start, pose([0.02 * progress, 0.0, 0.12 * progress]))
            # After grasping, the object and end effector maintain a fixed transform,
            # which Eq. (5) should identify as a link.
            object_pose = pose_compose(ee_pose, pose_inverse(grasp_offset))
            ee.append(ee_pose)
            object_frame.append(object_pose)
            target_frame.append(
                pose([0.68 - 0.013 * demo, 0.20 + 0.017 * demo, 0.08], -0.02 * demo)
            )
        ee_array = np.stack(ee)
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=ee_array,
                action_pose=ee_array.copy(),
                gripper=np.concatenate((np.ones(steps_per_skill), -np.ones(steps_per_skill))),
                frames={
                    "object": np.stack(object_frame),
                    "target": np.stack(target_frame),
                },
                skill=skill,
                name=f"synthetic_{demo}",
            )
        )
    return demonstrations


def test_pose_round_trip() -> None:
    frame = pose([0.3, -0.2, 0.1], yaw=0.7)
    value = pose([0.6, 0.4, -0.1], yaw=-0.2)
    np.testing.assert_allclose(pose_compose(frame, relative_pose(frame, value)), value)


def test_demonstration_rejects_fractional_nonfinite_skills_and_rank_three_gripper() -> None:
    poses = np.repeat(pose([0.0, 0.0, 0.0])[None], 3, axis=0)
    frames = {"world": poses.copy()}
    invalid_skills = (
        np.asarray([0.0, np.nan, 1.0]),
        np.asarray([0.0, 0.5, 1.0]),
    )
    for invalid_skill in invalid_skills:
        with np.testing.assert_raises(ValueError):
            DynaMACDemonstration(
                poses,
                poses,
                np.zeros(3),
                frames,
                invalid_skill,
            )
    with np.testing.assert_raises(ValueError):
        DynaMACDemonstration(
            poses,
            poses,
            np.zeros((3, 1, 1)),
            frames,
            np.zeros(3),
        )


def test_quaternion_log_accepts_scalar_negative_representative() -> None:
    np.testing.assert_allclose(quaternion_log(np.asarray([-1.0, 0.0, 0.0, 0.0])), 0.0)


def test_pose_resampling_uses_spherical_quaternion_interpolation() -> None:
    start = pose([0.0, 0.0, 0.0], yaw=0.0)
    end = pose([1.0, 0.0, 0.0], yaw=np.pi)
    midpoint = interpolate_poses(np.stack((start, end)), 3)[1]
    np.testing.assert_allclose(midpoint[:3], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(midpoint[3:7], pose([0.0, 0.0, 0.0], yaw=np.pi / 2)[3:7])


def test_equation_5_defaults_to_position_only_three_dimensional_gmsd() -> None:
    standard_deviation = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64) * 1.0e-3
    covariance = np.diag(np.square(standard_deviation))
    expected_position = float(np.prod(standard_deviation[:3]) ** (1.0 / 3.0))
    np.testing.assert_allclose(
        geometric_mean_standard_deviation(covariance),
        expected_position,
    )

    expected_full_pose = float(np.prod(standard_deviation) ** (1.0 / 6.0))
    np.testing.assert_allclose(
        geometric_mean_standard_deviation(
            covariance,
            position_weight=1.0,
            rotation_weight=1.0,
        ),
        expected_full_pose,
    )


def test_author_confirmed_defaults_and_schema_are_frozen() -> None:
    config = DynaMACConfig()
    assert MODEL_SCHEMA_VERSION == 13
    assert (
        SELECTION_SEMANTICS_ID
        == "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
    )
    assert config.policy_model == "time_state"
    assert config.kinematic_analysis_enabled is True
    assert config.tau_m == 0.005
    assert config.tau_omega == 0.5
    assert config.eq5_position_weight == 1.0
    assert config.eq5_rotation_weight == 0.0
    assert config.link_mask_scope == "skill_majority"
    assert config.modal_partition_method == "none"
    assert config.position_variance_floor == 1.0e-6
    assert config.rotation_variance_floor == 1.0e-6
    assert config.covariance_estimation_method == "diagonal_empirical_ridge"


def test_eq5_time_mask_uses_strict_majority_and_becomes_skill_constant() -> None:
    boundary_scale = np.asarray([0.001, 0.002, 0.006, 0.007])
    raw, skill_mask = _filter_link_mask(boundary_scale, DynaMACConfig())
    np.testing.assert_array_equal(raw, [True, True, False, False])
    # Exactly 50% is not linked: the author rule is strict > 0.5.
    np.testing.assert_array_equal(skill_mask, np.zeros(4, dtype=bool))

    raw, skill_mask = _filter_link_mask(
        np.asarray([0.001, 0.002, 0.003, 0.007]),
        DynaMACConfig(),
    )
    np.testing.assert_array_equal(raw, [True, True, True, False])
    np.testing.assert_array_equal(skill_mask, np.ones(4, dtype=bool))

    _, experimental_time_mask = _filter_link_mask(
        boundary_scale,
        DynaMACConfig(link_mask_scope="timestep"),
    )
    np.testing.assert_array_equal(experimental_time_mask, [True, True, False, False])


def test_equation_6_uses_maximum_relative_precision() -> None:
    covariance = {
        "a": np.stack((np.eye(6), np.eye(6) * 100.0)),
        "b": np.stack((np.eye(6) * 100.0, np.eye(6))),
    }
    scores = task_parameter_scores(
        covariance,
        availability={name: np.ones(2, dtype=bool) for name in covariance},
        candidate_kind={name: "dynamic" for name in covariance},
    )
    assert scores["a"] > 0.999999
    assert scores["b"] > 0.999999


def test_world_transform_and_product_of_experts() -> None:
    local = pose([0.1, 0.0, 0.0])
    marginal = transform_marginal(
        "moving",
        pose([0.5, 0.0, 0.0], yaw=np.pi / 2.0),
        local,
        np.eye(6) * 0.02,
    )
    np.testing.assert_allclose(marginal.mean[:3], [0.5, 0.1, 0.0], atol=1.0e-12)
    fused, covariance, weights = product_of_experts(
        [
            GaussianMarginal("left", pose([0.0, 0.0, 0.0]), np.eye(6)),
            GaussianMarginal("right", pose([2.0, 0.0, 0.0]), np.eye(6)),
        ]
    )
    np.testing.assert_allclose(fused[:3], [1.0, 0.0, 0.0], atol=1.0e-9)
    np.testing.assert_allclose(covariance, np.eye(6) * 0.5)
    assert weights == {"left": 0.5, "right": 0.5}


def test_algorithm_1_masks_linked_object_and_uses_virtual_frame(tmp_path) -> None:
    policy = DynaMAC(
        DynaMACConfig(
            maximum_modes=1,
            tau_m=0.001,
            tau_omega=0.19,
            position_variance_floor=1.0e-8,
            rotation_variance_floor=1.0e-8,
        )
    ).fit(synthetic_demonstrations())
    assert policy.skill_sequence == (10, 20)
    assert not policy.skills[0].link_diagnostics["object"]["linked"]
    assert policy.skills[1].link_diagnostics["object"]["linked"]
    assert policy.skills[1].link_diagnostics["object"]["fully_linked"]
    # Eq. (5) removes the fully linked real frame before Eq. (6)'s denominator;
    # the always-available virtual frame supplies the skill without a fallback.
    assert "object" not in policy.skills[1].selected_frames
    second_selection = policy.training_audit["skills"][1]["task_parameter_selection"]
    assert second_selection["eq5_filters_eq6_denominator"] is True
    np.testing.assert_array_equal(
        second_selection["eq5_availability"]["object"],
        np.zeros(policy.skills[1].duration, dtype=bool),
    )
    np.testing.assert_array_equal(
        second_selection["relative_precision"][0],
        np.zeros(policy.skills[1].duration),
    )
    assert second_selection["argmax_time"]["object"] is None
    assert "virtual_skill_20" in policy.skills[1].selected_frames
    for skill in policy.skills:
        assert sorted(
            index for members in skill.mode_demonstration_indices for index in members
        ) == list(range(len(synthetic_demonstrations())))
    assert all(
        frame == f"virtual_skill_{skill.label}"
        for skill in policy.skills
        for frame in skill.selected_frames
        if frame.startswith("virtual_skill_")
    )

    first = synthetic_demonstrations()[0]
    observation = DynaMACObservation(
        first.ee_pose[0], {name: values[0] for name, values in first.frames.items()}
    )
    policy.reset(observation)
    action = policy.act(observation)
    assert (
        action.diagnostics["selection_mode"]
        == "eq6_per_skill_with_eq5_skill_mask"
    )
    assert action.diagnostics["kinematic_link_granularity"] == (
        "offline_per_skill_strict_majority"
    )
    assert action.diagnostics["task_parameter_selection_granularity"] == (
        "offline_per_skill_max_over_time"
    )
    assert not action.diagnostics["online_link_detection"]

    checkpoint = tmp_path / "dynamac.npz"
    policy.save(checkpoint)
    restored = DynaMAC.load(checkpoint)
    assert restored.fingerprint() == policy.fingerprint()
    assert restored.summary() == policy.summary()


def test_no_kinematic_analysis_is_an_explicit_eq5_bypass() -> None:
    config = DynaMACConfig(
        kinematic_analysis_enabled=False,
        maximum_modes=1,
        tau_omega=0.19,
    )
    policy = DynaMAC(config).fit(synthetic_demonstrations())

    assert policy.config.tau_m == 0.005
    assert policy.training_audit["kinematic_analysis"] == {
        "enabled": False,
        "equation": "Eq. (5)",
        "disabled_behavior": "explicit_bypass_all_dynamic_candidates_available",
        "tau_m_retained": 0.005,
    }
    for skill, audit in zip(policy.skills, policy.training_audit["skills"], strict=True):
        link = skill.link_diagnostics["object"]
        assert link["analysis_performed"] is False
        assert link["disabled_behavior"] == (
            "explicit_bypass_all_dynamic_candidates_available"
        )
        assert link["linked"] is False
        np.testing.assert_array_equal(
            audit["task_parameter_selection"]["eq5_availability"]["object"],
            np.ones(skill.duration, dtype=bool),
        )
        assert audit["task_parameter_selection"]["eq5_bypass"] == (
            "all_dynamic_candidates_available"
        )
        assert audit["task_parameter_selection"]["eq5_filters_eq6_candidates"] is False
        assert audit["task_parameter_selection"]["eq5_filters_eq6_denominator"] is False


def test_link_detection_uses_measured_ee_not_commanded_action() -> None:
    demonstrations = []
    for demo_index, source in enumerate(synthetic_demonstrations()):
        action = source.action_pose.copy()
        offset = np.concatenate(
            (
                demo_index * np.asarray([0.03, -0.02, 0.025]),
                quaternion_exp(demo_index * np.asarray([0.05, -0.04, 0.03])),
            )
        )
        action[8:] = pose_compose(action[8:], offset)
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=source.ee_pose,
                action_pose=action,
                gripper=source.gripper,
                frames=source.frames,
                skill=source.skill,
                name=source.name,
            )
        )
    policy = DynaMAC(DynaMACConfig(maximum_modes=1, tau_omega=0.19)).fit(demonstrations)
    assert policy.skills[1].link_diagnostics["object"]["linked"]


def test_current_ee_time_state_drives_eq5_eq6_and_final_fit_not_next_action() -> None:
    demonstrations = []
    time_state = np.stack(
        [pose([0.0, 0.0, 0.0]), pose([0.1, 0.0, 0.0]), pose([0.2, 0.0, 0.0])]
    )
    world = np.repeat(pose([0.0, 0.0, 0.0])[None], len(time_state), axis=0)
    for demo_index in range(3):
        next_action = np.stack(
            [
                pose([1.0 + demo_index, 0.0, 0.0]),
                pose([1.2 + demo_index, 0.0, 0.0]),
                pose([1.4 + demo_index, 0.0, 0.0]),
            ]
        )
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=time_state,
                action_pose=next_action,
                gripper=np.zeros(len(time_state)),
                frames={"world": world},
                skill=np.zeros(len(time_state)),
                name=f"stream_{demo_index}",
            )
        )

    time_state_policy = DynaMAC(DynaMACConfig(tau_omega=0.0)).fit(demonstrations)
    action_policy = DynaMAC(
        DynaMACConfig(policy_model="action_pose", tau_omega=0.0)
    ).fit(demonstrations)
    time_state_stream = time_state_policy.skills[0].streams["virtual_skill_0"]
    action_stream = action_policy.skills[0].streams["virtual_skill_0"]
    assert time_state_policy.skills[0].link_diagnostics["world"]["linked"] is True
    assert action_policy.skills[0].link_diagnostics["world"]["linked"] is False
    np.testing.assert_allclose(time_state_stream.mean[0, :, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(action_stream.mean[0, :, 0], [2.0, 2.2, 2.4])
    np.testing.assert_allclose(
        time_state_policy.training_audit["skills"][0]["task_parameter_selection"][
            "candidate_covariance"
        ]["virtual_skill_0"],
        time_state_stream.covariance[0],
    )
    assert time_state_policy.training_audit["skills"][0]["policy_model"] == "time_state"
    np.testing.assert_allclose(
        time_state_policy.training_audit["skills"][0]["local_policy"][
            "virtual_skill_0"
        ][:, :, 0],
        np.broadcast_to([0.0, 0.1, 0.2], (3, 3)),
    )


def test_static_midigap_uses_the_same_author_confirmed_time_state_stream() -> None:
    demonstrations = []
    current_ee = np.stack(
        [pose([0.0, 0.0, 0.0]), pose([0.1, 0.0, 0.0]), pose([0.2, 0.0, 0.0])]
    )
    world = np.repeat(pose([0.0, 0.0, 0.0])[None], len(current_ee), axis=0)
    for demo_index in range(3):
        next_action = np.stack(
            [
                pose([1.0 + demo_index, 0.0, 0.0]),
                pose([1.2 + demo_index, 0.0, 0.0]),
                pose([1.4 + demo_index, 0.0, 0.0]),
            ]
        )
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=current_ee,
                action_pose=next_action,
                gripper=np.zeros(len(current_ee)),
                frames={"world": world},
                skill=np.zeros(len(current_ee)),
                name=f"static_stream_{demo_index}",
            )
        )

    time_state = TaskParameterizedMiDiGaP(
        DynaMACConfig(tau_omega=0.0)
    ).fit(demonstrations)
    action_pose = TaskParameterizedMiDiGaP(
        DynaMACConfig(policy_model="action_pose", tau_omega=0.0)
    ).fit(demonstrations)

    np.testing.assert_allclose(
        time_state.skills[0].streams["world"].mean[0, :, 0],
        [0.0, 0.1, 0.2],
    )
    np.testing.assert_allclose(
        action_pose.skills[0].streams["world"].mean[0, :, 0],
        [2.0, 2.2, 2.4],
    )


def test_default_unimodal_fit_skips_partition_and_uses_zero_labels() -> None:
    policy = DynaMAC().fit(synthetic_demonstrations())
    for skill, audit in zip(policy.skills, policy.training_audit["skills"], strict=True):
        np.testing.assert_array_equal(audit["mode_labels"], np.zeros(5, dtype=np.int64))
        np.testing.assert_allclose(skill.mode_priors, [1.0])
        assert skill.mode_demonstration_indices == ((0, 1, 2, 3, 4),)
        assert audit["modal_partition"]["method"] == "none"
        assert audit["modal_partition"]["partition_performed"] is False
        assert audit["modal_partition"]["unimodal"] is True


def test_midigap_transition_equation_and_global_map_path() -> None:
    transition = _transition_probabilities(
        np.asarray([0, 0, 1, 1, 1]),
        np.asarray([1, 1, 0, 0, 1]),
    )
    np.testing.assert_allclose(transition, [[0.0, 1.0], [2.0 / 3.0, 1.0 / 3.0]])

    policy = DynaMAC()
    policy.skills = [
        SkillModel(0, 1, (), np.asarray([0.6, 0.4]), {}, np.zeros((2, 1, 1))),
        SkillModel(
            1,
            1,
            (),
            np.asarray([0.5, 0.5]),
            {},
            np.zeros((2, 1, 1)),
            transition_from_previous=np.asarray([[0.1, 0.9], [0.95, 0.05]]),
        ),
    ]
    assert policy._select_mode_path("map") == (0, 1)


def test_inference_changes_marginals_not_offline_frame_set() -> None:
    policy = DynaMAC(DynaMACConfig(maximum_modes=1, tau_omega=0.19)).fit(synthetic_demonstrations())
    first = synthetic_demonstrations()[0]
    original = DynaMACObservation(
        first.ee_pose[0], {name: values[0].copy() for name, values in first.frames.items()}
    )
    moved_frames = {name: value.copy() for name, value in original.frames.items()}
    moved_frames["object"][:3] += [0.08, 0.0, 0.0]
    moved = DynaMACObservation(original.ee_pose.copy(), moved_frames)
    policy.reset(original)
    first_action = policy.act(original)
    policy.reset(moved)
    moved_action = policy.act(moved)
    assert (
        first_action.diagnostics["selected_frames"] == moved_action.diagnostics["selected_frames"]
    )
    if "object" in first_action.diagnostics["selected_frames"]:
        assert moved_action.pose[0] > first_action.pose[0]


def test_fit_and_refit_require_an_explicit_episode_reset() -> None:
    demonstrations = synthetic_demonstrations()
    policy = DynaMAC(DynaMACConfig(maximum_modes=1, tau_omega=0.19)).fit(demonstrations)
    first = demonstrations[0]
    observation = DynaMACObservation(
        first.ee_pose[0], {name: values[0] for name, values in first.frames.items()}
    )

    with np.testing.assert_raises_regex(RuntimeError, "reset"):
        policy.act(observation)
    policy.reset(observation, mode_strategy="map")
    policy.act(observation)

    policy.fit(demonstrations)
    with np.testing.assert_raises_regex(RuntimeError, "reset"):
        policy.act(observation)


def test_failed_refit_restores_model_audit_and_live_episode(monkeypatch) -> None:
    demonstrations = synthetic_demonstrations()
    policy = DynaMAC(DynaMACConfig(maximum_modes=1, tau_omega=0.19)).fit(
        demonstrations
    )
    first = demonstrations[0]
    observation = DynaMACObservation(
        first.ee_pose[0],
        {name: values[0] for name, values in first.frames.items()},
    )
    policy.reset(observation, mode_strategy="map")
    policy.act(observation)

    def freeze(value):
        if isinstance(value, np.ndarray):
            return ("array", value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, dict):
            return tuple((key, freeze(item)) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    expected_fingerprint = policy.fingerprint()
    expected_audit = freeze(policy.training_audit)
    expected_skill_index = policy._skill_index
    expected_time_index = policy._time_index
    expected_virtual_frames = {
        name: value.copy() for name, value in policy._virtual_frames.items()
    }

    def corrupt_then_fail(self, _demonstrations):
        self.frame_names = ("rejected_frame",)
        self.skill_sequence = (999,)
        self.skills = []
        self._training_audit = {"rejected": np.asarray([1.0])}
        self._skill_index = 999
        self._time_index = 999
        self._virtual_frames = {"rejected": pose([9.0, 9.0, 9.0])}
        self._episode_initialized = False
        raise RuntimeError("synthetic rejected refit")

    monkeypatch.setattr(DynaMAC, "_fit_in_place", corrupt_then_fail)
    with np.testing.assert_raises_regex(RuntimeError, "synthetic rejected refit"):
        policy.fit(demonstrations)

    assert policy.fingerprint() == expected_fingerprint
    assert freeze(policy.training_audit) == expected_audit
    assert policy._skill_index == expected_skill_index
    assert policy._time_index == expected_time_index
    assert policy._episode_initialized is True
    assert policy._virtual_frames.keys() == expected_virtual_frames.keys()
    for name, value in expected_virtual_frames.items():
        np.testing.assert_array_equal(policy._virtual_frames[name], value)

    next_action = policy.act(observation)
    assert next_action.pose.shape == (7,)
    assert next_action.covariance.shape == (6, 6)
    assert policy._time_index == expected_time_index + 1


def test_virtual_frame_is_captured_from_next_skill_start_observation() -> None:
    policy = DynaMAC(DynaMACConfig(maximum_modes=1, tau_omega=0.19)).fit(synthetic_demonstrations())
    first = synthetic_demonstrations()[0]
    initial = DynaMACObservation(
        first.ee_pose[0], {name: values[0] for name, values in first.frames.items()}
    )
    policy.reset(initial)
    for _ in range(policy.skills[0].duration):
        policy.act(initial)

    next_start_pose = pose([0.91, -0.17, 0.32], yaw=0.4)
    next_start = DynaMACObservation(next_start_pose, initial.frames)
    policy.act(next_start)
    np.testing.assert_allclose(policy._virtual_frames["virtual_skill_20"], next_start_pose)
