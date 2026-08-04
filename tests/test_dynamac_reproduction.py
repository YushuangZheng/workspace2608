"""DynaMAC 论文公式、Algorithm 1 和推理边界测试。"""

from __future__ import annotations

import numpy as np
from essay2608.policy.dynamac import (
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
    GaussianMarginal,
    SkillModel,
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
            # 抓取前，末端相对物体在六维上具有演示间变化，物体仍是外生帧。
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
            # 抓取后物体与末端保持同一固定变换，式 (5) 应识别链接。
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


def test_quaternion_log_accepts_scalar_negative_representative() -> None:
    np.testing.assert_allclose(quaternion_log(np.asarray([-1.0, 0.0, 0.0, 0.0])), 0.0)


def test_pose_resampling_uses_spherical_quaternion_interpolation() -> None:
    start = pose([0.0, 0.0, 0.0], yaw=0.0)
    end = pose([1.0, 0.0, 0.0], yaw=np.pi)
    midpoint = interpolate_poses(np.stack((start, end)), 3)[1]
    np.testing.assert_allclose(midpoint[:3], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(midpoint[3:7], pose([0.0, 0.0, 0.0], yaw=np.pi / 2)[3:7])


def test_equation_5_is_geometric_mean_standard_deviation() -> None:
    standard_deviation = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64) * 1.0e-3
    covariance = np.diag(np.square(standard_deviation))
    expected = float(np.prod(standard_deviation) ** (1.0 / 6.0))
    np.testing.assert_allclose(geometric_mean_standard_deviation(covariance), expected)


def test_equation_6_uses_maximum_relative_precision() -> None:
    covariance = {
        "a": np.stack((np.eye(6), np.eye(6) * 100.0)),
        "b": np.stack((np.eye(6) * 100.0, np.eye(6))),
    }
    scores = task_parameter_scores(covariance)
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
            tau_omega=0.19,
            link_fraction_threshold=0.5,
        )
    ).fit(synthetic_demonstrations())
    assert policy.skill_sequence == (10, 20)
    assert not policy.skills[0].link_diagnostics["object"]["linked"]
    assert policy.skills[1].link_diagnostics["object"]["linked"]
    assert "object" not in policy.skills[1].selected_frames
    assert "virtual_skill_20" in policy.skills[1].selected_frames

    first = synthetic_demonstrations()[0]
    observation = DynaMACObservation(
        first.ee_pose[0], {name: values[0] for name, values in first.frames.items()}
    )
    policy.reset(observation)
    action = policy.act(observation)
    assert action.diagnostics["selection_mode"] == "offline_skill_fixed"
    assert not action.diagnostics["online_link_detection"]

    checkpoint = tmp_path / "dynamac.npz"
    policy.save(checkpoint)
    restored = DynaMAC.load(checkpoint)
    assert restored.fingerprint() == policy.fingerprint()
    assert restored.summary() == policy.summary()


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
