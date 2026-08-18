"""Integration tests for DynaMAC persistence, configuration, and arm synchronization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from essay2608.policy.dynamac import (
    BimanualDynaMAC,
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
    SkillModel,
    StreamModel,
    pose_compose,
    quaternion_exp,
    task_parameter_scores,
)
from essay2608.policy.midigap import TaskParameterizedMiDiGaP


def pose(position, tangent=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.concatenate(
        (np.asarray(position, dtype=np.float64), quaternion_exp(np.asarray(tangent)))
    )


def test_schema_v13_checkpoint_roundtrips_explicit_timestep_component_masks(
    tmp_path: Path,
) -> None:
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=np.float32(0.001),
            link_mask_scope="timestep",
            preliminary_analysis="precluster_all_real_frame_product_mode_conditioned",
            maximum_modes=np.int64(2),
            clustering_variance_floor=np.float32(2.5e-4),
            clustering_restarts=np.int64(7),
            covariance_estimation_method="full_empirical_ridge",
            default_mode_strategy="map",
        )
    )
    policy.frame_names = ("object",)
    policy.skill_sequence = (0,)
    mean = np.repeat(pose([0.0, 0.0, 0.0])[None, None], 3, axis=1)
    mean = np.repeat(mean, 2, axis=0)
    covariance = np.repeat((np.eye(6) * 0.01)[None, None], 3, axis=1)
    covariance = np.repeat(covariance, 2, axis=0)
    component_active = np.asarray(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    policy.skills = [
        SkillModel(
            label=0,
            duration=3,
            selected_frames=("object", "virtual_skill_0"),
            mode_priors=np.asarray([0.5, 0.5]),
            streams={
                "object": StreamModel(
                    "object", mean.copy(), covariance.copy(), component_active
                ),
                "virtual_skill_0": StreamModel(
                    "virtual_skill_0",
                    mean.copy(),
                    covariance.copy(),
                    np.ones((2, 3), dtype=bool),
                ),
            },
            gripper=np.zeros((2, 3, 1)),
            mode_demonstration_indices=((0,), (1,)),
        )
    ]

    checkpoint = tmp_path / "masked_without_suffix"
    policy.save(checkpoint)
    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(".npz").exists()
    restored = DynaMAC.load(checkpoint)
    np.testing.assert_array_equal(
        restored.skills[0].streams["object"].active,
        component_active,
    )
    np.testing.assert_array_equal(
        restored.skills[0].streams["object"].availability,
        component_active,
    )
    np.testing.assert_array_equal(
        restored.skills[0].streams["object"].selected_by_eq6,
        [True, True],
    )
    assert restored.fingerprint() == policy.fingerprint()
    assert restored.summary()["model_schema_version"] == 13
    assert restored.summary()["selection_semantics_id"] == (
        "eq5_timestep_availability_before_eq6_and_poe_"
        "time_state_position3d_unimodal_v1"
    )
    assert restored.config == policy.config
    assert restored.summary()["config"] == asdict(policy.config)
    assert type(restored.config.tau_m) is float
    assert type(restored.config.maximum_modes) is int
    assert type(restored.config.clustering_variance_floor) is float
    assert type(restored.config.gripper_clustering_scale) is float
    assert type(restored.config.clustering_restarts) is int
    assert restored.config.covariance_estimation_method == "full_empirical_ridge"

    observation = DynaMACObservation(
        pose([0.0, 0.0, 0.0]),
        {"object": pose([0.0, 0.0, 0.0])},
    )
    restored.reset(
        observation,
        mode_strategy="map",
        mode_evidence=[np.asarray([1.0, 0.0])],
    )

    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    inconsistent_arrays = {name: value.copy() for name, value in arrays.items()}
    inconsistent_arrays[DynaMAC._array_key(0, "object", "active")] = np.ones_like(
        component_active
    )
    inconsistent = tmp_path / "schema_13_inconsistent_active.npz"
    np.savez_compressed(
        inconsistent,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        **inconsistent_arrays,
    )
    with pytest.raises(ValueError, match="active.*availability AND selected_by_eq6"):
        DynaMAC.load(inconsistent)
    for legacy_schema, message in (
        (6, "global gripper"),
        (7, "dimensional scale"),
        (9, "diagonal empirical covariance"),
        (10, r"Eq\. \(5\).*Eq\. \(6\)"),
        (11, r"Eq\. \(6\).*candidates unavailable under Eq\. \(5\)"),
        (12, "per-timestep link masks"),
    ):
        legacy_metadata = dict(metadata)
        legacy_metadata["model_schema_version"] = legacy_schema
        legacy = tmp_path / f"schema_{legacy_schema}.npz"
        np.savez_compressed(
            legacy,
            metadata_json=np.asarray(json.dumps(legacy_metadata, ensure_ascii=False)),
            **arrays,
        )
        with pytest.raises(ValueError, match=message):
            DynaMAC.load(legacy)
    assert "object" in restored.act(observation).diagnostics["active_frames"]
    restored.reset(
        observation,
        mode_strategy="map",
        mode_evidence=[np.asarray([0.0, 1.0])],
    )
    second_mode_action = restored.act(observation)
    assert second_mode_action.diagnostics["mode"] == 1
    assert "object" not in second_mode_action.diagnostics["active_frames"]


def test_checkpoint_policy_type_cannot_masquerade_as_dynamac(tmp_path: Path) -> None:
    static = TaskParameterizedMiDiGaP(DynaMACConfig(maximum_modes=1))
    static.frame_names = ("object",)
    static.skill_sequence = (0,)
    mean = pose([0.0, 0.0, 0.0])[None, None]
    covariance = (np.eye(6) * 0.01)[None, None]
    static.skills = [
        SkillModel(
            label=0,
            duration=1,
            selected_frames=("object",),
            mode_priors=np.ones(1),
            streams={"object": StreamModel("object", mean, covariance)},
            gripper=np.zeros((1, 1, 1)),
            mode_demonstration_indices=((0,),),
        )
    ]
    checkpoint = tmp_path / "static.npz"
    static.save(checkpoint)

    restored = TaskParameterizedMiDiGaP.load(checkpoint)
    assert restored.name == "midigap_static_frames"
    with pytest.raises(ValueError, match="checkpoint policy type"):
        DynaMAC.load(checkpoint)


def test_reset_does_not_require_permanently_rejected_distractor_frames() -> None:
    policy = DynaMAC(DynaMACConfig(maximum_modes=1, default_mode_strategy="map"))
    policy.frame_names = ("rejected_distractor",)
    policy.skill_sequence = (0,)
    mean = np.repeat(pose([0.0, 0.0, 0.0])[None, None], 2, axis=1)
    covariance = np.repeat((np.eye(6) * 0.01)[None, None], 2, axis=1)
    policy.skills = [
        SkillModel(
            label=0,
            duration=2,
            selected_frames=("virtual_skill_0",),
            mode_priors=np.ones(1),
            streams={
                "virtual_skill_0": StreamModel(
                    "virtual_skill_0", mean, covariance, np.ones(2, dtype=bool)
                )
            },
            gripper=np.zeros((1, 2, 1)),
        )
    ]
    observation = DynaMACObservation(pose([0.0, 0.0, 0.0]), {})
    policy.reset(observation, mode_strategy="map")
    action = policy.act(observation)
    assert action.pose.shape == (7,)
    assert action.covariance.shape == (6, 6)


def test_failed_skill_boundary_action_rolls_back_virtual_capture() -> None:
    policy = DynaMAC(DynaMACConfig(maximum_modes=1, default_mode_strategy="map"))
    policy.frame_names = ("object",)
    policy.skill_sequence = (0, 1)
    mean = pose([0.0, 0.0, 0.0])[None, None]
    covariance = (np.eye(6) * 0.01)[None, None]
    first_stream = StreamModel("virtual_skill_0", mean, covariance)
    second_streams = {
        "object": StreamModel("object", mean.copy(), covariance.copy()),
        "virtual_skill_1": StreamModel("virtual_skill_1", mean.copy(), covariance.copy()),
    }
    policy.skills = [
        SkillModel(
            0,
            1,
            ("virtual_skill_0",),
            np.ones(1),
            {"virtual_skill_0": first_stream},
            np.zeros((1, 1, 1)),
        ),
        SkillModel(
            1,
            1,
            ("object", "virtual_skill_1"),
            np.ones(1),
            second_streams,
            np.zeros((1, 1, 1)),
            transition_from_previous=np.ones((1, 1)),
        ),
    ]
    start = DynaMACObservation(pose([0.0, 0.0, 0.0]), {})
    policy.reset(start, mode_strategy="map")
    policy.act(start)
    assert policy._pending_virtual_capture

    with pytest.raises(ValueError, match="object"):
        policy.act(DynaMACObservation(pose([0.4, 0.0, 0.0]), {}))

    assert policy._pending_virtual_capture
    assert "virtual_skill_1" not in policy._virtual_frames
    assert policy._skill_index == 1
    assert policy._time_index == 0


def test_published_json_explicitly_freezes_every_dynamac_choice() -> None:
    path = Path(__file__).parents[1] / "configs" / "dynamac.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    configured = DynaMACConfig(**raw)
    assert set(raw) == set(asdict(DynaMACConfig()))
    assert configured.tau_omega == DynaMACConfig().tau_omega == 0.5
    assert configured.maximum_modes == 1


@pytest.mark.parametrize(
    "preliminary_analysis",
    [
        "paper_order_pooled",
        "precluster_all_real_frame_product_mode_conditioned",
    ],
)
def test_preliminary_analysis_accepts_both_supported_branches(
    preliminary_analysis: str,
) -> None:
    config = DynaMACConfig(preliminary_analysis=preliminary_analysis)
    assert config.preliminary_analysis == preliminary_analysis


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_variance_floor": np.nan},
        {"temporal_variance_threshold": np.nan},
        {"dbscan_epsilon": np.inf},
        {"gripper_clustering_scale": 0.0},
        {"maximum_modes": 1.5},
        {"random_seed": -1},
        {"preliminary_analysis": "unknown"},
        {"covariance_estimation_method": "diagonal_plus_ridge"},
        {"eq6_empty_selection": "unknown"},
    ],
)
def test_dynamac_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DynaMACConfig(**kwargs)


def test_default_threshold_uses_strict_author_equation_6_boundary() -> None:
    covariance = {
        "first": np.repeat(np.eye(6)[None], 2, axis=0),
        "second": np.repeat(np.eye(6)[None], 2, axis=0),
    }
    scores = task_parameter_scores(
        covariance,
        availability={name: np.ones(2, dtype=bool) for name in covariance},
        candidate_kind={name: "dynamic" for name in covariance},
    )
    assert scores == {"first": 0.5, "second": 0.5}
    assert all(not value > DynaMACConfig().tau_omega for value in scores.values())


def bimanual_demonstrations() -> tuple[list[DynaMACDemonstration], list[DynaMACDemonstration]]:
    left_demonstrations = []
    right_demonstrations = []
    duration = 6
    identity_frames = np.repeat(pose([0.0, 0.0, 0.0])[None], duration, axis=0)
    stale_cross_frames = np.repeat(pose([99.0, 99.0, 99.0])[None], duration, axis=0)
    for demo_index in range(7):
        centered = demo_index - 3
        left_ee = []
        right_ee = []
        left_action = []
        right_action = []
        for time_index in range(duration):
            right_pose = pose(
                [
                    0.4 + 0.025 * centered + 0.01 * time_index,
                    -0.15 + 0.012 * centered,
                    0.2 + 0.005 * time_index,
                ],
                [0.02 * centered, -0.015 * centered, 0.01 * time_index],
            )
            relative_left = pose(
                [
                    0.15 + 0.015 * centered,
                    -0.04 + 0.011 * centered,
                    0.03 + 0.009 * centered,
                ],
                [0.035 * centered, -0.03 * centered, 0.025 * centered],
            )
            left_pose = pose_compose(right_pose, relative_left)
            right_ee.append(right_pose)
            left_ee.append(left_pose)
            # Each commanded stream depends exactly on the other arm. The relative
            # measured end-effector pose varies across demonstrations, preventing
            # Eq. (5) from mistaking this coordination parameter for a rigid link.
            left_action.append(
                pose_compose(
                    right_pose,
                    pose(
                        [0.12 + 0.01 * time_index, 0.02, 0.05],
                        [0.01 * time_index, 0.0, 0.02],
                    ),
                )
            )
            right_action.append(
                pose_compose(
                    left_pose,
                    pose(
                        [-0.1 - 0.008 * time_index, -0.01, 0.04],
                        [0.0, -0.01 * time_index, -0.015],
                    ),
                )
            )
        labels = np.zeros(duration, dtype=np.int64)
        left_demonstrations.append(
            DynaMACDemonstration(
                np.stack(left_ee),
                np.stack(left_action),
                np.zeros(duration),
                {"right_ee": stale_cross_frames, "world": identity_frames},
                labels,
                f"left_{demo_index}",
            )
        )
        right_demonstrations.append(
            DynaMACDemonstration(
                np.stack(right_ee),
                np.stack(right_action),
                np.zeros(duration),
                {"left_ee": stale_cross_frames, "world": identity_frames},
                labels,
                f"right_{demo_index}",
            )
        )
    return left_demonstrations, right_demonstrations


def test_bimanual_policies_condition_both_ways_on_same_timestep_snapshot() -> None:
    left_demos, right_demos = bimanual_demonstrations()
    config = DynaMACConfig(
        tau_m=1.0e-12,
        tau_omega=0.2,
        policy_model="action_pose",
        eq5_rotation_weight=1.0,
        link_mask_scope="timestep",
        position_variance_floor=1.0e-8,
        rotation_variance_floor=1.0e-8,
        maximum_modes=1,
        clustering_length=6,
        resampling_method="interpolate",
        default_mode_strategy="map",
    )
    policy = BimanualDynaMAC(config=config).fit(left_demos, right_demos)
    # Remove and reinsert the caller's stale peer key to replace its value and fix
    # the PoE ordering.
    assert policy.left.frame_names == ("world", "right_ee")
    assert policy.right.frame_names == ("world", "left_ee")
    assert policy.left.skills[0].selected_frames == ("right_ee",)
    assert policy.right.skills[0].selected_frames == ("left_ee",)

    identity = pose([0.0, 0.0, 0.0])
    left = DynaMACObservation(left_demos[3].ee_pose[0], {"world": identity})
    right = DynaMACObservation(right_demos[3].ee_pose[0], {"world": identity})
    policy.reset(left, right, mode_strategy="map")
    baseline = policy.act(left, right)

    moved_right = DynaMACObservation(
        pose_compose(pose([0.08, -0.03, 0.02], [0.03, 0.0, 0.0]), right.ee_pose),
        # Supply a deliberately stale copy; the synchronized wrapper must replace it
        # with ``moved_right.ee_pose``.
        {"world": identity, "left_ee": pose([99.0, 99.0, 99.0])},
    )
    policy.reset(left, moved_right, mode_strategy="map")
    right_perturbed = policy.act(left, moved_right)
    assert np.linalg.norm(right_perturbed.left.pose[:3] - baseline.left.pose[:3]) > 0.05

    moved_left = DynaMACObservation(
        pose_compose(pose([-0.07, 0.04, 0.01], [0.0, 0.025, 0.0]), left.ee_pose),
        {"world": identity, "right_ee": pose([-99.0, -99.0, -99.0])},
    )
    policy.reset(moved_left, right, mode_strategy="map")
    left_perturbed = policy.act(moved_left, right)
    assert np.linalg.norm(left_perturbed.right.pose[:3] - baseline.right.pose[:3]) > 0.05


def test_bimanual_policies_fit_independent_skill_sequences_and_hold_completed_arm() -> None:
    left_demos, original_right_demos = bimanual_demonstrations()
    right_demos = []
    # Vary the right-arm boundary across demonstrations.  Its two per-skill
    # int(mean) durations sum to five, whereas the independent left policy has
    # one six-step skill.  This exercises truly asynchronous completion.
    for demo_index, demonstration in enumerate(original_right_demos):
        boundary = 2 if demo_index % 2 == 0 else 3
        labels = np.zeros(len(demonstration.skill), dtype=np.int64)
        labels[boundary:] = 1
        right_demos.append(
            DynaMACDemonstration(
                demonstration.ee_pose,
                demonstration.action_pose,
                demonstration.gripper,
                demonstration.frames,
                labels,
                demonstration.name,
            )
        )

    policy = BimanualDynaMAC(
        config=DynaMACConfig(
            tau_omega=0.2,
            maximum_modes=1,
            clustering_length=6,
            resampling_method="interpolate",
            default_mode_strategy="map",
        )
    ).fit(left_demos, right_demos)
    assert policy.left.skill_sequence == (0,)
    assert policy.right.skill_sequence == (0, 1)
    assert sum(skill.duration for skill in policy.left.skills) == 6
    assert sum(skill.duration for skill in policy.right.skills) == 5

    # Each Algorithm-1 instance owns a separate virtual-frame namespace and
    # captures only its own EE at its own skill boundaries.  The right arm's
    # second boundary must not leak a ``virtual_skill_1`` candidate into the
    # one-skill left policy.
    assert "virtual_skill_1" not in policy.left.skills[0].selection_scores
    assert "virtual_skill_1" in policy.right.skills[1].selection_scores
    np.testing.assert_allclose(
        policy.left.training_audit["skills"][0]["virtual_frame_start_poses"][
            "virtual_skill_0"
        ],
        np.stack([demo.ee_pose[0] for demo in left_demos]),
    )
    np.testing.assert_allclose(
        policy.right.training_audit["skills"][0]["virtual_frame_start_poses"][
            "virtual_skill_0"
        ],
        np.stack([demo.ee_pose[0] for demo in right_demos]),
    )

    identity = pose([0.0, 0.0, 0.0])
    left = DynaMACObservation(left_demos[0].ee_pose[0], {"world": identity})
    right = DynaMACObservation(right_demos[0].ee_pose[0], {"world": identity})
    policy.reset(left, right, mode_strategy="map")
    actions = [policy.act(left, right) for _ in range(6)]

    assert "complete_hold" not in actions[4].right.diagnostics
    assert actions[5].right.diagnostics["complete_hold"] is True
    np.testing.assert_allclose(actions[5].right.pose, actions[4].right.pose)
    np.testing.assert_allclose(actions[5].right.gripper, actions[4].right.gripper)
    assert policy.complete


def test_bimanual_rejects_aliasing_the_same_policy_instance() -> None:
    shared = DynaMAC()
    with pytest.raises(ValueError, match="two independent"):
        BimanualDynaMAC(left=shared, right=shared)


def test_bimanual_derives_reproducible_independent_rng_substreams(tmp_path: Path) -> None:
    config = DynaMACConfig(
        random_seed=2608,
        maximum_modes=1,
        clustering_length=6,
        resampling_method="interpolate",
    )
    left_demos, right_demos = bimanual_demonstrations()
    first = BimanualDynaMAC(config=config).fit(left_demos, right_demos)
    second = BimanualDynaMAC(config=config).fit(left_demos, right_demos)

    assert first.left.config.random_seed != first.right.config.random_seed
    assert first.left.config.random_seed == second.left.config.random_seed
    assert first.right.config.random_seed == second.right.config.random_seed
    assert first.left._rng.bit_generator.state != first.right._rng.bit_generator.state
    assert first.left._rng.bit_generator.state == second.left._rng.bit_generator.state
    assert first.right._rng.bit_generator.state == second.right._rng.bit_generator.state

    first_left_draws = first.left._rng.random(16)
    first_right_draws = first.right._rng.random(16)
    np.testing.assert_allclose(first_left_draws, second.left._rng.random(16))
    np.testing.assert_allclose(first_right_draws, second.right._rng.random(16))
    assert not np.array_equal(first_left_draws, first_right_draws)

    left_path = tmp_path / "left.npz"
    right_path = tmp_path / "right.npz"
    first.left.save(left_path)
    first.right.save(right_path)
    restored = BimanualDynaMAC(left=DynaMAC.load(left_path), right=DynaMAC.load(right_path))
    fresh = BimanualDynaMAC(config=config).fit(left_demos, right_demos)
    np.testing.assert_allclose(restored.left._rng.random(16), fresh.left._rng.random(16))
    np.testing.assert_allclose(restored.right._rng.random(16), fresh.right._rng.random(16))


def test_bimanual_rejects_explicitly_correlated_rng_streams() -> None:
    with pytest.raises(ValueError, match="same random stream"):
        BimanualDynaMAC(left=DynaMAC(), right=DynaMAC())


def test_bimanual_act_rolls_back_both_sides_when_one_prediction_fails() -> None:
    left_demos, right_demos = bimanual_demonstrations()
    policy = BimanualDynaMAC(
        config=DynaMACConfig(
            tau_omega=0.2,
            maximum_modes=1,
            clustering_length=6,
            resampling_method="interpolate",
            default_mode_strategy="map",
        )
    ).fit(left_demos, right_demos)
    identity = pose([0.0, 0.0, 0.0])
    left = DynaMACObservation(left_demos[3].ee_pose[0], {"world": identity})
    right = DynaMACObservation(right_demos[3].ee_pose[0], {"world": identity})
    policy.reset(left, right, mode_strategy="map")
    before = policy.left._capture_runtime_state()
    policy.right._complete = True

    with pytest.raises(RuntimeError, match="has completed"):
        policy.act(left, right)

    assert policy.left._skill_index == before["skill_index"]
    assert policy.left._time_index == before["time_index"]
    assert policy.left._complete == before["complete"]
    assert policy.right._complete
