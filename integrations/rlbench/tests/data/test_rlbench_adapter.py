from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from essay2608.policy.dynamac import DynaMACDemonstration
from essay2608.policy.tapas_segmentation import (
    TAPASSegmentationConfig,
    _single_grasp_contact_cycle_subset,
    align_tapas_boundaries,
    gripper_change_boundaries,
    segment_bimanual_trajectories,
    segment_trajectories,
    translation_direction_reversal_boundaries,
)

from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    DYNAMAC_GRIPPER_TARGET_TIMING,
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
    DYNAMAC_CURRENT_STATE_TIMING,
    TAPAS_DEFAULT_CONFIG_PATH,
    current_gripper_state,
    load_rlbench_segmentation_config,
)
from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
    TAPASSegmentationConfig as RLBenchTAPASSegmentationConfig,
)
from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
    segment_trajectories as rlbench_segment_trajectories,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    CANDIDATE_FRAME_POLICY,
    CANDIDATE_FRAME_POLICY_SOURCE_STATUS,
    get_task_spec,
)

TABLE_I_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "training"
    / "main"
)


def _positions(pauses: set[int], *, steps: int = 24, offset: float = 0.0) -> list[float]:
    values = [offset]
    for action_index in range(steps - 1):
        delta = 0.0 if action_index in pauses else 0.1
        values.append(values[-1] + delta)
    return values


def _core_trajectory(pauses: set[int], *, offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [[x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for x in _positions(pauses, offset=offset)],
        dtype=np.float64,
    )


def _xyzw_pose(x: float) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _task_state(frame_count: int, *, sample: int) -> np.ndarray:
    return np.concatenate([_xyzw_pose(20.0 + frame + sample / 100.0) for frame in range(frame_count)])


def _bimanual_episode(
    *,
    frame_count: int,
    left_pauses: set[int],
    right_pauses: set[int],
) -> list[SimpleNamespace]:
    left = _positions(left_pauses)
    right = _positions(right_pauses, offset=10.0)
    return [
        SimpleNamespace(
            left=SimpleNamespace(gripper_pose=_xyzw_pose(left[index]), gripper_open=1.0),
            right=SimpleNamespace(gripper_pose=_xyzw_pose(right[index]), gripper_open=1.0),
            task_low_dim_state=(_task_state(frame_count, sample=index),),
        )
        for index in range(len(left))
    ]


def _union_config() -> TAPASSegmentationConfig:
    return TAPASSegmentationConfig(
        distance_based=False,
        velocity_based=True,
        gripper_based=True,
        min_end_distance=1,
        max_idx_distance=1,
        min_len=1,
        repeat_final_step=0,
    )


def test_default_config_is_author_clarified_velocity_gripper_union() -> None:
    raw = json.loads(Path(TAPAS_DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    config = TAPASSegmentationConfig.from_mapping(raw)

    assert config.strategy == "velocity_gripper_union"
    assert config.velocity_based is True
    assert config.gripper_based is True
    assert config.distance_based is False
    assert raw["evidence"]["boundary_signals"].startswith("AUTHOR_EMAIL_EXPLICIT")
    assert raw["evidence"]["cross_demo_alignment"].endswith("20260814")

    store = config.for_task("bimanual_put_bottle_in_fridge")
    assert store.boundary_selection == "gripper_preferred_temporal_consensus"
    assert store.expected_boundary_count == 2
    assert store.provenance["active_task_profile_source_status"].startswith(
        "AUTHOR_ATTACHMENTS"
    )

    wipe = config.for_task("wipe_desk")
    assert wipe.boundary_selection == "temporal_consensus_require_gripper"
    assert wipe.expected_boundary_count == 8
    assert wipe.velocity_threshold == 0.002
    assert wipe.direction_reversal_based is True
    assert wipe.direction_reversal_threshold_degrees == 120.0
    assert wipe.candidate_merge_fraction == 0.04
    assert wipe.provenance["active_task_profile_source_status"].startswith(
        "CONTINUOUS_CARTESIAN_PATH"
    )

    handover = config.for_task("bimanual_handover_item")
    assert handover.expected_boundary_count == 6
    assert handover.candidate_merge_fraction == 0.05
    assert handover.velocity_threshold == 0.001

    place_cups = config.for_task("place_cups")
    assert place_cups.expected_boundary_count == 4
    assert place_cups.boundary_selection == "single_grasp_contact_cycle"
    assert place_cups.provenance["active_task_profile_source_status"] == (
        "LOCAL_CONTACT_PHASE_ALIGNMENT_VALIDATED_20260815"
    )

    for task_name, count in (
        ("bimanual_lift_tray", 2),
        ("bimanual_sweep_to_dustpan", 4),
    ):
        local = config.for_task(task_name)
        assert local.expected_boundary_count == count
        assert local.boundary_selection == "temporal_consensus"
        assert local.provenance["active_task_profile_source_status"] == (
            "LOCAL_EXPLICIT_DEFAULT_PENDING_AUTHOR_PLOT"
        )


def test_rlbench_segmentation_module_is_a_thin_core_compatibility_facade() -> None:
    assert rlbench_segment_trajectories is segment_trajectories
    loaded = load_rlbench_segmentation_config()
    assert isinstance(loaded, TAPASSegmentationConfig)
    assert isinstance(loaded, RLBenchTAPASSegmentationConfig)
    assert loaded.to_dict() == TAPASSegmentationConfig.from_json(
        TAPAS_DEFAULT_CONFIG_PATH
    ).to_dict()
    assert RLBenchTAPASSegmentationConfig.from_json().to_dict() == loaded.to_dict()
    with pytest.raises(FrozenInstanceError):
        loaded.unexpected_field = True


def test_current_gripper_target_uses_current_state_at_the_same_pose_sample() -> None:
    states = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0])
    assert gripper_change_boundaries(states, min_end_distance=0) == (2, 4)
    np.testing.assert_array_equal(
        current_gripper_state(states),
        np.asarray([1.0, 1.0, -1.0, -1.0, 1.0]),
    )

    episode = [
        SimpleNamespace(
            gripper_pose=_xyzw_pose(float(index)),
            gripper_open=float(state),
            task_low_dim_state=(_task_state(2, sample=index),),
        )
        for index, state in enumerate(states)
    ]
    segmentation = align_tapas_boundaries(((),), (len(states),))
    result = make_unimanual_demonstrations(
        [episode],
        "stack_wine",
        segmentation=segmentation,
    )

    np.testing.assert_array_equal(
        result.demonstrations[0].gripper[:, 0],
        np.asarray([1.0, 1.0, -1.0, -1.0, 1.0]),
    )
    assert result.audit["schema"] == "rlbench-dynamac-demo-adapter-v3"
    assert result.audit["pose_and_gripper_sample_aligned"] is True
    assert result.audit["gripper_action_timing"] == DYNAMAC_GRIPPER_TARGET_TIMING
    assert result.audit["action_timing"] == DYNAMAC_CURRENT_STATE_TIMING


def test_wipe_desk_uses_standard_physical_task_frames() -> None:
    spec = get_task_spec("wipe_desk")
    assert spec.frame_names == ("sponge", "dirt_boundary")
    assert spec.action_frame_names == spec.frame_names
    assert spec.scene_entity_names == ()
    assert spec.configuration_schema == {}
    assert spec.structural_bindings == {}
    states = [
        np.concatenate(
            (
                _xyzw_pose(float(index)),
                _xyzw_pose(10.0),
            )
        )
        for index in range(4)
    ]
    episode = [
        SimpleNamespace(
            gripper_pose=_xyzw_pose(float(index)),
            gripper_open=1.0,
            task_low_dim_state=(state,),
        )
        for index, state in enumerate(states)
    ]
    result = make_unimanual_demonstrations(
        [episode],
        spec,
        segmentation=align_tapas_boundaries(((),), (len(episode),)),
    )
    demo = result.demonstrations[0]
    assert tuple(demo.frames) == spec.action_frame_names
    assert tuple(demo.frames) == ("sponge", "dirt_boundary")
    assert tuple(demo.scene_entity_poses) == ()
    assert demo.structural_bindings == {}
    assert demo.entity_configurations == {}


def test_velocity_and_gripper_change_candidates_are_unioned() -> None:
    gripper = np.ones(24, dtype=np.float64)
    gripper[10:14] = 0.0

    result = segment_trajectories(
        [_core_trajectory({5})],
        gripper_states=[gripper],
        config=_union_config(),
    )

    assert result.boundaries == ((5, 10, 14),)
    assert result.audit["boundary_components"] == {
        "ee_translation_velocity": [[5]],
        "gripper_change": [[10, 14]],
    }
    assert result.audit["no_boundary_truncation"] is True


def test_direction_reversal_candidates_find_continuous_fold_without_stop() -> None:
    x = np.concatenate(
        [
            np.linspace(0.0, 1.0, 16, endpoint=False),
            np.linspace(1.0, 0.0, 17),
        ]
    )
    poses = np.asarray(
        [[value, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for value in x],
        dtype=np.float64,
    )

    assert translation_direction_reversal_boundaries(
        poses,
        angle_threshold_degrees=120.0,
        window=3,
        max_idx_distance=2,
        min_end_distance=2,
        min_chord_length=0.01,
    ) == (16,)


def test_direction_reversal_signal_is_generic_and_audited_as_component() -> None:
    x = np.concatenate(
        [
            np.linspace(0.0, 1.0, 16, endpoint=False),
            np.linspace(1.0, 0.0, 17),
        ]
    )
    poses = np.asarray(
        [[value, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for value in x],
        dtype=np.float64,
    )
    config = TAPASSegmentationConfig(
        distance_based=False,
        velocity_based=False,
        gripper_based=False,
        direction_reversal_based=True,
        direction_reversal_threshold_degrees=120.0,
        velocity_threshold=0.01,
        max_idx_distance=2,
        min_end_distance=2,
    )

    result = segment_trajectories([poses], config=config)

    assert result.boundaries == ((16,),)
    assert result.boundary_components["ee_translation_direction_reversal"] == (
        (16,),
    )


def test_cross_demo_boundary_count_mismatch_never_prefix_truncates() -> None:
    with pytest.raises(ValueError, match="forbids prefix/min-count truncation"):
        align_tapas_boundaries(
            ((5,), (5, 10)),
            (24, 24),
            truncate_to_minimum=True,
        )

    with pytest.raises(ValueError, match="forbids prefix/min-count truncation"):
        segment_trajectories(
            [_core_trajectory({5}), _core_trajectory({5, 10})],
            gripper_states=[np.ones(24), np.ones(24)],
            config=_union_config(),
        )


def test_place_cups_real_five_demo_contact_phases_are_aligned() -> None:
    episode_root = TABLE_I_DATA_ROOT / "place_cups" / "all_variations" / "episodes"
    paths = [episode_root / f"episode{index}" / "low_dim_obs.pkl" for index in range(5)]
    if not all(path.is_file() for path in paths):
        pytest.skip("local PlaceCups five-demo cohort is not installed")

    result = make_unimanual_demonstrations(
        load_low_dim_obs_pickles(paths),
        "place_cups",
        names=[f"episode{index}" for index in range(5)],
    )

    assert result.segmentation.boundaries == (
        (62, 80, 97, 165),
        (58, 71, 82, 137),
        (59, 81, 103, 179),
        (55, 69, 83, 172),
        (63, 77, 89, 152),
    )
    gripper_changes = result.segmentation.boundary_components["gripper_change"]
    for boundaries, changes in zip(
        result.segmentation.boundaries, gripper_changes, strict=True
    ):
        assert changes == (boundaries[1],)
        assert boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]


def test_single_grasp_contact_cycle_discards_only_internal_transport_stops() -> None:
    assert _single_grasp_contact_cycle_subset(
        (5, 10, 14, 18, 24),
        gripper_row=(10,),
        expected=4,
    ) == (5, 10, 14, 24)


@pytest.mark.parametrize(
    ("row", "gripper_row", "expected", "message"),
    [
        ((5, 10, 14, 24), (), 4, "exactly one retained gripper change"),
        ((5, 10, 14, 20, 24), (10, 20), 4, "exactly one retained gripper change"),
        ((5, 10, 24), (10,), 4, "at least two velocity stops after contact"),
        ((5, 10, 14, 24), (10,), 3, "exactly four boundaries"),
    ],
)
def test_single_grasp_contact_cycle_fails_closed_on_other_phase_structures(
    row: tuple[int, ...],
    gripper_row: tuple[int, ...],
    expected: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _single_grasp_contact_cycle_subset(
            row,
            gripper_row=gripper_row,
            expected=expected,
        )


def test_shared_union_applies_each_demo_arm_union_to_both_arms() -> None:
    result = segment_bimanual_trajectories(
        [_core_trajectory({5})],
        [_core_trajectory({9}, offset=10.0)],
        left_gripper_states=[np.ones(24)],
        right_gripper_states=[np.ones(24)],
        config=_union_config(),
        coordination="shared_union",
        coordination_source_status="AUTHOR_EMAIL_EXPLICIT_HANDOVER_SHARED_UNION_20260814",
    )

    assert result.left.boundaries == ((5, 9),)
    assert result.right.boundaries == ((5, 9),)
    assert result.audit["shared_boundaries_between_arms"] is True
    assert result.audit["pre_coordination_boundaries"] == {
        "left": [[5]],
        "right": [[9]],
    }


def test_task_registry_freezes_candidate_and_bimanual_coordination_provenance() -> None:
    for task_name in (
        "bimanual_put_bottle_in_fridge",
        "bimanual_handover_item",
        "bimanual_lift_tray",
        "bimanual_sweep_to_dustpan",
    ):
        spec = get_task_spec(task_name)
        assert spec.candidate_frame_policy == CANDIDATE_FRAME_POLICY
        assert spec.candidate_frame_policy_source_status == (
            CANDIDATE_FRAME_POLICY_SOURCE_STATUS
        )

    store = get_task_spec("bimanual_put_bottle_in_fridge")
    handover = get_task_spec("bimanual_handover_item")
    assert store.segmentation_coordination == "independent"
    assert store.segmentation_coordination_source_status.startswith("AUTHOR_EMAIL_EXPLICIT")
    assert handover.segmentation_coordination == "shared_union"
    for name in ("bimanual_lift_tray", "bimanual_sweep_to_dustpan"):
        inferred = get_task_spec(name)
        assert inferred.segmentation_coordination == "shared_union"
        assert inferred.segmentation_coordination_source_status == (
            "LOCAL_INTERACTION_INFERENCE_REQUIRES_DEBUG_PLOTS"
        )
        assert inferred.segmentation_debug_plots_required is True


def test_demo_adapter_routes_store_independent_and_handover_shared_union() -> None:
    config = _union_config()
    store = make_bimanual_demonstrations(
        [
            _bimanual_episode(
                frame_count=2,
                left_pauses={5},
                right_pauses={9},
            )
        ],
        "bimanual_put_bottle_in_fridge",
        config=config,
    )
    assert store.segmentation.left.boundaries == ((5,),)
    assert store.segmentation.right.boundaries == ((9,),)
    assert store.segmentation.coordination == "independent"
    assert store.audit["task_segmentation_coordination"] == "independent"
    assert type(store.left_demonstrations[0]) is DynaMACDemonstration

    handover = make_bimanual_demonstrations(
        [
            _bimanual_episode(
                frame_count=5,
                left_pauses={5},
                right_pauses={9},
            )
        ],
        "bimanual_handover_item",
        config=config,
    )
    assert handover.segmentation.left.boundaries == ((5, 9),)
    assert handover.segmentation.right.boundaries == ((5, 9),)
    assert handover.segmentation.coordination == "shared_union"
    assert tuple(handover.left_demonstrations[0].frames) == (
        "item0",
        "item1",
        "item2",
        "item3",
        "item4",
    )
    np.testing.assert_array_equal(
        handover.left_demonstrations[0].action_pose,
        handover.left_demonstrations[0].ee_pose,
    )
    assert handover.audit["pose_target_timing"] == "time-state current EE pose from obs[t]"
    assert handover.audit["gripper_action_timing"] == DYNAMAC_GRIPPER_TARGET_TIMING
