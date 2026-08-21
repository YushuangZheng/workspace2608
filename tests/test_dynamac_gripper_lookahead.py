"""Tests for the observation-free, task-independent gripper lookahead API."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from essay2608.policy.dynamac import (
    BimanualDynaMAC,
    DynaMAC,
    DynaMACConfig,
    DynaMACObservation,
    SkillModel,
    StreamModel,
)


def _pose(x: float = 0.0) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _skill(
    label: int,
    gripper_by_mode: list[list[float]],
    *,
    transition: np.ndarray | None = None,
) -> SkillModel:
    gripper = np.asarray(gripper_by_mode, dtype=np.float64)[..., None]
    modes, duration, _ = gripper.shape
    means = np.repeat(_pose()[None, None, :], modes, axis=0)
    means = np.repeat(means, duration, axis=1)
    covariance = np.repeat((np.eye(6) * 1.0e-3)[None, None], modes, axis=0)
    covariance = np.repeat(covariance, duration, axis=1)
    return SkillModel(
        label=label,
        duration=duration,
        selected_frames=("world",),
        mode_priors=np.full(modes, 1.0 / modes),
        streams={"world": StreamModel("world", means, covariance)},
        gripper=gripper,
        transition_from_previous=transition,
        mode_demonstration_indices=tuple((index,) for index in range(modes)),
    )


def _policy(
    schedules: list[list[list[float]]],
    *,
    random_seed: int = 1,
) -> DynaMAC:
    policy = DynaMAC(DynaMACConfig(random_seed=random_seed))
    modes = len(schedules[0])
    policy.frame_names = ("world",)
    policy.skill_sequence = tuple(range(len(schedules)))
    policy.skills = [
        _skill(
            index,
            schedule,
            transition=(
                None
                if index == 0
                else np.full((modes, len(schedule)), 1.0 / len(schedule))
            ),
        )
        for index, schedule in enumerate(schedules)
    ]
    return policy


def _observation() -> DynaMACObservation:
    return DynaMACObservation(_pose(), {"world": _pose()})


def _assert_runtime_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if key == "virtual_frames":
            assert left[key].keys() == right[key].keys()
            for name in left[key]:
                np.testing.assert_array_equal(left[key][name], right[key][name])
        elif key == "mode_evidence":
            assert len(left[key]) == len(right[key])
            for first, second in zip(left[key], right[key], strict=True):
                np.testing.assert_array_equal(first, second)
        else:
            assert left[key] == right[key]


def test_preview_uses_selected_mode_path_and_marks_only_real_skill_boundary() -> None:
    policy = _policy(
        [
            [[1.0, 1.0], [-1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]
    )
    policy.reset(
        _observation(),
        mode_strategy="map",
        mode_evidence=(np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0])),
    )
    assert policy._mode_path == (1, 0)

    before = policy._capture_runtime_state()
    internal = policy.preview_next_gripper()
    after = policy._capture_runtime_state()
    _assert_runtime_equal(before, after)
    np.testing.assert_array_equal(internal.gripper, [-1.0])
    assert internal.crosses_skill_boundary is False
    assert internal.repeats_terminal is False
    assert (internal.next_skill_index, internal.next_time_index, internal.next_mode) == (
        0,
        1,
        1,
    )

    policy.act(_observation())
    boundary = policy.preview_next_gripper()
    np.testing.assert_array_equal(boundary.gripper, [1.0])
    assert boundary.crosses_skill_boundary is True
    assert boundary.repeats_terminal is False
    assert boundary.next_skill_label == 1
    assert (boundary.next_skill_index, boundary.next_time_index, boundary.next_mode) == (
        1,
        0,
        0,
    )


def test_preview_repeats_final_command_before_and_after_policy_completion() -> None:
    policy = _policy([[[1.0, -1.0]]])
    policy.reset(_observation(), mode_strategy="map")

    first = policy.preview_next_gripper()
    np.testing.assert_array_equal(first.gripper, [-1.0])
    assert first.repeats_terminal is False
    policy.act(_observation())

    last = policy.preview_next_gripper()
    np.testing.assert_array_equal(last.gripper, [-1.0])
    assert last.repeats_terminal is True
    assert last.crosses_skill_boundary is False
    policy.act(_observation())
    assert policy.complete

    completed = policy.preview_next_gripper()
    np.testing.assert_array_equal(completed.gripper, [-1.0])
    assert completed.repeats_terminal is True
    assert completed.next_time_index == 1


def test_preview_and_transaction_rollback_reproduce_the_same_command() -> None:
    policy = _policy([[[1.0, -1.0, 1.0]]])
    policy.reset(_observation(), mode_strategy="map")
    transaction = policy._capture_runtime_state()
    first = policy.preview_next_gripper()
    policy.act(_observation())
    policy._restore_runtime_state(deepcopy(transaction))

    second = policy.preview_next_gripper()
    np.testing.assert_array_equal(second.gripper, first.gripper)
    assert second == first
    _assert_runtime_equal(policy._capture_runtime_state(), transaction)


def test_bimanual_preview_holds_completed_arm_while_peer_keeps_its_own_clock() -> None:
    left = _policy([[[1.0, -1.0]]], random_seed=11)
    right = _policy([[[1.0, 1.0, -1.0, 1.0]]], random_seed=22)
    policy = BimanualDynaMAC(left=left, right=right)
    observation = _observation()
    policy.reset(observation, observation, mode_strategy="map")
    policy.act(observation, observation)
    policy.act(observation, observation)
    assert policy.left.complete
    assert not policy.right.complete

    left_before = policy.left._capture_runtime_state()
    right_before = policy.right._capture_runtime_state()
    preview = policy.preview_next_gripper()
    _assert_runtime_equal(policy.left._capture_runtime_state(), left_before)
    _assert_runtime_equal(policy.right._capture_runtime_state(), right_before)

    np.testing.assert_array_equal(preview.left.gripper, [-1.0])
    assert preview.left.repeats_terminal is True
    np.testing.assert_array_equal(preview.right.gripper, [1.0])
    assert preview.right.repeats_terminal is False
    assert preview.right.next_time_index == 3


def test_preview_requires_fitted_reset_policy_but_never_an_observation() -> None:
    unfitted = DynaMAC()
    with pytest.raises(RuntimeError, match="尚未拟合"):
        unfitted.preview_next_gripper()

    fitted = _policy([[[1.0, -1.0]]])
    with pytest.raises(RuntimeError, match="尚未 reset"):
        fitted.preview_next_gripper()
    fitted.reset(_observation(), mode_strategy="map")
    assert fitted.preview_next_gripper().next_time_index == 1
