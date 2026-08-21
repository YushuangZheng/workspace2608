from __future__ import annotations

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.core import gripper_timing


def test_global_boundary_rule_advances_transition_and_preserves_other_9d_scalars():
    action = np.arange(9, dtype=np.float64)
    action[7] = 1.0

    emitted = gripper_timing.apply_global_gripper_timing(
        action,
        next_wire_gripper_by_index={7: 0.0},
        crosses_skill_boundary_by_index={7: True},
    )

    assert emitted[7] == 0.0
    keep = [index for index in range(9) if index != 7]
    np.testing.assert_array_equal(emitted[keep], action[keep])


def test_global_boundary_rule_does_not_advance_internal_transition():
    action = np.arange(9, dtype=np.float64)
    action[7] = 1.0

    emitted = gripper_timing.apply_global_gripper_timing(
        action,
        next_wire_gripper_by_index={7: 0.0},
        crosses_skill_boundary_by_index={7: False},
    )

    np.testing.assert_array_equal(emitted, action)


def test_bimanual_arms_use_independent_boundaries_in_right_first_layout():
    action = np.arange(18, dtype=np.float64)
    action[7] = 1.0
    action[16] = 0.0

    emitted = gripper_timing.apply_global_gripper_timing(
        action,
        next_wire_gripper_by_index={7: 0.0, 16: 1.0},
        crosses_skill_boundary_by_index={7: True, 16: False},
    )

    assert (emitted[7], emitted[16]) == (0.0, 0.0)
    keep = [index for index in range(18) if index not in {7, 16}]
    np.testing.assert_array_equal(emitted[keep], action[keep])


def test_terminal_same_command_is_a_noop_even_when_boundary_flag_is_true():
    action = np.zeros(9, dtype=np.float64)
    emitted = gripper_timing.apply_global_gripper_timing(
        action,
        next_wire_gripper_by_index={7: 0.0},
        crosses_skill_boundary_by_index={7: True},
    )
    np.testing.assert_array_equal(emitted, action)


def test_global_protocol_identity_forbids_task_specific_adaptation():
    metadata = gripper_timing.global_gripper_timing_metadata()
    assert metadata["rule"] == "skill_boundary_transition_lookahead"
    assert metadata["task_specific_branches"] is False
    assert metadata["task_name_or_tick_special_cases"] is False
    assert metadata["pose_predictions_per_policy_tick"] == 1
    assert metadata["training_labels_modified"] is False


@pytest.mark.parametrize(
    ("next_commands", "boundaries", "error"),
    [
        ({}, {7: True}, ValueError),
        ({7: 0.25}, {7: True}, ValueError),
        ({7: 0.0}, {7: 1}, TypeError),
    ],
)
def test_global_protocol_fails_closed_on_invalid_lookahead(
    next_commands, boundaries, error
):
    with pytest.raises(error):
        gripper_timing.apply_global_gripper_timing(
            np.zeros(9),
            next_wire_gripper_by_index=next_commands,
            crosses_skill_boundary_by_index=boundaries,
        )


def test_native_gripper_conversion_matches_checkpoint_signed_encoding():
    assert gripper_timing.native_gripper_to_wire([1.0]) == 1.0
    assert gripper_timing.native_gripper_to_wire([-1.0]) == 0.0
    assert gripper_timing.native_gripper_to_wire([0.0]) == 0.0
