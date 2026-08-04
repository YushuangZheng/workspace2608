from __future__ import annotations

import numpy as np
import pytest

from essay2608.data.handover_schema import HandoverState
from essay2608.policy.bimanual_recovery import (
    BimanualRecoveryConfig,
    BimanualRecoveryState,
    BimanualRecoveryTrigger,
    BimanualRelationRecoveryController,
)
from essay2608.policy.relation import RelationState


IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0])
LEFT = np.concatenate((np.asarray([0.40, 0.0, 0.30]), IDENTITY))
RIGHT = np.concatenate((np.asarray([0.60, 0.0, 0.30]), IDENTITY))
OBJECT = np.concatenate((np.asarray([0.50, 0.0, 0.302]), IDENTITY))
ACTION = np.concatenate((LEFT, [-1.0], RIGHT, [-1.0]))


def update(
    controller: BimanualRelationRecoveryController,
    *,
    task_state: HandoverState = HandoverState.RIGHT_GRASP,
    left_state: RelationState = RelationState.CONNECTED,
    right_state: RelationState = RelationState.DISCONNECTED,
    right_pose: np.ndarray = RIGHT,
    opening: float = 0.08,
):
    return controller.update(
        task_state=task_state,
        normal_action=ACTION,
        left_pose=LEFT,
        right_pose=right_pose,
        object_pose=OBJECT,
        right_gripper_opening_m=opening,
        left_relation_state=left_state,
        right_relation_state=right_state,
    )


def test_transfer_gate_waits_for_stable_receiver_connection() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            miss_verification_steps=10,
            minimum_verified_both_steps=3,
        )
    )
    decision = update(controller)
    assert decision.state == BimanualRecoveryState.NORMAL
    assert decision.transfer_gate_active and decision.pause_task_clock
    for _ in range(2):
        decision = update(controller, right_state=RelationState.CONNECTED, opening=0.06)
        assert decision.pause_task_clock
    decision = update(controller, right_state=RelationState.CONNECTED, opening=0.06)
    assert not decision.transfer_gate_active
    assert not decision.pause_task_clock


def test_receiver_miss_enters_bounded_recovery_without_releasing_giver() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(miss_verification_steps=2)
    )
    update(controller)
    decision = update(controller)
    assert decision.state == BimanualRecoveryState.RECEIVER_MISS_DETECTED
    assert decision.trigger == BimanualRecoveryTrigger.RECEIVER_MISS
    assert decision.action[7] < 0.0
    assert decision.action[15] > 0.0
    assert decision.pause_task_clock


def test_gate_only_waits_but_never_enters_recovery_graph() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            enable_recovery=False,
            miss_verification_steps=1,
        )
    )
    decision = update(controller)
    assert decision.state == BimanualRecoveryState.NORMAL
    assert decision.transfer_gate_active and decision.pause_task_clock
    assert decision.trigger == BimanualRecoveryTrigger.NONE


def test_transient_receiver_loss_cancels_before_retreat() -> None:
    controller = BimanualRelationRecoveryController()
    for _ in range(controller.config.minimum_verified_both_steps):
        update(controller, right_state=RelationState.CONNECTED, opening=0.06)
    decision = update(
        controller,
        task_state=HandoverState.TRANSFER,
        right_state=RelationState.CANDIDATE_LOST,
        opening=0.06,
    )
    assert decision.state == BimanualRecoveryState.RECEIVER_LOSS_DETECTED
    assert decision.trigger == BimanualRecoveryTrigger.RECEIVER_LOSS
    decision = update(
        controller,
        task_state=HandoverState.TRANSFER,
        right_state=RelationState.CONNECTED,
        opening=0.06,
    )
    assert decision.state == BimanualRecoveryState.NORMAL
    assert not decision.pause_task_clock


def test_miss_recovery_reapproaches_regrasps_and_verifies_both() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            miss_verification_steps=1,
            minimum_verified_both_steps=2,
        )
    )
    decision = update(controller)
    assert decision.state == BimanualRecoveryState.RECEIVER_MISS_DETECTED
    decision = update(controller)
    assert decision.state == BimanualRecoveryState.SAFE_RETREAT
    retreat_pose = RIGHT.copy()
    retreat_pose[:3] = controller.retreat_target
    decision = update(controller, right_pose=retreat_pose, opening=0.08)
    assert decision.state == BimanualRecoveryState.REAPPROACH

    pregrasp_pose = RIGHT.copy()
    pregrasp_pose[:3] = controller._object_site(
        OBJECT,
        tuple(
            np.asarray(controller.config.right_grasp_offset_xyz_m)
            + np.asarray(controller.config.right_pregrasp_clearance_xyz_m)
        ),
    )
    decision = update(controller, right_pose=pregrasp_pose, opening=0.08)
    assert decision.state == BimanualRecoveryState.REGRASP
    assert decision.regrasp_attempts == 1

    grasp_pose = RIGHT.copy()
    grasp_pose[:3] = controller._object_site(
        OBJECT,
        controller.config.right_grasp_offset_xyz_m,
    )
    decision = update(controller, right_pose=grasp_pose, opening=0.08)
    assert decision.state == BimanualRecoveryState.REGRASP
    assert decision.action[15] < 0.0
    decision = update(
        controller,
        right_pose=grasp_pose,
        opening=0.06,
        right_state=RelationState.CONNECTED,
    )
    assert decision.state == BimanualRecoveryState.VERIFY_BOTH
    for _ in range(2):
        decision = update(
            controller,
            right_pose=grasp_pose,
            opening=0.06,
            right_state=RelationState.CONNECTED,
        )
    assert decision.state == BimanualRecoveryState.RESUME_TASK
    decision = update(
        controller,
        right_pose=grasp_pose,
        opening=0.06,
        right_state=RelationState.CONNECTED,
    )
    assert decision.state == BimanualRecoveryState.NORMAL
    assert not decision.pause_task_clock


def test_verify_requires_both_edges_not_only_receiver() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(minimum_verified_both_steps=2)
    )
    controller.state = BimanualRecoveryState.VERIFY_BOTH
    decision = update(
        controller,
        left_state=RelationState.DISCONNECTED,
        right_state=RelationState.CONNECTED,
        opening=0.06,
    )
    assert decision.state == BimanualRecoveryState.VERIFY_BOTH
    assert decision.verified_both_steps == 0


def test_recovery_stops_at_global_bound() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            miss_verification_steps=1,
            maximum_recovery_steps=1,
            maximum_state_steps=10,
        )
    )
    update(controller)
    update(controller)
    decision = update(controller)
    assert decision.state == BimanualRecoveryState.RECOVERY_FAILED
    assert controller.failed


def test_post_release_loss_verifies_receiver_without_requiring_giver() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            loss_confirmation_steps=1,
            minimum_verified_both_steps=2,
        )
    )
    decision = update(
        controller,
        task_state=HandoverState.RIGHT_TO_TARGET,
        left_state=RelationState.DISCONNECTED,
        right_state=RelationState.CANDIDATE_LOST,
    )
    assert decision.state == BimanualRecoveryState.RECEIVER_LOSS_DETECTED
    assert not decision.requires_giver_connection
    update(
        controller,
        task_state=HandoverState.RIGHT_TO_TARGET,
        left_state=RelationState.DISCONNECTED,
        right_state=RelationState.DISCONNECTED,
    )
    controller.state = BimanualRecoveryState.REGRASP
    controller.regrasp_closing = True
    decision = update(
        controller,
        task_state=HandoverState.RIGHT_TO_TARGET,
        left_state=RelationState.DISCONNECTED,
        right_state=RelationState.CONNECTED,
        opening=0.06,
    )
    assert decision.state == BimanualRecoveryState.VERIFY_RECEIVER
    for _ in range(2):
        decision = update(
            controller,
            task_state=HandoverState.RIGHT_TO_TARGET,
            left_state=RelationState.DISCONNECTED,
            right_state=RelationState.CONNECTED,
            opening=0.06,
        )
    assert decision.state == BimanualRecoveryState.RESUME_TASK


def test_regrasp_timeout_uses_second_bounded_attempt() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            maximum_regrasp_attempts=2,
            maximum_state_steps=1,
        )
    )
    controller.state = BimanualRecoveryState.REGRASP
    controller.regrasp_attempts = 1
    controller.regrasp_closing = True
    decision = update(controller)
    assert decision.state == BimanualRecoveryState.SAFE_RETREAT
    assert not controller.failed


def test_recovery_target_is_bounded_relative_to_measured_pose() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            miss_verification_steps=1,
            maximum_arm_target_step_m=0.03,
        )
    )
    update(controller)
    decision = update(controller)
    assert np.linalg.norm(decision.action[8:11] - RIGHT[:3]) <= 0.03 + 1e-12


def test_recovery_output_limits_state_transition_target_jump() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(
            miss_verification_steps=2,
            maximum_arm_target_step_m=0.03,
        )
    )
    normal_action = ACTION.copy()
    normal_action[8:11] = RIGHT[:3] - np.asarray([0.10, 0.0, 0.0])
    first = controller.update(
        task_state=HandoverState.RIGHT_GRASP,
        normal_action=normal_action,
        left_pose=LEFT,
        right_pose=RIGHT,
        object_pose=OBJECT,
        right_gripper_opening_m=0.08,
        left_relation_state=RelationState.CONNECTED,
        right_relation_state=RelationState.DISCONNECTED,
    )
    second = controller.update(
        task_state=HandoverState.RIGHT_GRASP,
        normal_action=normal_action,
        left_pose=LEFT,
        right_pose=RIGHT,
        object_pose=OBJECT,
        right_gripper_opening_m=0.08,
        left_relation_state=RelationState.CONNECTED,
        right_relation_state=RelationState.DISCONNECTED,
    )
    assert second.action_overridden
    assert np.linalg.norm(second.action[8:11] - first.action[8:11]) <= 0.03 + 1e-12


def test_slew_limit_remains_active_until_normal_target_is_rejoined() -> None:
    controller = BimanualRelationRecoveryController(
        BimanualRecoveryConfig(maximum_arm_target_step_m=0.03)
    )
    controller.state = BimanualRecoveryState.RESUME_TASK
    controller.slew_limiter_active = True
    controller.last_left_target = LEFT[:3].copy()
    controller.last_right_target = RIGHT[:3].copy()
    far_action = ACTION.copy()
    far_action[:3] += np.asarray([0.10, 0.0, 0.0])
    first = controller.update(
        task_state=HandoverState.RIGHT_TO_TARGET,
        normal_action=far_action,
        left_pose=LEFT,
        right_pose=RIGHT,
        object_pose=OBJECT,
        right_gripper_opening_m=0.06,
        left_relation_state=RelationState.DISCONNECTED,
        right_relation_state=RelationState.CONNECTED,
    )
    assert first.state == BimanualRecoveryState.NORMAL
    assert first.action_overridden
    assert np.linalg.norm(first.action[:3] - LEFT[:3]) <= 0.03 + 1e-12
    second = controller.update(
        task_state=HandoverState.RIGHT_TO_TARGET,
        normal_action=far_action,
        left_pose=LEFT,
        right_pose=RIGHT,
        object_pose=OBJECT,
        right_gripper_opening_m=0.06,
        left_relation_state=RelationState.DISCONNECTED,
        right_relation_state=RelationState.CONNECTED,
    )
    assert second.action_overridden
    assert np.linalg.norm(second.action[:3] - first.action[:3]) <= 0.03 + 1e-12


def test_recovery_config_rejects_nonpositive_or_malformed_values() -> None:
    with pytest.raises(ValueError, match="必须为正数"):
        BimanualRecoveryConfig(miss_verification_steps=0)
    with pytest.raises(ValueError, match="三维向量"):
        BimanualRecoveryConfig(right_grasp_offset_xyz_m=(0.1, 0.0))
