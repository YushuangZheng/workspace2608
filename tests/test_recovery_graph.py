from __future__ import annotations

import numpy as np

from essay2608.policy.dynamac import OracleRelationRecoveryPolicy
from essay2608.policy.base import PolicyObservation
from essay2608.policy.recovery import (
    RecoveryConfig,
    RecoveryState,
    RecoveryTrigger,
    RelationRecoveryController,
    privileged_grasp_relation,
)
from essay2608.policy.relation import RelationEstimate, RelationState


def observation(
    ee=(0.0, 0.0, 0.10),
    obj=(0.30, 0.0, 0.02),
    opening=0.04,
    contact=False,
) -> PolicyObservation:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    return PolicyObservation(
        ee_pose=np.concatenate((np.asarray(ee, dtype=np.float64), identity)),
        object_pose=np.concatenate((np.asarray(obj, dtype=np.float64), identity)),
        target_pose=np.concatenate((np.asarray([0.55, 0.2, 0.08]), identity)),
        gripper_opening_m=opening,
        gripper_velocity_m_s=0.0,
        object_contact=contact,
    )


def estimate(state: RelationState) -> RelationEstimate:
    return RelationEstimate(
        state=state,
        connected=state in {RelationState.CONNECTED, RelationState.CANDIDATE_LOST},
        confidence=1.0 if state == RelationState.CONNECTED else 0.0,
        connection_score=0.0,
        loss_score=0.0,
        features={},
        transitioned=False,
    )


NORMAL_ACTION = np.asarray([0.4, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0, -1.0])


def advance_to_verify(controller: RelationRecoveryController) -> None:
    disconnected = estimate(RelationState.DISCONNECTED)
    controller.update(observation(), disconnected, 4, NORMAL_ACTION)
    decision = controller.update(observation(), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.MISS_DETECTED
    decision = controller.update(observation(), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.SAFE_RETREAT
    decision = controller.update(observation(ee=(0.0, 0.0, 0.14), opening=0.08), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.RELOCALIZE
    decision = controller.update(observation(ee=(0.30, 0.0, 0.14), opening=0.08), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.REAPPROACH
    decision = controller.update(observation(ee=(0.30, 0.0, 0.085), opening=0.08), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.REGRASP
    grasp = np.asarray((0.30, 0.0, 0.02)) + np.asarray(controller.config.grasp_offset_xyz_m)
    decision = controller.update(observation(ee=grasp, opening=0.08), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.REGRASP
    assert decision.action[7] < 0.0
    decision = controller.update(observation(ee=grasp, opening=0.04), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.VERIFY_GRASP


def test_miss_recovery_regrasp_verify_and_resume() -> None:
    controller = RelationRecoveryController(RecoveryConfig(miss_verification_steps=2))
    advance_to_verify(controller)
    decision = controller.update(observation(), estimate(RelationState.CONNECTED), 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.RESUME_TASK
    assert decision.resume_phase == 4
    assert decision.regrasp_attempts == 1
    decision = controller.update(observation(), estimate(RelationState.CONNECTED), 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.NORMAL
    assert not decision.pause_task_clock


def test_loss_freezes_then_confirms_safe_retreat() -> None:
    controller = RelationRecoveryController()
    controller.update(observation(), estimate(RelationState.CONNECTED), 5, NORMAL_ACTION)
    decision = controller.update(observation(), estimate(RelationState.CANDIDATE_LOST), 5, NORMAL_ACTION)
    assert decision.state == RecoveryState.LOSS_DETECTED
    assert decision.trigger == RecoveryTrigger.LOSS
    assert np.allclose(decision.action[:3], observation().ee_pose[:3])
    assert decision.pause_task_clock
    decision = controller.update(observation(), estimate(RelationState.DISCONNECTED), 5, NORMAL_ACTION)
    assert decision.state == RecoveryState.SAFE_RETREAT


def test_oracle_direct_disconnect_triggers_loss_without_future_state() -> None:
    controller = RelationRecoveryController()
    controller.update(observation(), estimate(RelationState.CONNECTED), 5, NORMAL_ACTION)
    decision = controller.update(observation(), estimate(RelationState.DISCONNECTED), 5, NORMAL_ACTION)
    assert decision.state == RecoveryState.LOSS_DETECTED
    assert decision.trigger == RecoveryTrigger.LOSS


def test_transient_loss_candidate_cancels_without_recovery_motion() -> None:
    controller = RelationRecoveryController()
    controller.update(observation(), estimate(RelationState.CONNECTED), 5, NORMAL_ACTION)
    controller.update(observation(), estimate(RelationState.CANDIDATE_LOST), 5, NORMAL_ACTION)
    decision = controller.update(observation(), estimate(RelationState.CONNECTED), 5, NORMAL_ACTION)
    assert decision.state == RecoveryState.NORMAL
    assert decision.trigger == RecoveryTrigger.NONE
    assert not decision.action_overridden


def test_normal_release_phase_does_not_trigger_loss_recovery() -> None:
    controller = RelationRecoveryController()
    controller.update(observation(), estimate(RelationState.CONNECTED), 6, NORMAL_ACTION)
    decision = controller.update(observation(), estimate(RelationState.CANDIDATE_LOST), 7, NORMAL_ACTION)
    assert decision.state == RecoveryState.NORMAL
    assert not decision.action_overridden


def test_failed_verification_stops_after_bounded_attempts() -> None:
    controller = RelationRecoveryController(
        RecoveryConfig(
            miss_verification_steps=2,
            maximum_regrasp_attempts=1,
            verify_grasp_steps=2,
            maximum_state_steps=10,
        )
    )
    advance_to_verify(controller)
    disconnected = estimate(RelationState.DISCONNECTED)
    controller.update(observation(), disconnected, 4, NORMAL_ACTION)
    decision = controller.update(observation(), disconnected, 4, NORMAL_ACTION)
    assert decision.state == RecoveryState.RECOVERY_FAILED
    assert controller.failed


def test_oracle_policy_uses_only_current_contact_boolean() -> None:
    policy = OracleRelationRecoveryPolicy()
    policy.phase = 4
    disconnected_observation = observation()
    policy._update_online_state(disconnected_observation, 0)
    assert policy.relation_estimate is not None
    assert policy.relation_estimate.state == RelationState.DISCONNECTED
    connected_observation = PolicyObservation(
        ee_pose=disconnected_observation.ee_pose,
        object_pose=disconnected_observation.object_pose,
        target_pose=disconnected_observation.target_pose,
        gripper_opening_m=0.045,
        gripper_velocity_m_s=0.0,
        object_contact=True,
    )
    policy._update_online_state(connected_observation, 0)
    assert policy.relation_estimate.state == RelationState.CONNECTED
    assert policy.virtual_frame_pose is not None
    assert policy._active_frames(connected_observation) == ["target", "virtual_ee"]


def test_privileged_grasp_predicate_rejects_empty_close_and_drop() -> None:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    obj = np.concatenate((np.asarray([0.50, 0.0, 0.021]), identity))
    held_ee = np.concatenate((np.asarray([0.50, 0.0, 0.031]), identity))
    dropped_ee = np.concatenate((np.asarray([0.50, -0.18, 0.15]), identity))
    assert privileged_grasp_relation(held_ee, obj, 0.045)
    assert not privileged_grasp_relation(held_ee, obj, 0.001)
    assert not privileged_grasp_relation(dropped_ee, obj, 0.045)
