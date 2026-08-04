from __future__ import annotations

import numpy as np

from essay2608.data.handover_schema import HandoverState
from essay2608.eval.bimanual_recovery_study import (
    BimanualRecoveryIntervention,
    fault_realization,
    score_bimanual_recovery_trace,
)


RIGHT = np.asarray([0.60, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0])
ACTION = np.asarray(
    [
        0.40,
        0.0,
        0.30,
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.60,
        0.0,
        0.30,
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
    ]
)


def test_miss_fault_is_single_transient_open_displacement() -> None:
    intervention = BimanualRecoveryIntervention("receiver_miss_once", 0.10)
    active = []
    for _ in range(12):
        decision = intervention.apply(
            action=ACTION,
            task_state=HandoverState.RIGHT_GRASP,
            right_pose=RIGHT,
        )
        active.append(decision.active)
        if decision.active:
            assert decision.action[15] > 0.0
            assert np.allclose(decision.action[8:11], RIGHT[:3] + [0.05, 0.0, 0.06])
    assert sum(active) == 8
    assert not active[-1]


def test_loss_fault_starts_after_declared_transfer_delay() -> None:
    intervention = BimanualRecoveryIntervention("receiver_loss_once", 0.10)
    decisions = [
        intervention.apply(
            action=ACTION,
            task_state=HandoverState.TRANSFER,
            right_pose=RIGHT,
        )
        for _ in range(12)
    ]
    assert not decisions[0].active
    assert not decisions[1].active
    assert decisions[2].active
    assert sum(decision.active for decision in decisions) == 8
    assert decisions[2].event == "receiver_forced_open_and_displaced"


def test_brief_loss_has_shorter_fixed_duration_and_never_restarts() -> None:
    intervention = BimanualRecoveryIntervention("receiver_brief_loss", 0.02)
    active = []
    for _ in range(40):
        decision = intervention.apply(
            action=ACTION,
            task_state=HandoverState.TRANSFER,
            right_pose=RIGHT,
        )
        active.append(decision.active)
    assert sum(active) == 6
    assert not any(active[21:])


def test_normal_condition_never_changes_action() -> None:
    intervention = BimanualRecoveryIntervention("normal", 0.02)
    decision = intervention.apply(
        action=ACTION,
        task_state=HandoverState.TRANSFER,
        right_pose=RIGHT,
    )
    assert not decision.active
    assert np.array_equal(decision.action, ACTION)


def test_fault_realization_uses_physical_loss_not_only_command() -> None:
    active = np.zeros(50, dtype=bool)
    active[15:21] = True
    truth_right = np.zeros(50, dtype=bool)
    truth_right[5:17] = True
    truth_right[25:] = True
    realization = fault_realization(
        "receiver_brief_loss",
        {
            "intervention_active": active,
            "truth_left_connected": np.ones(50, dtype=bool),
            "truth_right_connected": truth_right,
        },
        0.02,
    )
    assert realization["realized"]
    truth_right[17:25] = True
    unrealized = fault_realization(
        "receiver_brief_loss",
        {
            "intervention_active": active,
            "truth_left_connected": np.ones(50, dtype=bool),
            "truth_right_connected": truth_right,
        },
        0.02,
    )
    assert not unrealized["realized"]


def test_recovery_trace_scores_safe_release_latency_and_recovery_only_safety() -> None:
    steps = 7
    poses = np.zeros((steps, 7), dtype=np.float64)
    poses[:, 3] = 1.0
    actions = np.zeros((steps, 16), dtype=np.float64)
    actions[3, 8] = 0.02
    truth_right = np.asarray([0, 1, 1, 0, 0, 1, 1], dtype=bool)
    arrays = {
        "state": np.asarray(
            [
                HandoverState.RIGHT_GRASP,
                HandoverState.TRANSFER,
                HandoverState.TRANSFER,
                HandoverState.TRANSFER,
                HandoverState.TRANSFER,
                HandoverState.LEFT_RELEASE,
                HandoverState.RIGHT_TO_TARGET,
            ]
        ),
        "truth_left_connected": np.ones(steps, dtype=bool),
        "truth_right_connected": truth_right,
        "inferred_right_connected": truth_right,
        "intervention_active": np.asarray([0, 0, 1, 1, 0, 0, 0], dtype=bool),
        "recovery_state": np.asarray(
            ["NORMAL", "NORMAL", "NORMAL", "RECEIVER_LOSS_DETECTED", "VERIFY_BOTH", "RESUME_TASK", "NORMAL"]
        ),
        "recovery_transition": np.asarray(
            ["none", "none", "none", "NORMAL->RECEIVER_LOSS_DETECTED", "none", "VERIFY_BOTH->RESUME_TASK", "RESUME_TASK->NORMAL"]
        ),
        "recovery_requires_giver": np.asarray([0, 0, 0, 1, 1, 1, 0], dtype=bool),
        "left_ee_pose": poses,
        "right_ee_pose": poses,
        "applied_action": actions,
        "regrasp_attempts": np.asarray([0, 0, 0, 0, 1, 1, 1]),
        "phase_clock_held": np.asarray([0, 0, 0, 1, 1, 1, 0], dtype=bool),
    }
    metrics = score_bimanual_recovery_trace(
        arrays,
        "receiver_loss_once",
        0.02,
        method="relation_recovery",
        task_success=True,
    )
    assert metrics["safe_release"]
    assert not metrics["unsafe_release"]
    assert metrics["recovery_triggered"]
    assert metrics["time_to_recover_s"] == 0.04
    assert metrics["signed_detection_delay_from_truth_loss_s"] == 0.0
    assert metrics["giver_retained_during_recovery"]
    assert metrics["maximum_recovery_action_target_jump_m"] == 0.02


def test_after_release_fault_requires_physical_giver_release() -> None:
    active = np.zeros(60, dtype=bool)
    active[15:55] = True
    right = np.ones(60, dtype=bool)
    right[20:55] = False
    left = np.ones(60, dtype=bool)
    unrealized = fault_realization(
        "receiver_loss_after_release",
        {
            "intervention_active": active,
            "truth_left_connected": left,
            "truth_right_connected": right,
        },
        0.02,
    )
    assert not unrealized["realized"]
    left[10:] = False
    realized = fault_realization(
        "receiver_loss_after_release",
        {
            "intervention_active": active,
            "truth_left_connected": left,
            "truth_right_connected": right,
        },
        0.02,
    )
    assert realized["realized"]
