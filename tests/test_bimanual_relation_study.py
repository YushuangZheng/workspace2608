from __future__ import annotations

import numpy as np
import torch

from essay2608.data.handover_schema import HandoverState
from essay2608.eval.bimanual_relation_study import (
    BimanualRelationIntervention,
    condition_realization,
    score_bimanual_relation_trace,
)


def action() -> torch.Tensor:
    values = torch.zeros((1, 16), dtype=torch.float32)
    values[:, 7] = -1.0
    values[:, 15] = -1.0
    return values


def right_pose() -> torch.Tensor:
    return torch.asarray([[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]])


def test_receiver_delay_holds_grasp_clock_then_allows_settle() -> None:
    intervention = BimanualRelationIntervention("receiver_delayed", 0.1)
    decisions = [
        intervention.apply(
            action=action(),
            state=HandoverState.RIGHT_GRASP,
            truth_label="left_only",
            right_pose=right_pose(),
        )
        for _ in range(21)
    ]
    assert all(decision.hold_phase_clock for decision in decisions[:20])
    assert all(decision.action[0, 15].item() == 1.0 for decision in decisions[:10])
    assert all(decision.action[0, 15].item() == -1.0 for decision in decisions[10:])
    assert not decisions[-1].hold_phase_clock
    assert not decisions[-1].active


def test_one_arm_pause_freezes_pose_without_opening_gripper() -> None:
    intervention = BimanualRelationIntervention("one_arm_paused", 0.1)
    decision = intervention.apply(
        action=action(),
        state=HandoverState.RIGHT_TO_TARGET,
        truth_label="right_only",
        right_pose=right_pose(),
    )
    assert decision.active and decision.hold_phase_clock
    assert torch.equal(decision.action[:, 8:15], right_pose())
    assert decision.action[0, 15].item() == -1.0


def test_relation_metrics_keep_edges_separate() -> None:
    truth_left = np.asarray([0, 1, 1, 1, 0, 0], dtype=bool)
    truth_right = np.asarray([0, 0, 0, 1, 1, 0], dtype=bool)
    inferred_left = np.asarray([0, 0, 1, 1, 0, 0], dtype=bool)
    inferred_right = np.asarray([0, 0, 0, 0, 1, 0], dtype=bool)
    labels = np.asarray(["none", "left_only", "left_only", "both", "right_only", "none"])
    inferred_labels = np.asarray(
        ["none", "none", "left_only", "left_only", "right_only", "none"]
    )
    metrics = score_bimanual_relation_trace(
        truth_labels=labels,
        inferred_labels=inferred_labels,
        truth_left=truth_left,
        truth_right=truth_right,
        inferred_left=inferred_left,
        inferred_right=inferred_right,
        control_dt_s=0.1,
    )
    assert metrics["left"]["fn"] == 1
    assert metrics["right"]["fn"] == 1
    assert metrics["left"]["transitions"]["maximum_delay_s"] == 0.1
    assert metrics["right"]["transitions"]["maximum_delay_s"] == 0.1
    assert not metrics["privileged_contact_used_as_estimator_input"]


def test_condition_realization_requires_physical_effect() -> None:
    labels = np.asarray(["none", "left_only", "left_only", "none", "none"])
    result = condition_realization(
        "receiver_miss",
        truth_left=np.asarray([0, 1, 1, 0, 0], dtype=bool),
        truth_right=np.zeros(5, dtype=bool),
        truth_labels=labels,
        intervention_active=np.asarray([0, 0, 1, 1, 1], dtype=bool),
        intervention_event=np.asarray(
            ["none", "none", "receiver_forced_open", "receiver_forced_open", "receiver_forced_open"]
        ),
        control_dt_s=0.1,
    )
    assert result["realized"]

    not_realized = condition_realization(
        "receiver_miss",
        truth_left=np.asarray([0, 1, 1, 1, 1], dtype=bool),
        truth_right=np.asarray([0, 0, 0, 1, 1], dtype=bool),
        truth_labels=np.asarray(["none", "left_only", "left_only", "both", "both"]),
        intervention_active=np.asarray([0, 0, 1, 1, 1], dtype=bool),
        intervention_event=np.asarray(["none", "none", "x", "x", "x"]),
        control_dt_s=0.1,
    )
    assert not not_realized["realized"]
