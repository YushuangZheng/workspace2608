"""Tests for the minimal bimanual handover supervision schema."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from essay2608.data.dataset import audit_bimanual_demonstration, load_bimanual_demo
from essay2608.data.handover_schema import HandoverState, handover_relation_label


def test_handover_relation_labels_include_a_short_bimanual_interval() -> None:
    labels = [handover_relation_label(state) for state in HandoverState]
    sequence = [labels[0]]
    for label in labels[1:]:
        if label != sequence[-1]:
            sequence.append(label)

    assert sequence == ["none", "left_only", "both", "right_only", "none"]
    assert handover_relation_label(HandoverState.RIGHT_GRASP) == "left_only"
    assert handover_relation_label(HandoverState.TRANSFER) == "both"


def test_bimanual_loader_remains_compatible_with_frozen_v1() -> None:
    demo = load_bimanual_demo(Path("data/handover_static/v1/demo_000.npz"))
    result = audit_bimanual_demonstration(demo)

    assert demo.relation_label is None
    assert result["relation_schema"] == "legacy_carrier_only"


def test_bimanual_audit_rejects_incorrect_relation_supervision() -> None:
    demo = load_bimanual_demo(Path("data/handover_static/v1/demo_000.npz"))
    labels = np.asarray(
        [handover_relation_label(state) for state in demo.state],
        dtype="U16",
    )
    left_gripper = np.full((demo.steps, 2), 0.04, dtype=np.float32)
    right_gripper = left_gripper.copy()
    left_gripper[np.isin(demo.state, np.arange(2, 8))] = 0.0
    right_gripper[np.isin(demo.state, np.arange(6, 10))] = 0.0
    upgraded = replace(
        demo,
        relation_label=labels,
        left_gripper_state=left_gripper,
        right_gripper_state=right_gripper,
    )

    result = audit_bimanual_demonstration(upgraded)
    assert result["relation_sequence"] == ["none", "left_only", "both", "right_only", "none"]

    bad_labels = labels.copy()
    bad_labels[np.flatnonzero(demo.state == HandoverState.TRANSFER)[0]] = "left_only"
    with pytest.raises(ValueError, match="relation labels disagree"):
        audit_bimanual_demonstration(replace(upgraded, relation_label=bad_labels))
