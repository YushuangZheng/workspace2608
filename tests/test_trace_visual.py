from __future__ import annotations

import numpy as np

from essay2608.eval.trace_visual import audit_failure_taxonomy, reconstruct_active_frames


def test_reconstruct_active_frames_uses_persisted_switch_anchors() -> None:
    frames = reconstruct_active_frames(
        6,
        [
            {"step": 2, "before": ["object", "target"], "after": ["target"]},
            {"step": 4, "before": ["target"], "after": ["object", "target"]},
        ],
    )
    assert frames == [
        ("object", "target"),
        ("object", "target"),
        ("target",),
        ("target",),
        ("object", "target"),
        ("object", "target"),
    ]


def test_reconstruct_active_frames_does_not_guess_without_anchor() -> None:
    assert reconstruct_active_frames(3, []) == [None, None, None]


def test_failure_taxonomy_checks_stable_xy_failure() -> None:
    metadata = {
        "metrics": {
            "success": False,
            "failure_reason": "placement_xy_above_threshold",
            "final_xy_error_m": 0.02,
            "policy_complete": True,
            "environment_done": False,
            "gripper_released": True,
            "object_on_support": True,
            "stable_after_release": True,
            "success_criteria": {"xy_threshold_m": 0.01},
        }
    }
    arrays = {
        "object_position": np.asarray([[0.02, 0.0, 0.021]]),
        "target_position": np.asarray([[0.0, 0.0, 0.08]]),
    }
    audit = audit_failure_taxonomy(metadata, arrays)
    assert audit["consistent"]
    assert audit["checks"]["placement_failure_semantics"]


def test_failure_taxonomy_rejects_wrong_recorded_xy() -> None:
    metadata = {
        "metrics": {
            "success": True,
            "failure_reason": "success",
            "final_xy_error_m": 0.02,
            "policy_complete": True,
            "environment_done": False,
            "gripper_released": True,
            "object_on_support": True,
            "stable_after_release": True,
            "success_criteria": {"xy_threshold_m": 0.01},
        }
    }
    arrays = {
        "object_position": np.asarray([[0.001, 0.0, 0.021]]),
        "target_position": np.asarray([[0.0, 0.0, 0.08]]),
    }
    audit = audit_failure_taxonomy(metadata, arrays)
    assert audit["consistent"]
    assert not audit["terminal_trace_alignment"]


def test_failure_taxonomy_prefers_persisted_terminal_snapshot() -> None:
    metadata = {
        "metrics": {
            "success": True,
            "failure_reason": "success",
            "final_xy_error_m": 0.001,
            "policy_complete": True,
            "environment_done": False,
            "gripper_released": True,
            "object_on_support": True,
            "stable_after_release": True,
            "success_criteria": {"xy_threshold_m": 0.01},
        }
    }
    arrays = {
        "object_position": np.asarray([[0.02, 0.0, 0.021]]),
        "target_position": np.asarray([[0.0, 0.0, 0.08]]),
        "terminal_object_position": np.asarray([0.001, 0.0, 0.021]),
        "terminal_target_position": np.asarray([0.0, 0.0, 0.08]),
    }
    audit = audit_failure_taxonomy(metadata, arrays)
    assert audit["terminal_snapshot_persisted"]
    assert audit["terminal_trace_alignment"]
