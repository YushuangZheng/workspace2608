"""Tests for additive phase-level rollout attribution."""

from __future__ import annotations

import numpy as np

from essay2608.eval import analyze_phase_trace, compare_paired_methods, summarize_success_metrics


def _trial(method: str, scale: float = 1.0) -> tuple[dict[str, np.ndarray], dict]:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [scale, scale, 0.0], [2 * scale, scale, 0.0]]
    )
    phases = np.asarray([0, 1, 1, 2])
    arrays = {
        "ee_position": positions,
        "object_position": np.zeros_like(positions),
        "target_position": np.ones_like(positions),
        "action": np.concatenate((positions, np.zeros((4, 5))), axis=1),
        "phase": phases,
        "raw_action_position": positions,
        "policy_action_position": positions,
        "action_rate_limited": np.asarray([False, False, True, False]),
    }
    result = {
        "method": method,
        "condition": "static",
        "seed": 1,
        "metrics": {
            "path_length_m": 3.0 * scale,
            "forced_phase_transitions": 0,
            "frame_switch_diagnostics": [],
        },
    }
    return arrays, result


def test_phase_paths_add_to_saved_rollout_path() -> None:
    arrays, result = _trial("mask_only")
    analysis = analyze_phase_trace(arrays, result, control_dt=0.5)
    assert analysis["path_partition_residual_m"] == 0.0
    assert analysis["phases"]["approach_above_object"]["path_length_m"] == 2.0
    assert analysis["phases"]["approach_object"]["path_length_m"] == 1.0


def test_paired_comparison_reports_candidate_minus_baseline() -> None:
    baseline_arrays, baseline_result = _trial("mask_only", scale=1.0)
    candidate_arrays, candidate_result = _trial("full_dynamac", scale=0.5)
    comparison = compare_paired_methods(
        [
            analyze_phase_trace(baseline_arrays, baseline_result),
            analyze_phase_trace(candidate_arrays, candidate_result),
        ]
    )
    assert comparison["num_pairs"] == 1
    assert comparison["num_candidate_shorter"] == 1
    assert comparison["mean_candidate_minus_baseline_path_m"] == -1.5


def test_success_audit_exposes_legacy_and_stable_place_disagreement() -> None:
    _, result = _trial("mask_only")
    result["metrics"].update(
        {
            "policy_complete": True,
            "environment_done": False,
            "final_error_3d_m": 0.061,
            "final_xy_error_m": 0.002,
            "success": True,
            "object_on_support": True,
            "gripper_released": True,
            "stable_after_release": True,
        }
    )
    audit = summarize_success_metrics([result])["methods"]["mask_only"]
    assert audit["legacy_3d_success_rate"] == 0.0
    assert audit["stable_place_success_rate"] == 1.0
