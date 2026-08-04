from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from essay2608.eval.physical_handover_audit import (
    TASK_ID,
    _experiment_fingerprint,
    audit_physical_handover_run,
)


def _write_run(root: Path, *, corrupt_terminal: bool = False) -> Path:
    seed = 7
    source_sha = "frozen-source"
    labels = np.asarray(
        ["none", "left_only", "both", "right_only"] + ["none"] * 26,
        dtype="U16",
    )
    left = np.isin(labels, ["left_only", "both"])
    right = np.isin(labels, ["right_only", "both"])
    moving = np.asarray(
        [[0.0, 0.0, 0.18], [0.0, 0.0, 0.22], [0.0, 0.0, 0.30],
         [0.2, 0.0, 0.24], [0.2, 0.0, 0.181]],
        dtype=np.float32,
    )
    object_position = np.concatenate(
        [moving, np.repeat(moving[-1][None], 25, axis=0)], axis=0
    )
    target = np.asarray([0.2, 0.0, 0.181], dtype=np.float32)
    terminal = target.copy()
    fingerprint = _experiment_fingerprint(
        seed,
        source_sha,
        max_steps=1400,
        success_xy_threshold=0.04,
        minimum_both_duration_s=0.20,
    )
    trial = {
        "task_id": TASK_ID,
        "seed": seed,
        "experiment_fingerprint": fingerprint,
        "source_sha256": source_sha,
        "success": True,
        "failure_reason": "success",
        "expert_complete": True,
        "expert_failed": False,
        "steps": len(labels),
        "relation_sequence": ["none", "left_only", "both", "right_only", "none"],
        "both_duration_s": 0.2,
        "maximum_object_height_m": 0.30,
        "final_object_position_m": terminal.tolist(),
        "final_target_position_m": target.tolist(),
        "final_xy_error_m": 0.0,
        "object_on_support": True,
        "stable": True,
        "settling_displacement_m": 0.0,
        "video_requested": False,
    }
    trial_dir = root / "trials"
    trial_dir.mkdir(parents=True)
    stem = f"scripted_physical_handover__seed_{seed}"
    (trial_dir / f"{stem}.json").write_text(json.dumps(trial), encoding="utf-8")
    step = len(labels)
    trace = {
        "state": np.arange(step),
        "left_ee_position": np.zeros((step, 3)),
        "left_ee_orientation": np.zeros((step, 4)),
        "right_ee_position": np.zeros((step, 3)),
        "right_ee_orientation": np.zeros((step, 4)),
        "object_position": object_position,
        "object_orientation": np.zeros((step, 4)),
        "target_position": np.repeat(target[None], step, axis=0),
        "object_linear_velocity": np.zeros((step, 3)),
        "action": np.zeros((step, 16)),
        "left_finger_force": np.zeros((step, 2, 3)),
        "right_finger_force": np.zeros((step, 2, 3)),
        "left_finger_position": np.zeros((step, 2, 3)),
        "right_finger_position": np.zeros((step, 2, 3)),
        "left_connected": left,
        "right_connected": right,
        "left_confidence": np.zeros(step),
        "right_confidence": np.zeros(step),
        "relation_label": labels,
        "control_dt": np.asarray(0.2, dtype=np.float32),
        "terminal_object_position": terminal + ([0.1, 0.0, 0.0] if corrupt_terminal else 0.0),
        "terminal_target_position": target,
    }
    np.savez_compressed(trial_dir / f"{stem}.npz", **trace)
    summary = {
        "task_id": TASK_ID,
        "source_sha256": source_sha,
        "num_trials": 1,
        "num_successes": 1,
        "success_rate": 1.0,
        "seeds": [seed],
        "trials": [trial],
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_audit_accepts_complete_physical_handover_run(tmp_path: Path) -> None:
    summary = _write_run(tmp_path)
    result = audit_physical_handover_run(
        summary,
        expected_seeds=[7],
        expected_source_sha256="frozen-source",
        expected_successes=1,
    )
    assert result.trial_count == 1
    assert result.success_count == 1


def test_audit_rejects_terminal_position_mismatch(tmp_path: Path) -> None:
    summary = _write_run(tmp_path, corrupt_terminal=True)
    with pytest.raises(ValueError, match="终端物体位置"):
        audit_physical_handover_run(
            summary,
            expected_seeds=[7],
            expected_source_sha256="frozen-source",
            expected_successes=1,
        )
