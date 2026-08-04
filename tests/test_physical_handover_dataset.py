from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from essay2608.data.physical_handover import audit_physical_handover_dataset


def _write_dataset(root: Path, *, corrupt_relation: bool = False) -> Path:
    steps = 40
    dt = 0.2
    seed = 12
    source_sha = "test-source"
    states = np.concatenate((np.arange(12), np.full(steps - 12, 11))).astype(np.int64)
    labels = np.asarray(
        ["none"] * 2
        + ["left_only"] * 2
        + ["both"] * 2
        + ["right_only"] * 2
        + ["none"] * (steps - 8),
        dtype="U16",
    )
    left_connected = np.isin(labels, ["left_only", "both"])
    right_connected = np.isin(labels, ["both", "right_only"])
    if corrupt_relation:
        labels[3] = "right_only"

    moving = np.zeros((15, 3), dtype=np.float32)
    moving[:, 0] = np.linspace(0.0, 0.2, len(moving))
    moving[:, 2] = np.asarray(
        [0.18, 0.20, 0.24, 0.28, 0.30, 0.28, 0.26, 0.24, 0.22,
         0.20, 0.19, 0.185, 0.183, 0.182, 0.181],
        dtype=np.float32,
    )
    terminal = np.asarray([0.2, 0.0, 0.181], dtype=np.float32)
    object_position = np.concatenate(
        (moving, np.repeat(terminal[None], steps - len(moving), axis=0)), axis=0
    )
    quaternion = np.zeros((steps, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    target_position = np.repeat(terminal[None], steps, axis=0)
    left_force = np.zeros((steps, 2, 3), dtype=np.float32)
    right_force = np.zeros((steps, 2, 3), dtype=np.float32)
    left_force[left_connected, :, 0] = 1.0
    right_force[right_connected, :, 0] = 1.0

    arrays = {
        "time": np.arange(steps, dtype=np.float32) * dt,
        "state": states,
        "left_ee_position": object_position.copy(),
        "left_ee_orientation": quaternion.copy(),
        "right_ee_position": object_position.copy(),
        "right_ee_orientation": quaternion.copy(),
        "left_ee_pose": np.concatenate((object_position, quaternion), axis=-1),
        "right_ee_pose": np.concatenate((object_position, quaternion), axis=-1),
        "object_position": object_position,
        "object_orientation": quaternion.copy(),
        "object_pose": np.concatenate((object_position, quaternion), axis=-1),
        "target_position": target_position,
        "target_pose": np.concatenate((target_position, quaternion), axis=-1),
        "object_linear_velocity": np.zeros((steps, 3), dtype=np.float32),
        "action": np.zeros((steps, 16), dtype=np.float32),
        "left_finger_force": left_force,
        "right_finger_force": right_force,
        "left_finger_position": np.zeros((steps, 2, 3), dtype=np.float32),
        "right_finger_position": np.zeros((steps, 2, 3), dtype=np.float32),
        "left_connected": left_connected,
        "right_connected": right_connected,
        "left_confidence": left_connected.astype(np.float32),
        "right_confidence": right_connected.astype(np.float32),
        "relation_label": labels,
        "control_dt": np.asarray(dt, dtype=np.float32),
        "terminal_object_position": terminal,
        "terminal_target_position": terminal,
        "seed": np.asarray(seed, dtype=np.int64),
        "source_sha256": np.asarray(source_sha),
        "experiment_fingerprint": np.asarray("test-fingerprint"),
        "both_duration_s": np.asarray(0.4, dtype=np.float32),
        "maximum_object_height_m": np.asarray(0.30, dtype=np.float32),
        "final_xy_error_m": np.asarray(0.0, dtype=np.float32),
        "object_on_support": np.asarray(True),
        "stable": np.asarray(True),
        "settling_displacement_m": np.asarray(0.0, dtype=np.float32),
        "quaternion_order": np.asarray("wxyz"),
        "coordinate_frame": np.asarray("local_environment"),
    }
    demo_path = root / "demo_000.npz"
    np.savez_compressed(demo_path, **arrays)
    manifest = {
        "task_id": "Essay2608-Bimanual-Physical-Handover-v0",
        "dataset_schema_version": 1,
        "num_demos": 1,
        "requested_seeds": [seed],
        "demos": [{"file": demo_path.name, "seed": seed}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_physical_dataset_audit_accepts_phase_independent_relations(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    result = audit_physical_handover_dataset(
        dataset,
        expected_seeds=[12],
        expected_source_sha256="test-source",
    )
    assert result["num_demos"] == 1
    assert result["min_both_duration_s"] == pytest.approx(0.4)
    assert result["total_phase_disagreement_steps"] > 0


def test_physical_dataset_audit_rejects_relation_edge_mismatch(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, corrupt_relation=True)
    with pytest.raises(ValueError, match="关系"):
        audit_physical_handover_dataset(
            dataset,
            expected_seeds=[12],
            expected_source_sha256="test-source",
        )
