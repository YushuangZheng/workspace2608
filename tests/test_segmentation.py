"""Tests for velocity-based skill segmentation diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from essay2608.data.dataset import Demonstration, load_dataset
from essay2608.data.segmentation import (
    SegmentationConfig,
    analyze_segmentation,
    end_effector_speeds,
    segment_demonstration,
)


def synthetic_demo(positions: np.ndarray, quaternions: np.ndarray, dt: float = 0.02) -> Demonstration:
    steps = len(positions)
    ee_pose = np.concatenate((positions, quaternions), axis=-1)
    return Demonstration(
        path=Path("synthetic.npz"),
        time=np.arange(steps) * dt,
        state=np.minimum(np.arange(steps) // max(steps // 10, 1), 9),
        ee_pose=ee_pose,
        object_pose=ee_pose.copy(),
        target_pose=ee_pose.copy(),
        action=np.zeros((steps, 8)),
        joint_pos=np.zeros((steps, 9)),
        joint_vel=np.zeros((steps, 9)),
        control_dt=dt,
        final_error=0.0,
    )


def test_angular_speed_detects_rotation_without_translation() -> None:
    steps = 21
    positions = np.zeros((steps, 3))
    angles = np.linspace(0.0, np.pi / 2.0, steps)
    quaternions = np.stack(
        (np.cos(angles / 2.0), np.zeros(steps), np.zeros(steps), np.sin(angles / 2.0)),
        axis=-1,
    )
    linear, angular = end_effector_speeds(
        synthetic_demo(positions, quaternions),
        smoothing_duration_s=0.02,
    )
    np.testing.assert_allclose(linear, 0.0)
    np.testing.assert_allclose(angular, np.pi / 2.0 / (steps - 1) / 0.02)


def test_persistent_low_speed_intervals_create_coarse_boundaries() -> None:
    steps = 101
    positions = np.zeros((steps, 3))
    positions[20:40, 0] = np.linspace(0.01, 0.20, 20)
    positions[40:60, 0] = 0.20
    positions[60:80, 0] = np.linspace(0.21, 0.40, 20)
    positions[80:, 0] = 0.40
    quaternions = np.zeros((steps, 4))
    quaternions[:, 0] = 1.0
    config = SegmentationConfig(
        smoothing_duration_s=0.02,
        minimum_low_speed_duration_s=0.10,
        maximum_merge_gap_s=0.04,
        endpoint_tolerance_s=0.04,
    )
    trace = segment_demonstration(
        synthetic_demo(positions, quaternions),
        linear_speed_threshold_m_s=0.01,
        angular_speed_threshold_rad_s=0.01,
        config=config,
    )
    np.testing.assert_array_equal(trace.boundary_indices, [19, 49, 79])
    assert trace.summary["automatic_segment_count"] == 4


def test_frozen_dataset_segmentation_is_cross_demo_consistent() -> None:
    demonstrations, _ = load_dataset("data/pick_place_static/v1", verify_hashes=True)
    result, traces = analyze_segmentation(demonstrations)
    assert len(traces) == 5
    assert result["training_only"]
    assert not result["replaces_training_segmentation"]
    assert result["consistency"]["automatic_segment_counts"] == [5, 5, 5, 5, 5]
    assert result["consistency"]["fully_supported_boundary_clusters"] == 4
    assert all(cluster["support"] == 5 for cluster in result["aligned_boundaries"])
    assert all(len(trace.summary["manual_boundaries"]) == 9 for trace in traces)
