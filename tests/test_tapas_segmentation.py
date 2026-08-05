"""TAPAS 风格运动学分割回归测试。"""

from __future__ import annotations

import numpy as np

from robodojo_adapter.tapas import (
    TapasSegmentationConfig,
    tapas_skill_boundaries,
    tapas_skill_labels,
)


def _trajectory() -> np.ndarray:
    poses = np.zeros((90, 7), dtype=np.float64)
    poses[:, 3] = 1.0
    poses[10:25, 0] = np.linspace(0.0, 0.25, 15)
    poses[25:45, 0] = 0.25
    poses[45:60, 0] = np.linspace(0.25, 0.6, 15)
    poses[60:, 0] = 0.6
    return poses


def test_tapas_boundaries_follow_motion_valleys() -> None:
    config = TapasSegmentationConfig(minimum_skill_length=8, maximum_skills=3)
    boundaries = tapas_skill_boundaries(_trajectory(), config=config)
    assert boundaries[0] == 0
    assert boundaries[-1] == 90
    assert len(boundaries) == 4
    assert boundaries != (0, 27, 61, 90)
    assert all(
        right - left >= config.minimum_skill_length
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def test_tapas_labels_are_contiguous() -> None:
    labels = tapas_skill_labels(
        _trajectory(),
        config=TapasSegmentationConfig(minimum_skill_length=8, maximum_skills=3),
    )
    sequence = labels[np.r_[True, labels[1:] != labels[:-1]]]
    assert np.array_equal(sequence, np.arange(len(sequence)))
    assert labels.shape == (90,)
