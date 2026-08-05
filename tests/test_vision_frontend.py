"""内置 DINO/SAM RGB-D 前端的无权重回归测试。"""

from __future__ import annotations

import numpy as np
from essay2608.data.robodojo_pose import RGBDPoseEstimator
from essay2608.data.vision_frontend import DinoSamConfig, DinoSamPoseEstimator


def test_builtin_dino_sam_is_lazy_and_selectable(monkeypatch) -> None:
    monkeypatch.setenv("ESSAY2608_RGBD_POSE_ESTIMATOR", "builtin:dino_sam")
    estimator = RGBDPoseEstimator.from_environment()
    assert estimator.name == "builtin:dino_sam"
    assert isinstance(estimator.estimator, DinoSamPoseEstimator)


def test_dino_sam_config_rejects_invalid_candidate_settings() -> None:
    try:
        DinoSamConfig(grid_size=1)
    except ValueError as error:
        assert "网格" in str(error)
    else:  # pragma: no cover - 仅用于确保构造器确实拒绝非法配置
        raise AssertionError("非法 DINO/SAM 网格未被拒绝")


def test_dino_sam_config_is_independent_of_numpy_inputs() -> None:
    config = DinoSamConfig()
    assert config.maximum_candidates == 8
    assert np.asarray([config.grid_size]).dtype.kind in "iu"
