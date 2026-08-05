"""RoboDojo 任务物体位姿观测后端。

项目明确支持两条互斥路径：

* ``oracle_pose``：直接读取 RoboDojo layout manager 的真值；
* ``rgbd_pose``：只把 RGB-D 和相机标定交给用户提供的 PoseEstimator。

RGB-D 估计器通过 ``ESSAY2608_RGBD_POSE_ESTIMATOR=module:function`` 注入，也可使用
``builtin:dino_sam`` 启用可选的 Transformers DINOv2/SAM 候选前端；默认不会下载模型
或改变 Oracle Pose 基准。
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

PoseMap = dict[str, np.ndarray]


class PoseEstimator(Protocol):
    """RGB-D 任务位姿估计器的项目接口。"""

    def estimate(
        self,
        *,
        task_name: str,
        rgb: dict[str, Any],
        depth: dict[str, Any],
        camera: dict[str, dict[str, Any]],
        observation: dict,
    ) -> PoseMap:
        """由每个相机的 RGB-D 和标定返回 ``label -> xyz+wxyz``。"""


def _pose(value: Any, field: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"RGB-D 位姿 {field} 必须是有限的 xyz+wxyz 七维向量")
    return pose


def validate_pose_map(values: Any) -> PoseMap:
    if not isinstance(values, dict) or not values:
        raise ValueError("PoseEstimator 必须返回非空的 label -> pose 字典")
    return {str(name): _pose(value, f"{name}") for name, value in values.items()}


@dataclass(frozen=True)
class OraclePoseEstimator:
    """Oracle 路径的标识实现；实际 layout 读取仍由 GUI 适配层完成。"""

    name: str = "robodojo_layout_manager"


@dataclass
class RGBDPoseEstimator:
    """从环境变量加载用户的 RGB-D 位姿估计函数。"""

    estimator: Any
    name: str

    @classmethod
    def from_environment(cls) -> RGBDPoseEstimator:
        spec = os.environ.get("ESSAY2608_RGBD_POSE_ESTIMATOR", "").strip()
        if spec == "builtin:dino_sam":
            from .vision_frontend import DinoSamPoseEstimator

            return cls(estimator=DinoSamPoseEstimator(), name=spec)
        if not spec or ":" not in spec:
            raise RuntimeError(
                "rgbd_pose 模式需要设置 ESSAY2608_RGBD_POSE_ESTIMATOR=module:function；"
                "不能在未提供估计器时回退到 Oracle 位姿"
            )
        module_name, function_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        estimator = getattr(module, function_name, None)
        if estimator is None or not callable(estimator):
            raise RuntimeError(f"RGB-D PoseEstimator 不可调用：{spec}")
        return cls(estimator=estimator, name=spec)

    def estimate(
        self,
        *,
        task_name: str,
        rgb: dict[str, Any],
        depth: dict[str, Any],
        camera: dict[str, dict[str, Any]],
        observation: dict,
    ) -> PoseMap:
        result = self.estimator(
            task_name=task_name,
            rgb=rgb,
            depth=depth,
            camera=camera,
            observation=observation,
        )
        return validate_pose_map(result)


def rgbd_from_observation(observation: dict) -> tuple[dict, dict, dict]:
    vision = observation.get("vision")
    if not isinstance(vision, dict) or not vision:
        raise ValueError("RGB-D PoseEstimator 收到的观测没有 vision 相机数据")
    rgb: dict[str, Any] = {}
    depth: dict[str, Any] = {}
    camera: dict[str, dict[str, Any]] = {}
    for camera_name, values in vision.items():
        if not isinstance(values, dict):
            continue
        if "color" not in values or "depth" not in values:
            raise ValueError(
                f"相机 {camera_name} 缺少 color/depth；请使用 rgbd_pose 环境配置"
            )
        rgb[camera_name] = values["color"]
        depth[camera_name] = values["depth"]
        camera[camera_name] = {
            key: values[key]
            for key in ("intrinsic_matrix", "extrinsic_matrix", "shape")
            if key in values
        }
    if not rgb:
        raise ValueError("RGB-D PoseEstimator 没有可用相机")
    return rgb, depth, camera


def estimate_rgbd_pose(estimator: RGBDPoseEstimator, task_name: str, observation: dict) -> PoseMap:
    rgb, depth, camera = rgbd_from_observation(observation)
    return estimator.estimate(
        task_name=task_name,
        rgb=rgb,
        depth=depth,
        camera=camera,
        observation=observation,
    )


__all__ = [
    "OraclePoseEstimator",
    "PoseEstimator",
    "RGBDPoseEstimator",
    "estimate_rgbd_pose",
    "rgbd_from_observation",
    "validate_pose_map",
]
