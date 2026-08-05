"""可选的 DINO/SAM RGB-D 任务帧候选前端。

该模块不改变默认的 Oracle Pose 轨道。设置为 ``builtin:dino_sam`` 后，首次使用时
从 Hugging Face 加载 DINOv2 与 SAM 权重，用 SAM 网格提示产生候选物体掩码，再用
DINO 图像特征对候选排序；候选中心通过深度和相机标定反投影为 ``xyz+wxyz``。

它是可运行的视觉候选接口，但没有论文未公开的 TAPAS 标签/提示策略，因此输出名为
``visual_candidate_XXX``。正式的 DynaMAC 训练仍应使用稳定的任务标签或 Oracle Pose，
不能把自动候选前端的近似结果与论文专用标注混为一谈。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DinoSamConfig:
    """DINO/SAM 权重和候选筛选参数。"""

    dino_model: str = "facebook/dinov2-small"
    sam_model: str = "facebook/sam-vit-base"
    device: str = "auto"
    grid_size: int = 6
    minimum_area_ratio: float = 0.002
    maximum_candidates: int = 8

    def __post_init__(self) -> None:
        if self.grid_size < 2 or self.maximum_candidates < 1:
            raise ValueError("DINO/SAM 网格和候选数量必须为正")
        if not 0.0 < self.minimum_area_ratio < 1.0:
            raise ValueError("minimum_area_ratio 必须位于 (0,1)")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _rgb_uint8(value: Any) -> np.ndarray:
    image = _array(value)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB 图像必须为 [H,W,3]")
    if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _depth_float(value: Any, shape: tuple[int, int]) -> np.ndarray:
    depth = _array(value).squeeze()
    if depth.shape != shape:
        raise ValueError(f"深度尺寸 {depth.shape} 与 RGB 尺寸 {shape} 不一致")
    depth = depth.astype(np.float64)
    if not np.any(np.isfinite(depth) & (depth > 0.0)):
        raise ValueError("深度图没有有效正深度")
    return depth


def _camera_pose(values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = _array(values.get("intrinsic_matrix"))
    extrinsic = _array(values.get("extrinsic_matrix"))
    if intrinsic.shape == (3, 3):
        matrix = intrinsic
    elif intrinsic.size == 9:
        matrix = intrinsic.reshape(3, 3)
    else:
        raise ValueError("相机 intrinsic_matrix 必须为 [3,3]")
    if extrinsic.shape != (4, 4):
        raise ValueError("相机 extrinsic_matrix 必须为 [4,4]")
    return matrix.astype(np.float64), extrinsic.astype(np.float64)


class DinoSamPoseEstimator:
    """基于 Transformers DINOv2/SAM 的可选候选位姿估计器。"""

    name = "builtin:dino_sam"

    def __init__(self, config: DinoSamConfig = DinoSamConfig()) -> None:
        self.config = config
        self._processor = None
        self._sam = None
        self._dino_processor = None
        self._dino = None
        self._torch = None

    def _load(self) -> None:
        if self._sam is not None:
            return
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                Dinov2Model,
                SamModel,
                SamProcessor,
            )
        except ImportError as error:  # pragma: no cover - 可选路径
            raise RuntimeError(
                "builtin:dino_sam 需要安装 torch、transformers 和 pillow；"
                "默认 Oracle Pose 不需要这些依赖"
            ) from error
        device = (
            "cuda"
            if self.config.device == "auto" and torch.cuda.is_available()
            else self.config.device
        )
        self._torch = torch
        self._processor = SamProcessor.from_pretrained(self.config.sam_model)
        self._sam = SamModel.from_pretrained(self.config.sam_model).to(device).eval()
        self._dino_processor = AutoImageProcessor.from_pretrained(self.config.dino_model)
        self._dino = Dinov2Model.from_pretrained(self.config.dino_model).to(device).eval()

    def _masks(self, image: np.ndarray) -> list[np.ndarray]:
        self._load()
        height, width = image.shape[:2]
        points = [
            [
                [float(x * width / (self.config.grid_size + 1)), float(y * height / (self.config.grid_size + 1))]
                for y in range(1, self.config.grid_size + 1)
                for x in range(1, self.config.grid_size + 1)
            ]
        ]
        encoded = self._processor(image, input_points=points, return_tensors="pt")
        encoded = {name: value.to(self._sam.device) for name, value in encoded.items()}
        with self._torch.inference_mode():
            output = self._sam(**encoded, multimask_output=True)
        masks = self._processor.post_process_masks(
            output.pred_masks, encoded["original_sizes"], encoded["reshaped_input_sizes"]
        )[0]
        masks = _array(masks).astype(bool)
        minimum_area = image.shape[0] * image.shape[1] * self.config.minimum_area_ratio
        candidates = []
        for mask in masks.reshape(-1, *masks.shape[-2:]):
            area = int(mask.sum())
            if area >= minimum_area:
                candidates.append((area, mask))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[np.ndarray] = []
        for _, mask in candidates:
            overlap = [float(np.logical_and(mask, other).sum() / max(mask.sum(), 1)) for other in selected]
            if not overlap or max(overlap) < 0.75:
                selected.append(mask)
            if len(selected) >= self.config.maximum_candidates:
                break
        return selected

    def _dino_scores(self, image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
        self._load()
        crops = []
        for mask in masks:
            rows, columns = np.nonzero(mask)
            if len(rows) == 0:
                crops.append(image)
                continue
            crops.append(image[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1])
        encoded = self._dino_processor(images=crops, return_tensors="pt")
        encoded = {name: value.to(self._dino.device) for name, value in encoded.items()}
        with self._torch.inference_mode():
            hidden = self._dino(**encoded).last_hidden_state[:, 0]
        return _array(hidden.norm(dim=-1)).reshape(-1)

    def estimate(
        self,
        *,
        task_name: str,
        rgb: dict[str, Any],
        depth: dict[str, Any],
        camera: dict[str, dict[str, Any]],
        observation: dict,
    ) -> dict[str, np.ndarray]:
        del task_name, observation
        if not rgb:
            raise ValueError("DINO/SAM 需要至少一个 RGB 相机")
        camera_name = next(iter(rgb))
        image = _rgb_uint8(rgb[camera_name])
        depth_map = _depth_float(depth[camera_name], image.shape[:2])
        intrinsic, extrinsic = _camera_pose(camera[camera_name])
        masks = self._masks(image)
        if not masks:
            raise ValueError("SAM 没有生成有效候选掩码")
        scores = self._dino_scores(image, masks)
        order = np.argsort(-scores)
        result: dict[str, np.ndarray] = {}
        for candidate_index in order:
            mask = masks[int(candidate_index)]
            rows, columns = np.nonzero(mask)
            valid_depth = depth_map[rows, columns]
            valid = np.isfinite(valid_depth) & (valid_depth > 0.0)
            if not np.any(valid):
                continue
            u = float(np.median(columns[valid]))
            v = float(np.median(rows[valid]))
            z = float(np.median(valid_depth[valid]))
            ray = np.asarray(
                [(u - intrinsic[0, 2]) / intrinsic[0, 0], -(v - intrinsic[1, 2]) / intrinsic[1, 1], -1.0]
            )
            world = extrinsic[:3, :3] @ (ray * z) + extrinsic[:3, 3]
            result[f"visual_candidate_{len(result):03d}"] = np.asarray(
                [*world, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        if not result:
            raise ValueError("候选掩码没有有效 RGB-D 深度")
        return result


__all__ = ["DinoSamConfig", "DinoSamPoseEstimator"]
