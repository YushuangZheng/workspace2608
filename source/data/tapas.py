"""TAPAS 风格的轻量技能分割后端。

TAPAS 的视觉候选生成器属于外部感知系统，本模块只实现可以独立复现的运动学
分割与跨演示边界对齐：利用末端平移/旋转速度低谷和夹爪变化寻找技能边界，
不再把固定归一化时间切段冒充 TAPAS。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from essay2608.policy.dynamac import quaternion_conjugate, quaternion_log, quaternion_multiply


@dataclass(frozen=True)
class TapasSegmentationConfig:
    """可审计的运动学分割参数；论文未唯一指定这些数值，必须写入 provenance。"""

    minimum_skill_length: int = 12
    maximum_skills: int = 3
    velocity_quantile: float = 0.35
    smoothing_window: int = 9
    gripper_change_quantile: float = 0.80

    def __post_init__(self) -> None:
        if self.minimum_skill_length < 2:
            raise ValueError("minimum_skill_length 必须至少为 2")
        if self.maximum_skills < 1:
            raise ValueError("maximum_skills 必须为正")
        if not 0.0 < self.velocity_quantile < 1.0:
            raise ValueError("velocity_quantile 必须位于 (0, 1)")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window 必须为正")
        if not 0.0 < self.gripper_change_quantile <= 1.0:
            raise ValueError("gripper_change_quantile 必须位于 (0, 1]")


def _validate_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2:
        raise ValueError("末端轨迹必须为长度至少 2 的 [T,7] 位姿数组")
    if not np.all(np.isfinite(poses)):
        raise ValueError("末端轨迹不能包含非有限数值")
    return poses


def _robust_scale(values: np.ndarray) -> float:
    positive = values[values > 1.0e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def _motion_score(poses: np.ndarray, gripper: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    rotation = np.asarray(
        [
            np.linalg.norm(
                quaternion_log(
                    quaternion_multiply(
                        quaternion_conjugate(poses[index, 3:7]), poses[index + 1, 3:7]
                    )
                )
            )
            for index in range(len(poses) - 1)
        ],
        dtype=np.float64,
    )
    score = translation / _robust_scale(translation) + rotation / _robust_scale(rotation)
    grip_change = np.zeros(len(poses), dtype=np.float64)
    if gripper is not None:
        gripper = np.asarray(gripper, dtype=np.float64)
        if gripper.ndim != 2 or len(gripper) != len(poses):
            raise ValueError("夹爪轨迹必须为 [T,G] 且与末端轨迹等长")
        grip_change[1:] = np.linalg.norm(np.diff(gripper, axis=0), axis=1)
        score = score + 0.15 * grip_change[1:] / _robust_scale(grip_change[1:])
    return np.concatenate(([score[0]], score)), grip_change


def tapas_skill_boundaries(
    poses: np.ndarray,
    gripper: np.ndarray | None = None,
    config: TapasSegmentationConfig = TapasSegmentationConfig(),
) -> tuple[int, ...]:
    """根据速度低谷和夹爪变化返回 ``[0, ..., T]`` 技能边界。

    边界数量由轨迹长度和配置上限确定；候选位置来自观测运动，而不是固定时间比例。
    """

    poses = _validate_poses(poses)
    length = len(poses)
    segment_capacity = length // config.minimum_skill_length
    target_segments = min(config.maximum_skills, max(1, segment_capacity))
    if target_segments == 1:
        return (0, length)

    score, grip_change = _motion_score(poses, gripper)
    window = min(config.smoothing_window, length)
    if window % 2 == 0:
        window -= 1
    if window > 1:
        score = np.convolve(score, np.ones(window) / window, mode="same")
    low_threshold = float(np.quantile(score[1:-1], config.velocity_quantile))
    candidates: list[tuple[float, int]] = []
    for index in range(config.minimum_skill_length, length - config.minimum_skill_length + 1):
        local = score[index - 1 : index + 2]
        if score[index] <= low_threshold and score[index] <= float(np.min(local)):
            candidates.append((float(score[index]), index))

    if gripper is not None and np.any(grip_change > 0.0):
        grip_threshold = float(np.quantile(grip_change[grip_change > 0.0], config.gripper_change_quantile))
        for index in np.flatnonzero(grip_change >= grip_threshold):
            index = int(index)
            if config.minimum_skill_length <= index <= length - config.minimum_skill_length:
                candidates.append((-float(grip_change[index]), index))

    selected: list[int] = []
    for _, index in sorted(candidates):
        if all(abs(index - previous) >= config.minimum_skill_length for previous in selected):
            selected.append(index)
            if len(selected) == target_segments - 1:
                break

    # 运动过于平滑时仍然需要一个确定的分割；从各等长分区中选择最低速点，
    # 只作为候选不足时的退化策略，且位置仍由观测速度决定。
    for partition in range(1, target_segments):
        if len(selected) == target_segments - 1:
            break
        start = max(config.minimum_skill_length, partition * length // target_segments)
        end = min(length - config.minimum_skill_length, (partition + 1) * length // target_segments)
        if end <= start:
            continue
        index = int(start + np.argmin(score[start:end]))
        if all(abs(index - previous) >= config.minimum_skill_length for previous in selected):
            selected.append(index)
    selected.sort()
    return tuple([0, *selected, length])


def tapas_skill_labels(
    poses: np.ndarray,
    gripper: np.ndarray | None = None,
    config: TapasSegmentationConfig = TapasSegmentationConfig(),
) -> np.ndarray:
    """把 ``tapas_skill_boundaries`` 转换为连续整数技能标签。"""

    boundaries = tapas_skill_boundaries(poses, gripper, config)
    labels = np.empty(len(poses), dtype=np.int64)
    for label, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        labels[start:end] = label
    return labels


__all__ = ["TapasSegmentationConfig", "tapas_skill_boundaries", "tapas_skill_labels"]
