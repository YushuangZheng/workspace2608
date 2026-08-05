"""论文 DynaMAC 的自包含复现。

实现对应论文 Algorithm 1 与公式 (1)--(6)：

* 在 :math:`R^3 \\times S^3` 上把末端轨迹变换到任务参数坐标系；
* 以离散时间黎曼高斯（DiGaP）或其混合（MiDiGaP）拟合每条流；
* 用式 (5) 从演示离线识别技能级运动学链接；
* 在每个技能起点建立并累积冻结的虚拟末端坐标系；
* 用式 (6) 选择任务参数；
* 推理时把各 marginal 变回世界系，再在共同切空间做 PoE；
* 双臂由两套独立 DynaMAC 并发组成，对侧末端只是候选任务参数。

论文没有定义测试时在线链接重判、接触恢复或事件驱动技能识别。本实现也不会偷偷加入
这些扩展：链接与流集合只在 ``fit`` 时确定，推理阶段仅更新保留帧的当前位姿，并按离散
时间索引切换技能。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Array = np.ndarray
IDENTITY_POSE = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
MODEL_SCHEMA_VERSION = 2


def _as_float_array(value: Array | Sequence[float]) -> Array:
    return np.asarray(value, dtype=np.float64)


def normalize_quaternion(quaternion: Array) -> Array:
    """归一化 wxyz 四元数，并拒绝零四元数。"""

    quaternion = _as_float_array(quaternion)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("四元数范数必须非零")
    return quaternion / norm


def quaternion_conjugate(quaternion: Array) -> Array:
    result = normalize_quaternion(quaternion).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: Array, right: Array) -> Array:
    """计算 wxyz Hamilton 积，支持 NumPy 广播。"""

    left = _as_float_array(left)
    right = _as_float_array(right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_to_matrix(quaternion: Array) -> Array:
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def rotate_vector(quaternion: Array, vector: Array) -> Array:
    return np.einsum("...ij,...j->...i", quaternion_to_matrix(quaternion), vector)


def quaternion_log(quaternion: Array) -> Array:
    """把单位四元数映射为最短旋转向量。"""

    quaternion = normalize_quaternion(quaternion)
    quaternion = np.where(
        (quaternion[..., 0] < 0.0)[..., None],
        -quaternion,
        quaternion,
    )
    vector = quaternion[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quaternion[..., 0], 0.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1.0e-12,
    )
    return vector * scale[..., None]


def quaternion_exp(rotation_vector: Array) -> Array:
    rotation_vector = _as_float_array(rotation_vector)
    angle = np.linalg.norm(rotation_vector, axis=-1)
    half = 0.5 * angle
    scale = np.divide(
        np.sin(half),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1.0e-12,
    )
    return normalize_quaternion(
        np.concatenate((np.cos(half)[..., None], rotation_vector * scale[..., None]), axis=-1)
    )


def quaternion_mean(quaternions: Array) -> Array:
    """Markley 均值；符号翻转不改变结果。"""

    quaternions = normalize_quaternion(quaternions)
    accumulator = np.einsum("ni,nj->ij", quaternions, quaternions)
    _, eigenvectors = np.linalg.eigh(accumulator)
    result = eigenvectors[:, -1]
    if result[0] < 0.0:
        result *= -1.0
    return normalize_quaternion(result)


def pose_compose(left: Array, right: Array) -> Array:
    """SE(3) 左作用；位姿格式为 ``[x, y, z, qw, qx, qy, qz]``。"""

    left = _as_float_array(left)
    right = _as_float_array(right)
    position = left[..., :3] + rotate_vector(left[..., 3:7], right[..., :3])
    orientation = normalize_quaternion(quaternion_multiply(left[..., 3:7], right[..., 3:7]))
    return np.concatenate((position, orientation), axis=-1)


def pose_inverse(pose: Array) -> Array:
    pose = _as_float_array(pose)
    orientation = quaternion_conjugate(pose[..., 3:7])
    position = -rotate_vector(orientation, pose[..., :3])
    return np.concatenate((position, orientation), axis=-1)


def relative_pose(frame_pose: Array, world_pose: Array) -> Array:
    """公式 (1)：把世界位姿写到 ``frame_pose`` 局部坐标系。"""

    return pose_compose(pose_inverse(frame_pose), world_pose)


def pose_log_world(base: Array, point: Array) -> Array:
    """在 ``base`` 处取 Log，并用世界轴表示六维切向量。"""

    base = _as_float_array(base)
    point = _as_float_array(point)
    local_rotation = quaternion_log(
        quaternion_multiply(quaternion_conjugate(base[3:7]), point[3:7])
    )
    rotation_world = rotate_vector(base[3:7], local_rotation)
    return np.concatenate((point[:3] - base[:3], rotation_world))


def pose_exp_world(base: Array, tangent: Array) -> Array:
    """从世界轴六维切向量返回 :math:`R^3\\times S^3`。"""

    base = _as_float_array(base)
    tangent = _as_float_array(tangent)
    local_rotation = rotate_vector(quaternion_conjugate(base[3:7]), tangent[3:])
    orientation = quaternion_multiply(base[3:7], quaternion_exp(local_rotation))
    return np.concatenate((base[:3] + tangent[:3], normalize_quaternion(orientation)))


def interpolate_rows(values: Array, length: int) -> Array:
    values = _as_float_array(values)
    if length < 1:
        raise ValueError("重采样长度必须为正")
    if len(values) == 1:
        return np.repeat(values, length, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    flat = values.reshape(len(values), -1)
    result = np.stack(
        [np.interp(target, source, flat[:, index]) for index in range(flat.shape[1])],
        axis=-1,
    )
    return result.reshape((length,) + values.shape[1:])


def _quaternion_slerp(left: Array, right: Array, fraction: float) -> Array:
    left = normalize_quaternion(left)
    right = normalize_quaternion(right)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(left + fraction * (right - left))
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return normalize_quaternion(
        math.sin((1.0 - fraction) * angle) / denominator * left
        + math.sin(fraction * angle) / denominator * right
    )


def interpolate_poses(poses: Array, length: int) -> Array:
    poses = _as_float_array(poses)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("位姿轨迹必须为 [T, 7]")
    position = interpolate_rows(poses[:, :3], length)
    source = np.linspace(0.0, 1.0, len(poses))
    target = np.linspace(0.0, 1.0, length)
    quaternion = []
    for value in target:
        right_index = min(int(np.searchsorted(source, value, side="right")), len(poses) - 1)
        left_index = max(right_index - 1, 0)
        width = source[right_index] - source[left_index]
        fraction = 0.0 if width == 0.0 else float((value - source[left_index]) / width)
        quaternion.append(
            _quaternion_slerp(
                poses[left_index, 3:7],
                poses[right_index, 3:7],
                fraction,
            )
        )
    quaternion = np.stack(quaternion)
    return np.concatenate((position, quaternion), axis=-1)


def _pose_mean(poses: Array) -> Array:
    """在 R3 × S3 上迭代求 Karcher/Fréchet 均值。"""

    poses = _as_float_array(poses)
    mean = np.concatenate((np.mean(poses[:, :3], axis=0), quaternion_mean(poses[:, 3:7])))
    for _ in range(64):
        increment = np.mean(_pose_residuals(mean, poses), axis=0)
        mean = pose_exp_world(mean, increment)
        if np.linalg.norm(increment) < 1.0e-12:
            break
    return mean


def _pose_residuals(mean: Array, poses: Array) -> Array:
    return np.stack([pose_log_world(mean, pose) for pose in poses])


def _fit_pose_sequence(
    trajectories: Array,
    position_variance_floor: float,
    rotation_variance_floor: float,
) -> tuple[Array, Array]:
    """MiDiGaP 式 (6)：逐时刻 Fréchet 均值与对角切空间协方差。"""

    trajectories = _as_float_array(trajectories)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 7:
        raise ValueError("轨迹批次必须为 [N, T, 7]")
    means = np.zeros((trajectories.shape[1], 7), dtype=np.float64)
    covariance = np.zeros((trajectories.shape[1], 6, 6), dtype=np.float64)
    floor = np.asarray(
        [position_variance_floor] * 3 + [rotation_variance_floor] * 3,
        dtype=np.float64,
    )
    for time_index in range(trajectories.shape[1]):
        means[time_index] = _pose_mean(trajectories[:, time_index])
        residuals = _pose_residuals(means[time_index], trajectories[:, time_index])
        denominator = max(len(residuals) - 1, 1)
        diagonal = np.sum(np.square(residuals), axis=0) / denominator + floor
        covariance[time_index] = np.diag(diagonal)
    return means, covariance


def geometric_mean_standard_deviation(covariance: Array) -> Array:
    """DynaMAC 式 (5)，等价于 ``det(Sigma) ** (1 / 12)``。"""

    covariance = _as_float_array(covariance)
    if covariance.shape[-2:] != (6, 6):
        raise ValueError("式 (5) 需要 6x6 位姿协方差")
    sign, log_determinant = np.linalg.slogdet(covariance)
    if np.any(sign <= 0.0):
        raise ValueError("协方差必须正定")
    return np.exp(log_determinant / 12.0)


def task_parameter_scores(covariances: dict[str, Array]) -> dict[str, float]:
    """DynaMAC 式 (6)：对每条流取时间最大相对精度。"""

    if not covariances:
        raise ValueError("至少需要一个候选任务参数")
    names = list(covariances)
    values = np.stack([_as_float_array(covariances[name]) for name in names])
    if values.ndim != 4 or values.shape[-2:] != (6, 6):
        raise ValueError("候选协方差必须具有 [F, T, 6, 6] 形状")
    signs, log_determinants = np.linalg.slogdet(values)
    if np.any(signs <= 0.0):
        raise ValueError("协方差必须正定")
    log_precision = -log_determinants
    maximum = np.max(log_precision, axis=0, keepdims=True)
    relative = np.exp(log_precision - maximum)
    relative /= np.sum(relative, axis=0, keepdims=True)
    return {name: float(np.max(relative[index])) for index, name in enumerate(names)}


@dataclass(frozen=True)
class GaussianMarginal:
    """已经变换到世界系的一条高斯 marginal。"""

    frame: str
    mean: Array
    covariance: Array


def transform_marginal(
    frame_name: str,
    frame_pose: Array,
    local_mean: Array,
    local_covariance: Array,
) -> GaussianMarginal:
    """公式 (2)：用当前任务参数位姿把局部 marginal 变回世界系。"""

    frame_pose = _as_float_array(frame_pose)
    mean = pose_compose(frame_pose, local_mean)
    rotation = quaternion_to_matrix(frame_pose[3:7])
    tangent_rotation = np.zeros((6, 6), dtype=np.float64)
    tangent_rotation[:3, :3] = rotation
    tangent_rotation[3:, 3:] = rotation
    covariance = tangent_rotation @ local_covariance @ tangent_rotation.T
    return GaussianMarginal(frame_name, mean, covariance)


def product_of_experts(
    marginals: Sequence[GaussianMarginal],
    maximum_iterations: int = 32,
    tolerance: float = 1.0e-10,
) -> tuple[Array, Array, dict[str, float]]:
    """公式 (3)：在共同世界切空间融合黎曼高斯专家。

    ``joint_covariance @ information`` 是高斯乘积的闭式均值增量；这里通过
    Log/Exp 迭代把同一公式用于 :math:`R^3\\times S^3`。
    """

    if not marginals:
        raise ValueError("PoE 至少需要一个 marginal")
    precisions = [np.linalg.inv(item.covariance) for item in marginals]
    total_precision = np.sum(precisions, axis=0)
    joint_covariance = np.linalg.inv(total_precision)
    determinant_scores = np.asarray(
        [math.exp(-np.linalg.slogdet(item.covariance)[1]) for item in marginals]
    )
    initial = int(np.argmax(determinant_scores))
    mean = marginals[initial].mean.copy()
    for _ in range(maximum_iterations):
        information = np.sum(
            [
                precision @ pose_log_world(mean, marginal.mean)
                for precision, marginal in zip(precisions, marginals, strict=True)
            ],
            axis=0,
        )
        increment = joint_covariance @ information
        mean = pose_exp_world(mean, increment)
        if np.linalg.norm(increment) < tolerance:
            break
    weights = determinant_scores / np.sum(determinant_scores)
    return (
        mean,
        joint_covariance,
        {item.frame: float(weight) for item, weight in zip(marginals, weights, strict=True)},
    )


@dataclass(frozen=True)
class DynaMACDemonstration:
    """一条单智能体演示；所有位姿均为世界系 wxyz。"""

    ee_pose: Array
    action_pose: Array
    gripper: Array
    frames: dict[str, Array]
    skill: Array
    name: str = "demonstration"

    def __post_init__(self) -> None:
        ee_pose = _as_float_array(self.ee_pose)
        action_pose = _as_float_array(self.action_pose)
        gripper = _as_float_array(self.gripper)
        skill = np.asarray(self.skill, dtype=np.int64)
        frames = {name: _as_float_array(value) for name, value in self.frames.items()}
        steps = len(ee_pose)
        if ee_pose.shape != (steps, 7) or action_pose.shape != (steps, 7):
            raise ValueError(f"{self.name} 的末端/动作位姿必须为 [T, 7]")
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if gripper.shape[0] != steps or skill.shape != (steps,):
            raise ValueError(f"{self.name} 的数组长度不一致")
        if not frames or any(value.shape != (steps, 7) for value in frames.values()):
            raise ValueError(f"{self.name} 的任务参数必须为非空 [T, 7] 位姿字典")
        sequence = _compressed_skill_sequence(skill)
        if len(sequence) != len(set(sequence)):
            raise ValueError(f"{self.name} 的同一技能不能分成多个不连续区间")
        for poses in [ee_pose, action_pose, *frames.values()]:
            if not np.all(np.isfinite(poses)):
                raise ValueError(f"{self.name} 含非有限位姿")
            normalize_quaternion(poses[:, 3:7])
        object.__setattr__(self, "ee_pose", ee_pose)
        object.__setattr__(self, "action_pose", action_pose)
        object.__setattr__(self, "gripper", gripper)
        object.__setattr__(self, "skill", skill)
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class DynaMACObservation:
    ee_pose: Array
    frames: dict[str, Array]

    def __post_init__(self) -> None:
        ee_pose = _as_float_array(self.ee_pose)
        frames = {name: _as_float_array(value) for name, value in self.frames.items()}
        if ee_pose.shape != (7,) or any(value.shape != (7,) for value in frames.values()):
            raise ValueError("观测位姿必须为 [7]")
        object.__setattr__(self, "ee_pose", ee_pose)
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class DynaMACAction:
    pose: Array
    gripper: Array
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DynaMACConfig:
    """公开阈值与论文未指定、但复现必须显式冻结的数值选择。"""

    tau_m: float = 0.001
    tau_omega: float = 0.2
    link_fraction_threshold: float = 0.5
    minimum_link_run: int = 3
    position_variance_floor: float = 1.0e-8
    rotation_variance_floor: float = 1.0e-8
    maximum_modes: int = 3
    minimum_mode_size: int = 2
    clustering_length: int = 20
    random_seed: int = 2608

    def __post_init__(self) -> None:
        if not 0.0 < self.tau_m or not 0.0 <= self.tau_omega < 1.0:
            raise ValueError("tau_m/tau_omega 非法")
        if not 0.0 <= self.link_fraction_threshold <= 1.0:
            raise ValueError("link_fraction_threshold 必须位于 [0, 1]")
        if self.minimum_link_run < 1 or self.maximum_modes < 1:
            raise ValueError("窗口和模态数必须为正")
        if self.minimum_mode_size < 1 or self.clustering_length < 2:
            raise ValueError("模态最小样本数/聚类长度非法")


@dataclass
class StreamModel:
    frame: str
    mean: Array  # [M, T, 7]
    covariance: Array  # [M, T, 6, 6]


@dataclass
class SkillModel:
    label: int
    duration: int
    selected_frames: tuple[str, ...]
    mode_priors: Array
    streams: dict[str, StreamModel]
    gripper: Array  # [M, T, G]
    transition_from_previous: Array | None = None
    link_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    selection_scores: dict[str, float] = field(default_factory=dict)


def _compressed_skill_sequence(skill: Array) -> list[int]:
    if len(skill) == 0:
        return []
    result = [int(skill[0])]
    for value in skill[1:]:
        value = int(value)
        if value != result[-1]:
            result.append(value)
    return result


def _maximum_true_run(mask: Array) -> int:
    maximum = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _validate_demonstrations(
    demonstrations: Sequence[DynaMACDemonstration],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not demonstrations:
        raise ValueError("至少需要一条演示")
    frame_names = tuple(sorted(demonstrations[0].frames))
    skill_sequence = tuple(_compressed_skill_sequence(demonstrations[0].skill))
    if not skill_sequence:
        raise ValueError("演示没有技能")
    for demonstration in demonstrations[1:]:
        if tuple(sorted(demonstration.frames)) != frame_names:
            raise ValueError("所有演示必须包含相同任务参数")
        if tuple(_compressed_skill_sequence(demonstration.skill)) != skill_sequence:
            raise ValueError("所有演示必须具有相同技能顺序")
        if demonstration.gripper.shape[1] != demonstrations[0].gripper.shape[1]:
            raise ValueError("所有演示的夹爪维数必须一致")
    return frame_names, skill_sequence


def _skill_slice(demonstration: DynaMACDemonstration, label: int) -> Array:
    indices = np.flatnonzero(demonstration.skill == label)
    if not len(indices) or np.any(np.diff(indices) != 1):
        raise ValueError(f"{demonstration.name} 缺少连续技能 {label}")
    return indices


def _resampled_skill_data(
    demonstrations: Sequence[DynaMACDemonstration],
    label: int,
    duration: int,
    virtual_starts: dict[int, list[Array]],
) -> tuple[Array, Array, dict[str, Array], dict[str, Array]]:
    ee_trajectories = []
    action_trajectories = []
    grippers = []
    real_frames: dict[str, list[Array]] = {name: [] for name in demonstrations[0].frames}
    virtual_frames: dict[str, list[Array]] = {
        f"virtual_skill_{virtual_label}": [] for virtual_label in virtual_starts
    }
    for demo_index, demonstration in enumerate(demonstrations):
        indices = _skill_slice(demonstration, label)
        ee_trajectories.append(interpolate_poses(demonstration.ee_pose[indices], duration))
        action_trajectories.append(interpolate_poses(demonstration.action_pose[indices], duration))
        grippers.append(interpolate_rows(demonstration.gripper[indices], duration))
        for name, poses in demonstration.frames.items():
            real_frames[name].append(interpolate_poses(poses[indices], duration))
        for virtual_label, starts in virtual_starts.items():
            pose = starts[demo_index]
            virtual_frames[f"virtual_skill_{virtual_label}"].append(
                np.repeat(pose[None], duration, axis=0)
            )
    frames = {name: np.stack(values) for name, values in {**real_frames, **virtual_frames}.items()}
    return (
        np.stack(ee_trajectories),
        np.stack(action_trajectories),
        frames,
        {"gripper": np.stack(grippers)},
    )


def _local_trajectories(frame_trajectories: Array, poses: Array) -> Array:
    return np.stack(
        [
            relative_pose(frame_demo, pose_demo)
            for frame_demo, pose_demo in zip(frame_trajectories, poses, strict=True)
        ]
    )


def _trajectory_distances(trajectories: Array, centres: Array) -> Array:
    """M^T 上逐时刻测地距离平方之和。"""

    distances = np.empty((len(trajectories), len(centres)), dtype=np.float64)
    for trajectory_index, trajectory in enumerate(trajectories):
        for centre_index, centre in enumerate(centres):
            residuals = np.stack(
                [
                    pose_log_world(centre_pose, trajectory_pose)
                    for centre_pose, trajectory_pose in zip(centre, trajectory, strict=True)
                ]
            )
            distances[trajectory_index, centre_index] = float(np.sum(np.square(residuals)))
    return distances


def _deterministic_riemannian_kmeans(
    trajectories: Array,
    clusters: int,
) -> tuple[Array, Array, float]:
    """MiDiGaP Sec. IV-B 的 Riemannian k-means，使用确定性最远点初始化。"""

    centre_indices = [0]
    while len(centre_indices) < clusters:
        distance = np.min(_trajectory_distances(trajectories, trajectories[centre_indices]), axis=1)
        distance[centre_indices] = -1.0
        centre_indices.append(int(np.argmax(distance)))
    centres = trajectories[centre_indices].copy()
    labels = np.zeros(len(trajectories), dtype=np.int64)
    for iteration in range(100):
        distances = _trajectory_distances(trajectories, centres)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels) and iteration:
            break
        labels = updated
        if any(np.sum(labels == index) == 0 for index in range(clusters)):
            return labels, centres, float("inf")
        centres = np.stack(
            [
                np.stack(
                    [
                        _pose_mean(trajectories[labels == index, time_index])
                        for time_index in range(trajectories.shape[1])
                    ]
                )
                for index in range(clusters)
            ]
        )
    distances = _trajectory_distances(trajectories, centres)
    residual = float(np.sum(distances[np.arange(len(trajectories)), labels]))
    return labels, centres, residual


def _partition_modes(local_trajectories: Array, config: DynaMACConfig) -> Array:
    """MiDiGaP 的 M^T 轨迹级 Riemannian k-means+BIC 模态划分。"""

    trajectories = np.stack(
        [
            interpolate_poses(trajectory, config.clustering_length)
            for trajectory in local_trajectories
        ]
    )
    samples = len(trajectories)
    dimension = trajectories.shape[1] * 6
    maximum = min(config.maximum_modes, samples // config.minimum_mode_size)
    best_bic = float("inf")
    best_labels = np.zeros(samples, dtype=np.int64)
    for clusters in range(1, maximum + 1):
        labels, _, residual = _deterministic_riemannian_kmeans(trajectories, clusters)
        counts = np.asarray([np.sum(labels == index) for index in range(clusters)])
        if np.any(counts < config.minimum_mode_size) or not np.isfinite(residual):
            continue
        variance = max(residual / max(samples * dimension, 1), 1.0e-12)
        log_likelihood = -0.5 * samples * dimension * (math.log(2.0 * math.pi * variance) + 1.0)
        log_likelihood += float(np.sum(counts * np.log(counts / samples)))
        parameters = clusters * dimension + clusters - 1
        bic = -2.0 * log_likelihood + parameters * math.log(samples)
        if bic < best_bic:
            best_bic = bic
            best_labels = labels.copy()
    # 稳定重编号，使 checkpoint 与演示输入顺序确定。
    unique = sorted(
        np.unique(best_labels),
        key=lambda value: int(np.flatnonzero(best_labels == value)[0]),
    )
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in best_labels], dtype=np.int64)


def _transition_probabilities(previous: Array, current: Array) -> Array:
    """MiDiGaP 式 (12)：用演示集合交集估计相邻技能模态转移。"""

    previous = np.asarray(previous, dtype=np.int64)
    current = np.asarray(current, dtype=np.int64)
    if previous.shape != current.shape or previous.ndim != 1:
        raise ValueError("相邻技能的模态标签必须是一一对应的一维演示标签")
    result = np.zeros((int(np.max(previous)) + 1, int(np.max(current)) + 1))
    for source in range(result.shape[0]):
        source_mask = previous == source
        denominator = int(np.sum(source_mask))
        if denominator == 0:
            raise ValueError("前一技能存在空模态")
        for target in range(result.shape[1]):
            result[source, target] = np.sum(source_mask & (current == target)) / denominator
    return result


class DynaMAC:
    """论文忠实的单智能体 DynaMAC/Task-Parameterized MiDiGaP。"""

    name = "dynamac"

    def __init__(self, config: DynaMACConfig = DynaMACConfig()) -> None:
        self.config = config
        self.frame_names: tuple[str, ...] = ()
        self.skill_sequence: tuple[int, ...] = ()
        self.skills: list[SkillModel] = []
        self._skill_index = 0
        self._time_index = 0
        self._virtual_frames: dict[str, Array] = {}
        self._pending_virtual_capture = False
        self._mode_strategy: Literal["map", "sample"] = "map"
        self._mode_path: tuple[int, ...] = ()
        self._active_mode = 0
        self._complete = False
        self._rng = np.random.default_rng(config.random_seed)

    @property
    def fitted(self) -> bool:
        return bool(self.skills)

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def current_skill(self) -> SkillModel:
        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        return self.skills[self._skill_index]

    def fit(self, demonstrations: Sequence[DynaMACDemonstration]) -> DynaMAC:
        """执行 Algorithm 1；所有链接判断与流选择都发生在这里。"""

        self.frame_names, self.skill_sequence = _validate_demonstrations(demonstrations)
        self.skills = []
        virtual_starts: dict[int, list[Array]] = {}
        previous_mode_labels: Array | None = None
        for label in self.skill_sequence:
            # 虚拟帧只在对应技能开始时可观测。训练当前技能时把其它技能的
            # 起始位姿也塞进候选集，会让离线选择出推理时尚未捕获的未来帧，
            # 最终在 act() 中触发“虚拟帧尚未捕获”。
            virtual_starts[label] = [
                demonstration.ee_pose[_skill_slice(demonstration, label)[0]].copy()
                for demonstration in demonstrations
            ]
            lengths = [len(_skill_slice(demonstration, label)) for demonstration in demonstrations]
            duration = max(int(round(float(np.mean(lengths)))), 1)
            ee, actions, frames, extra = _resampled_skill_data(
                demonstrations, label, duration, {label: virtual_starts[label]}
            )

            local_ee = {
                name: _local_trajectories(frame_values, ee) for name, frame_values in frames.items()
            }
            local_actions = {
                name: _local_trajectories(frame_values, actions)
                for name, frame_values in frames.items()
            }
            policy_covariance: dict[str, Array] = {}
            for name, local in local_actions.items():
                _, policy_covariance[name] = _fit_pose_sequence(
                    local,
                    self.config.position_variance_floor,
                    self.config.rotation_variance_floor,
                )
            link_covariance: dict[str, Array] = {}
            for name in self.frame_names:
                _, link_covariance[name] = _fit_pose_sequence(
                    local_ee[name],
                    self.config.position_variance_floor,
                    self.config.rotation_variance_floor,
                )

            link_diagnostics: dict[str, dict[str, Any]] = {}
            valid_real_frames: list[str] = []
            for name in self.frame_names:
                scale = geometric_mean_standard_deviation(link_covariance[name])
                linked_mask = scale < self.config.tau_m
                linked_fraction = float(np.mean(linked_mask))
                maximum_run = _maximum_true_run(linked_mask)
                linked = (
                    linked_fraction >= self.config.link_fraction_threshold
                    and maximum_run >= self.config.minimum_link_run
                )
                link_diagnostics[name] = {
                    "linked": linked,
                    "linked_fraction": linked_fraction,
                    "maximum_link_run": maximum_run,
                    "minimum_m": float(np.min(scale)),
                    "median_m": float(np.median(scale)),
                }
                if not linked:
                    valid_real_frames.append(name)

            candidate_frames = [
                *valid_real_frames,
                f"virtual_skill_{label}",
            ]
            scores = task_parameter_scores(
                {name: policy_covariance[name] for name in candidate_frames}
            )
            selected = tuple(
                name for name in candidate_frames if scores[name] > self.config.tau_omega
            )
            if not selected:
                raise RuntimeError(
                    f"技能 {label} 没有任务参数通过 tau_omega={self.config.tau_omega}；"
                    "请修正演示覆盖或预注册阈值，不能静默回退"
                )

            clustering_frame = f"virtual_skill_{label}"
            mode_labels = _partition_modes(local_actions[clustering_frame], self.config)
            modes = int(np.max(mode_labels)) + 1
            priors = np.asarray(
                [np.mean(mode_labels == mode) for mode in range(modes)], dtype=np.float64
            )
            transition = (
                None
                if previous_mode_labels is None
                else _transition_probabilities(previous_mode_labels, mode_labels)
            )
            streams: dict[str, StreamModel] = {}
            gripper_models = []
            for mode in range(modes):
                members = mode_labels == mode
                gripper_models.append(np.mean(extra["gripper"][members], axis=0))
            for name in selected:
                means = []
                covariances = []
                for mode in range(modes):
                    members = mode_labels == mode
                    mean, covariance = _fit_pose_sequence(
                        local_actions[name][members],
                        self.config.position_variance_floor,
                        self.config.rotation_variance_floor,
                    )
                    means.append(mean)
                    covariances.append(covariance)
                streams[name] = StreamModel(name, np.stack(means), np.stack(covariances))
            self.skills.append(
                SkillModel(
                    label=label,
                    duration=duration,
                    selected_frames=selected,
                    mode_priors=priors,
                    streams=streams,
                    gripper=np.stack(gripper_models),
                    transition_from_previous=transition,
                    link_diagnostics=link_diagnostics,
                    selection_scores=scores,
                )
            )
            previous_mode_labels = mode_labels
        return self

    def _select_mode_path(
        self,
        strategy: Literal["map", "sample"],
    ) -> tuple[int, ...]:
        """按 MiDiGaP 式 (12)--(13) 选择整条技能模态路径。"""

        if strategy == "sample":
            path = [
                int(self._rng.choice(len(self.skills[0].mode_priors), p=self.skills[0].mode_priors))
            ]
            for skill in self.skills[1:]:
                if skill.transition_from_previous is None:
                    raise RuntimeError("MiDiGaP 技能缺少模态转移矩阵")
                probabilities = skill.transition_from_previous[path[-1]]
                path.append(int(self._rng.choice(len(probabilities), p=probabilities)))
            return tuple(path)
        if strategy != "map":
            raise ValueError(f"未知模态选择策略：{strategy}")

        scores = np.log(np.clip(self.skills[0].mode_priors, 1.0e-300, None))
        backpointers: list[Array] = []
        for skill in self.skills[1:]:
            if skill.transition_from_previous is None:
                raise RuntimeError("MiDiGaP 技能缺少模态转移矩阵")
            transition = np.log(np.clip(skill.transition_from_previous, 1.0e-300, None))
            candidates = scores[:, None] + transition
            backpointers.append(np.argmax(candidates, axis=0))
            scores = np.max(candidates, axis=0)
        path = [int(np.argmax(scores))]
        for backpointer in reversed(backpointers):
            path.append(int(backpointer[path[-1]]))
        return tuple(reversed(path))

    def reset(
        self,
        observation: DynaMACObservation,
        mode_strategy: Literal["map", "sample"] = "map",
    ) -> None:
        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        missing = set(self.frame_names) - set(observation.frames)
        if missing:
            raise ValueError(f"观测缺少任务参数：{sorted(missing)}")
        self._skill_index = 0
        self._time_index = 0
        self._complete = False
        self._virtual_frames = {
            f"virtual_skill_{self.current_skill.label}": observation.ee_pose.copy()
        }
        self._pending_virtual_capture = False
        self._mode_strategy = mode_strategy
        self._mode_path = self._select_mode_path(mode_strategy)
        self._active_mode = self._mode_path[0]

    def _frame_pose(self, name: str, observation: DynaMACObservation) -> Array:
        if name.startswith("virtual_skill_"):
            if name not in self._virtual_frames:
                raise RuntimeError(f"虚拟帧 {name} 尚未在技能边界捕获")
            return self._virtual_frames[name]
        if name not in observation.frames:
            raise ValueError(f"观测缺少已选择任务参数 {name}")
        return observation.frames[name]

    def act(self, observation: DynaMACObservation) -> DynaMACAction:
        """按固定离散时间执行；不读取接触或在线链接状态。"""

        if self._complete:
            raise RuntimeError("DynaMAC 已完成")
        if self._pending_virtual_capture:
            virtual_name = f"virtual_skill_{self.current_skill.label}"
            self._virtual_frames[virtual_name] = observation.ee_pose.copy()
            self._pending_virtual_capture = False
        skill = self.current_skill
        index = min(self._time_index, skill.duration - 1)
        marginals = []
        for name in skill.selected_frames:
            stream = skill.streams[name]
            marginals.append(
                transform_marginal(
                    name,
                    self._frame_pose(name, observation),
                    stream.mean[self._active_mode, index],
                    stream.covariance[self._active_mode, index],
                )
            )
        pose, covariance, weights = product_of_experts(marginals)
        gripper = skill.gripper[self._active_mode, index].copy()
        diagnostics = {
            "method": self.name,
            "skill_index": self._skill_index,
            "skill_label": skill.label,
            "time_index": index,
            "duration": skill.duration,
            "mode": self._active_mode,
            "mode_prior": float(skill.mode_priors[self._active_mode]),
            "modal_path": list(self._mode_path),
            "path_probability_factor": (
                float(skill.mode_priors[self._active_mode])
                if self._skill_index == 0
                else float(
                    skill.transition_from_previous[
                        self._mode_path[self._skill_index - 1], self._active_mode
                    ]
                )
            ),
            "selected_frames": list(skill.selected_frames),
            "marginal_means": {item.frame: item.mean.tolist() for item in marginals},
            "poe_weights": weights,
            "joint_covariance": covariance.tolist(),
            "selection_mode": "offline_skill_fixed",
            "online_link_detection": False,
        }
        self._time_index += 1
        if self._time_index >= skill.duration:
            if self._skill_index == len(self.skills) - 1:
                self._complete = True
            else:
                self._skill_index += 1
                self._time_index = 0
                # 下一技能的虚拟帧应取下一次 act 收到的技能起始观测，
                # 而不是上一技能最后一个控制周期的旧观测。
                self._pending_virtual_capture = True
                self._active_mode = self._mode_path[self._skill_index]
        return DynaMACAction(pose=pose, gripper=gripper, diagnostics=diagnostics)

    def summary(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        return {
            "implementation": "DynaMAC Algorithm 1 + task-parameterized MiDiGaP",
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "config": asdict(self.config),
            "frame_names": list(self.frame_names),
            "skill_sequence": list(self.skill_sequence),
            "skills": [
                {
                    "label": skill.label,
                    "duration": skill.duration,
                    "modes": len(skill.mode_priors),
                    "mode_priors": skill.mode_priors.tolist(),
                    "transition_from_previous": (
                        None
                        if skill.transition_from_previous is None
                        else skill.transition_from_previous.tolist()
                    ),
                    "selected_frames": list(skill.selected_frames),
                    "link_diagnostics": skill.link_diagnostics,
                    "selection_scores": skill.selection_scores,
                }
                for skill in self.skills
            ],
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.summary(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8"))
        for skill in self.skills:
            digest.update(skill.mode_priors.tobytes())
            digest.update(skill.gripper.tobytes())
            if skill.transition_from_previous is not None:
                digest.update(skill.transition_from_previous.tobytes())
            for name in skill.selected_frames:
                digest.update(skill.streams[name].mean.tobytes())
                digest.update(skill.streams[name].covariance.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _array_key(skill_index: int, frame: str, field_name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", frame)
        suffix = hashlib.sha256(frame.encode("utf-8")).hexdigest()[:8]
        return f"skill_{skill_index}__{safe}_{suffix}__{field_name}"

    def save(self, path: str | Path) -> None:
        """保存无 pickle 的单文件 checkpoint。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.summary()
        metadata["fingerprint"] = self.fingerprint()
        arrays: dict[str, Array] = {}
        for index, skill in enumerate(self.skills):
            arrays[f"skill_{index}__mode_priors"] = skill.mode_priors
            arrays[f"skill_{index}__gripper"] = skill.gripper
            if skill.transition_from_previous is not None:
                arrays[f"skill_{index}__transition"] = skill.transition_from_previous
            for name, stream in skill.streams.items():
                arrays[self._array_key(index, name, "mean")] = stream.mean
                arrays[self._array_key(index, name, "covariance")] = stream.covariance
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> DynaMAC:
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("model_schema_version") != MODEL_SCHEMA_VERSION:
                raise ValueError("不支持的 DynaMAC checkpoint schema")
            policy = cls(DynaMACConfig(**metadata["config"]))
            policy.frame_names = tuple(metadata["frame_names"])
            policy.skill_sequence = tuple(int(value) for value in metadata["skill_sequence"])
            for index, skill_meta in enumerate(metadata["skills"]):
                selected = tuple(skill_meta["selected_frames"])
                streams = {}
                for name in selected:
                    streams[name] = StreamModel(
                        name,
                        archive[policy._array_key(index, name, "mean")].copy(),
                        archive[policy._array_key(index, name, "covariance")].copy(),
                    )
                policy.skills.append(
                    SkillModel(
                        label=int(skill_meta["label"]),
                        duration=int(skill_meta["duration"]),
                        selected_frames=selected,
                        mode_priors=archive[f"skill_{index}__mode_priors"].copy(),
                        streams=streams,
                        gripper=archive[f"skill_{index}__gripper"].copy(),
                        transition_from_previous=(
                            None if index == 0 else archive[f"skill_{index}__transition"].copy()
                        ),
                        link_diagnostics=skill_meta["link_diagnostics"],
                        selection_scores={
                            name: float(value)
                            for name, value in skill_meta["selection_scores"].items()
                        },
                    )
                )
        if policy.fingerprint() != metadata.get("fingerprint"):
            raise ValueError("DynaMAC checkpoint 指纹不一致")
        return policy


@dataclass(frozen=True)
class BimanualDynaMACAction:
    left: DynaMACAction
    right: DynaMACAction


class BimanualDynaMAC:
    """论文 Sec. III-C：两套并发 DynaMAC，不设联合策略或固定 leader。"""

    def __init__(
        self,
        left: DynaMAC | None = None,
        right: DynaMAC | None = None,
        config: DynaMACConfig = DynaMACConfig(),
    ) -> None:
        self.left = left if left is not None else DynaMAC(config)
        self.right = right if right is not None else DynaMAC(config)

    def fit(
        self,
        left_demonstrations: Sequence[DynaMACDemonstration],
        right_demonstrations: Sequence[DynaMACDemonstration],
    ) -> BimanualDynaMAC:
        if len(left_demonstrations) != len(right_demonstrations):
            raise ValueError("左右臂演示数量必须一致")
        if any("right_ee" not in demo.frames for demo in left_demonstrations):
            raise ValueError("左臂候选任务参数必须包含 right_ee")
        if any("left_ee" not in demo.frames for demo in right_demonstrations):
            raise ValueError("右臂候选任务参数必须包含 left_ee")
        self.left.fit(left_demonstrations)
        self.right.fit(right_demonstrations)
        if self.left.skill_sequence != self.right.skill_sequence:
            raise ValueError("并发左右臂必须具有同一技能序列")
        if [skill.duration for skill in self.left.skills] != [
            skill.duration for skill in self.right.skills
        ]:
            raise ValueError("并发左右臂必须具有同步的技能时长")
        return self

    @property
    def complete(self) -> bool:
        return self.left.complete and self.right.complete

    def reset(
        self,
        left_observation: DynaMACObservation,
        right_observation: DynaMACObservation,
        mode_strategy: Literal["map", "sample"] = "map",
    ) -> None:
        self.left.reset(left_observation, mode_strategy)
        self.right.reset(right_observation, mode_strategy)

    def act(
        self,
        left_observation: DynaMACObservation,
        right_observation: DynaMACObservation,
    ) -> BimanualDynaMACAction:
        return BimanualDynaMACAction(
            left=self.left.act(left_observation),
            right=self.right.act(right_observation),
        )


# 清晰兼容名；不再把在线关系原型伪装成 DynaMAC。
DynaMACPolicy = DynaMAC


__all__ = [
    "BimanualDynaMAC",
    "BimanualDynaMACAction",
    "DynaMAC",
    "DynaMACAction",
    "DynaMACConfig",
    "DynaMACDemonstration",
    "DynaMACObservation",
    "DynaMACPolicy",
    "GaussianMarginal",
    "geometric_mean_standard_deviation",
    "interpolate_poses",
    "normalize_quaternion",
    "pose_compose",
    "pose_exp_world",
    "pose_inverse",
    "pose_log_world",
    "product_of_experts",
    "relative_pose",
    "task_parameter_scores",
    "transform_marginal",
]
