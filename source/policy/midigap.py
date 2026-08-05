"""MiDiGaP 的独立复现：多模态 DiGaP、约束更新、技能连接与 VAPOR。

位姿统一使用 ``[x, y, z, qw, qx, qy, qz]``，协方差是在世界轴表示的
``R3 × S3`` 六维切空间协方差。DynaMAC 会把 MiDiGaP 用在任务参数流内；本模块提供
论文中不依赖 DynaMAC 的完整 MiDiGaP 接口，包括式 (14)--(24) 和 VAPOR。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .dynamac import (
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    SkillModel,
    StreamModel,
    _fit_pose_sequence,
    _local_trajectories,
    _partition_modes,
    _pose_mean,
    _pose_residuals,
    _resampled_skill_data,
    _skill_slice,
    _transition_probabilities,
    _validate_demonstrations,
    interpolate_poses,
    pose_exp_world,
    pose_log_world,
    task_parameter_scores,
)

Array = np.ndarray


@dataclass(frozen=True)
class MiDiGaPConfig:
    """论文没有唯一指定的聚类与数值稳定参数。"""

    position_variance_floor: float = 1.0e-8
    rotation_variance_floor: float = 1.0e-8
    maximum_modes: int = 3
    minimum_mode_size: int = 2
    clustering_length: int = 20
    random_seed: int = 2608

    def __post_init__(self) -> None:
        if self.position_variance_floor <= 0.0 or self.rotation_variance_floor <= 0.0:
            raise ValueError("协方差下限必须为正")
        if self.maximum_modes < 1 or self.minimum_mode_size < 1:
            raise ValueError("模态数量参数必须为正")
        if self.clustering_length < 2:
            raise ValueError("聚类重采样长度至少为 2")

    def _dynamac_config(self) -> DynaMACConfig:
        return DynaMACConfig(
            position_variance_floor=self.position_variance_floor,
            rotation_variance_floor=self.rotation_variance_floor,
            maximum_modes=self.maximum_modes,
            minimum_mode_size=self.minimum_mode_size,
            clustering_length=self.clustering_length,
            random_seed=self.random_seed,
        )


@dataclass(frozen=True)
class MiDiGaPMode:
    mean: Array  # [T, 7]
    covariance: Array  # [T, 6, 6]
    prior: float
    demonstration_indices: tuple[int, ...]


class MiDiGaP:
    """MiDiGaP Sec. IV：在轨迹流形 ``M^T`` 上聚类后逐时刻拟合 DiGaP。"""

    name = "midigap"

    def __init__(self, config: MiDiGaPConfig = MiDiGaPConfig()) -> None:
        self.config = config
        self.duration = 0
        self.mode_labels = np.empty(0, dtype=np.int64)
        self.modes: tuple[MiDiGaPMode, ...] = ()
        self._rng = np.random.default_rng(config.random_seed)

    @property
    def fitted(self) -> bool:
        return bool(self.modes)

    @property
    def priors(self) -> Array:
        if not self.fitted:
            raise RuntimeError("MiDiGaP 尚未拟合")
        return np.asarray([mode.prior for mode in self.modes], dtype=np.float64)

    def fit(self, trajectories: Sequence[Array]) -> MiDiGaP:
        """拟合一组同语义、可变长度的末端位姿演示。"""

        values = [np.asarray(item, dtype=np.float64) for item in trajectories]
        if not values:
            raise ValueError("至少需要一条轨迹")
        if any(item.ndim != 2 or item.shape[1] != 7 or len(item) == 0 for item in values):
            raise ValueError("每条轨迹必须为非空 [T, 7]")
        self.duration = max(int(round(float(np.mean([len(item) for item in values])))), 1)
        resampled = np.stack([interpolate_poses(item, self.duration) for item in values])
        self.mode_labels = _partition_modes(resampled, self.config._dynamac_config())
        modes = []
        for label in range(int(np.max(self.mode_labels)) + 1):
            members = np.flatnonzero(self.mode_labels == label)
            mean, covariance = _fit_pose_sequence(
                resampled[members],
                self.config.position_variance_floor,
                self.config.rotation_variance_floor,
            )
            modes.append(
                MiDiGaPMode(
                    mean=mean,
                    covariance=covariance,
                    prior=float(len(members) / len(values)),
                    demonstration_indices=tuple(int(index) for index in members),
                )
            )
        self.modes = tuple(modes)
        return self

    def select_mode(self, strategy: str = "map") -> int:
        if strategy == "map":
            return int(np.argmax(self.priors))
        if strategy == "sample":
            return int(self._rng.choice(len(self.modes), p=self.priors))
        raise ValueError(f"未知模态选择策略：{strategy}")

    def most_likely_trajectory(self) -> Array:
        return self.modes[self.select_mode("map")].mean.copy()


class TaskParameterizedMiDiGaP(DynaMAC):
    """论文对比实验中的静态帧 MiDiGaP 策略。

    它和 DynaMAC 使用相同的任务参数流、MiDiGaP 模态与 PoE 推理，但不会检测运动学链接，
    也不会补充虚拟末端帧。因此任务对象在执行中被机器人带动时，会复现论文所述的
    exogeneity/causal-collapse 问题，而不是暗中获得 DynaMAC 的修复。
    """

    name = "midigap_static_frames"

    def fit(
        self,
        demonstrations: Sequence[DynaMACDemonstration],
    ) -> TaskParameterizedMiDiGaP:
        self.frame_names, self.skill_sequence = _validate_demonstrations(demonstrations)
        self.skills = []
        previous_mode_labels: Array | None = None
        for label in self.skill_sequence:
            lengths = [len(_skill_slice(item, label)) for item in demonstrations]
            duration = max(int(round(float(np.mean(lengths)))), 1)
            _, actions, frames, extra = _resampled_skill_data(
                demonstrations,
                label,
                duration,
                virtual_starts={},
            )
            local_actions = {
                name: _local_trajectories(frame_values, actions)
                for name, frame_values in frames.items()
            }
            fitted = {
                name: _fit_pose_sequence(
                    values,
                    self.config.position_variance_floor,
                    self.config.rotation_variance_floor,
                )
                for name, values in local_actions.items()
            }
            scores = task_parameter_scores(
                {name: covariance for name, (_, covariance) in fitted.items()}
            )
            selected = tuple(
                name for name in self.frame_names if scores[name] > self.config.tau_omega
            )
            if not selected:
                raise RuntimeError(
                    f"技能 {label} 没有静态任务参数通过 tau_omega={self.config.tau_omega}"
                )

            # 论文公开了 M^T 聚类，却没有唯一指定多流数据先在哪条流聚类；冻结为式 (6)
            # 得分最高的任务参数，避免把多条等价表示重复拼接进距离。
            clustering_frame = max(selected, key=scores.__getitem__)
            mode_labels = _partition_modes(local_actions[clustering_frame], self.config)
            mode_count = int(np.max(mode_labels)) + 1
            priors = np.asarray(
                [np.mean(mode_labels == mode) for mode in range(mode_count)],
                dtype=np.float64,
            )
            transition = (
                None
                if previous_mode_labels is None
                else _transition_probabilities(previous_mode_labels, mode_labels)
            )
            streams = {}
            gripper = []
            for mode in range(mode_count):
                members = mode_labels == mode
                gripper.append(np.mean(extra["gripper"][members], axis=0))
            for name in selected:
                means = []
                covariances = []
                for mode in range(mode_count):
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
                    label=int(label),
                    duration=duration,
                    selected_frames=selected,
                    mode_priors=priors,
                    streams=streams,
                    gripper=np.stack(gripper),
                    transition_from_previous=transition,
                    link_diagnostics={
                        name: {"linked": False, "reason": "static_frame_baseline"}
                        for name in self.frame_names
                    },
                    selection_scores=scores,
                )
            )
            previous_mode_labels = mode_labels
        return self


class PoseConstraint(Protocol):
    """MiDiGaP 可行域。"""

    supports_moment_matching: bool

    def contains(self, pose: Array) -> bool: ...

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool: ...


@dataclass(frozen=True)
class ReachabilitySphere:
    """式 (17) 的球形工作空间。"""

    center: Array
    radius: float
    supports_moment_matching: bool = True

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or self.radius <= 0.0:
            raise ValueError("可达球需要三维中心和正半径")
        object.__setattr__(self, "center", center)

    def contains(self, pose: Array) -> bool:
        return bool(np.linalg.norm(np.asarray(pose)[:3] - self.center) <= self.radius)

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        spread = z * math.sqrt(float(np.max(np.linalg.eigvalsh(covariance[:3, :3]))))
        return bool(np.linalg.norm(mean[:3] - self.center) - spread <= self.radius)


@dataclass(frozen=True)
class CollisionHalfSpace:
    """式 (18)--(19)：``normal @ (x - point) >= safety_distance``。"""

    point: Array
    normal: Array
    safety_distance: float = 0.0
    supports_moment_matching: bool = True

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if point.shape != (3,) or normal.shape != (3,) or norm < 1.0e-12:
            raise ValueError("碰撞半空间需要三维点和非零法向")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", normal / norm)

    def _margin(self, pose: Array) -> float:
        return float(self.normal @ (np.asarray(pose)[:3] - self.point))

    def contains(self, pose: Array) -> bool:
        return self._margin(pose) >= self.safety_distance

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        variance = float(self.normal @ covariance[:3, :3] @ self.normal)
        return self._margin(mean) + z * math.sqrt(max(variance, 0.0)) >= self.safety_distance


@dataclass(frozen=True)
class IntersectionConstraint:
    constraints: tuple[PoseConstraint, ...]

    def __init__(self, constraints: Sequence[PoseConstraint]) -> None:
        if not constraints:
            raise ValueError("交集约束不能为空")
        object.__setattr__(self, "constraints", tuple(constraints))

    @property
    def supports_moment_matching(self) -> bool:
        return all(item.supports_moment_matching for item in self.constraints)

    def contains(self, pose: Array) -> bool:
        return all(item.contains(pose) for item in self.constraints)

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        return all(item.confidence_intersects(mean, covariance, z) for item in self.constraints)


@dataclass(frozen=True)
class OccupancyConstraint:
    """式 (20)--(22) 的非凸占据约束；按论文只更新模态权重。"""

    occupancy: Callable[[Array], float]
    threshold: float
    supports_moment_matching: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("占据阈值必须位于 [0, 1]")

    def contains(self, pose: Array) -> bool:
        return float(self.occupancy(np.asarray(pose)[:3])) < self.threshold

    def confidence_intersects(self, mean: Array, covariance: Array, z: float) -> bool:
        # 任意形状占据场没有可靠的解析置信域相交判据，交给 Monte Carlo 估计。
        return True


@dataclass(frozen=True)
class TruncatedGaussian:
    mean: Array
    covariance: Array
    acceptance_probability: float
    accepted_samples: int


def sample_riemannian_gaussian(
    mean: Array,
    covariance: Array,
    sample_count: int,
    rng: np.random.Generator,
) -> Array:
    """在均值切空间采样并通过 Exp 映回 ``R3 × S3``。"""

    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.shape != (7,) or covariance.shape != (6, 6) or sample_count < 1:
        raise ValueError("采样需要 [7] 均值、[6,6] 协方差和正样本数")
    symmetric = 0.5 * (covariance + covariance.T)
    tangent = rng.multivariate_normal(np.zeros(6), symmetric, size=sample_count)
    return np.stack([pose_exp_world(mean, value) for value in tangent])


def truncate_riemannian_gaussian(
    mean: Array,
    covariance: Array,
    constraint: PoseConstraint,
    *,
    sample_count: int = 1000,
    rng: np.random.Generator | None = None,
    variance_floor: float = 1.0e-10,
) -> TruncatedGaussian | None:
    """式 (15)--(16)：Monte Carlo 截断后做 Fréchet 矩匹配。"""

    rng = np.random.default_rng(2608) if rng is None else rng
    samples = sample_riemannian_gaussian(mean, covariance, sample_count, rng)
    accepted = samples[[constraint.contains(sample) for sample in samples]]
    probability = float(len(accepted) / sample_count)
    if len(accepted) == 0:
        return None
    updated_mean = _pose_mean(accepted)
    residuals = _pose_residuals(updated_mean, accepted)
    denominator = max(len(accepted) - 1, 1)
    diagonal = np.sum(np.square(residuals), axis=0) / denominator
    updated_covariance = np.diag(np.maximum(diagonal, variance_floor))
    return TruncatedGaussian(updated_mean, updated_covariance, probability, len(accepted))


@dataclass(frozen=True)
class ConstraintUpdate:
    mean: Array
    covariance: Array
    priors: Array
    acceptance_probability: Array  # [M, T]
    feasible_modes: Array


def constrained_midigap_update(
    mean: Array,
    covariance: Array,
    priors: Array,
    constraint: PoseConstraint,
    *,
    sample_count: int = 1000,
    q: float = 1.0,
    confidence_z: float = 1.96,
    update_gaussians: bool = True,
    random_seed: int = 2608,
) -> ConstraintUpdate:
    """MiDiGaP 式 (15)--(24) 的约束更新。

    非凸 ``OccupancyConstraint`` 会自动保持高斯不变，仅按可行采样率更新模态权重。
    """

    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    if mean.ndim != 3 or mean.shape[-1] != 7:
        raise ValueError("均值必须为 [M,T,7]")
    if covariance.shape != mean.shape[:2] + (6, 6) or priors.shape != (len(mean),):
        raise ValueError("协方差或模态先验形状不匹配")
    if sample_count < 1 or q <= 0.0 or confidence_z <= 0.0:
        raise ValueError("sample_count、q、confidence_z 必须为正")
    if np.any(priors < 0.0) or not np.isclose(np.sum(priors), 1.0):
        raise ValueError("模态先验必须非负且和为 1")

    updated_mean = mean.copy()
    updated_covariance = covariance.copy()
    acceptance = np.zeros(mean.shape[:2], dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    for mode in range(mean.shape[0]):
        for time_index in range(mean.shape[1]):
            item_mean = mean[mode, time_index]
            item_covariance = covariance[mode, time_index]
            if not constraint.confidence_intersects(item_mean, item_covariance, confidence_z):
                continue
            truncated = truncate_riemannian_gaussian(
                item_mean,
                item_covariance,
                constraint,
                sample_count=sample_count,
                rng=rng,
            )
            if truncated is None:
                continue
            acceptance[mode, time_index] = truncated.acceptance_probability
            if update_gaussians and constraint.supports_moment_matching:
                updated_mean[mode, time_index] = truncated.mean
                updated_covariance[mode, time_index] = truncated.covariance

    evidence = np.mean(np.power(acceptance, q), axis=1) ** (1.0 / q)
    unnormalized = priors * evidence
    total = float(np.sum(unnormalized))
    if total <= 0.0:
        raise RuntimeError("约束消除了全部 MiDiGaP 模态")
    return ConstraintUpdate(
        mean=updated_mean,
        covariance=updated_covariance,
        priors=unnormalized / total,
        acceptance_probability=acceptance,
        feasible_modes=evidence > 0.0,
    )


def update_incoming_transitions(transition: Array, target_evidence: Array) -> Array:
    """把目标技能的新证据传播到式 (12) 的入边并逐行归一化。"""

    transition = np.asarray(transition, dtype=np.float64)
    target_evidence = np.asarray(target_evidence, dtype=np.float64)
    if transition.ndim != 2 or target_evidence.shape != (transition.shape[1],):
        raise ValueError("转移矩阵与目标证据形状不匹配")
    weighted = transition * target_evidence[None, :]
    row_sum = np.sum(weighted, axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("目标证据消除了某个来源模态的全部出边")
    return weighted / row_sum


def gaussian_pose_kl(
    source_mean: Array,
    source_covariance: Array,
    target_mean: Array,
    target_covariance: Array,
) -> float:
    """在目标均值切空间计算六维高斯 KL，供 MiDiGaP 式 (14) 使用。"""

    source_covariance = np.asarray(source_covariance, dtype=np.float64)
    target_covariance = np.asarray(target_covariance, dtype=np.float64)
    if source_covariance.shape != (6, 6) or target_covariance.shape != (6, 6):
        raise ValueError("KL 协方差必须为 [6,6]")
    target_precision = np.linalg.inv(target_covariance)
    delta = pose_log_world(np.asarray(target_mean), np.asarray(source_mean))
    source_sign, source_logdet = np.linalg.slogdet(source_covariance)
    target_sign, target_logdet = np.linalg.slogdet(target_covariance)
    if source_sign <= 0.0 or target_sign <= 0.0:
        raise ValueError("KL 协方差必须正定")
    value = 0.5 * (
        np.trace(target_precision @ source_covariance)
        + delta @ target_precision @ delta
        - 6.0
        + target_logdet
        - source_logdet
    )
    return float(max(value, 0.0))


def kl_transition_matrix(
    source_end_mean: Array,
    source_end_covariance: Array,
    target_start_mean: Array,
    target_start_covariance: Array,
) -> Array:
    """式 (14)：对未联合示范的技能按 ``exp(-KL)`` 建立模态转移。"""

    source_end_mean = np.asarray(source_end_mean, dtype=np.float64)
    source_end_covariance = np.asarray(source_end_covariance, dtype=np.float64)
    target_start_mean = np.asarray(target_start_mean, dtype=np.float64)
    target_start_covariance = np.asarray(target_start_covariance, dtype=np.float64)
    if source_end_mean.ndim != 2 or target_start_mean.ndim != 2:
        raise ValueError("技能边界均值必须为 [M,7]")
    if source_end_covariance.shape != (len(source_end_mean), 6, 6):
        raise ValueError("来源技能边界协方差形状不匹配")
    if target_start_covariance.shape != (len(target_start_mean), 6, 6):
        raise ValueError("目标技能边界协方差形状不匹配")
    divergence = np.asarray(
        [
            [
                gaussian_pose_kl(source_mean, source_cov, target_mean, target_cov)
                for target_mean, target_cov in zip(
                    target_start_mean, target_start_covariance, strict=True
                )
            ]
            for source_mean, source_cov in zip(source_end_mean, source_end_covariance, strict=True)
        ]
    )
    # 减去行最小值，避免小协方差产生的大 KL 令整行下溢。
    compatibility = np.exp(-(divergence - np.min(divergence, axis=1, keepdims=True)))
    return compatibility / np.sum(compatibility, axis=1, keepdims=True)


@dataclass(frozen=True)
class VAPORConfig:
    lambda_pose: float = 1.0
    lambda_joint: float = 0.1
    confidence_z: float = 1.96
    maximum_iterations: int = 300
    tolerance: float = 1.0e-7
    solver: Literal["slsqp", "augmented_lagrangian_fd"] = "slsqp"

    def __post_init__(self) -> None:
        if self.lambda_pose <= 0.0 or self.lambda_joint < 0.0:
            raise ValueError("VAPOR 代价权重非法")
        if self.confidence_z <= 0.0 or self.maximum_iterations < 1 or self.tolerance <= 0.0:
            raise ValueError("VAPOR 约束或求解器参数非法")
        if self.solver not in {"slsqp", "augmented_lagrangian_fd"}:
            raise ValueError("VAPOR solver 必须为 slsqp 或 augmented_lagrangian_fd")


@dataclass(frozen=True)
class VAPORResult:
    joint_trajectory: Array
    pose_trajectory: Array
    objective: float
    success: bool
    message: str
    maximum_normalized_deviation: float


def variance_aware_path_optimization(
    mean: Array,
    covariance: Array,
    initial_joint_position: Array,
    forward_kinematics: Callable[[Array], Array],
    joint_lower: Array,
    joint_upper: Array,
    config: VAPORConfig = VAPORConfig(),
) -> VAPORResult:
    """复现 VAPOR 式 (29)--(32) 的方差感知全轨迹关节优化。

    论文实现使用未公开的 Kineverse Jacobian 与增广拉格朗日求解器；默认用 SciPy SLSQP
    对同一目标、关节边界和 95% 分量置信约束求解，Jacobian 由有限差分获得。设置
    ``VAPORConfig(solver="augmented_lagrangian_fd")`` 可切换到同一约束的有限差分增广
    拉格朗日后端；它复现求解目标和更新形式，但不冒充 Kineverse 的符号 Jacobian。
    """

    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - 由可选依赖测试
        raise RuntimeError("VAPOR 需要安装可选依赖：pip install -e '.[midigap]'") from error

    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    initial_joint_position = np.asarray(initial_joint_position, dtype=np.float64)
    joint_lower = np.asarray(joint_lower, dtype=np.float64)
    joint_upper = np.asarray(joint_upper, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != 7 or covariance.shape != (len(mean), 6, 6):
        raise ValueError("VAPOR 轨迹必须为 [T,7] 与 [T,6,6]")
    joints = len(initial_joint_position)
    if joint_lower.shape != (joints,) or joint_upper.shape != (joints,):
        raise ValueError("关节边界形状不匹配")
    if np.any(joint_lower >= joint_upper):
        raise ValueError("关节下界必须小于上界")
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    if np.any(diagonal <= 0.0):
        raise ValueError("VAPOR 协方差对角线必须为正")
    sigma_max = float(np.max(diagonal))
    normalized_precision = np.stack([np.linalg.inv(item / sigma_max) for item in covariance])
    allowed = config.confidence_z * np.sqrt(diagonal)

    def pose_errors(flat: Array) -> Array:
        trajectory = flat.reshape(len(mean), joints)
        return np.stack(
            [pose_log_world(target, forward_kinematics(q)) for target, q in zip(mean, trajectory)]
        )

    def objective(flat: Array) -> float:
        trajectory = flat.reshape(len(mean), joints)
        errors = pose_errors(flat)
        pose_cost = sum(
            error @ precision @ error
            for error, precision in zip(errors, normalized_precision, strict=True)
        )
        smoothness = float(np.sum(np.square(np.diff(trajectory, axis=0))))
        return float(config.lambda_pose * pose_cost + config.lambda_joint * smoothness)

    # 论文初始化：先求最终目标 IK，再从当前关节位姿线性插值到该解。
    final_target = mean[-1]

    def final_ik_cost(q: Array) -> float:
        error = pose_log_world(final_target, forward_kinematics(q))
        return float(error @ normalized_precision[-1] @ error)

    final_ik = minimize(
        final_ik_cost,
        np.clip(initial_joint_position, joint_lower, joint_upper),
        method="L-BFGS-B",
        bounds=list(zip(joint_lower, joint_upper, strict=True)),
        options={"maxiter": config.maximum_iterations, "ftol": config.tolerance},
    )
    fractions = np.linspace(0.0, 1.0, len(mean))[:, None]
    initial_path = (
        initial_joint_position[None]
        + fractions * (np.asarray(final_ik.x) - initial_joint_position)[None]
    )
    bounds = list(
        zip(np.tile(joint_lower, len(mean)), np.tile(joint_upper, len(mean)), strict=True)
    )

    def confidence_constraint(flat: Array) -> Array:
        # SLSQP 约定 g(x) >= 0；绝对值拆成上下两个光滑线性不等式外壳。
        errors = pose_errors(flat)
        return np.concatenate(((allowed - errors).ravel(), (allowed + errors).ravel()))

    if config.solver == "slsqp":
        result = minimize(
            objective,
            initial_path.ravel(),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": confidence_constraint},
            options={
                "maxiter": config.maximum_iterations,
                "ftol": config.tolerance,
                "disp": False,
            },
        )
    else:
        # Kineverse 的公开论文描述使用增广拉格朗日；在没有其机器人模型/Jacobian
        # 对象时，仍可对本项目提供的 forward_kinematics 做同目标的有限差分 AL，
        # 这样求解器选择不会被错误地宣称成 Kineverse 原实现。
        flat = initial_path.ravel()
        multipliers = np.zeros_like(confidence_constraint(flat))
        penalty = 10.0
        result = None
        for _ in range(config.maximum_iterations):
            def augmented_objective(value):
                constraints = confidence_constraint(value)
                violation = np.minimum(constraints, 0.0)
                return float(
                    objective(value)
                    - multipliers @ constraints
                    + 0.5 * penalty * (violation @ violation)
                )

            result = minimize(
                augmented_objective,
                flat,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max(20, config.maximum_iterations // 4), "ftol": config.tolerance},
            )
            flat = np.asarray(result.x)
            constraints = confidence_constraint(flat)
            if float(np.max(np.maximum(-constraints, 0.0))) <= config.tolerance:
                break
            multipliers = np.maximum(0.0, multipliers - penalty * constraints)
            penalty *= 2.0
        assert result is not None
        final_constraints = confidence_constraint(flat)
        result.fun = objective(flat)
        result.success = bool(np.max(np.maximum(-final_constraints, 0.0)) <= config.tolerance)
        result.message = (
            "有限差分增广拉格朗日收敛"
            if result.success
            else "有限差分增广拉格朗日达到迭代上限"
        )
    joint_trajectory = np.asarray(result.x).reshape(len(mean), joints)
    pose_trajectory = np.stack([forward_kinematics(q) for q in joint_trajectory])
    deviations = np.abs(
        np.stack([pose_log_world(target, actual) for target, actual in zip(mean, pose_trajectory)])
    ) / np.sqrt(diagonal)
    return VAPORResult(
        joint_trajectory=joint_trajectory,
        pose_trajectory=pose_trajectory,
        objective=float(result.fun),
        success=bool(result.success and np.max(deviations) <= config.confidence_z + 1.0e-5),
        message=str(result.message),
        maximum_normalized_deviation=float(np.max(deviations)),
    )


__all__ = [
    "CollisionHalfSpace",
    "ConstraintUpdate",
    "IntersectionConstraint",
    "MiDiGaP",
    "MiDiGaPConfig",
    "MiDiGaPMode",
    "OccupancyConstraint",
    "PoseConstraint",
    "ReachabilitySphere",
    "TruncatedGaussian",
    "TaskParameterizedMiDiGaP",
    "VAPORConfig",
    "VAPORResult",
    "constrained_midigap_update",
    "gaussian_pose_kl",
    "kl_transition_matrix",
    "sample_riemannian_gaussian",
    "truncate_riemannian_gaussian",
    "update_incoming_transitions",
    "variance_aware_path_optimization",
]
