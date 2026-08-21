"""Bounded, current-seeded TRAC-IK implementation for RLBench Panda arms.

Formal evaluators import the public :mod:`trac_ik` facade.  This implementation
extracts the exact seven-joint chain from each live CoppeliaSim arm, limits the
search to the current joint neighbourhood, and verifies every returned
configuration with the same external forward-kinematics model.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

Array = np.ndarray
LIVE_CHAIN_SCHEMA = "coppeliasim-moving-frame-panda-chain-v1"
LIVE_CHAIN_SOURCE = "live_coppeliasim_moving_frame_segments"


class TracIKLike(Protocol):
    """Minimal interface supplied by the optional ROS-free binding."""

    joint_limits: tuple[Array, Array]

    def fk(self, q: Array) -> tuple[Array, Array]: ...

    def ik_with_bounds(
        self,
        tgt_pos: Array,
        tgt_rot: Array,
        seed_jnt_values: Array,
        bounds: Array,
    ) -> Array | None: ...


TracIKFactory = Callable[..., TracIKLike]


@dataclass(frozen=True)
class TracIKDistanceConfig:
    """Small, explicit search region used by the formal global controller."""

    timeout_s: float = 0.01
    epsilon: float = 1.0e-5
    translation_tolerance_m: float = 0.001
    rotation_tolerance_rad: float = math.radians(1.0)
    joint_window_rad: float = 0.35
    joint_delta_abs_max_rad: float = 0.35
    joint_delta_l2_max_rad: float = 0.50
    fk_translation_max_m: float = 0.002
    fk_rotation_max_rad: float = math.radians(2.0)

    def __post_init__(self) -> None:
        for name in (
            "timeout_s",
            "epsilon",
            "translation_tolerance_m",
            "rotation_tolerance_rad",
            "joint_window_rad",
            "joint_delta_abs_max_rad",
            "joint_delta_l2_max_rad",
            "fk_translation_max_m",
            "fk_rotation_max_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def cartesian_bounds(self) -> Array:
        return np.asarray(
            [
                self.translation_tolerance_m,
                self.translation_tolerance_m,
                self.translation_tolerance_m,
                self.rotation_tolerance_rad,
                self.rotation_tolerance_rad,
                self.rotation_tolerance_rad,
            ],
            dtype=np.float64,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "solver": "trac_ik_distance",
            "formal_default": True,
            "timeout_s": self.timeout_s,
            "epsilon": self.epsilon,
            "translation_tolerance_m": self.translation_tolerance_m,
            "rotation_tolerance_rad": self.rotation_tolerance_rad,
            "joint_window_rad": self.joint_window_rad,
            "joint_delta_abs_max_rad": self.joint_delta_abs_max_rad,
            "joint_delta_l2_max_rad": self.joint_delta_l2_max_rad,
            "fk_translation_max_m": self.fk_translation_max_m,
            "fk_rotation_max_rad": self.fk_rotation_max_rad,
            "random_global_sampling_fallback": False,
        }


@dataclass(frozen=True)
class TracIKDistanceResult:
    joints: Array
    elapsed_ms: float
    joint_delta_abs_rad: float
    joint_delta_l2_rad: float
    fk_translation_error_m: float
    fk_rotation_error_rad: float
    bounded_cartesian_api_used: bool


def default_rlbench_panda_urdf() -> Path:
    from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT

    return REPOSITORY_ROOT / "RLBench" / "urdfs" / "panda" / "panda.urdf"


def _matrix_from_xyzw_pose(pose: Any) -> Array:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("pose must be a finite xyz+xyzw vector")
    quaternion = value[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("pose quaternion norm must be positive")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = value[:3]
    return matrix


def _pose_error(predicted: Array, expected: Array) -> tuple[float, float]:
    translation = float(np.linalg.norm(predicted[:3, 3] - expected[:3, 3]))
    relative = predicted[:3, :3].T @ expected[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return translation, float(math.acos(cosine))


def _rigid_matrix(obj: Any, *, relative_to: Any | None = None) -> Array:
    matrix = np.asarray(obj.get_matrix(relative_to=relative_to), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("live Panda chain contains an invalid transform")
    return matrix


def _orthonormalized(transform: Array) -> Array:
    result = np.asarray(transform, dtype=np.float64).copy()
    u, _singular_values, vh = np.linalg.svd(result[:3, :3])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    result[:3, :3] = rotation
    result[3, :] = [0.0, 0.0, 0.0, 1.0]
    return result


def _matrix_to_rpy(rotation: Array) -> tuple[float, float, float]:
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy > 1.0e-12:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(-float(rotation[2, 0]), sy)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(-float(rotation[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def _urdf_origin(transform: Array) -> str:
    transform = _orthonormalized(transform)
    xyz = transform[:3, 3]
    rpy = _matrix_to_rpy(transform[:3, :3])
    xyz_text = " ".join(f"{float(value):.17g}" for value in xyz)
    rpy_text = " ".join(f"{float(value):.17g}" for value in rpy)
    return f'<origin xyz="{xyz_text}" rpy="{rpy_text}"/>'


def _live_panda_chain_urdf(arm: Any, lower: Array, upper: Array) -> tuple[str, Array]:
    """Extract one exact minimal chain without baking in current joint angles.

    In CoppeliaSim 4.1, requesting a downstream object's matrix relative to a
    joint returns it relative to that joint's *moving* frame.  It is therefore
    already the fixed post-joint segment used as the next URDF joint origin;
    applying ``inv(simGetJointMatrix)`` again would double-remove the joint
    angle.  Conversely, querying the joint object itself in world coordinates
    yields its fixed object/base frame, which is the model-root transform.
    """

    joints = tuple(getattr(arm, "joints", ()))
    if len(joints) != 7:
        raise RuntimeError("TRAC-IK live Panda chain must contain seven joints")
    tip = arm.get_tip()
    fixed_after_joint: list[Array] = []
    for index, joint in enumerate(joints):
        downstream = joints[index + 1] if index < 6 else tip
        fixed_after_joint.append(_orthonormalized(_rigid_matrix(downstream, relative_to=joint)))

    lines = ['<?xml version="1.0"?>', '<robot name="rlbench_live_panda">']
    lines.append('  <link name="base"/>')
    lines.extend(f'  <link name="link{index}"/>' for index in range(1, 8))
    lines.append('  <link name="tip"/>')
    for index in range(7):
        parent = "base" if index == 0 else f"link{index}"
        child = f"link{index + 1}"
        origin = np.eye(4) if index == 0 else fixed_after_joint[index - 1]
        lines.extend(
            [
                f'  <joint name="joint{index + 1}" type="revolute">',
                f'    <parent link="{parent}"/>',
                f'    <child link="{child}"/>',
                "    " + _urdf_origin(origin),
                '    <axis xyz="0 0 1"/>',
                (
                    f'    <limit lower="{float(lower[index]):.17g}" '
                    f'upper="{float(upper[index]):.17g}" '
                    'effort="100" velocity="3"/>'
                ),
                "  </joint>",
            ]
        )
    lines.extend(
        [
            '  <joint name="tip_fixed" type="fixed">',
            '    <parent link="link7"/>',
            '    <child link="tip"/>',
            "    " + _urdf_origin(fixed_after_joint[6]),
            "  </joint>",
            "</robot>",
        ]
    )
    return "\n".join(lines) + "\n", _rigid_matrix(joints[0])


def _joint_limits(arm: Any, current: Array) -> tuple[Array, Array]:
    cyclics, intervals = arm.get_joint_intervals()
    if len(cyclics) != current.size or len(intervals) != current.size:
        raise RuntimeError("TRAC-IK arm joint interval count is inconsistent")
    lower = []
    upper = []
    for cyclic, interval in zip(cyclics, intervals):
        if bool(cyclic):
            lower.append(-math.pi)
            upper.append(math.pi)
            continue
        values = np.asarray(interval, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise RuntimeError("TRAC-IK arm joint interval is invalid")
        lower.append(float(values[0]))
        upper.append(float(values[0] + values[1]))
    return np.asarray(lower), np.asarray(upper)


def _default_factory(**kwargs: Any) -> TracIKLike:
    try:
        from trac_ik import TracIK
    except ImportError as exc:  # pragma: no cover - depends on optional native build
        raise RuntimeError(
            "the formal global controller requires the pinned bounded pytracik build"
        ) from exc
    return TracIK(**kwargs)


class AlignedTracIKDistanceSolver:
    """One exact-chain, current-seeded solver for one live PyRep arm."""

    def __init__(
        self,
        arm: Any,
        *,
        config: TracIKDistanceConfig,
        urdf_path: Path | str | None = None,
        factory: TracIKFactory | None = None,
    ) -> None:
        self.arm = arm
        self.config = config
        current = self._current_joints()
        lower, upper = _joint_limits(arm, current)
        self.full_lower = lower
        self.full_upper = upper
        constructor = _default_factory if factory is None else factory
        if urdf_path is None:
            urdf_text, world_from_model_root = _live_panda_chain_urdf(arm, lower, upper)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="rlbench-live-panda-",
                suffix=".urdf",
            ) as temporary_urdf:
                temporary_urdf.write(urdf_text)
                temporary_urdf.flush()
                self.solver = constructor(
                    base_link_name="base",
                    tip_link_name="tip",
                    urdf_path=temporary_urdf.name,
                    timeout=config.timeout_s,
                    epsilon=config.epsilon,
                    solver_type="Distance",
                )
            self.world_from_model_root = world_from_model_root
            self.chain_schema = LIVE_CHAIN_SCHEMA
            self.chain_source = LIVE_CHAIN_SOURCE
        else:
            path = Path(urdf_path)
            if not path.is_file():
                raise RuntimeError(f"TRAC-IK Panda URDF is unavailable: {path}")
            self.solver = constructor(
                base_link_name="robot_base",
                tip_link_name="Pandatip",
                urdf_path=str(path),
                timeout=config.timeout_s,
                epsilon=config.epsilon,
                solver_type="Distance",
            )
            model_tip = self._model_fk(current)
            actual_tip = np.asarray(arm.get_tip().get_matrix(), dtype=np.float64)
            if actual_tip.shape != (4, 4) or not np.all(np.isfinite(actual_tip)):
                raise RuntimeError("live Panda tip matrix is invalid")
            self.world_from_model_root = actual_tip @ np.linalg.inv(model_tip)
            self.chain_source = "explicit_urdf_initial_pose_alignment"
            self.chain_schema = "explicit-urdf-initial-pose-alignment-legacy-v1"
        if not callable(getattr(self.solver, "ik_with_bounds", None)):
            raise RuntimeError(
                "the formal global controller requires pytracik.ik_with_bounds"
            )
        self.solver.joint_limits = (lower.copy(), upper.copy())
        self._verify_live_alignment(current)

    def _current_joints(self) -> Array:
        joints = np.asarray(self.arm.get_joint_positions(), dtype=np.float64)
        if joints.ndim != 1 or joints.size != 7 or not np.all(np.isfinite(joints)):
            raise RuntimeError("TRAC-IK expects seven finite Panda joints")
        return joints

    def _model_fk(self, joints: Array) -> Array:
        position, rotation = self.solver.fk(np.asarray(joints, dtype=np.float64))
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("TRAC-IK forward kinematics returned non-finite data")
        return matrix

    def _verify_live_alignment(self, current: Array) -> tuple[float, float]:
        predicted = self.world_from_model_root @ self._model_fk(current)
        actual = np.asarray(self.arm.get_tip().get_matrix(), dtype=np.float64)
        translation, rotation = _pose_error(predicted, actual)
        if (
            translation > self.config.fk_translation_max_m
            or rotation > self.config.fk_rotation_max_rad
        ):
            raise RuntimeError("TRAC-IK Panda model no longer matches the live arm")
        return translation, rotation

    def solve(self, target_pose: Any) -> TracIKDistanceResult | None:
        current = self._current_joints()
        self._verify_live_alignment(current)
        target_world = _matrix_from_xyzw_pose(target_pose)
        target_model = np.linalg.inv(self.world_from_model_root) @ target_world
        lower = np.maximum(self.full_lower, current - self.config.joint_window_rad)
        upper = np.minimum(self.full_upper, current + self.config.joint_window_rad)
        self.solver.joint_limits = (lower, upper)

        bounded = self.solver.ik_with_bounds
        started = time.perf_counter()
        candidate = bounded(
            target_model[:3, 3],
            target_model[:3, :3],
            seed_jnt_values=current.copy(),
            bounds=self.config.cartesian_bounds.copy(),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if candidate is None:
            return None
        joints = np.asarray(candidate, dtype=np.float64)
        if joints.shape != current.shape or not np.all(np.isfinite(joints)):
            return None
        if np.any(joints < lower - 1.0e-9) or np.any(joints > upper + 1.0e-9):
            return None
        delta = joints - current
        delta_abs = float(np.max(np.abs(delta), initial=0.0))
        delta_l2 = float(np.linalg.norm(delta))
        if (
            delta_abs > self.config.joint_delta_abs_max_rad
            or delta_l2 > self.config.joint_delta_l2_max_rad
        ):
            return None
        predicted_world = self.world_from_model_root @ self._model_fk(joints)
        translation, rotation = _pose_error(predicted_world, target_world)
        if (
            translation > self.config.fk_translation_max_m
            or rotation > self.config.fk_rotation_max_rad
        ):
            return None
        return TracIKDistanceResult(
            joints=joints.copy(),
            elapsed_ms=elapsed_ms,
            joint_delta_abs_rad=delta_abs,
            joint_delta_l2_rad=delta_l2,
            fk_translation_error_m=translation,
            fk_rotation_error_rad=rotation,
            bounded_cartesian_api_used=True,
        )
