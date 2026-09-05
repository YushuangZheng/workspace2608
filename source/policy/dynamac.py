"""Self-contained implementation of DynaMAC.

The implementation follows Algorithm 1 and Eqs. (1)--(6):

* transform end-effector trajectories into task-parameter frames on
  :math:`R^3 \\times S^3`;
* fit each stream with a discrete-time Riemannian Gaussian (DiGaP) or a mixture
  of them (MiDiGaP);
* identify kinematic links offline with either the archived skill-majority
  mask or the V3 strict-majority gate followed by per-time-state Eq. (5)
  availability;
* create and accumulate frozen virtual end-effector frames at skill starts;
* select task parameters with Eq. (6);
* transform marginals back to the world frame and combine them with a PoE in a
  common tangent space; and
* run two independent DynaMAC policies concurrently for bimanual control, using
  the peer end effector only as a candidate task parameter.

The paper does not define online link reclassification, contact recovery, or
event-driven skill recognition at evaluation time. Links and stream sets are
therefore fixed during ``fit``; inference only updates retained frame poses and
switches skills by discrete time index.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

Array = np.ndarray
IDENTITY_POSE = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
MODEL_SCHEMA_VERSION = 13
TAPAS_REFERENCE_COMMIT = "52e35214b9baa7b190b87196c36b9e98f4006149"
QUATERNION_BATCH_GAUGE_PROTOCOL_ID = (
    "per-timestep-sign-invariant-markley-anchor-temporal-continuity-v2"
)
RIEPY_MANIFOLD_REGULARIZATION = 1.0e-6
SELECTION_SEMANTICS_ID = (
    "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
)
TIMESTEP_SELECTION_SEMANTICS_ID = (
    "eq5_timestep_availability_before_eq6_and_poe_time_state_position3d_unimodal_v1"
)
MAJORITY_GATED_TIMESTEP_SELECTION_SEMANTICS_ID = (
    "eq5_skill_majority_gate_timestep_availability_before_eq6_and_poe_"
    "time_state_position3d_unimodal_v1"
)
CovarianceEstimationMethod = Literal[
    "diagonal_empirical_ridge",
    "diagonal_empirical_spd_floor",
    "full_empirical_ridge",
]
CandidateKind = Literal["dynamic", "virtual"]
Eq6EmptySelection = Literal["error", "keep_argmax"]
Eq6CovarianceScope = Literal["full_pose", "eq5_weighted_subspace"]


def _selection_semantics_id(config: Any) -> str:
    """Return an artifact identity without changing archived V2 identities."""

    if config.link_mask_scope == "skill_majority":
        return SELECTION_SEMANTICS_ID
    if config.link_mask_scope == "timestep":
        return TIMESTEP_SELECTION_SEMANTICS_ID
    if config.link_mask_scope == "skill_majority_gate_timestep":
        return MAJORITY_GATED_TIMESTEP_SELECTION_SEMANTICS_ID
    raise ValueError(f"未知 link_mask_scope：{config.link_mask_scope}")


def _as_float_array(value: Array | Sequence[float]) -> Array:
    return np.asarray(value, dtype=np.float64)


def normalize_quaternion(quaternion: Array) -> Array:
    """Normalize a wxyz quaternion and reject zero-norm inputs."""

    quaternion = _as_float_array(quaternion)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("quaternion norm must be nonzero")
    return quaternion / norm


def ensure_quaternion_continuity(quaternions: Array) -> Array:
    """Apply the quaternion sign-continuity preprocessing used by TAPAS.

    ``q`` and ``-q`` represent the same orientation, but mixing them directly can
    make the :math:`S^3` logarithmic map cross the wrong hemisphere. Each frame is
    represented with a nonnegative dot product against the preceding frame.
    """

    result = normalize_quaternion(quaternions).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("quaternion sequence must have shape [T, 4]")
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _normalize_pose_trajectory(poses: Array) -> Array:
    poses = _as_float_array(poses).copy()
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) == 0:
        raise ValueError("pose trajectory must be nonempty with shape [T, 7]")
    # Pinned TAPAS removes q/-q jumps only along time. It does not treat qw=0 as
    # an orientation boundary. Forcing each local trajectory to begin at qw>=0
    # would map physically adjacent 179- and 181-degree demonstrations to nearly
    # antipodal representatives on S3.
    poses[:, 3:7] = ensure_quaternion_continuity(poses[:, 3:7])
    return poses


def _prepare_pose_batch(trajectories: Array) -> Array:
    """Make trajectories continuous and choose a shared per-time ``q/-q`` gauge.

    Each demonstration first receives TAPAS-style temporal sign continuity.  A
    single whole-trajectory sign chosen at the first sample is nevertheless
    insufficient when physically equivalent demonstrations take different paths
    through :math:`SO(3)`: their continuous :math:`S^3` lifts can start in one
    shared hemisphere and finish in opposite hemispheres.  Mixing those antipodes
    makes the standard-arccos logarithmic map learn an orientation that no
    demonstration contains.

    At every time step, use the sign-invariant principal Markley eigenvector as a
    shared anchor and align every sample to its hemisphere.  The anchor itself is
    kept temporally continuous, so its arbitrary eigenvector sign cannot create a
    learned mean jump.  Only quaternion representatives change; physical poses,
    transforms, and commands do not.
    """

    values = _as_float_array(trajectories)
    if (
        values.ndim != 3
        or values.shape[-1] != 7
        or values.shape[0] == 0
        or values.shape[1] == 0
    ):
        raise ValueError("pose batch must be nonempty with shape [N, T, 7]")
    result = np.stack([_normalize_pose_trajectory(item) for item in values])
    previous_anchor: Array | None = None
    for time_index in range(result.shape[1]):
        quaternions = result[:, time_index, 3:7]
        accumulator = np.einsum("ni,nj->ij", quaternions, quaternions)
        _, eigenvectors = np.linalg.eigh(accumulator)
        anchor = eigenvectors[:, -1]
        # Markley eigenvectors have an arbitrary sign.  The first anchor follows
        # the first demonstration; subsequent anchors follow the previous anchor.
        reference = quaternions[0] if previous_anchor is None else previous_anchor
        if float(np.dot(anchor, reference)) < 0.0:
            anchor *= -1.0
        opposite = np.einsum("ni,i->n", quaternions, anchor) < 0.0
        result[opposite, time_index, 3:7] = -result[opposite, time_index, 3:7]
        previous_anchor = anchor
    return result


def quaternion_conjugate(quaternion: Array) -> Array:
    result = normalize_quaternion(quaternion).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: Array, right: Array) -> Array:
    """Compute the wxyz Hamilton product with NumPy broadcasting."""

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
    """Return the TAPAS/``riepybdlib`` logarithmic map on :math:`S^3`.

    The result is a three-dimensional tangent vector on the unit-quaternion sphere.
    Its norm is half the physical rotation angle, unlike a conventional axis-angle
    rotation vector whose norm is the full angle.
    """

    quaternion = normalize_quaternion(quaternion)
    vector = quaternion[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = np.arccos(np.clip(quaternion[..., 0], -1.0, 1.0))
    active = np.abs(quaternion[..., 0] - 1.0) > RIEPY_MANIFOLD_REGULARIZATION
    scale = np.divide(
        angle,
        vector_norm,
        out=np.zeros_like(angle),
        where=active & (vector_norm > np.finfo(np.float64).eps),
    )
    return vector * scale[..., None]


def quaternion_exp(rotation_vector: Array) -> Array:
    """Apply the TAPAS/``riepybdlib`` exponential map to a half-angle tangent."""

    rotation_vector = _as_float_array(rotation_vector)
    angle = np.linalg.norm(rotation_vector, axis=-1)
    scale = np.divide(
        np.sin(angle),
        angle,
        out=np.ones_like(angle),
        where=angle > 1.0e-12,
    )
    return normalize_quaternion(
        np.concatenate(
            (np.cos(angle)[..., None], rotation_vector * scale[..., None]), axis=-1
        )
    )


def quaternion_mean(quaternions: Array) -> Array:
    """Compute a Markley mean, which is invariant to quaternion sign flips."""

    quaternions = normalize_quaternion(quaternions)
    accumulator = np.einsum("ni,nj->ij", quaternions, quaternions)
    _, eigenvectors = np.linalg.eigh(accumulator)
    result = eigenvectors[:, -1]
    if result[0] < 0.0:
        result *= -1.0
    return normalize_quaternion(result)


def pose_compose(left: Array, right: Array) -> Array:
    """Apply an SE(3) left action to poses stored as ``[x, y, z, qw, qx, qy, qz]``."""

    left = _as_float_array(left)
    right = _as_float_array(right)
    position = left[..., :3] + rotate_vector(left[..., 3:7], right[..., :3])
    orientation = normalize_quaternion(
        quaternion_multiply(left[..., 3:7], right[..., 3:7])
    )
    return np.concatenate((position, orientation), axis=-1)


def pose_inverse(pose: Array) -> Array:
    pose = _as_float_array(pose)
    orientation = quaternion_conjugate(pose[..., 3:7])
    position = -rotate_vector(orientation, pose[..., :3])
    return np.concatenate((position, orientation), axis=-1)


def relative_pose(frame_pose: Array, world_pose: Array) -> Array:
    """Eq. (1): express a world-frame pose in the ``frame_pose`` coordinates."""

    return pose_compose(pose_inverse(frame_pose), world_pose)


def pose_log_world(base: Array, point: Array) -> Array:
    """Evaluate the TAPAS ``R3 x S3`` logarithmic map at ``base``.

    Translation uses a shared Euclidean basis, while rotation uses the
    ``riepybdlib`` quaternion body-tangent basis. The historical function name is
    retained for API compatibility.
    """

    base = _as_float_array(base)
    point = _as_float_array(point)
    rotation_tangent = quaternion_log(
        quaternion_multiply(quaternion_conjugate(base[3:7]), point[3:7])
    )
    return np.concatenate((point[:3] - base[:3], rotation_tangent))


def pose_log_nearest(base: Array, point: Array) -> Array:
    """Return a sign-invariant local pose residual for observation matching.

    DynaMAC fitting keeps a continuous lift on :math:`S^3`, so
    :func:`pose_log_world` deliberately does not change quaternion signs.  A
    runtime tracker, however, may represent the same physical orientation with
    either ``q`` or ``-q``.  Observation likelihoods and adjacent-cycle motion
    residuals must therefore align the observed representative to the model or
    previous pose before applying the existing TAPAS logarithmic map.
    """

    reference = _as_float_array(base)
    aligned = _as_float_array(point).copy()
    if reference.shape != (7,) or aligned.shape != (7,):
        raise ValueError("pose_log_nearest requires two [7] poses")
    if float(np.dot(reference[3:7], aligned[3:7])) < 0.0:
        aligned[3:7] *= -1.0
    return pose_log_world(reference, aligned)


def pose_exp_world(base: Array, tangent: Array) -> Array:
    """Map a six-dimensional TAPAS ``R3 x S3`` tangent vector to a pose."""

    base = _as_float_array(base)
    tangent = _as_float_array(tangent)
    orientation = quaternion_multiply(base[3:7], quaternion_exp(tangent[3:]))
    return np.concatenate((base[:3] + tangent[:3], normalize_quaternion(orientation)))


def _quaternion_left_matrix(quaternion: Array) -> Array:
    """Return the left Hamilton matrix; its last three columns form an S3 basis."""

    w, x, y, z = normalize_quaternion(quaternion)
    return np.asarray(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ],
        dtype=np.float64,
    )


def quaternion_parallel_transport(source: Array, target: Array) -> Array:
    """Transport a 3D tangent from ``source`` to ``target`` on the shortest S3 geodesic.

    The returned 3x3 orthogonal matrix uses the column-vector convention and
    matches ``quat_parallel_transport(...).T`` in the TAPAS ``riepybdlib`` version.
    """

    source = normalize_quaternion(source)
    target = normalize_quaternion(target)
    inner = float(np.clip(np.dot(source, target), -1.0, 1.0))
    distance = math.acos(inner)
    if distance < RIEPY_MANIFOLD_REGULARIZATION:
        return np.eye(3, dtype=np.float64)
    if 1.0 + inner < np.finfo(np.float64).eps:
        raise ValueError(
            "parallel transport between antipodal S3 quaternions is undefined"
        )
    # Levi-Civita parallel transport on the unit sphere in ambient coordinates.
    ambient = np.eye(4) - np.outer(source + target, target) / (1.0 + inner)
    source_basis = _quaternion_left_matrix(source)[:, 1:]
    target_basis = _quaternion_left_matrix(target)[:, 1:]
    result = target_basis.T @ ambient @ source_basis
    # Remove floating-point drift so covariance congruence remains positive definite.
    left, _, right = np.linalg.svd(result)
    result = left @ right
    # The pinned riepybdlib applies the same trace-sign correction on long S3 geodesics.
    if np.sign(np.trace(result)) == -1:
        result *= -1.0
    return result


def pose_parallel_transport(source: Array, target: Array) -> Array:
    """Return the 6x6 parallel-transport matrix on ``R3 x S3``."""

    result = np.eye(6, dtype=np.float64)
    result[3:, 3:] = quaternion_parallel_transport(source[3:7], target[3:7])
    return result


def interpolate_rows(values: Array, length: int) -> Array:
    values = _as_float_array(values)
    if length < 1:
        raise ValueError("resampling length must be positive")
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


def tapas_subsample_rows(values: Array, length: int) -> Array:
    """Align skill demonstrations using TAPAS ``Demos.get_idx_by_target_len``.

    The pinned training path uses ``round(linspace(...))`` for both downsampling
    and upsampling, so repeated samples are distributed across the trajectory.
    """

    values = _as_float_array(values)
    if length < 1 or len(values) < 1:
        raise ValueError("source and target trajectory lengths must be positive")
    # TAPAS inherits float32 when it calls CPU ``torch.linspace``. To reduce endpoint
    # error, that kernel computes the first half from ``start`` and the second half
    # backward from ``end``. This differs from np.linspace(dtype=float32) at some
    # half-integers, so the split calculation is mirrored here.
    if length == 1:
        samples = np.zeros(1, dtype=np.float32)
    else:
        end = np.float32(len(values) - 1)
        step = end / np.float32(length - 1)
        halfway = length // 2
        samples = np.empty(length, dtype=np.float32)
        samples[:halfway] = np.arange(halfway, dtype=np.float32) * step
        samples[halfway:] = (
            end
            - np.arange(
                length - halfway - 1,
                -1,
                -1,
                dtype=np.float32,
            )
            * step
        )
    indices = np.rint(samples).astype(np.int64)
    return values[indices].copy()


def tapas_subsample_poses(poses: Array, length: int) -> Array:
    poses = _normalize_pose_trajectory(poses)
    return tapas_subsample_rows(poses, length)


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
    poses = _normalize_pose_trajectory(poses)
    position = interpolate_rows(poses[:, :3], length)
    source = np.linspace(0.0, 1.0, len(poses))
    target = np.linspace(0.0, 1.0, length)
    quaternion = []
    for value in target:
        right_index = min(
            int(np.searchsorted(source, value, side="right")), len(poses) - 1
        )
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


def _pose_mean(poses: Array, weights: Array | None = None) -> Array:
    """Iteratively compute a weighted Karcher/Frechet mean on R3 x S3."""

    poses = _as_float_array(poses)
    if weights is None:
        weights = np.full(len(poses), 1.0 / len(poses), dtype=np.float64)
    else:
        weights = _as_float_array(weights)
        if (
            weights.shape != (len(poses),)
            or np.any(weights < 0.0)
            or np.sum(weights) <= 0.0
        ):
            raise ValueError(
                "Frechet mean weights must be a nonnegative vector with positive sum"
            )
        weights = weights / np.sum(weights)
    # Use weights for Markley initialization so EM soft responsibilities are retained.
    quaternions = normalize_quaternion(poses[:, 3:7])
    accumulator = np.einsum("n,ni,nj->ij", weights, quaternions, quaternions)
    _, eigenvectors = np.linalg.eigh(accumulator)
    initial_quaternion = eigenvectors[:, -1]
    # The standard-arccos logarithmic map on S3 is not a q/-q quotient. The Markley
    # eigenvector must follow the continuous data representative instead of forcing
    # qw positive after crossing 180 degrees.
    reference = quaternions[int(np.argmax(weights))]
    if float(np.dot(initial_quaternion, reference)) < 0.0:
        initial_quaternion *= -1.0
    mean = np.concatenate(
        (np.sum(weights[:, None] * poses[:, :3], axis=0), initial_quaternion)
    )
    for _ in range(64):
        increment = np.sum(weights[:, None] * _pose_residuals(mean, poses), axis=0)
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
    *,
    covariance_estimation_method: CovarianceEstimationMethod,
) -> tuple[Array, Array]:
    """MiDiGaP Eq. (6): per-step Frechet means and tangent-space covariances.

    The estimators keep the paper path and the older numerical completion
    distinguishable in saved configuration:

    * ``diagonal_empirical_ridge`` implements the MiDiGaP paper covariance
      ``diag(empirical) + lambda I``.  This is the paper-task default and matches
      the author's reported ``1e-5``/``1e-6`` covariance regularization;
    * ``diagonal_empirical_spd_floor`` uses diagonal empirical covariance and
      ``max(empirical_variance, floor)`` to make zero or small dimensions SPD.
      It remains available only to load and reproduce earlier local models;
    * ``full_empirical_ridge`` adds diagonal ridge to full empirical covariance.
    """

    trajectories = _as_float_array(trajectories)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 7:
        raise ValueError("trajectory batch must have shape [N, T, 7]")
    trajectories = _prepare_pose_batch(trajectories)
    means = np.zeros((trajectories.shape[1], 7), dtype=np.float64)
    covariance = np.zeros((trajectories.shape[1], 6, 6), dtype=np.float64)
    floor = np.asarray(
        [position_variance_floor] * 3 + [rotation_variance_floor] * 3,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(floor)) or np.any(floor <= 0.0):
        raise ValueError("covariance floors must be finite and positive")
    if covariance_estimation_method not in {
        "diagonal_empirical_ridge",
        "diagonal_empirical_spd_floor",
        "full_empirical_ridge",
    }:
        raise ValueError("unknown covariance_estimation_method")
    for time_index in range(trajectories.shape[1]):
        means[time_index] = _pose_mean(trajectories[:, time_index])
        if (
            time_index
            and float(np.dot(means[time_index - 1, 3:7], means[time_index, 3:7])) < 0.0
        ):
            means[time_index, 3:7] *= -1.0
        residuals = _pose_residuals(means[time_index], trajectories[:, time_index])
        denominator = max(len(residuals) - 1, 1)
        empirical = residuals.T @ residuals / denominator
        if covariance_estimation_method == "diagonal_empirical_ridge":
            covariance[time_index] = np.diag(np.diag(empirical) + floor)
        elif covariance_estimation_method == "diagonal_empirical_spd_floor":
            covariance[time_index] = np.diag(np.maximum(np.diag(empirical), floor))
        else:
            covariance[time_index] = empirical + np.diag(floor)
    return means, covariance


def geometric_mean_standard_deviation(
    covariance: Array,
    *,
    position_weight: float = 1.0,
    rotation_weight: float = 0.0,
) -> Array:
    """Compute the weighted geometric-mean standard deviation in DynaMAC Eq. (5).

    The default uses position only (position weight 1 and rotation weight 0), so
    ``d=3`` and the result is ``det(Sigma_pos) ** (1 / 6)``. Setting both weights
    to 1 enables the full-pose ``d=6`` variant. Positive weights scale their tangent
    coordinates; a zero weight removes that three-dimensional factor entirely.
    """

    weighted = _weighted_pose_covariance(
        covariance,
        position_weight=position_weight,
        rotation_weight=rotation_weight,
    )
    dimension = weighted.shape[-1]
    sign, log_determinant = np.linalg.slogdet(weighted)
    if np.any(sign <= 0.0):
        raise ValueError("covariance must be positive definite")
    return np.exp(log_determinant / (2.0 * dimension))


def _weighted_pose_covariance(
    covariance: Array,
    *,
    position_weight: float,
    rotation_weight: float,
) -> Array:
    """Select and weight the active position/rotation tangent subspace."""

    covariance = _as_float_array(covariance)
    if covariance.shape[-2:] != (6, 6):
        raise ValueError("pose statistics require a 6x6 tangent covariance")
    weights = np.asarray(
        [position_weight] * 3 + [rotation_weight] * 3,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(
            "Eq. (5) position and rotation weights must be finite and nonnegative"
        )
    active = weights > 0.0
    if not np.any(active):
        raise ValueError(
            "Eq. (5) requires at least one positive-weight tangent dimension"
        )
    selected = covariance[..., active, :][..., :, active]
    active_weights = weights[active]
    return selected * (active_weights[:, None] * active_weights[None, :])


def task_parameter_scores(
    covariances: dict[str, Array],
    *,
    availability: dict[str, Array],
    candidate_kind: dict[str, CandidateKind],
) -> dict[str, float]:
    """DynaMAC Eq. (6): maximize relative precision over time.

    ``availability[f][t]`` identifies candidates retained by Eq. (5) at time ``t``.
    Both metadata arguments are required. Static TP-MiDiGaP without an Eq. (5)
    stage must use :func:`static_task_parameter_scores` instead.
    """

    return task_parameter_score_details(
        covariances,
        availability=availability,
        candidate_kind=candidate_kind,
    )["scores"]


def task_parameter_score_details(
    covariances: dict[str, Array],
    *,
    availability: dict[str, Array],
    candidate_kind: dict[str, CandidateKind],
) -> dict[str, Any]:
    """返回式 (6) 的候选过滤、逐时刻归一化和 ``max_t`` 审计量。

    Eq. (5) 先确定每个时刻的可用候选集，Eq. (6) 的分母随后只在该集合
    内计算。不可用候选的相对精度严格为零，因此其时刻不可能成为该候选
    的 ``max_t``。任一时刻的可用集合为空时立即拒绝，而不猜测回退专家。
    """

    if not covariances:
        raise ValueError("至少需要一个候选任务参数")
    if availability is None:
        raise ValueError("Eq. (5) availability must be provided explicitly")
    if candidate_kind is None:
        raise ValueError("candidate_kind 必须显式提供")
    names = list(covariances)
    values = np.stack([_as_float_array(covariances[name]) for name in names])
    if (
        values.ndim != 4
        or values.shape[-1] == 0
        or values.shape[-2] != values.shape[-1]
    ):
        raise ValueError("候选协方差必须具有 [F, T, D, D] 方阵形状")
    if values.shape[1] == 0:
        raise ValueError("候选协方差的时间轴不能为空")
    if not np.all(np.isfinite(values)):
        raise ValueError("候选协方差必须全部有限")
    signs, log_determinants = np.linalg.slogdet(values)
    if np.any(signs <= 0.0) or not np.all(np.isfinite(log_determinants)):
        raise ValueError("协方差必须正定")

    duration = values.shape[1]
    if set(candidate_kind) != set(names):
        raise ValueError("candidate_kind 必须与候选协方差具有相同键")
    invalid_kinds = {
        name: kind
        for name, kind in candidate_kind.items()
        if kind not in {"dynamic", "virtual"}
    }
    if invalid_kinds:
        raise ValueError(f"candidate_kind 只允许 dynamic/virtual，收到 {invalid_kinds}")
    candidate_kinds: dict[str, CandidateKind] = {
        name: candidate_kind[name] for name in names
    }

    if set(availability) != set(names):
        raise ValueError("Eq. (5) availability 必须与候选协方差具有相同键")
    availability_rows = []
    for name in names:
        row = np.asarray(availability[name])
        if row.dtype.kind != "b" or row.shape != (duration,):
            raise ValueError("Eq. (5) availability 必须是与时间轴匹配的布尔 [T] 掩码")
        availability_rows.append(row)
    availability_values = np.stack(availability_rows)

    invalid_virtual_frames = [
        name
        for index, name in enumerate(names)
        if candidate_kinds[name] == "virtual" and not np.all(availability_values[index])
    ]
    if invalid_virtual_frames:
        raise ValueError(
            "virtual task-parameter frames must be available at every time step: "
            f"{invalid_virtual_frames}"
        )

    available_candidate_count = np.sum(availability_values, axis=0, dtype=np.int64)
    if np.any(available_candidate_count == 0):
        missing_times = [
            int(index) for index in np.flatnonzero(available_candidate_count == 0)
        ]
        raise RuntimeError(
            f"Eq. (6) time steps {missing_times} have no candidates available under "
            "Eq. (5); the paper does not define an empty-denominator fallback"
        )

    log_precision = -log_determinants
    available_log_precision = np.where(availability_values, log_precision, -np.inf)
    maximum = np.max(available_log_precision, axis=0, keepdims=True)
    shifted = np.full_like(log_precision, -np.inf)
    np.subtract(
        log_precision,
        maximum,
        out=shifted,
        where=availability_values,
    )
    relative = np.zeros_like(log_precision)
    np.exp(shifted, out=relative, where=availability_values)
    stabilized_denominator = np.sum(relative, axis=0, keepdims=True)
    log_precision_denominator = maximum[0] + np.log(stabilized_denominator[0])
    relative /= stabilized_denominator
    normalization_residual = np.sum(relative, axis=0) - 1.0
    normalization_valid_mask = (
        np.isfinite(log_precision_denominator)
        & np.isfinite(normalization_residual)
        & (np.abs(normalization_residual) <= 1.0e-12)
    )
    if not np.all(normalization_valid_mask):
        invalid_times = [
            int(index) for index in np.flatnonzero(~normalization_valid_mask)
        ]
        raise RuntimeError(f"Eq. (6) 在时刻 {invalid_times} 的归一化无效")
    ever_available = np.any(availability_values, axis=1)
    scores = {
        name: (
            float(np.max(relative[index, availability_values[index]]))
            if ever_available[index]
            else 0.0
        )
        for index, name in enumerate(names)
    }
    return {
        "semantics_id": SELECTION_SEMANTICS_ID,
        "precision_dimension": int(values.shape[-1]),
        "frame_names": tuple(names),
        "candidate_kind": candidate_kinds,
        "candidate_kind_source": "explicit",
        "logdet_covariance": log_determinants.copy(),
        "logdet_precision": log_precision.copy(),
        "availability": {
            name: availability_values[index].copy() for index, name in enumerate(names)
        },
        "availability_source": "explicit_eq5",
        "available_candidate_count": available_candidate_count.copy(),
        "log_precision_denominator": log_precision_denominator.copy(),
        "relative_precision": relative.copy(),
        "normalization_residual": normalization_residual.copy(),
        "normalization_valid_mask": normalization_valid_mask.copy(),
        "scores": scores,
        "normalization_scope": "eq5_available_candidate_frames_per_timestep",
        "eq5_filters_eq6_candidates": True,
        "eq5_filters_eq6_denominator": True,
        "unavailable_relative_precision_is_zero": True,
        "argmax_time_restricted_to_available": True,
        "virtual_frames_always_available": True,
        "rejection_reason": {
            name: (None if ever_available[index] else "eq5_never_available")
            for index, name in enumerate(names)
        },
        "argmax_time": {
            name: (
                int(
                    np.flatnonzero(availability_values[index])[
                        np.argmax(relative[index, availability_values[index]])
                    ]
                )
                if ever_available[index]
                else None
            )
            for index, name in enumerate(names)
        },
    }


def static_task_parameter_score_details(
    covariances: dict[str, Array],
) -> dict[str, Any]:
    """以静态 MiDiGaP 的显式“全候选、全动态”合同审计 Eq. (6)。

    这是唯一允许构造全 True availability 的入口。它与 DynaMAC 的 Eq. (5)
    路径分离，并在返回记录中明确标记默认值来源。
    """

    if not covariances:
        raise ValueError("至少需要一个候选任务参数")
    durations = {
        name: _as_float_array(covariance).shape[0]
        for name, covariance in covariances.items()
    }
    if len(set(durations.values())) != 1:
        raise ValueError("静态候选协方差必须具有相同时间长度")
    duration = next(iter(durations.values()))
    details = task_parameter_score_details(
        covariances,
        availability={name: np.ones(duration, dtype=bool) for name in covariances},
        candidate_kind={name: "dynamic" for name in covariances},
    )
    details["availability_source"] = "implicit_all_candidates_static_default"
    details["candidate_kind_source"] = "implicit_all_dynamic_static_default"
    details["static_all_candidates_default"] = True
    return details


def static_task_parameter_scores(covariances: dict[str, Array]) -> dict[str, float]:
    """返回静态 MiDiGaP 的全候选 Eq. (6) 得分。"""

    return static_task_parameter_score_details(covariances)["scores"]


def _eq6_skill_selection(
    covariances: dict[str, Array],
    tau_omega: float,
    *,
    availability: dict[str, Array],
    candidate_kind: dict[str, CandidateKind],
    empty_selection: Eq6EmptySelection = "error",
    semantics_id: str = SELECTION_SEMANTICS_ID,
) -> tuple[tuple[str, ...], dict[str, bool], dict[str, Any]]:
    """执行一次 Eq. (5) 候选过滤后的论文 Eq. (6) 技能级筛选。

    该函数只接受单个 skill/单个 mode 的 ``[F,T]`` 数据，并强制调用方
    显式传入 Eq. (5) availability 与候选类型。技能级选择在最终 PoE 仍会
    与逐时刻 availability 取交集。
    """

    if not 0.0 <= tau_omega < 1.0:
        raise ValueError("tau_omega 必须位于 [0,1)")
    if empty_selection not in {"error", "keep_argmax"}:
        raise ValueError("未知 Eq. (6) 空选择处理方式")
    details = task_parameter_score_details(
        covariances,
        availability=availability,
        candidate_kind=candidate_kind,
    )
    details["semantics_id"] = semantics_id
    scores = details["scores"]
    selected_by_threshold = {
        name: bool(scores[name] > tau_omega) for name in details["frame_names"]
    }
    selected_by_eq6 = selected_by_threshold.copy()
    fallback_selected_frames: tuple[str, ...] = ()
    if not any(selected_by_threshold.values()) and empty_selection == "keep_argmax":
        eligible = tuple(
            name
            for name in details["frame_names"]
            if details["rejection_reason"][name] != "eq5_never_available"
        )
        if not eligible:
            raise RuntimeError("Eq. (6) 没有任何曾经可用的候选帧")
        maximum_score = max(scores[name] for name in eligible)
        fallback_selected_frames = tuple(
            name
            for name in eligible
            if math.isclose(
                scores[name],
                maximum_score,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        )
        for name in fallback_selected_frames:
            selected_by_eq6[name] = True
    selected_frames = tuple(
        name for name in details["frame_names"] if selected_by_eq6[name]
    )
    details["tau_omega"] = float(tau_omega)
    details["selected_by_threshold"] = selected_by_threshold.copy()
    details["selected_by_eq6"] = selected_by_eq6.copy()
    details["empty_selection_policy"] = empty_selection
    details["empty_selection_policy_source_status"] = (
        "PAPER_EQ6_STRICT_THRESHOLD_FAIL_CLOSED"
        if empty_selection == "error"
        else "LOCAL_INFERENCE"
    )
    details["empty_selection_fallback_applied"] = bool(fallback_selected_frames)
    details["fallback_selected_frames"] = fallback_selected_frames
    details["fallback_reason"] = (
        "no_frame_score_strictly_above_tau_omega_keep_argmax"
        if fallback_selected_frames
        else None
    )
    details["rejection_reason"] = {
        name: (
            details["rejection_reason"][name]
            if details["rejection_reason"][name] is not None
            else None if selected_by_eq6[name] else "eq6_score_not_above_tau_omega"
        )
        for name in details["frame_names"]
    }
    return selected_frames, selected_by_eq6, details


def _compose_framewise_poe_participation(
    eq5_availability: dict[str, Array],
    eq6_selected: dict[str, Array],
) -> dict[str, Array]:
    """组合 Algorithm 1 的两层筛选，返回 ``[mode,T]`` PoE 参与掩码。

    ``eq5_availability[f][m,t]`` 表示参考系在该帧仍是外生变量；
    ``eq6_selected[f][m]`` 表示 Eq. (6) 是否把它保留在该技能/模态模型中。
    二者只能做逻辑与，不允许沿时间轴做 ``any/all``，因此天然支持同一
    技能内的屏蔽与解除屏蔽。
    """

    if not eq5_availability or set(eq5_availability) != set(eq6_selected):
        raise ValueError("Eq. (5) availability 与 Eq. (6) selection 必须具有相同非空键")
    result: dict[str, Array] = {}
    common_shape: tuple[int, int] | None = None
    for name in eq5_availability:
        available = np.asarray(eq5_availability[name], dtype=bool)
        selected = np.asarray(eq6_selected[name], dtype=bool)
        if available.ndim != 2:
            raise ValueError("Eq. (5) availability 必须具有 [mode,T] 形状")
        if selected.shape != (available.shape[0],):
            raise ValueError("Eq. (6) selection 必须具有 [mode] 形状")
        if common_shape is None:
            common_shape = available.shape
        elif available.shape != common_shape:
            raise ValueError("所有候选参考系必须共享相同 [mode,T] 形状")
        result[name] = available & selected[:, None]
    return result


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
    *,
    diagonalize: bool = False,
) -> GaussianMarginal:
    """公式 (2)：按 TAPAS 的 tangent action + parallel transport 变到世界系。"""

    frame_pose = _as_float_array(frame_pose)
    mean = pose_compose(frame_pose, local_mean)
    rotation = quaternion_to_matrix(frame_pose[3:7])
    tangent_action = np.eye(6, dtype=np.float64)
    tangent_action[:3, :3] = rotation
    # TAPAS 的 quaternion manifold 采用 body-tangent 坐标；左乘 frame quaternion
    # 对该三维切向量的 action 是恒等，而不是再乘一次三维旋转。
    transport = pose_parallel_transport(frame_pose, mean)
    transform = transport @ tangent_action
    covariance = transform @ local_covariance @ transform.T
    if diagonalize:
        covariance = np.diag(np.diag(covariance))
    return GaussianMarginal(frame_name, mean, covariance)


def product_of_experts(
    marginals: Sequence[GaussianMarginal],
    maximum_iterations: int = 50,
    tolerance: float = 1.0e-5,
    precision_weights: Sequence[float] | None = None,
) -> tuple[Array, Array, dict[str, float]]:
    """公式 (3)：按固定 TAPAS/riepy 语义左折叠黎曼高斯专家。

    TAPAS 的 ``multiply_iterable`` 对 marginal 依次调用二元 ``Gaussian.__mul__``；
    二元乘积从左操作数均值开始迭代，并在每轮把两项精度 parallel-transport 到
    当前共同切空间。保留该有序语义，避免三项以上时因曲率产生源码级差异。
    """

    if not marginals:
        raise ValueError("PoE 至少需要一个 marginal")
    if precision_weights is not None:
        weights_array = np.asarray(precision_weights, dtype=np.float64)
        if weights_array.shape != (len(marginals),):
            raise ValueError("PoE 精度权重必须与 marginal 数量一致")
        if np.any(~np.isfinite(weights_array)) or np.any(weights_array < 0.0):
            raise ValueError("PoE 精度权重必须为有限非负数")
        retained = [
            (marginal, float(weight))
            for marginal, weight in zip(marginals, weights_array, strict=True)
            if weight > 0.0
        ]
        if not retained:
            raise ValueError("PoE 至少需要一个正精度权重的 marginal")
        # 对高斯专家 p(x)^w 等价于把精度缩放为 w Lambda，即把协方差
        # 缩放为 Sigma / w。全 1 权重保持原对象和原运算路径，确保冻结的
        # baseline 数值语义不受新查询接口影响。
        if any(weight != 1.0 for _, weight in retained):
            marginals = tuple(
                GaussianMarginal(
                    marginal.frame,
                    marginal.mean,
                    marginal.covariance / weight,
                )
                for marginal, weight in retained
            )
        else:
            marginals = tuple(marginal for marginal, _ in retained)
    reference_quaternion = marginals[0].mean[3:7]
    aligned_marginals = []
    for marginal in marginals:
        mean = marginal.mean.copy()
        if float(np.dot(reference_quaternion, mean[3:7])) < 0.0:
            # 不同 tracker 可为同一 SO(3) 姿态给出 q/-q。body-tangent 坐标和
            # covariance 不变，只统一均值代表元；规范 TAPAS 输入上这是 no-op。
            mean[3:7] *= -1.0
        aligned_marginals.append(
            GaussianMarginal(marginal.frame, mean, marginal.covariance)
        )
    marginals = tuple(aligned_marginals)

    def robust_inverse(matrix: Array) -> Array:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.inv(matrix + np.eye(len(matrix)) * 1.0e-8)

    def multiply_pair(
        left: GaussianMarginal, right: GaussianMarginal
    ) -> GaussianMarginal:
        left_precision = robust_inverse(left.covariance)
        right_precision = robust_inverse(right.covariance)
        mean = left.mean.copy()
        covariance = left.covariance.copy()
        # riepy 的停止量是增量平方范数，且 ``it > 50`` 后退出（最多 51 次更新）。
        for _ in range(maximum_iterations + 1):
            left_transport = pose_parallel_transport(left.mean, mean)
            right_transport = pose_parallel_transport(right.mean, mean)
            transported_left = left_transport @ left_precision @ left_transport.T
            transported_right = right_transport @ right_precision @ right_transport.T
            covariance = robust_inverse(transported_left + transported_right)
            information = transported_left @ pose_log_world(
                mean, left.mean
            ) + transported_right @ pose_log_world(mean, right.mean)
            increment = covariance @ information
            mean = pose_exp_world(mean, increment)
            if float(increment @ increment) <= tolerance:
                break
        # 固定 TAPAS 依赖返回最后一次更新前共同切空间中的 covariance；这是
        # riepy 的源码语义，而非在新均值处额外做一次论文外重算。
        return GaussianMarginal(f"{left.frame}*{right.frame}", mean, covariance)

    joint = marginals[0]
    for marginal in marginals[1:]:
        joint = multiply_pair(joint, marginal)

    log_determinant_scores = np.asarray(
        [-np.linalg.slogdet(item.covariance)[1] for item in marginals], dtype=np.float64
    )
    shifted = log_determinant_scores - np.max(log_determinant_scores)
    determinant_scores = np.exp(shifted)
    weights = determinant_scores / np.sum(determinant_scores)
    return (
        joint.mean.copy(),
        joint.covariance.copy(),
        {
            item.frame: float(weight)
            for item, weight in zip(marginals, weights, strict=True)
        },
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
    entity_configurations: dict[str, dict[str, Array]] = field(default_factory=dict)
    scene_entity_poses: dict[str, Array] = field(default_factory=dict)
    structural_bindings: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ee_pose = _normalize_pose_trajectory(self.ee_pose)
        action_pose = _normalize_pose_trajectory(self.action_pose)
        gripper = _as_float_array(self.gripper)
        raw_skill = _as_float_array(self.skill)
        if raw_skill.ndim != 1 or not np.all(np.isfinite(raw_skill)):
            raise ValueError(f"{self.name} 的技能标签必须是一维有限整数")
        if np.any(raw_skill != np.rint(raw_skill)):
            raise ValueError(f"{self.name} 的技能标签不能含小数")
        int64 = np.iinfo(np.int64)
        if np.any(raw_skill < int64.min) or np.any(raw_skill > int64.max):
            raise ValueError(f"{self.name} 的技能标签超出 int64 范围")
        skill = raw_skill.astype(np.int64)
        if any(not isinstance(name, str) or not name for name in self.frames):
            raise ValueError(f"{self.name} 的任务参数名称必须为非空字符串")
        frames = {
            name: _normalize_pose_trajectory(value)
            for name, value in self.frames.items()
        }
        scene_entity_poses = {
            name: _normalize_pose_trajectory(value)
            for name, value in self.scene_entity_poses.items()
        }
        steps = len(ee_pose)
        if ee_pose.shape != (steps, 7) or action_pose.shape != (steps, 7):
            raise ValueError(f"{self.name} 的末端/动作位姿必须为 [T, 7]")
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if (
            gripper.ndim != 2
            or gripper.shape[1] == 0
            or gripper.shape[0] != steps
            or skill.shape != (steps,)
        ):
            raise ValueError(f"{self.name} 的数组长度不一致")
        if not frames or any(value.shape != (steps, 7) for value in frames.values()):
            raise ValueError(f"{self.name} 的任务参数必须为非空 [T, 7] 位姿字典")
        if any(value.shape != (steps, 7) for value in scene_entity_poses.values()):
            raise ValueError(f"{self.name} 的场景实体位姿必须为 [T, 7]")
        overlap = set(frames).intersection(scene_entity_poses)
        if overlap:
            raise ValueError(f"场景实体位姿不能重复任务参数：{sorted(overlap)}")
        if any(name.startswith("virtual_skill_") for name in frames):
            raise ValueError("真实任务参数名称不能使用保留前缀 virtual_skill_")
        entity_configurations: dict[str, dict[str, Array]] = {}
        known_entities = set(frames).union(scene_entity_poses)
        for entity, raw_fields in self.entity_configurations.items():
            if entity not in known_entities:
                raise ValueError(f"实体构型引用未知实体 {entity}")
            if not isinstance(raw_fields, dict) or not raw_fields:
                raise ValueError(f"实体 {entity} 的内部构型字段不能为空")
            fields: dict[str, Array] = {}
            for field_name, raw_values in raw_fields.items():
                if not isinstance(field_name, str) or not field_name:
                    raise ValueError("实体内部构型字段名必须为非空字符串")
                values = _as_float_array(raw_values)
                if values.ndim == 1:
                    values = values[:, None]
                if (
                    values.ndim != 2
                    or values.shape[0] != steps
                    or values.shape[1] == 0
                    or not np.all(np.isfinite(values))
                ):
                    raise ValueError(
                        f"实体 {entity} 的构型字段 {field_name} 必须为有限 [T,D] 数组"
                    )
                fields[field_name] = values.copy()
            entity_configurations[entity] = fields
        structural_bindings = dict(self.structural_bindings)
        for child, parent in structural_bindings.items():
            if (
                not isinstance(child, str)
                or not child
                or not isinstance(parent, str)
                or not parent
                or child == parent
            ):
                raise ValueError("直接结构绑定必须连接两个不同的非空实体")
            if child not in known_entities or parent not in known_entities:
                raise ValueError(f"直接结构绑定引用未知实体：{child}->{parent}")
        sequence = _compressed_skill_sequence(skill)
        if len(sequence) != len(set(sequence)):
            raise ValueError(f"{self.name} 的同一技能不能分成多个不连续区间")
        for poses in [ee_pose, action_pose, *frames.values()]:
            if not np.all(np.isfinite(poses)):
                raise ValueError(f"{self.name} 含非有限位姿")
        if not np.all(np.isfinite(gripper)):
            raise ValueError(f"{self.name} 含非有限夹爪动作")
        object.__setattr__(self, "ee_pose", ee_pose)
        object.__setattr__(self, "action_pose", action_pose)
        object.__setattr__(self, "gripper", gripper)
        object.__setattr__(self, "skill", skill)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "entity_configurations", entity_configurations)
        object.__setattr__(self, "scene_entity_poses", scene_entity_poses)
        object.__setattr__(self, "structural_bindings", structural_bindings)


@dataclass(frozen=True)
class DynaMACObservation:
    ee_pose: Array
    frames: dict[str, Array]

    def __post_init__(self) -> None:
        ee_pose = _as_float_array(self.ee_pose)
        frames = {name: _as_float_array(value) for name, value in self.frames.items()}
        if ee_pose.shape != (7,) or any(
            value.shape != (7,) for value in frames.values()
        ):
            raise ValueError("观测位姿必须为 [7]")
        if not np.all(np.isfinite(ee_pose)) or any(
            not np.all(np.isfinite(value)) for value in frames.values()
        ):
            raise ValueError("观测含非有限位姿")
        ee_pose = ee_pose.copy()
        ee_pose[3:7] = normalize_quaternion(ee_pose[3:7])
        frames = {name: value.copy() for name, value in frames.items()}
        for value in frames.values():
            value[3:7] = normalize_quaternion(value[3:7])
        object.__setattr__(self, "ee_pose", ee_pose)
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class DynaMACAction:
    pose: Array
    covariance: Array
    gripper: Array
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DynaMACGripperLookahead:
    """Read-only command for the policy tick following the current cursor.

    The preview is deliberately gripper-only: obtaining it does not evaluate a
    pose product-of-experts, request an observation, draw from the policy RNG,
    or advance any episode state.  ``repeats_terminal`` means that there is no
    later policy tick and the final command is repeated.  The remaining
    location fields describe the tick from which ``gripper`` is read.
    """

    gripper: Array
    crosses_skill_boundary: bool
    repeats_terminal: bool
    next_skill_index: int
    next_skill_label: int
    next_time_index: int
    next_mode: int

    def __post_init__(self) -> None:
        gripper = _as_float_array(self.gripper)
        if gripper.ndim != 1 or len(gripper) == 0:
            raise ValueError("gripper lookahead command must be a non-empty vector")
        if not np.all(np.isfinite(gripper)):
            raise ValueError("gripper lookahead command must be finite")
        object.__setattr__(self, "gripper", gripper.copy())
        for name in (
            "crosses_skill_boundary",
            "repeats_terminal",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in (
            "next_skill_index",
            "next_skill_label",
            "next_time_index",
            "next_mode",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if name != "next_skill_label" and value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class DynaMACConfig:
    """公开阈值与论文未指定、但复现必须显式冻结的数值选择。"""

    # 作者邮件确认的 paper-task 默认值。
    tau_m: float = 0.005
    tau_omega: float = 0.5
    # 论文 Eq. (6) 只保留严格超过阈值的 frames；``error`` 对空集保持
    # fail-closed。``keep_argmax`` 仅用于显式的本地鲁棒性实验。
    eq6_empty_selection: Eq6EmptySelection = "error"
    # 论文 Eq. (6) 写作完整 pose covariance determinant；position-only
    # 权重与 d=3 是作者为 Eq. (5)/tau_M 确认的设置。加权子空间仅保留
    # 为显式实验选项。
    eq6_covariance_scope: Eq6CovarianceScope = "full_pose"
    # Eq. (5)、Eq. (6)、modal input 与最终 DiGaP 必须读取同一条策略流。
    # ``action_pose`` 只保留为显式旧实验选项；默认 time-state 使用当前 EE。
    policy_model: Literal["time_state", "action_pose"] = "time_state"
    # 方法消融的显式 Eq. (5) 开关。关闭时不改动作者阈值，而是让
    # 所有 dynamic 候选帧在整个 skill 中可用，然后照常执行 Eq. (6)。
    kinematic_analysis_enabled: bool = True
    # 作者的 Eq. (5) 使用 position-only metric（d=3）。把 rotation weight
    # 显式改成正数可复现旧 full-pose 实验口径。
    eq5_position_weight: float = 1.0
    eq5_rotation_weight: float = 0.0
    # Eq. (5) 始终先产生逐时刻 raw mask。``skill_majority`` 冻结 V1/V2 的
    # 技能级常量口径；V3 的 ``skill_majority_gate_timestep`` 只用严格 >50%
    # 决定是否启用 raw mask，启用后仍逐时刻读取。
    link_mask_scope: Literal[
        "skill_majority",
        "timestep",
        "skill_majority_gate_timestep",
    ] = "skill_majority"
    link_filter: Literal["none", "temporal_variance"] = "none"
    temporal_variance_window: int = 5
    temporal_variance_threshold: float = 1.0e-4
    position_variance_floor: float = 1.0e-6
    rotation_variance_floor: float = 1.0e-6
    # MiDiGaP Eq. (6) diagonalizes the empirical covariance and recommends
    # additive ``lambda I`` regularization.  The author reports using 1e-5 or
    # 1e-6 for the DynaMAC experiments; 1e-6 is the frozen paper-task default.
    # The older SPD-floor and the full-covariance alternative remain explicit.
    covariance_estimation_method: CovarianceEstimationMethod = (
        "diagonal_empirical_ridge"
    )
    diagonalize_transformed_covariance: bool = True
    resampling_method: Literal["tapas_subsample", "interpolate"] = "tapas_subsample"
    # 作者确认论文任务均按单峰 DiGaP 运行；其他方法保留为显式实验选项。
    modal_partition_method: Literal[
        "none",
        "riemannian_kmeans_bic",
        "riemannian_gmm_bic",
        "dbscan",
    ] = "none"
    preliminary_analysis: Literal[
        "paper_order_pooled",
        "precluster_all_real_frame_product_mode_conditioned",
    ] = "paper_order_pooled"
    maximum_modes: int = 8
    minimum_mode_size: int = 2
    clustering_length: int = 20
    # The papers place the gripper in one global Euclidean factor, but do not
    # publish its scale relative to metres/radians.  Keep native gripper units
    # by default and let dataset protocols freeze an explicit positive scale.
    gripper_clustering_scale: float = 1.0
    # MiDiGaP specifies regularized diagonal covariances for modal clustering,
    # but does not publish the task-wise regularizer.  Keep it separate from
    # the final DiGaP covariance floor so a singleton candidate cannot win BIC
    # merely through an almost-zero policy covariance.
    clustering_variance_floor: float = 1.0e-6
    clustering_restarts: int = 64
    dbscan_epsilon: float = 0.1
    dbscan_min_samples: int = 2
    gmm_maximum_iterations: int = 100
    default_mode_strategy: Literal["map", "sample"] = "sample"
    random_seed: int = 2608

    def __post_init__(self) -> None:
        floating_names = (
            "tau_m",
            "tau_omega",
            "eq5_position_weight",
            "eq5_rotation_weight",
            "temporal_variance_threshold",
            "position_variance_floor",
            "rotation_variance_floor",
            "gripper_clustering_scale",
            "clustering_variance_floor",
            "dbscan_epsilon",
        )
        floating = tuple(getattr(self, name) for name in floating_names)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in floating
        ):
            raise ValueError("DynaMAC 浮点配置必须为实数")
        if not all(math.isfinite(float(value)) for value in floating):
            raise ValueError("DynaMAC 浮点配置必须为有限值")
        integer_names = (
            "temporal_variance_window",
            "maximum_modes",
            "minimum_mode_size",
            "clustering_length",
            "clustering_restarts",
            "dbscan_min_samples",
            "gmm_maximum_iterations",
            "random_seed",
        )
        integer = tuple(getattr(self, name) for name in integer_names)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in integer
        ):
            raise ValueError("DynaMAC 整数配置不能使用布尔值或小数")
        if self.random_seed < 0:
            raise ValueError("random_seed 必须为非负整数")
        if not isinstance(self.diagonalize_transformed_covariance, (bool, np.bool_)):
            raise ValueError("变换后协方差对角化开关必须为布尔值")
        if not isinstance(self.kinematic_analysis_enabled, (bool, np.bool_)):
            raise ValueError("运动学链接分析开关必须为布尔值")
        for name, value in zip(floating_names, floating, strict=True):
            object.__setattr__(self, name, float(value))
        for name, value in zip(integer_names, integer, strict=True):
            object.__setattr__(self, name, int(value))
        object.__setattr__(
            self,
            "diagonalize_transformed_covariance",
            bool(self.diagonalize_transformed_covariance),
        )
        object.__setattr__(
            self,
            "kinematic_analysis_enabled",
            bool(self.kinematic_analysis_enabled),
        )
        if not 0.0 < self.tau_m or not 0.0 <= self.tau_omega < 1.0:
            raise ValueError("tau_m/tau_omega 非法")
        if self.policy_model not in {"time_state", "action_pose"}:
            raise ValueError("未知 policy_model")
        if self.eq6_empty_selection not in {"error", "keep_argmax"}:
            raise ValueError("未知 eq6_empty_selection")
        if self.eq6_covariance_scope not in {
            "full_pose",
            "eq5_weighted_subspace",
        }:
            raise ValueError("未知 eq6_covariance_scope")
        if (
            self.eq5_position_weight < 0.0
            or self.eq5_rotation_weight < 0.0
            or self.eq5_position_weight + self.eq5_rotation_weight <= 0.0
        ):
            raise ValueError("Eq. (5) 位置/旋转权重必须非负且至少一个为正")
        if self.link_mask_scope not in {
            "skill_majority",
            "timestep",
            "skill_majority_gate_timestep",
        }:
            raise ValueError("未知 link_mask_scope")
        if self.link_filter not in {"none", "temporal_variance"}:
            raise ValueError("未知 link_filter")
        if (
            self.link_mask_scope == "skill_majority_gate_timestep"
            and self.link_filter != "none"
        ):
            raise ValueError(
                "skill_majority_gate_timestep 必须直接门控 raw Eq. (5) mask"
            )
        if self.maximum_modes < 1:
            raise ValueError("模态数必须为正")
        if self.temporal_variance_window < 1 or self.temporal_variance_threshold < 0.0:
            raise ValueError("temporal variance 参数非法")
        if self.position_variance_floor <= 0.0 or self.rotation_variance_floor <= 0.0:
            raise ValueError("协方差下限必须为正")
        if self.covariance_estimation_method not in {
            "diagonal_empirical_ridge",
            "diagonal_empirical_spd_floor",
            "full_empirical_ridge",
        }:
            raise ValueError("未知 covariance_estimation_method")
        if self.minimum_mode_size < 1 or self.clustering_length < 2:
            raise ValueError("模态最小样本数/聚类长度非法")
        if self.gripper_clustering_scale <= 0.0:
            raise ValueError("夹爪聚类尺度必须为正")
        if self.clustering_variance_floor <= 0.0 or self.clustering_restarts < 1:
            raise ValueError("聚类正则/重启次数必须为正")
        if self.dbscan_epsilon <= 0.0 or self.dbscan_min_samples < 1:
            raise ValueError("DBSCAN 参数非法")
        if self.gmm_maximum_iterations < 1:
            raise ValueError("GMM 迭代次数必须为正")
        if self.resampling_method not in {"tapas_subsample", "interpolate"}:
            raise ValueError("未知 resampling_method")
        if self.modal_partition_method not in {
            "none",
            "riemannian_kmeans_bic",
            "riemannian_gmm_bic",
            "dbscan",
        }:
            raise ValueError("未知 modal_partition_method")
        if self.preliminary_analysis not in {
            "paper_order_pooled",
            "precluster_all_real_frame_product_mode_conditioned",
        }:
            raise ValueError("未知 preliminary_analysis")
        if self.default_mode_strategy not in {"map", "sample"}:
            raise ValueError("未知 default_mode_strategy")


@dataclass
class StreamModel:
    frame: str
    mean: Array  # [M, T, 7]
    covariance: Array  # [M, T, 6, 6]
    # schema 13 同时可表达历史 skill-majority 与 V3 多数门控后的 timestep
    # availability；两者由 config 和 selection semantics identity 严格区分。
    # active 严格等于 Eq. (5) availability AND Eq. (6) skill selection。
    active: Array | None = None  # [T] 或 [M, T]
    availability: Array | None = None  # [T] 或 [M, T]
    selected_by_eq6: Array | None = None  # [M]

    def __post_init__(self) -> None:
        self.mean = _as_float_array(self.mean)
        self.covariance = _as_float_array(self.covariance)
        if self.mean.ndim != 3 or self.mean.shape[-1] != 7:
            raise ValueError("流均值必须具有 [M, T, 7] 形状")
        if self.covariance.shape != self.mean.shape[:2] + (6, 6):
            raise ValueError("流协方差必须具有 [M, T, 6, 6] 形状")
        if self.active is None:
            self.active = np.ones(self.mean.shape[1], dtype=bool)
        else:
            self.active = np.asarray(self.active, dtype=bool)
        valid_shapes = {
            (self.mean.shape[1],),
            (self.mean.shape[0], self.mean.shape[1]),
        }
        if self.active.shape not in valid_shapes:
            raise ValueError("流 active 掩码必须具有 [T] 或 [M, T] 形状")
        if self.availability is None:
            # 手工构造的静态流没有 Eq. (5) 阶段：以 active 作为全量可用性，
            # 并把 Eq. (6) selection 设为真，得到唯一保守分解。
            self.availability = self.active.copy()
        else:
            self.availability = np.asarray(self.availability, dtype=bool)
        if self.availability.shape not in valid_shapes:
            raise ValueError("流 availability 必须具有 [T] 或 [M, T] 形状")
        if self.selected_by_eq6 is None:
            self.selected_by_eq6 = np.ones(self.mean.shape[0], dtype=bool)
        else:
            self.selected_by_eq6 = np.asarray(self.selected_by_eq6, dtype=bool)
        if self.selected_by_eq6.shape != (self.mean.shape[0],):
            raise ValueError("流 selected_by_eq6 必须具有 [M] 形状")

        active_by_mode = (
            np.repeat(self.active[None, :], self.mean.shape[0], axis=0)
            if self.active.ndim == 1
            else self.active
        )
        availability_by_mode = (
            np.repeat(self.availability[None, :], self.mean.shape[0], axis=0)
            if self.availability.ndim == 1
            else self.availability
        )
        expected_active = availability_by_mode & self.selected_by_eq6[:, None]
        if not np.array_equal(active_by_mode, expected_active):
            raise ValueError("流 active 必须严格等于 availability AND selected_by_eq6")

    def is_active(self, mode: int, time_index: int) -> bool:
        if self.active.ndim == 1:
            return bool(self.active[time_index])
        return bool(self.active[mode, time_index])

    def is_available(self, mode: int, time_index: int) -> bool:
        if self.availability.ndim == 1:
            return bool(self.availability[time_index])
        return bool(self.availability[mode, time_index])

    def is_selected(self, mode: int) -> bool:
        return bool(self.selected_by_eq6[mode])


@dataclass
class SkillModel:
    label: int
    duration: int
    selected_frames: tuple[str, ...]
    mode_priors: Array
    streams: dict[str, StreamModel]
    gripper: Array  # [M, T, G]
    transition_from_previous: Array | None = None
    mode_demonstration_indices: tuple[tuple[int, ...], ...] = ()
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


def _temporal_variance_curve(
    local_mean: Array,
    window: int,
) -> tuple[Array, Array]:
    """计算论文脚注所允许、但未给出离散化细节的时序方差曲线。

    Eq. (5) 衡量同一时刻、跨演示的条件分布宽度；一条高度可重复但仍在
    运动的 pre-grasp 轨迹也可能因此产生低 GMSD。真正的刚性链接还要求
    帧与末端的相对均值在时间窗内近似不变。这里在居中窗内计算局部
    均值轨迹的 ``R3 x S3`` 切空间均方位移。技能边界使用截断窗口，
    而不是把边界自动判成“未连接”；单点技能的时序方差按零处理。

    论文脚注 1 只明确允许 temporal-variance filtering，没有公开窗口、
    度量或阈值；本离散化仍为 ``INFERRED_IMPLEMENTATION``，不得表述为
    作者源码真值。
    """

    local_mean = _normalize_pose_trajectory(local_mean)
    left_width = (window - 1) // 2
    right_width = window // 2
    curve = np.full(len(local_mean), np.nan, dtype=np.float64)
    valid = np.zeros(len(local_mean), dtype=bool)
    for index in range(len(local_mean)):
        start = max(0, index - left_width)
        stop = min(len(local_mean), index + right_width + 1)
        poses = local_mean[start:stop]
        if len(poses) == 0:
            continue
        centre = _pose_mean(poses)
        residuals = _pose_residuals(centre, poses)
        curve[index] = float(np.mean(np.sum(np.square(residuals), axis=1)))
        valid[index] = True
    return curve, valid


def _temporal_variance_filter(
    local_mean: Array,
    raw_mask: Array,
    window: int,
    threshold: float,
) -> Array:
    """用显式时序方差离散化过滤 Eq. (5) 的短暂误检。"""

    raw_mask = np.asarray(raw_mask, dtype=bool)
    curve, valid = _temporal_variance_curve(local_mean, window)
    if raw_mask.shape != curve.shape:
        raise ValueError("temporal variance 的均值轨迹与链接 mask 长度不一致")
    # Eq. (5) 仍逐时刻成立；temporal variance 只排除局部均值仍在相对
    # 运动的短暂精度峰。不得再叠加“窗口内全部为真”的连续点门槛，后者
    # 等价于论文未定义的 minimum_link_run。
    stable = valid & (curve <= threshold)
    return raw_mask & stable


def _filter_link_mask(
    scale: Array,
    config: DynaMACConfig,
    local_mean: Array | None = None,
) -> tuple[Array, Array]:
    """返回逐时刻 Eq. (5) 判定与最终链接掩码。

    ``link_mask_scope="timestep"`` 原样保留逐时刻判定。历史
    ``skill_majority`` 协议以严格 ``mean(linked) > 0.5`` 提升为整个 skill
    恒定的 mask。V3 ``skill_majority_gate_timestep`` 只用同一严格多数决定
    是否启用 raw mask：启用时保留逐时刻 raw 值，否则整段不链接。
    """

    raw = np.asarray(scale < config.tau_m, dtype=bool)
    if config.link_filter == "none":
        pointwise = raw.copy()
    elif config.link_filter == "temporal_variance":
        if local_mean is None:
            raise ValueError("temporal_variance link filter 需要局部相对位姿均值轨迹")
        pointwise = _temporal_variance_filter(
            local_mean,
            raw,
            config.temporal_variance_window,
            config.temporal_variance_threshold,
        )
    else:
        raise ValueError(f"未知 link_filter：{config.link_filter}")
    if config.link_mask_scope == "timestep":
        return raw, pointwise
    if config.link_mask_scope == "skill_majority":
        skill_linked = bool(float(np.mean(pointwise)) > 0.5)
        return raw, np.full(pointwise.shape, skill_linked, dtype=bool)
    if config.link_mask_scope == "skill_majority_gate_timestep":
        gate_enabled = _strict_majority_gate(raw)
        return raw, raw.copy() if gate_enabled else np.zeros(raw.shape, dtype=bool)
    raise ValueError(f"未知 link_mask_scope：{config.link_mask_scope}")


def _strict_majority_gate(raw_linked: Array) -> bool:
    """Return the author's strict skill gate over a non-empty raw time mask."""

    values = np.asarray(raw_linked, dtype=bool)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Eq. (5) raw linked mask 必须是非空 [T] 布尔序列")
    return bool(float(np.mean(values)) > 0.5)


def _majority_gate_audit(raw_linked: Array, config: DynaMACConfig) -> bool | None:
    """Expose the V3 gate without assigning that meaning to legacy scopes."""

    if (
        not config.kinematic_analysis_enabled
        or config.link_mask_scope != "skill_majority_gate_timestep"
    ):
        return None
    return _strict_majority_gate(raw_linked)


def _kinematic_link_masks(
    scale: Array,
    config: DynaMACConfig,
    local_mean: Array | None = None,
) -> tuple[Array, Array]:
    """返回实际训练路径使用的 Eq. (5) mask。

    ``kinematic_analysis_enabled=False`` 是论文方法消融，而不是一个
    超参数技巧。该路径显式跳过 Eq. (5)，因此原始与最终 linked
    mask 均为假，dynamic candidate 的 availability 均为真。
    """

    values = _as_float_array(scale)
    if not config.kinematic_analysis_enabled:
        disabled = np.zeros(values.shape, dtype=bool)
        return disabled.copy(), disabled
    return _filter_link_mask(values, config, local_mean=local_mean)


def _validate_demonstrations(
    demonstrations: Sequence[DynaMACDemonstration],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not demonstrations:
        raise ValueError("至少需要一条演示")
    # TAPAS 的有序 Gaussian 左折叠保留模型/任务参数配置顺序；首条演示的
    # mapping 插入顺序因此是数值语义的一部分，不能改成字母排序。
    frame_names = tuple(demonstrations[0].frames)
    skill_sequence = tuple(_compressed_skill_sequence(demonstrations[0].skill))
    if not skill_sequence:
        raise ValueError("演示没有技能")
    for demonstration in demonstrations[1:]:
        if set(demonstration.frames) != set(frame_names):
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


def _resample_rows(values: Array, duration: int, method: str) -> Array:
    if method == "tapas_subsample":
        return tapas_subsample_rows(values, duration)
    if method == "interpolate":
        return interpolate_rows(values, duration)
    raise ValueError(f"未知重采样方法：{method}")


def _resample_poses(poses: Array, duration: int, method: str) -> Array:
    if method == "tapas_subsample":
        return tapas_subsample_poses(poses, duration)
    if method == "interpolate":
        return interpolate_poses(poses, duration)
    raise ValueError(f"未知重采样方法：{method}")


def _resampled_skill_data(
    demonstrations: Sequence[DynaMACDemonstration],
    label: int,
    duration: int,
    virtual_starts: dict[int, list[Array]],
    resampling_method: str = "tapas_subsample",
) -> tuple[Array, Array, dict[str, Array], dict[str, Array]]:
    ee_trajectories = []
    action_trajectories = []
    grippers = []
    real_frames: dict[str, list[Array]] = {
        name: [] for name in demonstrations[0].frames
    }
    virtual_frames: dict[str, list[Array]] = {
        f"virtual_skill_{virtual_label}": [] for virtual_label in virtual_starts
    }
    for demo_index, demonstration in enumerate(demonstrations):
        indices = _skill_slice(demonstration, label)
        ee_trajectories.append(
            _resample_poses(demonstration.ee_pose[indices], duration, resampling_method)
        )
        action_trajectories.append(
            _resample_poses(
                demonstration.action_pose[indices], duration, resampling_method
            )
        )
        grippers.append(
            _resample_rows(demonstration.gripper[indices], duration, resampling_method)
        )
        for name, poses in demonstration.frames.items():
            real_frames[name].append(
                _resample_poses(poses[indices], duration, resampling_method)
            )
        for virtual_label, starts in virtual_starts.items():
            pose = starts[demo_index]
            virtual_frames[f"virtual_skill_{virtual_label}"].append(
                np.repeat(pose[None], duration, axis=0)
            )
    frames = {
        name: np.stack(values)
        for name, values in {**real_frames, **virtual_frames}.items()
    }
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


def _modal_euclidean_matrix(values: Array | None, samples: int) -> Array | None:
    """Validate optional global Euclidean factors of a modal trajectory.

    MiDiGaP Eq. (5)/(8) is defined for a generic product manifold.  DynaMAC
    omits gripper notation only for brevity, while TAPAS-GMM Sec. IV-B adds
    gripper width exactly once as a global :math:`R` action factor.  The
    temporal/global dimensions are flattened here without duplicating them for
    every task-parameter frame.
    """

    if values is None:
        return None
    matrix = _as_float_array(values)
    if matrix.ndim < 2 or len(matrix) != samples:
        raise ValueError("全局欧氏模态数据必须以演示维开头")
    matrix = matrix.reshape(samples, -1)
    if matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("全局欧氏模态数据必须非空且有限")
    return matrix


def _trajectory_distances(
    trajectories: Array,
    centres: Array,
    euclidean_trajectories: Array | None = None,
    euclidean_centres: Array | None = None,
) -> Array:
    """Product-manifold trajectory squared distance.

    ``trajectories`` contains the :math:`M_pose^T` factors.  Optional global
    Euclidean action factors (currently gripper width) are included once.
    """

    euclidean_trajectories = _modal_euclidean_matrix(
        euclidean_trajectories, len(trajectories)
    )
    euclidean_centres = _modal_euclidean_matrix(euclidean_centres, len(centres))
    if (euclidean_trajectories is None) != (euclidean_centres is None):
        raise ValueError("欧氏轨迹和中心必须同时提供")
    if (
        euclidean_trajectories is not None
        and euclidean_trajectories.shape[1] != euclidean_centres.shape[1]
    ):
        raise ValueError("欧氏轨迹和中心维数不一致")

    distances = np.empty((len(trajectories), len(centres)), dtype=np.float64)
    for trajectory_index, trajectory in enumerate(trajectories):
        for centre_index, centre in enumerate(centres):
            residuals = np.stack(
                [
                    pose_log_world(centre_pose, trajectory_pose)
                    for centre_pose, trajectory_pose in zip(
                        centre, trajectory, strict=True
                    )
                ]
            )
            squared = float(np.sum(np.square(residuals)))
            if euclidean_trajectories is not None:
                squared += float(
                    np.sum(
                        np.square(
                            euclidean_trajectories[trajectory_index]
                            - euclidean_centres[centre_index]
                        )
                    )
                )
            distances[trajectory_index, centre_index] = squared
    return distances


def _riemannian_kmeans_from_centres(
    trajectories: Array,
    centre_indices: Sequence[int],
    euclidean_trajectories: Array | None = None,
) -> tuple[Array, Array, float]:
    """从给定演示中心初始化一次 Riemannian k-means。"""

    centre_indices = tuple(int(index) for index in centre_indices)
    clusters = len(centre_indices)
    centres = trajectories[list(centre_indices)].copy()
    euclidean_trajectories = _modal_euclidean_matrix(
        euclidean_trajectories, len(trajectories)
    )
    euclidean_centres = (
        None
        if euclidean_trajectories is None
        else euclidean_trajectories[list(centre_indices)].copy()
    )
    labels = np.zeros(len(trajectories), dtype=np.int64)
    for iteration in range(100):
        distances = _trajectory_distances(
            trajectories,
            centres,
            euclidean_trajectories,
            euclidean_centres,
        )
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
        if euclidean_trajectories is not None:
            euclidean_centres = np.stack(
                [
                    np.mean(euclidean_trajectories[labels == index], axis=0)
                    for index in range(clusters)
                ]
            )
    distances = _trajectory_distances(
        trajectories,
        centres,
        euclidean_trajectories,
        euclidean_centres,
    )
    residual = float(np.sum(distances[np.arange(len(trajectories)), labels]))
    return labels, centres, residual


def _riemannian_kmeans_candidates(
    trajectories: Array,
    clusters: int,
    config: DynaMACConfig,
    euclidean_trajectories: Array | None = None,
) -> list[tuple[Array, Array, float]]:
    """生成确定且去重的多初始化 k-means 解。

    MiDiGaP 明确指出 GMM/k-means 对初始化敏感，但没有公开初始化策略。
    对论文常用的五条演示，穷举所有演示中心组合可消除任意固定首样本造成
    的局部最优；更大数据集则由显式 ``clustering_restarts`` 上限控制。
    这项工程选择标记为 ``INFERRED_IMPLEMENTATION``。
    """

    samples = len(trajectories)
    all_combinations = itertools.combinations(range(samples), clusters)
    initializations = list(
        itertools.islice(all_combinations, config.clustering_restarts)
    )
    candidates: list[tuple[Array, Array, float]] = []
    seen: set[tuple[int, ...]] = set()
    for centre_indices in initializations:
        labels, centres, residual = _riemannian_kmeans_from_centres(
            trajectories,
            centre_indices,
            euclidean_trajectories,
        )
        if not np.isfinite(residual):
            continue
        # Canonicalize by first member so label permutations are one candidate.
        unique = sorted(
            np.unique(labels),
            key=lambda value: int(np.flatnonzero(labels == value)[0]),
        )
        mapping = {int(old): new for new, old in enumerate(unique)}
        canonical = np.asarray(
            [mapping[int(value)] for value in labels], dtype=np.int64
        )
        key = tuple(int(value) for value in canonical)
        if key in seen:
            continue
        seen.add(key)
        canonical_centres = np.stack([centres[int(old)] for old in unique])
        candidates.append((canonical, canonical_centres, residual))
    return candidates


def _deterministic_riemannian_kmeans(
    trajectories: Array,
    clusters: int,
    config: DynaMACConfig | None = None,
    euclidean_trajectories: Array | None = None,
) -> tuple[Array, Array, float]:
    """兼容入口：返回多初始化中残差最小的 Riemannian k-means 解。"""

    active_config = DynaMACConfig() if config is None else config
    candidates = _riemannian_kmeans_candidates(
        trajectories,
        clusters,
        active_config,
        euclidean_trajectories,
    )
    if not candidates:
        return (
            np.zeros(len(trajectories), dtype=np.int64),
            trajectories[:clusters].copy(),
            float("inf"),
        )
    return min(candidates, key=lambda item: item[2])


def _logsumexp(values: Array, axis: int) -> Array:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def _mixture_statistics(
    trajectories: Array,
    responsibilities: Array,
    config: DynaMACConfig,
    euclidean_trajectories: Array | None = None,
) -> tuple[Array | tuple[Array, Array], Array, Array]:
    """在乘积流形 ``M^T`` 上拟合对角 Riemannian Gaussian mixture 参数。"""

    components = responsibilities.shape[1]
    effective = np.sum(responsibilities, axis=0)
    priors = effective / np.sum(effective)
    euclidean_trajectories = _modal_euclidean_matrix(
        euclidean_trajectories, len(trajectories)
    )
    centres = []
    euclidean_centres = []
    variances = []
    euclidean_dimension = (
        0 if euclidean_trajectories is None else euclidean_trajectories.shape[1]
    )
    floor = np.full(
        trajectories.shape[1] * 6 + euclidean_dimension,
        config.clustering_variance_floor,
        dtype=np.float64,
    )
    for component in range(components):
        weights = responsibilities[:, component]
        centre = np.stack(
            [
                _pose_mean(trajectories[:, time_index], weights)
                for time_index in range(trajectories.shape[1])
            ]
        )
        residuals = np.stack(
            [
                np.concatenate(
                    [
                        pose_log_world(centre_pose, sample_pose)
                        for centre_pose, sample_pose in zip(
                            centre, trajectory, strict=True
                        )
                    ]
                )
                for trajectory in trajectories
            ]
        )
        if euclidean_trajectories is not None:
            euclidean_centre = (
                np.sum(weights[:, None] * euclidean_trajectories, axis=0)
                / effective[component]
            )
            residuals = np.concatenate(
                (residuals, euclidean_trajectories - euclidean_centre), axis=1
            )
            euclidean_centres.append(euclidean_centre)
        variance = (
            np.sum(weights[:, None] * np.square(residuals), axis=0)
            / effective[component]
        )
        centres.append(centre)
        variances.append(variance + floor)
    pose_centres = np.stack(centres)
    combined_centres: Array | tuple[Array, Array] = pose_centres
    if euclidean_trajectories is not None:
        combined_centres = (pose_centres, np.stack(euclidean_centres))
    return combined_centres, np.stack(variances), priors


def _mixture_log_joint(
    trajectories: Array,
    centres: Array | tuple[Array, Array],
    variances: Array,
    priors: Array,
    euclidean_trajectories: Array | None = None,
) -> Array:
    euclidean_trajectories = _modal_euclidean_matrix(
        euclidean_trajectories, len(trajectories)
    )
    if isinstance(centres, tuple):
        pose_centres, euclidean_centres = centres
    else:
        pose_centres, euclidean_centres = centres, None
    if (euclidean_trajectories is None) != (euclidean_centres is None):
        raise ValueError("混合模型欧氏轨迹和中心必须同时提供")
    samples = len(trajectories)
    result = np.empty((samples, len(pose_centres)), dtype=np.float64)
    for component, (centre, variance, prior) in enumerate(
        zip(pose_centres, variances, priors, strict=True)
    ):
        residuals = np.stack(
            [
                np.concatenate(
                    [
                        pose_log_world(centre_pose, sample_pose)
                        for centre_pose, sample_pose in zip(
                            centre, trajectory, strict=True
                        )
                    ]
                )
                for trajectory in trajectories
            ]
        )
        if euclidean_trajectories is not None:
            residuals = np.concatenate(
                (
                    residuals,
                    euclidean_trajectories - euclidean_centres[component],
                ),
                axis=1,
            )
        result[:, component] = np.log(max(float(prior), 1.0e-300)) - 0.5 * np.sum(
            np.log(2.0 * math.pi * variance) + np.square(residuals) / variance,
            axis=1,
        )
    return result


def _mixture_bic(
    log_likelihood: float, clusters: int, dimension: int, samples: int
) -> float:
    # 每个 Riemannian 分量有 D 个均值自由度、D 个对角方差和 mixture priors。
    parameters = clusters * (2 * dimension) + clusters - 1
    return -2.0 * log_likelihood + parameters * math.log(max(samples, 2))


def _riemannian_kmeans_bic(
    trajectories: Array,
    config: DynaMACConfig,
    euclidean_trajectories: Array | None = None,
) -> Array:
    samples = len(trajectories)
    euclidean_trajectories = _modal_euclidean_matrix(euclidean_trajectories, samples)
    dimension = trajectories.shape[1] * 6 + (
        0 if euclidean_trajectories is None else euclidean_trajectories.shape[1]
    )
    maximum = max(1, min(config.maximum_modes, samples // config.minimum_mode_size))
    best_bic = float("inf")
    best_labels = np.zeros(samples, dtype=np.int64)
    for clusters in range(1, maximum + 1):
        for labels, _, residual in _riemannian_kmeans_candidates(
            trajectories,
            clusters,
            config,
            euclidean_trajectories,
        ):
            counts = np.asarray([np.sum(labels == index) for index in range(clusters)])
            if np.any(counts < config.minimum_mode_size) or not np.isfinite(residual):
                continue
            responsibilities = np.eye(clusters, dtype=np.float64)[labels]
            centres, variances, priors = _mixture_statistics(
                trajectories,
                responsibilities,
                config,
                euclidean_trajectories,
            )
            log_likelihood = float(
                np.sum(
                    _logsumexp(
                        _mixture_log_joint(
                            trajectories,
                            centres,
                            variances,
                            priors,
                            euclidean_trajectories,
                        ),
                        1,
                    )
                )
            )
            bic = _mixture_bic(log_likelihood, clusters, dimension, samples)
            if bic < best_bic:
                best_bic = bic
                best_labels = labels.copy()
    return best_labels


def _riemannian_gmm_bic(
    trajectories: Array,
    config: DynaMACConfig,
    euclidean_trajectories: Array | None = None,
) -> Array:
    """MiDiGaP Sec. IV-B：对角 Riemannian GMM + BIC。"""

    samples = len(trajectories)
    euclidean_trajectories = _modal_euclidean_matrix(euclidean_trajectories, samples)
    dimension = trajectories.shape[1] * 6 + (
        0 if euclidean_trajectories is None else euclidean_trajectories.shape[1]
    )
    maximum = max(1, min(config.maximum_modes, samples // config.minimum_mode_size))
    best_bic = float("inf")
    best_labels = np.zeros(samples, dtype=np.int64)
    for clusters in range(1, maximum + 1):
        for labels, _, residual in _riemannian_kmeans_candidates(
            trajectories,
            clusters,
            config,
            euclidean_trajectories,
        ):
            counts = np.asarray([np.sum(labels == index) for index in range(clusters)])
            if np.any(counts < config.minimum_mode_size) or not np.isfinite(residual):
                continue
            responsibilities = np.eye(clusters, dtype=np.float64)[labels]
            previous_likelihood = -np.inf
            likelihood = -np.inf
            valid = True
            for _ in range(config.gmm_maximum_iterations):
                effective = np.sum(responsibilities, axis=0)
                if np.any(effective < config.minimum_mode_size):
                    valid = False
                    break
                centres, variances, priors = _mixture_statistics(
                    trajectories,
                    responsibilities,
                    config,
                    euclidean_trajectories,
                )
                log_joint = _mixture_log_joint(
                    trajectories,
                    centres,
                    variances,
                    priors,
                    euclidean_trajectories,
                )
                normalizer = _logsumexp(log_joint, 1)
                likelihood = float(np.sum(normalizer))
                responsibilities = np.exp(log_joint - normalizer[:, None])
                if np.isfinite(previous_likelihood) and abs(
                    likelihood - previous_likelihood
                ) <= 1.0e-8 * (1.0 + abs(likelihood)):
                    break
                previous_likelihood = likelihood
            if not valid or not np.isfinite(likelihood):
                continue
            labels = np.argmax(responsibilities, axis=1).astype(np.int64)
            counts = np.asarray([np.sum(labels == index) for index in range(clusters)])
            if np.any(counts < config.minimum_mode_size):
                continue
            bic = _mixture_bic(likelihood, clusters, dimension, samples)
            if bic < best_bic:
                best_bic = bic
                best_labels = labels.copy()
    return best_labels


def _riemannian_dbscan(
    trajectories: Array,
    config: DynaMACConfig,
    euclidean_trajectories: Array | None = None,
) -> Array:
    """MiDiGaP Sec. IV-B：使用 ``M^T`` 测地距离的 DBSCAN。"""

    distances = np.sqrt(
        np.maximum(
            _trajectory_distances(
                trajectories,
                trajectories,
                euclidean_trajectories,
                euclidean_trajectories,
            ),
            0.0,
        )
    )
    labels, neighbours = _dbscan_raw_labels_from_distances(
        distances,
        config.dbscan_epsilon,
        config.dbscan_min_samples,
    )
    cluster = int(np.max(labels)) + 1 if np.any(labels >= 0) else 0
    # Eq. (7) requires a complete partition, while the paper does not specify
    # how DBSCAN noise is handled.  Fabricating a singleton Gaussian from each
    # noise point makes its covariance equal the numerical floor and can create
    # a false DynaMAC link.  Conservatively attach noise to its nearest detected
    # cluster; if DBSCAN found no core cluster at all, retain one pooled mode.
    # This completion is recorded as INFERRED_IMPLEMENTATION in the audit.
    noise = np.flatnonzero(labels < 0)
    if len(noise) and cluster == 0:
        labels[:] = 0
    elif len(noise):
        assigned = np.flatnonzero(labels >= 0)
        for index in noise:
            nearest = assigned[int(np.argmin(distances[index, assigned]))]
            labels[index] = labels[nearest]
    return labels


def _dbscan_raw_labels_from_distances(
    distances: Array,
    epsilon: float,
    min_samples: int,
) -> tuple[Array, list[Array]]:
    """标准 DBSCAN 标签（保留 ``-1`` 噪声）及邻域，供算法与审计共用。"""

    distances = _as_float_array(distances)
    if (
        distances.ndim != 2
        or distances.shape[0] != distances.shape[1]
        or not np.all(np.isfinite(distances))
    ):
        raise ValueError("DBSCAN 距离必须为有限方阵")
    neighbours = [
        np.flatnonzero(distances[index] <= epsilon) for index in range(len(distances))
    ]
    labels = np.full(len(distances), -1, dtype=np.int64)
    visited = np.zeros(len(distances), dtype=bool)
    cluster = 0
    for seed in range(len(distances)):
        if visited[seed]:
            continue
        visited[seed] = True
        if len(neighbours[seed]) < min_samples:
            continue
        labels[seed] = cluster
        queue = list(int(value) for value in neighbours[seed] if int(value) != seed)
        queued = set(queue)
        while queue:
            point = queue.pop(0)
            if not visited[point]:
                visited[point] = True
                if len(neighbours[point]) >= min_samples:
                    for neighbour in neighbours[point]:
                        neighbour = int(neighbour)
                        if neighbour not in queued:
                            queue.append(neighbour)
                            queued.add(neighbour)
            if labels[point] < 0:
                labels[point] = cluster
        cluster += 1
    return labels, neighbours


def _partition_modes(
    local_trajectories: Array,
    config: DynaMACConfig,
    global_euclidean_trajectories: Array | None = None,
) -> Array:
    """MiDiGaP 的 ``M^T`` 整轨迹模态划分。"""

    trajectories = _prepare_pose_batch(
        np.stack(
            [
                _resample_poses(
                    trajectory, config.clustering_length, config.resampling_method
                )
                for trajectory in local_trajectories
            ]
        )
    )
    euclidean = None
    if global_euclidean_trajectories is not None:
        values = _as_float_array(global_euclidean_trajectories)
        if values.ndim == 2:
            values = values[..., None]
        if values.ndim != 3 or len(values) != len(local_trajectories):
            raise ValueError("全局欧氏轨迹必须具有 [N, T, G] 形状")
        euclidean = np.stack(
            [
                _resample_rows(
                    trajectory, config.clustering_length, config.resampling_method
                )
                for trajectory in values
            ]
        ).reshape(len(values), -1)
    return _partition_resampled_modes(trajectories, config, euclidean)


def _partition_resampled_modes(
    trajectories: Array,
    config: DynaMACConfig,
    global_euclidean: Array | None = None,
) -> Array:
    """对已经统一长度的 ``M^T`` 或乘积流执行一次模态划分。"""

    trajectories = _as_float_array(trajectories)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 7:
        raise ValueError("聚类轨迹必须具有 [N, T, 7] 形状")
    global_euclidean = _modal_euclidean_matrix(global_euclidean, len(trajectories))
    if config.modal_partition_method == "none":
        best_labels = np.zeros(len(trajectories), dtype=np.int64)
    elif config.modal_partition_method == "riemannian_kmeans_bic":
        best_labels = _riemannian_kmeans_bic(trajectories, config, global_euclidean)
    elif config.modal_partition_method == "riemannian_gmm_bic":
        best_labels = _riemannian_gmm_bic(trajectories, config, global_euclidean)
    elif config.modal_partition_method == "dbscan":
        best_labels = _riemannian_dbscan(trajectories, config, global_euclidean)
    else:
        raise ValueError(f"未知模态划分方法：{config.modal_partition_method}")
    # 稳定重编号，使 checkpoint 与演示输入顺序确定。
    unique = sorted(
        np.unique(best_labels),
        key=lambda value: int(np.flatnonzero(best_labels == value)[0]),
    )
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in best_labels], dtype=np.int64)


def _gripper_modal_factor(gripper: Array, scale: float) -> Array:
    """Represent native gripper commands in the single global Euclidean factor.

    A constant offset is immaterial to Euclidean residuals, while the relative
    scale against pose coordinates is not.  The latter is therefore explicit
    in the config instead of assuming a particular environment's value range.
    """

    return float(scale) * _as_float_array(gripper)


def _partition_product_modes(
    local_streams: dict[str, Array],
    config: DynaMACConfig,
    global_streams: dict[str, Array] | None = None,
) -> Array:
    """在等权 per-frame product manifold 上划分整条策略轨迹。

    TAPAS 的 ``get_per_frame_data(flat=True)`` 与 MiDiGaP 的任务参数化说明都
    把一个样本表示为所有候选局部流的乘积。每个 frame 先独立重采样到
    ``clustering_length``，随后沿乘积维拼接；距离、似然与 BIC 因而对每个
    frame 的 ``M^T`` 项求和。论文没有公开 frame 权重，本实现采用等权并将
    该数值选择记录为 ``INFERRED_IMPLEMENTATION``。
    """

    products, global_euclidean = _resampled_product_modal_data(
        local_streams,
        config,
        global_streams,
    )
    return _partition_resampled_modes(products, config, global_euclidean)


def _resampled_product_modal_data(
    local_streams: dict[str, Array],
    config: DynaMACConfig,
    global_streams: dict[str, Array] | None = None,
) -> tuple[Array, Array | None]:
    """构造 MiDiGaP 式 (8) 的完整乘积流形样本，保留供审计使用。"""

    if not local_streams:
        raise ValueError("per-frame product 聚类至少需要一条局部流")
    sample_count: int | None = None
    products = []
    for name, stream in local_streams.items():
        values = _as_float_array(stream)
        if values.ndim != 3 or values.shape[-1] != 7 or len(values) == 0:
            raise ValueError(f"局部流 {name} 必须具有非空 [N, T, 7] 形状")
        if sample_count is None:
            sample_count = len(values)
        elif len(values) != sample_count:
            raise ValueError("per-frame product 的所有局部流必须演示一一对应")
        products.append(
            _prepare_pose_batch(
                np.stack(
                    [
                        _resample_poses(
                            trajectory,
                            config.clustering_length,
                            config.resampling_method,
                        )
                        for trajectory in values
                    ]
                )
            )
        )
    euclidean_products = []
    for name, stream in (global_streams or {}).items():
        values = _as_float_array(stream)
        if values.ndim == 2:
            values = values[..., None]
        if values.ndim != 3 or len(values) == 0:
            raise ValueError(f"全局流 {name} 必须具有非空 [N, T, G] 形状")
        if sample_count is not None and len(values) != sample_count:
            raise ValueError("全局流必须与局部流的演示一一对应")
        resampled = np.stack(
            [
                _resample_rows(
                    trajectory,
                    config.clustering_length,
                    config.resampling_method,
                )
                for trajectory in values
            ]
        )
        euclidean_products.append(resampled.reshape(len(values), -1))
    global_euclidean = (
        None if not euclidean_products else np.concatenate(euclidean_products, axis=1)
    )
    return np.concatenate(products, axis=1), global_euclidean


def _modal_partition_audit(
    trajectories: Array,
    global_euclidean: Array | None,
    labels: Array,
    config: DynaMACConfig,
) -> dict[str, Any]:
    """记录实际模态划分的输入、距离和方法诊断；不参与策略选择。"""

    trajectories = _as_float_array(trajectories)
    labels = np.asarray(labels, dtype=np.int64)
    squared_distances = _trajectory_distances(
        trajectories,
        trajectories,
        global_euclidean,
        global_euclidean,
    )
    distances = np.sqrt(np.maximum(squared_distances, 0.0))
    audit: dict[str, Any] = {
        "method": config.modal_partition_method,
        "partition_performed": config.modal_partition_method != "none",
        "paper_workflow": "unimodal DiGaP; no modal partition for reported tasks",
        "automatic_fallback": False,
        "resampled_pose_product": trajectories.copy(),
        "global_euclidean_product": (
            None
            if global_euclidean is None
            else _as_float_array(global_euclidean).copy()
        ),
        "geodesic_distance_matrix": distances,
        "final_labels": labels.copy(),
        "mode_sizes": np.asarray(
            [np.sum(labels == mode) for mode in range(int(np.max(labels)) + 1)],
            dtype=np.int64,
        ),
    }
    if config.modal_partition_method == "none":
        audit.update(
            {
                "unimodal": True,
                "partition_reason": "disabled_by_author_confirmed_paper_task_default",
            }
        )
    elif config.modal_partition_method == "dbscan":
        raw_labels, neighbours = _dbscan_raw_labels_from_distances(
            distances,
            config.dbscan_epsilon,
            config.dbscan_min_samples,
        )
        raw_clusters = int(np.max(raw_labels)) + 1 if np.any(raw_labels >= 0) else 0
        audit.update(
            {
                "dbscan_epsilon": config.dbscan_epsilon,
                "dbscan_min_samples": config.dbscan_min_samples,
                "dbscan_neighbour_counts": np.asarray(
                    [len(item) for item in neighbours], dtype=np.int64
                ),
                "dbscan_core_mask": np.asarray(
                    [len(item) >= config.dbscan_min_samples for item in neighbours],
                    dtype=bool,
                ),
                "dbscan_raw_labels": raw_labels,
                "dbscan_noise_indices": np.flatnonzero(raw_labels < 0),
                "dbscan_raw_cluster_count": raw_clusters,
                "dbscan_completion": (
                    "pooled_all_samples_INFERRED_IMPLEMENTATION"
                    if raw_clusters == 0 and np.any(raw_labels < 0)
                    else (
                        "nearest_detected_cluster_INFERRED_IMPLEMENTATION"
                        if np.any(raw_labels < 0)
                        else "none"
                    )
                ),
            }
        )
    else:
        audit.update(
            {
                "maximum_modes": config.maximum_modes,
                "minimum_mode_size": config.minimum_mode_size,
                "clustering_variance_floor": config.clustering_variance_floor,
                "clustering_restarts": config.clustering_restarts,
                "selection_criterion": "BIC",
            }
        )
    return audit


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
            result[source, target] = (
                np.sum(source_mask & (current == target)) / denominator
            )
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
        self._mode_strategy: Literal["map", "sample"] = config.default_mode_strategy
        self._mode_path: tuple[int, ...] = ()
        self._mode_evidence: tuple[Array, ...] = ()
        self._active_mode = 0
        self._complete = False
        self._episode_initialized = False
        self._rng = np.random.default_rng(config.random_seed)
        self._training_audit: dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return bool(self.skills)

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def selected_mode_path(self) -> tuple[int, ...]:
        """Return the read-only episode MiDiGaP mode path selected by reset."""

        if not self._episode_initialized:
            raise RuntimeError("DynaMAC 尚未 reset，不能读取 episode 模态路径")
        return tuple(self._mode_path)

    @property
    def selection_semantics_id(self) -> str:
        """Config-specific model identity; V1/V2 checkpoints keep their ID."""

        return _selection_semantics_id(self.config)

    @property
    def training_audit(self) -> dict[str, Any]:
        """最近一次 ``fit`` 的只读审计副本；checkpoint 加载后为空。"""

        return deepcopy(self._training_audit)

    @property
    def current_skill(self) -> SkillModel:
        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        return self.skills[self._skill_index]

    def _capture_runtime_state(self) -> dict[str, Any]:
        """为双臂同周期事务保存可变执行状态。"""

        return {
            "skill_index": self._skill_index,
            "time_index": self._time_index,
            "virtual_frames": {
                name: value.copy() for name, value in self._virtual_frames.items()
            },
            "pending_virtual_capture": self._pending_virtual_capture,
            "mode_strategy": self._mode_strategy,
            "mode_path": self._mode_path,
            "mode_evidence": tuple(value.copy() for value in self._mode_evidence),
            "active_mode": self._active_mode,
            "complete": self._complete,
            "episode_initialized": self._episode_initialized,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def _restore_runtime_state(self, state: dict[str, Any]) -> None:
        self._skill_index = state["skill_index"]
        self._time_index = state["time_index"]
        self._virtual_frames = {
            name: value.copy() for name, value in state["virtual_frames"].items()
        }
        self._pending_virtual_capture = state["pending_virtual_capture"]
        self._mode_strategy = state["mode_strategy"]
        self._mode_path = state["mode_path"]
        self._mode_evidence = tuple(value.copy() for value in state["mode_evidence"])
        self._active_mode = state["active_mode"]
        self._complete = state["complete"]
        self._episode_initialized = state["episode_initialized"]
        self._rng.bit_generator.state = deepcopy(state["rng_state"])

    def _invalidate_episode_after_fit(self) -> None:
        """使新模型必须经 ``reset`` 建立技能路径和虚拟帧。"""

        self._skill_index = 0
        self._time_index = 0
        self._virtual_frames = {}
        self._pending_virtual_capture = False
        self._mode_strategy = self.config.default_mode_strategy
        self._mode_path = ()
        self._mode_evidence = ()
        self._active_mode = 0
        self._complete = False
        self._episode_initialized = False
        self._rng = np.random.default_rng(self.config.random_seed)

    def fit(self, demonstrations: Sequence[DynaMACDemonstration]) -> DynaMAC:
        """Fit transactionally, preserving the previous policy after failure.

        Strict Eq. (5)/Eq. (6) validation can reject a demonstration cohort.
        A failed refit must not leave a previously usable policy paired with a
        partial audit from the rejected cohort.
        """

        previous_model = (self.frame_names, self.skill_sequence, self.skills)
        previous_runtime = self._capture_runtime_state()
        previous_audit = deepcopy(self._training_audit)
        try:
            return self._fit_in_place(demonstrations)
        except Exception:
            self.frame_names, self.skill_sequence, self.skills = previous_model
            self._restore_runtime_state(previous_runtime)
            self._training_audit = previous_audit
            raise

    def _fit_in_place(
        self,
        demonstrations: Sequence[DynaMACDemonstration],
    ) -> DynaMAC:
        """执行 Algorithm 1，学习 Eq. (5) 掩码与 Eq. (6) 技能集合。

        Eq. (5) 先生成逐时刻 raw mask。V3 以严格多数决定是否启用该
        frame/skill 的 mask：通过时保留 raw 的逐时刻结构，不通过时整段
        availability 为真；历史协议仍可显式折叠成 skill-majority 常量。
        Eq. (6) 与最终 DiGaP 使用同一条 time-state 策略流（当前 EE），
        不是 next action。
        """

        frame_names, skill_sequence = _validate_demonstrations(demonstrations)
        self._training_audit = {
            "audit_schema_version": 1,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "semantics_id": self.selection_semantics_id,
            "paper_scope": "DynaMAC Algorithm 1 + MiDiGaP + TAPAS skill inputs",
            "policy_feedback": False,
            "frame_names": frame_names,
            "skill_sequence": skill_sequence,
            "demonstration_names": tuple(item.name for item in demonstrations),
            "config": asdict(self.config),
            "kinematic_analysis": {
                "enabled": self.config.kinematic_analysis_enabled,
                "equation": "Eq. (5)",
                "disabled_behavior": (
                    None
                    if self.config.kinematic_analysis_enabled
                    else "explicit_bypass_all_dynamic_candidates_available"
                ),
                "tau_m_retained": self.config.tau_m,
            },
            "skills": [],
        }
        fitted_skills: list[SkillModel] = []
        virtual_starts: dict[int, list[Array]] = {}
        previous_mode_labels: Array | None = None
        for label in skill_sequence:
            # Algorithm 1: V <- V ∪ {EE_tk}。这里只含当前及过去的技能起点，
            # 因而训练选出的每个虚拟帧在执行该技能时都已经被捕获。
            virtual_starts[label] = [
                demonstration.ee_pose[_skill_slice(demonstration, label)[0]].copy()
                for demonstration in demonstrations
            ]
            lengths = [
                len(_skill_slice(demonstration, label))
                for demonstration in demonstrations
            ]
            # TAPAS Demos 使用 int(mean(lengths)) 与 round(linspace) 索引重采样。
            duration = max(int(float(np.mean(lengths))), 1)
            ee, actions, frames, extra = _resampled_skill_data(
                demonstrations,
                label,
                duration,
                virtual_starts,
                self.config.resampling_method,
            )

            policy_pose = ee if self.config.policy_model == "time_state" else actions
            local_policy = {
                name: _local_trajectories(frame_values, policy_pose)
                for name, frame_values in frames.items()
            }
            candidate_frames = [
                *frame_names,
                *(f"virtual_skill_{virtual_label}" for virtual_label in virtual_starts),
            ]
            candidate_kind: dict[str, CandidateKind] = {
                name: "dynamic" for name in frame_names
            }
            candidate_kind.update(
                {
                    f"virtual_skill_{virtual_label}": "virtual"
                    for virtual_label in virtual_starts
                }
            )
            if len(candidate_kind) != len(candidate_frames):
                raise ValueError("真实任务参数名不得与 DynaMAC 虚拟帧名冲突")
            skill_audit: dict[str, Any] = {
                "semantics_id": self.selection_semantics_id,
                "skill_label": int(label),
                "duration": int(duration),
                "source_skill_indices": tuple(
                    _skill_slice(demonstration, label).copy()
                    for demonstration in demonstrations
                ),
                "virtual_frame_history": tuple(
                    f"virtual_skill_{virtual_label}" for virtual_label in virtual_starts
                ),
                "virtual_frame_start_poses": {
                    f"virtual_skill_{virtual_label}": np.stack(starts)
                    for virtual_label, starts in virtual_starts.items()
                },
                "candidate_frames": tuple(candidate_frames),
                "candidate_kind": candidate_kind.copy(),
                "resampled_ee": ee.copy(),
                "resampled_action": actions.copy(),
                "policy_model": self.config.policy_model,
                "kinematic_analysis_enabled": self.config.kinematic_analysis_enabled,
                "resampled_policy_pose": policy_pose.copy(),
                "resampled_gripper": extra["gripper"].copy(),
                "resampled_frame_pose": {
                    name: values.copy() for name, values in frames.items()
                },
                "local_policy": {
                    name: values.copy() for name, values in local_policy.items()
                },
                "kinematic_links": {},
            }
            link_diagnostics: dict[str, dict[str, Any]] = {}
            if self.config.preliminary_analysis == "paper_order_pooled":
                # DynaMAC Algorithm 1 literally performs pooled Eq. (5), adds
                # virtual frames, performs pooled Eq. (6), and only then fits
                # the final policy.  Neither equation carries a mode subscript.
                # This is the strict paper-order baseline.
                pooled_availability = {
                    name: np.ones(duration, dtype=bool) for name in candidate_frames
                }
                for name in frame_names:
                    link_mean, covariance = _fit_pose_sequence(
                        local_policy[name],
                        self.config.position_variance_floor,
                        self.config.rotation_variance_floor,
                        covariance_estimation_method=self.config.covariance_estimation_method,
                    )
                    scale = geometric_mean_standard_deviation(
                        covariance,
                        position_weight=self.config.eq5_position_weight,
                        rotation_weight=self.config.eq5_rotation_weight,
                    )
                    raw_mask, linked_mask = _kinematic_link_masks(
                        scale,
                        self.config,
                        local_mean=link_mean,
                    )
                    temporal_curve, temporal_valid = _temporal_variance_curve(
                        link_mean,
                        self.config.temporal_variance_window,
                    )
                    pooled_availability[name] = ~linked_mask
                    link_diagnostics[name] = {
                        "linked": bool(np.any(linked_mask)),
                        "fully_linked": bool(np.all(linked_mask)),
                        "raw_linked_fraction": float(np.mean(raw_mask)),
                        "linked_fraction": float(np.mean(linked_mask)),
                        "raw_maximum_link_run": _maximum_true_run(raw_mask),
                        "maximum_link_run": _maximum_true_run(linked_mask),
                        "minimum_m": float(np.min(scale)),
                        "median_m": float(np.median(scale)),
                        "gmsd": scale.tolist(),
                        "raw_link_mask": raw_mask.tolist(),
                        "filtered_link_mask": linked_mask.tolist(),
                        "filter": self.config.link_filter,
                        "mask_scope": self.config.link_mask_scope,
                        **(
                            {
                                "majority_gate_enabled": _majority_gate_audit(
                                    raw_mask, self.config
                                ),
                                "majority_gate_rule": ("strict_mean_raw_linked_gt_0.5"),
                            }
                            if self.config.link_mask_scope
                            == "skill_majority_gate_timestep"
                            else {}
                        ),
                        "kinematic_analysis_enabled": (
                            self.config.kinematic_analysis_enabled
                        ),
                        "analysis_performed": self.config.kinematic_analysis_enabled,
                        "disabled_behavior": (
                            None
                            if self.config.kinematic_analysis_enabled
                            else "explicit_bypass_all_dynamic_candidates_available"
                        ),
                        "skill_linked": (
                            bool(linked_mask[0])
                            if self.config.link_mask_scope == "skill_majority"
                            else None
                        ),
                        "scope": "paper_order_pooled_preliminary_digap",
                        "per_mode": [],
                    }
                    sign_covariance, logdet_covariance = np.linalg.slogdet(covariance)
                    if np.any(sign_covariance <= 0.0):
                        raise RuntimeError("运动学连接分析产生非正定协方差")
                    skill_audit["kinematic_links"][name] = {
                        "preliminary_mean": link_mean.copy(),
                        "preliminary_covariance": covariance.copy(),
                        "preliminary_precision": np.linalg.inv(covariance),
                        "per_dimension_standard_deviation": np.sqrt(
                            np.diagonal(covariance, axis1=-2, axis2=-1)
                        ),
                        "gmsd": scale.copy(),
                        "logdet_covariance": logdet_covariance.copy(),
                        "logdet_precision": (-logdet_covariance).copy(),
                        "raw_link_mask": raw_mask.copy(),
                        "filtered_link_mask": linked_mask.copy(),
                        "temporal_variance": temporal_curve,
                        "temporal_variance_valid": temporal_valid,
                        "tau_m": self.config.tau_m,
                        "position_weight": self.config.eq5_position_weight,
                        "rotation_weight": self.config.eq5_rotation_weight,
                        "dimension": int(
                            3 * (self.config.eq5_position_weight > 0.0)
                            + 3 * (self.config.eq5_rotation_weight > 0.0)
                        ),
                        "filter": self.config.link_filter,
                        "mask_scope": self.config.link_mask_scope,
                        **(
                            {
                                "majority_gate_enabled": _majority_gate_audit(
                                    raw_mask, self.config
                                ),
                                "majority_gate_rule": ("strict_mean_raw_linked_gt_0.5"),
                            }
                            if self.config.link_mask_scope
                            == "skill_majority_gate_timestep"
                            else {}
                        ),
                        "kinematic_analysis_enabled": (
                            self.config.kinematic_analysis_enabled
                        ),
                        "analysis_performed": self.config.kinematic_analysis_enabled,
                        "disabled_behavior": (
                            None
                            if self.config.kinematic_analysis_enabled
                            else "explicit_bypass_all_dynamic_candidates_available"
                        ),
                        "skill_linked": (
                            bool(linked_mask[0])
                            if self.config.link_mask_scope == "skill_majority"
                            else None
                        ),
                        "filter_status": (
                            "METHOD_ABLATION_EQ5_DISABLED"
                            if not self.config.kinematic_analysis_enabled
                            else (
                                "PAPER_EQ5_EXACT"
                                if self.config.link_filter == "none"
                                else "PAPER_ALLOWED_NUMERICS_INFERRED"
                            )
                        ),
                    }
                policy_covariance: dict[str, Array] = {}
                for name in candidate_frames:
                    _, covariance = _fit_pose_sequence(
                        local_policy[name],
                        self.config.position_variance_floor,
                        self.config.rotation_variance_floor,
                        covariance_estimation_method=self.config.covariance_estimation_method,
                    )
                    policy_covariance[name] = covariance
                selection_covariance = (
                    policy_covariance
                    if self.config.eq6_covariance_scope == "full_pose"
                    else {
                        name: _weighted_pose_covariance(
                            covariance,
                            position_weight=self.config.eq5_position_weight,
                            rotation_weight=self.config.eq5_rotation_weight,
                        )
                        for name, covariance in policy_covariance.items()
                    }
                )
                selected, pooled_eq6_selected, selection_details = _eq6_skill_selection(
                    selection_covariance,
                    self.config.tau_omega,
                    availability=pooled_availability,
                    candidate_kind=candidate_kind,
                    empty_selection=self.config.eq6_empty_selection,
                    semantics_id=self.selection_semantics_id,
                )
                scores = selection_details["scores"]
                skill_audit["task_parameter_selection"] = {
                    **selection_details,
                    "candidate_covariance": {
                        name: covariance.copy()
                        for name, covariance in policy_covariance.items()
                    },
                    "selection_covariance": {
                        name: covariance.copy()
                        for name, covariance in selection_covariance.items()
                    },
                    "eq6_covariance_scope": self.config.eq6_covariance_scope,
                    "eq6_covariance_scope_source_status": (
                        "LOCAL_AUTHOR_EMAIL_INTERPRETATION"
                        if self.config.eq6_covariance_scope == "eq5_weighted_subspace"
                        else "PAPER_EQ6_FULL_POSE"
                    ),
                    "eq5_availability": {
                        name: mask.copy() for name, mask in pooled_availability.items()
                    },
                    "tau_omega": self.config.tau_omega,
                    "kinematic_analysis_enabled": (
                        self.config.kinematic_analysis_enabled
                    ),
                    "eq5_bypass": (
                        None
                        if self.config.kinematic_analysis_enabled
                        else "all_dynamic_candidates_available"
                    ),
                }
                if not selected:
                    score_summary = ", ".join(
                        f"{name}={scores[name]:.6g}" for name in candidate_frames
                    )
                    raise RuntimeError(
                        f"skill {label} has no task parameter above "
                        f"tau_omega={self.config.tau_omega}; fix demonstration coverage "
                        "or the configured threshold rather than applying a silent "
                        f"fallback; scores: {score_summary}"
                    )
                # Algorithm 1 line 7 uses the selected per-frame product as the
                # task-parameterized MiDiGaP input.  Labels are discovered once.
                modal_local_streams = {name: local_policy[name] for name in selected}
                modal_global_streams = {
                    "gripper": _gripper_modal_factor(
                        extra["gripper"], self.config.gripper_clustering_scale
                    )
                }
                modal_product, modal_euclidean = _resampled_product_modal_data(
                    modal_local_streams,
                    self.config,
                    # TAPAS-GMM Sec. IV-B adds gripper width once as a global
                    # R factor.  The unpublished relative unit is an explicit
                    # config value; emitted commands remain in native units.
                    modal_global_streams,
                )
                mode_labels = _partition_resampled_modes(
                    modal_product,
                    self.config,
                    modal_euclidean,
                )
                skill_audit["modal_partition"] = _modal_partition_audit(
                    modal_product,
                    modal_euclidean,
                    mode_labels,
                    self.config,
                )
                modes = int(np.max(mode_labels)) + 1
                eq5_availability = {
                    name: np.repeat(pooled_availability[name][None, :], modes, axis=0)
                    for name in candidate_frames
                }
                eq6_selected = {
                    name: np.full(modes, pooled_eq6_selected[name], dtype=bool)
                    for name in candidate_frames
                }
            else:
                # A pooled Gaussian can erase the active/idle structure needed
                # by Eq. (5)/(6).  The papers do not resolve that multimodal
                # corner case.  For the explicit inferred option, cluster the
                # product of *all initial real candidate frames* exactly once,
                # before Algorithm 1 line 5 introduces virtual frames; freeze
                # those labels for the mode-conditioned preliminary DiGaPs and
                # the final streams.  No selected-frame feedback loop is used.
                modal_product, modal_euclidean = _resampled_product_modal_data(
                    {name: local_policy[name] for name in frame_names},
                    self.config,
                    {
                        "gripper": _gripper_modal_factor(
                            extra["gripper"], self.config.gripper_clustering_scale
                        )
                    },
                )
                mode_labels = _partition_resampled_modes(
                    modal_product,
                    self.config,
                    modal_euclidean,
                )
                skill_audit["modal_partition"] = _modal_partition_audit(
                    modal_product,
                    modal_euclidean,
                    mode_labels,
                    self.config,
                )
                modes = int(np.max(mode_labels)) + 1

                policy_covariance: dict[str, Array] = {}
                for name in candidate_frames:
                    per_mode_covariance = []
                    for mode in range(modes):
                        _, covariance = _fit_pose_sequence(
                            local_policy[name][mode_labels == mode],
                            self.config.position_variance_floor,
                            self.config.rotation_variance_floor,
                            covariance_estimation_method=self.config.covariance_estimation_method,
                        )
                        per_mode_covariance.append(covariance)
                    policy_covariance[name] = np.stack(per_mode_covariance)
                selection_covariance = (
                    policy_covariance
                    if self.config.eq6_covariance_scope == "full_pose"
                    else {
                        name: _weighted_pose_covariance(
                            covariance,
                            position_weight=self.config.eq5_position_weight,
                            rotation_weight=self.config.eq5_rotation_weight,
                        )
                        for name, covariance in policy_covariance.items()
                    }
                )

                eq5_availability = {
                    name: np.ones((modes, duration), dtype=bool)
                    for name in candidate_frames
                }
                for name in frame_names:
                    mode_scales = []
                    mode_raw_masks = []
                    mode_linked_masks = []
                    per_mode_diagnostics = []
                    for mode in range(modes):
                        link_mean, covariance = _fit_pose_sequence(
                            local_policy[name][mode_labels == mode],
                            self.config.position_variance_floor,
                            self.config.rotation_variance_floor,
                            covariance_estimation_method=self.config.covariance_estimation_method,
                        )
                        scale = geometric_mean_standard_deviation(
                            covariance,
                            position_weight=self.config.eq5_position_weight,
                            rotation_weight=self.config.eq5_rotation_weight,
                        )
                        raw_mask, linked_mask = _kinematic_link_masks(
                            scale,
                            self.config,
                            local_mean=link_mean,
                        )
                        temporal_curve, temporal_valid = _temporal_variance_curve(
                            link_mean,
                            self.config.temporal_variance_window,
                        )
                        eq5_availability[name][mode] = ~linked_mask
                        mode_scales.append(scale)
                        mode_raw_masks.append(raw_mask)
                        mode_linked_masks.append(linked_mask)
                        per_mode_diagnostics.append(
                            {
                                "mode": mode,
                                "demonstration_indices": [
                                    int(index)
                                    for index in np.flatnonzero(mode_labels == mode)
                                ],
                                "linked": bool(np.any(linked_mask)),
                                "fully_linked": bool(np.all(linked_mask)),
                                "raw_linked_fraction": float(np.mean(raw_mask)),
                                "linked_fraction": float(np.mean(linked_mask)),
                                "raw_maximum_link_run": _maximum_true_run(raw_mask),
                                "maximum_link_run": _maximum_true_run(linked_mask),
                                "minimum_m": float(np.min(scale)),
                                "median_m": float(np.median(scale)),
                                "gmsd": scale.tolist(),
                                "raw_link_mask": raw_mask.tolist(),
                                "filtered_link_mask": linked_mask.tolist(),
                                "mask_scope": self.config.link_mask_scope,
                                **(
                                    {
                                        "majority_gate_enabled": _majority_gate_audit(
                                            raw_mask, self.config
                                        ),
                                        "majority_gate_rule": (
                                            "strict_mean_raw_linked_gt_0.5"
                                        ),
                                    }
                                    if self.config.link_mask_scope
                                    == "skill_majority_gate_timestep"
                                    else {}
                                ),
                                "kinematic_analysis_enabled": (
                                    self.config.kinematic_analysis_enabled
                                ),
                                "analysis_performed": (
                                    self.config.kinematic_analysis_enabled
                                ),
                                "disabled_behavior": (
                                    None
                                    if self.config.kinematic_analysis_enabled
                                    else "explicit_bypass_all_dynamic_candidates_available"
                                ),
                                "skill_linked": (
                                    bool(linked_mask[0])
                                    if self.config.link_mask_scope == "skill_majority"
                                    else None
                                ),
                            }
                        )
                    scales = np.stack(mode_scales)
                    raw_masks = np.stack(mode_raw_masks)
                    linked_masks = np.stack(mode_linked_masks)
                    link_diagnostics[name] = {
                        "linked": bool(np.any(linked_masks)),
                        "fully_linked": bool(np.all(linked_masks)),
                        "raw_linked_fraction": float(np.mean(raw_masks)),
                        "linked_fraction": float(np.mean(linked_masks)),
                        "raw_maximum_link_run": max(
                            _maximum_true_run(mask) for mask in raw_masks
                        ),
                        "maximum_link_run": max(
                            _maximum_true_run(mask) for mask in linked_masks
                        ),
                        "minimum_m": float(np.min(scales)),
                        "median_m": float(np.median(scales)),
                        "gmsd": (scales[0] if modes == 1 else scales).tolist(),
                        "raw_link_mask": (
                            raw_masks[0] if modes == 1 else raw_masks
                        ).tolist(),
                        "filtered_link_mask": (
                            linked_masks[0] if modes == 1 else linked_masks
                        ).tolist(),
                        "filter": self.config.link_filter,
                        "mask_scope": self.config.link_mask_scope,
                        **(
                            {
                                "majority_gate_enabled": (
                                    [
                                        _majority_gate_audit(mask, self.config)
                                        for mask in mode_raw_masks
                                    ]
                                    if modes > 1
                                    else _majority_gate_audit(
                                        mode_raw_masks[0], self.config
                                    )
                                ),
                                "majority_gate_rule": ("strict_mean_raw_linked_gt_0.5"),
                            }
                            if self.config.link_mask_scope
                            == "skill_majority_gate_timestep"
                            else {}
                        ),
                        "kinematic_analysis_enabled": (
                            self.config.kinematic_analysis_enabled
                        ),
                        "analysis_performed": self.config.kinematic_analysis_enabled,
                        "disabled_behavior": (
                            None
                            if self.config.kinematic_analysis_enabled
                            else "explicit_bypass_all_dynamic_candidates_available"
                        ),
                        "scope": (
                            "precluster_all_real_frame_product_mode_conditioned_"
                            "preliminary_digap_INFERRED_IMPLEMENTATION"
                        ),
                        "per_mode": per_mode_diagnostics,
                    }
                    skill_audit["kinematic_links"][name] = {
                        "scope": "mode_conditioned_INFERRED_IMPLEMENTATION",
                        "per_mode": deepcopy(per_mode_diagnostics),
                    }

                per_mode_selection = [
                    _eq6_skill_selection(
                        {
                            name: selection_covariance[name][mode]
                            for name in candidate_frames
                        },
                        self.config.tau_omega,
                        availability={
                            name: eq5_availability[name][mode]
                            for name in candidate_frames
                        },
                        candidate_kind=candidate_kind,
                        empty_selection=self.config.eq6_empty_selection,
                        semantics_id=self.selection_semantics_id,
                    )
                    for mode in range(modes)
                ]
                per_mode_scores = [item[2]["scores"] for item in per_mode_selection]
                scores = {
                    name: max(mode_scores[name] for mode_scores in per_mode_scores)
                    for name in candidate_frames
                }
                skill_audit["task_parameter_selection"] = {
                    "scope": "mode_conditioned_INFERRED_IMPLEMENTATION",
                    "per_mode_scores": deepcopy(per_mode_scores),
                    "per_mode_details": [
                        deepcopy(item[2]) for item in per_mode_selection
                    ],
                    "scores": deepcopy(scores),
                    "candidate_covariance": {
                        name: covariance.copy()
                        for name, covariance in policy_covariance.items()
                    },
                    "selection_covariance": {
                        name: covariance.copy()
                        for name, covariance in selection_covariance.items()
                    },
                    "eq6_covariance_scope": self.config.eq6_covariance_scope,
                    "eq6_covariance_scope_source_status": (
                        "LOCAL_AUTHOR_EMAIL_INTERPRETATION"
                        if self.config.eq6_covariance_scope == "eq5_weighted_subspace"
                        else "PAPER_EQ6_FULL_POSE"
                    ),
                    "eq5_availability": {
                        name: mask.copy() for name, mask in eq5_availability.items()
                    },
                    "normalization_scope": "eq5_available_candidate_frames_per_timestep",
                    "eq5_filters_eq6_candidates": (
                        self.config.kinematic_analysis_enabled
                    ),
                    "eq5_filters_eq6_denominator": (
                        self.config.kinematic_analysis_enabled
                    ),
                    "tau_omega": self.config.tau_omega,
                    "kinematic_analysis_enabled": (
                        self.config.kinematic_analysis_enabled
                    ),
                    "eq5_bypass": (
                        None
                        if self.config.kinematic_analysis_enabled
                        else "all_dynamic_candidates_available"
                    ),
                }
                # The union is serialized, while Eq. (6)-rejected frames stay
                # inactive in the corresponding component's PoE.  DynaMAC does
                # not publish this [M,F] mask, so it remains part of the named
                # inferred multimodal option rather than the strict baseline.
                eq6_selected = {
                    name: np.asarray(
                        [item[1][name] for item in per_mode_selection],
                        dtype=bool,
                    )
                    for name in candidate_frames
                }
                selected = tuple(
                    name for name in candidate_frames if np.any(eq6_selected[name])
                )
            if not selected:
                score_summary = ", ".join(
                    f"{name}={scores[name]:.6g}" for name in candidate_frames
                )
                raise RuntimeError(
                    f"skill {label} has no task parameter above "
                    f"tau_omega={self.config.tau_omega}; fix demonstration coverage "
                    "or the configured threshold rather than applying a silent "
                    f"fallback; scores: {score_summary}"
                )
            # Eq. (6) 决定技能级 frame selection；最终 PoE 再与 Eq. (5)
            # 的配置粒度 availability（V3 为多数门控后的逐时刻 raw mask）
            # 严格取交集。
            framewise_participation = _compose_framewise_poe_participation(
                eq5_availability,
                eq6_selected,
            )
            selection_audit = skill_audit["task_parameter_selection"]
            selection_audit.update(
                {
                    "kinematic_link_granularity": (
                        "disabled_all_dynamic_candidates_available"
                        if not self.config.kinematic_analysis_enabled
                        else (
                            "per_skill_strict_majority_eq5"
                            if self.config.link_mask_scope == "skill_majority"
                            else (
                                "skill_majority_gate_then_raw_per_timestep_eq5"
                                if self.config.link_mask_scope
                                == "skill_majority_gate_timestep"
                                else "per_timestep_within_skill_eq5"
                            )
                        )
                    ),
                    "task_parameter_selection_granularity": "per_skill_max_over_time_eq6",
                    "eq5_filters_eq6_candidates": (
                        self.config.kinematic_analysis_enabled
                    ),
                    "eq5_filters_eq6_denominator": (
                        self.config.kinematic_analysis_enabled
                    ),
                    "eq6_normalization_scope": (
                        "eq5_available_candidate_frames_per_timestep"
                    ),
                    "selected_by_eq6": {
                        name: mask.copy() for name, mask in eq6_selected.items()
                    },
                    "poe_participation_mask": {
                        name: mask.copy()
                        for name, mask in framewise_participation.items()
                    },
                }
            )
            selected_coverage = np.zeros((modes, duration), dtype=bool)
            for name in selected:
                selected_coverage |= framewise_participation[name]
            if not np.all(selected_coverage):
                missing_mode_times = [
                    [int(mode), int(time_index)]
                    for mode, time_index in np.argwhere(~selected_coverage)
                ]
                raise RuntimeError(
                    f"skill {label} selected task parameters are all linked at "
                    f"[mode, time] {missing_mode_times}; the DynaMAC paper does not "
                    "define a no-expert fallback"
                )

            streams = {}
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
                        local_policy[name][members],
                        self.config.position_variance_floor,
                        self.config.rotation_variance_floor,
                        covariance_estimation_method=self.config.covariance_estimation_method,
                    )
                    means.append(mean)
                    covariances.append(covariance)
                streams[name] = StreamModel(
                    name,
                    np.stack(means),
                    np.stack(covariances),
                    (
                        framewise_participation[name][0].copy()
                        if modes == 1
                        else framewise_participation[name].copy()
                    ),
                    (
                        eq5_availability[name][0].copy()
                        if modes == 1
                        else eq5_availability[name].copy()
                    ),
                    eq6_selected[name].copy(),
                )

            priors = np.asarray(
                [np.mean(mode_labels == mode) for mode in range(modes)],
                dtype=np.float64,
            )
            mode_demonstration_indices = tuple(
                tuple(int(index) for index in np.flatnonzero(mode_labels == mode))
                for mode in range(modes)
            )
            transition = (
                None
                if previous_mode_labels is None
                else _transition_probabilities(previous_mode_labels, mode_labels)
            )
            fitted_skills.append(
                SkillModel(
                    label=label,
                    duration=duration,
                    selected_frames=selected,
                    mode_priors=priors,
                    streams=streams,
                    gripper=np.stack(gripper_models),
                    transition_from_previous=transition,
                    mode_demonstration_indices=mode_demonstration_indices,
                    link_diagnostics=link_diagnostics,
                    selection_scores=scores,
                )
            )
            skill_audit.update(
                {
                    "selected_frames": tuple(selected),
                    "mode_labels": mode_labels.copy(),
                    "mode_priors": priors.copy(),
                    "mode_demonstration_indices": mode_demonstration_indices,
                    "transition_from_previous": (
                        None if transition is None else transition.copy()
                    ),
                    "final_streams": {
                        name: {
                            "mean": stream.mean.copy(),
                            "covariance": stream.covariance.copy(),
                            "precision": np.linalg.inv(stream.covariance),
                            "active": stream.active.copy(),
                            "availability": stream.availability.copy(),
                            "selected_by_eq6": stream.selected_by_eq6.copy(),
                            "semantics_id": self.selection_semantics_id,
                        }
                        for name, stream in streams.items()
                    },
                    "gripper_model": np.stack(gripper_models),
                }
            )
            self._training_audit["skills"].append(skill_audit)
            previous_mode_labels = mode_labels
        self.frame_names = frame_names
        self.skill_sequence = skill_sequence
        self.skills = fitted_skills
        self._invalidate_episode_after_fit()
        return self

    def _select_mode_path(
        self,
        strategy: Literal["map", "sample"],
        mode_evidence: Sequence[Array] | None = None,
    ) -> tuple[int, ...]:
        """按 MiDiGaP 式 (12)--(13)/(24) 选择整条技能模态路径。

        ``mode_evidence[k][m]`` 是第 ``k`` 个技能的模态 ``m`` 似然；
        它可由 MiDiGaP 的可达性、碰撞或运动学可行性证据产生。未提供时
        全为 1，严格退化为论文式 (13)。
        """

        if not self.skills:
            raise RuntimeError("DynaMAC 没有已拟合技能")
        if mode_evidence is None:
            evidence = tuple(np.ones_like(skill.mode_priors) for skill in self.skills)
        else:
            if len(mode_evidence) != len(self.skills):
                raise ValueError("模态证据必须与技能序列等长")
            validated = []
            for skill, values in zip(self.skills, mode_evidence, strict=True):
                current = _as_float_array(values)
                if (
                    current.shape != skill.mode_priors.shape
                    or np.any(~np.isfinite(current))
                    or np.any(current < 0.0)
                ):
                    raise ValueError("每个技能的模态证据必须为有限非负 [M] 向量")
                maximum = float(np.max(current))
                if maximum <= 0.0:
                    raise RuntimeError("模态证据排除了某个技能的全部模态")
                # Eq. (24) only depends on relative likelihoods.  Scaling first
                # prevents otherwise equivalent 1e-200/1e200 evidence from
                # underflowing or overflowing before the row normalization.
                validated.append(current / maximum)
            evidence = tuple(validated)

        # MiDiGaP Eq. (24): update the first-skill prior, then multiply every
        # incoming edge by the evidence of its target and normalize each source
        # row.  A zero row denotes a source mode with no evidence-compatible
        # continuation; it is pruned by the path recursion below.
        priors = self.skills[0].mode_priors * evidence[0]
        prior_total = float(np.sum(priors))
        if prior_total <= 0.0:
            raise RuntimeError("模态证据排除了所有 MiDiGaP 路径")
        priors = priors / prior_total
        transitions: list[Array] = []
        for index, skill in enumerate(self.skills[1:], start=1):
            original = skill.transition_from_previous
            if original is None:
                raise RuntimeError("MiDiGaP 技能缺少模态转移矩阵")
            weighted = original * evidence[index][None, :]
            row_sum = np.sum(weighted, axis=1, keepdims=True)
            updated = np.divide(
                weighted,
                row_sum,
                out=np.zeros_like(weighted),
                where=row_sum > 0.0,
            )
            transitions.append(updated)

        if strategy == "sample":
            # Sample exactly from the finite Eq. (13) path distribution after
            # the local Eq. (24) updates.  Normalizing every backward message
            # is scale-invariant and prevents long-horizon underflow.
            backward: list[Array] = [np.empty(0)] * len(self.skills)
            backward[-1] = np.ones_like(self.skills[-1].mode_priors)
            for index in range(len(self.skills) - 2, -1, -1):
                message = transitions[index] @ backward[index + 1]
                maximum = float(np.max(message))
                backward[index] = message / maximum if maximum > 0.0 else message
            probabilities = priors * backward[0]
            total = float(np.sum(probabilities))
            if total <= 0.0:
                raise RuntimeError("模态证据排除了所有 MiDiGaP 路径")
            path = [int(self._rng.choice(len(probabilities), p=probabilities / total))]
            for index, transition in enumerate(transitions, start=1):
                probabilities = transition[path[-1]] * backward[index]
                total = float(np.sum(probabilities))
                if total <= 0.0:
                    raise RuntimeError("模态证据排除了已选前缀的所有后续路径")
                path.append(
                    int(self._rng.choice(len(probabilities), p=probabilities / total))
                )
            return tuple(path)
        if strategy != "map":
            raise ValueError(f"未知模态选择策略：{strategy}")

        scores = np.full(priors.shape, -np.inf, dtype=np.float64)
        np.log(priors, out=scores, where=priors > 0.0)
        backpointers: list[Array] = []
        for updated_transition in transitions:
            transition = np.full(updated_transition.shape, -np.inf, dtype=np.float64)
            np.log(
                updated_transition,
                out=transition,
                where=updated_transition > 0.0,
            )
            candidates = scores[:, None] + transition
            backpointers.append(np.argmax(candidates, axis=0))
            scores = np.max(candidates, axis=0)
        if not np.any(np.isfinite(scores)):
            raise RuntimeError("模态证据排除了所有 MiDiGaP 路径")
        path = [int(np.argmax(scores))]
        for backpointer in reversed(backpointers):
            path.append(int(backpointer[path[-1]]))
        return tuple(reversed(path))

    def reset(
        self,
        observation: DynaMACObservation,
        mode_strategy: Literal["map", "sample"] | None = None,
        mode_evidence: Sequence[Array] | None = None,
    ) -> None:
        """开始一个 episode；失败时保持原执行状态和 RNG 状态。"""

        state = self._capture_runtime_state()
        try:
            self._reset_impl(observation, mode_strategy, mode_evidence)
        except Exception:
            self._restore_runtime_state(state)
            raise

    def restart_current_skill_reference(self) -> tuple[int, int]:
        """Reset only the fixed-clock action reference to this skill's entry.

        This is an evaluation recovery interface used by generic Skill-Retry;
        it does not reset the simulator, resample a mode, recapture an object
        frame, or alter any learned distribution.  The ordinary DynaMAC path
        never calls it.
        """

        if not self._episode_initialized:
            raise RuntimeError("DynaMAC 尚未 reset，不能重启技能引用")
        self._time_index = 0
        self._complete = False
        self._pending_virtual_capture = False
        self._active_mode = self._mode_path[self._skill_index]
        return self._skill_index, self._time_index

    def _reset_impl(
        self,
        observation: DynaMACObservation,
        mode_strategy: Literal["map", "sample"] | None,
        mode_evidence: Sequence[Array] | None,
    ) -> None:
        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        selected_strategy = (
            self.config.default_mode_strategy
            if mode_strategy is None
            else mode_strategy
        )
        mode_path = self._select_mode_path(selected_strategy, mode_evidence)
        evidence = (
            tuple(np.ones_like(skill.mode_priors) for skill in self.skills)
            if mode_evidence is None
            else tuple(_as_float_array(value).copy() for value in mode_evidence)
        )
        first_skill = self.skills[0]
        missing = {
            name
            for name in first_skill.selected_frames
            if not name.startswith("virtual_skill_")
            and first_skill.streams[name].is_active(mode_path[0], 0)
            and name not in observation.frames
        }
        if missing:
            raise ValueError(f"观测缺少首时刻已选择任务参数：{sorted(missing)}")
        self._skill_index = 0
        self._time_index = 0
        self._complete = False
        self._virtual_frames = {
            f"virtual_skill_{self.current_skill.label}": observation.ee_pose.copy()
        }
        self._pending_virtual_capture = False
        self._mode_strategy = selected_strategy
        self._mode_path = mode_path
        self._mode_evidence = evidence
        self._active_mode = self._mode_path[0]
        self._episode_initialized = True

    def _frame_pose(self, name: str, observation: DynaMACObservation) -> Array:
        if name.startswith("virtual_skill_"):
            if name in self._virtual_frames:
                return self._virtual_frames[name]
            # The closed-loop controller may query a legal realignment/reentry
            # state whose virtual frame is owned by its runtime snapshot rather
            # than the baseline cursor.  Supplying that captured frame through
            # the observation keeps query_state read-only and leaves baseline
            # skill-boundary capture semantics unchanged.
            if name in observation.frames:
                return observation.frames[name]
            raise RuntimeError(f"虚拟帧 {name} 尚未在技能边界捕获或随观测提供")
        if name not in observation.frames:
            raise ValueError(f"观测缺少已选择任务参数 {name}")
        return observation.frames[name]

    @staticmethod
    def _query_state_components(
        state_id: Any,
        mode_index: int | None,
    ) -> tuple[int, int, int | None]:
        """Normalize progress state and the independent mode component."""

        if all(hasattr(state_id, name) for name in ("skill_index", "local_index")):
            values = (state_id.skill_index, state_id.local_index)
        else:
            try:
                raw_values = tuple(state_id)
            except TypeError as exc:
                raise TypeError("state_id 必须是 StateId 或二元组") from exc
            if len(raw_values) != 2:
                raise ValueError("state_id 必须只包含 skill_index 和 local_index")
            values = raw_values
        if len(values) != 2:
            raise ValueError("state_id 必须包含 skill_index 和 local_index")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in values
        ):
            raise TypeError("state_id 的两个分量必须为整数")
        mode = mode_index
        if mode is not None and (
            isinstance(mode, (bool, np.bool_))
            or not isinstance(mode, (int, np.integer))
        ):
            raise TypeError("mode_index 必须为整数")
        return int(values[0]), int(values[1]), None if mode is None else int(mode)

    def query_state(
        self,
        observation: DynaMACObservation,
        state_id: Any,
        stream_weights: dict[str, float] | None = None,
        *,
        mode_index: int | None = None,
    ) -> DynaMACAction:
        """Query one fitted skill state without advancing any episode cursor.

        ``stream_weights=None`` follows the frozen DynaMAC Eq. (5)/Eq. (6)
        participation mask.  An explicit mapping is the closed-loop path: only
        named, positive-weight streams participate, and the supplied values
        scale their precisions.  Eq. (6)-rejected component streams can never
        be re-enabled by a runtime weight.
        """

        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        if not self._episode_initialized:
            raise RuntimeError("DynaMAC 尚未 reset，不能查询状态")
        skill_index, index, mode = self._query_state_components(state_id, mode_index)
        if skill_index < 0 or skill_index >= len(self.skills):
            raise IndexError("state_id 的 skill_index 超出范围")
        skill = self.skills[skill_index]
        if mode is None:
            mode = self._mode_path[skill_index]
        if index < 0 or index >= skill.duration:
            raise IndexError("state_id 的 local_index 超出范围")
        if mode < 0 or mode >= len(skill.mode_priors):
            raise IndexError("state_id 的 mode 超出范围")
        if stream_weights is not None:
            unknown = set(stream_weights).difference(skill.selected_frames)
            if unknown:
                raise ValueError(f"流权重包含当前技能未建模的参考系：{sorted(unknown)}")
            normalized_weights: dict[str, float] = {}
            for name, value in stream_weights.items():
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError("流精度权重必须为有限非负实数")
                normalized_weights[name] = float(value)
        else:
            normalized_weights = {}

        marginals: list[GaussianMarginal] = []
        precision_weights: list[float] = []
        inactive_frames: list[str] = []
        effective_weights: dict[str, float] = {}
        mask_indices: dict[str, int] = {}
        for name in skill.selected_frames:
            stream = skill.streams[name]
            mask_index = index
            if self.config.link_mask_scope == "skill_majority":
                active_mask = (
                    stream.active if stream.active.ndim == 1 else stream.active[mode]
                )
                availability_mask = (
                    stream.availability
                    if stream.availability.ndim == 1
                    else stream.availability[mode]
                )
                if not (
                    np.all(active_mask == active_mask[0])
                    and np.all(availability_mask == availability_mask[0])
                ):
                    raise RuntimeError("schema 13 skill mask 在技能内必须恒定")
                mask_index = 0
            mask_indices[name] = mask_index
            selected = stream.is_selected(mode)
            weight = (
                float(stream.is_active(mode, mask_index))
                if stream_weights is None
                else normalized_weights.get(name, 0.0)
            )
            if not selected:
                weight = 0.0
            effective_weights[name] = weight
            if weight <= 0.0:
                inactive_frames.append(name)
                continue
            marginals.append(
                transform_marginal(
                    name,
                    self._frame_pose(name, observation),
                    stream.mean[mode, index],
                    stream.covariance[mode, index],
                    diagonalize=self.config.diagonalize_transformed_covariance,
                )
            )
            precision_weights.append(weight)
        if not marginals:
            raise RuntimeError(
                f"技能 {skill.label} 在状态 ({index}, {mode}) 没有正权重执行流"
            )
        pose, covariance, weights = product_of_experts(
            marginals,
            precision_weights=precision_weights,
        )
        gripper = skill.gripper[mode, index].copy()
        diagnostics = {
            "method": self.name,
            "selection_semantics_id": self.selection_semantics_id,
            "skill_index": skill_index,
            "skill_label": skill.label,
            "time_index": index,
            "duration": skill.duration,
            "mode": mode,
            "mode_prior": float(skill.mode_priors[mode]),
            "mode_evidence": float(self._mode_evidence[skill_index][mode]),
            "modal_path": list(self._mode_path),
            "path_probability_factor": (
                float(skill.mode_priors[mode])
                if skill_index == 0
                else float(
                    skill.transition_from_previous[
                        self._mode_path[skill_index - 1], mode
                    ]
                )
            ),
            "selected_frames": list(skill.selected_frames),
            "active_frames": [item.frame for item in marginals],
            "inactive_linked_frames": inactive_frames,
            "frame_status": {
                name: {
                    "selected_by_eq6_for_skill": skill.streams[name].is_selected(mode),
                    "exogenous_for_skill_by_eq5": (
                        skill.streams[name].is_available(mode, mask_indices[name])
                        if self.config.link_mask_scope == "skill_majority"
                        else None
                    ),
                    "exogenous_at_t_by_eq5": skill.streams[name].is_available(
                        mode, mask_indices[name]
                    ),
                    "participates_in_poe_at_t": effective_weights[name] > 0.0,
                }
                for name in skill.selected_frames
            },
            "marginal_means": {item.frame: item.mean.tolist() for item in marginals},
            "marginal_covariances": {
                item.frame: item.covariance.tolist() for item in marginals
            },
            "frame_poses": {
                item.frame: self._frame_pose(item.frame, observation).tolist()
                for item in marginals
            },
            "captured_virtual_frames": {
                name: pose.tolist() for name, pose in self._virtual_frames.items()
            },
            "poe_weights": weights,
            "joint_covariance": covariance.tolist(),
            "selection_mode": (
                "eq6_with_kinematic_analysis_disabled"
                if not self.config.kinematic_analysis_enabled
                else (
                    "eq6_per_skill_with_eq5_skill_mask"
                    if self.config.link_mask_scope == "skill_majority"
                    else (
                        "eq6_per_skill_with_majority_gated_eq5_framewise_participation"
                        if self.config.link_mask_scope == "skill_majority_gate_timestep"
                        else "eq6_per_skill_with_eq5_framewise_participation"
                    )
                )
            ),
            "kinematic_link_granularity": (
                "disabled_all_dynamic_candidates_available"
                if not self.config.kinematic_analysis_enabled
                else (
                    "offline_per_skill_strict_majority"
                    if self.config.link_mask_scope == "skill_majority"
                    else (
                        "offline_skill_majority_gate_then_raw_per_timestep"
                        if self.config.link_mask_scope == "skill_majority_gate_timestep"
                        else "offline_per_timestep_within_skill"
                    )
                )
            ),
            "kinematic_analysis_enabled": self.config.kinematic_analysis_enabled,
            "task_parameter_selection_granularity": "offline_per_skill_max_over_time",
            "online_link_detection": False,
            "query_advances_clock": False,
        }
        if stream_weights is not None:
            diagnostics.update(
                {
                    "requested_stream_weights": normalized_weights,
                    "effective_stream_weights": effective_weights,
                }
            )
        return DynaMACAction(
            pose=pose,
            covariance=covariance,
            gripper=gripper,
            diagnostics=diagnostics,
        )

    def preview_next_gripper(self) -> DynaMACGripperLookahead:
        """Preview ``g[t + 1]`` without predicting another pose or moving time.

        Call this immediately before :meth:`act` while the runtime cursor still
        denotes tick ``t``.  Internal transitions read the following sample in
        the same skill.  At a skill boundary the preview follows the already
        selected episode ``mode_path`` into the next skill.  At the end of the
        episode (and after completion) it repeats the final command so a
        uniform lookahead never invents a new terminal state.

        The method has no observation argument by design.  It is safe to use
        inside a tentative action transaction because it is a pure read of the
        discrete gripper model and leaves the captured runtime byte-for-byte
        unchanged.
        """

        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        if not self._episode_initialized:
            raise RuntimeError("DynaMAC 尚未 reset，不能预览 gripper")
        if len(self._mode_path) != len(self.skills):
            raise RuntimeError("DynaMAC episode mode path is incomplete")

        source_skill_index = self._skill_index
        source_skill = self.skills[source_skill_index]
        source_time_index = min(self._time_index, source_skill.duration - 1)
        crosses_boundary = False
        repeats_terminal = False
        if self._complete:
            next_skill_index = len(self.skills) - 1
            next_time_index = self.skills[next_skill_index].duration - 1
            repeats_terminal = True
        elif source_time_index + 1 < source_skill.duration:
            next_skill_index = source_skill_index
            next_time_index = source_time_index + 1
        elif source_skill_index + 1 < len(self.skills):
            next_skill_index = source_skill_index + 1
            next_time_index = 0
            crosses_boundary = True
        else:
            next_skill_index = source_skill_index
            next_time_index = source_skill.duration - 1
            repeats_terminal = True

        next_skill = self.skills[next_skill_index]
        next_mode = self._mode_path[next_skill_index]
        if next_mode < 0 or next_mode >= next_skill.gripper.shape[0]:
            raise RuntimeError("DynaMAC episode mode path references a missing mode")
        if (
            next_skill.duration < 1
            or next_skill.gripper.shape[1] != next_skill.duration
        ):
            raise RuntimeError("DynaMAC skill has an invalid gripper duration")
        return DynaMACGripperLookahead(
            gripper=next_skill.gripper[next_mode, next_time_index].copy(),
            crosses_skill_boundary=crosses_boundary,
            repeats_terminal=repeats_terminal,
            next_skill_index=next_skill_index,
            next_skill_label=next_skill.label,
            next_time_index=next_time_index,
            next_mode=next_mode,
        )

    def act(self, observation: DynaMACObservation) -> DynaMACAction:
        """按固定离散时间执行；失败时不推进技能、时间或虚拟帧状态。"""

        state = self._capture_runtime_state()
        try:
            return self._act_impl(observation)
        except Exception:
            self._restore_runtime_state(state)
            raise

    def _act_impl(self, observation: DynaMACObservation) -> DynaMACAction:
        """非事务执行内核；不读取接触或在线链接状态。"""

        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        if not self._episode_initialized:
            raise RuntimeError("DynaMAC 尚未 reset，不能执行 act")
        if self._complete:
            raise RuntimeError("DynaMAC 已完成")
        if self._pending_virtual_capture:
            virtual_name = f"virtual_skill_{self.current_skill.label}"
            self._virtual_frames[virtual_name] = observation.ee_pose.copy()
            self._pending_virtual_capture = False
        skill = self.current_skill
        index = min(self._time_index, skill.duration - 1)
        action = self.query_state(
            observation,
            (self._skill_index, index),
            mode_index=self._active_mode,
        )
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
        return action

    def summary(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("DynaMAC 尚未拟合")
        demonstration_count = self._validate_and_get_demonstration_count()
        return {
            "implementation": (
                "DynaMAC Algorithm 1 + task-parameterized MiDiGaP"
                if self.name == "dynamac"
                else "static-frame task-parameterized MiDiGaP baseline"
            ),
            "policy_type": self.name,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "selection_semantics_id": self.selection_semantics_id,
            "demonstration_count": demonstration_count,
            "tapas_reference_commit": TAPAS_REFERENCE_COMMIT,
            "config": asdict(self.config),
            "frame_names": list(self.frame_names),
            "skill_sequence": list(self.skill_sequence),
            "skills": [
                {
                    "label": skill.label,
                    "duration": skill.duration,
                    "modes": len(skill.mode_priors),
                    "mode_priors": skill.mode_priors.tolist(),
                    "mode_demonstration_indices": [
                        list(indices) for indices in skill.mode_demonstration_indices
                    ],
                    "transition_from_previous": (
                        None
                        if skill.transition_from_previous is None
                        else skill.transition_from_previous.tolist()
                    ),
                    "selected_frames": list(skill.selected_frames),
                    "active_fraction": {
                        name: float(np.mean(skill.streams[name].active))
                        for name in skill.selected_frames
                    },
                    "availability_fraction": {
                        name: float(np.mean(skill.streams[name].availability))
                        for name in skill.selected_frames
                    },
                    "selected_by_eq6": {
                        name: skill.streams[name].selected_by_eq6.tolist()
                        for name in skill.selected_frames
                    },
                    "link_diagnostics": skill.link_diagnostics,
                    "selection_scores": skill.selection_scores,
                }
                for skill in self.skills
            ],
        }

    def _validate_and_get_demonstration_count(self) -> int:
        """验证 checkpoint 中用于重算 MiDiGaP 式 (12) 的模式成员分区。"""

        demonstration_count: int | None = None
        for skill in self.skills:
            membership = skill.mode_demonstration_indices
            if (
                not membership
                or len(membership) != len(skill.mode_priors)
                or any(not members for members in membership)
            ):
                raise ValueError(
                    f"技能 {skill.label} 的模式演示成员必须与模态一一对应且均非空"
                )
            flattened = [index for members in membership for index in members]
            if any(index < 0 for index in flattened) or len(set(flattened)) != len(
                flattened
            ):
                raise ValueError(f"技能 {skill.label} 的模式演示成员含负数或重复 index")
            if sorted(flattened) != list(range(len(flattened))):
                raise ValueError(
                    f"技能 {skill.label} 的模式演示成员必须完整覆盖连续 index"
                )
            if demonstration_count is None:
                demonstration_count = len(flattened)
            elif len(flattened) != demonstration_count:
                raise ValueError("所有技能的模式演示成员必须覆盖同一批演示")
            expected_priors = np.asarray(
                [len(members) / len(flattened) for members in membership],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(skill.mode_priors)) or not np.allclose(
                skill.mode_priors, expected_priors
            ):
                raise ValueError(f"技能 {skill.label} 的模态先验与演示成员比例不一致")
        if demonstration_count is None:
            raise ValueError("DynaMAC checkpoint 没有技能")
        return demonstration_count

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
                digest.update(skill.streams[name].active.tobytes())
                digest.update(skill.streams[name].availability.tobytes())
                digest.update(skill.streams[name].selected_by_eq6.tobytes())
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
                arrays[self._array_key(index, name, "active")] = stream.active
                arrays[self._array_key(index, name, "availability")] = (
                    stream.availability
                )
                arrays[self._array_key(index, name, "selected_by_eq6")] = (
                    stream.selected_by_eq6
                )
        # 传入文件对象可阻止 NumPy 在无后缀路径后静默追加 ``.npz``，使
        # ``save(path)`` 与 ``load(path)`` 对任意合法路径都严格互逆。
        with path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                metadata_json=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                ),
                **arrays,
            )

    @classmethod
    def load(cls, path: str | Path) -> DynaMAC:
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            schema = metadata.get("model_schema_version")
            if schema == 2:
                raise ValueError(
                    "DynaMAC schema 2 使用旧 full-angle/world-tangent 数学，且未保存逐时刻 "
                    "active mask/虚拟帧历史，无法无损迁移；请用当前实现从演示重新拟合"
                )
            if schema == 3:
                raise ValueError(
                    "DynaMAC schema 3 未保存可审计的模态演示成员集合；"
                    f"请用当前实现从演示重新拟合为 schema {MODEL_SCHEMA_VERSION}"
                )
            if schema in {4, 5}:
                raise ValueError(
                    f"DynaMAC schema {schema} 早于当前 preliminary-analysis/product-"
                    "clustering 配置语义，无法验证 fingerprint；"
                    f"请从演示重新拟合为 schema {MODEL_SCHEMA_VERSION}"
                )
            if schema == 6:
                raise ValueError(
                    "DynaMAC schema 6 excludes the global gripper action from the modal "
                    "product and can correlate both arms through the same seed; refit the "
                    f"demonstrations with schema {MODEL_SCHEMA_VERSION}"
                )
            if schema == 7:
                raise ValueError(
                    "DynaMAC schema 7 does not store the dimensional scale between the "
                    "gripper Euclidean factor and pose factors; refit the demonstrations "
                    f"with schema {MODEL_SCHEMA_VERSION}"
                )
            if schema == 9:
                raise ValueError(
                    "DynaMAC schema 9 silently mixes diagonal empirical covariance with "
                    "additive ridge regularization; refit the demonstrations with schema "
                    f"{MODEL_SCHEMA_VERSION}"
                )
            if schema == 10:
                raise ValueError(
                    "DynaMAC schema 10 predates the current Eq. (5)/Eq. (6) candidate "
                    "filtering and empty-denominator rules; refit the demonstrations with "
                    f"schema {MODEL_SCHEMA_VERSION}"
                )
            if schema == 11:
                raise ValueError(
                    "DynaMAC schema 11 applies Eq. (6) normalization to candidates "
                    "unavailable under Eq. (5) and applies availability only in the final "
                    "PoE; refit "
                    f"the demonstrations with schema {MODEL_SCHEMA_VERSION}"
                )
            if schema == 12:
                raise ValueError(
                    "DynaMAC schema 12 uses per-timestep link masks, full-pose Eq. (5), "
                    "a next-action policy stream, and default modal partitioning; refit "
                    "the demonstrations with the current schema"
                )
            if schema != MODEL_SCHEMA_VERSION:
                raise ValueError("unsupported DynaMAC checkpoint schema")
            expected_policy_type = cls.name
            if metadata.get("policy_type") != expected_policy_type:
                raise ValueError(
                    f"checkpoint policy type {metadata.get('policy_type')!r} cannot be "
                    f"loaded by {expected_policy_type!r}"
                )
            config = DynaMACConfig(**metadata["config"])
            if metadata.get("selection_semantics_id") != _selection_semantics_id(
                config
            ):
                raise ValueError(
                    "DynaMAC checkpoint 的任务参数选择 semantics_id 不匹配"
                )
            policy = cls(config)
            policy.frame_names = tuple(metadata["frame_names"])
            policy.skill_sequence = tuple(
                int(value) for value in metadata["skill_sequence"]
            )
            for index, skill_meta in enumerate(metadata["skills"]):
                if "mode_demonstration_indices" not in skill_meta:
                    raise ValueError("DynaMAC checkpoint 缺少模式演示成员")
                selected = tuple(skill_meta["selected_frames"])
                streams = {}
                for name in selected:
                    streams[name] = StreamModel(
                        name,
                        archive[policy._array_key(index, name, "mean")].copy(),
                        archive[policy._array_key(index, name, "covariance")].copy(),
                        archive[policy._array_key(index, name, "active")].copy(),
                        archive[policy._array_key(index, name, "availability")].copy(),
                        archive[
                            policy._array_key(index, name, "selected_by_eq6")
                        ].copy(),
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
                            None
                            if index == 0
                            else archive[f"skill_{index}__transition"].copy()
                        ),
                        mode_demonstration_indices=tuple(
                            tuple(int(value) for value in indices)
                            for indices in skill_meta["mode_demonstration_indices"]
                        ),
                        link_diagnostics=skill_meta["link_diagnostics"],
                        selection_scores={
                            name: float(value)
                            for name, value in skill_meta["selection_scores"].items()
                        },
                    )
                )
        if policy._validate_and_get_demonstration_count() != metadata.get(
            "demonstration_count"
        ):
            raise ValueError("DynaMAC checkpoint 的演示数量与模式成员不一致")
        if policy.fingerprint() != metadata.get("fingerprint"):
            raise ValueError("DynaMAC checkpoint 指纹不一致")
        return policy


@dataclass(frozen=True)
class BimanualDynaMACAction:
    left: DynaMACAction
    right: DynaMACAction


@dataclass(frozen=True)
class BimanualDynaMACGripperLookahead:
    """Independent read-only next-tick gripper previews for both arms."""

    left: DynaMACGripperLookahead
    right: DynaMACGripperLookahead


def synchronized_bimanual_demonstrations(
    left_demonstrations: Sequence[DynaMACDemonstration],
    right_demonstrations: Sequence[DynaMACDemonstration],
) -> tuple[list[DynaMACDemonstration], list[DynaMACDemonstration]]:
    """Inject each paired end effector as the other arm's synchronized frame."""

    if len(left_demonstrations) != len(right_demonstrations):
        raise ValueError("左右臂演示数量必须一致")
    if not left_demonstrations:
        raise ValueError("双臂 DynaMAC 至少需要一对演示")
    paired_left = []
    paired_right = []
    for left_demo, right_demo in zip(
        left_demonstrations, right_demonstrations, strict=True
    ):
        if len(left_demo.ee_pose) != len(right_demo.ee_pose):
            raise ValueError("成对双臂演示必须逐时刻对齐")
        left_frames = {
            name: value
            for name, value in left_demo.frames.items()
            if name != "right_ee"
        }
        left_frames["right_ee"] = right_demo.ee_pose
        right_frames = {
            name: value
            for name, value in right_demo.frames.items()
            if name != "left_ee"
        }
        right_frames["left_ee"] = left_demo.ee_pose
        paired_left.append(
            DynaMACDemonstration(
                ee_pose=left_demo.ee_pose,
                action_pose=left_demo.action_pose,
                gripper=left_demo.gripper,
                frames=left_frames,
                skill=left_demo.skill,
                name=left_demo.name,
                entity_configurations=left_demo.entity_configurations,
                scene_entity_poses=left_demo.scene_entity_poses,
                structural_bindings=left_demo.structural_bindings,
            )
        )
        paired_right.append(
            DynaMACDemonstration(
                ee_pose=right_demo.ee_pose,
                action_pose=right_demo.action_pose,
                gripper=right_demo.gripper,
                frames=right_frames,
                skill=right_demo.skill,
                name=right_demo.name,
                entity_configurations=right_demo.entity_configurations,
                scene_entity_poses=right_demo.scene_entity_poses,
                structural_bindings=right_demo.structural_bindings,
            )
        )
    return paired_left, paired_right


class BimanualDynaMAC:
    """论文 Sec. III-C：两套独立并发 DynaMAC，不共享技能或固定 leader。"""

    def __init__(
        self,
        left: DynaMAC | None = None,
        right: DynaMAC | None = None,
        config: DynaMACConfig = DynaMACConfig(),
    ) -> None:
        if left is not None and left is right:
            raise ValueError("the two arms must use two independent DynaMAC instances")
        # Sec. III-C specifies two independent concurrent policies.  Reusing
        # the same integer seed for equal-length MiDiGaP paths makes both RNGs
        # advance in lock-step and silently introduces a coordination device
        # that is absent from the paper.  SeedSequence gives deterministic,
        # independent substreams while retaining one user-facing base seed.
        child_sequences = np.random.SeedSequence(config.random_seed).spawn(2)
        child_seeds = tuple(
            int(sequence.generate_state(1, dtype=np.uint64)[0])
            for sequence in child_sequences
        )
        self.left = (
            left
            if left is not None
            else DynaMAC(replace(config, random_seed=child_seeds[0]))
        )
        self.right = (
            right
            if right is not None
            else DynaMAC(replace(config, random_seed=child_seeds[1]))
        )
        if self.left.config.random_seed == self.right.config.random_seed:
            raise ValueError(
                "independent arm policies cannot use the same random stream seed"
            )
        self._last_left_action: DynaMACAction | None = None
        self._last_right_action: DynaMACAction | None = None

    def fit(
        self,
        left_demonstrations: Sequence[DynaMACDemonstration],
        right_demonstrations: Sequence[DynaMACDemonstration],
    ) -> BimanualDynaMAC:
        paired_left, paired_right = synchronized_bimanual_demonstrations(
            left_demonstrations,
            right_demonstrations,
        )
        left_model = (self.left.frame_names, self.left.skill_sequence, self.left.skills)
        right_model = (
            self.right.frame_names,
            self.right.skill_sequence,
            self.right.skills,
        )
        left_runtime = self.left._capture_runtime_state()
        right_runtime = self.right._capture_runtime_state()
        try:
            self.left.fit(paired_left)
            self.right.fit(paired_right)
        except Exception:
            self.left.frame_names, self.left.skill_sequence, self.left.skills = (
                left_model
            )
            self.right.frame_names, self.right.skill_sequence, self.right.skills = (
                right_model
            )
            self.left._restore_runtime_state(left_runtime)
            self.right._restore_runtime_state(right_runtime)
            raise
        self._last_left_action = None
        self._last_right_action = None
        return self

    @property
    def complete(self) -> bool:
        return self.left.complete and self.right.complete

    def reset(
        self,
        left_observation: DynaMACObservation,
        right_observation: DynaMACObservation,
        mode_strategy: Literal["map", "sample"] | None = None,
        mode_evidence: Sequence[Array] | None = None,
        *,
        left_mode_evidence: Sequence[Array] | None = None,
        right_mode_evidence: Sequence[Array] | None = None,
    ) -> None:
        """Reset the two paper-defined independent DynaMAC policies.

        ``mode_evidence`` remains a convenience for symmetric models.  When
        the independently discovered left/right MiDiGaP partitions differ,
        callers supply side-specific evidence instead; no joint mode mapping
        or shared latent controller is introduced.
        """

        if mode_evidence is not None and (
            left_mode_evidence is not None or right_mode_evidence is not None
        ):
            raise ValueError("共享与分臂模态证据不能同时提供")
        if mode_evidence is not None:
            left_mode_evidence = mode_evidence
            right_mode_evidence = mode_evidence
        left_snapshot, right_snapshot = self._synchronous_observations(
            left_observation, right_observation
        )
        left_state = self.left._capture_runtime_state()
        right_state = self.right._capture_runtime_state()
        try:
            self.left.reset(left_snapshot, mode_strategy, left_mode_evidence)
            self.right.reset(right_snapshot, mode_strategy, right_mode_evidence)
        except Exception:
            self.left._restore_runtime_state(left_state)
            self.right._restore_runtime_state(right_state)
            raise
        self._last_left_action = None
        self._last_right_action = None

    def restart_current_skill_reference(self) -> dict[str, tuple[int, int]]:
        """Apply the same generic reference reset independently to both arms."""

        result = {
            "left": self.left.restart_current_skill_reference(),
            "right": self.right.restart_current_skill_reference(),
        }
        self._last_left_action = None
        self._last_right_action = None
        return result

    @staticmethod
    def _synchronous_observations(
        left_observation: DynaMACObservation,
        right_observation: DynaMACObservation,
    ) -> tuple[DynaMACObservation, DynaMACObservation]:
        """从同一控制周期快照注入对侧末端帧，杜绝顺序更新与陈旧副本。"""

        left_frames = {**left_observation.frames, "right_ee": right_observation.ee_pose}
        right_frames = {**right_observation.frames, "left_ee": left_observation.ee_pose}
        return (
            DynaMACObservation(left_observation.ee_pose, left_frames),
            DynaMACObservation(right_observation.ee_pose, right_frames),
        )

    def preview_next_gripper(self) -> BimanualDynaMACGripperLookahead:
        """Preview each independent arm's next command without advancing either.

        An arm that has already completed repeats its own final gripper command
        while the other arm continues.  Consequently asynchronous bimanual
        completion does not borrow a command or a boundary from the peer arm.
        """

        return BimanualDynaMACGripperLookahead(
            left=self.left.preview_next_gripper(),
            right=self.right.preview_next_gripper(),
        )

    def act(
        self,
        left_observation: DynaMACObservation,
        right_observation: DynaMACObservation,
    ) -> BimanualDynaMACAction:
        if self.complete:
            raise RuntimeError("bimanual DynaMAC has completed")
        left_snapshot, right_snapshot = self._synchronous_observations(
            left_observation, right_observation
        )
        # 两个动作都只读这一份 pre-action 快照；任一预测不会成为另一侧的当前帧。
        left_state = self.left._capture_runtime_state()
        right_state = self.right._capture_runtime_state()
        try:
            left_action = (
                self._completed_hold_action(self._last_left_action, "left")
                if self.left.complete
                else self.left.act(left_snapshot)
            )
            right_action = (
                self._completed_hold_action(self._last_right_action, "right")
                if self.right.complete
                else self.right.act(right_snapshot)
            )
        except Exception:
            self.left._restore_runtime_state(left_state)
            self.right._restore_runtime_state(right_state)
            raise
        self._last_left_action = left_action
        self._last_right_action = right_action
        return BimanualDynaMACAction(left=left_action, right=right_action)

    @staticmethod
    def _completed_hold_action(
        previous: DynaMACAction | None,
        arm: str,
    ) -> DynaMACAction:
        """Hold an independently completed arm while its peer finishes.

        This is concurrent-execution glue, not another policy or a recovery
        strategy: the completed DynaMAC emits no new prediction and its final
        command is repeated unchanged.
        """

        if previous is None:
            raise RuntimeError(
                f"{arm} DynaMAC has completed without a final action to hold"
            )
        diagnostics = deepcopy(previous.diagnostics)
        diagnostics["complete_hold"] = True
        return DynaMACAction(
            pose=previous.pose.copy(),
            covariance=previous.covariance.copy(),
            gripper=previous.gripper.copy(),
            diagnostics=diagnostics,
        )


# 清晰兼容名；不再把在线关系原型伪装成 DynaMAC。
DynaMACPolicy = DynaMAC


__all__ = [
    "BimanualDynaMAC",
    "BimanualDynaMACAction",
    "BimanualDynaMACGripperLookahead",
    "DynaMAC",
    "DynaMACAction",
    "DynaMACConfig",
    "DynaMACDemonstration",
    "DynaMACGripperLookahead",
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
    "pose_log_nearest",
    "product_of_experts",
    "relative_pose",
    "static_task_parameter_score_details",
    "static_task_parameter_scores",
    "synchronized_bimanual_demonstrations",
    "task_parameter_score_details",
    "task_parameter_scores",
    "transform_marginal",
]
