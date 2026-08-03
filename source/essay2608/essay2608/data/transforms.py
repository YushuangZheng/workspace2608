"""NumPy SE(3) helpers using wxyz quaternions."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize one or more wxyz quaternions."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("Cannot normalize a zero quaternion.")
    return quaternion / norm


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """Return the conjugate of one or more wxyz quaternions."""

    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of broadcastable wxyz quaternions."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
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


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized wxyz quaternions to rotation matrices."""

    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by broadcastable wxyz quaternions."""

    rotation = quaternion_to_matrix(quaternion)
    return np.einsum("...ij,...j->...i", rotation, vector)


def pose_inverse(pose: np.ndarray) -> np.ndarray:
    """Invert one or more poses stored as xyz + wxyz."""

    pose = np.asarray(pose, dtype=np.float64)
    inverse_quaternion = quaternion_conjugate(normalize_quaternion(pose[..., 3:7]))
    inverse_position = -rotate_vector(inverse_quaternion, pose[..., :3])
    return np.concatenate((inverse_position, inverse_quaternion), axis=-1)


def pose_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose broadcastable poses stored as xyz + wxyz."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    position = left[..., :3] + rotate_vector(left[..., 3:7], right[..., :3])
    quaternion = normalize_quaternion(quaternion_multiply(left[..., 3:7], right[..., 3:7]))
    return np.concatenate((position, quaternion), axis=-1)


def relative_pose(frame_pose: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Express ``pose`` in ``frame_pose`` coordinates."""

    return pose_multiply(pose_inverse(frame_pose), pose)


def quaternion_mean(quaternions: np.ndarray) -> np.ndarray:
    """Return a sign-invariant Markley mean for wxyz quaternions."""

    quaternions = normalize_quaternion(quaternions)
    accumulator = np.einsum("ni,nj->ij", quaternions, quaternions)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    result = eigenvectors[:, np.argmax(eigenvalues)]
    if result[0] < 0.0:
        result *= -1.0
    return normalize_quaternion(result)


def quaternion_distance_radians(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Shortest geodesic angular distance between wxyz quaternions."""

    left = normalize_quaternion(left)
    right = normalize_quaternion(right)
    dot = np.abs(np.sum(left * right, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def interpolate_rows(values: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample an array along its first axis."""

    values = np.asarray(values, dtype=np.float64)
    if length <= 0:
        raise ValueError("Resample length must be positive.")
    if len(values) == 1:
        return np.repeat(values, length, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    flattened = values.reshape(len(values), -1)
    result = np.stack([np.interp(target, source, flattened[:, index]) for index in range(flattened.shape[1])], axis=1)
    return result.reshape((length,) + values.shape[1:])


def interpolate_poses(poses: np.ndarray, length: int) -> np.ndarray:
    """Resample poses with linear position and normalized quaternion interpolation."""

    poses = np.asarray(poses, dtype=np.float64)
    position = interpolate_rows(poses[:, :3], length)
    quaternion = interpolate_rows(poses[:, 3:7], length)
    reference = quaternion[0]
    quaternion[np.sum(quaternion * reference, axis=-1) < 0.0] *= -1.0
    quaternion = normalize_quaternion(quaternion)
    return np.concatenate((position, quaternion), axis=-1)
