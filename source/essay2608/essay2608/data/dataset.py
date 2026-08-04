"""Versioned demonstration loading and dataset acceptance checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import quaternion_distance_radians, quaternion_mean, relative_pose


EXPECTED_KEYS = {
    "time",
    "state",
    "ee_pose",
    "object_pose",
    "target_pose",
    "action",
    "joint_pos",
    "joint_vel",
    "control_dt",
    "final_error",
    "quaternion_order",
    "coordinate_frame",
}
EXPECTED_STATE_SEQUENCE = list(range(10))
BIMANUAL_EXPECTED_KEYS = {
    "time",
    "state",
    "left_ee_pose",
    "right_ee_pose",
    "object_pose",
    "target_pose",
    "action",
    "joint_pos",
    "joint_vel",
    "carrier",
    "control_dt",
    "final_error",
    "quaternion_order",
    "coordinate_frame",
}
BIMANUAL_EXPECTED_STATE_SEQUENCE = list(range(13))
TRAY_EXPECTED_STATE_SEQUENCE = list(range(9))


@dataclass(frozen=True)
class Demonstration:
    """One immutable in-memory pick-and-place demonstration."""

    path: Path
    time: np.ndarray
    state: np.ndarray
    ee_pose: np.ndarray
    object_pose: np.ndarray
    target_pose: np.ndarray
    action: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    control_dt: float
    final_error: float

    @property
    def steps(self) -> int:
        return len(self.state)

    def phase_indices(self, phase: int) -> np.ndarray:
        """Return indices belonging to one scripted phase."""

        return np.flatnonzero(self.state == phase)


@dataclass(frozen=True)
class BimanualDemonstration:
    """One immutable bimanual handover demonstration."""

    path: Path
    time: np.ndarray
    state: np.ndarray
    left_ee_pose: np.ndarray
    right_ee_pose: np.ndarray
    object_pose: np.ndarray
    target_pose: np.ndarray
    action: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    carrier: np.ndarray
    control_dt: float
    final_error: float

    @property
    def steps(self) -> int:
        return len(self.state)

    def phase_indices(self, phase: int) -> np.ndarray:
        return np.flatnonzero(self.state == phase)


def sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_demo(path: Path) -> Demonstration:
    """Load and copy one NPZ demonstration."""

    with np.load(path, allow_pickle=False) as archive:
        missing = EXPECTED_KEYS - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        if _text_scalar(archive["quaternion_order"]) != "wxyz":
            raise ValueError(f"{path} does not use wxyz quaternions.")
        if _text_scalar(archive["coordinate_frame"]) != "local_environment":
            raise ValueError(f"{path} is not in the local environment frame.")
        arrays = {
            name: archive[name].copy()
            for name in EXPECTED_KEYS
            if name not in {"quaternion_order", "coordinate_frame"}
        }

    return Demonstration(
        path=path,
        time=arrays["time"],
        state=arrays["state"],
        ee_pose=arrays["ee_pose"],
        object_pose=arrays["object_pose"],
        target_pose=arrays["target_pose"],
        action=arrays["action"],
        joint_pos=arrays["joint_pos"],
        joint_vel=arrays["joint_vel"],
        control_dt=float(arrays["control_dt"]),
        final_error=float(arrays["final_error"]),
    )


def load_bimanual_demo(path: Path) -> BimanualDemonstration:
    """Load and copy one bimanual NPZ demonstration."""

    with np.load(path, allow_pickle=False) as archive:
        missing = BIMANUAL_EXPECTED_KEYS - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        if _text_scalar(archive["quaternion_order"]) != "wxyz":
            raise ValueError(f"{path} does not use wxyz quaternions.")
        if _text_scalar(archive["coordinate_frame"]) != "local_environment":
            raise ValueError(f"{path} is not in the local environment frame.")
        arrays = {
            name: archive[name].copy()
            for name in BIMANUAL_EXPECTED_KEYS
            if name not in {"quaternion_order", "coordinate_frame"}
        }
    return BimanualDemonstration(
        path=path,
        time=arrays["time"],
        state=arrays["state"],
        left_ee_pose=arrays["left_ee_pose"],
        right_ee_pose=arrays["right_ee_pose"],
        object_pose=arrays["object_pose"],
        target_pose=arrays["target_pose"],
        action=arrays["action"],
        joint_pos=arrays["joint_pos"],
        joint_vel=arrays["joint_vel"],
        carrier=arrays["carrier"],
        control_dt=float(arrays["control_dt"]),
        final_error=float(arrays["final_error"]),
    )


def load_dataset(dataset_dir: str | Path, verify_hashes: bool = True) -> tuple[list[Demonstration], dict[str, Any]]:
    """Load a dataset directory and optionally verify its frozen hashes."""

    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("demos", [])
    if len(entries) != manifest.get("num_demos"):
        raise ValueError("Manifest demo count is inconsistent.")

    demonstrations: list[Demonstration] = []
    for entry in entries:
        path = dataset_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_hashes and "sha256" in entry:
            actual_hash = sha256_file(path)
            if actual_hash != entry["sha256"]:
                raise ValueError(f"Frozen hash mismatch for {path.name}.")
        demonstrations.append(load_demo(path))
    return demonstrations, manifest


def load_bimanual_dataset(
    dataset_dir: str | Path,
    verify_hashes: bool = True,
) -> tuple[list[BimanualDemonstration], dict[str, Any]]:
    """Load a frozen bimanual dataset and verify its file identities."""

    dataset_dir = Path(dataset_dir).resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("demos", [])
    if len(entries) != manifest.get("num_demos"):
        raise ValueError("Manifest demo count is inconsistent.")
    demonstrations = []
    for entry in entries:
        path = dataset_dir / entry["file"]
        if verify_hashes and entry.get("sha256") != sha256_file(path):
            raise ValueError(f"Frozen hash mismatch for {path.name}.")
        demonstrations.append(load_bimanual_demo(path))
    return demonstrations, manifest


def _state_sequence(states: np.ndarray) -> list[int]:
    sequence = [int(states[0])]
    for value in states[1:]:
        if int(value) != sequence[-1]:
            sequence.append(int(value))
    return sequence


def _max_position_jump(poses: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=-1)))


def _pose_stability(poses: np.ndarray) -> dict[str, float]:
    position_std = np.std(poses[:, :3], axis=0)
    mean_quaternion = quaternion_mean(poses[:, 3:7])
    angular_error = quaternion_distance_radians(poses[:, 3:7], mean_quaternion)
    return {
        "position_rms_std_m": float(np.sqrt(np.mean(np.square(position_std)))),
        "rotation_rms_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(angular_error))))),
    }


def audit_demonstration(
    demonstration: Demonstration,
    success_threshold: float = 0.06,
    reset_jump_threshold: float = 0.05,
    connection_position_threshold: float = 0.002,
) -> dict[str, Any]:
    """Validate one trajectory and return manifest-ready diagnostics."""

    lengths = {
        len(demonstration.time),
        len(demonstration.state),
        len(demonstration.ee_pose),
        len(demonstration.object_pose),
        len(demonstration.target_pose),
        len(demonstration.action),
        len(demonstration.joint_pos),
        len(demonstration.joint_vel),
    }
    if lengths != {demonstration.steps}:
        raise ValueError(f"{demonstration.path.name} has inconsistent array lengths: {lengths}")
    if demonstration.ee_pose.shape[1] != 7 or demonstration.action.shape[1] != 8:
        raise ValueError(f"{demonstration.path.name} has an invalid pose or action shape.")
    for name in ("time", "ee_pose", "object_pose", "target_pose", "action", "joint_pos", "joint_vel"):
        if not np.all(np.isfinite(getattr(demonstration, name))):
            raise ValueError(f"{demonstration.path.name} contains non-finite {name} values.")

    sequence = _state_sequence(demonstration.state)
    if sequence != EXPECTED_STATE_SEQUENCE:
        raise ValueError(f"{demonstration.path.name} has incomplete states: {sequence}")
    expected_time = np.arange(demonstration.steps) * demonstration.control_dt
    if not np.allclose(demonstration.time, expected_time, atol=1.0e-5):
        raise ValueError(f"{demonstration.path.name} has a discontinuous time base.")
    if demonstration.final_error >= success_threshold:
        raise ValueError(f"{demonstration.path.name} exceeds the success threshold.")

    jumps = {
        "ee_m": _max_position_jump(demonstration.ee_pose),
        "object_m": _max_position_jump(demonstration.object_pose),
        "target_m": _max_position_jump(demonstration.target_pose),
    }
    if max(jumps.values()) >= reset_jump_threshold:
        raise ValueError(f"{demonstration.path.name} contains a reset-like pose jump: {jumps}")

    connected = np.isin(demonstration.state, [4, 5, 6])
    object_to_ee = relative_pose(demonstration.object_pose[connected], demonstration.ee_pose[connected])
    connection_stability = _pose_stability(object_to_ee)
    if connection_stability["position_rms_std_m"] >= connection_position_threshold:
        raise ValueError(f"{demonstration.path.name} does not exhibit a stable grasp connection.")

    return {
        "file": demonstration.path.name,
        "sha256": sha256_file(demonstration.path),
        "steps": demonstration.steps,
        "control_dt": demonstration.control_dt,
        "state_sequence": sequence,
        "initial_ee_pose": demonstration.ee_pose[0].astype(float).tolist(),
        "initial_object_pose": demonstration.object_pose[0].astype(float).tolist(),
        "target_pose": demonstration.target_pose[0].astype(float).tolist(),
        "final_error": demonstration.final_error,
        "max_step_position_jump": jumps,
        "postgrasp_object_to_ee_stability": connection_stability,
    }


def dataset_digest(entries: list[dict[str, Any]]) -> str:
    """Hash the ordered immutable demo file identities."""

    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['file']}:{entry['sha256']}\n".encode())
    return digest.hexdigest()


def audit_dataset(
    dataset_dir: str | Path,
    seeds: list[int] | None = None,
    success_threshold: float = 0.06,
) -> dict[str, Any]:
    """Audit every NPZ file and return aggregate acceptance metrics."""

    dataset_dir = Path(dataset_dir).resolve()
    paths = sorted(dataset_dir.glob("demo_*.npz"))
    if not paths:
        raise ValueError(f"No demonstrations found in {dataset_dir}.")
    if seeds is not None and len(seeds) != len(paths):
        raise ValueError("The number of seeds must match the number of demonstrations.")

    entries = [audit_demonstration(load_demo(path), success_threshold=success_threshold) for path in paths]
    initial_positions = np.asarray([entry["initial_object_pose"][:3] for entry in entries])
    pairwise = np.linalg.norm(initial_positions[:, None] - initial_positions[None, :], axis=-1)
    positive_distances = pairwise[np.triu_indices(len(entries), k=1)]
    minimum_distance = float(np.min(positive_distances)) if len(positive_distances) else float("inf")
    if minimum_distance <= 0.01:
        raise ValueError("Initial object positions are duplicate or insufficiently distinct.")

    if seeds is not None:
        for entry, seed in zip(entries, seeds, strict=True):
            entry["seed"] = seed
            entry["seed_provenance"] = "reconstructed from the original base_seed + demo_index launcher"

    return {
        "entries": entries,
        "dataset_sha256": dataset_digest(entries),
        "minimum_initial_object_distance_m": minimum_distance,
        "max_final_error_m": max(entry["final_error"] for entry in entries),
        "max_step_position_jump_m": max(
            max(entry["max_step_position_jump"].values()) for entry in entries
        ),
        "max_postgrasp_position_rms_std_m": max(
            entry["postgrasp_object_to_ee_stability"]["position_rms_std_m"] for entry in entries
        ),
    }


def audit_bimanual_demonstration(
    demonstration: BimanualDemonstration,
    success_threshold: float = 0.06,
    reset_jump_threshold: float = 0.15,
) -> dict[str, Any]:
    """Validate one complete handover and return manifest diagnostics."""

    arrays = (
        demonstration.time,
        demonstration.state,
        demonstration.left_ee_pose,
        demonstration.right_ee_pose,
        demonstration.object_pose,
        demonstration.target_pose,
        demonstration.action,
        demonstration.joint_pos,
        demonstration.joint_vel,
        demonstration.carrier,
    )
    if {len(array) for array in arrays} != {demonstration.steps}:
        raise ValueError(f"{demonstration.path.name} has inconsistent array lengths.")
    if demonstration.action.shape[1] != 16:
        raise ValueError(f"{demonstration.path.name} does not have a 16-D bimanual action.")
    if any(pose.shape[1] != 7 for pose in (demonstration.left_ee_pose, demonstration.right_ee_pose)):
        raise ValueError(f"{demonstration.path.name} has an invalid EE pose shape.")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{demonstration.path.name} contains non-finite values.")
    sequence = _state_sequence(demonstration.state)
    if sequence != BIMANUAL_EXPECTED_STATE_SEQUENCE:
        raise ValueError(f"{demonstration.path.name} has incomplete states: {sequence}")
    if demonstration.final_error >= success_threshold:
        raise ValueError(f"{demonstration.path.name} exceeds the success threshold.")
    expected_time = np.arange(demonstration.steps) * demonstration.control_dt
    if not np.allclose(demonstration.time, expected_time, atol=1.0e-5):
        raise ValueError(f"{demonstration.path.name} has a discontinuous time base.")

    jumps = {
        "left_ee_m": _max_position_jump(demonstration.left_ee_pose),
        "right_ee_m": _max_position_jump(demonstration.right_ee_pose),
        "object_m": _max_position_jump(demonstration.object_pose),
    }
    if max(jumps.values()) >= reset_jump_threshold:
        raise ValueError(f"{demonstration.path.name} contains a reset-like jump: {jumps}")

    left_mask = demonstration.carrier == 1
    right_mask = demonstration.carrier == 2
    if not left_mask.any() or not right_mask.any():
        raise ValueError(f"{demonstration.path.name} does not contain both carrier assignments.")
    left_connection = _pose_stability(
        relative_pose(demonstration.left_ee_pose[left_mask], demonstration.object_pose[left_mask])
    )
    right_connection = _pose_stability(
        relative_pose(demonstration.right_ee_pose[right_mask], demonstration.object_pose[right_mask])
    )
    cross_mask = np.isin(demonstration.state, [5, 6, 7])
    cross_arm = _pose_stability(
        relative_pose(demonstration.left_ee_pose[cross_mask], demonstration.right_ee_pose[cross_mask])
    )
    return {
        "file": demonstration.path.name,
        "sha256": sha256_file(demonstration.path),
        "steps": demonstration.steps,
        "control_dt": demonstration.control_dt,
        "state_sequence": sequence,
        "initial_left_ee_pose": demonstration.left_ee_pose[0].astype(float).tolist(),
        "initial_right_ee_pose": demonstration.right_ee_pose[0].astype(float).tolist(),
        "initial_object_pose": demonstration.object_pose[0].astype(float).tolist(),
        "target_pose": demonstration.target_pose[0].astype(float).tolist(),
        "final_error": demonstration.final_error,
        "max_step_position_jump": jumps,
        "left_object_connection_stability": left_connection,
        "right_object_connection_stability": right_connection,
        "cross_arm_handover_stability": cross_arm,
    }


def audit_bimanual_dataset(
    dataset_dir: str | Path,
    success_threshold: float = 0.06,
) -> dict[str, Any]:
    """Audit and fingerprint every bimanual demonstration in a directory."""

    dataset_dir = Path(dataset_dir).resolve()
    original_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    original_by_file = {entry["file"]: entry for entry in original_manifest.get("demos", [])}
    entries = []
    for path in sorted(dataset_dir.glob("demo_*.npz")):
        entry = audit_bimanual_demonstration(load_bimanual_demo(path), success_threshold)
        if path.name in original_by_file:
            for key in ("attempt", "seed"):
                if key in original_by_file[path.name]:
                    entry[key] = original_by_file[path.name][key]
        entries.append(entry)
    if not entries:
        raise ValueError(f"No handover demonstrations found in {dataset_dir}.")
    initial = np.asarray([entry["initial_object_pose"][:3] for entry in entries])
    pairwise = np.linalg.norm(initial[:, None] - initial[None, :], axis=-1)
    distances = pairwise[np.triu_indices(len(entries), k=1)]
    minimum_distance = float(np.min(distances)) if len(distances) else float("inf")
    if minimum_distance <= 0.003:
        raise ValueError("Bimanual initial object poses are duplicate or insufficiently distinct.")
    return {
        "entries": entries,
        "dataset_sha256": dataset_digest(entries),
        "minimum_initial_object_distance_m": minimum_distance,
        "max_final_error_m": max(entry["final_error"] for entry in entries),
        "max_step_position_jump_m": max(max(entry["max_step_position_jump"].values()) for entry in entries),
        "max_left_connection_std_m": max(
            entry["left_object_connection_stability"]["position_rms_std_m"] for entry in entries
        ),
        "max_right_connection_std_m": max(
            entry["right_object_connection_stability"]["position_rms_std_m"] for entry in entries
        ),
    }


def audit_tray_demonstration(
    demonstration: BimanualDemonstration,
    success_threshold: float = 0.06,
    reset_jump_threshold: float = 0.15,
) -> dict[str, Any]:
    """Validate one simultaneous bilateral tray trajectory."""

    arrays = (
        demonstration.time,
        demonstration.state,
        demonstration.left_ee_pose,
        demonstration.right_ee_pose,
        demonstration.object_pose,
        demonstration.target_pose,
        demonstration.action,
        demonstration.joint_pos,
        demonstration.joint_vel,
        demonstration.carrier,
    )
    if {len(array) for array in arrays} != {demonstration.steps}:
        raise ValueError(f"{demonstration.path.name} has inconsistent array lengths.")
    if demonstration.action.shape[1] != 16 or not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError(f"{demonstration.path.name} has invalid or non-finite arrays.")
    sequence = _state_sequence(demonstration.state)
    if sequence != TRAY_EXPECTED_STATE_SEQUENCE:
        raise ValueError(f"{demonstration.path.name} has incomplete tray states: {sequence}")
    if demonstration.final_error >= success_threshold:
        raise ValueError(f"{demonstration.path.name} exceeds the tray success threshold.")
    expected_time = np.arange(demonstration.steps) * demonstration.control_dt
    if not np.allclose(demonstration.time, expected_time, atol=1.0e-5):
        raise ValueError(f"{demonstration.path.name} has a discontinuous time base.")
    jumps = {
        "left_ee_m": _max_position_jump(demonstration.left_ee_pose),
        "right_ee_m": _max_position_jump(demonstration.right_ee_pose),
        "tray_m": _max_position_jump(demonstration.object_pose),
    }
    if max(jumps.values()) >= reset_jump_threshold:
        raise ValueError(f"{demonstration.path.name} contains a reset-like jump: {jumps}")
    connected = demonstration.carrier == 3
    if not connected.any():
        raise ValueError(f"{demonstration.path.name} does not contain bilateral attachment.")
    left_connection = _pose_stability(
        relative_pose(demonstration.left_ee_pose[connected], demonstration.object_pose[connected])
    )
    right_connection = _pose_stability(
        relative_pose(demonstration.right_ee_pose[connected], demonstration.object_pose[connected])
    )
    cross_arm = _pose_stability(
        relative_pose(demonstration.left_ee_pose[connected], demonstration.right_ee_pose[connected])
    )
    return {
        "file": demonstration.path.name,
        "sha256": sha256_file(demonstration.path),
        "steps": demonstration.steps,
        "control_dt": demonstration.control_dt,
        "state_sequence": sequence,
        "initial_object_pose": demonstration.object_pose[0].astype(float).tolist(),
        "target_pose": demonstration.target_pose[0].astype(float).tolist(),
        "final_error": demonstration.final_error,
        "max_step_position_jump": jumps,
        "left_tray_connection_stability": left_connection,
        "right_tray_connection_stability": right_connection,
        "cross_arm_stability": cross_arm,
    }


def audit_tray_dataset(dataset_dir: str | Path, success_threshold: float = 0.06) -> dict[str, Any]:
    """Audit and fingerprint a versioned lift-tray dataset."""

    dataset_dir = Path(dataset_dir).resolve()
    original = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    original_by_file = {entry["file"]: entry for entry in original.get("demos", [])}
    entries = []
    for path in sorted(dataset_dir.glob("demo_*.npz")):
        entry = audit_tray_demonstration(load_bimanual_demo(path), success_threshold)
        for key in ("attempt", "seed"):
            if key in original_by_file.get(path.name, {}):
                entry[key] = original_by_file[path.name][key]
        entries.append(entry)
    if not entries:
        raise ValueError(f"No tray demonstrations found in {dataset_dir}.")
    initial = np.asarray([entry["initial_object_pose"][:3] for entry in entries])
    pairwise = np.linalg.norm(initial[:, None] - initial[None, :], axis=-1)
    distances = pairwise[np.triu_indices(len(entries), k=1)]
    minimum_distance = float(np.min(distances)) if len(distances) else float("inf")
    if minimum_distance <= 0.003:
        raise ValueError("Tray initial poses are duplicate or insufficiently distinct.")
    return {
        "entries": entries,
        "dataset_sha256": dataset_digest(entries),
        "minimum_initial_object_distance_m": minimum_distance,
        "max_final_error_m": max(entry["final_error"] for entry in entries),
        "max_step_position_jump_m": max(
            max(entry["max_step_position_jump"].values()) for entry in entries
        ),
        "max_cross_arm_position_std_m": max(
            entry["cross_arm_stability"]["position_rms_std_m"] for entry in entries
        ),
    }
