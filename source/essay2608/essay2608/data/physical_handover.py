"""Schema and immutable audits for contact-rich physical handover demonstrations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from essay2608.eval.physical_handover_audit import (
    EXPECTED_LIFECYCLE,
    STEP_ALIGNED_KEYS,
    TASK_ID,
    V3_SOURCE_SHA256,
    _audit_trace,
)


PHYSICAL_DATASET_SCHEMA_VERSION = 1
EXPECTED_STATE_SEQUENCE = tuple(range(12))
DATASET_REQUIRED_KEYS = STEP_ALIGNED_KEYS | {
    "time",
    "left_ee_pose",
    "right_ee_pose",
    "object_pose",
    "target_pose",
    "control_dt",
    "terminal_object_position",
    "terminal_target_position",
    "seed",
    "source_sha256",
    "experiment_fingerprint",
    "both_duration_s",
    "maximum_object_height_m",
    "final_xy_error_m",
    "object_on_support",
    "stable",
    "settling_displacement_m",
    "quaternion_order",
    "coordinate_frame",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(archive: Any, key: str) -> Any:
    return np.asarray(archive[key]).item()


def _compressed(values: Iterable[Any]) -> tuple[Any, ...]:
    sequence: list[Any] = []
    for value in values:
        value = value.item() if isinstance(value, np.generic) else value
        if not sequence or value != sequence[-1]:
            sequence.append(value)
    return tuple(sequence)


def physical_dataset_digest(entries: list[dict[str, Any]]) -> str:
    """Hash the ordered physical demonstration file identities."""

    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['file']}:{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def audit_physical_handover_demonstration(
    path: Path,
    *,
    expected_source_sha256: str = V3_SOURCE_SHA256,
    success_xy_threshold: float = 0.04,
    minimum_both_duration_s: float = 0.20,
) -> dict[str, Any]:
    """Validate one successful phase-independent physical handover trace."""

    path = path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        missing = DATASET_REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path.name} 缺少物理数据字段：{sorted(missing)!r}")
        steps = len(archive["state"])
        aligned_keys = STEP_ALIGNED_KEYS | {
            "time",
            "left_ee_pose",
            "right_ee_pose",
            "object_pose",
            "target_pose",
        }
        lengths = {key: len(archive[key]) for key in aligned_keys}
        if set(lengths.values()) != {steps} or steps <= 0:
            raise ValueError(f"{path.name} 的逐 step 字段未对齐：{lengths!r}")
        if archive["left_ee_pose"].shape != (steps, 7):
            raise ValueError(f"{path.name} 的 left_ee_pose 形状错误")
        if archive["right_ee_pose"].shape != (steps, 7):
            raise ValueError(f"{path.name} 的 right_ee_pose 形状错误")
        if archive["object_pose"].shape != (steps, 7):
            raise ValueError(f"{path.name} 的 object_pose 形状错误")
        if archive["target_pose"].shape != (steps, 7):
            raise ValueError(f"{path.name} 的 target_pose 形状错误")
        if archive["action"].shape != (steps, 16):
            raise ValueError(f"{path.name} 的动作不是 16 维")
        if archive["left_finger_force"].shape != (steps, 2, 3):
            raise ValueError(f"{path.name} 的左侧双指接触力形状错误")
        if archive["right_finger_force"].shape != (steps, 2, 3):
            raise ValueError(f"{path.name} 的右侧双指接触力形状错误")

        numeric_keys = aligned_keys.difference({"relation_label"}) | {
            "left_confidence",
            "right_confidence",
        }
        for key in numeric_keys:
            if not np.all(np.isfinite(archive[key])):
                raise ValueError(f"{path.name} 的 {key} 含非有限值")

        if str(_scalar(archive, "quaternion_order")) != "wxyz":
            raise ValueError(f"{path.name} 不是 wxyz 四元数顺序")
        if str(_scalar(archive, "coordinate_frame")) != "local_environment":
            raise ValueError(f"{path.name} 不是局部环境坐标")
        source_sha = str(_scalar(archive, "source_sha256"))
        if source_sha != expected_source_sha256:
            raise ValueError(f"{path.name} 的源码指纹与冻结专家不一致")

        control_dt = float(_scalar(archive, "control_dt"))
        expected_time = np.arange(steps, dtype=np.float64) * control_dt
        if not np.allclose(archive["time"], expected_time, atol=1e-5, rtol=0.0):
            raise ValueError(f"{path.name} 的时间轴不连续")
        state_sequence = _compressed(archive["state"])
        if state_sequence != EXPECTED_STATE_SEQUENCE:
            raise ValueError(f"{path.name} 的专家状态不完整：{state_sequence!r}")

        labels = np.asarray(archive["relation_label"]).astype("U16")
        relation_sequence = _compressed(labels)
        if relation_sequence != EXPECTED_LIFECYCLE:
            raise ValueError(f"{path.name} 的物理关系生命周期不完整")
        left_connected = np.asarray(archive["left_connected"], dtype=bool)
        right_connected = np.asarray(archive["right_connected"], dtype=bool)
        derived = np.where(
            left_connected & right_connected,
            "both",
            np.where(
                left_connected,
                "left_only",
                np.where(right_connected, "right_only", "none"),
            ),
        )
        if not np.array_equal(labels, derived):
            raise ValueError(f"{path.name} 的关系标签不是由两条独立物理边组成")

        trial = {
            "steps": steps,
            "relation_sequence": list(relation_sequence),
            "both_duration_s": float(_scalar(archive, "both_duration_s")),
            "maximum_object_height_m": float(
                _scalar(archive, "maximum_object_height_m")
            ),
            "final_object_position_m": np.asarray(
                archive["terminal_object_position"], dtype=float
            ).tolist(),
            "final_target_position_m": np.asarray(
                archive["terminal_target_position"], dtype=float
            ).tolist(),
            "final_xy_error_m": float(_scalar(archive, "final_xy_error_m")),
            "stable": bool(_scalar(archive, "stable")),
            "settling_displacement_m": float(
                _scalar(archive, "settling_displacement_m")
            ),
            "success": True,
        }
    _audit_trace(
        path,
        trial,
        minimum_both_duration_s=minimum_both_duration_s,
    )

    with np.load(path, allow_pickle=False) as archive:
        if float(_scalar(archive, "final_xy_error_m")) >= success_xy_threshold:
            raise ValueError(f"{path.name} 的最终 XY 误差超过数据门槛")
        if not bool(_scalar(archive, "object_on_support")):
            raise ValueError(f"{path.name} 的物体最终不在支撑面")
        if not bool(_scalar(archive, "stable")):
            raise ValueError(f"{path.name} 的物体末段不稳定")

        left_forces = np.linalg.norm(archive["left_finger_force"], axis=-1)
        right_forces = np.linalg.norm(archive["right_finger_force"], axis=-1)
        physical_contact = {
            "left": np.all(left_forces >= 0.15, axis=-1),
            "right": np.all(right_forces >= 0.15, axis=-1),
        }
        contact_agreement: dict[str, float] = {}
        for side, connected in (
            ("left", np.asarray(archive["left_connected"], dtype=bool)),
            ("right", np.asarray(archive["right_connected"], dtype=bool)),
        ):
            agreement = float(np.mean(physical_contact[side][connected]))
            if not np.isfinite(agreement) or agreement < 0.90:
                raise ValueError(f"{path.name} 的 {side} 关系与双指接触一致率过低")
            contact_agreement[side] = agreement

        initial_object_pose = archive["object_pose"][0].astype(float).tolist()
        max_jumps = {
            key: float(np.max(np.linalg.norm(np.diff(archive[key][:, :3], axis=0), axis=-1)))
            for key in ("left_ee_pose", "right_ee_pose", "object_pose")
        }
        if max(max_jumps.values()) >= 0.15:
            raise ValueError(f"{path.name} 存在复位式位姿跳变：{max_jumps!r}")
        phase_labels = np.asarray(
            [
                "left_only" if 2 <= int(state) <= 6 else
                "both" if int(state) == 7 else
                "right_only" if 8 <= int(state) <= 9 else "none"
                for state in archive["state"]
            ],
            dtype="U16",
        )
        phase_disagreement_steps = int(np.count_nonzero(labels != phase_labels))
        return {
            "file": path.name,
            "sha256": _sha256(path),
            "seed": int(_scalar(archive, "seed")),
            "steps": steps,
            "control_dt": control_dt,
            "source_sha256": source_sha,
            "experiment_fingerprint": str(_scalar(archive, "experiment_fingerprint")),
            "state_sequence": list(state_sequence),
            "relation_sequence": list(relation_sequence),
            "relation_source": "physical_contact_proximity_relative_motion",
            "phase_disagreement_steps": phase_disagreement_steps,
            "both_duration_s": float(_scalar(archive, "both_duration_s")),
            "final_xy_error_m": float(_scalar(archive, "final_xy_error_m")),
            "settling_displacement_m": float(
                _scalar(archive, "settling_displacement_m")
            ),
            "initial_object_pose": initial_object_pose,
            "contact_agreement_when_connected": contact_agreement,
            "max_step_position_jump_m": max_jumps,
        }


def audit_physical_handover_dataset(
    dataset_dir: str | Path,
    *,
    expected_seeds: Iterable[int] | None = None,
    expected_source_sha256: str = V3_SOURCE_SHA256,
) -> dict[str, Any]:
    """Audit membership, provenance, and every physical demonstration."""

    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task_id") != TASK_ID:
        raise ValueError("物理数据集 task_id 不正确")
    if int(manifest.get("dataset_schema_version", -1)) != PHYSICAL_DATASET_SCHEMA_VERSION:
        raise ValueError("物理数据集 schema 版本不正确")

    paths = sorted(dataset_dir.glob("demo_*.npz"))
    manifest_entries = manifest.get("demos", [])
    manifest_files = [entry["file"] for entry in manifest_entries]
    if [path.name for path in paths] != manifest_files:
        raise ValueError("manifest 与物理演示文件集合或顺序不一致")
    if int(manifest.get("num_demos", -1)) != len(paths) or not paths:
        raise ValueError("manifest 的物理演示数量不正确")

    entries = [
        audit_physical_handover_demonstration(
            path,
            expected_source_sha256=expected_source_sha256,
        )
        for path in paths
    ]
    seeds = tuple(int(entry["seed"]) for entry in entries)
    if len(seeds) != len(set(seeds)):
        raise ValueError("物理数据集 seed 存在重复")
    requested = tuple(int(seed) for seed in manifest.get("requested_seeds", []))
    if seeds != requested:
        raise ValueError("演示 seed 与预注册 requested_seeds 不一致")
    if expected_seeds is not None and seeds != tuple(int(seed) for seed in expected_seeds):
        raise ValueError("物理数据集 seed 与外部冻结协议不一致")
    if any(entry["source_sha256"] != expected_source_sha256 for entry in entries):
        raise ValueError("物理数据集混入了其他源码指纹")

    initial = np.asarray([entry["initial_object_pose"][:2] for entry in entries])
    ranges = np.ptp(initial, axis=0)
    if len(entries) > 1 and np.any(ranges < 0.015):
        raise ValueError("物理数据集没有覆盖足够的初始 XY 扰动范围")
    digest = physical_dataset_digest(entries)
    if manifest.get("frozen"):
        if manifest.get("dataset_sha256") != digest:
            raise ValueError("冻结 manifest 的数据集哈希不一致")
        if not (dataset_dir / "FROZEN").is_file():
            raise ValueError("manifest 声称冻结但缺少 FROZEN 标记")
        by_file = {entry["file"]: entry for entry in manifest_entries}
        for entry in entries:
            if by_file[entry["file"]].get("sha256") != entry["sha256"]:
                raise ValueError(f"{entry['file']} 的冻结文件哈希不一致")

    return {
        "entries": entries,
        "num_demos": len(entries),
        "dataset_sha256": digest,
        "source_sha256": expected_source_sha256,
        "seeds": list(seeds),
        "initial_xy_range_m": ranges.astype(float).tolist(),
        "min_both_duration_s": min(entry["both_duration_s"] for entry in entries),
        "max_final_xy_error_m": max(entry["final_xy_error_m"] for entry in entries),
        "max_settling_displacement_m": max(
            entry["settling_displacement_m"] for entry in entries
        ),
        "min_contact_agreement_when_connected": {
            side: min(
                entry["contact_agreement_when_connected"][side] for entry in entries
            )
            for side in ("left", "right")
        },
        "total_phase_disagreement_steps": sum(
            entry["phase_disagreement_steps"] for entry in entries
        ),
    }
