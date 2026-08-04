"""Audit immutable physical-handover JSON/NPZ evidence without launching simulation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASK_ID = "Essay2608-Bimanual-Physical-Handover-v0"
V2_SEEDS = tuple(range(7800, 7820))
V2_SOURCE_SHA256 = "e8622016ff485fced50d0f3b32ccefaad06f34dd7748d3a5e1cca950066b6801"
V3_SEEDS = tuple(range(8000, 8020))
V3_SOURCE_SHA256 = "2e52bf2a0c961e5c79a4ca4a709bcb6416c3cca3e8f6ccf17dcc11339889d31c"
EXPECTED_LIFECYCLE = ("none", "left_only", "both", "right_only", "none")

STEP_ALIGNED_KEYS = {
    "state",
    "left_ee_position",
    "left_ee_orientation",
    "right_ee_position",
    "right_ee_orientation",
    "object_position",
    "object_orientation",
    "target_position",
    "object_linear_velocity",
    "action",
    "left_finger_force",
    "right_finger_force",
    "left_finger_position",
    "right_finger_position",
    "left_connected",
    "right_connected",
    "left_confidence",
    "right_confidence",
    "relation_label",
}
REQUIRED_TRACE_KEYS = STEP_ALIGNED_KEYS | {
    "control_dt",
    "terminal_object_position",
    "terminal_target_position",
}


@dataclass(frozen=True)
class PhysicalHandoverAuditResult:
    """Identity and aggregate result returned after all hard checks pass."""

    trial_count: int
    success_count: int
    source_sha256: str
    summary_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_count": self.trial_count,
            "success_count": self.success_count,
            "source_sha256": self.source_sha256,
            "summary_sha256": self.summary_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compressed(values: Iterable[str]) -> tuple[str, ...]:
    sequence: list[str] = []
    for value in values:
        value = str(value)
        if not sequence or value != sequence[-1]:
            sequence.append(value)
    return tuple(sequence)


def _experiment_fingerprint(
    seed: int,
    source_sha256: str,
    *,
    max_steps: int,
    success_xy_threshold: float,
    minimum_both_duration_s: float,
) -> str:
    payload = {
        "task_id": TASK_ID,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "success_xy_threshold": float(success_xy_threshold),
        "minimum_both_duration_s": float(minimum_both_duration_s),
        "source_sha256": source_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_trace(
    path: Path,
    trial: dict[str, Any],
    *,
    minimum_both_duration_s: float,
) -> None:
    with np.load(path, allow_pickle=False) as trace:
        missing = REQUIRED_TRACE_KEYS.difference(trace.files)
        if missing:
            raise ValueError(f"{path.name} 缺少 trace 字段：{sorted(missing)!r}")

        lengths = {key: len(trace[key]) for key in STEP_ALIGNED_KEYS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path.name} 的逐 step 字段未对齐：{lengths!r}")
        steps = next(iter(lengths.values()))
        if steps <= 0 or steps != int(trial["steps"]):
            raise ValueError(f"{path.name} 的步数与 JSON 不一致")

        terminal_object = np.asarray(trace["terminal_object_position"], dtype=np.float64)
        terminal_target = np.asarray(trace["terminal_target_position"], dtype=np.float64)
        if terminal_object.shape != (3,) or terminal_target.shape != (3,):
            raise ValueError(f"{path.name} 的终端位置形状不是 (3,)")
        if not np.allclose(terminal_object, trial["final_object_position_m"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{path.name} 的终端物体位置与 JSON 不一致")
        if not np.allclose(terminal_target, trial["final_target_position_m"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{path.name} 的终端目标位置与 JSON 不一致")

        labels = tuple(str(value) for value in trace["relation_label"].tolist())
        if _compressed(labels) != tuple(trial["relation_sequence"]):
            raise ValueError(f"{path.name} 的关系生命周期与 JSON 不一致")
        left = np.asarray(trace["left_connected"], dtype=bool)
        right = np.asarray(trace["right_connected"], dtype=bool)
        derived_labels = np.where(
            left & right,
            "both",
            np.where(left, "left_only", np.where(right, "right_only", "none")),
        )
        if not np.array_equal(derived_labels, np.asarray(labels)):
            raise ValueError(f"{path.name} 的 relation_label 与双边连接状态不一致")

        control_dt = float(np.asarray(trace["control_dt"]).item())
        both_duration = labels.count("both") * control_dt
        if not np.isclose(both_duration, trial["both_duration_s"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{path.name} 的 both 持续时间与 JSON 不一致")
        maximum_height = float(np.max(trace["object_position"][:, 2]))
        if not np.isclose(maximum_height, trial["maximum_object_height_m"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{path.name} 的最大高度与 JSON 不一致")
        final_xy_error = float(np.linalg.norm(terminal_object[:2] - terminal_target[:2]))
        if not np.isclose(final_xy_error, trial["final_xy_error_m"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{path.name} 的最终 XY 误差与 JSON 不一致")

        positions = np.asarray(trace["object_position"], dtype=np.float64)
        settling = positions[-25:]
        displacement = float(np.max(np.linalg.norm(settling - settling[-1], axis=-1)))
        if not np.isclose(displacement, trial["settling_displacement_m"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{path.name} 的末段位移与 JSON 不一致")
        if bool(displacement <= 0.01) != bool(trial["stable"]):
            raise ValueError(f"{path.name} 的稳定性标志与 trace 不一致")

        if trial["success"]:
            if _compressed(labels) != EXPECTED_LIFECYCLE:
                raise ValueError(f"{path.name} 的成功 trial 缺少精确关系生命周期")
            if both_duration + 1e-6 < minimum_both_duration_s:
                raise ValueError(f"{path.name} 的成功 trial 共同持物时间不足")


def audit_physical_handover_run(
    summary_path: Path,
    *,
    expected_seeds: Iterable[int] = V2_SEEDS,
    expected_source_sha256: str = V2_SOURCE_SHA256,
    expected_successes: int | None = 18,
    max_steps: int = 1400,
    success_xy_threshold: float = 0.04,
    minimum_both_duration_s: float = 0.20,
) -> PhysicalHandoverAuditResult:
    """Validate v2 membership, fingerprints, aggregates, and every persisted trace."""

    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in expected_seeds)
    trials = summary.get("trials", [])

    if summary.get("task_id") != TASK_ID:
        raise ValueError("summary 的 task_id 不正确")
    if tuple(int(seed) for seed in summary.get("seeds", [])) != seeds:
        raise ValueError("summary 的 seed 顺序或成员与冻结协议不一致")
    if len(seeds) != len(set(seeds)) or len(trials) != len(seeds):
        raise ValueError("正式 seed 必须唯一且每个 seed 恰有一个 trial")
    by_seed = {int(trial["seed"]): trial for trial in trials}
    if len(by_seed) != len(trials) or set(by_seed) != set(seeds):
        raise ValueError("trial 的 seed 集合与冻结协议不一致")
    if summary.get("source_sha256") != expected_source_sha256:
        raise ValueError("summary 的源码指纹与冻结协议不一致")

    successes = sum(bool(trial.get("success")) for trial in trials)
    if int(summary.get("num_trials", -1)) != len(trials):
        raise ValueError("summary 的 num_trials 与 trial 数量不一致")
    if int(summary.get("num_successes", -1)) != successes:
        raise ValueError("summary 的 num_successes 与 trial 结果不一致")
    if not np.isclose(summary.get("success_rate", -1.0), successes / len(trials), atol=0.0, rtol=0.0):
        raise ValueError("summary 的 success_rate 与 trial 结果不一致")
    if expected_successes is not None and successes != expected_successes:
        raise ValueError(f"正式成功数被改写：预期={expected_successes}，实际={successes}")

    fingerprints: set[str] = set()
    trial_dir = summary_path.parent / "trials"
    expected_stems = {f"scripted_physical_handover__seed_{seed}" for seed in seeds}
    json_paths = sorted(trial_dir.glob("*.json"))
    npz_paths = sorted(trial_dir.glob("*.npz"))
    if {path.stem for path in json_paths} != expected_stems:
        raise ValueError("逐 trial JSON 文件集合不完整")
    if {path.stem for path in npz_paths} != expected_stems:
        raise ValueError("逐 trial NPZ 文件集合不完整")

    for seed in seeds:
        trial = by_seed[seed]
        if trial.get("task_id") != TASK_ID or trial.get("source_sha256") != expected_source_sha256:
            raise ValueError(f"seed {seed} 的任务或源码指纹不一致")
        expected_fingerprint = _experiment_fingerprint(
            seed,
            expected_source_sha256,
            max_steps=max_steps,
            success_xy_threshold=success_xy_threshold,
            minimum_both_duration_s=minimum_both_duration_s,
        )
        if trial.get("experiment_fingerprint") != expected_fingerprint:
            raise ValueError(f"seed {seed} 的实验指纹与冻结参数不一致")
        fingerprints.add(expected_fingerprint)
        if (trial.get("failure_reason") == "success") != bool(trial.get("success")):
            raise ValueError(f"seed {seed} 的 success 与 failure_reason 矛盾")
        if trial["success"]:
            if not trial.get("expert_complete") or trial.get("expert_failed"):
                raise ValueError(f"seed {seed} 成功但专家状态不一致")
            if float(trial["final_xy_error_m"]) >= success_xy_threshold:
                raise ValueError(f"seed {seed} 成功但 XY 误差越界")
            if not trial.get("object_on_support") or not trial.get("stable"):
                raise ValueError(f"seed {seed} 成功但支撑或稳定性不成立")

        stem = f"scripted_physical_handover__seed_{seed}"
        record = json.loads((trial_dir / f"{stem}.json").read_text(encoding="utf-8"))
        if record != trial:
            raise ValueError(f"seed {seed} 的逐 trial JSON 与 summary 不一致")
        _audit_trace(
            trial_dir / f"{stem}.npz",
            trial,
            minimum_both_duration_s=minimum_both_duration_s,
        )

    if len(fingerprints) != len(trials):
        raise ValueError("experiment_fingerprint 存在重复")
    return PhysicalHandoverAuditResult(
        trial_count=len(trials),
        success_count=successes,
        source_sha256=expected_source_sha256,
        summary_sha256=_sha256(summary_path),
    )
