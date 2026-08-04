"""Audit the immutable evidence produced by the recovery protocol."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_TRACE_KEYS = {
    "ee_position",
    "object_position",
    "target_position",
    "action",
    "phase",
    "connected",
    "perturbation_event",
    "raw_action_position",
    "policy_action_position",
    "relation_state",
    "relation_confidence",
    "active_frames",
    "recovery_state",
    "recovery_trigger",
    "regrasp_attempts",
    "terminal_object_position",
    "terminal_target_position",
    "terminal_ee_position",
}

STEP_ALIGNED_TRACE_KEYS = {
    "ee_position",
    "object_position",
    "target_position",
    "action",
    "phase",
    "connected",
    "perturbation_event",
    "raw_action_position",
    "policy_action_position",
    "relation_state",
    "relation_confidence",
    "active_frames",
    "recovery_state",
    "recovery_trigger",
    "regrasp_attempts",
}


@dataclass(frozen=True)
class RecoveryAuditResult:
    """Compact evidence identity returned after every hard check passes."""

    trial_count: int
    fingerprint_count: int
    source_git_commit: str
    source_sha256: str
    dataset_sha256: str
    summary_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_count": self.trial_count,
            "fingerprint_count": self.fingerprint_count,
            "source_git_commit": self.source_git_commit,
            "source_sha256": self.source_sha256,
            "dataset_sha256": self.dataset_sha256,
            "summary_sha256": self.summary_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_value(values: set[Any], name: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"{name} 必须唯一，实际为 {sorted(values, key=str)!r}")
    return next(iter(values))


def _audit_trace(npz_path: Path, trial: dict[str, Any]) -> None:
    with np.load(npz_path, allow_pickle=False) as trace:
        missing = REQUIRED_TRACE_KEYS.difference(trace.files)
        if missing:
            raise ValueError(f"{npz_path.name} 缺少 trace 字段：{sorted(missing)!r}")

        lengths = {key: len(trace[key]) for key in STEP_ALIGNED_TRACE_KEYS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{npz_path.name} 的逐 step 字段未对齐：{lengths!r}")
        if next(iter(lengths.values())) <= 0:
            raise ValueError(f"{npz_path.name} 是空 trace")

        for key in (
            "terminal_object_position",
            "terminal_target_position",
            "terminal_ee_position",
        ):
            if trace[key].shape != (3,):
                raise ValueError(f"{npz_path.name} 的 {key} 形状不是 (3,)")

        metrics = trial["metrics"]
        if not np.allclose(
            trace["terminal_object_position"],
            np.asarray(metrics["final_object_position_m"], dtype=np.float64),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(f"{npz_path.name} 的终端物体位置与 JSON 指标不一致")
        if not np.allclose(
            trace["terminal_target_position"],
            np.asarray(metrics["final_target_position_m"], dtype=np.float64),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(f"{npz_path.name} 的终端目标位置与 JSON 指标不一致")


def audit_recovery_run(
    summary_path: Path,
    protocol_path: Path,
    *,
    expected_source_commit: str | None = None,
) -> RecoveryAuditResult:
    """Validate the complete Cartesian protocol, fingerprints, and trace schema."""

    summary_path = summary_path.resolve()
    protocol_path = protocol_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    trials = summary.get("trials", [])

    methods = tuple(protocol["methods"])
    conditions = tuple(protocol["conditions"])
    seeds = tuple(int(seed) for seed in protocol["held_out_test_seeds"])
    expected_keys = set(itertools.product(methods, conditions, seeds))
    actual_keys = {
        (trial["method"], trial["condition"], int(trial["seed"])) for trial in trials
    }
    if len(trials) != len(expected_keys) or actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        unexpected = sorted(actual_keys.difference(expected_keys))
        raise ValueError(
            "试验笛卡尔积不完整："
            f"记录={len(trials)}，预期={len(expected_keys)}，"
            f"缺失={missing[:5]!r}，额外={unexpected[:5]!r}"
        )

    fingerprints = [trial["experiment_fingerprint"] for trial in trials]
    if len(set(fingerprints)) != len(trials):
        raise ValueError("experiment_fingerprint 存在重复")

    source_commits = {
        trial["experiment_config"]["source_git_commit"] for trial in trials
    }
    source_commit = _single_value(source_commits, "source_git_commit")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ValueError(
            f"源码提交不匹配：结果={source_commit}，协议 tag={expected_source_commit}"
        )
    source_sha = _single_value(
        {trial["experiment_config"]["source_sha256"] for trial in trials},
        "source_sha256",
    )
    dataset_sha = _single_value(
        {trial["dataset_sha256"] for trial in trials}, "dataset_sha256"
    )

    expected_steps = int(protocol["maximum_steps"])
    expected_threshold = float(protocol["success_xy_threshold_m"])
    for trial in trials:
        config = trial["experiment_config"]
        if int(config["schema_version"]) != int(summary["evaluation_schema_version"]):
            raise ValueError("trial 与 summary 的 schema_version 不一致")
        if int(config["max_steps"]) != expected_steps:
            raise ValueError("trial 的 max_steps 与预注册协议不一致")
        if not np.isclose(
            float(config["success_xy_threshold_m"]), expected_threshold, atol=0.0, rtol=0.0
        ):
            raise ValueError("trial 的 success_xy_threshold_m 与预注册协议不一致")
        if config["dataset_sha256"] != dataset_sha:
            raise ValueError("experiment_config 与 trial 顶层数据哈希不一致")

    drop_trials = [trial for trial in trials if trial["condition"].startswith("drop_")]
    drop_configs = [trial["experiment_config"]["perturbation_config"] for trial in drop_trials]
    if {float(config["distance_m"]) for config in drop_configs} != {
        float(value) for value in protocol["drop_distances_m"]
    }:
        raise ValueError("drop 距离覆盖与预注册协议不一致")
    if {config["direction"] for config in drop_configs} != set(protocol["drop_directions"]):
        raise ValueError("drop 方向覆盖与预注册协议不一致")
    if {int(config["force_open_steps"]) for config in drop_configs} != {
        int(value) for value in protocol["drop_force_open_steps"]
    }:
        raise ValueError("drop 夹爪行为覆盖与预注册协议不一致")

    trial_dir = summary_path.parent / "trials"
    json_paths = sorted(trial_dir.glob("*.json"))
    npz_paths = sorted(trial_dir.glob("*.npz"))
    if len(json_paths) != len(trials) or len(npz_paths) != len(trials):
        raise ValueError(
            f"逐试验文件数量错误：JSON={len(json_paths)}，NPZ={len(npz_paths)}"
        )

    by_name = {
        f"{trial['method']}__{trial['condition']}__seed_{int(trial['seed'])}": trial
        for trial in trials
    }
    if {path.stem for path in json_paths} != set(by_name):
        raise ValueError("逐试验 JSON 文件名集合与 summary 不一致")
    if {path.stem for path in npz_paths} != set(by_name):
        raise ValueError("逐试验 NPZ 文件名集合与 summary 不一致")

    for path in json_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = by_name[path.stem]
        if record != expected:
            raise ValueError(f"{path.name} 与 summary 中对应记录不一致")
    for path in npz_paths:
        _audit_trace(path, by_name[path.stem])

    return RecoveryAuditResult(
        trial_count=len(trials),
        fingerprint_count=len(set(fingerprints)),
        source_git_commit=source_commit,
        source_sha256=source_sha,
        dataset_sha256=dataset_sha,
        summary_sha256=_sha256(summary_path),
    )
