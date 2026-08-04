"""双臂关系恢复正式结果的只读硬审计。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from essay2608.eval.bimanual_recovery_study import (
    fault_realization,
    score_bimanual_recovery_trace,
    task_outcome_from_trace,
)
from essay2608.eval.bimanual_relation_study import score_bimanual_relation_trace


STRING_STEP_KEYS = {
    "truth_label",
    "inferred_left_state",
    "inferred_right_state",
    "inferred_label",
    "control_left_state",
    "control_right_state",
    "recovery_state",
    "recovery_trigger",
    "recovery_transition",
    "expert_rebase_event",
    "intervention_event",
}
STEP_KEYS = {
    "state",
    "left_ee_pose",
    "right_ee_pose",
    "object_pose",
    "target_pose",
    "left_finger_force",
    "right_finger_force",
    "left_finger_position",
    "right_finger_position",
    "left_finger_distance_m",
    "right_finger_distance_m",
    "left_finger_velocity_m_s",
    "right_finger_velocity_m_s",
    "base_action",
    "supervised_action",
    "applied_action",
    "truth_left_connected",
    "truth_right_connected",
    "truth_label",
    "inferred_left_connected",
    "inferred_right_connected",
    "inferred_left_confidence",
    "inferred_right_confidence",
    "inferred_left_connection_score",
    "inferred_right_connection_score",
    "inferred_left_loss_score",
    "inferred_right_loss_score",
    "inferred_left_state",
    "inferred_right_state",
    "inferred_label",
    "control_left_state",
    "control_right_state",
    "recovery_state",
    "recovery_trigger",
    "recovery_transition",
    "recovery_requires_giver",
    "expert_rebase_event",
    "recovery_action_overridden",
    "transfer_gate_active",
    "regrasp_attempts",
    "phase_clock_held",
    "intervention_active",
    "intervention_event",
}
SCALAR_KEYS = {
    "control_dt",
    "method",
    "condition",
    "seed",
    "source_sha256",
    "relation_config_sha256",
    "recovery_config_sha256",
    "dataset_sha256",
    "maximum_steps",
    "privileged_relation_used_for_control",
    "expert_complete",
    "expert_failed",
    "expert_failure_reason",
    "recovery_failed",
    "environment_done",
}
TERMINAL_KEYS = {
    "terminal_left_pose",
    "terminal_right_pose",
    "terminal_object_pose",
    "terminal_target_pose",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _canonical_sha256(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fingerprint(repository: Path) -> str:
    sources = [
        repository / "scripts/eval_bimanual_recovery.py",
        repository / "source/essay2608/essay2608/bimanual.py",
        repository / "source/essay2608/essay2608/bimanual_physical.py",
        repository / "source/essay2608/essay2608/eval/bimanual_relation.py",
        repository / "source/essay2608/essay2608/eval/bimanual_relation_study.py",
        repository / "source/essay2608/essay2608/eval/bimanual_recovery_study.py",
        repository / "source/essay2608/essay2608/policy/relation.py",
        repository / "source/essay2608/essay2608/policy/bimanual_relation.py",
        repository / "source/essay2608/essay2608/policy/bimanual_recovery.py",
        repository
        / "source/essay2608/essay2608/tasks/manager_based/bimanual_physical_handover"
        / "physical_handover_env_cfg.py",
    ]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(repository)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _experiment_fingerprint(
    protocol: dict[str, Any], method: str, condition: str, seed: int
) -> str:
    payload = {
        "task_id": protocol["task_id"],
        "method": method,
        "condition": condition,
        "seed": int(seed),
        "max_steps": int(protocol["maximum_steps"]),
        "source_sha256": protocol["source_sha256"],
        "relation_config_sha256": protocol["relation_config_sha256"],
        "recovery_config_sha256": protocol["recovery_config_sha256"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compressed(values: np.ndarray) -> list[str]:
    return [
        str(value)
        for index, value in enumerate(values)
        if index == 0 or value != values[index - 1]
    ]


def _assert_nested_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path} 的字典键不一致")
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} 的列表长度不一致")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_nested_close(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if actual is None or not np.isclose(
            float(actual), expected, atol=1e-7, rtol=1e-7
        ):
            raise ValueError(f"{path} 的浮点值不一致：{actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"{path} 不一致：{actual!r} != {expected!r}")


def _trace_arrays(archive: Any) -> dict[str, np.ndarray]:
    return {key: np.asarray(archive[key]) for key in STEP_KEYS}


def _maximum_recovery_supervised_target_jump(
    arrays: dict[str, np.ndarray],
) -> float | None:
    recovery_active = ~np.isin(
        arrays["recovery_state"].astype("U64"),
        ["NOT_APPLICABLE", "NORMAL"],
    )
    if not np.any(recovery_active) or len(recovery_active) < 2:
        return None
    boundary_mask = recovery_active[:-1] | recovery_active[1:]
    action = np.asarray(arrays["supervised_action"], dtype=np.float64)
    values = []
    for target in (action[:, :3], action[:, 8:11]):
        differences = np.linalg.norm(np.diff(target, axis=0), axis=-1)
        selected = differences[boundary_mask]
        if len(selected):
            values.append(float(np.max(selected)))
    return max(values) if values else None


def _audit_information_boundary(
    *,
    path: Path,
    method: str,
    arrays: dict[str, np.ndarray],
) -> None:
    left_control = arrays["control_left_state"].astype("U32")
    right_control = arrays["control_right_state"].astype("U32")
    if method == "clocked_expert":
        if not np.all(left_control == "NOT_APPLICABLE") or not np.all(
            right_control == "NOT_APPLICABLE"
        ):
            raise ValueError(f"{path.name} 的时钟专家不应读取关系边")
        return
    if method == "oracle_relation_recovery":
        expected_left = np.where(arrays["truth_left_connected"], "CONNECTED", "DISCONNECTED")
        expected_right = np.where(
            arrays["truth_right_connected"], "CONNECTED", "DISCONNECTED"
        )
    else:
        expected_left = arrays["inferred_left_state"].astype("U32")
        expected_right = arrays["inferred_right_state"].astype("U32")
    if not np.array_equal(left_control, expected_left) or not np.array_equal(
        right_control, expected_right
    ):
        raise ValueError(f"{path.name} 的控制关系边违反方法信息边界")


def _audit_recovery_gates(
    *,
    path: Path,
    method: str,
    condition: str,
    task_success: bool,
    metrics: dict[str, Any],
    relation_metrics: dict[str, Any],
    maximum_supervised_jump: float | None,
    protocol: dict[str, Any],
) -> None:
    if method in {"relation_gate", "relation_recovery", "oracle_relation_recovery"}:
        if metrics["unsafe_release"]:
            raise ValueError(f"{path.name} 发生关系监督下的不安全释放")
    if method not in {"relation_recovery", "oracle_relation_recovery"}:
        return
    if condition == "normal":
        if metrics["false_recovery_trigger"] or metrics["recovery_triggered"]:
            raise ValueError(f"{path.name} 在正常工况误触发恢复")
        return
    if not task_success or not metrics["safe_release"]:
        raise ValueError(f"{path.name} 的关系恢复没有安全完成任务")
    if not metrics["recovery_success"] or not metrics["relation_reestablished_after_fault"]:
        raise ValueError(f"{path.name} 没有完成可验证的关系恢复")
    if metrics["maximum_regrasp_attempts_observed"] > protocol["maximum_regrasp_attempts"]:
        raise ValueError(f"{path.name} 的重抓次数超过协议上限")
    if metrics["time_to_recover_s"] > protocol["maximum_recovery_time_s"][condition]:
        raise ValueError(f"{path.name} 的恢复时间超过协议门")
    maximum_jump = metrics["maximum_recovery_action_target_jump_m"]
    if maximum_jump is not None and maximum_jump > protocol[
        "maximum_recovery_action_target_jump_m"
    ]:
        raise ValueError(f"{path.name} 的恢复动作跳变超过协议门")
    if maximum_supervised_jump is not None and maximum_supervised_jump > protocol[
        "maximum_recovery_supervised_target_jump_m"
    ]:
        raise ValueError(f"{path.name} 的恢复监督动作跳变超过协议门")
    maximum_speed = metrics["maximum_recovery_ee_speed_m_s"]
    if maximum_speed is not None and maximum_speed > protocol[
        "maximum_recovery_ee_speed_m_s"
    ]:
        raise ValueError(f"{path.name} 的恢复末端速度超过协议门")
    if condition == "receiver_brief_loss":
        if metrics["geometry_recovery_executed"] or not metrics["transient_loss_cancelled"]:
            raise ValueError(f"{path.name} 对短暂丢失执行了不必要的几何恢复")
    else:
        if not metrics["geometry_recovery_executed"]:
            raise ValueError(f"{path.name} 的强故障未执行几何恢复")
    if condition == "receiver_loss_after_release":
        if metrics["giver_retention_required"]:
            raise ValueError(f"{path.name} 在发送臂已释放后错误要求 giver 边")
    elif not metrics["giver_retention_required"] or not metrics[
        "giver_retained_during_recovery"
    ]:
        raise ValueError(f"{path.name} 在交接恢复时没有保持发送边")
    retained = metrics["receiver_retained_until_intended_release"]
    if retained is not None and not retained:
        raise ValueError(f"{path.name} 恢复后在计划释放前再次丢失接收边")
    if method == "relation_recovery" and relation_metrics["four_value_accuracy"] < protocol[
        "minimum_online_four_value_accuracy"
    ]:
        raise ValueError(f"{path.name} 的在线关系准确率低于协议门")


def _audit_trial(
    trace_path: Path,
    result: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    method = str(result["method"])
    condition = str(result["condition"])
    seed = int(result["seed"])
    if result.get("artifact_type") != "bimanual_relation_recovery_trial":
        raise ValueError(f"{trace_path.name} 的 artifact_type 错误")
    if result.get("task_id") != protocol["task_id"]:
        raise ValueError(f"{trace_path.name} 的 task_id 错误")
    expected_values = {
        "source_sha256": protocol["source_sha256"],
        "relation_config_sha256": protocol["relation_config_sha256"],
        "recovery_config_sha256": protocol["recovery_config_sha256"],
        "dataset_sha256": protocol["dataset_sha256"],
        "maximum_steps": protocol["maximum_steps"],
    }
    for key, expected in expected_values.items():
        if result.get(key) != expected:
            raise ValueError(f"{trace_path.name} 的 {key} 与协议不一致")
    expected_fingerprint = _experiment_fingerprint(protocol, method, condition, seed)
    if result.get("experiment_fingerprint") != expected_fingerprint:
        raise ValueError(f"{trace_path.name} 的实验指纹错误")
    privileged = method == "oracle_relation_recovery"
    if bool(result.get("privileged_relation_used_for_control")) != privileged:
        raise ValueError(f"{trace_path.name} 的 privileged 控制声明错误")
    if result.get("privileged_relation_used_by_online_methods"):
        raise ValueError(f"{trace_path.name} 声称在线方法使用了 privileged relation")

    with np.load(trace_path, allow_pickle=False) as archive:
        missing = (STEP_KEYS | SCALAR_KEYS | TERMINAL_KEYS).difference(archive.files)
        if missing:
            raise ValueError(f"{trace_path.name} 缺少字段：{sorted(missing)!r}")
        steps = len(archive["state"])
        lengths = {key: len(archive[key]) for key in STEP_KEYS}
        if steps <= 0 or set(lengths.values()) != {steps}:
            raise ValueError(f"{trace_path.name} 的逐步数组未对齐：{lengths!r}")
        expected_shapes = {
            "left_ee_pose": (steps, 7),
            "right_ee_pose": (steps, 7),
            "object_pose": (steps, 7),
            "target_pose": (steps, 7),
            "left_finger_force": (steps, 2, 3),
            "right_finger_force": (steps, 2, 3),
            "left_finger_position": (steps, 2, 3),
            "right_finger_position": (steps, 2, 3),
            "base_action": (steps, 16),
            "supervised_action": (steps, 16),
            "applied_action": (steps, 16),
            "terminal_left_pose": (7,),
            "terminal_right_pose": (7,),
            "terminal_object_pose": (7,),
            "terminal_target_pose": (7,),
        }
        for key, shape in expected_shapes.items():
            if archive[key].shape != shape:
                raise ValueError(f"{trace_path.name} 的 {key} 形状错误：{archive[key].shape}")
        for key in STEP_KEYS.difference(STRING_STEP_KEYS):
            if not np.all(np.isfinite(archive[key])):
                raise ValueError(f"{trace_path.name} 的 {key} 含非有限值")
        for key in TERMINAL_KEYS:
            if not np.all(np.isfinite(archive[key])):
                raise ValueError(f"{trace_path.name} 的 {key} 含非有限值")

        scalar_expected = {
            "method": method,
            "condition": condition,
            "seed": seed,
            "source_sha256": protocol["source_sha256"],
            "relation_config_sha256": protocol["relation_config_sha256"],
            "recovery_config_sha256": protocol["recovery_config_sha256"],
            "dataset_sha256": protocol["dataset_sha256"],
            "maximum_steps": protocol["maximum_steps"],
            "privileged_relation_used_for_control": privileged,
        }
        for key, expected in scalar_expected.items():
            if np.asarray(archive[key]).item() != expected:
                raise ValueError(f"{trace_path.name} 的 trace {key} 错误")
        control_dt = float(np.asarray(archive["control_dt"]).item())
        if not np.isclose(control_dt, protocol["control_dt_s"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{trace_path.name} 的控制周期错误")

        arrays = _trace_arrays(archive)
        truth_left = arrays["truth_left_connected"].astype(bool)
        truth_right = arrays["truth_right_connected"].astype(bool)
        inferred_left = arrays["inferred_left_connected"].astype(bool)
        inferred_right = arrays["inferred_right_connected"].astype(bool)
        truth_labels = arrays["truth_label"].astype("U16")
        inferred_labels = arrays["inferred_label"].astype("U16")
        derived_truth = np.where(
            truth_left & truth_right,
            "both",
            np.where(truth_left, "left_only", np.where(truth_right, "right_only", "none")),
        )
        derived_inferred = np.where(
            inferred_left & inferred_right,
            "both",
            np.where(
                inferred_left,
                "left_only",
                np.where(inferred_right, "right_only", "none"),
            ),
        )
        if not np.array_equal(truth_labels, derived_truth):
            raise ValueError(f"{trace_path.name} 的物理关系标签无法由两条边重算")
        if not np.array_equal(inferred_labels, derived_inferred):
            raise ValueError(f"{trace_path.name} 的推断关系标签无法由两条边重算")
        _audit_information_boundary(path=trace_path, method=method, arrays=arrays)

        expert_complete = bool(np.asarray(archive["expert_complete"]).item())
        expert_failed = bool(np.asarray(archive["expert_failed"]).item())
        failure_reason_value = str(np.asarray(archive["expert_failure_reason"]).item())
        expert_failure_reason = None if failure_reason_value == "none" else failure_reason_value
        recovery_failed = bool(np.asarray(archive["recovery_failed"]).item())
        environment_done = bool(np.asarray(archive["environment_done"]).item())
        task_success, failure_reason, outcome = task_outcome_from_trace(
            expert_complete=expert_complete,
            expert_failed=expert_failed,
            expert_failure_reason=expert_failure_reason,
            recovery_failed=recovery_failed,
            environment_done=environment_done,
            object_positions=arrays["object_pose"][:, :3],
            final_position=np.asarray(archive["terminal_object_pose"][:3]),
            target_position=np.asarray(archive["terminal_target_pose"][:3]),
        )
        metrics = score_bimanual_recovery_trace(
            arrays,
            condition,
            control_dt,
            method=method,
            task_success=task_success,
        )
        maximum_supervised_jump = _maximum_recovery_supervised_target_jump(arrays)
        relation_metrics = score_bimanual_relation_trace(
            truth_labels=truth_labels,
            inferred_labels=inferred_labels,
            truth_left=truth_left,
            truth_right=truth_right,
            inferred_left=inferred_left,
            inferred_right=inferred_right,
            control_dt_s=control_dt,
        )
        realization = fault_realization(condition, arrays, control_dt)

    if int(result.get("steps", -1)) != steps:
        raise ValueError(f"{trace_path.name} 的 JSON steps 错误")
    if result.get("task_success") != task_success or result.get(
        "task_failure_reason"
    ) != failure_reason:
        raise ValueError(f"{trace_path.name} 的任务结论无法由 trace 重算")
    for key, expected in {
        "expert_complete": expert_complete,
        "expert_failed": expert_failed,
        "recovery_failed": recovery_failed,
    }.items():
        if result.get(key) != expected:
            raise ValueError(f"{trace_path.name} 的 {key} 与 trace 不一致")
    _assert_nested_close(result.get("metrics"), metrics, "metrics")
    _assert_nested_close(result.get("relation_metrics"), relation_metrics, "relation_metrics")
    _assert_nested_close(result.get("fault_realization"), realization, "fault_realization")
    for key, expected in outcome.items():
        _assert_nested_close(result.get(key), expected, key)
    if not realization["realized"]:
        raise ValueError(f"{trace_path.name} 的物理故障未按定义成立")
    truth_sequence = _compressed(truth_labels)
    inferred_sequence = _compressed(inferred_labels)
    if result.get("truth_relation_sequence") != truth_sequence:
        raise ValueError(f"{trace_path.name} 的 JSON 物理序列错误")
    if result.get("inferred_relation_sequence") != inferred_sequence:
        raise ValueError(f"{trace_path.name} 的 JSON 推断序列错误")
    _audit_recovery_gates(
        path=trace_path,
        method=method,
        condition=condition,
        task_success=task_success,
        metrics=metrics,
        relation_metrics=relation_metrics,
        maximum_supervised_jump=maximum_supervised_jump,
        protocol=protocol,
    )
    return {
        "method": method,
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "task_success": task_success,
        "unsafe_release": metrics["unsafe_release"],
        "safe_release": metrics["safe_release"],
        "recovery_triggered": metrics["recovery_triggered"],
        "recovery_success": metrics["recovery_success"],
        "geometry_recovery_executed": metrics["geometry_recovery_executed"],
        "transient_loss_cancelled": metrics["transient_loss_cancelled"],
        "time_to_recover_s": metrics["time_to_recover_s"],
        "maximum_recovery_supervised_target_jump_m": maximum_supervised_jump,
        "left_path_length_m": metrics["left_path_length_m"],
        "right_path_length_m": metrics["right_path_length_m"],
        "four_value_accuracy": relation_metrics["four_value_accuracy"],
        "trace_sha256": _sha256(trace_path),
        "experiment_fingerprint": expected_fingerprint,
    }


def _aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_trials": len(entries),
        "num_task_successes": sum(row["task_success"] for row in entries),
        "num_safe_releases": sum(row["safe_release"] for row in entries),
        "num_unsafe_releases": sum(row["unsafe_release"] for row in entries),
        "num_recovery_triggers": sum(row["recovery_triggered"] for row in entries),
        "num_recovery_successes": sum(row["recovery_success"] is True for row in entries),
        "num_geometry_recoveries": sum(row["geometry_recovery_executed"] for row in entries),
        "num_transient_loss_cancellations": sum(
            row["transient_loss_cancelled"] for row in entries
        ),
        "mean_steps": float(np.mean([row["steps"] for row in entries])),
        "mean_left_path_length_m": float(
            np.mean([row["left_path_length_m"] for row in entries])
        ),
        "mean_right_path_length_m": float(
            np.mean([row["right_path_length_m"] for row in entries])
        ),
        "mean_four_value_accuracy": float(
            np.mean([row["four_value_accuracy"] for row in entries])
        ),
        "maximum_recovery_supervised_target_jump_m": max(
            (
                row["maximum_recovery_supervised_target_jump_m"]
                for row in entries
                if row["maximum_recovery_supervised_target_jump_m"] is not None
            ),
            default=None,
        ),
    }


def _controller_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "num_trials": count,
        "num_faults_realized": sum(
            row["fault_realization"]["realized"] for row in rows
        ),
        "num_task_successes": sum(row["task_success"] for row in rows),
        "num_safe_releases": sum(row["metrics"]["safe_release"] for row in rows),
        "num_unsafe_releases": sum(row["metrics"]["unsafe_release"] for row in rows),
        "num_recovery_triggers": sum(
            row["metrics"]["recovery_triggered"] for row in rows
        ),
        "num_geometry_recoveries": sum(
            row["metrics"]["geometry_recovery_executed"] for row in rows
        ),
        "num_transient_loss_cancellations": sum(
            row["metrics"]["transient_loss_cancelled"] for row in rows
        ),
        "num_recovery_successes": sum(
            row["metrics"]["recovery_success"] is True for row in rows
        ),
        "num_false_recovery_triggers": sum(
            row["metrics"]["false_recovery_trigger"] for row in rows
        ),
        "mean_final_xy_error_m": (
            sum(row["final_xy_error_m"] for row in rows) / count if count else None
        ),
        "mean_four_value_accuracy": (
            sum(row["relation_metrics"]["four_value_accuracy"] for row in rows) / count
            if count
            else None
        ),
    }


def audit_bimanual_recovery_results(
    results_dir: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """审计精确成员、身份、信息边界、原始 trace、指标和整体验收门。"""

    results_dir = Path(results_dir).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    repository = protocol_path.parents[2]
    relation_config = repository / protocol["relation_config"]
    recovery_config = repository / protocol["recovery_config"]
    if _sha256(relation_config) != protocol["relation_config_sha256"]:
        raise ValueError("当前关系配置哈希与预注册协议不一致")
    recovery_values = json.loads(recovery_config.read_text(encoding="utf-8"))
    if _canonical_sha256(recovery_values) != protocol["recovery_config_sha256"]:
        raise ValueError("当前恢复配置哈希与预注册协议不一致")
    if protocol.get("verify_current_sources", True):
        if _source_fingerprint(repository) != protocol["source_sha256"]:
            raise ValueError("当前双臂恢复源码指纹与预注册协议不一致")

    expected = [
        (method, condition, int(seed))
        for method in protocol["methods"]
        for condition in protocol["conditions"]
        for seed in protocol["formal_seeds"]
    ]
    trial_dir = results_dir / "trials"
    expected_json = {
        f"bimanual_recovery__{method}__{condition}__seed_{seed}.json"
        for method, condition, seed in expected
    }
    expected_npz = {name.replace(".json", ".npz") for name in expected_json}
    actual_json = {path.name for path in trial_dir.glob("*.json")}
    actual_npz = {path.name for path in trial_dir.glob("*.npz")}
    if actual_json != expected_json or actual_npz != expected_npz:
        raise ValueError(
            "正式结果成员不精确："
            f"JSON 缺失={sorted(expected_json - actual_json)!r} "
            f"JSON 额外={sorted(actual_json - expected_json)!r} "
            f"NPZ 缺失={sorted(expected_npz - actual_npz)!r} "
            f"NPZ 额外={sorted(actual_npz - expected_npz)!r}"
        )

    entries = []
    summary_trials = []
    for method, condition, seed in expected:
        stem = f"bimanual_recovery__{method}__{condition}__seed_{seed}"
        result = json.loads((trial_dir / f"{stem}.json").read_text(encoding="utf-8"))
        summary_trials.append({**result, "worker_returncode": 0})
        entries.append(_audit_trial(trial_dir / f"{stem}.npz", result, protocol))

    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_expected = {
        "artifact_type": protocol["summary_artifact_type"],
        "task_id": protocol["task_id"],
        "source_sha256": protocol["source_sha256"],
        "relation_config_sha256": protocol["relation_config_sha256"],
        "recovery_config_sha256": protocol["recovery_config_sha256"],
        "dataset_sha256": protocol["dataset_sha256"],
        "maximum_steps": protocol["maximum_steps"],
        "methods": protocol["methods"],
        "conditions": protocol["conditions"],
        "seeds": protocol["formal_seeds"],
        "num_expected_trials": len(expected),
        "num_valid_trials": len(expected),
        "all_faults_physically_realized": True,
        "control_dt_s": protocol["control_dt_s"],
    }
    for key, value in summary_expected.items():
        if summary.get(key) != value:
            raise ValueError(f"正式 summary 的 {key} 错误")
    actual_pairs = [
        (row.get("method"), row.get("condition"), row.get("seed"))
        for row in summary.get("trials", [])
    ]
    if actual_pairs != expected:
        raise ValueError("正式 summary 的 trial 成员或顺序错误")
    if any(row.get("worker_returncode") != 0 for row in summary.get("trials", [])):
        raise ValueError("正式 summary 存在非零 worker 返回码")
    if summary.get("trials") != summary_trials:
        raise ValueError("正式 summary 的 trial 内容与逐条 JSON 不一致")
    if summary.get("recovery_config") != recovery_values:
        raise ValueError("正式 summary 的恢复配置内容错误")
    expected_controller_aggregate = {
        method: {
            condition: _controller_aggregate(
                [
                    row
                    for row in summary_trials
                    if row["method"] == method and row["condition"] == condition
                ]
            )
            for condition in protocol["conditions"]
        }
        for method in protocol["methods"]
    }
    _assert_nested_close(
        summary.get("by_method_condition"),
        expected_controller_aggregate,
        "summary.by_method_condition",
    )

    by_method_condition: dict[str, dict[str, Any]] = {}
    for method in protocol["methods"]:
        by_method_condition[method] = {}
        for condition in protocol["conditions"]:
            rows = [
                row
                for row in entries
                if row["method"] == method and row["condition"] == condition
            ]
            aggregate = _aggregate(rows)
            minimum = protocol["minimum_task_successes"].get(method, {}).get(condition)
            if minimum is not None and aggregate["num_task_successes"] < minimum:
                raise ValueError(
                    f"{method}/{condition} 的任务成功数低于协议门："
                    f"{aggregate['num_task_successes']} < {minimum}"
                )
            by_method_condition[method][condition] = aggregate

    recovery_methods = ("relation_recovery", "oracle_relation_recovery")
    strong_conditions = tuple(protocol["strong_fault_conditions"])
    paired_online_wins = 0
    paired_online_losses = 0
    if {"clocked_expert", "relation_recovery"}.issubset(protocol["methods"]):
        for condition in strong_conditions:
            for seed in protocol["formal_seeds"]:
                clocked = next(
                    row
                    for row in entries
                    if row["method"] == "clocked_expert"
                    and row["condition"] == condition
                    and row["seed"] == seed
                )
                online = next(
                    row
                    for row in entries
                    if row["method"] == "relation_recovery"
                    and row["condition"] == condition
                    and row["seed"] == seed
                )
                paired_online_wins += int(
                    online["task_success"] and not clocked["task_success"]
                )
                paired_online_losses += int(
                    clocked["task_success"] and not online["task_success"]
                )
        if paired_online_wins < protocol["minimum_paired_strong_recovery_wins"]:
            raise ValueError("在线关系恢复相对时钟专家的配对强故障胜场不足")
        if paired_online_losses > protocol["maximum_paired_strong_recovery_losses"]:
            raise ValueError("在线关系恢复相对时钟专家出现过多配对退化")

    fault_conditions = [
        condition for condition in protocol["conditions"] if condition != "normal"
    ]
    recovery_successes = {
        method: sum(
            row["task_success"]
            for row in entries
            if row["method"] == method and row["condition"] in fault_conditions
        )
        for method in recovery_methods
    }
    oracle_gap = 0
    if set(recovery_methods).issubset(protocol["methods"]):
        oracle_gap = recovery_successes["oracle_relation_recovery"] - recovery_successes[
            "relation_recovery"
        ]
        if oracle_gap > protocol["maximum_online_oracle_success_gap"]:
            raise ValueError("在线关系恢复与 Oracle 的故障成功数差距超过协议门")

    return {
        "protocol_version": protocol["protocol_version"],
        "trial_count": len(entries),
        "all_trials_passed": True,
        "source_sha256": protocol["source_sha256"],
        "relation_config_sha256": protocol["relation_config_sha256"],
        "recovery_config_sha256": protocol["recovery_config_sha256"],
        "dataset_sha256": protocol["dataset_sha256"],
        "summary_sha256": _sha256(summary_path),
        "paired_online_strong_wins": paired_online_wins,
        "paired_online_strong_losses": paired_online_losses,
        "online_oracle_fault_success_gap": oracle_gap,
        "by_method_condition": by_method_condition,
        "entries": entries,
    }
