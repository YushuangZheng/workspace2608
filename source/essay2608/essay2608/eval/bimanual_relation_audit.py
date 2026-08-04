"""Read-only audit for pre-registered online bimanual-relation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from essay2608.eval.bimanual_relation_study import (
    condition_realization,
    score_bimanual_relation_trace,
)


STRING_KEYS = {
    "truth_label",
    "inferred_label",
    "inferred_left_state",
    "inferred_right_state",
    "intervention_event",
}
STEP_KEYS = {
    "state",
    "left_ee_pose",
    "right_ee_pose",
    "object_pose",
    "left_finger_force",
    "right_finger_force",
    "left_finger_position",
    "right_finger_position",
    "left_finger_distance_m",
    "right_finger_distance_m",
    "left_finger_velocity_m_s",
    "right_finger_velocity_m_s",
    "base_action",
    "applied_action",
    "truth_left_connected",
    "truth_right_connected",
    "truth_left_confidence",
    "truth_right_confidence",
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
    "intervention_active",
    "intervention_event",
    "phase_clock_held",
}
SCALAR_KEYS = {
    "control_dt",
    "seed",
    "condition",
    "source_sha256",
    "config_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_fingerprint(repository: Path) -> str:
    sources = [
        repository / "scripts/eval_bimanual_relation.py",
        repository / "source/essay2608/essay2608/bimanual.py",
        repository / "source/essay2608/essay2608/bimanual_physical.py",
        repository / "source/essay2608/essay2608/eval/bimanual_relation.py",
        repository / "source/essay2608/essay2608/eval/bimanual_relation_study.py",
        repository / "source/essay2608/essay2608/policy/relation.py",
        repository / "source/essay2608/essay2608/policy/bimanual_relation.py",
        repository
        / "source/essay2608/essay2608/tasks/manager_based/bimanual_physical_handover"
        / "physical_handover_env_cfg.py",
    ]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(repository)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _experiment_fingerprint(protocol: dict[str, Any], condition: str, seed: int) -> str:
    payload = {
        "task_id": protocol["task_id"],
        "seed": int(seed),
        "condition": condition,
        "max_steps": int(protocol["max_steps"]),
        "config_sha256": protocol["config_sha256"],
        "source_sha256": protocol["source_sha256"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        if actual is None or not np.isclose(float(actual), expected, atol=1e-10, rtol=1e-9):
            raise ValueError(f"{path} 的浮点值不一致：{actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"{path} 不一致：{actual!r} != {expected!r}")


def _edge_transition_gate(
    metrics: dict[str, Any],
    *,
    maximum_delay_s: float,
    label: str,
) -> None:
    transitions = metrics["transitions"]
    if transitions["num_matched_transitions"] != transitions["num_truth_transitions"]:
        raise ValueError(f"{label} 存在未匹配的物理关系转移")
    maximum = transitions["maximum_delay_s"]
    if maximum is not None and maximum > maximum_delay_s + 1e-10:
        raise ValueError(f"{label} 最大转移延迟 {maximum:.3f}s 超过协议门")


def _audit_trial(
    trace_path: Path,
    result: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    condition = str(result["condition"])
    seed = int(result["seed"])
    if result.get("artifact_type") != "bimanual_relation_online_trial":
        raise ValueError(f"{trace_path.name} 的 artifact_type 错误")
    if result.get("task_id") != protocol["task_id"]:
        raise ValueError(f"{trace_path.name} 的 task_id 错误")
    for key in ("source_sha256", "config_sha256", "dataset_sha256"):
        expected_key = "dataset_sha256" if key == "dataset_sha256" else key
        if result.get(key) != protocol[expected_key]:
            raise ValueError(f"{trace_path.name} 的 {key} 与协议不一致")
    expected_fingerprint = _experiment_fingerprint(protocol, condition, seed)
    if result.get("experiment_fingerprint") != expected_fingerprint:
        raise ValueError(f"{trace_path.name} 的实验指纹错误")

    with np.load(trace_path, allow_pickle=False) as archive:
        missing = (STEP_KEYS | SCALAR_KEYS).difference(archive.files)
        if missing:
            raise ValueError(f"{trace_path.name} 缺少字段：{sorted(missing)!r}")
        steps = len(archive["truth_label"])
        lengths = {key: len(archive[key]) for key in STEP_KEYS}
        if steps <= 0 or set(lengths.values()) != {steps}:
            raise ValueError(f"{trace_path.name} 的逐步数组未对齐：{lengths!r}")
        expected_shapes = {
            "left_ee_pose": (steps, 7),
            "right_ee_pose": (steps, 7),
            "object_pose": (steps, 7),
            "left_finger_force": (steps, 2, 3),
            "right_finger_force": (steps, 2, 3),
            "left_finger_position": (steps, 2, 3),
            "right_finger_position": (steps, 2, 3),
            "base_action": (steps, 16),
            "applied_action": (steps, 16),
        }
        for key, shape in expected_shapes.items():
            if archive[key].shape != shape:
                raise ValueError(f"{trace_path.name} 的 {key} 形状错误：{archive[key].shape}")
        for key in STEP_KEYS.difference(STRING_KEYS):
            if not np.all(np.isfinite(archive[key])):
                raise ValueError(f"{trace_path.name} 的 {key} 含非有限值")

        scalar_expectations = {
            "seed": seed,
            "condition": condition,
            "source_sha256": protocol["source_sha256"],
            "config_sha256": protocol["config_sha256"],
        }
        for key, expected in scalar_expectations.items():
            if np.asarray(archive[key]).item() != expected:
                raise ValueError(f"{trace_path.name} 的 trace {key} 错误")
        control_dt = float(np.asarray(archive["control_dt"]).item())
        if not np.isclose(control_dt, protocol["control_dt_s"], atol=1e-6, rtol=0.0):
            raise ValueError(f"{trace_path.name} 的控制周期错误")

        truth_left = archive["truth_left_connected"].astype(bool)
        truth_right = archive["truth_right_connected"].astype(bool)
        inferred_left = archive["inferred_left_connected"].astype(bool)
        inferred_right = archive["inferred_right_connected"].astype(bool)
        truth_labels = archive["truth_label"].astype("U16")
        inferred_labels = archive["inferred_label"].astype("U16")
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
            raise ValueError(f"{trace_path.name} 的物理标签不是由两条真值边组成")
        if not np.array_equal(inferred_labels, derived_inferred):
            raise ValueError(f"{trace_path.name} 的推断标签不是由两条估计边组成")

        metrics = score_bimanual_relation_trace(
            truth_labels=truth_labels,
            inferred_labels=inferred_labels,
            truth_left=truth_left,
            truth_right=truth_right,
            inferred_left=inferred_left,
            inferred_right=inferred_right,
            control_dt_s=control_dt,
        )
        realization = condition_realization(
            condition,
            truth_left=truth_left,
            truth_right=truth_right,
            truth_labels=truth_labels,
            intervention_active=archive["intervention_active"],
            intervention_event=archive["intervention_event"],
            control_dt_s=control_dt,
        )
        intervention_active = archive["intervention_active"].astype(bool)
        intervention_event = archive["intervention_event"].astype("U32")

    if int(result.get("steps", -1)) != steps:
        raise ValueError(f"{trace_path.name} 的 JSON steps 错误")
    _assert_nested_close(result.get("relation_metrics"), metrics, "relation_metrics")
    _assert_nested_close(result.get("condition_realization"), realization, "condition_realization")
    if not realization["realized"]:
        raise ValueError(f"{trace_path.name} 的物理干预没有按定义成立")
    if result.get("relation_metrics", {}).get(
        "privileged_contact_used_as_estimator_input"
    ):
        raise ValueError(f"{trace_path.name} 声称估计器使用了 privileged contact")

    expected_sequence = protocol["expected_inferred_sequences"][condition]
    truth_sequence = _compressed(truth_labels)
    inferred_sequence = _compressed(inferred_labels)
    if truth_sequence != expected_sequence:
        raise ValueError(f"{trace_path.name} 的物理序列偏离协议：{truth_sequence!r}")
    if inferred_sequence != expected_sequence:
        raise ValueError(f"{trace_path.name} 的推断序列偏离协议：{inferred_sequence!r}")
    if result.get("truth_relation_sequence") != truth_sequence:
        raise ValueError(f"{trace_path.name} 的 JSON 物理序列错误")
    if result.get("inferred_relation_sequence") != inferred_sequence:
        raise ValueError(f"{trace_path.name} 的 JSON 推断序列错误")

    if metrics["four_value_accuracy"] < protocol["minimum_four_value_accuracy"]:
        raise ValueError(f"{trace_path.name} 的四值准确率低于协议门")
    if metrics["left"]["f1"] < protocol["minimum_left_f1"]:
        raise ValueError(f"{trace_path.name} 的左边 F1 低于协议门")
    _edge_transition_gate(
        metrics["left"],
        maximum_delay_s=protocol["maximum_left_transition_delay_s"],
        label=f"{trace_path.name} 左边",
    )

    if condition == "receiver_miss":
        if metrics["right"]["fp"] > protocol[
            "receiver_miss_maximum_right_false_positive_steps"
        ]:
            raise ValueError(f"{trace_path.name} 的空抓右边存在假阳性")
        if np.any(inferred_right):
            raise ValueError(f"{trace_path.name} 的空抓曾推断右边连接")
    else:
        if metrics["right"]["f1"] < protocol["minimum_right_f1"][condition]:
            raise ValueError(f"{trace_path.name} 的右边 F1 低于协议门")
        _edge_transition_gate(
            metrics["right"],
            maximum_delay_s=protocol["maximum_right_transition_delay_s"][condition],
            label=f"{trace_path.name} 右边",
        )

    if condition == "receiver_delayed":
        forced_open = intervention_event == "receiver_delay_open"
        if np.any(inferred_right[forced_open]):
            raise ValueError(f"{trace_path.name} 在强制延迟张开时推断了右边")
    elif condition == "prolonged_both_hold":
        if not np.all(inferred_labels[intervention_active] == "both"):
            raise ValueError(f"{trace_path.name} 在延长双持时没有保持推断 both")
    elif condition == "one_arm_paused":
        if not np.all(inferred_right[intervention_active]):
            raise ValueError(f"{trace_path.name} 在接收臂暂停时丢失推断右边")

    return {
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "task_success": bool(result["task_success"]),
        "four_value_accuracy": metrics["four_value_accuracy"],
        "left": {key: metrics["left"][key] for key in ("tp", "fp", "fn", "f1")},
        "right": {key: metrics["right"][key] for key in ("tp", "fp", "fn", "f1")},
        "truth_sequence": truth_sequence,
        "inferred_sequence": inferred_sequence,
        "condition_realized": True,
        "experiment_fingerprint": expected_fingerprint,
        "trace_sha256": _sha256(trace_path),
    }


def _aggregate_edge(entries: list[dict[str, Any]], side: str) -> dict[str, Any]:
    tp = sum(entry[side]["tp"] for entry in entries)
    fp = sum(entry[side]["fp"] for entry in entries)
    fn = sum(entry[side]["fn"] for entry in entries)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def audit_bimanual_relation_results(
    results_dir: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Hard-audit exact membership, provenance, traces, metrics, and gates."""

    results_dir = Path(results_dir).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    repository = protocol_path.parents[2]
    config = Path(protocol["estimator_config"])
    config = config if config.is_absolute() else repository / config
    if _sha256(config) != protocol["config_sha256"]:
        raise ValueError("当前估计配置哈希与预注册协议不一致")
    if protocol.get("verify_current_sources", True):
        current_source = _source_fingerprint(repository)
        if current_source != protocol["source_sha256"]:
            raise ValueError("当前在线评测源码指纹与预注册协议不一致")

    expected = [
        (condition, int(seed))
        for condition in protocol["conditions"]
        for seed in protocol["formal_seeds"]
    ]
    trial_dir = results_dir / "trials"
    expected_json = {
        f"bimanual_relation__{condition}__seed_{seed}.json"
        for condition, seed in expected
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
    for condition, seed in expected:
        stem = f"bimanual_relation__{condition}__seed_{seed}"
        result = json.loads((trial_dir / f"{stem}.json").read_text(encoding="utf-8"))
        entries.append(_audit_trial(trial_dir / f"{stem}.npz", result, protocol))

    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("正式 summary 的源码指纹错误")
    if summary.get("config_sha256") != protocol["config_sha256"]:
        raise ValueError("正式 summary 的配置哈希错误")
    if summary.get("conditions") != protocol["conditions"]:
        raise ValueError("正式 summary 的条件顺序错误")
    if summary.get("seeds") != protocol["formal_seeds"]:
        raise ValueError("正式 summary 的 seed 顺序错误")
    if summary.get("num_expected_trials") != len(expected):
        raise ValueError("正式 summary 的试验计数错误")
    if summary.get("num_valid_trials") != len(expected):
        raise ValueError("正式 summary 存在缺失 worker")
    if not summary.get("all_conditions_physically_realized"):
        raise ValueError("正式 summary 存在未实现的物理条件")
    summary_pairs = [
        (row.get("condition"), row.get("seed")) for row in summary.get("trials", [])
    ]
    if summary_pairs != expected:
        raise ValueError("正式 summary 的 trial 成员或顺序错误")

    by_condition: dict[str, Any] = {}
    for condition in protocol["conditions"]:
        rows = [entry for entry in entries if entry["condition"] == condition]
        task_successes = sum(entry["task_success"] for entry in rows)
        if condition in protocol["task_success_conditions"] and task_successes < protocol[
            "minimum_task_successes_per_required_condition"
        ]:
            raise ValueError(f"{condition} 的任务成功数低于协议门：{task_successes}")
        by_condition[condition] = {
            "num_trials": len(rows),
            "num_task_successes": task_successes,
            "mean_four_value_accuracy": float(
                np.mean([entry["four_value_accuracy"] for entry in rows])
            ),
            "minimum_four_value_accuracy": min(
                entry["four_value_accuracy"] for entry in rows
            ),
            "minimum_left_f1": min(entry["left"]["f1"] for entry in rows),
            "minimum_right_f1": min(entry["right"]["f1"] for entry in rows),
        }

    return {
        "protocol_version": protocol["protocol_version"],
        "trial_count": len(entries),
        "all_trials_passed": True,
        "source_sha256": protocol["source_sha256"],
        "config_sha256": protocol["config_sha256"],
        "dataset_sha256": protocol["dataset_sha256"],
        "summary_sha256": _sha256(summary_path),
        "weighted_four_value_accuracy": float(
            np.average(
                [entry["four_value_accuracy"] for entry in entries],
                weights=[entry["steps"] for entry in entries],
            )
        ),
        "left": _aggregate_edge(entries, "left"),
        "right": _aggregate_edge(entries, "right"),
        "by_condition": by_condition,
        "entries": entries,
    }
