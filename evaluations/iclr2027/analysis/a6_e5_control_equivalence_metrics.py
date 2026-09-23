"""Cycle-level evidence for the control-equivalence ablation.

This analysis never runs RLBench.  It uses the retained Stress-4 episode and
cycle records that underlie the E5 Full versus no-control-equivalence
comparison.  The compact records directly support the operational hold,
repeat, alarm, recovery, and completion metrics.  The one internal quantity
that was not compacted into the cycle record -- accepted control-equivalence
classes -- is reconstructed by deterministic policy-only replay.  Such a
reconstruction is accepted only when every replayed command is bitwise-close
to the command retained by the formal run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
E5_ROOT = EVAL_ROOT / "results" / "controlled" / "e5"
FULL_ROOT = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "end_to_end" / "m5"
ABLATION_ROOT = E5_ROOT / "fine" / "no_control_equivalence"
PLAN = E5_ROOT / "A6_E5_FINE_RUN_PLAN.json"
OUTPUT = E5_ROOT / "derived" / "control_equivalence"
CURRENT_FREEZE = EVAL_ROOT / "results" / "a6_execution" / "M5_PENDING_DIRECT_IDENTITY_FREEZE.json"
STRESS4 = (
    "open_drawer",
    "place_cups_3",
    "bimanual_handover_item",
    "bimanual_lift_tray",
)
CONDITIONS = ("nominal", "perturbed")
METHODS = ("full", "no_control_equivalence")
REPLAY_ATOL = 1.0e-10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20270921


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_identity(paths: Iterable[Path]) -> dict[str, Any]:
    rows = []
    aggregate = hashlib.sha256()
    for path in sorted(set(Path(value) for value in paths), key=lambda value: str(value)):
        relative = str(path.resolve().relative_to(ROOT.resolve()))
        digest = _sha256(path)
        rows.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "files": len(rows),
        "aggregate_sha256": aggregate.hexdigest(),
        "entries": rows,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe(episode_id: str) -> str:
    return str(episode_id).replace("/", "__")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _cycle_path(root: Path, episode: Mapping[str, Any]) -> Path:
    raw = episode.get("cycle_file")
    if not isinstance(raw, str):
        raise RuntimeError(f"episode has no retained cycle file: {episode.get('episode_id')}")
    source = Path(str(episode["_source_path"]))
    path = (source.parent / raw).resolve()
    if not path.is_file():
        raise RuntimeError(f"missing retained cycle file: {path}")
    expected = episode.get("cycle_file_sha256")
    if expected is not None and _sha256(path) != expected:
        raise RuntimeError(f"cycle-file hash mismatch: {path}")
    return path


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    path = root / "episodes" / f"{_safe(episode_id)}.json"
    value = _load_json(path)
    if value.get("episode_id") != episode_id:
        raise RuntimeError(f"episode identity mismatch: {path}")
    value["_source_path"] = str(path)
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        values = [json.loads(line) for line in stream if line.strip()]
    for index, value in enumerate(values):
        if int(value.get("cycle", -1)) != index:
            raise RuntimeError(f"non-contiguous cycle record: {path}:{index}")
    return values


def _manifests() -> dict[str, list[dict[str, Any]]]:
    plan = _load_json(PLAN)
    result = {}
    for condition in CONDITIONS:
        entry = plan["manifests"][condition]
        path = ROOT / str(entry["path"])
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"E5 manifest changed after run-plan freeze: {path}")
        rows = _jsonl(path)
        if len(rows) != int(entry["episodes"]):
            raise RuntimeError(f"E5 manifest denominator mismatch: {path}")
        result[condition] = rows
    return result


def _paired_episodes() -> dict[tuple[str, str], list[dict[str, Any]]]:
    manifests = _manifests()
    output: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for condition, rows in manifests.items():
        full = []
        ablation = []
        for row in rows:
            full_id = str(row.get("source_episode_id") or row["episode_id"])
            full_episode = _episode(FULL_ROOT / condition, full_id)
            ablation_episode = _episode(ABLATION_ROOT / condition, str(row["episode_id"]))
            for value in (full_episode, ablation_episode):
                if (
                    value.get("task") != row.get("task")
                    or int(value.get("seed")) != int(row["seed"])
                    or int(value.get("variation")) != int(row["variation"])
                    or value.get("condition") != condition
                    or value.get("fault_family") != row.get("fault_family")
                    or value.get("fault_severity") != row.get("fault_severity")
                    or value.get("trigger_stage") != row.get("trigger_stage")
                ):
                    raise RuntimeError(
                        f"paired E5 identity mismatch: {condition}/{row['episode_id']}"
                    )
            full.append(full_episode)
            ablation.append(ablation_episode)
        output[("full", condition)] = full
        output[("no_control_equivalence", condition)] = ablation
    return output


def _reference_tokens(policy_state: Mapping[str, Any]) -> tuple[tuple[str, int, int], ...]:
    raw = policy_state.get("reference_state_if_available")
    if not isinstance(raw, Mapping):
        return ()
    if "skill" in raw and "progress" in raw:
        values = {"single": raw}
    else:
        values = raw
    output = []
    for arm in sorted(values):
        state = values[arm]
        if not isinstance(state, Mapping):
            continue
        if state.get("skill") is None or state.get("progress") is None:
            continue
        output.append((str(arm), int(state["skill"]), int(state["progress"])))
    return tuple(output)


def _all_task_mode(record: Mapping[str, Any]) -> bool:
    arms = record.get("execution", {}).get("policy_audit", {}).get("arms", {})
    return bool(arms) and all(
        str(value.get("mode_after", "")).lower() == "task"
        for value in arms.values()
    )


def _recovery_entry(record: Mapping[str, Any]) -> bool:
    arms = record.get("execution", {}).get("policy_audit", {}).get("arms", {})
    active = {"verify_link", "recovery", "reentry"}
    return any(
        str(value.get("mode_after", "")).lower() in active
        and str(value.get("mode_before", "")).lower() not in active
        for value in arms.values()
    )


def _violation_active(record: Mapping[str, Any]) -> bool:
    audit = record.get("audit", {})
    cycle = int(record["cycle"])
    onset = audit.get("violation_onset_cycle")
    end = audit.get("violation_end_cycle")
    if onset is None or cycle < int(onset):
        return False
    return end is None or cycle <= int(end)


def _same_target(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return bool(
        len(left) == len(right)
        and np.allclose(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-9,
        )
    )


def _longest_run(values: Sequence[Any]) -> int:
    if not values:
        return 0
    longest = current = 1
    for previous, value in zip(values, values[1:]):
        if value == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _episode_metrics(
    method: str,
    condition: str,
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    cycle_path = _cycle_path(
        FULL_ROOT / condition if method == "full" else ABLATION_ROOT / condition,
        episode,
    )
    records = _records(cycle_path)
    if len(records) != int(episode.get("cycle_records", episode.get("cycles", -1))):
        raise RuntimeError(f"cycle denominator mismatch: {cycle_path}")
    references = [
        _reference_tokens(value.get("feature", {}).get("policy_state", {}))
        for value in records
    ]
    actions = [value.get("feature", {}).get("action", ()) for value in records]
    repeated_reference = [False]
    repeated_target = [False]
    for index in range(1, len(records)):
        repeated_reference.append(bool(references[index]) and references[index] == references[index - 1])
        repeated_target.append(_same_target(actions[index], actions[index - 1]))

    alarms = []
    reasons_by_cycle = []
    for value in records:
        monitor = value.get("feature", {}).get("policy_state", {}).get("monitor", {})
        alarms.append(bool(monitor.get("alarm", False)))
        reasons_by_cycle.append(tuple(str(reason) for reason in monitor.get("reasons", ())))
    no_plausible = ["no_plausible_state" in reasons for reasons in reasons_by_cycle]
    mismatch = [any("mismatch" in reason for reason in reasons) for reasons in reasons_by_cycle]
    recovery_entries = [_recovery_entry(value) for value in records]
    off_violation_entries = [
        entered and not _violation_active(value)
        for entered, value in zip(recovery_entries, records)
    ]
    redundant_hold = []
    for index, value in enumerate(records):
        resolution = value.get("execution", {}).get("action_resolution", {})
        policy_audit = value.get("execution", {}).get("policy_audit", {})
        any_reentry = any(
            bool(arm.get("reentry_committed", False))
            for arm in policy_audit.get("arms", {}).values()
        )
        redundant_hold.append(
            index > 0
            and repeated_reference[index]
            and repeated_target[index]
            and resolution.get("aggregate") == "reached"
            and _all_task_mode(value)
            and not alarms[index]
            and not any_reentry
        )

    denominator = len(records)
    target_tokens = [
        tuple(round(float(component), 9) for component in action)
        for action in actions
    ]
    pair_key = "|".join(
        str(value)
        for value in (
            episode["task"],
            episode["seed"],
            episode["variation"],
            condition,
            episode.get("fault_family"),
            episode.get("fault_severity"),
            episode.get("trigger_stage"),
        )
    )
    return {
        "pair_key": pair_key,
        "method": method,
        "condition": condition,
        "task": episode["task"],
        "seed": int(episode["seed"]),
        "variation": int(episode["variation"]),
        "episode_id": episode["episode_id"],
        "success": int(bool(episode["final_success"])),
        "cycles": denominator,
        "reference_repeat_cycles": sum(repeated_reference),
        "reference_repeat_rate": sum(repeated_reference) / denominator if denominator else 0.0,
        "same_target_repeat_cycles": sum(repeated_target),
        "same_target_repeat_rate": sum(repeated_target) / denominator if denominator else 0.0,
        "redundant_hold_proxy_cycles": sum(redundant_hold),
        "redundant_hold_proxy_rate": sum(redundant_hold) / denominator if denominator else 0.0,
        "longest_same_reference_run": _longest_run(references),
        "longest_same_target_run": _longest_run(target_tokens),
        "no_plausible_reason_cycles": sum(no_plausible),
        "no_plausible_reason_rate": sum(no_plausible) / denominator if denominator else 0.0,
        "mismatch_reason_cycles": sum(mismatch),
        "mismatch_reason_rate": sum(mismatch) / denominator if denominator else 0.0,
        "alarm_cycles": sum(alarms),
        "recovery_entries": sum(recovery_entries),
        "off_violation_recovery_entries": sum(off_violation_entries),
        "false_interventions": int(episode.get("false_interventions", 0)),
        "infrastructure_error": int(episode.get("termination_reason") == "infrastructure_error"),
        "cycle_path": str(cycle_path.relative_to(ROOT)),
        "cycle_sha256": _sha256(cycle_path),
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _bootstrap_interval(values: Sequence[float], seed_offset: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_REPLICATES, array.size))
    means = np.mean(array[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def direct() -> dict[str, Any]:
    paired = _paired_episodes()
    rows = []
    episode_paths = []
    for method in METHODS:
        for condition in CONDITIONS:
            for episode in paired[(method, condition)]:
                rows.append(_episode_metrics(method, condition, episode))
                episode_paths.append(Path(str(episode["_source_path"])))

    expected = 2 * (400 + 800)
    if len(rows) != expected:
        raise RuntimeError(f"cycle-metric denominator must be {expected}, got {len(rows)}")
    by_key = {(row["method"], row["pair_key"]): row for row in rows}
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate control-equivalence episode key")

    metrics = (
        "reference_repeat_rate",
        "same_target_repeat_rate",
        "redundant_hold_proxy_rate",
        "longest_same_reference_run",
        "longest_same_target_run",
        "no_plausible_reason_rate",
        "mismatch_reason_rate",
        "recovery_entries",
        "off_violation_recovery_entries",
        "false_interventions",
    )
    summary_rows = []
    paired_rows = []
    seed_offset = 0
    for condition in CONDITIONS:
        for task in (*STRESS4, "ALL"):
            selected = [
                row
                for row in rows
                if row["condition"] == condition and (task == "ALL" or row["task"] == task)
            ]
            for method in METHODS:
                cell = [row for row in selected if row["method"] == method]
                cycles = sum(int(row["cycles"]) for row in cell)
                successes = sum(int(row["success"]) for row in cell)
                successful_cycles = [int(row["cycles"]) for row in cell if row["success"]]
                summary_rows.append({
                    "condition": condition,
                    "task": task,
                    "method": method,
                    "episodes": len(cell),
                    "cycles": cycles,
                    "successes": successes,
                    "success_rate": successes / len(cell),
                    "reference_repeat_cycles": sum(int(row["reference_repeat_cycles"]) for row in cell),
                    "reference_repeat_rate": sum(int(row["reference_repeat_cycles"]) for row in cell) / cycles,
                    "same_target_repeat_cycles": sum(int(row["same_target_repeat_cycles"]) for row in cell),
                    "same_target_repeat_rate": sum(int(row["same_target_repeat_cycles"]) for row in cell) / cycles,
                    "redundant_hold_proxy_cycles": sum(int(row["redundant_hold_proxy_cycles"]) for row in cell),
                    "redundant_hold_proxy_rate": sum(int(row["redundant_hold_proxy_cycles"]) for row in cell) / cycles,
                    "no_plausible_reason_cycles": sum(int(row["no_plausible_reason_cycles"]) for row in cell),
                    "mismatch_reason_cycles": sum(int(row["mismatch_reason_cycles"]) for row in cell),
                    "alarm_cycles": sum(int(row["alarm_cycles"]) for row in cell),
                    "recovery_entries": sum(int(row["recovery_entries"]) for row in cell),
                    "off_violation_recovery_entries": sum(int(row["off_violation_recovery_entries"]) for row in cell),
                    "false_interventions": sum(int(row["false_interventions"]) for row in cell),
                    "successful_episodes": len(successful_cycles),
                    "successful_completion_cycles_mean": _mean(successful_cycles),
                    "successful_completion_cycles_median": _median(successful_cycles),
                })

            full = {row["pair_key"]: row for row in selected if row["method"] == "full"}
            no_equivalence = {
                row["pair_key"]: row
                for row in selected
                if row["method"] == "no_control_equivalence"
            }
            if set(full) != set(no_equivalence):
                raise RuntimeError(f"unpaired control-equivalence cell: {condition}/{task}")
            paired_keys = sorted(full)
            for metric in metrics:
                differences = [
                    float(full[key][metric]) - float(no_equivalence[key][metric])
                    for key in paired_keys
                ]
                low, high = _bootstrap_interval(differences, seed_offset)
                seed_offset += 1
                paired_rows.append({
                    "condition": condition,
                    "task": task,
                    "metric": metric,
                    "paired_episodes": len(paired_keys),
                    "full_mean": _mean([float(full[key][metric]) for key in paired_keys]),
                    "no_control_equivalence_mean": _mean([float(no_equivalence[key][metric]) for key in paired_keys]),
                    "paired_mean_difference_full_minus_ablation": _mean(differences),
                    "paired_bootstrap_95ci_low": low,
                    "paired_bootstrap_95ci_high": high,
                })
            both_success = [
                key
                for key in paired_keys
                if full[key]["success"] and no_equivalence[key]["success"]
            ]
            cycle_differences = [
                int(full[key]["cycles"]) - int(no_equivalence[key]["cycles"])
                for key in both_success
            ]
            low, high = _bootstrap_interval(cycle_differences, seed_offset)
            seed_offset += 1
            paired_rows.append({
                "condition": condition,
                "task": task,
                "metric": "successful_completion_cycles_both_success",
                "paired_episodes": len(both_success),
                "full_mean": _mean([int(full[key]["cycles"]) for key in both_success]),
                "no_control_equivalence_mean": _mean([int(no_equivalence[key]["cycles"]) for key in both_success]),
                "paired_mean_difference_full_minus_ablation": _mean(cycle_differences),
                "paired_bootstrap_95ci_low": low,
                "paired_bootstrap_95ci_high": high,
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    episode_csv = OUTPUT / "control_equivalence_episode_metrics.csv"
    summary_csv = OUTPUT / "appendix_control_equivalence_cycle_metrics.csv"
    paired_csv = OUTPUT / "appendix_control_equivalence_paired_effects.csv"
    _atomic_csv(episode_csv, rows)
    _atomic_csv(summary_csv, summary_rows)
    _atomic_csv(paired_csv, paired_rows)
    definitions = {
        "reference_repeat_cycles": "Cycles after the first in which every available arm retained the same (skill, progress) action reference as the preceding cycle.",
        "same_target_repeat_cycles": "Cycles after the first in which the complete commanded action target matched the preceding target to absolute tolerance 1e-9.",
        "redundant_hold_proxy_cycles": "Conservative operational proxy: reference and target both repeated, the target resolved as reached, every arm remained in TASK mode, no alarm fired, and no re-entry committed. It is not treated as semantic ground truth for every hold.",
        "no_plausible_reason_cycles": "Cycles whose retained monitor trigger reasons explicitly include no_plausible_state.",
        "mismatch_reason_cycles": "Cycles whose retained monitor trigger reasons include a reason containing mismatch.",
        "recovery_entries": "Cycles with at least one arm transitioning from outside VERIFY_LINK/RECOVERY/REENTRY into one of those modes.",
        "off_violation_recovery_entries": "Recovery entries occurring outside the independently audited physical-violation interval; nominal entries are therefore false interventions by construction.",
        "successful_completion_cycles": "Episode control-cycle count, reported only for task-success episodes; the paired contrast additionally restricts to pairs in which both variants succeeded.",
        "success_rate": "Downstream task-success check retained to detect an internal-metric improvement that harms the task outcome.",
    }
    direct_summary = {
        "schema": "essay2608.iclr2027.e5-control-equivalence-cycle-metrics.v1",
        "status": "PASS_DIRECT_LOG_METRICS",
        "design": "paired retained-cycle analysis; no simulator rerun",
        "episodes": len(rows),
        "paired_episode_pairs": len(rows) // 2,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired episode",
        },
        "metric_definitions": definitions,
        "inputs": {
            "run_plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": _sha256(PLAN)},
            "current_freeze": {"path": str(CURRENT_FREEZE.relative_to(ROOT)), "sha256": _sha256(CURRENT_FREEZE)},
            "episode_files": _aggregate_identity(episode_paths),
            "cycle_files": {
                "files": len(rows),
                "aggregate_sha256": hashlib.sha256(
                    "".join(
                        f"{row['cycle_path']}\0{row['cycle_sha256']}\n"
                        for row in sorted(rows, key=lambda value: (value["method"], value["pair_key"]))
                    ).encode("utf-8")
                ).hexdigest(),
            },
        },
        "outputs": {
            episode_csv.name: _sha256(episode_csv),
            summary_csv.name: _sha256(summary_csv),
            paired_csv.name: _sha256(paired_csv),
        },
    }
    _atomic_json(OUTPUT / "CONTROL_EQUIVALENCE_DIRECT.json", direct_summary)
    print(json.dumps(direct_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return direct_summary


def finalize_direct_only() -> dict[str, Any]:
    """Freeze the retained-log analysis without optional policy replay.

    The formal cycle records do not contain the internal accepted-equivalence
    class.  Reconstructing that one diagnostic requires a comparatively
    expensive policy-only replay, so it is deliberately omitted rather than
    inferred from downstream behavior.  All reported metrics in this record
    are read directly from the retained formal logs.
    """

    direct_path = OUTPUT / "CONTROL_EQUIVALENCE_DIRECT.json"
    direct_result = _load_json(direct_path)
    outputs = {
        name: digest
        for name, digest in direct_result["outputs"].items()
        if name in {
            "control_equivalence_episode_metrics.csv",
            "appendix_control_equivalence_cycle_metrics.csv",
            "appendix_control_equivalence_paired_effects.csv",
        }
    }
    final = {
        "schema": "essay2608.iclr2027.e5-control-equivalence-analysis.v1",
        "status": "PASS_DIRECT_LOG_METRICS",
        "design": "paired retained-cycle analysis; no simulator or policy replay",
        "simulator_reruns": 0,
        "policy_replays": 0,
        "episode_pairs": 1200,
        "cycle_metric_episodes": 2400,
        "inputs": {
            "direct_analysis": {
                "path": str(direct_path.relative_to(ROOT)),
                "sha256": _sha256(direct_path),
            }
        },
        "outputs": outputs,
        "metric_definitions": direct_result["metric_definitions"],
        "unreported_internal_metric": {
            "name": "accepted_control_equivalence_merge_count",
            "reason": "not retained in the formal cycle record; optional full policy replay was omitted because the direct cycle metrics already test the mechanism and replay would not change an end-to-end result",
            "no_proxy_or_extrapolation_used": True,
        },
        "claim_boundary": (
            "Success is a downstream non-harm check. The mechanism claim is "
            "limited to the paired cycle-level repeat, long-run, conservative "
            "redundant-hold, monitor-reason, recovery-entry, and completion "
            "metrics directly present in the retained formal logs."
        ),
    }
    _atomic_json(OUTPUT / "CONTROL_EQUIVALENCE_ANALYSIS.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return final


def _payload(record: Mapping[str, Any]) -> dict[str, Any]:
    feature = record["feature"]
    arms = feature["arms"]
    payload: dict[str, Any] = {"task_low_dim_state": feature["task_state"]}
    if "single" in arms:
        payload.update({
            "gripper_pose": arms["single"]["ee_pose_xyzw"],
            "gripper_open": arms["single"]["gripper_open"],
        })
    else:
        for arm in ("left", "right"):
            payload[arm] = {
                "gripper_pose": arms[arm]["ee_pose_xyzw"],
                "gripper_open": arms[arm]["gripper_open"],
            }
    return payload


def replay_task(task: str) -> dict[str, Any]:
    """Replay Full-policy records for one task without starting a simulator."""

    if task not in STRESS4:
        raise ValueError(f"task must be one of {STRESS4}")
    if np.__version__ != "1.26.4":
        raise RuntimeError(
            "internal replay requires the formal RoboTwin NumPy 1.26.4 environment; "
            f"found {np.__version__}"
        )
    from evaluations.iclr2027.runners.shared_episode import (
        BIMANUAL_MODELS,
        CLOSED_LOOP_MODELS,
        REPOSITORY_ROOT,
        SINGLE_MODELS,
    )
    from integrations.rlbench.iclr2027.task_registry import experiment_task
    from integrations.rlbench.rlbench_tsf.policy_server import TSFPolicyServer

    spec = experiment_task(task).spec
    base_models = BIMANUAL_MODELS if spec.bimanual else SINGLE_MODELS
    boundary = (
        REPOSITORY_ROOT
        / "evaluations/iclr2027/artifacts/calibration/normal_task_boundaries/main10/runtime_configs"
        / f"{task}.json"
    )
    server = TSFPolicyServer(
        task,
        CLOSED_LOOP_MODELS,
        base_models,
        feature_profile="full",
        task_spec=spec,
        boundary_config=boundary,
    )
    paired = _paired_episodes()
    output_rows = []
    for condition in CONDITIONS:
        episodes = [value for value in paired[("full", condition)] if value["task"] == task]
        for episode in episodes:
            cycle_path = _cycle_path(FULL_ROOT / condition, episode)
            records = _records(cycle_path)
            exact = True
            failure = None
            max_action_error = 0.0
            exact_cycles = 0
            assessed_arm_cycles = 0
            accepted_arm_cycles = 0
            merged_additional_states = 0
            accepted_low_confidence = 0
            accepted_reference_lag = 0
            try:
                for index, record in enumerate(records):
                    payload = _payload(record)
                    if index == 0:
                        server._reset(payload)
                    response = server._act(payload)
                    recorded_action = np.asarray(record["feature"]["action"], dtype=np.float64)
                    replayed_action = np.asarray(response["action"], dtype=np.float64)
                    error = float(np.max(np.abs(replayed_action - recorded_action)))
                    max_action_error = max(max_action_error, error)
                    if not np.allclose(replayed_action, recorded_action, rtol=0.0, atol=REPLAY_ATOL):
                        exact = False
                        failure = f"action_mismatch_cycle_{index}"
                        break
                    diagnostic = server.policy.diagnostics.records[-1]
                    for arm in sorted(diagnostic["arms"]):
                        execution = diagnostic["arms"][arm].get("execution")
                        assessment = None if execution is None else execution.get("control_equivalence")
                        if not isinstance(assessment, Mapping):
                            continue
                        assessed_arm_cycles += int(bool(assessment.get("evaluated", False)))
                        if bool(assessment.get("accepted", False)):
                            accepted_arm_cycles += 1
                            merged_additional_states += max(
                                0, len(assessment.get("equivalent_states", ())) - 1
                            )
                            reason = assessment.get("reason")
                            accepted_low_confidence += int(
                                reason == "control_equivalent_progress_uncertainty"
                            )
                            accepted_reference_lag += int(
                                reason == "control_equivalent_reference_lag"
                            )
                    resolution = record["execution"]["action_resolution"]
                    server._resolve(
                        {
                            "transaction_id": response["transaction_id"],
                            "primary_action_status": resolution["aggregate"],
                            "primary_action_statuses": resolution["per_arm"],
                            "primary_action_applied": resolution["primary_action_applied"],
                        },
                        commit=True,
                    )
                    exact_cycles += 1
            except Exception as error:
                exact = False
                failure = f"{type(error).__name__}: {error}"
            output_rows.append({
                "condition": condition,
                "task": task,
                "episode_id": episode["episode_id"],
                "seed": int(episode["seed"]),
                "success": int(bool(episode["final_success"])),
                "recorded_cycles": len(records),
                "exact_replay": int(exact and exact_cycles == len(records)),
                "exact_cycles": exact_cycles,
                "max_action_error": max_action_error,
                "failure": failure,
                "assessed_arm_cycles": assessed_arm_cycles,
                "accepted_arm_cycles": accepted_arm_cycles,
                "merged_additional_states": merged_additional_states,
                "accepted_progress_uncertainty": accepted_low_confidence,
                "accepted_reference_lag": accepted_reference_lag,
                "cycle_path": str(cycle_path.relative_to(ROOT)),
                "cycle_sha256": _sha256(cycle_path),
            })
    if len(output_rows) != 300:
        raise RuntimeError(f"expected 300 Full episodes for {task}, got {len(output_rows)}")
    destination = OUTPUT / "replay" / f"{task}.json"
    payload = {
        "schema": "essay2608.iclr2027.e5-control-equivalence-internal-replay-task.v1",
        "status": "COMPLETE",
        "task": task,
        "python": sys.executable,
        "numpy": np.__version__,
        "action_tolerance": REPLAY_ATOL,
        "episodes": len(output_rows),
        "exact_episodes": sum(int(row["exact_replay"]) for row in output_rows),
        "recorded_cycles": sum(int(row["recorded_cycles"]) for row in output_rows),
        "exact_cycles": sum(int(row["exact_cycles"]) for row in output_rows),
        "rows": output_rows,
    }
    _atomic_json(destination, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return payload


def finalize() -> dict[str, Any]:
    direct_path = OUTPUT / "CONTROL_EQUIVALENCE_DIRECT.json"
    direct_result = _load_json(direct_path)
    replay_rows = []
    replay_paths = []
    for task in STRESS4:
        path = OUTPUT / "replay" / f"{task}.json"
        value = _load_json(path)
        if value.get("task") != task or value.get("status") != "COMPLETE":
            raise RuntimeError(f"invalid replay shard: {path}")
        replay_rows.extend(value["rows"])
        replay_paths.append(path)
    if len(replay_rows) != 1200:
        raise RuntimeError(f"internal replay denominator must be 1200, got {len(replay_rows)}")

    summary_rows = []
    for condition in CONDITIONS:
        for task in (*STRESS4, "ALL"):
            rows = [
                value
                for value in replay_rows
                if value["condition"] == condition and (task == "ALL" or value["task"] == task)
            ]
            exact = [value for value in rows if value["exact_replay"]]
            summary_rows.append({
                "condition": condition,
                "task": task,
                "episodes": len(rows),
                "exact_replay_episodes": len(exact),
                "exact_replay_episode_rate": len(exact) / len(rows),
                "recorded_cycles": sum(int(value["recorded_cycles"]) for value in rows),
                "exact_replay_cycles": sum(int(value["exact_cycles"]) for value in rows),
                "assessed_arm_cycles_in_exact_episodes": sum(int(value["assessed_arm_cycles"]) for value in exact),
                "accepted_merge_arm_cycles_in_exact_episodes": sum(int(value["accepted_arm_cycles"]) for value in exact),
                "additional_states_merged_in_exact_episodes": sum(int(value["merged_additional_states"]) for value in exact),
                "accepted_progress_uncertainty_in_exact_episodes": sum(int(value["accepted_progress_uncertainty"]) for value in exact),
                "accepted_reference_lag_in_exact_episodes": sum(int(value["accepted_reference_lag"]) for value in exact),
            })
    replay_csv = OUTPUT / "control_equivalence_internal_replay_episodes.csv"
    summary_csv = OUTPUT / "appendix_control_equivalence_internal_merges.csv"
    _atomic_csv(replay_csv, replay_rows)
    _atomic_csv(summary_csv, summary_rows)

    exact_episodes = sum(int(value["exact_replay"]) for value in replay_rows)
    exact_all = exact_episodes == len(replay_rows)
    final = {
        "schema": "essay2608.iclr2027.e5-control-equivalence-analysis.v1",
        "status": "PASS" if exact_all else "PASS_DIRECT_METRICS_INTERNAL_COUNT_PARTIAL",
        "design": "paired retained-cycle analysis plus simulator-free deterministic policy replay",
        "simulator_reruns": 0,
        "episode_pairs": 1200,
        "cycle_metric_episodes": 2400,
        "internal_replay": {
            "environment": {"python": replay_rows and _load_json(replay_paths[0])["python"], "numpy": "1.26.4"},
            "action_tolerance": REPLAY_ATOL,
            "episodes": len(replay_rows),
            "exact_episodes": exact_episodes,
            "exact_episode_rate": exact_episodes / len(replay_rows),
            "interpretation": (
                "Complete actual accepted-merge count over the retained Full trajectories."
                if exact_all
                else "Accepted-merge counts are reported only for action-exact replay episodes and are not extrapolated to the remaining episodes."
            ),
        },
        "inputs": {
            "direct_analysis": {"path": str(direct_path.relative_to(ROOT)), "sha256": _sha256(direct_path)},
            "replay_shards": _aggregate_identity(replay_paths),
        },
        "outputs": {
            "control_equivalence_episode_metrics.csv": _sha256(OUTPUT / "control_equivalence_episode_metrics.csv"),
            "appendix_control_equivalence_cycle_metrics.csv": _sha256(OUTPUT / "appendix_control_equivalence_cycle_metrics.csv"),
            "appendix_control_equivalence_paired_effects.csv": _sha256(OUTPUT / "appendix_control_equivalence_paired_effects.csv"),
            replay_csv.name: _sha256(replay_csv),
            summary_csv.name: _sha256(summary_csv),
        },
        "metric_definitions": direct_result["metric_definitions"],
        "claim_boundary": "Success is a downstream non-harm check. The mechanism claim is supported by paired cycle-level hold/repeat/mismatch/recovery metrics and action-exact internal merge counts; the redundant-hold measure remains an operational proxy.",
    }
    _atomic_json(OUTPUT / "CONTROL_EQUIVALENCE_ANALYSIS.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("direct")
    subparsers.add_parser("finalize-direct-only")
    replay = subparsers.add_parser("replay-task")
    replay.add_argument("--task", choices=STRESS4, required=True)
    subparsers.add_parser("finalize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "direct":
        direct()
    elif args.command == "finalize-direct-only":
        finalize_direct_only()
    elif args.command == "replay-task":
        replay_task(args.task)
    elif args.command == "finalize":
        finalize()
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
