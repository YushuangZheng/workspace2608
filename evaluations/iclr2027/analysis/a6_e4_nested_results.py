"""Validate and aggregate the corrected nested-paired A6 E4 evaluation.

The frozen nominal and exactly-one-event cells are reused.  For the
per-interaction condition, one-stage tasks reuse the exactly-one-event cell
after the real-simulator equivalence gate, while the five multi-stage tasks
come from the corrected nested-paired formal run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
E4 = EVAL / "results" / "controlled" / "e4"
OLD_PLAN_ACTIVE = E4 / "A6_E4_RUN_PLAN.json"
OLD_PLAN_ARCHIVE = E4 / "invalidated" / "independent_per_stage_protocol_20260918" / "A6_E4_RUN_PLAN.json"
NESTED = E4 / "nested_per_stage"
NESTED_PLAN = NESTED / "A6_E4_NESTED_RUN_PLAN.json"
GATE = NESTED / "E4_NESTED_DEVELOPMENT_GATE.json"
OUTPUT = NESTED / "derived"
LEGACY_FREEZE = EVAL / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
CURRENT_FREEZE = (
    EVAL / "results" / "a6_execution" / "M5_PENDING_DIRECT_IDENTITY_FREEZE.json"
)
E4_REFRESH_PROMOTION = (
    EVAL
    / "results"
    / "controlled"
    / "pending_direct_identity_refresh"
    / "promotions"
    / "E4_PROMOTION.json"
)
CONDITIONS = ("nominal", "single_event", "per_stage")
TASK_FAMILY = {
    "place_cups_1": "place_cups",
    "place_cups_2": "place_cups",
    "place_cups_3": "place_cups",
    "remove_cups_1": "remove_cups",
    "remove_cups_2": "remove_cups",
    "push_buttons_1": "push_buttons",
    "push_buttons_2": "push_buttons",
    "push_buttons_3": "push_buttons",
}
ONE_STAGE = {task for task in TASK_FAMILY if task.endswith("_1")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    path = root / "episodes" / (episode_id.replace("/", "__") + ".json")
    if not path.is_file():
        raise RuntimeError(f"missing E4 episode: {path}")
    return _json(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_frozen_cell(
    method: Mapping[str, Any],
    condition: str,
    manifest: list[dict[str, Any]],
    *,
    accepted_config_sha256: set[str] | None = None,
) -> list[dict[str, Any]]:
    root = E4 / "end_to_end" / str(method["key"]) / condition
    expected = {str(row["episode_id"]).replace("/", "__") for row in manifest}
    actual = {path.stem for path in (root / "episodes").glob("*.json")}
    if actual != expected:
        raise RuntimeError(f"frozen E4 cell mismatch: {method['key']}/{condition}")
    calibration = method.get("calibration")
    if calibration is not None:
        path = ROOT / str(calibration["path"])
        if not path.is_file() or _sha256(path) != calibration["sha256"]:
            raise RuntimeError(f"frozen calibration changed: {method['key']}")
    accepted_hashes = accepted_config_sha256 or {str(method["config_sha256"])}
    values = []
    for row in manifest:
        result = _episode(root, str(row["episode_id"]))
        identity = result.get("method_config_identity") or {}
        if (
            result.get("episode_id") != row["episode_id"]
            or result.get("task") != row["task"]
            or int(result.get("task_level", -1)) != int(row["task_level"])
            or int(result.get("seed", -1)) != int(row["seed"])
            or int(result.get("variation", -1)) != int(row["variation"])
            or identity.get("sha256") not in accepted_hashes
            or (calibration is not None and identity.get("monitor_calibration") != calibration)
        ):
            raise RuntimeError(f"frozen E4 result identity mismatch: {row['episode_id']}")
        values.append(result)
    return values


def _load_nested_cell(
    method: Mapping[str, Any], manifest: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    root = NESTED / "end_to_end" / str(method["key"])
    expected = {str(row["episode_id"]).replace("/", "__") for row in manifest}
    actual = {path.stem for path in (root / "episodes").glob("*.json")}
    if actual != expected:
        raise RuntimeError(f"nested E4 result set mismatch: {method['key']}")
    calibration = method.get("calibration")
    if calibration is not None:
        path = ROOT / str(calibration["path"])
        if not path.is_file() or _sha256(path) != calibration["sha256"]:
            raise RuntimeError(f"nested calibration changed: {method['key']}")
    values = []
    for row in manifest:
        result = _episode(root, str(row["episode_id"]))
        identity = result.get("method_config_identity") or {}
        audit = result.get("horizon_audit") or {}
        if (
            result.get("episode_id") != row["episode_id"]
            or result.get("task") != row["task"]
            or int(result.get("task_level", -1)) != int(row["task_level"])
            or int(result.get("seed", -1)) != int(row["seed"])
            or int(result.get("variation", -1)) != int(row["variation"])
            or identity.get("sha256") != method["config_sha256"]
            or (calibration is not None and identity.get("monitor_calibration") != calibration)
            or (
                result.get("termination_reason") != "infrastructure_error"
                and (
                    int(audit.get("stage_count", -1)) != int(row["task_level"])
                    or int(audit.get("event_opportunity_count", -1)) < 1
                    or int(audit.get("event_opportunity_count", -1)) > int(row["task_level"])
                    or (
                        audit.get("paired_first_event_triggered") is not True
                        and (
                            int(audit.get("later_event_opportunities", -1)) != 0
                            or int(audit.get("actual_triggered_event_count", -1)) != 0
                            or int(audit.get("event_opportunity_count", -1)) != 1
                        )
                    )
                )
            )
        ):
            raise RuntimeError(f"nested E4 result identity mismatch: {row['episode_id']}")
        values.append(result)
    return values


def _event_counts(result: Mapping[str, Any], condition: str) -> tuple[int, int]:
    if condition == "nominal":
        return 0, 0
    if condition == "single_event" or result.get("_one_stage_reuse") is True:
        protocol = result.get("fault_protocol") or {}
        return 1, int(protocol.get("triggered") is True)
    if result.get("termination_reason") == "infrastructure_error":
        return 0, 0
    audit = result.get("horizon_audit") or {}
    return int(audit["event_opportunity_count"]), int(audit["actual_triggered_event_count"])


def generate() -> dict[str, Any]:
    old_plan_path = OLD_PLAN_ACTIVE if OLD_PLAN_ACTIVE.is_file() else OLD_PLAN_ARCHIVE
    old_plan = _json(old_plan_path)
    nested_plan = _json(NESTED_PLAN)
    gate = _json(GATE)
    legacy_freeze = _json(LEGACY_FREEZE)
    current_freeze = _json(CURRENT_FREEZE)
    refresh_promotion = _json(E4_REFRESH_PROMOTION)
    if (
        gate.get("status") != "PASS"
        or gate.get("formal_manifest_sha256") != nested_plan["manifest"]["sha256"]
        or gate.get("protocol_files") != nested_plan["protocol_files"]
        or nested_plan["m5_post_e4_freeze"]["sha256"] != _sha256(LEGACY_FREEZE)
        or legacy_freeze.get("status") != "PASS"
        or current_freeze.get("status") != "PASS"
        or (current_freeze.get("supersedes") or {}).get("sha256")
        != _sha256(LEGACY_FREEZE)
        or refresh_promotion.get("status") != "PASS_PROMOTED"
        or refresh_promotion.get("current_m5_freeze_sha256")
        != _sha256(CURRENT_FREEZE)
    ):
        raise RuntimeError("nested E4 gate/freeze identity is not valid")

    old_methods = {str(value["key"]): value for value in old_plan["methods"]}
    nested_methods = {str(value["key"]): value for value in nested_plan["methods"]}
    if set(old_methods) != set(nested_methods):
        raise RuntimeError("old and nested E4 method sets differ")
    if (
        (current_freeze.get("official_config") or {}).get("sha256")
        != nested_methods["m5"]["config_sha256"]
    ):
        raise RuntimeError("current M5 freeze does not match the accepted E4 config")
    old_manifests = {
        condition: _rows(ROOT / old_plan["manifests"][condition]["path"])
        for condition in ("nominal", "single_event")
    }
    nested_manifest = _rows(ROOT / nested_plan["manifest"]["path"])

    results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, method in old_methods.items():
        accepted_hashes = None
        if key == "m5":
            accepted_hashes = {
                str(method["config_sha256"]),
                str(nested_methods[key]["config_sha256"]),
            }
        # The M5 composite cells span the legacy frozen config and its current
        # official successor.  The two immutable freezes and the complete-cell
        # promotion above prove this exact succession.  Accept only those two
        # recorded hashes for both nominal and single-event results; all other
        # methods remain pinned to one config hash.
        results[(key, "nominal")] = _load_frozen_cell(
            method,
            "nominal",
            old_manifests["nominal"],
            accepted_config_sha256=accepted_hashes,
        )
        results[(key, "single_event")] = _load_frozen_cell(
            method,
            "single_event",
            old_manifests["single_event"],
            accepted_config_sha256=accepted_hashes,
        )
    for key, method in nested_methods.items():
        multistage = _load_nested_cell(method, nested_manifest)
        one_stage = []
        for result in results[(key, "single_event")]:
            if str(result["task"]) in ONE_STAGE:
                copied = dict(result)
                copied["_one_stage_reuse"] = True
                one_stage.append(copied)
        combined = one_stage + multistage
        counts = defaultdict(int)
        for value in combined:
            counts[str(value["task"])] += 1
        if len(combined) != 800 or set(counts.values()) != {100} or set(counts) != set(TASK_FAMILY):
            raise RuntimeError(f"composite nested E4 cell is invalid: {key}: {dict(counts)}")
        results[(key, "per_stage")] = combined

    success_by_stage: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for key in old_methods:
        for condition in CONDITIONS:
            for result in results[(key, condition)]:
                groups[(key, condition, TASK_FAMILY[str(result["task"])], int(result["task_level"]))].append(result)
    for (method, condition, family, stage_count), values in sorted(groups.items()):
        successes = sum(bool(value["final_success"]) for value in values)
        success_by_stage.append({
            "method": method,
            "condition": condition,
            "task_family": family,
            "stage_count": stage_count,
            "successes": successes,
            "episodes": len(values),
            "success_rate": successes / len(values),
            "actual_triggered_events": sum(_event_counts(value, condition)[1] for value in values),
        })

    scheduled: list[dict[str, Any]] = []
    groups2: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for key in old_methods:
        for condition in ("single_event", "per_stage"):
            for result in results[(key, condition)]:
                stage_count = int(result["task_level"])
                groups2[(key, TASK_FAMILY[str(result["task"])], stage_count, condition)].append(result)
    for (method, family, stage_count, condition), values in sorted(groups2.items()):
        successes = sum(bool(value["final_success"]) for value in values)
        opened = [_event_counts(value, condition)[0] for value in values]
        triggered = [_event_counts(value, condition)[1] for value in values]
        paired_first_triggered = sum(
            bool((value.get("horizon_audit") or {}).get("paired_first_event_triggered"))
            for value in values
            if condition == "per_stage" and value.get("_one_stage_reuse") is not True
        )
        later_opened = sum(
            int((value.get("horizon_audit") or {}).get("later_event_opportunities") or 0)
            for value in values
            if condition == "per_stage" and value.get("_one_stage_reuse") is not True
        )
        scheduled.append({
            "method": method,
            "task_family": family,
            "stage_count": stage_count,
            "condition": condition,
            # This is the design upper bound.  The strict nested protocol
            # opens a later opportunity only after the paired first event has
            # triggered and a later interaction is actually reached.
            "maximum_event_opportunities": 1 if condition == "single_event" else stage_count,
            "opened_event_opportunities": sum(opened),
            "mean_opened_event_opportunities": sum(opened) / len(values),
            "paired_first_triggered_episodes": paired_first_triggered,
            "later_event_opportunities_opened": later_opened,
            "successes": successes,
            "episodes": len(values),
            "success_rate": successes / len(values),
            "actual_triggered_events": sum(triggered),
            "infrastructure_errors": sum(value.get("termination_reason") == "infrastructure_error" for value in values),
            "nested_paired": condition == "per_stage" and stage_count > 1,
            "one_stage_reused": condition == "per_stage" and stage_count == 1,
        })

    actual: list[dict[str, Any]] = []
    groups3: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for key in old_methods:
        for result in results[(key, "per_stage")]:
            triggered = _event_counts(result, "per_stage")[1]
            groups3[(key, TASK_FAMILY[str(result["task"])], int(result["task_level"]), triggered)].append(result)
    for (method, family, stages, triggered), values in sorted(groups3.items()):
        successes = sum(bool(value["final_success"]) for value in values)
        actual.append({
            "method": method,
            "task_family": family,
            "stage_count": stages,
            "actual_triggered_event_count": triggered,
            "successes": successes,
            "episodes": len(values),
            "success_rate": successes / len(values),
            "analysis_role": "descriptive_only_survivor_stratified",
        })

    remaining: list[dict[str, Any]] = []
    groups4: dict[tuple[str, str, int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for key in old_methods:
        for result in results[(key, "per_stage")]:
            if result.get("_one_stage_reuse") is True or result.get("termination_reason") == "infrastructure_error":
                continue
            records = sorted((result.get("horizon_audit") or {}).get("remaining_stages_after_repair", ()), key=lambda value: int(value["restoration_policy_step"]))
            for index, record in enumerate(records, start=1):
                groups4[(key, TASK_FAMILY[str(result["task"])], int(result["task_level"]), index, int(record["remaining_stages_after_repair"]))].append(record)
    for (method, family, stages, index, stages_remaining), values in sorted(groups4.items()):
        completed = sum(bool(value["completed_episode_after_repair"]) for value in values)
        remaining.append({
            "method": method,
            "task_family": family,
            "stage_count": stages,
            "repair_index": index,
            "remaining_stages_after_repair": stages_remaining,
            "completed_episodes": completed,
            "restored_events": len(values),
            "completion_rate_after_repair": completed / len(values),
        })

    appendix: list[dict[str, Any]] = []
    for key in old_methods:
        for condition in CONDITIONS:
            by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for result in results[(key, condition)]:
                by_task[str(result["task"])].append(result)
            for task, values in sorted(by_task.items()):
                successes = sum(bool(value["final_success"]) for value in values)
                counts = [_event_counts(value, condition) for value in values]
                appendix.append({
                    "method": key,
                    "condition": condition,
                    "task": task,
                    "task_family": TASK_FAMILY[task],
                    "stage_count": int(values[0]["task_level"]),
                    "successes": successes,
                    "episodes": len(values),
                    "success_rate": successes / len(values),
                    "opened_event_opportunities": sum(value[0] for value in counts),
                    "actual_triggered_events": sum(value[1] for value in counts),
                    "infrastructure_errors": sum(value.get("termination_reason") == "infrastructure_error" for value in values),
                    "protocol_source": (
                        "frozen_nominal"
                        if condition == "nominal"
                        else "frozen_single_event"
                        if condition == "single_event"
                        else "single_event_reuse_after_equivalence_gate"
                        if int(values[0]["task_level"]) == 1
                        else "nested_paired_formal"
                    ),
                })

    outputs = {
        "fig5_success_by_stage_count.csv": (success_by_stage, ("method", "condition", "task_family", "stage_count", "successes", "episodes", "success_rate", "actual_triggered_events")),
        "fig5_success_by_scheduled_opportunities.csv": (scheduled, ("method", "task_family", "stage_count", "condition", "maximum_event_opportunities", "opened_event_opportunities", "mean_opened_event_opportunities", "paired_first_triggered_episodes", "later_event_opportunities_opened", "successes", "episodes", "success_rate", "actual_triggered_events", "infrastructure_errors", "nested_paired", "one_stage_reused")),
        "appendix_success_by_actual_trigger_count.csv": (actual, ("method", "task_family", "stage_count", "actual_triggered_event_count", "successes", "episodes", "success_rate", "analysis_role")),
        "fig5_remaining_completion.csv": (remaining, ("method", "task_family", "stage_count", "repair_index", "remaining_stages_after_repair", "completed_episodes", "restored_events", "completion_rate_after_repair")),
        "appendix_horizon_complete.csv": (appendix, ("method", "condition", "task", "task_family", "stage_count", "successes", "episodes", "success_rate", "opened_event_opportunities", "actual_triggered_events", "infrastructure_errors", "protocol_source")),
    }
    for name, (rows, fields) in outputs.items():
        _atomic_csv(OUTPUT / name, rows, fields)
    summary = {
        "schema": "essay2608.iclr2027.a6-e4-nested-analysis.v2",
        "status": "PASS",
        "episodes": 9600,
        "unique_executed_episodes": 8400,
        "analyzed_condition_cells": 9600,
        "new_nested_formal_episodes": int(nested_plan["expected_episodes"]),
        "one_stage_reused_condition_cells": 1200,
        "comparison": "paired_single_event_plus_later_interaction_events",
        "selection_is_outcome_independent": True,
        "stage_event_and_opportunity_axes_separate": True,
        "actual_trigger_count_role": "descriptive_appendix_only",
        "old_plan_path": str(old_plan_path.relative_to(ROOT)),
        "old_plan_sha256": _sha256(old_plan_path),
        "nested_plan_sha256": _sha256(NESTED_PLAN),
        "development_gate_sha256": _sha256(GATE),
        "m5_post_e4_freeze_sha256": _sha256(LEGACY_FREEZE),
        "current_m5_freeze_sha256": _sha256(CURRENT_FREEZE),
        "e4_refresh_promotion_sha256": _sha256(E4_REFRESH_PROMOTION),
        "outputs": {name: _sha256(OUTPUT / name) for name in outputs},
    }
    _atomic_json(OUTPUT / "E4_ANALYSIS.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
