"""Validate and aggregate the A6 E4 Horizon-3 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
E4_ROOT = EVAL_ROOT / "results" / "controlled" / "e4"
PLAN_PATH = E4_ROOT / "A6_E4_RUN_PLAN.json"
OUTPUT = E4_ROOT / "derived"
FINAL_ACCEPTANCE = E4_ROOT / "A6_E4_FINAL_ACCEPTANCE.json"
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
CONDITIONS = ("nominal", "single_event", "per_stage")


def _validate_current_final_acceptance() -> dict[str, Any] | None:
    """Validate the promoted nested E4 result after the old plan was retired.

    The original independent per-stage protocol and its run plan were moved
    under ``invalidated/`` once the nested paired protocol was accepted.  Keep
    this public analysis entry point useful without silently regenerating the
    retired comparison.
    """

    if PLAN_PATH.is_file() or not FINAL_ACCEPTANCE.is_file():
        return None
    acceptance = json.loads(FINAL_ACCEPTANCE.read_text(encoding="utf-8"))
    if (
        acceptance.get("status") != "PASS"
        or acceptance.get("active_protocol")
        != "paired_single_event_plus_later_interaction_events"
        or acceptance.get("old_independent_per_stage_protocol_retired") is not True
    ):
        raise RuntimeError("E4 final acceptance does not identify the nested protocol")
    outputs = acceptance.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise RuntimeError("E4 final acceptance has no output identity")
    for name, expected in outputs.items():
        path = OUTPUT / str(name)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"promoted E4 output identity mismatch: {path}")
    analysis_entry = acceptance.get("analysis")
    if not isinstance(analysis_entry, Mapping):
        raise RuntimeError("E4 final acceptance has no analysis identity")
    analysis_path = ROOT / str(analysis_entry["path"])
    if not analysis_path.is_file() or _sha256(analysis_path) != analysis_entry["sha256"]:
        raise RuntimeError("promoted E4 analysis identity mismatch")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "PASS":
        raise RuntimeError("promoted E4 analysis is not PASS")
    return {
        **analysis,
        "validated_via_final_acceptance": True,
        "final_acceptance": {
            "path": str(FINAL_ACCEPTANCE.relative_to(ROOT)),
            "sha256": _sha256(FINAL_ACCEPTANCE),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    path = root / "episodes" / (episode_id.replace("/", "__") + ".json")
    if not path.is_file():
        raise RuntimeError(f"missing E4 episode: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    fieldnames: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_cell(
    method: Mapping[str, Any],
    condition: str,
    manifest: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = E4_ROOT / "end_to_end" / str(method["key"]) / condition
    manifest_rows = list(manifest)
    expected_files = {
        str(row["episode_id"]).replace("/", "__") for row in manifest_rows
    }
    actual_files = {path.stem for path in (root / "episodes").glob("*.json")}
    if actual_files != expected_files:
        raise RuntimeError(
            f"E4 {method['key']}/{condition} result set mismatch: "
            f"missing={len(expected_files - actual_files)}, "
            f"extra={len(actual_files - expected_files)}"
        )
    calibration = method.get("calibration")
    if calibration is not None:
        calibration_path = ROOT / str(calibration["path"])
        if (
            not calibration_path.is_file()
            or _sha256(calibration_path) != calibration["sha256"]
        ):
            raise RuntimeError(f"E4 calibration identity changed: {method['key']}")
    rows = []
    for manifest_row in manifest_rows:
        result = _episode(root, str(manifest_row["episode_id"]))
        method_identity = result.get("method_config_identity", {})
        if (
            result.get("episode_id") != manifest_row["episode_id"]
            or result.get("task") != manifest_row["task"]
            or int(result.get("task_level")) != int(manifest_row["task_level"])
            or int(result.get("seed")) != int(manifest_row["seed"])
            or int(result.get("variation")) != int(manifest_row["variation"])
            or result.get("condition") != manifest_row["condition"]
            or method_identity.get("sha256") != method["config_sha256"]
            or (
                calibration is not None
                and method_identity.get("monitor_calibration") != calibration
            )
        ):
            raise RuntimeError(f"E4 result identity mismatch: {manifest_row['episode_id']}")
        infrastructure_error = result.get("termination_reason") == "infrastructure_error"
        if condition == "per_stage" and not infrastructure_error:
            audit = result.get("horizon_audit") or {}
            if (
                int(audit.get("stage_count", -1)) != int(manifest_row["task_level"])
                or int(audit.get("event_opportunity_count", -1)) < 1
                or int(audit.get("event_opportunity_count", -1))
                > int(manifest_row["task_level"])
            ):
                raise RuntimeError(f"invalid E4 repeated-event audit: {manifest_row['episode_id']}")
        rows.append(result)
    return rows


def _event_counts(result: Mapping[str, Any], condition: str) -> tuple[int, int]:
    if condition == "nominal":
        return 0, 0
    if condition == "per_stage":
        if result.get("termination_reason") == "infrastructure_error":
            # The queue-level ITT record is a valid failed episode, but no simulator
            # execution occurred from which event opportunities could be audited.
            return 0, 0
        audit = result["horizon_audit"]
        return int(audit["event_opportunity_count"]), int(
            audit["actual_triggered_event_count"]
        )
    protocol = result.get("fault_protocol") or {}
    return 1, int(protocol.get("triggered") is True)


def generate() -> dict[str, Any]:
    current = _validate_current_final_acceptance()
    if current is not None:
        return current
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    manifests = {
        condition: _rows(ROOT / str(plan["manifests"][condition]["path"]))
        for condition in CONDITIONS
    }
    results = {
        (str(method["key"]), condition): _load_cell(
            method, condition, manifests[condition]
        )
        for method in plan["methods"]
        for condition in CONDITIONS
    }

    nominal_identity = {
        (
            row["task"], int(row["task_level"]), int(row["variation"]),
            int(row["seed"]), row["pair_id"], int(row["horizon"]),
        )
        for row in manifests["nominal"]
    }
    single_identity = {
        (
            row["task"], int(row["task_level"]), int(row["variation"]),
            int(row["seed"]), row["pair_id"], int(row["horizon"]),
        )
        for row in manifests["single_event"]
    }
    if nominal_identity != single_identity:
        raise RuntimeError("E4 nominal and single-event results are not paired")

    success_by_stage = []
    stage_groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for method in plan["methods"]:
        key = str(method["key"])
        for condition in ("nominal", "single_event"):
            for result in results[(key, condition)]:
                stage_groups[
                    (key, condition, TASK_FAMILY[result["task"]], int(result["task_level"]))
                ].append(result)
    for (method, condition, family, stage_count), values in sorted(stage_groups.items()):
        success = sum(bool(value["final_success"]) for value in values)
        success_by_stage.append(
            {
                "method": method,
                "condition": condition,
                "task_family": family,
                "stage_count": stage_count,
                "successes": success,
                "episodes": len(values),
                "success_rate": success / len(values),
                "physically_triggered": sum(
                    _event_counts(value, condition)[1] for value in values
                ),
            }
        )

    # Figure 5 uses the number of disturbance opportunities assigned before
    # execution.  Actual trigger count is a post-treatment quantity: episodes
    # that fail early cannot reach later triggers, so grouping the main result
    # by that count would condition on survival.
    by_scheduled_opportunities = []
    scheduled_groups: dict[
        tuple[str, str, int, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for method in plan["methods"]:
        key = str(method["key"])
        for condition in ("single_event", "per_stage"):
            for result in results[(key, condition)]:
                stage_count = int(result["task_level"])
                scheduled = 1 if condition == "single_event" else stage_count
                scheduled_groups[
                    (
                        key,
                        TASK_FAMILY[result["task"]],
                        stage_count,
                        condition,
                        scheduled,
                    )
                ].append(result)
    for (
        method,
        family,
        stage_count,
        condition,
        scheduled_opportunities,
    ), values in sorted(scheduled_groups.items()):
        success = sum(bool(value["final_success"]) for value in values)
        event_counts = [_event_counts(value, condition) for value in values]
        by_scheduled_opportunities.append(
            {
                "method": method,
                "task_family": family,
                "stage_count": stage_count,
                "condition": condition,
                "scheduled_event_opportunities": scheduled_opportunities,
                "successes": success,
                "episodes": len(values),
                "success_rate": success / len(values),
                "actual_triggered_events": sum(value[1] for value in event_counts),
                "infrastructure_errors": sum(
                    value.get("termination_reason") == "infrastructure_error"
                    for value in values
                ),
            }
        )

    # Retain actual trigger count only as a descriptive appendix diagnostic.
    by_actual_trigger = []
    actual_groups: dict[
        tuple[str, str, int, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for method in plan["methods"]:
        key = str(method["key"])
        for result in results[(key, "per_stage")]:
            _opportunities_reached, triggered = _event_counts(result, "per_stage")
            actual_groups[
                (key, TASK_FAMILY[result["task"]], int(result["task_level"]), triggered)
            ].append(result)
    for (method, family, stage_count, event_count), values in sorted(actual_groups.items()):
        success = sum(bool(value["final_success"]) for value in values)
        by_actual_trigger.append(
            {
                "method": method,
                "task_family": family,
                "stage_count": stage_count,
                "actual_triggered_event_count": event_count,
                "successes": success,
                "episodes": len(values),
                "success_rate": success / len(values),
                "analysis_role": "descriptive_only_survivor_stratified",
            }
        )

    remaining = []
    repair_groups: dict[tuple[str, str, int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for method in plan["methods"]:
        key = str(method["key"])
        for result in results[(key, "per_stage")]:
            if result.get("termination_reason") == "infrastructure_error":
                continue
            records = sorted(
                result["horizon_audit"]["remaining_stages_after_repair"],
                key=lambda value: int(value["restoration_policy_step"]),
            )
            for repair_index, record in enumerate(records, start=1):
                repair_groups[
                    (
                        key,
                        TASK_FAMILY[result["task"]],
                        int(result["task_level"]),
                        repair_index,
                        int(record["remaining_stages_after_repair"]),
                    )
                ].append(record)
    for (method, family, stages, repair_index, stages_remaining), values in sorted(repair_groups.items()):
        completed = sum(bool(value["completed_episode_after_repair"]) for value in values)
        remaining.append(
            {
                "method": method,
                "task_family": family,
                "stage_count": stages,
                "repair_index": repair_index,
                "remaining_stages_after_repair": stages_remaining,
                "completed_episodes": completed,
                "restored_events": len(values),
                "completion_rate_after_repair": completed / len(values),
            }
        )

    appendix = []
    for method in plan["methods"]:
        key = str(method["key"])
        for condition in CONDITIONS:
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for result in results[(key, condition)]:
                grouped[str(result["task"])].append(result)
            for task, values in sorted(grouped.items()):
                success = sum(bool(value["final_success"]) for value in values)
                counts = [_event_counts(value, condition) for value in values]
                appendix.append(
                    {
                        "method": key,
                        "condition": condition,
                        "task": task,
                        "task_family": TASK_FAMILY[task],
                        "stage_count": int(values[0]["task_level"]),
                        "successes": success,
                        "episodes": len(values),
                        "success_rate": success / len(values),
                        "event_opportunities": sum(value[0] for value in counts),
                        "actual_triggered_events": sum(value[1] for value in counts),
                        "infrastructure_errors": sum(
                            value.get("termination_reason") == "infrastructure_error"
                            for value in values
                        ),
                    }
                )

    outputs = {
        "fig5_success_by_stage_count.csv": (
            success_by_stage,
            (
                "method", "condition", "task_family", "stage_count", "successes",
                "episodes", "success_rate", "physically_triggered",
            ),
        ),
        "fig5_success_by_scheduled_opportunities.csv": (
            by_scheduled_opportunities,
            (
                "method", "task_family", "stage_count", "condition",
                "scheduled_event_opportunities", "successes", "episodes",
                "success_rate", "actual_triggered_events",
                "infrastructure_errors",
            ),
        ),
        "appendix_success_by_actual_trigger_count.csv": (
            by_actual_trigger,
            (
                "method", "task_family", "stage_count",
                "actual_triggered_event_count", "successes", "episodes",
                "success_rate", "analysis_role",
            ),
        ),
        "fig5_remaining_completion.csv": (
            remaining,
            (
                "method", "task_family", "stage_count", "repair_index",
                "remaining_stages_after_repair", "completed_episodes",
                "restored_events", "completion_rate_after_repair",
            ),
        ),
        "appendix_horizon_complete.csv": (
            appendix,
            (
                "method", "condition", "task", "task_family", "stage_count",
                "successes", "episodes", "success_rate", "event_opportunities",
                "actual_triggered_events", "infrastructure_errors",
            ),
        ),
    }
    for name, (rows, fieldnames) in outputs.items():
        _atomic_csv(OUTPUT / name, rows, fieldnames=fieldnames)
    summary = {
        "schema": "essay2608.iclr2027.a6-e4-analysis.v1",
        "status": "PASS",
        "episodes": int(plan["expected_new_episodes"]),
        "stage_event_and_opportunity_axes_separate": True,
        "primary_disturbance_axis": "scheduled_event_opportunities",
        "actual_trigger_count_role": "descriptive_appendix_only",
        "outputs": {name: _sha256(OUTPUT / name) for name in outputs},
    }
    _atomic_json(OUTPUT / "E4_ANALYSIS.json", summary)
    return summary


def main() -> int:
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
