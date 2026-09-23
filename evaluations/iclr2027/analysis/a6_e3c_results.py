"""Validate and aggregate E3-C severity, timing, and composition results."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.analysis.rebuild_a5_audit import (
    AUDITOR_PATH,
    reaudit_episode,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
E3_ROOT = EVAL_ROOT / "results" / "controlled" / "e3"
RESULT_ROOT = E3_ROOT / "e3_c_conditions"
DERIVED = E3_ROOT / "derived"
PLAN = RESULT_ROOT / "A6_E3C_RUN_PLAN.json"
A5_ROOT = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "end_to_end"
MANIFEST = EVAL_ROOT / "manifests" / "stress4_severity.jsonl"
STRESS4_BASE = EVAL_ROOT / "manifests" / "stress4_failure_budget_test.jsonl"
METHODS = {
    "m0": ("DynaMAC", 0, None),
    "m3": ("FAIL-Detect + Retry", 0, None),
    "m4_seed1103": ("Failure-Supervised + Retry", 200, 1103),
    "m4_seed2207": ("Failure-Supervised + Retry", 200, 2207),
    "m4_seed3301": ("Failure-Supervised + Retry", 200, 3301),
    "m5": ("Full method", 0, None),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing E3-C input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe(value: str) -> str:
    return str(value).replace("/", "__")


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    path = root / "episodes" / f"{_safe(episode_id)}.json"
    if not path.is_file():
        raise RuntimeError(f"missing E3-C episode: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_source_episode_path"] = str(path)
    return value


_AUDIT_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _current_audit(method_key: str, result: Mapping[str, Any]) -> dict[str, Any]:
    path = str(result["_source_episode_path"])
    key = (method_key, path)
    if key not in _AUDIT_CACHE:
        _AUDIT_CACHE[key] = reaudit_episode(method_key, Path(path), result)
    return _AUDIT_CACHE[key]


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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


def _a5_key(method_key: str) -> str:
    return method_key


def _validate_result(
    result: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    expected_episode_id: str,
    expected_config_sha256: str | None,
    expected_calibration: Mapping[str, Any] | None = None,
) -> None:
    if (
        result.get("episode_id") != expected_episode_id
        or result.get("task") != row["task"]
        or int(result.get("seed")) != int(row["seed"])
        or int(result.get("variation")) != int(row["variation"])
        or result.get("fault_family") != row["fault_family"]
    ):
        raise RuntimeError(f"E3-C episode identity mismatch: {expected_episode_id}")
    if (
        expected_config_sha256 is not None
        and result.get("method_config_identity", {}).get("sha256")
        != expected_config_sha256
    ):
        raise RuntimeError(f"E3-C method config mismatch: {expected_episode_id}")
    if expected_calibration is not None:
        calibration_path = ROOT / str(expected_calibration["path"])
        if (
            not calibration_path.is_file()
            or _sha256(calibration_path) != expected_calibration["sha256"]
            or result.get("method_config_identity", {}).get("monitor_calibration")
            != dict(expected_calibration)
        ):
            raise RuntimeError(
                f"E3-C monitor calibration mismatch: {expected_episode_id}"
            )


def _new_results(
    method_key: str,
    manifest: list[Mapping[str, Any]],
    config_sha256: str,
    expected_calibration: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    root = RESULT_ROOT / method_key
    expected_files = {_safe(str(row["episode_id"])) for row in manifest}
    actual_files = {path.stem for path in (root / "episodes").glob("*.json")}
    if actual_files != expected_files:
        raise RuntimeError(
            f"E3-C {method_key} result set mismatch: "
            f"missing={len(expected_files - actual_files)}, extra={len(actual_files - expected_files)}"
        )
    output = []
    for row in manifest:
        result = _episode(root, str(row["episode_id"]))
        _validate_result(
            result,
            row,
            expected_episode_id=str(row["episode_id"]),
            expected_config_sha256=config_sha256,
            expected_calibration=expected_calibration,
        )
        output.append(result)
    return output


def _a5_result(method_key: str, row: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row.get("source_episode_id") or row.get("pair_id") or row["episode_id"])
    result = _episode(A5_ROOT / _a5_key(method_key) / "perturbed", source)
    _validate_result(
        result,
        row,
        expected_episode_id=source,
        expected_config_sha256=None,
    )
    return result


def _record(
    *,
    axis: str,
    method_key: str,
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    severity: str,
    trigger_stage: str,
    event_condition: str,
) -> dict[str, Any]:
    method, budget, seed = METHODS[method_key]
    audit = _current_audit(method_key, result)
    # Each aggregate panel varies exactly one experimental axis.  Preserve the
    # original row attributes for traceability, but neutralize the other axes
    # in the grouping columns so, for example, a severity point is not split
    # into the early/middle/late assignments inherited from E1.
    grouped_severity = severity if axis == "severity" else ""
    grouped_stage = trigger_stage if axis == "trigger_stage" else ""
    grouped_event_condition = event_condition if axis == "composition" else ""
    return {
        "axis": axis,
        "method_key": "m4" if method_key.startswith("m4_seed") else method_key,
        "evaluation_instance": method_key,
        "method": method,
        "failure_trajectories_per_task": budget,
        "training_seed": "" if seed is None else seed,
        "task": row["task"],
        "fault_family": row["fault_family"],
        "severity": grouped_severity,
        "trigger_stage": grouped_stage,
        "event_condition": grouped_event_condition,
        "manifest_severity": row.get("fault_severity", ""),
        "manifest_trigger_stage": row.get("trigger_stage", ""),
        "source_episode_id": row.get("source_episode_id", ""),
        "episode_id": result["episode_id"],
        "success": int(bool(result["final_success"])),
        "physically_triggered": int(bool(audit.get("physically_triggered"))),
        "infrastructure_error": int(
            result.get("termination_reason") == "infrastructure_error"
        ),
    }


def _summary_rows(complete: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in complete:
        key = (
            row["axis"],
            row["method_key"],
            row["evaluation_instance"],
            row["method"],
            row["failure_trajectories_per_task"],
            row["training_seed"],
            row["severity"],
            row["trigger_stage"],
            row["event_condition"],
        )
        grouped[key].append(row)
    per_instance = []
    for key, values in grouped.items():
        task_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            task_groups[str(row["task"])].append(row)
        task_rates = [
            sum(int(row["success"]) for row in task_rows) / len(task_rows)
            for task_rows in task_groups.values()
        ]
        per_instance.append(
            {
                "axis": key[0],
                "method_key": key[1],
                "evaluation_instance": key[2],
                "method": key[3],
                "failure_trajectories_per_task": key[4],
                "training_seed": key[5],
                "severity": key[6],
                "trigger_stage": key[7],
                "event_condition": key[8],
                "tasks": len(task_groups),
                "episodes": len(values),
                "task_macro_success": mean(task_rates),
                "physically_triggered": sum(int(row["physically_triggered"]) for row in values),
                "infrastructure_errors": sum(int(row["infrastructure_error"]) for row in values),
            }
        )

    final_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_instance:
        key = (
            row["axis"],
            row["method_key"],
            row["method"],
            row["failure_trajectories_per_task"],
            row["severity"],
            row["trigger_stage"],
            row["event_condition"],
        )
        final_groups[key].append(row)
    output = []
    for key, values in final_groups.items():
        rates = [float(row["task_macro_success"]) for row in values]
        output.append(
            {
                "axis": key[0],
                "method_key": key[1],
                "method": key[2],
                "failure_trajectories_per_task": key[3],
                "severity": key[4],
                "trigger_stage": key[5],
                "event_condition": key[6],
                "evaluation_instances": len(values),
                "tasks_per_instance": values[0]["tasks"],
                "episodes_per_instance": values[0]["episodes"],
                "task_macro_success_mean": mean(rates),
                "task_macro_success_seed_sd": pstdev(rates),
                "physically_triggered_total": sum(int(row["physically_triggered"]) for row in values),
                "infrastructure_errors_total": sum(int(row["infrastructure_errors"]) for row in values),
            }
        )
    return output


def generate() -> dict[str, Any]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    manifest = _rows(MANIFEST)
    if len(manifest) != 2000:
        raise RuntimeError("E3-C manifest must contain 2000 rows")
    method_identities = {str(item["key"]): item for item in plan["methods"]}
    new = {
        key: _new_results(
            key,
            manifest,
            str(method_identities[key]["config_sha256"]),
            method_identities[key].get("calibration"),
        )
        for key in METHODS
    }
    by_method_episode = {
        key: {str(result["episode_id"]): result for result in results}
        for key, results in new.items()
    }

    complete = []
    for method_key in METHODS:
        for row in manifest:
            result = by_method_episode[method_key][str(row["episode_id"])]
            axis = str(row["e3c_axis"])
            complete.append(
                _record(
                    axis=axis,
                    method_key=method_key,
                    row=row,
                    result=result,
                    severity=str(row["fault_severity"]),
                    trigger_stage=str(row["trigger_stage"]),
                    event_condition=("composed" if axis == "composition" else "single"),
                )
            )

        # Medium severity is not rerun.  Reuse exactly the 50 E1 source rows
        # for each family represented by the low/high curve.
        severity_rows = [row for row in manifest if row["e3c_axis"] == "severity"]
        seen_sources = {}
        for row in severity_rows:
            key = (row["task"], row["fault_family"], row["source_episode_id"])
            seen_sources.setdefault(key, row)
        for row in seen_sources.values():
            result = _a5_result(method_key, row)
            complete.append(
                _record(
                    axis="severity",
                    method_key=method_key,
                    row=row,
                    result=result,
                    severity="medium",
                    trigger_stage=str(row["trigger_stage"]),
                    event_condition="single",
                )
            )

        # The single-event comparison for composition is the complete frozen
        # Stress-4 E1 medium set, not a selected subset of successful runs.
        for row in _rows(STRESS4_BASE):
            result = _a5_result(method_key, row)
            complete.append(
                _record(
                    axis="composition",
                    method_key=method_key,
                    row=row,
                    result=result,
                    severity="medium",
                    trigger_stage=str(row["trigger_stage"]),
                    event_condition="single",
                )
            )

    summary_rows = _summary_rows(complete)
    figure = [
        row for row in summary_rows if row["axis"] in {"severity", "composition"}
    ]
    trigger = [row for row in summary_rows if row["axis"] == "trigger_stage"]
    _atomic_csv(DERIVED / "fig4_severity_composition.csv", figure)
    _atomic_csv(DERIVED / "appendix_trigger_stage.csv", trigger)
    _atomic_csv(DERIVED / "appendix_e3c_complete.csv", complete)
    infrastructure = sum(int(row["infrastructure_error"]) for row in complete)
    summary = {
        "schema": "essay2608.iclr2027.a6-e3c-analysis.v1",
        "status": "PASS" if infrastructure == 0 else "PASS_WITH_ITT_INFRASTRUCTURE_FAILURES",
        "new_episodes": int(plan["expected_new_episodes"]),
        "medium_severity_reused_from_e1": True,
        "single_event_composition_reference_reused_from_e1": True,
        "physical_audit": {
            "path": str(AUDITOR_PATH.relative_to(ROOT)),
            "sha256": _sha256(AUDITOR_PATH),
            "source": "reconstructed_from_retained_fault_protocol_and_cycle_evidence",
        },
        "infrastructure_errors_including_reused_rows": infrastructure,
        "outputs": {
            name: _sha256(DERIVED / name)
            for name in (
                "fig4_severity_composition.csv",
                "appendix_trigger_stage.csv",
                "appendix_e3c_complete.csv",
            )
        },
    }
    _atomic_json(DERIVED / "E3C_ANALYSIS.json", summary)
    return summary


def main() -> int:
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
