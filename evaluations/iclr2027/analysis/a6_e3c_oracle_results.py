"""Validate and summarize the E3-C high-severity oracle-timing diagnostic."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.analysis.rebuild_a5_audit import (
    AUDITOR_PATH,
    reaudit_episode,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
ORACLE_ROOT = (
    EVAL_ROOT
    / "results"
    / "controlled"
    / "e3"
    / "e3_c_oracle_feasibility"
    / "high"
)
STANDARD_ROOT = (
    EVAL_ROOT
    / "results"
    / "controlled"
    / "e3"
    / "e3_c_conditions"
    / "m5"
)
DERIVED = EVAL_ROOT / "results" / "controlled" / "e3" / "derived"
PLAN = ORACLE_ROOT / "A6_E3C_ORACLE_RUN_PLAN.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe(episode_id: str) -> str:
    return str(episode_id).replace("/", "__")


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    path = root / "episodes" / f"{_safe(episode_id)}.json"
    if not path.is_file():
        raise RuntimeError(f"missing oracle-feasibility episode: {path}")
    value = _read_json(path)
    value["_source_episode_path"] = str(path)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty oracle-feasibility table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _validate_oracle_result(
    result: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    method_sha256: str,
    current_audit: Mapping[str, Any],
) -> None:
    episode_id = str(row["episode_id"])
    if (
        result.get("episode_id") != episode_id
        or result.get("task") != row["task"]
        or int(result.get("seed")) != int(row["seed"])
        or int(result.get("variation")) != int(row["variation"])
        or result.get("fault_family") != row["fault_family"]
        or result.get("fault_severity") != "high"
    ):
        raise RuntimeError(f"oracle episode identity mismatch: {episode_id}")
    identity = result.get("method_config_identity") or {}
    if identity.get("sha256") != method_sha256:
        raise RuntimeError(f"oracle method identity mismatch: {episode_id}")
    # Queue-synthesized ITT failures never entered the episode process, so they
    # intentionally have no policy-model/oracle identity.  Validate that they
    # are genuine empty infrastructure records, retain them as failed episodes,
    # and do not mistake the absent child-process metadata for an information
    # boundary violation.
    if result.get("termination_reason") == "infrastructure_error":
        error = result.get("error") or {}
        if (
            result.get("reason") != "infrastructure_error"
            or int(result.get("cycles", -1)) != 0
            or int(result.get("cycle_records", -1)) != 0
            or identity.get("policy_model") is not None
            or error.get("type")
            not in {"EpisodeProcessTimeout", "EpisodeProcessError"}
            or result.get("oracle_timing_diagnostic") is not None
        ):
            raise RuntimeError(f"invalid oracle infrastructure record: {episode_id}")
        episode_path = ORACLE_ROOT / "episodes" / f"{_safe(episode_id)}.json"
        cycle_path = (episode_path.parent / str(result["cycle_file"])).resolve()
        if (
            not cycle_path.is_file()
            or _sha256(cycle_path) != result.get("cycle_file_sha256")
        ):
            raise RuntimeError(f"oracle infrastructure cycle mismatch: {episode_id}")
        with gzip.open(cycle_path, "rt", encoding="utf-8") as stream:
            if next(stream, None) is not None:
                raise RuntimeError(
                    f"oracle infrastructure cycle is not empty: {episode_id}"
                )
        return
    policy_identity = identity.get("policy_model") or {}
    oracle_identity = policy_identity.get("oracle_timing_diagnostic") or {}
    if (
        oracle_identity.get("input_fields") != ["violation_onset_cycle"]
        or oracle_identity.get("fault_identity_visible") is not False
        or oracle_identity.get("repair_target_visible") is not False
        or oracle_identity.get("reentry_state_visible") is not False
        or oracle_identity.get("formal_m5_path_modified") is not False
    ):
        raise RuntimeError(f"oracle information boundary mismatch: {episode_id}")
    diagnostic = result.get("oracle_timing_diagnostic") or {}
    embedded_audit = result.get("audit") or {}
    current_triggered = bool(current_audit.get("physically_triggered"))
    embedded_triggered = bool(embedded_audit.get("physically_triggered"))
    if current_triggered != embedded_triggered:
        raise RuntimeError(
            f"oracle online/offline trigger decision disagrees: {episode_id}"
        )
    # The oracle receives the onset produced online by the public physical
    # auditor in this very rollout.  The lossless offline re-audit reconstructs
    # trigger validity from cumulative injector/cycle evidence; it is the
    # authoritative trigger decision for analysis, but its first-valid cycle
    # is not guaranteed to equal the online observer's onset cycle.  Requiring
    # those two clocks to be numerically identical incorrectly rejects valid
    # oracle runs.  The frozen implementation hash below proves which online
    # auditor generated the embedded onset.
    onset = embedded_audit.get("violation_onset_cycle")
    if current_triggered:
        if onset is None or diagnostic.get("event_forwarded") is not True:
            raise RuntimeError(f"triggered episode omitted oracle onset: {episode_id}")
        if diagnostic.get("input", {}).get("violation_onset_cycle") != onset:
            raise RuntimeError(f"oracle onset disagrees with physical audit: {episode_id}")
    forbidden = {
        "fault_family",
        "target_object",
        "target_objects",
        "repair_target",
        "reentry_state",
    }
    server_status = diagnostic.get("server_status") or {}
    if forbidden.intersection(server_status):
        raise RuntimeError(f"oracle server status leaked semantic fields: {episode_id}")


def main() -> int:
    plan = _read_json(PLAN)
    auditor_key = "evaluations/iclr2027/audit/physical_events.py"
    if (
        (plan.get("implementation_identity") or {}).get(auditor_key)
        != _sha256(AUDITOR_PATH)
    ):
        raise RuntimeError("oracle run did not use the current public physical auditor")
    manifest = _rows(ROOT / str(plan["manifest"]))
    expected = {_safe(row["episode_id"]) for row in manifest}
    actual = {path.stem for path in (ORACLE_ROOT / "episodes").glob("*.json")}
    if actual != expected:
        raise RuntimeError(
            "oracle result set mismatch: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    groups: dict[
        tuple[str, str],
        list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    for row in manifest:
        episode_id = str(row["episode_id"])
        oracle = _episode(ORACLE_ROOT, episode_id)
        standard = _episode(STANDARD_ROOT, episode_id)
        oracle_audit = reaudit_episode(
            "oracle_m5", Path(str(oracle["_source_episode_path"])), oracle
        )
        standard_audit = reaudit_episode(
            "m5", Path(str(standard["_source_episode_path"])), standard
        )
        _validate_oracle_result(
            oracle,
            row,
            method_sha256=str(plan["method_config_sha256"]),
            current_audit=oracle_audit,
        )
        if (
            standard.get("episode_id") != episode_id
            or standard.get("task") != row["task"]
            or standard.get("fault_family") != row["fault_family"]
            or standard.get("fault_severity") != "high"
        ):
            raise RuntimeError(f"standard Full episode mismatch: {episode_id}")
        groups[(str(row["task"]), str(row["fault_family"]))].append(
            (oracle, standard, oracle_audit, standard_audit)
        )
    output = []
    for (task, family), values in sorted(groups.items()):
        if len(values) != 50:
            raise RuntimeError(f"oracle feasibility cell is not 50 episodes: {task}/{family}")
        physically_triggered = sum(
            int(bool(oracle_audit.get("physically_triggered")))
            for _oracle, _standard, oracle_audit, _standard_audit in values
        )
        forwarded = sum(
            int(bool((oracle.get("oracle_timing_diagnostic") or {}).get("event_forwarded")))
            for oracle, _standard, _oracle_audit, _standard_audit in values
        )
        oracle_success = sum(
            int(bool(oracle["final_success"]))
            for oracle, _standard, _oracle_audit, _standard_audit in values
        )
        standard_success = sum(
            int(bool(standard["final_success"]))
            for _oracle, standard, _oracle_audit, _standard_audit in values
        )
        oracle_infrastructure = sum(
            int(oracle.get("termination_reason") == "infrastructure_error")
            for oracle, _standard, _oracle_audit, _standard_audit in values
        )
        standard_infrastructure = sum(
            int(standard.get("termination_reason") == "infrastructure_error")
            for _oracle, standard, _oracle_audit, _standard_audit in values
        )
        triggered_oracle_success = sum(
            int(
                bool(oracle_audit.get("physically_triggered"))
                and bool(oracle["final_success"])
            )
            for oracle, _standard, oracle_audit, _standard_audit in values
        )
        if physically_triggered == 0:
            feasibility = "unassessable_no_physical_trigger"
        elif triggered_oracle_success == 0:
            feasibility = "oracle_infeasible_observed"
        else:
            feasibility = "oracle_feasible_observed"
        output.append(
            {
                "task": task,
                "fault_family": family,
                "severity": "high",
                "scheduled_episodes": len(values),
                "physically_triggered_episodes": physically_triggered,
                "oracle_onset_forwarded_episodes": forwarded,
                "standard_full_successes": standard_success,
                "standard_full_success_rate": standard_success / len(values),
                "standard_full_infrastructure_errors": standard_infrastructure,
                "oracle_timing_full_successes": oracle_success,
                "oracle_timing_full_success_rate": oracle_success / len(values),
                "oracle_timing_infrastructure_errors": oracle_infrastructure,
                "triggered_oracle_successes": triggered_oracle_success,
                "feasibility_marker": feasibility,
            }
        )
    if len(output) != 12:
        raise RuntimeError("oracle feasibility output must contain 12 task-family cells")
    csv_path = DERIVED / "fig4_oracle_feasibility.csv"
    json_path = DERIVED / "E3C_ORACLE_FEASIBILITY.json"
    _atomic_csv(csv_path, output)
    relative_csv = csv_path.name
    summary = {
        "schema": "essay2608.iclr2027.a6-e3c-oracle-analysis.v1",
        "status": "PASS",
        "diagnostic_only": True,
        "formal_method_result": False,
        "oracle_inputs": ["violation_onset_cycle"],
        "episodes": len(manifest),
        "cells": len(output),
        "oracle_infrastructure_errors": sum(
            int(row["oracle_timing_infrastructure_errors"]) for row in output
        ),
        "standard_full_infrastructure_errors": sum(
            int(row["standard_full_infrastructure_errors"]) for row in output
        ),
        "physical_audit": {
            "path": str(AUDITOR_PATH.relative_to(ROOT)),
            "sha256": _sha256(AUDITOR_PATH),
            "source": "reconstructed_from_retained_fault_protocol_and_cycle_evidence",
        },
        "oracle_infeasible_cells": [
            f"{row['task']}/{row['fault_family']}"
            for row in output
            if row["feasibility_marker"] == "oracle_infeasible_observed"
        ],
        "outputs": {relative_csv: _sha256(csv_path)},
    }
    _atomic_json(json_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
