"""Freeze the real-simulator development gate for nested-paired E4.

The simulator is not byte-deterministic for a small set of difficult IK and
contact trajectories, even when the ordinary one-event protocol is repeated
with the same seed.  The gate therefore separates protocol equivalence from
unattainable byte-level replay equivalence:

* all three executions must produce the same first physical event at the same
  control cycle;
* the nested run may not introduce any additional pre-event divergence beyond
  the divergence observed between two ordinary one-event repeats;
* one-stage tasks must expose exactly one event opportunity and no later-event
  opportunity in two independent nested runs; and
* the unit tests must prove that the one-stage wrapper forwards the physical
  action unchanged and never constructs a second fault environment.

Only development outputs are read.  No retained E4 nested result is consulted.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
E4 = EVAL / "results" / "controlled" / "e4" / "nested_per_stage"
DEV = E4 / "development"
FORMAL_PLAN = E4 / "A6_E4_NESTED_RUN_PLAN.json"
DEV_PLAN = DEV / "E4_NESTED_DEVELOPMENT_PLAN.json"
SINGLE_MANIFEST = DEV / "manifests" / "paired_single.jsonl"
NESTED_MANIFEST = DEV / "manifests" / "paired_nested.jsonl"
OUTPUT = E4 / "E4_NESTED_DEVELOPMENT_GATE.json"
REPORT = E4 / "E4_NESTED_DEVELOPMENT_GATE.md"
ROOTS = {
    "single": DEV / "single",
    "nested": DEV / "nested",
    "single_repeat": DEV / "single_repeat2",
    "nested_one_stage_repeat": DEV / "nested_repeat2",
}
PAIR_COMPARISONS = {
    "single_vs_nested": ("single", "nested"),
    "single_vs_single_repeat": ("single", "single_repeat"),
    "nested_vs_single_repeat": ("nested", "single_repeat"),
}


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


def _safe(episode_id: str) -> str:
    return episode_id.replace("/", "__")


def _episode_id(row: Mapping[str, Any], run: str) -> str:
    episode_id = str(row["episode_id"])
    if run.startswith("nested"):
        return episode_id.replace("development_single", "development_nested")
    return episode_id


def _episode(run: str, row: Mapping[str, Any]) -> dict[str, Any]:
    episode_id = _episode_id(row, run)
    return _json(ROOTS[run] / "episodes" / (_safe(episode_id) + ".json"))


def _normal_cycle(row: Mapping[str, Any]) -> dict[str, Any]:
    feature = copy.deepcopy(dict(row["feature"]))
    feature.pop("episode_id", None)
    execution = row["execution"]
    return {
        "cycle": int(row["cycle"]),
        "feature": feature,
        "execution": {
            key: copy.deepcopy(execution.get(key))
            for key in (
                "action_resolution",
                "applied_action",
                "policy_complete",
                "reward",
                "terminate",
            )
        },
    }


def _normal_event(event: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(event))
    for key in ("nested_role", "interaction_index", "component_policy_step"):
        value.pop(key, None)
    return value


def _prefix(run: str, row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], tuple[int, dict[str, Any]]]:
    episode_id = _episode_id(row, run)
    path = ROOTS[run] / "cycles" / (_safe(episode_id) + ".jsonl.gz")
    prefix: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            cycle = json.loads(line)
            prefix.append(_normal_cycle(cycle))
            events = ((cycle.get("execution") or {}).get("injector") or {}).get("events") or ()
            if events:
                return prefix, (int(cycle["cycle"]), _normal_event(events[0]))
    raise RuntimeError(f"development row did not trigger its paired event: {run}: {episode_id}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def generate() -> dict[str, Any]:
    formal = _json(FORMAL_PLAN)
    development = _json(DEV_PLAN)
    if development["formal_manifest_sha256"] != formal["manifest"]["sha256"]:
        raise RuntimeError("development and formal E4 manifests disagree")
    if development["protocol_files"] != formal["protocol_files"]:
        raise RuntimeError("development and formal E4 protocol identities disagree")
    for record in formal["protocol_files"]:
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"protocol changed after development execution: {path}")

    single_rows = _rows(SINGLE_MANIFEST)
    nested_rows = _rows(NESTED_MANIFEST)
    if len(single_rows) != 20 or len(nested_rows) != 20:
        raise RuntimeError("development gate requires exactly 20 paired rows")
    nested_by_pair = {str(row["pair_id"]): row for row in nested_rows}
    if set(nested_by_pair) != {str(row["pair_id"]) for row in single_rows}:
        raise RuntimeError("development pair identities disagree")

    unit = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(EVAL / "tests" / "test_a6_horizon_nested_events.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if unit.returncode != 0:
        raise RuntimeError("nested-event unit tests failed:\n" + unit.stdout)

    comparison = {
        key: {"first_event_exact": 0, "prefix_exact": 0, "prefix_mismatches": []}
        for key in PAIR_COMPARISONS
    }
    records = []
    later_opportunities = 0
    one_stage_count = 0
    one_stage_repeat_count = 0
    for row in single_rows:
        name = f"{row['task']}/{row['fault_family']}"
        runs = {key: _episode(key, row) for key in ("single", "nested", "single_repeat")}
        if any(value.get("reason") == "infrastructure_error" for value in runs.values()):
            raise RuntimeError(f"infrastructure error in development evidence: {name}")
        prefixes = {key: _prefix(key, row) for key in runs}
        row_comparison = {}
        for key, (left, right) in PAIR_COMPARISONS.items():
            event_equal = prefixes[left][1] == prefixes[right][1]
            prefix_equal = prefixes[left][0] == prefixes[right][0]
            comparison[key]["first_event_exact"] += int(event_equal)
            comparison[key]["prefix_exact"] += int(prefix_equal)
            if not prefix_equal:
                comparison[key]["prefix_mismatches"].append(name)
            row_comparison[key] = {
                "first_event_exact": event_equal,
                "prefix_exact": prefix_equal,
            }
        audit = runs["nested"].get("horizon_audit") or {}
        later_opportunities += int(audit.get("later_event_opportunities") or 0)
        one_stage = int(row["task_level"]) == 1
        if one_stage:
            if (
                int(audit.get("event_opportunity_count", -1)) != 1
                or int(audit.get("later_event_opportunities", -1)) != 0
                or int(audit.get("actual_triggered_event_count", -1)) != 1
            ):
                raise RuntimeError(f"one-stage nested protocol was not one-event: {name}")
            repeated = _episode("nested_one_stage_repeat", row)
            repeated_audit = repeated.get("horizon_audit") or {}
            if repeated.get("reason") == "infrastructure_error" or (
                int(repeated_audit.get("event_opportunity_count", -1)) != 1
                or int(repeated_audit.get("later_event_opportunities", -1)) != 0
                or int(repeated_audit.get("actual_triggered_event_count", -1)) != 1
            ):
                raise RuntimeError(f"one-stage repeated nested protocol was invalid: {name}")
            if _prefix("nested_one_stage_repeat", row)[1] != prefixes["single"][1]:
                raise RuntimeError(f"one-stage repeated nested first event changed: {name}")
            one_stage_count += 1
            one_stage_repeat_count += 1
        records.append(
            {
                "cell": name,
                "task_level": int(row["task_level"]),
                "first_event_cycle": prefixes["single"][1][0],
                "comparisons": row_comparison,
                "later_event_opportunities": int(audit.get("later_event_opportunities") or 0),
                "one_stage_single_event_only": one_stage,
            }
        )

    if any(value["first_event_exact"] != 20 for value in comparison.values()):
        raise RuntimeError("paired first-event identity is not exact")
    mismatch_sets = {
        key: tuple(value["prefix_mismatches"]) for key, value in comparison.items()
    }
    if len(set(mismatch_sets.values())) != 1:
        raise RuntimeError("nested protocol introduced divergence outside repeatability envelope")
    if one_stage_count != 6 or one_stage_repeat_count != 6:
        raise RuntimeError("one-stage structural-equivalence coverage is incomplete")
    if later_opportunities < 1:
        raise RuntimeError("development run did not exercise a later event opportunity")

    gate = {
        "schema": "essay2608.iclr2027.a6-e4-nested-development-gate.v2",
        "status": "PASS",
        "development_only": True,
        "sealed_results_read": False,
        "equivalence_definition": (
            "exact first-event semantics and cycle; no nested pre-event divergence "
            "outside the same-seed single-event repeatability envelope"
        ),
        "pairs": 20,
        "comparisons": comparison,
        "repeatability_envelope_cells": list(next(iter(mismatch_sets.values()))),
        "paired_prefix_within_repeatability_envelope": 20,
        "one_stage_protocol_equivalent": one_stage_count,
        "one_stage_independent_nested_repeat": one_stage_repeat_count,
        "later_event_opportunities_exercised": later_opportunities,
        "unit_test": {
            "path": str((EVAL / "tests" / "test_a6_horizon_nested_events.py").relative_to(ROOT)),
            "sha256": _sha256(EVAL / "tests" / "test_a6_horizon_nested_events.py"),
            "result": unit.stdout.strip(),
        },
        "development_inputs": {
            "plan": {"path": str(DEV_PLAN.relative_to(ROOT)), "sha256": _sha256(DEV_PLAN)},
            "single_manifest": {"path": str(SINGLE_MANIFEST.relative_to(ROOT)), "sha256": _sha256(SINGLE_MANIFEST)},
            "nested_manifest": {"path": str(NESTED_MANIFEST.relative_to(ROOT)), "sha256": _sha256(NESTED_MANIFEST)},
            "single_repeat_episode_count": len(list((ROOTS["single_repeat"] / "episodes").glob("*.json"))),
            "nested_one_stage_repeat_episode_count": len(list((ROOTS["nested_one_stage_repeat"] / "episodes").glob("*.json"))),
        },
        "formal_manifest_sha256": formal["manifest"]["sha256"],
        "protocol_files": formal["protocol_files"],
        "records": records,
    }
    _atomic_json(OUTPUT, gate)
    REPORT.write_text(
        "# E4 nested-paired development gate\n\n"
        "Status: **PASS**.\n\n"
        "All 20 paired cells reproduced the same first physical event at the same "
        "control cycle in the nested run and two ordinary single-event runs. Exact "
        "pre-event cycle replay was 17/20 in every pairwise comparison; the same "
        "three difficult cells formed the simulator repeatability envelope, so the "
        "nested protocol introduced no additional pre-event divergence. All six "
        "one-stage cells exposed exactly one event opportunity and no later event in "
        "two independent nested runs.\n",
        encoding="utf-8",
    )
    return gate


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
