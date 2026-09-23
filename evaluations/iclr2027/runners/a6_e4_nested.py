"""Prepare and run the corrected nested-paired E4 per-stage panel.

Nominal and exactly-one-event results remain frozen.  One-stage tasks are
definitionally identical to their paired single-event condition and are reused
after the development equivalence gate.  Only the five multi-stage tasks are
executed: 500 episodes for each of M0, M1, M3, and M5 (2,000 total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.audit.horizon_nested_events import NESTED_EVENT_SCHEDULE
from evaluations.iclr2027.runners.a6_endpoint_compatibility import (
    assert_frozen_m5_post_e4,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
E4 = EVAL / "results" / "controlled" / "e4"
OUTPUT = E4 / "nested_per_stage"
SOURCE = E4 / "manifest_views" / "exactly_one_event_100_per_level.jsonl"
MANIFEST = OUTPUT / "manifest_views" / "nested_multistage_100_per_level.jsonl"
PLAN = OUTPUT / "A6_E4_NESTED_RUN_PLAN.json"
DEVELOPMENT_GATE = OUTPUT / "E4_NESTED_DEVELOPMENT_GATE.json"
FREEZE = EVAL / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
OLD_PLAN_ACTIVE = E4 / "A6_E4_RUN_PLAN.json"
OLD_PLAN_ARCHIVE = (
    E4
    / "invalidated"
    / "independent_per_stage_protocol_20260918"
    / "A6_E4_RUN_PLAN.json"
)
TASKS = (
    "place_cups_2",
    "place_cups_3",
    "push_buttons_2",
    "push_buttons_3",
    "remove_cups_2",
)
METHODS = {
    "m0": EVAL / "configs" / "methods" / "m0_dynamac.json",
    "m1": EVAL / "configs" / "methods" / "m1_restart.json",
    "m3": EVAL / "configs" / "methods" / "m3_fail_detect_runtime.json",
    "m5": EVAL / "configs" / "methods" / "m5_full.json",
}
M3_CALIBRATION = (
    EVAL / "artifacts" / "calibration" / "monitors" / "m3" / "horizon_v1" / "calibration.json"
)
WORKERS = 32
TIMEOUT_SECONDS = 900.0


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nested_rows() -> list[dict[str, Any]]:
    selected = [row for row in _rows(SOURCE) if str(row["task"]) in TASKS]
    counts = Counter(str(row["task"]) for row in selected)
    if len(selected) != 500 or counts != Counter({task: 100 for task in TASKS}):
        raise RuntimeError(f"nested E4 population is not five tasks x 100: {counts}")
    nested = []
    paired_fields = (
        "task",
        "task_level",
        "variation",
        "seed",
        "pair_id",
        "horizon",
        "fault_family",
        "fault_severity",
        "trigger_stage",
        "recovery_budget",
    )
    for source in selected:
        if int(source.get("task_level") or 0) < 2:
            raise RuntimeError("one-stage task leaked into nested E4 execution set")
        row = dict(source)
        suffix = str(source["episode_id"]).split("/", 1)[1]
        row.update(
            {
                "episode_id": f"horizon3_nested_per_stage/{suffix}",
                "split": "horizon3_nested_per_stage",
                "event_schedule": NESTED_EVENT_SCHEDULE,
                "paired_single_event_episode_id": source["episode_id"],
            }
        )
        if any(row.get(field) != source.get(field) for field in paired_fields):
            raise RuntimeError("nested manifest changed a paired first-event field")
        nested.append(row)
    _atomic_jsonl(MANIFEST, nested)
    return nested


def prepare() -> dict[str, Any]:
    freeze = assert_frozen_m5_post_e4()
    old_plan_path = OLD_PLAN_ACTIVE if OLD_PLAN_ACTIVE.is_file() else OLD_PLAN_ARCHIVE
    old = _json(old_plan_path)
    old_methods = {str(value["key"]): value for value in old["methods"]}
    nested = _nested_rows()
    methods = []
    for key, path in METHODS.items():
        digest = _sha256(path)
        if key != "m5" and digest != old_methods[key]["config_sha256"]:
            raise RuntimeError(f"{key} differs from the accepted E4 method identity")
        if key == "m5" and digest != freeze["official_config_alias"]["sha256"]:
            raise RuntimeError("M5 differs from the final post-E4 freeze")
        methods.append(
            {
                "key": key,
                "config": str(path.relative_to(ROOT)),
                "config_sha256": digest,
                "calibration": (
                    None
                    if key != "m3"
                    else {
                        "path": str(M3_CALIBRATION.relative_to(ROOT)),
                        "sha256": _sha256(M3_CALIBRATION),
                    }
                ),
            }
        )
    protocol_paths = (
        EVAL / "audit" / "horizon_nested_events.py",
        EVAL / "runners" / "a6_horizon_nested_episode.py",
        EVAL / "runners" / "a6_horizon_nested_launch.py",
        EVAL / "runners" / "a6_e4_nested_development.py",
    )
    plan = {
        "schema": "essay2608.iclr2027.a6-e4-nested-run-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "BLOCKED_ON_DEVELOPMENT_GATE",
        "selection_is_outcome_independent": True,
        "comparison": "paired_single_event_plus_later_interaction_events",
        "paired_prefix_fields_unchanged": True,
        "one_stage_reuse_rule": (
            "reuse paired single-event cell only after real-simulator equivalence gate"
        ),
        "source_single_event": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": _sha256(SOURCE),
        },
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": _sha256(MANIFEST),
            "episodes": len(nested),
            "tasks": dict(Counter(str(row["task"]) for row in nested)),
        },
        "methods": methods,
        "m5_post_e4_freeze": {
            "path": str(FREEZE.relative_to(ROOT)),
            "sha256": _sha256(FREEZE),
            "model_identity": freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
        },
        "protocol_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in protocol_paths
        ],
        "development_gate": str(DEVELOPMENT_GATE.relative_to(ROOT)),
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "automatic_in_job_infrastructure_retries": 1,
        "expected_episodes": len(methods) * len(nested),
    }
    if PLAN.is_file():
        previous = _json(PLAN)
        left, right = dict(previous), dict(plan)
        left.pop("created_utc", None)
        right.pop("created_utc", None)
        if left != right:
            retained = list((OUTPUT / "end_to_end").glob("*/episodes/*.json"))
            retained += list((OUTPUT / "end_to_end").glob("*/cycles/*.jsonl.gz"))
            if retained or DEVELOPMENT_GATE.is_file():
                raise RuntimeError("existing nested E4 plan disagrees with frozen inputs")
            # A failed preflight may create the blocked plan before launching
            # any simulator episode.  In that evidence-free state it is safe
            # to replace the plan atomically after a runner-only correction.
            _atomic_json(PLAN, plan)
            return plan
        return previous
    _atomic_json(PLAN, plan)
    return plan


def _assert_development_gate(plan: Mapping[str, Any]) -> None:
    if not DEVELOPMENT_GATE.is_file():
        raise RuntimeError("nested E4 formal run is blocked on the development gate")
    gate = _json(DEVELOPMENT_GATE)
    if gate.get("status") != "PASS":
        raise RuntimeError("nested E4 development gate did not pass")
    if gate.get("formal_manifest_sha256") != plan["manifest"]["sha256"]:
        raise RuntimeError("nested E4 development gate belongs to another manifest")
    if gate.get("protocol_files") != plan["protocol_files"]:
        raise RuntimeError("nested E4 protocol changed after the development gate")


def run() -> None:
    plan = prepare()
    _assert_development_gate(plan)
    for method in plan["methods"]:
        output = OUTPUT / "end_to_end" / method["key"]
        command = [
            sys.executable,
            "-m",
            "evaluations.iclr2027.runners.a6_horizon_nested_launch",
            "--manifest",
            str(MANIFEST),
            "--output-root",
            str(output),
            "--workers",
            str(WORKERS),
            "--episode-timeout-seconds",
            str(TIMEOUT_SECONDS),
            "--retry-infrastructure",
            "1",
            "--method",
            str(ROOT / method["config"]),
        ]
        if method["calibration"] is not None:
            command.extend(
                ["--calibration-artifact", str(ROOT / method["calibration"]["path"])]
            )
        completed = subprocess.run(command, cwd=ROOT)
        queue = _json(output / "QUEUE_STATUS.json")
        if (
            completed.returncode not in (0, 2)
            or int(queue.get("completed_episode_count", -1)) != 500
            or int(queue.get("missing_selected_count", -1)) != 0
        ):
            raise SystemExit(completed.returncode or 4)


def status() -> dict[str, Any]:
    plan = prepare()
    cells = {}
    total = 0
    for method in plan["methods"]:
        count = len(list((OUTPUT / "end_to_end" / method["key"] / "episodes").glob("*.json")))
        cells[method["key"]] = {"completed": count, "expected": 500}
        total += count
    return {"completed": total, "expected": 2000, "cells": cells}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    args = parser.parse_args(argv)
    if args.command == "run":
        run()
        value = status()
    elif args.command == "prepare":
        value = prepare()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
