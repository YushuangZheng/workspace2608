"""Run the E2 state-aware recovery attribution cell required by Figure 3(c).

The middle system in the attribution is not generic Skill-Retry.  It keeps
the frozen task-state monitor and relation repair from Full, but resumes the
pre-recovery task state instead of selecting a legal re-entry state.  The
result is stored in the canonical E5 fine-ablation directory so the later E5
analysis reuses, rather than repeats, these episodes.
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

from evaluations.iclr2027.runners.a6_endpoint_compatibility import (
    assert_frozen_m5_post_e4,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
RESULT_ROOT = EVAL_ROOT / "results" / "controlled" / "e1_e2"
E5_ROOT = EVAL_ROOT / "results" / "controlled" / "e5"
PLAN_PATH = RESULT_ROOT / "A6_E2_RECOVERY_ATTRIBUTION_RUN_PLAN.json"
A5_ACCEPTANCE = RESULT_ROOT / "A5_ACCEPTANCE.json"
M5_POST_E4_FREEZE = EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
NOMINAL = EVAL_ROOT / "manifests" / "main10_nominal.jsonl"
PERTURBED = EVAL_ROOT / "manifests" / "ablation4.jsonl"
METHOD = EVAL_ROOT / "configs" / "methods" / "ablation_retry_same_state.json"
OUTPUT_ROOT = E5_ROOT / "fine" / "retry_same_state"
VIEW_PATH = E5_ROOT / "manifest_views" / "stress4_nominal.jsonl"
STRESS4 = (
    "open_drawer",
    "place_cups_3",
    "bimanual_handover_item",
    "bimanual_lift_tray",
)
WORKERS = 32
TIMEOUT_SECONDS = 900.0


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
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _implementation_identity() -> dict[str, Any]:
    paths = sorted((ROOT / "source" / "policy" / "tsf").rglob("*.py"))
    entries = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in paths
    ]
    aggregate = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in entries).encode()
    ).hexdigest()
    return {"entries": entries, "aggregate_sha256": aggregate}


def prepare() -> dict[str, Any]:
    from source.policy.tsf.ablation import TSFFeatureProfile

    acceptance = _read_json(A5_ACCEPTANCE)
    if not str(acceptance.get("status", "")).startswith("PASS"):
        raise RuntimeError("A5 acceptance is not PASS")
    freeze = assert_frozen_m5_post_e4()
    config = _read_json(METHOD)
    if config.get("method_id") != "ablation_retry_same_state":
        raise RuntimeError("unexpected E2 middle-system method identity")
    if config.get("recovery", {}).get("kind") != "relation_repair_and_same_state_reentry":
        raise RuntimeError("E2 middle system is not relation repair plus same-state resume")
    if config.get("feature_profile") not in TSFFeatureProfile.names():
        raise RuntimeError("E2 middle-system feature profile is not implemented")

    nominal_by_task = {task: [] for task in STRESS4}
    for row in _rows(NOMINAL):
        task = str(row["task"])
        if task in nominal_by_task and len(nominal_by_task[task]) < 100:
            nominal_by_task[task].append(row)
    nominal_rows = [row for task in STRESS4 for row in nominal_by_task[task]]
    nominal_counts = Counter(str(row["task"]) for row in nominal_rows)
    if len(nominal_rows) != 400 or nominal_counts != Counter({task: 100 for task in STRESS4}):
        raise RuntimeError("E2 nominal view must be Stress-4 x 100")
    perturbed_rows = _rows(PERTURBED)
    perturbed_counts = Counter(str(row["task"]) for row in perturbed_rows)
    if len(perturbed_rows) != 800 or perturbed_counts != Counter({task: 200 for task in STRESS4}):
        raise RuntimeError("E2 perturbed view must be Stress-4 x 200")
    _atomic_jsonl(VIEW_PATH, nominal_rows)

    plan = {
        "schema": "essay2608.iclr2027.a6-e2-recovery-attribution-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "A6",
        "experiment": "E2_Figure3c_relation_repair_same_state_resume",
        "paper_system": "TSF-Monitor + Relation Repair + Same-State Resume",
        "not_generic_skill_retry": True,
        "sealed_test": True,
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "one_episode_per_job": True,
        "dynamic_global_queue": True,
        "a5_acceptance": {
            "path": str(A5_ACCEPTANCE.relative_to(ROOT)),
            "sha256": _sha256(A5_ACCEPTANCE),
        },
        "m5_post_e4_freeze": {
            "path": str(M5_POST_E4_FREEZE.relative_to(ROOT)),
            "sha256": _sha256(M5_POST_E4_FREEZE),
            "model_identity": freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
        },
        "method": {
            "path": str(METHOD.relative_to(ROOT)),
            "sha256": _sha256(METHOD),
            "method_id": config["method_id"],
            "feature_profile": config["feature_profile"],
            "recovery_kind": config["recovery"]["kind"],
        },
        "implementation_identity": _implementation_identity(),
        "manifests": {
            "nominal": {
                "path": str(VIEW_PATH.relative_to(ROOT)),
                "sha256": _sha256(VIEW_PATH),
                "episodes": 400,
            },
            "perturbed": {
                "path": str(PERTURBED.relative_to(ROOT)),
                "sha256": _sha256(PERTURBED),
                "episodes": 800,
            },
        },
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "expected_new_episodes": 1200,
        "later_e5_reuse": True,
    }
    if PLAN_PATH.is_file():
        previous = _read_json(PLAN_PATH)
        comparable = dict(plan)
        old_comparable = dict(previous)
        comparable.pop("created_utc", None)
        old_comparable.pop("created_utc", None)
        if comparable != old_comparable:
            raise RuntimeError("existing E2 recovery-attribution plan disagrees with frozen inputs")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def _assert_identity(plan: Mapping[str, Any]) -> None:
    assert_frozen_m5_post_e4()
    if _sha256(ROOT / str(plan["method"]["path"])) != plan["method"]["sha256"]:
        raise RuntimeError("E2 middle-system method config changed after plan freeze")
    if _implementation_identity() != plan["implementation_identity"]:
        raise RuntimeError("TSF implementation changed after E2 plan freeze")


def _run_cell(condition: str, plan: Mapping[str, Any]) -> None:
    manifest = ROOT / str(plan["manifests"][condition]["path"])
    expected = int(plan["manifests"][condition]["episodes"])
    command = [
        sys.executable,
        "-m",
        "evaluations.iclr2027.runners.launch",
        "--manifest",
        str(manifest),
        "--output-root",
        str(OUTPUT_ROOT / condition),
        "--workers",
        str(WORKERS),
        "--episode-timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--retry-infrastructure",
        "1",
        "--method",
        str(METHOD),
    ]
    print(json.dumps({"event": "cell_start", "condition": condition, "episodes": expected}), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    status_path = OUTPUT_ROOT / condition / "QUEUE_STATUS.json"
    queue = _read_json(status_path) if status_path.is_file() else {}
    complete = (
        int(queue.get("completed_episode_count", -1)) == expected
        and int(queue.get("missing_selected_count", -1)) == 0
    )
    if completed.returncode not in (0, 2) or not complete:
        raise SystemExit(completed.returncode or 4)
    print(
        json.dumps(
            {
                "event": "cell_complete",
                "condition": condition,
                "episodes": expected,
                "infrastructure_errors": int(queue.get("infrastructure_errors", 0)),
            }
        ),
        flush=True,
    )


def run() -> None:
    plan = prepare()
    _assert_identity(plan)
    for condition in ("nominal", "perturbed"):
        _run_cell(condition, plan)


def status() -> dict[str, Any]:
    cells = {}
    total = 0
    for condition, expected in (("nominal", 400), ("perturbed", 800)):
        completed = len(list((OUTPUT_ROOT / condition / "episodes").glob("*.json")))
        cells[condition] = {"completed": completed, "expected": expected}
        total += completed
    return {
        "stage": "A6",
        "experiment": "E2_Figure3c_relation_repair_same_state_resume",
        "completed_episodes": total,
        "expected_episodes": 1200,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "run":
        run()
        result = status()
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
