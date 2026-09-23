"""Run the diagnostic-only oracle-timing feasibility check for E3-C high severity."""

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
from typing import Any, Mapping

from evaluations.iclr2027.runners.a6_endpoint_compatibility import (
    assert_frozen_m5_post_e4,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
SOURCE_MANIFEST = EVAL_ROOT / "manifests" / "stress4_severity.jsonl"
ORACLE_MANIFEST = EVAL_ROOT / "manifests" / "stress4_severity_high_oracle.jsonl"
RESULT_ROOT = (
    EVAL_ROOT
    / "results"
    / "controlled"
    / "e3"
    / "e3_c_oracle_feasibility"
    / "high"
)
PLAN_PATH = RESULT_ROOT / "A6_E3C_ORACLE_RUN_PLAN.json"
A5_PLAN = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_RUN_PLAN.json"
M5_POST_E4_FREEZE = EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
METHOD_CONFIG = EVAL_ROOT / "configs" / "methods" / "m5_full.json"
WORKERS = 32
TIMEOUT_SECONDS = 900.0
EXPECTED_ROWS = 600
IMPLEMENTATION_FILES = (
    "integrations/rlbench/rlbench_tsf/oracle_timing_server.py",
    "evaluations/iclr2027/runners/a6_oracle_timing_episode.py",
    "evaluations/iclr2027/runners/a6_oracle_timing_launch.py",
    "evaluations/iclr2027/runners/a6_e3c_oracle.py",
    "evaluations/iclr2027/audit/physical_events.py",
    "evaluations/iclr2027/runners/shared_episode.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare() -> dict[str, Any]:
    source = _rows(SOURCE_MANIFEST)
    selected = [
        row
        for row in source
        if row.get("e3c_axis") == "severity"
        and row.get("fault_severity") == "high"
    ]
    cells = Counter((str(row["task"]), str(row["fault_family"])) for row in selected)
    if len(selected) != EXPECTED_ROWS or set(cells.values()) != {50} or len(cells) != 12:
        raise RuntimeError(f"oracle high-severity view is not 12 x 50: {cells}")
    if len({str(row["episode_id"]) for row in selected}) != EXPECTED_ROWS:
        raise RuntimeError("oracle high-severity view contains duplicate episode IDs")
    if ORACLE_MANIFEST.is_file():
        if _rows(ORACLE_MANIFEST) != selected:
            raise RuntimeError(
                "existing oracle manifest disagrees with the frozen source view"
            )
    else:
        _atomic_jsonl(ORACLE_MANIFEST, selected)
    a5 = _read_json(A5_PLAN)
    freeze = assert_frozen_m5_post_e4()
    frozen_m5 = next(entry for entry in a5["methods"] if entry["key"] == "m5")
    if _sha256(METHOD_CONFIG) != frozen_m5["config_sha256"]:
        raise RuntimeError("frozen M5 method config changed")
    plan = {
        "schema": "essay2608.iclr2027.a6-e3c-oracle-run-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "A6",
        "diagnostic_only": True,
        "formal_method_result": False,
        "purpose": "mark high-severity cells where oracle event timing plus frozen Full recovery cannot complete",
        "oracle_inputs": ["violation_onset_cycle"],
        "forbidden_oracle_inputs": [
            "fault_family",
            "target_object",
            "repair_target",
            "reentry_state",
        ],
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "manifest": str(ORACLE_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(ORACLE_MANIFEST),
        "manifest_rows": len(selected),
        "cell_counts": {f"{task}/{family}": count for (task, family), count in sorted(cells.items())},
        "method_config": str(METHOD_CONFIG.relative_to(ROOT)),
        "method_config_sha256": _sha256(METHOD_CONFIG),
        "a5_endpoint_identity": a5["endpoint_code_identity"]["aggregate_sha256"],
        "m5_post_e4_freeze": {
            "path": str(M5_POST_E4_FREEZE.relative_to(ROOT)),
            "sha256": _sha256(M5_POST_E4_FREEZE),
            "model_identity": freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
        },
        "episode_module": "evaluations.iclr2027.runners.a6_oracle_timing_episode",
        "implementation_identity": {
            relative: _sha256(ROOT / relative)
            for relative in IMPLEMENTATION_FILES
        },
    }
    if PLAN_PATH.is_file():
        previous = _read_json(PLAN_PATH)
        old = dict(previous)
        new = dict(plan)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError("existing oracle run plan disagrees with frozen inputs")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def run() -> None:
    plan = prepare()
    command = [
        sys.executable,
        "-m",
        "evaluations.iclr2027.runners.a6_oracle_timing_launch",
        "--manifest",
        str(ORACLE_MANIFEST),
        "--output-root",
        str(RESULT_ROOT),
        "--workers",
        str(WORKERS),
        "--episode-timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--method",
        str(METHOD_CONFIG),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    status_path = RESULT_ROOT / "QUEUE_STATUS.json"
    queue_status = _read_json(status_path) if status_path.is_file() else {}
    complete_itt = (
        int(queue_status.get("completed_episode_count", -1))
        == int(plan["manifest_rows"])
        and int(queue_status.get("missing_selected_count", -1)) == 0
    )
    if completed.returncode not in (0, 2) or not complete_itt:
        raise SystemExit(completed.returncode or 4)


def status() -> dict[str, Any]:
    plan = prepare()
    completed = len(list((RESULT_ROOT / "episodes").glob("*.json")))
    return {
        "stage": "A6",
        "phase": "E3-C oracle feasibility diagnostic",
        "completed_episodes": completed,
        "expected_episodes": int(plan["manifest_rows"]),
        "diagnostic_only": True,
    }


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
