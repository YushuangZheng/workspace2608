"""Run the frozen E3-C severity, trigger-stage, and composition cells."""

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
MANIFEST = EVAL_ROOT / "manifests" / "stress4_severity.jsonl"
RESULT_ROOT = EVAL_ROOT / "results" / "controlled" / "e3" / "e3_c_conditions"
PLAN_PATH = RESULT_ROOT / "A6_E3C_RUN_PLAN.json"
A5_PLAN = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_RUN_PLAN.json"
M5_POST_E4_FREEZE = EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
FAULT_CONFIG = EVAL_ROOT / "configs" / "shared" / "faults.json"
WORKERS = 32
TIMEOUT_SECONDS = 900.0

METHODS = (
    ("m0", EVAL_ROOT / "configs" / "methods" / "m0_dynamac.json", None),
    (
        "m3",
        EVAL_ROOT / "configs" / "methods" / "m3_fail_detect_runtime.json",
        EVAL_ROOT / "artifacts" / "calibration" / "monitors" / "m3" / "v1" / "calibration.json",
    ),
    (
        "m4_seed1103",
        EVAL_ROOT / "configs" / "methods" / "m4_failure_supervised_runtime.json",
        EVAL_ROOT / "artifacts" / "calibration" / "monitors" / "m4" / "seed_1103" / "v1" / "calibration.json",
    ),
    (
        "m4_seed2207",
        EVAL_ROOT / "configs" / "methods" / "m4_failure_supervised_seed_2207_runtime.json",
        EVAL_ROOT / "artifacts" / "calibration" / "monitors" / "m4" / "seed_2207" / "v1" / "calibration.json",
    ),
    (
        "m4_seed3301",
        EVAL_ROOT / "configs" / "methods" / "m4_failure_supervised_seed_3301_runtime.json",
        EVAL_ROOT / "artifacts" / "calibration" / "monitors" / "m4" / "seed_3301" / "v1" / "calibration.json",
    ),
    ("m5", EVAL_ROOT / "configs" / "methods" / "m5_full.json", None),
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare() -> dict[str, Any]:
    rows = _rows(MANIFEST)
    axes = Counter(str(row.get("e3c_axis")) for row in rows)
    if len(rows) != 2000 or axes != Counter(
        {"severity": 1200, "trigger_stage": 600, "composition": 200}
    ):
        raise RuntimeError(f"E3-C manifest is not the frozen 2000-row design: {axes}")
    a5 = _read_json(A5_PLAN)
    freeze = assert_frozen_m5_post_e4()
    frozen_methods = {entry["key"]: entry for entry in a5["methods"]}
    identities = []
    for key, config, calibration in METHODS:
        if not config.is_file() or (calibration is not None and not calibration.is_file()):
            raise RuntimeError(f"missing frozen E3-C input for {key}")
        if key in frozen_methods and _sha256(config) != frozen_methods[key]["config_sha256"]:
            raise RuntimeError(f"A5 method config changed for {key}")
        identities.append(
            {
                "key": key,
                "config": str(config.relative_to(ROOT)),
                "config_sha256": _sha256(config),
                "calibration": None
                if calibration is None
                else {
                    "path": str(calibration.relative_to(ROOT)),
                    "sha256": _sha256(calibration),
                },
            }
        )
    plan = {
        "schema": "essay2608.iclr2027.a6-e3c-run-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "A6",
        "sealed_test": True,
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "one_episode_per_job": True,
        "dynamic_global_queue": True,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "manifest_rows": len(rows),
        "condition_counts": dict(axes),
        "fault_config_sha256": _sha256(FAULT_CONFIG),
        "a5_run_plan": {
            "path": str(A5_PLAN.relative_to(ROOT)),
            "sha256": _sha256(A5_PLAN),
            "endpoint_identity": a5["endpoint_code_identity"]["aggregate_sha256"],
        },
        "m5_post_e4_freeze": {
            "path": str(M5_POST_E4_FREEZE.relative_to(ROOT)),
            "sha256": _sha256(M5_POST_E4_FREEZE),
            "model_identity": freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
        },
        "methods": identities,
        "expected_new_episodes": len(rows) * len(identities),
    }
    if PLAN_PATH.is_file():
        previous = _read_json(PLAN_PATH)
        old = dict(previous)
        new = dict(plan)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError("existing E3-C run plan disagrees with frozen inputs")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def run(method_keys: set[str] | None = None) -> None:
    plan = prepare()
    expected = int(plan["manifest_rows"])
    for entry in plan["methods"]:
        if method_keys is not None and str(entry["key"]) not in method_keys:
            continue
        output = RESULT_ROOT / str(entry["key"])
        command = [
            sys.executable,
            "-m",
            "evaluations.iclr2027.runners.launch",
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
            str(ROOT / entry["config"]),
        ]
        if entry["calibration"] is not None:
            command.extend(
                ["--calibration-artifact", str(ROOT / entry["calibration"]["path"])]
            )
        completed = subprocess.run(command, cwd=ROOT)
        status_path = output / "QUEUE_STATUS.json"
        status = _read_json(status_path) if status_path.is_file() else {}
        complete_itt = (
            int(status.get("completed_episode_count", -1)) == expected
            and int(status.get("missing_selected_count", -1)) == 0
        )
        if completed.returncode not in (0, 2) or not complete_itt:
            raise SystemExit(completed.returncode or 4)


def status() -> dict[str, Any]:
    plan = prepare()
    jobs = {}
    completed = 0
    for entry in plan["methods"]:
        output = RESULT_ROOT / str(entry["key"])
        count = len(list((output / "episodes").glob("*.json")))
        jobs[entry["key"]] = {"completed": count, "expected": plan["manifest_rows"]}
        completed += count
    return {
        "stage": "A6",
        "phase": "E3-C",
        "completed_episodes": completed,
        "expected_episodes": plan["expected_new_episodes"],
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--method-key", action="append")
    args = parser.parse_args(argv)
    if args.command == "run":
        requested = None if not args.method_key else set(args.method_key)
        unknown = set() if requested is None else requested.difference(key for key, *_ in METHODS)
        if unknown:
            parser.error("unknown --method-key: " + ", ".join(sorted(unknown)))
        run(requested)
        value = status()
    elif args.command == "prepare":
        value = prepare()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
