"""Freeze and run Full TSF versus an equal-capability simple state controller.

Both systems use the same Stress-4 episodes, observations, five successful
demonstrations, DynaMAC policy, boundary guards, Link/Unlink repair, legal
re-entry, and 400-cycle recovery budget.  The only intended difference is the
state-inference rule selected by the feature profile.
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

from source.policy.tsf.ablation import TSFFeatureProfile


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
RESULT_ROOT = EVAL / "results" / "reviewer_decisive" / "simple_state"
PLAN = RESULT_ROOT / "RUN_PLAN.json"
NOMINAL_SOURCE = EVAL / "manifests" / "main10_nominal.jsonl"
PERTURBED = EVAL / "manifests" / "ablation4.jsonl"
NOMINAL_VIEW = RESULT_ROOT / "manifest_views" / "stress4_nominal.jsonl"
METHODS = {
    "full": EVAL / "configs" / "methods" / "m5_full.json",
    "simple_state": (
        EVAL / "configs" / "methods" / "ablation_simple_state_controller.json"
    ),
}
STRESS4 = (
    "open_drawer",
    "place_cups_3",
    "bimanual_handover_item",
    "bimanual_lift_tray",
)
EXPECTED = {"nominal": 400, "perturbed": 800}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
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
    paths += [
        ROOT / "integrations" / "rlbench" / "rlbench_tsf" / "policy_server.py",
        ROOT / "evaluations" / "iclr2027" / "runners" / "shared_episode.py",
    ]
    entries = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in paths
    ]
    aggregate = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in entries).encode()
    ).hexdigest()
    return {"aggregate_sha256": aggregate, "entries": entries}


def _profile_difference() -> dict[str, Any]:
    full = TSFFeatureProfile.named("full").to_dict()
    simple = TSFFeatureProfile.named("simple_state_controller").to_dict()
    keys = set(full) | set(simple)
    difference = {
        key: {"full": full.get(key), "simple_state": simple.get(key)}
        for key in sorted(keys)
        if key != "name" and full.get(key) != simple.get(key)
    }
    if difference != {
        "state_inference": {
            "full": None,
            "simple_state": "nearest_demo_guard",
        }
    }:
        raise RuntimeError(f"unexpected capability difference: {difference}")
    return difference


def prepare() -> dict[str, Any]:
    nominal_by_task = {task: [] for task in STRESS4}
    for row in _rows(NOMINAL_SOURCE):
        task = str(row["task"])
        if task in nominal_by_task and len(nominal_by_task[task]) < 100:
            nominal_by_task[task].append(row)
    nominal = [row for task in STRESS4 for row in nominal_by_task[task]]
    perturbed = _rows(PERTURBED)
    expected_counts = Counter({task: 100 for task in STRESS4})
    if Counter(str(row["task"]) for row in nominal) != expected_counts:
        raise RuntimeError("nominal view is not Stress-4 x 100")
    if Counter(str(row["task"]) for row in perturbed) != Counter(
        {task: 200 for task in STRESS4}
    ):
        raise RuntimeError("perturbed manifest is not Stress-4 x 200")
    _atomic_jsonl(NOMINAL_VIEW, nominal)

    methods = {}
    for key, path in METHODS.items():
        value = _json(path)
        methods[key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "method_id": value["method_id"],
            "feature_profile": value["feature_profile"],
            "recovery": value["recovery"],
            "runtime": value["runtime"],
        }
    if methods["full"]["recovery"] != methods["simple_state"]["recovery"]:
        raise RuntimeError("recovery capability differs between compared systems")
    if methods["full"]["runtime"] != methods["simple_state"]["runtime"]:
        raise RuntimeError("runtime assets differ between compared systems")

    plan = {
        "schema": "essay2608.iclr2027.decisive-simple-state-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sealed_test": True,
        "comparison": "coupled_belief_vs_nearest_demo_direct_relation_guard",
        "controlled_capability_difference": _profile_difference(),
        "shared": {
            "observations": "identical causal state interface",
            "demonstrations": "episodes 0--4 per task",
            "action_policy": "same frozen DynaMAC assets",
            "boundary_guards": True,
            "relation_repair": "same Link/Unlink executor",
            "legal_reentry": True,
            "recovery_budget_cycles": 400,
        },
        "methods": methods,
        "implementation_identity": _implementation_identity(),
        "manifests": {
            "nominal": {
                "path": str(NOMINAL_VIEW.relative_to(ROOT)),
                "sha256": _sha256(NOMINAL_VIEW),
                "episodes": 400,
                "selection": "first 100 rows per task in immutable source order",
            },
            "perturbed": {
                "path": str(PERTURBED.relative_to(ROOT)),
                "sha256": _sha256(PERTURBED),
                "episodes": 800,
            },
        },
        "primary_endpoint": "paired intention-to-treat task success",
        "secondary_endpoints": [
            "physically-triggered recovery success",
            "trigger-detect-repair-reentry-success path",
        ],
    }
    if PLAN.is_file():
        previous = _json(PLAN)
        old = dict(previous)
        new = dict(plan)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError("existing frozen plan differs from current inputs")
        return previous
    _atomic_json(PLAN, plan)
    return plan


def _assert_frozen(plan: Mapping[str, Any]) -> None:
    if _implementation_identity() != plan["implementation_identity"]:
        raise RuntimeError("implementation changed after plan freeze")
    for method in plan["methods"].values():
        if _sha256(ROOT / str(method["path"])) != method["sha256"]:
            raise RuntimeError("method config changed after plan freeze")
    for manifest in plan["manifests"].values():
        if _sha256(ROOT / str(manifest["path"])) != manifest["sha256"]:
            raise RuntimeError("manifest changed after plan freeze")


def run(*, workers: int, timeout: float) -> None:
    plan = prepare()
    _assert_frozen(plan)
    for method_key, method in plan["methods"].items():
        for condition, manifest in plan["manifests"].items():
            output = RESULT_ROOT / method_key / condition
            command = [
                sys.executable,
                "-m",
                "evaluations.iclr2027.runners.launch",
                "--manifest",
                str(ROOT / str(manifest["path"])),
                "--output-root",
                str(output),
                "--workers",
                str(workers),
                "--episode-timeout-seconds",
                str(timeout),
                "--retry-infrastructure",
                "1",
                "--method",
                str(ROOT / str(method["path"])),
            ]
            completed = subprocess.run(command, cwd=ROOT)
            if completed.returncode not in (0, 2):
                raise SystemExit(completed.returncode)


def status() -> dict[str, Any]:
    cells = {}
    complete = True
    for method in METHODS:
        for condition, expected in EXPECTED.items():
            root = RESULT_ROOT / method / condition / "episodes"
            count = len(list(root.glob("*.json"))) if root.is_dir() else 0
            cells[f"{method}/{condition}"] = {
                "completed": count,
                "expected": expected,
            }
            complete &= count == expected
    return {
        "prepared": PLAN.is_file(),
        "complete": complete,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--episode-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "run":
        run(workers=args.workers, timeout=args.episode_timeout_seconds)
        result = status()
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
