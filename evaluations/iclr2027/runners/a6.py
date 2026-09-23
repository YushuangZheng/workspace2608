"""Prepare and run the E3 budget/LOFO shadow evaluation.

Failure-budget and leave-one-family-out experiments measure monitoring quality
on the same frozen M0 trajectories.  They never launch a simulator, modify an
action, or report Skill-Retry task success as evidence of monitor quality.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
RESULT_ROOT = EVAL_ROOT / "results" / "controlled" / "e3"
CONFIG_ROOT = RESULT_ROOT / "method_configs"
VIEW_ROOT = RESULT_ROOT / "manifest_views"
CALIBRATION_ROOT = EVAL_ROOT / "artifacts" / "calibration" / "monitors" / "m4" / "e3"
PLAN_PATH = RESULT_ROOT / "A6_E3_RUN_PLAN.json"

WORKERS = 32
CALIBRATION_WORKERS = 8
TIMEOUT_SECONDS = 900.0
SEEDS = (1103, 2207, 3301)
BUDGETS = (20, 50, 100)
STRESS4 = (
    "open_drawer",
    "place_cups_3",
    "bimanual_handover_item",
    "bimanual_lift_tray",
)
FAMILIES = (
    "actuation_delay",
    "missed_interaction",
    "relation_loss",
    "environment_change",
    "coordination_delay",
)

BASE_METHOD_CONFIG = EVAL_ROOT / "configs" / "methods" / "m4_failure_supervised_runtime.json"
BACKEND_CONFIG = EVAL_ROOT / "configs" / "methods" / "m4_failure_supervised.json"
CALIBRATION_CONFIG = EVAL_ROOT / "configs" / "shared" / "monitor_calibration.json"
NORMAL_CALIBRATION = EVAL_ROOT / "manifests" / "main10_normal_calibration.jsonl"
NORMAL_RESULTS = EVAL_ROOT / "datasets" / "normal_calibration_candidates"
BUDGET_TEST = EVAL_ROOT / "manifests" / "stress4_failure_budget_test.jsonl"
LOFO_TEST = EVAL_ROOT / "manifests" / "stress4_leave_one_family_out.jsonl"
CHECKPOINT_ROOT = EVAL_ROOT / "artifacts" / "checkpoints" / "m4"
INFERENCE_VALIDATION = EVAL_ROOT / "artifacts" / "training" / "m4" / "INFERENCE_VALIDATION.json"
A5_ACCEPTANCE = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_ACCEPTANCE.json"
MAIN10_NOMINAL = EVAL_ROOT / "manifests" / "main10_nominal.jsonl"
M0_ROOT = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "end_to_end" / "m0"
SHADOW_ROOT = RESULT_ROOT / "shadow"


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


def _read_rows(path: Path) -> list[dict[str, Any]]:
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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _method_config_path(*, seed: int, budget: int = 200, family: str | None = None) -> Path:
    stem = f"budget_{budget}" if family is None else f"lofo_{family}"
    return CONFIG_ROOT / stem / f"seed_{seed}.json"


def _calibration_root(*, seed: int, budget: int = 200, family: str | None = None) -> Path:
    stem = f"budget_{budget}" if family is None else f"lofo_{family}"
    return CALIBRATION_ROOT / stem / f"seed_{seed}" / "v1"


def _eligible_tasks(family: str) -> tuple[str, ...]:
    tasks = []
    for task in STRESS4:
        directory = CHECKPOINT_ROOT / task / f"lofo_{family}"
        if all((directory / f"seed_{seed}" / "model.pt").is_file() for seed in SEEDS):
            tasks.append(task)
    return tuple(tasks)


def _write_method_config(*, seed: int, budget: int = 200, family: str | None = None) -> Path:
    value = _read_json(BASE_METHOD_CONFIG)
    monitor = dict(value["monitor"])
    monitor["training_budget"] = budget
    monitor["training_seed"] = seed
    monitor["held_out_family"] = family
    value["monitor"] = monitor
    path = _method_config_path(seed=seed, budget=budget, family=family)
    _atomic_json(path, value)
    return path


def _prepare_views() -> dict[str, Path]:
    normal = _read_rows(NORMAL_CALIBRATION)
    stress_normal = [row for row in normal if row["task"] in STRESS4]
    if len(stress_normal) != 200 or Counter(row["task"] for row in stress_normal) != Counter(
        {task: 50 for task in STRESS4}
    ):
        raise RuntimeError("Stress-4 normal calibration view is not 4 tasks x 50")
    views = {"normal_stress4": VIEW_ROOT / "normal_calibration_stress4.jsonl"}
    _atomic_jsonl(views["normal_stress4"], stress_normal)

    sealed_nominal = [
        row for row in _read_rows(MAIN10_NOMINAL) if row["task"] in STRESS4
    ]
    if len(sealed_nominal) != 800 or Counter(row["task"] for row in sealed_nominal) != Counter(
        {task: 200 for task in STRESS4}
    ):
        raise RuntimeError("Stress-4 sealed nominal view is not 4 tasks x 200")
    budget_rows = _read_rows(BUDGET_TEST)
    shadow_budget_perturbed = []
    for row in budget_rows:
        value = dict(row)
        value["view_episode_id"] = value["episode_id"]
        value["episode_id"] = value["source_episode_id"]
        shadow_budget_perturbed.append(value)
    views["shadow_budget_nominal"] = VIEW_ROOT / "shadow_budget_nominal.jsonl"
    views["shadow_budget_perturbed"] = VIEW_ROOT / "shadow_budget_perturbed.jsonl"
    _atomic_jsonl(views["shadow_budget_nominal"], sealed_nominal)
    _atomic_jsonl(views["shadow_budget_perturbed"], shadow_budget_perturbed)

    lofo_rows = _read_rows(LOFO_TEST)
    for family in FAMILIES:
        eligible = _eligible_tasks(family)
        if not eligible:
            raise RuntimeError(f"no eligible LOFO task for {family}")
        calibration_rows = [row for row in normal if row["task"] in eligible]
        test_rows = [row for row in lofo_rows if row["fault_family"] == family]
        if len(calibration_rows) != 50 * len(eligible):
            raise RuntimeError(f"invalid normal calibration view for {family}")
        if len(test_rows) != 50 * len(eligible):
            raise RuntimeError(f"invalid LOFO test view for {family}")
        calibration_path = VIEW_ROOT / f"normal_calibration_lofo_{family}.jsonl"
        test_path = VIEW_ROOT / f"shadow_lofo_{family}_perturbed.jsonl"
        _atomic_jsonl(calibration_path, calibration_rows)
        source_test_rows = []
        for row in test_rows:
            value = dict(row)
            value["view_episode_id"] = value["episode_id"]
            value["episode_id"] = value["source_episode_id"]
            source_test_rows.append(value)
        _atomic_jsonl(test_path, source_test_rows)
        nominal_rows = []
        counts: Counter[str] = Counter()
        for row in sealed_nominal:
            task = str(row["task"])
            if task not in eligible or counts[task] >= 50:
                continue
            nominal_rows.append(row)
            counts[task] += 1
        if len(nominal_rows) != len(source_test_rows):
            raise RuntimeError(f"LOFO nominal/test imbalance for {family}")
        nominal_path = VIEW_ROOT / f"shadow_lofo_{family}_nominal.jsonl"
        _atomic_jsonl(nominal_path, nominal_rows)
        views[f"normal_{family}"] = calibration_path
        views[f"shadow_{family}_nominal"] = nominal_path
        views[f"shadow_{family}_perturbed"] = test_path
    return views


def _assert_checkpoint_identity(*, seed: int, budget: int = 200, family: str | None = None) -> None:
    tasks = STRESS4 if family is None else _eligible_tasks(family)
    subdirectory = f"budget_{budget}" if family is None else f"lofo_{family}"
    for task in tasks:
        directory = CHECKPOINT_ROOT / task / subdirectory / f"seed_{seed}"
        manifest = _read_json(directory / "checkpoint_manifest.json")
        model = directory / "model.pt"
        if (
            manifest["task"] != task
            or int(manifest["training_seed"]) != seed
            or int(manifest["training_budget"]) != budget
            or manifest.get("held_out_family") != family
            or manifest["checkpoint_sha256"] != _sha256(model)
            or manifest["config_sha256"] != _sha256(BACKEND_CONFIG)
        ):
            raise RuntimeError(f"checkpoint manifest mismatch: {directory}")


def prepare() -> dict[str, Any]:
    validation = _read_json(INFERENCE_VALIDATION)
    if validation.get("status") != "pass" or validation.get("checkpoints") != 114:
        raise RuntimeError("current 114-checkpoint inference validation has not passed")
    a5 = _read_json(A5_ACCEPTANCE)
    if not str(a5.get("status", "")).startswith("PASS"):
        raise RuntimeError("A5 acceptance is not frozen PASS")
    views = _prepare_views()
    configs = []
    for budget in BUDGETS:
        for seed in SEEDS:
            _assert_checkpoint_identity(seed=seed, budget=budget)
            path = _write_method_config(seed=seed, budget=budget)
            configs.append(
                {
                    "kind": "budget",
                    "budget": budget,
                    "seed": seed,
                    "path": _relative(path),
                    "sha256": _sha256(path),
                }
            )
    for family in FAMILIES:
        for seed in SEEDS:
            _assert_checkpoint_identity(seed=seed, family=family)
            path = _write_method_config(seed=seed, family=family)
            configs.append(
                {
                    "kind": "lofo",
                    "family": family,
                    "seed": seed,
                    "eligible_tasks": list(_eligible_tasks(family)),
                    "path": _relative(path),
                    "sha256": _sha256(path),
                }
            )
    manifests = {
        name: {"path": _relative(path), "rows": len(_read_rows(path)), "sha256": _sha256(path)}
        for name, path in views.items()
    }
    manifests["budget_test"] = {
        "path": _relative(BUDGET_TEST),
        "rows": len(_read_rows(BUDGET_TEST)),
        "sha256": _sha256(BUDGET_TEST),
    }
    plan = {
        "schema": "essay2608.iclr2027.a6-e3-shadow-plan.v2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "A6",
        "authorization": "user_confirmed_after_A5_acceptance",
        "sealed_test": True,
        "workers": WORKERS,
        "execution_mode": "offline_no_op_shadow",
        "simulator_launched": False,
        "action_authority": False,
        "a5_acceptance": {"path": _relative(A5_ACCEPTANCE), "sha256": _sha256(A5_ACCEPTANCE)},
        "checkpoint_validation": {
            "path": _relative(INFERENCE_VALIDATION),
            "sha256": _sha256(INFERENCE_VALIDATION),
            "checkpoints": 114,
        },
        "manifests": manifests,
        "method_configs": configs,
        "source_result_roots": {
            condition: _relative(M0_ROOT / condition) for condition in ("nominal", "perturbed")
        },
        "expected_new_physical_episodes": 0,
        "expected_shadow_replays": {
            "failure_budget": 14400,
            "lofo": 4800,
            "total": 19200
        },
    }
    if PLAN_PATH.exists():
        previous = _read_json(PLAN_PATH)
        comparable = dict(plan)
        old_comparable = dict(previous)
        comparable.pop("created_utc", None)
        old_comparable.pop("created_utc", None)
        if comparable != old_comparable:
            raise RuntimeError("existing A6 E3 run plan disagrees with current frozen inputs")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def _calibrate_entry(entry: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    """Calibrate one immutable checkpoint identity in an isolated process."""

    config_path = ROOT / str(entry["path"])
    if entry["kind"] == "budget":
        output = _calibration_root(seed=entry["seed"], budget=entry["budget"])
        manifest = ROOT / plan["manifests"]["normal_stress4"]["path"]
        key = f"budget_{entry['budget']}_seed_{entry['seed']}"
    else:
        output = _calibration_root(seed=entry["seed"], family=entry["family"])
        manifest = ROOT / plan["manifests"][f"normal_{entry['family']}"]["path"]
        key = f"lofo_{entry['family']}_seed_{entry['seed']}"
    artifact_path = output / "calibration.json"
    if artifact_path.is_file():
        artifact = _read_json(artifact_path)
        if artifact.get("method_config_identity", {}).get("sha256") == entry["sha256"]:
            return key
        raise RuntimeError(f"stale calibration artifact exists: {artifact_path}")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluations.iclr2027.calibration.m4",
            "--manifest",
            str(manifest),
            "--results",
            str(NORMAL_RESULTS),
            "--output",
            str(output),
            "--config",
            str(CALIBRATION_CONFIG),
            "--method",
            str(config_path),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"M4 calibration failed: {key}")
    return key


def calibrate() -> None:
    plan = prepare()
    with ThreadPoolExecutor(max_workers=CALIBRATION_WORKERS) as pool:
        futures = [pool.submit(_calibrate_entry, entry, plan) for entry in plan["method_configs"]]
        for completed, future in enumerate(as_completed(futures), start=1):
            key = future.result()
            print(
                json.dumps(
                    {"event": "calibration_complete", "key": key, "completed": completed, "total": len(futures)}
                ),
                flush=True,
            )


def _run_job(entry: Mapping[str, Any]) -> None:
    config_path = ROOT / str(entry["path"])
    if entry["kind"] == "budget":
        key = f"budget_{entry['budget']}_seed_{entry['seed']}"
        calibration = _calibration_root(seed=entry["seed"], budget=entry["budget"]) / "calibration.json"
        manifests = {
            condition: VIEW_ROOT / f"shadow_budget_{condition}.jsonl"
            for condition in ("nominal", "perturbed")
        }
        output_root = SHADOW_ROOT / "failure_budget" / f"budget_{entry['budget']}" / f"seed_{entry['seed']}"
    else:
        key = f"lofo_{entry['family']}_seed_{entry['seed']}"
        calibration = _calibration_root(seed=entry["seed"], family=entry["family"]) / "calibration.json"
        manifests = {
            condition: VIEW_ROOT / f"shadow_lofo_{entry['family']}_{condition}.jsonl"
            for condition in ("nominal", "perturbed")
        }
        output_root = SHADOW_ROOT / "lofo" / entry["family"] / f"seed_{entry['seed']}"
    if not calibration.is_file():
        raise RuntimeError(f"missing A-only calibration: {calibration}")
    for condition, manifest in manifests.items():
        expected = len(_read_rows(manifest))
        output = output_root / condition
        index_path = output / "score_index.json"
        if index_path.is_file():
            index = _read_json(index_path)
            if int(index.get("episodes", -1)) == expected and bool(index.get("action_passthrough_verified")):
                print(json.dumps({"event": "shadow_skip_complete", "key": key, "condition": condition, "episodes": expected}), flush=True)
                continue
        print(json.dumps({"event": "shadow_start", "key": key, "condition": condition, "episodes": expected}), flush=True)
        command = [
            sys.executable,
            "-m",
            "evaluations.iclr2027.runners.shadow_parallel",
            "--manifest",
            str(manifest),
            "--result-root",
            str(M0_ROOT / condition),
            "--output-root",
            str(output),
            "--method",
            str(config_path),
            "--calibration-artifact",
            str(calibration),
            "--condition",
            condition,
            "--workers",
            str(WORKERS),
        ]
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        index = _read_json(index_path)
        if int(index.get("episodes", -1)) != expected or not bool(index.get("action_passthrough_verified")):
            raise RuntimeError(f"incomplete or action-changing shadow output: {key}/{condition}")
        print(json.dumps({"event": "shadow_complete", "key": key, "condition": condition, "episodes": expected}), flush=True)


def run(phase: str) -> None:
    plan = prepare()
    calibrate()
    selected = [entry for entry in plan["method_configs"] if phase == "all" or entry["kind"] == phase]
    for entry in selected:
        _run_job(entry)


def status() -> dict[str, Any]:
    jobs: dict[str, dict[str, Any]] = {}
    completed = 0
    expected_total = 0
    if not PLAN_PATH.is_file():
        return {"stage": "A6", "prepared": False}
    plan = _read_json(PLAN_PATH)
    for entry in plan["method_configs"]:
        if entry["kind"] == "budget":
            key = f"budget_{entry['budget']}_seed_{entry['seed']}"
            base = SHADOW_ROOT / "failure_budget" / f"budget_{entry['budget']}" / f"seed_{entry['seed']}"
        else:
            key = f"lofo_{entry['family']}_seed_{entry['seed']}"
            base = SHADOW_ROOT / "lofo" / entry["family"] / f"seed_{entry['seed']}"
        jobs[key] = {}
        for condition in ("nominal", "perturbed"):
            if entry["kind"] == "budget":
                manifest = VIEW_ROOT / f"shadow_budget_{condition}.jsonl"
            else:
                manifest = VIEW_ROOT / f"shadow_lofo_{entry['family']}_{condition}.jsonl"
            expected = len(_read_rows(manifest)) if manifest.is_file() else 0
            index_path = base / condition / "score_index.json"
            count = int(_read_json(index_path).get("episodes", 0)) if index_path.is_file() else 0
            jobs[key][condition] = {"completed": count, "expected": expected}
            completed += count
            expected_total += expected
    calibrations = len(list(CALIBRATION_ROOT.glob("**/calibration.json")))
    return {
        "stage": "A6",
        "prepared": True,
        "calibrations": calibrations,
        "expected_calibrations": 24,
        "completed_shadow_replays": completed,
        "expected_shadow_replays": expected_total,
        "new_physical_episodes": 0,
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("calibrate")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--phase", choices=("budget", "lofo", "all"), default="all")
    subparsers.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "calibrate":
        calibrate()
        result = status()
    elif args.command == "run":
        run(args.phase)
        result = status()
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
