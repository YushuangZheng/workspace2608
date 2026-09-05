"""Build deterministic, mutually exclusive ICLR 2027 episode manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from integrations.rlbench.iclr2027.task_registry import experiment_task_set
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT.parent / "configs" / "shared"
PROTOCOL_PATH = CONFIG_ROOT / "protocol.json"
FAULTS_PATH = CONFIG_ROOT / "faults.json"
MANIFEST_SCHEMA = "essay2608.iclr2027.episode-manifest.v1"
INDEX_SCHEMA = "essay2608.iclr2027.manifest-index.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _row(
    *,
    split: str,
    task: Any,
    index: int,
    seed: int,
    condition: str,
    fault_family: Optional[str] = None,
    severity: Optional[str] = None,
    trigger_stage: Optional[str] = None,
) -> dict[str, Any]:
    public_variations = {
        "close_jar": 20,
        "open_drawer": 3,
        "insert_onto_square_peg": 20,
        "place_cups_3": 1,
        "stack_cups": 20,
        "sweep_to_dustpan": 1,
        "bimanual_handover_item": 5,
        "bimanual_lift_tray": 1,
        "bimanual_sweep_to_dustpan": 1,
        "bimanual_put_bottle_in_fridge": 1,
        "place_cups_1": 1,
        "place_cups_2": 1,
        "remove_cups_1": 1,
        "remove_cups_2": 1,
        "push_buttons_1": 1,
        "push_buttons_2": 1,
        "push_buttons_3": 1,
    }
    if task.task_id not in public_variations:
        raise KeyError(f"variation count is not frozen for {task.task_id}")
    return {
        "schema": MANIFEST_SCHEMA,
        "episode_id": f"{split}/{task.task_id}/{index:04d}",
        "split": split,
        "task": task.task_id,
        "task_level": task.task_level,
        "variation": index % public_variations[task.task_id],
        "seed": seed,
        "condition": condition,
        "fault_family": fault_family,
        "fault_severity": severity,
        "trigger_stage": trigger_stage,
        # The row is method-independent: every compared method consumes this
        # same pair identity, initialization, and physical assignment.
        "pair_id": f"{split}/{task.task_id}/{index:04d}",
        "horizon": 1000,
        "recovery_budget": 400,
    }


def _faults_for(task: Any) -> tuple[str, ...]:
    return tuple(
        value
        for value in task.compatible_faults
        if value != "composed_event"
    )


def _task_rows(
    split: str,
    tasks: Iterable[Any],
    count: int,
    seed_base: int,
    *,
    perturbed: bool,
) -> list[dict[str, Any]]:
    rows = []
    stages = ("early", "middle", "late")
    for task_offset, task in enumerate(tasks):
        faults = _faults_for(task)
        for index in range(count):
            fault = faults[index % len(faults)] if perturbed else None
            rows.append(
                _row(
                    split=split,
                    task=task,
                    index=index,
                    seed=seed_base + task_offset * 100_000 + index,
                    condition="perturbed" if perturbed else "nominal",
                    fault_family=fault,
                    severity="medium" if perturbed else None,
                    trigger_stage=(
                        stages[(index // len(faults)) % len(stages)]
                        if perturbed
                        else None
                    ),
                )
            )
    return rows


def _readonly_view(
    rows: Iterable[Mapping[str, Any]],
    *,
    view_name: str,
    task_ids: set[str],
    fault_family: Optional[str] = None,
) -> list[dict[str, Any]]:
    result = []
    for source in rows:
        if source["task"] not in task_ids:
            continue
        if fault_family is not None and source["fault_family"] != fault_family:
            continue
        result.append(
            {
                **source,
                "source_episode_id": source["episode_id"],
                "episode_id": f"{view_name}/{source['task']}/{len(result):04d}",
                "split": view_name,
                "readonly_view": True,
            }
        )
    return result


def build_all_manifests(root: Path = ROOT) -> dict[str, Any]:
    protocol = _load(PROTOCOL_PATH)
    seeds = protocol["seed_namespaces"]
    main10 = tuple(experiment_task_set("main10"))
    stress4 = tuple(experiment_task_set("stress4"))
    horizon3 = tuple(experiment_task_set("horizon3"))
    native6 = tuple(experiment_task_set("native6"))

    manifests: dict[str, list[dict[str, Any]]] = {}
    manifests["main10_normal_calibration_candidates.jsonl"] = _task_rows(
        "normal_calibration_candidates",
        main10,
        int(protocol["normal_calibration_candidate_limit_per_task"]),
        int(seeds["normal_calibration_candidates"]),
        perturbed=False,
    )
    # The retained 50 successful rows are materialized after rollout.  Preserve
    # an already materialized read-only view so rebuilding unrelated manifests
    # cannot silently erase the frozen calibration selection.
    calibration_path = root / "main10_normal_calibration.jsonl"
    manifests["main10_normal_calibration.jsonl"] = (
        _read_jsonl(calibration_path)
        if calibration_path.is_file() and calibration_path.stat().st_size
        else []
    )
    manifests["main10_failure_train.jsonl"] = _task_rows(
        "failure_train",
        main10,
        int(protocol["failure_train_episodes_per_task"]),
        int(seeds["failure_train"]),
        perturbed=True,
    )
    development = []
    development.extend(
        _task_rows(
            "development_nominal",
            main10,
            int(protocol["development_nominal_per_task"]),
            int(seeds["development"]),
            perturbed=False,
        )
    )
    development.extend(
        _task_rows(
            "development_perturbed",
            main10,
            int(protocol["development_perturbed_per_task"]),
            int(seeds["development"]) + 50_000,
            perturbed=True,
        )
    )
    manifests["main10_development.jsonl"] = development
    nominal = _task_rows(
        "sealed_nominal",
        main10,
        int(protocol["sealed_nominal_per_task"]),
        int(seeds["sealed_nominal"]),
        perturbed=False,
    )
    perturbed = _task_rows(
        "sealed_perturbed",
        main10,
        int(protocol["sealed_perturbed_per_task"]),
        int(seeds["sealed_perturbed"]),
        perturbed=True,
    )
    manifests["main10_nominal.jsonl"] = nominal
    manifests["main10_perturbed.jsonl"] = perturbed

    stress_ids = {task.task_id for task in stress4}
    manifests["stress4_failure_budget_test.jsonl"] = _readonly_view(
        perturbed,
        view_name="stress4_failure_budget_test",
        task_ids=stress_ids,
    )
    lofo_rows = []
    for family in (
        "actuation_delay",
        "missed_interaction",
        "relation_loss",
        "environment_change",
        "coordination_delay",
    ):
        lofo_rows.extend(
            _readonly_view(
                perturbed,
                view_name=f"stress4_lofo_{family}",
                task_ids=stress_ids,
                fault_family=family,
            )
        )
    manifests["stress4_leave_one_family_out.jsonl"] = lofo_rows

    extension_seed = int(seeds["extension"])
    severity_rows = []
    for severity_offset, severity in enumerate(("low", "high", "composed")):
        rows = _task_rows(
            f"stress4_severity_{severity}",
            stress4,
            200,
            extension_seed + severity_offset * 1_000_000,
            perturbed=True,
        )
        for row in rows:
            row["fault_severity"] = severity
            if severity == "composed":
                row["fault_family"] = "composed_event"
        severity_rows.extend(rows)
    manifests["stress4_severity.jsonl"] = severity_rows

    horizon_ids = {task.task_id for task in horizon3}
    manifests["horizon3_single_event.jsonl"] = _task_rows(
        "horizon3_single_event",
        horizon3,
        200,
        extension_seed + 4_000_000,
        perturbed=True,
    )
    per_stage = _task_rows(
        "horizon3_per_stage",
        horizon3,
        200,
        extension_seed + 5_000_000,
        perturbed=True,
    )
    for row in per_stage:
        row["event_schedule"] = "one_eligible_event_per_interaction_stage"
    manifests["horizon3_per_stage.jsonl"] = per_stage
    manifests["ablation4.jsonl"] = _readonly_view(
        perturbed,
        view_name="ablation4",
        task_ids=stress_ids,
    )
    native_ids = {task.task_id for task in native6}
    manifests["native6_nominal.jsonl"] = _readonly_view(
        nominal,
        view_name="native6_nominal",
        task_ids=native_ids,
    )
    manifests["native6_perturbed.jsonl"] = _readonly_view(
        perturbed,
        view_name="native6_perturbed",
        task_ids=native_ids,
    )

    for name, rows in manifests.items():
        _write_jsonl(root / name, rows)
    index_path = root / "MANIFEST_INDEX.json"
    index = {
        "schema": INDEX_SCHEMA,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "faults_sha256": _sha256(FAULTS_PATH),
        "manifests": {
            name: {"rows": len(rows), "sha256": _sha256(root / name)}
            for name, rows in manifests.items()
        },
        "sealed_executed": False,
        "result_based_task_selection": False,
    }
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_all_manifests(root)
    return index


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_all_manifests(root: Path = ROOT) -> dict[str, int]:
    paths = sorted(root.glob("*.jsonl"))
    seen_episode_ids: dict[str, str] = {}
    split_keys: dict[str, set[tuple[Any, ...]]] = {}
    counts = {}
    for path in paths:
        rows = _read_jsonl(path)
        counts[path.name] = len(rows)
        keys = set()
        for row in rows:
            if row.get("schema") != MANIFEST_SCHEMA:
                raise ValueError(f"invalid row schema in {path}")
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"invalid episode id in {path}")
            if episode_id in seen_episode_ids:
                raise ValueError(
                    f"episode id reused by {path.name} and {seen_episode_ids[episode_id]}"
                )
            seen_episode_ids[episode_id] = path.name
            key = (row["task"], row["variation"], row["seed"], row["condition"])
            if key in keys:
                raise ValueError(f"duplicate task/variation/seed/condition in {path}")
            keys.add(key)
            if row["condition"] == "nominal" and any(
                row.get(name) is not None
                for name in ("fault_family", "fault_severity", "trigger_stage")
            ):
                raise ValueError(f"nominal row carries a fault assignment in {path}")
        split_keys[path.name] = keys
    independent = [
        name
        for name in split_keys
        if name
        in {
            "main10_normal_calibration_candidates.jsonl",
            "main10_failure_train.jsonl",
            "main10_development.jsonl",
            "main10_nominal.jsonl",
            "main10_perturbed.jsonl",
            "stress4_severity.jsonl",
            "horizon3_single_event.jsonl",
            "horizon3_per_stage.jsonl",
        }
    ]
    for index, left in enumerate(independent):
        for right in independent[index + 1 :]:
            seed_overlap = {
                (task, seed)
                for task, _variation, seed, _condition in split_keys[left]
            } & {
                (task, seed)
                for task, _variation, seed, _condition in split_keys[right]
            }
            if seed_overlap:
                raise ValueError(f"independent split seeds overlap: {left}, {right}")
    return counts


if __name__ == "__main__":
    print(json.dumps(build_all_manifests(), indent=2, sort_keys=True))
