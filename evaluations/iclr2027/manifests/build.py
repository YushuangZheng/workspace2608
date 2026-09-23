"""Build deterministic, mutually exclusive ICLR 2027 episode manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from integrations.rlbench.iclr2027.task_registry import experiment_task_set
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from evaluations.iclr2027.manifests.native6 import (
    build_development_rows as build_native6_development_rows,
    build_nominal_view as build_native6_nominal_view,
    build_perturbed_view as build_native6_perturbed_view,
)

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


def _apply_seed_replacements(
    rows: Iterable[Mapping[str, Any]],
    replacements: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply protocol-recorded infrastructure replacements without changing row order."""

    result = [dict(row) for row in rows]
    by_episode = {row["episode_id"]: row for row in result}
    for episode_id, spec in replacements.items():
        if episode_id not in by_episode:
            raise KeyError(f"seed replacement refers to unknown episode: {episode_id}")
        row = by_episode[episode_id]
        original_seed = int(spec["original_seed"])
        replacement_seed = int(spec["replacement_seed"])
        if int(row["seed"]) != original_seed:
            raise ValueError(
                f"seed replacement source mismatch for {episode_id}: "
                f"{row['seed']} != {original_seed}"
            )
        for field in ("fault_family", "trigger_stage"):
            expected = spec.get(field)
            if expected is not None and row.get(field) != expected:
                raise ValueError(
                    f"seed replacement {field} mismatch for {episode_id}: "
                    f"{row.get(field)} != {expected}"
                )
        row["seed"] = replacement_seed
    return result


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


def _stress4_severity_rows(
    tasks: Iterable[Any],
    main_perturbed: Iterable[Mapping[str, Any]],
    extension_seed: int,
) -> list[dict[str, Any]]:
    """Build the paired E3-C severity, stage, and composition conditions.

    Low/high severity and early/middle/late timing deliberately reuse the
    corresponding E1 initialization.  They change only the physical condition,
    so they are paired conditions rather than statistically independent splits.
    Composed events use the disjoint extension seed namespace.
    """

    source_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in main_perturbed:
        key = (str(source["task"]), str(source["fault_family"]))
        source_rows.setdefault(key, []).append(dict(source))
    for rows in source_rows.values():
        rows.sort(key=lambda row: str(row["episode_id"]))

    result: list[dict[str, Any]] = []
    for task_offset, task in enumerate(tasks):
        task_id = str(task.task_id)
        third_family = (
            "coordination_delay"
            if task_id.startswith("bimanual_")
            else "environment_change"
        )
        continuous_families = (
            "actuation_delay",
            "relation_loss",
            third_family,
        )

        # Severity changes are paired to the 50 E1 rows of the same task and
        # family, preserving variation, seed, and trigger stage.
        for family in continuous_families:
            sources = source_rows.get((task_id, family), [])
            if len(sources) != 50:
                raise ValueError(
                    f"E3-C requires exactly 50 E1 sources for {task_id}/{family}; "
                    f"found {len(sources)}"
                )
            for severity in ("low", "high"):
                for index, source in enumerate(sources):
                    episode_id = (
                        f"stress4_severity/{task_id}/{family}/{severity}/{index:04d}"
                    )
                    result.append(
                        {
                            **source,
                            "episode_id": episode_id,
                            "pair_id": episode_id,
                            "split": "stress4_severity",
                            "source_episode_id": source["episode_id"],
                            "fault_severity": severity,
                            "e3c_axis": "severity",
                        }
                    )

        # One common, physically eligible family isolates trigger timing.  All
        # three stages are new paired executions because E1 interleaves stages
        # within each family and therefore cannot supply 50 middle-stage rows.
        stage_sources = source_rows.get((task_id, "actuation_delay"), [])
        if len(stage_sources) != 50:
            raise ValueError(
                f"E3-C requires exactly 50 E1 actuation sources for {task_id}; "
                f"found {len(stage_sources)}"
            )
        for stage in ("early", "middle", "late"):
            for index, source in enumerate(stage_sources):
                episode_id = (
                    f"stress4_trigger_stage/{task_id}/actuation_delay/"
                    f"{stage}/{index:04d}"
                )
                result.append(
                    {
                        **source,
                        "episode_id": episode_id,
                        "pair_id": episode_id,
                        "split": "stress4_trigger_stage",
                        "source_episode_id": source["episode_id"],
                        "fault_severity": "medium",
                        "trigger_stage": stage,
                        "e3c_axis": "trigger_stage",
                    }
                )

        # Composed events are a genuinely new initialization split.
        for index in range(50):
            row = _row(
                split="stress4_composed",
                task=task,
                index=index,
                seed=extension_seed + task_offset * 100_000 + index,
                condition="perturbed",
                fault_family="composed_event",
                severity="composed",
                trigger_stage="early",
            )
            row["e3c_axis"] = "composition"
            row["event_schedule"] = "actuation_delay_early_then_relation_loss_late"
            result.append(row)
    return result


def build_all_manifests(root: Path = ROOT) -> dict[str, Any]:
    protocol = _load(PROTOCOL_PATH)
    seeds = protocol["seed_namespaces"]
    main10 = tuple(experiment_task_set("main10"))
    stress4 = tuple(experiment_task_set("stress4"))
    horizon3 = tuple(experiment_task_set("horizon3"))

    manifests: dict[str, list[dict[str, Any]]] = {}
    manifests["main10_normal_calibration_candidates.jsonl"] = _task_rows(
        "normal_calibration_candidates",
        main10,
        int(protocol["normal_calibration_candidate_limit_per_task"]),
        int(seeds["normal_calibration_candidates"]),
        perturbed=False,
    )
    calibration_extensions: dict[str, list[dict[str, Any]]] = {}
    task_offsets = {task.task_id: offset for offset, task in enumerate(main10)}
    task_by_id = {task.task_id: task for task in main10}
    base_candidate_limit = int(protocol["normal_calibration_candidate_limit_per_task"])
    for task_id, spec in protocol.get("normal_calibration_candidate_extensions", {}).items():
        if task_id not in task_by_id:
            raise KeyError(f"normal calibration extension refers to unknown task: {task_id}")
        start = int(spec["start_index"])
        count = int(spec["count"])
        if start < base_candidate_limit or count <= 0:
            raise ValueError(f"invalid normal calibration extension for {task_id}")
        relative_path = str(spec["manifest"])
        if not relative_path.startswith("amendments/") or not relative_path.endswith(".jsonl"):
            raise ValueError(f"invalid calibration extension manifest path: {relative_path}")
        seed_base = int(seeds["normal_calibration_candidates"]) + task_offsets[task_id] * 100_000
        calibration_extensions[relative_path] = [
            _row(
                split="normal_calibration_candidates",
                task=task_by_id[task_id],
                index=index,
                seed=seed_base + index,
                condition="nominal",
            )
            for index in range(start, start + count)
        ]
    # The retained 50 successful rows are materialized after rollout.  Preserve
    # an already materialized read-only view so rebuilding unrelated manifests
    # cannot silently erase the frozen calibration selection.
    calibration_path = root / "main10_normal_calibration.jsonl"
    manifests["main10_normal_calibration.jsonl"] = (
        _read_jsonl(calibration_path)
        if calibration_path.is_file() and calibration_path.stat().st_size
        else []
    )
    manifests["main10_failure_train.jsonl"] = _apply_seed_replacements(
        _task_rows(
            "failure_train",
            main10,
            int(protocol["failure_train_episodes_per_task"]),
            int(seeds["failure_train"]),
            perturbed=True,
        ),
        protocol.get("failure_train_seed_replacements", {}),
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
    manifests["stress4_severity.jsonl"] = _stress4_severity_rows(
        stress4,
        perturbed,
        extension_seed,
    )

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
    # E6 is a smaller, system-level comparison.  Its formal manifests are
    # deterministic result-blind subsets of E1, and its development split is
    # disjoint from every formal seed.  Keep this logic shared with the
    # isolated ``build_native6`` command so a later full rebuild cannot restore
    # the obsolete 200-per-task/four-fault views.
    manifests["native6_development.jsonl"] = build_native6_development_rows(
        nominal
    )
    manifests["native6_nominal.jsonl"] = build_native6_nominal_view(nominal)
    manifests["native6_perturbed.jsonl"] = build_native6_perturbed_view(
        perturbed
    )

    for name, rows in manifests.items():
        _write_jsonl(root / name, rows)
    for name, rows in calibration_extensions.items():
        _write_jsonl(root / name, rows)
    index_path = root / "MANIFEST_INDEX.json"
    # Rebuilding deterministic manifests after sealed execution may update a
    # protocol/configuration identity, but it must never make the sealed split
    # look unused again.  Carry the immutable first-execution provenance
    # forward when an existing index has already crossed that boundary.
    existing_index = (
        _load(index_path)
        if index_path.is_file() and index_path.stat().st_size
        else {}
    )
    sealed_executed = bool(existing_index.get("sealed_executed", False))
    index = {
        "schema": INDEX_SCHEMA,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "faults_sha256": _sha256(FAULTS_PATH),
        "manifests": {
            name: {"rows": len(rows), "sha256": _sha256(root / name)}
            for name, rows in manifests.items()
        },
        "calibration_candidate_extensions": {
            name: {"rows": len(rows), "sha256": _sha256(root / name)}
            for name, rows in calibration_extensions.items()
        },
        "sealed_executed": sealed_executed,
        "result_based_task_selection": False,
    }
    if sealed_executed:
        for field in ("sealed_first_started_stage", "sealed_first_started_utc"):
            if field not in existing_index:
                raise ValueError(f"sealed manifest index is missing {field}")
            index[field] = existing_index[field]
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_all_manifests(root)
    return index


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_all_manifests(root: Path = ROOT) -> dict[str, int]:
    # Native-6 v3 has its own event-grounded schema and verifier.  Those files
    # intentionally live beside the common episode manifests, but they are not
    # inputs to this legacy-schema validator.
    paths = sorted(
        path
        for path in root.glob("*.jsonl")
        if not path.name.startswith("native6_v3_")
    )
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
            key = (
                row["task"],
                row["variation"],
                row["seed"],
                row["condition"],
                row.get("fault_family"),
                row.get("fault_severity"),
                row.get("trigger_stage"),
            )
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
            "horizon3_single_event.jsonl",
            "horizon3_per_stage.jsonl",
        }
    ]
    for index, left in enumerate(independent):
        for right in independent[index + 1 :]:
            seed_overlap = {
                (task, seed)
                for task, _variation, seed, _condition, *_fault in split_keys[left]
            } & {
                (task, seed)
                for task, _variation, seed, _condition, *_fault in split_keys[right]
            }
            if seed_overlap:
                raise ValueError(f"independent split seeds overlap: {left}, {right}")
    return counts


if __name__ == "__main__":
    print(json.dumps(build_all_manifests(), indent=2, sort_keys=True))
