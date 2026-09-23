"""Materialize only the E6 Native-6 assets from frozen E1 manifests.

Unlike ``build.py``, this command cannot rewrite Main-10, Stress-4, Horizon-3,
calibration, or failure-training manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.manifests.native6 import (
    NATIVE6_FAULTS,
    NATIVE6_TASK_IDS,
    build_development_rows,
    build_nominal_view,
    build_perturbed_view,
    validate_native6_rows,
)
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT


ROOT = REPOSITORY_ROOT / "evaluations" / "iclr2027" / "manifests"
CONFIG_ROOT = REPOSITORY_ROOT / "evaluations" / "iclr2027" / "configs" / "shared"
FIXTURE_ROOT = REPOSITORY_ROOT / "evaluations" / "iclr2027" / "tests" / "fixtures" / "native6"
SOURCE_NOMINAL = ROOT / "main10_nominal.jsonl"
SOURCE_PERTURBED = ROOT / "main10_perturbed.jsonl"
DEVELOPMENT_PATH = ROOT / "native6_development.jsonl"
NOMINAL_PATH = ROOT / "native6_nominal.jsonl"
PERTURBED_PATH = ROOT / "native6_perturbed.jsonl"
INDEX_PATH = ROOT / "NATIVE6_MANIFEST_INDEX.json"
GLOBAL_INDEX_PATH = ROOT / "MANIFEST_INDEX.json"
CONTRACT_PATH = CONFIG_ROOT / "native6_contract.json"
FAULTS_PATH = CONFIG_ROOT / "faults.json"
RESULT_SCHEMA_PATH = CONFIG_ROOT / "native6_result_schema.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _build_fixtures(
    development: list[dict[str, Any]], nominal: list[dict[str, Any]]
) -> list[Path]:
    contract = _load_json(CONTRACT_PATH)
    faults = _load_json(FAULTS_PATH)
    initialization_rows = []
    for task in NATIVE6_TASK_IDS:
        row = next(
            item
            for item in development
            if item["task"] == task and item["condition"] == "nominal"
        )
        initialization_rows.append(
            {
                "schema": "essay2608.iclr2027.native6-initialization-fixture.v1",
                "episode_id": row["episode_id"],
                "task": task,
                "variation": row["variation"],
                "seed": row["seed"],
                "base_task": "place_cups" if task == "place_cups_3" else task,
                "fixed_base_variation": 2 if task == "place_cups_3" else None,
                "expected_success_source": "rlbench_task.success",
            }
        )

    fault_rows = []
    for task in NATIVE6_TASK_IDS:
        for family in NATIVE6_FAULTS:
            row = next(
                item
                for item in development
                if item["task"] == task and item["fault_family"] == family
            )
            stage_fraction = float(faults["trigger_stages"][row["trigger_stage"]])
            expected = {
                "earliest_cycle": int(round(stage_fraction * int(row["horizon"]))),
                "target_arm": "single",
                "severity": "medium",
            }
            if family == "actuation_delay":
                expected["duration_cycles"] = int(
                    faults["medium"]["actuation_delay_cycles"]
                )
            elif family == "missed_interaction":
                expected["suppressed_occurrences"] = int(
                    faults["medium"]["missed_interaction_occurrences"]
                )
            else:
                expected["minimum_stable_relation_cycles"] = int(
                    faults["eligibility"]["minimum_stable_relation_cycles"]
                )
                expected["translation_m"] = float(faults["medium"]["translation_m"])
            fault_rows.append(
                {
                    "schema": "essay2608.iclr2027.native6-fault-fixture.v1",
                    "episode_id": row["episode_id"],
                    "task": task,
                    "fault_family": family,
                    "trigger_stage": row["trigger_stage"],
                    "expected": expected,
                }
            )

    manifest_row = nominal[0]
    valid_result = {
        "schema": "essay2608.iclr2027.native-system-episode-result.v1",
        "episode_id": manifest_row["episode_id"],
        "source_episode_id": manifest_row["source_episode_id"],
        "pair_id": manifest_row["pair_id"],
        "method_id": "rvt",
        "task": manifest_row["task"],
        "variation": manifest_row["variation"],
        "seed": manifest_row["seed"],
        "condition": manifest_row["condition"],
        "fault_family": manifest_row["fault_family"],
        "fault_severity": manifest_row["fault_severity"],
        "trigger_stage": manifest_row["trigger_stage"],
        "horizon": manifest_row["horizon"],
        "config_identity": "fixture-config-sha256",
        "checkpoint_identity": "fixture-checkpoint-sha256",
        "environment_identity": "fixture-environment-sha256",
        "eligible": False,
        "physically_triggered": False,
        "violation_onset_cycle": None,
        "violation_end_cycle": None,
        "relation_restored_cycle": None,
        "final_success": True,
        "cycles": 10,
        "termination_reason": "success",
        "infrastructure_error": False,
        "wall_seconds": 1.0,
        "peak_memory_kib": 1024,
        "cycle_file": "cycles/fixture.jsonl.gz",
        "cycle_file_sha256": "0" * 64,
        "cycle_records": 10,
    }

    initialization_path = FIXTURE_ROOT / "initialization_cases.jsonl"
    fault_path = FIXTURE_ROOT / "fault_cases.jsonl"
    result_path = FIXTURE_ROOT / "valid_native_result.json"
    _write_jsonl(initialization_path, initialization_rows)
    _write_jsonl(fault_path, fault_rows)
    _write_json(
        result_path,
        {
            "schema": "essay2608.iclr2027.native6-result-fixture.v1",
            "contract_schema": contract["schema"],
            "manifest_row": manifest_row,
            "result": valid_result,
        },
    )
    return [initialization_path, fault_path, result_path]


def materialize(root: Path = ROOT) -> dict[str, Any]:
    if root != ROOT:
        source_nominal_path = root / SOURCE_NOMINAL.name
        source_perturbed_path = root / SOURCE_PERTURBED.name
        development_path = root / DEVELOPMENT_PATH.name
        nominal_path = root / NOMINAL_PATH.name
        perturbed_path = root / PERTURBED_PATH.name
        index_path = root / INDEX_PATH.name
    else:
        source_nominal_path = SOURCE_NOMINAL
        source_perturbed_path = SOURCE_PERTURBED
        development_path = DEVELOPMENT_PATH
        nominal_path = NOMINAL_PATH
        perturbed_path = PERTURBED_PATH
        index_path = INDEX_PATH

    before = {
        "main10_nominal.jsonl": _sha256(source_nominal_path),
        "main10_perturbed.jsonl": _sha256(source_perturbed_path),
    }
    source_nominal = _load_jsonl(source_nominal_path)
    source_perturbed = _load_jsonl(source_perturbed_path)
    development = build_development_rows(source_nominal)
    nominal = build_nominal_view(source_nominal)
    perturbed = build_perturbed_view(source_perturbed)
    validation = validate_native6_rows(
        development,
        nominal,
        perturbed,
        source_nominal_ids={str(row["episode_id"]) for row in source_nominal},
        source_perturbed_ids={str(row["episode_id"]) for row in source_perturbed},
    )
    _write_jsonl(development_path, development)
    _write_jsonl(nominal_path, nominal)
    _write_jsonl(perturbed_path, perturbed)

    after = {
        "main10_nominal.jsonl": _sha256(source_nominal_path),
        "main10_perturbed.jsonl": _sha256(source_perturbed_path),
    }
    if before != after:
        raise RuntimeError("isolated Native-6 materialization changed an E1 manifest")

    fixture_paths = _build_fixtures(development, nominal) if root == ROOT else []
    manifests = {
        development_path.name: {
            "rows": len(development),
            "sha256": _sha256(development_path),
            "purpose": "development_only_every_task_10_plus_10",
        },
        nominal_path.name: {
            "rows": len(nominal),
            "sha256": _sha256(nominal_path),
            "purpose": "formal_e6_readonly_view",
        },
        perturbed_path.name: {
            "rows": len(perturbed),
            "sha256": _sha256(perturbed_path),
            "purpose": "formal_e6_readonly_view",
        },
    }
    index = {
        "schema": "essay2608.iclr2027.native6-manifest-index.v1",
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "faults_path": str(FAULTS_PATH.relative_to(REPOSITORY_ROOT)),
        "faults_sha256": _sha256(FAULTS_PATH),
        "result_schema_path": str(RESULT_SCHEMA_PATH.relative_to(REPOSITORY_ROOT)),
        "result_schema_sha256": _sha256(RESULT_SCHEMA_PATH),
        "source_e1_manifests": before,
        "source_e1_manifests_unchanged": True,
        "selection_read_episode_results": False,
        "manifests": manifests,
        "fixtures": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
            for path in fixture_paths
        },
        "validation": validation,
    }
    _write_json(index_path, index)

    if root == ROOT and GLOBAL_INDEX_PATH.is_file():
        global_index = _load_json(GLOBAL_INDEX_PATH)
        global_manifests = global_index.setdefault("manifests", {})
        global_manifests.update(manifests)
        global_index["native6_manifest_index"] = {
            "path": str(INDEX_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(INDEX_PATH),
            "e1_sources_unchanged": True,
            "materialized_before_first_e6_formal_episode": True,
        }
        _write_json(GLOBAL_INDEX_PATH, global_index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    index = materialize(args.output_root.resolve())
    print(json.dumps(index["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
