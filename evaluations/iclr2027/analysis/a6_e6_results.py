"""Validate and aggregate the final Native-6 v3 system comparison.

RVT and RACER retain their accepted original nominal executions and use only
the Amendment-4-verified v3 perturbed results.  Ours reuses the exact frozen
M5 nominal source episodes selected by ``native6_nominal.jsonl`` and consumes
the dedicated v3 perturbed execution.  Legacy cycle-floor perturbed results
are never read by this analysis.
"""

from __future__ import annotations

import csv
import copy
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
NATIVE = EVAL_ROOT / "results" / "native"
NATIVE_V3 = EVAL_ROOT / "results" / "native_v3"
A5_M5_NOMINAL = (
    EVAL_ROOT / "results" / "controlled" / "e1_e2" / "end_to_end" / "m5" / "nominal"
)
NOMINAL_MANIFEST = EVAL_ROOT / "manifests" / "native6_nominal.jsonl"
PERTURBED_MANIFEST = EVAL_ROOT / "manifests" / "native6_v3_perturbed.jsonl"
OUTPUT = NATIVE_V3 / "derived"
AMENDMENT_5 = (
    EVAL_ROOT / "configs" / "shared" / "native6_physical_protocol_v3_amendment_5.json"
)
M5_POST_E4_FREEZE = EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
M5_POST_E4_IMPACT = EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_IMPACT_SCOPE.json"
CURRENT_M5_FREEZE = (
    EVAL_ROOT / "results" / "a6_execution" / "M5_PENDING_DIRECT_IDENTITY_FREEZE.json"
)
CURRENT_E6_EXCLUSION = (
    EVAL_ROOT
    / "results"
    / "controlled"
    / "pending_direct_identity_refresh"
    / "e6"
    / "NO_RERUN.json"
)
SYSTEMS = ("rvt", "racer", "ours")
TASKS = (
    "close_jar",
    "open_drawer",
    "insert_onto_square_peg",
    "place_cups_3",
    "stack_cups",
    "sweep_to_dustpan",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing E6 input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe(episode_id: str) -> str:
    return str(episode_id).replace("/", "__")


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    path = root / "episodes" / f"{_safe(episode_id)}.json"
    if not path.is_file():
        raise RuntimeError(f"missing frozen M5 nominal episode: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _table_tex(rows: Iterable[Mapping[str, Any]]) -> str:
    names = {"rvt": "RVT", "racer": "RACER", "ours": "Ours"}
    supervision = {
        "rvt": "none",
        "racer": "recovery trajectories",
        "ours": "none",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Method & Recovery supervision & Nominal & Perturbed & Retention \\",
        r"\midrule",
    ]
    for row in rows:
        system = str(row["system"])
        lines.append(
            f"{names[system]} & {supervision[system]} & "
            f"{100.0 * float(row['nominal_task_macro_success']):.1f} & "
            f"{100.0 * float(row['perturbed_task_macro_success']):.1f} & "
            f"{100.0 * float(row['task_macro_retention']):.1f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _validate_manifest(rows: list[Mapping[str, Any]], *, condition: str) -> None:
    if len(rows) != 600:
        raise RuntimeError(f"Native-6 {condition} manifest must contain 600 rows")
    if Counter(str(row["task"]) for row in rows) != Counter({task: 100 for task in TASKS}):
        raise RuntimeError(f"Native-6 {condition} manifest is not six tasks x 100")
    if any(row.get("condition") != condition for row in rows):
        raise RuntimeError(f"Native-6 {condition} manifest contains another condition")


def _external_results(system: str, condition: str) -> list[dict[str, Any]]:
    path = (
        NATIVE / system / "formal" / "nominal_episodes.jsonl"
        if condition == "nominal"
        else NATIVE_V3 / system / "formal" / "perturbed_episodes.jsonl"
    )
    rows = _rows(path)
    if len(rows) != 600:
        raise RuntimeError(f"{system}/{condition} does not contain 600 results")
    if any(row.get("method_id") != system for row in rows):
        raise RuntimeError(f"{system}/{condition} method identity mismatch")
    if condition == "perturbed" and any(
        row.get("protocol_revision") != "native6_event_grounded_physics_v3"
        for row in rows
    ):
        raise RuntimeError(f"{system} perturbed results include an obsolete protocol")
    return rows


def _ours_nominal(manifest: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for row in manifest:
        source = str(row["source_episode_id"])
        result = _episode(A5_M5_NOMINAL, source)
        if (
            result.get("episode_id") != source
            or result.get("task") != row["task"]
            or int(result.get("seed")) != int(row["seed"])
            or int(result.get("variation")) != int(row["variation"])
        ):
            raise RuntimeError(f"Ours nominal source identity mismatch: {source}")
        results.append(
            {
                **result,
                "method_id": "ours",
                "condition": "nominal",
                "native6_episode_id": row["episode_id"],
                "source_episode_id": source,
            }
        )
    return results


def _ours_perturbed() -> list[dict[str, Any]]:
    rows = _rows(NATIVE_V3 / "ours" / "formal" / "episodes.jsonl")
    if len(rows) != 600:
        raise RuntimeError("Ours Native-v3 formal result does not contain 600 rows")
    if any(
        row.get("method_id") != "ours"
        or row.get("protocol_revision") != "native6_event_grounded_physics_v3"
        for row in rows
    ):
        raise RuntimeError("Ours Native-v3 formal identity mismatch")
    return rows


def _completed_physical_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row["cycle_file"]))
    if not path.is_absolute():
        path = ROOT / path
    events: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            cycle = json.loads(line)
            if "physical_events" in cycle:
                events.extend(dict(value) for value in cycle["physical_events"])
            else:
                injector = (cycle.get("execution") or {}).get("injector") or {}
                events.extend(dict(value) for value in injector.get("events") or ())
    return events


def _reaudit_completed_step_effects(
    rows: Iterable[Mapping[str, Any]], *, system: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corrected = []
    corrections = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        if (
            row.get("task") == "open_drawer"
            and row.get("fault_family") == "relation_loss"
            and bool(row.get("injection_triggered"))
            and not bool(row.get("physical_effect_confirmed"))
            and not bool(row.get("infrastructure_error"))
        ):
            events = _completed_physical_events(row)
            trigger = int(row["trigger_sim_step"])
            prior = [
                event
                for event in events
                if int(event.get("sim_step", -1)) <= trigger
                and event.get("contact_objects")
            ]
            targets = set(prior[-1]["contact_objects"]) if prior else set()
            effect = next(
                (
                    event
                    for event in events
                    if int(event.get("sim_step", -1)) >= trigger
                    and targets
                    and all(
                        float(value) > 0.9
                        for value in event.get("actual_open_amount", ())
                    )
                    and not targets.intersection(event.get("contact_objects", ()))
                ),
                None,
            )
            if effect is not None:
                row["physical_effect_confirmed"] = True
                row["physically_triggered"] = True
                row["effect_sim_step"] = int(effect["sim_step"])
                row["effect_time_s"] = float(effect["simulation_time_s"])
                row["violation_onset_sim_step"] = int(effect["sim_step"])
                row["violation_onset_time_s"] = float(effect["simulation_time_s"])
                corrections.append(
                    {
                        "system": system,
                        "episode_id": row["episode_id"],
                        "targets": sorted(targets),
                        "effect_sim_step": int(effect["sim_step"]),
                        "task_success_unchanged": bool(row["final_success"]),
                    }
                )
        corrected.append(row)
    return corrected, corrections


def _match(
    results: Iterable[Mapping[str, Any]],
    manifest: Iterable[Mapping[str, Any]],
    *,
    system: str,
    condition: str,
) -> list[dict[str, Any]]:
    result_rows = [dict(row) for row in results]
    manifest_rows = [dict(row) for row in manifest]
    by_id = {
        str(row.get("native6_episode_id") or row["episode_id"]): row
        for row in result_rows
    }
    if len(by_id) != len(result_rows):
        raise RuntimeError(f"duplicate {system}/{condition} episode IDs")
    expected = {str(row["episode_id"]): row for row in manifest_rows}
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))[:3]
        extra = sorted(set(by_id) - set(expected))[:3]
        raise RuntimeError(
            f"{system}/{condition} manifest mismatch; missing={missing}, extra={extra}"
        )
    ordered = []
    for episode_id, manifest_row in expected.items():
        result = by_id[episode_id]
        if (
            result.get("task") != manifest_row["task"]
            or int(result.get("seed")) != int(manifest_row["seed"])
            or int(result.get("variation")) != int(manifest_row["variation"])
        ):
            raise RuntimeError(f"{system}/{condition} row identity mismatch: {episode_id}")
        ordered.append(result)
    return ordered


def generate() -> dict[str, Any]:
    freeze = json.loads(M5_POST_E4_FREEZE.read_text(encoding="utf-8"))
    impact = json.loads(M5_POST_E4_IMPACT.read_text(encoding="utf-8"))
    current_freeze = json.loads(CURRENT_M5_FREEZE.read_text(encoding="utf-8"))
    current_exclusion = json.loads(CURRENT_E6_EXCLUSION.read_text(encoding="utf-8"))
    affected_native = sorted(
        set(TASKS).intersection(
            set(impact["affected_tasks"]) | set(current_freeze["affected_tasks"])
        )
    )
    if affected_native:
        raise RuntimeError(
            "a frozen M5 revision changed Native-6 task semantics: "
            + ", ".join(affected_native)
        )
    if (
        current_freeze.get("status") != "PASS"
        or (current_freeze.get("supersedes") or {}).get("sha256")
        != _sha256(M5_POST_E4_FREEZE)
        or current_exclusion.get("status") != "PASS_NO_RERUN"
        or current_exclusion.get("freeze_sha256") != _sha256(CURRENT_M5_FREEZE)
    ):
        raise RuntimeError("current M5 freeze or Native-6 exclusion is invalid")
    ours_identity_path = NATIVE_V3 / "ours" / "formal" / "SYSTEM_IDENTITY.json"
    ours_identity = json.loads(ours_identity_path.read_text(encoding="utf-8"))
    if (
        ours_identity.get("checkpoint_identity")
        != freeze["model_tree_identity"]["aggregate_sha256"]
        or ours_identity.get("environment_identity")
        != freeze["algorithm_source_identity"]["aggregate_sha256"]
        or ours_identity.get("m5_post_e4_freeze_sha256")
        != _sha256(M5_POST_E4_FREEZE)
    ):
        raise RuntimeError("Ours E6 formal result does not use the post-E4 M5 freeze")
    nominal_manifest = _rows(NOMINAL_MANIFEST)
    perturbed_manifest = _rows(PERTURBED_MANIFEST)
    _validate_manifest(nominal_manifest, condition="nominal")
    _validate_manifest(perturbed_manifest, condition="perturbed")

    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for system in ("rvt", "racer"):
        all_results[(system, "nominal")] = _match(
            _external_results(system, "nominal"),
            nominal_manifest,
            system=system,
            condition="nominal",
        )
        all_results[(system, "perturbed")] = _match(
            _external_results(system, "perturbed"),
            perturbed_manifest,
            system=system,
            condition="perturbed",
        )
    all_results[("ours", "nominal")] = _match(
        _ours_nominal(nominal_manifest),
        nominal_manifest,
        system="ours",
        condition="nominal",
    )
    all_results[("ours", "perturbed")] = _match(
        _ours_perturbed(),
        perturbed_manifest,
        system="ours",
        condition="perturbed",
    )

    audit_corrections = []
    for system in SYSTEMS:
        corrected, changes = _reaudit_completed_step_effects(
            all_results[(system, "perturbed")], system=system
        )
        all_results[(system, "perturbed")] = corrected
        audit_corrections.extend(changes)

    complete = []
    table = []
    for system in SYSTEMS:
        per_condition: dict[str, dict[str, float]] = defaultdict(dict)
        for condition in ("nominal", "perturbed"):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in all_results[(system, condition)]:
                grouped[str(row["task"])].append(row)
            for task in TASKS:
                values = grouped[task]
                if len(values) != 100:
                    raise RuntimeError(f"{system}/{condition}/{task} is not 100 episodes")
                successes = sum(bool(row["final_success"]) for row in values)
                rate = successes / len(values)
                per_condition[condition][task] = rate
                complete.append(
                    {
                        "system": system,
                        "task": task,
                        "condition": condition,
                        "successes": successes,
                        "episodes": len(values),
                        "success_rate": rate,
                        "infrastructure_errors": sum(
                            bool(row.get("infrastructure_error")) for row in values
                        ),
                        "eligible": sum(bool(row.get("eligible")) for row in values),
                        "injection_triggered": sum(
                            bool(row.get("injection_triggered")) for row in values
                        ),
                        "physical_effect_confirmed": sum(
                            bool(row.get("physical_effect_confirmed")) for row in values
                        ),
                    }
                )
        retention = []
        for task in TASKS:
            nominal = per_condition["nominal"][task]
            perturbed = per_condition["perturbed"][task]
            if nominal <= 0.0:
                raise RuntimeError(f"retention undefined for {system}/{task}: nominal success is zero")
            retention.append(perturbed / nominal)
        table.append(
            {
                "system": system,
                "nominal_task_macro_success": mean(per_condition["nominal"].values()),
                "perturbed_task_macro_success": mean(per_condition["perturbed"].values()),
                "task_macro_retention": mean(retention),
            }
        )

    _atomic_csv(OUTPUT / "table2_native_systems.csv", table)
    _atomic_text(OUTPUT / "table2_native_systems.tex", _table_tex(table))
    _atomic_csv(OUTPUT / "appendix_native_complete.csv", complete)
    _atomic_json(
        OUTPUT / "E6_COMPLETED_STEP_REAUDIT.json",
        {
            "schema": "essay2608.iclr2027.native6-v3-completed-step-reaudit.v1",
            "protocol_amendment": "native6_event_grounded_physics_v3_amendment_5",
            "raw_artifacts_mutated": False,
            "task_success_mutated": False,
            "corrections": audit_corrections,
        },
    )
    inputs = (
        NOMINAL_MANIFEST,
        PERTURBED_MANIFEST,
        NATIVE / "rvt" / "formal" / "nominal_episodes.jsonl",
        NATIVE / "racer" / "formal" / "nominal_episodes.jsonl",
        NATIVE_V3 / "rvt" / "formal" / "perturbed_episodes.jsonl",
        NATIVE_V3 / "racer" / "formal" / "perturbed_episodes.jsonl",
        NATIVE_V3 / "ours" / "formal" / "episodes.jsonl",
        ours_identity_path,
        AMENDMENT_5,
        M5_POST_E4_FREEZE,
        M5_POST_E4_IMPACT,
        CURRENT_M5_FREEZE,
        CURRENT_E6_EXCLUSION,
    )
    summary = {
        "schema": "essay2608.iclr2027.a6-e6-analysis.v2",
        "status": "PASS",
        "systems": len(SYSTEMS),
        "episodes": sum(len(rows) for rows in all_results.values()),
        "legacy_perturbed_protocol_read": False,
        "ours_nominal_reused_from_frozen_m5": True,
        "ours_nominal_reuse_justification": {
            "selection_rule": impact["selection_rule"],
            "affected_native6_tasks": affected_native,
            "all_native6_tasks_semantically_unchanged": True,
            "impact_scope_sha256": _sha256(M5_POST_E4_IMPACT),
            "current_m5_freeze_sha256": _sha256(CURRENT_M5_FREEZE),
            "current_e6_exclusion_sha256": _sha256(CURRENT_E6_EXCLUSION),
        },
        "completed_step_reaudit": {
            "protocol_amendment": "native6_event_grounded_physics_v3_amendment_5",
            "corrections": len(audit_corrections),
            "raw_artifacts_mutated": False,
            "task_success_mutated": False,
        },
        "retention_definition": "perturbed_rate_divided_by_nominal_rate_per_task_then_macro_mean",
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in inputs
        ],
        "outputs": {
            "table2_native_systems.csv": _sha256(OUTPUT / "table2_native_systems.csv"),
            "table2_native_systems.tex": _sha256(OUTPUT / "table2_native_systems.tex"),
            "appendix_native_complete.csv": _sha256(OUTPUT / "appendix_native_complete.csv"),
            "E6_COMPLETED_STEP_REAUDIT.json": _sha256(
                OUTPUT / "E6_COMPLETED_STEP_REAUDIT.json"
            ),
        },
    }
    _atomic_json(OUTPUT / "E6_ANALYSIS.json", summary)
    return summary


def main() -> int:
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
