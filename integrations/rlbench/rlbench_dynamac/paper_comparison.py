"""Build a consolidated DynaMAC paper-to-local experiment comparison.

The updater discovers completed evaluation JSON files below ``results/`` and
writes Markdown, CSV, and JSON summaries.  Paper values are fixed constants
transcribed from Tables I--IV of arXiv:2607.22119v1.  Local dynamic runs are
reported as diagnostics because the paper's DynaBench movement and arm-
perturbation defaults are not published.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from essay2608.policy import DynaMACConfig

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results"
DEFAULT_RELEASE = "v1"
PAPER_REFERENCE = "DynaMAC, arXiv:2607.22119v1"
EXPECTED_LOCAL_CONFIG = asdict(
    DynaMACConfig(
        **json.loads(
            (INTEGRATION_ROOT / "configs" / "dynamac_rlbench_local.json").read_text(
                encoding="utf-8"
            )
        )
    )
)
EXPECTED_MODEL_SCHEMA_VERSION = 13
EXPECTED_TAPAS_COMMIT = "52e35214b9baa7b190b87196c36b9e98f4006149"
EXPECTED_SELECTION_SEMANTICS_ID = (
    "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
)
EXPECTED_EVALUATION_PROTOCOL_ID = (
    "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-noop-clock-v2"
)


@dataclass(frozen=True)
class PaperCell:
    table: str
    condition: str
    task: str
    paper_rate: float
    local_task: str | None = None
    local_scenarios: tuple[str, ...] = ()
    result_family: str | None = None
    note: str = ""
    unavailable_reason: str = ""
    stopped_reason: str = ""


@dataclass(frozen=True)
class LocalRun:
    path: Path
    task: str
    scenario: str
    seed: int
    episodes: int
    horizon: int
    variation: int
    successes: int
    success_rate: float
    payload: dict[str, Any]


PAPER_AVERAGES = {
    "I": 0.94,
    "II": 0.95,
    "III": 0.96,
    "IV": 0.95,
}

TABLE_SCOPES = {
    "I": "Unimanual RLBench",
    "II": "Static bimanual RLBench",
    "III": "Dynamic bimanual RLBench",
    "IV": "Real-world bimanual hardware",
}

WIPE_DESK_DYNAMIC_UNAVAILABLE = (
    "Unavailable: the public generic Scene movement path cannot restore WipeDesk "
    "after dirt objects are changed mid-episode."
)


def _cells() -> tuple[PaperCell, ...]:
    table_i_tasks = (
        ("stack_wine", "StackWine", 1.00, 1.00, 1.00),
        ("place_cups", "PlaceCups", 0.99, 0.97, 0.99),
        ("open_microwave", "OpenMicrowave", 0.99, 0.99, 0.97),
        ("wipe_desk", "WipeDesk", 1.00, 0.66, 0.69),
    )
    cells: list[PaperCell] = []
    for task, label, static, smooth, teleport in table_i_tasks:
        for condition, scenario, paper_rate in (
            ("Static", "static", static),
            ("Smooth dynamics", "smooth", smooth),
            ("Teleportation", "teleport", teleport),
        ):
            unavailable_reason = (
                WIPE_DESK_DYNAMIC_UNAVAILABLE
                if task == "wipe_desk" and scenario != "static"
                else ""
            )
            stopped_reason = ""
            cells.append(
                PaperCell(
                    table="I",
                    condition=condition,
                    task=label,
                    paper_rate=paper_rate,
                    local_task=None if unavailable_reason else task,
                    local_scenarios=() if unavailable_reason else (scenario,),
                    result_family="table_i",
                    note=(
                        "Independent five-demonstration cohort."
                        if scenario == "static"
                        else "Local public-RLBench motion schedule; paper defaults unpublished."
                    ),
                    unavailable_reason=unavailable_reason,
                    stopped_reason=stopped_reason,
                )
            )

    table_ii_tasks = (
        ("bimanual_put_bottle_in_fridge", "StoreBottle", 0.82),
        ("bimanual_handover_item", "HandOver", 0.97),
        ("bimanual_sweep_to_dustpan", "SweepDust", 1.00),
        ("bimanual_lift_tray", "LiftTray", 1.00),
    )
    for task, label, paper_rate in table_ii_tasks:
        cells.append(
            PaperCell(
                table="II",
                condition="Static",
                task=label,
                paper_rate=paper_rate,
                local_task=task,
                local_scenarios=("static",),
                result_family="bimanual_static",
                note=(
                    "Independent five-demonstration cohort. "
                    "[SweepDust diagnosis](sweep_dust_diagnosis.md)."
                    if task == "bimanual_sweep_to_dustpan"
                    else "Independent five-demonstration cohort."
                ),
            )
        )

    cells.extend(
        (
            PaperCell(
                table="III",
                condition="Coordination",
                task="Hand Left",
                paper_rate=0.97,
                local_task="bimanual_handover_item_dynamic",
                local_scenarios=(
                    "coordination_hand_left",
                    "coordination_left",
                    "hand_left",
                    "perturb_left",
                    "left_arm_perturbed",
                ),
                result_family="bimanual_coordination",
                note="Arm-perturbation magnitude and timing are unpublished.",
            ),
            PaperCell(
                table="III",
                condition="Coordination",
                task="Hand Right",
                paper_rate=0.97,
                local_task="bimanual_handover_item_dynamic",
                local_scenarios=(
                    "coordination_hand_right",
                    "coordination_right",
                    "hand_right",
                    "perturb_right",
                    "right_arm_perturbed",
                ),
                result_family="bimanual_coordination",
                note="Arm-perturbation magnitude and timing are unpublished.",
            ),
        )
    )
    for task, label, paper_rate in table_ii_tasks:
        cells.append(
            PaperCell(
                table="III",
                condition="Dynamic environment",
                task=label,
                paper_rate=paper_rate,
                local_task=task,
                # The paper does not identify its Table III motion type. Keep
                # the local paper-cell selector fixed to teleport.
                local_scenarios=("teleport",),
                result_family="bimanual_dynamic",
                note="Local public-RLBench intervention; paper DynaBench defaults unpublished.",
            )
        )

    for condition, task, paper_rate in (
        ("Static", "StoreItem", 0.96),
        ("Static", "HandOver", 0.96),
        ("Static", "SweepBlocks", 0.96),
        ("Static", "LiftJointly", 1.00),
        ("Dynamic environment", "StoreItem", 0.92),
        ("Dynamic environment", "HandOver", 0.88),
        ("Coordination", "Hand Left", 0.92),
        ("Coordination", "Hand Right", 0.96),
    ):
        cells.append(
            PaperCell(
                table="IV",
                condition=condition,
                task=task,
                paper_rate=paper_rate,
                note="Physical dual-Franka setup unavailable locally.",
            )
        )
    return tuple(cells)


PAPER_CELLS = _cells()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_run(path: Path) -> tuple[LocalRun | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path}: unreadable JSON ({exc})"
    if not isinstance(payload, dict):
        return None, None
    required = ("task", "scenario", "seed", "episodes", "horizon", "successes")
    if not all(key in payload for key in required) or "success_rate" not in payload:
        return None, None
    if not isinstance(payload["task"], str) or not isinstance(payload["scenario"], str):
        return None, f"{path}: task/scenario must be strings"
    integer_fields = ("seed", "episodes", "horizon", "successes")
    if not all(isinstance(payload[key], int) and not isinstance(payload[key], bool) for key in integer_fields):
        return None, f"{path}: seed/episodes/horizon/successes must be integers"
    if not _is_number(payload["success_rate"]):
        return None, f"{path}: success_rate must be numeric"
    episodes = int(payload["episodes"])
    variation = payload.get("variation", 0)
    if not isinstance(variation, int) or isinstance(variation, bool) or variation < 0:
        return None, f"{path}: variation must be a non-negative integer"
    successes = int(payload["successes"])
    success_rate = float(payload["success_rate"])
    if episodes < 1 or not 0 <= successes <= episodes:
        return None, f"{path}: invalid episode totals"
    if abs(success_rate - successes / episodes) > 1e-9:
        return None, f"{path}: success_rate disagrees with successes/episodes"
    episode_rows = payload.get("results")
    if isinstance(episode_rows, list):
        if len(episode_rows) != episodes:
            return None, f"{path}: result row count disagrees with episodes"
        counted = sum(bool(row.get("success")) for row in episode_rows if isinstance(row, dict))
        if counted != successes:
            return None, f"{path}: result rows disagree with successes"
    return (
        LocalRun(
            path=path,
            task=payload["task"],
            scenario=payload["scenario"],
            seed=int(payload["seed"]),
            episodes=episodes,
            horizon=int(payload["horizon"]),
            variation=variation,
            successes=successes,
            success_rate=success_rate,
            payload=payload,
        ),
        None,
    )


def discover_runs(results_dir: Path) -> tuple[list[LocalRun], list[str]]:
    runs: list[LocalRun] = []
    warnings: list[str] = []
    generated = {"paper_comparison.json"}
    for path in sorted(results_dir.rglob("*.json")):
        if path.name in generated:
            continue
        # One-episode coordination smoke files predate the evaluation-result
        # schema (their top-level ``episodes`` field is a row list).  They are
        # launcher diagnostics, never candidates for a paper cell, so exclude
        # them without presenting a misleading malformed-result warning.
        if path.name.startswith("smoke_") and "_n1" in path.stem:
            continue
        run, warning = _load_run(path)
        if run is not None:
            runs.append(run)
        if warning is not None:
            warnings.append(warning)
    return runs, warnings


def _family_matches(cell: PaperCell, run: LocalRun, results_dir: Path) -> bool:
    try:
        relative_parts = run.path.relative_to(results_dir).parts
    except ValueError:
        relative_parts = run.path.parts
    schema = str(run.payload.get("schema", ""))
    in_table_i = "table_i" in relative_parts or schema.startswith("dynamac-table-i-")
    if cell.result_family == "table_i":
        return in_table_i
    if cell.result_family in {
        "bimanual_static",
        "bimanual_dynamic",
        "bimanual_coordination",
    }:
        return not in_table_i
    return True


def _model_identity_rank(run: LocalRun) -> int:
    """Rank exact corrected runs before mismatched and legacy results."""

    identity = run.payload.get("model_identity")
    if not isinstance(identity, dict):
        return 2
    fingerprint_present = bool(identity.get("fingerprint")) or bool(
        identity.get("left_fingerprint") and identity.get("right_fingerprint")
    )
    if (
        identity.get("manifest_authenticated") is True
        and identity.get("training_config") == EXPECTED_LOCAL_CONFIG
        and identity.get("model_schema_version") == EXPECTED_MODEL_SCHEMA_VERSION
        and identity.get("selection_semantics_id")
        == EXPECTED_SELECTION_SEMANTICS_ID
        and identity.get("tapas_reference_commit") == EXPECTED_TAPAS_COMMIT
        and fingerprint_present
        and run.payload.get("evaluation_protocol_id")
        == EXPECTED_EVALUATION_PROTOCOL_ID
    ):
        return 0
    return 1


def _model_fingerprint_key(run: LocalRun) -> tuple[str, ...]:
    identity = run.payload.get("model_identity")
    if not isinstance(identity, dict):
        return ()
    if identity.get("fingerprint"):
        return (str(identity["fingerprint"]),)
    if identity.get("left_fingerprint") and identity.get("right_fingerprint"):
        return (
            str(identity["left_fingerprint"]),
            str(identity["right_fingerprint"]),
        )
    return ()


def _select_run(
    cell: PaperCell,
    runs: Iterable[LocalRun],
    results_dir: Path,
    *,
    seed: int,
    episodes: int,
    horizon: int,
) -> LocalRun | None:
    if cell.local_task is None:
        return None
    candidates = [
        run
        for run in runs
        if run.task == cell.local_task
        and run.scenario in cell.local_scenarios
        and run.seed == seed
        and run.episodes == episodes
        and run.horizon == horizon
        and run.variation == 0
        and _family_matches(cell, run, results_dir)
    ]
    if not candidates:
        return None
    scenario_rank = {name: index for index, name in enumerate(cell.local_scenarios)}
    candidates.sort(
        key=lambda run: (
            scenario_rank[run.scenario],
            _model_identity_rank(run),
            str(run.path),
        )
    )
    best_key = (
        scenario_rank[candidates[0].scenario],
        _model_identity_rank(candidates[0]),
    )
    equally_ranked = [
        run
        for run in candidates
        if (scenario_rank[run.scenario], _model_identity_rank(run)) == best_key
    ]
    if best_key[1] == 0 and len(equally_ranked) > 1:
        identities = {_model_fingerprint_key(run) for run in equally_ranked}
        paths = ", ".join(str(run.path) for run in equally_ranked)
        raise RuntimeError(
            "multiple corrected results match one paper cell; select an exact "
            f"checkpoint identity explicitly ({len(identities)} identities): {paths}"
        )
    return candidates[0]


def _protocol_metadata(run: LocalRun) -> dict[str, Any]:
    for key in ("scenario_protocol", "protocol", "coordination_protocol"):
        value = run.payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _applied_events(run: LocalRun) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    rows = run.payload.get("results")
    if not isinstance(rows, list):
        return events
    for row in rows:
        if not isinstance(row, dict):
            continue
        scenario_events = row.get("scenario_events")
        if not isinstance(scenario_events, list):
            continue
        events.extend(
            event
            for event in scenario_events
            if isinstance(event, dict) and event.get("applied") is True
        )
    return events


def _numeric_event_values(events: Sequence[dict[str, Any]], key: str) -> list[float]:
    return sorted(
        float(event[key])
        for event in events
        if _is_number(event.get(key))
    )


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def _material_protocol_note(cell: PaperCell, run: LocalRun | None) -> str:
    if run is None or cell.table != "III" or cell.condition != "Dynamic environment":
        return ""
    events = _applied_events(run)
    state_values = _numeric_event_values(events, "task_state_l2")
    root_values = _numeric_event_values(events, "root_pose_l2")
    state_median = _median(state_values)
    root_median = _median(root_values)
    state_max = max(state_values) if state_values else None
    root_max = max(root_values) if root_values else None
    if cell.task == "StoreBottle":
        if state_median is None:
            return "Material evidence unavailable; public static-workspace kidnap() reinitializes task objects."
        return (
            f"Material object-state change (median task-state L2 {state_median:.3f}), "
            "but task-root motion is zero; public static-workspace kidnap() "
            "reinitializes the task."
        )
    if cell.task == "HandOver":
        state_text = "unknown" if state_median is None else f"{state_median:.3f}"
        return (
            f"Material episode reinitialization (median task-state L2 {state_text}; "
            "task-root motion zero); public kidnap() can release an already-grasped "
            "item, so this is not the paper intervention."
        )
    if cell.task == "SweepDust":
        state_text = "unknown" if state_max is None else f"{state_max:.2e}"
        root_text = "unknown" if root_max is None else f"{root_max:.2e}"
        return (
            f"No material candidate/root motion (maximum task-state L2 {state_text}; "
            f"maximum root-pose L2 {root_text}); the upstream 1e-9 effectiveness "
            "flag is only numerically true."
        )
    if cell.task == "LiftTray":
        root_text = "unknown" if root_median is None else f"{root_median:.3f}"
        metadata = _protocol_metadata(run)
        effective = metadata.get("episodes_with_effective_intervention")
        effective_text = "unknown" if not isinstance(effective, int) else str(effective)
        return (
            f"Material task-root motion (median root-pose L2 {root_text}; "
            f"{effective_text}/{run.episodes} interventions effective)."
        )
    return ""


def _status(cell: PaperCell, run: LocalRun | None) -> str:
    if cell.unavailable_reason:
        return "unavailable"
    if cell.table == "IV":
        return "hardware unavailable"
    if run is None:
        if cell.stopped_reason:
            return "stopped after 0% preliminary run"
        return "pending"
    identity_rank = _model_identity_rank(run)
    if identity_rank == 2:
        return "historical pre-fix result"
    if identity_rank == 1:
        return "configuration-mismatch diagnostic"
    metadata = _protocol_metadata(run)
    if metadata.get("protocol_valid") is False:
        return "invalid diagnostic"
    # Static Table I has the same interpretation boundary as static Table II:
    # it is a reproduction of the paper cell using an independent local
    # five-demonstration cohort, not a byte-identical rerun of author data.
    # The Table I evaluator conservatively marks the whole local protocol
    # family paper_comparable=false because its dynamic schedules are local;
    # do not let that family-level flag demote the static condition.
    if cell.table == "I" and cell.condition == "Static":
        return "local reproduction"
    if metadata.get("paper_comparable") is False:
        return "non-comparable diagnostic"
    if cell.table == "I" and cell.condition != "Static":
        return "non-comparable diagnostic"
    if cell.table == "III":
        return "non-comparable diagnostic"
    return "local reproduction"


def build_records(
    results_dir: Path,
    *,
    seed: int,
    episodes: int,
    horizon: int,
    release: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if release is not None:
        if not release or Path(release).name != release or release in {".", ".."}:
            raise ValueError("release must be one directory name")
        discovery_root = results_dir / release
    else:
        discovery_root = results_dir
    runs, warnings = discover_runs(discovery_root)
    records: list[dict[str, Any]] = []
    for cell in PAPER_CELLS:
        run = _select_run(
            cell,
            runs,
            results_dir,
            seed=seed,
            episodes=episodes,
            horizon=horizon,
        )
        source = None
        if run is not None:
            try:
                source = run.path.relative_to(results_dir).as_posix()
            except ValueError:
                source = str(run.path)
        records.append(
            {
                "table": cell.table,
                "condition": cell.condition,
                "task": cell.task,
                "paper_success_rate": cell.paper_rate,
                "local_success_rate": run.success_rate if run is not None else None,
                "difference": run.success_rate - cell.paper_rate if run is not None else None,
                "local_successes": run.successes if run is not None else None,
                "local_episodes": run.episodes if run is not None else None,
                "status": _status(cell, run),
                "source_file": source,
                "notes": (
                    cell.unavailable_reason
                    or (cell.stopped_reason if run is None else "")
                    or cell.note
                ),
                "protocol_note": _material_protocol_note(cell, run),
            }
        )
    return records, warnings


def _table_summary(table: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in records if row["table"] == table]
    accepted_statuses = (
        {"local reproduction"}
        if table in {"I", "II"}
        else {"non-comparable diagnostic"}
        if table == "III"
        else set()
    )
    available = [
        row
        for row in rows
        if row["local_success_rate"] is not None
        and row["status"] in accepted_statuses
    ]
    statuses = sorted({str(row["status"]) for row in rows})
    return {
        "table": table,
        "scope": TABLE_SCOPES[table],
        "paper_average": PAPER_AVERAGES[table],
        "local_available_mean": (
            sum(float(row["local_success_rate"]) for row in available) / len(available)
            if available
            else None
        ),
        "local_cells_available": len(available),
        "total_cells": len(rows),
        "local_statuses": statuses,
    }


def build_document(
    records: Sequence[dict[str, Any]],
    warnings: Sequence[str],
    *,
    seed: int,
    episodes: int,
    horizon: int,
    release: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "dynamac-paper-comparison-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_reference": PAPER_REFERENCE,
        "selection": {
            "release": release,
            "seed": seed,
            "episodes": episodes,
            "horizon": horizon,
        },
        "status_definitions": {
            "local reproduction": "Same paper cell, independent local demonstrations/configuration.",
            "non-comparable diagnostic": "A local run whose intervention protocol is not the unpublished paper protocol.",
            "invalid diagnostic": "The requested local intervention did not produce a valid/effective protocol.",
            "pending": "No matching completed local result was found.",
            "stopped after 0% preliminary run": (
                "No 200-episode result is claimed; a deterministic local "
                "implementation issue was confirmed from a stopped preliminary run."
            ),
            "hardware unavailable": "Requires the paper's physical robot, perception, and demonstration setup.",
            "unavailable": "The public simulator path cannot execute this cell reliably.",
        },
        "table_summaries": [_table_summary(table, records) for table in ("I", "II", "III", "IV")],
        "records": list(records),
        "discovery_warnings": list(warnings),
    }


def _rate(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _difference(value: object) -> str:
    return "—" if value is None else f"{float(value):+.3f}"


def markdown(document: dict[str, Any]) -> str:
    selection = document["selection"]
    lines = [
        "# DynaMAC Paper Comparison",
        "",
        "Published values are transcribed from Tables I–IV of "
        "[DynaMAC (arXiv:2607.22119v1)](https://arxiv.org/abs/2607.22119). "
        "Local results use the selected complete files only: "
        + (
            f"release `{selection['release']}`, "
            if selection.get("release") is not None
            else ""
        )
        + f"seed `{selection['seed']}`, `{selection['episodes']}` episodes, horizon "
        f"`{selection['horizon']}`.",
        "",
        "Dynamic simulator values are retained as diagnostics, not claimed as paper "
        "reproductions: the task-motion and arm-perturbation defaults used for the "
        "paper are not published. Table IV requires physical hardware and is not run "
        "locally.",
        "",
        "## Overview",
        "",
        "| Paper table | Scope | Paper DynaMAC average | Available local mean | Coverage | Interpretation |",
        "|---|---|---:|---:|---:|---|",
    ]
    for summary in document["table_summaries"]:
        coverage = f"{summary['local_cells_available']}/{summary['total_cells']}"
        statuses = ", ".join(summary["local_statuses"]) or (
            "hardware unavailable" if summary["table"] == "IV" else "pending"
        )
        lines.append(
            f"| {summary['table']} | {summary['scope']} | "
            f"{summary['paper_average']:.2f} | {_rate(summary['local_available_mean'])} | "
            f"{coverage} | {statuses} |"
        )

    records = document["records"]
    for table in ("I", "II", "III", "IV"):
        lines.extend(
            [
                "",
                f"## Table {table}: {TABLE_SCOPES[table]}",
                "",
                "| Condition | Task | Paper DynaMAC | Local | Difference | Status | Protocol note | Evidence |",
                "|---|---|---:|---:|---:|---|---|---|",
            ]
        )
        for row in (item for item in records if item["table"] == table):
            evidence = "—"
            if row["source_file"]:
                evidence = f"[JSON]({row['source_file']})"
            lines.append(
                f"| {row['condition']} | {row['task']} | "
                f"{float(row['paper_success_rate']):.2f} | "
                f"{_rate(row['local_success_rate'])} | {_difference(row['difference'])} | "
                f"{row['status']} | {row.get('protocol_note') or row['notes']} | {evidence} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A numerical difference is descriptive only. Even the static local runs use "
            "independently sampled demonstrations and are not byte-for-byte reruns of the "
            "authors' data. Differences beside diagnostic rows must not be interpreted as "
            "reproduction gaps.",
            "",
        ]
    )
    if document["discovery_warnings"]:
        lines.extend(["## Ignored result files", ""])
        lines.extend(f"- {warning}" for warning in document["discovery_warnings"])
        lines.append("")
    return "\n".join(lines)


CSV_FIELDS = (
    "table",
    "condition",
    "task",
    "paper_success_rate",
    "local_success_rate",
    "difference",
    "local_successes",
    "local_episodes",
    "status",
    "source_file",
    "notes",
    "protocol_note",
)


def write_outputs(
    document: dict[str, Any],
    *,
    markdown_path: Path,
    csv_path: Path,
    json_path: Path,
) -> None:
    for path in (markdown_path, csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(document), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS}
            for row in document["records"]
        )
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help="Version directory below results-dir to select (default: v1).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "paper_comparison.md",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "paper_comparison.csv",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "paper_comparison.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0 or args.episodes < 1 or args.horizon < 1:
        raise ValueError("seed must be non-negative; episodes and horizon must be positive")
    records, warnings = build_records(
        args.results_dir,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        release=args.release,
    )
    document = build_document(
        records,
        warnings,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        release=args.release,
    )
    write_outputs(
        document,
        markdown_path=args.markdown_output,
        csv_path=args.csv_output,
        json_path=args.json_output,
    )
    print(markdown(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
