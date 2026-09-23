"""Generate the complete current E1 success and infrastructure tables (M0--M6)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "evaluations" / "iclr2027" / "results" / "controlled" / "e1_e2"
ENDPOINT = BASE / "end_to_end"
DERIVED = BASE / "derived"
REPORT = DERIVED / "A6_E1_COMPLETE_RESULTS.json"
MARKDOWN = DERIVED / "A6_E1_COMPLETE_RESULTS.md"
TASKS = (
    ("bimanual_handover_item", "Handover Item"),
    ("bimanual_lift_tray", "Lift Tray"),
    ("bimanual_put_bottle_in_fridge", "Put Bottle in Fridge"),
    ("bimanual_sweep_to_dustpan", "Bimanual Sweep to Dustpan"),
    ("close_jar", "Close Jar"),
    ("insert_onto_square_peg", "Insert Onto Square Peg"),
    ("open_drawer", "Open Drawer"),
    ("place_cups_3", "Place Cups/3"),
    ("stack_cups", "Stack Cups"),
    ("sweep_to_dustpan", "Sweep to Dustpan"),
)
METHODS = (
    ("m0", ("m0",), "M0"),
    ("m1", ("m1",), "M1"),
    ("m2", ("m2",), "M2"),
    ("m3", ("m3",), "M3"),
    ("m4", ("m4_seed1103", "m4_seed2207", "m4_seed3301"), "M4 (3 seeds)"),
    ("m5", ("m5",), "M5"),
    ("m6", ("m6",), "M6"),
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_cell(methods: tuple[str, ...], condition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        root = ENDPOINT / method / condition
        episodes = [_json(path) for path in sorted((root / "episodes").glob("*.json"))]
        if len(episodes) != 2000:
            raise RuntimeError(f"E1 endpoint is not complete: {method}/{condition}")
        counts = Counter(str(row["task"]) for row in episodes)
        if set(counts) != {task for task, _name in TASKS} or set(counts.values()) != {200}:
            raise RuntimeError(f"E1 task population is invalid: {method}/{condition}")
        rows.extend(episodes)
    return rows


def _format_cell(successes: int, episodes: int) -> str:
    return f"{successes}/{episodes} ({100.0 * successes / episodes:.1f}%)"


def generate() -> dict[str, Any]:
    output: dict[str, Any] = {}
    md = [
        "# E1 complete results (current accepted endpoint)",
        "",
        "M4 aggregates the three frozen training seeds; its denominator is therefore 600 per task and 6,000 overall.",
        "",
    ]
    for condition in ("nominal", "perturbed"):
        by_method: dict[str, list[dict[str, Any]]] = {}
        method_records: dict[str, Any] = {}
        for key, members, label in METHODS:
            rows = _load_cell(members, condition)
            by_method[key] = rows
            method_records[key] = {
                "label": label,
                "members": list(members),
                "episodes": len(rows),
                "successes": sum(bool(row.get("final_success")) for row in rows),
                "infrastructure_errors": sum(
                    row.get("termination_reason") == "infrastructure_error" for row in rows
                ),
                "tasks": {},
            }
            for task, name in TASKS:
                task_rows = [row for row in rows if row["task"] == task]
                method_records[key]["tasks"][task] = {
                    "name": name,
                    "episodes": len(task_rows),
                    "successes": sum(bool(row.get("final_success")) for row in task_rows),
                    "infrastructure_errors": sum(
                        row.get("termination_reason") == "infrastructure_error"
                        for row in task_rows
                    ),
                }
        output[condition] = method_records

        md.extend(
            [
                f"## {condition.title()}",
                "",
                "| Task | " + " | ".join(label for _key, _members, label in METHODS) + " |",
                "|--|" + "--:|" * len(METHODS),
            ]
        )
        for task, name in TASKS:
            cells = []
            for key, _members, _label in METHODS:
                record = method_records[key]["tasks"][task]
                cells.append(_format_cell(record["successes"], record["episodes"]))
            md.append("| " + name + " | " + " | ".join(cells) + " |")
        totals = []
        for key, _members, _label in METHODS:
            record = method_records[key]
            totals.append(_format_cell(record["successes"], record["episodes"]))
        md.append("| **Overall** | " + " | ".join(f"**{cell}**" for cell in totals) + " |")
        md.extend(["", "Infrastructure errors:", ""])
        md.append(
            "- "
            + "; ".join(
                f"{label}: {method_records[key]['infrastructure_errors']}/{method_records[key]['episodes']}"
                for key, _members, label in METHODS
            )
        )
        md.append("")

    report = {
        "schema": "essay2608.iclr2027.a6-e1-complete-results.v1",
        "status": "PASS",
        "m4_aggregates_three_training_seeds": True,
        "conditions": output,
    }
    _atomic_json(REPORT, report)
    MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    temporary = MARKDOWN.with_name(MARKDOWN.name + ".tmp")
    temporary.write_text("\n".join(md) + "\n", encoding="utf-8")
    os.replace(temporary, MARKDOWN)
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
