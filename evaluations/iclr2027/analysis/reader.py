"""Uniform result reader; no values are copied by hand into paper tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evaluations.iclr2027.interfaces.feature_schema import EPISODE_SCHEMA


def read_results(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for path in sorted((Path(root) / "episodes").glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema") != EPISODE_SCHEMA:
                raise ValueError("unknown result schema: %s" % path)
            rows.append(value)
    return rows


def summarize_results(roots: Iterable[Path]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in read_results(roots):
        key = (row["split"], row["task"], row["method_id"], row["condition"])
        groups[key].append(row)
    result = []
    for key, values in sorted(groups.items()):
        success = sum(bool(value["success"]) for value in values)
        triggered = sum(
            bool(value.get("audit", {}).get("physically_triggered"))
            for value in values
        )
        eligible = sum(
            bool(value.get("audit", {}).get("eligible")) for value in values
        )
        result.append(
            {
                "split": key[0],
                "task": key[1],
                "method_id": key[2],
                "condition": key[3],
                "episodes": len(values),
                "successes": success,
                "success_rate": success / len(values),
                "eligible": eligible,
                "physically_triggered": triggered,
                "trigger_rate_given_eligible": (
                    triggered / eligible if eligible else None
                ),
                "mean_cycles": sum(value["cycles"] for value in values)
                / len(values),
            }
        )
    return result


def write_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["read_results", "summarize_results", "write_summary_csv"]
