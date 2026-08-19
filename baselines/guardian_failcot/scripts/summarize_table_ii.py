#!/usr/bin/env python3
"""Rebuild the Guardian Table-II comparison input from official JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Cell:
    benchmark: str
    mode: str
    relative_output: str
    expected_n: int
    paper_accuracy: float


CELLS = (
    Cell(
        "RoboFail",
        "execution",
        "execution_thinking/DATASET_robofail.jsonl",
        153,
        0.86,
    ),
    Cell(
        "RoboFail",
        "planning",
        "planning_thinking/DATASET_robofail.jsonl",
        30,
        0.70,
    ),
    Cell(
        "UR5-Fail",
        "execution",
        "execution_thinking/DATASET_ur5fail_test.jsonl",
        140,
        0.77,
    ),
    Cell(
        "UR5-Fail",
        "planning",
        "planning_thinking/DATASET_ur5fail_test.jsonl",
        140,
        0.89,
    ),
    Cell(
        "RoboVQA",
        "execution",
        "execution_thinking/DATASET_robovqa.jsonl",
        357,
        0.85,
    ),
)


def parse_output(path: Path, expected_n: int) -> tuple[int, int]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            correct = record.get("correct")
            if correct not in (0, 1, False, True):
                raise ValueError(
                    f"{path}:{line_number}: correct must be binary, got {correct!r}"
                )
            records.append(record)

    if len(records) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} rows, found {len(records)}")
    ids = [record.get("id") for record in records]
    if len(set(ids)) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} unique ids")
    return len(records), sum(int(bool(record["correct"])) for record in records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=BASELINE_ROOT / "results/table_ii/guardian-thinking",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASELINE_ROOT / "results/comparison/comparison.csv",
    )
    args = parser.parse_args()

    rows = []
    for cell in CELLS:
        total, correct = parse_output(
            args.results_root / cell.relative_output, cell.expected_n
        )
        reproduced = correct / total
        rows.append(
            {
                "benchmark": cell.benchmark,
                "mode": cell.mode,
                "n": total,
                "paper_accuracy": f"{cell.paper_accuracy:.6f}",
                "reproduced_accuracy": f"{reproduced:.10f}",
                "correct": correct,
                "delta_pp": f"{100.0 * (reproduced - cell.paper_accuracy):.6f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    os.replace(temporary, args.output)

    print(
        f"Wrote {len(rows)} verified Table-II cells "
        f"({sum(int(row['n']) for row in rows)} rows) to {args.output}"
    )


if __name__ == "__main__":
    main()
