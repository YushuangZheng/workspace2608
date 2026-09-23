"""Calibration-only sensitivity of the frozen normal-task boundary threshold.

This analysis uses the immutable local-score trace that produced the frozen
``theta_local`` and ``H`` values.  For each global threshold multiplier it
recomputes pre-terminal false-ready runs and terminal-hold ready runs.  It does
not replay current source code and reads no failure, monitor-calibration, or
sealed-evaluation episode.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
CALIBRATION = (
    EVAL_ROOT
    / "artifacts"
    / "calibration"
    / "normal_task_boundaries"
    / "main10"
)
OUTPUT = EVAL_ROOT / "results" / "controlled" / "e5" / "derived"
MULTIPLIERS = (0.85, 0.925, 1.0, 1.075, 1.15)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
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


def _trial_runs(
    rows: Sequence[Mapping[str, str]], threshold: float
) -> tuple[int, int]:
    recorded = sorted(
        (
            row
            for row in rows
            if row["phase"] == "recorded"
            and not int(row["truth_in_terminal_window"])
        ),
        key=lambda row: int(row["tick"]),
    )
    held = sorted(
        (row for row in rows if row["phase"] == "terminal_hold"),
        key=lambda row: int(row["hold_cycle"]),
    )
    preterminal_run = _longest_true_run(
        bool(int(row["local_evidence_available"]))
        and float(row["local_score"]) > threshold
        for row in recorded
    )
    terminal_run = _longest_true_run(
        bool(int(row["local_evidence_available"]))
        and float(row["local_score"]) > threshold
        for row in held
    )
    return preterminal_run, terminal_run


def generate() -> dict[str, Any]:
    trace = _rows(CALIBRATION / "control_tick_trace.csv.gz")
    calibration_rows = _rows(CALIBRATION / "boundary_calibration.csv")
    frozen_trials = _rows(CALIBRATION / "normal_trials.csv")
    calibration = {
        (row["task"], row["arm"], row["boundary"]): row
        for row in calibration_rows
    }
    grouped: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        grouped[
            (
                row["task"],
                row["arm"],
                row["boundary"],
                int(row["demonstration"]),
            )
        ].append(row)
    if len(calibration) != 60 or len(grouped) != 300:
        raise RuntimeError(
            f"unexpected frozen coverage: {len(calibration)} boundaries, "
            f"{len(grouped)} boundary-demo instances"
        )

    summary_rows: list[dict[str, Any]] = []
    complete_rows: list[dict[str, Any]] = []
    for multiplier in MULTIPLIERS:
        by_boundary: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for key, members in sorted(grouped.items()):
            task, arm, boundary, demonstration = key
            frozen = calibration[(task, arm, boundary)]
            base_threshold = float(frozen["local_score_threshold"])
            threshold = base_threshold * multiplier
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"threshold multiplier {multiplier} leaves [0,1] for "
                    f"{task}/{arm}/{boundary}"
                )
            required = int(frozen["confirmation_cycles"])
            preterminal_run, terminal_run = _trial_runs(members, threshold)
            row = {
                "threshold_multiplier": multiplier,
                "task": task,
                "arm": arm,
                "boundary": boundary,
                "demonstration": demonstration,
                "base_local_score_threshold": base_threshold,
                "counterfactual_local_score_threshold": threshold,
                "confirmation_cycles": required,
                "maximum_preterminal_ready_run": preterminal_run,
                "terminal_hold_ready_run": terminal_run,
                "premature_local_completion": int(preterminal_run >= required),
                "normal_terminal_local_completion": int(terminal_run >= required),
            }
            complete_rows.append(row)
            by_boundary[(task, arm, boundary)].append(row)

        multiplier_rows = [
            row for row in complete_rows if row["threshold_multiplier"] == multiplier
        ]
        boundary_passes = sum(
            all(not row["premature_local_completion"] for row in rows)
            and all(row["normal_terminal_local_completion"] for row in rows)
            for rows in by_boundary.values()
        )
        summary_rows.append(
            {
                "threshold_multiplier": multiplier,
                "boundary_demo_instances": len(multiplier_rows),
                "premature_local_completion_instances": sum(
                    row["premature_local_completion"] for row in multiplier_rows
                ),
                "premature_local_completion_rate": mean(
                    row["premature_local_completion"] for row in multiplier_rows
                ),
                "normal_terminal_local_completion_instances": sum(
                    row["normal_terminal_local_completion"] for row in multiplier_rows
                ),
                "normal_terminal_local_completion_rate": mean(
                    row["normal_terminal_local_completion"] for row in multiplier_rows
                ),
                "boundaries_passing_both_criteria": boundary_passes,
                "boundaries": len(by_boundary),
                "boundary_pass_rate": boundary_passes / len(by_boundary),
            }
        )

    # The 1.0 point must exactly regenerate the two frozen run-length tables.
    baseline = {
        (row["task"], row["arm"], row["boundary"], int(row["demonstration"])): row
        for row in complete_rows
        if row["threshold_multiplier"] == 1.0
    }
    for expected in frozen_trials:
        key = (
            expected["task"],
            expected["arm"],
            expected["boundary"],
            int(expected["demonstration"]),
        )
        actual = baseline.get(key)
        if actual is None or any(
            int(actual[field]) != int(expected[field])
            for field in (
                "maximum_preterminal_ready_run",
                "terminal_hold_ready_run",
            )
        ):
            raise RuntimeError(f"frozen-point run-length mismatch at {key}")
    baseline_summary = next(
        row for row in summary_rows if row["threshold_multiplier"] == 1.0
    )
    if (
        baseline_summary["premature_local_completion_instances"] != 0
        or baseline_summary["normal_terminal_local_completion_instances"] != 300
        or baseline_summary["boundaries_passing_both_criteria"] != 60
    ):
        raise RuntimeError("frozen threshold no longer satisfies its calibration criteria")

    main_output = OUTPUT / "appendix_threshold_sensitivity.csv"
    complete_output = OUTPUT / "appendix_threshold_sensitivity_complete.csv"
    _atomic_csv(main_output, summary_rows)
    _atomic_csv(complete_output, complete_rows)
    summary = {
        "schema": "essay2608.iclr2027.a6-e5-threshold-sensitivity.v1",
        "status": "PASS",
        "calibration_only": True,
        "source_replay_used": False,
        "failure_episodes_read": 0,
        "monitor_calibration_episodes_read": 0,
        "sealed_episodes_read": 0,
        "boundaries": 60,
        "boundary_demo_instances": 300,
        "threshold_multipliers": list(MULTIPLIERS),
        "changed_parameter": "local_score_threshold",
        "unchanged": [
            "confirmation_cycles",
            "relation_and_scene_guards",
            "task_models",
            "motion_policy",
            "runtime_mechanisms",
        ],
        "frozen_inputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                CALIBRATION / "control_tick_trace.csv.gz",
                CALIBRATION / "boundary_calibration.csv",
                CALIBRATION / "normal_trials.csv",
            )
        },
        "outputs": {
            main_output.name: _sha256(main_output),
            complete_output.name: _sha256(complete_output),
        },
    }
    _atomic_json(OUTPUT / "E5_THRESHOLD_SENSITIVITY.json", summary)
    return summary


def main() -> int:
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
