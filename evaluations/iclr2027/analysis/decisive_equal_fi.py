"""Independent-calibration monitoring comparison at fixed FI budgets.

Operating points are selected only from the disjoint normal-calibration
split.  The retained test set is then used once to report the actual false
interventions (FI) per 1000 nominal cycles, event recall, missed-event rate,
and detection delay conditional on detection.  PR curves remain descriptive;
they are never used to select a reported operating point.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import precision_recall_curve

from evaluations.iclr2027.analysis.rebuild_a5_audit import load_current_reaudit
from evaluations.iclr2027.methods.fail_detect.conformal import (
    TimeVaryingConformalBand,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
A5 = EVAL / "results" / "controlled" / "e1_e2"
DECISIVE = EVAL / "results" / "reviewer_decisive" / "equal_fi"
DERIVED = DECISIVE / "derived"
M6_SHADOW = DECISIVE / "shadow" / "m6"
TARGET_FI = (0.5, 1.0, 2.0)
PRIMARY_TARGET_FI = 1.0
THRESHOLD_MULTIPLIER_GRID = tuple(
    float(value) for value in np.geomspace(0.05, 20.0, 401)
)


@dataclass(frozen=True)
class MonitorSpec:
    key: str
    label: str
    score_name: str
    calibration: Path | None
    calibration_index: Path
    nominal_index: Path
    perturbed_index: Path
    inclusive: bool = False


SPECS = (
    MonitorSpec(
        "m2",
        "Trajectory-Likelihood",
        "standardized_nll",
        EVAL / "artifacts/calibration/monitors/m2/v1/calibration.json",
        EVAL / "artifacts/calibration/monitors/m2/v1/score_index.json",
        A5 / "shadow/m2/nominal/score_index.json",
        A5 / "shadow/m2/perturbed/score_index.json",
    ),
    MonitorSpec(
        "m3",
        "FAIL-Detect",
        "logpzo",
        EVAL / "artifacts/calibration/monitors/m3/v1/calibration.json",
        EVAL / "artifacts/calibration/monitors/m3/v1/score_index.json",
        A5 / "shadow/m3/nominal/score_index.json",
        A5 / "shadow/m3/perturbed/score_index.json",
    ),
    *tuple(
        MonitorSpec(
            f"m4_seed{seed}",
            f"Supervised GRU [{seed}]",
            "violation_probability",
            EVAL
            / f"artifacts/calibration/monitors/m4/seed_{seed}/v1/calibration.json",
            EVAL
            / f"artifacts/calibration/monitors/m4/seed_{seed}/v1/score_index.json",
            A5 / f"shadow/m4_seed{seed}/nominal/score_index.json",
            A5 / f"shadow/m4_seed{seed}/perturbed/score_index.json",
        )
        for seed in (1103, 2207, 3301)
    ),
    MonitorSpec(
        "m6",
        "TSF-Monitor",
        "task_state_mismatch",
        None,
        M6_SHADOW / "calibration/score_index.json",
        M6_SHADOW / "nominal/score_index.json",
        M6_SHADOW / "perturbed/score_index.json",
        inclusive=True,
    ),
)

_CALIBRATION_TRACE_CACHE: dict[
    tuple[MonitorSpec, frozenset[str] | None],
    dict[str, list[tuple[np.ndarray, np.ndarray]]],
] = {}
_TEST_TRACE_CACHE: dict[
    tuple[MonitorSpec, str, frozenset[str] | None],
    list[dict[str, Any]],
] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _gzip_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _persistence(spec: MonitorSpec, task: str) -> int:
    if spec.calibration is None:
        # The continuous score is already a counter/native-threshold ratio.
        return 1
    assert spec.calibration is not None
    return int(_json(spec.calibration)["tasks"][task]["persistence_cycles"])


@lru_cache(maxsize=None)
def _fixed_or_scheduled_threshold(
    calibration: Path,
    task: str,
) -> tuple[float | None, TimeVaryingConformalBand | None]:
    task_config = _json(calibration)["tasks"][task]
    if "threshold" in task_config:
        return float(task_config["threshold"]), None
    return None, TimeVaryingConformalBand.from_dict(
        task_config["threshold_schedule"]
    )


def _calibration_threshold(
    spec: MonitorSpec,
    task: str,
    row: Mapping[str, Any],
) -> float:
    if spec.calibration is None:
        return 1.0
    fixed, schedule = _fixed_or_scheduled_threshold(spec.calibration, task)
    if fixed is not None:
        return fixed
    assert schedule is not None
    return float(schedule.threshold(int(row["policy_step"])))


def _calibration_traces(
    spec: MonitorSpec,
    allowed_episode_ids: set[str] | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    population = (
        None if allowed_episode_ids is None else frozenset(allowed_episode_ids)
    )
    cache_key = (spec, population)
    if cache_key in _CALIBRATION_TRACE_CACHE:
        return _CALIBRATION_TRACE_CACHE[cache_key]
    index = _json(spec.calibration_index)
    traces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for entry in index["files"]:
        if allowed_episode_ids is not None and str(entry["episode_id"]) not in allowed_episode_ids:
            continue
        task = str(entry["task"])
        score_path = entry.get("path", entry.get("score_path"))
        if score_path is None:
            raise KeyError(f"calibration index row has no score path: {entry}")
        rows = _gzip_rows(ROOT / str(score_path))
        normalized = []
        available = []
        for row in rows:
            raw = (
                float(row["scores"][spec.score_name])
                if "scores" in row
                else float(row["score"])
            )
            base = _calibration_threshold(spec, task, row)
            if not math.isfinite(base) or base <= 0.0:
                raise ValueError(f"invalid native threshold: {spec.key}/{task}")
            normalized.append(raw / base)
            available.append(bool(row.get("available", True)))
        traces.setdefault(task, []).append(
            (
                np.asarray(normalized, dtype=np.float64),
                np.asarray(available, dtype=np.bool_),
            )
        )
    _CALIBRATION_TRACE_CACHE[cache_key] = traces
    return traces


def _alarm_mask(
    values: np.ndarray,
    available: np.ndarray,
    threshold: float,
    persistence: int,
    *,
    inclusive: bool,
) -> np.ndarray:
    exceeded = (values >= threshold) if inclusive else (values > threshold)
    exceeded &= available
    alarm = np.zeros(len(values), dtype=np.bool_)
    streak = 0
    for index, active in enumerate(exceeded):
        streak = streak + 1 if bool(active) else 0
        alarm[index] = streak >= persistence
    return alarm


def _rising_edges(mask: np.ndarray) -> int:
    if not len(mask):
        return 0
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))


def _select_thresholds(
    spec: MonitorSpec,
    target_fi: float,
    allowed_episode_ids: set[str] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    selected = {}
    audit = []
    for task, traces in sorted(_calibration_traces(spec, allowed_episode_ids).items()):
        persistence = _persistence(spec, task)
        total_cycles = sum(len(values) for values, _ in traces)
        choice = math.inf
        choice_fi = 0.0
        for multiplier in (*THRESHOLD_MULTIPLIER_GRID, math.inf):
            interventions = sum(
                _rising_edges(
                    _alarm_mask(
                        values,
                        available,
                        multiplier,
                        persistence,
                        inclusive=spec.inclusive,
                    )
                )
                for values, available in traces
            )
            rate = 1000.0 * interventions / total_cycles
            if rate <= target_fi:
                choice = multiplier
                choice_fi = rate
                break
        selected[task] = choice
        audit.append(
            {
                "method_key": spec.key,
                "monitor": spec.label,
                "target_fi_per_1000_cycles": target_fi,
                "task": task,
                "threshold_multiplier": choice,
                "persistence_cycles": persistence,
                "calibration_episodes": len(traces),
                "calibration_cycles": total_cycles,
                "calibration_false_interventions_per_1000_cycles": choice_fi,
            }
        )
    return selected, audit


def _test_traces(
    spec: MonitorSpec,
    condition: str,
    allowed_episode_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    population = (
        None if allowed_episode_ids is None else frozenset(allowed_episode_ids)
    )
    cache_key = (spec, condition, population)
    if cache_key in _TEST_TRACE_CACHE:
        return _TEST_TRACE_CACHE[cache_key]
    index_path = spec.nominal_index if condition == "nominal" else spec.perturbed_index
    index = _json(index_path)
    traces = []
    for entry in index["files"]:
        episode_id = str(entry["episode_id"])
        if allowed_episode_ids is not None and episode_id not in allowed_episode_ids:
            continue
        task = str(entry["task"])
        rows = _gzip_rows(ROOT / str(entry["score_path"]))
        values = []
        available = []
        for row in rows:
            native = row.get("threshold")
            if native is None:
                if spec.calibration is None:
                    native = 1.0
                else:
                    raise RuntimeError(
                        f"retained score lacks native threshold: {spec.key}/{episode_id}"
                    )
            values.append(float(row["scores"][spec.score_name]) / float(native))
            row_available = row.get("available")
            if row_available is None and spec.key == "m2":
                row_available = float(
                    row["scores"].get("available_streams", 0.0)
                ) > 0.0
            available.append(bool(True if row_available is None else row_available))
        value_array = np.asarray(values, dtype=np.float64)
        available_array = np.asarray(available, dtype=np.bool_)
        traces.append(
            {
                "episode_id": episode_id,
                "task": task,
                "rows": rows,
                "values": value_array,
                "available": available_array,
            }
        )
    _TEST_TRACE_CACHE[cache_key] = traces
    return traces


def _test_records(
    spec: MonitorSpec,
    condition: str,
    thresholds: Mapping[str, float],
    audits: Mapping[tuple[str, str], Mapping[str, Any]],
    allowed_episode_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for trace in _test_traces(spec, condition, allowed_episode_ids):
        episode_id = str(trace["episode_id"])
        task = str(trace["task"])
        rows = trace["rows"]
        value_array = trace["values"]
        available_array = trace["available"]
        alarm = _alarm_mask(
            value_array,
            available_array,
            thresholds[task],
            _persistence(spec, task),
            inclusive=spec.inclusive,
        )
        alarm_cycles = [
            int(row["cycle"])
            for row, active in zip(rows, alarm)
            if bool(active)
        ]
        source_episode_path = (
            A5
            / "end_to_end"
            / "m0"
            / condition
            / "episodes"
            / f"{episode_id.replace('/', '__')}.json"
        )
        source_episode = _json(source_episode_path)
        if source_episode.get("task") != task:
            raise RuntimeError(f"source task mismatch: {spec.key}/{episode_id}")
        audit = audits.get(("m0", episode_id), {}) if condition == "perturbed" else {}
        onset = audit.get("violation_onset_cycle")
        triggered = bool(audit.get("physically_triggered", False))
        valid = (
            []
            if onset is None
            else [cycle for cycle in alarm_cycles if cycle >= int(onset)]
        )
        records.append(
            {
                "episode_id": episode_id,
                "task": task,
                "fault_family": source_episode.get("fault_family"),
                "cycles": len(rows),
                "scorable": bool(len(value_array)),
                "episode_score": float(np.max(value_array)) if len(value_array) else -math.inf,
                "alarm_rising_edges": _rising_edges(alarm),
                "triggered": triggered,
                "onset": onset,
                "detected": bool(triggered and valid),
                "first_valid_alarm": valid[0] if valid else None,
            }
        )
    return records


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def _metrics(
    spec: MonitorSpec,
    target: float,
    nominal: Sequence[Mapping[str, Any]],
    perturbed: Sequence[Mapping[str, Any]],
    *,
    population: str,
    family: str = "all",
) -> dict[str, Any]:
    nominal_scorable = [row for row in nominal if row["scorable"]]
    perturbed_scorable = [row for row in perturbed if row["scorable"]]
    triggered = [row for row in perturbed_scorable if row["triggered"]]
    detected = [row for row in triggered if row["detected"]]
    delays = [
        int(row["first_valid_alarm"]) - int(row["onset"])
        for row in detected
    ]
    interventions = sum(int(row["alarm_rising_edges"]) for row in nominal_scorable)
    cycles = sum(int(row["cycles"]) for row in nominal_scorable)
    low, high = _wilson(len(detected), len(triggered))
    return {
        "population": population,
        "fault_family": family,
        "method_key": spec.key,
        "monitor": spec.label,
        "target_calibration_fi_per_1000_cycles": target,
        "scheduled_nominal_episodes": len(nominal),
        "scored_nominal_episodes": len(nominal_scorable),
        "unscored_nominal_infrastructure_episodes": len(nominal) - len(nominal_scorable),
        "scheduled_perturbed_episodes": len(perturbed),
        "scored_perturbed_episodes": len(perturbed_scorable),
        "unscored_perturbed_infrastructure_episodes": len(perturbed) - len(perturbed_scorable),
        "test_actual_fi_per_1000_nominal_cycles": 1000.0 * interventions / cycles,
        "nominal_false_interventions": interventions,
        "nominal_cycles": cycles,
        "physically_triggered_events": len(triggered),
        "detected_events": len(detected),
        "event_recall": len(detected) / len(triggered) if triggered else math.nan,
        "event_recall_ci_low": low,
        "event_recall_ci_high": high,
        "undetected_events": len(triggered) - len(detected),
        "undetected_rate": 1.0 - len(detected) / len(triggered) if triggered else math.nan,
        "median_delay_detected_only_cycles": median(delays) if delays else math.nan,
        "delay_q1_detected_only_cycles": float(np.quantile(delays, 0.25)) if delays else math.nan,
        "delay_q3_detected_only_cycles": float(np.quantile(delays, 0.75)) if delays else math.nan,
        "delay_denominator_detected_events": len(delays),
    }


def _family_metrics(
    spec: MonitorSpec,
    target: float,
    nominal: Sequence[Mapping[str, Any]],
    perturbed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for family in sorted(
        {str(row["fault_family"]) for row in perturbed if row["fault_family"]}
    ):
        family_rows = [row for row in perturbed if row["fault_family"] == family]
        tasks = {str(row["task"]) for row in family_rows}
        family_nominal = [row for row in nominal if row["task"] in tasks]
        result.append(
            _metrics(
                spec,
                target,
                family_nominal,
                family_rows,
                population="main10_by_family",
                family=family,
            )
        )
    return result


def _pr_rows(
    spec: MonitorSpec,
    nominal: Sequence[Mapping[str, Any]],
    perturbed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    nominal = [
        row
        for row in nominal
        if row["scorable"] and math.isfinite(float(row["episode_score"]))
    ]
    perturbed = [
        row
        for row in perturbed
        if row["scorable"]
        and math.isfinite(float(row["episode_score"]))
    ]
    labels = np.asarray(
        [0] * len(nominal) + [int(row["triggered"]) for row in perturbed],
        dtype=np.int8,
    )
    scores = np.asarray(
        [row["episode_score"] for row in (*nominal, *perturbed)],
        dtype=np.float64,
    )
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return [
        {
            "method_key": spec.key,
            "monitor": spec.label,
            "normalized_threshold": float(threshold),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "selected_for_operating_point": False,
        }
        for index, threshold in enumerate(thresholds)
    ]


def generate() -> dict[str, Any]:
    missing = [
        str(path.relative_to(ROOT))
        for spec in SPECS
        for path in (
            spec.calibration_index,
            spec.nominal_index,
            spec.perturbed_index,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing equal-FI inputs: " + ", ".join(missing))
    audits = load_current_reaudit()
    calibration_rows = []
    result_rows = []
    family_rows = []
    pr_rows = []
    for spec in SPECS:
        primary_nominal = None
        primary_perturbed = None
        for target in TARGET_FI:
            thresholds, selected = _select_thresholds(spec, target)
            calibration_rows.extend(selected)
            nominal = _test_records(spec, "nominal", thresholds, audits)
            perturbed = _test_records(spec, "perturbed", thresholds, audits)
            result_rows.append(
                _metrics(
                    spec,
                    target,
                    nominal,
                    perturbed,
                    population="main10",
                )
            )
            family_rows.extend(_family_metrics(spec, target, nominal, perturbed))
            if target == PRIMARY_TARGET_FI:
                primary_nominal = nominal
                primary_perturbed = perturbed
        assert primary_nominal is not None and primary_perturbed is not None
        pr_rows.extend(_pr_rows(spec, primary_nominal, primary_perturbed))

    _atomic_csv(DERIVED / "calibration_operating_points.csv", calibration_rows)
    _atomic_csv(DERIVED / "main10_recall_at_fixed_fi.csv", result_rows)
    _atomic_csv(DERIVED / "main10_recall_at_fixed_fi_by_family.csv", family_rows)
    _atomic_csv(DERIVED / "main10_pr_curve.csv", pr_rows)
    protocol = {
        "schema": "essay2608.iclr2027.decisive-equal-fi-analysis.v1",
        "status": "PASS",
        "primary_target_fi_per_1000_nominal_cycles": PRIMARY_TARGET_FI,
        "sensitivity_target_fi_per_1000_nominal_cycles": list(TARGET_FI),
        "threshold_selection_split": "disjoint normal calibration only",
        "test_threshold_selection_forbidden": True,
        "test_reports_actual_not_target_fi": True,
        "delay_population": "detected physically-triggered events only",
        "missed_events_reported_separately": True,
        "pr_curve_used_for_threshold_selection": False,
        "threshold_multiplier_grid": {
            "kind": "geometric",
            "minimum": THRESHOLD_MULTIPLIER_GRID[0],
            "maximum": THRESHOLD_MULTIPLIER_GRID[-1],
            "count": len(THRESHOLD_MULTIPLIER_GRID),
        },
        "inputs": {
            spec.key: {
                "calibration_index": str(spec.calibration_index.relative_to(ROOT)),
                "calibration_index_sha256": _sha256(spec.calibration_index),
                "nominal_index": str(spec.nominal_index.relative_to(ROOT)),
                "nominal_index_sha256": _sha256(spec.nominal_index),
                "perturbed_index": str(spec.perturbed_index.relative_to(ROOT)),
                "perturbed_index_sha256": _sha256(spec.perturbed_index),
            }
            for spec in SPECS
        },
        "outputs": {
            path.name: _sha256(path)
            for path in (
                DERIVED / "calibration_operating_points.csv",
                DERIVED / "main10_recall_at_fixed_fi.csv",
                DERIVED / "main10_recall_at_fixed_fi_by_family.csv",
                DERIVED / "main10_pr_curve.csv",
            )
        },
    }
    _atomic_json(DERIVED / "ANALYSIS.json", protocol)
    return protocol


def status() -> dict[str, Any]:
    inputs = {
        spec.key: {
            "calibration": spec.calibration_index.is_file(),
            "nominal": spec.nominal_index.is_file(),
            "perturbed": spec.perturbed_index.is_file(),
        }
        for spec in SPECS
    }
    return {
        "ready": all(all(value.values()) for value in inputs.values()),
        "inputs": inputs,
        "analysis_exists": (DERIVED / "ANALYSIS.json").is_file(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "status"))
    args = parser.parse_args(argv)
    result = generate() if args.command == "generate" else status()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
