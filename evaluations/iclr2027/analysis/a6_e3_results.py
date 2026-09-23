"""Aggregate E3 failure-budget and LOFO no-op shadow monitoring results."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from evaluations.iclr2027.analysis.rebuild_a5_audit import load_current_reaudit


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
A5_ROOT = EVAL_ROOT / "results" / "controlled" / "e1_e2"
E3_ROOT = EVAL_ROOT / "results" / "controlled" / "e3"
VIEW_ROOT = E3_ROOT / "manifest_views"
SHADOW_ROOT = E3_ROOT / "shadow"
DERIVED = E3_ROOT / "derived"
PLAN = E3_ROOT / "A6_E3_RUN_PLAN.json"
SEEDS = (1103, 2207, 3301)
BUDGETS = (20, 50, 100, 200)
FAMILIES = (
    "actuation_delay",
    "missed_interaction",
    "relation_loss",
    "environment_change",
    "coordination_delay",
)
ZERO_FAILURE_METHODS = {
    "m2": ("Trajectory-Likelihood", "standardized_nll"),
    "m3": ("FAIL-Detect", "logpzo"),
    "m6": ("Ours-Monitor", "task_state_mismatch"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gzip_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


def _index(path: Path) -> dict[str, Any]:
    value = _json(path)
    if not value.get("action_passthrough_verified"):
        raise RuntimeError(f"shadow action passthrough not verified: {path}")
    if value.get("source_files_modified") is not False:
        raise RuntimeError(f"shadow source modification status invalid: {path}")
    files = value.get("files")
    if not isinstance(files, list):
        raise RuntimeError(f"shadow file index missing: {path}")
    value["_by_episode"] = {str(item["episode_id"]): item for item in files}
    return value


def _entry_scores(index: Mapping[str, Any], episode_id: str, score_name: str) -> list[dict[str, Any]]:
    entry = index["_by_episode"].get(episode_id)
    if entry is None:
        raise RuntimeError(f"shadow episode missing from index: {episode_id}")
    if not entry.get("action_passthrough_verified"):
        raise RuntimeError(f"shadow action mismatch: {episode_id}")
    return _gzip_rows(ROOT / str(entry["score_path"]))


def _rising_edges(cycles: Iterable[int]) -> int:
    values = sorted(set(int(value) for value in cycles))
    return sum(index == 0 or value != values[index - 1] + 1 for index, value in enumerate(values))


_REAUDIT: dict[tuple[str, str], dict[str, Any]] | None = None


def _audit(episode_id: str) -> Mapping[str, Any]:
    global _REAUDIT
    if _REAUDIT is None:
        _REAUDIT = load_current_reaudit()
    key = ("m0", episode_id)
    if key not in _REAUDIT:
        raise RuntimeError(f"current M0 physical audit missing: {episode_id}")
    return _REAUDIT[key]


def _safe_metric(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else math.nan


def _evaluate(
    nominal_index: Mapping[str, Any],
    perturbed_index: Mapping[str, Any],
    nominal_manifest: Iterable[Mapping[str, Any]],
    perturbed_manifest: Iterable[Mapping[str, Any]],
    score_name: str,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    nominal_records = []
    for row in nominal_manifest:
        if task is not None and row["task"] != task:
            continue
        episode_id = str(row["episode_id"])
        scores = _entry_scores(nominal_index, episode_id, score_name)
        if not scores:
            continue
        alarms = [int(item["cycle"]) for item in scores if item["alarm"]]
        values = [float(item["scores"][score_name]) for item in scores]
        nominal_records.append(
            {
                "episode_id": episode_id,
                "predicted": bool(alarms),
                "score": max(values),
                "cycles": len(scores),
                "alarm_events": _rising_edges(alarms),
            }
        )

    positive_records = []
    scheduled_perturbed = 0
    for row in perturbed_manifest:
        if task is not None and row["task"] != task:
            continue
        scheduled_perturbed += 1
        episode_id = str(row["episode_id"])
        audit = _audit(episode_id)
        if not bool(audit.get("physically_triggered")):
            continue
        onset = audit.get("violation_onset_cycle")
        if onset is None:
            raise RuntimeError(f"triggered event has no onset: {episode_id}")
        onset = int(onset)
        scores = _entry_scores(perturbed_index, episode_id, score_name)
        if not scores:
            continue
        post = [item for item in scores if int(item["cycle"]) >= onset]
        valid_alarms = [int(item["cycle"]) for item in post if item["alarm"]]
        premature = [int(item["cycle"]) for item in scores if item["alarm"] and int(item["cycle"]) < onset]
        values = [float(item["scores"][score_name]) for item in post]
        positive_records.append(
            {
                "episode_id": episode_id,
                "detected": bool(valid_alarms),
                "premature_alarm": bool(premature),
                "score": max(values) if values else -math.inf,
                "delay": valid_alarms[0] - onset if valid_alarms else None,
            }
        )

    if not nominal_records or not positive_records:
        raise RuntimeError(f"empty monitoring population for task={task}")
    true_positive = sum(item["detected"] for item in positive_records)
    false_negative = len(positive_records) - true_positive
    nominal_false_positive = sum(item["predicted"] for item in nominal_records)
    premature_false_positive = sum(item["premature_alarm"] for item in positive_records)
    false_positive = nominal_false_positive + premature_false_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / len(positive_records)
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    labels = np.asarray([0] * len(nominal_records) + [1] * len(positive_records), dtype=np.int8)
    predictions = np.asarray(
        [int(item["predicted"]) for item in nominal_records]
        + [int(item["detected"]) for item in positive_records],
        dtype=np.int8,
    )
    raw_scores = [item["score"] for item in nominal_records + positive_records]
    finite = [float(value) for value in raw_scores if math.isfinite(float(value))]
    floor = min(finite) - 1.0 if finite else -1.0
    episode_scores = np.asarray(
        [float(value) if math.isfinite(float(value)) else floor for value in raw_scores],
        dtype=np.float64,
    )
    delays = [int(item["delay"]) for item in positive_records if item["delay"] is not None]
    nominal_cycles = sum(int(item["cycles"]) for item in nominal_records)
    false_interventions = sum(int(item["alarm_events"]) for item in nominal_records)
    return {
        "scheduled_nominal_episodes": len(list(nominal_manifest)) if task is None else sum(row["task"] == task for row in nominal_manifest),
        "scored_nominal_episodes": len(nominal_records),
        "scheduled_perturbed_episodes": scheduled_perturbed,
        "physically_triggered_events": len(positive_records),
        "excluded_untriggered_episodes": scheduled_perturbed - len(positive_records),
        "true_positive_events": true_positive,
        "false_positive_events": false_positive,
        "nominal_false_positive_episodes": nominal_false_positive,
        "premature_false_positive_events": premature_false_positive,
        "false_negative_events": false_negative,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": event_f1,
        "episode_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "episode_auprc": float(average_precision_score(labels, episode_scores)),
        "median_detection_delay_cycles": median(delays) if delays else math.nan,
        "nominal_false_interventions": false_interventions,
        "nominal_cycles": nominal_cycles,
        "false_interventions_per_1000_nominal_cycles": 1000.0 * false_interventions / nominal_cycles,
    }


def _task_rows(
    *,
    nominal_index: Mapping[str, Any],
    perturbed_index: Mapping[str, Any],
    nominal_manifest: list[dict[str, Any]],
    perturbed_manifest: list[dict[str, Any]],
    score_name: str,
    base: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = sorted({str(row["task"]) for row in perturbed_manifest})
    rows = []
    for task in tasks:
        rows.append(
            {
                **base,
                "aggregation": "task",
                "task": task,
                **_evaluate(
                    nominal_index,
                    perturbed_index,
                    nominal_manifest,
                    perturbed_manifest,
                    score_name,
                    task=task,
                ),
            }
        )
    pooled = {
        **base,
        "aggregation": "pooled",
        "task": "ALL",
        **_evaluate(
            nominal_index,
            perturbed_index,
            nominal_manifest,
            perturbed_manifest,
            score_name,
        ),
    }
    return rows, pooled


def _macro(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values = list(rows)
    fields = (
        "event_precision",
        "event_recall",
        "event_f1",
        "episode_balanced_accuracy",
        "episode_auprc",
        "median_detection_delay_cycles",
        "false_interventions_per_1000_nominal_cycles",
    )
    output = {}
    for field in fields:
        finite = [float(row[field]) for row in values if math.isfinite(float(row[field]))]
        output[field] = mean(finite) if finite else math.nan
    return output


def _figure_row(base: Mapping[str, Any], seed_macros: list[Mapping[str, float]]) -> dict[str, Any]:
    output = {**base, "training_seeds": len(seed_macros)}
    for field in (
        "event_precision",
        "event_recall",
        "event_f1",
        "episode_balanced_accuracy",
        "episode_auprc",
        "median_detection_delay_cycles",
        "false_interventions_per_1000_nominal_cycles",
    ):
        values = [float(row[field]) for row in seed_macros if math.isfinite(float(row[field]))]
        output[f"task_macro_{field}_mean"] = mean(values) if values else math.nan
        output[f"task_macro_{field}_seed_sd"] = pstdev(values) if len(values) > 1 else 0.0
    return output


def _a5_index(method: str, condition: str) -> Path:
    return A5_ROOT / "shadow" / method / condition / "score_index.json"


def _new_budget_index(budget: int, seed: int, condition: str) -> Path:
    return SHADOW_ROOT / "failure_budget" / f"budget_{budget}" / f"seed_{seed}" / condition / "score_index.json"


def _new_lofo_index(family: str, seed: int, condition: str) -> Path:
    return SHADOW_ROOT / "lofo" / family / f"seed_{seed}" / condition / "score_index.json"


def generate() -> dict[str, Any]:
    plan = _json(PLAN)
    if plan.get("schema") != "essay2608.iclr2027.a6-e3-shadow-plan.v2":
        raise RuntimeError("E3 shadow plan is not current")
    budget_nominal = _rows(VIEW_ROOT / "shadow_budget_nominal.jsonl")
    budget_perturbed = _rows(VIEW_ROOT / "shadow_budget_perturbed.jsonl")

    budget_complete: list[dict[str, Any]] = []
    budget_pooled: list[dict[str, Any]] = []
    budget_figure: list[dict[str, Any]] = []
    for budget in BUDGETS:
        seed_macros = []
        for seed in SEEDS:
            nominal_path = _a5_index(f"m4_seed{seed}", "nominal") if budget == 200 else _new_budget_index(budget, seed, "nominal")
            perturbed_path = _a5_index(f"m4_seed{seed}", "perturbed") if budget == 200 else _new_budget_index(budget, seed, "perturbed")
            base = {
                "experiment": "failure_budget_shadow",
                "method_key": "m4",
                "monitor": "Failure-Supervised Monitor",
                "failure_trajectories_per_task": budget,
                "training_seed": seed,
                "held_out_family": "",
            }
            task_rows, pooled = _task_rows(
                nominal_index=_index(nominal_path),
                perturbed_index=_index(perturbed_path),
                nominal_manifest=budget_nominal,
                perturbed_manifest=budget_perturbed,
                score_name="violation_probability",
                base=base,
            )
            budget_complete.extend(task_rows)
            budget_pooled.append(pooled)
            seed_macros.append(_macro(task_rows))
        budget_figure.append(
            _figure_row(
                {
                    "method_key": "m4",
                    "monitor": "Failure-Supervised Monitor",
                    "failure_trajectories_per_task": budget,
                },
                seed_macros,
            )
        )
    for method_key, (name, score_name) in ZERO_FAILURE_METHODS.items():
        base = {
            "experiment": "failure_budget_shadow",
            "method_key": method_key,
            "monitor": name,
            "failure_trajectories_per_task": 0,
            "training_seed": "frozen",
            "held_out_family": "",
        }
        task_rows, pooled = _task_rows(
            nominal_index=_index(_a5_index(method_key, "nominal")),
            perturbed_index=_index(_a5_index(method_key, "perturbed")),
            nominal_manifest=budget_nominal,
            perturbed_manifest=budget_perturbed,
            score_name=score_name,
            base=base,
        )
        budget_complete.extend(task_rows)
        budget_pooled.append(pooled)
        budget_figure.append(
            _figure_row(
                {
                    "method_key": method_key,
                    "monitor": name,
                    "failure_trajectories_per_task": 0,
                },
                [_macro(task_rows)],
            )
        )

    lofo_complete: list[dict[str, Any]] = []
    lofo_pooled: list[dict[str, Any]] = []
    lofo_figure: list[dict[str, Any]] = []
    for family in FAMILIES:
        nominal_manifest = _rows(VIEW_ROOT / f"shadow_lofo_{family}_nominal.jsonl")
        perturbed_manifest = _rows(VIEW_ROOT / f"shadow_lofo_{family}_perturbed.jsonl")
        identities = [
            (
                "m4_lofo",
                "Failure-Supervised Monitor (held-out family)",
                "violation_probability",
                seed,
                _new_lofo_index(family, seed, "nominal"),
                _new_lofo_index(family, seed, "perturbed"),
            )
            for seed in SEEDS
        ] + [
            (
                "m4_all_family",
                "Failure-Supervised Monitor (all families)",
                "violation_probability",
                seed,
                _a5_index(f"m4_seed{seed}", "nominal"),
                _a5_index(f"m4_seed{seed}", "perturbed"),
            )
            for seed in SEEDS
        ] + [
            (
                method_key,
                name,
                score_name,
                "frozen",
                _a5_index(method_key, "nominal"),
                _a5_index(method_key, "perturbed"),
            )
            for method_key, (name, score_name) in ZERO_FAILURE_METHODS.items()
        ]
        macros: dict[tuple[str, str], list[dict[str, float]]] = {}
        labels: dict[tuple[str, str], str] = {}
        for method_key, name, score_name, seed, nominal_path, perturbed_path in identities:
            base = {
                "experiment": "lofo_shadow",
                "method_key": method_key,
                "monitor": name,
                "failure_trajectories_per_task": 200 if method_key.startswith("m4") else 0,
                "training_seed": seed,
                "held_out_family": family,
            }
            task_rows, pooled = _task_rows(
                nominal_index=_index(nominal_path),
                perturbed_index=_index(perturbed_path),
                nominal_manifest=nominal_manifest,
                perturbed_manifest=perturbed_manifest,
                score_name=score_name,
                base=base,
            )
            lofo_complete.extend(task_rows)
            lofo_pooled.append(pooled)
            key = (method_key, name)
            macros.setdefault(key, []).append(_macro(task_rows))
            labels[key] = name
        for (method_key, name), seed_macros in macros.items():
            lofo_figure.append(
                _figure_row(
                    {
                        "held_out_family": family,
                        "method_key": method_key,
                        "monitor": name,
                        "failure_trajectories_per_task": 200 if method_key.startswith("m4") else 0,
                        "eligible_tasks": len({row["task"] for row in lofo_complete if row["held_out_family"] == family}),
                    },
                    seed_macros,
                )
            )

    outputs = {
        "fig4_failure_budget.csv": budget_figure,
        "appendix_failure_budget_monitoring.csv": budget_complete + budget_pooled,
        "fig4_leave_one_family_out.csv": lofo_figure,
        "appendix_heldout_family_monitoring.csv": lofo_complete + lofo_pooled,
    }
    for name, rows in outputs.items():
        _atomic_csv(DERIVED / name, rows)
    summary = {
        "schema": "essay2608.iclr2027.a6-e3-shadow-analysis.v2",
        "status": "PASS",
        "new_physical_episodes": 0,
        "new_shadow_replays": int(plan["expected_shadow_replays"]["total"]),
        "reused_a5_shadow_source": True,
        "task_success_used_for_budget_or_lofo_claim": False,
        "primary_population": "physically triggered events plus fixed nominal negatives",
        "outputs": list(outputs),
        "output_sha256": {name: _sha256(DERIVED / name) for name in outputs},
    }
    _atomic_json(DERIVED / "E3_AB_ANALYSIS.json", summary)
    return summary


def main() -> int:
    result = generate()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
