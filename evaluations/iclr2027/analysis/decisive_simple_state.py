"""Analyze the paired Full TSF versus simple-state controller experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from evaluations.iclr2027.runners.decisive_simple_state import (
    EXPECTED,
    METHODS,
    PLAN,
    RESULT_ROOT,
    ROOT,
    STRESS4,
    _assert_frozen,
    _json,
)
from evaluations.iclr2027.runners.episode_io import load_cycles, resolve_cycle_file


DERIVED = RESULT_ROOT / "derived"
RELATION_FAULTS = {"missed_interaction", "relation_loss"}
TEMPORAL_FAULTS = {"actuation_delay", "coordination_delay"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cell(method: str, condition: str) -> dict[tuple[str, int], dict[str, Any]]:
    root = RESULT_ROOT / method / condition / "episodes"
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
    ]
    if len(episodes) != EXPECTED[condition]:
        raise RuntimeError(
            f"incomplete simple-state cell {method}/{condition}: {len(episodes)}"
        )
    result = {}
    for episode in episodes:
        pair = (str(episode["task"]), int(episode["seed"]))
        if pair in result:
            raise RuntimeError(f"duplicate paired episode: {method}/{condition}/{pair}")
        result[pair] = episode
    return result


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def _paired_row(
    condition: str,
    population: str,
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    full = np.asarray([bool(pair[0]["success"]) for pair in pairs], dtype=np.int8)
    simple = np.asarray([bool(pair[1]["success"]) for pair in pairs], dtype=np.int8)
    difference = full.astype(np.float64) - simple.astype(np.float64)
    full_successes = int(np.sum(full))
    simple_successes = int(np.sum(simple))
    full_low, full_high = _wilson(full_successes, len(pairs))
    simple_low, simple_high = _wilson(simple_successes, len(pairs))
    full_only = int(np.sum((full == 1) & (simple == 0)))
    simple_only = int(np.sum((full == 0) & (simple == 1)))
    discordant = full_only + simple_only
    mean = float(np.mean(difference))
    standard_error = (
        float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
        if len(difference) > 1
        else math.nan
    )
    return {
        "condition": condition,
        "population": population,
        "paired_episodes": len(pairs),
        "full_successes": full_successes,
        "full_success_rate": full_successes / len(pairs),
        "full_success_ci_low": full_low,
        "full_success_ci_high": full_high,
        "simple_successes": simple_successes,
        "simple_success_rate": simple_successes / len(pairs),
        "simple_success_ci_low": simple_low,
        "simple_success_ci_high": simple_high,
        "paired_risk_difference_full_minus_simple": mean,
        "paired_difference_ci_low": max(-1.0, mean - 1.959963984540054 * standard_error),
        "paired_difference_ci_high": min(1.0, mean + 1.959963984540054 * standard_error),
        "difference_ci_method": "paired Wald; episode-level differences",
        "full_only_success": full_only,
        "simple_only_success": simple_only,
        "exact_mcnemar_p": (
            float(binomtest(full_only, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
    }


def _audit(episode: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = episode.get("audit", {})
    if audit.get("schema") == "essay2608.iclr2027.physical-event-audit.v1":
        return audit
    if (
        episode.get("termination_reason") == "infrastructure_error"
        and not bool(audit.get("physically_triggered"))
    ):
        return {"physically_triggered": False}
    raise RuntimeError(f"episode lacks current physical audit: {episode['episode_id']}")


def _path_record(
    method: str,
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _audit(episode)
    triggered = bool(audit.get("physically_triggered"))
    onset = audit.get("violation_onset_cycle")
    alarm_cycles = []
    if triggered:
        episode_path = (
            RESULT_ROOT
            / method
            / "perturbed"
            / "episodes"
            / f"{str(episode['episode_id']).replace('/', '__')}.json"
        )
        cycle_path = resolve_cycle_file(episode_path, episode)
        for row in load_cycles(cycle_path):
            alarm = (
                row.get("feature", {})
                .get("policy_state", {})
                .get("monitor", {})
                .get("alarm", False)
            )
            if alarm and onset is not None and int(row["cycle"]) >= int(onset):
                alarm_cycles.append(int(row["cycle"]))
    detected = bool(alarm_cycles)
    first_alarm = alarm_cycles[0] if alarm_cycles else None
    family = episode.get("fault_family")
    if family in RELATION_FAULTS:
        restored_cycle = audit.get("relation_restored_cycle")
    elif family in TEMPORAL_FAULTS:
        restored_cycle = audit.get("violation_end_cycle")
    else:
        restored_cycle = episode.get("legal_reentry_cycle")
        if restored_cycle is None and episode.get("success"):
            restored_cycle = episode.get("cycles")
    restored = bool(
        detected
        and restored_cycle is not None
        and int(restored_cycle) >= int(first_alarm)
    )
    reentry_cycle = episode.get("legal_reentry_cycle")
    reentered = bool(restored and reentry_cycle is not None)
    succeeded = bool(episode["success"])
    if not triggered:
        category = "not_physically_triggered"
    elif not detected:
        category = "triggered_undetected"
    elif not restored:
        category = "detected_not_restored"
    elif not reentered:
        category = "restored_not_reentered"
    elif succeeded:
        category = "reentered_final_success"
    else:
        category = "reentered_final_failure"
    return {
        "physically_triggered": triggered,
        "detected": detected,
        "condition_restored_after_detection": restored,
        "legal_reentry_after_restoration": reentered,
        "final_success": succeeded,
        "path_category": category,
    }


def generate() -> dict[str, Any]:
    if not PLAN.is_file():
        raise FileNotFoundError("run plan is not frozen; execute prepare first")
    plan = _json(PLAN)
    _assert_frozen(plan)
    cells = {
        (method, condition): _cell(method, condition)
        for method in METHODS
        for condition in EXPECTED
    }
    paired_rows = []
    infrastructure_sensitivity_rows = []
    for condition in EXPECTED:
        full = cells[("full", condition)]
        simple = cells[("simple_state", condition)]
        if set(full) != set(simple):
            raise RuntimeError(f"unpaired method cells: {condition}")
        ordered = sorted(full)
        paired_rows.append(
            _paired_row(condition, "Stress-4", [(full[key], simple[key]) for key in ordered])
        )
        for task in STRESS4:
            task_keys = [key for key in ordered if key[0] == task]
            paired_rows.append(
                _paired_row(
                    condition,
                    task,
                    [(full[key], simple[key]) for key in task_keys],
                )
            )
        full_infrastructure = {
            key
            for key in ordered
            if full[key].get("termination_reason") == "infrastructure_error"
        }
        simple_infrastructure = {
            key
            for key in ordered
            if simple[key].get("termination_reason") == "infrastructure_error"
        }
        retained = [
            key
            for key in ordered
            if key not in full_infrastructure and key not in simple_infrastructure
        ]
        sensitivity = _paired_row(
            condition,
            "Stress-4; neither paired episode is an infrastructure error",
            [(full[key], simple[key]) for key in retained],
        )
        sensitivity.update(
            {
                "scheduled_pairs": len(ordered),
                "full_infrastructure_errors": len(full_infrastructure),
                "simple_infrastructure_errors": len(simple_infrastructure),
                "overlapping_infrastructure_errors": len(
                    full_infrastructure & simple_infrastructure
                ),
                "pairs_excluded_either_infrastructure": len(ordered) - len(retained),
            }
        )
        infrastructure_sensitivity_rows.append(sensitivity)

    path_rows = []
    records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for method in METHODS:
        by_pair = {}
        for pair, episode in cells[(method, "perturbed")].items():
            record = _path_record(method, episode)
            by_pair[pair] = record
            path_rows.append(
                {
                    "method": method,
                    "task": pair[0],
                    "seed": pair[1],
                    **record,
                }
            )
        records[method] = by_pair

    funnel_rows = []
    for method in METHODS:
        values = list(records[method].values())
        triggered = [row for row in values if row["physically_triggered"]]
        detected = [row for row in triggered if row["detected"]]
        restored = [row for row in detected if row["condition_restored_after_detection"]]
        reentered = [row for row in restored if row["legal_reentry_after_restoration"]]
        stages = (
            ("physically_triggered", len(triggered), len(values), "scheduled_itt"),
            ("detected", len(detected), len(triggered), "physically_triggered"),
            ("condition_restored", len(restored), len(detected), "detected"),
            ("legal_reentry", len(reentered), len(restored), "condition_restored"),
            (
                "final_success_after_reentry",
                sum(row["final_success"] for row in reentered),
                len(reentered),
                "legal_reentry",
            ),
            (
                "final_success_itt",
                sum(row["final_success"] for row in values),
                len(values),
                "scheduled_itt",
            ),
        )
        for stage, numerator, denominator, population in stages:
            funnel_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "numerator": numerator,
                    "denominator": denominator,
                    "denominator_population": population,
                    "rate": numerator / denominator if denominator else math.nan,
                }
            )

    common = [
        pair
        for pair in records["full"]
        if records["full"][pair]["physically_triggered"]
        and records["simple_state"][pair]["physically_triggered"]
    ]
    common_rows = []
    for method in METHODS:
        successes = sum(records[method][pair]["final_success"] for pair in common)
        common_rows.append(
            {
                "population": "both_methods_physically_triggered_intersection",
                "method": method,
                "successes": successes,
                "denominator": len(common),
                "success_rate": successes / len(common) if common else math.nan,
            }
        )

    _atomic_csv(DERIVED / "paired_itt_success.csv", paired_rows)
    _atomic_csv(
        DERIVED / "paired_infrastructure_sensitivity.csv",
        infrastructure_sensitivity_rows,
    )
    _atomic_csv(DERIVED / "paired_episode_paths.csv", path_rows)
    _atomic_csv(DERIVED / "recovery_funnel.csv", funnel_rows)
    _atomic_csv(DERIVED / "common_trigger_success.csv", common_rows)
    protocol = {
        "schema": "essay2608.iclr2027.decisive-simple-state-analysis.v1",
        "status": "PASS",
        "primary_endpoint": "paired intention-to-treat task success",
        "paired_exact_test": "two-sided exact McNemar",
        "path_denominators_explicit": True,
        "common_trigger_intersection_reported": True,
        "paired_infrastructure_sensitivity_reported": True,
        "run_plan": str(PLAN.relative_to(ROOT)),
        "run_plan_sha256": _sha256(PLAN),
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(DERIVED.glob("*.csv"))
        },
    }
    _atomic_json(DERIVED / "ANALYSIS.json", protocol)
    return protocol


def status() -> dict[str, Any]:
    cells = {}
    ready = PLAN.is_file()
    for method in METHODS:
        for condition, expected in EXPECTED.items():
            root = RESULT_ROOT / method / condition / "episodes"
            found = len(list(root.glob("*.json"))) if root.is_dir() else 0
            cells[f"{method}/{condition}"] = {"found": found, "expected": expected}
            ready &= found == expected
    return {
        "ready": ready,
        "cells": cells,
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
