"""Paired recovery-path cross statistics for the decisive reviewer analysis.

The analysis keeps the 800 pre-registered Stress-4 task/seed pairs as the
intention-to-treat (ITT) population.  It reports both method-specific trigger
subgroups and their intersections, so recovery rates with different physical
trigger membership cannot be compared as though they shared a denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluations.iclr2027.analysis.a5_detection import (
    _current_audit,
    _endpoint_episode_path,
    _method_events,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
MANIFEST = EVAL / "manifests" / "ablation4.jsonl"
DERIVED = EVAL / "results" / "reviewer_decisive" / "recovery_paths" / "derived"
METHODS = {
    "m6": "TSF monitor + skill retry",
    "retry_same_state": "TSF monitor + relation repair + same-state resume",
    "m5": "Full TSF",
}


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


def _manifest_keys() -> list[tuple[str, int]]:
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [(str(row["task"]), int(row["seed"])) for row in rows]
    if len(keys) != 800 or len(set(keys)) != 800:
        raise RuntimeError("ablation4 must contain 800 unique task/seed pairs")
    return keys


def _load_method(key: str, expected: set[tuple[str, int]]) -> dict[tuple[str, int], dict[str, Any]]:
    episode_root = _endpoint_episode_path(key, "perturbed", "placeholder").parent
    all_episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(episode_root.glob("*.json"))
    ]
    source_expected = 800 if key == "retry_same_state" else 2000
    if len(all_episodes) != source_expected:
        raise RuntimeError(
            f"incomplete source cell {key}: expected {source_expected}, found {len(all_episodes)}"
        )
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for episode in all_episodes:
        pair = (str(episode["task"]), int(episode["seed"]))
        if pair not in expected:
            continue
        if pair in selected:
            raise RuntimeError(f"duplicate paired episode: {key}/{pair}")
        selected[pair] = episode
    if set(selected) != expected:
        missing = sorted(expected - set(selected))[:5]
        raise RuntimeError(f"{key} is not paired on Stress-4; examples missing: {missing}")
    return selected


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def generate() -> dict[str, Any]:
    ordered_pairs = _manifest_keys()
    expected = set(ordered_pairs)
    episodes = {key: _load_method(key, expected) for key in METHODS}

    pair_rows: list[dict[str, Any]] = []
    method_records: dict[str, list[dict[str, Any]]] = {key: [] for key in METHODS}
    for task, seed in ordered_pairs:
        pair = (task, seed)
        wide: dict[str, Any] = {"task": task, "seed": seed}
        membership_bits = []
        for key in METHODS:
            episode = episodes[key][pair]
            audit = _current_audit(key, episode)
            triggered = bool(audit.get("physically_triggered"))
            event = _method_events(key, episode, audit=audit) if triggered else None
            detected = bool(event and event["detected"])
            restored = bool(
                event
                and detected
                and event["condition_restored"]
                and int(event["condition_restored_cycle"]) >= int(event["first_alarm"])
            )
            resumed = bool(event and restored and event["execution_resumed"])
            succeeded = bool(episode["success"])
            if not triggered:
                category = "not_physically_triggered"
            elif not detected:
                category = "triggered_undetected"
            elif not restored:
                category = "detected_not_restored"
            elif not resumed:
                category = "restored_not_resumed"
            elif not succeeded:
                category = "resumed_final_failure"
            else:
                category = "resumed_final_success"
            record = {
                "task": task,
                "seed": seed,
                "method_key": key,
                "method": METHODS[key],
                "physically_triggered": triggered,
                "detected": detected,
                "condition_restored_after_detection": restored,
                "execution_resumed_after_restoration": resumed,
                "final_success": succeeded,
                "path_category": category,
            }
            method_records[key].append(record)
            membership_bits.append(key if triggered else "-")
            wide[f"{key}_triggered"] = triggered
            wide[f"{key}_detected"] = detected
            wide[f"{key}_restored"] = restored
            wide[f"{key}_resumed"] = resumed
            wide[f"{key}_success"] = succeeded
            wide[f"{key}_path"] = category
        wide["trigger_membership"] = "|".join(membership_bits)
        pair_rows.append(wide)

    funnel_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for key, label in METHODS.items():
        records = method_records[key]
        triggered = [row for row in records if row["physically_triggered"]]
        detected = [row for row in triggered if row["detected"]]
        restored = [row for row in detected if row["condition_restored_after_detection"]]
        resumed = [row for row in restored if row["execution_resumed_after_restoration"]]
        stages = (
            ("physically_triggered", len(triggered), len(records), "scheduled_itt"),
            ("detected", len(detected), len(triggered), "physically_triggered"),
            ("condition_restored", len(restored), len(detected), "detected"),
            ("execution_resumed", len(resumed), len(restored), "condition_restored"),
            (
                "final_success_after_resume",
                sum(row["final_success"] for row in resumed),
                len(resumed),
                "execution_resumed",
            ),
            (
                "final_success_itt",
                sum(row["final_success"] for row in records),
                len(records),
                "scheduled_itt",
            ),
            (
                "final_success_among_triggered",
                sum(row["final_success"] for row in triggered),
                len(triggered),
                "physically_triggered",
            ),
            (
                "final_success_among_detected",
                sum(row["final_success"] for row in detected),
                len(detected),
                "detected",
            ),
            (
                "final_success_among_restored",
                sum(row["final_success"] for row in restored),
                len(restored),
                "condition_restored",
            ),
        )
        for stage, numerator, denominator, denominator_name in stages:
            funnel_rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "stage": stage,
                    "numerator": numerator,
                    "denominator": denominator,
                    "denominator_population": denominator_name,
                    "rate": _divide(numerator, denominator),
                }
            )
        categories = sorted({str(row["path_category"]) for row in records})
        for category in categories:
            subgroup = [row for row in records if row["path_category"] == category]
            count = len(subgroup)
            successes = sum(bool(row["final_success"]) for row in subgroup)
            category_rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "path_category": category,
                    "count": count,
                    "rate_among_scheduled_itt": count / len(records),
                    "final_successes": successes,
                    "final_failures": count - successes,
                    "final_success_rate_within_path": successes / count,
                }
            )

    membership_rows: list[dict[str, Any]] = []
    memberships = sorted({str(row["trigger_membership"]) for row in pair_rows})
    for membership in memberships:
        subgroup = [row for row in pair_rows if row["trigger_membership"] == membership]
        for key, label in METHODS.items():
            successes = sum(bool(row[f"{key}_success"]) for row in subgroup)
            membership_rows.append(
                {
                    "trigger_membership": membership,
                    "episodes": len(subgroup),
                    "method_key": key,
                    "method": label,
                    "final_successes": successes,
                    "final_success_rate": successes / len(subgroup),
                }
            )

    common_trigger = [
        row for row in pair_rows if all(bool(row[f"{key}_triggered"]) for key in METHODS)
    ]
    common_rows = []
    for key, label in METHODS.items():
        successes = sum(bool(row[f"{key}_success"]) for row in common_trigger)
        common_rows.append(
            {
                "population": "all_three_physically_triggered_intersection",
                "method_key": key,
                "method": label,
                "final_successes": successes,
                "denominator": len(common_trigger),
                "final_success_rate": _divide(successes, len(common_trigger)),
            }
        )

    if len(pair_rows) != 800:
        raise RuntimeError("paired path table does not conserve the ITT population")
    for key in METHODS:
        if sum(row["count"] for row in category_rows if row["method_key"] == key) != 800:
            raise RuntimeError(f"path categories do not sum to 800: {key}")
    if any(int(row["numerator"]) > int(row["denominator"]) for row in funnel_rows):
        raise RuntimeError("recovery funnel contains numerator greater than denominator")

    _atomic_csv(DERIVED / "paired_episode_paths.csv", pair_rows)
    _atomic_csv(DERIVED / "recovery_funnel_with_denominators.csv", funnel_rows)
    _atomic_csv(DERIVED / "path_outcome_categories.csv", category_rows)
    _atomic_csv(DERIVED / "trigger_membership_cross_stats.csv", membership_rows)
    _atomic_csv(DERIVED / "common_trigger_success.csv", common_rows)
    protocol = {
        "schema": "essay2608.iclr2027.decisive-recovery-paths.v1",
        "status": "PASS",
        "paired_population": "pre-registered Stress-4 task/seed ITT",
        "scheduled_episodes_per_method": len(ordered_pairs),
        "method_specific_trigger_denominators_reported": True,
        "common_trigger_intersection_reported": True,
        "stage_denominators_explicit": True,
        "path_by_final_outcome_cross_statistics_reported": True,
        "conservation_checks_passed": True,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(DERIVED.glob("*.csv"))
        },
    }
    _atomic_json(DERIVED / "ANALYSIS.json", protocol)
    return protocol


def status() -> dict[str, Any]:
    return {
        "ready": MANIFEST.is_file()
        and all(_endpoint_episode_path(key, "perturbed", "placeholder").parent.is_dir() for key in METHODS),
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
