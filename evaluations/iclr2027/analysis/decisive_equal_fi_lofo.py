"""Held-out-fault-family comparison at independently calibrated FI budgets.

For every held-out family, the supervised GRU is the corresponding LOFO
model.  TSF is training-free.  Both methods select their task-wise operating
points only on the same disjoint family-eligible normal-calibration view and
are then evaluated on the same nominal and family-specific trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.iclr2027.analysis.decisive_equal_fi import (
    A5,
    EVAL,
    M6_SHADOW,
    PRIMARY_TARGET_FI,
    ROOT,
    TARGET_FI,
    MonitorSpec,
    _atomic_csv,
    _atomic_json,
    _json,
    _metrics,
    _pr_rows,
    _select_thresholds,
    _sha256,
    _test_records,
)
from evaluations.iclr2027.analysis.rebuild_a5_audit import load_current_reaudit


E3 = EVAL / "results" / "controlled" / "e3"
DERIVED = EVAL / "results" / "reviewer_decisive" / "equal_fi" / "derived"
FAMILIES = (
    "actuation_delay",
    "coordination_delay",
    "environment_change",
    "missed_interaction",
    "relation_loss",
)
SEEDS = (1103, 2207, 3301)


def _manifest_ids(path: Path) -> set[str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["episode_id"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate episode IDs in {path}")
    return result


def _views(family: str) -> tuple[Path, Path, Path]:
    root = E3 / "manifest_views"
    return (
        root / f"normal_calibration_lofo_{family}.jsonl",
        root / f"shadow_lofo_{family}_nominal.jsonl",
        root / f"shadow_lofo_{family}_perturbed.jsonl",
    )


def _specs(family: str) -> tuple[MonitorSpec, ...]:
    result = []
    for seed in SEEDS:
        calibration_root = (
            EVAL
            / "artifacts"
            / "calibration"
            / "monitors"
            / "m4"
            / "e3"
            / f"lofo_{family}"
            / f"seed_{seed}"
            / "v1"
        )
        shadow_root = E3 / "shadow" / "lofo" / family / f"seed_{seed}"
        result.append(
            MonitorSpec(
                key=f"m4_lofo_{family}_seed{seed}",
                label=f"Supervised GRU LOFO-{family} [{seed}]",
                score_name="violation_probability",
                calibration=calibration_root / "calibration.json",
                calibration_index=calibration_root / "score_index.json",
                nominal_index=shadow_root / "nominal" / "score_index.json",
                perturbed_index=shadow_root / "perturbed" / "score_index.json",
            )
        )
    result.append(
        MonitorSpec(
            key=f"m6_lofo_{family}",
            label="TSF-Monitor (training-free)",
            score_name="task_state_mismatch",
            calibration=None,
            calibration_index=M6_SHADOW / "calibration" / "score_index.json",
            nominal_index=M6_SHADOW / "nominal" / "score_index.json",
            perturbed_index=M6_SHADOW / "perturbed" / "score_index.json",
            inclusive=True,
        )
    )
    return tuple(result)


def _validate_index_population(
    spec: MonitorSpec,
    calibration_ids: set[str],
    nominal_ids: set[str],
    perturbed_ids: set[str],
) -> None:
    expected = (
        (spec.calibration_index, calibration_ids),
        (spec.nominal_index, nominal_ids),
        (spec.perturbed_index, perturbed_ids),
    )
    for path, required in expected:
        present = {str(row["episode_id"]) for row in _json(path)["files"]}
        if not required <= present:
            examples = sorted(required - present)[:5]
            raise RuntimeError(f"{spec.key} index misses held-out view IDs: {examples}")


def generate() -> dict[str, Any]:
    audits = load_current_reaudit()
    operating_points = []
    results = []
    pr_rows = []
    view_hashes: dict[str, str] = {}
    view_counts: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        calibration_view, nominal_view, perturbed_view = _views(family)
        for path in (calibration_view, nominal_view, perturbed_view):
            view_hashes[str(path.relative_to(ROOT))] = _sha256(path)
        calibration_ids = _manifest_ids(calibration_view)
        nominal_ids = _manifest_ids(nominal_view)
        perturbed_ids = _manifest_ids(perturbed_view)
        sizes = (len(calibration_ids), len(nominal_ids), len(perturbed_ids))
        if len(set(sizes)) != 1 or sizes[0] not in {100, 200}:
            raise RuntimeError(f"unexpected LOFO view sizes for {family}")
        view_counts[family] = {
            "calibration": sizes[0],
            "nominal": sizes[1],
            "perturbed": sizes[2],
        }
        for spec in _specs(family):
            _validate_index_population(
                spec,
                calibration_ids,
                nominal_ids,
                perturbed_ids,
            )
            for target in TARGET_FI:
                thresholds, selected = _select_thresholds(
                    spec,
                    target,
                    allowed_episode_ids=calibration_ids,
                )
                for row in selected:
                    row["held_out_fault_family"] = family
                operating_points.extend(selected)
                nominal = _test_records(
                    spec,
                    "nominal",
                    thresholds,
                    audits,
                    allowed_episode_ids=nominal_ids,
                )
                perturbed = _test_records(
                    spec,
                    "perturbed",
                    thresholds,
                    audits,
                    allowed_episode_ids=perturbed_ids,
                )
                observed_families = {
                    str(row["fault_family"])
                    for row in perturbed
                    if row["fault_family"] is not None
                }
                if observed_families != {family}:
                    raise RuntimeError(
                        f"held-out test contamination for {family}: {observed_families}"
                    )
                row = _metrics(
                    spec,
                    target,
                    nominal,
                    perturbed,
                    population="held_out_fault_family",
                    family=family,
                )
                row["monitor_class"] = (
                    "tsf_training_free" if spec.key.startswith("m6_") else "supervised_gru_lofo"
                )
                row["calibration_episodes"] = len(calibration_ids)
                row["nominal_test_episodes"] = len(nominal)
                row["perturbed_test_episodes"] = len(perturbed)
                results.append(row)
                if target == PRIMARY_TARGET_FI:
                    for pr_row in _pr_rows(spec, nominal, perturbed):
                        pr_row["held_out_fault_family"] = family
                        pr_rows.append(pr_row)

    _atomic_csv(DERIVED / "lofo_calibration_operating_points.csv", operating_points)
    _atomic_csv(DERIVED / "lofo_recall_at_fixed_fi.csv", results)
    _atomic_csv(DERIVED / "lofo_pr_curve.csv", pr_rows)
    protocol = {
        "schema": "essay2608.iclr2027.decisive-equal-fi-lofo.v1",
        "status": "PASS",
        "held_out_families": list(FAMILIES),
        "supervised_gru_seeds": list(SEEDS),
        "tsf_training_free": True,
        "shared_episode_counts_by_family": view_counts,
        "operating_points_selected_on_test": False,
        "actual_test_fi_reported": True,
        "delay_population": "detected physically-triggered events only",
        "view_sha256": view_hashes,
        "outputs": {
            name: _sha256(DERIVED / name)
            for name in (
                "lofo_calibration_operating_points.csv",
                "lofo_pr_curve.csv",
                "lofo_recall_at_fixed_fi.csv",
            )
        },
    }
    _atomic_json(DERIVED / "LOFO_ANALYSIS.json", protocol)
    return protocol


def status() -> dict[str, Any]:
    paths = [path for family in FAMILIES for path in _views(family)]
    paths.extend(
        path
        for family in FAMILIES
        for spec in _specs(family)
        for path in (spec.calibration_index, spec.nominal_index, spec.perturbed_index)
    )
    return {
        "ready": all(path.is_file() for path in paths),
        "missing": [str(path.relative_to(ROOT)) for path in paths if not path.is_file()],
        "analysis_exists": (DERIVED / "LOFO_ANALYSIS.json").is_file(),
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
