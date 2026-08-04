"""Calibrate and replay the two-edge estimator on a declared frozen-data split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from essay2608.data.physical_handover import audit_physical_handover_dataset
from essay2608.policy.bimanual_relation import (
    calibrate_bimanual_relation_estimator,
    replay_bimanual_relation_estimator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data/handover_physical/v1")
    )
    parser.add_argument(
        "--calibration_seeds", nargs="+", type=int, default=list(range(8200, 8210))
    )
    parser.add_argument(
        "--evaluation_seeds", nargs="+", type=int, default=list(range(8210, 8220))
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/bimanual_relation/offline_dev_v1"),
    )
    return parser.parse_args()


def source_fingerprint(repository: Path) -> str:
    paths = [
        Path(__file__).resolve(),
        repository / "source/essay2608/essay2608/policy/relation.py",
        repository / "source/essay2608/essay2608/policy/bimanual_relation.py",
        repository / "source/essay2608/essay2608/data/physical_handover.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repository)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def aggregate_edges(rows: list[dict], side: str) -> dict:
    tp = sum(row[side]["tp"] for row in rows)
    fp = sum(row[side]["fp"] for row in rows)
    fn = sum(row[side]["fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_demo_f1": float(np.mean([row[side]["f1"] for row in rows])),
        "minimum_demo_f1": float(np.min([row[side]["f1"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"拒绝覆盖已有双臂关系分析目录：{output_dir}")
    if set(args.calibration_seeds) & set(args.evaluation_seeds):
        raise ValueError("标定 seed 与开发评测 seed 必须互斥")
    audit = audit_physical_handover_dataset(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    by_seed = {
        int(entry["seed"]): data_dir / entry["file"] for entry in manifest["demos"]
    }
    requested = set(args.calibration_seeds) | set(args.evaluation_seeds)
    if not requested.issubset(by_seed):
        raise ValueError(f"数据集缺少请求 seed：{sorted(requested.difference(by_seed))}")

    calibration_paths = [by_seed[seed] for seed in args.calibration_seeds]
    config, calibration = calibrate_bimanual_relation_estimator(
        calibration_paths,
        dataset_sha256=audit["dataset_sha256"],
    )
    output_dir.mkdir(parents=True)
    trial_dir = output_dir / "trials"
    trial_dir.mkdir()
    rows = []
    for seed in args.evaluation_seeds:
        summary, arrays = replay_bimanual_relation_estimator(by_seed[seed], config)
        summary = {"seed": seed, **summary}
        rows.append(summary)
        (trial_dir / f"seed_{seed}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(trial_dir / f"seed_{seed}.npz", **arrays)

    source_sha = source_fingerprint(Path(__file__).resolve().parents[1])
    aggregate = {
        "artifact_type": "bimanual_relation_offline_development_replay",
        "source_sha256": source_sha,
        "dataset_sha256": audit["dataset_sha256"],
        "calibration_seeds": list(args.calibration_seeds),
        "evaluation_seeds": list(args.evaluation_seeds),
        "privileged_contact_used_as_estimator_input": False,
        "num_evaluation_demonstrations": len(rows),
        "four_value_accuracy": {
            "mean": float(np.mean([row["four_value_accuracy"] for row in rows])),
            "minimum": float(np.min([row["four_value_accuracy"] for row in rows])),
            "maximum": float(np.max([row["four_value_accuracy"] for row in rows])),
        },
        "left": aggregate_edges(rows, "left"),
        "right": aggregate_edges(rows, "right"),
        "truth_both_steps": sum(row["truth_both_steps"] for row in rows),
        "inferred_both_steps": sum(row["inferred_both_steps"] for row in rows),
        "trials": rows,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in aggregate.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
