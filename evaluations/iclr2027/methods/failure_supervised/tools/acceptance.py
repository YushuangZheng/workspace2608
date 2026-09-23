"""Portable CPU forward/golden acceptance. Requires no training data or simulator.

Run from the repository root. This checks supplied artifacts, never trains or
calibrates a monitor. It writes no files unless --output is explicitly given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA, validate_feature_record
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.failure_supervised.runtime import (
    CanonicalFailureSupervisedMonitor,
    sha256,
)

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "evaluations/iclr2027"


def repository_path(relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError("expected a canonical repository-relative path")
    result = ROOT / p
    if not result.is_file() or result.is_symlink() or ROOT not in result.resolve().parents:
        raise ValueError(f"missing or noncanonical file: {relative}")
    return result


def check_one(manifest_path: Path, fixture: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != "essay2608.iclr2027.monitor-checkpoint.v1"
        or manifest["method_id"] != "M4"
    ):
        raise ValueError("not an M4 checkpoint manifest")
    checkpoint = repository_path(manifest["checkpoint_relative_path"])
    config = repository_path(manifest["config_relative_path"])
    if (
        sha256(checkpoint) != manifest["checkpoint_sha256"]
        or sha256(config) != manifest["config_sha256"]
    ):
        raise ValueError("checkpoint/config byte identity mismatch")
    monitor = CanonicalFailureSupervisedMonitor(checkpoint, config, device="cpu")
    for key in ("task", "training_budget", "training_seed", "held_out_family", "feature_schema"):
        if manifest[key] != monitor.metadata[key]:
            raise ValueError(f"checkpoint payload identity differs from manifest: {key}")
    # This deterministic tensor forward is an availability/shape check only.
    # It is deliberately not labelled a task rollout or a golden fixture.
    with torch.inference_mode():
        logits, _ = monitor.scorer.model(torch.zeros(1, 3, monitor.layout.input_dim))
    if logits.shape != (1, 3) or not torch.isfinite(logits).all():
        raise ValueError("CPU checkpoint forward failed")
    identity = checkpoint.parent.relative_to(BASE / "artifacts/checkpoints/m4")
    golden_path = BASE / "artifacts/development_golden/m4" / identity / "golden.json"
    golden = json.loads(golden_path.read_text())
    if (
        golden["checkpoint_sha256"] != monitor.checkpoint_hash
        or golden["config_sha256"] != monitor.config_hash
        or golden["fixture_sha256"] != sha256(fixture)
    ):
        raise ValueError("golden/checkpoint/config/fixture identity mismatch")
    features = [
        validate_feature_record(json.loads(line))
        for line in fixture.read_text().splitlines()
        if line.strip()
    ]
    selected = [r for r in features if r["episode_id"].split("/")[1] == manifest["task"]]
    if len(selected) != len(golden["outputs"]):
        raise ValueError("golden does not cover exactly the available task fixture records")
    errors = []
    previous_id, previous_cycle = None, None
    for record, expected in zip(selected, golden["outputs"]):
        reset = record["episode_id"] != previous_id or record["cycle"] != previous_cycle + 1
        if (
            expected["episode_id"] != record["episode_id"]
            or expected["cycle"] != record["cycle"]
            or expected["reset_before"] != reset
        ):
            raise ValueError("golden identity/order/reset differs from sparse source fixture")
        if reset:
            monitor.reset(
                EpisodeContext(
                    record["episode_id"],
                    manifest["task"],
                    "M4",
                    len(monitor.layout.arms) == 2,
                    1000,
                    FEATURE_SCHEMA,
                    monitor.config_hash,
                    monitor.checkpoint_hash,
                )
            )
        monitor.observe_record(record)
        actual = monitor.cycle_output()
        for key in ("cycle", "alarm", "threshold", "persistence_count", "metadata"):
            if actual[key] != expected[key]:
                raise ValueError(f"golden output differs: {key}")
        if set(actual["scores"]) != set(expected["scores"]):
            raise ValueError("golden score names differ")
        for key, score in actual["scores"].items():
            difference = abs(score - expected["scores"][key])
            errors.append(difference)
            if not np.isclose(
                score,
                expected["scores"][key],
                atol=golden["comparison_atol"],
                rtol=golden["comparison_rtol"],
            ):
                raise ValueError(f"golden score mismatch: {manifest['task']}/{key}: {difference}")
        previous_id, previous_cycle = record["episode_id"], record["cycle"]
    return {
        "status": "pass",
        "task": manifest["task"],
        "training_seed": manifest["training_seed"],
        "training_budget": manifest["training_budget"],
        "held_out_family": manifest["held_out_family"],
        "checkpoint_sha256": monitor.checkpoint_hash,
        "cpu_forward": "pass",
        "golden_records": len(selected),
        "golden_status": "pass" if selected else "fixture_not_available_for_task",
        "max_score_error": max(errors, default=None),
        "formal_development_rollout_gate": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fixture_dir = BASE / "tests/fixtures/development_examples"
    fixture = fixture_dir / "causal_records.jsonl"
    index = json.loads((fixture_dir / "FIXTURE_INDEX.json").read_text())
    if index["contains_calibration_or_sealed_records"] or index["contains_evaluator_labels"]:
        raise ValueError("not a permitted development-only fixture")
    if sha256(fixture) != index["causal_records_sha256"]:
        raise ValueError("fixture differs from A's fixture index")
    paths = (
        sorted((BASE / "artifacts/checkpoints/m4").glob("*/*/*/checkpoint_manifest.json"))
        if args.all
        else [args.checkpoint_manifest]
    )
    if not paths:
        raise ValueError("no checkpoint manifests")
    results = [check_one(path, fixture) for path in paths]
    external = sorted(
        {
            str(getattr(module, "__file__", ""))
            for module in sys.modules.values()
            if "/_external/" in str(getattr(module, "__file__", ""))
        }
    )
    if external:
        raise ValueError(
            f"pure inference unexpectedly imported external reproduction source: {external}"
        )
    report = {
        "status": "pass",
        "scope": "local_cpu_forward_and_available_sparse_golden_only",
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "external_reproduction_source_imports": external,
        "training_data_read": False,
        "calibration_or_sealed_data_read": False,
        "checkpoints_verified": len(results),
        "checkpoints_with_golden": sum(r["golden_records"] > 0 for r in results),
        "formal_development_rollout_gate": False,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, allow_nan=False))


if __name__ == "__main__":
    main()
