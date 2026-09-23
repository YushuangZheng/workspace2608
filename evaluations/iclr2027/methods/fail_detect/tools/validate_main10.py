"""Validate A's real Main-10 M3 checkpoints on fixed development records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluations.iclr2027.interfaces.feature_schema import (
    FEATURE_SCHEMA,
    validate_feature_record,
)
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.registry import build_monitor, load_method_spec


ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "evaluations/iclr2027"
FIXTURE = BASE / "tests/fixtures/development_examples/causal_records.jsonl"
OUTPUT = BASE / "artifacts/development_golden/m3/main10_scorer_golden.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate() -> dict:
    spec = load_method_spec("m3_fail_detect_runtime")
    records = [
        validate_feature_record(json.loads(line))
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outputs = []
    task_identities = {}
    for task in sorted({record["episode_id"].split("/")[1] for record in records}):
        monitor = build_monitor(spec, task_id=task)
        if monitor is None:
            raise RuntimeError("M3 factory returned no monitor")
        selected = [r for r in records if r["episode_id"].split("/")[1] == task]
        previous_id, previous_cycle = None, None
        for record in selected:
            reset = (
                record["episode_id"] != previous_id
                or record["cycle"] != previous_cycle + 1
            )
            if reset:
                monitor.reset(
                    EpisodeContext(
                        episode_id=record["episode_id"],
                        task_id=task,
                        method_id=spec.method_id,
                        bimanual=len(record["arms"]) == 2,
                        horizon=1000,
                        feature_schema=FEATURE_SCHEMA,
                        method_config_hash=spec.config_sha256,
                        checkpoint_hash=monitor.checkpoint_hash,
                    )
                )
            monitor.observe_record(record)
            outputs.append(
                {
                    "task": task,
                    "episode_id": record["episode_id"],
                    "cycle": record["cycle"],
                    "reset_before": reset,
                    "output": monitor.cycle_output(),
                }
            )
            previous_id, previous_cycle = record["episode_id"], record["cycle"]
        task_identities[task] = {
            "checkpoint_sha256": monitor.checkpoint_hash,
            "backend_config_sha256": monitor.config_hash,
        }
    return {
        "schema": "essay2608.iclr2027.m3-main10-development-golden.v1",
        "status": "pass",
        "scope": "real_project_scorer_sparse_development_records",
        "method_config_sha256": spec.config_sha256,
        "fixture_sha256": _sha256(FIXTURE),
        "tasks": task_identities,
        "records": len(outputs),
        "training_data": "five_successful_demonstrations_per_task",
        "failure_train_read": False,
        "normal_calibration_read": False,
        "sealed_test_read": False,
        "comparison_atol": 2e-5,
        "comparison_rtol": 2e-5,
        "outputs": outputs,
    }


def _compare(expected: dict, actual: dict) -> float:
    if {k: v for k, v in expected.items() if k != "outputs"} != {
        k: v for k, v in actual.items() if k != "outputs"
    }:
        raise ValueError("M3 golden identity or coverage changed")
    errors = []
    for old, new in zip(expected["outputs"], actual["outputs"], strict=True):
        score_name = next(iter(old["output"]["scores"]))
        if set(old["output"]["scores"]) != {score_name} or set(
            new["output"]["scores"]
        ) != {score_name}:
            raise ValueError("M3 golden score name changed")
        old_score = old["output"]["scores"][score_name]
        new_score = new["output"]["scores"][score_name]
        old_without_score = json.loads(json.dumps(old))
        new_without_score = json.loads(json.dumps(new))
        del old_without_score["output"]["scores"][score_name]
        del new_without_score["output"]["scores"][score_name]
        if old_without_score != new_without_score:
            raise ValueError("M3 golden output contract changed")
        error = abs(old_score - new_score)
        errors.append(error)
        if not np.isclose(
            old_score,
            new_score,
            atol=expected["comparison_atol"],
            rtol=expected["comparison_rtol"],
        ):
            raise ValueError(f"M3 golden score changed by {error}")
    return max(errors, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = _generate()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        error = 0.0
    else:
        if not OUTPUT.is_file():
            raise FileNotFoundError("M3 Main-10 golden is missing; create it with --write")
        error = _compare(json.loads(OUTPUT.read_text(encoding="utf-8")), actual)
    print(
        json.dumps(
            {
                "status": "pass",
                "tasks": len(actual["tasks"]),
                "records": actual["records"],
                "maximum_score_error": error,
                "output": str(OUTPUT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
