"""Replay all A development examples for adapter-contract verification only.

The explicit analytic test backend below is NOT a trained Main-10 logpZO model.
This command cannot mark M3 formal scoring or the complete A4 delivery ready.
It trains nothing, fits no normalizer or threshold, and reads no failure pool,
calibration or sealed data. Expected scores also have an independent NumPy oracle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA, validate_feature_record
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext

from ..adapter import TorchLogpZOScorer
from ..preprocessing import ENCODER_SCHEMA, FeatureLayout
from ..runtime import CanonicalFailDetectMonitor, _sha256

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "evaluations/iclr2027"
CONFIG = BASE / "configs/methods/m3_fail_detect.json"
FIXTURES = BASE / "tests/fixtures/development_examples"
OUTPUT = BASE / "artifacts/development_golden/m3/adapter_contract_golden.json"
BACKEND = {
    "scope": "analytic_test_double_not_a_trained_logpzo_model",
    "velocity": "0.125*x + 0.01*(flattened_index_mod_7 - 3)",
    "normalizer": "identity_not_fitted",
    "checkpoint": None,
}
BACKEND_HASH = hashlib.sha256(json.dumps(BACKEND, sort_keys=True).encode()).hexdigest()


class AnalyticTestVelocity(torch.nn.Module):
    """Known arithmetic for testing padding/score/runtime, never formal inference."""

    def forward(self, x, timestep):
        if not torch.all(timestep == 0):
            raise ValueError("logpZO must use zero timestep")
        index = torch.arange(x[0].numel(), device=x.device).reshape(x.shape[1:])
        bias = (index % 7 - 3).to(x.dtype) * 0.01
        return x * 0.125 + bias


class ZeroTestThreshold:
    def threshold(self, cycle):
        return 0.0  # fixed test case, NOT a fitted or official threshold


def reference_score(vector: np.ndarray, channels: int) -> float:
    """Independent arithmetic: does not call adapter padding or scoring code."""
    width = math.ceil(vector.size / (channels * 4)) * channels * 4
    flat = np.zeros(width, dtype=np.float32)
    flat[: vector.size] = vector
    bias = (np.arange(width) % 7 - 3).astype(np.float32) * np.float32(0.01)
    velocity = flat * np.float32(0.125) + bias
    return float(np.square(flat + velocity).sum())


def load_examples():
    index = json.loads((FIXTURES / "FIXTURE_INDEX.json").read_text())
    fixture = FIXTURES / "causal_records.jsonl"
    if index["contains_calibration_or_sealed_records"] or index["contains_evaluator_labels"]:
        raise ValueError("only development examples without evaluator labels are allowed")
    if _sha256(fixture) != index["causal_records_sha256"]:
        raise ValueError("A's development example checksum differs")
    records = [
        validate_feature_record(json.loads(line))
        for line in fixture.read_text().splitlines()
        if line.strip()
    ]
    expected = [
        (source["episode_id"], cycle)
        for source in index["sources"]
        for cycle in source["selected_feature_cycles"]
    ]
    if (
        len(records) != index["records"]
        or [(r["episode_id"], r["cycle"]) for r in records] != expected
    ):
        raise ValueError("fixture count, episode ordering or cycle coverage differs from index")
    return records, index["causal_records_sha256"]


def evaluate_contract(config=CONFIG):
    records, fixture_hash = load_examples()
    config_data = json.loads(config.read_text())
    config_hash = _sha256(config)
    channels = config_data["unet_input_channels"]
    torch.set_num_threads(1)
    model = AnalyticTestVelocity().eval().requires_grad_(False)
    scorer = TorchLogpZOScorer(model, input_dim=channels, device="cpu")
    outputs = {"score_only": [], "test_threshold_zero": []}
    max_error = 0.0
    dimensions = {}
    # Test doubles are refused by the public adapter unless explicitly opted in.
    development_default_rejected = False
    for mode in outputs:
        previous = None
        monitor = None
        count = 0
        first_alarm = None
        for record in records:
            task = record["episode_id"].split("/")[1]
            layout = FeatureLayout.from_record(record)
            reset = (
                previous is None
                or record["episode_id"] != previous[0]
                or record["cycle"] != previous[1] + 1
            )
            if reset:
                binding = {
                    "development_only": True,
                    "task": task,
                    "feature_schema": FEATURE_SCHEMA,
                    "encoder_schema": ENCODER_SCHEMA,
                    "config_sha256": config_hash,
                    "scorer_sha256": BACKEND_HASH,
                    "score_definition": config_data["score_definition"],
                    "layout": layout.to_dict(),
                    "normalizer": {
                        "mean": [0.0] * layout.input_dim,
                        "std": [1.0] * layout.input_dim,
                    },
                }
                if not development_default_rejected:
                    try:
                        CanonicalFailDetectMonitor.from_scorer(scorer, config, binding=binding)
                    except ValueError as exc:
                        if "development-only" not in str(exc):
                            raise
                        development_default_rejected = True
                    else:
                        raise ValueError("test backend was incorrectly accepted for formal use")
                monitor = CanonicalFailDetectMonitor.from_scorer(
                    scorer,
                    config,
                    binding=binding,
                    allow_development_scorer=True,
                    threshold_schedule=None if mode == "score_only" else ZeroTestThreshold(),
                )
                monitor.reset(
                    EpisodeContext(
                        record["episode_id"],
                        task,
                        "M3",
                        len(layout.arms) == 2,
                        1000,
                        FEATURE_SCHEMA,
                        config_hash,
                        None,
                    )
                )
                count = 0
                first_alarm = None
            before = copy.deepcopy(record)
            vector = layout.encode(record)
            normalized = np.clip(
                vector, -config_data["normalizer"]["clip"], config_data["normalizer"]["clip"]
            )
            expected = reference_score(normalized, channels)
            monitor.observe_record(record)
            actual = monitor.cycle_output()
            if record != before:
                raise ValueError("shadow adapter mutated its observation/action input")
            error = abs(actual["scores"]["logpzo"] - expected)
            max_error = max(max_error, error)
            if not math.isclose(actual["scores"]["logpzo"], expected, abs_tol=1e-5, rel_tol=1e-6):
                raise ValueError("adapter score differs from independent arithmetic oracle")
            threshold = None if mode == "score_only" else 0.0
            count = count + 1 if threshold is not None and expected > threshold else 0
            alarm = count >= config_data["persistence"]
            if alarm and first_alarm is None:
                first_alarm = record["cycle"]
            if (
                actual["cycle"],
                actual["threshold"],
                actual["alarm"],
                actual["persistence_count"],
                actual["metadata"]["first_alarm_cycle"],
            ) != (record["cycle"], threshold, alarm, count, first_alarm):
                raise ValueError("runtime reset, timestamp, persistence or alarm differs")
            outputs[mode].append(
                {
                    "episode_id": record["episode_id"],
                    "reset_before": reset,
                    "encoded_input_sha256": hashlib.sha256(
                        vector.astype("<f4").tobytes()
                    ).hexdigest(),
                    "reference_score": expected,
                    **actual,
                }
            )
            dimensions[task] = layout.input_dim
            previous = (record["episode_id"], record["cycle"])
    return {
        "schema": "essay2608.iclr2027.m3-adapter-contract-golden.v1",
        "status": "pass",
        "scope": "adapter_contract_only",
        "m3_formal_scorer_bound": False,
        "complete_A4_delivery": False,
        "warning": "Analytic test backend; these are NOT trained Main-10 M3 scores.",
        "backend": BACKEND,
        "backend_sha256": BACKEND_HASH,
        "config_sha256": config_hash,
        "fixture_sha256": fixture_hash,
        "preprocessing_sha256": _sha256(BASE / "methods/fail_detect/preprocessing.py"),
        "records": len(records),
        "episodes": len({r["episode_id"] for r in records}),
        "task_input_dimensions": dimensions,
        "max_reference_score_error": max_error,
        "default_test_backend_rejection": development_default_rejected,
        "training_performed": False,
        "normalizer_fitted": False,
        "calibration_performed": False,
        "calibration_or_sealed_read": False,
        "shadow_inputs_unchanged": True,
        "outputs": outputs,
    }


def verify_contract(expected, actual):
    """Compare a frozen golden; all identity and non-score fields must match."""
    expected = copy.deepcopy(expected)
    actual = copy.deepcopy(actual)
    for mode in expected["outputs"]:
        a, b = expected["outputs"][mode], actual["outputs"][mode]
        if len(a) != len(b):
            raise ValueError("golden coverage differs")
        for old, new in zip(a, b):
            if not math.isclose(
                old["scores"]["logpzo"], new["scores"]["logpzo"], abs_tol=1e-5, rel_tol=1e-6
            ):
                raise ValueError("golden score differs")
            old.pop("scores")
            new.pop("scores")
    expected.pop("max_reference_score_error")
    actual.pop("max_reference_score_error")
    if expected != actual:
        raise ValueError("golden identity, oracle, reset or output fields differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="write the fixed contract-only artifact if absent"
    )
    args = parser.parse_args()
    report = evaluate_contract()
    if OUTPUT.exists():
        verify_contract(json.loads(OUTPUT.read_text()), report)
    elif args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        raise FileNotFoundError("contract golden missing; use --write for initial creation")
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
