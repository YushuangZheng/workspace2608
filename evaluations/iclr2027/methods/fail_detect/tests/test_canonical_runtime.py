"""Synthetic adapter checks, explicitly not trained M3 or development golden."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext, RuntimeMonitor
from evaluations.iclr2027.methods.fail_detect.preprocessing import ENCODER_SCHEMA, FeatureLayout
from evaluations.iclr2027.methods.fail_detect.runtime import CanonicalFailDetectMonitor, _sha256


class ZeroVelocity(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_frozen_m3_schema_and_untrained_checkpoint_rejection(tmp_path):
    base = Path(__file__).resolve().parents[3]
    config = base / "configs/methods/m3_fail_detect.json"
    record = json.loads(
        (base / "tests/fixtures/development_examples/causal_records.jsonl")
        .read_text()
        .splitlines()[0]
    )
    layout = FeatureLayout.from_record(record)
    checkpoint = tmp_path / "toy_unit_test.pt"
    torch.save(
        {
            "schema": "essay2608.iclr2027.m3-logpzo.v1",
            "model_state_dict": {},
            "metadata": {
                "development_only": True,
                "task": "close_jar",
                "feature_schema": FEATURE_SCHEMA,
                "encoder_schema": ENCODER_SCHEMA,
                "config_sha256": _sha256(config),
                "fit_split": "normal_demonstrations",
                "unet_input_channels": 10,
                "layout": layout.to_dict(),
                "normalizer": {"mean": [0.0] * layout.input_dim, "std": [1.0] * layout.input_dim},
            },
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="development-only"):
        CanonicalFailDetectMonitor(ZeroVelocity(), checkpoint, config)
    m = CanonicalFailDetectMonitor(
        ZeroVelocity(), checkpoint, config, allow_development_checkpoint=True
    )
    assert isinstance(m, RuntimeMonitor)
    ctx = EpisodeContext(
        record["episode_id"],
        "close_jar",
        "M3",
        False,
        1000,
        FEATURE_SCHEMA,
        m.config_hash,
        m.checkpoint_hash,
    )
    m.reset(ctx)
    m.observe_record(record)
    expected = float(np.square(np.clip(layout.encode(record), -8, 8)).sum())
    assert m.score()["logpzo"] == pytest.approx(expected, rel=1e-6)
    assert m.cycle_output()["threshold"] is None and not m.alarm()
    with pytest.raises(ValueError, match="non-contiguous"):
        m.observe_record(record)
    m.reset(ctx)
    m.observe_record(record)
    assert m.score()["logpzo"] == pytest.approx(expected, rel=1e-6)
