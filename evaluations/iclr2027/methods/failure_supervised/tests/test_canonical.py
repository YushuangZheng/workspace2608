"""Server-B method-scoped tests. A-owned fixture/interface files stay read-only."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from evaluations.iclr2027.interfaces.failure_train import (
    causal_violation_labels,
    select_failure_train_rows,
)
from evaluations.iclr2027.interfaces.feature_schema import AUDIT_SCHEMA, FEATURE_SCHEMA
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext, RuntimeMonitor
from evaluations.iclr2027.methods.fail_detect.preprocessing import FeatureLayout
from evaluations.iclr2027.methods.failure_supervised.model import (
    CausalGRUClassifier,
    masked_binary_cross_entropy,
)
from evaluations.iclr2027.methods.failure_supervised.runtime import (
    CanonicalFailureSupervisedMonitor,
    sha256,
)
from evaluations.iclr2027.methods.failure_supervised.train_cli import BASE, CONFIG, config
from evaluations.iclr2027.methods.failure_supervised.training import save_training_checkpoint


def fixture():
    return json.loads(
        (BASE / "tests/fixtures/development_examples/causal_records.jsonl")
        .read_text()
        .splitlines()[0]
    )


def monitor(tmp_path):
    record = fixture()
    layout = FeatureLayout.from_record(record)
    model = CausalGRUClassifier(layout.input_dim, hidden_dim=4)
    path = tmp_path / "model.pt"
    metadata = {
        "task": "close_jar",
        "config_sha256": sha256(CONFIG),
        "feature_schema": FEATURE_SCHEMA,
        "layout": layout.to_dict(),
        "normalizer": {"mean": [0.0] * layout.input_dim, "std": [1.0] * layout.input_dim},
    }
    save_training_checkpoint(path, model, training_step=0, metadata=metadata)
    return CanonicalFailureSupervisedMonitor(path, CONFIG), record


def context(monitor, record):
    return EpisodeContext(
        record["episode_id"],
        "close_jar",
        "M4",
        False,
        1000,
        FEATURE_SCHEMA,
        monitor.config_hash,
        monitor.checkpoint_hash,
    )


def test_layout_deterministic_and_no_labels():
    record = fixture()
    layout = FeatureLayout.from_record(record)
    expected = layout.encode(record)
    assert len(expected) == layout.input_dim
    assert np.array_equal(expected, FeatureLayout.from_dict(layout.to_dict()).encode(record))
    extra = {
        **record,
        "audit": {"violation_onset_cycle": 1},
        "fault_family": "do_not_use",
        "episode_id": "different_id",
        "cycle": 700,
        "observation_timestamp": 700,
        "action_timestamp": 700,
    }
    assert np.array_equal(expected, layout.encode(extra))
    bad = copy.deepcopy(record)
    bad["policy_state"]["hidden"] = {"fault_family": "forbidden"}
    with pytest.raises(ValueError, match="evaluator-only"):
        layout.encode(bad)
    bad = copy.deepcopy(record)
    bad["task_state"].append(0)
    with pytest.raises(ValueError, match="dimensions"):
        layout.encode(bad)


def test_progressed_is_effective_response_not_stopped():
    record = fixture()
    layout = FeatureLayout.from_record(record)
    reached = copy.deepcopy(record)
    progressed = copy.deepcopy(record)
    stopped = copy.deepcopy(record)
    for value, status in (
        (reached, "reached"),
        (progressed, "progressed"),
        (stopped, "stopped"),
    ):
        value["action_resolution"] = {
            "aggregate": status,
            "per_arm": {"single": status},
            "primary_action_applied": True,
        }
    reached_vector = layout.encode(reached)
    progressed_vector = layout.encode(progressed)
    stopped_vector = layout.encode(stopped)
    assert np.array_equal(reached_vector, progressed_vector)
    assert not np.array_equal(reached_vector, stopped_vector)
    assert len(progressed_vector) == layout.input_dim


def test_label_lag_not_same_cycle():
    audits = [
        {"schema": AUDIT_SCHEMA, "violation_onset_cycle": onset, "violation_end_cycle": end}
        for onset, end in [(None, None), (1, None), (1, 2), (1, 2)]
    ]
    assert causal_violation_labels(audits) == (0, 0, 1, 0)


def test_budget_and_lofo_preserve_fixed_order():
    rows = [
        {"task": "test", "episode_id": str(i), "fault_family": "a" if i % 4 == 0 else "b"}
        for i in range(200)
    ]
    assert select_failure_train_rows(rows, task="test", budget=20) == rows[:20]
    assert select_failure_train_rows(rows, task="test", budget=50)[:20] == rows[:20]
    lofo = select_failure_train_rows(rows, task="test", held_out_family="a")
    assert len(lofo) == 150 and all(r["fault_family"] == "b" for r in lofo)
    with pytest.raises(ValueError):
        select_failure_train_rows(rows, task="test", budget=200, held_out_family="a")


def test_causality_mask_and_streaming():
    torch.manual_seed(4)
    model = CausalGRUClassifier(5, hidden_dim=8).eval()
    x = torch.randn(2, 12, 5)
    altered = x.clone()
    altered[:, 7:] += 100
    with torch.no_grad():
        logits, _ = model(x)
        changed, _ = model(altered)
        hidden, pieces = None, []
        for i in range(12):
            value, hidden = model(x[:, i : i + 1], hidden)
            pieces.append(value)
    torch.testing.assert_close(logits[:, :7], changed[:, :7])
    torch.testing.assert_close(logits, torch.cat(pieces, dim=1))
    mask = torch.arange(12)[None] < torch.tensor([12, 5])[:, None]
    labels = torch.zeros_like(logits)
    baseline = masked_binary_cross_entropy(logits, labels, mask)
    padded = logits.clone()
    padded[~mask] = 100
    torch.testing.assert_close(baseline, masked_binary_cross_entropy(padded, labels, mask))


def test_runtime_identity_reset_gap_and_no_calibration(tmp_path):
    m, record = monitor(tmp_path)
    assert isinstance(m, RuntimeMonitor)
    ctx = context(m, record)
    m.reset(ctx)
    m.observe_record(record)
    first = m.score()
    assert m.cycle_output()["threshold"] is None and m.alarm() is False
    assert m.cycle_output()["persistence_count"] == 0
    next_record = {
        **record,
        "cycle": record["cycle"] + 1,
        "observation_timestamp": record["cycle"] + 1,
        "action_timestamp": record["cycle"] + 1,
    }
    m.observe_record(next_record)
    with pytest.raises(ValueError, match="non-contiguous"):
        m.observe_record(next_record)
    m.reset(ctx)
    m.observe_record(record)
    assert first == m.score()
    wrong = EpisodeContext(
        ctx.episode_id,
        "open_drawer",
        "M4",
        False,
        1000,
        FEATURE_SCHEMA,
        m.config_hash,
        m.checkpoint_hash,
    )
    with pytest.raises(ValueError, match="identity"):
        m.reset(wrong)


def test_runtime_persistence_and_abi(tmp_path):
    m, record = monitor(tmp_path)

    class Schedule:
        def threshold(self, cycle):
            return -1.0

    m.schedule = Schedule()
    m.reset(context(m, record))
    for i in range(config()["persistence"]):
        obs = {**record, "cycle": i, "observation_timestamp": i}
        m.observe(obs, {"action": record["action"], "action_timestamp": i}, record["policy_state"])
        assert m.alarm() == (i == config()["persistence"] - 1)
    assert m.cycle_output()["metadata"]["first_alarm_cycle"] == config()["persistence"] - 1
