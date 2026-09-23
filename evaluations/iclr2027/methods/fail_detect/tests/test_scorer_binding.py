"""Contract tests with explicit test doubles, NOT trained M3 golden outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.fail_detect.preprocessing import ENCODER_SCHEMA, FeatureLayout
from evaluations.iclr2027.methods.fail_detect.runtime import CanonicalFailDetectMonitor, _sha256


@pytest.fixture
def example():
    base = Path(__file__).resolve().parents[3]
    config = base / "configs/methods/m3_fail_detect.json"
    records = [
        json.loads(line)
        for line in (base / "tests/fixtures/development_examples/causal_records.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    record = records[0]
    layout = FeatureLayout.from_record(record)
    binding = {
        "development_only": True,
        "task": record["episode_id"].split("/")[1],
        "feature_schema": FEATURE_SCHEMA,
        "encoder_schema": ENCODER_SCHEMA,
        "config_sha256": _sha256(config),
        "scorer_sha256": "a" * 64,
        "score_definition": json.loads(config.read_text())["score_definition"],
        "layout": layout.to_dict(),
        "normalizer": {"mean": [0.0] * layout.input_dim, "std": [1.0] * layout.input_dim},
    }
    return config, records, binding


def make_monitor(example, scorer=lambda x: float(np.square(x).sum()), **kwargs):
    config, records, binding = example
    monitor = CanonicalFailDetectMonitor.from_scorer(
        scorer, config, binding=binding, allow_development_scorer=True, **kwargs
    )
    context = EpisodeContext(
        records[0]["episode_id"],
        binding["task"],
        "M3",
        False,
        1000,
        FEATURE_SCHEMA,
        monitor.config_hash,
        None,
    )
    return monitor, context


def test_existing_scorer_needs_no_new_checkpoint_and_does_not_mutate_inputs(example):
    config, records, binding = example
    with pytest.raises(ValueError, match="development-only"):
        CanonicalFailDetectMonitor.from_scorer(lambda x: 0.0, config, binding=binding)
    monitor, context = make_monitor(example)
    with pytest.raises(RuntimeError, match="reset"):
        monitor.observe_record(records[0])
    monitor.reset(context)
    before = copy.deepcopy(records[0])
    monitor.observe_record(records[0])
    assert records[0] == before
    output = monitor.cycle_output()
    assert output["metadata"]["development_only"] is True
    assert output["metadata"]["checkpoint_sha256"] is None
    assert output["metadata"]["scorer_sha256"] == "a" * 64
    assert output["threshold"] is None and not output["alarm"]
    score = monitor.score()
    monitor.reset(context)
    monitor.observe_record(records[0])
    assert monitor.score() == score
    # Caller changes cannot alter the frozen identity/normalizer after binding.
    binding["task"] = "different_task"
    binding["normalizer"]["mean"][0] = 999.0
    assert monitor.metadata["task"] == context.task_id
    assert monitor.mean[0] == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_schema", "wrong"),
        ("encoder_schema", "wrong"),
        ("config_sha256", "b" * 64),
        ("scorer_sha256", None),
        ("scorer_sha256", "not-a-hash"),
        ("checkpoint_sha256", "bad"),
        ("score_definition", "some_other_score"),
        ("task", ""),
    ],
)
def test_rejects_unbound_or_incompatible_scorer(example, field, value):
    example[2][field] = value
    with pytest.raises(ValueError):
        make_monitor(example)


def test_causal_validation_precedes_scorer_and_sparse_reset_is_explicit(example):
    calls = []
    monitor, context = make_monitor(example, scorer=lambda x: calls.append(x.copy()) or 1.0)
    monitor.reset(context)
    bad = copy.deepcopy(example[1][0])
    bad["policy_state"]["audit_label"] = True
    with pytest.raises(ValueError):
        monitor.observe_record(bad)
    assert not calls
    monitor.observe_record(example[1][0])
    with pytest.raises(ValueError, match="non-contiguous"):
        monitor.observe_record(example[1][0])
    sparse = copy.deepcopy(example[1][0])
    for key in ("cycle", "observation_timestamp", "action_timestamp"):
        sparse[key] = 20
    with pytest.raises(ValueError, match="non-contiguous"):
        monitor.observe_record(sparse)
    monitor.reset(context)
    monitor.observe_record(sparse)
    assert len(calls) == 2


def test_unselected_top_level_metadata_cannot_reach_scorer(example):
    calls = []
    monitor, context = make_monitor(example, scorer=lambda x: calls.append(x.copy()) or 1.0)
    record = copy.deepcopy(example[1][0])
    monitor.reset(context)
    monitor.observe_record(record)
    record["audit"] = {"failure": True}
    monitor.reset(context)
    monitor.observe_record(record)
    assert np.array_equal(calls[0], calls[1])


def test_strict_threshold_persistence_and_first_alarm_reset(example):
    class Schedule:
        def threshold(self, cycle):
            return 1.0

    scores = iter([1.0, 2.0, 2.0, 2.0, 0.0])
    monitor, context = make_monitor(
        example, scorer=lambda x: next(scores), threshold_schedule=Schedule()
    )
    monitor.reset(context)
    for cycle, expected_count in enumerate([0, 1, 2, 3, 0]):
        record = copy.deepcopy(example[1][0])
        for key in ("cycle", "observation_timestamp", "action_timestamp"):
            record[key] = cycle
        monitor.observe_record(record)
        assert monitor.cycle_output()["persistence_count"] == expected_count
        assert monitor.alarm() == (cycle == 3)
    assert monitor.cycle_output()["metadata"]["first_alarm_cycle"] == 3
    monitor.reset(context)
    assert monitor.first_alarm_cycle is None and not monitor.alarm()


def test_nonfinite_backend_output_is_not_published(example):
    monitor, context = make_monitor(example, scorer=lambda x: float("nan"))
    monitor.reset(context)
    with pytest.raises(ValueError, match="nonfinite"):
        monitor.observe_record(example[1][0])
    with pytest.raises(RuntimeError, match="no observed"):
        monitor.cycle_output()
