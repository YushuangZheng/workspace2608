"""All supplied examples, not only the first single-arm record."""

import copy

import pytest

from evaluations.iclr2027.methods.fail_detect.tools.validate_adapter import (
    evaluate_contract,
    verify_contract,
)


@pytest.fixture(scope="module")
def report():
    return evaluate_contract()


def test_all_examples_and_explicit_scope(report):
    assert report["records"] == 18 and report["episodes"] == 4
    assert set(report["task_input_dimensions"]) == {"close_jar", "bimanual_handover_item"}
    assert report["scope"] == "adapter_contract_only"
    assert not report["m3_formal_scorer_bound"]
    assert not report["formal_evaluation_ready"]
    assert report["default_test_backend_rejection"]
    assert report["shadow_inputs_unchanged"]
    assert not report["training_performed"] and not report["calibration_or_sealed_read"]
    assert len(report["outputs"]["score_only"]) == 18
    assert len(report["outputs"]["test_threshold_zero"]) == 18
    verify_contract(report, evaluate_contract())


@pytest.mark.parametrize("mode", ["score_only", "test_threshold_zero"])
def test_rejects_changed_golden_score(report, mode):
    changed = copy.deepcopy(report)
    changed["outputs"][mode][0]["scores"]["logpzo"] += 1.0
    with pytest.raises(ValueError, match="score"):
        verify_contract(changed, report)


def test_rejects_changed_input_or_reset_identity(report):
    changed = copy.deepcopy(report)
    changed["outputs"]["score_only"][0]["reset_before"] = False
    with pytest.raises(ValueError, match="identity"):
        verify_contract(changed, report)
