import json
from copy import deepcopy

import pytest

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.protocols.tsf_dynamic_protocol import (
    build_tsf_dynamic_trigger_evidence,
    resolve_tsf_dynamic_trigger,
)


def _manifest(task):
    path = INTEGRATION_ROOT / "models" / "dynamac_backbone_v1" / task / "training.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "task,expected_tick,expected_schema",
    [
        ("stack_wine", 58, "dynamac-direct-training-v3"),
        ("wipe_desk", 52, "dynamac-direct-static-training-v1"),
        ("bimanual_handover_item", 50, "dynamac-direct-training-v3"),
    ],
)
def test_tsf_trigger_is_bound_to_current_checkpoint(
    task, expected_tick, expected_schema
):
    manifest = _manifest(task)
    evidence = build_tsf_dynamic_trigger_evidence(
        task,
        manifest["checkpoint_trigger_audit"],
        manifest,
    )
    identity = {
        "manifest_authenticated": True,
        "training_manifest_schema": manifest["manifest_schema"],
        "checkpoint_trigger_audit_digest": manifest["checkpoint_trigger_audit"][
            "fingerprint"
        ],
        "tsf_dynamic_trigger_evidence": evidence,
    }

    resolved = resolve_tsf_dynamic_trigger(identity, task, 10)

    assert manifest["manifest_schema"] == expected_schema
    assert resolved["trigger_step"] == expected_tick
    assert evidence["required_active_window"] == [expected_tick, expected_tick + 9]
    assert evidence["task_parameter_selected"] == [True]


def test_tsf_trigger_rejects_evidence_from_another_model():
    manifest = _manifest("wipe_desk")
    evidence = build_tsf_dynamic_trigger_evidence(
        "wipe_desk",
        manifest["checkpoint_trigger_audit"],
        manifest,
    )
    evidence = deepcopy(evidence)
    evidence["resolved_global_tick"] += 1
    identity = {
        "manifest_authenticated": True,
        "training_manifest_schema": manifest["manifest_schema"],
        "checkpoint_trigger_audit_fingerprint": manifest["checkpoint_trigger_audit"][
            "fingerprint"
        ],
        "tsf_dynamic_trigger_evidence": evidence,
    }

    with pytest.raises(RuntimeError, match="evidence is invalid"):
        resolve_tsf_dynamic_trigger(identity, "wipe_desk", 10)
