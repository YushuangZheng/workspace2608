import copy
import hashlib
import json
import pytest

from integrations.rlbench.rlbench_dynamac.eval import eval_set
from integrations.rlbench.rlbench_dynamac.eval.evaluation_split import (
    EVALUATION_SET_V2_ID,
    load_evaluation_set_v2_spec,
)


def _resign(payload):
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _components(tag="a"):
    return {
        role: {"schema": f"fixture-{role}-v1", "fingerprint": tag * 64}
        for role in ("task_semantics", "motion_source", "intervention")
    }


def test_v2_spec_is_complete_and_has_no_result_run_state():
    spec = load_evaluation_set_v2_spec()

    assert spec["evaluation_set_id"] == EVALUATION_SET_V2_ID
    assert set(spec["dynamic_environment"]) == eval_set.ENVIRONMENT_TASKS
    assert {
        task
        for task, profile in spec["dynamic_environment"].items()
        if profile["artifact_origin"] == "regenerated_v2"
    } == {
        "open_microwave",
        "bimanual_put_bottle_in_fridge",
        "bimanual_lift_tray",
    }
    assert all(
        profile["artifact_origin"] == "reused_legacy"
        for task, profile in spec["dynamic_environment"].items()
        if task
        not in {
            "open_microwave",
            "bimanual_put_bottle_in_fridge",
            "bimanual_lift_tray",
        }
    )
    assert spec["legacy_import"]["source_artifacts_remain_external"] is False
    assert spec["isolation"]["evaluation_artifact_root"] == "data/evaluation"
    assert spec["isolation"]["training_data_root"] == "data/training"
    assert "NOT_RUN" not in json.dumps(spec)


def test_v2_spec_rejects_not_run_even_after_resigning(tmp_path):
    spec = copy.deepcopy(load_evaluation_set_v2_spec())
    spec["dynamic_environment"]["stack_wine"]["NOT_RUN"] = True
    _resign(spec)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="NOT_RUN"):
        load_evaluation_set_v2_spec(path)


def test_task_scoped_envelope_authenticates_only_its_own_task():
    identity = eval_set.build_task_scoped_identity(
        task_name="bimanual_lift_tray",
        components=_components(),
    )
    inner_body = {
        "task_name": "bimanual_lift_tray",
        "base_seed": 2_608_000_000,
        "episodes": 2,
        "variation_schedule": [0, 0],
        "plans": [{"episode": 0}, {"episode": 1}],
    }
    inner = {
        **inner_body,
        "batch_fingerprint": eval_set.canonical_fingerprint(inner_body),
    }
    envelope = eval_set.build_task_scoped_plan_batch(
        task_name="bimanual_lift_tray",
        task_identity=identity,
        runtime_loader="fixture-v1",
        runtime_batch=inner,
    )

    plans = eval_set.load_task_scoped_plan_batch(
        envelope,
        runtime_loaders={"fixture-v1": lambda payload: payload["plans"]},
    )
    assert plans == inner["plans"]

    forged = copy.deepcopy(envelope)
    forged["task_name"] = "bimanual_put_bottle_in_fridge"
    forged_body = {
        key: value for key, value in forged.items() if key != "batch_fingerprint"
    }
    forged["batch_fingerprint"] = eval_set.canonical_fingerprint(forged_body)
    with pytest.raises(ValueError, match="identity"):
        eval_set.load_task_scoped_plan_batch(
            forged,
            runtime_loaders={"fixture-v1": lambda payload: payload["plans"]},
        )


def test_current_v2_loader_authenticates_all_local_artifacts():
    loaded = eval_set.load_evaluation_set_v2_manifest(
        full_preflight=True,
        verify_training_files=True,
    )

    root = eval_set.EVAL_SET_ROOT.resolve()
    assert loaded["manifest_path"].parent == root
    assert set(loaded["environment_batches"]) == eval_set.ENVIRONMENT_TASKS
    assert all(
        batch["path"].resolve().is_relative_to(root)
        for batch in loaded["environment_batches"].values()
    )
    assert len(loaded["coordination_source_batch"]["plans"]) == 200
    assert loaded["training_identity"]["path"] == "data/training/manifest.json"
    assert loaded["training_identity"]["training_file_count"] == 125


def test_current_loader_rejects_retired_v1_id():
    with pytest.raises(ValueError, match="current rlbench_eval_v2"):
        eval_set.load_fixed_eval_set_manifest("rlbench_fixed_v1")
