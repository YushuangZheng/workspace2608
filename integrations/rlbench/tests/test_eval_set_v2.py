import copy
import hashlib
import json
from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac import eval_set
from integrations.rlbench.rlbench_dynamac.evaluation_split import (
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
    } == {"bimanual_put_bottle_in_fridge", "bimanual_lift_tray"}
    assert all(
        profile["artifact_origin"] == "reused_legacy"
        for task, profile in spec["dynamic_environment"].items()
        if task not in {"bimanual_put_bottle_in_fridge", "bimanual_lift_tray"}
    )
    assert spec["legacy_import"]["source_artifacts_remain_external"] is True
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


def test_v2_draft_keeps_legacy_batches_as_zero_copy_canonical_references(
    tmp_path,
    monkeypatch,
):
    evaluation_sets = tmp_path / "evaluation_sets"
    source = evaluation_sets / "rlbench_fixed_v1"
    source.mkdir(parents=True)
    real_source = (
        Path(eval_set.__file__).resolve().parents[1]
        / "evaluation_sets"
        / "rlbench_fixed_v1"
        / "manifest.json"
    )
    (source / "manifest.json").write_bytes(real_source.read_bytes())
    monkeypatch.setattr(eval_set, "EVAL_SET_ROOT", evaluation_sets)

    root = eval_set.build_evaluation_set_v2_draft()

    assert root == evaluation_sets / "rlbench_eval_v2"
    assert (root / "spec.json").is_file()
    assert not any(
        path.is_file()
        for path in (root / "plans" / "environment").glob("*.json")
    )
    assert (source / "manifest.json").read_bytes() == real_source.read_bytes()


def test_store_training_identity_proves_seed_disjointness_without_file_reads():
    demo_hash = "1" * 64
    body = {
        "schema": eval_set.STORE_TRAINING_IDENTITY_SCHEMA,
        "task": "bimanual_put_bottle_in_fridge",
        "collection_manifest": {
            "path": "data/v4/store_bottle/collection_manifest.json",
            "sha256": "2" * 64,
            "fingerprint": "3" * 64,
        },
        "demonstrations": [
            {
                "episode": episode,
                "seed": 4_104_000_000 + episode,
                "variation": 0,
                "path": (
                    "data/v4/store_bottle/bimanual_put_bottle_in_fridge/"
                    f"all_variations/episodes/episode{episode}/low_dim_obs.pkl"
                ),
                "bytes": 1,
                "sha256": demo_hash,
            }
            for episode in range(5)
        ],
        "model_release_manifest": {
            "path": "models/v4/release_manifest.json",
            "sha256": "4" * 64,
            "fingerprint": "5" * 64,
        },
        "evaluation_seed_range": [2_608_000_000, 2_608_000_199],
        "training_evaluation_seed_ranges_disjoint": True,
    }
    identity = {**body, "fingerprint": eval_set.canonical_fingerprint(body)}

    assert eval_set.validate_store_training_identity(
        identity,
        verify_files=False,
    )["demonstrations"][0]["seed"] == 4_104_000_000


def test_v2_seal_and_loader_mix_zero_copy_legacy_with_task_scoped_batches(
    tmp_path,
    monkeypatch,
):
    integration_root = Path(eval_set.__file__).resolve().parents[1]
    evaluation_sets = tmp_path / "evaluation_sets"
    source = evaluation_sets / "rlbench_fixed_v1"
    source.mkdir(parents=True)
    real_source = integration_root / "evaluation_sets" / "rlbench_fixed_v1"
    for name in ("manifest.json", "training_split_manifest.json"):
        (source / name).write_bytes((real_source / name).read_bytes())
    monkeypatch.setattr(eval_set, "EVAL_SET_ROOT", evaluation_sets)
    root = eval_set.build_evaluation_set_v2_draft()

    source_manifest = json.loads((source / "manifest.json").read_text())
    legacy_spec = eval_set.load_evaluation_set_spec(real_source / "spec.json")
    legacy_batches = {}
    for task, profile in legacy_spec["dynamic_environment"].items():
        reference = source_manifest["environment_plan_batches"][task]
        legacy_batches[task] = {
            "path": source / profile["artifact_path"],
            "payload": {
                "schema": "dynamac-rlbench-staged-motion-plan-batch-v3.4",
                "protocol_id": (
                    "rlbench-deterministic-source-staging-waypoint-validated-"
                    "boundary-root-v3.4"
                ),
                "task_name": task,
                "batch_fingerprint": reference["batch_fingerprint"],
            },
            "plans": [object()] * 200,
        }
    coordination_profile = legacy_spec["coordination"][eval_set.COORDINATION_TASK]
    coordination_reference = source_manifest["coordination_source_batch"]
    fake_legacy = {
        "manifest_path": source / "manifest.json",
        "manifest_sha256": eval_set.file_sha256(source / "manifest.json"),
        "payload": source_manifest,
        "spec": legacy_spec,
        "training_split": {},
        "environment_batches": legacy_batches,
        "coordination_source_batch": {
            "resolved_path": source / coordination_profile["artifact_path"],
            "payload": {
                "schema": "dynamac-rlbench-staged-source-plan-batch-v1",
                "protocol_id": "rlbench-deterministic-source-a-only-v1",
                "batch_fingerprint": coordination_reference["batch_fingerprint"],
            },
            "plans": [object()] * 200,
        },
    }
    monkeypatch.setattr(
        eval_set,
        "_load_v1_fixed_eval_set_manifest",
        lambda *args, **kwargs: fake_legacy,
    )

    fixture_loader = lambda payload: payload["plans"]
    for task in ("bimanual_put_bottle_in_fridge", "bimanual_lift_tray"):
        inner_body = {
            "task_name": task,
            "base_seed": 2_608_000_000,
            "episodes": 200,
            "variation_schedule": [0] * 200,
            "plans": [{"episode": episode} for episode in range(200)],
        }
        inner = {
            **inner_body,
            "batch_fingerprint": eval_set.canonical_fingerprint(inner_body),
        }
        identity = eval_set.build_task_scoped_identity(
            task_name=task,
            components=_components("a" if task.startswith("bimanual_put") else "b"),
        )
        envelope = eval_set.build_task_scoped_plan_batch(
            task_name=task,
            task_identity=identity,
            runtime_loader="fixture-v1",
            runtime_batch=inner,
        )
        profile = load_evaluation_set_v2_spec(root / "spec.json")[
            "dynamic_environment"
        ][task]
        path = root / profile["artifact_path"]
        path.write_text(json.dumps(envelope), encoding="utf-8")

    store_body = {
        "schema": eval_set.STORE_TRAINING_IDENTITY_SCHEMA,
        "task": "bimanual_put_bottle_in_fridge",
        "collection_manifest": {
            "path": "data/v4/store_bottle/collection_manifest.json",
            "sha256": "2" * 64,
            "fingerprint": "3" * 64,
        },
        "demonstrations": [
            {
                "episode": episode,
                "seed": 4_104_000_000 + episode,
                "variation": 0,
                "path": f"data/v4/store_bottle/episode{episode}/low_dim_obs.pkl",
                "bytes": 1,
                "sha256": "1" * 64,
            }
            for episode in range(5)
        ],
        "model_release_manifest": {
            "path": "models/v4/release_manifest.json",
            "sha256": "4" * 64,
            "fingerprint": "5" * 64,
        },
        "evaluation_seed_range": [2_608_000_000, 2_608_000_199],
        "training_evaluation_seed_ranges_disjoint": True,
    }
    store_identity = {
        **store_body,
        "fingerprint": eval_set.canonical_fingerprint(store_body),
    }
    monkeypatch.setattr(
        eval_set,
        "build_store_training_identity",
        lambda **kwargs: store_identity,
    )

    manifest_path = eval_set.seal_evaluation_set_v2(
        store_collection_manifest=Path("unused-collection.json"),
        store_model_release_manifest=Path("unused-model.json"),
        runtime_loaders={"fixture-v1": fixture_loader},
    )
    loaded = eval_set.load_evaluation_set_v2_manifest(
        selected_task="bimanual_lift_tray",
        runtime_loaders={"fixture-v1": fixture_loader},
    )

    assert manifest_path == root / "manifest.json"
    assert loaded["environment_batches"]["bimanual_lift_tray"][
        "artifact_origin"
    ] == "regenerated_v2"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["environment_plan_batches"]["stack_wine"][
        "artifact_origin"
    ] == "reused_legacy"
    assert manifest["environment_plan_batches"]["stack_wine"][
        "source_evaluation_set_id"
    ] == "rlbench_fixed_v1"
    assert not (root / "plans/environment/stack_wine_a_b_n200.json").exists()
