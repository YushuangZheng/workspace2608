from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rlbench_dynamac import store_bottle_eval_v4 as protocol
from rlbench_dynamac import store_bottle_plan_rebind as rebind
from rlbench_dynamac.eval_set import (
    build_task_scoped_identity,
    build_task_scoped_plan_batch,
    canonical_fingerprint,
    load_task_scoped_plan_batch,
)
from rlbench_dynamac.store_bottle_semantics import (
    load_store_bottle_semantic_spec,
)


SOURCE_POSE = (0.1, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0)


def _source_components():
    semantics = load_store_bottle_semantic_spec()
    motion = protocol.load_v4_store_motion_source_protocol(
        verify_semantics_file=False
    )
    intervention = protocol.load_v4_store_intervention_protocol(
        verify_evidence_files=False
    )
    return {
        "task_semantics": {
            "schema": semantics.schema,
            "fingerprint": semantics.fingerprint,
        },
        "motion_source": {
            "schema": motion["schema"],
            "fingerprint": motion["fingerprint"],
        },
        "intervention": {
            "schema": intervention["schema"],
            "fingerprint": intervention["fingerprint"],
        },
    }


def _plan(index: int, components, *, base_seed: int = 50):
    mode = protocol.store_mode_for_episode(index)
    moved = protocol.V4_STORE_MODE_MOVED_ENTITIES[mode]
    candidate_seed = 123_000 + index
    entities = []
    for name in protocol.V4_STORE_ENTITY_ORDER:
        goal = (
            protocol.sample_v4_store_entity_goal_pose(
                SOURCE_POSE,
                candidate_seed,
                entity=name,
            )
            if name in moved
            else np.asarray(SOURCE_POSE, dtype=np.float64)
        )
        entities.append(
            protocol.StoreBottleEntityMotion(
                name=name,
                root_name=protocol.V4_STORE_ENTITY_ROOTS[name],
                frame_name=protocol.V4_STORE_ENTITY_FRAMES[name],
                source_pose=SOURCE_POSE,
                goal_pose=tuple(goal),
                moved=name in moved,
                candidate_seed=candidate_seed if name in moved else None,
            )
        )
    return protocol.StoreBottleMultiEntityPlan(
        task_name=protocol.STORE_BOTTLE_TASK_NAME,
        episode_index=index,
        episode_seed=base_seed + index,
        variation=0,
        mode=mode,
        entities=tuple(entities),
        source_low_dim_state=SOURCE_POSE + SOURCE_POSE,
        validation={
            "schema": protocol.V4_STORE_PLAN_VALIDATION_SCHEMA,
            "source_seed": base_seed + index,
            "source_waypoint_validated": True,
            "goal_waypoint_validated": True,
            "goal_sampling_max_attempts": 100,
            "sampling_attempts": 1,
            "motion_source_fingerprint": components["motion_source"][
                "fingerprint"
            ],
            "intervention_fingerprint": components["intervention"][
                "fingerprint"
            ],
            "policy_result_fields_read": False,
        },
    )


def _source_envelope():
    components = _source_components()
    plans = [_plan(index, components) for index in range(6)]
    inner = protocol.store_bottle_motion_plan_batch(
        base_seed=50,
        variations=[0] * 6,
        plans=plans,
    )
    identity = build_task_scoped_identity(
        task_name=protocol.STORE_BOTTLE_TASK_NAME,
        components=components,
    )
    outer = build_task_scoped_plan_batch(
        task_name=protocol.STORE_BOTTLE_TASK_NAME,
        task_identity=identity,
        runtime_loader=protocol.V4_STORE_RUNTIME_LOADER_ID,
        runtime_batch=inner,
    )
    return outer, components


def _install_target_intervention(monkeypatch, components):
    target = copy.deepcopy(components)
    target["intervention"]["fingerprint"] = "9" * 64
    original_loader = protocol.load_v4_store_intervention_protocol

    def load_current(*args, **kwargs):
        value = original_loader(*args, **kwargs)
        return {**value, "fingerprint": target["intervention"]["fingerprint"]}

    monkeypatch.setattr(
        protocol,
        "load_v4_store_intervention_protocol",
        load_current,
    )
    monkeypatch.setattr(
        rebind,
        "v4_store_task_identity_components",
        lambda: copy.deepcopy(target),
    )
    return target


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rebind_preserves_every_non_identity_runtime_field(monkeypatch):
    source, components = _source_envelope()
    target = _install_target_intervention(monkeypatch, components)

    rebound = rebind.rebind_v4_store_plan_envelope(
        source,
        source_file_sha256="a" * 64,
    )

    assert (
        rebind.store_runtime_non_identity_projection(source["runtime_batch"])
        == rebind.store_runtime_non_identity_projection(rebound["runtime_batch"])
    )
    assert rebound["task_identity"]["components"] == target
    assert rebound["task_identity"]["fingerprint"] != source["task_identity"][
        "fingerprint"
    ]
    assert rebound["runtime_batch"]["batch_fingerprint"] != source[
        "runtime_batch"
    ]["batch_fingerprint"]
    assert rebound["batch_fingerprint"] != source["batch_fingerprint"]
    assert all(
        plan["validation"]["intervention_fingerprint"] == "9" * 64
        for plan in rebound["runtime_batch"]["plans"]
    )
    provenance = rebound["rebind_provenance"]
    assert provenance["source_file_sha256"] == "a" * 64
    assert provenance["source_outer_batch_fingerprint"] == source[
        "batch_fingerprint"
    ]
    assert provenance["source_inner_batch_fingerprint"] == source[
        "runtime_batch"
    ]["batch_fingerprint"]
    assert len(
        load_task_scoped_plan_batch(
            rebound,
            runtime_loaders=protocol.v4_store_runtime_loaders(),
        )
    ) == 6


def test_rebind_rejects_any_internal_non_identity_rewrite(monkeypatch):
    source, components = _source_envelope()
    _install_target_intervention(monkeypatch, components)
    original = rebind._rewrite_runtime_intervention_identity

    def rewrite_with_tamper(*args, **kwargs):
        value = original(*args, **kwargs)
        value["plans"][0]["source_low_dim_state"][0] += 0.001
        plan = value["plans"][0]
        plan["fingerprint"] = canonical_fingerprint(
            {key: item for key, item in plan.items() if key != "fingerprint"}
        )
        value["batch_fingerprint"] = canonical_fingerprint(
            {
                key: item
                for key, item in value.items()
                if key != "batch_fingerprint"
            }
        )
        return value

    monkeypatch.setattr(
        rebind,
        "_rewrite_runtime_intervention_identity",
        rewrite_with_tamper,
    )
    with pytest.raises(RuntimeError, match="non-identity runtime plan evidence"):
        rebind.rebind_v4_store_plan_envelope(
            source,
            source_file_sha256="a" * 64,
        )


def test_file_rebind_is_hash_pinned_canonical_and_atomic(monkeypatch, tmp_path):
    source, components = _source_envelope()
    _install_target_intervention(monkeypatch, components)
    archive_root = tmp_path / "results" / "_archive" / "v4"
    source_path = archive_root / "old" / "store.json"
    output_path = tmp_path / "evaluation_sets" / "rlbench_eval_v2" / "store.json"
    _write_json(source_path, source)
    expected_sha256 = _sha256(source_path)
    monkeypatch.setattr(rebind, "V4_REBIND_ARCHIVE_ROOT", archive_root)
    monkeypatch.setattr(rebind, "CANONICAL_STORE_PLAN", output_path)

    # Even a self-consistent forged envelope is rejected if its archived bytes
    # no longer match the independently pinned source file hash.
    forged = copy.deepcopy(source)
    forged["runtime_batch"]["plans"][0]["source_low_dim_state"][0] += 0.001
    plan = forged["runtime_batch"]["plans"][0]
    plan["fingerprint"] = canonical_fingerprint(
        {key: item for key, item in plan.items() if key != "fingerprint"}
    )
    inner = forged["runtime_batch"]
    inner["batch_fingerprint"] = canonical_fingerprint(
        {key: item for key, item in inner.items() if key != "batch_fingerprint"}
    )
    forged["batch_fingerprint"] = canonical_fingerprint(
        {key: item for key, item in forged.items() if key != "batch_fingerprint"}
    )
    _write_json(source_path, forged)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rebind.rebind_v4_store_plan_file(
            source_path,
            expected_source_sha256=expected_sha256,
            output=output_path,
        )
    assert not output_path.exists()

    _write_json(source_path, source)
    expected_sha256 = _sha256(source_path)
    result = rebind.rebind_v4_store_plan_file(
        source_path,
        expected_source_sha256=expected_sha256,
        output=output_path,
    )
    assert result["sha256"] == _sha256(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "rebind_provenance"
    ]["source_file_sha256"] == expected_sha256
    assert not output_path.with_name(output_path.name + ".lock").exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rebind.rebind_v4_store_plan_file(
            source_path,
            expected_source_sha256=expected_sha256,
            output=output_path,
        )
