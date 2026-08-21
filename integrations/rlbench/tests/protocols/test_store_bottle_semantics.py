from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    STORE_BOTTLE_CONFIG_PATH,
    STORE_BOTTLE_SEMANTIC_SCHEMA,
    STORE_BOTTLE_SEMANTIC_VERSION,
    extract_store_bottle_semantic_episode,
    load_store_bottle_semantic_spec,
    store_bottle_semantic_observations_from_rlbench,
    validate_store_bottle_scene_hierarchy,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import get_task_spec

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CURRENT_PLAN_BATCH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "environment"
    / "bimanual_put_bottle_in_fridge_a_b_n200.json"
)
LIVE_TASK_MODULE = (
    REPOSITORY_ROOT
    / "integrations"
    / "rlbench"
    / "rlbench_dynamac"
    / "store_bottle_live_v4.py"
)


def _xyzw_pose(x: float) -> np.ndarray:
    return np.asarray([x, x + 1.0, x + 2.0, 0.0, 0.0, 0.0, 1.0])


def _observation(sample: int = 0) -> SimpleNamespace:
    low_dim = np.concatenate((_xyzw_pose(10.0 + sample), _xyzw_pose(20.0 + sample)))
    return SimpleNamespace(
        left=SimpleNamespace(gripper_pose=_xyzw_pose(1.0 + sample), gripper_open=1.0),
        right=SimpleNamespace(gripper_pose=_xyzw_pose(2.0 + sample), gripper_open=1.0),
        task_low_dim_state=(low_dim,),
    )


def _sealed_source_task_tree_rows() -> list[dict[str, object]]:
    envelope = json.loads(CURRENT_PLAN_BATCH.read_text(encoding="utf-8"))
    return envelope["runtime_batch"]["plans"][0]["validation"][
        "source_task_tree_relative_state"
    ]


def test_v4_store_semantics_reuse_the_pinned_ttm_without_touching_v1() -> None:
    spec = load_store_bottle_semantic_spec()

    assert STORE_BOTTLE_CONFIG_PATH == (
        REPOSITORY_ROOT
        / "integrations"
        / "rlbench"
        / "configs"
        / "v4"
        / "store_bottle_semantics.json"
    )
    assert spec.schema == STORE_BOTTLE_SEMANTIC_SCHEMA
    assert spec.semantic_version == STORE_BOTTLE_SEMANTIC_VERSION
    assert spec.release_name == "v4"
    assert spec.training_data_root == (
        "integrations/rlbench/data/training/main/bimanual_put_bottle_in_fridge"
    )
    assert spec.models_root == "integrations/rlbench/models/v4"
    assert spec.evaluation_set_id == "rlbench_eval_v2"
    assert spec.legacy_ttm_sha256 == (
        "845ac4c0f68a809fb33c06a5d2a57a92a6964315ecc43acb488d4618613505ae"
    )
    assert not (
        REPOSITORY_ROOT
        / "RLBench"
        / "rlbench"
        / "task_ttms"
        / "bimanual_put_bottle_in_fridge_semantic_v4.ttm"
    ).exists()

    legacy = get_task_spec("bimanual_put_bottle_in_fridge")
    assert legacy.frame_names == ("bottle", "fridge_root")
    assert legacy.class_name == "BimanualPutBottleInFridge"
    assert spec.task_spec.frame_names == ("bottle", "fridge")
    assert spec.task_spec.module == (
        "integrations.rlbench.rlbench_dynamac.store_bottle_live_v4"
    )
    assert spec.task_spec.class_name == "BimanualPutBottleInFridgeSemanticV4"


def test_v4_live_task_class_explicitly_reuses_the_legacy_model_name() -> None:
    tree = ast.parse(LIVE_TASK_MODULE.read_text(encoding="utf-8"))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    source = LIVE_TASK_MODULE.read_text(encoding="utf-8")

    assert "BimanualPutBottleInFridgeSemanticV4" in classes
    assert "name=LEGACY_STORE_BOTTLE_MODEL_NAME" in source
    assert 'BOTTLE_MOTION_ROOT_OBJECT = "fridge_root"' in source
    assert 'FRIDGE_MOTION_ROOT_OBJECT = "fridge_base"' in source
    assert "self.fridge_motion_root().get_pose()" in source


def test_semantic_groups_match_the_audited_legacy_ttm_hierarchy() -> None:
    audit = validate_store_bottle_scene_hierarchy(
        _sealed_source_task_tree_rows()
    )

    assert audit["passed"] is True
    assert audit["missing"] == []
    assert audit["parent_mismatches"] == {}
    assert audit["groups"]["bottle"]["scene_root_name"] == "fridge_root"
    assert audit["groups"]["fridge"]["scene_root_name"] == "fridge_base"
    assert "success" in audit["groups"]["fridge"]["members"]
    assert "success" not in audit["groups"]["bottle"]["members"]
    assert set(audit["groups"]["bottle"]["members"]).isdisjoint(
        audit["groups"]["fridge"]["members"]
    )


def test_hierarchy_audit_rejects_a_waypoint_attached_to_the_wrong_group() -> None:
    rows = [dict(row) for row in _sealed_source_task_tree_rows()]
    next(row for row in rows if row["name"] == "waypoint7")["parent"] = "bottle"

    audit = validate_store_bottle_scene_hierarchy(rows)

    assert audit["passed"] is False
    assert audit["parent_mismatches"] == {
        "waypoint7": {"expected": "fridge_base", "actual": "bottle"}
    }


def test_v4_episode_adapter_names_the_true_fridge_frame() -> None:
    arrays = extract_store_bottle_semantic_episode([_observation(0), _observation(1)])

    assert tuple(arrays.left.frames) == ("bottle", "fridge")
    assert tuple(arrays.right.frames) == ("bottle", "fridge")
    np.testing.assert_allclose(arrays.left.frames["bottle"][:, 0], [10.0, 11.0])
    np.testing.assert_allclose(arrays.left.frames["fridge"][:, 0], [20.0, 21.0])
    np.testing.assert_allclose(
        arrays.left.frames["fridge"], arrays.right.frames["fridge"]
    )
    assert arrays.left.frames["fridge"] is not arrays.right.frames["fridge"]


def test_v4_runtime_adapter_gives_both_arms_independent_frame_copies() -> None:
    left, right = store_bottle_semantic_observations_from_rlbench(_observation())

    assert tuple(left.frames) == ("bottle", "fridge")
    assert tuple(right.frames) == ("bottle", "fridge")
    np.testing.assert_allclose(left.frames["bottle"][:3], [10.0, 11.0, 12.0])
    np.testing.assert_allclose(left.frames["fridge"][:3], [20.0, 21.0, 22.0])
    left.frames["fridge"][0] = -1.0
    assert right.frames["fridge"][0] == 20.0


def test_v4_store_model_config_is_independent_of_the_v3_path() -> None:
    v3_path = (
        REPOSITORY_ROOT
        / "integrations"
        / "rlbench"
        / "configs"
        / "dynamac_rlbench_v3.json"
    )
    v4_path = STORE_BOTTLE_CONFIG_PATH.parent / "dynamac_store_bottle.json"

    assert v4_path.is_file()
    assert v4_path != v3_path
    assert json.loads(v4_path.read_text(encoding="utf-8")) == json.loads(
        v3_path.read_text(encoding="utf-8")
    )
