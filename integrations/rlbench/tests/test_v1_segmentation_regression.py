"""Artifact-backed proof that the core-module move preserves v1 segmentation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.tapas_segmentation import (
    load_rlbench_segmentation_config,
)

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = INTEGRATION_ROOT / "models" / "v1"
TABLE_I_ROOT = INTEGRATION_ROOT / "data" / "dynamac_table_i_live_g5_seed0"
TABLE_II_ROOT = (
    INTEGRATION_ROOT
    / "data"
    / "dynamac_table_ii_g5_a51b4e_128x128_seed0_20260811"
    / "stage_5_demos"
)
TABLE_III_ROOT = INTEGRATION_ROOT / "data" / "table_iii_coordination" / "g5_seed0"
TABLE_III_PROTOCOL = INTEGRATION_ROOT / "configs" / "table_iii_coordination_local.json"

STANDARD_COHORTS = (
    ("stack_wine", TABLE_I_ROOT),
    ("place_cups", TABLE_I_ROOT),
    ("open_microwave", TABLE_I_ROOT),
    ("wipe_desk", TABLE_I_ROOT),
    ("bimanual_put_bottle_in_fridge", TABLE_II_ROOT),
    ("bimanual_handover_item", TABLE_II_ROOT),
    ("bimanual_lift_tray", TABLE_II_ROOT),
    ("bimanual_sweep_to_dustpan", TABLE_II_ROOT),
)


def _episode_paths(data_root: Path, task: str) -> list[Path]:
    episode_root = data_root / task / "all_variations" / "episodes"
    return [episode_root / f"episode{index}" / "low_dim_obs.pkl" for index in range(5)]


def _assert_cohort_matches_manifest(
    task: str,
    data_root: Path,
    model_root: Path,
    *,
    segmentation_config=None,
) -> int:
    paths = _episode_paths(data_root, task)
    manifest_path = model_root / task / "training.json"
    if not manifest_path.is_file() or not all(path.is_file() for path in paths):
        pytest.skip("the retained v1 models and low-dimensional demonstrations are required")
    episodes = load_low_dim_obs_pickles(paths)
    names = [f"episode{index}" for index in range(5)]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["bimanual"]:
        converted = make_bimanual_demonstrations(
            episodes,
            task,
            names=names,
            config=segmentation_config,
        )
    else:
        converted = make_unimanual_demonstrations(
            episodes,
            task,
            names=names,
            config=segmentation_config,
        )
    assert converted.audit["segmentation"] == manifest["adapter"]["segmentation"]
    return len(episodes)


def test_all_45_retained_demonstrations_reproduce_v1_segmentation_exactly() -> None:
    demonstration_count = sum(
        _assert_cohort_matches_manifest(task, data_root, MODELS_ROOT)
        for task, data_root in STANDARD_COHORTS
    )

    protocol = json.loads(TABLE_III_PROTOCOL.read_text(encoding="utf-8"))["segmentation"]
    base = load_rlbench_segmentation_config().for_task("bimanual_handover_item")
    table_iii_config = replace(
        base,
        boundary_selection=protocol["boundary_selection"],
        expected_boundary_count=protocol["expected_boundary_count"],
        provenance={
            **{key: value for key, value in base.provenance.items() if key != "task_profiles"},
            "task_profiles": {},
            "coordination_local_protocol": protocol,
        },
    )
    demonstration_count += _assert_cohort_matches_manifest(
        "bimanual_handover_item",
        TABLE_III_ROOT,
        MODELS_ROOT / "table_iii",
        segmentation_config=table_iii_config,
    )
    assert demonstration_count == 45
