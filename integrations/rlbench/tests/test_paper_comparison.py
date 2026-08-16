from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from essay2608.policy import DynaMACConfig

from integrations.rlbench.rlbench_dynamac.paper_comparison import (
    EXPECTED_EVALUATION_PROTOCOL_ID,
    EXPECTED_LOCAL_CONFIG,
    EXPECTED_SELECTION_SEMANTICS_ID,
    EXPECTED_TAPAS_COMMIT,
    INTEGRATION_ROOT,
    build_document,
    build_records,
    discover_runs,
    markdown,
)


def _write_run(
    path,
    *,
    task,
    scenario,
    rate,
    family=None,
    paper_comparable=None,
    event=None,
    effective_interventions=None,
    covariance_method=None,
    corrected=True,
    variation=0,
    fingerprint="test-fingerprint",
):
    successes = int(rate * 200)
    payload = {
        "task": task,
        "scenario": scenario,
        "seed": 0,
        "episodes": 200,
        "horizon": 1000,
        "variation": variation,
        "successes": successes,
        "success_rate": successes / 200,
        "results": [],
    }
    for episode in range(200):
        row = {"episode": episode, "success": episode < successes}
        if event is not None:
            row["scenario_events"] = [dict(event)]
        payload["results"].append(row)
    if family == "table_i":
        payload["schema"] = "dynamac-table-i-evaluation-v1"
    if corrected:
        payload["model_identity"] = {
            "model_schema_version": 13,
            "selection_semantics_id": EXPECTED_SELECTION_SEMANTICS_ID,
            "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
            "training_config": dict(EXPECTED_LOCAL_CONFIG),
            "fingerprint": fingerprint,
            "manifest_authenticated": True,
        }
        payload["evaluation_protocol_id"] = EXPECTED_EVALUATION_PROTOCOL_ID
    elif covariance_method is not None:
        payload["model_identity"] = {
            "model_schema_version": 13,
            "selection_semantics_id": EXPECTED_SELECTION_SEMANTICS_ID,
            "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
            "training_config": {
                **EXPECTED_LOCAL_CONFIG,
                "covariance_estimation_method": covariance_method,
            },
            "fingerprint": fingerprint,
            "manifest_authenticated": True,
        }
        payload["evaluation_protocol_id"] = EXPECTED_EVALUATION_PROTOCOL_ID
    if paper_comparable is not None:
        payload["scenario_protocol"] = {
            "protocol_valid": True,
            "paper_comparable": paper_comparable,
        }
        if effective_interventions is not None:
            payload["scenario_protocol"][
                "episodes_with_effective_intervention"
            ] = effective_interventions
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_consolidated_report_separates_reproduction_and_diagnostic(tmp_path):
    _write_run(
        tmp_path / "table_i" / "stack_wine_static_seed0_n200_h1000.json",
        task="stack_wine",
        scenario="static",
        rate=1.0,
        family="table_i",
    )
    _write_run(
        tmp_path / "bimanual_put_bottle_in_fridge_static_seed0_n200_h1000.json",
        task="bimanual_put_bottle_in_fridge",
        scenario="static",
        rate=0.85,
    )
    _write_run(
        tmp_path / "bimanual_put_bottle_in_fridge_teleport_seed0_n200_h1000.json",
        task="bimanual_put_bottle_in_fridge",
        scenario="teleport",
        rate=0.80,
        paper_comparable=False,
    )

    records, warnings = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    document = build_document(
        records, warnings, seed=0, episodes=200, horizon=1000
    )
    text = markdown(document)

    stack_static = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "StackWine"
    )
    store_static = next(
        row for row in records if row["table"] == "II" and row["task"] == "StoreBottle"
    )
    store_dynamic = next(
        row for row in records if row["table"] == "III" and row["task"] == "StoreBottle"
    )
    hardware = next(row for row in records if row["table"] == "IV")

    assert stack_static["status"] == "local reproduction"
    assert store_static["status"] == "local reproduction"
    assert store_dynamic["status"] == "non-comparable diagnostic"
    assert hardware["status"] == "hardware unavailable"
    assert "0.94" in text
    assert "non-comparable diagnostic" in text


def test_sweepdust_row_links_the_diagnosis_without_changing_selection(tmp_path):
    _write_run(
        tmp_path / "bimanual_sweep_to_dustpan_static_seed0_n200_h1000.json",
        task="bimanual_sweep_to_dustpan",
        scenario="static",
        rate=0.025,
    )

    records, warnings = build_records(
        tmp_path, seed=0, episodes=200, horizon=1000
    )
    document = build_document(
        records, warnings, seed=0, episodes=200, horizon=1000
    )
    sweep = next(
        row for row in records if row["table"] == "II" and row["task"] == "SweepDust"
    )

    assert sweep["local_success_rate"] == 0.025
    assert "[SweepDust diagnosis](sweep_dust_diagnosis.md)" in markdown(document)


def test_table_i_static_is_reproduction_despite_family_level_dynamic_flag(tmp_path):
    _write_run(
        tmp_path / "table_i" / "stack_wine_static_seed0_n200_h1000.json",
        task="stack_wine",
        scenario="static",
        rate=1.0,
        family="table_i",
        paper_comparable=False,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack_static = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "StackWine"
    )

    assert stack_static["status"] == "local reproduction"


def test_wrong_episode_count_is_not_selected(tmp_path):
    path = tmp_path / "bimanual_handover_item_static_seed0_n100_h1000.json"
    successes = 90
    payload = {
        "task": "bimanual_handover_item",
        "scenario": "static",
        "seed": 0,
        "episodes": 100,
        "horizon": 1000,
        "successes": successes,
        "success_rate": 0.9,
        "results": [
            {"episode": episode, "success": episode < successes}
            for episode in range(100)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    handover = next(
        row for row in records if row["table"] == "II" and row["task"] == "HandOver"
    )
    assert handover["status"] == "pending"
    assert handover["local_success_rate"] is None


def test_table_i_selection_requires_variation_zero(tmp_path):
    _write_run(
        tmp_path / "table_i" / "place_cups_static_variation1_seed0_n200_h1000.json",
        task="place_cups",
        scenario="static",
        rate=1.0,
        family="table_i",
        variation=1,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    place = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "PlaceCups"
    )

    assert place["status"] == "pending"
    assert place["local_success_rate"] is None


def test_corrected_self_identifying_model_is_preferred_over_legacy_result(tmp_path):
    task = "bimanual_put_bottle_in_fridge"
    _write_run(
        tmp_path / f"{task}_static_seed0_n200_h1000.json",
        task=task,
        scenario="static",
        rate=0.85,
        corrected=False,
    )
    _write_run(
        tmp_path / "ridge_local" / f"{task}_static_seed0_n200_h1000.json",
        task=task,
        scenario="static",
        rate=0.80,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    store = next(
        row for row in records if row["table"] == "II" and row["task"] == "StoreBottle"
    )

    assert store["local_success_rate"] == 0.80
    assert store["source_file"].startswith("ridge_local/")


def test_policy_server_full_config_is_recognized_as_corrected(tmp_path):
    raw_config = json.loads(
        (INTEGRATION_ROOT / "configs" / "dynamac_rlbench_local.json").read_text(
            encoding="utf-8"
        )
    )
    policy_server_config = asdict(DynaMACConfig(**raw_config))
    assert len(policy_server_config) > len(raw_config)

    path = tmp_path / "table_i" / "place_cups_static_seed0_n200_h1000.json"
    _write_run(
        path,
        task="place_cups",
        scenario="static",
        rate=0.99,
        family="table_i",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_identity"]["training_config"] = policy_server_config
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    place = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "PlaceCups"
    )

    assert place["status"] == "local reproduction"
    assert place["local_success_rate"] == 0.99


def test_two_corrected_checkpoint_identities_fail_closed(tmp_path):
    task = "bimanual_put_bottle_in_fridge"
    for directory, fingerprint in (("run_a", "fingerprint-a"), ("run_b", "fingerprint-b")):
        _write_run(
            tmp_path / directory / f"{task}_static_seed0_n200_h1000.json",
            task=task,
            scenario="static",
            rate=0.80,
            fingerprint=fingerprint,
        )

    with pytest.raises(RuntimeError, match="multiple corrected results"):
        build_records(tmp_path, seed=0, episodes=200, horizon=1000)


def test_release_filter_allows_versioned_results_to_coexist(tmp_path):
    task = "bimanual_put_bottle_in_fridge"
    for release, fingerprint in (("v1", "fingerprint-v1"), ("v2", "fingerprint-v2")):
        _write_run(
            tmp_path / release / f"{task}_static_seed0_n200_h1000.json",
            task=task,
            scenario="static",
            rate=0.80,
            fingerprint=fingerprint,
        )

    records, _ = build_records(
        tmp_path,
        seed=0,
        episodes=200,
        horizon=1000,
        release="v1",
    )
    store = next(
        row for row in records if row["table"] == "II" and row["task"] == "StoreBottle"
    )

    assert store["source_file"].startswith("v1/")


def test_authenticated_coordination_result_is_a_table_iii_diagnostic(tmp_path):
    _write_run(
        tmp_path / "table_iii_coordination" / "hand_left_seed0_n200_h1000.json",
        task="bimanual_handover_item_dynamic",
        scenario="coordination_hand_left",
        rate=0.75,
        paper_comparable=False,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    hand_left = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Coordination"
        and row["task"] == "Hand Left"
    )

    assert hand_left["status"] == "non-comparable diagnostic"
    assert hand_left["local_success_rate"] == 0.75


def test_legacy_coordination_n1_smoke_is_silently_ignored(tmp_path):
    smoke = tmp_path / "table_iii_coordination" / "smoke_left_seed10000_n1.json"
    smoke.parent.mkdir(parents=True)
    smoke.write_text(
        json.dumps(
            {
                "task": "bimanual_handover_item_dynamic",
                "scenario": "coordination_hand_left",
                "seed": 10000,
                "episodes": [{"episode": 0, "success": False}],
                "horizon": 1000,
                "successes": 0,
                "success_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )

    runs, warnings = discover_runs(tmp_path)

    assert runs == []
    assert warnings == []


def test_wipe_desk_dynamic_cells_are_unavailable_not_pending(tmp_path):
    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)

    rows = [
        row
        for row in records
        if row["table"] == "I"
        and row["task"] == "WipeDesk"
        and row["condition"] != "Static"
    ]

    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"unavailable"}
    assert all("cannot restore WipeDesk" in row["notes"] for row in rows)


def test_place_cups_without_a_complete_corrected_run_is_pending(tmp_path):
    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    rows = [
        row
        for row in records
        if row["table"] == "I" and row["task"] == "PlaceCups"
    ]

    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"pending"}
    assert all(row["local_success_rate"] is None for row in rows)
    assert all(row["local_episodes"] is None for row in rows)


def test_table_iii_uses_teleport_and_ignores_extra_smooth_run(tmp_path):
    task = "bimanual_lift_tray"
    _write_run(
        tmp_path / f"{task}_teleport_seed0_n200_h1000.json",
        task=task,
        scenario="teleport",
        rate=0.90,
        paper_comparable=False,
        event={"applied": True, "task_state_l2": 0.3, "root_pose_l2": 0.2},
        effective_interventions=200,
    )
    _write_run(
        tmp_path / f"{task}_smooth_seed0_n200_h1000.json",
        task=task,
        scenario="smooth",
        rate=0.92,
        paper_comparable=False,
        event={
            "applied": True,
            "task_state_l2": 0.2,
            "root_pose_l2": 0.02,
            "planned_root_translation_m": 0.15,
        },
        effective_interventions=200,
    )

    records, warnings = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    document = build_document(
        records,
        warnings,
        seed=0,
        episodes=200,
        horizon=1000,
    )
    lift = next(
        row
        for row in records
        if row["table"] == "III" and row["task"] == "LiftTray"
    )

    assert lift["local_success_rate"] == 0.90
    assert lift["source_file"].endswith("_teleport_seed0_n200_h1000.json")
    assert "median root-pose L2 0.200" in lift["protocol_note"]
    assert "Additional local diagnostics" not in markdown(document)


def test_table_iii_teleport_rows_include_task_specific_material_notes(tmp_path):
    cases = (
        ("bimanual_put_bottle_in_fridge", "StoreBottle", 0.8, 0.0),
        ("bimanual_handover_item", "HandOver", 2.0, 0.0),
        ("bimanual_sweep_to_dustpan", "SweepDust", 1.0e-6, 0.0),
        ("bimanual_lift_tray", "LiftTray", 0.3, 0.2),
    )
    for task, _label, state_l2, root_l2 in cases:
        _write_run(
            tmp_path / f"{task}_teleport_seed0_n200_h1000.json",
            task=task,
            scenario="teleport",
            rate=0.5,
            paper_comparable=False,
            event={
                "applied": True,
                "task_state_l2": state_l2,
                "root_pose_l2": root_l2,
            },
            effective_interventions=200,
        )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    notes = {
        row["task"]: row["protocol_note"]
        for row in records
        if row["table"] == "III" and row["condition"] == "Dynamic environment"
    }

    assert "static-workspace kidnap()" in notes["StoreBottle"]
    assert "release an already-grasped item" in notes["HandOver"]
    assert "No material candidate/root motion" in notes["SweepDust"]
    assert "Material task-root motion" in notes["LiftTray"]
