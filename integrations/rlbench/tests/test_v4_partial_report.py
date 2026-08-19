from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac import v4_partial_report
from integrations.rlbench.rlbench_dynamac.evaluation_videos import (
    CAPTURE_CONFIG_SCHEMA,
    DEFAULT_SELECTION_SEED,
    SELECTION_PROTOCOL_ID,
    SELECTION_SCHEMA,
    retention_quota,
)
from integrations.rlbench.rlbench_dynamac.store_bottle_eval_v4 import (
    V4_STORE_MOTION_PROTOCOL_ID,
    V4_STORE_RUNTIME_LOADER_ID,
)
from integrations.rlbench.rlbench_dynamac.v4_partial_report import (
    EVALUATION_PROTOCOL_ID,
    MULTI_FACTOR_NOTE,
    PartialReportValidationError,
    build_report,
    render_markdown,
    write_report,
)


STORE_TASK = "bimanual_put_bottle_in_fridge"
STORE_CELL = f"{STORE_TASK}/static"
STORE_RESULT_NAME = (
    "bimanual_put_bottle_in_fridge_static_seed2608000000_n200_h1000.json"
)


def _canonical_eval_identity():
    return {
        "schema": v4_partial_report.CANONICAL_EVAL_IDENTITY_SCHEMA,
        "evaluation_set_id": "rlbench_eval_v2",
        "manifest_sha256": "a" * 64,
        "manifest_fingerprint": "1" * 64,
        "spec_sha256": "b" * 64,
        "spec_fingerprint": "2" * 64,
        "selected_batches": {
            cell.cell_id: {"sha256": "c" * 64, "fingerprint": "d" * 64}
            for cell in v4_partial_report.TARGET_CELLS
        },
    }


@pytest.fixture(autouse=True)
def _bind_results_to_fixture_canonical_seal(monkeypatch):
    monkeypatch.setattr(
        v4_partial_report,
        "load_canonical_eval_identity",
        _canonical_eval_identity,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_identity_projection_maps_each_formal_cell_to_its_sealed_batch():
    loaded = {
        "manifest_sha256": "a" * 64,
        "payload": {
            "evaluation_set_id": "rlbench_eval_v2",
            "fingerprint": "1" * 64,
            "spec": {"sha256": "b" * 64, "fingerprint": "2" * 64},
            "environment_plan_batches": {
                STORE_TASK: {
                    "sha256": "c" * 64,
                    "batch_fingerprint": "d" * 64,
                },
                "bimanual_lift_tray": {
                    "sha256": "e" * 64,
                    "batch_fingerprint": "f" * 64,
                },
            },
            "coordination_source_batch": {
                "sha256": "3" * 64,
                "batch_fingerprint": "4" * 64,
            },
        },
    }

    identity = v4_partial_report.canonical_eval_identity_from_loaded_manifest(
        loaded
    )

    assert identity["selected_batches"][f"{STORE_TASK}/static"] == {
        "sha256": "c" * 64,
        "fingerprint": "d" * 64,
    }
    assert identity["selected_batches"]["bimanual_lift_tray/teleport"] == {
        "sha256": "e" * 64,
        "fingerprint": "f" * 64,
    }
    assert identity["selected_batches"][
        "bimanual_handover_item_dynamic/coordination_hand_left"
    ] == {"sha256": "3" * 64, "fingerprint": "4" * 64}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _episode_rows(successes: int):
    return [
        {
            "episode": episode,
            "success": episode < successes,
            "reason": "success" if episode < successes else "horizon",
            "invalid_actions": int(episode % 5 == 0),
        }
        for episode in range(200)
    ]


def _store_model_identity():
    training_identity = {
        "schema": "rlbench-store-bottle-training-identity-v4",
        "policy_spec": {
            "task": STORE_TASK,
            "semantic_schema": "rlbench-store-bottle-semantic-scene-v4",
            "semantic_version": "store_bottle_clean_v4",
            "semantic_fingerprint": "1" * 64,
            "frame_names": ["bottle", "fridge"],
            "bimanual": True,
        },
        "collection": {
            "demonstrations": 5,
            "all_success_verified": True,
            "manifest_sha256": "2" * 64,
            "manifest_fingerprint": "3" * 64,
            "seeds": [4_104_000_000 + index for index in range(5)],
        },
        "policy_config": {"sha256": "4" * 64},
        "evaluation_artifacts_included": False,
        "tasks_trained": [STORE_TASK],
        "other_tasks_trained": False,
    }
    encoded = json.dumps(
        training_identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    training_identity["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return {
        "training_manifest_schema": "dynamac-direct-training-v4",
        "manifest_authenticated": True,
        "training_identity": training_identity,
    }


def _video_evidence(
    results_root: Path,
    *,
    task: str,
    scenario: str,
    successes: int,
    paper_target: float,
):
    cell_id = f"{task}/{scenario}"
    cell_dir = results_root / "evaluation_videos" / task / scenario
    cell_dir.mkdir(parents=True)
    rate = successes / 200.0
    quota = retention_quota(rate, paper_target)
    selected_successes = list(range(min(successes, quota.successes)))
    selected_failures = list(
        range(successes, successes + min(200 - successes, quota.failures))
    )
    selected_ids = set(selected_successes + selected_failures)
    selected = []
    deleted = []
    for episode in range(200):
        base = {
            "episode": episode,
            "episode_seed": 2_608_000_000 + episode,
            "outcome": "success" if episode < successes else "failure",
            "video": f"episode_{episode:03d}.mp4",
            "companions": [f"episode_{episode:03d}.json"],
        }
        if episode in selected_ids:
            video = cell_dir / base["video"]
            sidecar = cell_dir / base["companions"][0]
            video.write_bytes(f"video-{episode}".encode("ascii"))
            sidecar.write_text(
                json.dumps({"episode": episode}), encoding="utf-8"
            )
            selected.append(
                {
                    **base,
                    "video_sha256": _sha256(video),
                    "video_bytes": video.stat().st_size,
                }
            )
        else:
            deleted.append(base)
    retained = {
        "successes": len(selected_successes),
        "failures": len(selected_failures),
    }
    selection = {
        "protocol_id": SELECTION_PROTOCOL_ID,
        "seed": DEFAULT_SELECTION_SEED,
        "tier": quota.tier,
        "available": {"successes": successes, "failures": 200 - successes},
        "requested": {
            "successes": quota.successes,
            "failures": quota.failures,
        },
        "retained": retained,
        "paper_close_enough_for_zero": quota.paper_close_enough_for_zero,
    }
    manifest = {
        "schema": SELECTION_SCHEMA,
        "cell_key": cell_id,
        "cell_metadata": {},
        "cell_result": {
            "successes": successes,
            "episodes": 200,
            "success_rate": rate,
            "paper_success_rate": paper_target,
        },
        "selection": selection,
        "selected": selected,
        "deleted": deleted,
        "all_episodes_recorded_before_selection": True,
        "unselected_artifacts_deleted": True,
    }
    manifest_path = cell_dir / "video_selection_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "release": "v4",
        "cell_key": cell_id,
        "cell_dir": str(cell_dir.resolve()),
        "episodes_recorded": 200,
        "capture_config": {
            "schema": CAPTURE_CONFIG_SCHEMA,
            "camera": "front",
            "capture_granularity": "returned_high_level_observations",
            "streamed_without_frame_buffer": True,
        },
        "paper_success_rate": paper_target,
        "selection_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "schema": SELECTION_SCHEMA,
            "selection": selection,
            "selected_episodes": [row["episode"] for row in selected],
            "all_episodes_recorded_before_selection": True,
        },
        "formal_result_committed_after_video_selection": True,
    }


def _store_result(results_root: Path, *, successes: int = 164):
    rows = _episode_rows(successes)
    rate = successes / 200.0
    return {
        "release": "v4",
        "task": STORE_TASK,
        "scenario": "static",
        "episodes": 200,
        "episodes_requested": 200,
        "episodes_completed": 200,
        "seed": 2_608_000_000,
        "successes": successes,
        "success_rate": rate,
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "fixed_eval_set": {
            "evaluation_set_id": "rlbench_eval_v2",
            "manifest_sha256": "a" * 64,
            "spec_sha256": "b" * 64,
            "selected_batch_sha256": "c" * 64,
            "selected_batch_fingerprint": "d" * 64,
            "formal_access": "canonical_id_read_only_no_generation",
        },
        "controller": {
            "primary_ik": "jacobian",
            "fallback_ik": "sampling",
            "sampling_ignore_collisions": False,
            "sampling_candidate_selection": "nearest_current_joint_l2",
            "ik_candidate_validation": "shape_finite_noncyclic_joint_limits",
            "primary_action_retry": {
                "max_primary_action_attempts_per_policy_tick": 3,
            },
        },
        "scenario_protocol": {
            "status": "STATIC_REFERENCE",
            "motion_protocol": {"protocol_id": V4_STORE_MOTION_PROTOCOL_ID},
            "staged_motion_plan_cache": {
                "runtime_protocol_id": V4_STORE_MOTION_PROTOCOL_ID,
                "task_scoped_envelope": {
                    "runtime_loader": V4_STORE_RUNTIME_LOADER_ID,
                    "task_identity_fingerprint": "e" * 64,
                    "batch_fingerprint": "f" * 64,
                },
            },
        },
        "episode_accounting": {
            "planned_episode_denominator": 200,
            "completed_episode_count": 200,
            "successes_in_planned_denominator": successes,
            "success_rate_all_planned_episodes": rate,
        },
        "ik_execution_diagnostics": {
            "jacobian_failures": 12,
            "sampling_fallback_successes": 8,
            "sampling_fallback_failures": 4,
        },
        "model_identity": _store_model_identity(),
        "store_mode_subgroups": {
            "bottle_only": {
                "planned": 67,
                "completed": 67,
                "successes": 55,
                "success_rate": 55 / 67.0,
                "intervention_reached": 50,
            },
            "fridge_only": {
                "planned": 67,
                "completed": 67,
                "successes": 55,
                "success_rate": 55 / 67.0,
            },
            "both": {
                "planned": 66,
                "completed": 66,
                "successes": successes - 110,
                "success_rate": (successes - 110) / 66.0,
            },
        },
        "results": rows,
        "evaluation_video_capture": _video_evidence(
            results_root,
            task=STORE_TASK,
            scenario="static",
            successes=successes,
            paper_target=0.82,
        ),
    }


def _write_v3_store(v3_root: Path, successes: int = 159):
    payload = {
        "task": STORE_TASK,
        "scenario": "static",
        "episodes": 200,
        "episodes_requested": 200,
        "episodes_completed": 200,
        "successes": successes,
        "success_rate": successes / 200.0,
        "results": _episode_rows(successes),
    }
    path = (
        v3_root
        / "table_ii"
        / "bimanual_put_bottle_in_fridge_static_seed2608000000_n200_h1000.json"
    )
    _write_json(path, payload)


def _write_store_result(results_root: Path, payload) -> Path:
    path = results_root / "table_ii" / STORE_RESULT_NAME
    _write_json(path, payload)
    return path


def test_missing_results_stay_not_run_and_never_form_a_full_table(tmp_path):
    report = build_report(
        results_root=tmp_path / "absent-v4",
        v3_results_root=tmp_path / "absent-v3",
    )

    assert report["scope"]["completed_validated_cell_count"] == 0
    assert report["scope"]["not_run_target_cell_count"] == 6
    assert report["scope"]["full_tables_generated"] is False
    assert report["scope"]["cross_cell_average_computed"] is False
    assert report["cross_cell_aggregate"]["computed"] is False
    assert len(report["cells"]) == 6
    assert all(cell["status"] == "NOT_RUN" for cell in report["cells"])
    assert len(report["formal_cell_status"]) == 25
    assert all(
        row["status"] == "NOT_RUN" for row in report["formal_cell_status"]
    )
    markdown = render_markdown(report)
    assert "Tables I–III" in markdown
    assert "no cross-cell average" in markdown


def test_present_cell_is_strictly_validated_and_reported_with_v3_caveat(tmp_path):
    results_root = tmp_path / "results" / "v4"
    v3_root = tmp_path / "results" / "v3"
    _write_store_result(results_root, _store_result(results_root))
    _write_v3_store(v3_root)

    report = build_report(results_root=results_root, v3_results_root=v3_root)
    store = next(cell for cell in report["cells"] if cell["cell_id"] == STORE_CELL)

    assert report["scope"]["completed_validated_cell_count"] == 1
    assert store["status"] == "COMPLETED_VALIDATED"
    assert store["successes"] == 164
    assert store["success_rate"] == pytest.approx(0.82)
    assert store["gap_to_paper_percentage_points"] == pytest.approx(0.0)
    assert store["terminal_reason_counts"] == [
        {"reason": "success", "count": 164},
        {"reason": "horizon", "count": 36},
    ]
    assert store["store_subgroups"]["source_field"] == "store_mode_subgroups"
    assert store["videos"]["retained"] == {"successes": 0, "failures": 0}
    assert store["videos"]["retained_video_paths"] == []
    comparison = store["v3_comparison"]
    assert comparison["success_rate"] == pytest.approx(0.795)
    assert comparison["v4_minus_v3_percentage_points"] == pytest.approx(2.5)
    assert comparison["causal_attribution"] is False
    assert comparison["interpretation"] == MULTI_FACTOR_NOTE

    json_output = tmp_path / "out" / "partial.json"
    markdown_output = tmp_path / "out" / "partial.md"
    write_report(
        report,
        json_output=json_output,
        markdown_output=markdown_output,
    )
    assert json.loads(json_output.read_text(encoding="utf-8")) == report
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "multi-factor" in markdown.lower()
    assert "bottle_only" in markdown


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("episodes_completed", 199, "episodes_completed"),
        ("evaluation_protocol_id", "legacy-v3", "collision-aware V4"),
    ),
)
def test_present_result_fails_closed_on_core_identity_or_accounting(
    tmp_path, field, bad_value, message
):
    results_root = tmp_path / "results" / "v4"
    payload = _store_result(results_root)
    payload[field] = bad_value
    _write_store_result(results_root, payload)

    with pytest.raises(PartialReportValidationError, match=message):
        build_report(results_root=results_root, v3_results_root=tmp_path / "v3")


def test_present_result_fails_closed_on_fixed_eval_identity(tmp_path):
    results_root = tmp_path / "results" / "v4"
    payload = _store_result(results_root)
    payload["fixed_eval_set"]["evaluation_set_id"] = "rlbench_fixed_v1"
    _write_store_result(results_root, payload)

    with pytest.raises(PartialReportValidationError, match="rlbench_eval_v2"):
        build_report(results_root=results_root, v3_results_root=tmp_path / "v3")


@pytest.mark.parametrize(
    "field",
    (
        "manifest_sha256",
        "spec_sha256",
        "selected_batch_sha256",
        "selected_batch_fingerprint",
    ),
)
def test_present_result_must_match_each_current_canonical_hash(tmp_path, field):
    results_root = tmp_path / "results" / "v4"
    payload = _store_result(results_root)
    payload["fixed_eval_set"][field] = "9" * 64
    _write_store_result(results_root, payload)

    with pytest.raises(
        PartialReportValidationError,
        match=rf"fixed_eval_set\.{field}.*current canonical",
    ):
        build_report(results_root=results_root, v3_results_root=tmp_path / "v3")


def test_store_stat_fields_are_flexible_but_training_identity_is_fail_closed(tmp_path):
    results_root = tmp_path / "results" / "v4"
    payload = _store_result(results_root)
    payload["store_mode_subgroups"] = {
        "bottle_only": {"episodes": 67, "successes": 55, "rate": 55 / 67.0}
    }
    _write_store_result(results_root, payload)
    report = build_report(results_root=results_root, v3_results_root=tmp_path / "v3")
    store = next(cell for cell in report["cells"] if cell["cell_id"] == STORE_CELL)
    subgroup = store["store_subgroups"]["groups"]["bottle_only"]
    assert subgroup["completed"] == 67
    assert subgroup["success_rate"] == pytest.approx(55 / 67.0)

    payload["model_identity"]["training_identity"]["policy_spec"][
        "semantic_version"
    ] = "legacy"
    _write_store_result(results_root, payload)

    with pytest.raises(PartialReportValidationError, match="training identity"):
        build_report(results_root=results_root, v3_results_root=tmp_path / "v3")


def test_present_result_fails_closed_on_video_manifest_hash(tmp_path):
    results_root = tmp_path / "results" / "v4"
    payload = _store_result(results_root)
    payload["evaluation_video_capture"]["selection_manifest"]["sha256"] = "0" * 64
    _write_store_result(results_root, payload)

    with pytest.raises(PartialReportValidationError, match="manifest hash mismatch"):
        build_report(results_root=results_root, v3_results_root=tmp_path / "v3")


def test_retained_video_path_and_hash_are_reported(tmp_path):
    results_root = tmp_path / "results" / "v4"
    payload = _store_result(results_root, successes=160)
    _write_store_result(results_root, payload)

    report = build_report(results_root=results_root, v3_results_root=tmp_path / "v3")
    store = next(cell for cell in report["cells"] if cell["cell_id"] == STORE_CELL)

    assert store["videos"]["retained"] == {"successes": 3, "failures": 3}
    assert len(store["videos"]["retained_video_paths"]) == 6
    assert all(path.endswith(".mp4") for path in store["videos"]["retained_video_paths"])

    retained = results_root / store["videos"]["retained_video_paths"][0]
    retained.write_bytes(b"tampered")
    with pytest.raises(PartialReportValidationError, match="hash/size mismatch"):
        build_report(results_root=results_root, v3_results_root=tmp_path / "v3")


def test_writer_refuses_to_put_result_state_in_evaluation_set(monkeypatch, tmp_path):
    report = build_report(
        results_root=tmp_path / "results" / "v4",
        v3_results_root=tmp_path / "results" / "v3",
    )
    monkeypatch.setattr(v4_partial_report, "INTEGRATION_ROOT", tmp_path)
    forbidden = tmp_path / "evaluation_sets" / "rlbench_eval_v2" / "report.json"

    with pytest.raises(PartialReportValidationError, match="never be written"):
        write_report(
            report,
            json_output=forbidden,
            markdown_output=tmp_path / "report.md",
        )
    assert not forbidden.exists()
