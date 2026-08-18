import hashlib
import json

import pytest

from integrations.rlbench.rlbench_dynamac.evaluation_split import (
    EVALUATION_SET_ROOT,
    load_evaluation_set_spec,
    load_training_split_manifest,
    validate_fixed_evaluation_split,
)


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def test_training_split_authenticates_exact_nine_cohorts_and_130_files():
    manifest = load_training_split_manifest()

    assert manifest["training_model_group_count"] == 9
    assert manifest["training_episode_count"] == 45
    assert manifest["training_file_count"] == 130
    assert sum(len(cohort["episodes"]) for cohort in manifest["cohorts"]) == 45
    assert sum(
        len(episode["files"])
        for cohort in manifest["cohorts"]
        for episode in cohort["episodes"]
    ) == 130
    table_ii = [
        cohort for cohort in manifest["cohorts"] if cohort["cohort_id"] == "table_ii"
    ]
    assert len(table_ii) == 4
    assert all(
        cohort["seed_provenance"]["status"]
        == "unknown_directory_label_unverified"
        and cohort["seed_provenance"]["conservative_reserved_seed_range"] == [0, 199]
        and cohort["seed_provenance"]["blocks_disjoint_high_seed_evaluation"]
        is False
        for cohort in table_ii
    )
    assert all(
        "evaluation_sets" not in record["path"].split("/")
        and "results" not in record["path"].split("/")
        for cohort in manifest["cohorts"]
        for episode in cohort["episodes"]
        for record in episode["files"].values()
    )


def test_fixed_spec_freezes_high_seed_variations_and_two_stage_publication():
    spec = load_evaluation_set_spec()

    assert spec["seed_namespace"]["base_seed"] == 2_608_000_000
    assert spec["episode_count_per_task"] == 200
    assert len(spec["dynamic_environment"]) == 8
    assert spec["dynamic_environment"]["place_cups"]["task_variation_count"] == 3
    assert spec["dynamic_environment"]["place_cups"][
        "evaluation_variation_schedule"
    ] == {"kind": "fixed", "value": 0}
    assert spec["coordination"]["bimanual_handover_item_dynamic"][
        "evaluation_variation_schedule"
    ] == {"kind": "episode_index_mod_task_variation_count"}
    assert spec["sealing"]["formal_generation_policy"] == (
        "require_prebuilt_sealed_artifacts_fail_if_missing"
    )
    assert spec["sealing"]["builder_may_read_results"] is False
    assert spec["sealing"]["result_based_candidate_selection_forbidden"] is True
    assert spec["sealing"]["result_based_scenario_tuning_forbidden"] is True
    assert spec["sealing"]["sealed_manifest_path"] == "manifest.json"


def test_fixed_split_enumerates_logical_and_derived_seed_separation():
    evidence = validate_fixed_evaluation_split()
    spec = load_evaluation_set_spec()

    assert evidence["validated"] is True
    assert spec["derived_rng_seed_namespace"]["source_selection"][
        "maximum_attempts"
    ] == 20
    assert spec["derived_rng_seed_namespace"]["goal_sampling"][
        "maximum_attempts"
    ] == 100
    assert evidence["evaluation_seed_minimum"] == 2_608_000_000
    assert evidence["evaluation_seed_maximum"] == 2_608_000_199
    assert evidence["derived_rng_seed_combinations_enumerated"] == 196_000
    assert evidence["derived_source_seed_minimum"] > 199
    assert evidence["derived_goal_seed_minimum"] > 199
    assert evidence["initial_state_zero_overlap_required"] is False
    assert (
        evidence["table_ii_unknown_seed_provenance_blocks_high_seed_evaluation"]
        is False
    )


def test_training_file_tamper_fails_closed_after_manifest_resigning(tmp_path):
    manifest = load_training_split_manifest(verify_files=False)
    first = manifest["cohorts"][0]["episodes"][0]["files"]["low_dim_obs"]
    first["sha256"] = "0" * 64
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
    manifest["fingerprint"] = _canonical_sha256(unsigned)
    path = tmp_path / "training_split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="training file hash mismatch"):
        load_training_split_manifest(path)


def test_training_manifest_cannot_be_resigned_to_read_results(tmp_path):
    manifest = load_training_split_manifest(verify_files=False)
    first = manifest["cohorts"][0]["episodes"][0]["files"]["low_dim_obs"]
    result_path = EVALUATION_SET_ROOT.parents[1] / "results" / "paper_comparison.json"
    first["path"] = "results/paper_comparison.json"
    first["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
    manifest["fingerprint"] = _canonical_sha256(unsigned)
    path = tmp_path / "training_split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="training episode file must be below data"):
        load_training_split_manifest(path)


def test_eval_set_paths_are_version_neutral_and_separate_from_training():
    spec = load_evaluation_set_spec()
    assert EVALUATION_SET_ROOT.name == "rlbench_fixed_v1"
    assert "evaluation_sets" in EVALUATION_SET_ROOT.parts
    assert spec["release_scope"] == "version_neutral_cross_model_evaluation"
