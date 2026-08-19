"""Version-neutral RLBench training/evaluation split authentication.

This module contains no simulator imports and is Python 3.8 compatible.  It
authenticates the 45 demonstrations used by the nine fitted model groups and
the preregistered high-seed evaluation-set specification.  It deliberately
does not inspect evaluation results or require initial-state inequality.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SET_ROOT = INTEGRATION_ROOT / "evaluation_sets" / "rlbench_fixed_v1"
TRAINING_SPLIT_MANIFEST = EVALUATION_SET_ROOT / "training_split_manifest.json"
EVALUATION_SET_SPEC = EVALUATION_SET_ROOT / "spec.json"
EVALUATION_SET_V2_ID = "rlbench_eval_v2"
EVALUATION_SET_V2_CONFIG_SPEC = (
    INTEGRATION_ROOT / "configs" / "v4" / "evaluation_set_spec.json"
)

TRAINING_SPLIT_SCHEMA = "dynamac-rlbench-training-split-manifest-v1"
EVALUATION_SET_SPEC_SCHEMA = "dynamac-rlbench-fixed-evaluation-spec-v1"
EVALUATION_SET_V2_SPEC_SCHEMA = "dynamac-rlbench-evaluation-spec-v2"

_TASK_SCOPED_IDENTITY_SCHEMA = "dynamac-rlbench-task-scoped-identity-v2"
_TASK_SCOPED_BATCH_SCHEMA = "dynamac-rlbench-task-scoped-motion-plan-batch-v2"
_TASK_SCOPED_BATCH_PROTOCOL_ID = (
    "rlbench-task-scoped-staged-motion-plan-envelope-v2"
)
_STORE_TRAINING_IDENTITY_SCHEMA = (
    "dynamac-rlbench-store-bottle-eval-binding-v4"
)
_REGENERATED_V2_TASKS = frozenset(
    {"bimanual_put_bottle_in_fridge", "bimanual_lift_tray"}
)

_DYNAMIC_TASKS = frozenset(
    {
        "stack_wine",
        "place_cups",
        "open_microwave",
        "wipe_desk",
        "bimanual_put_bottle_in_fridge",
        "bimanual_handover_item",
        "bimanual_lift_tray",
        "bimanual_sweep_to_dustpan",
    }
)


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split payload must be a JSON object")
    return payload


def _validate_fingerprint(payload, label):
    fingerprint = payload.get("fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "fingerprint"}
    if fingerprint != _canonical_sha256(unsigned):
        raise ValueError(f"{label} fingerprint mismatch")


def _resolve_training_path(relative_path):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("training file path must be a non-empty string")
    parts = Path(relative_path).parts
    if not parts or parts[0] in {"evaluation_sets", "results"}:
        raise ValueError("training file points into a reserved evaluation/output root")
    candidate = (INTEGRATION_ROOT / relative_path).resolve()
    root = INTEGRATION_ROOT.resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise ValueError("training file escapes the integration root")
    evaluation_root = EVALUATION_SET_ROOT.resolve()
    if os.path.commonpath((str(evaluation_root), str(candidate))) == str(
        evaluation_root
    ):
        raise ValueError("training file points into the evaluation-set directory")
    return candidate


def validate_training_entry_paths(*paths):
    """Reject training inputs/outputs that overlap sealed sets or results."""

    forbidden = (
        (INTEGRATION_ROOT / "evaluation_sets").resolve(),
        (INTEGRATION_ROOT / "results").resolve(),
    )
    for value in paths:
        candidate = Path(value).resolve()
        for root in forbidden:
            if os.path.commonpath((str(root), str(candidate))) == str(root):
                raise ValueError(
                    "training path overlaps evaluation artifacts or results"
                )


def _scheduled_variation(profile, episode_index):
    schedule = profile["evaluation_variation_schedule"]
    if schedule["kind"] == "fixed":
        return schedule["value"]
    return episode_index % profile["task_variation_count"]


def _derived_rng_seeds(logical_seed, variation):
    modulus = 2**32 - 1
    source = [logical_seed]
    source.extend(
        (logical_seed * 1_000_003 + variation * 9_176 + attempt * 104_729)
        % modulus
        for attempt in range(2, 21)
    )
    goals = [
        (logical_seed * 1_000_003 + variation * 9_176 + attempt * 7_919)
        % modulus
        for attempt in range(1, 101)
    ]
    return source, goals


def load_training_split_manifest(path=TRAINING_SPLIT_MANIFEST, verify_files=True):
    """Load the authenticated 45-demonstration training inventory."""

    payload = _load_object(path)
    if payload.get("schema") != TRAINING_SPLIT_SCHEMA:
        raise ValueError("unsupported training split schema")
    _validate_fingerprint(payload, "training split")
    if (
        set(payload)
        != {
            "schema",
            "split_id",
            "hash_algorithm",
            "training_episode_count",
            "training_file_count",
            "training_model_group_count",
            "evaluation_artifacts_included",
            "claim_boundary",
            "source_collection_manifests",
            "configuration_identity",
            "rlbench_source",
            "cohorts",
            "fingerprint",
        }
        or
        payload.get("split_id") != "rlbench_fixed_v1_training_inputs"
        or payload.get("hash_algorithm") != "sha256"
        or payload.get("evaluation_artifacts_included") is not False
        or payload.get("training_episode_count") != 45
        or payload.get("training_file_count") != 130
        or payload.get("training_model_group_count") != 9
    ):
        raise ValueError("training split header is invalid")
    cohorts = payload.get("cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != 9:
        raise ValueError("training split must contain nine cohorts")
    authenticated_inputs = {
        "source_collection_manifests": {
            "data/dynamac_table_i_live_g5_seed0/collection_manifest.json",
            "data/table_iii_coordination/g5_seed0/collection_manifest.json",
        },
        "configuration_identity": {
            "configs/dynamac_rlbench_v3.json",
            "configs/tapas_segmentation.json",
            "configs/tasks.json",
        },
    }
    for field, expected_paths in authenticated_inputs.items():
        records = payload.get(field)
        if not isinstance(records, list) or len(records) != len(expected_paths):
            raise ValueError(f"training split {field} is invalid")
        actual_paths = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError(f"training split {field} record is invalid")
            relative = record["path"]
            actual_paths.add(relative)
            resolved = _resolve_training_path(relative)
            if verify_files and (
                not resolved.is_file() or _file_sha256(resolved) != record["sha256"]
            ):
                raise ValueError(f"training identity file hash mismatch: {relative}")
        if actual_paths != expected_paths:
            raise ValueError(f"training split {field} path set is invalid")
    rlbench = payload.get("rlbench_source")
    if (
        not isinstance(rlbench, dict)
        or rlbench.get("repository") != "https://github.com/vonHartz/RLBench"
        or rlbench.get("branch") != "tapas"
        or rlbench.get("commit") != "a51b4e609dc5c3e1a8c06046bd87a9da24723da4"
    ):
        raise ValueError("training RLBench source identity is invalid")
    seen_tasks = []
    seen_paths = set()
    total = 0
    for cohort in cohorts:
        if not isinstance(cohort, dict):
            raise ValueError("training cohort must be an object")
        if set(cohort) != {
            "cohort_id",
            "task",
            "policy_task_alias",
            "descriptions_available",
            "seed_provenance",
            "episodes",
        }:
            raise ValueError("training cohort fields are invalid")
        task = cohort.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("training cohort task is invalid")
        seen_tasks.append((cohort.get("cohort_id"), task))
        provenance = cohort.get("seed_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("training cohort seed provenance is missing")
        status = provenance.get("status")
        if status not in {"explicit_collection_manifest", "unknown_directory_label_unverified"}:
            raise ValueError("training cohort seed provenance is invalid")
        if status == "unknown_directory_label_unverified":
            if provenance.get("conservative_reserved_seed_range") != [0, 199]:
                raise ValueError("unknown training seeds require the conservative reservation")
            if provenance.get("blocks_disjoint_high_seed_evaluation") is not False:
                raise ValueError("unknown low seeds must not block disjoint high-seed evaluation")
        episodes = cohort.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 5:
            raise ValueError("each training cohort must contain five episodes")
        if [episode.get("episode") for episode in episodes] != list(range(5)):
            raise ValueError("training episodes must be ordered zero through four")
        for episode in episodes:
            if set(episode) != {"episode", "seed", "variation", "files"}:
                raise ValueError("training episode fields are invalid")
            seed = episode.get("seed")
            if status == "explicit_collection_manifest":
                if not isinstance(seed, int) or isinstance(seed, bool):
                    raise ValueError("known training seed is invalid")
            elif seed is not None:
                raise ValueError("unknown training seed must be null")
            variation = episode.get("variation")
            if not isinstance(variation, int) or isinstance(variation, bool):
                raise ValueError("training variation is invalid")
            files = episode.get("files")
            required = {"low_dim_obs", "variation_number"}
            if cohort.get("descriptions_available") is True:
                required.add("variation_descriptions")
            if not isinstance(files, dict) or set(files) != required:
                raise ValueError("training episode file roles are invalid")
            for record in files.values():
                if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                    raise ValueError("training file record is invalid")
                relative = record["path"]
                digest = record["sha256"]
                if not Path(relative).parts or Path(relative).parts[0] != "data":
                    raise ValueError("training episode file must be below data/")
                if relative in seen_paths:
                    raise ValueError("training file path is duplicated")
                seen_paths.add(relative)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                ):
                    raise ValueError("training file hash is invalid")
                resolved = _resolve_training_path(relative)
                if verify_files and (not resolved.is_file() or _file_sha256(resolved) != digest):
                    raise ValueError(f"training file hash mismatch: {relative}")
            total += 1
    expected = {
        ("table_i", task)
        for task in ("stack_wine", "place_cups", "open_microwave", "wipe_desk")
    } | {
        ("table_ii", task)
        for task in (
            "bimanual_put_bottle_in_fridge",
            "bimanual_handover_item",
            "bimanual_lift_tray",
            "bimanual_sweep_to_dustpan",
        )
    } | {("table_iii_coordination", "bimanual_handover_item_dynamic")}
    if set(seen_tasks) != expected or total != 45 or len(seen_paths) != 130:
        raise ValueError("training cohort identity set is invalid")
    return payload


def load_evaluation_set_spec(path=EVALUATION_SET_SPEC):
    """Load the preregistered, result-independent fixed evaluation spec."""

    payload = _load_object(path)
    if payload.get("schema") != EVALUATION_SET_SPEC_SCHEMA:
        raise ValueError("unsupported fixed evaluation-set schema")
    _validate_fingerprint(payload, "fixed evaluation spec")
    if (
        set(payload)
        != {
            "schema",
            "evaluation_set_id",
            "release_scope",
            "episode_count_per_task",
            "protocol_fingerprints",
            "seed_namespace",
            "derived_rng_seed_namespace",
            "dynamic_environment",
            "coordination",
            "sealing",
            "isolation",
            "fingerprint",
        }
        or
        payload.get("evaluation_set_id") != "rlbench_fixed_v1"
        or payload.get("release_scope") != "version_neutral_cross_model_evaluation"
        or payload.get("episode_count_per_task") != 200
    ):
        raise ValueError("fixed evaluation-set header is invalid")
    seed = payload.get("seed_namespace")
    if (
        not isinstance(seed, dict)
        or set(seed)
        != {
            "name",
            "base_seed",
            "derivation",
            "minimum_seed",
            "maximum_seed",
            "shared_numeric_seed_values_across_task_classes",
        }
        or seed.get("base_seed") != 2608000000
        or seed.get("derivation") != "base_seed + episode_index"
        or seed.get("minimum_seed") != 2608000000
        or seed.get("maximum_seed") != 2608000199
        or seed.get("shared_numeric_seed_values_across_task_classes") is not True
    ):
        raise ValueError("fixed evaluation seed namespace is invalid")
    if payload.get("protocol_fingerprints") != {
        "motion_source": {
            "schema": "rlbench-dynamac-v3-motion-sources-v1",
            "fingerprint": "884021e9e0c4da39ffe176c38ed05eb49e2279b60c1b1f1907897ea197754ddc",
        },
        "staged_motion_plan_batch": {
            "schema": "dynamac-rlbench-staged-motion-plan-batch-v3.4",
            "protocol_id": "rlbench-deterministic-source-staging-waypoint-validated-boundary-root-v3.4",
        },
    }:
        raise ValueError("fixed evaluation protocol fingerprints are invalid")
    derived = payload.get("derived_rng_seed_namespace")
    if derived != {
        "modulus": 4294967295,
        "source_selection": {
            "maximum_attempts": 20,
            "attempt_1": "logical_episode_seed",
            "fallback_attempts_2_through_20": "(logical_episode_seed*1000003 + variation*9176 + attempt*104729) % 4294967295",
        },
        "goal_sampling": {
            "maximum_attempts": 100,
            "attempts_1_through_100": "(logical_episode_seed*1000003 + variation*9176 + attempt*7919) % 4294967295",
        },
        "validator_enumerates_all_episode_variation_attempt_combinations": True,
    }:
        raise ValueError("fixed evaluation derived RNG namespace is invalid")
    dynamic = payload.get("dynamic_environment")
    if not isinstance(dynamic, dict) or frozenset(dynamic) != _DYNAMIC_TASKS:
        raise ValueError("fixed evaluation dynamic task set is invalid")
    paths = set()
    fixed_zero = {"kind": "fixed", "value": 0}
    episode_mod = {"kind": "episode_index_mod_task_variation_count"}
    expected_variations = {
        "stack_wine": (1, fixed_zero),
        "place_cups": (3, fixed_zero),
        "open_microwave": (1, fixed_zero),
        "wipe_desk": (1, fixed_zero),
        "bimanual_put_bottle_in_fridge": (1, episode_mod),
        "bimanual_handover_item": (5, episode_mod),
        "bimanual_lift_tray": (1, episode_mod),
        "bimanual_sweep_to_dustpan": (1, episode_mod),
    }
    for task, profile in dynamic.items():
        expected_count, expected_schedule = expected_variations[task]
        if (
            not isinstance(profile, dict)
            or set(profile)
            != {
                "task_variation_count",
                "evaluation_variation_schedule",
                "artifact_kind",
                "artifact_path",
                "consumers",
            }
            or profile.get("artifact_kind") != "A_B"
            or profile.get("task_variation_count") != expected_count
            or profile.get("evaluation_variation_schedule") != expected_schedule
            or profile.get("consumers") != ["static", "smooth", "teleport"]
        ):
            raise ValueError(f"fixed evaluation profile is invalid: {task}")
        artifact_path = profile.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path.startswith(
            "plans/environment/"
        ):
            raise ValueError("fixed evaluation A/B path is invalid")
        paths.add(artifact_path)
    coordination = payload.get("coordination")
    profile = (
        coordination.get("bimanual_handover_item_dynamic")
        if isinstance(coordination, dict)
        else None
    )
    if (
        not isinstance(profile, dict)
        or set(profile)
        != {
            "policy_task_alias",
            "task_variation_count",
            "evaluation_variation_schedule",
            "artifact_kind",
            "artifact_path",
            "consumers",
        }
        or profile.get("artifact_kind") != "A_only"
        or profile.get("task_variation_count") != 5
        or profile.get("evaluation_variation_schedule") != episode_mod
        or profile.get("consumers") != ["none", "hand_left", "hand_right"]
        or not isinstance(profile.get("artifact_path"), str)
        or not profile["artifact_path"].startswith("initializations/coordination/")
    ):
        raise ValueError("fixed coordination initialization profile is invalid")
    paths.add(profile["artifact_path"])
    if len(paths) != 9:
        raise ValueError("fixed evaluation artifact paths must be unique")
    sealing = payload.get("sealing")
    if (
        not isinstance(sealing, dict)
        or set(sealing)
        != {
            "sealed_manifest_path",
            "artifact_hashes_absent_until_builder_publication",
            "formal_generation_policy",
            "same_A_B_required_across_smooth_and_teleport",
            "same_A_required_across_all_scenarios",
            "builder_may_read_results",
            "result_based_candidate_selection_forbidden",
            "result_based_scenario_tuning_forbidden",
        }
        or sealing.get("sealed_manifest_path") != "manifest.json"
        or sealing.get("artifact_hashes_absent_until_builder_publication") is not True
        or sealing.get("formal_generation_policy")
        != "require_prebuilt_sealed_artifacts_fail_if_missing"
        or sealing.get("builder_may_read_results") is not False
        or sealing.get("result_based_candidate_selection_forbidden") is not True
        or sealing.get("result_based_scenario_tuning_forbidden") is not True
    ):
        raise ValueError("fixed evaluation sealing policy is invalid")
    isolation = payload.get("isolation")
    if (
        not isinstance(isolation, dict)
        or set(isolation)
        != {
            "training_data_root",
            "evaluation_artifact_root",
            "results_root",
            "training_must_not_read_evaluation_artifacts_or_results",
            "results_reference_sealed_manifest_and_artifact_hashes",
            "results_must_not_embed_or_modify_evaluation_artifacts",
        }
        or isolation.get("training_must_not_read_evaluation_artifacts_or_results")
        is not True
        or isolation.get("results_reference_sealed_manifest_and_artifact_hashes")
        is not True
        or isolation.get("results_must_not_embed_or_modify_evaluation_artifacts")
        is not True
    ):
        raise ValueError("fixed evaluation isolation policy is invalid")
    return payload


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_evaluation_result_state(value):
    """Keep result/run state out of a sealed evaluation-input definition."""

    forbidden = {
        "not_run",
        "result",
        "results",
        "run_status",
        "success",
        "successes",
        "success_rate",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in forbidden:
                raise ValueError(
                    "evaluation-input spec must not contain result or NOT_RUN state"
                )
            _reject_evaluation_result_state(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_evaluation_result_state(nested)
    elif isinstance(value, str) and value.strip().upper() == "NOT_RUN":
        raise ValueError("evaluation-input spec must not contain NOT_RUN state")


def load_evaluation_set_v2_spec(path=EVALUATION_SET_V2_CONFIG_SPEC):
    """Load the complete, result-independent ``rlbench_eval_v2`` input spec.

    V2 deliberately separates byte-identical legacy imports from task-scoped
    regenerated batches.  It describes available evaluation inputs only; the
    subset of cells selected for a particular report belongs in result
    metadata and is therefore rejected here.
    """

    payload = _load_object(path)
    if payload.get("schema") != EVALUATION_SET_V2_SPEC_SCHEMA:
        raise ValueError("unsupported evaluation-set v2 schema")
    _validate_fingerprint(payload, "evaluation-set v2 spec")
    _reject_evaluation_result_state(payload)
    if (
        set(payload)
        != {
            "schema",
            "evaluation_set_id",
            "release_scope",
            "episode_count_per_task",
            "training_identity_contract",
            "legacy_import",
            "protocol_contracts",
            "seed_namespace",
            "derived_rng_seed_namespace",
            "dynamic_environment",
            "coordination",
            "sealing",
            "isolation",
            "fingerprint",
        }
        or payload.get("evaluation_set_id") != EVALUATION_SET_V2_ID
        or payload.get("release_scope")
        != "v4_complete_task_scoped_evaluation_inputs"
        or payload.get("episode_count_per_task") != 200
    ):
        raise ValueError("evaluation-set v2 header is invalid")

    if payload.get("training_identity_contract") != {
        "kind": "legacy_non_store_plus_store_v4",
        "legacy_non_store": {
            "source": "rlbench_fixed_v1/training_split_manifest.json",
            "identity_mode": "sealed_v1_training_split_reference",
            "excluded_tasks": ["bimanual_put_bottle_in_fridge"],
        },
        "store_bottle_v4": {
            "binding_schema": _STORE_TRAINING_IDENTITY_SCHEMA,
            "demonstration_count": 5,
            "collection_root": "data/v4/store_bottle",
            "model_root": "models/v4",
            "required_model_manifest": True,
            "must_be_disjoint_from_evaluation_seed_range": [
                2_608_000_000,
                2_608_000_199,
            ],
        },
    }:
        raise ValueError("evaluation-set v2 training identity contract is invalid")

    legacy = payload.get("legacy_import")
    if (
        not isinstance(legacy, dict)
        or set(legacy)
        != {
            "source_evaluation_set_id",
            "source_manifest_sha256",
            "source_manifest_fingerprint",
            "source_manifest_path",
            "reference_mode",
            "source_is_read_only",
            "source_artifacts_remain_external",
            "preserve_legacy_internal_identity",
        }
        or legacy.get("source_evaluation_set_id") != "rlbench_fixed_v1"
        or not _is_sha256(legacy.get("source_manifest_sha256"))
        or not _is_sha256(legacy.get("source_manifest_fingerprint"))
        or legacy.get("source_manifest_path") != "manifest.json"
        or legacy.get("reference_mode")
        != "external_canonical_sealed_eval_set"
        or legacy.get("source_is_read_only") is not True
        or legacy.get("source_artifacts_remain_external") is not True
        or legacy.get("preserve_legacy_internal_identity") is not True
    ):
        raise ValueError("evaluation-set v2 legacy-import contract is invalid")

    protocols = payload.get("protocol_contracts")
    if protocols != {
        "reused_legacy": {
            "batch_schema": "dynamac-rlbench-staged-motion-plan-batch-v3.4",
            "protocol_id": (
                "rlbench-deterministic-source-staging-waypoint-validated-"
                "boundary-root-v3.4"
            ),
            "identity_mode": "legacy_global_identity_preserved",
        },
        "regenerated_v2": {
            "envelope_schema": _TASK_SCOPED_BATCH_SCHEMA,
            "protocol_id": _TASK_SCOPED_BATCH_PROTOCOL_ID,
            "identity_schema": _TASK_SCOPED_IDENTITY_SCHEMA,
            "identity_mode": "task_scoped_v2",
            "required_identity_components": [
                "task_semantics",
                "motion_source",
                "intervention",
            ],
        },
    }:
        raise ValueError("evaluation-set v2 protocol contracts are invalid")

    seed = payload.get("seed_namespace")
    if (
        not isinstance(seed, dict)
        or set(seed)
        != {
            "name",
            "base_seed",
            "derivation",
            "minimum_seed",
            "maximum_seed",
            "shared_numeric_seed_values_across_task_classes",
        }
        or seed.get("name") != "dynamac-heldout-evaluation-high-seed-v2"
        or seed.get("base_seed") != 2_608_000_000
        or seed.get("derivation") != "base_seed + episode_index"
        or seed.get("minimum_seed") != 2_608_000_000
        or seed.get("maximum_seed") != 2_608_000_199
        or seed.get("shared_numeric_seed_values_across_task_classes") is not True
    ):
        raise ValueError("evaluation-set v2 seed namespace is invalid")
    if payload.get("derived_rng_seed_namespace") != {
        "modulus": 4_294_967_295,
        "source_selection": {
            "maximum_attempts": 20,
            "attempt_1": "logical_episode_seed",
            "fallback_attempts_2_through_20": "(logical_episode_seed*1000003 + variation*9176 + attempt*104729) % 4294967295",
        },
        "goal_sampling": {
            "maximum_attempts": 100,
            "attempts_1_through_100": "(logical_episode_seed*1000003 + variation*9176 + attempt*7919) % 4294967295",
        },
        "validator_enumerates_all_episode_variation_attempt_combinations": True,
    }:
        raise ValueError("evaluation-set v2 derived RNG namespace is invalid")

    dynamic = payload.get("dynamic_environment")
    if not isinstance(dynamic, dict) or frozenset(dynamic) != _DYNAMIC_TASKS:
        raise ValueError("evaluation-set v2 dynamic task set is invalid")
    fixed_zero = {"kind": "fixed", "value": 0}
    episode_mod = {"kind": "episode_index_mod_task_variation_count"}
    expected_variations = {
        "stack_wine": (1, fixed_zero),
        "place_cups": (3, fixed_zero),
        "open_microwave": (1, fixed_zero),
        "wipe_desk": (1, fixed_zero),
        "bimanual_put_bottle_in_fridge": (1, episode_mod),
        "bimanual_handover_item": (5, episode_mod),
        "bimanual_lift_tray": (1, episode_mod),
        "bimanual_sweep_to_dustpan": (1, episode_mod),
    }
    artifact_paths = set()
    for task, profile in dynamic.items():
        regenerated = task in _REGENERATED_V2_TASKS
        common_fields = {
            "task_variation_count",
            "evaluation_variation_schedule",
            "artifact_kind",
            "artifact_path",
            "consumers",
            "artifact_origin",
        }
        expected_fields = common_fields | (
            {"task_identity_contract"}
            if regenerated
            else {"legacy_source_artifact_path"}
        )
        expected_count, expected_schedule = expected_variations[task]
        if (
            not isinstance(profile, dict)
            or set(profile) != expected_fields
            or profile.get("task_variation_count") != expected_count
            or profile.get("evaluation_variation_schedule") != expected_schedule
            or profile.get("artifact_kind") != "A_B"
            or profile.get("consumers")
            != (
                ["static", "teleport"]
                if regenerated
                else ["static", "smooth", "teleport"]
            )
            or profile.get("artifact_origin")
            != ("regenerated_v2" if regenerated else "reused_legacy")
        ):
            raise ValueError(f"evaluation-set v2 profile is invalid: {task}")
        artifact_path = profile.get("artifact_path")
        if (
            not isinstance(artifact_path, str)
            or not artifact_path.startswith("plans/environment/")
        ):
            raise ValueError("evaluation-set v2 A/B path is invalid")
        artifact_paths.add(artifact_path)
        if regenerated:
            if profile.get("task_identity_contract") != {
                "schema": _TASK_SCOPED_IDENTITY_SCHEMA,
                "scope": task,
                "required_components": [
                    "task_semantics",
                    "motion_source",
                    "intervention",
                ],
            }:
                raise ValueError(
                    f"evaluation-set v2 task identity is invalid: {task}"
                )
        elif profile.get("legacy_source_artifact_path") != artifact_path:
            raise ValueError(
                f"evaluation-set v2 legacy import path is invalid: {task}"
            )

    coordination = payload.get("coordination")
    coordination_profile = (
        coordination.get("bimanual_handover_item_dynamic")
        if isinstance(coordination, dict)
        else None
    )
    if (
        not isinstance(coordination_profile, dict)
        or set(coordination_profile)
        != {
            "policy_task_alias",
            "task_variation_count",
            "evaluation_variation_schedule",
            "artifact_kind",
            "artifact_path",
            "consumers",
            "artifact_origin",
            "legacy_source_artifact_path",
        }
        or coordination_profile.get("policy_task_alias")
        != "bimanual_handover_item"
        or coordination_profile.get("task_variation_count") != 5
        or coordination_profile.get("evaluation_variation_schedule") != episode_mod
        or coordination_profile.get("artifact_kind") != "A_only"
        or coordination_profile.get("consumers")
        != ["none", "hand_left", "hand_right"]
        or coordination_profile.get("artifact_origin") != "reused_legacy"
        or coordination_profile.get("legacy_source_artifact_path")
        != coordination_profile.get("artifact_path")
        or not isinstance(coordination_profile.get("artifact_path"), str)
        or not coordination_profile["artifact_path"].startswith(
            "initializations/coordination/"
        )
    ):
        raise ValueError("evaluation-set v2 coordination profile is invalid")
    artifact_paths.add(coordination_profile["artifact_path"])
    if len(artifact_paths) != 9:
        raise ValueError("evaluation-set v2 artifact paths must be unique")

    if payload.get("sealing") != {
        "sealed_manifest_path": "manifest.json",
        "formal_generation_policy": (
            "require_prebuilt_sealed_artifacts_fail_if_missing"
        ),
        "same_A_B_required_across_supported_scenarios": True,
        "same_A_required_across_coordination_scenarios": True,
        "builder_may_read_results": False,
        "result_based_candidate_selection_forbidden": True,
        "result_based_scenario_tuning_forbidden": True,
        "legacy_imports_verified_before_seal": True,
        "regenerated_batches_require_task_scoped_identity": True,
    }:
        raise ValueError("evaluation-set v2 sealing policy is invalid")
    if payload.get("isolation") != {
        "training_data_root": "data/v4",
        "evaluation_artifact_root": "evaluation_sets/rlbench_eval_v2",
        "results_root": "results/v4",
        "training_must_not_read_evaluation_artifacts_or_results": True,
        "results_reference_sealed_manifest_and_artifact_hashes": True,
        "results_must_not_embed_or_modify_evaluation_artifacts": True,
    }:
        raise ValueError("evaluation-set v2 isolation policy is invalid")
    return payload


def validate_fixed_evaluation_split(
    training_path=TRAINING_SPLIT_MANIFEST,
    spec_path=EVALUATION_SET_SPEC,
    verify_training_files=True,
):
    """Authenticate both assets and enforce only the declared seed separation."""

    training = load_training_split_manifest(
        training_path,
        verify_files=verify_training_files,
    )
    spec = load_evaluation_set_spec(spec_path)
    minimum = spec["seed_namespace"]["minimum_seed"]
    maximum = spec["seed_namespace"]["maximum_seed"]
    reserved_ranges = []
    for cohort in training["cohorts"]:
        provenance = cohort["seed_provenance"]
        if provenance["status"] == "unknown_directory_label_unverified":
            reserved = provenance["conservative_reserved_seed_range"]
            reserved_ranges.append(reserved)
            if not (maximum < reserved[0] or minimum > reserved[1]):
                raise ValueError("evaluation seeds overlap conservatively reserved training seeds")
        for episode in cohort["episodes"]:
            seed = episode["seed"]
            if seed is not None and minimum <= seed <= maximum:
                raise ValueError("evaluation seeds overlap an explicit training seed")
    derived_source = []
    derived_goal = []
    profiles = [
        (profile, True) for profile in spec["dynamic_environment"].values()
    ] + [
        (spec["coordination"]["bimanual_handover_item_dynamic"], False)
    ]
    for profile, has_goal_sampling in profiles:
        for episode_index in range(spec["episode_count_per_task"]):
            logical_seed = spec["seed_namespace"]["base_seed"] + episode_index
            variation = _scheduled_variation(profile, episode_index)
            source_seeds, goal_seeds = _derived_rng_seeds(logical_seed, variation)
            derived_source.extend(source_seeds)
            if has_goal_sampling:
                derived_goal.extend(goal_seeds)
    for lower, upper in reserved_ranges:
        if any(lower <= value <= upper for value in derived_source + derived_goal):
            raise ValueError("derived evaluation RNG seed overlaps reserved training seeds")
    evidence = {
        "schema": "dynamac-rlbench-fixed-split-validation-v1",
        "training_split_fingerprint": training["fingerprint"],
        "evaluation_spec_fingerprint": spec["fingerprint"],
        "training_episode_count": 45,
        "evaluation_seed_minimum": minimum,
        "evaluation_seed_maximum": maximum,
        "derived_source_seed_minimum": min(derived_source),
        "derived_source_seed_maximum": max(derived_source),
        "derived_goal_seed_minimum": min(derived_goal),
        "derived_goal_seed_maximum": max(derived_goal),
        "derived_rng_seed_combinations_enumerated": len(derived_source)
        + len(derived_goal),
        "initial_state_zero_overlap_required": False,
        "table_ii_unknown_seed_provenance_blocks_high_seed_evaluation": False,
        "validated": True,
    }
    evidence["fingerprint"] = _canonical_sha256(evidence)
    return evidence


__all__ = [
    "EVALUATION_SET_ROOT",
    "EVALUATION_SET_SPEC",
    "EVALUATION_SET_SPEC_SCHEMA",
    "EVALUATION_SET_V2_CONFIG_SPEC",
    "EVALUATION_SET_V2_ID",
    "EVALUATION_SET_V2_SPEC_SCHEMA",
    "TRAINING_SPLIT_MANIFEST",
    "TRAINING_SPLIT_SCHEMA",
    "load_evaluation_set_spec",
    "load_evaluation_set_v2_spec",
    "load_training_split_manifest",
    "validate_fixed_evaluation_split",
    "validate_training_entry_paths",
]
