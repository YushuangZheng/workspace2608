"""Build a consolidated DynaMAC paper-to-local experiment comparison.

The updater discovers completed evaluation JSON files below ``results/`` and
writes Markdown, CSV, and JSON summaries.  Paper values are fixed constants
transcribed from Tables I--IV of arXiv:2607.22119v1.  Local dynamic runs are
reported as diagnostics because the paper's DynaBench movement and arm-
perturbation defaults are not published.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from essay2608.policy import DynaMACConfig

from .runtime import (
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
    LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    LOW_DIM_STATE_ROUNDTRIP_ATOL,
    PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
    ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD,
    ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M,
    ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
    ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
    DiscreteGripperProtocol,
)
from .unimanual_evaluate import (
    DYNAMIC_EPISODE_ACCOUNTING_SCHEMA,
    EXPECTED_UNIMANUAL_BASE_SCENE_SHA256,
    EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT,
    LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
)

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results"
DEFAULT_RELEASE = "v2"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / DEFAULT_RELEASE
PAPER_REFERENCE = "DynaMAC, arXiv:2607.22119v1"


def _canonical_config(path: Path) -> dict[str, Any]:
    return asdict(
        DynaMACConfig(**json.loads(path.read_text(encoding="utf-8")))
    )


EXPECTED_RELEASE_CONFIGS = {
    "v1": _canonical_config(
        INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v1.json"
    ),
    "v2": _canonical_config(
        INTEGRATION_ROOT / "configs" / "dynamac_rlbench_local.json"
    ),
}
EXPECTED_LOCAL_CONFIG = EXPECTED_RELEASE_CONFIGS[DEFAULT_RELEASE]
EXPECTED_MODEL_SCHEMA_VERSION = 13
EXPECTED_TAPAS_COMMIT = "52e35214b9baa7b190b87196c36b9e98f4006149"
EXPECTED_SELECTION_SEMANTICS_ID = (
    "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
)
LEGACY_V1_EVALUATION_PROTOCOL_ID = (
    "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-noop-clock-v2"
)
V2_EVALUATION_PROTOCOL_BASE_ID = (
    "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-"
    "noop-retry-same-policy-tick-fresh-observation-"
    f"primary-attempt{DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS}-v4"
)
EXPECTED_RELEASE_EVALUATION_PROTOCOL_IDS = {
    "v1": {
        "unimanual": LEGACY_V1_EVALUATION_PROTOCOL_ID,
        "bimanual": LEGACY_V1_EVALUATION_PROTOCOL_ID,
    },
    "v2": {
        "unimanual": DiscreteGripperProtocol(
            bimanual=False
        ).extend_evaluation_protocol_id(V2_EVALUATION_PROTOCOL_BASE_ID),
        "bimanual": DiscreteGripperProtocol(
            bimanual=True
        ).extend_evaluation_protocol_id(V2_EVALUATION_PROTOCOL_BASE_ID),
    },
}
# Backward-compatible name for callers that construct bimanual default-release
# fixtures.  Real selection uses the task-aware mapping above.
EXPECTED_EVALUATION_PROTOCOL_ID = EXPECTED_RELEASE_EVALUATION_PROTOCOL_IDS[
    DEFAULT_RELEASE
]["bimanual"]

UNIMANUAL_TASKS = {
    "stack_wine",
    "place_cups",
    "open_microwave",
    "wipe_desk",
}


@dataclass(frozen=True)
class PaperCell:
    table: str
    condition: str
    task: str
    paper_rate: float
    local_task: str | None = None
    local_scenarios: tuple[str, ...] = ()
    result_family: str | None = None
    note: str = ""
    unavailable_reason: str = ""
    stopped_reason: str = ""


@dataclass(frozen=True)
class LocalRun:
    path: Path
    task: str
    scenario: str
    seed: int
    episodes: int
    horizon: int
    variation: int
    successes: int
    success_rate: float
    payload: dict[str, Any]


PAPER_AVERAGES = {
    "I": 0.94,
    "II": 0.95,
    "III": 0.96,
    "IV": 0.95,
}

TABLE_SCOPES = {
    "I": "Unimanual RLBench",
    "II": "Static bimanual RLBench",
    "III": "Dynamic bimanual RLBench",
    "IV": "Real-world bimanual hardware",
}

def _cells() -> tuple[PaperCell, ...]:
    table_i_tasks = (
        ("stack_wine", "StackWine", 1.00, 1.00, 1.00),
        ("place_cups", "PlaceCups", 0.99, 0.97, 0.99),
        ("open_microwave", "OpenMicrowave", 0.99, 0.99, 0.97),
        ("wipe_desk", "WipeDesk", 1.00, 0.66, 0.69),
    )
    cells: list[PaperCell] = []
    for task, label, static, smooth, teleport in table_i_tasks:
        for condition, scenario, paper_rate in (
            ("Static", "static", static),
            ("Smooth dynamics", "smooth", smooth),
            ("Teleportation", "teleport", teleport),
        ):
            stopped_reason = ""
            cells.append(
                PaperCell(
                    table="I",
                    condition=condition,
                    task=label,
                    paper_rate=paper_rate,
                    local_task=task,
                    local_scenarios=(scenario,),
                    result_family="table_i",
                    note=(
                        "Independent five-demonstration cohort."
                        if scenario == "static"
                        else "Local preserve-instance task-root motion schedule; paper defaults unpublished."
                    ),
                    stopped_reason=stopped_reason,
                )
            )

    table_ii_tasks = (
        ("bimanual_put_bottle_in_fridge", "StoreBottle", 0.82),
        ("bimanual_handover_item", "HandOver", 0.97),
        ("bimanual_sweep_to_dustpan", "SweepDust", 1.00),
        ("bimanual_lift_tray", "LiftTray", 1.00),
    )
    for task, label, paper_rate in table_ii_tasks:
        cells.append(
            PaperCell(
                table="II",
                condition="Static",
                task=label,
                paper_rate=paper_rate,
                local_task=task,
                local_scenarios=("static",),
                result_family="bimanual_static",
                note=(
                    "Independent five-demonstration cohort. "
                    "[SweepDust diagnosis](sweep_dust_diagnosis.md)."
                    if task == "bimanual_sweep_to_dustpan"
                    else "Independent five-demonstration cohort."
                ),
            )
        )

    cells.extend(
        (
            PaperCell(
                table="III",
                condition="Coordination",
                task="Hand Left",
                paper_rate=0.97,
                local_task="bimanual_handover_item_dynamic",
                local_scenarios=(
                    "coordination_hand_left",
                    "coordination_left",
                    "hand_left",
                    "perturb_left",
                    "left_arm_perturbed",
                ),
                result_family="bimanual_coordination",
                note="Arm-perturbation magnitude and timing are unpublished.",
            ),
            PaperCell(
                table="III",
                condition="Coordination",
                task="Hand Right",
                paper_rate=0.97,
                local_task="bimanual_handover_item_dynamic",
                local_scenarios=(
                    "coordination_hand_right",
                    "coordination_right",
                    "hand_right",
                    "perturb_right",
                    "right_arm_perturbed",
                ),
                result_family="bimanual_coordination",
                note="Arm-perturbation magnitude and timing are unpublished.",
            ),
        )
    )
    for task, label, paper_rate in table_ii_tasks:
        cells.append(
            PaperCell(
                table="III",
                condition="Dynamic environment",
                task=label,
                paper_rate=paper_rate,
                local_task=task,
                # The paper does not identify its Table III motion type. Keep
                # the local paper-cell selector fixed to teleport.
                local_scenarios=("teleport",),
                result_family="bimanual_dynamic",
                note="Local public-RLBench intervention; paper DynaBench defaults unpublished.",
            )
        )

    for condition, task, paper_rate in (
        ("Static", "StoreItem", 0.96),
        ("Static", "HandOver", 0.96),
        ("Static", "SweepBlocks", 0.96),
        ("Static", "LiftJointly", 1.00),
        ("Dynamic environment", "StoreItem", 0.92),
        ("Dynamic environment", "HandOver", 0.88),
        ("Coordination", "Hand Left", 0.92),
        ("Coordination", "Hand Right", 0.96),
    ):
        cells.append(
            PaperCell(
                table="IV",
                condition=condition,
                task=task,
                paper_rate=paper_rate,
                note="Physical dual-Franka setup unavailable locally.",
            )
        )
    return tuple(cells)


PAPER_CELLS = _cells()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_run(path: Path) -> tuple[LocalRun | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path}: unreadable JSON ({exc})"
    if not isinstance(payload, dict):
        return None, None
    required = ("task", "scenario", "seed", "episodes", "horizon", "successes")
    if not all(key in payload for key in required) or "success_rate" not in payload:
        return None, None
    if not isinstance(payload["task"], str) or not isinstance(payload["scenario"], str):
        return None, f"{path}: task/scenario must be strings"
    integer_fields = ("seed", "episodes", "horizon", "successes")
    if not all(isinstance(payload[key], int) and not isinstance(payload[key], bool) for key in integer_fields):
        return None, f"{path}: seed/episodes/horizon/successes must be integers"
    if not _is_number(payload["success_rate"]):
        return None, f"{path}: success_rate must be numeric"
    episodes = int(payload["episodes"])
    variation = payload.get("variation", 0)
    if not isinstance(variation, int) or isinstance(variation, bool) or variation < 0:
        return None, f"{path}: variation must be a non-negative integer"
    successes = int(payload["successes"])
    success_rate = float(payload["success_rate"])
    if episodes < 1 or not 0 <= successes <= episodes:
        return None, f"{path}: invalid episode totals"
    if abs(success_rate - successes / episodes) > 1e-9:
        return None, f"{path}: success_rate disagrees with successes/episodes"
    episode_rows = payload.get("results")
    if isinstance(episode_rows, list):
        if len(episode_rows) != episodes:
            return None, f"{path}: result row count disagrees with episodes"
        counted = sum(bool(row.get("success")) for row in episode_rows if isinstance(row, dict))
        if counted != successes:
            return None, f"{path}: result rows disagree with successes"
    return (
        LocalRun(
            path=path,
            task=payload["task"],
            scenario=payload["scenario"],
            seed=int(payload["seed"]),
            episodes=episodes,
            horizon=int(payload["horizon"]),
            variation=variation,
            successes=successes,
            success_rate=success_rate,
            payload=payload,
        ),
        None,
    )


def discover_runs(results_dir: Path) -> tuple[list[LocalRun], list[str]]:
    runs: list[LocalRun] = []
    warnings: list[str] = []
    generated = {"paper_comparison.json"}
    for path in sorted(results_dir.rglob("*.json")):
        if path.name in generated:
            continue
        # One-episode coordination smoke files predate the evaluation-result
        # schema (their top-level ``episodes`` field is a row list).  They are
        # launcher diagnostics, never candidates for a paper cell, so exclude
        # them without presenting a misleading malformed-result warning.
        if path.name.startswith("smoke_") and "_n1" in path.stem:
            continue
        run, warning = _load_run(path)
        if run is not None:
            runs.append(run)
        if warning is not None:
            warnings.append(warning)
    return runs, warnings


def _family_matches(cell: PaperCell, run: LocalRun, results_dir: Path) -> bool:
    try:
        relative_parts = run.path.relative_to(results_dir).parts
    except ValueError:
        relative_parts = run.path.parts
    schema = str(run.payload.get("schema", ""))
    in_table_i = "table_i" in relative_parts or schema.startswith("dynamac-table-i-")
    if cell.result_family == "table_i":
        return in_table_i
    if cell.result_family in {
        "bimanual_static",
        "bimanual_dynamic",
        "bimanual_coordination",
    }:
        return not in_table_i
    return True


def _expected_training_config(release: str | None) -> dict[str, Any] | None:
    if release is None:
        return EXPECTED_LOCAL_CONFIG
    return EXPECTED_RELEASE_CONFIGS.get(release)


def _expected_evaluation_protocol_ids(
    release: str | None,
) -> dict[str, str] | None:
    selected_release = DEFAULT_RELEASE if release is None else release
    return EXPECTED_RELEASE_EVALUATION_PROTOCOL_IDS.get(selected_release)


def expected_evaluation_protocol_id(
    task: str,
    *,
    release: str | None = DEFAULT_RELEASE,
) -> str | None:
    """Return the authenticated evaluator ID for a task/release pair."""

    protocols = _expected_evaluation_protocol_ids(release)
    if protocols is None:
        return None
    layout = "unimanual" if task in UNIMANUAL_TASKS else "bimanual"
    return protocols[layout]


def _model_identity_rank(
    run: LocalRun,
    expected_training_config: dict[str, Any] | None = EXPECTED_LOCAL_CONFIG,
    expected_evaluation_protocol_ids: dict[str, str] | None = (
        EXPECTED_RELEASE_EVALUATION_PROTOCOL_IDS[DEFAULT_RELEASE]
    ),
) -> int:
    """Rank exact corrected runs before mismatched and legacy results."""

    identity = run.payload.get("model_identity")
    if not isinstance(identity, dict):
        return 2
    fingerprint_present = bool(identity.get("fingerprint")) or bool(
        identity.get("left_fingerprint") and identity.get("right_fingerprint")
    )
    layout = "unimanual" if run.task in UNIMANUAL_TASKS else "bimanual"
    expected_protocol_id = (
        expected_evaluation_protocol_ids.get(layout)
        if expected_evaluation_protocol_ids is not None
        else None
    )
    if (
        identity.get("manifest_authenticated") is True
        and expected_training_config is not None
        and expected_protocol_id is not None
        and identity.get("training_config") == expected_training_config
        and identity.get("model_schema_version") == EXPECTED_MODEL_SCHEMA_VERSION
        and identity.get("selection_semantics_id")
        == EXPECTED_SELECTION_SEMANTICS_ID
        and identity.get("tapas_reference_commit") == EXPECTED_TAPAS_COMMIT
        and fingerprint_present
        and run.payload.get("evaluation_protocol_id") == expected_protocol_id
    ):
        return 0
    return 1


def _model_fingerprint_key(run: LocalRun) -> tuple[str, ...]:
    identity = run.payload.get("model_identity")
    if not isinstance(identity, dict):
        return ()
    if identity.get("fingerprint"):
        return (str(identity["fingerprint"]),)
    if identity.get("left_fingerprint") and identity.get("right_fingerprint"):
        return (
            str(identity["left_fingerprint"]),
            str(identity["right_fingerprint"]),
        )
    return ()


def _select_run(
    cell: PaperCell,
    runs: Iterable[LocalRun],
    results_dir: Path,
    *,
    seed: int,
    episodes: int,
    horizon: int,
    expected_training_config: dict[str, Any] | None = EXPECTED_LOCAL_CONFIG,
    expected_evaluation_protocol_ids: dict[str, str] | None = (
        EXPECTED_RELEASE_EVALUATION_PROTOCOL_IDS[DEFAULT_RELEASE]
    ),
) -> LocalRun | None:
    if cell.local_task is None:
        return None
    candidates = [
        run
        for run in runs
        if run.task == cell.local_task
        and run.scenario in cell.local_scenarios
        and run.seed == seed
        and run.episodes == episodes
        and run.horizon == horizon
        and run.variation == 0
        and _family_matches(cell, run, results_dir)
    ]
    if not candidates:
        return None
    scenario_rank = {name: index for index, name in enumerate(cell.local_scenarios)}
    candidates.sort(
        key=lambda run: (
            scenario_rank[run.scenario],
            _model_identity_rank(
                run,
                expected_training_config,
                expected_evaluation_protocol_ids,
            ),
            str(run.path),
        )
    )
    best_key = (
        scenario_rank[candidates[0].scenario],
        _model_identity_rank(
            candidates[0],
            expected_training_config,
            expected_evaluation_protocol_ids,
        ),
    )
    equally_ranked = [
        run
        for run in candidates
        if (
            scenario_rank[run.scenario],
            _model_identity_rank(
                run,
                expected_training_config,
                expected_evaluation_protocol_ids,
            ),
        )
        == best_key
    ]
    if best_key[1] == 0 and len(equally_ranked) > 1:
        identities = {_model_fingerprint_key(run) for run in equally_ranked}
        paths = ", ".join(str(run.path) for run in equally_ranked)
        raise RuntimeError(
            "multiple corrected results match one paper cell; select an exact "
            f"checkpoint identity explicitly ({len(identities)} identities): {paths}"
        )
    return candidates[0]


def _protocol_metadata(run: LocalRun) -> dict[str, Any]:
    for key in ("scenario_protocol", "protocol", "coordination_protocol"):
        value = run.payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _valid_v2_unimanual_scene_launch(run: LocalRun) -> bool:
    controller = run.payload.get("controller")
    scene_launch = (
        controller.get("scene_launch") if isinstance(controller, dict) else None
    )
    if not isinstance(scene_launch, dict):
        return False
    handling = scene_launch.get("vision_sensor_handling")
    derived_sha256 = scene_launch.get("derived_scene_sha256")
    if not isinstance(handling, list) or not isinstance(derived_sha256, str):
        return False
    try:
        int(derived_sha256, 16)
    except ValueError:
        return False
    return (
        len(derived_sha256) == 64
        and scene_launch.get("protocol_id")
        == LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID
        and scene_launch.get("applied") is True
        and scene_launch.get("source_scene_sha256")
        == EXPECTED_UNIMANUAL_BASE_SCENE_SHA256
        and scene_launch.get("vision_sensor_count")
        == EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT
        and len(handling) == EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT
        and all(
            isinstance(item, dict) and item.get("after") == 1
            for item in handling
        )
        and scene_launch.get("populated_scene_steps_before_patch") == 0
        and scene_launch.get("camera_observations_requested") is False
        and scene_launch.get("task_model_loaded_during_rewrite") is False
        and scene_launch.get("physics_modified") is False
        and scene_launch.get("task_modified") is False
        and scene_launch.get("policy_input_modified") is False
        and scene_launch.get("qt_qpa_platform") == "offscreen"
    )


def _applied_events(run: LocalRun) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    rows = run.payload.get("results")
    if not isinstance(rows, list):
        return events
    for row in rows:
        if not isinstance(row, dict):
            continue
        events.extend(_row_applied_events(row))
    return events


def _row_applied_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the shared controller events from either evaluator schema."""

    events: list[dict[str, Any]] = []
    for key in ("scenario_events", "interventions"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        events.extend(
            event
            for event in values
            if isinstance(event, dict) and event.get("applied") is True
        )
    return events


def _expected_v4_root_motion_protocol() -> dict[str, Any]:
    """Return the complete authenticated preserve-instance v4 contract."""

    return {
        "protocol_id": PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
        "episode_instance_semantics": "preserve_initialized_episode",
        "goal_object": "task.boundary_root()",
        "goal_sampling": "scene_workspace_boundary_without_task_reinitialization",
        "sampling_rollback": "task_configuration_tree_only_live_robot_untouched",
        "sampling_rollback_frequency": "after_each_attempt_and_outer_finally",
        "task_configuration_tree_restore_api": "Task.get_state/restore_state",
        "task_tree_object_count_guard": True,
        "live_robot_state_during_goal_sampling": "untouched",
        "live_robot_configuration_tree_access": "none",
        "online_task_waypoint_validation": "disabled_to_preserve_live_robot_state",
        "calls_task_validate": False,
        "grasp_membership_and_parentage_audited": True,
        "robot_collision_validation": (
            "reject_candidate_external_pairs_absent_at_source"
        ),
        "robot_collision_pair_granularity": (
            "named_arm_collection_x_external_collidable_scene_shape"
        ),
        "source_robot_contacts_allowed": True,
        "grasped_tool_collision_semantics": (
            "current_arm_collection_membership_without_task_filters"
        ),
        "self_collision_semantics": (
            "current_arm_collection_members_excluded_matching_all_other"
        ),
        "low_dim_state_roundtrip_comparison": (
            "valid_pose_chunks_sign_invariant_else_scalar_max_abs"
        ),
        "low_dim_state_roundtrip_scalar_tolerance": (
            LOW_DIM_STATE_ROUNDTRIP_ATOL
        ),
        "low_dim_state_roundtrip_pose_translation_tolerance_m": (
            LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
        ),
        "low_dim_state_roundtrip_pose_rotation_tolerance_rad": (
            LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
        ),
        "root_application_validation": (
            "planned_motion_and_actual_motion_and_commanded_pose_reached"
        ),
        "root_actual_motion_translation_tolerance_m": (
            ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
        ),
        "root_actual_motion_rotation_tolerance_rad": (
            ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
        ),
        "root_command_translation_tolerance_m": (
            ROOT_COMMAND_TRANSLATION_TOLERANCE_M
        ),
        "root_command_rotation_tolerance_rad": (
            ROOT_COMMAND_ROTATION_TOLERANCE_RAD
        ),
        "dynamic_state_note": (
            "the task configuration tree restores task poses and joints and "
            "resets task dynamics; live robot trees remain untouched; the "
            "subsequent root-motion intervention resets moved task dynamics"
        ),
        "goal_validation": "workspace_fit_no_new_robot_external_collision_pairs",
        "calls_task_init_episode": False,
        "calls_scene_kidnap": False,
        "calls_scene_move_task_smoothly": False,
        "smooth_schedule": "fractions_1_over_n_through_n_over_n",
        "smooth_endpoint_validation": "final_goal_pose_reached",
        "smooth_endpoint_guaranteed": True,
    }


def _finite_nonnegative_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) >= 0.0


def _collision_pair_keys(
    value: object,
    *,
    expected_arms: frozenset[str],
) -> tuple[tuple[str, int, str], ...] | None:
    """Validate and normalize JSON collision-pair evidence."""

    if not isinstance(value, list):
        return None
    keys: list[tuple[str, int, str]] = []
    expected_fields = {
        "arm",
        "external_object_handle",
        "external_object_name",
    }
    for row in value:
        if not isinstance(row, dict) or set(row) != expected_fields:
            return None
        arm = row.get("arm")
        handle = row.get("external_object_handle")
        name = row.get("external_object_name")
        if (
            arm not in expected_arms
            or not isinstance(handle, int)
            or isinstance(handle, bool)
            or handle < 0
            or not isinstance(name, str)
            or not name
        ):
            return None
        keys.append((arm, handle, name))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        return None
    return tuple(keys)


def _valid_v4_preservation_evidence(
    preservation: object,
    *,
    expected_arms: frozenset[str],
    max_sampling_attempts: int,
) -> bool:
    if not isinstance(preservation, dict):
        return False
    if (
        preservation.get("initialized_episode_preserved") is not True
        or preservation.get("task_init_episode_called") is not False
        or preservation.get("task_validate_called") is not False
        or preservation.get("low_dim_state_roundtrip_preserved") is not True
        or preservation.get("condition_and_grasp_registry_identity_preserved")
        is not True
        or preservation.get("gripper_grasp_membership_and_parentage_preserved")
        is not True
        or preservation.get("task_configuration_tree_restored") is not True
        or preservation.get("live_robot_state_untouched") is not True
        or preservation.get("live_robot_configuration_trees_accessed") is not False
        or preservation.get("waypoint_cache_identity_preserved") is not True
        or preservation.get("configuration_tree_rollback")
        != "task_only_after_each_attempt_and_outer_finally"
        or preservation.get("robot_collision_pair_policy")
        != "reject_candidate_external_pairs_absent_at_source"
        or preservation.get("robot_collision_pair_granularity")
        != "named_arm_collection_x_external_collidable_scene_shape"
    ):
        return False

    roundtrip_l2 = preservation.get("low_dim_state_roundtrip_l2")
    roundtrip_max_abs = preservation.get("low_dim_state_roundtrip_max_abs")
    comparison_mode = preservation.get(
        "low_dim_state_roundtrip_comparison_mode"
    )
    chunk_count = preservation.get("low_dim_state_roundtrip_chunk_count")
    max_translation = preservation.get(
        "low_dim_state_roundtrip_max_translation_m"
    )
    max_rotation = preservation.get("low_dim_state_roundtrip_max_rotation_rad")
    attempts = preservation.get("sampling_attempts")
    rejected = preservation.get(
        "sampling_attempts_rejected_for_new_robot_collision_pairs"
    )
    if (
        not _finite_nonnegative_number(roundtrip_l2)
        or not _finite_nonnegative_number(roundtrip_max_abs)
        or not isinstance(chunk_count, int)
        or isinstance(chunk_count, bool)
        or chunk_count < 0
        or not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or not 1 <= attempts <= max_sampling_attempts
        or not isinstance(rejected, int)
        or isinstance(rejected, bool)
        or not 0 <= rejected < attempts
    ):
        return False

    if comparison_mode == "pose_chunks_sign_invariant":
        if (
            chunk_count < 1
            or not _finite_nonnegative_number(max_translation)
            or not _finite_nonnegative_number(max_rotation)
            or float(max_translation) > LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
            or float(max_rotation) > LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
        ):
            return False
    elif comparison_mode == "scalar_max_abs":
        if (
            chunk_count != 0
            or max_translation is not None
            or max_rotation is not None
            or float(roundtrip_max_abs) > LOW_DIM_STATE_ROUNDTRIP_ATOL
        ):
            return False
    else:
        return False

    source_pairs = _collision_pair_keys(
        preservation.get("source_robot_external_collision_pairs"),
        expected_arms=expected_arms,
    )
    goal_pairs = _collision_pair_keys(
        preservation.get("goal_robot_external_collision_pairs"),
        expected_arms=expected_arms,
    )
    new_pairs = _collision_pair_keys(
        preservation.get("goal_new_robot_external_collision_pairs"),
        expected_arms=expected_arms,
    )
    if source_pairs is None or goal_pairs is None or new_pairs is None:
        return False
    return (
        new_pairs == ()
        and frozenset(goal_pairs).issubset(frozenset(source_pairs))
        and frozenset(goal_pairs) - frozenset(source_pairs) == frozenset(new_pairs)
    )


def _valid_v4_root_application(
    event: dict[str, Any],
    *,
    required_goal_reached: bool | None,
) -> bool:
    numeric_fields = (
        "planned_root_translation_m",
        "planned_root_rotation_rad",
        "actual_root_translation_m",
        "actual_root_rotation_rad",
        "commanded_root_translation_residual_m",
        "commanded_root_rotation_residual_rad",
        "goal_root_translation_residual_m",
        "goal_root_rotation_residual_rad",
    )
    if any(
        not _finite_nonnegative_number(event.get(field))
        for field in numeric_fields
    ):
        return False

    planned_translation = float(event["planned_root_translation_m"])
    planned_rotation = float(event["planned_root_rotation_rad"])
    actual_translation = float(event["actual_root_translation_m"])
    actual_rotation = float(event["actual_root_rotation_rad"])
    commanded_translation = float(event["commanded_root_translation_residual_m"])
    commanded_rotation = float(event["commanded_root_rotation_residual_rad"])
    goal_translation = float(event["goal_root_translation_residual_m"])
    goal_rotation = float(event["goal_root_rotation_residual_rad"])
    planned_motion = (
        planned_translation > ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
        or planned_rotation > ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
    )
    actual_motion = (
        actual_translation > ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
        or actual_rotation > ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
    )
    command_reached = (
        commanded_translation <= ROOT_COMMAND_TRANSLATION_TOLERANCE_M
        and commanded_rotation <= ROOT_COMMAND_ROTATION_TOLERANCE_RAD
    )
    measured_goal_reached = (
        goal_translation <= ROOT_COMMAND_TRANSLATION_TOLERANCE_M
        and goal_rotation <= ROOT_COMMAND_ROTATION_TOLERANCE_RAD
    )
    return (
        event.get("planned_root_motion") is True
        and planned_motion
        and event.get("actual_root_motion") is True
        and actual_motion
        and event.get("commanded_root_pose_reached") is True
        and command_reached
        and event.get("goal_root_pose_reached") is measured_goal_reached
        and (
            required_goal_reached is None
            or measured_goal_reached is required_goal_reached
        )
        and event.get("protocol_effective") is True
    )


def _valid_v2_root_motion_protocol(run: LocalRun) -> bool:
    """Authenticate every preserve-instance task-root intervention.

    Rows that terminate with failure before the scheduled trigger are retained
    but are explicitly ineligible for intervention.  Eligible teleport rows
    require one event; eligible smooth rows require a strict effective prefix,
    with the exact endpoint whenever the episode survives the full window.
    Summary counts are recomputed from those row-level claims so legacy fields
    or forged pre-trigger exemptions cannot authenticate a run.
    """

    metadata = _protocol_metadata(run)
    expected_motion = _expected_v4_root_motion_protocol()
    if metadata.get("motion_protocol") != expected_motion:
        return False
    if metadata.get("dynamic_episode_accounting_schema") != (
        DYNAMIC_EPISODE_ACCOUNTING_SCHEMA
    ):
        return False
    if (
        metadata.get("pre_intervention_failure_policy")
        != "retain_failure_with_null_intervention_effectiveness"
        or metadata.get("pre_intervention_success_policy")
        != "fail_closed_unexercised_dynamic_condition"
        or metadata.get("smooth_terminal_progress_policy")
        != "strict_effective_prefix_until_episode_terminal"
    ):
        return False

    rows = run.payload.get("results")
    if not isinstance(rows, list) or len(rows) != run.episodes:
        return False
    bimanual = run.task.startswith("bimanual_")
    expected_arms = (
        frozenset({"right_arm", "left_arm"})
        if bimanual
        else frozenset({"arm"})
    )
    max_attempts_key = (
        "max_sampling_attempts" if bimanual else "intervention_max_attempts"
    )
    max_sampling_attempts = metadata.get(max_attempts_key)
    if (
        not isinstance(max_sampling_attempts, int)
        or isinstance(max_sampling_attempts, bool)
        or max_sampling_attempts < 1
    ):
        return False
    expected_smooth_calls = 10
    if run.scenario == "smooth":
        smooth_calls_key = (
            "smooth_interpolation_calls" if bimanual else "smooth_motion_calls"
        )
        if metadata.get(smooth_calls_key) != expected_smooth_calls:
            return False

    trigger_step = metadata.get("trigger_control_step")
    if (
        not isinstance(trigger_step, int)
        or isinstance(trigger_step, bool)
        or trigger_step < 0
    ):
        return False

    event_key = "scenario_events" if bimanual else "interventions"
    other_event_key = "interventions" if bimanual else "scenario_events"
    eligible_count = 0
    preterminal_count = 0
    intervention_count = 0
    effective_count = 0

    for row in rows:
        if not isinstance(row, dict):
            return False
        raw_events = row.get(event_key)
        if (
            not isinstance(raw_events, list)
            or other_event_key in row
            or any(not isinstance(event, dict) for event in raw_events)
        ):
            return False
        events = [
            event for event in raw_events if event.get("applied") is True
        ]
        if len(events) != len(raw_events):
            return False

        eligible = row.get("intervention_eligible")
        reached = row.get("intervention_reached")
        preterminal = row.get("pre_intervention_terminal")
        effective = row.get("intervention_effective")
        complete = row.get("intervention_complete")
        steps = row.get("steps")
        success = row.get("success")
        reason = row.get("reason")
        terminal_reasons = {
            "success",
            "terminate",
            "policy_complete",
            "primary_action_retry_exhausted",
            "noop_failed",
            "horizon",
        }
        if (
            row.get("trigger_step") != trigger_step
            or not isinstance(eligible, bool)
            or not isinstance(reached, bool)
            or not isinstance(preterminal, bool)
            or eligible is not reached
            or preterminal is reached
            or not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps < 0
            or not isinstance(success, bool)
            or reason not in terminal_reasons
            or success is not (reason == "success")
        ):
            return False

        if preterminal:
            preterminal_count += 1
            if (
                events
                or effective is not None
                or complete is not None
                or success is not False
                or steps > trigger_step
            ):
                return False
            continue

        eligible_count += 1
        if effective is not True or not events:
            return False
        intervention_count += 1
        effective_count += 1
        if run.scenario == "teleport":
            if len(events) != 1 or complete is not True:
                return False
        elif run.scenario == "smooth":
            if not 1 <= len(events) <= expected_smooth_calls:
                return False
            expected_complete = len(events) == expected_smooth_calls
            if complete is not expected_complete:
                return False
            if not expected_complete:
                last_event_step = trigger_step + len(events) - 1
                if steps not in {last_event_step, last_event_step + 1}:
                    return False
        else:
            return False
        for event in events:
            event_motion = event.get("motion_protocol")
            preservation = event.get("instance_preservation")
            if (
                not isinstance(event.get("step"), int)
                or isinstance(event.get("step"), bool)
                or not isinstance(event.get("trigger_step"), int)
                or isinstance(event.get("trigger_step"), bool)
                or event_motion != expected_motion
                or event.get("policy_observation_refreshed") is not True
                or not _valid_v4_preservation_evidence(
                    preservation,
                    expected_arms=expected_arms,
                    max_sampling_attempts=max_sampling_attempts,
                )
            ):
                return False
        if steps < events[-1]["step"]:
            return False
        if any(
            event.get("instance_preservation")
            != events[0].get("instance_preservation")
            for event in events[1:]
        ):
            return False
        if run.scenario == "teleport":
            event = events[0]
            if (
                event.get("kind") != "teleport_task"
                or event.get("step") != trigger_step
                or event.get("trigger_step") != trigger_step
                or not _valid_v4_root_application(
                    event,
                    required_goal_reached=True,
                )
            ):
                return False
        if run.scenario == "smooth":
            for index, event in enumerate(events, start=1):
                fraction = index / expected_smooth_calls
                expected_endpoint = index == expected_smooth_calls
                complete = event.get("complete")
                endpoint_applied = event.get("endpoint_applied")
                if (
                    event.get("kind") != "smooth_task_motion"
                    or event.get("step") != trigger_step + index - 1
                    or event.get("trigger_step") != trigger_step
                    or not isinstance(event.get("smooth_call"), int)
                    or isinstance(event.get("smooth_call"), bool)
                    or event.get("smooth_call") != index
                    or not _is_number(event.get("endpoint_fraction"))
                    or not math.isfinite(float(event["endpoint_fraction"]))
                    or abs(float(event["endpoint_fraction"]) - fraction) > 1.0e-12
                    or not isinstance(complete, bool)
                    or complete is not expected_endpoint
                    or not isinstance(endpoint_applied, bool)
                    or endpoint_applied is not expected_endpoint
                    or not _valid_v4_root_application(
                        event,
                        required_goal_reached=(True if expected_endpoint else None),
                    )
                ):
                    return False

    all_episodes_intervened = intervention_count == run.episodes
    all_interventions_effective = effective_count == intervention_count
    all_eligible_effective = effective_count == eligible_count
    all_rows_covered = eligible_count + preterminal_count == run.episodes
    return (
        all_rows_covered
        and metadata.get("protocol_valid") is True
        and metadata.get("episodes_intervention_eligible") == eligible_count
        and metadata.get("episodes_pre_intervention_terminal")
        == preterminal_count
        and metadata.get("episodes_with_intervention") == intervention_count
        and metadata.get("episodes_with_effective_intervention")
        == effective_count
        and metadata.get("all_episodes_intervened")
        is all_episodes_intervened
        and metadata.get("all_interventions_effective")
        is all_interventions_effective
        and metadata.get("all_eligible_interventions_effective")
        is all_eligible_effective
    )


def _numeric_event_values(events: Sequence[dict[str, Any]], key: str) -> list[float]:
    return sorted(
        float(event[key])
        for event in events
        if _is_number(event.get(key))
    )


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def _material_protocol_note(cell: PaperCell, run: LocalRun | None) -> str:
    if run is None or cell.table != "III" or cell.condition != "Dynamic environment":
        return ""
    events = _applied_events(run)
    state_values = _numeric_event_values(events, "task_state_l2")
    root_values = _numeric_event_values(events, "root_pose_l2")
    state_median = _median(state_values)
    root_median = _median(root_values)
    state_text = "unknown" if state_median is None else f"{state_median:.3f}"
    root_text = "unknown" if root_median is None else f"{root_median:.3f}"
    metadata = _protocol_metadata(run)
    effective = metadata.get("episodes_with_effective_intervention")
    effective_text = "unknown" if not isinstance(effective, int) else str(effective)
    eligible = metadata.get("episodes_intervention_eligible")
    eligible_text = "unknown" if not isinstance(eligible, int) else str(eligible)
    preterminal = metadata.get("episodes_pre_intervention_terminal")
    preterminal_text = (
        "unknown" if not isinstance(preterminal, int) else str(preterminal)
    )
    motion = metadata.get("motion_protocol")
    motion_id = motion.get("protocol_id") if isinstance(motion, dict) else "unknown"
    return (
        f"Task-root intervention (median root-pose L2 {root_text}; median task-state "
        f"L2 {state_text}; {effective_text}/{eligible_text} eligible episodes "
        f"effective; {preterminal_text} failures ended before the trigger; "
        f"protocol {motion_id})."
    )


def _status(
    cell: PaperCell,
    run: LocalRun | None,
    expected_training_config: dict[str, Any] | None = EXPECTED_LOCAL_CONFIG,
    expected_evaluation_protocol_ids: dict[str, str] | None = (
        EXPECTED_RELEASE_EVALUATION_PROTOCOL_IDS[DEFAULT_RELEASE]
    ),
) -> str:
    if cell.unavailable_reason:
        return "unavailable"
    if cell.table == "IV":
        return "hardware unavailable"
    if run is None:
        if cell.stopped_reason:
            return "stopped after 0% preliminary run"
        return "pending"
    identity_rank = _model_identity_rank(
        run,
        expected_training_config,
        expected_evaluation_protocol_ids,
    )
    if identity_rank == 2:
        return "historical pre-fix result"
    if identity_rank == 1:
        return "configuration-mismatch diagnostic"
    metadata = _protocol_metadata(run)
    uses_v2_config = expected_training_config == EXPECTED_RELEASE_CONFIGS["v2"]
    root_motion_cell = (
        (cell.table == "I" and cell.condition != "Static")
        or (cell.table == "III" and cell.condition == "Dynamic environment")
    )
    if uses_v2_config and cell.table == "I":
        if not _valid_v2_unimanual_scene_launch(run):
            return "invalid diagnostic"
    if uses_v2_config and root_motion_cell:
        if not _valid_v2_root_motion_protocol(run):
            return "invalid diagnostic"
    if metadata.get("protocol_valid") is False:
        return "invalid diagnostic"
    # Static Table I has the same interpretation boundary as static Table II:
    # it is a reproduction of the paper cell using an independent local
    # five-demonstration cohort, not a byte-identical rerun of author data.
    # The Table I evaluator conservatively marks the whole local protocol
    # family paper_comparable=false because its dynamic schedules are local;
    # do not let that family-level flag demote the static condition.
    if cell.table == "I" and cell.condition == "Static":
        return "local reproduction"
    if metadata.get("paper_comparable") is False:
        return "non-comparable diagnostic"
    if cell.table == "I" and cell.condition != "Static":
        return "non-comparable diagnostic"
    if cell.table == "III":
        return "non-comparable diagnostic"
    return "local reproduction"


def build_records(
    results_dir: Path,
    *,
    seed: int,
    episodes: int,
    horizon: int,
    release: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if release is not None:
        if not release or Path(release).name != release or release in {".", ".."}:
            raise ValueError("release must be one directory name")
        discovery_root = results_dir / release
    else:
        discovery_root = results_dir
    runs, warnings = discover_runs(discovery_root)
    expected_training_config = _expected_training_config(release)
    expected_protocol_ids = _expected_evaluation_protocol_ids(release)
    records: list[dict[str, Any]] = []
    for cell in PAPER_CELLS:
        run = _select_run(
            cell,
            runs,
            results_dir,
            seed=seed,
            episodes=episodes,
            horizon=horizon,
            expected_training_config=expected_training_config,
            expected_evaluation_protocol_ids=expected_protocol_ids,
        )
        source = None
        if run is not None:
            try:
                source = run.path.relative_to(results_dir).as_posix()
            except ValueError:
                source = str(run.path)
        records.append(
            {
                "table": cell.table,
                "condition": cell.condition,
                "task": cell.task,
                "paper_success_rate": cell.paper_rate,
                "local_success_rate": run.success_rate if run is not None else None,
                "difference": run.success_rate - cell.paper_rate if run is not None else None,
                "local_successes": run.successes if run is not None else None,
                "local_episodes": run.episodes if run is not None else None,
                "status": _status(
                    cell,
                    run,
                    expected_training_config,
                    expected_protocol_ids,
                ),
                "source_file": source,
                "notes": (
                    cell.unavailable_reason
                    or (cell.stopped_reason if run is None else "")
                    or cell.note
                ),
                "protocol_note": _material_protocol_note(cell, run),
            }
        )
    return records, warnings


def _table_summary(table: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in records if row["table"] == table]
    accepted_statuses = (
        {"local reproduction"}
        if table in {"I", "II"}
        else {"non-comparable diagnostic"}
        if table == "III"
        else set()
    )
    available = [
        row
        for row in rows
        if row["local_success_rate"] is not None
        and row["status"] in accepted_statuses
    ]
    statuses = sorted({str(row["status"]) for row in rows})
    return {
        "table": table,
        "scope": TABLE_SCOPES[table],
        "paper_average": PAPER_AVERAGES[table],
        "local_available_mean": (
            sum(float(row["local_success_rate"]) for row in available) / len(available)
            if available
            else None
        ),
        "local_cells_available": len(available),
        "total_cells": len(rows),
        "local_statuses": statuses,
    }


def build_document(
    records: Sequence[dict[str, Any]],
    warnings: Sequence[str],
    *,
    seed: int,
    episodes: int,
    horizon: int,
    release: str | None = None,
) -> dict[str, Any]:
    expected_training_config = _expected_training_config(release)
    expected_protocol_ids = _expected_evaluation_protocol_ids(release)
    return {
        "schema": "dynamac-paper-comparison-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_reference": PAPER_REFERENCE,
        "selection": {
            "release": release,
            "seed": seed,
            "episodes": episodes,
            "horizon": horizon,
            "expected_model_identity": {
                "model_schema_version": EXPECTED_MODEL_SCHEMA_VERSION,
                "selection_semantics_id": EXPECTED_SELECTION_SEMANTICS_ID,
                "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
                "training_config": expected_training_config,
                "evaluation_protocol_ids": expected_protocol_ids,
            },
        },
        "status_definitions": {
            "local reproduction": "Same paper cell, independent local demonstrations/configuration.",
            "non-comparable diagnostic": "A local run whose intervention protocol is not the unpublished paper protocol.",
            "invalid diagnostic": "The requested local intervention did not produce a valid/effective protocol.",
            "pending": "No matching completed local result was found.",
            "stopped after 0% preliminary run": (
                "No 200-episode result is claimed; a deterministic local "
                "implementation issue was confirmed from a stopped preliminary run."
            ),
            "hardware unavailable": "Requires the paper's physical robot, perception, and demonstration setup.",
            "unavailable": "The public simulator path cannot execute this cell reliably.",
        },
        "table_summaries": [_table_summary(table, records) for table in ("I", "II", "III", "IV")],
        "records": list(records),
        "discovery_warnings": list(warnings),
    }


def _rate(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _difference(value: object) -> str:
    return "—" if value is None else f"{float(value):+.3f}"


def markdown(document: dict[str, Any]) -> str:
    selection = document["selection"]
    lines = [
        "# DynaMAC Paper Comparison",
        "",
        "Published values are transcribed from Tables I–IV of "
        "[DynaMAC (arXiv:2607.22119v1)](https://arxiv.org/abs/2607.22119). "
        "Local results use the selected complete files only: "
        + (
            f"release `{selection['release']}`, "
            if selection.get("release") is not None
            else ""
        )
        + f"seed `{selection['seed']}`, `{selection['episodes']}` episodes, horizon "
        f"`{selection['horizon']}`.",
        "",
        "Dynamic simulator values are retained as diagnostics, not claimed as paper "
        "reproductions: the task-motion and arm-perturbation defaults used for the "
        "paper are not published. Table IV requires physical hardware and is not run "
        "locally.",
        "",
        "## Overview",
        "",
        "| Paper table | Scope | Paper DynaMAC average | Available local mean | Coverage | Interpretation |",
        "|---|---|---:|---:|---:|---|",
    ]
    for summary in document["table_summaries"]:
        coverage = f"{summary['local_cells_available']}/{summary['total_cells']}"
        statuses = ", ".join(summary["local_statuses"]) or (
            "hardware unavailable" if summary["table"] == "IV" else "pending"
        )
        lines.append(
            f"| {summary['table']} | {summary['scope']} | "
            f"{summary['paper_average']:.2f} | {_rate(summary['local_available_mean'])} | "
            f"{coverage} | {statuses} |"
        )

    records = document["records"]
    for table in ("I", "II", "III", "IV"):
        lines.extend(
            [
                "",
                f"## Table {table}: {TABLE_SCOPES[table]}",
                "",
                "| Condition | Task | Paper DynaMAC | Local | Difference | Status | Protocol note | Evidence |",
                "|---|---|---:|---:|---:|---|---|---|",
            ]
        )
        for row in (item for item in records if item["table"] == table):
            evidence = "—"
            if row["source_file"]:
                evidence = f"[JSON]({row['source_file']})"
            lines.append(
                f"| {row['condition']} | {row['task']} | "
                f"{float(row['paper_success_rate']):.2f} | "
                f"{_rate(row['local_success_rate'])} | {_difference(row['difference'])} | "
                f"{row['status']} | {row.get('protocol_note') or row['notes']} | {evidence} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A numerical difference is descriptive only. Even the static local runs use "
            "independently sampled demonstrations and are not byte-for-byte reruns of the "
            "authors' data. Differences beside diagnostic rows must not be interpreted as "
            "reproduction gaps.",
            "",
        ]
    )
    if document["discovery_warnings"]:
        lines.extend(["## Ignored result files", ""])
        lines.extend(f"- {warning}" for warning in document["discovery_warnings"])
        lines.append("")
    return "\n".join(lines)


CSV_FIELDS = (
    "table",
    "condition",
    "task",
    "paper_success_rate",
    "local_success_rate",
    "difference",
    "local_successes",
    "local_episodes",
    "status",
    "source_file",
    "notes",
    "protocol_note",
)


def write_outputs(
    document: dict[str, Any],
    *,
    markdown_path: Path,
    csv_path: Path,
    json_path: Path,
) -> None:
    for path in (markdown_path, csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(document), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS}
            for row in document["records"]
        )
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help=f"Version directory below results-dir to select (default: {DEFAULT_RELEASE}).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "paper_comparison.md",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "paper_comparison.csv",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "paper_comparison.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0 or args.episodes < 1 or args.horizon < 1:
        raise ValueError("seed must be non-negative; episodes and horizon must be positive")
    records, warnings = build_records(
        args.results_dir,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        release=args.release,
    )
    document = build_document(
        records,
        warnings,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        release=args.release,
    )
    write_outputs(
        document,
        markdown_path=args.markdown_output,
        csv_path=args.csv_output,
        json_path=args.json_output,
    )
    print(markdown(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
