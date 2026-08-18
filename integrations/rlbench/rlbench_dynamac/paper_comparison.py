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

from .direct_evaluate import evaluation_protocol_id as bimanual_evaluation_protocol_id
from .direct_policy import (
    TRAINING_MANIFEST_SCHEMA_V3,
    V3_ADAPTER_PROTOCOL,
)
from .eval_set import (
    GLOBAL_EVAL_SEED_START,
    fixed_coordination_sources,
    fixed_environment_plans,
)
from .runtime import (
    _CONDITION_STRUCTURAL_FIELDS,
    CROSS_INITIALIZATION_JOINT_TOLERANCE,
    CROSS_INITIALIZATION_REPRODUCIBILITY_SCHEMA,
    CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
    CROSS_INITIALIZATION_SCALAR_TOLERANCE,
    CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    DETERMINISTIC_SOURCE_RESET_EVIDENCE_SCHEMA,
    DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
    FORMAL_INTERVENTION_COLLISION_PAIR_POLICY,
    FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA,
    FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
    FRESH_TASK_GENERATION_PROTOCOL_ID,
    LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
    LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    LOW_DIM_STATE_ROUNDTRIP_ATOL,
    PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
    QUATERNION_ROTATION_METRIC,
    ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD,
    ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M,
    ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
    ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
    STAGED_VALIDATED_MOTION_PROTOCOL_ID,
    TASK_SEMANTIC_SIGNATURE_SCHEMA,
    TASK_TREE_STATE_SCHEMA,
    DiscreteGripperProtocol,
    ScenarioController,
    _canonical_json_fingerprint,
    _compact_task_tree_comparison,
    _root_motion_metrics,
    final_settling_metadata,
)
from .task_specs import get_task_spec
from .unimanual_evaluate import (
    DYNAMIC_EPISODE_ACCOUNTING_SCHEMA,
    EXPECTED_UNIMANUAL_BASE_SCENE_SHA256,
    EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT,
    LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
)
from .unimanual_evaluate import (
    evaluation_protocol_id as unimanual_evaluation_protocol_id,
)
from .v3_protocol import (
    V3_SELECTION_SEMANTICS_ID,
    load_v3_intervention_protocol,
    load_v3_motion_source_protocol,
    resolve_authenticated_v3_trigger,
)

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results"
DEFAULT_RELEASE = "v3"
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
        INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v2.json"
    ),
    "v3": _canonical_config(
        INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v3.json"
    ),
}
EXPECTED_LOCAL_CONFIG = EXPECTED_RELEASE_CONFIGS[DEFAULT_RELEASE]
EXPECTED_MODEL_SCHEMA_VERSION = 13
EXPECTED_TAPAS_COMMIT = "52e35214b9baa7b190b87196c36b9e98f4006149"
V1_V2_SELECTION_SEMANTICS_ID = (
    "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
)
EXPECTED_RELEASE_SELECTION_SEMANTICS_IDS = {
    "v1": V1_V2_SELECTION_SEMANTICS_ID,
    "v2": V1_V2_SELECTION_SEMANTICS_ID,
    "v3": V3_SELECTION_SEMANTICS_ID,
}
EXPECTED_SELECTION_SEMANTICS_ID = EXPECTED_RELEASE_SELECTION_SEMANTICS_IDS[
    DEFAULT_RELEASE
]
LEGACY_V1_EVALUATION_PROTOCOL_ID = (
    "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-noop-clock-v2"
)
V2_EVALUATION_PROTOCOL_BASE_ID = (
    "rlbench-absolute-ee-ik-jacobian-sampling5-timeout200-"
    "noop-retry-same-policy-tick-fresh-observation-"
    f"primary-attempt{DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS}-v4"
)
V2_DYNAMIC_EPISODE_ACCOUNTING_SCHEMA = "trigger-eligibility-smooth-prefix-v1"
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
    "v3": {
        "unimanual": unimanual_evaluation_protocol_id(
            DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
        ),
        "bimanual": bimanual_evaluation_protocol_id(
            DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
        ),
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
EXPECTED_COORDINATION_VARIATION_COUNT = 5


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


def _release_for_training_config(
    expected_training_config: dict[str, Any] | None,
) -> str | None:
    if expected_training_config is None:
        return None
    matches = [
        release
        for release, config in EXPECTED_RELEASE_CONFIGS.items()
        if config == expected_training_config
    ]
    return matches[0] if len(matches) == 1 else None


def _valid_v3_model_protocol_identity(run: LocalRun, identity: dict[str, Any]) -> bool:
    if (
        identity.get("training_manifest_schema") != TRAINING_MANIFEST_SCHEMA_V3
        or identity.get("training_adapter_protocol") != V3_ADAPTER_PROTOCOL
        or not isinstance(identity.get("checkpoint_trigger_audit_fingerprint"), str)
        or not identity.get("checkpoint_trigger_audit_fingerprint")
    ):
        return False
    try:
        if run.scenario in {"coordination_hand_left", "coordination_hand_right"}:
            authentication = resolve_authenticated_v3_trigger(
                identity,
                scenario=run.scenario,
            )
        else:
            authentication = resolve_authenticated_v3_trigger(
                identity,
                task=run.task,
            )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False
    return (
        authentication.get("evidence_fingerprint")
        == identity.get("v3_trigger_anchor_evidence", {}).get("fingerprint")
    )


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
    release = _release_for_training_config(expected_training_config)
    expected_selection_semantics_id = (
        EXPECTED_RELEASE_SELECTION_SEMANTICS_IDS.get(release)
        if release is not None
        else None
    )
    if (
        identity.get("manifest_authenticated") is True
        and expected_training_config is not None
        and expected_protocol_id is not None
        and identity.get("training_config") == expected_training_config
        and identity.get("model_schema_version") == EXPECTED_MODEL_SCHEMA_VERSION
        and identity.get("selection_semantics_id")
        == expected_selection_semantics_id
        and identity.get("tapas_reference_commit") == EXPECTED_TAPAS_COMMIT
        and fingerprint_present
        and run.payload.get("evaluation_protocol_id") == expected_protocol_id
        and (
            release != "v3"
            or _valid_v3_model_protocol_identity(run, identity)
        )
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
        "sampling_rollback": (
            "task_configuration_tree_only_live_robot_untouched"
        ),
        "sampling_rollback_frequency": "after_each_attempt_and_outer_finally",
        "task_configuration_tree_restore_api": "Task.get_state/restore_state",
        "task_tree_object_count_guard": True,
        "live_robot_state_during_goal_sampling": "untouched",
        "live_robot_configuration_tree_access": "none",
        "online_task_waypoint_validation": (
            "disabled_to_preserve_live_robot_state"
        ),
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


def _finite_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _v3_derived_motion_metric_matches(value: object, expected: object) -> bool:
    """Compare redundant JSON/recomputed root metrics at roundoff precision."""

    return bool(
        _finite_number(value)
        and _finite_number(expected)
        and math.isclose(
            float(value),
            float(expected),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    )


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
    expected_chunk_count: int,
    expected_arms: frozenset[str],
    max_sampling_attempts: int,
) -> bool:
    if not isinstance(preservation, dict):
        return False
    expected_fields = {
        "initialized_episode_preserved",
        "task_init_episode_called",
        "task_validate_called",
        "low_dim_state_roundtrip_preserved",
        "low_dim_state_roundtrip_comparison_mode",
        "low_dim_state_roundtrip_chunk_count",
        "low_dim_state_roundtrip_l2",
        "low_dim_state_roundtrip_max_abs",
        "low_dim_state_roundtrip_max_translation_m",
        "low_dim_state_roundtrip_max_rotation_rad",
        "condition_and_grasp_registry_identity_preserved",
        "gripper_grasp_membership_and_parentage_preserved",
        "configuration_tree_rollback",
        "task_configuration_tree_restored",
        "live_robot_state_untouched",
        "live_robot_configuration_trees_accessed",
        "robot_collision_pair_policy",
        "robot_collision_pair_granularity",
        "source_robot_external_collision_pairs",
        "goal_robot_external_collision_pairs",
        "goal_new_robot_external_collision_pairs",
        "sampling_attempts_rejected_for_new_robot_collision_pairs",
        "sampling_attempts",
        "waypoint_cache_identity_preserved",
    }
    if (
        set(preservation) != expected_fields
        or preservation.get("initialized_episode_preserved") is not True
        or preservation.get("task_init_episode_called") is not False
        or preservation.get("task_validate_called") is not False
        or preservation.get("low_dim_state_roundtrip_preserved") is not True
        or preservation.get("low_dim_state_roundtrip_comparison_mode")
        != "pose_chunks_sign_invariant"
        or preservation.get("low_dim_state_roundtrip_chunk_count")
        != expected_chunk_count
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
    roundtrip_max_translation = preservation.get(
        "low_dim_state_roundtrip_max_translation_m"
    )
    roundtrip_max_rotation = preservation.get(
        "low_dim_state_roundtrip_max_rotation_rad"
    )
    attempts = preservation.get("sampling_attempts")
    rejected = preservation.get(
        "sampling_attempts_rejected_for_new_robot_collision_pairs"
    )
    if (
        not _finite_nonnegative_number(roundtrip_l2)
        or not _finite_nonnegative_number(roundtrip_max_abs)
        or not _finite_nonnegative_number(roundtrip_max_translation)
        or float(roundtrip_max_translation)
        > LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
        or not _finite_nonnegative_number(roundtrip_max_rotation)
        or float(roundtrip_max_rotation) > LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
        or not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or not 1 <= attempts <= max_sampling_attempts
        or not isinstance(rejected, int)
        or isinstance(rejected, bool)
        or not 0 <= rejected < attempts
    ):
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
        V2_DYNAMIC_EPISODE_ACCOUNTING_SCHEMA
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
    try:
        expected_chunk_count = len(get_task_spec(run.task).pose_chunks)
    except (KeyError, ValueError):
        return False
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
                    expected_chunk_count=expected_chunk_count,
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


def _valid_v3_final_settling(run: LocalRun) -> bool:
    expected = final_settling_metadata(DEFAULT_FINAL_SETTLING_PHYSICS_STEPS)
    controller = run.payload.get("controller")
    if (
        not isinstance(controller, dict)
        or controller.get("final_settling") != expected
        or run.payload.get("final_settling_protocol") != expected
        or controller.get("policy_clock_rollback") is not True
        or controller.get("policy_clock_semantics_id")
        != "policy-tick-transaction-commit-on-primary-action-success-v1"
    ):
        return False
    rows = run.payload.get("results")
    if not isinstance(rows, list) or len(rows) != run.episodes:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        settling = row.get("final_settling")
        if not isinstance(settling, dict) or any(
            settling.get(key) != value for key, value in expected.items()
        ):
            return False
        attempted = settling.get("attempted")
        available = settling.get("available")
        steps = settling.get("steps_executed")
        first_terminal = settling.get("first_terminal_step")
        success = settling.get("success")
        terminate = settling.get("terminate")
        stop_reason = settling.get("stop_reason")
        if (
            not isinstance(attempted, bool)
            or available is not True
            or not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 0 <= steps <= DEFAULT_FINAL_SETTLING_PHYSICS_STEPS
            or not isinstance(success, bool)
            or not isinstance(terminate, bool)
        ):
            return False
        if not attempted:
            if (
                steps != 0
                or first_terminal is not None
                or stop_reason != "not_entered"
                or success
                or terminate
                or row.get("reason")
                in {
                    "policy_complete",
                    "policy_complete_after_final_settling",
                    "success_after_final_settling",
                    "terminate_during_final_settling",
                }
            ):
                return False
            continue
        if not 1 <= steps <= DEFAULT_FINAL_SETTLING_PHYSICS_STEPS:
            return False
        if success or terminate:
            if (
                first_terminal != steps
                or stop_reason
                != ("success" if success else "explicit_terminate")
            ):
                return False
        elif (
            steps != DEFAULT_FINAL_SETTLING_PHYSICS_STEPS
            or first_terminal is not None
            or stop_reason != "maximum_physics_steps_reached"
        ):
            return False
        expected_reason = (
            "success_after_final_settling"
            if success
            else (
                "terminate_during_final_settling"
                if terminate
                else "policy_complete_after_final_settling"
            )
        )
        if row.get("reason") != expected_reason or row.get("success") is not success:
            return False
    return True


def _valid_v3_episode_accounting(run: LocalRun) -> bool:
    rows = run.payload.get("results")
    if not isinstance(rows, list) or len(rows) != run.episodes:
        return False
    complete_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("intervention_complete") is True
    ]
    complete_successes = sum(row.get("success") is True for row in complete_rows)
    expected = {
        "schema": DYNAMIC_EPISODE_ACCOUNTING_SCHEMA,
        "planned_episode_denominator": run.episodes,
        "completed_episode_count": run.episodes,
        "successes_in_planned_denominator": run.successes,
        "success_rate_all_planned_episodes": run.success_rate,
        "trigger_reached_count": sum(
            isinstance(row, dict) and row.get("intervention_reached") is True
            for row in rows
        ),
        "intervention_complete_count": len(complete_rows),
        "dynamic_condition_unexercised_count": sum(
            isinstance(row, dict)
            and row.get("dynamic_condition_unexercised") is True
            for row in rows
        ),
        "pre_trigger_success_count": sum(
            isinstance(row, dict)
            and row.get("pre_intervention_terminal") is True
            and row.get("success") is True
            for row in rows
        ),
        "complete_intervention_subset_count": len(complete_rows),
        "successes_in_complete_intervention_subset": complete_successes,
        "success_rate_in_complete_intervention_subset": (
            complete_successes / float(len(complete_rows))
            if complete_rows
            else None
        ),
    }
    return run.payload.get("episode_accounting") == expected


def _v3_trigger_authentication(run: LocalRun) -> dict[str, Any] | None:
    identity = run.payload.get("model_identity")
    if not isinstance(identity, dict):
        return None
    try:
        if run.scenario in {"coordination_hand_left", "coordination_hand_right"}:
            return resolve_authenticated_v3_trigger(
                identity,
                scenario=run.scenario,
            )
        return resolve_authenticated_v3_trigger(identity, task=run.task)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None


def _valid_v3_trigger_metadata(run: LocalRun) -> bool:
    metadata = _protocol_metadata(run)
    if run.scenario == "local_baseline":
        identity = run.payload.get("model_identity")
        try:
            authentication = {
                side: resolve_authenticated_v3_trigger(
                    identity,
                    scenario=f"coordination_hand_{side}",
                )
                for side in ("left", "right")
            }
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False
        protocol = load_v3_intervention_protocol()
        return bool(
            metadata.get("trigger_authentication") == authentication
            and metadata.get("trigger_policy_step") is None
            and metadata.get("intervention_registry_schema") == protocol["schema"]
            and metadata.get("intervention_registry_fingerprint")
            == protocol["fingerprint"]
            and metadata.get("trigger_reference_domain")
            == "successfully_committed_policy_ticks"
        )
    authentication = (
        metadata.get("trigger_authentication")
        if run.scenario == "local_baseline"
        else _v3_trigger_authentication(run)
    )
    if not isinstance(authentication, dict):
        return False
    protocol = load_v3_intervention_protocol()
    expected_step = (
        None if run.scenario == "static" else authentication["trigger_step"]
    )
    return (
        metadata.get("trigger_authentication") == authentication
        and metadata.get("trigger_policy_step") == expected_step
        and metadata.get("intervention_registry_schema") == protocol["schema"]
        and metadata.get("intervention_registry_fingerprint")
        == protocol["fingerprint"]
        and metadata.get("trigger_reference_domain")
        == "successfully_committed_policy_ticks"
    )


def _v3_cross_initialization_tolerances() -> dict[str, float]:
    return {
        "root_translation_m": ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
        "root_rotation_rad": ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
        "task_pose_translation_m": (
            CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
        ),
        "task_pose_rotation_rad": CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
        "scalar_state": CROSS_INITIALIZATION_SCALAR_TOLERANCE,
        "joint_position": CROSS_INITIALIZATION_JOINT_TOLERANCE,
    }


def _valid_v3_pose(value: Any) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 7
        or any(not _finite_number(item) for item in value)
    ):
        return False
    quaternion_norm = math.sqrt(sum(float(item) ** 2 for item in value[3:7]))
    return quaternion_norm > 0.0 and math.isclose(
        quaternion_norm,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-3,
    )


def _valid_v3_task_tree_state(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    required = {
        "name",
        "type",
        "parent",
        "in_boundary_root_subtree",
        "world_pose",
        "pose_relative_to_boundary_root",
    }
    keys: list[tuple[str, str, str]] = []
    for row in value:
        row_keys = set(row) if isinstance(row, dict) else set()
        if (
            not isinstance(row, dict)
            or (
                row_keys != required
                and row_keys != required | {"joint_position"}
            )
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or not isinstance(row.get("type"), str)
            or not row["type"]
            or not (
                row.get("parent") is None
                or isinstance(row.get("parent"), str)
            )
            or not isinstance(row.get("in_boundary_root_subtree"), bool)
            or not _valid_v3_pose(row.get("world_pose"))
            or not _valid_v3_pose(row.get("pose_relative_to_boundary_root"))
            or (
                "joint_position" in row
                and not _finite_number(row.get("joint_position"))
            )
        ):
            return False
        keys.append((row["name"], row["type"], row.get("parent") or ""))
    identities = [(name, object_type) for name, object_type, _parent in keys]
    return keys == sorted(keys) and len(identities) == len(set(identities))


def _valid_v3_stable_collision_records(
    value: Any,
    *,
    expected_arms: frozenset[str],
) -> bool:
    if not isinstance(value, list):
        return False
    keys: list[tuple[str, str]] = []
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != {"arm", "external_object_name"}
            or row.get("arm") not in expected_arms
            or not isinstance(row.get("external_object_name"), str)
            or not row["external_object_name"]
        ):
            return False
        keys.append((row["arm"], row["external_object_name"]))
    return keys == sorted(keys) and len(keys) == len(set(keys))


def _valid_v3_root_reproducibility(
    value: Any,
    *,
    translation_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> bool:
    expected_fields = {
        "preserved",
        "translation_error_m",
        "rotation_error_rad",
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "quaternion_rotation_metric",
    }
    return bool(
        isinstance(value, dict)
        and set(value) == expected_fields
        and value.get("preserved") is True
        and value.get("translation_tolerance_m") == translation_tolerance_m
        and value.get("rotation_tolerance_rad") == rotation_tolerance_rad
        and value.get("quaternion_rotation_metric")
        == QUATERNION_ROTATION_METRIC
        and _finite_nonnegative_number(value.get("translation_error_m"))
        and float(value["translation_error_m"]) <= translation_tolerance_m
        and _finite_nonnegative_number(value.get("rotation_error_rad"))
        and float(value["rotation_error_rad"]) <= rotation_tolerance_rad
    )


def _valid_v3_low_dim_reproducibility(
    value: Any,
    *,
    expected_chunk_count: int,
    translation_tolerance_m: float,
    rotation_tolerance_rad: float,
    scalar_tolerance: float,
) -> bool:
    expected_fields = {
        "preserved",
        "comparison_mode",
        "chunk_count",
        "raw_l2",
        "raw_max_abs",
        "max_translation_m",
        "max_rotation_rad",
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "scalar_tolerance",
        "quaternion_rotation_metric",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("preserved") is not True
        or value.get("translation_tolerance_m") != translation_tolerance_m
        or value.get("rotation_tolerance_rad") != rotation_tolerance_rad
        or value.get("scalar_tolerance") != scalar_tolerance
        or value.get("quaternion_rotation_metric")
        != QUATERNION_ROTATION_METRIC
        or not _finite_nonnegative_number(value.get("raw_l2"))
        or not _finite_nonnegative_number(value.get("raw_max_abs"))
    ):
        return False
    if value.get("comparison_mode") == "pose_chunks_sign_invariant":
        return bool(
            value.get("chunk_count") == expected_chunk_count
            and expected_chunk_count > 0
            and _finite_nonnegative_number(value.get("max_translation_m"))
            and float(value["max_translation_m"]) <= translation_tolerance_m
            and _finite_nonnegative_number(value.get("max_rotation_rad"))
            and float(value["max_rotation_rad"]) <= rotation_tolerance_rad
        )
    return bool(
        value.get("comparison_mode") == "scalar_max_abs"
        and expected_chunk_count == 0
        and value.get("chunk_count") == 0
        and value.get("max_translation_m") is None
        and value.get("max_rotation_rad") is None
        and float(value["raw_max_abs"]) <= scalar_tolerance
    )


def _valid_v3_compact_task_tree_comparison(
    value: Any,
    *,
    expected_object_count: int,
    translation_tolerance_m: float,
    rotation_tolerance_rad: float,
    joint_tolerance: float,
) -> bool:
    expected_fields = {
        "matched",
        "topology_matched",
        "comparison_mode",
        "expected_object_count",
        "actual_object_count",
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "joint_tolerance",
        "quaternion_rotation_metric",
        "all_parents_matched",
        "all_subtree_memberships_matched",
        "max_translation_error_m",
        "max_rotation_error_rad",
        "max_joint_position_error",
    }
    return bool(
        isinstance(value, dict)
        and set(value) == expected_fields
        and value.get("matched") is True
        and value.get("topology_matched") is True
        and value.get("comparison_mode") == "all_objects_world"
        and value.get("expected_object_count") == expected_object_count
        and value.get("actual_object_count") == expected_object_count
        and value.get("translation_tolerance_m") == translation_tolerance_m
        and value.get("rotation_tolerance_rad") == rotation_tolerance_rad
        and value.get("joint_tolerance") == joint_tolerance
        and value.get("quaternion_rotation_metric")
        == QUATERNION_ROTATION_METRIC
        and value.get("all_parents_matched") is True
        and value.get("all_subtree_memberships_matched") is True
        and _finite_nonnegative_number(value.get("max_translation_error_m"))
        and float(value["max_translation_error_m"]) <= translation_tolerance_m
        and _finite_nonnegative_number(value.get("max_rotation_error_rad"))
        and float(value["max_rotation_error_rad"]) <= rotation_tolerance_rad
        and _finite_nonnegative_number(value.get("max_joint_position_error"))
        and float(value["max_joint_position_error"]) <= joint_tolerance
    )


def _valid_v3_full_task_tree_comparison(
    value: Any,
    *,
    expected_mode: str,
    expected_object_count: int,
    translation_tolerance_m: float,
    rotation_tolerance_rad: float,
    joint_tolerance: float,
    expected_identities: list[tuple[str, str]] | None = None,
) -> bool:
    expected_fields = {
        "matched",
        "topology_matched",
        "comparison_mode",
        "expected_object_count",
        "actual_object_count",
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "joint_tolerance",
        "quaternion_rotation_metric",
        "objects",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("matched") is not True
        or value.get("topology_matched") is not True
        or value.get("comparison_mode") != expected_mode
        or value.get("expected_object_count") != expected_object_count
        or value.get("actual_object_count") != expected_object_count
        or value.get("translation_tolerance_m") != translation_tolerance_m
        or value.get("rotation_tolerance_rad") != rotation_tolerance_rad
        or value.get("joint_tolerance") != joint_tolerance
        or value.get("quaternion_rotation_metric")
        != QUATERNION_ROTATION_METRIC
        or not isinstance(value.get("objects"), list)
        or len(value["objects"]) != expected_object_count
    ):
        return False
    row_fields = {
        "name",
        "type",
        "matched",
        "parent_matched",
        "subtree_membership_matched",
        "in_boundary_root_subtree",
        "pose_comparison_frame",
        "translation_error_m",
        "rotation_error_rad",
        "world_translation_error_m",
        "world_rotation_error_rad",
        "boundary_root_relative_translation_error_m",
        "boundary_root_relative_rotation_error_rad",
        "joint_position_error",
    }
    identities: list[tuple[str, str]] = []
    for row in value["objects"]:
        if (
            not isinstance(row, dict)
            or set(row) != row_fields
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or not isinstance(row.get("type"), str)
            or not row["type"]
            or row.get("matched") is not True
            or row.get("parent_matched") is not True
            or row.get("subtree_membership_matched") is not True
            or not isinstance(row.get("in_boundary_root_subtree"), bool)
            or any(
                not _finite_nonnegative_number(row.get(field))
                for field in (
                    "translation_error_m",
                    "rotation_error_rad",
                    "world_translation_error_m",
                    "world_rotation_error_rad",
                    "boundary_root_relative_translation_error_m",
                    "boundary_root_relative_rotation_error_rad",
                )
            )
            or float(row["translation_error_m"]) > translation_tolerance_m
            or float(row["rotation_error_rad"]) > rotation_tolerance_rad
            or (
                row.get("joint_position_error") is not None
                and (
                    not _finite_nonnegative_number(row["joint_position_error"])
                    or float(row["joint_position_error"]) > joint_tolerance
                )
            )
        ):
            return False
        expected_frame = (
            "boundary_root"
            if expected_mode == "boundary_root_subtree_relative_else_world"
            and row["in_boundary_root_subtree"]
            else "world"
        )
        selected_prefix = (
            "boundary_root_relative" if expected_frame == "boundary_root" else "world"
        )
        if (
            row.get("pose_comparison_frame") != expected_frame
            or row.get("translation_error_m")
            != row.get(f"{selected_prefix}_translation_error_m")
            or row.get("rotation_error_rad")
            != row.get(f"{selected_prefix}_rotation_error_rad")
        ):
            return False
        identities.append((row["name"], row["type"]))
    return bool(
        identities == sorted(identities)
        and len(identities) == len(set(identities))
        and (expected_identities is None or identities == expected_identities)
    )


def _valid_v3_velocity_summary(
    value: Any,
    *,
    expected_object_count: int | None = None,
) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value)
        == {
            "schema",
            "compared_for_identity",
            "diagnostic_only",
            "object_count",
            "all_finite",
            "max_linear_speed_m_s",
            "max_angular_speed_rad_s",
        }
        and value.get("schema") == "rlbench-task-tree-velocity-summary-v1"
        and value.get("compared_for_identity") is False
        and value.get("diagnostic_only") is True
        and isinstance(value.get("object_count"), int)
        and not isinstance(value.get("object_count"), bool)
        and value["object_count"] > 0
        and (
            expected_object_count is None
            or value["object_count"] == expected_object_count
        )
        and value.get("all_finite") is True
        and isinstance(value.get("max_linear_speed_m_s"), (int, float))
        and not isinstance(value.get("max_linear_speed_m_s"), bool)
        and math.isfinite(float(value["max_linear_speed_m_s"]))
        and float(value["max_linear_speed_m_s"]) >= 0.0
        and isinstance(value.get("max_angular_speed_rad_s"), (int, float))
        and not isinstance(value.get("max_angular_speed_rad_s"), bool)
        and math.isfinite(float(value["max_angular_speed_rad_s"]))
        and float(value["max_angular_speed_rad_s"]) >= 0.0
    )


def _valid_v3_fingerprint(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_v3_fresh_task_generation_evidence(
    value: Any,
    *,
    generation_index: int | None = None,
    episode_seed: int | None = None,
    variation: int | None = None,
    task_name: str | None = None,
    verify_instance: bool = True,
) -> bool:
    expected_fields = {
        "schema",
        "protocol_id",
        "generation_index",
        "episode_seed",
        "variation",
        "task_name",
        "physics_running_before_stop",
        "physics_stopped_before_task_reload",
        "previous_task_present",
        "previous_task_unloaded_before_stop",
        "previous_task_unloaded_while_physics_running",
        "scene_task_absent_before_stop",
        "task_model_loaded_fresh",
        "fresh_task_python_instance_created",
        "task_model_only_reloaded",
        "base_scene_reloaded",
        "physics_started_by_task_environment",
        "rng_seeded_after_reload_immediately_before_reset",
        "variation_set_after_seed_before_reset",
        "task_environment_reset_calls",
        "reset_verify_instance",
        "fingerprint",
    }
    if not verify_instance:
        expected_fields.update(
            {
                "reset_random_placement_expected",
                "reset_robot_collision_check_count",
                "reset_robot_collision_check_results",
            }
        )
    if not isinstance(value, dict) or set(value) != expected_fields:
        return False
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    return bool(
        value.get("schema")
        == (
            FRESH_TASK_GENERATION_EVIDENCE_SCHEMA
            if verify_instance
            else DETERMINISTIC_SOURCE_RESET_EVIDENCE_SCHEMA
        )
        and value.get("protocol_id")
        == (
            FRESH_TASK_GENERATION_PROTOCOL_ID
            if verify_instance
            else DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
        )
        and isinstance(value.get("generation_index"), int)
        and not isinstance(value.get("generation_index"), bool)
        and value["generation_index"] >= 1
        and (generation_index is None or value["generation_index"] == generation_index)
        and isinstance(value.get("episode_seed"), int)
        and not isinstance(value.get("episode_seed"), bool)
        and value["episode_seed"] >= 0
        and (episode_seed is None or value["episode_seed"] == episode_seed)
        and isinstance(value.get("variation"), int)
        and not isinstance(value.get("variation"), bool)
        and value["variation"] >= 0
        and (variation is None or value["variation"] == variation)
        and isinstance(value.get("task_name"), str)
        and bool(value["task_name"])
        and (task_name is None or value["task_name"] == task_name)
        and isinstance(value.get("physics_running_before_stop"), bool)
        and isinstance(value.get("previous_task_present"), bool)
        and value.get("physics_running_before_stop")
        is value.get("previous_task_present")
        and value.get("physics_stopped_before_task_reload") is True
        and value.get("previous_task_unloaded_before_stop")
        is value.get("previous_task_present")
        and value.get("previous_task_unloaded_while_physics_running")
        is value.get("previous_task_present")
        and value.get("scene_task_absent_before_stop") is True
        and value.get("task_model_loaded_fresh") is True
        and value.get("fresh_task_python_instance_created") is True
        and value.get("task_model_only_reloaded") is True
        and value.get("base_scene_reloaded") is False
        and value.get("physics_started_by_task_environment") is True
        and value.get("rng_seeded_after_reload_immediately_before_reset") is True
        and value.get("variation_set_after_seed_before_reset") is True
        and isinstance(value.get("task_environment_reset_calls"), int)
        and not isinstance(value.get("task_environment_reset_calls"), bool)
        and value.get("task_environment_reset_calls") == 1
        and value.get("reset_verify_instance") is verify_instance
        and (
            verify_instance
            or (
                isinstance(value.get("reset_random_placement_expected"), bool)
                and value.get("reset_robot_collision_check_results")
                == (
                    [False]
                    if value["reset_random_placement_expected"]
                    else []
                )
                and value.get("reset_robot_collision_check_count")
                == len(value["reset_robot_collision_check_results"])
            )
        )
        and value.get("fingerprint") == _canonical_json_fingerprint(body)
    )


def _valid_v3_formal_fresh_task_generations(run: LocalRun) -> bool:
    rows = run.payload.get("results")
    envelope = run.payload.get("fresh_task_generation")
    controller = run.payload.get("controller")
    variations = run.payload.get("variation_schedule")
    variation_count = run.payload.get("variation_count")
    deterministic_dynamic = (
        run.scenario in {"static", "smooth", "teleport", "local_baseline"}
        or run.scenario.startswith("coordination_hand_")
    ) and isinstance(run.payload.get("fixed_eval_set"), dict)
    expected_lifecycle = (
        DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
        if deterministic_dynamic
        else FRESH_TASK_GENERATION_PROTOCOL_ID
    )
    if (
        not isinstance(rows, list)
        or len(rows) != run.episodes
        or not isinstance(envelope, dict)
        or set(envelope)
        != {
            "required_per_formal_episode",
            "all_episodes_recorded",
            "evidence",
        }
        or envelope.get("required_per_formal_episode") is not True
        or envelope.get("all_episodes_recorded") is not True
        or not isinstance(envelope.get("evidence"), list)
        or len(envelope["evidence"]) != run.episodes
        or not isinstance(controller, dict)
        or controller.get("formal_episode_initialization")
        != expected_lifecycle
        or not isinstance(variations, list)
        or len(variations) != run.episodes
        or not isinstance(variation_count, int)
        or isinstance(variation_count, bool)
        or variation_count < 1
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < variation_count
            for value in variations
        )
    ):
        return False
    if run.task in UNIMANUAL_TASKS:
        if variations != [run.variation] * run.episodes:
            return False
    elif not run.scenario.startswith("coordination_hand_"):
        if variations != [
            episode % variation_count for episode in range(run.episodes)
        ]:
            return False
    for episode, (row, evidence, variation) in enumerate(
        zip(rows, envelope["evidence"], variations)
    ):
        if (
            not isinstance(row, dict)
            or row.get("fresh_task_generation") != evidence
            or not _valid_v3_fresh_task_generation_evidence(
                evidence,
                generation_index=episode + 1,
                episode_seed=(None if deterministic_dynamic else run.seed + episode),
                variation=variation,
                task_name=run.task,
                verify_instance=not deterministic_dynamic,
            )
            or evidence.get("previous_task_present") is not (episode > 0)
            or evidence.get("physics_running_before_stop") is not (episode > 0)
        ):
            return False
    return True


_V3_SEMANTIC_RESERVED_FIELDS = frozenset(
    {
        "type",
        "name",
        "object_name",
        "fields",
        "structural_fields",
        "excluded_runtime_progress_fields",
    }
)


def _valid_v3_semantic_type(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "." in value


def _valid_v3_semantic_condition(value: Any, *, depth: int) -> bool:
    if (
        not isinstance(value, dict)
        or set(value)
        != {"type", "structural_fields", "excluded_runtime_progress_fields"}
        or not _valid_v3_semantic_type(value.get("type"))
        or not isinstance(value.get("structural_fields"), dict)
        or not isinstance(value.get("excluded_runtime_progress_fields"), list)
    ):
        return False
    condition_type = value["type"]
    structural_fields = value["structural_fields"]
    excluded_fields = value["excluded_runtime_progress_fields"]
    schema = _CONDITION_STRUCTURAL_FIELDS.get(condition_type)
    if schema is not None:
        expected_structural_fields, expected_runtime_fields = schema
        if (
            set(structural_fields) != set(expected_structural_fields)
            or excluded_fields != list(expected_runtime_fields)
        ):
            return False
    elif condition_type.startswith("rlbench.backend.conditions."):
        return False
    elif excluded_fields:
        # Custom conditions accepted by the runtime inherit the base no-op
        # reset and therefore cannot declare excluded execution progress.
        return False
    if condition_type in {
        "rlbench.backend.conditions.ConditionSet",
        "rlbench.backend.conditions.OrConditions",
    }:
        nested_conditions = structural_fields.get("_conditions")
        if not isinstance(nested_conditions, list) or any(
            not isinstance(item, dict)
            or set(item)
            not in (
                {
                    "type",
                    "structural_fields",
                    "excluded_runtime_progress_fields",
                },
                {"type", "fields"},
            )
            or not _valid_v3_semantic_value(item, depth=depth + 1)
            for item in nested_conditions
        ):
            return False
    return all(
        isinstance(name, str)
        and bool(name)
        and _valid_v3_semantic_value(item, depth=depth + 1)
        for name, item in structural_fields.items()
    )


def _valid_v3_semantic_value(value: Any, *, depth: int = 0) -> bool:
    if depth > 16:
        return False
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(
            _valid_v3_semantic_value(item, depth=depth + 1)
            for item in value
        )
    if not isinstance(value, dict):
        return False
    fields = set(value)
    if fields == {"object_name"}:
        return isinstance(value["object_name"], str) and bool(value["object_name"])
    if fields == {"type", "name"}:
        return bool(
            _valid_v3_semantic_type(value["type"])
            and not value["type"].startswith("rlbench.backend.conditions.")
            and (
                value["name"] is None
                or isinstance(value["name"], str)
                and bool(value["name"])
            )
        )
    if fields == {
        "type",
        "structural_fields",
        "excluded_runtime_progress_fields",
    }:
        return _valid_v3_semantic_condition(value, depth=depth)
    if fields == {"type", "fields"}:
        return bool(
            _valid_v3_semantic_type(value["type"])
            and not value["type"].startswith("rlbench.backend.conditions.")
            and isinstance(value["fields"], dict)
            and all(
                isinstance(name, str)
                and bool(name)
                and _valid_v3_semantic_value(item, depth=depth + 1)
                for name, item in value["fields"].items()
            )
        )
    if fields & _V3_SEMANTIC_RESERVED_FIELDS:
        return False
    return all(
        isinstance(name, str)
        and _valid_v3_semantic_value(item, depth=depth + 1)
        for name, item in value.items()
    )


def _valid_v3_task_semantic_signature(value: Any) -> bool:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema",
            "task_class",
            "success_conditions",
            "fail_conditions",
            "graspable_objects",
        }
        or value.get("schema") != TASK_SEMANTIC_SIGNATURE_SCHEMA
        or not _valid_v3_semantic_type(value.get("task_class"))
        or any(
            not isinstance(value.get(name), list)
            for name in (
                "success_conditions",
                "fail_conditions",
                "graspable_objects",
            )
        )
    ):
        return False
    return bool(
        all(
            isinstance(item, dict)
            and _valid_v3_semantic_condition(item, depth=0)
            for name in ("success_conditions", "fail_conditions")
            for item in value[name]
        )
        and all(
            isinstance(item, dict)
            and set(item) == {"type", "name"}
            and _valid_v3_semantic_value(item)
            for item in value["graspable_objects"]
        )
    )


def _valid_v3_cross_initialization_audit(
    validation: dict[str, Any],
    sampling_attempts: int,
    *,
    episode_seed: int,
    variation: int,
    task_name: str,
    expected_object_count: int,
    expected_chunk_count: int,
) -> bool:
    audit = validation.get("cross_initialization_reproducibility")
    if not isinstance(audit, dict):
        return False
    attempts = audit.get("attempts")
    if (
        set(audit)
        != {
            "schema",
            "comparison_class",
            "fresh_task_generation_protocol_id",
            "candidate_source_policy",
            "reference_attempt",
            "selected_attempt",
            "reference_source_fingerprint",
            "selected_source_fingerprint",
            "all_attempts_passed",
            "attempts",
            "tolerances",
            "quaternion_rotation_metric",
            "worst_observed",
            "tolerance_role",
        }
        or audit.get("schema") != CROSS_INITIALIZATION_REPRODUCIBILITY_SCHEMA
        or audit.get("comparison_class")
        != "same_seed_variation_independent_fresh_task_generation"
        or audit.get("fresh_task_generation_protocol_id")
        != FRESH_TASK_GENERATION_PROTOCOL_ID
        or audit.get("candidate_source_policy")
        != "each_candidate_uses_its_same_fresh_generation_A"
        or audit.get("reference_attempt") != 1
        or audit.get("selected_attempt") != sampling_attempts
        or audit.get("selected_source_fingerprint")
        != validation.get("selected_source_fingerprint")
        or audit.get("all_attempts_passed") is not True
        or audit.get("tolerances") != _v3_cross_initialization_tolerances()
        or audit.get("quaternion_rotation_metric")
        != QUATERNION_ROTATION_METRIC
        or not isinstance(attempts, list)
        or len(attempts) != sampling_attempts
        or not all(isinstance(row, dict) for row in attempts)
        or [row.get("attempt") for row in attempts]
        != list(range(1, sampling_attempts + 1))
        or attempts[0].get("source_fingerprint")
        != audit.get("reference_source_fingerprint")
        or attempts[-1].get("source_fingerprint")
        != audit.get("selected_source_fingerprint")
        or not _valid_v3_fingerprint(audit.get("reference_source_fingerprint"))
        or not _valid_v3_fingerprint(audit.get("selected_source_fingerprint"))
        or audit.get("tolerance_role")
        != "fail_closed_cross_generation_watchdog"
    ):
        return False
    for row in attempts:
        root = row.get("root")
        low_dim = row.get("low_dim_state")
        task_tree = row.get("task_tree")
        if (
            set(row)
            != {
                "attempt",
                "fresh_task_generation",
                "source_fingerprint",
                "root",
                "low_dim_state",
                "task_tree",
                "task_semantics_matched",
                "task_descriptions_matched",
                "robot_external_collision_pairs_matched",
                "task_object_velocities_finite",
                "passed",
            }
            or row.get("passed") is not True
            or row.get("task_semantics_matched") is not True
            or row.get("task_descriptions_matched") is not True
            or row.get("robot_external_collision_pairs_matched") is not True
            or row.get("task_object_velocities_finite") is not True
            or not _valid_v3_fingerprint(row.get("source_fingerprint"))
            or not _valid_v3_fresh_task_generation_evidence(
                row.get("fresh_task_generation"),
                episode_seed=episode_seed,
                variation=variation,
                task_name=task_name,
            )
            or not _valid_v3_root_reproducibility(
                root,
                translation_tolerance_m=ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
                rotation_tolerance_rad=ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
            )
            or not _valid_v3_low_dim_reproducibility(
                low_dim,
                expected_chunk_count=expected_chunk_count,
                translation_tolerance_m=(
                    CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
                ),
                rotation_tolerance_rad=(
                    CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD
                ),
                scalar_tolerance=CROSS_INITIALIZATION_SCALAR_TOLERANCE,
            )
            or not _valid_v3_compact_task_tree_comparison(
                task_tree,
                expected_object_count=expected_object_count,
                translation_tolerance_m=(
                    CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
                ),
                rotation_tolerance_rad=(
                    CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD
                ),
                joint_tolerance=CROSS_INITIALIZATION_JOINT_TOLERANCE,
            )
        ):
            return False
    worst = audit.get("worst_observed")
    worst_keys = {
        "root_translation_m",
        "root_rotation_rad",
        "low_dim_pose_translation_m",
        "low_dim_pose_rotation_rad",
        "low_dim_raw_max_abs",
        "task_tree_translation_m",
        "task_tree_rotation_rad",
        "task_tree_joint_position",
    }
    pose_attempts = [
        row["low_dim_state"]
        for row in attempts
        if row["low_dim_state"]["comparison_mode"]
        == "pose_chunks_sign_invariant"
    ]
    recomputed_worst = {
        "root_translation_m": max(
            float(row["root"]["translation_error_m"]) for row in attempts
        ),
        "root_rotation_rad": max(
            float(row["root"]["rotation_error_rad"]) for row in attempts
        ),
        "low_dim_pose_translation_m": max(
            (float(row["max_translation_m"]) for row in pose_attempts),
            default=0.0,
        ),
        "low_dim_pose_rotation_rad": max(
            (float(row["max_rotation_rad"]) for row in pose_attempts),
            default=0.0,
        ),
        "low_dim_raw_max_abs": max(
            float(row["low_dim_state"]["raw_max_abs"]) for row in attempts
        ),
        "task_tree_translation_m": max(
            float(row["task_tree"]["max_translation_error_m"])
            for row in attempts
        ),
        "task_tree_rotation_rad": max(
            float(row["task_tree"]["max_rotation_error_rad"])
            for row in attempts
        ),
        "task_tree_joint_position": max(
            float(row["task_tree"]["max_joint_position_error"])
            for row in attempts
        ),
    }
    return bool(
        isinstance(worst, dict)
        and set(worst) == worst_keys
        and all(
            isinstance(worst.get(key), (int, float))
            and not isinstance(worst.get(key), bool)
            and math.isfinite(float(worst[key]))
            and float(worst[key]) >= 0.0
            for key in worst_keys
        )
        and worst == recomputed_worst
        and worst.get("root_translation_m")
        <= ROOT_COMMAND_TRANSLATION_TOLERANCE_M
        and worst.get("root_rotation_rad")
        <= ROOT_COMMAND_ROTATION_TOLERANCE_RAD
        and worst.get("low_dim_pose_translation_m")
        <= CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
        and worst.get("low_dim_pose_rotation_rad")
        <= CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD
        and worst.get("task_tree_translation_m")
        <= CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
        and worst.get("task_tree_rotation_rad")
        <= CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD
        and worst.get("task_tree_joint_position")
        <= CROSS_INITIALIZATION_JOINT_TOLERANCE
        and all(
            left.get("generation_index") < right.get("generation_index")
            for left, right in zip(
                (row["fresh_task_generation"] for row in attempts),
                (row["fresh_task_generation"] for row in attempts[1:]),
            )
        )
        and validation.get("selected_source_fresh_task_generation")
        == attempts[-1].get("fresh_task_generation")
    )


def _valid_v3_task_frame_rigid_motion(value: Any, spec: Any) -> bool:
    expected_fields = {
        "task_spec",
        "source_expression",
        "checked_frames",
        "all_pose_chunks_follow_boundary_root_rigid_transform",
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "frames",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("task_spec") != spec.task_name
        or value.get("source_expression") != spec.source_expression
        or value.get("checked_frames") != list(spec.frame_names)
        or value.get("all_pose_chunks_follow_boundary_root_rigid_transform")
        is not True
        or value.get("translation_tolerance_m")
        != LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
        or value.get("rotation_tolerance_rad")
        != LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
        or not isinstance(value.get("frames"), list)
        or len(value["frames"]) != len(spec.frame_names)
    ):
        return False
    for expected_name, row in zip(spec.frame_names, value["frames"]):
        if (
            not isinstance(row, dict)
            or set(row)
            != {"frame", "translation_error_m", "rotation_error_rad", "preserved"}
            or row.get("frame") != expected_name
            or row.get("preserved") is not True
            or not _finite_nonnegative_number(row.get("translation_error_m"))
            or float(row["translation_error_m"])
            > LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
            or not _finite_nonnegative_number(row.get("rotation_error_rad"))
            or float(row["rotation_error_rad"])
            > LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
        ):
            return False
    return True


def _load_v3_staged_plans(run: LocalRun) -> list[Any] | None:
    fixed = run.payload.get("fixed_eval_set")
    if not isinstance(fixed, dict):
        return None
    eval_set_id = fixed.get("evaluation_set_id")
    if not isinstance(eval_set_id, str) or not eval_set_id:
        return None
    try:
        manifest, selected = fixed_environment_plans(eval_set_id, run.task)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    payload = selected["payload"]
    plans = selected["plans"]
    manifest_reference = manifest["payload"]["environment_plan_batches"][run.task]
    if (
        fixed
        != {
            "evaluation_set_id": eval_set_id,
            "manifest_sha256": manifest["manifest_sha256"],
            "spec_sha256": manifest["payload"]["spec"]["sha256"],
            "selected_batch_sha256": manifest_reference["sha256"],
            "selected_batch_fingerprint": payload["batch_fingerprint"],
            "formal_access": "canonical_id_read_only_no_generation",
        }
        or run.payload.get("motion_plan_batch_fingerprint")
        != payload.get("batch_fingerprint")
        or run.seed != payload.get("base_seed")
        or run.episodes != payload.get("episodes")
    ):
        return None
    return plans

def _valid_v3_staged_source_binding(row: dict[str, Any], plan: Any) -> bool:
    binding = row.get("staged_source_binding")
    reconstruction = (
        binding.get("deterministic_source_reconstruction")
        if isinstance(binding, dict)
        else None
    )
    evidence = row.get("motion_plan_evidence")
    formal_generation = row.get("fresh_task_generation")
    if not (
        isinstance(binding, dict)
        and binding.get("required") is True
        and binding.get("matched") is True
        and binding.get("formal_source_bound") is True
        and binding.get("source_seed") == plan.validation.get("source_seed")
        and binding.get("formal_sampling_or_restore") is False
        and binding.get("formal_task_validate_calls") == 0
        and binding.get("formal_waypoint_cache_state") == "none"
        and binding.get("staging_source_certification_reused") is True
        and binding.get("formal_observation_refreshed_after_binding") is True
        and binding.get("fresh_task_generation_protocol_id")
        == DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID
        and binding.get("selected_source_fresh_task_generation")
        == plan.validation.get("selected_source_fresh_task_generation")
        and binding.get("formal_source_fresh_task_generation") == formal_generation
        and _valid_v3_fresh_task_generation_evidence(
            formal_generation,
            episode_seed=plan.validation.get("source_seed"),
            variation=plan.variation,
            task_name=plan.task_name,
            verify_instance=False,
        )
        and isinstance(reconstruction, dict)
        and reconstruction.get("schema")
        == "dynamac-rlbench-source-reconstruction-audit-v1"
        and reconstruction.get("passed") is True
        and reconstruction.get("source_waypoint_cache_state") is None
        and binding.get("task_name") == plan.task_name
        and binding.get("task_semantics_matched") is True
        and binding.get("task_tree_matched") is True
        and binding.get("task_descriptions_matched") is True
        and binding.get("robot_external_collision_pairs_matched") is True
        and binding.get("selected_source_fingerprint")
        == plan.validation.get("selected_source_fingerprint")
        and _valid_v3_fingerprint(binding.get("formal_source_fingerprint"))
        and binding.get("motion_plan_fingerprint") == plan.fingerprint()
        and row.get("motion_plan_fingerprint") == plan.fingerprint()
        and row.get("motion_plan_protocol_id") == STAGED_VALIDATED_MOTION_PROTOCOL_ID
        and evidence
        == {
            "plan_fingerprint": plan.fingerprint(),
            "source_waypoint_validated": True,
            "goal_waypoint_validated": True,
            "formal_rollout_sample_or_restore": False,
            "formal_source_bound": True,
            "formal_task_name_bound": plan.task_name,
            "formal_task_semantics_matched": True,
            "formal_task_tree_matched": True,
            "formal_deterministic_source_reconstruction_passed": True,
            "formal_task_validate_calls": 0,
            "formal_observation_refreshed_after_binding": True,
            "formal_robot_external_collision_pairs_matched": True,
            "selected_source_fingerprint": binding.get(
                "selected_source_fingerprint"
            ),
            "formal_source_fingerprint": binding.get("formal_source_fingerprint"),
        }
    ):
        return False
    return True

    # Legacy V3.3 binding checks below are unreachable for V3.4 plans.
    root = binding.get("root") if isinstance(binding, dict) else None
    low_dim = binding.get("low_dim_state") if isinstance(binding, dict) else None
    task_tree_match = (
        binding.get("task_tree_match") if isinstance(binding, dict) else None
    )
    cross_initialization = (
        binding.get("cross_initialization_reproducibility")
        if isinstance(binding, dict)
        else None
    )
    evidence = row.get("motion_plan_evidence")
    source_object_count = plan.validation.get("source_task_tree_object_count")
    source_task_tree = plan.validation.get("source_task_tree_relative_state")
    source_identities = (
        [(item["name"], item["type"]) for item in source_task_tree]
        if _valid_v3_task_tree_state(source_task_tree)
        else []
    )
    try:
        spec = get_task_spec(plan.task_name)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        isinstance(binding, dict)
        and binding.get("required") is True
        and binding.get("matched") is True
        and binding.get("formal_source_bound") is True
        and binding.get("formal_sampling_or_restore") is False
        and binding.get("fresh_task_generation_protocol_id")
        == FRESH_TASK_GENERATION_PROTOCOL_ID
        and binding.get("selected_source_fresh_task_generation")
        == plan.validation.get("selected_source_fresh_task_generation")
        and binding.get("formal_source_fresh_task_generation")
        == row.get("fresh_task_generation")
        and binding.get("task_name") == plan.task_name
        and binding.get("task_semantics_matched") is True
        and binding.get("task_tree_matched") is True
        and binding.get("task_tree_state_schema") == TASK_TREE_STATE_SCHEMA
        and isinstance(source_object_count, int)
        and not isinstance(source_object_count, bool)
        and source_object_count > 0
        and _valid_v3_full_task_tree_comparison(
            task_tree_match,
            expected_mode="all_objects_world",
            expected_object_count=source_object_count,
            translation_tolerance_m=CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
            rotation_tolerance_rad=CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
            joint_tolerance=CROSS_INITIALIZATION_JOINT_TOLERANCE,
            expected_identities=source_identities,
        )
        and binding.get("task_descriptions_matched") is True
        and _valid_v3_root_reproducibility(
            root,
            translation_tolerance_m=ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
            rotation_tolerance_rad=ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
        )
        and _valid_v3_low_dim_reproducibility(
            low_dim,
            expected_chunk_count=len(spec.pose_chunks),
            translation_tolerance_m=CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
            rotation_tolerance_rad=CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
            scalar_tolerance=CROSS_INITIALIZATION_SCALAR_TOLERANCE,
        )
        and binding.get("robot_external_collision_pairs_matched") is True
        and binding.get("selected_source_fingerprint")
        == plan.validation.get("selected_source_fingerprint")
        and _valid_v3_fingerprint(binding.get("formal_source_fingerprint"))
        and isinstance(cross_initialization, dict)
        and cross_initialization.get("schema")
        == CROSS_INITIALIZATION_REPRODUCIBILITY_SCHEMA
        and cross_initialization.get("comparison_class")
        == (
            "selected_staging_fresh_A_to_formal_fresh_"
            "same_seed_variation_A"
        )
        and cross_initialization.get("fresh_task_generation_protocol_id")
        == FRESH_TASK_GENERATION_PROTOCOL_ID
        and cross_initialization.get("selected_source_fresh_task_generation")
        == binding.get("selected_source_fresh_task_generation")
        and cross_initialization.get("formal_source_fresh_task_generation")
        == binding.get("formal_source_fresh_task_generation")
        and cross_initialization.get("selected_source_fingerprint")
        == binding.get("selected_source_fingerprint")
        and cross_initialization.get("formal_source_fingerprint")
        == binding.get("formal_source_fingerprint")
        and cross_initialization.get("fingerprints_compared_for_identity") is False
        and cross_initialization.get("tolerances")
        == _v3_cross_initialization_tolerances()
        and cross_initialization.get("root") == root
        and cross_initialization.get("low_dim_state") == low_dim
        and cross_initialization.get("task_tree")
        == _compact_task_tree_comparison(task_tree_match)
        and _valid_v3_compact_task_tree_comparison(
            cross_initialization.get("task_tree"),
            expected_object_count=source_object_count,
            translation_tolerance_m=CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
            rotation_tolerance_rad=CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
            joint_tolerance=CROSS_INITIALIZATION_JOINT_TOLERANCE,
        )
        and cross_initialization.get("task_semantics_matched") is True
        and cross_initialization.get("task_descriptions_matched") is True
        and cross_initialization.get("robot_external_collision_pairs_matched")
        is True
        and cross_initialization.get(
            "task_object_velocities_compared_for_identity"
        )
        is False
        and cross_initialization.get("task_object_velocities_finite") is True
        and cross_initialization.get("selected_task_object_velocity_summary")
        == plan.validation.get("selected_source_task_object_velocity_summary")
        and _valid_v3_velocity_summary(
            cross_initialization.get("selected_task_object_velocity_summary"),
            expected_object_count=source_object_count,
        )
        and _valid_v3_velocity_summary(
            cross_initialization.get("formal_task_object_velocity_summary"),
            expected_object_count=source_object_count,
        )
        and cross_initialization.get("passed") is True
        and binding.get("motion_plan_fingerprint") == plan.fingerprint()
        and row.get("motion_plan_fingerprint") == plan.fingerprint()
        and row.get("motion_plan_protocol_id")
        == STAGED_VALIDATED_MOTION_PROTOCOL_ID
        and evidence
        == {
            "plan_fingerprint": plan.fingerprint(),
            "source_waypoint_validated": True,
            "goal_waypoint_validated": True,
            "formal_rollout_sample_or_restore": False,
            "formal_source_bound": True,
            "formal_task_name_bound": plan.task_name,
            "formal_task_semantics_matched": True,
            "formal_task_tree_matched": True,
            "formal_cross_initialization_reproducibility_passed": True,
            "formal_robot_external_collision_pairs_matched": True,
            "selected_source_fingerprint": binding.get(
                "selected_source_fingerprint"
            ),
            "formal_source_fingerprint": binding.get("formal_source_fingerprint"),
        }
    )


def _valid_v3_formal_intervention_state_audit(
    event: dict[str, Any],
    *,
    expected_arms: frozenset[str],
) -> bool:
    audit = event.get("formal_intervention_state_audit")
    if not isinstance(audit, dict):
        return False
    expected_fields = {
        "schema",
        "comparison_class",
        "reference_state",
        "task_tree_state_schema",
        "task_tree",
        "task_semantics_matched",
        "condition_and_grasp_registry_identity_preserved",
        "gripper_grasp_membership_and_parentage_preserved",
        "robot_external_collision_pair_policy",
        "before_robot_external_collision_pairs",
        "after_robot_external_collision_pairs",
        "new_robot_external_collision_pairs",
        "no_new_robot_external_collision_pairs",
        "passed",
    }
    task_tree = audit.get("task_tree")
    object_count = (
        task_tree.get("expected_object_count")
        if isinstance(task_tree, dict)
        else None
    )
    before = audit.get("before_robot_external_collision_pairs")
    after = audit.get("after_robot_external_collision_pairs")
    new = audit.get("new_robot_external_collision_pairs")
    if (
        set(audit) != expected_fields
        or audit.get("schema") != FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA
        or audit.get("comparison_class")
        != (
            "same_formal_initialized_task_instance_immediate_pre_to_post_"
            "boundary_root_command"
        )
        or audit.get("reference_state")
        != "current_policy_evolved_formal_state"
        or audit.get("task_tree_state_schema") != TASK_TREE_STATE_SCHEMA
        or not isinstance(object_count, int)
        or isinstance(object_count, bool)
        or object_count < 1
        or not _valid_v3_full_task_tree_comparison(
            task_tree,
            expected_mode="boundary_root_subtree_relative_else_world",
            expected_object_count=object_count,
            translation_tolerance_m=LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
            rotation_tolerance_rad=LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
            joint_tolerance=LOW_DIM_STATE_ROUNDTRIP_ATOL,
        )
        or audit.get("task_semantics_matched") is not True
        or audit.get("condition_and_grasp_registry_identity_preserved")
        is not True
        or audit.get("gripper_grasp_membership_and_parentage_preserved")
        is not True
        or audit.get("robot_external_collision_pair_policy")
        != FORMAL_INTERVENTION_COLLISION_PAIR_POLICY
        or not _valid_v3_stable_collision_records(
            before,
            expected_arms=expected_arms,
        )
        or not _valid_v3_stable_collision_records(
            after,
            expected_arms=expected_arms,
        )
        or not _valid_v3_stable_collision_records(
            new,
            expected_arms=expected_arms,
        )
    ):
        return False
    before_keys = {
        (row["arm"], row["external_object_name"])
        for row in before
    }
    after_keys = {
        (row["arm"], row["external_object_name"])
        for row in after
    }
    expected_new = [
        {"arm": arm, "external_object_name": name}
        for arm, name in sorted(after_keys - before_keys)
    ]
    expected_no_new = not expected_new
    return bool(
        new == expected_new
        and audit.get("no_new_robot_external_collision_pairs") is expected_no_new
        and audit.get("passed") is True
    )


def _valid_v3_dynamic_protocol(run: LocalRun) -> bool:
    """Authenticate V3 staged motion, committed clock, and episode accounting."""

    if run.scenario not in {"smooth", "teleport"}:
        return False
    if (
        not _valid_v3_trigger_metadata(run)
        or not _valid_v3_final_settling(run)
        or not _valid_v3_episode_accounting(run)
        or not _valid_v3_formal_fresh_task_generations(run)
        or (
            run.task in UNIMANUAL_TASKS
            and not _valid_v2_unimanual_scene_launch(run)
        )
    ):
        return False
    plans = _load_v3_staged_plans(run)
    if plans is None:
        return False
    try:
        task_spec = get_task_spec(run.task)
    except (KeyError, TypeError, ValueError):
        return False
    expected_arms = (
        frozenset({"right_arm", "left_arm"})
        if task_spec.bimanual
        else frozenset({"arm"})
    )
    metadata = _protocol_metadata(run)
    authentication = (
        _protocol_metadata(run).get("trigger_authentication")
        if run.scenario == "local_baseline"
        else _v3_trigger_authentication(run)
    )
    if not isinstance(authentication, dict):
        return False
    trigger_step = authentication["trigger_step"]
    expected_motion = ScenarioController(
        "smooth_task_motion" if run.scenario == "smooth" else "teleport_task",
        trigger_step=trigger_step,
        total_steps=10,
        motion_plan=plans[0],
    ).protocol_metadata()
    smooth_key = (
        "smooth_motion_calls" if run.task in UNIMANUAL_TASKS
        else "smooth_interpolation_calls"
    )
    attempts_key = (
        "intervention_max_attempts"
        if run.task in UNIMANUAL_TASKS
        else "max_sampling_attempts"
    )
    motion_source_protocol = load_v3_motion_source_protocol()
    if (
        metadata.get("status")
        != "V3_PREREGISTERED_CHECKPOINT_AUTHENTICATED"
        and run.task not in UNIMANUAL_TASKS
    ):
        return False
    if (
        metadata.get("motion_protocol") != expected_motion
        or metadata.get("dynamic_episode_accounting_schema")
        != DYNAMIC_EPISODE_ACCOUNTING_SCHEMA
        or metadata.get("pre_intervention_failure_policy")
        != "retain_failure_with_null_intervention_effectiveness"
        or metadata.get("pre_intervention_success_policy")
        != "retain_success_in_planned_denominator_with_unexercised_condition"
        or metadata.get("smooth_terminal_progress_policy")
        != "strict_effective_prefix_until_episode_terminal"
        or metadata.get(attempts_key)
        != motion_source_protocol["goal_sampling_max_attempts"]
        or (
            run.scenario == "smooth" and metadata.get(smooth_key) != 10
        )
    ):
        return False
    controller = run.payload.get("controller")
    if (
        not isinstance(controller, dict)
        or controller.get("dynamic_clock_semantics")
        != "advance_only_after_policy_commit"
    ):
        return False
    rows = run.payload.get("results")
    if not isinstance(rows, list) or len(rows) != run.episodes:
        return False
    event_key = "interventions" if run.task in UNIMANUAL_TASKS else "scenario_events"
    eligible_count = 0
    preterminal_count = 0
    pretrigger_successes = 0
    intervention_count = 0
    complete_count = 0
    complete_successes = 0
    for episode, (row, plan) in enumerate(zip(rows, plans)):
        if not isinstance(row, dict) or not _valid_v3_staged_source_binding(row, plan):
            return False
        events = row.get(event_key)
        if (
            not isinstance(events, list)
            or any(not isinstance(event, dict) for event in events)
            or any(event.get("applied") is not True for event in events)
        ):
            return False
        committed = row.get("committed_policy_steps")
        if (
            not isinstance(committed, int)
            or isinstance(committed, bool)
            or committed < 0
            or row.get("dynamic_clock_steps") != committed
            or row.get("trigger_step") != trigger_step
            or not isinstance(row.get("success"), bool)
        ):
            return False
        reached = row.get("intervention_reached")
        eligible = row.get("intervention_eligible")
        preterminal = row.get("pre_intervention_terminal")
        if not all(isinstance(value, bool) for value in (reached, eligible, preterminal)):
            return False
        if not reached:
            if (
                eligible
                or not preterminal
                or committed > trigger_step
                or events
                or row.get("pre_intervention_terminal_outcome")
                != ("success" if row["success"] else "failure")
                or row.get("dynamic_condition_exercised") is not False
                or row.get("dynamic_condition_unexercised") is not True
                or row.get("intervention_effective") is not None
                or row.get("intervention_complete") is not None
            ):
                return False
            preterminal_count += 1
            pretrigger_successes += int(row["success"])
            continue
        if (
            not eligible
            or preterminal
            or committed < trigger_step
            or row.get("pre_intervention_terminal_outcome") is not None
            or row.get("dynamic_condition_exercised") is not True
            or row.get("dynamic_condition_unexercised") is not False
            or row.get("intervention_effective") is not True
            or not events
        ):
            return False
        eligible_count += 1
        intervention_count += 1
        complete = run.scenario == "teleport" or len(events) == 10
        if row.get("intervention_complete") is not complete:
            return False
        if complete:
            complete_count += 1
            complete_successes += int(row["success"])
        plan_root_motion = _root_motion_metrics(
            plan.source_pose,
            plan.goal_pose,
        )
        for index, event in enumerate(events, start=1):
            required_goal_reached = (
                True
                if run.scenario == "teleport" or index == 10
                else None
            )
            if (
                event.get("clock_domain") != "committed_policy_ticks"
                or event.get("trigger_step") != trigger_step
                or event.get("step") != trigger_step + index - 1
                or event.get("motion_protocol")
                != {
                    "protocol_id": expected_motion["protocol_id"],
                    "metadata_fingerprint": _canonical_json_fingerprint(
                        expected_motion
                    ),
                }
                or event.get("policy_observation_refreshed") is not True
                or event.get("motion_plan_reference")
                != {
                    "motion_plan_fingerprint": plan.fingerprint(),
                    "validation_fingerprint": _canonical_json_fingerprint(
                        plan.validation
                    ),
                }
                or not _v3_derived_motion_metric_matches(
                    event.get("planned_root_translation_m"),
                    plan_root_motion["planned_root_translation_m"],
                )
                or not _v3_derived_motion_metric_matches(
                    event.get("planned_root_rotation_rad"),
                    plan_root_motion["planned_root_rotation_rad"],
                )
                or not _valid_v3_formal_intervention_state_audit(
                    event,
                    expected_arms=expected_arms,
                )
                or not _valid_v4_root_application(
                    event,
                    required_goal_reached=required_goal_reached,
                )
            ):
                return False
            if run.scenario == "teleport":
                if index != 1 or event.get("kind") != "teleport_task":
                    return False
            elif (
                event.get("kind") != "smooth_task_motion"
                or event.get("smooth_call") != index
                or not math.isclose(
                    float(event.get("endpoint_fraction", math.nan)),
                    index / 10.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or event.get("complete") is not (index == 10)
                or event.get("endpoint_applied") is not (index == 10)
            ):
                return False
        if run.scenario == "teleport" and len(events) != 1:
            return False
        if run.scenario == "smooth" and not 1 <= len(events) <= 10:
            return False
        if run.scenario == "smooth" and len(events) < 10:
            final_event_step = trigger_step + len(events) - 1
            if committed not in {final_event_step, final_event_step + 1}:
                return False
    complete_rate = (
        complete_successes / float(complete_count) if complete_count else None
    )
    return (
        eligible_count + preterminal_count == run.episodes
        and metadata.get("planned_episode_denominator") == run.episodes
        and metadata.get("completed_episode_count") == run.episodes
        and metadata.get("episodes_intervention_eligible") == eligible_count
        and metadata.get("episodes_pre_intervention_terminal") == preterminal_count
        and metadata.get("episodes_dynamic_condition_unexercised")
        == preterminal_count
        and metadata.get("pre_trigger_successes") == pretrigger_successes
        and metadata.get("episodes_with_intervention") == intervention_count
        and metadata.get("episodes_with_effective_intervention")
        == intervention_count
        and metadata.get("episodes_with_complete_intervention") == complete_count
        and metadata.get("successes_in_complete_intervention_subset")
        == complete_successes
        and metadata.get("success_rate_in_complete_intervention_subset")
        == complete_rate
        and metadata.get("all_episodes_intervened")
        is (intervention_count == run.episodes)
        and metadata.get("all_interventions_effective") is True
        and metadata.get("all_eligible_interventions_effective") is True
        and metadata.get("protocol_valid") is True
    )


def _valid_v3_static_protocol(run: LocalRun) -> bool:
    if (
        not _valid_v3_trigger_metadata(run)
        or not _valid_v3_final_settling(run)
        or not _valid_v3_episode_accounting(run)
        or not _valid_v3_formal_fresh_task_generations(run)
    ):
        return False
    plans = _load_v3_staged_plans(run)
    if plans is None:
        return False
    rows = run.payload.get("results")
    if not isinstance(rows, list) or len(rows) != run.episodes:
        return False
    event_key = "interventions" if run.task in UNIMANUAL_TASKS else "scenario_events"
    for row, plan in zip(rows, plans):
        if (
            not isinstance(row, dict)
            or not _valid_v3_staged_source_binding(row, plan)
            or row.get(event_key) != []
            or row.get("trigger_step") is not None
            or row.get("intervention_eligible") is not False
            or row.get("intervention_reached") is not False
            or row.get("pre_intervention_terminal") is not False
            or row.get("dynamic_condition_exercised") is not False
            or row.get("dynamic_condition_unexercised") is not None
            or row.get("intervention_effective") is not None
            or row.get("intervention_complete") is not None
        ):
            return False
    if run.task in UNIMANUAL_TASKS and not _valid_v2_unimanual_scene_launch(run):
        return False
    return True


def _valid_v3_coordination_protocol(run: LocalRun) -> bool:
    if (
        run.scenario
        not in {"coordination_hand_left", "coordination_hand_right", "local_baseline"}
        or not _valid_v3_trigger_metadata(run)
        or not _valid_v3_final_settling(run)
        or not _valid_v3_formal_fresh_task_generations(run)
    ):
        return False
    fixed = run.payload.get("fixed_eval_set")
    if not isinstance(fixed, dict):
        return False
    try:
        eval_set, selected = fixed_coordination_sources(
            fixed.get("evaluation_set_id")
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    plans = selected["plans"]
    if fixed != {
        "evaluation_set_id": eval_set["payload"]["evaluation_set_id"],
        "manifest_sha256": eval_set["manifest_sha256"],
        "spec_sha256": eval_set["payload"]["spec"]["sha256"],
        "selected_batch_sha256": selected["sha256"],
        "selected_batch_fingerprint": selected["batch_fingerprint"],
        "formal_access": "canonical_id_read_only_no_generation",
    }:
        return False
    metadata = _protocol_metadata(run)
    authentication = (
        metadata.get("trigger_authentication")
        if run.scenario == "local_baseline"
        else _v3_trigger_authentication(run)
    )
    if not isinstance(authentication, dict):
        return False
    arm = (
        "none"
        if run.scenario == "local_baseline"
        else run.scenario.removeprefix("coordination_hand_")
    )
    trigger = None if arm == "none" else authentication["trigger_step"]
    controller = run.payload.get("controller")
    variation_count = run.payload.get("variation_count")
    variation_schedule = run.payload.get("variation_schedule")
    expected_variation_schedule = [
        episode % EXPECTED_COORDINATION_VARIATION_COUNT
        for episode in range(run.episodes)
    ]
    if (
        metadata.get("protocol_valid") is not True
        or metadata.get("perturbed_arm") != arm
        or (
            arm != "none"
            and authentication.get("profile", {}).get("perturbed_arm") != arm
        )
        or metadata.get("translation_world_m") != [0.0, 0.0, 0.03]
        or metadata.get("application")
        != "persistent offset on every predicted EE target from trigger"
        or metadata.get("trigger_policy_step") != trigger
        or metadata.get("legacy_one_third_trigger_disabled") is not True
        or not isinstance(controller, dict)
        or controller.get("coordination_trigger_clock")
        != "successfully_committed_policy_ticks"
        or not isinstance(variation_count, int)
        or isinstance(variation_count, bool)
        or variation_count != EXPECTED_COORDINATION_VARIATION_COUNT
        or not isinstance(variation_schedule, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in variation_schedule
        )
        or variation_schedule != expected_variation_schedule
    ):
        return False
    rows = run.payload.get("results")
    if not isinstance(rows, list) or len(rows) != run.episodes:
        return False
    for episode, (row, plan) in enumerate(zip(rows, plans)):
        if not isinstance(row, dict):
            return False
        binding = row.get("staged_source_binding")
        if (
            not isinstance(binding, dict)
            or binding.get("schema")
            != "dynamac-rlbench-formal-source-a-binding-v1"
            or binding.get("required") is not True
            or binding.get("matched") is not True
            or binding.get("source_seed") != plan.validation.get("source_seed")
            or binding.get("fresh_task_generation")
            != row.get("fresh_task_generation")
            or binding.get("task_validate_calls") != 0
            or binding.get("source_reconstruction", {}).get("passed") is not True
            or binding.get("plan_fingerprint") != plan.fingerprint()
            or plan.episode_seed != run.seed + episode
            or plan.variation != expected_variation_schedule[episode]
        ):
            return False
        committed = row.get("committed_policy_steps")
        perturbed = row.get("perturbed_steps")
        attempts = row.get("perturbed_attempts")
        if (
            not isinstance(committed, int)
            or isinstance(committed, bool)
            or committed < 0
            or (
                arm == "none"
                and (perturbed != 0 or attempts != 0)
            )
            or (
                arm != "none"
                and perturbed != max(0, committed - trigger)
            )
            or not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < perturbed
            or not isinstance(row.get("variation"), int)
            or isinstance(row.get("variation"), bool)
            or row.get("variation") != expected_variation_schedule[episode]
        ):
            return False
    return (
        run.payload.get("episodes_requested") == run.episodes
        and run.payload.get("episodes_completed") == run.episodes
    )


def _v3_plan_pair_fingerprints(run: LocalRun) -> tuple[str, ...] | None:
    metadata = _protocol_metadata(run)
    cache = metadata.get("staged_motion_plan_cache")
    fingerprints = cache.get("plan_fingerprints") if isinstance(cache, dict) else None
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != run.episodes
        or any(not isinstance(item, str) or not item for item in fingerprints)
    ):
        return None
    return tuple(fingerprints)


def _v3_table_i_plan_pair_valid(run: LocalRun, runs: Sequence[LocalRun]) -> bool:
    scenarios = {"static", "smooth", "teleport"}
    if run.task not in UNIMANUAL_TASKS or run.scenario not in scenarios:
        return True
    peers = [
        peer
        for peer in runs
        if peer.task == run.task
        and peer.scenario in scenarios - {run.scenario}
        and peer.seed == run.seed
        and peer.episodes == run.episodes
        and peer.horizon == run.horizon
        and peer.variation == run.variation
        and _model_identity_rank(peer) == 0
    ]
    if len(peers) != 2 or {peer.scenario for peer in peers} != scenarios - {
        run.scenario
    }:
        return False
    return (
        all(
            run.payload.get("motion_plan_batch_fingerprint")
            == peer.payload.get("motion_plan_batch_fingerprint")
            and _v3_plan_pair_fingerprints(run)
            == _v3_plan_pair_fingerprints(peer)
            and run.payload.get("fixed_eval_set") == peer.payload.get("fixed_eval_set")
            and _model_fingerprint_key(run) == _model_fingerprint_key(peer)
            for peer in peers
        )
    )


def _v3_dynamic_accounting(run: LocalRun | None) -> dict[str, Any]:
    empty = {
        "planned_episode_denominator": None,
        "trigger_reached_episodes": None,
        "complete_intervention_episodes": None,
        "incomplete_intervention_episodes": None,
        "pretrigger_terminal_episodes": None,
        "pretrigger_successes": None,
        "successes_in_complete_intervention_subset": None,
        "success_rate_in_complete_intervention_subset": None,
    }
    if run is None or run.scenario not in {"smooth", "teleport"}:
        return empty
    rows = run.payload.get("results")
    if not isinstance(rows, list):
        return empty
    reached = sum(row.get("intervention_reached") is True for row in rows if isinstance(row, dict))
    complete_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("intervention_complete") is True
    ]
    complete = len(complete_rows)
    preterminal_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("pre_intervention_terminal") is True
    ]
    successes = sum(row.get("success") is True for row in complete_rows)
    return {
        "planned_episode_denominator": run.episodes,
        "trigger_reached_episodes": reached,
        "complete_intervention_episodes": complete,
        "incomplete_intervention_episodes": reached - complete,
        "pretrigger_terminal_episodes": len(preterminal_rows),
        "pretrigger_successes": sum(row.get("success") is True for row in preterminal_rows),
        "successes_in_complete_intervention_subset": successes,
        "success_rate_in_complete_intervention_subset": (
            successes / float(complete) if complete else None
        ),
    }


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
        f"effective; {preterminal_text} episodes ended before the trigger; "
        f"protocol {motion_id})."
    )


def _status(
    cell: PaperCell,
    run: LocalRun | None,
    expected_training_config: dict[str, Any] | None = EXPECTED_LOCAL_CONFIG,
    expected_evaluation_protocol_ids: dict[str, str] | None = (
        EXPECTED_RELEASE_EVALUATION_PROTOCOL_IDS[DEFAULT_RELEASE]
    ),
    all_runs: Sequence[LocalRun] = (),
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
    uses_v3_config = expected_training_config == EXPECTED_RELEASE_CONFIGS["v3"]
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
    if uses_v3_config:
        if cell.condition == "Static":
            if not _valid_v3_static_protocol(run):
                return "invalid diagnostic"
        elif cell.condition == "Dynamic environment" or (
            cell.table == "I" and cell.condition in {"Smooth dynamics", "Teleportation"}
        ):
            if not _valid_v3_dynamic_protocol(run):
                return "invalid diagnostic"
            if cell.table == "I" and not _v3_table_i_plan_pair_valid(run, all_runs):
                return "invalid diagnostic"
        elif cell.condition == "Coordination":
            if not _valid_v3_coordination_protocol(run):
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
    selected_release = release
    if selected_release is None:
        inferred_releases = {
            candidate_release
            for run in runs
            for candidate_release, config in EXPECTED_RELEASE_CONFIGS.items()
            if isinstance(run.payload.get("model_identity"), dict)
            and run.payload["model_identity"].get("training_config") == config
        }
        # Programmatic callers that point directly at one historical result
        # directory keep working.  The CLI always supplies its explicit V3
        # default, so releases can never be mixed in a published report.
        selected_release = (
            next(iter(inferred_releases))
            if len(inferred_releases) == 1
            else DEFAULT_RELEASE
        )
    expected_training_config = _expected_training_config(selected_release)
    expected_protocol_ids = _expected_evaluation_protocol_ids(selected_release)
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
                    runs,
                ),
                "source_file": source,
                "notes": (
                    cell.unavailable_reason
                    or (cell.stopped_reason if run is None else "")
                    or cell.note
                ),
                "protocol_note": _material_protocol_note(cell, run),
                **_v3_dynamic_accounting(run),
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
    selected_release = DEFAULT_RELEASE if release is None else release
    intervention_protocol = (
        load_v3_intervention_protocol() if selected_release == "v3" else None
    )
    return {
        "schema": "dynamac-paper-comparison-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_reference": PAPER_REFERENCE,
        "selection": {
            "release": release,
            "seed": seed,
            "episodes": episodes,
            "horizon": horizon,
            "expected_model_identity": {
                "model_schema_version": EXPECTED_MODEL_SCHEMA_VERSION,
                "selection_semantics_id": (
                    EXPECTED_RELEASE_SELECTION_SEMANTICS_IDS.get(selected_release)
                ),
                "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
                "training_config": expected_training_config,
                "training_manifest_schema": (
                    TRAINING_MANIFEST_SCHEMA_V3
                    if selected_release == "v3"
                    else None
                ),
                "training_adapter_protocol": (
                    V3_ADAPTER_PROTOCOL if selected_release == "v3" else None
                ),
                "evaluation_protocol_ids": expected_protocol_ids,
                "intervention_protocol": (
                    {
                        "schema": intervention_protocol["schema"],
                        "fingerprint": intervention_protocol["fingerprint"],
                    }
                    if intervention_protocol is not None
                    else None
                ),
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

    dynamic_rows = [
        row
        for row in records
        if row.get("planned_episode_denominator") is not None
    ]
    if dynamic_rows:
        lines.extend(
            [
                "",
                "## V3 dynamic accounting",
                "",
                "The primary success rate always uses the planned episode denominator. "
                "The conditional rate below uses only episodes that completed the full "
                "intervention; pre-trigger successes remain in the primary denominator "
                "and are marked as an unexercised dynamic condition.",
                "",
                "| Task / condition | Planned | Trigger reached | Complete | Incomplete | Pre-trigger terminal (success) | Complete-subset success |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in dynamic_rows:
            conditional = row.get("success_rate_in_complete_intervention_subset")
            conditional_text = "—" if conditional is None else f"{float(conditional):.3f}"
            lines.append(
                f"| {row['task']} / {row['condition']} | "
                f"{row['planned_episode_denominator']} | "
                f"{row['trigger_reached_episodes']} | "
                f"{row['complete_intervention_episodes']} | "
                f"{row['incomplete_intervention_episodes']} | "
                f"{row['pretrigger_terminal_episodes']} "
                f"({row['pretrigger_successes']}) | {conditional_text} |"
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
    "planned_episode_denominator",
    "trigger_reached_episodes",
    "complete_intervention_episodes",
    "incomplete_intervention_episodes",
    "pretrigger_terminal_episodes",
    "pretrigger_successes",
    "successes_in_complete_intervention_subset",
    "success_rate_in_complete_intervention_subset",
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
    parser.add_argument("--seed", type=int, default=GLOBAL_EVAL_SEED_START)
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
