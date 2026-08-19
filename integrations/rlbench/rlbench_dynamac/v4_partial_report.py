"""Validate and report the six deliberately selected V4 formal cells.

This is a result-side tool.  It never opens an RLBench simulator and never
writes into an evaluation-set directory.  A missing target result remains
``NOT_RUN``.  Once a target result exists, however, its episode accounting,
V4 identity, fixed evaluation-set binding, and retained-video manifest are
all authenticated before any number is reported.

The report is intentionally partial: it does not recreate Tables I--III and
does not compute an average across cells.  Its only V4 result cells are
StoreBottle static/teleport, LiftTray static/teleport, and coordination
hand-left/hand-right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .direct_evaluate import EVALUATION_PROTOCOL_ID
from .evaluation_split import (
    EVALUATION_SET_V2_CONFIG_SPEC,
    EVALUATION_SET_V2_ID,
    load_evaluation_set_v2_spec,
)
from .evaluation_videos import (
    CAPTURE_CONFIG_SCHEMA,
    DEFAULT_SELECTION_SEED,
    SELECTION_PROTOCOL_ID,
    SELECTION_SCHEMA,
    retention_quota,
)
from .records import atomic_json, json_ready
from .store_bottle_eval_v4 import (
    V4_STORE_MODE_ORDER,
    V4_STORE_MOTION_PROTOCOL_ID,
    V4_STORE_RUNTIME_LOADER_ID,
)
from .store_bottle_semantics import (
    STORE_BOTTLE_SEMANTIC_SCHEMA,
    STORE_BOTTLE_SEMANTIC_VERSION,
)
from .store_bottle_v4 import TRAINING_IDENTITY_SCHEMA
from .v4_dynamic_protocol import (
    V4_COORDINATION_CARTESIAN_SUBSTEPS,
    V4_COORDINATION_PROTOCOL_ID,
    V4_COORDINATION_TRANSLATION_METERS,
    V4_COORDINATION_TRIGGER_STEP,
    V4_LIFT_MOTION_PROTOCOL_ID,
    V4_LIFT_RUNTIME_LOADER_ID,
)


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = INTEGRATION_ROOT / "results" / "v4"
DEFAULT_V3_RESULTS_ROOT = INTEGRATION_ROOT / "results" / "v3"
DEFAULT_JSON_OUTPUT = DEFAULT_RESULTS_ROOT / "reports" / "six_cell_partial.json"
DEFAULT_MARKDOWN_OUTPUT = (
    DEFAULT_RESULTS_ROOT / "reports" / "six_cell_partial.md"
)

REPORT_SCHEMA = "dynamac-v4-six-cell-partial-report-v1"
CANONICAL_EVAL_IDENTITY_SCHEMA = "dynamac-v4-canonical-evaluation-identity-v1"
EXPECTED_EPISODES = 200
EXPECTED_SEED = 2_608_000_000
SHA256_HEX_LENGTH = 64
MULTI_FACTOR_NOTE = (
    "Descriptive multi-factor V4-versus-V3 comparison only.  The releases "
    "differ in more than one factor (including evaluation-set identity, task/intervention "
    "semantics, IK execution, and result/video protocol), so the delta must "
    "not be attributed to any single change."
)


class PartialReportValidationError(ValueError):
    """A present result cannot be admitted into the partial report."""


@dataclass(frozen=True)
class CellSpec:
    task: str
    scenario: str
    paper_target: float
    v3_relative_path: str

    @property
    def cell_id(self) -> str:
        return f"{self.task}/{self.scenario}"


TARGET_CELLS: Tuple[CellSpec, ...] = (
    CellSpec(
        "bimanual_put_bottle_in_fridge",
        "static",
        0.82,
        "table_ii/bimanual_put_bottle_in_fridge_static_"
        "seed2608000000_n200_h1000.json",
    ),
    CellSpec(
        "bimanual_put_bottle_in_fridge",
        "teleport",
        0.82,
        "table_iii_environment/bimanual_put_bottle_in_fridge_teleport_"
        "seed2608000000_n200_h1000.json",
    ),
    CellSpec(
        "bimanual_lift_tray",
        "static",
        1.0,
        "table_ii/bimanual_lift_tray_static_"
        "seed2608000000_n200_h1000.json",
    ),
    CellSpec(
        "bimanual_lift_tray",
        "teleport",
        1.0,
        "table_iii_environment/bimanual_lift_tray_teleport_"
        "seed2608000000_n200_h1000.json",
    ),
    CellSpec(
        "bimanual_handover_item_dynamic",
        "coordination_hand_left",
        0.97,
        "table_iii_coordination/coordination_hand_left_preregistered_trigger_"
        "seed2608000000_n200_h1000.json",
    ),
    CellSpec(
        "bimanual_handover_item_dynamic",
        "coordination_hand_right",
        0.97,
        "table_iii_coordination/coordination_hand_right_preregistered_trigger_"
        "seed2608000000_n200_h1000.json",
    ),
)
TARGET_BY_ID = {cell.cell_id: cell for cell in TARGET_CELLS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_canonical_eval_identity(identity: Any) -> Dict[str, Any]:
    """Validate the current-seal snapshot used to admit result artifacts."""

    expected_fields = {
        "schema",
        "evaluation_set_id",
        "manifest_sha256",
        "manifest_fingerprint",
        "spec_sha256",
        "spec_fingerprint",
        "selected_batches",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != expected_fields
        or identity.get("schema") != CANONICAL_EVAL_IDENTITY_SCHEMA
        or identity.get("evaluation_set_id") != EVALUATION_SET_V2_ID
    ):
        raise PartialReportValidationError(
            "canonical rlbench_eval_v2 identity is invalid"
        )
    for field in (
        "manifest_sha256",
        "manifest_fingerprint",
        "spec_sha256",
        "spec_fingerprint",
    ):
        if not _is_sha256(identity.get(field)):
            raise PartialReportValidationError(
                f"canonical rlbench_eval_v2 {field} is invalid"
            )
    selected = identity.get("selected_batches")
    if not isinstance(selected, dict) or set(selected) != set(TARGET_BY_ID):
        raise PartialReportValidationError(
            "canonical rlbench_eval_v2 six-cell batch map is invalid"
        )
    for cell_id, binding in selected.items():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"sha256", "fingerprint"}
            or not _is_sha256(binding.get("sha256"))
            or not _is_sha256(binding.get("fingerprint"))
        ):
            raise PartialReportValidationError(
                f"canonical rlbench_eval_v2 batch identity is invalid: {cell_id}"
            )
    return identity


def canonical_eval_identity_from_loaded_manifest(
    loaded: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project a fully authenticated eval-set load into a result admission key."""

    payload = loaded.get("payload") if isinstance(loaded, Mapping) else None
    spec = payload.get("spec") if isinstance(payload, dict) else None
    environment = (
        payload.get("environment_plan_batches")
        if isinstance(payload, dict)
        else None
    )
    coordination = (
        payload.get("coordination_source_batch")
        if isinstance(payload, dict)
        else None
    )
    if (
        not isinstance(payload, dict)
        or payload.get("evaluation_set_id") != EVALUATION_SET_V2_ID
        or not isinstance(spec, dict)
        or not isinstance(environment, dict)
        or not isinstance(coordination, dict)
    ):
        raise PartialReportValidationError(
            "loaded canonical rlbench_eval_v2 manifest is incomplete"
        )
    selected: Dict[str, Dict[str, str]] = {}
    for cell in TARGET_CELLS:
        reference = (
            coordination
            if cell.task == "bimanual_handover_item_dynamic"
            else environment.get(cell.task)
        )
        if not isinstance(reference, dict):
            raise PartialReportValidationError(
                f"canonical batch reference is absent: {cell.cell_id}"
            )
        selected[cell.cell_id] = {
            "sha256": reference.get("sha256"),
            "fingerprint": reference.get("batch_fingerprint"),
        }
    identity = {
        "schema": CANONICAL_EVAL_IDENTITY_SCHEMA,
        "evaluation_set_id": EVALUATION_SET_V2_ID,
        "manifest_sha256": loaded.get("manifest_sha256"),
        "manifest_fingerprint": payload.get("fingerprint"),
        "spec_sha256": spec.get("sha256"),
        "spec_fingerprint": spec.get("fingerprint"),
        "selected_batches": selected,
    }
    return _validate_canonical_eval_identity(identity)


def load_canonical_eval_identity() -> Dict[str, Any]:
    """Deep-authenticate the canonical seal and return its exact result key."""

    from .eval_set import load_fixed_eval_set_manifest

    try:
        loaded = load_fixed_eval_set_manifest(
            EVALUATION_SET_V2_ID,
            full_preflight=True,
            verify_training_files=True,
        )
    except Exception as error:
        raise PartialReportValidationError(
            "canonical rlbench_eval_v2 preflight failed"
        ) from error
    return canonical_eval_identity_from_loaded_manifest(loaded)


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartialReportValidationError(
            f"{label} is not readable strict JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise PartialReportValidationError(f"{label} must be a JSON object: {path}")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PartialReportValidationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PartialReportValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise PartialReportValidationError(f"{label} must be finite and in [0, 1]")
    return result


def _same_float(actual: Any, expected: float, label: str) -> float:
    value = _rate(actual, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise PartialReportValidationError(
            f"{label} is inconsistent: expected {expected}, found {value}"
        )
    return value


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _require_regular_inside(root: Path, path: Path, label: str) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise PartialReportValidationError(f"{label} must not be a symbolic link")
    resolved = unresolved.resolve()
    if not _inside(root.resolve(), resolved):
        raise PartialReportValidationError(f"{label} escapes its declared root")
    if not resolved.is_file():
        raise PartialReportValidationError(f"{label} is not a regular file: {path}")
    return resolved


def _relative_display(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _formal_cells(evaluation_spec: Mapping[str, Any]) -> Tuple[str, ...]:
    cells = []
    for task, profile in evaluation_spec["dynamic_environment"].items():
        for scenario in profile["consumers"]:
            cells.append(f"{task}/{scenario}")
    scenario_for_consumer = {
        "none": "local_baseline",
        "hand_left": "coordination_hand_left",
        "hand_right": "coordination_hand_right",
    }
    for task, profile in evaluation_spec["coordination"].items():
        for consumer in profile["consumers"]:
            if consumer not in scenario_for_consumer:
                raise PartialReportValidationError(
                    f"unknown coordination consumer in V4 spec: {consumer}"
                )
            cells.append(f"{task}/{scenario_for_consumer[consumer]}")
    if len(cells) != len(set(cells)):
        raise PartialReportValidationError("V4 evaluation spec contains duplicate cells")
    missing_targets = set(TARGET_BY_ID) - set(cells)
    if missing_targets:
        raise PartialReportValidationError(
            f"six-cell scope is absent from the V4 spec: {sorted(missing_targets)}"
        )
    return tuple(cells)


def _default_result_candidates(results_root: Path, cell: CellSpec) -> Tuple[Path, ...]:
    task = cell.task
    if cell.scenario == "static":
        return (
            results_root
            / "table_ii"
            / f"{task}_static_seed2608000000_n200_h1000.json",
        )
    if cell.scenario == "teleport":
        return (
            results_root
            / "table_iii_environment"
            / f"{task}_teleport_seed2608000000_n200_h1000.json",
        )
    arm = cell.scenario.removeprefix("coordination_hand_")
    return (
        results_root
        / "table_iii_coordination"
        / (
            f"coordination_hand_{arm}_v4_cartesian_tick235_"
            "seed2608000000_n200_h1000.json"
        ),
        results_root
        / "table_iii_coordination"
        / (
            f"coordination_hand_{arm}_preregistered_trigger_"
            "seed2608000000_n200_h1000.json"
        ),
    )


def _candidate_identity(path: Path) -> Optional[Tuple[str, str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    task = value.get("task")
    scenario = value.get("scenario")
    if isinstance(task, str) and isinstance(scenario, str):
        return task, scenario
    return None


def _discover_result_paths(
    results_root: Path,
    overrides: Mapping[str, Path],
) -> Dict[str, Optional[Path]]:
    unknown = set(overrides) - set(TARGET_BY_ID)
    if unknown:
        raise PartialReportValidationError(
            f"--cell-result names cells outside the six-cell scope: {sorted(unknown)}"
        )
    root = Path(results_root).resolve()
    discovered = {cell.cell_id: None for cell in TARGET_CELLS}
    if not root.exists():
        for cell_id, path in overrides.items():
            if Path(path).exists():
                raise PartialReportValidationError(
                    f"result override is outside missing results root: {cell_id}"
                )
        return discovered
    if not root.is_dir():
        raise PartialReportValidationError(f"results root is not a directory: {root}")

    scanned = {cell.cell_id: [] for cell in TARGET_CELLS}
    for path in root.rglob("*.json"):
        relative = path.relative_to(root)
        if any(part in {"evaluation_videos", "reports"} for part in relative.parts):
            continue
        identity = _candidate_identity(path)
        if identity is None:
            continue
        cell_id = f"{identity[0]}/{identity[1]}"
        if cell_id in scanned:
            scanned[cell_id].append(path.resolve())

    for cell in TARGET_CELLS:
        cell_id = cell.cell_id
        override = overrides.get(cell_id)
        if override is not None:
            path = Path(override)
            if not path.exists():
                continue
            discovered[cell_id] = _require_regular_inside(
                root, path, f"{cell_id} result"
            )
            continue

        canonical = [path for path in _default_result_candidates(root, cell) if path.exists()]
        for path in canonical:
            # A canonical filename is an asserted result artifact, so malformed
            # JSON must fail rather than silently look like NOT_RUN.
            _read_object(path, f"{cell_id} result")
        matches = set(scanned[cell_id]) | {path.resolve() for path in canonical}
        if len(matches) > 1:
            raise PartialReportValidationError(
                f"multiple V4 results claim {cell_id}: "
                f"{[str(path) for path in sorted(matches)]}"
            )
        if matches:
            discovered[cell_id] = _require_regular_inside(
                root, next(iter(matches)), f"{cell_id} result"
            )
    return discovered


def _validate_episode_results(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_episodes: int = EXPECTED_EPISODES,
) -> Tuple[Sequence[Mapping[str, Any]], int, float, Counter]:
    for field in ("episodes", "episodes_requested", "episodes_completed"):
        if _integer(payload.get(field), f"{label}.{field}") != expected_episodes:
            raise PartialReportValidationError(
                f"{label}.{field} must equal {expected_episodes}"
            )
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != expected_episodes:
        raise PartialReportValidationError(
            f"{label}.results must contain exactly {expected_episodes} rows"
        )
    episodes = []
    successes = 0
    reasons = Counter()
    invalid_actions = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PartialReportValidationError(f"{label}.results[{index}] is not an object")
        episode = _integer(row.get("episode"), f"{label}.results[{index}].episode")
        success = row.get("success")
        if not isinstance(success, bool):
            raise PartialReportValidationError(
                f"{label}.results[{index}].success must be boolean"
            )
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            raise PartialReportValidationError(
                f"{label}.results[{index}].reason must be a non-empty string"
            )
        invalid = _integer(
            row.get("invalid_actions", 0),
            f"{label}.results[{index}].invalid_actions",
        )
        episodes.append(episode)
        successes += int(success)
        reasons[reason] += 1
        invalid_actions += invalid
    if sorted(episodes) != list(range(expected_episodes)):
        raise PartialReportValidationError(
            f"{label}.results episode IDs must be exactly 0..{expected_episodes - 1}"
        )
    if _integer(payload.get("successes"), f"{label}.successes") != successes:
        raise PartialReportValidationError(f"{label}.successes disagrees with results")
    success_rate = successes / float(expected_episodes)
    _same_float(payload.get("success_rate"), success_rate, f"{label}.success_rate")

    accounting = payload.get("episode_accounting")
    if accounting is not None:
        if not isinstance(accounting, dict):
            raise PartialReportValidationError(f"{label}.episode_accounting is not an object")
        expected_accounting = {
            "planned_episode_denominator": expected_episodes,
            "completed_episode_count": expected_episodes,
            "successes_in_planned_denominator": successes,
        }
        for field, expected in expected_accounting.items():
            if field in accounting and accounting[field] != expected:
                raise PartialReportValidationError(
                    f"{label}.episode_accounting.{field} is inconsistent"
                )
        if "success_rate_all_planned_episodes" in accounting:
            _same_float(
                accounting["success_rate_all_planned_episodes"],
                success_rate,
                f"{label}.episode_accounting.success_rate_all_planned_episodes",
            )
    if "invalid_actions" in payload and payload["invalid_actions"] != invalid_actions:
        raise PartialReportValidationError(
            f"{label}.invalid_actions disagrees with episode rows"
        )
    return rows, successes, success_rate, reasons


def _validate_fixed_eval_identity(
    payload: Mapping[str, Any],
    label: str,
    *,
    cell: CellSpec,
    canonical_eval_identity: Mapping[str, Any],
) -> None:
    """Require the result to name the exact current seal and selected batch."""

    canonical = _validate_canonical_eval_identity(dict(canonical_eval_identity))
    selected = canonical["selected_batches"][cell.cell_id]
    expected = {
        "evaluation_set_id": EVALUATION_SET_V2_ID,
        "manifest_sha256": canonical["manifest_sha256"],
        "spec_sha256": canonical["spec_sha256"],
        "selected_batch_sha256": selected["sha256"],
        "selected_batch_fingerprint": selected["fingerprint"],
        "formal_access": "canonical_id_read_only_no_generation",
    }
    identity = payload.get("fixed_eval_set")
    if not isinstance(identity, dict):
        raise PartialReportValidationError(f"{label}.fixed_eval_set is required")
    if set(identity) != set(expected):
        raise PartialReportValidationError(
            f"{label}.fixed_eval_set fields do not match the canonical schema"
        )
    for field, value in expected.items():
        if identity.get(field) != value:
            raise PartialReportValidationError(
                f"{label}.fixed_eval_set.{field} does not match the current "
                f"canonical {EVALUATION_SET_V2_ID}"
            )


def _validate_controller(payload: Mapping[str, Any], label: str) -> None:
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise PartialReportValidationError(f"{label}.controller is required")
    expected = {
        "primary_ik": "jacobian",
        "fallback_ik": "sampling",
        "sampling_ignore_collisions": False,
        "sampling_candidate_selection": "nearest_current_joint_l2",
        "ik_candidate_validation": "shape_finite_noncyclic_joint_limits",
    }
    for field, value in expected.items():
        if controller.get(field) != value:
            raise PartialReportValidationError(
                f"{label}.controller.{field} is not the V4 setting"
            )
    retry = controller.get("primary_action_retry")
    if (
        not isinstance(retry, dict)
        or retry.get("max_primary_action_attempts_per_policy_tick") != 3
    ):
        raise PartialReportValidationError(
            f"{label} does not preserve the three-attempt retry budget"
        )


def _validate_store_model_identity(
    payload: Mapping[str, Any], cell: CellSpec
) -> str:
    """Authenticate the Store-only retrained V4 policy identity."""

    model = payload.get("model_identity")
    if not isinstance(model, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id}.model_identity is required for StoreBottle V4"
        )
    identity = model.get("training_identity")
    if (
        model.get("training_manifest_schema") != "dynamac-direct-training-v4"
        or model.get("manifest_authenticated") is not True
        or not isinstance(identity, dict)
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} is not served by an authenticated V4 StoreBottle model"
        )
    fingerprint = identity.get("fingerprint")
    unsigned = {key: value for key, value in identity.items() if key != "fingerprint"}
    policy_spec = identity.get("policy_spec")
    collection = identity.get("collection")
    policy_config = identity.get("policy_config")
    if (
        identity.get("schema") != TRAINING_IDENTITY_SCHEMA
        or not _is_sha256(fingerprint)
        or fingerprint != _canonical_sha256(unsigned)
        or not isinstance(policy_spec, dict)
        or policy_spec.get("task") != cell.task
        or policy_spec.get("semantic_schema") != STORE_BOTTLE_SEMANTIC_SCHEMA
        or policy_spec.get("semantic_version") != STORE_BOTTLE_SEMANTIC_VERSION
        or not _is_sha256(policy_spec.get("semantic_fingerprint"))
        or policy_spec.get("frame_names") != ["bottle", "fridge"]
        or policy_spec.get("bimanual") is not True
        or not isinstance(collection, dict)
        or collection.get("demonstrations") != 5
        or collection.get("all_success_verified") is not True
        or not _is_sha256(collection.get("manifest_sha256"))
        or not _is_sha256(collection.get("manifest_fingerprint"))
        or not isinstance(policy_config, dict)
        or not _is_sha256(policy_config.get("sha256"))
        or identity.get("evaluation_artifacts_included") is not False
        or identity.get("tasks_trained") != [cell.task]
        or identity.get("other_tasks_trained") is not False
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has an invalid V4 StoreBottle training identity"
        )
    seeds = collection.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or any(EXPECTED_SEED <= seed < EXPECTED_SEED + EXPECTED_EPISODES for seed in seeds)
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} StoreBottle training/evaluation seed identity overlaps"
        )
    return fingerprint


def _validate_direct_task_protocol(payload: Mapping[str, Any], cell: CellSpec) -> None:
    scenario = payload.get("scenario_protocol")
    if not isinstance(scenario, dict):
        raise PartialReportValidationError(f"{cell.cell_id}.scenario_protocol is required")
    motion = scenario.get("motion_protocol")
    cache = scenario.get("staged_motion_plan_cache")
    if not isinstance(motion, dict) or not isinstance(cache, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id} lacks its V4 task-scoped motion identity"
        )
    envelope = cache.get("task_scoped_envelope")
    if not isinstance(envelope, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id} lacks its task-scoped plan envelope"
        )
    if cell.task == "bimanual_lift_tray":
        expected_status = (
            "STATIC_REFERENCE"
            if cell.scenario == "static"
            else "V4_LIFT_SKILL0_TICK35_TASK_SCOPED"
        )
        expected_protocol = V4_LIFT_MOTION_PROTOCOL_ID
        expected_loader = V4_LIFT_RUNTIME_LOADER_ID
    elif cell.task == "bimanual_put_bottle_in_fridge":
        expected_status = (
            "STATIC_REFERENCE"
            if cell.scenario == "static"
            else "V4_STORE_INDEPENDENT_ENTITY_TRIGGERS_TASK_SCOPED"
        )
        expected_protocol = V4_STORE_MOTION_PROTOCOL_ID
        expected_loader = V4_STORE_RUNTIME_LOADER_ID
    else:
        raise PartialReportValidationError(f"unsupported direct V4 task: {cell.task}")
    if (
        scenario.get("status") != expected_status
        or motion.get("protocol_id") != expected_protocol
        or cache.get("runtime_protocol_id") != expected_protocol
        or envelope.get("runtime_loader") != expected_loader
        or not _is_sha256(envelope.get("task_identity_fingerprint"))
        or not _is_sha256(envelope.get("batch_fingerprint"))
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has the wrong task-scoped V4 protocol identity"
        )


def _validate_coordination_protocol(payload: Mapping[str, Any], cell: CellSpec) -> None:
    if payload.get("schema") != "dynamac-table-iii-coordination-local-v4":
        raise PartialReportValidationError(f"{cell.cell_id} has the wrong result schema")
    protocol = payload.get("coordination_protocol")
    if not isinstance(protocol, dict):
        raise PartialReportValidationError(f"{cell.cell_id}.coordination_protocol is required")
    expected_arm = cell.scenario.removeprefix("coordination_hand_")
    if (
        protocol.get("protocol_id") != V4_COORDINATION_PROTOCOL_ID
        or protocol.get("perturbed_arm") != expected_arm
        or protocol.get("trigger_policy_step") != V4_COORDINATION_TRIGGER_STEP
        or protocol.get("translation_world_m")
        != list(V4_COORDINATION_TRANSLATION_METERS)
        or protocol.get("cartesian_substeps") != V4_COORDINATION_CARTESIAN_SUBSTEPS
        or protocol.get("persistent_policy_target_offset") is not False
        or protocol.get("policy_clock_advances_during_intervention") is not False
        or protocol.get("protocol_valid") is not True
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has the wrong one-shot V4 coordination protocol"
        )


def _artifact_path(cell_dir: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PartialReportValidationError(f"{label} must be a relative path")
    path = Path(relative)
    if path.is_absolute():
        raise PartialReportValidationError(f"{label} must be relative to the video cell")
    unresolved = cell_dir / path
    if unresolved.is_symlink():
        raise PartialReportValidationError(f"{label} must not be a symbolic link")
    resolved = unresolved.resolve()
    if not _inside(cell_dir, resolved):
        raise PartialReportValidationError(f"{label} escapes the video cell")
    return resolved


def _validate_video_manifest(
    payload: Mapping[str, Any],
    *,
    cell: CellSpec,
    rows: Sequence[Mapping[str, Any]],
    successes: int,
    success_rate: float,
    results_root: Path,
) -> Dict[str, Any]:
    capture = payload.get("evaluation_video_capture")
    if not isinstance(capture, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id} lacks required V4 episode-video evidence"
        )
    if (
        capture.get("release") != "v4"
        or capture.get("cell_key") != cell.cell_id
        or capture.get("episodes_recorded") != EXPECTED_EPISODES
        or capture.get("formal_result_committed_after_video_selection") is not True
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has inconsistent V4 video capture metadata"
        )
    _same_float(
        capture.get("paper_success_rate"),
        cell.paper_target,
        f"{cell.cell_id}.evaluation_video_capture.paper_success_rate",
    )
    capture_config = capture.get("capture_config")
    if (
        not isinstance(capture_config, dict)
        or capture_config.get("schema") != CAPTURE_CONFIG_SCHEMA
        or capture_config.get("camera") != "front"
        or capture_config.get("capture_granularity")
        != "returned_high_level_observations"
        or capture_config.get("streamed_without_frame_buffer") is not True
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has the wrong lightweight capture protocol"
        )

    root = Path(results_root).resolve()
    cell_dir_raw = capture.get("cell_dir")
    if not isinstance(cell_dir_raw, str) or not cell_dir_raw:
        raise PartialReportValidationError(f"{cell.cell_id}.cell_dir is missing")
    cell_dir_path = Path(cell_dir_raw)
    if not cell_dir_path.is_absolute():
        cell_dir_path = root / cell_dir_path
    if cell_dir_path.is_symlink():
        raise PartialReportValidationError(
            f"{cell.cell_id} video cell must not be a symbolic link"
        )
    cell_dir = cell_dir_path.resolve()
    if not _inside(root, cell_dir) or not cell_dir.is_dir():
        raise PartialReportValidationError(
            f"{cell.cell_id} video cell is not a real directory inside results/v4"
        )

    audit = capture.get("selection_manifest")
    if not isinstance(audit, dict):
        raise PartialReportValidationError(f"{cell.cell_id} video selection audit is missing")
    path_raw = audit.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        raise PartialReportValidationError(f"{cell.cell_id} selection path is missing")
    manifest_path = Path(path_raw)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = _require_regular_inside(
        root, manifest_path, f"{cell.cell_id} video selection manifest"
    )
    if manifest_path.parent != cell_dir:
        raise PartialReportValidationError(
            f"{cell.cell_id} selection manifest is outside its declared video cell"
        )
    if not _is_sha256(audit.get("sha256")) or _sha256(manifest_path) != audit["sha256"]:
        raise PartialReportValidationError(
            f"{cell.cell_id} video selection manifest hash mismatch"
        )

    manifest = _read_object(manifest_path, f"{cell.cell_id} video selection manifest")
    if (
        manifest.get("schema") != SELECTION_SCHEMA
        or audit.get("schema") != SELECTION_SCHEMA
        or manifest.get("cell_key") != cell.cell_id
        or manifest.get("all_episodes_recorded_before_selection") is not True
        or manifest.get("unselected_artifacts_deleted") is not True
        or audit.get("all_episodes_recorded_before_selection") is not True
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has an invalid video selection manifest header"
        )
    cell_result = manifest.get("cell_result")
    if not isinstance(cell_result, dict):
        raise PartialReportValidationError(f"{cell.cell_id} video cell_result is missing")
    if (
        cell_result.get("successes") != successes
        or cell_result.get("episodes") != EXPECTED_EPISODES
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} video aggregate disagrees with formal results"
        )
    _same_float(
        cell_result.get("success_rate"),
        success_rate,
        f"{cell.cell_id}.video.success_rate",
    )
    _same_float(
        cell_result.get("paper_success_rate"),
        cell.paper_target,
        f"{cell.cell_id}.video.paper_success_rate",
    )

    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("protocol_id") != SELECTION_PROTOCOL_ID:
        raise PartialReportValidationError(
            f"{cell.cell_id} has the wrong video selection protocol"
        )
    quota = retention_quota(success_rate, cell.paper_target)
    expected_available = {
        "successes": successes,
        "failures": EXPECTED_EPISODES - successes,
    }
    expected_requested = {
        "successes": quota.successes,
        "failures": quota.failures,
    }
    expected_retained = {
        "successes": min(successes, quota.successes),
        "failures": min(EXPECTED_EPISODES - successes, quota.failures),
    }
    if (
        selection.get("seed") != DEFAULT_SELECTION_SEED
        or selection.get("tier") != quota.tier
        or selection.get("available") != expected_available
        or selection.get("requested") != expected_requested
        or selection.get("retained") != expected_retained
        or selection.get("paper_close_enough_for_zero")
        is not quota.paper_close_enough_for_zero
        or audit.get("selection") != selection
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} video retention quota/audit is inconsistent"
        )

    selected = manifest.get("selected")
    deleted = manifest.get("deleted")
    if not isinstance(selected, list) or not isinstance(deleted, list):
        raise PartialReportValidationError(
            f"{cell.cell_id} video inventory must have selected/deleted lists"
        )
    result_success = {row["episode"]: row["success"] for row in rows}
    inventory = []
    retained_paths = []
    retained_counts = Counter()
    for retained, values in ((True, selected), (False, deleted)):
        for index, row in enumerate(values):
            label = f"{cell.cell_id}.video.{('selected' if retained else 'deleted')}[{index}]"
            if not isinstance(row, dict):
                raise PartialReportValidationError(f"{label} is not an object")
            episode = _integer(row.get("episode"), f"{label}.episode")
            if row.get("episode_seed") != EXPECTED_SEED + episode:
                raise PartialReportValidationError(
                    f"{label}.episode_seed disagrees with the fixed seed namespace"
                )
            expected_outcome = "success" if result_success.get(episode) is True else "failure"
            if episode not in result_success or row.get("outcome") != expected_outcome:
                raise PartialReportValidationError(
                    f"{label} outcome disagrees with the formal episode"
                )
            video = _artifact_path(cell_dir, row.get("video"), f"{label}.video")
            companions = row.get("companions")
            if not isinstance(companions, list):
                raise PartialReportValidationError(f"{label}.companions must be a list")
            companion_paths = [
                _artifact_path(cell_dir, value, f"{label}.companions")
                for value in companions
            ]
            if retained:
                if video.is_symlink() or not video.is_file():
                    raise PartialReportValidationError(f"{label}.video is not retained")
                if (
                    not _is_sha256(row.get("video_sha256"))
                    or _sha256(video) != row["video_sha256"]
                    or row.get("video_bytes") != video.stat().st_size
                ):
                    raise PartialReportValidationError(f"{label}.video hash/size mismatch")
                for companion in companion_paths:
                    if companion.is_symlink() or not companion.is_file():
                        raise PartialReportValidationError(
                            f"{label} retained companion is missing"
                        )
                retained_counts[expected_outcome] += 1
                retained_paths.append(video.relative_to(root).as_posix())
            elif video.exists() or any(path.exists() for path in companion_paths):
                raise PartialReportValidationError(
                    f"{label} claims deletion but an artifact still exists"
                )
            inventory.append(episode)
    if sorted(inventory) != list(range(EXPECTED_EPISODES)):
        raise PartialReportValidationError(
            f"{cell.cell_id} video inventory must cover episodes 0..199 exactly once"
        )
    if (
        retained_counts["success"] != expected_retained["successes"]
        or retained_counts["failure"] != expected_retained["failures"]
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} retained video counts disagree with selection"
        )
    selected_episodes = [row["episode"] for row in selected]
    if audit.get("selected_episodes") != selected_episodes:
        raise PartialReportValidationError(
            f"{cell.cell_id} selected episode audit disagrees with manifest"
        )
    return {
        "selection_manifest_path": manifest_path.relative_to(root).as_posix(),
        "selection_manifest_sha256": audit["sha256"],
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        "retention_tier": quota.tier,
        "retained": expected_retained,
        "retained_video_paths": retained_paths,
    }


def _json_statistics(value: Any, label: str) -> Any:
    """Admit JSON statistics while rejecting NaN and opaque objects."""

    if isinstance(value, dict):
        return {str(key): _json_statistics(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_json_statistics(item, f"{label}[]") for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise PartialReportValidationError(f"{label} contains a non-JSON or non-finite value")


def _store_subgroups(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Read evolving Store subgroup statistics without weakening run identity."""

    source_name = None
    source = None
    for key in (
        "store_mode_subgroups",
        "store_subgroups",
        "subgroup_results",
        "subgroups",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            source_name, source = key, value
            break
    if source is None:
        return None
    normalized = {}
    aliases = {
        "planned": ("planned", "episodes_planned", "planned_episodes"),
        "completed": (
            "completed",
            "episodes_completed",
            "completed_episodes",
            "episodes",
        ),
        "successes": ("successes",),
        "success_rate": ("success_rate", "rate"),
    }
    for name, row in source.items():
        if not isinstance(name, str) or not isinstance(row, dict):
            raise PartialReportValidationError("Store subgroup rows must be named objects")
        item = {"reported": _json_statistics(row, f"Store subgroup {name}")}
        for canonical, candidates in aliases.items():
            for candidate in candidates:
                if candidate in row:
                    item[canonical] = row[candidate]
                    break
        for field in ("planned", "completed", "successes"):
            if field in item:
                item[field] = _integer(item[field], f"Store subgroup {name}.{field}")
        if "success_rate" in item:
            item["success_rate"] = _rate(
                item["success_rate"], f"Store subgroup {name}.success_rate"
            )
        if all(field in item for field in ("completed", "successes", "success_rate")):
            if item["successes"] > item["completed"]:
                raise PartialReportValidationError(
                    f"Store subgroup {name} has more successes than completed episodes"
                )
            expected = (
                item["successes"] / float(item["completed"])
                if item["completed"]
                else 0.0
            )
            if not math.isclose(
                item["success_rate"], expected, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise PartialReportValidationError(
                    f"Store subgroup {name} success rate is inconsistent"
                )
        normalized[name] = item
    ordered = {
        name: normalized[name]
        for name in V4_STORE_MODE_ORDER
        if name in normalized
    }
    ordered.update(
        (name, normalized[name]) for name in sorted(normalized) if name not in ordered
    )
    return {"source_field": source_name, "groups": ordered}


def _validate_v4_result(
    path: Path,
    cell: CellSpec,
    results_root: Path,
    *,
    canonical_eval_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    label = cell.cell_id
    payload = _read_object(path, f"{label} result")
    if (
        payload.get("release") != "v4"
        or payload.get("task") != cell.task
        or payload.get("scenario") != cell.scenario
        or payload.get("seed") != EXPECTED_SEED
    ):
        raise PartialReportValidationError(f"{label} has the wrong V4 cell identity")
    if payload.get("evaluation_protocol_id") != EVALUATION_PROTOCOL_ID:
        raise PartialReportValidationError(
            f"{label} does not use the collision-aware V4 evaluation protocol"
        )
    rows, successes, success_rate, reasons = _validate_episode_results(
        payload, label=label
    )
    _validate_fixed_eval_identity(
        payload,
        label,
        cell=cell,
        canonical_eval_identity=canonical_eval_identity,
    )
    _validate_controller(payload, label)
    if cell.task == "bimanual_handover_item_dynamic":
        _validate_coordination_protocol(payload, cell)
        store_training_identity_fingerprint = None
    else:
        _validate_direct_task_protocol(payload, cell)
        store_training_identity_fingerprint = (
            _validate_store_model_identity(payload, cell)
            if cell.task == "bimanual_put_bottle_in_fridge"
            else None
        )
    diagnostics = payload.get("ik_execution_diagnostics")
    if not isinstance(diagnostics, dict):
        raise PartialReportValidationError(f"{label}.ik_execution_diagnostics is required")
    diagnostics = _json_statistics(diagnostics, f"{label}.ik_execution_diagnostics")
    videos = _validate_video_manifest(
        payload,
        cell=cell,
        rows=rows,
        successes=successes,
        success_rate=success_rate,
        results_root=results_root,
    )
    invalid_actions = sum(int(row.get("invalid_actions", 0)) for row in rows)
    subgroups = (
        _store_subgroups(payload)
        if cell.task == "bimanual_put_bottle_in_fridge"
        else None
    )
    return {
        "status": "COMPLETED_VALIDATED",
        "result_path": _relative_display(path, results_root),
        "result_sha256": _sha256(path),
        "successes": successes,
        "episodes": EXPECTED_EPISODES,
        "success_rate": success_rate,
        "paper_target": cell.paper_target,
        "gap_to_paper_percentage_points": (success_rate - cell.paper_target) * 100.0,
        "validated_identity": {
            "release": "v4",
            "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
            "evaluation_set_id": EVALUATION_SET_V2_ID,
            "store_training_identity_fingerprint": (
                store_training_identity_fingerprint
            ),
        },
        "terminal_reason_counts": [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                reasons.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "invalid_actions": invalid_actions,
        "ik_execution_diagnostics": diagnostics,
        "store_subgroups": subgroups,
        "videos": videos,
    }


def _v3_comparison(
    v3_results_root: Path,
    cell: CellSpec,
    *,
    v4_rate: Optional[float],
) -> Dict[str, Any]:
    path = Path(v3_results_root) / cell.v3_relative_path
    base = {
        "causal_attribution": False,
        "interpretation": MULTI_FACTOR_NOTE,
    }
    if not path.is_file():
        return {
            **base,
            "status": "NOT_AVAILABLE",
            "result_path": cell.v3_relative_path,
            "successes": None,
            "episodes": None,
            "success_rate": None,
            "v4_minus_v3_percentage_points": None,
        }
    payload = _read_object(path, f"V3 comparison {cell.cell_id}")
    if payload.get("task") != cell.task or payload.get("scenario") != cell.scenario:
        raise PartialReportValidationError(
            f"V3 comparison artifact has the wrong identity: {path}"
        )
    _, successes, rate, _ = _validate_episode_results(
        payload, label=f"V3 comparison {cell.cell_id}"
    )
    return {
        **base,
        "status": "AVAILABLE",
        "result_path": _relative_display(path, v3_results_root),
        "successes": successes,
        "episodes": EXPECTED_EPISODES,
        "success_rate": rate,
        "v4_minus_v3_percentage_points": (
            None if v4_rate is None else (v4_rate - rate) * 100.0
        ),
    }


def build_report(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    v3_results_root: Path = DEFAULT_V3_RESULTS_ROOT,
    evaluation_spec_path: Path = EVALUATION_SET_V2_CONFIG_SPEC,
    cell_results: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    """Build a validated in-memory report without writing any file."""

    spec = load_evaluation_set_v2_spec(evaluation_spec_path)
    formal_cells = _formal_cells(spec)
    paths = _discover_result_paths(Path(results_root), cell_results or {})
    canonical_eval_identity = (
        load_canonical_eval_identity()
        if any(path is not None for path in paths.values())
        else None
    )
    cells = []
    completed = 0
    for cell in TARGET_CELLS:
        path = paths[cell.cell_id]
        if path is None:
            value = {
                "status": "NOT_RUN",
                "result_path": None,
                "successes": None,
                "episodes": None,
                "success_rate": None,
                "paper_target": cell.paper_target,
                "gap_to_paper_percentage_points": None,
                "validated_identity": None,
                "terminal_reason_counts": [],
                "invalid_actions": None,
                "ik_execution_diagnostics": None,
                "store_subgroups": None,
                "videos": None,
            }
        else:
            if canonical_eval_identity is None:  # pragma: no cover - defensive
                raise PartialReportValidationError(
                    "present V4 result lacks a canonical evaluation identity"
                )
            value = _validate_v4_result(
                path,
                cell,
                Path(results_root),
                canonical_eval_identity=canonical_eval_identity,
            )
            completed += 1
        value = {
            "cell_id": cell.cell_id,
            "task": cell.task,
            "scenario": cell.scenario,
            **value,
        }
        value["v3_comparison"] = _v3_comparison(
            Path(v3_results_root),
            cell,
            v4_rate=value["success_rate"],
        )
        cells.append(value)

    target_status = {row["cell_id"]: row["status"] for row in cells}
    formal_scope = []
    for cell_id in formal_cells:
        target = cell_id in TARGET_BY_ID
        formal_scope.append(
            {
                "cell_id": cell_id,
                "in_six_cell_scope": target,
                "status": target_status.get(cell_id, "NOT_RUN"),
                "reason": (
                    None
                    if target_status.get(cell_id) == "COMPLETED_VALIDATED"
                    else (
                        "target result artifact is absent"
                        if target
                        else "outside this six-cell partial run"
                    )
                ),
            }
        )
    return json_ready(
        {
            "schema": REPORT_SCHEMA,
            "release": "v4",
            "report_kind": "six_cell_partial_validation",
            "evaluation_set_id": EVALUATION_SET_V2_ID,
            "evaluation_spec_fingerprint": spec["fingerprint"],
            "results_root": str(Path(results_root).resolve()),
            "scope": {
                "target_cell_count": len(TARGET_CELLS),
                "completed_validated_cell_count": completed,
                "not_run_target_cell_count": len(TARGET_CELLS) - completed,
                "full_tables_generated": False,
                "cross_cell_average_computed": False,
                "evaluation_set_modified": False,
            },
            "cells": cells,
            "formal_cell_status": formal_scope,
            "cross_cell_aggregate": {
                "computed": False,
                "value": None,
                "reason": "a partial six-cell report must not imply a full-release average",
            },
            "v3_comparison_caveat": MULTI_FACTOR_NOTE,
        }
    )


def _markdown_rate(value: Optional[float]) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _markdown_pp(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:+.1f} pp"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the partial report without constructing the paper's full tables."""

    lines = [
        "# DynaMAC V4 six-cell partial report",
        "",
        (
            "This report covers only StoreBottle static/teleport, LiftTray "
            "static/teleport, and coordination hand-left/hand-right. Missing "
            "results remain **NOT_RUN**. It is not Tables I–III and no "
            "cross-cell average is computed."
        ),
        "",
        "| Cell | Status | Successes | Rate | Paper target | Gap | V3 rate | V4−V3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        comparison = cell["v3_comparison"]
        successes = (
            "—"
            if cell["successes"] is None
            else f"{cell['successes']}/{cell['episodes']}"
        )
        lines.append(
            "| {cell} | {status} | {successes} | {rate} | {paper} | {gap} | "
            "{v3} | {delta} |".format(
                cell=_markdown_escape(cell["cell_id"]),
                status=cell["status"],
                successes=successes,
                rate=_markdown_rate(cell["success_rate"]),
                paper=_markdown_rate(cell["paper_target"]),
                gap=_markdown_pp(cell["gap_to_paper_percentage_points"]),
                v3=_markdown_rate(comparison["success_rate"]),
                delta=_markdown_pp(comparison["v4_minus_v3_percentage_points"]),
            )
        )

    lines.extend(["", f"> {report['v3_comparison_caveat']}", ""])
    for cell in report["cells"]:
        lines.extend([f"## {_markdown_escape(cell['cell_id'])}", ""])
        if cell["status"] == "NOT_RUN":
            lines.extend(["Status: **NOT_RUN** — no V4 result artifact was found.", ""])
            continue
        reason_text = ", ".join(
            f"{row['reason']}={row['count']}"
            for row in cell["terminal_reason_counts"]
        )
        lines.extend(
            [
                f"- Result: `{_markdown_escape(cell['result_path'])}`",
                f"- Terminal reasons: {reason_text}",
                f"- Invalid actions: {cell['invalid_actions']}",
                "- IK diagnostics: `" + json.dumps(
                    cell["ik_execution_diagnostics"],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "`",
            ]
        )
        if cell["store_subgroups"] is not None:
            lines.append(
                "- Store subgroups: `"
                + json.dumps(
                    cell["store_subgroups"],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "`"
            )
        videos = cell["videos"]
        lines.append(
            "- Retained videos: "
            f"success={videos['retained']['successes']}, "
            f"failure={videos['retained']['failures']}; manifest "
            f"`{_markdown_escape(videos['selection_manifest_path'])}`"
        )
        if videos["retained_video_paths"]:
            lines.append("- Retained paths: " + ", ".join(
                f"`{_markdown_escape(path)}`" for path in videos["retained_video_paths"]
            ))
        else:
            lines.append("- Retained paths: none (the declared retention tier selected zero).")
        lines.append("")

    lines.extend(
        [
            "## Formal V4 cell status",
            "",
            "Cells outside the six-cell scope are reported as NOT_RUN here; this state is "
            "kept in the result-side report and is never written into the sealed evaluation set.",
            "",
            "| Formal cell | Six-cell scope | Status | Reason |",
            "|---|---:|---:|---|",
        ]
    )
    for row in report["formal_cell_status"]:
        lines.append(
            "| {cell} | {scope} | {status} | {reason} |".format(
                cell=_markdown_escape(row["cell_id"]),
                scope="yes" if row["in_six_cell_scope"] else "no",
                status=row["status"],
                reason=_markdown_escape(row["reason"] or "—"),
            )
        )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent), text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _forbid_evaluation_set_output(path: Path) -> None:
    root = (INTEGRATION_ROOT / "evaluation_sets").resolve()
    candidate = Path(path).resolve()
    if _inside(root, candidate):
        raise PartialReportValidationError(
            "partial reports are result state and must never be written into evaluation_sets"
        )


def write_report(
    report: Mapping[str, Any],
    *,
    json_output: Path,
    markdown_output: Path,
) -> None:
    """Atomically write both result-side report formats."""

    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    if json_path.resolve() == markdown_path.resolve():
        raise PartialReportValidationError("JSON and Markdown outputs must differ")
    _forbid_evaluation_set_output(json_path)
    _forbid_evaluation_set_output(markdown_path)
    markdown = render_markdown(report)
    atomic_json(json_path, report)
    _atomic_text(markdown_path, markdown)


def _parse_cell_results(values: Iterable[str]) -> Dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise PartialReportValidationError(
                "--cell-result must use CELL_ID=/path/to/result.json"
            )
        cell_id, path = value.split("=", 1)
        if not cell_id or not path or cell_id in parsed:
            raise PartialReportValidationError(
                "--cell-result must name each non-empty cell exactly once"
            )
        parsed[cell_id] = Path(path)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--v3-results-root", type=Path, default=DEFAULT_V3_RESULTS_ROOT
    )
    parser.add_argument(
        "--evaluation-spec", type=Path, default=EVALUATION_SET_V2_CONFIG_SPEC
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT
    )
    parser.add_argument(
        "--cell-result",
        action="append",
        default=[],
        metavar="CELL_ID=PATH",
        help=(
            "Override discovery for one of the six cells. A missing override "
            "path remains NOT_RUN; a present file is strictly validated."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = _parse_cell_results(args.cell_result)
    report = build_report(
        results_root=args.results_root,
        v3_results_root=args.v3_results_root,
        evaluation_spec_path=args.evaluation_spec,
        cell_results=overrides,
    )
    write_report(
        report,
        json_output=args.output_json,
        markdown_output=args.output_markdown,
    )
    print(
        "wrote {json_path} and {markdown_path} "
        "({completed}/6 V4 cells validated)".format(
            json_path=args.output_json,
            markdown_path=args.output_markdown,
            completed=report["scope"]["completed_validated_cell_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_EVAL_IDENTITY_SCHEMA",
    "DEFAULT_JSON_OUTPUT",
    "DEFAULT_MARKDOWN_OUTPUT",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_V3_RESULTS_ROOT",
    "MULTI_FACTOR_NOTE",
    "PartialReportValidationError",
    "REPORT_SCHEMA",
    "TARGET_CELLS",
    "build_parser",
    "build_report",
    "canonical_eval_identity_from_loaded_manifest",
    "load_canonical_eval_identity",
    "main",
    "render_markdown",
    "write_report",
]
