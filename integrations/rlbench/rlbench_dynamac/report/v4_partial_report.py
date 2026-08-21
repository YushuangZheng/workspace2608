"""Validate and report the complete 22-cell local V4 paper matrix.

This is a result-side tool.  It never opens an RLBench simulator and never
writes into an evaluation-set directory.  A missing target result remains
``NOT_RUN``.  Once a target result exists, however, its episode accounting,
V4 identity and fixed evaluation-set binding are authenticated before any
number is reported.  Video evidence is produced by a separate post-evaluation
replay workflow; a missing replay is reported as pending and never invalidates
an otherwise complete formal result.

The cells match the locally runnable entries returned by
``report.matrix.PAPER_CELLS``: 12 Table-I cells, four Table-II cells, and six
Table-III cells.  No cross-cell average is inferred.
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

from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import EVALUATION_PROTOCOL_ID as BIMANUAL_EVALUATION_PROTOCOL_ID
from integrations.rlbench.rlbench_dynamac.eval.evaluation_split import (
    EVALUATION_SET_V2_CONFIG_SPEC,
    EVALUATION_SET_V2_ID,
    load_evaluation_set_v2_spec,
)
from integrations.rlbench.rlbench_dynamac.report.evaluation_videos import (
    CAPTURE_CONFIG_SCHEMA,
    DEFAULT_SELECTION_SEED,
    SELECTION_PROTOCOL_ID,
    SELECTION_SCHEMA,
    retention_quota,
)
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json, json_ready
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    FORMAL_POLICY_CLOCK_SEMANTICS_ID,
    GLOBAL_IK_CONTROLLER_PROFILE,
    STAGED_MOTION_PLAN_BATCH_SCHEMA,
    STAGED_VALIDATED_MOTION_PROTOCOL_ID,
    global_ik_controller_metadata,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_eval_v4 import (
    V4_STORE_MODE_ORDER,
    V4_STORE_MOTION_PROTOCOL_ID,
    V4_STORE_RUNTIME_LOADER_ID,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    STORE_BOTTLE_SEMANTIC_SCHEMA,
    STORE_BOTTLE_SEMANTIC_VERSION,
)
from integrations.rlbench.rlbench_dynamac.data.store_bottle_v4 import TRAINING_IDENTITY_SCHEMA
from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import (
    V4_COORDINATION_PROTOCOL_ID,
    V4_COORDINATION_SMOOTH_POLICY_TICKS,
    V4_COORDINATION_TRANSLATION_METERS,
    V4_COORDINATION_TRIGGER_STEP,
    V4_LIFT_MOTION_PROTOCOL_ID,
    V4_LIFT_RUNTIME_LOADER_ID,
)
from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
    EVALUATION_PROTOCOL_ID as UNIMANUAL_EVALUATION_PROTOCOL_ID,
)


from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
DEFAULT_RESULTS_ROOT = INTEGRATION_ROOT / "results" / "v4"
DEFAULT_V3_RESULTS_ROOT = INTEGRATION_ROOT / "results" / "v3"
DEFAULT_JSON_OUTPUT = DEFAULT_RESULTS_ROOT / "reports" / "full_22_cell.json"
DEFAULT_MARKDOWN_OUTPUT = (
    DEFAULT_RESULTS_ROOT / "reports" / "full_22_cell.md"
)

REPORT_SCHEMA = "dynamac-v4-full-22-cell-report-v3"
CANONICAL_EVAL_IDENTITY_SCHEMA = "dynamac-v4-canonical-evaluation-identity-v1"
EXPECTED_EPISODES = 200
EXPECTED_SEED = 2_608_000_000
SHA256_HEX_LENGTH = 64
POST_EVALUATION_REPLAY_PENDING = "POST_EVALUATION_REPLAY_PENDING"
REPLAY_VALIDATED = "REPLAY_VALIDATED"
STALE_RESULT_PENDING = "PENDING"
EVALUATION_PROTOCOL_ID = BIMANUAL_EVALUATION_PROTOCOL_ID
UNIMANUAL_TASKS = frozenset(
    {"stack_wine", "place_cups", "open_microwave", "wipe_desk"}
)
MULTI_FACTOR_NOTE = (
    "Descriptive multi-factor V4-versus-V3 comparison only.  The releases "
    "differ in more than one factor (including evaluation-set identity, task/intervention "
    "semantics, IK execution, and result/video protocol), so the delta must "
    "not be attributed to any single change."
)


class PartialReportValidationError(ValueError):
    """A present result cannot be admitted into the partial report."""


class StaleEvaluationBatchError(PartialReportValidationError):
    """A result is validly formed but names an older batch for this cell."""


@dataclass(frozen=True)
class CellSpec:
    task: str
    scenario: str
    paper_target: float
    v3_relative_path: str
    table: str
    result_family: str

    @property
    def cell_id(self) -> str:
        return f"{self.task}/{self.scenario}"


def _v3_relative_path(task: str, scenario: str, result_family: str) -> str:
    suffix = "seed2608000000_n200_h1000.json"
    if result_family == "table_i":
        family = "table_i" if scenario == "static" else "table_i_dynamic"
        return f"{family}/{task}_{scenario}_variation0_{suffix}"
    if result_family == "bimanual_static":
        return f"table_ii/{task}_static_{suffix}"
    if result_family == "bimanual_dynamic":
        return f"table_iii_environment/{task}_teleport_{suffix}"
    if result_family == "bimanual_coordination":
        return (
            "table_iii_coordination/"
            f"{scenario}_preregistered_trigger_{suffix}"
        )
    raise PartialReportValidationError(
        f"unsupported paper-comparison result family: {result_family}"
    )


def _target_cells_from_matrix() -> Tuple[CellSpec, ...]:
    from integrations.rlbench.rlbench_dynamac.report.matrix import PAPER_CELLS

    selected = []
    for paper_cell in PAPER_CELLS:
        if not paper_cell.local_scenarios:
            raise PartialReportValidationError(
                f"paper cell lacks a local scenario: {paper_cell.task}"
            )
        scenario = paper_cell.local_scenarios[0]
        selected.append(
            CellSpec(
                task=paper_cell.local_task,
                scenario=scenario,
                paper_target=paper_cell.paper_rate,
                v3_relative_path=_v3_relative_path(
                    paper_cell.local_task,
                    scenario,
                    paper_cell.result_family,
                ),
                table=paper_cell.table,
                result_family=paper_cell.result_family,
            )
        )
    if len(selected) != 22 or len({cell.cell_id for cell in selected}) != 22:
        raise PartialReportValidationError(
            "report.matrix.PAPER_CELLS must expose exactly 22 unique local cells"
        )
    return tuple(selected)


TARGET_CELLS: Tuple[CellSpec, ...] = _target_cells_from_matrix()
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
            "canonical rlbench_eval_v2 22-cell batch map is invalid"
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

    from integrations.rlbench.rlbench_dynamac.eval.eval_set import load_fixed_eval_set_manifest

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
            f"22-cell scope is absent from the V4 spec: {sorted(missing_targets)}"
        )
    return tuple(cells)


def _default_result_candidates(results_root: Path, cell: CellSpec) -> Tuple[Path, ...]:
    task = cell.task
    if cell.result_family == "table_i":
        family = "table_i" if cell.scenario == "static" else "table_i_dynamic"
        return (
            results_root
            / family
            / (
                f"{task}_{cell.scenario}_variation0_seed2608000000_"
                "n200_h1000.json"
            ),
        )
    if cell.result_family == "bimanual_static":
        return (
            results_root
            / "table_ii"
            / f"{task}_static_seed2608000000_n200_h1000.json",
        )
    if cell.result_family == "bimanual_dynamic":
        return (
            results_root
            / "table_iii_environment"
            / f"{task}_teleport_seed2608000000_n200_h1000.json",
        )
    if cell.result_family != "bimanual_coordination":
        raise PartialReportValidationError(
            f"unsupported V4 result family: {cell.result_family}"
        )
    arm = cell.scenario.removeprefix("coordination_hand_")
    return (
        results_root
        / "table_iii_coordination"
        / (
            f"coordination_hand_{arm}_v4_smooth_clock_tick235_"
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
            f"--cell-result names cells outside the 22-cell scope: {sorted(unknown)}"
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
        if any(
            part in {"evaluation_videos", "replay_video", "reports"}
            for part in relative.parts
        ):
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
        # Coordination changed its formal runtime semantics and filename.  Old
        # coordination JSONs remain historical artifacts, not alternate names
        # for the current cell.  Other result families retain identity-based
        # discovery for explicitly redirected but otherwise valid outputs.
        matches = {path.resolve() for path in canonical}
        if cell.result_family != "bimanual_coordination":
            matches.update(scanned[cell_id])
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
    require_formal_clock: bool = True,
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
        if require_formal_clock:
            if row.get("policy_clock_semantics_id") != FORMAL_POLICY_CLOCK_SEMANTICS_ID:
                raise PartialReportValidationError(
                    f"{label}.results[{index}] has stale policy-clock semantics"
                )
            committed = _integer(
                row.get("committed_policy_steps"),
                f"{label}.results[{index}].committed_policy_steps",
            )
            holds = _integer(
                row.get("primary_failure_joint_hold_commits"),
                f"{label}.results[{index}].primary_failure_joint_hold_commits",
            )
            if holds > committed:
                raise PartialReportValidationError(
                    f"{label}.results[{index}] has invalid committed-clock accounting"
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
    """Admit a result by its cell-local input identity.

    The manifest/spec hashes remain mandatory provenance records, but they are
    deliberately not global admission keys: changing an unrelated task in the
    same sealed evaluation set must not invalidate this cell.  Only the batch
    selected by this cell is compared with the current canonical manifest.
    """

    canonical = _validate_canonical_eval_identity(dict(canonical_eval_identity))
    selected = canonical["selected_batches"][cell.cell_id]
    expected_fields = {
        "evaluation_set_id",
        "manifest_sha256",
        "spec_sha256",
        "selected_batch_sha256",
        "selected_batch_fingerprint",
        "formal_access",
    }
    identity = payload.get("fixed_eval_set")
    if not isinstance(identity, dict):
        raise PartialReportValidationError(f"{label}.fixed_eval_set is required")
    if set(identity) != expected_fields:
        raise PartialReportValidationError(
            f"{label}.fixed_eval_set fields do not match the canonical schema"
        )
    if identity.get("evaluation_set_id") != EVALUATION_SET_V2_ID:
        raise PartialReportValidationError(
            f"{label}.fixed_eval_set.evaluation_set_id is not "
            f"{EVALUATION_SET_V2_ID}"
        )
    if identity.get("formal_access") != "canonical_id_read_only_no_generation":
        raise PartialReportValidationError(
            f"{label}.fixed_eval_set.formal_access is not the formal read-only mode"
        )
    for field in ("manifest_sha256", "spec_sha256"):
        if not _is_sha256(identity.get(field)):
            raise PartialReportValidationError(
                f"{label}.fixed_eval_set.{field} is not a SHA-256 provenance record"
            )
    for field, expected_value in (
        ("selected_batch_sha256", selected["sha256"]),
        ("selected_batch_fingerprint", selected["fingerprint"]),
    ):
        if not _is_sha256(identity.get(field)):
            raise PartialReportValidationError(
                f"{label}.fixed_eval_set.{field} is not a SHA-256 cell identity"
            )
        if identity.get(field) != expected_value:
            raise StaleEvaluationBatchError(
                f"{label}.fixed_eval_set.{field} does not match the current "
                "cell-selected evaluation batch"
            )


def _validate_controller(payload: Mapping[str, Any], label: str) -> None:
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise PartialReportValidationError(f"{label}.controller is required")
    expected = global_ik_controller_metadata()
    for field, value in expected.items():
        if controller.get(field) != value:
            raise PartialReportValidationError(
                f"{label}.controller.{field} is not the formal global setting"
            )
    if controller.get("worker_clock_handshake_id") != FORMAL_POLICY_CLOCK_SEMANTICS_ID:
        raise PartialReportValidationError(
            f"{label} worker/controller clock identities disagree"
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


def _validate_open_microwave_task_scoped_scenario_protocol(
    payload: Mapping[str, Any], cell: CellSpec
) -> None:
    """Authenticate OpenMicrowave's sealed outer envelope and inner runtime batch.

    The canonical evaluation-set preflight authenticates the task-scoped outer
    envelope named by ``selected_batch_fingerprint``.  The unimanual evaluator
    then reports that outer fingerprint at result level and the selectively
    composed inner V3.4 runtime batch in ``staged_motion_plan_cache``.  They are
    intentionally distinct identities.
    """

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise PartialReportValidationError(f"{cell.cell_id}.protocol is required")
    motion = protocol.get("motion_protocol")
    cache = protocol.get("staged_motion_plan_cache")
    if not isinstance(motion, dict) or not isinstance(cache, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id} lacks its task-scoped sealed motion-plan identity"
        )
    expected_kind = {
        "static": "static",
        "smooth": "smooth_task_motion",
        "teleport": "teleport_task",
    }[cell.scenario]
    selected_fingerprint = payload.get("fixed_eval_set", {}).get(
        "selected_batch_fingerprint"
    )
    runtime_fingerprint = cache.get("batch_fingerprint")
    cache_key = cache.get("cache_key")
    if (
        cell.task != "open_microwave"
        or protocol.get("dynamic_method") != expected_kind
        or protocol.get("protocol_valid") is not True
        or motion.get("protocol_id") != STAGED_VALIDATED_MOTION_PROTOCOL_ID
        or cache.get("schema") != STAGED_MOTION_PLAN_BATCH_SCHEMA
        or cache.get("protocol_id") != STAGED_VALIDATED_MOTION_PROTOCOL_ID
        or not _is_sha256(selected_fingerprint)
        or payload.get("motion_plan_batch_fingerprint") != selected_fingerprint
        or not _is_sha256(runtime_fingerprint)
        or runtime_fingerprint == selected_fingerprint
        or cache.get("scenario_independent") is not True
        or cache.get("formal_access") != "canonical_eval_set_read_only"
        or not isinstance(cache_key, dict)
        or cache_key.get("task") != cell.task
        or cache_key.get("base_seed") != EXPECTED_SEED
        or cache_key.get("episodes") != EXPECTED_EPISODES
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has the wrong task-scoped OpenMicrowave scenario protocol"
        )


def _validate_legacy_reused_scenario_protocol(
    payload: Mapping[str, Any], cell: CellSpec
) -> None:
    """Authenticate a legacy shared sealed-plan contract or task-scoped repair."""

    if cell.task == "open_microwave":
        _validate_open_microwave_task_scoped_scenario_protocol(payload, cell)
        return

    unimanual = cell.task in UNIMANUAL_TASKS
    protocol_field = "protocol" if unimanual else "scenario_protocol"
    protocol = payload.get(protocol_field)
    if not isinstance(protocol, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id}.{protocol_field} is required"
        )
    motion = protocol.get("motion_protocol")
    cache = protocol.get("staged_motion_plan_cache")
    if not isinstance(motion, dict) or not isinstance(cache, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id} lacks its generic sealed motion-plan identity"
        )
    expected_kind = {
        "static": "static",
        "smooth": "smooth_task_motion",
        "teleport": "teleport_task",
    }[cell.scenario]
    reported_kind = (
        protocol.get("dynamic_method")
        if unimanual
        else protocol.get("motion_kind")
    )
    selected_fingerprint = payload.get("fixed_eval_set", {}).get(
        "selected_batch_fingerprint"
    )
    cache_key = cache.get("cache_key")
    if (
        reported_kind != expected_kind
        or protocol.get("protocol_valid") is not True
        or motion.get("protocol_id") != STAGED_VALIDATED_MOTION_PROTOCOL_ID
        or cache.get("schema") != STAGED_MOTION_PLAN_BATCH_SCHEMA
        or cache.get("protocol_id") != STAGED_VALIDATED_MOTION_PROTOCOL_ID
        or not _is_sha256(selected_fingerprint)
        or cache.get("batch_fingerprint") != selected_fingerprint
        or payload.get("motion_plan_batch_fingerprint") != selected_fingerprint
        or cache.get("scenario_independent") is not True
        or cache.get("formal_access") != "canonical_eval_set_read_only"
        or not isinstance(cache_key, dict)
        or cache_key.get("task") != cell.task
        or cache_key.get("base_seed") != EXPECTED_SEED
        or cache_key.get("episodes") != EXPECTED_EPISODES
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has the wrong generic legacy-reused scenario protocol"
        )
    if not unimanual:
        expected_status = (
            "STATIC_REFERENCE"
            if cell.scenario == "static"
            else "V3_PREREGISTERED_CHECKPOINT_AUTHENTICATED"
        )
        if protocol.get("status") != expected_status:
            raise PartialReportValidationError(
                f"{cell.cell_id} has the wrong generic scenario status"
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
        or protocol.get("smooth_policy_ticks")
        != V4_COORDINATION_SMOOTH_POLICY_TICKS
        or protocol.get("persistent_policy_target_offset") is not True
        or protocol.get("policy_clock_advances_during_intervention") is not True
        or protocol.get("protocol_valid") is not True
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} has the wrong smooth V4 coordination protocol"
        )
    controller = payload.get("controller")
    execution = (
        controller.get("coordination_intervention_execution")
        if isinstance(controller, dict)
        else None
    )
    if (
        not isinstance(execution, dict)
        or execution.get("uses_same_global_action_mode_ik_chain") is not True
        or execution.get("policy_transaction") is not True
        or execution.get("intervention_and_policy_share_action") is not True
        or execution.get("policy_requests")
        != V4_COORDINATION_SMOOTH_POLICY_TICKS
        or execution.get("policy_clock_advances")
        != V4_COORDINATION_SMOOTH_POLICY_TICKS
        or execution.get("max_primary_action_attempts_per_policy_tick") != 1
        or execution.get("raw_joint_hold_commit_on_failure") is not True
    ):
        raise PartialReportValidationError(
            f"{cell.cell_id} lacks the coordination execution-chain audit"
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
    if capture is None:
        return {
            "status": POST_EVALUATION_REPLAY_PENDING,
            "selection_manifest_path": None,
            "selection_manifest_sha256": None,
            "selection_protocol_id": None,
            "retention_tier": None,
            "retained": None,
            "retained_video_paths": [],
            "reason": (
                "formal evaluation is complete; outcome-matched replay videos "
                "have not yet been generated"
            ),
        }
    if not isinstance(capture, dict):
        raise PartialReportValidationError(
            f"{cell.cell_id}.evaluation_video_capture must be an object or null"
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
        "status": REPLAY_VALIDATED,
        "selection_manifest_path": manifest_path.relative_to(root).as_posix(),
        "selection_manifest_sha256": audit["sha256"],
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        "retention_tier": quota.tier,
        "retained": expected_retained,
        "retained_video_paths": retained_paths,
    }


def _validated_post_evaluation_replays(
    *,
    results_root: Path,
    completed_cell_count: int,
) -> Dict[str, Dict[str, Any]]:
    """Validate and summarize the separate post-evaluation replay release.

    The replay planner is intentionally imported lazily: it authenticates the
    same formal results through :mod:`v4_formal_launch`, whose admission gate
    calls this module.  Calling it only after the report's first validation
    pass avoids a dependency cycle while keeping one replay source of truth.
    """

    root = Path(results_root).resolve()
    if (
        completed_cell_count != len(TARGET_CELLS)
        or root != DEFAULT_RESULTS_ROOT.resolve()
    ):
        return {}

    from . import v4_post_evaluation_replays as post_replays

    try:
        plan = post_replays.build_validated_plan(
            selection_seed=DEFAULT_SELECTION_SEED
        )
    except RuntimeError as error:
        raise PartialReportValidationError(
            f"cannot authenticate the post-evaluation replay plan: {error}"
        ) from error

    jobs_by_cell: Dict[str, list[Any]] = {}
    for job in plan.jobs:
        jobs_by_cell.setdefault(job.cell_id, []).append(job)

    summaries: Dict[str, Dict[str, Any]] = {}
    for selection in plan.selections:
        jobs = jobs_by_cell.get(selection.cell_id, [])
        # An absent target is ordinary pending replay state.  A published
        # target, however, must pass the recorder's complete manifest/hash
        # admission gate before the report may cite it.
        if any(
            not (job.target.exists() or job.target.is_symlink()) for job in jobs
        ):
            continue

        retained = {
            "successes": selection.required_successes,
            "failures": selection.required_failures,
        }
        outcome_manifests: Dict[str, Dict[str, Any]] = {}
        retained_video_paths = []
        for job in jobs:
            try:
                manifest = post_replays._validate_job_output(job)
            except RuntimeError as error:
                raise PartialReportValidationError(
                    f"invalid post-evaluation replay for {selection.cell_id}: {error}"
                ) from error
            manifest_path = job.target / "manifest.json"
            rows = manifest["episodes"]
            video_paths = [
                _relative_display(job.target / row["video"], root)
                for row in rows
            ]
            retained_video_paths.extend(video_paths)
            outcome_manifests[job.outcome] = {
                "manifest_path": _relative_display(manifest_path, root),
                "manifest_sha256": _sha256(manifest_path),
                "replay_directory": _relative_display(job.target, root),
                "confirmed_trajectories": len(rows),
                "video_paths": video_paths,
            }

        summaries[selection.cell_id] = {
            "status": REPLAY_VALIDATED,
            "workflow": "post_evaluation_outcome_replay",
            # There is one manifest per non-empty outcome, rather than the
            # legacy single-cell capture manifest embedded in formal results.
            "selection_manifest_path": None,
            "selection_manifest_sha256": None,
            "selection_protocol_id": post_replays.evaluation_videos.SELECTION_PROTOCOL_ID,
            "selection_seed": plan.selection_seed,
            "retention_tier": selection.quota_tier,
            "retained": retained,
            "retained_video_paths": retained_video_paths,
            "outcome_manifests": outcome_manifests,
        }
    return summaries


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
            if item["success_rate"] is None:
                if item.get("completed") != 0:
                    raise PartialReportValidationError(
                        f"Store subgroup {name}.success_rate must be numeric"
                    )
            else:
                item["success_rate"] = _rate(
                    item["success_rate"], f"Store subgroup {name}.success_rate"
                )
        if all(field in item for field in ("completed", "successes")):
            if item["successes"] > item["completed"]:
                raise PartialReportValidationError(
                    f"Store subgroup {name} has more successes than completed episodes"
                )
            if item["completed"] == 0 and item["successes"] != 0:
                raise PartialReportValidationError(
                    f"Store subgroup {name} zero-count statistics are inconsistent"
                )
            expected = (
                item["successes"] / float(item["completed"])
                if item["completed"]
                else None
            )
            if (
                "success_rate" in item
                and item["success_rate"] is not None
                and (
                    not math.isclose(
                        item["success_rate"],
                        0.0 if expected is None else expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
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
    expected_evaluation_protocol_id = (
        UNIMANUAL_EVALUATION_PROTOCOL_ID
        if cell.task in UNIMANUAL_TASKS
        else BIMANUAL_EVALUATION_PROTOCOL_ID
    )
    if payload.get("evaluation_protocol_id") != expected_evaluation_protocol_id:
        raise PartialReportValidationError(
            f"{label} does not use the formal V4 evaluation protocol"
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
        if cell.task in {
            "bimanual_put_bottle_in_fridge",
            "bimanual_lift_tray",
        }:
            _validate_direct_task_protocol(payload, cell)
        else:
            _validate_legacy_reused_scenario_protocol(payload, cell)
        store_training_identity_fingerprint = (
            _validate_store_model_identity(payload, cell)
            if cell.task == "bimanual_put_bottle_in_fridge"
            else None
        )
    diagnostics = payload.get("ik_execution_diagnostics")
    if not isinstance(diagnostics, dict):
        raise PartialReportValidationError(f"{label}.ik_execution_diagnostics is required")
    diagnostics = _json_statistics(diagnostics, f"{label}.ik_execution_diagnostics")
    if (
        diagnostics.get("controller_profile") != GLOBAL_IK_CONTROLLER_PROFILE
        or diagnostics.get("trac_ik_distance_unbounded_cartesian_api_uses") != 0
    ):
        raise PartialReportValidationError(
            f"{label}.ik_execution_diagnostics has stale or unbounded IK identity"
        )
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
    result_status = (
        POST_EVALUATION_REPLAY_PENDING
        if videos["status"] == POST_EVALUATION_REPLAY_PENDING
        else "COMPLETED_VALIDATED"
    )
    return {
        "status": result_status,
        "formal_result_status": "COMPLETED_VALIDATED",
        "result_path": _relative_display(path, results_root),
        "result_sha256": _sha256(path),
        "successes": successes,
        "episodes": EXPECTED_EPISODES,
        "success_rate": success_rate,
        "paper_target": cell.paper_target,
        "gap_to_paper_percentage_points": (success_rate - cell.paper_target) * 100.0,
        "validated_identity": {
            "release": "v4",
            "evaluation_protocol_id": expected_evaluation_protocol_id,
            "evaluation_set_id": EVALUATION_SET_V2_ID,
            "controller_profile": GLOBAL_IK_CONTROLLER_PROFILE,
            "policy_clock_semantics_id": FORMAL_POLICY_CLOCK_SEMANTICS_ID,
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
        payload,
        label=f"V3 comparison {cell.cell_id}",
        require_formal_clock=False,
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
                "formal_result_status": "NOT_RUN",
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
            try:
                value = _validate_v4_result(
                    path,
                    cell,
                    Path(results_root),
                    canonical_eval_identity=canonical_eval_identity,
                )
            except StaleEvaluationBatchError as error:
                value = {
                    "status": STALE_RESULT_PENDING,
                    "formal_result_status": STALE_RESULT_PENDING,
                    "result_path": _relative_display(path, Path(results_root)),
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
                    "pending_reason": str(error),
                }
            else:
                completed += 1
        value = {
            "cell_id": cell.cell_id,
            "table": cell.table,
            "result_family": cell.result_family,
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

    post_evaluation_replays = _validated_post_evaluation_replays(
        results_root=Path(results_root),
        completed_cell_count=completed,
    )
    for row in cells:
        replay = post_evaluation_replays.get(row["cell_id"])
        if replay is not None:
            row["videos"] = replay
            row["status"] = "COMPLETED_VALIDATED"

    target_status = {row["cell_id"]: row["status"] for row in cells}
    replay_pending = sum(
        row["status"] == POST_EVALUATION_REPLAY_PENDING for row in cells
    )
    replay_validated = sum(
        row.get("videos", {}).get("status") == REPLAY_VALIDATED
        for row in cells
        if isinstance(row.get("videos"), dict)
    )
    formal_scope = []
    for cell_id in formal_cells:
        target = cell_id in TARGET_BY_ID
        formal_scope.append(
            {
                "cell_id": cell_id,
                "in_22_cell_scope": target,
                "status": target_status.get(cell_id, "NOT_RUN"),
                "reason": (
                    None
                    if target_status.get(cell_id) == "COMPLETED_VALIDATED"
                    else (
                        (
                            "formal result validated; post-evaluation replay pending"
                            if target_status.get(cell_id)
                            == POST_EVALUATION_REPLAY_PENDING
                            else (
                                "the existing result names an older batch for "
                                "this cell"
                                if target_status.get(cell_id)
                                == STALE_RESULT_PENDING
                                else "target result artifact is absent"
                            )
                        )
                        if target
                        else "outside the 22-cell local paper matrix"
                    )
                ),
            }
        )
    return json_ready(
        {
            "schema": REPORT_SCHEMA,
            "release": "v4",
            "report_kind": "full_22_cell_validation",
            "evaluation_set_id": EVALUATION_SET_V2_ID,
            "evaluation_spec_fingerprint": spec["fingerprint"],
            "results_root": str(Path(results_root).resolve()),
            "scope": {
                "target_cell_count": len(TARGET_CELLS),
                "completed_validated_cell_count": completed,
                "post_evaluation_replay_pending_cell_count": replay_pending,
                "replay_validated_cell_count": replay_validated,
                "not_run_target_cell_count": len(TARGET_CELLS) - completed,
                "full_tables_generated": True,
                "cross_cell_average_computed": False,
                "evaluation_set_modified": False,
            },
            "cells": cells,
            "formal_cell_status": formal_scope,
            "cross_cell_aggregate": {
                "computed": False,
                "value": None,
                "reason": "the 22-cell validation report does not infer a cross-cell average",
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
    """Render the complete local 22-cell matrix."""

    lines = [
        "# DynaMAC V4 full 22-cell report",
        "",
        (
            "This report covers all 22 locally runnable cells corresponding to "
            "paper Tables I–III. Missing formal results remain **NOT_RUN**; "
            "results bound to an older batch for their own cell remain "
            f"**{STALE_RESULT_PENDING}**; validated formal results without "
            "post-evaluation replay are "
            f"**{POST_EVALUATION_REPLAY_PENDING}**. No cross-cell average is computed."
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
        if cell["status"] == STALE_RESULT_PENDING:
            lines.extend(
                [
                    "Status: **PENDING** — "
                    + _markdown_escape(cell["pending_reason"])
                    + ".",
                    "",
                ]
            )
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
        if videos["status"] == POST_EVALUATION_REPLAY_PENDING:
            lines.append(
                "- Replay videos: **POST_EVALUATION_REPLAY_PENDING** — "
                + videos["reason"]
            )
        else:
            retained_text = (
                "- Retained replay videos: "
                f"success={videos['retained']['successes']}, "
                f"failure={videos['retained']['failures']}"
            )
            if videos.get("outcome_manifests") is None:
                retained_text += (
                    "; manifest "
                    f"`{_markdown_escape(videos['selection_manifest_path'])}`"
                )
            lines.append(retained_text)
            for outcome, manifest in videos.get("outcome_manifests", {}).items():
                lines.append(
                    f"- {outcome.capitalize()} replay manifest: "
                    f"`{_markdown_escape(manifest['manifest_path'])}` "
                    f"(count={manifest['confirmed_trajectories']}, "
                    f"sha256={manifest['manifest_sha256']})"
                )
            if videos["retained_video_paths"]:
                lines.append("- Retained paths: " + ", ".join(
                    f"`{_markdown_escape(path)}`"
                    for path in videos["retained_video_paths"]
                ))
            else:
                lines.append(
                    "- Retained paths: none (the declared retention tier selected zero)."
                )
        lines.append("")

    lines.extend(
        [
            "## Formal V4 cell status",
            "",
            "The evaluation spec has three consumers outside the 22-cell local paper "
            "matrix. Report state remains result-side and is never written into the seal.",
            "",
            "| Formal cell | 22-cell scope | Status | Reason |",
            "|---|---:|---:|---|",
        ]
    )
    for row in report["formal_cell_status"]:
        lines.append(
            "| {cell} | {scope} | {status} | {reason} |".format(
                cell=_markdown_escape(row["cell_id"]),
                scope="yes" if row["in_22_cell_scope"] else "no",
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
    root = (INTEGRATION_ROOT / "data" / "evaluation").resolve()
    candidate = Path(path).resolve()
    if _inside(root, candidate):
        raise PartialReportValidationError(
            "partial reports are result state and must never be written into data/evaluation"
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
            "Override discovery for one of the 22 cells. A missing override "
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
        "({completed}/22 V4 cells validated)".format(
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
    "STALE_RESULT_PENDING",
    "StaleEvaluationBatchError",
    "TARGET_CELLS",
    "build_parser",
    "build_report",
    "canonical_eval_identity_from_loaded_manifest",
    "load_canonical_eval_identity",
    "main",
    "render_markdown",
    "write_report",
]
