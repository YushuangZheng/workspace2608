"""Read-only authentication for the materialized current RLBench eval set."""

from __future__ import annotations

import hashlib
import json
import re
import argparse
from pathlib import Path
from typing import Any, Callable, Mapping

from integrations.rlbench.rlbench_dynamac.eval.evaluation_split import (
    EVALUATION_SET_V2_ID,
    EVALUATION_SET_V2_SPEC_SCHEMA,
    load_evaluation_set_v2_spec,
)
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    StagedMotionPlan,
    load_staged_motion_plan_batch,
    load_staged_source_plan_batch,
    stage_source_plan,
    staged_motion_plan_batch,
    staged_source_plan_batch,
)

EVAL_SET_V2_MANIFEST_SCHEMA = "dynamac-rlbench-sealed-evaluation-manifest-v2"
EVAL_SET_V2_PROTOCOL_ID = "rlbench-v4-task-scoped-eval-set-v2"
TASK_SCOPED_IDENTITY_SCHEMA = "dynamac-rlbench-task-scoped-identity-v2"
TASK_SCOPED_PLAN_BATCH_SCHEMA = (
    "dynamac-rlbench-task-scoped-motion-plan-batch-v2"
)
TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID = (
    "rlbench-task-scoped-staged-motion-plan-envelope-v2"
)
TASK_SCOPED_REBIND_PROVENANCE_SCHEMA = (
    "dynamac-rlbench-task-scoped-plan-identity-rebind-v1"
)
TRAINING_DATA_BINDING_SCHEMA = "dynamac-rlbench-training-data-binding-v1"
LEGACY_RUNTIME_BATCH_LOADER = "staged_motion_plan_batch_v3_4"
SELECTIVE_COMPOSED_RUNTIME_BATCH_LOADER = (
    "selective_composed_staged_motion_plan_batch_v3_4"
)
SELECTIVE_COMPOSITION_PROVENANCE_SCHEMA = (
    "dynamac-rlbench-selective-plan-composition-v1"
)
SELECTIVE_COMPOSITION_PROTOCOL_ID = (
    "rlbench-copy-base-replace-listed-plans-v1"
)
OPEN_MICROWAVE_TASK = "open_microwave"
GLOBAL_EVAL_SEED_START = 2_608_000_000
FIXED_EVAL_EPISODES = 200
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
EVAL_SET_ROOT = INTEGRATION_ROOT / "data" / "evaluation"
ENVIRONMENT_TASKS = frozenset(
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
COORDINATION_TASK = "bimanual_handover_item_dynamic"
def _builtin_v4_runtime_loaders() -> dict[
    str,
    Callable[[dict[str, Any]], list[Any]],
]:
    """Return the authenticated V4 plan loaders without import cycles.

    The V4 protocol modules import this module only inside their envelope
    builders, so importing their stable loader registries lazily here keeps
    policy-only validation lightweight and makes the sealed evaluation set
    self-loading in both the CLI and formal evaluators.
    """

    from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_eval_v4 import v4_store_runtime_loaders
    from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import v4_runtime_loaders

    loaders = {
        **v4_runtime_loaders(),
        **v4_store_runtime_loaders(),
    }
    if len(loaders) != 2:
        raise RuntimeError("V4 runtime loader registry is incomplete")
    return loaders


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_task_scoped_identity(
    *,
    task_name: str,
    components: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build the small task-local identity bound to one regenerated batch."""

    required = {"task_semantics", "motion_source", "intervention"}
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("task-scoped identity task is invalid")
    if set(components) != required:
        raise ValueError("task-scoped identity components are incomplete")
    normalized: dict[str, dict[str, str]] = {}
    for role in sorted(required):
        component = components[role]
        if (
            not isinstance(component, Mapping)
            or set(component) != {"schema", "fingerprint"}
            or not isinstance(component.get("schema"), str)
            or not component["schema"]
            or not _is_sha256(component.get("fingerprint"))
        ):
            raise ValueError(f"task-scoped identity component is invalid: {role}")
        normalized[role] = {
            "schema": component["schema"],
            "fingerprint": component["fingerprint"],
        }
    body = {
        "schema": TASK_SCOPED_IDENTITY_SCHEMA,
        "scope": task_name,
        "components": normalized,
    }
    return {**body, "fingerprint": canonical_fingerprint(body)}


def _validate_task_scoped_identity(
    identity: Any,
    *,
    task_name: str,
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("task-scoped identity must be an object")
    components = identity.get("components")
    expected = build_task_scoped_identity(
        task_name=task_name,
        components=components if isinstance(components, dict) else {},
    )
    if identity != expected:
        raise ValueError("task-scoped identity authentication failed")
    return identity


def build_task_scoped_plan_batch(
    *,
    task_name: str,
    task_identity: Mapping[str, Any],
    runtime_loader: str,
    runtime_batch: Mapping[str, Any],
    rebind_provenance: Mapping[str, Any] | None = None,
    composition_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a runtime plan batch to only the task protocols that produced it.

    The inner runtime representation remains owned by the staging/runtime
    implementation.  This envelope prevents a StoreBottle or LiftTray change
    from changing the identity of unrelated imported legacy batches.
    """

    identity = _validate_task_scoped_identity(dict(task_identity), task_name=task_name)
    if (
        not isinstance(runtime_loader, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", runtime_loader) is None
    ):
        raise ValueError("task-scoped runtime loader ID is invalid")
    if not isinstance(runtime_batch, Mapping):
        raise ValueError("task-scoped runtime batch must be an object")
    inner = dict(runtime_batch)
    variation_schedule = inner.get("variation_schedule")
    episodes = inner.get("episodes")
    base_seed = inner.get("base_seed")
    if (
        inner.get("task_name") != task_name
        or isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or isinstance(episodes, bool)
        or not isinstance(episodes, int)
        or episodes < 1
        or not isinstance(variation_schedule, list)
        or len(variation_schedule) != episodes
        or not _is_sha256(inner.get("batch_fingerprint"))
    ):
        raise ValueError("task-scoped runtime batch schedule is invalid")
    body = {
        "schema": TASK_SCOPED_PLAN_BATCH_SCHEMA,
        "protocol_id": TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID,
        "task_name": task_name,
        "base_seed": base_seed,
        "episodes": episodes,
        "variation_schedule": variation_schedule,
        "scenario_independent": True,
        "task_identity": identity,
        "runtime_loader": runtime_loader,
        "runtime_batch": inner,
    }
    if rebind_provenance is not None:
        body["rebind_provenance"] = _validate_task_scoped_rebind_provenance(
            rebind_provenance,
            task_name=task_name,
            target_task_identity_fingerprint=identity["fingerprint"],
        )
    if composition_provenance is not None:
        if rebind_provenance is not None:
            raise ValueError("task-scoped batch cannot be rebound and composed")
        body["composition_provenance"] = _validate_selective_composition_provenance(
            composition_provenance,
            task_name=task_name,
            runtime_batch=inner,
        )
    return {**body, "batch_fingerprint": canonical_fingerprint(body)}


def _validate_selective_composition_provenance(
    provenance: Any,
    *,
    task_name: str,
    runtime_batch: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "protocol_id",
        "task_name",
        "base_source_evaluation_set_id",
        "base_source_artifact_path",
        "base_source_file_sha256",
        "base_source_batch_fingerprint",
        "replaced_episodes",
        "replacements",
        "generation_indices_rewritten",
    }
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != expected_fields
        or provenance.get("schema") != SELECTIVE_COMPOSITION_PROVENANCE_SCHEMA
        or provenance.get("protocol_id") != SELECTIVE_COMPOSITION_PROTOCOL_ID
        or provenance.get("task_name") != task_name
        or provenance.get("base_source_evaluation_set_id") != "rlbench_fixed_v1"
        or not isinstance(provenance.get("base_source_artifact_path"), str)
        or not provenance["base_source_artifact_path"].startswith(
            "plans/environment/"
        )
        or not _is_sha256(provenance.get("base_source_file_sha256"))
        or not _is_sha256(provenance.get("base_source_batch_fingerprint"))
        or provenance.get("generation_indices_rewritten") is not False
    ):
        raise ValueError("selective composition provenance is invalid")
    replaced = provenance.get("replaced_episodes")
    rows = provenance.get("replacements")
    if (
        not isinstance(replaced, list)
        or not replaced
        or replaced != sorted(set(replaced))
        or any(isinstance(value, bool) or not isinstance(value, int) for value in replaced)
        or not isinstance(rows, list)
        or [row.get("episode_index") for row in rows if isinstance(row, Mapping)]
        != replaced
    ):
        raise ValueError("selective composition replacement ledger is invalid")
    raw_plans = runtime_batch.get("plans")
    if not isinstance(raw_plans, list):
        raise ValueError("selective composition runtime plans are invalid")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "episode_index",
                "base_plan_fingerprint",
                "replacement_plan_fingerprint",
            }
            or row["episode_index"] < 0
            or row["episode_index"] >= len(raw_plans)
            or not _is_sha256(row.get("base_plan_fingerprint"))
            or not _is_sha256(row.get("replacement_plan_fingerprint"))
            or row["base_plan_fingerprint"] == row["replacement_plan_fingerprint"]
            or raw_plans[row["episode_index"]].get("fingerprint")
            != row["replacement_plan_fingerprint"]
        ):
            raise ValueError("selective composition replacement record is invalid")
    return dict(provenance)


def _validate_task_scoped_rebind_provenance(
    provenance: Any,
    *,
    task_name: str,
    target_task_identity_fingerprint: str,
) -> dict[str, Any]:
    """Authenticate the closed identity-only rewrite record on an envelope."""

    expected_fields = {
        "schema",
        "protocol_id",
        "task_name",
        "source_file_sha256",
        "source_outer_batch_fingerprint",
        "source_inner_batch_fingerprint",
        "source_task_identity_fingerprint",
        "target_task_identity_fingerprint",
        "non_identity_projection_fingerprint",
        "rewritten_fields",
        "policy_result_fields_read",
        "simulator_started",
    }
    expected_rewritten = [
        "runtime_batch.plans[*].validation.intervention_fingerprint",
        "runtime_batch.plans[*].fingerprint",
        "runtime_batch.batch_fingerprint",
        "task_identity",
        "rebind_provenance",
        "batch_fingerprint",
    ]
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != expected_fields
        or provenance.get("schema") != TASK_SCOPED_REBIND_PROVENANCE_SCHEMA
        or not isinstance(provenance.get("protocol_id"), str)
        or not provenance["protocol_id"]
        or provenance.get("task_name") != task_name
        or provenance.get("target_task_identity_fingerprint")
        != target_task_identity_fingerprint
        or provenance.get("source_task_identity_fingerprint")
        == target_task_identity_fingerprint
        or provenance.get("rewritten_fields") != expected_rewritten
        or provenance.get("policy_result_fields_read") is not False
        or provenance.get("simulator_started") is not False
    ):
        raise ValueError("task-scoped rebind provenance is invalid")
    for field in (
        "source_file_sha256",
        "source_outer_batch_fingerprint",
        "source_inner_batch_fingerprint",
        "source_task_identity_fingerprint",
        "target_task_identity_fingerprint",
        "non_identity_projection_fingerprint",
    ):
        if not _is_sha256(provenance.get(field)):
            raise ValueError(f"task-scoped rebind {field} is invalid")
    return dict(provenance)


def load_selective_composed_staged_motion_plan_batch(
    payload: dict[str, Any],
) -> list[StagedMotionPlan]:
    """Load a v3.4 batch composed from independently staged plan subsets.

    Every plan and all batch schedule/fingerprint fields remain fully
    authenticated.  Only the legacy assertion that generation indices form
    one process-global sequence across all 200 plans is omitted; the indices
    inside each independently staged plan are still validated by
    ``StagedMotionPlan.from_json``.
    """

    if not isinstance(payload, dict):
        raise ValueError("selective composed plan batch must be an object")
    body = {key: value for key, value in payload.items() if key != "batch_fingerprint"}
    if payload.get("batch_fingerprint") != canonical_fingerprint(body):
        raise ValueError("selective composed plan batch fingerprint is invalid")
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, list):
        raise ValueError("selective composed plan list is invalid")
    plans = [StagedMotionPlan.from_json(row) for row in raw_plans]
    expected = staged_motion_plan_batch(
        task_name=payload.get("task_name"),
        base_seed=payload.get("base_seed"),
        variations=payload.get("variation_schedule"),
        plans=plans,
    )
    if payload != expected:
        raise ValueError("selective composed plan batch fields are inconsistent")
    return plans


def build_selective_composition_provenance(
    *,
    task_name: str,
    base_runtime_batch: Mapping[str, Any],
    composed_runtime_batch: Mapping[str, Any],
    base_source_file_sha256: str,
    base_source_artifact_path: str,
) -> dict[str, Any]:
    """Bind an exact copy-plus-listed-replacements batch to its legacy base."""

    base = dict(base_runtime_batch)
    composed = dict(composed_runtime_batch)
    base_plans = load_staged_motion_plan_batch(base)
    composed_plans = load_selective_composed_staged_motion_plan_batch(composed)
    if (
        base.get("task_name") != task_name
        or composed.get("task_name") != task_name
        or base.get("base_seed") != composed.get("base_seed")
        or base.get("episodes") != composed.get("episodes")
        or base.get("variation_schedule") != composed.get("variation_schedule")
        or len(base_plans) != len(composed_plans)
    ):
        raise ValueError("selective composition differs from its base schedule")
    rows = []
    for episode, (base_plan, replacement_plan) in enumerate(
        zip(base_plans, composed_plans)
    ):
        base_fingerprint = base_plan.fingerprint()
        replacement_fingerprint = replacement_plan.fingerprint()
        if base_fingerprint != replacement_fingerprint:
            rows.append(
                {
                    "episode_index": episode,
                    "base_plan_fingerprint": base_fingerprint,
                    "replacement_plan_fingerprint": replacement_fingerprint,
                }
            )
    if not rows:
        raise ValueError("selective composition contains no replacements")
    provenance = {
        "schema": SELECTIVE_COMPOSITION_PROVENANCE_SCHEMA,
        "protocol_id": SELECTIVE_COMPOSITION_PROTOCOL_ID,
        "task_name": task_name,
        "base_source_evaluation_set_id": "rlbench_fixed_v1",
        "base_source_artifact_path": base_source_artifact_path,
        "base_source_file_sha256": base_source_file_sha256,
        "base_source_batch_fingerprint": base["batch_fingerprint"],
        "replaced_episodes": [row["episode_index"] for row in rows],
        "replacements": rows,
        "generation_indices_rewritten": False,
    }
    return _validate_selective_composition_provenance(
        provenance,
        task_name=task_name,
        runtime_batch=composed,
    )


def open_microwave_task_identity_components(
    runtime_batch: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Derive OpenMicrowave's task-local identity from frozen V3 contracts."""

    plans = load_selective_composed_staged_motion_plan_batch(dict(runtime_batch))
    if not plans or any(plan.task_name != OPEN_MICROWAVE_TASK for plan in plans):
        raise ValueError("OpenMicrowave runtime batch task is invalid")
    semantic_schema = plans[0].validation["task_semantic_signature"]["schema"]
    semantic_fingerprint = plans[0].validation["task_semantic_fingerprint"]
    if any(
        plan.validation.get("task_semantic_fingerprint") != semantic_fingerprint
        or plan.validation.get("task_semantic_signature", {}).get("schema")
        != semantic_schema
        for plan in plans
    ):
        raise ValueError("OpenMicrowave task semantics differ across plans")
    from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
        dynamic_trigger_profile,
        load_v3_intervention_protocol,
        load_v3_motion_source_protocol,
        motion_source_profile,
    )

    motion = load_v3_motion_source_protocol()
    intervention = load_v3_intervention_protocol()
    motion_body = {
        "protocol_schema": motion["schema"],
        "protocol_fingerprint": motion["fingerprint"],
        "task": OPEN_MICROWAVE_TASK,
        "profile": motion_source_profile(OPEN_MICROWAVE_TASK, motion),
    }
    intervention_body = {
        "protocol_schema": intervention["schema"],
        "protocol_fingerprint": intervention["fingerprint"],
        "task": OPEN_MICROWAVE_TASK,
        "profile": dynamic_trigger_profile(OPEN_MICROWAVE_TASK, intervention),
    }
    return {
        "task_semantics": {
            "schema": semantic_schema,
            "fingerprint": semantic_fingerprint,
        },
        "motion_source": {
            "schema": "dynamac-rlbench-task-motion-source-profile-v1",
            "fingerprint": canonical_fingerprint(motion_body),
        },
        "intervention": {
            "schema": "dynamac-rlbench-task-intervention-profile-v1",
            "fingerprint": canonical_fingerprint(intervention_body),
        },
    }


def build_open_microwave_task_scoped_plan_batch(
    *,
    runtime_batch: Mapping[str, Any],
    composition_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap a selectively repaired OpenMicrowave v3.4 batch for eval_v2."""

    inner = dict(runtime_batch)
    identity = build_task_scoped_identity(
        task_name=OPEN_MICROWAVE_TASK,
        components=open_microwave_task_identity_components(inner),
    )
    return build_task_scoped_plan_batch(
        task_name=OPEN_MICROWAVE_TASK,
        task_identity=identity,
        runtime_loader=SELECTIVE_COMPOSED_RUNTIME_BATCH_LOADER,
        runtime_batch=inner,
        composition_provenance=composition_provenance,
    )


def validate_selective_composition_against_base(
    envelope: Mapping[str, Any],
    *,
    base_runtime_batch: Mapping[str, Any],
    base_source_file_sha256: str,
    base_source_artifact_path: str,
) -> None:
    """Prove that only the provenance-listed plans differ from the base."""

    raw = dict(envelope)
    composed = raw.get("runtime_batch")
    provenance = _validate_selective_composition_provenance(
        raw.get("composition_provenance"),
        task_name=raw.get("task_name"),
        runtime_batch=composed if isinstance(composed, Mapping) else {},
    )
    base = dict(base_runtime_batch)
    if (
        provenance["base_source_file_sha256"] != base_source_file_sha256
        or provenance["base_source_artifact_path"] != base_source_artifact_path
        or provenance["base_source_batch_fingerprint"]
        != base.get("batch_fingerprint")
    ):
        raise ValueError("selective composition legacy-base binding is invalid")
    load_staged_motion_plan_batch(base)
    load_selective_composed_staged_motion_plan_batch(dict(composed))
    base_rows = base.get("plans")
    composed_rows = composed.get("plans")
    if not isinstance(base_rows, list) or not isinstance(composed_rows, list):
        raise ValueError("selective composition plan rows are invalid")
    replacements = {
        row["episode_index"]: row for row in provenance["replacements"]
    }
    for episode, (base_row, composed_row) in enumerate(
        zip(base_rows, composed_rows)
    ):
        record = replacements.get(episode)
        if record is None:
            if composed_row != base_row:
                raise ValueError("unlisted selective composition plan changed")
        elif (
            base_row.get("fingerprint") != record["base_plan_fingerprint"]
            or composed_row.get("fingerprint")
            != record["replacement_plan_fingerprint"]
        ):
            raise ValueError("selective composition plan binding is invalid")


def load_task_scoped_plan_batch(
    payload: Mapping[str, Any],
    *,
    runtime_loaders: Mapping[str, Callable[[dict[str, Any]], list[Any]]] | None = None,
) -> list[Any]:
    """Authenticate a task-scoped envelope and deserialize its inner plans."""

    if not isinstance(payload, Mapping):
        raise ValueError("task-scoped plan batch must be an object")
    raw = dict(payload)
    base_fields = {
        "schema",
        "protocol_id",
        "task_name",
        "base_seed",
        "episodes",
        "variation_schedule",
        "scenario_independent",
        "task_identity",
        "runtime_loader",
        "runtime_batch",
        "batch_fingerprint",
    }
    expected_field_sets = {
        frozenset(base_fields),
        frozenset(base_fields | {"rebind_provenance"}),
        frozenset(base_fields | {"composition_provenance"}),
    }
    body = {key: value for key, value in raw.items() if key != "batch_fingerprint"}
    task_name = raw.get("task_name")
    if (
        frozenset(raw) not in expected_field_sets
        or raw.get("schema") != TASK_SCOPED_PLAN_BATCH_SCHEMA
        or raw.get("protocol_id") != TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID
        or raw.get("scenario_independent") is not True
        or raw.get("batch_fingerprint") != canonical_fingerprint(body)
        or not isinstance(task_name, str)
        or not task_name
    ):
        raise ValueError("task-scoped plan batch authentication failed")
    identity = _validate_task_scoped_identity(
        raw.get("task_identity"),
        task_name=task_name,
    )
    if "rebind_provenance" in raw:
        _validate_task_scoped_rebind_provenance(
            raw["rebind_provenance"],
            task_name=task_name,
            target_task_identity_fingerprint=identity["fingerprint"],
        )
    if "composition_provenance" in raw:
        _validate_selective_composition_provenance(
            raw["composition_provenance"],
            task_name=task_name,
            runtime_batch=(
                raw["runtime_batch"]
                if isinstance(raw.get("runtime_batch"), Mapping)
                else {}
            ),
        )
    inner = raw.get("runtime_batch")
    if not isinstance(inner, dict):
        raise ValueError("task-scoped runtime batch is invalid")
    if (
        inner.get("task_name") != task_name
        or inner.get("base_seed") != raw.get("base_seed")
        or inner.get("episodes") != raw.get("episodes")
        or inner.get("variation_schedule") != raw.get("variation_schedule")
    ):
        raise ValueError("task-scoped inner batch identity is inconsistent")
    available: dict[str, Callable[[dict[str, Any]], list[Any]]] = {
        LEGACY_RUNTIME_BATCH_LOADER: load_staged_motion_plan_batch,
        SELECTIVE_COMPOSED_RUNTIME_BATCH_LOADER: (
            load_selective_composed_staged_motion_plan_batch
        ),
        **_builtin_v4_runtime_loaders(),
    }
    if runtime_loaders is not None:
        available.update(runtime_loaders)
    loader_id = raw.get("runtime_loader")
    loader = available.get(loader_id)
    if loader is None:
        raise ValueError(f"unsupported task-scoped runtime loader: {loader_id}")
    plans = loader(inner)
    if not isinstance(plans, list) or len(plans) != raw.get("episodes"):
        raise ValueError("task-scoped runtime loader returned an invalid plan count")
    # ``identity`` was authenticated above. Keeping the local variable makes
    # this explicit for static checkers and documents the admission sequence.
    if identity["scope"] != task_name:  # pragma: no cover - defensive
        raise ValueError("task-scoped identity scope changed during loading")
    return plans


def validate_training_data_binding(
    binding: Any,
    *,
    verify_file: bool,
) -> dict[str, Any]:
    """Authenticate the single current training-manifest reference.

    Evaluation inputs do not own or duplicate training inventory bytes.  This
    small binding replaces the former V1/Store special-case composite and can
    be checked without importing training code.
    """

    expected_fields = {
        "schema",
        "path",
        "sha256",
        "fingerprint",
        "training_episode_count",
        "training_file_count",
        "training_model_group_count",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != expected_fields
        or binding.get("schema") != TRAINING_DATA_BINDING_SCHEMA
        or binding.get("path") != "data/training/manifest.json"
        or not _is_sha256(binding.get("sha256"))
        or not _is_sha256(binding.get("fingerprint"))
        or binding.get("training_episode_count") != 45
        or binding.get("training_file_count") != 125
        or binding.get("training_model_group_count") != 9
    ):
        raise ValueError("evaluation-set training-data binding is invalid")
    if verify_file:
        path = (INTEGRATION_ROOT / binding["path"]).resolve()
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise ValueError("bound training manifest SHA-256 changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("fingerprint") != binding["fingerprint"]
            or payload.get("training_episode_count") != 45
            or payload.get("training_file_count") != 125
            or payload.get("training_model_group_count") != 9
        ):
            raise ValueError("bound training manifest identity changed")
    return binding


def resolve_eval_set_root(eval_set_id: str) -> Path:
    if (
        not isinstance(eval_set_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", eval_set_id) is None
    ):
        raise ValueError("fixed eval-set ID is invalid")
    if eval_set_id != EVALUATION_SET_V2_ID:
        raise ValueError("only the current rlbench_eval_v2 set is materialized")
    return EVAL_SET_ROOT.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_formal_artifact_paths(*, output: Path, models_dir: Path) -> None:
    """Keep sealed inputs, model inputs, and formal outputs disjoint."""

    output_path = Path(output).resolve()
    model_path = Path(models_dir).resolve()
    results_root = (INTEGRATION_ROOT / "results").resolve()
    evaluation_root = EVAL_SET_ROOT.resolve()
    if not _is_within(output_path, results_root):
        raise ValueError("formal evaluation output must be below the results root")
    if _is_within(output_path, evaluation_root):
        raise ValueError("formal output cannot modify a sealed evaluation set")
    if _is_within(model_path, evaluation_root) or _is_within(model_path, results_root):
        raise ValueError("formal model input overlaps evaluation artifacts or results")


def _validate_bound_json(
    *,
    reference: Any,
    path: Path,
    expected_schema: str,
) -> dict[str, Any]:
    if not isinstance(reference, dict) or set(reference) != {"sha256", "fingerprint"}:
        raise ValueError("fixed eval-set JSON reference is invalid")
    if file_sha256(path) != reference["sha256"]:
        raise ValueError(f"fixed eval-set SHA-256 mismatch: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"fixed eval-set JSON must be an object: {path.name}")
    if (
        payload.get("schema") != expected_schema
        or payload.get("fingerprint") != reference["fingerprint"]
    ):
        raise ValueError(f"fixed eval-set fingerprint mismatch: {path.name}")
    return payload


def _artifact_path(root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("fixed eval-set artifact path is invalid")
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("fixed eval-set artifact escapes its canonical root")
    return resolved


def _load_v2_manifest(eval_set_id: str) -> tuple[Path, dict[str, Any]]:
    if eval_set_id != EVALUATION_SET_V2_ID:
        raise ValueError("evaluation-set v2 ID must be rlbench_eval_v2")
    root = resolve_eval_set_root(eval_set_id)
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation-set v2 manifest must be a JSON object")
    expected_fields = {
        "schema",
        "protocol_id",
        "evaluation_set_id",
        "spec",
        "legacy_source_manifest",
        "training_identity",
        "environment_plan_batches",
        "coordination_source_batch",
        "sealed_without_evaluation_results",
        "fingerprint",
    }
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if (
        set(payload) != expected_fields
        or payload.get("schema") != EVAL_SET_V2_MANIFEST_SCHEMA
        or payload.get("protocol_id") != EVAL_SET_V2_PROTOCOL_ID
        or payload.get("evaluation_set_id") != EVALUATION_SET_V2_ID
        or payload.get("sealed_without_evaluation_results") is not True
        or payload.get("fingerprint") != canonical_fingerprint(body)
    ):
        raise ValueError("evaluation-set v2 manifest is invalid")
    return path, payload


def _validate_v2_legacy_reference(
    reference: Any,
    *,
    source_artifact_path: str,
    source_manifest_sha256: str,
    source_batch_reference: Mapping[str, Any],
    source_batch_payload: Mapping[str, Any],
) -> None:
    expected_fields = {
        "artifact_origin",
        "source_evaluation_set_id",
        "source_artifact_path",
        "source_manifest_sha256",
        "sha256",
        "batch_schema",
        "protocol_id",
        "batch_fingerprint",
    }
    if (
        not isinstance(reference, dict)
        or set(reference) != expected_fields
        or reference.get("artifact_origin") != "reused_legacy"
        or reference.get("source_evaluation_set_id") != "rlbench_fixed_v1"
        or reference.get("source_artifact_path") != source_artifact_path
        or reference.get("source_manifest_sha256") != source_manifest_sha256
        or reference.get("sha256") != source_batch_reference.get("sha256")
        or reference.get("batch_fingerprint")
        != source_batch_reference.get("batch_fingerprint")
        or reference.get("batch_schema") != source_batch_payload.get("schema")
        or reference.get("protocol_id") != source_batch_payload.get("protocol_id")
    ):
        raise ValueError("evaluation-set v2 legacy reference is invalid")


def _load_v2_materialized_legacy_batch(
    *,
    root: Path,
    task: str,
    profile: Mapping[str, Any],
    reference: Any,
) -> dict[str, Any]:
    """Load byte-identical V1-origin bytes from the current local set."""

    expected_fields = {
        "artifact_origin",
        "artifact_path",
        "source_evaluation_set_id",
        "source_artifact_path",
        "source_manifest_sha256",
        "sha256",
        "batch_schema",
        "protocol_id",
        "batch_fingerprint",
    }
    if (
        not isinstance(reference, dict)
        or set(reference) != expected_fields
        or reference.get("artifact_origin") != "reused_legacy"
        or reference.get("artifact_path") != profile.get("artifact_path")
        or reference.get("source_evaluation_set_id") != "rlbench_fixed_v1"
        or reference.get("source_artifact_path")
        != profile.get("legacy_source_artifact_path")
        or not _is_sha256(reference.get("source_manifest_sha256"))
    ):
        raise ValueError(
            f"evaluation-set materialized legacy reference is invalid: {task}"
        )
    path = _artifact_path(root, profile["artifact_path"])
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise ValueError(
            f"evaluation-set materialized batch SHA-256 is invalid: {task}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    plans = load_staged_motion_plan_batch(payload)
    if (
        payload.get("schema") != reference.get("batch_schema")
        or payload.get("protocol_id") != reference.get("protocol_id")
        or payload.get("batch_fingerprint") != reference.get("batch_fingerprint")
    ):
        raise ValueError(
            f"evaluation-set materialized batch identity is invalid: {task}"
        )
    return {
        "path": path,
        "payload": payload,
        "plans": plans,
        "artifact_origin": "reused_legacy",
        "source_evaluation_set_id": "rlbench_fixed_v1",
    }


def _load_v2_regenerated_batch(
    *,
    root: Path,
    task: str,
    profile: Mapping[str, Any],
    reference: Any,
    runtime_loaders: Mapping[
        str,
        Callable[[dict[str, Any]], list[Any]],
    ]
    | None,
) -> dict[str, Any]:
    expected_reference_fields = {
        "artifact_origin",
        "artifact_path",
        "sha256",
        "batch_schema",
        "protocol_id",
        "batch_fingerprint",
        "task_identity_fingerprint",
        "runtime_loader",
    }
    if (
        not isinstance(reference, dict)
        or set(reference) != expected_reference_fields
        or reference.get("artifact_origin") != "regenerated_v2"
        or reference.get("artifact_path") != profile.get("artifact_path")
        or (
            task == OPEN_MICROWAVE_TASK
            and reference.get("runtime_loader")
            != SELECTIVE_COMPOSED_RUNTIME_BATCH_LOADER
        )
    ):
        raise ValueError(f"evaluation-set v2 regenerated reference is invalid: {task}")
    path = _artifact_path(root, profile["artifact_path"])
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise ValueError(
            f"evaluation-set v2 regenerated batch SHA-256 is invalid: {task}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    plans = load_task_scoped_plan_batch(payload, runtime_loaders=runtime_loaders)
    identity = payload.get("task_identity", {})
    if (
        payload.get("schema") != reference.get("batch_schema")
        or payload.get("protocol_id") != reference.get("protocol_id")
        or payload.get("batch_fingerprint")
        != reference.get("batch_fingerprint")
        or identity.get("fingerprint")
        != reference.get("task_identity_fingerprint")
        or payload.get("runtime_loader") != reference.get("runtime_loader")
    ):
        raise ValueError(
            f"evaluation-set v2 regenerated batch identity is invalid: {task}"
        )
    return {"path": path, "payload": payload, "plans": plans}


def _expected_variation_schedule(
    profile: Mapping[str, Any],
    *,
    episodes: int,
) -> list[int]:
    schedule = profile["evaluation_variation_schedule"]
    if schedule["kind"] == "fixed":
        return [int(schedule["value"])] * episodes
    return [
        episode % int(profile["task_variation_count"])
        for episode in range(episodes)
    ]


def load_evaluation_set_v2_manifest(
    eval_set_id: str = EVALUATION_SET_V2_ID,
    *,
    selected_task: str | None = None,
    full_preflight: bool = False,
    verify_training_files: bool = False,
    runtime_loaders: Mapping[
        str,
        Callable[[dict[str, Any]], list[Any]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Load the single self-contained V2 set from ``data/evaluation``."""

    root = resolve_eval_set_root(eval_set_id)
    manifest_path, manifest = _load_v2_manifest(eval_set_id)
    spec_path = root / "spec.json"
    _validate_bound_json(
        reference=manifest.get("spec"),
        path=spec_path,
        expected_schema=EVALUATION_SET_V2_SPEC_SCHEMA,
    )
    spec = load_evaluation_set_v2_spec(spec_path)
    episodes = int(spec["episode_count_per_task"])

    source_contract = spec["legacy_import"]
    source_reference = manifest.get("legacy_source_manifest")
    if (
        not isinstance(source_reference, dict)
        or set(source_reference)
        != {"evaluation_set_id", "manifest_sha256", "manifest_fingerprint", "access"}
        or source_reference.get("evaluation_set_id")
        != source_contract["source_evaluation_set_id"]
        or source_reference.get("manifest_sha256")
        != source_contract["source_manifest_sha256"]
        or source_reference.get("manifest_fingerprint")
        != source_contract["source_manifest_fingerprint"]
        or source_reference.get("access")
        != "historical_provenance_no_runtime_dependency"
    ):
        raise ValueError("evaluation-set legacy provenance is invalid")
    training_identity = validate_training_data_binding(
        manifest.get("training_identity"),
        verify_file=verify_training_files,
    )

    references = manifest.get("environment_plan_batches")
    if not isinstance(references, dict) or frozenset(references) != ENVIRONMENT_TASKS:
        raise ValueError("evaluation-set v2 environment task set is incomplete")
    for task, profile in spec["dynamic_environment"].items():
        reference = references[task]
        if not isinstance(reference, dict) or reference.get("artifact_origin") != profile.get(
            "artifact_origin"
        ):
            raise ValueError(f"evaluation-set v2 artifact origin is invalid: {task}")
        if reference.get("artifact_path") != profile.get("artifact_path"):
            raise ValueError(f"evaluation-set local artifact path is invalid: {task}")
        if profile["artifact_origin"] == "reused_legacy":
            if (
                reference.get("source_evaluation_set_id")
                != source_contract["source_evaluation_set_id"]
                or reference.get("source_artifact_path")
                != profile["legacy_source_artifact_path"]
                or reference.get("source_manifest_sha256")
                != source_contract["source_manifest_sha256"]
            ):
                raise ValueError(f"evaluation-set legacy provenance differs: {task}")
        elif (
            reference.get("batch_schema") != TASK_SCOPED_PLAN_BATCH_SCHEMA
            or reference.get("protocol_id") != TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID
        ):
            raise ValueError(f"evaluation-set regenerated binding is invalid: {task}")

    tasks_to_load = (
        sorted(ENVIRONMENT_TASKS)
        if full_preflight
        else [selected_task]
        if selected_task is not None
        else []
    )
    if any(task not in ENVIRONMENT_TASKS for task in tasks_to_load):
        raise ValueError("requested task is absent from evaluation-set v2")

    loaded_batches: dict[str, dict[str, Any]] = {}
    for task in tasks_to_load:
        profile = spec["dynamic_environment"][task]
        reference = references[task]
        if profile["artifact_origin"] == "reused_legacy":
            selected = _load_v2_materialized_legacy_batch(
                root=root,
                task=task,
                profile=profile,
                reference=reference,
            )
        else:
            selected = _load_v2_regenerated_batch(
                root=root,
                task=task,
                profile=profile,
                reference=reference,
                runtime_loaders=runtime_loaders,
            )
            selected = {**selected, "artifact_origin": "regenerated_v2"}
        payload = selected["payload"]
        if (
            payload.get("task_name") != task
            or payload.get("base_seed") != GLOBAL_EVAL_SEED_START
            or payload.get("episodes") != episodes
            or payload.get("variation_schedule")
            != _expected_variation_schedule(profile, episodes=episodes)
            or len(selected["plans"]) != episodes
        ):
            raise ValueError(f"evaluation-set batch schedule is invalid: {task}")
        loaded_batches[task] = selected

    coordination_reference = manifest.get("coordination_source_batch")
    coordination_profile = spec["coordination"][COORDINATION_TASK]
    expected_coordination_fields = {
        "artifact_origin",
        "artifact_path",
        "source_evaluation_set_id",
        "source_artifact_path",
        "source_manifest_sha256",
        "sha256",
        "batch_schema",
        "protocol_id",
        "batch_fingerprint",
    }
    if (
        not isinstance(coordination_reference, dict)
        or set(coordination_reference) != expected_coordination_fields
        or coordination_reference.get("artifact_origin") != "reused_legacy"
        or coordination_reference.get("artifact_path")
        != coordination_profile["artifact_path"]
        or coordination_reference.get("source_evaluation_set_id")
        != source_contract["source_evaluation_set_id"]
        or coordination_reference.get("source_artifact_path")
        != coordination_profile["legacy_source_artifact_path"]
        or coordination_reference.get("source_manifest_sha256")
        != source_contract["source_manifest_sha256"]
    ):
        raise ValueError("evaluation-set coordination reference is invalid")
    coordination_path = _artifact_path(root, coordination_profile["artifact_path"])
    coordination_payload = None
    coordination_plans = None
    if selected_task is None or full_preflight:
        if (
            not coordination_path.is_file()
            or file_sha256(coordination_path) != coordination_reference["sha256"]
        ):
            raise ValueError("evaluation-set coordination SHA-256 is invalid")
        coordination_payload = json.loads(coordination_path.read_text(encoding="utf-8"))
        coordination_plans = load_staged_source_plan_batch(coordination_payload)
        if (
            coordination_payload.get("schema") != coordination_reference["batch_schema"]
            or coordination_payload.get("protocol_id")
            != coordination_reference["protocol_id"]
            or coordination_payload.get("batch_fingerprint")
            != coordination_reference["batch_fingerprint"]
            or coordination_payload.get("base_seed") != GLOBAL_EVAL_SEED_START
            or coordination_payload.get("episodes") != episodes
            or coordination_payload.get("variation_schedule")
            != [episode % 5 for episode in range(episodes)]
            or len(coordination_plans) != episodes
        ):
            raise ValueError("evaluation-set coordination identity is invalid")

    return {
        "manifest_version": 2,
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "payload": manifest,
        "spec": spec,
        "training_identity": training_identity,
        "training_split": training_identity,
        "legacy_source_manifest": source_reference,
        "environment_batches": loaded_batches,
        "coordination_source_batch": {
            **coordination_reference,
            "resolved_path": coordination_path,
            "payload": coordination_payload,
            "plans": coordination_plans,
            "artifact_origin": "reused_legacy",
        },
    }


def load_fixed_eval_set_manifest(
    eval_set_id: str,
    *,
    selected_task: str | None = None,
    full_preflight: bool = False,
    verify_training_files: bool = False,
    runtime_loaders: Mapping[
        str,
        Callable[[dict[str, Any]], list[Any]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Load the sole materialized current evaluation set."""

    if eval_set_id != EVALUATION_SET_V2_ID:
        raise ValueError("only the current rlbench_eval_v2 set is supported")
    return load_evaluation_set_v2_manifest(
        eval_set_id,
        selected_task=selected_task,
        full_preflight=full_preflight,
        verify_training_files=verify_training_files,
        runtime_loaders=runtime_loaders,
    )


def fixed_environment_plans(
    eval_set_id: str,
    task: str,
    *,
    runtime_loaders: Mapping[
        str,
        Callable[[dict[str, Any]], list[Any]],
    ]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_fixed_eval_set_manifest(
        eval_set_id,
        selected_task=task,
        runtime_loaders=runtime_loaders,
    )
    return manifest, manifest["environment_batches"][task]


def fixed_coordination_sources(eval_set_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_fixed_eval_set_manifest(eval_set_id)
    return manifest, manifest["coordination_source_batch"]


def build_coordination_source_batch(eval_set_id: str, *, headless: bool = True) -> Path:
    """Offline-only builder for the preregistered dynamic-HandOver A batch."""

    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import _make_action_mode
    from integrations.rlbench.rlbench_dynamac.core.records import atomic_json, reserve_output

    root = resolve_eval_set_root(eval_set_id)
    spec = load_evaluation_set_v2_spec(root / "spec.json")
    profile = spec["coordination"][COORDINATION_TASK]
    output = _artifact_path(root, profile["artifact_path"])
    module_name = "rlbench.bimanual_tasks.bimanual_handover_item_dynamic"
    class_name = "BimanualHandoverItemDynamic"
    import importlib

    task_class = getattr(importlib.import_module(module_name), class_name)
    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    environment = Environment(
        action_mode=_make_action_mode(),
        obs_config=observation_config,
        headless=headless,
        robot_setup="dual_panda",
    )
    plans = []
    variations = [episode % 5 for episode in range(FIXED_EVAL_EPISODES)]
    launched = False
    with reserve_output(output):
        try:
            environment.launch()
            launched = True
            for episode, variation in enumerate(variations):
                plans.append(
                    stage_source_plan(
                        environment,
                        task_class,
                        task_name=COORDINATION_TASK,
                        episode_seed=GLOBAL_EVAL_SEED_START + episode,
                        variation=variation,
                    )
                )
                print(
                    f"staged coordination A {episode + 1}/{FIXED_EVAL_EPISODES}",
                    flush=True,
                )
        finally:
            if launched:
                environment.shutdown()
        payload = staged_source_plan_batch(
            task_name=COORDINATION_TASK,
            task_module=module_name,
            task_class=class_name,
            base_seed=GLOBAL_EVAL_SEED_START,
            variations=variations,
            plans=plans,
        )
        atomic_json(output, payload)
    return output


def seal_evaluation_set_v2(
    eval_set_id: str = EVALUATION_SET_V2_ID,
    *,
    runtime_loaders: Mapping[
        str,
        Callable[[dict[str, Any]], list[Any]],
    ]
    | None = None,
) -> Path:
    """Authenticate the already materialized local V2 seal.

    Evaluation inputs are now published directly under ``data/evaluation``;
    there is no external V1 import or draft-to-seal builder workflow.
    """

    loaded = load_evaluation_set_v2_manifest(
        eval_set_id,
        full_preflight=True,
        verify_training_files=True,
        runtime_loaders=runtime_loaders,
    )
    return loaded["manifest_path"]


def seal_fixed_eval_set(eval_set_id: str) -> Path:
    """Compatibility entry point for authenticating the current local seal."""

    if eval_set_id != EVALUATION_SET_V2_ID:
        raise ValueError("only the current rlbench_eval_v2 set is supported")
    return seal_evaluation_set_v2(eval_set_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    coordination = subparsers.add_parser("build-coordination")
    coordination.add_argument("--eval-set-id", required=True)
    display = coordination.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    coordination.set_defaults(headless=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--eval-set-id", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--eval-set-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-coordination":
        print(build_coordination_source_batch(args.eval_set_id, headless=args.headless))
    elif args.command == "seal":
        print(seal_fixed_eval_set(args.eval_set_id))
    else:
        load_fixed_eval_set_manifest(
            args.eval_set_id,
            full_preflight=True,
            verify_training_files=True,
        )
        print("fixed evaluation set preflight passed")
    return 0


__all__ = [
    "COORDINATION_TASK",
    "EVAL_SET_ROOT",
    "EVAL_SET_V2_MANIFEST_SCHEMA",
    "EVAL_SET_V2_PROTOCOL_ID",
    "ENVIRONMENT_TASKS",
    "FIXED_EVAL_EPISODES",
    "GLOBAL_EVAL_SEED_START",
    "LEGACY_RUNTIME_BATCH_LOADER",
    "OPEN_MICROWAVE_TASK",
    "SELECTIVE_COMPOSED_RUNTIME_BATCH_LOADER",
    "SELECTIVE_COMPOSITION_PROVENANCE_SCHEMA",
    "SELECTIVE_COMPOSITION_PROTOCOL_ID",
    "TRAINING_DATA_BINDING_SCHEMA",
    "TASK_SCOPED_IDENTITY_SCHEMA",
    "TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID",
    "TASK_SCOPED_PLAN_BATCH_SCHEMA",
    "TASK_SCOPED_REBIND_PROVENANCE_SCHEMA",
    "build_open_microwave_task_scoped_plan_batch",
    "build_selective_composition_provenance",
    "build_task_scoped_identity",
    "build_task_scoped_plan_batch",
    "canonical_fingerprint",
    "file_sha256",
    "fixed_environment_plans",
    "fixed_coordination_sources",
    "load_fixed_eval_set_manifest",
    "load_evaluation_set_v2_manifest",
    "load_selective_composed_staged_motion_plan_batch",
    "load_task_scoped_plan_batch",
    "resolve_eval_set_root",
    "seal_evaluation_set_v2",
    "seal_fixed_eval_set",
    "validate_formal_artifact_paths",
    "validate_selective_composition_against_base",
    "validate_training_data_binding",
]


if __name__ == "__main__":
    raise SystemExit(main())
