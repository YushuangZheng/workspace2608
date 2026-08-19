"""Read-only authentication for the version-neutral fixed RLBench eval set."""

from __future__ import annotations

import hashlib
import json
import re
import argparse
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from .evaluation_split import (
    EVALUATION_SET_SPEC_SCHEMA,
    EVALUATION_SET_V2_CONFIG_SPEC,
    EVALUATION_SET_V2_ID,
    EVALUATION_SET_V2_SPEC_SCHEMA,
    TRAINING_SPLIT_SCHEMA,
    load_evaluation_set_spec,
    load_evaluation_set_v2_spec,
    load_training_split_manifest,
    validate_fixed_evaluation_split,
)
from .runtime import (
    load_staged_motion_plan_batch,
    load_staged_source_plan_batch,
    stage_source_plan,
    staged_source_plan_batch,
)

FIXED_EVAL_SET_MANIFEST_SCHEMA = "dynamac-rlbench-sealed-evaluation-manifest-v1"
FIXED_EVAL_SET_PROTOCOL_ID = "rlbench-version-neutral-fixed-eval-set-v1"
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
STORE_TRAINING_IDENTITY_SCHEMA = (
    "dynamac-rlbench-store-bottle-eval-binding-v4"
)
COMPOSITE_TRAINING_IDENTITY_SCHEMA = (
    "dynamac-rlbench-composite-training-identity-v2"
)
LEGACY_RUNTIME_BATCH_LOADER = "staged_motion_plan_batch_v3_4"
GLOBAL_EVAL_SEED_START = 2_608_000_000
FIXED_EVAL_EPISODES = 200
EVAL_SET_ROOT = Path(__file__).resolve().parents[1] / "evaluation_sets"
INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
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
LEGACY_V4_MODEL_TASKS = (
    "bimanual_handover_item",
    "bimanual_handover_item_dynamic",
    "bimanual_lift_tray",
    "bimanual_sweep_to_dustpan",
    "open_microwave",
    "place_cups",
    "stack_wine",
    "wipe_desk",
)


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

    from .store_bottle_eval_v4 import v4_store_runtime_loaders
    from .v4_dynamic_protocol import v4_runtime_loaders

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
    return {**body, "batch_fingerprint": canonical_fingerprint(body)}


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


def _relative_scoped_file(
    path: Path,
    *,
    root: Path,
    expected_name: str | None = None,
) -> tuple[Path, str]:
    resolved = Path(path).resolve()
    scoped_root = Path(root).resolve()
    try:
        relative_to_scope = resolved.relative_to(scoped_root)
        relative_to_integration = resolved.relative_to(INTEGRATION_ROOT.resolve())
    except ValueError as error:
        raise ValueError("training identity file escapes its V4 scope") from error
    if expected_name is not None and relative_to_scope.as_posix() != expected_name:
        raise ValueError("training identity file has an unexpected scoped path")
    if not resolved.is_file():
        raise ValueError(f"training identity file does not exist: {resolved}")
    return resolved, relative_to_integration.as_posix()


def _load_fingerprinted_object(path: Path, *, expected_schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"training identity schema is invalid: {path.name}")
    fingerprint = payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if fingerprint != canonical_fingerprint(body):
        raise ValueError(f"training identity fingerprint is invalid: {path.name}")
    return payload


def _collection_episode_low_dim_record(
    episode: Mapping[str, Any],
) -> Mapping[str, Any]:
    files = episode.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("StoreBottle collection episode has no file inventory")
    record = files.get("low_dim_obs")
    if not isinstance(record, Mapping):
        # Accept the explicit filename key used by some collection tools while
        # normalizing the sealed identity to one stable role name.
        record = files.get("low_dim_obs.pkl")
    if not isinstance(record, Mapping):
        raise ValueError("StoreBottle collection episode lacks low_dim_obs.pkl")
    return record


def build_store_training_identity(
    *,
    collection_manifest_path: Path,
    model_release_manifest_path: Path,
) -> dict[str, Any]:
    """Bind the five new StoreBottle demos and the completed V4 model release."""

    collection_root = INTEGRATION_ROOT / "data" / "v4" / "store_bottle"
    collection_path, collection_relative = _relative_scoped_file(
        Path(collection_manifest_path),
        root=collection_root,
        expected_name="collection_manifest.json",
    )
    collection = _load_fingerprinted_object(
        collection_path,
        expected_schema="rlbench-store-bottle-v4-static-demonstrations-v1",
    )
    episodes = collection.get("episodes")
    if (
        collection.get("demonstrations") != 5
        or collection.get("base_seed") != 4_104_000_000
        or collection.get("variation") != 0
        or not isinstance(episodes, list)
        or len(episodes) != 5
    ):
        raise ValueError("StoreBottle collection identity is invalid")
    demo_records = []
    expected_seeds = list(range(4_104_000_000, 4_104_000_005))
    for expected_episode, (episode, expected_seed) in enumerate(
        zip(episodes, expected_seeds)
    ):
        if (
            not isinstance(episode, Mapping)
            or episode.get("episode") != expected_episode
            or episode.get("seed") != expected_seed
            or episode.get("variation") != 0
            or episode.get("success_verified") is not True
        ):
            raise ValueError("StoreBottle collection episode identity is invalid")
        low_dim = _collection_episode_low_dim_record(episode)
        relative = low_dim.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(low_dim.get("bytes"), bool)
            or not isinstance(low_dim.get("bytes"), int)
            or low_dim["bytes"] < 1
            or not _is_sha256(low_dim.get("sha256"))
        ):
            raise ValueError("StoreBottle low-dimensional demo record is invalid")
        demo_path = (collection_root / relative).resolve()
        try:
            demo_path.relative_to(collection_root.resolve())
        except ValueError as error:
            raise ValueError(
                "StoreBottle demo path escapes its collection root"
            ) from error
        if (
            not demo_path.is_file()
            or demo_path.stat().st_size != low_dim["bytes"]
            or file_sha256(demo_path) != low_dim["sha256"]
        ):
            raise ValueError("StoreBottle demo file hash is invalid")
        demo_records.append(
            {
                "episode": expected_episode,
                "seed": expected_seed,
                "variation": 0,
                "path": (
                    demo_path.relative_to(INTEGRATION_ROOT.resolve()).as_posix()
                ),
                "bytes": low_dim["bytes"],
                "sha256": low_dim["sha256"],
            }
        )

    model_root = INTEGRATION_ROOT / "models" / "v4"
    model_path, model_relative = _relative_scoped_file(
        Path(model_release_manifest_path),
        root=model_root,
        expected_name="release_manifest.json",
    )
    model_release = _load_fingerprinted_object(
        model_path,
        expected_schema="dynamac-rlbench-model-release-manifest-v4",
    )
    entries = model_release.get("entries")
    store_entries = (
        [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("model_id") == "bimanual_put_bottle_in_fridge"
        ]
        if isinstance(entries, list)
        else []
    )
    if (
        model_release.get("complete") is not True
        or model_release.get("status") != "verified"
        or len(store_entries) != 1
        or store_entries[0].get("mode") != "retrained_store_bottle_v4"
        or store_entries[0].get("target_inventory", {}).get("complete") is not True
    ):
        raise ValueError("StoreBottle V4 model release is incomplete")
    artifacts = store_entries[0]["target_inventory"].get("artifacts")
    artifact_names = (
        {
            record.get("name")
            for record in artifacts
            if isinstance(record, dict)
            and isinstance(record.get("bytes"), int)
            and record["bytes"] > 0
            and _is_sha256(record.get("sha256"))
        }
        if isinstance(artifacts, list)
        else set()
    )
    if artifact_names != {"left.npz", "right.npz", "training.json"}:
        raise ValueError("StoreBottle V4 model inventory is invalid")

    evaluation_seed_range = [GLOBAL_EVAL_SEED_START, GLOBAL_EVAL_SEED_START + 199]
    if any(
        evaluation_seed_range[0] <= seed <= evaluation_seed_range[1]
        for seed in expected_seeds
    ):
        raise ValueError("StoreBottle training seeds overlap evaluation seeds")
    body = {
        "schema": STORE_TRAINING_IDENTITY_SCHEMA,
        "task": "bimanual_put_bottle_in_fridge",
        "collection_manifest": {
            "path": collection_relative,
            "sha256": file_sha256(collection_path),
            "fingerprint": collection["fingerprint"],
        },
        "demonstrations": demo_records,
        "model_release_manifest": {
            "path": model_relative,
            "sha256": file_sha256(model_path),
            "fingerprint": model_release["fingerprint"],
        },
        "evaluation_seed_range": evaluation_seed_range,
        "training_evaluation_seed_ranges_disjoint": True,
    }
    return {**body, "fingerprint": canonical_fingerprint(body)}


def _resolve_bound_training_file(relative: Any, *, required_root: Path) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("bound training path must be a relative path")
    path = (INTEGRATION_ROOT / relative).resolve()
    try:
        path.relative_to(required_root.resolve())
    except ValueError as error:
        raise ValueError("bound training path escapes its declared root") from error
    return path


def validate_store_training_identity(
    identity: Any,
    *,
    verify_files: bool,
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("StoreBottle training identity must be an object")
    expected_fields = {
        "schema",
        "task",
        "collection_manifest",
        "demonstrations",
        "model_release_manifest",
        "evaluation_seed_range",
        "training_evaluation_seed_ranges_disjoint",
        "fingerprint",
    }
    body = {key: value for key, value in identity.items() if key != "fingerprint"}
    demos = identity.get("demonstrations")
    if (
        set(identity) != expected_fields
        or identity.get("schema") != STORE_TRAINING_IDENTITY_SCHEMA
        or identity.get("task") != "bimanual_put_bottle_in_fridge"
        or identity.get("evaluation_seed_range")
        != [GLOBAL_EVAL_SEED_START, GLOBAL_EVAL_SEED_START + 199]
        or identity.get("training_evaluation_seed_ranges_disjoint") is not True
        or identity.get("fingerprint") != canonical_fingerprint(body)
        or not isinstance(demos, list)
        or len(demos) != 5
    ):
        raise ValueError("StoreBottle training identity authentication failed")
    expected_seeds = list(range(4_104_000_000, 4_104_000_005))
    for episode, (record, seed) in enumerate(zip(demos, expected_seeds)):
        if (
            not isinstance(record, dict)
            or set(record)
            != {"episode", "seed", "variation", "path", "bytes", "sha256"}
            or record.get("episode") != episode
            or record.get("seed") != seed
            or record.get("variation") != 0
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 1
            or not _is_sha256(record.get("sha256"))
        ):
            raise ValueError("StoreBottle training demonstration binding is invalid")
        path = _resolve_bound_training_file(
            record["path"],
            required_root=INTEGRATION_ROOT / "data" / "v4" / "store_bottle",
        )
        if verify_files and (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError("StoreBottle bound demonstration hash changed")
    for field, root, name in (
        (
            "collection_manifest",
            INTEGRATION_ROOT / "data" / "v4" / "store_bottle",
            "collection_manifest.json",
        ),
        (
            "model_release_manifest",
            INTEGRATION_ROOT / "models" / "v4",
            "release_manifest.json",
        ),
    ):
        reference = identity.get(field)
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256", "fingerprint"}
            or not _is_sha256(reference.get("sha256"))
            or not _is_sha256(reference.get("fingerprint"))
        ):
            raise ValueError(f"StoreBottle {field} binding is invalid")
        path = _resolve_bound_training_file(reference["path"], required_root=root)
        if path.name != name:
            raise ValueError(f"StoreBottle {field} path is invalid")
        if verify_files and (
            not path.is_file() or file_sha256(path) != reference["sha256"]
        ):
            raise ValueError(f"StoreBottle {field} hash changed")
    return identity


def build_composite_training_identity(
    *,
    legacy_manifest: Mapping[str, Any],
    legacy_manifest_sha256: str,
    store_training_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Scope V1 training to inherited models and Store V4 to Store only."""

    training_reference = legacy_manifest.get("training_split_manifest")
    if (
        not isinstance(training_reference, Mapping)
        or set(training_reference) != {"sha256", "fingerprint"}
        or not _is_sha256(legacy_manifest_sha256)
    ):
        raise ValueError("legacy training-split reference is invalid")
    store = validate_store_training_identity(
        dict(store_training_identity),
        verify_files=False,
    )
    body = {
        "schema": COMPOSITE_TRAINING_IDENTITY_SCHEMA,
        "legacy_non_store": {
            "applies_to_tasks": list(LEGACY_V4_MODEL_TASKS),
            "excluded_tasks": ["bimanual_put_bottle_in_fridge"],
            "source_evaluation_set_id": "rlbench_fixed_v1",
            "source_artifact_path": "training_split_manifest.json",
            "source_manifest_sha256": legacy_manifest_sha256,
            "sha256": training_reference["sha256"],
            "fingerprint": training_reference["fingerprint"],
        },
        "store_bottle_v4": store,
    }
    return {**body, "fingerprint": canonical_fingerprint(body)}


def validate_composite_training_identity(
    identity: Any,
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    verify_training_files: bool,
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("composite training identity must be an object")
    expected_fields = {
        "schema",
        "legacy_non_store",
        "store_bottle_v4",
        "fingerprint",
    }
    body = {key: value for key, value in identity.items() if key != "fingerprint"}
    if (
        set(identity) != expected_fields
        or identity.get("schema") != COMPOSITE_TRAINING_IDENTITY_SCHEMA
        or identity.get("fingerprint") != canonical_fingerprint(body)
    ):
        raise ValueError("composite training identity authentication failed")
    legacy = identity.get("legacy_non_store")
    reference = source_manifest.get("training_split_manifest")
    expected_legacy = {
        "applies_to_tasks": list(LEGACY_V4_MODEL_TASKS),
        "excluded_tasks": ["bimanual_put_bottle_in_fridge"],
        "source_evaluation_set_id": "rlbench_fixed_v1",
        "source_artifact_path": "training_split_manifest.json",
        "source_manifest_sha256": source_manifest_sha256,
        "sha256": reference.get("sha256") if isinstance(reference, Mapping) else None,
        "fingerprint": (
            reference.get("fingerprint")
            if isinstance(reference, Mapping)
            else None
        ),
    }
    if legacy != expected_legacy:
        raise ValueError("composite legacy training identity is invalid")
    source_split = (
        resolve_eval_set_root("rlbench_fixed_v1") / "training_split_manifest.json"
    )
    if (
        not source_split.is_file()
        or file_sha256(source_split) != legacy["sha256"]
    ):
        raise ValueError("composite legacy training split changed")
    validate_store_training_identity(
        identity.get("store_bottle_v4"),
        verify_files=verify_training_files,
    )
    return identity


def resolve_eval_set_root(eval_set_id: str) -> Path:
    if (
        not isinstance(eval_set_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", eval_set_id) is None
    ):
        raise ValueError("fixed eval-set ID is invalid")
    canonical_root = EVAL_SET_ROOT.resolve()
    resolved = (canonical_root / eval_set_id).resolve()
    try:
        resolved.relative_to(canonical_root)
    except ValueError as error:
        raise ValueError("fixed eval-set ID escapes the canonical root") from error
    return resolved


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


def _load_manifest(eval_set_id: str) -> tuple[Path, dict[str, Any]]:
    root = resolve_eval_set_root(eval_set_id)
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixed eval-set manifest must be a JSON object")
    expected_fields = {
        "schema",
        "protocol_id",
        "evaluation_set_id",
        "spec",
        "training_split_manifest",
        "environment_plan_batches",
        "coordination_source_batch",
        "sealed_without_evaluation_results",
        "fingerprint",
    }
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if (
        set(payload) != expected_fields
        or payload.get("schema") != FIXED_EVAL_SET_MANIFEST_SCHEMA
        or payload.get("protocol_id") != FIXED_EVAL_SET_PROTOCOL_ID
        or payload.get("evaluation_set_id") != eval_set_id
        or payload.get("sealed_without_evaluation_results") is not True
        or payload.get("fingerprint") != canonical_fingerprint(body)
    ):
        raise ValueError("fixed eval-set manifest is invalid")
    return path, payload


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


def _load_v1_fixed_eval_set_manifest(
    eval_set_id: str,
    *,
    selected_task: str | None = None,
    full_preflight: bool = False,
    verify_training_files: bool = False,
) -> dict[str, Any]:
    """Validate the seal, then deep-load a selected batch exactly once."""

    root = resolve_eval_set_root(eval_set_id)
    manifest_path, manifest = _load_manifest(eval_set_id)
    spec_path = root / "spec.json"
    _validate_bound_json(
        reference=manifest["spec"],
        path=spec_path,
        expected_schema=EVALUATION_SET_SPEC_SCHEMA,
    )
    spec = load_evaluation_set_spec(spec_path)
    split_path = root / "training_split_manifest.json"
    _validate_bound_json(
        reference=manifest["training_split_manifest"],
        path=split_path,
        expected_schema=TRAINING_SPLIT_SCHEMA,
    )
    training_split = load_training_split_manifest(
        split_path,
        verify_files=verify_training_files,
    )
    references = manifest.get("environment_plan_batches")
    if not isinstance(references, dict) or frozenset(references) != ENVIRONMENT_TASKS:
        raise ValueError("fixed eval-set environment task set is incomplete")
    tasks_to_load = (
        sorted(ENVIRONMENT_TASKS)
        if full_preflight
        else [selected_task] if selected_task is not None else []
    )
    if any(task not in ENVIRONMENT_TASKS for task in tasks_to_load):
        raise ValueError("requested task is absent from the fixed evaluation set")
    loaded_batches: dict[str, dict[str, Any]] = {}
    for task in tasks_to_load:
        reference = references[task]
        if not isinstance(reference, dict) or set(reference) != {
            "sha256",
            "batch_fingerprint",
        }:
            raise ValueError(f"fixed eval-set reference for {task!r} is invalid")
        profile = spec["dynamic_environment"][task]
        batch_path = _artifact_path(root, profile["artifact_path"])
        if file_sha256(batch_path) != reference["sha256"]:
            raise ValueError(f"fixed eval-set file SHA-256 for {task!r} is invalid")
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        plans = load_staged_motion_plan_batch(payload)
        schedule = profile["evaluation_variation_schedule"]
        expected_schedule = (
            [int(schedule["value"])] * FIXED_EVAL_EPISODES
            if schedule["kind"] == "fixed"
            else [
                episode % profile["task_variation_count"]
                for episode in range(FIXED_EVAL_EPISODES)
            ]
        )
        if (
            payload.get("task_name") != task
            or payload.get("base_seed") != GLOBAL_EVAL_SEED_START
            or payload.get("episodes") != FIXED_EVAL_EPISODES
            or payload.get("variation_schedule") != expected_schedule
            or payload.get("batch_fingerprint") != reference["batch_fingerprint"]
            or len(plans) != FIXED_EVAL_EPISODES
        ):
            raise ValueError(f"fixed eval-set plan batch for {task!r} is invalid")
        loaded_batches[task] = {
            "path": batch_path,
            "payload": payload,
            "plans": plans,
        }
    coordination_ref = manifest.get("coordination_source_batch")
    if not isinstance(coordination_ref, dict) or set(coordination_ref) != {
        "sha256",
        "batch_fingerprint",
    }:
        raise ValueError("fixed coordination source-batch reference is invalid")
    coordination_profile = spec["coordination"][COORDINATION_TASK]
    coordination_path = _artifact_path(root, coordination_profile["artifact_path"])
    coordination_payload = None
    coordination_plans = None
    if selected_task is None or full_preflight:
        if not coordination_path.is_file() or file_sha256(coordination_path) != coordination_ref["sha256"]:
            raise ValueError("fixed coordination source-batch SHA-256 is invalid")
        coordination_payload = json.loads(coordination_path.read_text(encoding="utf-8"))
        coordination_plans = load_staged_source_plan_batch(coordination_payload)
        expected_coord_schedule = [episode % 5 for episode in range(FIXED_EVAL_EPISODES)]
        if (
            coordination_payload.get("task_name") != COORDINATION_TASK
            or coordination_payload.get("base_seed") != GLOBAL_EVAL_SEED_START
            or coordination_payload.get("episodes") != FIXED_EVAL_EPISODES
            or coordination_payload.get("variation_schedule") != expected_coord_schedule
            or coordination_payload.get("batch_fingerprint")
            != coordination_ref["batch_fingerprint"]
            or len(coordination_plans) != FIXED_EVAL_EPISODES
        ):
            raise ValueError("fixed coordination source batch is invalid")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "payload": manifest,
        "spec": spec,
        "training_split": training_split,
        "environment_batches": loaded_batches,
        "coordination_source_batch": {
            **coordination_ref,
            "resolved_path": coordination_path,
            "payload": coordination_payload,
            "plans": coordination_plans,
        },
    }


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
        "composite_training_identity",
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
    """Load V2 inputs without copying or rewriting legacy sealed batches."""

    root = resolve_eval_set_root(eval_set_id)
    manifest_path, manifest = _load_v2_manifest(eval_set_id)
    spec_path = root / "spec.json"
    spec_reference = manifest.get("spec")
    spec = _validate_bound_json(
        reference=spec_reference,
        path=spec_path,
        expected_schema=EVALUATION_SET_V2_SPEC_SCHEMA,
    )
    # The V2 loader is intentionally stricter than the generic JSON binding:
    # it also enforces origin maps, seed schedules, and the no-result schema.
    spec = load_evaluation_set_v2_spec(spec_path)
    episodes = int(spec["episode_count_per_task"])

    source_contract = spec["legacy_import"]
    source_manifest_reference = manifest.get("legacy_source_manifest")
    expected_source_reference_fields = {
        "evaluation_set_id",
        "manifest_sha256",
        "manifest_fingerprint",
        "access",
    }
    if (
        not isinstance(source_manifest_reference, dict)
        or set(source_manifest_reference) != expected_source_reference_fields
        or source_manifest_reference.get("evaluation_set_id")
        != source_contract["source_evaluation_set_id"]
        or source_manifest_reference.get("manifest_sha256")
        != source_contract["source_manifest_sha256"]
        or source_manifest_reference.get("manifest_fingerprint")
        != source_contract["source_manifest_fingerprint"]
        or source_manifest_reference.get("access")
        != "canonical_read_only_external_reference"
    ):
        raise ValueError("evaluation-set v2 legacy source binding is invalid")
    source_manifest_path, source_manifest_payload = _load_manifest(
        source_contract["source_evaluation_set_id"]
    )
    source_manifest_sha256 = file_sha256(source_manifest_path)
    if (
        source_manifest_sha256 != source_contract["source_manifest_sha256"]
        or source_manifest_payload.get("fingerprint")
        != source_contract["source_manifest_fingerprint"]
    ):
        raise ValueError("evaluation-set v2 legacy source seal changed")
    composite_training_identity = validate_composite_training_identity(
        manifest.get("composite_training_identity"),
        source_manifest=source_manifest_payload,
        source_manifest_sha256=source_manifest_sha256,
        verify_training_files=verify_training_files,
    )

    references = manifest.get("environment_plan_batches")
    if not isinstance(references, dict) or frozenset(references) != ENVIRONMENT_TASKS:
        raise ValueError("evaluation-set v2 environment task set is incomplete")
    for task, profile in spec["dynamic_environment"].items():
        reference = references[task]
        if not isinstance(reference, dict) or reference.get(
            "artifact_origin"
        ) != profile.get("artifact_origin"):
            raise ValueError(f"evaluation-set v2 artifact origin is invalid: {task}")
        if profile["artifact_origin"] == "reused_legacy":
            if (
                reference.get("source_evaluation_set_id")
                != source_contract["source_evaluation_set_id"]
                or reference.get("source_artifact_path")
                != profile["legacy_source_artifact_path"]
                or reference.get("source_manifest_sha256")
                != source_manifest_sha256
            ):
                raise ValueError(
                    f"evaluation-set v2 canonical legacy path is invalid: {task}"
                )
        elif (
            reference.get("artifact_path") != profile["artifact_path"]
            or reference.get("batch_schema") != TASK_SCOPED_PLAN_BATCH_SCHEMA
            or reference.get("protocol_id")
            != TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID
        ):
            raise ValueError(
                f"evaluation-set v2 regenerated binding is invalid: {task}"
            )
    tasks_to_load = (
        sorted(ENVIRONMENT_TASKS)
        if full_preflight
        else [selected_task]
        if selected_task is not None
        else []
    )
    if any(task not in ENVIRONMENT_TASKS for task in tasks_to_load):
        raise ValueError("requested task is absent from evaluation-set v2")

    legacy_tasks_to_load = [
        task
        for task in tasks_to_load
        if spec["dynamic_environment"][task]["artifact_origin"] == "reused_legacy"
    ]
    legacy_full = None
    if full_preflight:
        legacy_full = _load_v1_fixed_eval_set_manifest(
            source_contract["source_evaluation_set_id"],
            full_preflight=True,
            verify_training_files=verify_training_files,
        )
    loaded_batches: dict[str, dict[str, Any]] = {}
    for task in tasks_to_load:
        profile = spec["dynamic_environment"][task]
        reference = references[task]
        if task in legacy_tasks_to_load:
            source = (
                legacy_full
                if legacy_full is not None
                else _load_v1_fixed_eval_set_manifest(
                    source_contract["source_evaluation_set_id"],
                    selected_task=task,
                    verify_training_files=verify_training_files,
                )
            )
            selected = source["environment_batches"][task]
            source_profile = source["spec"]["dynamic_environment"][task]
            if (
                source_profile["artifact_path"]
                != profile["legacy_source_artifact_path"]
            ):
                raise ValueError(
                    f"evaluation-set v2 source path differs from v1 spec: {task}"
                )
            _validate_v2_legacy_reference(
                reference,
                source_artifact_path=source_profile["artifact_path"],
                source_manifest_sha256=source_manifest_sha256,
                source_batch_reference=source_manifest_payload[
                    "environment_plan_batches"
                ][task],
                source_batch_payload=selected["payload"],
            )
            loaded_batches[task] = {
                **selected,
                "artifact_origin": "reused_legacy",
                "source_evaluation_set_id": source_contract[
                    "source_evaluation_set_id"
                ],
            }
        else:
            selected = _load_v2_regenerated_batch(
                root=root,
                task=task,
                profile=profile,
                reference=reference,
                runtime_loaders=runtime_loaders,
            )
            payload = selected["payload"]
            expected_schedule = _expected_variation_schedule(
                profile,
                episodes=episodes,
            )
            if (
                payload.get("task_name") != task
                or payload.get("base_seed") != GLOBAL_EVAL_SEED_START
                or payload.get("episodes") != episodes
                or payload.get("variation_schedule") != expected_schedule
                or len(selected["plans"]) != episodes
            ):
                raise ValueError(
                    f"evaluation-set v2 regenerated batch schedule is invalid: {task}"
                )
            loaded_batches[task] = {
                **selected,
                "artifact_origin": "regenerated_v2",
            }

    coordination_reference = manifest.get("coordination_source_batch")
    coordination_profile = spec["coordination"][COORDINATION_TASK]
    if (
        not isinstance(coordination_reference, dict)
        or coordination_reference.get("artifact_origin") != "reused_legacy"
        or coordination_reference.get("source_evaluation_set_id")
        != source_contract["source_evaluation_set_id"]
        or coordination_reference.get("source_artifact_path")
        != coordination_profile["legacy_source_artifact_path"]
        or coordination_reference.get("source_manifest_sha256")
        != source_manifest_sha256
    ):
        raise ValueError("evaluation-set v2 coordination reference is invalid")
    coordination_payload = None
    coordination_plans = None
    coordination_path = (
        resolve_eval_set_root(source_contract["source_evaluation_set_id"])
        / coordination_profile["legacy_source_artifact_path"]
    ).resolve()
    if selected_task is None or full_preflight:
        source = (
            legacy_full
            if legacy_full is not None
            else _load_v1_fixed_eval_set_manifest(
                source_contract["source_evaluation_set_id"],
                verify_training_files=verify_training_files,
            )
        )
        selected_coordination = source["coordination_source_batch"]
        source_coordination_profile = source["spec"]["coordination"][
            COORDINATION_TASK
        ]
        if (
            source_coordination_profile["artifact_path"]
            != coordination_profile["legacy_source_artifact_path"]
        ):
            raise ValueError("evaluation-set v2 coordination source path changed")
        _validate_v2_legacy_reference(
            coordination_reference,
            source_artifact_path=source_coordination_profile["artifact_path"],
            source_manifest_sha256=source_manifest_sha256,
            source_batch_reference=source_manifest_payload[
                "coordination_source_batch"
            ],
            source_batch_payload=selected_coordination["payload"],
        )
        coordination_path = selected_coordination["resolved_path"]
        coordination_payload = selected_coordination["payload"]
        coordination_plans = selected_coordination["plans"]
    return {
        "manifest_version": 2,
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "payload": manifest,
        "spec": spec,
        "training_split": composite_training_identity,
        "legacy_source_manifest": {
            **source_manifest_reference,
            "resolved_path": source_manifest_path,
        },
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
    """Load either immutable V1 inputs or the V2 task-scoped extension."""

    if eval_set_id == EVALUATION_SET_V2_ID:
        return load_evaluation_set_v2_manifest(
            eval_set_id,
            selected_task=selected_task,
            full_preflight=full_preflight,
            verify_training_files=verify_training_files,
            runtime_loaders=runtime_loaders,
        )
    return _load_v1_fixed_eval_set_manifest(
        eval_set_id,
        selected_task=selected_task,
        full_preflight=full_preflight,
        verify_training_files=verify_training_files,
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

    from .direct_evaluate import _make_action_mode
    from .records import atomic_json, reserve_output

    root = resolve_eval_set_root(eval_set_id)
    spec = load_evaluation_set_spec(root / "spec.json")
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


def _seal_v1_fixed_eval_set(eval_set_id: str) -> Path:
    """Deep-authenticate all preregistered artifacts and atomically seal them."""

    from .records import atomic_json, reserve_output

    root = resolve_eval_set_root(eval_set_id)
    spec_path = root / "spec.json"
    split_path = root / "training_split_manifest.json"
    spec = load_evaluation_set_spec(spec_path)
    training = load_training_split_manifest(split_path, verify_files=False)
    split_evidence = validate_fixed_evaluation_split(
        training_path=split_path,
        spec_path=spec_path,
        verify_training_files=True,
    )
    if split_evidence.get("validated") is not True:
        raise ValueError("fixed training/evaluation split validation failed")
    environment_references = {}
    for task, profile in spec["dynamic_environment"].items():
        path = _artifact_path(root, profile["artifact_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        plans = load_staged_motion_plan_batch(payload)
        schedule = profile["evaluation_variation_schedule"]
        expected_schedule = (
            [schedule["value"]] * FIXED_EVAL_EPISODES
            if schedule["kind"] == "fixed"
            else [
                episode % profile["task_variation_count"]
                for episode in range(FIXED_EVAL_EPISODES)
            ]
        )
        if (
            payload.get("task_name") != task
            or payload.get("base_seed") != GLOBAL_EVAL_SEED_START
            or payload.get("episodes") != FIXED_EVAL_EPISODES
            or payload.get("variation_schedule") != expected_schedule
            or len(plans) != FIXED_EVAL_EPISODES
        ):
            raise ValueError(f"cannot seal invalid environment batch: {task}")
        environment_references[task] = {
            "sha256": file_sha256(path),
            "batch_fingerprint": payload["batch_fingerprint"],
        }
    coord_profile = spec["coordination"][COORDINATION_TASK]
    coord_path = _artifact_path(root, coord_profile["artifact_path"])
    coord_payload = json.loads(coord_path.read_text(encoding="utf-8"))
    coord_plans = load_staged_source_plan_batch(coord_payload)
    if (
        coord_payload.get("task_name") != COORDINATION_TASK
        or coord_payload.get("base_seed") != GLOBAL_EVAL_SEED_START
        or coord_payload.get("variation_schedule")
        != [episode % 5 for episode in range(FIXED_EVAL_EPISODES)]
        or len(coord_plans) != FIXED_EVAL_EPISODES
    ):
        raise ValueError("cannot seal invalid coordination source batch")
    body = {
        "schema": FIXED_EVAL_SET_MANIFEST_SCHEMA,
        "protocol_id": FIXED_EVAL_SET_PROTOCOL_ID,
        "evaluation_set_id": eval_set_id,
        "spec": {
            "sha256": file_sha256(spec_path),
            "fingerprint": spec["fingerprint"],
        },
        "training_split_manifest": {
            "sha256": file_sha256(split_path),
            "fingerprint": training["fingerprint"],
        },
        "environment_plan_batches": environment_references,
        "coordination_source_batch": {
            "sha256": file_sha256(coord_path),
            "batch_fingerprint": coord_payload["batch_fingerprint"],
        },
        "sealed_without_evaluation_results": True,
    }
    manifest = {**body, "fingerprint": canonical_fingerprint(body)}
    output = root / "manifest.json"
    with reserve_output(output):
        atomic_json(output, manifest)
    # One full post-write preflight is the publication gate.
    load_fixed_eval_set_manifest(
        eval_set_id,
        full_preflight=True,
        verify_training_files=True,
    )
    return output


def build_evaluation_set_v2_draft(
    eval_set_id: str = EVALUATION_SET_V2_ID,
    *,
    spec_path: Path = EVALUATION_SET_V2_CONFIG_SPEC,
) -> Path:
    """Create a V2 draft without copying any large legacy plan artifact.

    The only files created are the authenticated V2 spec and empty parent
    directories for the two regenerated task batches.  Every unchanged task
    remains an external, read-only reference to the canonical sealed V1 set.
    """

    if eval_set_id != EVALUATION_SET_V2_ID:
        raise ValueError("evaluation-set v2 draft ID must be rlbench_eval_v2")
    spec = load_evaluation_set_v2_spec(spec_path)
    root = resolve_eval_set_root(eval_set_id)
    if root.exists():
        raise FileExistsError(f"evaluation-set v2 draft already exists: {root}")
    source_id = spec["legacy_import"]["source_evaluation_set_id"]
    source_manifest_path, source_manifest = _load_manifest(source_id)
    if (
        file_sha256(source_manifest_path)
        != spec["legacy_import"]["source_manifest_sha256"]
        or source_manifest.get("fingerprint")
        != spec["legacy_import"]["source_manifest_fingerprint"]
    ):
        raise ValueError("canonical legacy source seal differs from V2 preregistration")
    root.mkdir(parents=False, exist_ok=False)
    destination_spec = root / "spec.json"
    shutil.copyfile(spec_path, destination_spec)
    for task, profile in spec["dynamic_environment"].items():
        if profile["artifact_origin"] != "regenerated_v2":
            continue
        artifact = _artifact_path(root, profile["artifact_path"])
        artifact.parent.mkdir(parents=True, exist_ok=True)
    load_evaluation_set_v2_spec(destination_spec)
    return root


def seal_evaluation_set_v2(
    eval_set_id: str = EVALUATION_SET_V2_ID,
    *,
    store_collection_manifest: Path,
    store_model_release_manifest: Path,
    runtime_loaders: Mapping[
        str,
        Callable[[dict[str, Any]], list[Any]],
    ]
    | None = None,
) -> Path:
    """Seal regenerated V2 batches and canonical V1 references together."""

    from .records import atomic_json, reserve_output

    if eval_set_id != EVALUATION_SET_V2_ID:
        raise ValueError("evaluation-set v2 seal ID must be rlbench_eval_v2")
    root = resolve_eval_set_root(eval_set_id)
    spec_path = root / "spec.json"
    spec = load_evaluation_set_v2_spec(spec_path)
    source_contract = spec["legacy_import"]
    source_manifest_path, source_manifest = _load_manifest(
        source_contract["source_evaluation_set_id"]
    )
    source_manifest_sha256 = file_sha256(source_manifest_path)
    if (
        source_manifest_sha256 != source_contract["source_manifest_sha256"]
        or source_manifest.get("fingerprint")
        != source_contract["source_manifest_fingerprint"]
    ):
        raise ValueError("canonical legacy source seal differs from V2 preregistration")
    # A full V1 preflight is the import admission gate. It authenticates all
    # referenced bytes and their original internal V3.4 identities in place.
    legacy = _load_v1_fixed_eval_set_manifest(
        source_contract["source_evaluation_set_id"],
        full_preflight=True,
        verify_training_files=False,
    )
    store_training_identity = build_store_training_identity(
        collection_manifest_path=store_collection_manifest,
        model_release_manifest_path=store_model_release_manifest,
    )
    composite_training_identity = build_composite_training_identity(
        legacy_manifest=source_manifest,
        legacy_manifest_sha256=source_manifest_sha256,
        store_training_identity=store_training_identity,
    )
    episodes = int(spec["episode_count_per_task"])
    environment_references: dict[str, dict[str, Any]] = {}
    for task, profile in spec["dynamic_environment"].items():
        if profile["artifact_origin"] == "reused_legacy":
            selected = legacy["environment_batches"][task]
            source_profile = legacy["spec"]["dynamic_environment"][task]
            source_reference = source_manifest["environment_plan_batches"][task]
            if (
                source_profile["artifact_path"]
                != profile["legacy_source_artifact_path"]
            ):
                raise ValueError(f"V2 legacy source path changed for {task}")
            environment_references[task] = {
                "artifact_origin": "reused_legacy",
                "source_evaluation_set_id": source_contract[
                    "source_evaluation_set_id"
                ],
                "source_artifact_path": source_profile["artifact_path"],
                "source_manifest_sha256": source_manifest_sha256,
                "sha256": source_reference["sha256"],
                "batch_schema": selected["payload"]["schema"],
                "protocol_id": selected["payload"]["protocol_id"],
                "batch_fingerprint": source_reference["batch_fingerprint"],
            }
            continue
        path = _artifact_path(root, profile["artifact_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        plans = load_task_scoped_plan_batch(
            payload,
            runtime_loaders=runtime_loaders,
        )
        expected_schedule = _expected_variation_schedule(
            profile,
            episodes=episodes,
        )
        identity = payload.get("task_identity", {})
        if (
            payload.get("task_name") != task
            or payload.get("base_seed") != GLOBAL_EVAL_SEED_START
            or payload.get("episodes") != episodes
            or payload.get("variation_schedule") != expected_schedule
            or len(plans) != episodes
            or identity.get("scope") != task
        ):
            raise ValueError(f"cannot seal invalid regenerated V2 batch: {task}")
        environment_references[task] = {
            "artifact_origin": "regenerated_v2",
            "artifact_path": profile["artifact_path"],
            "sha256": file_sha256(path),
            "batch_schema": payload["schema"],
            "protocol_id": payload["protocol_id"],
            "batch_fingerprint": payload["batch_fingerprint"],
            "task_identity_fingerprint": identity["fingerprint"],
            "runtime_loader": payload["runtime_loader"],
        }

    coordination_profile = spec["coordination"][COORDINATION_TASK]
    source_coordination_profile = legacy["spec"]["coordination"][COORDINATION_TASK]
    source_coordination = legacy["coordination_source_batch"]
    source_coordination_reference = source_manifest["coordination_source_batch"]
    if (
        source_coordination_profile["artifact_path"]
        != coordination_profile["legacy_source_artifact_path"]
    ):
        raise ValueError("V2 coordination legacy source path changed")
    coordination_reference = {
        "artifact_origin": "reused_legacy",
        "source_evaluation_set_id": source_contract["source_evaluation_set_id"],
        "source_artifact_path": source_coordination_profile["artifact_path"],
        "source_manifest_sha256": source_manifest_sha256,
        "sha256": source_coordination_reference["sha256"],
        "batch_schema": source_coordination["payload"]["schema"],
        "protocol_id": source_coordination["payload"]["protocol_id"],
        "batch_fingerprint": source_coordination_reference["batch_fingerprint"],
    }
    body = {
        "schema": EVAL_SET_V2_MANIFEST_SCHEMA,
        "protocol_id": EVAL_SET_V2_PROTOCOL_ID,
        "evaluation_set_id": EVALUATION_SET_V2_ID,
        "spec": {
            "sha256": file_sha256(spec_path),
            "fingerprint": spec["fingerprint"],
        },
        "legacy_source_manifest": {
            "evaluation_set_id": source_contract["source_evaluation_set_id"],
            "manifest_sha256": source_manifest_sha256,
            "manifest_fingerprint": source_manifest["fingerprint"],
            "access": "canonical_read_only_external_reference",
        },
        "composite_training_identity": composite_training_identity,
        "environment_plan_batches": environment_references,
        "coordination_source_batch": coordination_reference,
        "sealed_without_evaluation_results": True,
    }
    manifest = {**body, "fingerprint": canonical_fingerprint(body)}
    output = root / "manifest.json"
    with reserve_output(output):
        atomic_json(output, manifest)
    load_evaluation_set_v2_manifest(
        eval_set_id,
        full_preflight=True,
        verify_training_files=False,
        runtime_loaders=runtime_loaders,
    )
    return output


def seal_fixed_eval_set(
    eval_set_id: str,
    *,
    store_collection_manifest: Path | None = None,
    store_model_release_manifest: Path | None = None,
) -> Path:
    """Seal the requested schema while keeping the V1 implementation intact."""

    if eval_set_id == EVALUATION_SET_V2_ID:
        if store_collection_manifest is None or store_model_release_manifest is None:
            raise ValueError(
                "V2 seal requires StoreBottle collection and model release manifests"
            )
        return seal_evaluation_set_v2(
            eval_set_id,
            store_collection_manifest=store_collection_manifest,
            store_model_release_manifest=store_model_release_manifest,
        )
    return _seal_v1_fixed_eval_set(eval_set_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    coordination = subparsers.add_parser("build-coordination")
    coordination.add_argument("--eval-set-id", required=True)
    display = coordination.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    coordination.set_defaults(headless=True)
    draft = subparsers.add_parser("build-v2-draft")
    draft.add_argument("--eval-set-id", default=EVALUATION_SET_V2_ID)
    draft.add_argument(
        "--spec",
        type=Path,
        default=EVALUATION_SET_V2_CONFIG_SPEC,
    )
    seal = subparsers.add_parser("seal")
    seal.add_argument("--eval-set-id", required=True)
    seal.add_argument("--store-collection-manifest", type=Path)
    seal.add_argument("--store-model-release-manifest", type=Path)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--eval-set-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-coordination":
        print(build_coordination_source_batch(args.eval_set_id, headless=args.headless))
    elif args.command == "build-v2-draft":
        print(build_evaluation_set_v2_draft(args.eval_set_id, spec_path=args.spec))
    elif args.command == "seal":
        print(
            seal_fixed_eval_set(
                args.eval_set_id,
                store_collection_manifest=args.store_collection_manifest,
                store_model_release_manifest=args.store_model_release_manifest,
            )
        )
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
    "COMPOSITE_TRAINING_IDENTITY_SCHEMA",
    "EVAL_SET_ROOT",
    "EVAL_SET_V2_MANIFEST_SCHEMA",
    "EVAL_SET_V2_PROTOCOL_ID",
    "ENVIRONMENT_TASKS",
    "FIXED_EVAL_EPISODES",
    "FIXED_EVAL_SET_MANIFEST_SCHEMA",
    "FIXED_EVAL_SET_PROTOCOL_ID",
    "GLOBAL_EVAL_SEED_START",
    "LEGACY_RUNTIME_BATCH_LOADER",
    "STORE_TRAINING_IDENTITY_SCHEMA",
    "TASK_SCOPED_IDENTITY_SCHEMA",
    "TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID",
    "TASK_SCOPED_PLAN_BATCH_SCHEMA",
    "TASK_SCOPED_REBIND_PROVENANCE_SCHEMA",
    "build_composite_training_identity",
    "build_evaluation_set_v2_draft",
    "build_store_training_identity",
    "build_task_scoped_identity",
    "build_task_scoped_plan_batch",
    "canonical_fingerprint",
    "file_sha256",
    "fixed_environment_plans",
    "fixed_coordination_sources",
    "load_fixed_eval_set_manifest",
    "load_evaluation_set_v2_manifest",
    "load_task_scoped_plan_batch",
    "resolve_eval_set_root",
    "seal_evaluation_set_v2",
    "seal_fixed_eval_set",
    "validate_composite_training_identity",
    "validate_formal_artifact_paths",
    "validate_store_training_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
