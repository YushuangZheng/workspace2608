"""V4 model release inventory for retrained and inherited checkpoints.

The API records which checkpoint groups must be retrained and which are
inherited byte-for-byte from V3.  It never copies checkpoints and never starts
training; callers may inspect a pending manifest before any large artifacts
are produced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    STORE_BOTTLE_TASK_NAME,
    load_store_bottle_semantic_spec,
)

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
MODEL_RELEASE_PLAN_PATH = INTEGRATION_ROOT / "configs" / "v4" / "model_release.json"
DEFAULT_V3_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3"
DEFAULT_V4_MODELS_DIR = INTEGRATION_ROOT / "models" / "v4"
MODEL_RELEASE_PLAN_SCHEMA = "dynamac-rlbench-model-release-plan-v4"
MODEL_RELEASE_MANIFEST_SCHEMA = "dynamac-rlbench-model-release-manifest-v4"
RETRAINED_MODE = "retrained_store_bottle_v4"
SWEEP_RETRAINED_MODE = "retrained_sweep_dust_v4"
INHERITED_MODE = "inherited_v3_byte_for_byte"
SWEEP_DUST_TASK_NAME = "bimanual_sweep_to_dustpan"
SWEEP_TRAINING_DATA_SPEC_PATH = (
    INTEGRATION_ROOT
    / "data"
    / "training"
    / "main"
    / SWEEP_DUST_TASK_NAME
    / "augmentation_manifest.json"
)
SWEEP_TRAINING_AUGMENTATION_SCHEMA = (
    "sweep-dust-cartesian-training-augmentation-v1"
)

_EXPECTED_INHERITED_MODELS = frozenset(
    {
        "bimanual_handover_item",
        "bimanual_lift_tray",
        "open_microwave",
        "place_cups",
        "stack_wine",
        "wipe_desk",
        "table_iii/bimanual_handover_item",
    }
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _safe_relative_directory(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{label} must stay inside its release root")
    return path.as_posix()


def load_model_release_plan(
    path: str | Path = MODEL_RELEASE_PLAN_PATH,
) -> dict[str, Any]:
    """Load the closed V4 inventory without touching checkpoint files."""

    plan_path = Path(path)
    plan = _load_object(plan_path)
    if set(plan) != {
        "schema",
        "release",
        "source_release",
        "models_root",
        "copy_or_training_performed_by_manifest_api",
        "entries",
    }:
        raise ValueError("V4 model release plan fields are invalid")
    if (
        plan.get("schema") != MODEL_RELEASE_PLAN_SCHEMA
        or plan.get("release") != "v4"
        or plan.get("source_release") != "v3"
        or plan.get("models_root") != "integrations/rlbench/models/v4"
        or plan.get("copy_or_training_performed_by_manifest_api") is not False
    ):
        raise ValueError("V4 model release plan header is invalid")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != 9:
        raise ValueError("V4 release plan must define exactly nine model groups")

    seen_ids: set[str] = set()
    inherited: set[str] = set()
    retrained: list[str] = []
    sweep_retrained: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "model_id",
            "task",
            "mode",
            "source",
            "target",
            "required_artifacts",
        }:
            raise ValueError("V4 model release entry fields are invalid")
        model_id = _safe_relative_directory(entry["model_id"], "model_id")
        if model_id in seen_ids:
            raise ValueError("V4 model release contains a duplicate model_id")
        seen_ids.add(model_id)
        target = _safe_relative_directory(entry["target"], "target")
        if target != model_id:
            raise ValueError("V4 model targets must equal their model_id")
        artifacts = entry["required_artifacts"]
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or len(set(artifacts)) != len(artifacts)
            or any(
                _safe_relative_directory(item, "required artifact") != item
                or len(Path(item).parts) != 1
                for item in artifacts
            )
        ):
            raise ValueError("V4 required-artifact list is invalid")
        mode = entry["mode"]
        if mode == RETRAINED_MODE:
            if entry["source"] is not None or entry["task"] != STORE_BOTTLE_TASK_NAME:
                raise ValueError("only StoreBottle may be retrained in V4")
            retrained.append(model_id)
        elif mode == SWEEP_RETRAINED_MODE:
            if entry["source"] is not None or entry["task"] != SWEEP_DUST_TASK_NAME:
                raise ValueError("only SweepDust may use its V4 retrained mode")
            sweep_retrained.append(model_id)
        elif mode == INHERITED_MODE:
            source = _safe_relative_directory(entry["source"], "source")
            if source != model_id or entry["task"] == STORE_BOTTLE_TASK_NAME:
                raise ValueError("V4 inherited model identity is invalid")
            inherited.add(model_id)
        else:
            raise ValueError("V4 model release mode is invalid")
    if retrained != [STORE_BOTTLE_TASK_NAME]:
        raise ValueError("V4 must retrain StoreBottle exactly once")
    if sweep_retrained != [SWEEP_DUST_TASK_NAME]:
        raise ValueError("V4 must retrain SweepDust exactly once")
    if inherited != _EXPECTED_INHERITED_MODELS:
        raise ValueError("V4 inherited model set is incomplete")
    return plan


def _inventory(directory: Path, artifacts: list[str]) -> dict[str, Any]:
    records = []
    missing = []
    for name in artifacts:
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        records.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "directory_present": directory.is_dir(),
        "artifacts": records,
        "missing": missing,
        "complete": not missing,
    }


def _hashes(inventory: dict[str, Any]) -> dict[str, str]:
    return {record["name"]: record["sha256"] for record in inventory["artifacts"]}


def _validate_retrained_store_manifest(
    target_dir: Path,
    semantic_fingerprint: str,
) -> None:
    manifest = _load_object(target_dir / "training.json")
    identity = manifest.get("training_identity")
    policy_spec = identity.get("policy_spec") if isinstance(identity, dict) else None
    if (
        manifest.get("manifest_schema") != "dynamac-direct-training-v4"
        or not isinstance(policy_spec, dict)
        or policy_spec.get("semantic_fingerprint") != semantic_fingerprint
        or policy_spec.get("task") != STORE_BOTTLE_TASK_NAME
        or policy_spec.get("frame_names") != ["bottle", "fridge"]
    ):
        raise RuntimeError("V4 StoreBottle checkpoint has the wrong semantic identity")


def _repository_path(relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a repository-relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the repository")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return resolved


def sweep_training_input_identity(
    path: str | Path = SWEEP_TRAINING_DATA_SPEC_PATH,
) -> dict[str, Any]:
    """Authenticate the five augmented SweepDust inputs used for retraining."""

    spec_path = Path(path)
    spec = _load_object(spec_path)
    episodes = spec.get("episodes")
    verification = spec.get("verification")
    identity_at_fit = spec.get("training_identity_at_fit")
    if (
        spec.get("schema") != SWEEP_TRAINING_AUGMENTATION_SCHEMA
        or spec.get("task") != SWEEP_DUST_TASK_NAME
        or spec.get("active_for_next_sweep_retrain") is not True
        or not isinstance(spec.get("status"), str)
        or not spec["status"]
        or not isinstance(episodes, list)
        or len(episodes) != 5
        or [row.get("episode") for row in episodes if isinstance(row, dict)]
        != list(range(5))
        or not isinstance(verification, dict)
        or not isinstance(identity_at_fit, dict)
        or set(identity_at_fit) != {"manifest_path", "manifest_sha256", "data_root"}
        or verification.get("restricted_pickle_loader") != "passed"
        or verification.get("dynamac_bimanual_demo_adapter") != "passed"
        or verification.get("episode_0_unchanged") is not True
    ):
        raise ValueError("SweepDust augmented training-data status is invalid")
    data_root = _repository_path(
        spec.get("training_cli_data_root"),
        "SweepDust training_cli_data_root",
    )
    records = []
    for episode, row in enumerate(episodes):
        expected_sha = row.get("sha256" if episode == 0 else "augmented_sha256")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise ValueError("SweepDust augmented input SHA-256 is invalid")
        relative = (
            Path(SWEEP_DUST_TASK_NAME)
            / "all_variations"
            / "episodes"
            / f"episode{episode}"
            / "low_dim_obs.pkl"
        )
        input_path = data_root / relative
        if not input_path.is_file() or _file_sha256(input_path) != expected_sha:
            raise RuntimeError(
                f"SweepDust augmented episode{episode} differs from its data spec"
            )
        records.append({"episode": episode, "sha256": expected_sha})
    current_manifest_sha256 = _file_sha256(spec_path)
    if not current_manifest_sha256:
        raise RuntimeError("SweepDust relocation manifest is unreadable")
    return {
        "schema": spec["schema"],
        # These three values are part of the already released checkpoint
        # identity. Moving byte-identical inputs must not rewrite the model.
        "manifest_path": identity_at_fit["manifest_path"],
        "manifest_sha256": identity_at_fit["manifest_sha256"],
        "data_root": identity_at_fit["data_root"],
        "inputs": records,
        "status_at_fit": spec["status"],
    }


def _validate_retrained_sweep_manifest(
    target_dir: Path,
    expected_identity: dict[str, Any],
) -> None:
    manifest = _load_object(target_dir / "training.json")
    expected_augmentation = {
        "schema": expected_identity["schema"],
        "manifest_path": expected_identity["manifest_path"],
        "manifest_sha256": expected_identity["manifest_sha256"],
        "data_root": expected_identity["data_root"],
        "episodes": expected_identity["inputs"],
        "status_at_fit": expected_identity["status_at_fit"],
    }
    if (
        manifest.get("manifest_schema") != "dynamac-direct-training-v3"
        or manifest.get("task") != SWEEP_DUST_TASK_NAME
        or manifest.get("bimanual") is not True
        or manifest.get("demonstrations")
        != [f"episode{episode}" for episode in range(5)]
        or manifest.get("training_data_augmentation") != expected_augmentation
    ):
        raise RuntimeError(
            "V4 SweepDust checkpoint is not bound to the five augmented inputs"
        )


def build_model_release_manifest(
    *,
    plan_path: str | Path = MODEL_RELEASE_PLAN_PATH,
    source_models_dir: str | Path = DEFAULT_V3_MODELS_DIR,
    target_models_dir: str | Path = DEFAULT_V4_MODELS_DIR,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Inventory V3/V4 artifacts without copying or training any model."""

    plan_file = Path(plan_path)
    plan = load_model_release_plan(plan_file)
    source_root = Path(source_models_dir)
    target_root = Path(target_models_dir)
    semantic = load_store_bottle_semantic_spec()
    sweep_training_inputs = sweep_training_input_identity()
    records = []
    for entry in plan["entries"]:
        target_dir = target_root / entry["target"]
        target = _inventory(target_dir, entry["required_artifacts"])
        record = {
            "model_id": entry["model_id"],
            "task": entry["task"],
            "mode": entry["mode"],
            "target": entry["target"],
            "required_artifacts": list(entry["required_artifacts"]),
            "target_inventory": target,
        }
        if entry["mode"] == RETRAINED_MODE:
            record["source"] = None
            if target["complete"]:
                _validate_retrained_store_manifest(
                    target_dir,
                    semantic.fingerprint,
                )
                status = "retrained_store_bottle_v4_verified"
            else:
                status = "pending_store_bottle_v4_training"
        elif entry["mode"] == SWEEP_RETRAINED_MODE:
            record["source"] = None
            record["training_data_augmentation"] = sweep_training_inputs
            if target["complete"]:
                _validate_retrained_sweep_manifest(
                    target_dir,
                    sweep_training_inputs,
                )
                status = "retrained_sweep_dust_v4_verified"
            else:
                status = "pending_sweep_dust_v4_training"
        else:
            source_dir = source_root / entry["source"]
            source = _inventory(source_dir, entry["required_artifacts"])
            record["source"] = entry["source"]
            record["source_inventory"] = source
            if not source["complete"]:
                status = "pending_v3_source"
            elif not target["complete"]:
                status = "pending_byte_copy_from_v3"
            elif _hashes(source) != _hashes(target):
                raise RuntimeError(
                    f"{entry['model_id']}: V4 inherited artifacts differ from V3"
                )
            else:
                status = "inherited_v3_byte_for_byte_verified"
        record["status"] = status
        records.append(record)

    complete = all(record["status"].endswith("_verified") for record in records)
    if require_complete and not complete:
        pending = [
            record["model_id"]
            for record in records
            if not record["status"].endswith("_verified")
        ]
        raise RuntimeError(f"V4 model release is incomplete: {pending}")
    manifest = {
        "schema": MODEL_RELEASE_MANIFEST_SCHEMA,
        "release": "v4",
        "source_release": "v3",
        "plan_sha256": _file_sha256(plan_file),
        "semantic_version": semantic.semantic_version,
        "semantic_fingerprint": semantic.fingerprint,
        "copy_or_training_performed": False,
        "complete": complete,
        "status": "verified" if complete else "pending",
        "entries": records,
    }
    manifest["fingerprint"] = _canonical_sha256(manifest)
    return manifest


def write_model_release_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write an already-built small inventory; checkpoint files are untouched."""

    if manifest.get("schema") != MODEL_RELEASE_MANIFEST_SCHEMA:
        raise ValueError("cannot write an unsupported model release manifest")
    expected = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "fingerprint"}
    )
    if manifest.get("fingerprint") != expected:
        raise ValueError("model release manifest fingerprint mismatch")
    atomic_json(Path(path), manifest)


__all__ = [
    "DEFAULT_V3_MODELS_DIR",
    "DEFAULT_V4_MODELS_DIR",
    "INHERITED_MODE",
    "MODEL_RELEASE_MANIFEST_SCHEMA",
    "MODEL_RELEASE_PLAN_PATH",
    "MODEL_RELEASE_PLAN_SCHEMA",
    "RETRAINED_MODE",
    "SWEEP_DUST_TASK_NAME",
    "SWEEP_RETRAINED_MODE",
    "SWEEP_TRAINING_DATA_SPEC_PATH",
    "build_model_release_manifest",
    "load_model_release_plan",
    "sweep_training_input_identity",
    "write_model_release_manifest",
]
