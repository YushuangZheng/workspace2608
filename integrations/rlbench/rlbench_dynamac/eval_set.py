"""Read-only authentication for the version-neutral fixed RLBench eval set."""

from __future__ import annotations

import hashlib
import json
import re
import argparse
from pathlib import Path
from typing import Any

from .evaluation_split import (
    EVALUATION_SET_SPEC_SCHEMA,
    TRAINING_SPLIT_SCHEMA,
    load_evaluation_set_spec,
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


def load_fixed_eval_set_manifest(
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


def fixed_environment_plans(
    eval_set_id: str,
    task: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_fixed_eval_set_manifest(eval_set_id, selected_task=task)
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


def seal_fixed_eval_set(eval_set_id: str) -> Path:
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
    "ENVIRONMENT_TASKS",
    "FIXED_EVAL_EPISODES",
    "FIXED_EVAL_SET_MANIFEST_SCHEMA",
    "FIXED_EVAL_SET_PROTOCOL_ID",
    "GLOBAL_EVAL_SEED_START",
    "canonical_fingerprint",
    "file_sha256",
    "fixed_environment_plans",
    "fixed_coordination_sources",
    "load_fixed_eval_set_manifest",
    "resolve_eval_set_root",
    "validate_formal_artifact_paths",
]


if __name__ == "__main__":
    raise SystemExit(main())
