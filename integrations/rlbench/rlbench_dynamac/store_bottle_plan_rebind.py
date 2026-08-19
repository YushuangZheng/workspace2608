"""Identity-only StoreBottle plan rebind after an authenticated V4 retrain.

This command never launches RLBench and never samples a new scene.  It admits
one archived, hash-pinned StoreBottle task-scoped envelope, rewrites only the
intervention identity carried by each plan, and publishes a new envelope bound
to the current StoreBottle task identity.  A projection comparison proves that
all scene, seed, schedule, geometry, candidate, and validation evidence outside
those identity fields is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eval_set import (
    TASK_SCOPED_IDENTITY_SCHEMA,
    TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID,
    TASK_SCOPED_PLAN_BATCH_SCHEMA,
    TASK_SCOPED_REBIND_PROVENANCE_SCHEMA,
    build_task_scoped_identity,
    build_task_scoped_plan_batch,
    canonical_fingerprint,
    load_task_scoped_plan_batch,
)
from .records import atomic_json, reserve_output
from .store_bottle_eval_v4 import (
    STORE_BOTTLE_TASK_NAME,
    V4_STORE_BATCH_SCHEMA,
    V4_STORE_INTERVENTION_SCHEMA,
    V4_STORE_MODE_ORDER,
    V4_STORE_MOTION_PROTOCOL_ID,
    V4_STORE_PLAN_SCHEMA,
    V4_STORE_RUNTIME_LOADER_ID,
    load_v4_store_motion_plan_batch,
    store_mode_for_episode,
    v4_store_runtime_loaders,
    v4_store_task_identity_components,
)


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
V4_REBIND_ARCHIVE_ROOT = INTEGRATION_ROOT / "results" / "_archive" / "v4"
DEFAULT_ARCHIVED_ENVELOPE = (
    V4_REBIND_ARCHIVE_ROOT
    / "aborted_seal_2db53d670a1b_20260819T051019Z"
    / "evaluation_set"
    / "plans"
    / "environment"
    / "bimanual_put_bottle_in_fridge_a_b_n200.json"
)
DEFAULT_ARCHIVED_ENVELOPE_SHA256 = (
    "4767f199f3af2b1464b47194bb7a8de8e9c0932482c2ff8ea227fd89f6310a81"
)
CANONICAL_STORE_PLAN = (
    INTEGRATION_ROOT
    / "evaluation_sets"
    / "rlbench_eval_v2"
    / "plans"
    / "environment"
    / "bimanual_put_bottle_in_fridge_a_b_n200.json"
)
STORE_REBIND_PROTOCOL_ID = "store-bottle-identity-only-plan-rebind-v1"
_SHA256_LENGTH = 64


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"archived StoreBottle envelope is not strict JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("archived StoreBottle envelope must be a JSON object")
    return value


def _authenticate_source_envelope(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    """Authenticate the old outer/inner fingerprints without current-ID loading."""

    raw = copy.deepcopy(dict(source))
    expected_outer_fields = {
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
    outer_body = {
        key: value for key, value in raw.items() if key != "batch_fingerprint"
    }
    if (
        set(raw) != expected_outer_fields
        or raw.get("schema") != TASK_SCOPED_PLAN_BATCH_SCHEMA
        or raw.get("protocol_id") != TASK_SCOPED_PLAN_BATCH_PROTOCOL_ID
        or raw.get("task_name") != STORE_BOTTLE_TASK_NAME
        or raw.get("scenario_independent") is not True
        or raw.get("runtime_loader") != V4_STORE_RUNTIME_LOADER_ID
        or raw.get("batch_fingerprint") != canonical_fingerprint(outer_body)
    ):
        raise ValueError("archived StoreBottle outer envelope authentication failed")

    source_identity = raw.get("task_identity")
    source_components = (
        source_identity.get("components")
        if isinstance(source_identity, dict)
        else None
    )
    if not isinstance(source_components, dict):
        raise ValueError("archived StoreBottle task identity is absent")
    expected_source_identity = build_task_scoped_identity(
        task_name=STORE_BOTTLE_TASK_NAME,
        components=source_components,
    )
    if (
        source_identity != expected_source_identity
        or source_identity.get("schema") != TASK_SCOPED_IDENTITY_SCHEMA
    ):
        raise ValueError("archived StoreBottle task identity authentication failed")

    target_components = v4_store_task_identity_components()
    for role in ("task_semantics", "motion_source"):
        if source_components.get(role) != target_components.get(role):
            raise ValueError(
                f"StoreBottle {role} changed; identity-only rebind is forbidden"
            )
    source_intervention = source_components.get("intervention")
    target_intervention = target_components.get("intervention")
    if (
        not isinstance(source_intervention, dict)
        or not isinstance(target_intervention, dict)
        or source_intervention.get("schema") != V4_STORE_INTERVENTION_SCHEMA
        or target_intervention.get("schema") != V4_STORE_INTERVENTION_SCHEMA
        or not _is_sha256(source_intervention.get("fingerprint"))
        or not _is_sha256(target_intervention.get("fingerprint"))
        or source_intervention["fingerprint"]
        == target_intervention["fingerprint"]
    ):
        raise ValueError(
            "StoreBottle rebind requires one authenticated intervention-only "
            "identity change"
        )

    inner = raw.get("runtime_batch")
    if not isinstance(inner, dict):
        raise ValueError("archived StoreBottle runtime batch is absent")
    _authenticate_source_runtime_batch(
        inner,
        source_intervention_fingerprint=source_intervention["fingerprint"],
        source_motion_fingerprint=source_components["motion_source"]["fingerprint"],
    )
    if (
        raw.get("base_seed") != inner.get("base_seed")
        or raw.get("episodes") != inner.get("episodes")
        or raw.get("variation_schedule") != inner.get("variation_schedule")
    ):
        raise ValueError("archived StoreBottle outer/inner schedules disagree")
    return raw, inner, target_components


def _authenticate_source_runtime_batch(
    inner: Mapping[str, Any],
    *,
    source_intervention_fingerprint: str,
    source_motion_fingerprint: str,
) -> None:
    expected_inner_fields = {
        "schema",
        "protocol_id",
        "task_name",
        "base_seed",
        "episodes",
        "variation_schedule",
        "scenario_independent",
        "seed_domain",
        "mode_schedule",
        "mode_counts",
        "plans",
        "batch_fingerprint",
    }
    body = {
        key: value for key, value in inner.items() if key != "batch_fingerprint"
    }
    plans = inner.get("plans")
    variations = inner.get("variation_schedule")
    episodes = inner.get("episodes")
    base_seed = inner.get("base_seed")
    if (
        set(inner) != expected_inner_fields
        or inner.get("schema") != V4_STORE_BATCH_SCHEMA
        or inner.get("protocol_id") != V4_STORE_MOTION_PROTOCOL_ID
        or inner.get("task_name") != STORE_BOTTLE_TASK_NAME
        or inner.get("scenario_independent") is not True
        or inner.get("batch_fingerprint") != canonical_fingerprint(body)
        or isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or isinstance(episodes, bool)
        or not isinstance(episodes, int)
        or episodes < 1
        or not isinstance(variations, list)
        or len(variations) != episodes
        or not isinstance(plans, list)
        or len(plans) != episodes
    ):
        raise ValueError("archived StoreBottle runtime batch authentication failed")

    counts = {mode: 0 for mode in V4_STORE_MODE_ORDER}
    expected_plan_fields = {
        "schema",
        "protocol_id",
        "task_name",
        "episode_index",
        "episode_seed",
        "variation",
        "mode",
        "entities",
        "source_low_dim_state",
        "validation",
        "fingerprint",
    }
    for episode, (variation, plan) in enumerate(zip(variations, plans)):
        if not isinstance(plan, dict):
            raise ValueError("archived StoreBottle plan must be an object")
        plan_body = {
            key: value for key, value in plan.items() if key != "fingerprint"
        }
        validation = plan.get("validation")
        mode = store_mode_for_episode(episode)
        if (
            set(plan) != expected_plan_fields
            or plan.get("schema") != V4_STORE_PLAN_SCHEMA
            or plan.get("protocol_id") != V4_STORE_MOTION_PROTOCOL_ID
            or plan.get("task_name") != STORE_BOTTLE_TASK_NAME
            or plan.get("episode_index") != episode
            or plan.get("episode_seed") != base_seed + episode
            or plan.get("variation") != variation
            or plan.get("mode") != mode
            or plan.get("fingerprint") != canonical_fingerprint(plan_body)
            or not isinstance(validation, dict)
            or validation.get("motion_source_fingerprint")
            != source_motion_fingerprint
            or validation.get("intervention_fingerprint")
            != source_intervention_fingerprint
            or validation.get("policy_result_fields_read") is not False
        ):
            raise ValueError(
                f"archived StoreBottle plan authentication failed: episode {episode}"
            )
        counts[mode] += 1
    if inner.get("mode_counts") != counts:
        raise ValueError("archived StoreBottle mode counts are inconsistent")


def store_runtime_non_identity_projection(
    runtime_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove exactly the identity fields that an admitted rebind may rewrite."""

    projection = copy.deepcopy(dict(runtime_batch))
    projection.pop("batch_fingerprint", None)
    plans = projection.get("plans")
    if not isinstance(plans, list):
        raise ValueError("StoreBottle runtime projection lacks plans")
    for plan in plans:
        if not isinstance(plan, dict):
            raise ValueError("StoreBottle runtime projection plan is invalid")
        plan.pop("fingerprint", None)
        validation = plan.get("validation")
        if not isinstance(validation, dict):
            raise ValueError("StoreBottle runtime projection validation is invalid")
        validation.pop("intervention_fingerprint", None)
    return projection


def _rewrite_runtime_intervention_identity(
    source_inner: Mapping[str, Any],
    *,
    target_intervention_fingerprint: str,
) -> dict[str, Any]:
    rewritten = copy.deepcopy(dict(source_inner))
    for plan in rewritten["plans"]:
        plan["validation"][
            "intervention_fingerprint"
        ] = target_intervention_fingerprint
        plan_body = {
            key: value for key, value in plan.items() if key != "fingerprint"
        }
        plan["fingerprint"] = canonical_fingerprint(plan_body)
    batch_body = {
        key: value for key, value in rewritten.items() if key != "batch_fingerprint"
    }
    rewritten["batch_fingerprint"] = canonical_fingerprint(batch_body)
    return rewritten


def rebind_v4_store_plan_envelope(
    source: Mapping[str, Any],
    *,
    source_file_sha256: str,
) -> dict[str, Any]:
    """Build a current-identity envelope while preserving all non-ID evidence."""

    if not _is_sha256(source_file_sha256):
        raise ValueError("archived StoreBottle source SHA-256 is invalid")
    old_outer, old_inner, target_components = _authenticate_source_envelope(source)
    target_identity = build_task_scoped_identity(
        task_name=STORE_BOTTLE_TASK_NAME,
        components=target_components,
    )
    target_intervention = target_components["intervention"]["fingerprint"]
    new_inner = _rewrite_runtime_intervention_identity(
        old_inner,
        target_intervention_fingerprint=target_intervention,
    )
    old_projection = store_runtime_non_identity_projection(old_inner)
    new_projection = store_runtime_non_identity_projection(new_inner)
    if old_projection != new_projection:
        raise RuntimeError(
            "StoreBottle rebind changed non-identity runtime plan evidence"
        )
    projection_fingerprint = canonical_fingerprint(old_projection)

    # This invokes the ordinary current runtime loader.  Geometry, seeds,
    # schedules, semantic evidence, and every rewritten current fingerprint
    # must therefore pass the same gate used by formal evaluation.
    load_v4_store_motion_plan_batch(new_inner)
    provenance = {
        "schema": TASK_SCOPED_REBIND_PROVENANCE_SCHEMA,
        "protocol_id": STORE_REBIND_PROTOCOL_ID,
        "task_name": STORE_BOTTLE_TASK_NAME,
        "source_file_sha256": source_file_sha256,
        "source_outer_batch_fingerprint": old_outer["batch_fingerprint"],
        "source_inner_batch_fingerprint": old_inner["batch_fingerprint"],
        "source_task_identity_fingerprint": old_outer["task_identity"][
            "fingerprint"
        ],
        "target_task_identity_fingerprint": target_identity["fingerprint"],
        "non_identity_projection_fingerprint": projection_fingerprint,
        "rewritten_fields": [
            "runtime_batch.plans[*].validation.intervention_fingerprint",
            "runtime_batch.plans[*].fingerprint",
            "runtime_batch.batch_fingerprint",
            "task_identity",
            "rebind_provenance",
            "batch_fingerprint",
        ],
        "policy_result_fields_read": False,
        "simulator_started": False,
    }
    rebound = build_task_scoped_plan_batch(
        task_name=STORE_BOTTLE_TASK_NAME,
        task_identity=target_identity,
        runtime_loader=V4_STORE_RUNTIME_LOADER_ID,
        runtime_batch=new_inner,
        rebind_provenance=provenance,
    )
    if (
        store_runtime_non_identity_projection(rebound["runtime_batch"])
        != old_projection
    ):
        raise RuntimeError("StoreBottle outer rebind changed the runtime projection")
    load_task_scoped_plan_batch(
        rebound,
        runtime_loaders=v4_store_runtime_loaders(),
    )
    return rebound


def _require_archived_source(path: Path) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("StoreBottle rebind source must not be a symbolic link")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(V4_REBIND_ARCHIVE_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "StoreBottle rebind source must stay in the V4 archive"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _require_canonical_output(path: Path) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("StoreBottle canonical plan output must not be a symlink")
    resolved = unresolved.resolve()
    if resolved != CANONICAL_STORE_PLAN.resolve():
        raise ValueError(
            "StoreBottle rebind output must be the canonical rlbench_eval_v2 plan"
        )
    return resolved


def rebind_v4_store_plan_file(
    source: Path = DEFAULT_ARCHIVED_ENVELOPE,
    *,
    expected_source_sha256: str = DEFAULT_ARCHIVED_ENVELOPE_SHA256,
    output: Path = CANONICAL_STORE_PLAN,
) -> dict[str, Any]:
    """Hash-pin one archived envelope and atomically publish its rebind."""

    source_path = _require_archived_source(Path(source))
    output_path = _require_canonical_output(Path(output))
    actual_source_sha256 = _file_sha256(source_path)
    if (
        not _is_sha256(expected_source_sha256)
        or actual_source_sha256 != expected_source_sha256
    ):
        raise ValueError("archived StoreBottle envelope SHA-256 mismatch")
    rebound = rebind_v4_store_plan_envelope(
        _load_object(source_path),
        source_file_sha256=actual_source_sha256,
    )
    with reserve_output(output_path):
        atomic_json(output_path, rebound)
    return {
        "output": str(output_path),
        "sha256": _file_sha256(output_path),
        "batch_fingerprint": rebound["batch_fingerprint"],
        "task_identity_fingerprint": rebound["task_identity"]["fingerprint"],
        "runtime_batch_fingerprint": rebound["runtime_batch"][
            "batch_fingerprint"
        ],
        "rebind_provenance": rebound["rebind_provenance"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_ARCHIVED_ENVELOPE)
    parser.add_argument(
        "--expected-source-sha256",
        default=DEFAULT_ARCHIVED_ENVELOPE_SHA256,
    )
    parser.add_argument("--output", type=Path, default=CANONICAL_STORE_PLAN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = rebind_v4_store_plan_file(
        args.source,
        expected_source_sha256=args.expected_source_sha256,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "CANONICAL_STORE_PLAN",
    "DEFAULT_ARCHIVED_ENVELOPE",
    "DEFAULT_ARCHIVED_ENVELOPE_SHA256",
    "STORE_REBIND_PROTOCOL_ID",
    "build_parser",
    "main",
    "rebind_v4_store_plan_envelope",
    "rebind_v4_store_plan_file",
    "store_runtime_non_identity_projection",
]


if __name__ == "__main__":
    raise SystemExit(main())
