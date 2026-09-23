"""Pure helpers for the frozen Native-6 v3 task/contact mapping.

The helpers intentionally do not import PyRep.  Native-system adapters resolve
their own live objects and pass names plus ancestry here, which keeps the
semantic mapping identical across RVT, RACER, and Ours.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


COPY_SUFFIX = re.compile(r"#[0-9]+$")


def load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_mapping(value)
    return value


def normalize_coppelia_name(value: str) -> str:
    """Remove only CoppeliaSim's model-copy suffix from an object name."""

    if not isinstance(value, str) or not value:
        raise ValueError("scene object names must be non-empty strings")
    return COPY_SUFFIX.sub("", value)


def _task_spec(mapping: Mapping[str, Any], task_id: str) -> Mapping[str, Any]:
    tasks = mapping.get("tasks")
    if not isinstance(tasks, Mapping) or task_id not in tasks:
        raise ValueError(f"task is absent from the frozen mapping: {task_id}")
    spec = tasks[task_id]
    if not isinstance(spec, Mapping):
        raise TypeError(f"task mapping must be an object: {task_id}")
    return spec


def active_roots(
    mapping: Mapping[str, Any], task_id: str, variation: int
) -> tuple[dict[str, Any], ...]:
    """Return the semantic interaction roots active for one task episode."""

    spec = _task_spec(mapping, task_id)
    roots = spec.get("interaction_roots")
    if isinstance(roots, list):
        return tuple(dict(item) for item in roots)
    by_variation = spec.get("interaction_roots_by_variation")
    if not isinstance(by_variation, Mapping):
        raise ValueError(f"task has no interaction roots: {task_id}")
    key = str(int(variation))
    if key not in by_variation:
        raise ValueError(
            f"variation {variation} is absent from the frozen mapping for {task_id}"
        )
    selected = by_variation[key]
    if not isinstance(selected, list):
        raise TypeError(f"variation roots must be a list: {task_id}/{variation}")
    return tuple(dict(item) for item in selected)


def canonical_task_object(
    mapping: Mapping[str, Any],
    *,
    task_id: str,
    variation: int,
    object_name: str,
    ancestry_names: Iterable[str] = (),
) -> str | None:
    """Map a live task shape to one frozen semantic interaction object.

    ``ancestry_names`` must contain only the live object's parent chain.  Name
    prefixes and arbitrary scene-wide substring matching are deliberately not
    accepted.
    """

    names = {
        normalize_coppelia_name(name)
        for name in (object_name, *tuple(ancestry_names))
    }
    matches = []
    for root in active_roots(mapping, task_id, variation):
        scene_root = normalize_coppelia_name(str(root["scene_root_name"]))
        if scene_root in names:
            matches.append(str(root["canonical_id"]))
    if len(matches) > 1:
        raise ValueError(
            f"task object maps to multiple semantic roots: {task_id}/{object_name}"
        )
    return matches[0] if matches else None


def is_allowed_gripper_collision_shape(
    mapping: Mapping[str, Any], object_name: str
) -> bool:
    gripper = mapping.get("gripper_contact_scope")
    if not isinstance(gripper, Mapping):
        raise ValueError("mapping misses gripper_contact_scope")
    allowed = {
        str(name) for name in gripper.get("allowed_collision_shape_base_names", ())
    }
    return normalize_coppelia_name(object_name) in allowed


def validate_mapping(mapping: Mapping[str, Any]) -> None:
    if mapping.get("schema") != "essay2608.iclr2027.native6-object-contact-mapping.v1":
        raise ValueError("unsupported Native-6 object/contact mapping schema")
    gripper = mapping.get("gripper_contact_scope")
    if not isinstance(gripper, Mapping):
        raise ValueError("mapping misses gripper_contact_scope")
    allowed = gripper.get("allowed_collision_shape_base_names")
    if not isinstance(allowed, list) or len(allowed) != 4 or len(set(allowed)) != 4:
        raise ValueError("exactly four unique Panda finger collision shapes are required")
    expected_tasks = {
        "close_jar",
        "open_drawer",
        "insert_onto_square_peg",
        "place_cups_3",
        "stack_cups",
        "sweep_to_dustpan",
    }
    tasks = mapping.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != expected_tasks:
        raise ValueError("object mapping must define exactly the frozen Native-6 tasks")
    for task_id in sorted(expected_tasks):
        spec = _task_spec(mapping, task_id)
        variations = spec.get("preflight_variations")
        if not isinstance(variations, list) or not variations:
            raise ValueError(f"preflight variations are missing: {task_id}")
        seen: set[str] = set()
        for variation in variations:
            for root in active_roots(mapping, task_id, int(variation)):
                canonical = str(root.get("canonical_id", ""))
                scene_name = str(root.get("scene_root_name", ""))
                mode = str(root.get("missed_interaction_mode", ""))
                if not canonical or not scene_name:
                    raise ValueError(f"incomplete interaction root: {task_id}")
                if canonical in seen and spec.get("interaction_roots_by_variation") is None:
                    raise ValueError(f"duplicate canonical interaction root: {task_id}/{canonical}")
                seen.add(canonical)
                if mode not in {
                    "attachment_suppression",
                    "maintained_contact_close_suppression",
                }:
                    raise ValueError(f"invalid missed-interaction mode: {task_id}/{mode}")


def validate_mapping_preflight(
    report: Mapping[str, Any],
    mapping: Mapping[str, Any],
    *,
    system_id: str,
    expected_mapping_identity: str | None = None,
) -> None:
    """Fail closed on a B-side live-scene mapping preflight report."""

    validate_mapping(mapping)
    if report.get("schema") != "essay2608.iclr2027.native6-object-mapping-preflight.v1":
        raise ValueError("unsupported mapping preflight schema")
    if report.get("system_id") != system_id or report.get("status") != "PASS":
        raise ValueError("mapping preflight system/status mismatch")
    if expected_mapping_identity is not None and report.get(
        "object_mapping_identity"
    ) != expected_mapping_identity:
        raise ValueError("mapping preflight binds another object mapping")

    gripper_records = report.get("gripper_collision_shapes")
    if not isinstance(gripper_records, list):
        raise ValueError("mapping preflight misses gripper collision shapes")
    gripper_names = []
    for record in gripper_records:
        if not isinstance(record, Mapping):
            raise TypeError("gripper collision records must be objects")
        for field in ("name", "handle", "object_type"):
            if field not in record:
                raise ValueError(f"gripper collision record misses {field}")
        if str(record["object_type"]).lower() != "shape":
            raise ValueError("gripper contact entities must be shapes")
        if not is_allowed_gripper_collision_shape(mapping, str(record["name"])):
            raise ValueError(f"unfrozen gripper collision shape: {record['name']}")
        gripper_names.append(normalize_coppelia_name(str(record["name"])))
    expected_gripper = sorted(
        str(name)
        for name in mapping["gripper_contact_scope"][
            "allowed_collision_shape_base_names"
        ]
    )
    if sorted(gripper_names) != expected_gripper:
        raise ValueError("preflight must resolve each frozen finger collision shape once")

    expected_probe_keys = {
        (task_id, int(variation))
        for task_id, spec in mapping["tasks"].items()
        for variation in spec["preflight_variations"]
    }
    probes = report.get("task_probes")
    if not isinstance(probes, list):
        raise ValueError("mapping preflight misses task probes")
    observed_probe_keys = set()
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise TypeError("task probes must be objects")
        task_id = str(probe.get("task"))
        variation = int(probe.get("variation"))
        key = (task_id, variation)
        if key in observed_probe_keys:
            raise ValueError(f"duplicate mapping task probe: {key}")
        observed_probe_keys.add(key)
        expected_roots = {
            (str(root["canonical_id"]), str(root["scene_root_name"]))
            for root in active_roots(mapping, task_id, variation)
        }
        roots = probe.get("active_roots")
        if not isinstance(roots, list):
            raise ValueError(f"task probe misses active roots: {key}")
        observed_roots = set()
        for root in roots:
            if not isinstance(root, Mapping):
                raise TypeError("resolved roots must be objects")
            for field in (
                "canonical_id",
                "scene_root_name",
                "resolved_name",
                "handle",
                "object_type",
                "parent_chain",
                "resolved_count",
            ):
                if field not in root:
                    raise ValueError(f"resolved root misses {field}: {key}")
            if int(root["resolved_count"]) != 1:
                raise ValueError(f"semantic root did not resolve exactly once: {key}")
            if normalize_coppelia_name(str(root["resolved_name"])) != str(
                root["scene_root_name"]
            ):
                raise ValueError(f"resolved root name differs from frozen root: {key}")
            observed_roots.add(
                (str(root["canonical_id"]), str(root["scene_root_name"]))
            )
        if observed_roots != expected_roots:
            raise ValueError(f"resolved roots differ from frozen mapping: {key}")
    if observed_probe_keys != expected_probe_keys:
        raise ValueError("mapping preflight task/variation coverage is incomplete")

    contacts = report.get("open_drawer_contact_evidence")
    if not isinstance(contacts, list):
        raise ValueError("mapping preflight misses Open Drawer contact evidence")
    covered_variations = set()
    for contact in contacts:
        if not isinstance(contact, Mapping):
            raise TypeError("Open Drawer contact evidence must be objects")
        for field in (
            "variation",
            "finger_shape_name",
            "task_shape_name",
            "task_shape_parent_chain",
            "canonical_id",
            "sim_step",
            "simulation_time_s",
        ):
            if field not in contact:
                raise ValueError(f"Open Drawer contact evidence misses {field}")
        variation = int(contact["variation"])
        if not is_allowed_gripper_collision_shape(
            mapping, str(contact["finger_shape_name"])
        ):
            raise ValueError("Open Drawer evidence uses an unfrozen gripper shape")
        canonical = canonical_task_object(
            mapping,
            task_id="open_drawer",
            variation=variation,
            object_name=str(contact["task_shape_name"]),
            ancestry_names=tuple(contact["task_shape_parent_chain"]),
        )
        if canonical is None or canonical != str(contact["canonical_id"]):
            raise ValueError("Open Drawer contact does not map to its active drawer")
        if int(contact["sim_step"]) < 0 or float(contact["simulation_time_s"]) < 0.0:
            raise ValueError("Open Drawer contact has invalid physical time")
        covered_variations.add(variation)
    expected_drawer_variations = {
        int(value) for value in mapping["tasks"]["open_drawer"]["preflight_variations"]
    }
    if covered_variations != expected_drawer_variations:
        raise ValueError("Open Drawer contact evidence must cover every frozen variation")

    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("mapping preflight misses checks")
    required_checks = {
        "semantic_roots_unique",
        "finger_shapes_exact",
        "drawer_contacts_verified",
        "excluded_objects_absent",
        "proximity_not_used_as_contact",
    }
    if set(checks) != required_checks or not all(
        checks[name] is True for name in required_checks
    ):
        raise ValueError("mapping preflight checks are incomplete or failed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "active_roots",
    "canonical_task_object",
    "is_allowed_gripper_collision_shape",
    "load_mapping",
    "normalize_coppelia_name",
    "sha256_file",
    "validate_mapping",
    "validate_mapping_preflight",
]
