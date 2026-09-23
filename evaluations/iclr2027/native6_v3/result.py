"""Strict episode-result validation for the event-grounded E6 rerun."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .events import FAULT_FAMILIES


RESULT_SCHEMA = "essay2608.iclr2027.native6-v3-episode-result.v1"
PROTOCOL_REVISION = "native6_event_grounded_physics_v3"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def validate_native6_v3_result(
    value: Mapping[str, Any], manifest_row: Mapping[str, Any]
) -> Mapping[str, Any]:
    required = {
        "schema",
        "episode_id",
        "pair_id",
        "method_id",
        "task",
        "variation",
        "seed",
        "condition",
        "fault_family",
        "fault_severity",
        "trigger_rule",
        "event_ordinal",
        "horizon",
        "protocol_revision",
        "config_identity",
        "checkpoint_identity",
        "environment_identity",
        "fault_adapter_identity",
        "fault_config_identity",
        "audit_protocol_revision",
        "manifest_identity",
        "eligible",
        "injection_triggered",
        "physical_effect_confirmed",
        "physically_triggered",
        "eligible_sim_step",
        "trigger_sim_step",
        "effect_sim_step",
        "eligible_time_s",
        "trigger_time_s",
        "effect_time_s",
        "violation_onset_sim_step",
        "violation_onset_time_s",
        "violation_end_sim_step",
        "violation_end_time_s",
        "relation_restored_sim_step",
        "relation_restored_time_s",
        "completed_sim_steps",
        "final_simulation_time_s",
        "final_success",
        "cycles",
        "termination_reason",
        "infrastructure_error",
        "wall_seconds",
        "peak_memory_kib",
        "cycle_file",
        "cycle_file_sha256",
        "cycle_records",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Native-6 v3 result misses fields: {missing}")
    if value["schema"] != RESULT_SCHEMA:
        raise ValueError("unsupported Native-6 v3 result schema")
    if value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("result uses the wrong event-grounded protocol revision")
    manifest_fields = (
        "episode_id",
        "pair_id",
        "task",
        "variation",
        "seed",
        "condition",
        "fault_family",
        "fault_severity",
        "trigger_rule",
        "event_ordinal",
        "horizon",
    )
    for field in manifest_fields:
        if value.get(field) != manifest_row.get(field):
            raise ValueError(f"result differs from manifest field: {field}")
    if value["condition"] != "perturbed":
        raise ValueError("Native-6 v3 result validator accepts perturbed rows only")
    if value["fault_family"] not in FAULT_FAMILIES:
        raise ValueError("unsupported Native-6 v3 fault family")
    if value["trigger_rule"] != "first_eligible_physical_event":
        raise ValueError("Native-6 v3 uses only first eligible physical events")
    if value["event_ordinal"] != 1:
        raise ValueError("Native-6 v3 event ordinal must be one")
    for field in (
        "config_identity",
        "checkpoint_identity",
        "environment_identity",
        "fault_adapter_identity",
        "fault_config_identity",
        "audit_protocol_revision",
        "manifest_identity",
    ):
        _nonempty(value[field], field)
    for field in (
        "eligible",
        "injection_triggered",
        "physical_effect_confirmed",
        "physically_triggered",
        "final_success",
        "infrastructure_error",
    ):
        if not isinstance(value[field], bool):
            raise TypeError(f"{field} must be boolean")
    if value["injection_triggered"] and not value["eligible"]:
        raise ValueError("injection cannot precede eligibility")
    if value["physical_effect_confirmed"] and not value["injection_triggered"]:
        raise ValueError("physical effect cannot precede injection")
    if value["physically_triggered"] != value["physical_effect_confirmed"]:
        raise ValueError("physically_triggered must alias physical_effect_confirmed")
    triplets = (
        ("eligible", "eligible_sim_step", "eligible_time_s"),
        ("injection_triggered", "trigger_sim_step", "trigger_time_s"),
        ("physical_effect_confirmed", "effect_sim_step", "effect_time_s"),
    )
    for flag, step, seconds in triplets:
        if value[flag]:
            if isinstance(value[step], bool) or not isinstance(value[step], int) or value[step] < 0:
                raise ValueError(f"{step} must be a non-negative integer")
            if not isinstance(value[seconds], (int, float)) or value[seconds] < 0:
                raise ValueError(f"{seconds} must be non-negative")
        elif value[step] is not None or value[seconds] is not None:
            raise ValueError(f"{step}/{seconds} must be null when {flag} is false")
    ordered_steps = [
        value[field]
        for field in ("eligible_sim_step", "trigger_sim_step", "effect_sim_step")
        if value[field] is not None
    ]
    ordered_times = [
        float(value[field])
        for field in ("eligible_time_s", "trigger_time_s", "effect_time_s")
        if value[field] is not None
    ]
    if ordered_steps != sorted(ordered_steps) or ordered_times != sorted(ordered_times):
        raise ValueError("eligibility, injection, and physical effect must be time ordered")
    for field in ("completed_sim_steps", "cycles", "cycle_records"):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if value["cycle_records"] > value["cycles"]:
        raise ValueError("cycle_records exceeds high-level cycles")
    if not isinstance(value["final_simulation_time_s"], (int, float)) or value["final_simulation_time_s"] < 0:
        raise ValueError("final_simulation_time_s must be non-negative")
    for step, seconds in (
        ("violation_onset_sim_step", "violation_onset_time_s"),
        ("violation_end_sim_step", "violation_end_time_s"),
        ("relation_restored_sim_step", "relation_restored_time_s"),
    ):
        if (value[step] is None) != (value[seconds] is None):
            raise ValueError(f"{step} and {seconds} must both be null or populated")
        if value[step] is not None:
            if isinstance(value[step], bool) or not isinstance(value[step], int) or value[step] < 0:
                raise ValueError(f"{step} must be a non-negative integer")
            if not isinstance(value[seconds], (int, float)) or value[seconds] < 0:
                raise ValueError(f"{seconds} must be non-negative")
    if value["physical_effect_confirmed"]:
        if value["violation_onset_sim_step"] is None:
            raise ValueError("confirmed physical effect requires violation onset")
    elif any(
        value[field] is not None
        for field in (
            "violation_onset_sim_step",
            "violation_onset_time_s",
            "violation_end_sim_step",
            "violation_end_time_s",
            "relation_restored_sim_step",
            "relation_restored_time_s",
        )
    ):
        raise ValueError("unconfirmed faults cannot report violation or restoration times")
    ordered_physical_steps = [
        value[field]
        for field in (
            "violation_onset_sim_step",
            "violation_end_sim_step",
            "relation_restored_sim_step",
        )
        if value[field] is not None
    ]
    ordered_physical_times = [
        float(value[field])
        for field in (
            "violation_onset_time_s",
            "violation_end_time_s",
            "relation_restored_time_s",
        )
        if value[field] is not None
    ]
    if (
        ordered_physical_steps != sorted(ordered_physical_steps)
        or ordered_physical_times != sorted(ordered_physical_times)
    ):
        raise ValueError("violation and restoration events must be time ordered")
    all_steps = ordered_steps + ordered_physical_steps
    if all_steps and max(all_steps) > value["completed_sim_steps"]:
        raise ValueError("physical event occurs after the final simulator step")
    if not isinstance(value["wall_seconds"], (int, float)) or value["wall_seconds"] < 0:
        raise ValueError("wall_seconds must be non-negative")
    if isinstance(value["peak_memory_kib"], bool) or not isinstance(value["peak_memory_kib"], int) or value["peak_memory_kib"] < 0:
        raise ValueError("peak_memory_kib must be a non-negative integer")
    digest = _nonempty(value["cycle_file_sha256"], "cycle_file_sha256")
    if SHA256.fullmatch(digest) is None:
        raise ValueError("cycle_file_sha256 must be lowercase SHA256")
    _nonempty(value["cycle_file"], "cycle_file")
    return value


__all__ = ["PROTOCOL_REVISION", "RESULT_SCHEMA", "validate_native6_v3_result"]
