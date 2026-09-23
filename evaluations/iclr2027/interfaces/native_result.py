"""Validation for RVT/RACER/Ours E6 native-system episode results."""

from __future__ import annotations

import re
from typing import Any, Mapping


NATIVE_RESULT_SCHEMA = "essay2608.iclr2027.native-system-episode-result.v1"
ALLOWED_METHODS = frozenset({"rvt", "racer", "ours"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_native_episode_result(
    value: Mapping[str, Any], manifest_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one raw E6 result against the exact assigned episode row."""

    if value.get("schema") != NATIVE_RESULT_SCHEMA:
        raise ValueError("unsupported Native-6 episode result schema")
    method_id = _nonempty_string(value.get("method_id"), "method_id")
    if method_id not in ALLOWED_METHODS:
        raise ValueError(f"unsupported Native-6 method_id: {method_id}")

    for field in (
        "episode_id",
        "pair_id",
        "task",
        "condition",
        "fault_family",
        "fault_severity",
        "trigger_stage",
    ):
        expected = manifest_row.get(field)
        observed = value.get(field)
        if observed != expected:
            raise ValueError(
                f"result {field}={observed!r} disagrees with manifest {expected!r}"
            )
    source_episode_id = manifest_row.get("source_episode_id")
    if value.get("source_episode_id") != source_episode_id:
        raise ValueError("result source_episode_id disagrees with manifest")
    for field in ("variation", "seed", "horizon"):
        if int(value.get(field, -1)) != int(manifest_row[field]):
            raise ValueError(f"result {field} disagrees with manifest")

    for field in ("config_identity", "checkpoint_identity", "environment_identity"):
        _nonempty_string(value.get(field), field)
    for field in ("eligible", "physically_triggered", "final_success", "infrastructure_error"):
        if not isinstance(value.get(field), bool):
            raise ValueError(f"{field} must be boolean")
    for field in ("cycles", "peak_memory_kib", "cycle_records"):
        _nonnegative_integer(value.get(field), field)
    wall_seconds = value.get("wall_seconds")
    if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)) or wall_seconds < 0:
        raise ValueError("wall_seconds must be a non-negative number")
    for field in (
        "violation_onset_cycle",
        "violation_end_cycle",
        "relation_restored_cycle",
    ):
        item = value.get(field)
        if item is not None:
            _nonnegative_integer(item, field)
    _nonempty_string(value.get("termination_reason"), "termination_reason")
    _nonempty_string(value.get("cycle_file"), "cycle_file")
    digest = _nonempty_string(value.get("cycle_file_sha256"), "cycle_file_sha256")
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError("cycle_file_sha256 must be a lowercase SHA256 digest")
    if value["cycles"] > value["horizon"]:
        raise ValueError("cycles exceeds the frozen horizon")
    if value["cycle_records"] > value["cycles"]:
        raise ValueError("cycle_records exceeds executed cycles")
    if value["physically_triggered"] and not value["eligible"]:
        raise ValueError("a physically triggered event must be eligible")
    if value["condition"] == "nominal":
        if value["eligible"] or value["physically_triggered"]:
            raise ValueError("nominal result claims a fault event")
    return dict(value)


__all__ = ["ALLOWED_METHODS", "NATIVE_RESULT_SCHEMA", "validate_native_episode_result"]
