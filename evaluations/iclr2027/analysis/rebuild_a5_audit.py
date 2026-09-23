"""Rebuild the A5 physical-event audit from retained physical evidence.

The retained endpoint files are immutable experimental evidence.  Some early
M0/M5 rows embedded the pre-fix relation auditor, while their fault-protocol
records already contain the physical trigger/effect/restoration observations
needed by the frozen auditor.  This module validates those observations and
writes one versioned, authoritative re-audit artifact without rewriting the
source episodes or cycle logs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "evaluations/iclr2027/results/controlled/e1_e2"
ENDPOINT_ROOT = RESULT_ROOT / "end_to_end"
OUTPUT_ROOT = RESULT_ROOT / "reaudit/current"
EVENTS_PATH = OUTPUT_ROOT / "events.jsonl.gz"
INDEX_PATH = OUTPUT_ROOT / "REAUDIT_INDEX.json"
AUDITOR_PATH = ROOT / "evaluations/iclr2027/audit/physical_events.py"
SCHEMA = "essay2608.iclr2027.retained-physical-event-reaudit.v2"
FAMILIES = {
    "actuation_delay",
    "composed_event",
    "coordination_delay",
    "environment_change",
    "missed_interaction",
    "relation_loss",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _events(protocol: Mapping[str, Any], *kinds: str) -> list[Mapping[str, Any]]:
    wanted = set(kinds)
    return [
        event
        for event in protocol.get("events", ())
        if isinstance(event, Mapping) and event.get("kind") in wanted
    ]


def _effective(event: Mapping[str, Any]) -> bool:
    return event.get("protocol_effective") is True


def _validate_effect(family: str, protocol: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if protocol.get("family") != family:
        reasons.append("family_mismatch")
    if protocol.get("triggered") is not True:
        reasons.append("injector_not_triggered")
    if protocol.get("physical_effect_observed") is not True:
        reasons.append("physical_effect_not_observed")

    if family == "composed_event":
        components = [
            value
            for value in protocol.get("components", ())
            if isinstance(value, Mapping)
        ]
        if not components:
            reasons.append("missing_composed_components")
        for index, component in enumerate(components):
            component_family = str(component.get("family"))
            if component_family == "composed_event":
                reasons.append(f"component_{index}:nested_composed_event")
                continue
            _valid, component_reasons = _validate_effect(
                component_family, component
            )
            reasons.extend(
                f"component_{index}:{reason}" for reason in component_reasons
            )
    elif family in {"actuation_delay", "coordination_delay"}:
        starts = [event for event in _events(protocol, "time_stall") if _effective(event)]
        if not starts:
            reasons.append("missing_effective_time_stall")
        elif not any(
            max((float(value) for value in event.get("requested_translation_distance", {}).values()), default=0.0)
            > 0.0
            for event in starts
        ):
            reasons.append("time_stall_without_requested_motion")
    elif family == "environment_change":
        changes = [
            event for event in _events(protocol, "environment_change") if _effective(event)
        ]
        if not changes:
            reasons.append("missing_effective_environment_change")
        elif not any(float(event.get("task_state_l2_change", 0.0)) > 0.0 for event in changes):
            reasons.append("environment_change_without_state_change")
    elif family == "missed_interaction":
        failures = [event for event in _events(protocol, "grasp_failure") if _effective(event)]
        if not failures:
            reasons.append("missing_effective_grasp_failure")
        elif not any(event.get("target_objects") for event in failures):
            reasons.append("grasp_failure_without_target")
    elif family == "relation_loss":
        losses = [
            event
            for event in _events(protocol, "relation_mismatch", "unexpected_drop")
            if _effective(event)
        ]
        effects = [
            event for event in _events(protocol, "physical_fault_effect_audit") if _effective(event)
        ]
        if not losses:
            reasons.append("missing_effective_relation_loss")
        else:
            if not any(int(event.get("stable_interaction_cycles", 0)) >= 3 for event in losses):
                reasons.append("insufficient_stable_interaction_history")
            if not any(event.get("released_objects") for event in losses):
                reasons.append("relation_loss_without_target")
        if not effects:
            reasons.append("missing_physical_relation_effect_audit")
        elif not any(event.get("detached") is True for event in effects):
            reasons.append("relation_not_detached")
    else:
        reasons.append("unknown_family")
    return not reasons, reasons


def _first_cycle(events: Iterable[Mapping[str, Any]], kind: str) -> int | None:
    values = [
        int(event["policy_step"])
        for event in events
        if event.get("kind") == kind
        and _effective(event)
        and event.get("policy_step") is not None
    ]
    return min(values) if values else None


def _cycle_evidence(
    episode_path: Path,
    episode: Mapping[str, Any],
    family: str,
) -> dict[str, Any]:
    reference_value = episode.get("cycle_file")
    inferred_reference = reference_value is None
    if inferred_reference:
        # The isolated oracle-timing wrapper historically rewrote the episode
        # summary after EpisodeWriter.finalize().  Its retained cycle artifact
        # is still committed atomically under the canonical sibling filename,
        # but the second summary write omitted the three cycle-reference
        # fields.  Recover that lossless storage reference from the episode
        # filename; do not infer any physical event from the summary itself.
        cycle_path = (
            episode_path.parent.parent / "cycles" / (episode_path.stem + ".jsonl.gz")
        ).resolve()
    else:
        reference = Path(str(reference_value))
        cycle_path = (episode_path.parent / reference).resolve()
    if not cycle_path.is_file():
        raise RuntimeError(f"missing A5 cycle evidence: {cycle_path}")
    cycle_sha256 = _sha256(cycle_path)
    if not inferred_reference and cycle_sha256 != episode.get("cycle_file_sha256"):
        raise RuntimeError(f"A5 cycle evidence hash mismatch: {cycle_path}")
    onset = None
    temporal_end = None
    relation_restored = None
    targets: set[str] = set()
    cycle_records = 0
    with gzip.open(cycle_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            cycle_records += 1
            row = json.loads(line)
            cycle = int(row["cycle"])
            injector = row.get("execution", {}).get("injector")
            if not isinstance(injector, Mapping):
                continue
            targets.update(_target_objects(injector))
            valid, _reasons = _validate_effect(family, injector)
            if onset is None and valid:
                onset = cycle
            for event in injector.get("events", ()):
                if not isinstance(event, Mapping) or not _effective(event):
                    continue
                if onset is not None and event.get("kind") == "time_stall_ended":
                    temporal_end = cycle
                if onset is not None and event.get("kind") == "relation_restored":
                    relation_restored = cycle
    expected_records = episode.get("cycle_records", episode.get("cycles"))
    if expected_records is not None and cycle_records != int(expected_records):
        raise RuntimeError(
            "A5 cycle evidence length mismatch: "
            f"{cycle_path} ({cycle_records} != {expected_records})"
        )
    return {
        "cycle_path": cycle_path,
        "cycle_sha256": cycle_sha256,
        "cycle_reference_inferred": inferred_reference,
        "cycle_records": cycle_records,
        "onset": onset,
        "temporal_end": temporal_end,
        "relation_restored": relation_restored,
        "target_objects": sorted(targets),
    }


def _target_objects(protocol: Mapping[str, Any]) -> list[str]:
    values = list(protocol.get("target_objects") or ())
    if not values:
        for event in protocol.get("events", ()):
            values.extend(event.get("target_objects") or ())
            values.extend(event.get("released_objects") or ())
    if not values:
        for component in protocol.get("components", ()):
            if isinstance(component, Mapping):
                values.extend(_target_objects(component))
    return sorted({str(value) for value in values})


def _reaudit(method_key: str, episode_path: Path, episode: Mapping[str, Any]) -> dict[str, Any]:
    family = str(episode.get("fault_family"))
    protocol = episode.get("fault_protocol")
    if family not in FAMILIES:
        raise ValueError(f"unknown A5 fault family: {family}")
    if not isinstance(protocol, Mapping):
        protocol = {}
    valid_effect, invalid_reasons = _validate_effect(family, protocol)
    cycle_evidence = _cycle_evidence(episode_path, episode, family)
    onset = cycle_evidence["onset"] if valid_effect else None
    physically_triggered = bool(valid_effect and onset is not None)
    if valid_effect and onset is None:
        invalid_reasons.append("physical_effect_missing_from_retained_cycle")
    violation_end = None
    relation_restored = None
    if physically_triggered and family in {"actuation_delay", "coordination_delay"}:
        violation_end = cycle_evidence["temporal_end"]
    elif physically_triggered and family in {"missed_interaction", "relation_loss"}:
        relation_restored = cycle_evidence["relation_restored"]
        if relation_restored is not None and relation_restored >= int(onset):
            violation_end = relation_restored
        else:
            relation_restored = None

    source_audit = episode.get("audit") if isinstance(episode.get("audit"), Mapping) else {}
    reconstructed = {
        "schema": SCHEMA,
        "method_key": method_key,
        "episode_id": str(episode["episode_id"]),
        "task": str(episode["task"]),
        "fault_family": family,
        "eligible": bool(source_audit.get("eligible") or protocol.get("triggered")),
        "physically_triggered": physically_triggered,
        "violation_onset_cycle": onset,
        "violation_end_cycle": violation_end,
        "relation_restored_cycle": relation_restored,
        "legal_reentry_cycle": episode.get("legal_reentry_cycle"),
        "target_objects": (
            cycle_evidence["target_objects"] or _target_objects(protocol)
        ) if physically_triggered else [],
        "oracle_recoverable": source_audit.get("oracle_recoverable"),
        "effect_evidence_valid": valid_effect,
        "effect_evidence_failures": invalid_reasons,
        "source": {
            "episode_path": str(episode_path.relative_to(ROOT)),
            "episode_sha256": _sha256(episode_path),
            "cycle_file_sha256": cycle_evidence["cycle_sha256"],
            "cycle_path": str(cycle_evidence["cycle_path"].relative_to(ROOT)),
            "cycle_reference_inferred": cycle_evidence[
                "cycle_reference_inferred"
            ],
            "cycle_records": cycle_evidence["cycle_records"],
            "embedded_audit_sha256": _canonical_sha256(source_audit),
        },
    }
    compared = (
        "eligible",
        "physically_triggered",
        "violation_onset_cycle",
        "violation_end_cycle",
        "relation_restored_cycle",
        "target_objects",
    )
    reconstructed["changed_embedded_fields"] = [
        field for field in compared if source_audit.get(field) != reconstructed.get(field)
    ]
    return reconstructed


def reaudit_episode(
    method_key: str,
    episode_path: Path,
    episode: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Apply the current public physical audit to one retained episode."""

    path = Path(episode_path)
    value = (
        json.loads(path.read_text(encoding="utf-8"))
        if episode is None
        else episode
    )
    return _reaudit(str(method_key), path, value)


def rebuild() -> dict[str, Any]:
    plan = json.loads((RESULT_ROOT / "A5_RUN_PLAN.json").read_text(encoding="utf-8"))
    methods = tuple(str(item["key"]) for item in plan["methods"])
    rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for method_key in methods:
        root = ENDPOINT_ROOT / method_key / "perturbed" / "episodes"
        paths = sorted(root.glob("*.json"))
        if len(paths) != 2000:
            raise RuntimeError(f"incomplete perturbed endpoint cell: {method_key} ({len(paths)})")
        changed = 0
        triggered = 0
        by_family: dict[str, int] = {family: 0 for family in sorted(FAMILIES)}
        for path in paths:
            episode = json.loads(path.read_text(encoding="utf-8"))
            row = _reaudit(method_key, path, episode)
            rows.append(row)
            changed += bool(row["changed_embedded_fields"])
            if row["physically_triggered"]:
                triggered += 1
                by_family[row["fault_family"]] += 1
        counts[method_key] = {
            "episodes": len(paths),
            "changed_from_embedded_audit": changed,
            "physically_triggered": triggered,
            "triggered_by_family": by_family,
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = EVENTS_PATH.with_name(".events.jsonl.gz.tmp")
    with gzip.open(staging, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(staging, EVENTS_PATH)
    index = {
        "schema": SCHEMA,
        "status": "CURRENT",
        "source_rows_are_immutable": True,
        "replaces_embedded_audit_for_all_a5_derived_event_metrics": True,
        "auditor_implementation": {
            "path": str(AUDITOR_PATH.relative_to(ROOT)),
            "sha256": _sha256(AUDITOR_PATH),
        },
        "events_path": str(EVENTS_PATH.relative_to(ROOT)),
        "events_sha256": _sha256(EVENTS_PATH),
        "episodes": len(rows),
        "methods": counts,
        "legacy_policy": (
            "Embedded source audits remain only as immutable provenance; all current A5 "
            "event, delay, detection, restoration, and re-entry summaries must use this artifact."
        ),
    }
    temp = INDEX_PATH.with_name(".REAUDIT_INDEX.json.tmp")
    temp.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, INDEX_PATH)
    return index


def load_current_reaudit() -> dict[tuple[str, str], dict[str, Any]]:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if index.get("schema") != SCHEMA or index.get("status") != "CURRENT":
        raise RuntimeError("A5 current re-audit index is missing or stale")
    path = ROOT / str(index["events_path"])
    if _sha256(path) != index.get("events_sha256"):
        raise RuntimeError("A5 current re-audit hash mismatch")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["method_key"]), str(row["episode_id"]))
            if key in result:
                raise RuntimeError(f"duplicate A5 re-audit row: {key}")
            result[key] = row
    if len(result) != int(index["episodes"]):
        raise RuntimeError("A5 re-audit row count mismatch")
    return result


def main() -> int:
    print(json.dumps(rebuild(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
