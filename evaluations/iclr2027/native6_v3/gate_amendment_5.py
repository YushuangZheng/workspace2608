"""Native-6 v3 completed-step re-audit and development gate Amendment 5."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gate_amendment_4 import build_development_gate_amendment_4


TIMEOUT_TYPE = "EpisodeProcessTimeout"


def _completed_step_effect(
    events: Sequence[Mapping[str, Any]],
    *,
    trigger_sim_step: int,
    targets: set[str],
) -> Mapping[str, Any] | None:
    for event in events:
        if int(event.get("sim_step", -1)) < int(trigger_sim_step):
            continue
        if not targets:
            return None
        if not all(float(value) > 0.9 for value in event.get("actual_open_amount", ())):
            continue
        if targets.intersection(event.get("contact_objects", ())):
            continue
        return event
    return None


def reaudit_ours_completed_step_effects(
    records: Iterable[Mapping[str, Any]],
    *,
    episode_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Correct audit metadata from immutable completed-step observations.

    Only the known maintained-contact case is eligible for correction.  No
    action, task-success value, termination, or raw artifact is modified.
    """

    corrected: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for source in records:
        row = copy.deepcopy(dict(source))
        eligible = (
            row.get("task") == "open_drawer"
            and row.get("fault_family") == "relation_loss"
            and bool(row.get("injection_triggered"))
            and not bool(row.get("physical_effect_confirmed"))
            and not bool(row.get("infrastructure_error"))
        )
        if eligible:
            path = Path(episode_root) / (
                str(row["episode_id"]).replace("/", "__") + ".json"
            )
            episode = json.loads(path.read_text(encoding="utf-8"))
            protocol = episode.get("fault_protocol") or {}
            targets = set(protocol.get("target_objects") or ())
            event = _completed_step_effect(
                protocol.get("events") or (),
                trigger_sim_step=int(row["trigger_sim_step"]),
                targets=targets,
            )
            if event is not None:
                before = {
                    key: row.get(key)
                    for key in (
                        "physical_effect_confirmed",
                        "physically_triggered",
                        "effect_sim_step",
                        "effect_time_s",
                        "violation_onset_sim_step",
                        "violation_onset_time_s",
                    )
                }
                row["physical_effect_confirmed"] = True
                row["physically_triggered"] = True
                row["effect_sim_step"] = int(event["sim_step"])
                row["effect_time_s"] = float(event["simulation_time_s"])
                row["violation_onset_sim_step"] = int(event["sim_step"])
                row["violation_onset_time_s"] = float(event["simulation_time_s"])
                corrections.append(
                    {
                        "episode_id": row["episode_id"],
                        "targets": sorted(targets),
                        "before": before,
                        "after": {
                            key: row.get(key)
                            for key in before
                        },
                        "task_success_unchanged": row.get("final_success"),
                    }
                )
        corrected.append(row)
    return corrected, corrections


def build_development_gate_amendment_5(
    records: Iterable[Mapping[str, Any]],
    *,
    system_id: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    gate = build_development_gate_amendment_4(rows, system_id=system_id)
    timeout_rows = [
        row
        for row in rows
        if bool(row.get("infrastructure_error"))
        and (row.get("error_detail") or {}).get("type") == TIMEOUT_TYPE
    ]
    hard_infrastructure = [
        row
        for row in rows
        if bool(row.get("infrastructure_error")) and row not in timeout_rows
    ]
    reasons = [
        reason
        for reason in gate["failure_reasons"]
        if reason != "one or more infrastructure errors"
    ]
    if hard_infrastructure:
        reasons.append("one or more non-timeout infrastructure errors")
    gate.update(
        schema="essay2608.iclr2027.native6-v3-development-gate.v3",
        protocol_amendment="native6_event_grounded_physics_v3_amendment_5",
        status="PASS" if not reasons else "FAIL",
        failure_reasons=reasons,
        infrastructure={
            "wall_time_exhaustions_report_only": len(timeout_rows),
            "hard_infrastructure_errors": len(hard_infrastructure),
            "wall_time_episode_ids": sorted(
                str(row["episode_id"]) for row in timeout_rows
            ),
        },
    )
    return gate


__all__ = [
    "build_development_gate_amendment_5",
    "reaudit_ours_completed_step_effects",
]
