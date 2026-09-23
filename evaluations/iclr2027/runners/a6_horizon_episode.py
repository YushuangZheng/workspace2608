"""A6-only episode entry point for Horizon-3 repeated-event schedules.

The frozen A5 episode runner remains untouched.  This module installs an
episode-local fault factory only for rows that explicitly request the E4
per-interaction schedule, then delegates all policy, simulator, monitor, and
result handling to the frozen runner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.audit.horizon_events import (
    EVENT_SCHEDULE,
    InteractionStageFaultEnvironment,
)
from evaluations.iclr2027.runners import shared_episode


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _per_interaction_steps(policy_steps: int, task_level: int) -> int:
    if policy_steps < 1 or task_level < 1:
        raise ValueError("Horizon-3 policy steps and task level must be positive")
    return max(1, int(round(float(policy_steps) / float(task_level))))


def _annotate_horizon_summary(output_root: Path, row: Mapping[str, Any]) -> dict:
    path = (
        Path(output_root)
        / "episodes"
        / (str(row["episode_id"]).replace("/", "__") + ".json")
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    protocol = value.get("fault_protocol") or {}
    if protocol.get("event_schedule") != EVENT_SCHEDULE:
        raise RuntimeError("per-stage episode did not emit Horizon-3 protocol metadata")
    interaction_count = int(protocol["interaction_count"])
    completed = list(protocol.get("completed_interactions", ()))
    restoration_records = []
    for component in protocol.get("components", ()):
        restoration_step = component.get("relation_restoration_policy_step")
        if component.get("relation_restored") is not True or restoration_step is None:
            continue
        restoration_step = int(restoration_step)
        completed_at_restoration = sum(
            int(item["completion_policy_step"]) <= restoration_step
            for item in completed
        )
        restoration_records.append(
            {
                "interaction_index": int(component["interaction_index"]),
                "restoration_policy_step": restoration_step,
                "completed_interactions_at_restoration": completed_at_restoration,
                "remaining_stages_after_repair": max(
                    0, interaction_count - completed_at_restoration
                ),
                "completed_episode_after_repair": bool(value["final_success"]),
            }
        )
    horizon_audit = {
        "schema": "essay2608.iclr2027.horizon-audit.v1",
        "stage_count": interaction_count,
        "event_opportunity_count": int(protocol["interaction_stages_reached"]),
        "actual_triggered_event_count": int(protocol["triggered_event_count"]),
        "completed_interaction_count": int(protocol["interaction_stages_completed"]),
        "remaining_stages_after_repair": restoration_records,
    }
    value["horizon_audit"] = horizon_audit
    # The common single-event auditor cannot represent multiple intervals.
    # Preserve its detailed first-event fields, but make the aggregate trigger
    # flag agree with the authoritative repeated-event protocol.
    value.setdefault("audit", {})["physically_triggered"] = bool(
        horizon_audit["actual_triggered_event_count"]
    )
    _atomic_json(path, value)
    return value


def run_episode(
    row: Mapping[str, Any],
    output_root: Path,
    **kwargs: Any,
) -> dict:
    if row.get("event_schedule") != EVENT_SCHEDULE:
        raise ValueError("A6 Horizon episode entry point requires a per-stage row")
    level = int(row.get("task_level") or 0)
    original_builder = shared_episode.build_fault_environment

    def repeated_builder(task_environment: Any, task: Any, **builder_kwargs: Any) -> Any:
        if task.task_level is None or int(task.task_level) != level:
            raise RuntimeError("manifest and registry Horizon-3 task levels disagree")
        return InteractionStageFaultEnvironment(
            task_environment,
            task,
            family=str(builder_kwargs["family"]),
            severity=str(builder_kwargs.get("severity") or "medium"),
            trigger_stage=str(builder_kwargs["trigger_stage"]),
            per_interaction_policy_steps=_per_interaction_steps(
                int(builder_kwargs["policy_steps"]), level
            ),
            config=builder_kwargs["config"],
            fault_builder=original_builder,
        )

    shared_episode.build_fault_environment = repeated_builder
    try:
        result = shared_episode.run_episode(row, output_root, **kwargs)
    finally:
        shared_episode.build_fault_environment = original_builder
    if result.get("reason") == "infrastructure_error":
        return result
    return _annotate_horizon_summary(Path(output_root), row)


def main(argv: list[str] | None = None) -> int:
    args = shared_episode._parser().parse_args(argv)
    row = shared_episode._load_manifest_row(args.manifest, args.episode_id)
    result = run_episode(
        row,
        args.output_root,
        policy_python=args.policy_python,
        method=args.method,
        calibration_artifact=args.calibration_artifact,
        policy_diagnostics_dir=args.policy_diagnostics_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["reason"] != "infrastructure_error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
