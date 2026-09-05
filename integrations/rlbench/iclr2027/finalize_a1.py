"""Finalize the A1 task/model/development evidence without paper statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from integrations.rlbench.iclr2027.asset_audit import audit_all_task_assets
from integrations.rlbench.iclr2027.task_registry import TASKS
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import (
    atomic_json,
    reserve_output,
)

SCHEMA = "essay2608.iclr2027.a1-acceptance.v1"
DEFAULT_GATE_ROOT = INTEGRATION_ROOT / "results" / "iclr2027" / "a1_development_gate"
DEFAULT_PHASE6_ROOT = INTEGRATION_ROOT / "results" / "phase6_formal_v1" / "normal"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(INTEGRATION_ROOT))
    except ValueError:
        return str(path)


def _unimanual_gate(task_id: str, parts_root: Path) -> dict[str, Any]:
    paths = sorted(parts_root.glob(f"{task_id}_part*.json"))
    if not paths:
        raise RuntimeError(f"{task_id}: no A1 development-gate shards")
    rows = []
    shard_status = []
    for path in paths:
        payload = _load(path)
        if payload.get("schema") != "essay2608.iclr2027.a1-development-gate.v1":
            raise RuntimeError(f"{task_id}: invalid development-gate schema")
        if payload.get("task_id") != task_id:
            raise RuntimeError(f"{task_id}: shard task identity mismatch")
        shard_status.append(payload.get("status"))
        rows.extend(payload.get("results", ()))
    episodes = [int(row["episode"]) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    if sorted(episodes) != list(range(20)) or len(seeds) != len(set(seeds)):
        raise RuntimeError(f"{task_id}: development episodes are incomplete or overlap")
    infrastructure_errors = sum(
        row.get("reason") in {"joint_hold_failed", "infrastructure_error"}
        for row in rows
    )
    incomplete_episodes = sum(
        row.get("reason") == "development_horizon" for row in rows
    )
    invalid_actions = sum(int(row.get("invalid_actions", 0)) for row in rows)
    episodes_with_invalid_actions = sum(
        int(row.get("invalid_actions", 0)) > 0 for row in rows
    )
    successes = sum(bool(row.get("success")) for row in rows)
    task_api_readable = all(
        row.get("initial_audit", {}).get("task_low_dim_finite") is True
        and bool(row.get("initial_audit", {}).get("success_conditions"))
        and bool(row.get("initial_audit", {}).get("relation_observation"))
        for row in rows
    )
    status = (
        "PASS"
        if set(shard_status) == {"PASS"}
        and infrastructure_errors == 0
        and incomplete_episodes == 0
        and task_api_readable
        else "FAIL"
    )
    return {
        "status": status,
        "evidence_source": "A1_NEW_DEVELOPMENT_EPISODES",
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "policy_outcome_classification": (
            "SUCCESS_OBSERVED"
            if successes > 0
            else "FROZEN_BACKBONE_LIMITATION_OBSERVED"
        ),
        "infrastructure_errors": infrastructure_errors,
        "incomplete_episodes": incomplete_episodes,
        "invalid_actions": invalid_actions,
        "episodes_with_invalid_actions": episodes_with_invalid_actions,
        "task_api_readable": task_api_readable,
        "reason_counts": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "shards": [_display_path(path) for path in paths],
    }


def _reused_bimanual_gate(task_id: str, phase6_root: Path) -> dict[str, Any]:
    path = phase6_root / task_id / "full_n200.json"
    payload = _load(path)
    rows = payload.get("results")
    if (
        payload.get("task") != task_id
        or payload.get("episodes") != 200
        or payload.get("episodes_completed") != 200
        or not isinstance(rows, list)
        or len(rows) != 200
    ):
        raise RuntimeError(f"{task_id}: reused phase-six evidence is incomplete")
    infrastructure_reasons = {
        "joint_hold_failed",
        "policy_error",
        "simulator_error",
        "infrastructure_error",
    }
    infrastructure_errors = sum(
        row.get("reason") in infrastructure_reasons for row in rows
    )
    successes = sum(bool(row.get("success")) for row in rows)
    return {
        "status": "PASS" if infrastructure_errors == 0 and successes > 0 else "FAIL",
        "evidence_source": "REUSED_AUTHENTICATED_PHASE6_DEVELOPMENT",
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "infrastructure_errors": infrastructure_errors,
        "reason_counts": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "source": str(path.relative_to(INTEGRATION_ROOT)),
        "paper_result_reuse": False,
    }


def finalize(parts_root: Path, phase6_root: Path) -> dict[str, Any]:
    assets = audit_all_task_assets()
    gates = {}
    for task_id, task in TASKS.items():
        gates[task_id] = (
            _reused_bimanual_gate(task_id, phase6_root)
            if task.spec.bimanual
            else _unimanual_gate(task_id, parts_root)
        )
    failed = sorted(
        task_id for task_id, gate in gates.items() if gate["status"] != "PASS"
    )
    backbone_limitations = sorted(
        task_id
        for task_id, gate in gates.items()
        if gate.get("policy_outcome_classification")
        == "FROZEN_BACKBONE_LIMITATION_OBSERVED"
    )
    return {
        "schema": SCHEMA,
        "status": "PASS" if assets["status"] == "PASS" and not failed else "FAIL",
        "purpose": "A1_TASK_ASSET_ACCEPTANCE_NOT_PAPER_RESULTS",
        "phase6_reuse_decisions": {
            "reused_exact_task_demonstrations_and_base_models": [
                "bimanual_handover_item",
                "bimanual_lift_tray",
                "bimanual_sweep_to_dustpan",
                "bimanual_put_bottle_in_fridge",
            ],
            "rebuilt_closed_loop_sidecars_with_current_normal_task_model": list(TASKS),
            "rebuilt_from_new_five_demonstrations": [
                task_id for task_id, task in TASKS.items() if not task.spec.bimanual
            ],
            "not_reused": {
                "phase6_place_cups": (
                    "Phase-six PlaceCups is a one-cup task; Main-10 and Native-6 "
                    "freeze the native three-cup repeated-interaction level."
                ),
                "phase6_numeric_results": (
                    "Development evidence only; no value is inserted into E1-E6."
                ),
            },
        },
        "task_asset_audit": assets,
        "development_gates": gates,
        "failed_tasks": failed,
        "backbone_limitations": backbone_limitations,
        "fault_trigger_qualification": {
            "status": "SCHEDULED_IN_A2",
            "reason": (
                "Physical trigger-rate qualification requires the frozen common "
                "fault layer and auditor built in A2; it remains mandatory before "
                "any sealed test."
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, default=DEFAULT_GATE_ROOT / "parts")
    parser.add_argument("--phase6-root", type=Path, default=DEFAULT_PHASE6_ROOT)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_GATE_ROOT / "A1_ACCEPTANCE.json"
    )
    args = parser.parse_args(argv)
    payload = finalize(args.parts_root, args.phase6_root)
    with reserve_output(args.output):
        atomic_json(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "failed_tasks": payload["failed_tasks"]}
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
