"""Event-conditioned Native-6 v3 development gate under Amendment 4."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

from .events import FAULT_FAMILIES
from .gate import TASK_IDS


def build_development_gate_amendment_4(
    results: Union[Sequence[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    *,
    system_id: str,
    expected_per_cell: int = 10,
    minimum_eligible_events_per_cell: int = 1,
    minimum_eligible_to_injected: float = 0.8,
    minimum_injected_to_effect: float = 0.8,
) -> dict[str, Any]:
    """Gate adapter reliability conditional on a physical event being observed.

    ``assigned_to_eligible`` remains in every cell as native-policy event
    reachability, but it is not thresholded. All assigned episodes remain in
    later intention-to-treat evaluation.
    """

    rows = [dict(row) for row in results]
    expected_total = len(TASK_IDS) * len(FAULT_FAMILIES) * expected_per_cell
    failures: list[str] = []
    if len(rows) != expected_total:
        failures.append(f"expected {expected_total} rows, observed {len(rows)}")
    if len({row.get("episode_id") for row in rows}) != len(rows):
        failures.append("duplicate episode IDs")
    if any(row.get("method_id") != system_id for row in rows):
        failures.append("result method_id differs from the gated system")
    if any(bool(row.get("infrastructure_error")) for row in rows):
        failures.append("one or more infrastructure errors")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task")), str(row.get("fault_family")))].append(row)

    cells: dict[str, dict[str, Any]] = {}
    for task in TASK_IDS:
        for family in FAULT_FAMILIES:
            cell_rows = grouped[(task, family)]
            assigned = len(cell_rows)
            eligible = sum(bool(row.get("eligible")) for row in cell_rows)
            injected = sum(bool(row.get("injection_triggered")) for row in cell_rows)
            effect = sum(bool(row.get("physical_effect_confirmed")) for row in cell_rows)
            rates = {
                "assigned_to_eligible": eligible / assigned if assigned else 0.0,
                "eligible_to_injected": injected / eligible if eligible else 0.0,
                "injected_to_effect": effect / injected if injected else 0.0,
            }
            key = f"{task}/{family}"
            cells[key] = {
                "assigned": assigned,
                "eligible": eligible,
                "injection_triggered": injected,
                "physical_effect_confirmed": effect,
                **rates,
            }
            if assigned != expected_per_cell:
                failures.append(f"{key}: expected {expected_per_cell} assignments")
            if eligible < minimum_eligible_events_per_cell:
                failures.append(f"{key}: no eligible physical-event evidence")
            if rates["eligible_to_injected"] < minimum_eligible_to_injected:
                failures.append(f"{key}: insufficient eligible-to-injected coverage")
            if rates["injected_to_effect"] < minimum_injected_to_effect:
                failures.append(f"{key}: insufficient injected-to-effect coverage")

    identities = {
        field: sorted({str(row.get(field)) for row in rows})
        for field in (
            "config_identity",
            "checkpoint_identity",
            "environment_identity",
            "fault_adapter_identity",
            "fault_config_identity",
            "audit_protocol_revision",
            "manifest_identity",
        )
    }
    for field, values in identities.items():
        if len(values) != 1 or values == ["None"]:
            failures.append(f"{field} is not one non-empty frozen identity")

    return {
        "schema": "essay2608.iclr2027.native6-v3-development-gate.v2",
        "protocol_amendment": "native6_event_grounded_physics_v3_amendment_4",
        "system_id": system_id,
        "status": "PASS" if not failures else "FAIL",
        "episodes": len(rows),
        "expected_episodes": expected_total,
        "gate": {
            "minimum_eligible_events_per_cell": minimum_eligible_events_per_cell,
            "minimum_eligible_to_injected": minimum_eligible_to_injected,
            "minimum_injected_to_effect": minimum_injected_to_effect,
            "assigned_to_eligible_role": "REPORT_ONLY",
        },
        "cells": cells,
        "identity_values": identities,
        "failure_reasons": failures,
        "fault_counts": dict(Counter(str(row.get("fault_family")) for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("rvt", "racer", "ours"), required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate = build_development_gate_amendment_4(rows, system_id=args.system)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({"status": gate["status"], "episodes": gate["episodes"]}))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
