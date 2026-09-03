"""Materialize the immutable, human-auditable E0 dynamic/fault contract.

This is a packaging layer over the already sealed ``rlbench_eval_v2`` plans.
It never samples scenes and never reads evaluation results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT

from .run_cell import PROTOCOL, load_protocol


OUTPUT = Path(__file__).with_name("frozen")
EVALUATION_ROOT = REPOSITORY_ROOT / "integrations/rlbench/data/evaluation"
PAPER_FAULT_NAMES = {
    "time_stall": "motion_stall",
    "grasp_failure": "initial_grasp_failure",
    "relation_mismatch": "post_link_displacement",
    "unexpected_drop": "unexpected_drop",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _episode_records(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    first, last = protocol["evaluation_set"]["fault_episode_index_range"]
    records = []
    for task in protocol["tasks"]:
        batch_path = EVALUATION_ROOT / "environment" / f"{task}_a_b_n200.json"
        envelope = json.loads(batch_path.read_text(encoding="utf-8"))
        batch = envelope.get("runtime_batch", envelope)
        plans = batch.get("plans")
        if not isinstance(plans, list) or len(plans) != 200:
            raise RuntimeError(f"sealed evaluation batch is incomplete: {batch_path}")
        for episode_index in range(int(first), int(last) + 1):
            plan = plans[episode_index]
            records.append(
                {
                    "schema": "essay2608.iclr2027_core8_dynamic_episode.v1",
                    "evaluation_set_id": protocol["evaluation_set"]["id"],
                    "task": task,
                    "episode_index": episode_index,
                    "episode_seed": int(plan["episode_seed"]),
                    "variation": int(plan["variation"]),
                    "plan_fingerprint": str(plan["fingerprint"]),
                    "batch_fingerprint": str(envelope["batch_fingerprint"]),
                }
            )
    return records


def materialize() -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    episode_records = _episode_records(protocol)
    episodes_path = OUTPUT / "episodes.jsonl"
    episodes_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in episode_records
        ),
        encoding="utf-8",
    )

    faults_path = OUTPUT / "faults.json"
    _write_json(
        faults_path,
        {
            "schema": "essay2608.iclr2027_core8_faults.v1",
            "trigger_basis": "external_physical_or_action_predicates",
            "policy_internal_state_available_to_injector": False,
            "intention_to_treat": True,
            "background": {
                "scenario": protocol["shared_execution"]["scenario_by_experiment"][
                    "fault"
                ],
                "smooth_steps": protocol["shared_execution"]["smooth_steps"],
                "fault_label": False,
                "fault_eligibility": "after_completed_scheduled_background_segments",
                "policy_state_available_to_scheduler": False,
            },
            "faults": {
                name: {"paper_name": PAPER_FAULT_NAMES[name], **configuration}
                for name, configuration in protocol["faults"].items()
            },
        },
    )

    schema_path = OUTPUT / "result_schema.json"
    _write_json(
        schema_path,
        {
            "schema": "essay2608.iclr2027_core8_result_contract.v1",
            "required_episode_fields": [
                "formal_episode_index",
                "formal_episode_seed",
                "success",
                "reason",
                "scenario_events",
                "recovery_audit",
            ],
            "fault_episode_required_fields": ["physical_fault"],
            "physical_fault_schema": "essay2608.rlbench.physical_fault.v2",
            "physical_background_fields": [
                "required",
                "configured",
                "scenario",
                "expected_segments",
                "completed_segments",
                "ready",
                "ready_policy_step",
                "ready_before_fault",
                "pre_background_policy_steps",
                "events",
            ],
            "physical_audit_fields": [
                "effect_observed",
                "effect_policy_step",
                "fault_end_policy_step",
                "target_arm",
                "target_objects",
                "relation_restored",
                "relation_restoration_policy_step",
                "cycles_to_relation_restoration",
            ],
            "recovery_audit_schema": "essay2608.phase6_recovery_outcome.v1",
            "recovery_audit_fields": [
                "fault_trigger_policy_step",
                "fault_effect_observed",
                "first_post_fault_alarm_policy_step",
                "alarm_delay_cycles",
                "intervention_entries",
                "intervention_cycles",
                "post_fault_recovery_cycles",
                "relation_restored",
                "legal_reentry",
                "legal_reentry_policy_step",
                "post_reentry_completion",
            ],
            "missing_value_rule": "null means not applicable or not physically observed; false means evaluated and not achieved",
        },
    )

    manifest_path = OUTPUT / "manifest.json"
    source_manifest = EVALUATION_ROOT / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "essay2608.iclr2027_core8_frozen_contract.v1",
            "evaluation_set_id": protocol["evaluation_set"]["id"],
            "tasks": list(protocol["tasks"]),
            "methods": list(protocol["methods"]),
            "fault_episode_count_per_task": int(
                protocol["evaluation_set"]["fault_episode_index_range"][1]
            )
            + 1,
            "dynamic_no_fault_episode_count_per_task": int(
                protocol["evaluation_set"]["dynamic_episode_index_range"][1]
            )
            + 1,
            "background_scenarios": dict(
                protocol["shared_execution"]["scenario_by_experiment"]
            ),
            "episode_records": len(episode_records),
            "source_evaluation_manifest": source_manifest.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "source_evaluation_manifest_sha256": _sha256(source_manifest),
            "formal_protocol": PROTOCOL.relative_to(REPOSITORY_ROOT).as_posix(),
            "formal_protocol_sha256": _sha256(PROTOCOL),
            "files": {
                path.name: _sha256(path)
                for path in (episodes_path, faults_path, schema_path)
            },
            "result_based_selection": False,
        },
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(materialize(), ensure_ascii=False, indent=2, sort_keys=True))
