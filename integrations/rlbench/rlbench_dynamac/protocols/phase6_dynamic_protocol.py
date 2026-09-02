"""Model-scoped authentication for the Stage-six smooth background.

The physical intervention profiles were frozen with the V3 evaluation set,
but Stage six may load a checkpoint retrained from the same successful
demonstrations (for example, the corrected WipeDesk segmentation).  A trigger
therefore belongs to the *loaded checkpoint*, not to a historical manifest
name.  This module verifies the frozen pre-interaction phase against the
loaded checkpoint audit and resolves it on that checkpoint's policy clock.

The module is Python 3.8 compatible because it is imported by both simulator
and policy-worker processes.
"""

from __future__ import annotations

import hashlib
import json

from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
    V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA,
    V3_SELECTION_SEMANTICS_ID,
    dynamic_trigger_profile,
    load_v3_intervention_protocol,
)


PHASE6_DYNAMIC_TRIGGER_EVIDENCE_SCHEMA = (
    "essay2608.phase6_model_scoped_smooth_trigger.v1"
)


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_contains(runs, first, last):
    return any(run[0] <= first and run[1] >= last for run in runs)


def _arm_audit(checkpoint_audit, arm):
    if arm == "single":
        return checkpoint_audit
    arms = checkpoint_audit.get("arms")
    if not isinstance(arms, dict) or arm not in arms:
        raise RuntimeError("Stage-six dynamic anchor arm is absent")
    return arms[arm]


def _validate_audit(audit):
    unsigned = {key: value for key, value in audit.items() if key != "fingerprint"}
    if (
        audit.get("schema") != V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA
        or audit.get("selection_semantics_id") != V3_SELECTION_SEMANTICS_ID
        or audit.get("link_mask_scope") != "skill_majority_gate_timestep"
        or audit.get("link_filter") != "none"
        or audit.get("fingerprint") != _canonical_sha256(unsigned)
    ):
        raise RuntimeError("loaded checkpoint trigger audit is invalid")


def build_phase6_dynamic_trigger_evidence(task, checkpoint_audit, manifest):
    """Bind the frozen physical phase to the checkpoint being loaded.

    The phase is mapped to the current skill duration.  The complete smooth
    window must remain Equation-(5)-available and Equation-(6)-selected in the
    loaded model.  No historical V3 duration is claimed for a retrained model.
    """

    protocol = load_v3_intervention_protocol()
    profile = dynamic_trigger_profile(task, protocol)
    audit = _arm_audit(checkpoint_audit, profile["anchor_arm"])
    _validate_audit(audit)

    skills = audit.get("skills")
    matches = (
        [skill for skill in skills if skill.get("label") == profile["skill_label"]]
        if isinstance(skills, list)
        else []
    )
    if len(matches) != 1:
        raise RuntimeError("Stage-six dynamic trigger skill is absent or duplicated")
    skill = matches[0]
    duration = skill.get("duration")
    if not isinstance(duration, int) or isinstance(duration, bool) or duration < 2:
        raise RuntimeError("Stage-six dynamic trigger skill duration is invalid")

    local_tick = int(round(float(profile["phase"]) * float(duration - 1)))
    smooth_steps = int(protocol["smooth_steps"])
    last_tick = local_tick + smooth_steps - 1
    if local_tick < 0 or last_tick >= duration:
        raise RuntimeError("Stage-six smooth window lies outside the current skill")

    frame = skill.get("frames", {}).get(profile["evidence_frame"])
    if not isinstance(frame, dict):
        raise RuntimeError("Stage-six dynamic evidence frame is absent")
    selected = frame.get("selected_by_eq6")
    availability = frame.get("availability_runs")
    active = frame.get("poe_active_runs")
    if (
        selected != [True]
        or not isinstance(availability, list)
        or len(availability) != 1
        or not isinstance(active, list)
        or len(active) != 1
        or not _run_contains(availability[0], local_tick, last_tick)
        or not _run_contains(active[0], local_tick, last_tick)
    ):
        raise RuntimeError(
            "Stage-six smooth window is not continuously observable and active"
        )

    earlier_duration = 0
    for candidate in skills:
        if candidate is skill:
            break
        earlier_duration += int(candidate["duration"])
    global_tick = earlier_duration + local_tick
    evidence = {
        "schema": PHASE6_DYNAMIC_TRIGGER_EVIDENCE_SCHEMA,
        "task": task,
        "manifest_schema": manifest.get("manifest_schema"),
        "training_fingerprint": manifest.get("fingerprint"),
        "checkpoint_trigger_audit_fingerprint": checkpoint_audit.get("fingerprint"),
        "source_intervention_protocol_schema": protocol["schema"],
        "source_intervention_protocol_fingerprint": protocol["fingerprint"],
        "anchor_arm": profile["anchor_arm"],
        "skill_label": int(profile["skill_label"]),
        "skill_duration": int(duration),
        "evidence_frame": profile["evidence_frame"],
        "frozen_preinteraction_phase": float(profile["phase"]),
        "resolved_local_tick": int(local_tick),
        "resolved_global_tick": int(global_tick),
        "required_active_window": [int(local_tick), int(last_tick)],
        "smooth_steps": smooth_steps,
        "selected_by_eq6": list(selected),
        "availability_runs": availability,
        "poe_active_runs": active,
        "interaction_arm": profile["interaction_arm"],
        "interaction_object": profile["interaction_object"],
        "interaction_event": profile["interaction_event"],
        "expected_gripper_state": profile["expected_gripper_state"],
        "validated": True,
    }
    evidence["fingerprint"] = _canonical_sha256(evidence)
    return evidence


def resolve_phase6_dynamic_trigger(model_identity, task, smooth_steps):
    """Authenticate a model-scoped Stage-six trigger at evaluation time."""

    if not isinstance(model_identity, dict):
        raise RuntimeError("Stage-six model identity is missing")
    if model_identity.get("manifest_authenticated") is not True:
        raise RuntimeError(
            "Stage-six dynamic evaluation requires an authenticated model"
        )
    evidence = model_identity.get("phase6_dynamic_trigger_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("Stage-six model-scoped dynamic trigger evidence is missing")
    fingerprint = evidence.get("fingerprint")
    unsigned = {key: value for key, value in evidence.items() if key != "fingerprint"}
    if (
        evidence.get("schema") != PHASE6_DYNAMIC_TRIGGER_EVIDENCE_SCHEMA
        or evidence.get("task") != task
        or evidence.get("validated") is not True
        or evidence.get("manifest_schema")
        != model_identity.get("training_manifest_schema")
        or evidence.get("checkpoint_trigger_audit_fingerprint")
        != model_identity.get("checkpoint_trigger_audit_fingerprint")
        or evidence.get("smooth_steps") != smooth_steps
        or fingerprint != _canonical_sha256(unsigned)
    ):
        raise RuntimeError("Stage-six model-scoped dynamic trigger evidence is invalid")
    trigger = evidence.get("resolved_global_tick")
    if not isinstance(trigger, int) or isinstance(trigger, bool) or trigger < 0:
        raise RuntimeError("Stage-six resolved dynamic trigger is invalid")
    return {
        "schema": "essay2608.phase6_dynamic_trigger_authentication.v1",
        "trigger_step": trigger,
        "profile_key": task,
        "evidence": evidence,
        "evidence_fingerprint": fingerprint,
        "source_intervention_protocol_fingerprint": evidence[
            "source_intervention_protocol_fingerprint"
        ],
    }


__all__ = [
    "PHASE6_DYNAMIC_TRIGGER_EVIDENCE_SCHEMA",
    "build_phase6_dynamic_trigger_evidence",
    "resolve_phase6_dynamic_trigger",
]
