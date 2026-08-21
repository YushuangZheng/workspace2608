"""Frozen V3 release profiles and checkpoint-backed trigger authentication.

This module is deliberately Python 3.8 compatible because both the simulator
evaluators and the Python 3.10 policy worker import it.  It contains no RLBench
or PyRep imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
V3_INTERVENTION_CONFIG = INTEGRATION_ROOT / "configs" / "v3_interventions.json"
V3_MOTION_SOURCE_CONFIG = INTEGRATION_ROOT / "configs" / "v3_motion_sources.json"
V3_INTERVENTION_SCHEMA = "rlbench-dynamac-v3-interventions-v2"
V3_MOTION_SOURCE_SCHEMA = "rlbench-dynamac-v3-motion-sources-v1"
V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA = "dynamac-checkpoint-trigger-audit-v1"
V3_TRIGGER_ANCHOR_EVIDENCE_SCHEMA = "dynamac-v3-trigger-anchor-evidence-v1"
V3_SELECTION_SEMANTICS_ID = (
    "eq5_skill_majority_gate_timestep_availability_before_eq6_and_poe_"
    "time_state_position3d_unimodal_v1"
)

_DYNAMIC_TASKS = frozenset(
    {
        "stack_wine",
        "place_cups",
        "open_microwave",
        "wipe_desk",
        "bimanual_put_bottle_in_fridge",
        "bimanual_handover_item",
        "bimanual_lift_tray",
        "bimanual_sweep_to_dustpan",
    }
)
_COORDINATION_SCENARIOS = frozenset(
    {"coordination_hand_left", "coordination_hand_right"}
)


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(value, field):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value, field):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validate_phase(profile, scope):
    duration = _positive_integer(profile.get("expected_duration"), scope + ".expected_duration")
    if duration < 2:
        raise ValueError(f"{scope}.expected_duration must be at least two")
    tick = _nonnegative_integer(profile.get("local_tick"), scope + ".local_tick")
    if tick >= duration:
        raise ValueError(f"{scope}.local_tick is outside its skill")
    phase = profile.get("phase")
    if (
        not isinstance(phase, (int, float))
        or isinstance(phase, bool)
        or not math.isfinite(float(phase))
        or not math.isclose(
            float(phase),
            tick / float(duration - 1),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError(f"{scope}.phase does not equal local_tick/(duration-1)")


def _validate_anchor_common(profile, scope, allowed_arms):
    if not isinstance(profile, dict):
        raise ValueError(f"{scope} must be an object")
    if profile.get("anchor_arm") not in allowed_arms:
        raise ValueError(f"{scope}.anchor_arm is invalid")
    _nonnegative_integer(profile.get("skill_label"), scope + ".skill_label")
    frame = profile.get("evidence_frame")
    if not isinstance(frame, str) or not frame:
        raise ValueError(f"{scope}.evidence_frame must be a non-empty string")
    _validate_phase(profile, scope)


def _validate_dynamic_semantics(profile, scope, allowed_arms):
    interaction_arm = profile.get("interaction_arm")
    if interaction_arm not in allowed_arms:
        raise ValueError(f"{scope}.interaction_arm is invalid")
    for field in ("interaction_object", "interaction_event"):
        value = profile.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{scope}.{field} must be a non-empty string")
    if profile.get("expected_gripper_state") not in {"open", "closed"}:
        raise ValueError(f"{scope}.expected_gripper_state is invalid")


def load_v3_intervention_protocol(path=V3_INTERVENTION_CONFIG):
    """Load and fail-closed validate the frozen V3 intervention registry."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != V3_INTERVENTION_SCHEMA:
        raise ValueError("unsupported V3 intervention protocol schema")
    if set(payload) != {
        "schema",
        "release",
        "phase_formula",
        "smooth_steps",
        "final_settling_physics_steps",
        "dynamic_environment",
        "coordination",
        "provenance",
    }:
        raise ValueError("V3 intervention protocol fields are invalid")
    if payload.get("release") != "v3":
        raise ValueError("V3 intervention registry has the wrong release")
    if payload.get("phase_formula") != "local_tick / (skill_duration - 1)":
        raise ValueError("V3 trigger phase formula is not frozen")
    _positive_integer(payload.get("smooth_steps"), "smooth_steps")
    _positive_integer(
        payload.get("final_settling_physics_steps"),
        "final_settling_physics_steps",
    )
    dynamic = payload.get("dynamic_environment")
    if not isinstance(dynamic, dict) or frozenset(dynamic) != _DYNAMIC_TASKS:
        raise ValueError("V3 dynamic trigger registry task set is incomplete")
    for task, profile in dynamic.items():
        if not isinstance(profile, dict) or set(profile) != {
            "anchor_arm",
            "evidence_frame",
            "expected_duration",
            "expected_gripper_state",
            "interaction_arm",
            "interaction_event",
            "interaction_object",
            "local_tick",
            "phase",
            "required_active_window",
            "skill_label",
        }:
            raise ValueError(f"dynamic_environment.{task} fields are invalid")
        allowed_arms = {"left", "right"} if task.startswith("bimanual_") else {"single"}
        _validate_anchor_common(profile, f"dynamic_environment.{task}", allowed_arms)
        _validate_dynamic_semantics(
            profile,
            f"dynamic_environment.{task}",
            allowed_arms,
        )
        window = profile.get("required_active_window")
        if (
            not isinstance(window, list)
            or len(window) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in window)
            or window[0] != profile["local_tick"]
            or window[1] != window[0] + payload["smooth_steps"] - 1
            or window[1] >= profile["expected_duration"]
        ):
            raise ValueError(f"dynamic_environment.{task} active window is invalid")
    coordination = payload.get("coordination")
    if not isinstance(coordination, dict) or frozenset(coordination) != _COORDINATION_SCENARIOS:
        raise ValueError("V3 coordination trigger registry is incomplete")
    for scenario, profile in coordination.items():
        if not isinstance(profile, dict) or set(profile) != {
            "anchor_arm",
            "evidence_frame",
            "expected_duration",
            "expected_gripper_states",
            "global_tick",
            "handover_stage",
            "interaction_event",
            "interaction_object",
            "local_tick",
            "perturbed_arm",
            "phase",
            "skill_label",
        }:
            raise ValueError(f"coordination.{scenario} fields are invalid")
        _validate_anchor_common(profile, f"coordination.{scenario}", {"left", "right"})
        if profile.get("perturbed_arm") not in {"left", "right"}:
            raise ValueError(f"coordination.{scenario}.perturbed_arm is invalid")
        for field in ("interaction_object", "interaction_event", "handover_stage"):
            value = profile.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"coordination.{scenario}.{field} must be a non-empty string"
                )
        grippers = profile.get("expected_gripper_states")
        if (
            not isinstance(grippers, dict)
            or set(grippers) != {"left", "right"}
            or any(value not in {"open", "closed"} for value in grippers.values())
        ):
            raise ValueError(
                f"coordination.{scenario}.expected_gripper_states is invalid"
            )
        _nonnegative_integer(profile.get("global_tick"), f"coordination.{scenario}.global_tick")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not all(
        provenance.get(key) is True
        for key in (
            "frozen_before_v3_formal_evaluation",
            "manifests_reauthenticated",
            "integer_ticks_authoritative",
            "result_based_retuning_forbidden",
        )
    ) or provenance.get("model_weights_retrained_for_trigger_change") is not False:
        raise ValueError("V3 intervention provenance is not frozen")
    payload["fingerprint"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    return payload


def load_v3_motion_source_protocol(path=V3_MOTION_SOURCE_CONFIG):
    """Load the spatial source registry, independent of temporal triggers."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != V3_MOTION_SOURCE_SCHEMA:
        raise ValueError("unsupported V3 motion-source protocol schema")
    if set(payload) != {
        "schema",
        "release",
        "source_selection_max_attempts",
        "goal_sampling_max_attempts",
        "selection_policy",
        "sampling_scope",
        "result_based_retuning_forbidden",
        "tasks",
    }:
        raise ValueError("V3 motion-source protocol fields are invalid")
    if (
        payload.get("release") != "v3"
        or payload.get("source_selection_max_attempts") != 20
        or payload.get("goal_sampling_max_attempts") != 100
        or payload.get("selection_policy")
        != "manual_spatial_task_semantics_before_formal_results"
        or payload.get("sampling_scope")
        != "workspace_boundary_root_rigid_transform"
        or payload.get("result_based_retuning_forbidden") is not True
    ):
        raise ValueError("V3 motion-source protocol header is invalid")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or frozenset(tasks) != _DYNAMIC_TASKS:
        raise ValueError("V3 motion-source task set is incomplete")
    common = {
        "spatial_root_name",
        "spatial_root_type",
        "moved_entity_scope",
        "spatial_selection_reason",
    }
    for task, profile in tasks.items():
        if not isinstance(profile, dict):
            raise ValueError(f"motion-source profile {task!r} is invalid")
        if set(profile) != common:
            raise ValueError(f"motion-source profile {task!r} fields are invalid")
        if (
            not isinstance(profile["spatial_root_name"], str)
            or not profile["spatial_root_name"]
            or profile["spatial_root_type"] not in {"DUMMY", "SHAPE"}
            or not isinstance(profile["moved_entity_scope"], str)
            or not profile["moved_entity_scope"]
            or not isinstance(profile["spatial_selection_reason"], str)
            or not profile["spatial_selection_reason"]
        ):
            raise ValueError(f"motion-source profile {task!r} values are invalid")
    payload["fingerprint"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    return payload


def motion_source_profile(task, protocol=None):
    selected = load_v3_motion_source_protocol() if protocol is None else protocol
    try:
        return dict(selected["tasks"][task])
    except KeyError as exc:
        raise KeyError(f"no frozen V3 motion source for {task!r}") from exc


def dynamic_trigger_profile(task, protocol=None):
    selected = load_v3_intervention_protocol() if protocol is None else protocol
    try:
        return dict(selected["dynamic_environment"][task])
    except KeyError as exc:
        raise KeyError(f"no frozen V3 dynamic trigger for {task!r}") from exc


def coordination_trigger_profile(scenario, protocol=None):
    selected = load_v3_intervention_protocol() if protocol is None else protocol
    try:
        return dict(selected["coordination"][scenario])
    except KeyError as exc:
        raise KeyError(f"no frozen V3 coordination trigger for {scenario!r}") from exc


def _true_runs(values):
    mask = [bool(value) for value in values]
    runs = []
    start = None
    for index, active in enumerate(mask + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append([start, index - 1])
            start = None
    return runs


def checkpoint_trigger_audit(policy):
    """Return a compact, exact digest input for one fitted DynaMAC checkpoint."""

    skills = []
    for skill in policy.skills:
        frames = {}
        for name in sorted(skill.streams):
            stream = skill.streams[name]
            availability = np.asarray(stream.availability, dtype=bool)
            active = np.asarray(stream.active, dtype=bool)
            if availability.ndim == 1:
                availability = availability[None, :]
            if active.ndim == 1:
                active = active[None, :]
            selected = np.asarray(stream.selected_by_eq6, dtype=bool)
            if (
                availability.ndim != 2
                or active.shape != availability.shape
                or selected.shape != (availability.shape[0],)
                or availability.shape[1] != skill.duration
            ):
                raise RuntimeError("checkpoint trigger mask shape is invalid")
            expected_active = availability & selected[:, None]
            if not np.array_equal(active, expected_active):
                raise RuntimeError("checkpoint PoE mask is not Eq5 availability AND Eq6 selection")
            diagnostic = skill.link_diagnostics.get(name, {})
            raw = diagnostic.get("raw_link_mask") if isinstance(diagnostic, dict) else None
            raw_runs = None
            majority_gate_enabled = None
            if raw is not None:
                raw_mask = np.asarray(raw, dtype=bool)
                if raw_mask.ndim == 1:
                    raw_mask = raw_mask[None, :]
                if raw_mask.shape != availability.shape:
                    raise RuntimeError("checkpoint raw Eq5 mask shape is invalid")
                raw_runs = [_true_runs(row) for row in raw_mask]
                majority_gate_enabled = [
                    bool(float(np.mean(row)) > 0.5) for row in raw_mask
                ]
                if policy.config.link_mask_scope == "skill_majority_gate_timestep":
                    expected_availability = np.stack(
                        [
                            ~row if enabled else np.ones(row.shape, dtype=bool)
                            for row, enabled in zip(
                                raw_mask,
                                majority_gate_enabled,
                            )
                        ]
                    )
                    if not np.array_equal(availability, expected_availability):
                        raise RuntimeError(
                            "checkpoint availability is not majority-gated raw Eq5"
                        )
            frames[name] = {
                "selected_by_eq6": selected.tolist(),
                "raw_link_runs": raw_runs,
                "majority_gate_enabled": majority_gate_enabled,
                "availability_runs": [_true_runs(row) for row in availability],
                "poe_active_runs": [_true_runs(row) for row in active],
            }
        skills.append(
            {
                "label": int(skill.label),
                "duration": int(skill.duration),
                "frames": frames,
            }
        )
    audit = {
        "schema": V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA,
        "selection_semantics_id": policy.selection_semantics_id,
        "link_mask_scope": policy.config.link_mask_scope,
        "link_filter": policy.config.link_filter,
        "skill_sequence": [int(label) for label in policy.skill_sequence],
        "skills": skills,
    }
    audit["fingerprint"] = _canonical_sha256(audit)
    return audit


def bimanual_checkpoint_trigger_audit(policy):
    audit = {
        "schema": V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA,
        "arms": {
            "left": checkpoint_trigger_audit(policy.left),
            "right": checkpoint_trigger_audit(policy.right),
        },
    }
    audit["fingerprint"] = _canonical_sha256(audit)
    return audit


def _run_contains(runs, first, last):
    return any(run[0] <= first and run[1] >= last for run in runs)


def _anchor_from_arm_audit(arm_audit, profile, require_full_window):
    skills = arm_audit.get("skills")
    matches = [
        skill
        for skill in skills if skill.get("label") == profile.get("skill_label")
    ] if isinstance(skills, list) else []
    if len(matches) != 1:
        raise RuntimeError("V3 trigger skill is absent or duplicated in checkpoint")
    skill = matches[0]
    duration = skill.get("duration")
    if duration != profile.get("expected_duration"):
        raise RuntimeError("V3 trigger skill duration differs from preregistration")
    frame = skill.get("frames", {}).get(profile.get("evidence_frame"))
    if not isinstance(frame, dict):
        raise RuntimeError("V3 trigger evidence frame was rejected by Equation (6)")
    selected = frame.get("selected_by_eq6")
    availability_runs = frame.get("availability_runs")
    active_runs = frame.get("poe_active_runs")
    if (
        not isinstance(selected, list)
        or selected != [True]
        or not isinstance(availability_runs, list)
        or len(availability_runs) != 1
        or not isinstance(active_runs, list)
        or len(active_runs) != 1
    ):
        raise RuntimeError("V3 trigger requires one authenticated selected policy mode")
    first = int(profile["local_tick"])
    last = first
    if require_full_window:
        first, last = [int(value) for value in profile["required_active_window"]]
    if not _run_contains(availability_runs[0], first, last):
        raise RuntimeError("V3 trigger window is not Equation (5)-available")
    if not _run_contains(active_runs[0], first, last):
        raise RuntimeError("V3 trigger window does not participate in the final PoE")
    earlier_duration = 0
    for candidate in skills:
        if candidate is skill:
            break
        earlier_duration += int(candidate["duration"])
    resolved_global_tick = earlier_duration + int(profile["local_tick"])
    evidence = {
        "anchor_arm": profile["anchor_arm"],
        "skill_label": int(profile["skill_label"]),
        "duration": int(duration),
        "evidence_frame": profile["evidence_frame"],
        "local_tick": int(profile["local_tick"]),
        "phase": float(profile["phase"]),
        "phase_formula": "local_tick / (skill_duration - 1)",
        "resolved_global_tick": resolved_global_tick,
        "selected_by_eq6": list(selected),
        "availability_runs": availability_runs,
        "poe_active_runs": active_runs,
        "required_active_window": (
            list(profile["required_active_window"])
            if require_full_window
            else [first, last]
        ),
        "validated": True,
    }
    if require_full_window:
        evidence.update(
            {
                "interaction_arm": profile["interaction_arm"],
                "interaction_object": profile["interaction_object"],
                "interaction_event": profile["interaction_event"],
                "expected_gripper_state": profile["expected_gripper_state"],
            }
        )
    return evidence


def _arm_audit(checkpoint_audit, arm):
    if arm == "single":
        return checkpoint_audit
    arms = checkpoint_audit.get("arms")
    if not isinstance(arms, dict) or arm not in arms:
        raise RuntimeError("V3 trigger anchor arm is absent from checkpoint audit")
    return arms[arm]


def build_v3_trigger_anchor_evidence(task, checkpoint_audit, manifest):
    """Validate preregistered anchors against one loaded checkpoint audit."""

    protocol = load_v3_intervention_protocol()
    audits = (
        checkpoint_audit.get("arms", {}).values()
        if isinstance(checkpoint_audit.get("arms"), dict)
        else (checkpoint_audit,)
    )
    if any(
        audit.get("selection_semantics_id") != V3_SELECTION_SEMANTICS_ID
        or audit.get("link_mask_scope") != "skill_majority_gate_timestep"
        or audit.get("link_filter") != "none"
        for audit in audits
    ):
        raise RuntimeError("checkpoint does not implement the frozen V3 Eq5 gate")
    coordination_cohort = manifest.get("training_task") == "bimanual_handover_item_dynamic"
    anchors = {}
    if coordination_cohort:
        for scenario, profile in sorted(protocol["coordination"].items()):
            arm_audit = _arm_audit(checkpoint_audit, profile["anchor_arm"])
            evidence = _anchor_from_arm_audit(
                arm_audit,
                profile,
                require_full_window=False,
            )
            if evidence["resolved_global_tick"] != profile["global_tick"]:
                raise RuntimeError("V3 coordination global trigger tick changed")
            evidence["perturbed_arm"] = profile["perturbed_arm"]
            evidence["interaction_object"] = profile["interaction_object"]
            evidence["interaction_event"] = profile["interaction_event"]
            evidence["expected_gripper_states"] = dict(
                profile["expected_gripper_states"]
            )
            evidence["handover_stage"] = profile["handover_stage"]
            anchors[scenario] = evidence
        profile_family = "coordination"
    else:
        profile = dynamic_trigger_profile(task, protocol)
        arm_audit = _arm_audit(checkpoint_audit, profile["anchor_arm"])
        anchors[task] = _anchor_from_arm_audit(
            arm_audit,
            profile,
            require_full_window=True,
        )
        profile_family = "dynamic_environment"
    evidence = {
        "schema": V3_TRIGGER_ANCHOR_EVIDENCE_SCHEMA,
        "intervention_protocol_schema": protocol["schema"],
        "intervention_protocol_fingerprint": protocol["fingerprint"],
        "profile_family": profile_family,
        "checkpoint_trigger_audit_fingerprint": checkpoint_audit["fingerprint"],
        "anchors": anchors,
        "validated": bool(anchors) and all(item["validated"] for item in anchors.values()),
    }
    evidence["fingerprint"] = _canonical_sha256(evidence)
    return evidence


def resolve_authenticated_v3_trigger(model_identity, task=None, scenario=None):
    """Authenticate one evaluator trigger against the loaded V3 checkpoint.

    Exactly one of ``task`` (dynamic environment) or ``scenario``
    (``coordination_hand_left/right``) must be supplied.  The returned
    ``trigger_step`` is on the shared committed-policy clock.
    """

    if not isinstance(model_identity, dict):
        raise RuntimeError("V3 model identity is missing")
    if model_identity.get("manifest_authenticated") is not True:
        raise RuntimeError("V3 training manifest is not authenticated")
    if model_identity.get("training_manifest_schema") != "dynamac-direct-training-v3":
        raise RuntimeError("V3 evaluator received a non-V3 training manifest")
    if (task is None) == (scenario is None):
        raise ValueError("supply exactly one of task or coordination scenario")

    protocol = load_v3_intervention_protocol()
    if task is not None:
        if not isinstance(task, str) or not task:
            raise ValueError("task must be a non-empty string")
        key = task
        profile = dynamic_trigger_profile(task, protocol)
        family = "dynamic_environment"
    else:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        key = scenario
        profile = coordination_trigger_profile(scenario, protocol)
        family = "coordination"

    envelope = model_identity.get("v3_trigger_anchor_evidence")
    if not isinstance(envelope, dict):
        raise RuntimeError("V3 checkpoint trigger evidence is missing")
    fingerprint = envelope.get("fingerprint")
    unsigned = {name: value for name, value in envelope.items() if name != "fingerprint"}
    if (
        envelope.get("schema") != V3_TRIGGER_ANCHOR_EVIDENCE_SCHEMA
        or envelope.get("validated") is not True
        or envelope.get("profile_family") != family
        or envelope.get("intervention_protocol_schema") != protocol["schema"]
        or envelope.get("intervention_protocol_fingerprint") != protocol["fingerprint"]
        or envelope.get("checkpoint_trigger_audit_fingerprint")
        != model_identity.get("checkpoint_trigger_audit_fingerprint")
        or fingerprint != _canonical_sha256(unsigned)
    ):
        raise RuntimeError("V3 checkpoint trigger evidence envelope is invalid")
    anchors = envelope.get("anchors")
    anchor = anchors.get(key) if isinstance(anchors, dict) else None
    if not isinstance(anchor, dict) or anchor.get("validated") is not True:
        raise RuntimeError("V3 checkpoint trigger anchor is missing or invalid")

    expected = {
        "anchor_arm": profile["anchor_arm"],
        "skill_label": profile["skill_label"],
        "duration": profile["expected_duration"],
        "evidence_frame": profile["evidence_frame"],
        "local_tick": profile["local_tick"],
        "phase": profile["phase"],
        "phase_formula": "local_tick / (skill_duration - 1)",
    }
    if family == "dynamic_environment":
        expected.update(
            {
                "interaction_arm": profile["interaction_arm"],
                "interaction_object": profile["interaction_object"],
                "interaction_event": profile["interaction_event"],
                "expected_gripper_state": profile["expected_gripper_state"],
            }
        )
    if any(anchor.get(name) != value for name, value in expected.items()):
        raise RuntimeError("V3 checkpoint trigger anchor differs from preregistration")
    if family == "dynamic_environment":
        required_window = profile["required_active_window"]
    else:
        required_window = [profile["local_tick"], profile["local_tick"]]
        if (
            anchor.get("perturbed_arm") != profile["perturbed_arm"]
            or anchor.get("resolved_global_tick") != profile["global_tick"]
            or anchor.get("interaction_object") != profile["interaction_object"]
            or anchor.get("interaction_event") != profile["interaction_event"]
            or anchor.get("expected_gripper_states")
            != profile["expected_gripper_states"]
            or anchor.get("handover_stage") != profile["handover_stage"]
        ):
            raise RuntimeError("V3 coordination trigger differs from preregistration")
    if anchor.get("required_active_window") != required_window:
        raise RuntimeError("V3 checkpoint trigger active window is invalid")
    if anchor.get("selected_by_eq6") != [True]:
        raise RuntimeError("V3 trigger evidence frame is not Equation (6)-selected")
    availability = anchor.get("availability_runs")
    active = anchor.get("poe_active_runs")
    if (
        not isinstance(availability, list)
        or len(availability) != 1
        or not _run_contains(availability[0], required_window[0], required_window[1])
        or not isinstance(active, list)
        or len(active) != 1
        or not _run_contains(active[0], required_window[0], required_window[1])
    ):
        raise RuntimeError("V3 checkpoint trigger window is not active")
    trigger_step = anchor.get("resolved_global_tick")
    if not isinstance(trigger_step, int) or isinstance(trigger_step, bool) or trigger_step < 0:
        raise RuntimeError("V3 resolved committed trigger step is invalid")
    return {
        "trigger_step": trigger_step,
        "profile_key": key,
        "profile": profile,
        "evidence": anchor,
        "evidence_fingerprint": fingerprint,
        "intervention_protocol_fingerprint": protocol["fingerprint"],
    }


__all__ = [
    "V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA",
    "V3_INTERVENTION_CONFIG",
    "V3_INTERVENTION_SCHEMA",
    "V3_TRIGGER_ANCHOR_EVIDENCE_SCHEMA",
    "V3_SELECTION_SEMANTICS_ID",
    "bimanual_checkpoint_trigger_audit",
    "build_v3_trigger_anchor_evidence",
    "checkpoint_trigger_audit",
    "coordination_trigger_profile",
    "dynamic_trigger_profile",
    "load_v3_intervention_protocol",
    "resolve_authenticated_v3_trigger",
]
