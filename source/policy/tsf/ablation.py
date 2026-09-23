"""Explicit, task-agnostic profiles for controlled method comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal


@dataclass(frozen=True)
class TSFFeatureProfile:
    """Select already implemented mechanisms without changing learned models.

    These profiles are experiment controls, not alternative task mechanisms.
    The profile changes only which already-defined evidence and control
    mechanisms are authoritative.  It never changes a learned task model or
    introduces task-specific behavior.
    """

    name: str
    dynamic_frame_roles: bool
    relation_scene_boundary_guards: bool
    active_relation_verification: bool
    state_aware_recovery_reentry: bool
    complete_state_progress_evidence: bool = True
    belief_driven_progress: bool = True
    boundary_gated_advancement: bool = True
    relation_progress_evidence: bool = True
    scene_progress_evidence: bool = True
    legal_reentry_selection: bool = True
    control_equivalence_aggregation: bool = True
    state_inference: Literal["coupled_belief", "nearest_demo_guard"] = (
        "coupled_belief"
    )

    _PROFILES: ClassVar[dict[str, dict[str, object]]] = {
        "progress_only": {
            "dynamic_frame_roles": False,
            "relation_scene_boundary_guards": False,
            "active_relation_verification": False,
            "state_aware_recovery_reentry": False,
        },
        "progress_dynamic_roles": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": False,
            "active_relation_verification": False,
            "state_aware_recovery_reentry": False,
        },
        "full": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
        },
        # Equal-capability reviewer control: only the state inference rule is
        # replaced. Observation, action, boundary, repair, re-entry and budget
        # capabilities remain identical to Full.
        "simple_state_controller": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "state_inference": "nearest_demo_guard",
        },
        # E5: the task-state posterior is intentionally motion-only.  Fixed
        # candidate streams prevent relation evidence from leaking back into
        # action routing, and relation/scene boundary conditions are neutral.
        "motion_only": {
            "dynamic_frame_roles": False,
            "relation_scene_boundary_guards": False,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "complete_state_progress_evidence": False,
        },
        # E5: infer the complete task state for monitoring, while the action
        # reference follows the demonstrated state clock and boundaries do not
        # consume learned readiness conditions.
        "open_loop_progress": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "belief_driven_progress": False,
            "boundary_gated_advancement": False,
        },
        # M6 and E5 generic-retry retain active relation verification because
        # it is part of deciding whether an unresolved relation is a genuine
        # mismatch.  Only the post-alarm relation repair and legal re-entry
        # actuator are replaced by the shared generic Skill-Retry executor.
        "generic_retry": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": False,
        },
        # Appendix controls below each remove exactly one task-agnostic
        # component from Full.  They share the same learned models and do not
        # introduce task-specific branches or thresholds.
        "no_relation_evidence": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "relation_progress_evidence": False,
        },
        "no_scene_evidence": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "scene_progress_evidence": False,
        },
        "static_stream_roles": {
            "dynamic_frame_roles": False,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
        },
        "no_boundary_guards": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": False,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "boundary_gated_advancement": False,
        },
        "no_active_verification": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": False,
            "state_aware_recovery_reentry": True,
        },
        "retry_same_state": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "legal_reentry_selection": False,
        },
        "no_control_equivalence": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "active_relation_verification": True,
            "state_aware_recovery_reentry": True,
            "control_equivalence_aggregation": False,
        },
    }

    @classmethod
    def named(cls, name: str) -> "TSFFeatureProfile":
        try:
            fields = cls._PROFILES[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown TSF feature profile: {name}; "
                f"expected one of {sorted(cls._PROFILES)}"
            ) from exc
        return cls(name=name, **fields)

    def __post_init__(self) -> None:
        if self.state_inference not in {"coupled_belief", "nearest_demo_guard"}:
            raise ValueError(f"unknown state inference mode: {self.state_inference}")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls._PROFILES)

    def to_dict(self) -> dict[str, object]:
        # Preserve the published identity of the original three profiles.
        # New non-default controls are serialized only when an ablation
        # actually changes them.
        result = {
            "name": self.name,
            "dynamic_frame_roles": self.dynamic_frame_roles,
            "relation_scene_boundary_guards": self.relation_scene_boundary_guards,
            "active_relation_verification": self.active_relation_verification,
            "state_aware_recovery_reentry": self.state_aware_recovery_reentry,
        }
        for name in (
            "complete_state_progress_evidence",
            "belief_driven_progress",
            "boundary_gated_advancement",
            "relation_progress_evidence",
            "scene_progress_evidence",
            "legal_reentry_selection",
            "control_equivalence_aggregation",
        ):
            value = bool(getattr(self, name))
            if not value:
                result[name] = value
        if self.state_inference != "coupled_belief":
            result["state_inference"] = self.state_inference
        return result


__all__ = ["TSFFeatureProfile"]
