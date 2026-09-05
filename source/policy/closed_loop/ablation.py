"""Explicit, task-agnostic profiles for controlled method comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ClosedLoopFeatureProfile:
    """Select already implemented mechanisms without changing learned models.

    These profiles are experiment controls, not alternative task mechanisms.
    The profile changes only which already-defined evidence and control
    mechanisms are authoritative.  It never changes a learned task model or
    introduces task-specific behavior.
    """

    name: str
    dynamic_frame_roles: bool
    relation_scene_boundary_guards: bool
    auxiliary_verification_recovery: bool
    complete_state_progress_evidence: bool = True
    belief_driven_progress: bool = True
    boundary_gated_advancement: bool = True

    _PROFILES: ClassVar[dict[str, dict[str, bool]]] = {
        "progress_only": {
            "dynamic_frame_roles": False,
            "relation_scene_boundary_guards": False,
            "auxiliary_verification_recovery": False,
        },
        "progress_dynamic_roles": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": False,
            "auxiliary_verification_recovery": False,
        },
        "full": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "auxiliary_verification_recovery": True,
        },
        # E5: the task-state posterior is intentionally motion-only.  Fixed
        # candidate streams prevent relation evidence from leaking back into
        # action routing, and relation/scene boundary conditions are neutral.
        "motion_only": {
            "dynamic_frame_roles": False,
            "relation_scene_boundary_guards": False,
            "auxiliary_verification_recovery": True,
            "complete_state_progress_evidence": False,
        },
        # E5: infer the complete task state for monitoring, while the action
        # reference follows the demonstrated state clock and boundaries do not
        # consume learned readiness conditions.
        "open_loop_progress": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "auxiliary_verification_recovery": True,
            "belief_driven_progress": False,
            "boundary_gated_advancement": False,
        },
        # M6 and E5 generic-retry share the exact same complete-state alarm;
        # only the in-policy verification/repair/re-entry actuator is disabled.
        "generic_retry": {
            "dynamic_frame_roles": True,
            "relation_scene_boundary_guards": True,
            "auxiliary_verification_recovery": False,
        },
    }

    @classmethod
    def named(cls, name: str) -> "ClosedLoopFeatureProfile":
        try:
            fields = cls._PROFILES[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown closed-loop feature profile: {name}; "
                f"expected one of {sorted(cls._PROFILES)}"
            ) from exc
        return cls(name=name, **fields)

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
            "auxiliary_verification_recovery": (
                self.auxiliary_verification_recovery
            ),
        }
        for name in (
            "complete_state_progress_evidence",
            "belief_driven_progress",
            "boundary_gated_advancement",
        ):
            value = bool(getattr(self, name))
            if not value:
                result[name] = value
        return result


__all__ = ["ClosedLoopFeatureProfile"]
