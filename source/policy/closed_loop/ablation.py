"""Explicit feature profiles for controlled stage-six ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ClosedLoopFeatureProfile:
    """Select already implemented mechanisms without changing learned models.

    These profiles are experiment controls, not alternative task mechanisms.
    Every closed-loop profile keeps the phase-two progress posterior and
    HOLD/REALIGN/ADVANCE controller.  Later profiles add mechanisms in the
    order used by the component comparison.
    """

    name: str
    dynamic_frame_roles: bool
    relation_scene_boundary_guards: bool
    auxiliary_verification_recovery: bool

    _PROFILES: ClassVar[dict[str, tuple[bool, bool, bool]]] = {
        "progress_only": (False, False, False),
        "progress_dynamic_roles": (True, False, False),
        "full": (True, True, True),
    }

    @classmethod
    def named(cls, name: str) -> "ClosedLoopFeatureProfile":
        try:
            dynamic, guards, recovery = cls._PROFILES[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown closed-loop feature profile: {name}; "
                f"expected one of {sorted(cls._PROFILES)}"
            ) from exc
        return cls(name, dynamic, guards, recovery)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls._PROFILES)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["ClosedLoopFeatureProfile"]
