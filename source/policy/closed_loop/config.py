"""Configuration composition for the phase-six closed-loop policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .inference.belief_updater import BeliefUpdaterConfig
from .control.boundary_runtime import BoundaryRuntimeConfig
from .control.execution_controller import ClosedLoopExecutionConfig
from .recovery.manager import ClosedLoopRecoveryConfig


@dataclass(frozen=True)
class ClosedLoopPolicyConfig:
    """The four already-validated stage configurations used by one task."""

    belief: BeliefUpdaterConfig
    execution: ClosedLoopExecutionConfig
    boundary: BoundaryRuntimeConfig
    recovery: ClosedLoopRecoveryConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClosedLoopPolicyConfig:
        known = {"belief", "execution", "boundary", "recovery"}
        unknown = set(value).difference(known)
        if unknown or set(value) != known:
            raise ValueError(f"阶段六策略配置分区不完整或包含未知项：{sorted(unknown)}")
        sections = {name: value[name] for name in known}
        if any(not isinstance(section, Mapping) for section in sections.values()):
            raise TypeError("阶段六策略配置分区必须为对象")
        return cls(
            belief=BeliefUpdaterConfig.from_mapping(sections["belief"]),
            execution=ClosedLoopExecutionConfig.from_mapping(sections["execution"]),
            boundary=BoundaryRuntimeConfig.from_mapping(sections["boundary"]),
            recovery=ClosedLoopRecoveryConfig.from_mapping(sections["recovery"]),
        )

    @classmethod
    def from_files(
        cls,
        *,
        belief: str | Path,
        execution: str | Path,
        boundary: str | Path,
        recovery: str | Path,
    ) -> ClosedLoopPolicyConfig:
        return cls(
            belief=BeliefUpdaterConfig.from_json(belief),
            execution=ClosedLoopExecutionConfig.from_json(execution),
            boundary=BoundaryRuntimeConfig.from_json(boundary),
            recovery=ClosedLoopRecoveryConfig.from_json(recovery),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief": self.belief.to_dict(),
            "execution": self.execution.to_dict(),
            "boundary": self.boundary.to_dict(),
            "recovery": self.recovery.to_dict(),
        }


__all__ = ["ClosedLoopPolicyConfig"]
