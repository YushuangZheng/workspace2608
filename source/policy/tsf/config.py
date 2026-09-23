"""Compose the inference, execution, boundary, and recovery settings for TSF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .inference.belief_updater import BeliefUpdaterConfig
from .control.boundary_runtime import BoundaryRuntimeConfig
from .control.execution_controller import TSFExecutionConfig
from .recovery.manager import TSFRecoveryConfig


@dataclass(frozen=True)
class TSFPolicyConfig:
    """All runtime configuration sections required by one TSF task."""

    belief: BeliefUpdaterConfig
    execution: TSFExecutionConfig
    boundary: BoundaryRuntimeConfig
    recovery: TSFRecoveryConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TSFPolicyConfig:
        known = {"belief", "execution", "boundary", "recovery"}
        unknown = set(value).difference(known)
        if unknown or set(value) != known:
            raise ValueError(f"TSF 策略配置分区不完整或包含未知项：{sorted(unknown)}")
        sections = {name: value[name] for name in known}
        if any(not isinstance(section, Mapping) for section in sections.values()):
            raise TypeError("TSF 策略配置分区必须为对象")
        return cls(
            belief=BeliefUpdaterConfig.from_mapping(sections["belief"]),
            execution=TSFExecutionConfig.from_mapping(sections["execution"]),
            boundary=BoundaryRuntimeConfig.from_mapping(sections["boundary"]),
            recovery=TSFRecoveryConfig.from_mapping(sections["recovery"]),
        )

    @classmethod
    def from_files(
        cls,
        *,
        belief: str | Path,
        execution: str | Path,
        boundary: str | Path,
        recovery: str | Path,
    ) -> TSFPolicyConfig:
        return cls(
            belief=BeliefUpdaterConfig.from_json(belief),
            execution=TSFExecutionConfig.from_json(execution),
            boundary=BoundaryRuntimeConfig.from_json(boundary),
            recovery=TSFRecoveryConfig.from_json(recovery),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief": self.belief.to_dict(),
            "execution": self.execution.to_dict(),
            "boundary": self.boundary.to_dict(),
            "recovery": self.recovery.to_dict(),
        }


__all__ = ["TSFPolicyConfig"]
