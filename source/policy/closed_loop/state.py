"""Runtime records shared by the core policy and environment adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .belief_updater import ClosedLoopBelief
from .bimanual_controller import BoundaryCycleResult
from .execution_controller import ExecutionCycleResult
from .recovery import ExecutionMode, RecoveryManagerResult

Array = np.ndarray


class PolicyLifecycle(str, Enum):
    RUNNING = "running"
    FAILED = "failed"


@dataclass(frozen=True)
class ArmCommand:
    """Environment-neutral absolute end-effector and gripper command."""

    pose: Array
    covariance: Array
    gripper: Array
    source: str
    # ``None`` preserves the executor's legacy pose-completion sequencing for
    # auxiliary or frozen commands.  TASK commands use an explicit Boolean:
    # task-state/boundary semantics, rather than Cartesian millimetre error,
    # decide whether the accompanying gripper transition may be committed.
    gripper_authorized: bool | None = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        gripper = np.asarray(self.gripper, dtype=np.float64)
        if pose.shape != (7,) or covariance.shape != (6, 6):
            raise ValueError("顶层策略动作必须使用 [7] 位姿和 [6,6] 协方差")
        if gripper.ndim == 0:
            gripper = gripper.reshape(1)
        if gripper.ndim != 1 or not len(gripper):
            raise ValueError("顶层策略夹爪动作必须为非空一维数组")
        if not (
            np.all(np.isfinite(pose))
            and np.all(np.isfinite(covariance))
            and np.all(np.isfinite(gripper))
        ):
            raise ValueError("顶层策略动作包含非有限值")
        if not self.source:
            raise ValueError("顶层策略动作必须标识来源")
        if self.gripper_authorized is not None and not isinstance(
            self.gripper_authorized, (bool, np.bool_)
        ):
            raise TypeError("夹爪授权必须为布尔值或 None")
        object.__setattr__(self, "pose", pose.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "gripper", gripper.copy())
        if self.gripper_authorized is not None:
            object.__setattr__(
                self, "gripper_authorized", bool(self.gripper_authorized)
            )


@dataclass(frozen=True)
class ArmCycleResult:
    arm_id: str
    mode_before: ExecutionMode
    mode_after: ExecutionMode
    belief: ClosedLoopBelief
    command: ArmCommand
    execution: ExecutionCycleResult | None = None
    recovery: RecoveryManagerResult | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class PolicyCycleResult:
    tick: int
    arms: dict[str, ArmCycleResult]
    boundary: BoundaryCycleResult | None
    lifecycle: PolicyLifecycle
    diagnostics: dict[str, object]

    @property
    def commands(self) -> dict[str, ArmCommand]:
        return {arm: result.command for arm, result in self.arms.items()}

    @property
    def failed(self) -> bool:
        return self.lifecycle == PolicyLifecycle.FAILED


__all__ = [
    "ArmCommand",
    "ArmCycleResult",
    "PolicyCycleResult",
    "PolicyLifecycle",
]
