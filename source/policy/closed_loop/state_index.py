"""Unified discrete state identifiers for the closed-loop task model."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, order=True)
class StateId:
    """One task-progress state ``(skill_index, local_index)``."""

    skill_index: int
    local_index: int

    def __post_init__(self) -> None:
        for name in ("skill_index", "local_index"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} 必须为整数")
            if value < 0:
                raise ValueError(f"{name} 必须为非负整数")
            object.__setattr__(self, name, int(value))

    def as_tuple(self) -> tuple[int, int]:
        return self.skill_index, self.local_index

    @classmethod
    def from_tuple(cls, value: tuple[int, int] | list[int]) -> StateId:
        if len(value) != 2:
            raise ValueError("StateId 序列必须恰好包含两个分量")
        return cls(*value)


@dataclass(frozen=True)
class StateTopology:
    """Predecessor/successor graph metadata for one ``StateId``."""

    predecessors: tuple[StateId, ...]
    successors: tuple[StateId, ...]
    skill_terminal: bool
    has_cross_skill_successor: bool


def build_state_topology(policy: object) -> dict[StateId, StateTopology]:
    """Build the only task-state graph consumed by later closed-loop modules."""

    if not policy.fitted:
        raise RuntimeError("基础 DynaMAC 尚未拟合")
    predecessors: dict[StateId, list[StateId]] = {}
    successors: dict[StateId, list[StateId]] = {}
    all_states: list[StateId] = []
    for skill_index, skill in enumerate(policy.skills):
        for local_index in range(skill.duration):
            state = StateId(skill_index, local_index)
            all_states.append(state)
            predecessors[state] = []
            successors[state] = []

    for state in all_states:
        skill = policy.skills[state.skill_index]
        if state.local_index + 1 < skill.duration:
            target = StateId(state.skill_index, state.local_index + 1)
            successors[state].append(target)
            predecessors[target].append(state)
            continue
        if state.skill_index + 1 >= len(policy.skills):
            continue
        target = StateId(state.skill_index + 1, 0)
        successors[state].append(target)
        predecessors[target].append(state)

    result = {}
    for state in all_states:
        terminal = state.local_index == policy.skills[state.skill_index].duration - 1
        result[state] = StateTopology(
            predecessors=tuple(sorted(predecessors[state])),
            successors=tuple(sorted(successors[state])),
            skill_terminal=terminal,
            has_cross_skill_successor=terminal
            and any(
                item.skill_index != state.skill_index for item in successors[state]
            ),
        )
    return result


__all__ = ["StateId", "StateTopology", "build_state_topology"]
