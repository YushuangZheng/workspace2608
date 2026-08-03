"""Simulator-independent schema for scripted bimanual handover supervision."""

from __future__ import annotations

from enum import IntEnum


class HandoverState(IntEnum):
    REST = 0
    LEFT_APPROACH = 1
    LEFT_GRASP = 2
    LEFT_LIFT = 3
    LEFT_TO_HANDOVER = 4
    RIGHT_APPROACH = 5
    RIGHT_GRASP = 6
    TRANSFER = 7
    LEFT_RELEASE = 8
    RIGHT_TO_TARGET = 9
    RIGHT_RELEASE = 10
    RETREAT = 11
    COMPLETE = 12


RELATION_LABELS = ("none", "left_only", "both", "right_only")
RELATION_SEQUENCE = ("none", "left_only", "both", "right_only", "none")


def handover_relation_label(state: HandoverState | int) -> str:
    """Return the state-aligned scripted object/arm relation."""

    state = HandoverState(state)
    if HandoverState.LEFT_GRASP <= state <= HandoverState.RIGHT_GRASP:
        return "left_only"
    if state == HandoverState.TRANSFER:
        return "both"
    if HandoverState.LEFT_RELEASE <= state <= HandoverState.RIGHT_TO_TARGET:
        return "right_only"
    return "none"
