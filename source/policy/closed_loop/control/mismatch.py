"""Consecutive evidence counters that emit intents without executing recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..inference.belief_updater import ClosedLoopBelief
from .execution_cursor import ClosedLoopCursor, ExecutionDecision
from .frame_roles import FrameRoleSnapshot, RelationRecoveryIntent
from ..inference.progress_filter import ProgressStatus
from ..model.state_index import StateId


class MismatchKind(str, Enum):
    NO_PLAUSIBLE_STATE = "no_plausible_state"
    RELATION_MISMATCH = "relation_mismatch"
    PERSISTENT_HOLD = "persistent_hold"
    STALLED_PROGRESS = "stalled_progress"


@dataclass(frozen=True)
class MismatchEvent:
    kind: MismatchKind
    tick: int
    state_id: StateId
    consecutive_cycles: int
    frame_ids: tuple[str, ...] = ()
    recovery_intents: tuple[RelationRecoveryIntent, ...] = ()

    def __post_init__(self) -> None:
        if self.tick < 0 or self.consecutive_cycles < 1:
            raise ValueError("失配事件 tick 和累计周期必须为正")


@dataclass(frozen=True)
class MismatchCounters:
    no_plausible_state: int = 0
    relation_mismatch: int = 0
    persistent_hold: int = 0
    stalled_progress: int = 0


@dataclass(frozen=True)
class MismatchUpdate:
    counters: MismatchCounters
    events: tuple[MismatchEvent, ...]


@dataclass(frozen=True)
class MismatchConfig:
    # RLBench runs at 20 Hz.  Require one second of continuous absence of any
    # plausible state before recovery so ordinary contact/servo transients
    # remain HOLD/closed-loop reservo rather than becoming false recoveries.
    no_plausible_cycles: int = 20
    # A demonstrated LINK/UNLINK changes the expected relation immediately,
    # while the action-conditioned posterior needs several real control
    # cycles to accumulate post-transition motion evidence.  Five cycles is
    # the normal-data confirmation horizon; it remains one shared setting for
    # every task and relation rather than an event-specific grace branch.
    relation_mismatch_cycles: int = 5
    persistent_hold_cycles: int = 20
    stalled_progress_cycles: int = 20

    def __post_init__(self) -> None:
        if any(
            value < 1
            for value in (
                self.no_plausible_cycles,
                self.relation_mismatch_cycles,
                self.persistent_hold_cycles,
                self.stalled_progress_cycles,
            )
        ):
            raise ValueError("失配累计周期必须为正整数")


class MismatchTracker:
    """Accumulate only persistent evidence; single-frame anomalies do not emit."""

    def __init__(self, config: MismatchConfig = MismatchConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._counters = MismatchCounters()
        self._previous_estimated_state: StateId | None = None
        self._emitted: set[MismatchKind] = set()

    @staticmethod
    def _next(current: int, active: bool) -> int:
        return current + 1 if active else 0

    def update(
        self,
        belief: ClosedLoopBelief,
        cursor: ClosedLoopCursor,
        decision: ExecutionDecision,
        roles: FrameRoleSnapshot,
        *,
        action_executed: bool = True,
    ) -> MismatchUpdate:
        if not isinstance(action_executed, bool):
            raise TypeError("action_executed 必须为布尔值")
        # A controller-level rejection means the requested task command never
        # produced the current observation: the environment committed a joint
        # hold instead.  That observation is valid feedback for HOLD/retry, but
        # it is not evidence that an *executed* task action led outside the
        # learned state model.  Counting it as NO_PLAUSIBLE_STATE would turn a
        # known low-level IK rejection into a false task-level recovery.
        no_plausible = bool(
            action_executed
            and belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE
        )
        relation_mismatch = bool(roles.recovery_intents)
        hold = decision == ExecutionDecision.HOLD
        stalled = bool(
            hold
            and self._previous_estimated_state is not None
            and cursor.estimated_state == self._previous_estimated_state
        )
        counters = MismatchCounters(
            no_plausible_state=self._next(
                self._counters.no_plausible_state, no_plausible
            ),
            relation_mismatch=self._next(
                self._counters.relation_mismatch, relation_mismatch
            ),
            persistent_hold=self._next(self._counters.persistent_hold, hold),
            stalled_progress=self._next(self._counters.stalled_progress, stalled),
        )

        active = {
            MismatchKind.NO_PLAUSIBLE_STATE: no_plausible,
            MismatchKind.RELATION_MISMATCH: relation_mismatch,
            MismatchKind.PERSISTENT_HOLD: hold,
            MismatchKind.STALLED_PROGRESS: stalled,
        }
        thresholds = {
            MismatchKind.NO_PLAUSIBLE_STATE: self.config.no_plausible_cycles,
            MismatchKind.RELATION_MISMATCH: self.config.relation_mismatch_cycles,
            MismatchKind.PERSISTENT_HOLD: self.config.persistent_hold_cycles,
            MismatchKind.STALLED_PROGRESS: self.config.stalled_progress_cycles,
        }
        values = {
            MismatchKind.NO_PLAUSIBLE_STATE: counters.no_plausible_state,
            MismatchKind.RELATION_MISMATCH: counters.relation_mismatch,
            MismatchKind.PERSISTENT_HOLD: counters.persistent_hold,
            MismatchKind.STALLED_PROGRESS: counters.stalled_progress,
        }
        for kind, is_active in active.items():
            if not is_active:
                self._emitted.discard(kind)

        events = []
        frames = tuple(sorted(intent.frame_id for intent in roles.recovery_intents))
        for kind in MismatchKind:
            if values[kind] < thresholds[kind] or kind in self._emitted:
                continue
            events.append(
                MismatchEvent(
                    kind=kind,
                    tick=belief.tick,
                    state_id=cursor.estimated_state,
                    consecutive_cycles=values[kind],
                    frame_ids=(
                        frames if kind == MismatchKind.RELATION_MISMATCH else ()
                    ),
                    recovery_intents=(
                        roles.recovery_intents
                        if kind == MismatchKind.RELATION_MISMATCH
                        else ()
                    ),
                )
            )
            self._emitted.add(kind)

        self._counters = counters
        self._previous_estimated_state = cursor.estimated_state
        return MismatchUpdate(counters=counters, events=tuple(events))


__all__ = [
    "MismatchConfig",
    "MismatchCounters",
    "MismatchEvent",
    "MismatchKind",
    "MismatchTracker",
    "MismatchUpdate",
]
