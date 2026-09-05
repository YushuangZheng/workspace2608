"""Frozen generic Skill-Retry contract used by M2, M3, M4 and M6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SkillRetryDecision:
    requested: bool
    reference_state: Optional[Mapping[str, int]]
    reason: str
    remaining_budget: int


class SkillRetry:
    """Bounded reference reset without simulator or object-state mutation.

    The executor records only confirmed skill entries supplied by the shared
    policy interface.  It never receives a fault family, repair target, scene
    snapshot, or correct recovery direction.
    """

    def __init__(self, recovery_budget: int, maximum_retries: int = 1) -> None:
        if recovery_budget < 1 or maximum_retries < 1:
            raise ValueError("Skill-Retry budgets must be positive")
        self.initial_budget = int(recovery_budget)
        self.maximum_retries = int(maximum_retries)
        self.reset()

    def reset(self) -> None:
        self.remaining_budget = self.initial_budget
        self.retries = 0
        self.current_entry = None
        self.in_recovery = False

    def confirm_skill_entry(self, state: Mapping[str, int]) -> None:
        skill = int(state["skill"])
        progress = int(state["progress"])
        self.current_entry = {"skill": skill, "progress": progress}

    def request(self, alarm: bool) -> SkillRetryDecision:
        if not alarm:
            return SkillRetryDecision(False, None, "no_alarm", self.remaining_budget)
        if self.current_entry is None:
            return SkillRetryDecision(False, None, "no_confirmed_entry", self.remaining_budget)
        if self.retries >= self.maximum_retries or self.remaining_budget <= 0:
            return SkillRetryDecision(False, None, "budget_exhausted", self.remaining_budget)
        self.retries += 1
        self.in_recovery = True
        return SkillRetryDecision(
            True,
            dict(self.current_entry),
            "alarm_retry_current_skill",
            self.remaining_budget,
        )

    def consume_cycle(self) -> None:
        if self.in_recovery:
            self.remaining_budget = max(0, self.remaining_budget - 1)
            if self.remaining_budget == 0:
                self.in_recovery = False

    def finish(self) -> None:
        self.in_recovery = False


__all__ = ["SkillRetry", "SkillRetryDecision"]
