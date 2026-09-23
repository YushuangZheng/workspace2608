"""Event-grounded Native-6 E6 protocol.

This package is deliberately isolated from the legacy E6 runtime.  Importing
it has no side effects and cannot alter an already-running Main-10 evaluation.
"""

from .events import EventGroundedFaultState, FaultDecision
from .gate import build_development_gate
from .physical_clock import SimulatorStepClock
from .result import validate_native6_v3_result

__all__ = [
    "EventGroundedFaultState",
    "FaultDecision",
    "SimulatorStepClock",
    "build_development_gate",
    "validate_native6_v3_result",
]
