"""Side-effect-minimal simulator-step telemetry for Native-6 v3.

RVT and RACER execute one policy output as a complete RLBench planning and
gripper transaction.  Consequently, a policy-cycle counter is not a physical
clock.  This helper wraps the concrete ``PyRep.step`` instance so it observes
both ``Scene.step`` calls and direct gripper ``pyrep.step`` calls.  It never
installs a planner callback and never changes the return value of a successful
simulator step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Callable, Optional


class SimulatorClockError(RuntimeError):
    """Raised at a high-level boundary when simulator telemetry is invalid."""


@dataclass(frozen=True)
class SimulatorStepRecord:
    completed_step: int
    time_before_s: float
    time_after_s: float
    observed_dt_s: float
    configured_dt_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimulatorStepClock:
    """Count completed ``PyRep.step`` calls without changing their semantics.

    Telemetry callbacks must be observational and must not call ``step``.  A
    callback error is retained and raised only when ``raise_if_failed`` is
    called by the outer executor, after the underlying physical transaction
    has returned.  This prevents logging failures from interrupting a path or
    gripper actuation halfway through a simulator step.
    """

    _MARKER = "_iclr2027_native6_v3_step_clock"

    def __init__(
        self,
        pyrep: Any,
        *,
        simulation_time: Callable[[], float],
        simulation_timestep: Callable[[], float],
        on_completed_step: Optional[Callable[[SimulatorStepRecord], None]] = None,
    ) -> None:
        self._pyrep = pyrep
        self._simulation_time = simulation_time
        self._simulation_timestep = simulation_timestep
        self._callback = on_completed_step
        self._original_step: Optional[Callable[..., Any]] = None
        self._completed_steps = 0
        self._records: list[SimulatorStepRecord] = []
        self._telemetry_errors: list[str] = []
        self._inside_callback = False

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def records(self) -> tuple[SimulatorStepRecord, ...]:
        return tuple(self._records)

    @property
    def installed(self) -> bool:
        return self._original_step is not None

    def install(self) -> "SimulatorStepClock":
        if self.installed:
            return self
        owner = getattr(self._pyrep, self._MARKER, None)
        if owner is not None and owner is not self:
            raise SimulatorClockError("a different Native-6 v3 clock is installed")
        original = getattr(self._pyrep, "step", None)
        if not callable(original):
            raise SimulatorClockError("PyRep instance does not expose step()")
        self._original_step = original

        @wraps(original)
        def observed_step(*args: Any, **kwargs: Any) -> Any:
            if self._inside_callback:
                raise SimulatorClockError("telemetry callback attempted a recursive step")
            before = float(self._simulation_time())
            configured_dt = float(self._simulation_timestep())
            result = original(*args, **kwargs)
            after = float(self._simulation_time())
            self._completed_steps += 1
            record = SimulatorStepRecord(
                completed_step=self._completed_steps,
                time_before_s=before,
                time_after_s=after,
                observed_dt_s=after - before,
                configured_dt_s=configured_dt,
            )
            self._records.append(record)
            if self._callback is not None:
                self._inside_callback = True
                try:
                    self._callback(record)
                except Exception as exc:  # defer until the high-level boundary
                    self._telemetry_errors.append(
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    self._inside_callback = False
            return result

        setattr(self._pyrep, "step", observed_step)
        setattr(self._pyrep, self._MARKER, self)
        return self

    def uninstall(self) -> None:
        if not self.installed:
            return
        if getattr(self._pyrep, self._MARKER, None) is not self:
            raise SimulatorClockError("Native-6 v3 clock ownership changed")
        assert self._original_step is not None
        setattr(self._pyrep, "step", self._original_step)
        delattr(self._pyrep, self._MARKER)
        self._original_step = None

    def raise_if_failed(self) -> None:
        if self._telemetry_errors:
            raise SimulatorClockError(
                "simulator telemetry failed: " + "; ".join(self._telemetry_errors)
            )
        for record in self._records:
            if record.configured_dt_s <= 0.0:
                raise SimulatorClockError("simulation timestep must be positive")
            if record.observed_dt_s < 0.0:
                raise SimulatorClockError("simulation time moved backwards")

    def records_since(self, completed_step: int) -> tuple[SimulatorStepRecord, ...]:
        if completed_step < 0 or completed_step > self._completed_steps:
            raise ValueError("completed_step is outside the recorded interval")
        return tuple(self._records[completed_step:])

    def __enter__(self) -> "SimulatorStepClock":
        return self.install()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.uninstall()


__all__ = ["SimulatorClockError", "SimulatorStepClock", "SimulatorStepRecord"]
