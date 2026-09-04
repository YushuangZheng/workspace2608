from __future__ import annotations

import numpy as np

from evaluations.iclr2027.methods.failure_supervised import (
    FailureSupervisedMonitor,
    FailureSupervisedMonitorConfig,
)


class ConstantThreshold:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def threshold(self, score_index: int) -> float:
        return self.values[score_index]


def test_supervised_monitor_resets_state_and_tracks_first_alarm() -> None:
    class ProbabilitySequence:
        def __init__(self) -> None:
            self.values = iter(())
            self.reset_count = 0

        def reset(self) -> None:
            self.values = iter((0.7, 0.8, 0.2))
            self.reset_count += 1

        def __call__(self, features: np.ndarray) -> float:
            assert features.shape == (2,)
            return next(self.values)

    model = ProbabilitySequence()
    monitor = FailureSupervisedMonitor(
        model,
        ConstantThreshold((0.5, 0.5, 0.5)),
        config=FailureSupervisedMonitorConfig(persistence=2),
    )

    assert all(
        callable(getattr(monitor, method)) for method in ("reset", "observe", "score", "alarm")
    )
    monitor.reset({"episode_id": "failure-train-development"})
    assert model.reset_count == 1
    monitor.observe(np.asarray([1.0, 2.0]), None, None)
    assert monitor.alarm() is False
    monitor.observe(np.asarray([2.0, 3.0]), None, None)
    assert monitor.alarm() is True
    assert monitor.score()["first_alarm_index"] == 1.0
    monitor.observe(np.asarray([3.0, 4.0]), None, None)
    assert monitor.alarm() is False
    assert monitor.score()["first_alarm_index"] == 1.0

    monitor.reset({})
    assert model.reset_count == 2
    assert monitor.score()["first_alarm_index"] == -1.0
