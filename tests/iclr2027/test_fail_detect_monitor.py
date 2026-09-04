from __future__ import annotations

import numpy as np
import pytest

from evaluations.iclr2027.monitors import (
    FailDetectMonitor,
    FailDetectMonitorConfig,
    FailureSupervisedMonitor,
    FailureSupervisedMonitorConfig,
    RuntimeMonitor,
    TimeVaryingConformalBand,
    prepare_logpzo_input,
)


def test_time_varying_upper_band_matches_public_algorithm() -> None:
    mean_scores = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.2, 2.1, 3.3],
            [0.8, 1.9, 2.7],
            [1.1, 2.2, 3.1],
        ]
    )
    width_scores = np.asarray(
        [
            [1.3, 2.4, 3.5],
            [0.9, 2.0, 3.2],
            [1.4, 2.3, 3.7],
        ]
    )
    alpha = 0.2

    band = TimeVaryingConformalBand.fit(
        mean_scores,
        width_scores,
        alpha=alpha,
    )

    mean = mean_scores.mean(axis=0)
    residual = np.abs(mean_scores - mean)
    maximum = residual.max(axis=1)
    rank = int(np.ceil((len(mean_scores) + 1) * (1.0 - alpha)))
    gamma = np.sort(maximum)[rank - 1]
    modulation = residual[maximum <= gamma].max(axis=0) + 1.0e-8
    nonconformity = ((width_scores - mean) / modulation).max(axis=1)
    width = np.quantile(nonconformity, 1.0 - alpha)

    assert np.allclose(band.mean, mean)
    assert np.allclose(band.modulation, modulation)
    assert band.band_width == pytest.approx(width)
    assert np.allclose(band.upper, mean + width * modulation)


def test_conformal_artifact_round_trip_is_strict() -> None:
    band = TimeVaryingConformalBand.fit(
        [[0.0, 1.0], [0.1, 1.2]],
        [[0.2, 1.4], [0.3, 1.1]],
        alpha=0.25,
        modulation_kind="constant",
    )

    restored = TimeVaryingConformalBand.from_dict(band.to_dict())

    assert restored.to_dict() == band.to_dict()
    with pytest.raises(ValueError, match="unknown"):
        TimeVaryingConformalBand.from_dict({**band.to_dict(), "extra": True})
    with pytest.raises(IndexError, match="calibrated horizon"):
        restored.threshold(restored.horizon)


def test_logpzo_input_shape_matches_official_adjust_xshape() -> None:
    square_feature = np.arange(274, dtype=np.float32).reshape(1, 274)

    adjusted = prepare_logpzo_input(square_feature, input_dim=10)

    assert adjusted.shape == (28, 10)
    assert np.array_equal(adjusted.reshape(-1)[:274], square_feature.reshape(-1))
    assert np.array_equal(adjusted.reshape(-1)[274:], np.zeros(6))

    short_window = np.arange(6, dtype=np.float32).reshape(2, 3)
    short_adjusted = prepare_logpzo_input(short_window, input_dim=4)
    assert short_adjusted.shape == (4, 4)
    assert np.array_equal(short_adjusted.reshape(-1)[:6], short_window.reshape(-1))
    assert np.array_equal(short_adjusted.reshape(-1)[6:], np.zeros(10))


def test_monitor_is_causal_and_applies_persistence() -> None:
    band = TimeVaryingConformalBand(
        alpha=0.05,
        mean=np.zeros(3),
        modulation=np.ones(3),
        band_width=1.0,
        upper=np.ones(3),
        mean_episode_count=2,
        width_episode_count=2,
    )

    def model(window: np.ndarray) -> float:
        return float(window[-1, 0])

    monitor = FailDetectMonitor(
        model,
        band,
        config=FailDetectMonitorConfig(observation_window=1, persistence=2),
    )

    assert isinstance(monitor, RuntimeMonitor)
    monitor.reset({"episode_id": "development-1"})
    monitor.observe(np.asarray([1.5]), None, None)
    assert monitor.alarm() is False
    assert monitor.score()["consecutive_exceedances"] == 1.0

    monitor.observe(np.asarray([1.2]), None, None)
    assert monitor.alarm() is True
    assert monitor.score()["margin"] == pytest.approx(0.2)
    assert monitor.score()["first_alarm_index"] == 1.0

    monitor.observe(np.asarray([0.5]), None, None)
    assert monitor.alarm() is False
    assert monitor.score()["consecutive_exceedances"] == 0.0
    assert monitor.score()["first_alarm_index"] == 1.0


def test_monitor_warmup_and_feature_shape_are_explicit() -> None:
    band = TimeVaryingConformalBand(
        alpha=0.05,
        mean=np.zeros(1),
        modulation=np.ones(1),
        band_width=1.0,
        upper=np.ones(1),
        mean_episode_count=2,
        width_episode_count=2,
    )
    monitor = FailDetectMonitor(
        lambda window: float(np.sum(window)),
        band,
        config=FailDetectMonitorConfig(observation_window=2),
    )
    monitor.reset({})

    monitor.observe(np.asarray([0.2, 0.3]), None, None)
    assert monitor.score() == {
        "ready": 0.0,
        "score_index": -1.0,
        "consecutive_exceedances": 0.0,
        "alarm": 0.0,
        "first_alarm_index": -1.0,
    }
    with pytest.raises(ValueError, match="shape changed"):
        monitor.observe(np.asarray([0.2]), None, None)


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

    band = TimeVaryingConformalBand(
        alpha=0.05,
        mean=np.zeros(3),
        modulation=np.ones(3),
        band_width=0.5,
        upper=np.full(3, 0.5),
        mean_episode_count=2,
        width_episode_count=2,
    )
    model = ProbabilitySequence()
    monitor = FailureSupervisedMonitor(
        model,
        band,
        config=FailureSupervisedMonitorConfig(persistence=2),
    )

    assert isinstance(monitor, RuntimeMonitor)
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
