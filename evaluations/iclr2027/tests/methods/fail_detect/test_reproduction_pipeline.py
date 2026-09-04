from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evaluations.iclr2027.methods.fail_detect.reproduction.calibrate_square_logpzo import (  # noqa: E402
    _decisions,
    _fit_upstream_split,
)
from evaluations.iclr2027.methods.fail_detect.reproduction.distributed_logpzo_train import (  # noqa: E402
    _adjust_xshape,
)


def test_official_success_filter_split_and_alarm_metrics() -> None:
    calibration = [
        {"seed": index, "success": 1, "logpzo": [0.0, 0.0, 0.0]}
        for index in range(10)
    ]
    band, split = _fit_upstream_split(
        calibration,
        alpha=0.1,
        num_train=3,
        num_cal=7,
    )

    assert split["mean_episodes"] == 3
    assert split["width_episodes"] == 7
    assert np.array_equal(band.upper, np.zeros(3))

    metrics, decisions = _decisions(
        [
            {"seed": 10, "success": 1, "logpzo": [-1.0, -1.0, -1.0]},
            {"seed": 11, "success": 0, "logpzo": [-1.0, 1.0, 1.0]},
        ],
        band,
    )

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["mean_detected_failure_delay_env_steps"] == 8.0
    assert decisions[1]["first_alarm_score_index"] == 1


def test_logpzo_feature_padding_matches_upstream_shape_rule() -> None:
    features = torch.arange(2 * 548, dtype=torch.float32).reshape(2, 548)
    adjusted = _adjust_xshape(features, input_dim=10)

    assert adjusted.shape == (2, 56, 10)
    assert torch.equal(adjusted.reshape(2, -1)[:, :548], features)
    assert torch.count_nonzero(adjusted.reshape(2, -1)[:, 548:]) == 0
