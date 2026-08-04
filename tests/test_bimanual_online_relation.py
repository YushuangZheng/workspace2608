from __future__ import annotations

import numpy as np

from essay2608.policy.bimanual_relation import (
    BimanualOnlineRelationEstimator,
    BimanualRelationEstimatorConfig,
    BimanualRelationSample,
)
from essay2608.policy.relation import RelationEstimatorConfig


IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0])


def pose(x: float) -> np.ndarray:
    return np.concatenate((np.asarray([x, 0.0, 0.2]), IDENTITY))


def edge_config() -> RelationEstimatorConfig:
    return RelationEstimatorConfig(
        window_steps=5,
        occupied_opening_min_m=0.02,
        occupied_opening_nominal_m=0.04,
        occupied_opening_max_m=0.06,
        release_opening_m=0.07,
        maximum_gripper_speed_m_s=0.02,
        maximum_relative_linear_speed_m_s=0.02,
        maximum_relative_angular_speed_rad_s=0.05,
        maximum_relative_position_rms_std_m=0.001,
        maximum_relative_orientation_span_rad=0.01,
        maximum_object_distance_m=0.08,
        minimum_comotion_speed_m_s=0.02,
        minimum_velocity_correlation=0.80,
        lost_relative_linear_speed_m_s=0.10,
        lost_relative_angular_speed_rad_s=0.20,
        lost_relative_position_rms_std_m=0.01,
        lost_relative_orientation_span_rad=0.10,
        lost_object_distance_m=0.12,
        establish_steps=2,
        lost_steps=2,
        confidence_ema_alpha=0.5,
        calibration_source="bimanual_unit_test",
    )


def estimator() -> BimanualOnlineRelationEstimator:
    config = BimanualRelationEstimatorConfig(
        left=edge_config(),
        right=edge_config(),
        calibration_source="unit_test",
        calibration_seeds=(),
        dataset_sha256="unit-test",
    )
    return BimanualOnlineRelationEstimator(config)


def test_persisted_config_round_trip_and_distance_contract() -> None:
    config = estimator().config
    assert BimanualRelationEstimatorConfig.from_dict(config.as_dict()) == config

    values = edge_config().as_dict()
    values["lost_object_distance_m"] = None
    with np.testing.assert_raises_regex(ValueError, "必须同时设置"):
        RelationEstimatorConfig.from_dict(values)


def update(
    model: BimanualOnlineRelationEstimator,
    *,
    left_x: float,
    right_x: float,
    object_x: float,
    left_opening: float,
    right_opening: float,
) -> str:
    return model.update(
        BimanualRelationSample(
            left_ee_pose=pose(left_x),
            right_ee_pose=pose(right_x),
            object_pose=pose(object_x),
            left_finger_distance_m=left_opening,
            right_finger_distance_m=right_opening,
            left_finger_velocity_m_s=0.0,
            right_finger_velocity_m_s=0.0,
            control_dt_s=0.1,
        )
    ).label


def establish_left(model: BimanualOnlineRelationEstimator, start: int = 0) -> int:
    for index in range(start, start + 10):
        x = index * 0.01
        label = update(
            model,
            left_x=x,
            right_x=0.05,
            object_x=x,
            left_opening=0.04,
            right_opening=0.08,
        )
    assert label == "left_only"
    return start + 10


def establish_both(model: BimanualOnlineRelationEstimator) -> int:
    index = establish_left(model)
    for index in range(index, index + 10):
        x = index * 0.01
        label = update(
            model,
            left_x=x,
            right_x=x + 0.05,
            object_x=x,
            left_opening=0.04,
            right_opening=0.04,
        )
    assert label == "both"
    return index + 1


def test_receiver_miss_and_delay_do_not_create_premature_edge() -> None:
    model = estimator()
    index = establish_left(model)
    for index in range(index, index + 12):
        x = index * 0.01
        label = update(
            model,
            left_x=x,
            right_x=0.05,
            object_x=x,
            left_opening=0.04,
            right_opening=0.0,
        )
    assert label == "left_only"

    for index in range(index + 1, index + 11):
        x = index * 0.01
        label = update(
            model,
            left_x=x,
            right_x=x + 0.05,
            object_x=x,
            left_opening=0.04,
            right_opening=0.04,
        )
    assert label == "both"


def test_giver_early_release_exposes_no_holder_interval() -> None:
    model = estimator()
    index = establish_left(model)
    for index in range(index, index + 4):
        x = index * 0.01
        label = update(
            model,
            left_x=x,
            right_x=0.05,
            object_x=x,
            left_opening=0.08,
            right_opening=0.08,
        )
    assert label == "none"


def test_receiver_grasp_then_loss_returns_to_left_only() -> None:
    model = estimator()
    index = establish_both(model)
    paused_right = index * 0.01 + 0.05
    for index in range(index, index + 5):
        x = index * 0.04
        label = update(
            model,
            left_x=x,
            right_x=paused_right,
            object_x=x,
            left_opening=0.04,
            right_opening=0.04,
        )
    assert label == "left_only"


def test_prolonged_both_hold_does_not_time_out() -> None:
    model = estimator()
    index = establish_both(model)
    labels = []
    for index in range(index, index + 50):
        x = index * 0.005
        labels.append(
            update(
                model,
                left_x=x,
                right_x=x + 0.05,
                object_x=x,
                left_opening=0.04,
                right_opening=0.04,
            )
        )
    assert set(labels) == {"both"}


def test_paused_arm_loses_only_its_own_edge() -> None:
    model = estimator()
    index = establish_both(model)
    paused_left = index * 0.01
    for index in range(index, index + 5):
        x = index * 0.04
        label = update(
            model,
            left_x=paused_left,
            right_x=x + 0.05,
            object_x=x,
            left_opening=0.04,
            right_opening=0.04,
        )
    assert label == "right_only"
