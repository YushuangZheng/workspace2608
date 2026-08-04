"""Counterexample tests for bidirectional online relation estimation."""

from __future__ import annotations

import numpy as np

from essay2608.data.dataset import load_dataset
from essay2608.policy.relation import (
    OnlineRelationEstimator,
    RelationEstimatorConfig,
    RelationSample,
    RelationState,
    calibrate_relation_estimator,
)


IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0])


def pose(position: list[float]) -> np.ndarray:
    return np.concatenate((np.asarray(position, dtype=np.float64), IDENTITY))


def relation_test_config() -> RelationEstimatorConfig:
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
        minimum_comotion_speed_m_s=0.02,
        minimum_velocity_correlation=0.80,
        lost_relative_linear_speed_m_s=0.10,
        lost_relative_angular_speed_rad_s=0.20,
        lost_relative_position_rms_std_m=0.01,
        lost_relative_orientation_span_rad=0.10,
        establish_steps=2,
        lost_steps=2,
        confidence_ema_alpha=0.5,
        calibration_source="unit_test",
    )


def sample(
    ee_x: float,
    object_x: float,
    opening: float,
    velocity: float = 0.0,
    contact: bool | None = None,
) -> RelationSample:
    return RelationSample(
        ee_pose=pose([ee_x, 0.0, 0.20]),
        object_pose=pose([object_x, 0.0, 0.15]),
        gripper_opening_m=opening,
        gripper_velocity_m_s=velocity,
        control_dt_s=0.10,
        contact=contact,
    )


def establish_relation(estimator: OnlineRelationEstimator) -> list[RelationState]:
    states = []
    for index in range(10):
        estimate = estimator.update(sample(index * 0.01, index * 0.01, 0.04))
        states.append(estimate.state)
    return states


def test_gripper_closes_but_misses_object() -> None:
    estimator = OnlineRelationEstimator(relation_test_config())
    states = []
    for index in range(15):
        estimate = estimator.update(sample(index * 0.01, 0.0, 0.0))
        states.append(estimate.state)
    assert set(states) == {RelationState.DISCONNECTED}
    assert not estimator.connected
    assert estimator.last_estimate.features["score_components"]["occupied_gripper"] == 0.0


def test_successful_grasp_and_transport_establishes_relation() -> None:
    estimator = OnlineRelationEstimator(relation_test_config())
    states = establish_relation(estimator)
    assert RelationState.CANDIDATE_CONNECTED in states
    assert states[-1] == RelationState.CONNECTED
    assert estimator.connected
    assert estimator.last_estimate.connection_score > 0.9
    assert estimator.last_estimate.confidence > 0.5


def test_forced_drop_disconnects_while_gripper_remains_closed() -> None:
    estimator = OnlineRelationEstimator(relation_test_config())
    establish_relation(estimator)
    states = []
    for index in range(4):
        estimate = estimator.update(sample(0.10 + index * 0.01, 0.10 - index * 0.04, 0.04))
        states.append(estimate.state)
    assert RelationState.CANDIDATE_LOST in states
    assert states[-1] == RelationState.DISCONNECTED
    assert not estimator.connected


def test_external_object_motion_does_not_create_relation() -> None:
    estimator = OnlineRelationEstimator(relation_test_config())
    states = []
    for index in range(15):
        estimate = estimator.update(sample(0.0, index * 0.02, 0.08))
        states.append(estimate.state)
    assert set(states) == {RelationState.DISCONNECTED}
    assert not estimator.connected


def test_frozen_demo_calibration_is_training_only_and_well_ordered() -> None:
    demonstrations, _ = load_dataset("data/pick_place_static/v1", verify_hashes=True)
    config, calibration = calibrate_relation_estimator(demonstrations)
    assert calibration["num_demonstrations"] == 5
    assert calibration["rules"]["test_seeds_used"] is False
    assert config.occupied_opening_min_m < config.occupied_opening_nominal_m
    assert config.occupied_opening_nominal_m < config.occupied_opening_max_m
    assert config.occupied_opening_max_m < config.release_opening_m
    assert config.maximum_relative_linear_speed_m_s < config.lost_relative_linear_speed_m_s
    assert config.maximum_relative_angular_speed_rad_s < config.lost_relative_angular_speed_rad_s
