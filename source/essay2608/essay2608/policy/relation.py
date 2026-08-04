"""Bidirectional phase-independent online object/end-effector relation estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from essay2608.data.dataset import Demonstration
from essay2608.data.transforms import quaternion_distance_radians, relative_pose


class RelationState(str, Enum):
    """Hysteretic lifecycle of one candidate kinematic relation."""

    DISCONNECTED = "DISCONNECTED"
    CANDIDATE_CONNECTED = "CANDIDATE_CONNECTED"
    CONNECTED = "CONNECTED"
    CANDIDATE_LOST = "CANDIDATE_LOST"


@dataclass(frozen=True)
class RelationSample:
    """One sensor sample used by the estimator."""

    ee_pose: np.ndarray
    object_pose: np.ndarray
    gripper_opening_m: float
    gripper_velocity_m_s: float
    control_dt_s: float
    contact: bool | None = None


@dataclass(frozen=True)
class RelationEstimatorConfig:
    """Frozen-data calibration plus explicit state-machine hysteresis."""

    window_steps: int = 10
    occupied_opening_min_m: float = 0.020
    occupied_opening_nominal_m: float = 0.045
    occupied_opening_max_m: float = 0.063
    occupied_opening_plateau_min_m: float | None = None
    occupied_opening_plateau_max_m: float | None = None
    release_opening_m: float = 0.071
    maximum_gripper_speed_m_s: float = 0.025
    maximum_relative_linear_speed_m_s: float = 0.025
    maximum_relative_angular_speed_rad_s: float = 0.035
    maximum_relative_position_rms_std_m: float = 0.0005
    maximum_relative_orientation_span_rad: float = 0.005
    maximum_object_distance_m: float | None = None
    minimum_comotion_speed_m_s: float = 0.03
    minimum_velocity_correlation: float = 0.80
    lost_relative_linear_speed_m_s: float = 0.075
    lost_relative_angular_speed_rad_s: float = 0.105
    lost_relative_position_rms_std_m: float = 0.002
    lost_relative_orientation_span_rad: float = 0.03
    lost_object_distance_m: float | None = None
    kinematic_loss_requires_window_break: bool = False
    establish_confidence: float = 0.65
    cancel_candidate_confidence: float = 0.40
    lost_confidence: float = 0.70
    recover_confidence: float = 0.35
    establish_steps: int = 3
    lost_steps: int = 3
    confidence_ema_alpha: float = 0.30
    calibration_source: str = "defaults"

    def __post_init__(self) -> None:
        plateau_pair = (
            self.occupied_opening_plateau_min_m,
            self.occupied_opening_plateau_max_m,
        )
        if (plateau_pair[0] is None) != (plateau_pair[1] is None):
            raise ValueError("夹爪占用平台上下界必须同时设置或同时省略")
        if plateau_pair[0] is not None and plateau_pair[1] is not None:
            if not (
                self.occupied_opening_min_m
                < plateau_pair[0]
                <= self.occupied_opening_nominal_m
                <= plateau_pair[1]
                < self.occupied_opening_max_m
            ):
                raise ValueError("夹爪占用平台必须位于占用外边界内并包含标称开度")
        distance_pair = (
            self.maximum_object_distance_m,
            self.lost_object_distance_m,
        )
        if (distance_pair[0] is None) != (distance_pair[1] is None):
            raise ValueError("建立与解除的物体距离阈值必须同时设置或同时省略")
        if (
            distance_pair[0] is not None
            and distance_pair[1] is not None
            and distance_pair[1] <= distance_pair[0]
        ):
            raise ValueError("解除距离阈值必须大于建立距离阈值")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RelationEstimatorConfig:
        """Load a persisted estimator configuration with full validation."""

        return cls(**values)


@dataclass(frozen=True)
class RelationEstimate:
    """One state-machine output with continuous evidence and diagnostics."""

    state: RelationState
    connected: bool
    confidence: float
    connection_score: float
    loss_score: float
    features: dict[str, Any]
    transitioned: bool


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _inverse_ramp(value: float, good: float, bad: float) -> float:
    return 1.0 - _ramp(value, good, bad)


def _relative_series(demonstration: Demonstration) -> tuple[np.ndarray, ...]:
    relative = relative_pose(demonstration.ee_pose, demonstration.object_pose)
    dt = demonstration.control_dt
    relative_linear = np.concatenate(
        (
            [0.0],
            np.linalg.norm(np.diff(relative[:, :3], axis=0), axis=-1) / dt,
        )
    )
    relative_angular = np.concatenate(
        (
            [0.0],
            quaternion_distance_radians(relative[1:, 3:7], relative[:-1, 3:7]) / dt,
        )
    )
    opening = np.sum(demonstration.joint_pos[:, -2:], axis=-1)
    opening_velocity = np.sum(demonstration.joint_vel[:, -2:], axis=-1)
    return relative, relative_linear, relative_angular, opening, opening_velocity


def calibrate_relation_estimator(
    demonstrations: list[Demonstration],
    window_steps: int = 10,
) -> tuple[RelationEstimatorConfig, dict[str, Any]]:
    """Calibrate thresholds from all five frozen demonstrations, never test seeds."""

    if not demonstrations:
        raise ValueError("At least one demonstration is required for calibration.")
    connected_opening = []
    open_gripper = []
    gripper_speed = []
    relative_linear = []
    relative_angular = []
    position_rms = []
    orientation_span = []
    velocity_correlation = []
    comotion_speed = []

    for demonstration in demonstrations:
        relative, rel_linear, rel_angular, opening, opening_velocity = _relative_series(
            demonstration
        )
        connected = np.isin(demonstration.state, [4, 5, 6])
        exogenous_open = np.isin(demonstration.state, [0, 1, 2, 8, 9])
        connected_opening.extend(opening[connected])
        open_gripper.extend(opening[exogenous_open])
        for index in range(window_steps - 1, demonstration.steps):
            start = index - window_steps + 1
            if not np.all(connected[start : index + 1]):
                continue
            relative_window = relative[start : index + 1]
            ee_velocity = np.diff(demonstration.ee_pose[start : index + 1, :3], axis=0)
            object_velocity = np.diff(
                demonstration.object_pose[start : index + 1, :3],
                axis=0,
            )
            denominator = np.linalg.norm(ee_velocity) * np.linalg.norm(object_velocity)
            correlation = (
                float(np.sum(ee_velocity * object_velocity) / denominator)
                if denominator > 1.0e-12
                else 0.0
            )
            elapsed = max((window_steps - 1) * demonstration.control_dt, 1.0e-12)
            ee_speed = np.linalg.norm(
                demonstration.ee_pose[index, :3] - demonstration.ee_pose[start, :3]
            ) / elapsed
            object_speed = np.linalg.norm(
                demonstration.object_pose[index, :3]
                - demonstration.object_pose[start, :3]
            ) / elapsed
            gripper_speed.append(abs(float(opening_velocity[index])))
            relative_linear.append(float(rel_linear[index]))
            relative_angular.append(float(rel_angular[index]))
            position_rms.append(
                float(np.sqrt(np.mean(np.var(relative_window[:, :3], axis=0))))
            )
            orientation_span.append(
                float(
                    np.max(
                        quaternion_distance_radians(
                            relative_window[:, 3:7],
                            relative_window[0, 3:7],
                        )
                    )
                )
            )
            velocity_correlation.append(correlation)
            comotion_speed.append(float(min(ee_speed, object_speed)))

    arrays = {
        "connected_opening_m": np.asarray(connected_opening),
        "open_gripper_m": np.asarray(open_gripper),
        "gripper_speed_m_s": np.asarray(gripper_speed),
        "relative_linear_speed_m_s": np.asarray(relative_linear),
        "relative_angular_speed_rad_s": np.asarray(relative_angular),
        "relative_position_rms_std_m": np.asarray(position_rms),
        "relative_orientation_span_rad": np.asarray(orientation_span),
        "velocity_correlation": np.asarray(velocity_correlation),
        "comotion_speed_m_s": np.asarray(comotion_speed),
    }
    if any(not len(values) for values in arrays.values()):
        raise ValueError("Calibration demonstrations do not contain complete connected windows.")

    opening_low = float(np.quantile(arrays["connected_opening_m"], 0.01))
    opening_nominal = float(np.median(arrays["connected_opening_m"]))
    opening_high = float(np.quantile(arrays["connected_opening_m"], 0.99))
    open_low = float(np.quantile(arrays["open_gripper_m"], 0.01))
    occupied_min = max(0.005, opening_low * 0.5)
    occupied_max = (opening_high + open_low) * 0.5
    release_opening = (occupied_max + open_low) * 0.5
    maximum_gripper_speed = max(
        0.01,
        float(np.quantile(arrays["gripper_speed_m_s"], 0.99)) * 1.25,
    )
    maximum_relative_linear = max(
        0.005,
        float(np.quantile(arrays["relative_linear_speed_m_s"], 0.99)) * 1.25,
    )
    maximum_relative_angular = max(
        0.01,
        float(np.quantile(arrays["relative_angular_speed_rad_s"], 0.99)) * 1.25,
    )
    maximum_position_rms = max(
        0.0005,
        float(np.quantile(arrays["relative_position_rms_std_m"], 0.99)) * 1.25,
    )
    maximum_orientation_span = max(
        0.005,
        float(np.quantile(arrays["relative_orientation_span_rad"], 0.99)) * 1.25,
    )
    minimum_comotion = max(
        0.01,
        float(np.quantile(arrays["comotion_speed_m_s"], 0.01)) * 0.5,
    )
    minimum_correlation = max(
        0.50,
        float(np.quantile(arrays["velocity_correlation"], 0.01)) * 0.80,
    )
    config = RelationEstimatorConfig(
        window_steps=window_steps,
        occupied_opening_min_m=occupied_min,
        occupied_opening_nominal_m=opening_nominal,
        occupied_opening_max_m=occupied_max,
        release_opening_m=release_opening,
        maximum_gripper_speed_m_s=maximum_gripper_speed,
        maximum_relative_linear_speed_m_s=maximum_relative_linear,
        maximum_relative_angular_speed_rad_s=maximum_relative_angular,
        maximum_relative_position_rms_std_m=maximum_position_rms,
        maximum_relative_orientation_span_rad=maximum_orientation_span,
        minimum_comotion_speed_m_s=minimum_comotion,
        minimum_velocity_correlation=minimum_correlation,
        lost_relative_linear_speed_m_s=max(0.075, maximum_relative_linear * 3.0),
        lost_relative_angular_speed_rad_s=max(0.105, maximum_relative_angular * 3.0),
        lost_relative_position_rms_std_m=max(0.002, maximum_position_rms * 4.0),
        lost_relative_orientation_span_rad=max(0.03, maximum_orientation_span * 6.0),
        calibration_source="frozen_demonstrations_states_4_to_6",
    )
    calibration = {
        "num_demonstrations": len(demonstrations),
        "connected_window_count": len(arrays["relative_linear_speed_m_s"]),
        "rules": {
            "connected_calibration_states": [4, 5, 6],
            "open_gripper_calibration_states": [0, 1, 2, 8, 9],
            "connection_quantile": 0.99,
            "margin_multiplier": 1.25,
            "minimum_motion_quantile": 0.01,
            "test_seeds_used": False,
        },
        "quantiles": {
            name: {
                "q01": float(np.quantile(values, 0.01)),
                "median": float(np.median(values)),
                "q99": float(np.quantile(values, 0.99)),
            }
            for name, values in arrays.items()
        },
        "config": config.as_dict(),
    }
    return config, calibration


class OnlineRelationEstimator:
    """Estimate establishment and loss without consulting task phase labels."""

    def __init__(self, config: RelationEstimatorConfig = RelationEstimatorConfig()) -> None:
        self.config = config
        self.samples: deque[RelationSample] = deque(maxlen=config.window_steps)
        self.state = RelationState.DISCONNECTED
        self.confidence = 0.0
        self._state_steps = 0
        self.last_estimate: RelationEstimate | None = None

    @property
    def connected(self) -> bool:
        return self.state in {RelationState.CONNECTED, RelationState.CANDIDATE_LOST}

    def reset(self) -> None:
        self.samples.clear()
        self.state = RelationState.DISCONNECTED
        self.confidence = 0.0
        self._state_steps = 0
        self.last_estimate = None

    def _window_features(self) -> dict[str, Any]:
        sample = self.samples[-1]
        relative = relative_pose(sample.ee_pose, sample.object_pose)
        features: dict[str, Any] = {
            "gripper_opening_m": float(sample.gripper_opening_m),
            "gripper_velocity_m_s": float(sample.gripper_velocity_m_s),
            "relative_position_m": relative[:3].astype(float).tolist(),
            "relative_orientation_wxyz": relative[3:7].astype(float).tolist(),
            "contact": sample.contact,
            "window_ready": len(self.samples) == self.config.window_steps,
        }
        if len(self.samples) < 2:
            features.update(
                {
                    "relative_linear_speed_m_s": 0.0,
                    "relative_angular_speed_rad_s": 0.0,
                    "relative_position_rms_std_m": 0.0,
                    "relative_orientation_span_rad": 0.0,
                    "velocity_correlation": 0.0,
                    "comotion_speed_m_s": 0.0,
                }
            )
            return features

        samples = list(self.samples)
        relatives = np.stack(
            [relative_pose(item.ee_pose, item.object_pose) for item in samples]
        )
        dt = max(float(sample.control_dt_s), 1.0e-12)
        relative_linear_speed = float(
            np.linalg.norm(relatives[-1, :3] - relatives[-2, :3]) / dt
        )
        relative_angular_speed = float(
            quaternion_distance_radians(relatives[-1, 3:7], relatives[-2, 3:7]) / dt
        )
        ee_positions = np.stack([item.ee_pose[:3] for item in samples])
        object_positions = np.stack([item.object_pose[:3] for item in samples])
        ee_velocity = np.diff(ee_positions, axis=0)
        object_velocity = np.diff(object_positions, axis=0)
        denominator = np.linalg.norm(ee_velocity) * np.linalg.norm(object_velocity)
        correlation = (
            float(np.sum(ee_velocity * object_velocity) / denominator)
            if denominator > 1.0e-12
            else 0.0
        )
        elapsed = max((len(samples) - 1) * dt, 1.0e-12)
        comotion_speed = float(
            min(
                np.linalg.norm(ee_positions[-1] - ee_positions[0]),
                np.linalg.norm(object_positions[-1] - object_positions[0]),
            )
            / elapsed
        )
        features.update(
            {
                "relative_linear_speed_m_s": relative_linear_speed,
                "relative_angular_speed_rad_s": relative_angular_speed,
                "relative_position_rms_std_m": float(
                    np.sqrt(np.mean(np.var(relatives[:, :3], axis=0)))
                ),
                "relative_orientation_span_rad": float(
                    np.max(
                        quaternion_distance_radians(
                            relatives[:, 3:7],
                            relatives[0, 3:7],
                        )
                    )
                ),
                "velocity_correlation": correlation,
                "comotion_speed_m_s": comotion_speed,
            }
        )
        return features

    def _scores(self, features: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
        config = self.config
        opening = features["gripper_opening_m"]
        plateau_min = (
            config.occupied_opening_nominal_m
            if config.occupied_opening_plateau_min_m is None
            else config.occupied_opening_plateau_min_m
        )
        plateau_max = (
            config.occupied_opening_nominal_m
            if config.occupied_opening_plateau_max_m is None
            else config.occupied_opening_plateau_max_m
        )
        lower_occupancy = _ramp(
            opening,
            config.occupied_opening_min_m,
            plateau_min,
        )
        upper_occupancy = _inverse_ramp(
            opening,
            plateau_max,
            config.occupied_opening_max_m,
        )
        occupancy = min(lower_occupancy, upper_occupancy)
        if features["contact"] is True:
            occupancy = max(occupancy, 1.0)
        components = {
            "occupied_gripper": occupancy,
            "settled_gripper": _inverse_ramp(
                abs(features["gripper_velocity_m_s"]),
                0.0,
                config.maximum_gripper_speed_m_s,
            ),
            "relative_linear_rigidity": _inverse_ramp(
                features["relative_linear_speed_m_s"],
                0.0,
                config.maximum_relative_linear_speed_m_s,
            ),
            "relative_angular_rigidity": _inverse_ramp(
                features["relative_angular_speed_rad_s"],
                0.0,
                config.maximum_relative_angular_speed_rad_s,
            ),
            "relative_position_stability": _inverse_ramp(
                features["relative_position_rms_std_m"],
                0.0,
                config.maximum_relative_position_rms_std_m,
            ),
            "relative_orientation_stability": _inverse_ramp(
                features["relative_orientation_span_rad"],
                0.0,
                config.maximum_relative_orientation_span_rad,
            ),
            "velocity_correlation": _ramp(
                features["velocity_correlation"],
                config.minimum_velocity_correlation,
                1.0,
            ),
            "comotion": _ramp(
                features["comotion_speed_m_s"],
                0.0,
                config.minimum_comotion_speed_m_s,
            ),
        }
        if config.maximum_object_distance_m is not None:
            distance = float(np.linalg.norm(features["relative_position_m"]))
            components["proximity"] = _inverse_ramp(
                distance,
                config.maximum_object_distance_m * 0.8,
                config.maximum_object_distance_m,
            )
        if not features["window_ready"]:
            connection_score = 0.0
        else:
            connection_score = components["occupied_gripper"] * float(
                np.mean([value for name, value in components.items() if name != "occupied_gripper"])
            )

        loss_components = {
            "gripper_open": _ramp(
                opening,
                config.occupied_opening_max_m,
                config.release_opening_m,
            ),
            "gripper_empty_closed": _inverse_ramp(
                opening,
                config.occupied_opening_min_m * 0.25,
                config.occupied_opening_min_m,
            ),
        }
        instantaneous_break = max(
            _ramp(
                features["relative_linear_speed_m_s"],
                config.maximum_relative_linear_speed_m_s,
                config.lost_relative_linear_speed_m_s,
            ),
            _ramp(
                features["relative_angular_speed_rad_s"],
                config.maximum_relative_angular_speed_rad_s,
                config.lost_relative_angular_speed_rad_s,
            ),
        )
        window_break = max(
            _ramp(
                features["relative_position_rms_std_m"],
                config.maximum_relative_position_rms_std_m,
                config.lost_relative_position_rms_std_m,
            ),
            _ramp(
                features["relative_orientation_span_rad"],
                config.maximum_relative_orientation_span_rad,
                config.lost_relative_orientation_span_rad,
            ),
        )
        if config.kinematic_loss_requires_window_break:
            loss_components["kinematic_break"] = min(
                instantaneous_break,
                window_break,
            )
        else:
            loss_components["instantaneous_kinematic_break"] = instantaneous_break
            loss_components["window_kinematic_break"] = window_break
        if config.lost_object_distance_m is not None:
            distance = float(np.linalg.norm(features["relative_position_m"]))
            loss_components["distance_break"] = _ramp(
                distance,
                config.maximum_object_distance_m,
                config.lost_object_distance_m,
            )
        loss_score = float(max(loss_components.values()))
        components.update({f"loss_{name}": value for name, value in loss_components.items()})
        return float(connection_score), loss_score, components

    def update(self, sample: RelationSample) -> RelationEstimate:
        """Consume one actual sensor sample and update the hysteretic state."""

        self.samples.append(sample)
        features = self._window_features()
        connection_score, loss_score, score_components = self._scores(features)
        previous = self.state

        if self.state == RelationState.DISCONNECTED:
            if connection_score >= self.config.establish_confidence:
                self.state = RelationState.CANDIDATE_CONNECTED
                self._state_steps = 1
        elif self.state == RelationState.CANDIDATE_CONNECTED:
            if connection_score >= self.config.establish_confidence:
                self._state_steps += 1
                if self._state_steps >= self.config.establish_steps:
                    self.state = RelationState.CONNECTED
                    self._state_steps = 0
            elif connection_score < self.config.cancel_candidate_confidence:
                self.state = RelationState.DISCONNECTED
                self._state_steps = 0
        elif self.state == RelationState.CONNECTED:
            if loss_score >= self.config.lost_confidence:
                self.state = RelationState.CANDIDATE_LOST
                self._state_steps = 1
        elif self.state == RelationState.CANDIDATE_LOST:
            if loss_score >= self.config.lost_confidence:
                self._state_steps += 1
                if self._state_steps >= self.config.lost_steps:
                    self.state = RelationState.DISCONNECTED
                    self._state_steps = 0
                    self.samples.clear()
            elif loss_score <= self.config.recover_confidence:
                self.state = RelationState.CONNECTED
                self._state_steps = 0

        confidence_target = (
            1.0 - loss_score if self.connected else connection_score
        )
        alpha = self.config.confidence_ema_alpha
        self.confidence = float(
            np.clip((1.0 - alpha) * self.confidence + alpha * confidence_target, 0.0, 1.0)
        )
        features = {
            **features,
            "score_components": score_components,
            "state_steps": self._state_steps,
        }
        estimate = RelationEstimate(
            state=self.state,
            connected=self.connected,
            confidence=self.confidence,
            connection_score=connection_score,
            loss_score=loss_score,
            features=features,
            transitioned=self.state != previous,
        )
        self.last_estimate = estimate
        return estimate


def replay_relation_estimator(
    demonstration: Demonstration,
    config: RelationEstimatorConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Replay actual recorded sensors without exposing phase labels to the estimator."""

    estimator = OnlineRelationEstimator(config)
    states = []
    connected = []
    confidence = []
    connection_score = []
    loss_score = []
    transitions = []
    opening = np.sum(demonstration.joint_pos[:, -2:], axis=-1)
    opening_velocity = np.sum(demonstration.joint_vel[:, -2:], axis=-1)
    for index in range(demonstration.steps):
        estimate = estimator.update(
            RelationSample(
                ee_pose=demonstration.ee_pose[index],
                object_pose=demonstration.object_pose[index],
                gripper_opening_m=float(opening[index]),
                gripper_velocity_m_s=float(opening_velocity[index]),
                control_dt_s=demonstration.control_dt,
            )
        )
        states.append(estimate.state.value)
        connected.append(estimate.connected)
        confidence.append(estimate.confidence)
        connection_score.append(estimate.connection_score)
        loss_score.append(estimate.loss_score)
        if estimate.transitioned:
            transitions.append(
                {
                    "index": index,
                    "time_s": float(demonstration.time[index]),
                    "state": estimate.state.value,
                }
            )

    connected_array = np.asarray(connected, dtype=bool)
    expected = np.isin(demonstration.state, [4, 5, 6])
    expected_onset = int(np.flatnonzero(expected)[0])
    observed_indices = np.flatnonzero(connected_array)
    observed_onset = int(observed_indices[0]) if len(observed_indices) else None
    expected_release = int(np.flatnonzero(expected[:-1] & ~expected[1:])[0] + 1)
    release_indices = np.flatnonzero(~connected_array[expected_release:])
    observed_release = (
        expected_release + int(release_indices[0])
        if observed_onset is not None and len(release_indices)
        else None
    )
    summary = {
        "demonstration": demonstration.path.name,
        "steps": demonstration.steps,
        "expected_onset_step": expected_onset,
        "observed_onset_step": observed_onset,
        "onset_delay_steps": (
            observed_onset - expected_onset if observed_onset is not None else None
        ),
        "onset_delay_s": (
            (observed_onset - expected_onset) * demonstration.control_dt
            if observed_onset is not None
            else None
        ),
        "expected_release_step": expected_release,
        "observed_release_step": observed_release,
        "release_delay_steps": (
            observed_release - expected_release if observed_release is not None else None
        ),
        "release_delay_s": (
            (observed_release - expected_release) * demonstration.control_dt
            if observed_release is not None
            else None
        ),
        "false_positive_fraction": float(np.mean(connected_array & ~expected)),
        "false_negative_fraction": float(np.mean(~connected_array & expected)),
        "maximum_confidence": float(np.max(confidence)),
        "state_transitions": transitions,
    }
    arrays = {
        "time": demonstration.time.astype(np.float64),
        "manual_state": demonstration.state.astype(np.int64),
        "relation_state": np.asarray(states, dtype="U32"),
        "connected": connected_array,
        "confidence": np.asarray(confidence, dtype=np.float64),
        "connection_score": np.asarray(connection_score, dtype=np.float64),
        "loss_score": np.asarray(loss_score, dtype=np.float64),
        "gripper_opening_m": opening.astype(np.float64),
        "gripper_velocity_m_s": opening_velocity.astype(np.float64),
    }
    return summary, arrays
