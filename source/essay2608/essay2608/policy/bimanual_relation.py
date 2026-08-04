"""Two-edge online relation lifecycle estimation for physical handover."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from essay2608.data.transforms import quaternion_distance_radians, relative_pose
from essay2608.policy.relation import (
    OnlineRelationEstimator,
    RelationEstimate,
    RelationEstimatorConfig,
    RelationSample,
)


RELATION_LABELS = ("none", "left_only", "both", "right_only")


@dataclass(frozen=True)
class BimanualRelationEstimatorConfig:
    """Independent left/right estimator settings and calibration identity."""

    left: RelationEstimatorConfig
    right: RelationEstimatorConfig
    calibration_source: str
    calibration_seeds: tuple[int, ...]
    dataset_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "calibration_source": self.calibration_source,
            "calibration_seeds": list(self.calibration_seeds),
            "dataset_sha256": self.dataset_sha256,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BimanualRelationEstimatorConfig:
        """Load the exact two-edge calibration persisted by the analyzer."""

        return cls(
            left=RelationEstimatorConfig.from_dict(values["left"]),
            right=RelationEstimatorConfig.from_dict(values["right"]),
            calibration_source=str(values["calibration_source"]),
            calibration_seeds=tuple(int(seed) for seed in values["calibration_seeds"]),
            dataset_sha256=str(values["dataset_sha256"]),
        )


@dataclass(frozen=True)
class BimanualRelationSample:
    """One phase-free sample for both candidate arm-object edges."""

    left_ee_pose: np.ndarray
    right_ee_pose: np.ndarray
    object_pose: np.ndarray
    left_finger_distance_m: float
    right_finger_distance_m: float
    left_finger_velocity_m_s: float
    right_finger_velocity_m_s: float
    control_dt_s: float


@dataclass(frozen=True)
class BimanualRelationEstimate:
    """Four-value lifecycle inferred from two independent estimators."""

    label: str
    left: RelationEstimate
    right: RelationEstimate
    transitioned: bool

    @property
    def left_connected(self) -> bool:
        return self.left.connected

    @property
    def right_connected(self) -> bool:
        return self.right.connected


class BimanualOnlineRelationEstimator:
    """Run independent online estimators and compose their four-value label."""

    def __init__(self, config: BimanualRelationEstimatorConfig) -> None:
        self.config = config
        self.left = OnlineRelationEstimator(config.left)
        self.right = OnlineRelationEstimator(config.right)
        self.label = "none"
        self.last_estimate: BimanualRelationEstimate | None = None

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()
        self.label = "none"
        self.last_estimate = None

    def update(self, sample: BimanualRelationSample) -> BimanualRelationEstimate:
        """Update both edges without contact truth, task phase, or future state."""

        left = self.left.update(
            RelationSample(
                ee_pose=sample.left_ee_pose,
                object_pose=sample.object_pose,
                gripper_opening_m=sample.left_finger_distance_m,
                gripper_velocity_m_s=sample.left_finger_velocity_m_s,
                control_dt_s=sample.control_dt_s,
            )
        )
        right = self.right.update(
            RelationSample(
                ee_pose=sample.right_ee_pose,
                object_pose=sample.object_pose,
                gripper_opening_m=sample.right_finger_distance_m,
                gripper_velocity_m_s=sample.right_finger_velocity_m_s,
                control_dt_s=sample.control_dt_s,
            )
        )
        if left.connected and right.connected:
            label = "both"
        elif left.connected:
            label = "left_only"
        elif right.connected:
            label = "right_only"
        else:
            label = "none"
        estimate = BimanualRelationEstimate(
            label=label,
            left=left,
            right=right,
            transitioned=label != self.label,
        )
        self.label = label
        self.last_estimate = estimate
        return estimate


def _finger_distance(archive: Any, side: str) -> np.ndarray:
    positions = archive[f"{side}_finger_position"]
    distance = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=-1)
    distance = distance.reshape(len(distance), -1)
    if distance.shape[1] != 1:
        raise ValueError(f"{side} 指体位置不能化为唯一指间距：{positions.shape}")
    return distance[:, 0]


def _calibrate_arm(
    paths: list[Path],
    side: str,
    *,
    window_steps: int,
) -> tuple[RelationEstimatorConfig, dict[str, Any]]:
    connected_opening: list[float] = []
    disconnected_opening: list[float] = []
    distances: list[float] = []
    values: dict[str, list[float]] = {
        "gripper_speed_m_s": [],
        "relative_linear_speed_m_s": [],
        "relative_angular_speed_rad_s": [],
        "relative_position_rms_std_m": [],
        "relative_orientation_span_rad": [],
        "velocity_correlation": [],
        "comotion_speed_m_s": [],
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            ee_pose = archive[f"{side}_ee_pose"].astype(np.float64)
            object_pose = archive["object_pose"].astype(np.float64)
            connected = archive[f"{side}_connected"].astype(bool)
            opening = _finger_distance(archive, side).astype(np.float64)
            dt = float(np.asarray(archive["control_dt"]).item())
        opening_velocity = np.concatenate(
            ([0.0], np.diff(opening) / max(dt, 1e-12))
        )
        connected_opening.extend(opening[connected].tolist())
        disconnected_opening.extend(opening[~connected].tolist())
        relative = relative_pose(ee_pose, object_pose)
        distances.extend(np.linalg.norm(relative[connected, :3], axis=-1).tolist())
        for index in range(window_steps - 1, len(connected)):
            start = index - window_steps + 1
            if not np.all(connected[start : index + 1]):
                continue
            relative_window = relative[start : index + 1]
            ee_velocity = np.diff(ee_pose[start : index + 1, :3], axis=0)
            object_velocity = np.diff(object_pose[start : index + 1, :3], axis=0)
            denominator = np.linalg.norm(ee_velocity) * np.linalg.norm(object_velocity)
            correlation = (
                float(np.sum(ee_velocity * object_velocity) / denominator)
                if denominator > 1e-12
                else 0.0
            )
            elapsed = max((window_steps - 1) * dt, 1e-12)
            values["gripper_speed_m_s"].append(abs(float(opening_velocity[index])))
            values["relative_linear_speed_m_s"].append(
                float(np.linalg.norm(relative[index, :3] - relative[index - 1, :3]) / dt)
            )
            values["relative_angular_speed_rad_s"].append(
                float(
                    quaternion_distance_radians(
                        relative[index, 3:7], relative[index - 1, 3:7]
                    )
                    / dt
                )
            )
            values["relative_position_rms_std_m"].append(
                float(np.sqrt(np.mean(np.var(relative_window[:, :3], axis=0))))
            )
            values["relative_orientation_span_rad"].append(
                float(
                    np.max(
                        quaternion_distance_radians(
                            relative_window[:, 3:7], relative_window[0, 3:7]
                        )
                    )
                )
            )
            values["velocity_correlation"].append(correlation)
            values["comotion_speed_m_s"].append(
                float(
                    min(
                        np.linalg.norm(ee_pose[index, :3] - ee_pose[start, :3]),
                        np.linalg.norm(object_pose[index, :3] - object_pose[start, :3]),
                    )
                    / elapsed
                )
            )

    connected_array = np.asarray(connected_opening)
    disconnected_array = np.asarray(disconnected_opening)
    arrays = {name: np.asarray(items) for name, items in values.items()}
    if not len(connected_array) or any(not len(items) for items in arrays.values()):
        raise ValueError(f"{side} 物理标定窗口为空")
    occupied_q01, occupied_median, occupied_q99 = np.quantile(
        connected_array, [0.01, 0.50, 0.99]
    )
    lower_negative = disconnected_array[disconnected_array < occupied_q01]
    occupied_min = (
        float((np.quantile(lower_negative, 0.99) + occupied_q01) * 0.5)
        if len(lower_negative) >= 10
        else float(np.min(connected_array) * 0.90)
    )
    open_negative = disconnected_array[disconnected_array > occupied_q99]
    if len(open_negative) < 10:
        raise ValueError(f"{side} 标定缺少张开夹爪负样本")
    open_q01 = float(np.quantile(open_negative, 0.01))
    occupied_max = float((occupied_q99 + open_q01) * 0.5)
    release_opening = float((occupied_max + open_q01) * 0.5)

    def upper(name: str, floor: float) -> float:
        return max(floor, float(np.quantile(arrays[name], 0.95)) * 1.5)

    maximum_relative_linear = upper("relative_linear_speed_m_s", 0.01)
    maximum_relative_angular = upper("relative_angular_speed_rad_s", 0.03)
    maximum_position_rms = upper("relative_position_rms_std_m", 0.001)
    maximum_orientation_span = upper("relative_orientation_span_rad", 0.01)
    maximum_distance = float(np.quantile(distances, 0.99) * 1.15)
    config = RelationEstimatorConfig(
        window_steps=window_steps,
        occupied_opening_min_m=occupied_min,
        occupied_opening_nominal_m=float(occupied_median),
        occupied_opening_max_m=occupied_max,
        occupied_opening_plateau_min_m=float(occupied_q01),
        occupied_opening_plateau_max_m=float(occupied_q99),
        release_opening_m=release_opening,
        maximum_gripper_speed_m_s=upper("gripper_speed_m_s", 0.02),
        maximum_relative_linear_speed_m_s=maximum_relative_linear,
        maximum_relative_angular_speed_rad_s=maximum_relative_angular,
        maximum_relative_position_rms_std_m=maximum_position_rms,
        maximum_relative_orientation_span_rad=maximum_orientation_span,
        maximum_object_distance_m=maximum_distance,
        minimum_comotion_speed_m_s=max(
            0.005, float(np.quantile(arrays["comotion_speed_m_s"], 0.05)) * 0.5
        ),
        minimum_velocity_correlation=max(
            0.50, float(np.quantile(arrays["velocity_correlation"], 0.05)) * 0.8
        ),
        lost_relative_linear_speed_m_s=max(0.10, maximum_relative_linear * 3.0),
        lost_relative_angular_speed_rad_s=max(0.20, maximum_relative_angular * 3.0),
        lost_relative_position_rms_std_m=max(0.01, maximum_position_rms * 4.0),
        lost_relative_orientation_span_rad=max(0.10, maximum_orientation_span * 6.0),
        lost_object_distance_m=maximum_distance * 1.5,
        kinematic_loss_requires_window_break=True,
        establish_steps=3,
        lost_steps=3,
        calibration_source=f"handover_physical_v1_{side}_truth_windows",
    )
    diagnostics = {
        "side": side,
        "num_demonstrations": len(paths),
        "connected_samples": len(connected_array),
        "connected_windows": len(arrays["relative_linear_speed_m_s"]),
        "opening_quantiles_m": {
            "connected_q01": float(occupied_q01),
            "connected_median": float(occupied_median),
            "connected_q99": float(occupied_q99),
            "open_negative_q01": open_q01,
        },
        "feature_quantiles": {
            name: {
                "q05": float(np.quantile(items, 0.05)),
                "median": float(np.median(items)),
                "q95": float(np.quantile(items, 0.95)),
            }
            for name, items in arrays.items()
        },
        "config": config.as_dict(),
    }
    return config, diagnostics


def calibrate_bimanual_relation_estimator(
    paths: Iterable[Path],
    *,
    dataset_sha256: str,
    window_steps: int = 10,
) -> tuple[BimanualRelationEstimatorConfig, dict[str, Any]]:
    """Calibrate two edges from a declared physical-data subset."""

    paths = [Path(path).resolve() for path in paths]
    if not paths:
        raise ValueError("双臂关系标定至少需要一条演示")
    seeds = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            seeds.append(int(np.asarray(archive["seed"]).item()))
    left, left_diagnostics = _calibrate_arm(paths, "left", window_steps=window_steps)
    right, right_diagnostics = _calibrate_arm(paths, "right", window_steps=window_steps)
    config = BimanualRelationEstimatorConfig(
        left=left,
        right=right,
        calibration_source="handover_physical_v1_declared_subset",
        calibration_seeds=tuple(seeds),
        dataset_sha256=dataset_sha256,
    )
    return config, {
        "calibration_paths": [path.name for path in paths],
        "calibration_seeds": seeds,
        "dataset_sha256": dataset_sha256,
        "window_steps": window_steps,
        "privileged_relation_used_for_calibration_only": True,
        "left": left_diagnostics,
        "right": right_diagnostics,
        "config": config.as_dict(),
    }


def replay_bimanual_relation_estimator(
    path: Path,
    config: BimanualRelationEstimatorConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Replay measured geometry only and compare with persisted physics truth."""

    path = Path(path).resolve()
    estimator = BimanualOnlineRelationEstimator(config)
    with np.load(path, allow_pickle=False) as archive:
        left_pose = archive["left_ee_pose"].astype(np.float64)
        right_pose = archive["right_ee_pose"].astype(np.float64)
        object_pose = archive["object_pose"].astype(np.float64)
        truth_labels = archive["relation_label"].astype("U16")
        truth_left = archive["left_connected"].astype(bool)
        truth_right = archive["right_connected"].astype(bool)
        time = archive["time"].astype(np.float64)
        dt = float(np.asarray(archive["control_dt"]).item())
        left_opening = _finger_distance(archive, "left").astype(np.float64)
        right_opening = _finger_distance(archive, "right").astype(np.float64)
    left_velocity = np.concatenate(([0.0], np.diff(left_opening) / dt))
    right_velocity = np.concatenate(([0.0], np.diff(right_opening) / dt))
    inferred_labels = []
    inferred_left = []
    inferred_right = []
    left_confidence = []
    right_confidence = []
    for index in range(len(time)):
        estimate = estimator.update(
            BimanualRelationSample(
                left_ee_pose=left_pose[index],
                right_ee_pose=right_pose[index],
                object_pose=object_pose[index],
                left_finger_distance_m=float(left_opening[index]),
                right_finger_distance_m=float(right_opening[index]),
                left_finger_velocity_m_s=float(left_velocity[index]),
                right_finger_velocity_m_s=float(right_velocity[index]),
                control_dt_s=dt,
            )
        )
        inferred_labels.append(estimate.label)
        inferred_left.append(estimate.left_connected)
        inferred_right.append(estimate.right_connected)
        left_confidence.append(estimate.left.confidence)
        right_confidence.append(estimate.right.confidence)

    inferred_labels_array = np.asarray(inferred_labels, dtype="U16")
    inferred_left_array = np.asarray(inferred_left, dtype=bool)
    inferred_right_array = np.asarray(inferred_right, dtype=bool)

    def edge_metrics(truth: np.ndarray, inferred: np.ndarray) -> dict[str, Any]:
        tp = int(np.count_nonzero(truth & inferred))
        fp = int(np.count_nonzero(~truth & inferred))
        fn = int(np.count_nonzero(truth & ~inferred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }

    summary = {
        "demonstration": path.name,
        "steps": len(time),
        "four_value_accuracy": float(np.mean(inferred_labels_array == truth_labels)),
        "left": edge_metrics(truth_left, inferred_left_array),
        "right": edge_metrics(truth_right, inferred_right_array),
        "truth_both_steps": int(np.count_nonzero(truth_labels == "both")),
        "inferred_both_steps": int(np.count_nonzero(inferred_labels_array == "both")),
        "privileged_contact_used_as_input": False,
    }
    arrays = {
        "time": time,
        "truth_label": truth_labels,
        "inferred_label": inferred_labels_array,
        "truth_left_connected": truth_left,
        "truth_right_connected": truth_right,
        "inferred_left_connected": inferred_left_array,
        "inferred_right_connected": inferred_right_array,
        "left_confidence": np.asarray(left_confidence),
        "right_confidence": np.asarray(right_confidence),
        "left_finger_distance_m": left_opening,
        "right_finger_distance_m": right_opening,
    }
    return summary, arrays
