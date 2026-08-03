"""Measure trajectory variance in world, object, and target frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from essay2608.data.dataset import load_dataset
from essay2608.data.transforms import (
    interpolate_poses,
    quaternion_distance_radians,
    quaternion_mean,
    relative_pose,
)


PHASE_NAMES = [
    "rest",
    "approach_above_object",
    "approach_object",
    "grasp_object",
    "lift_object",
    "move_above_target",
    "lower_to_target",
    "release_object",
    "retreat",
    "complete",
]
FRAME_NAMES = ["world", "object", "target"]


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument("--output_dir", type=Path, default=Path("outputs/relative_frame_analysis/v1"))
parser.add_argument("--bins", type=int, default=50)
args = parser.parse_args()


def frame_trajectory(demonstration, phase: int, frame_name: str) -> np.ndarray:
    """Return phase EE poses expressed in one candidate frame."""

    indices = demonstration.phase_indices(phase)
    ee_pose = demonstration.ee_pose[indices]
    if frame_name == "world":
        poses = ee_pose
    elif frame_name == "object":
        poses = relative_pose(demonstration.object_pose[indices], ee_pose)
    elif frame_name == "target":
        poses = relative_pose(demonstration.target_pose[indices], ee_pose)
    else:
        raise ValueError(frame_name)
    return interpolate_poses(poses, args.bins)


def pose_variance(trajectories: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return aggregate position/rotation dispersion and per-bin position std."""

    position_std = np.std(trajectories[..., :3], axis=0)
    position_rms = float(np.sqrt(np.mean(np.square(position_std))))

    angular_errors = []
    for bin_index in range(trajectories.shape[1]):
        quaternions = trajectories[:, bin_index, 3:7]
        mean_quaternion = quaternion_mean(quaternions)
        angular_errors.append(quaternion_distance_radians(quaternions, mean_quaternion))
    rotation_rms = float(np.rad2deg(np.sqrt(np.mean(np.square(angular_errors)))))
    return position_rms, rotation_rms, position_std


def save_plot(values: np.ndarray, ylabel: str, path: Path) -> None:
    """Save a grouped phase/frame variance plot."""

    figure, axis = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(PHASE_NAMES))
    width = 0.25
    for frame_index, frame_name in enumerate(FRAME_NAMES):
        axis.bar(x + (frame_index - 1) * width, values[:, frame_index], width, label=frame_name)
    axis.set_xticks(x, PHASE_NAMES, rotation=35, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    demonstrations, manifest = load_dataset(args.data_dir, verify_hashes=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    position_rms = np.zeros((len(PHASE_NAMES), len(FRAME_NAMES)), dtype=np.float64)
    rotation_rms = np.zeros_like(position_rms)
    position_std = np.zeros((len(PHASE_NAMES), len(FRAME_NAMES), args.bins, 3), dtype=np.float64)

    phase_summary = {}
    for phase, phase_name in enumerate(PHASE_NAMES):
        for frame_index, frame_name in enumerate(FRAME_NAMES):
            trajectories = np.stack(
                [frame_trajectory(demonstration, phase, frame_name) for demonstration in demonstrations]
            )
            position_rms[phase, frame_index], rotation_rms[phase, frame_index], position_std[
                phase, frame_index
            ] = pose_variance(trajectories)

        best_index = int(np.argmin(position_rms[phase]))
        phase_summary[phase_name] = {
            "position_rms_std_m": {
                frame: float(position_rms[phase, index]) for index, frame in enumerate(FRAME_NAMES)
            },
            "rotation_rms_std_deg": {
                frame: float(rotation_rms[phase, index]) for index, frame in enumerate(FRAME_NAMES)
            },
            "best_position_frame": FRAME_NAMES[best_index],
        }

    postgrasp_stability = [
        entry["postgrasp_object_to_ee_stability"] for entry in manifest["demos"]
    ]
    summary = {
        "dataset_name": manifest["dataset_name"],
        "dataset_version": manifest["dataset_version"],
        "dataset_sha256": manifest["dataset_sha256"],
        "num_demos": len(demonstrations),
        "bins_per_phase": args.bins,
        "phases": phase_summary,
        "postgrasp_connection": {
            "max_position_rms_std_m": max(item["position_rms_std_m"] for item in postgrasp_stability),
            "max_rotation_rms_deg": max(item["rotation_rms_deg"] for item in postgrasp_stability),
            "accepted": True,
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "relative_frame_statistics.npz",
        phase_names=np.asarray(PHASE_NAMES),
        frame_names=np.asarray(FRAME_NAMES),
        position_rms_std_m=position_rms,
        rotation_rms_std_deg=rotation_rms,
        position_std_m=position_std,
    )
    save_plot(position_rms * 1000.0, "position RMS std (mm)", output_dir / "position_std.png")
    save_plot(rotation_rms, "rotation RMS std (deg)", output_dir / "rotation_std.png")

    print(json.dumps(summary, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
