"""Run and visualize velocity-based skill segmentation diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from essay2608.data import SegmentationConfig, analyze_segmentation, load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("outputs/single_arm_scientific/segmentation_v1"),
)
parser.add_argument("--smoothing_duration", type=float, default=0.10)
parser.add_argument("--minimum_low_speed_duration", type=float, default=0.12)
parser.add_argument("--maximum_merge_gap", type=float, default=0.08)
parser.add_argument("--endpoint_tolerance", type=float, default=0.06)
parser.add_argument("--threshold_quantile", type=float, default=0.40)
parser.add_argument("--alignment_tolerance_normalized", type=float, default=0.05)
args = parser.parse_args()


def source_identity() -> tuple[str, str]:
    """Return the current Git revision and a content hash of this diagnostic."""

    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "source" / "essay2608" / "essay2608" / "data" / "segmentation.py",
        root / "source" / "essay2608" / "essay2608" / "data" / "transforms.py",
        root / "source" / "essay2608" / "essay2608" / "data" / "dataset.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, digest.hexdigest()


def plot_velocity_traces(traces, thresholds: dict, output_path: Path) -> None:
    figure, axes = plt.subplots(len(traces), 1, figsize=(12, 2.25 * len(traces)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, trace in zip(axes, traces):
        axis.plot(trace.time, trace.linear_speed_m_s, color="#1f77b4", label="linear m/s")
        axis.plot(trace.time, trace.angular_speed_rad_s, color="#ff7f0e", label="angular rad/s")
        axis.axhline(
            thresholds["linear_speed_m_s"],
            color="#1f77b4",
            linestyle="--",
            linewidth=0.9,
        )
        axis.axhline(
            thresholds["angular_speed_rad_s"],
            color="#ff7f0e",
            linestyle="--",
            linewidth=0.9,
        )
        for start, end in trace.low_speed_intervals:
            axis.axvspan(
                trace.time[start],
                trace.time[min(end, len(trace.time) - 1)],
                color="#2ca02c",
                alpha=0.15,
            )
        for boundary in trace.manual_boundary_indices:
            axis.axvline(trace.time[boundary], color="0.72", linewidth=0.7)
        for boundary in trace.boundary_indices:
            axis.axvline(trace.time[boundary], color="black", linewidth=1.3)
        axis.set_title(trace.demonstration_name)
        axis.set_ylabel("speed")
        axis.grid(alpha=0.15)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(
        "Low-speed segmentation: green intervals, black candidates, gray manual state boundaries",
        y=0.998,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_alignment(alignment: list[dict], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 3.8))
    demo_names = sorted(
        {member["demonstration"] for cluster in alignment for member in cluster["members"]}
    )
    demo_index = {name: index for index, name in enumerate(demo_names)}
    for cluster in alignment:
        axis.axvline(cluster["mean_normalized_time"], color="0.75", linewidth=1.0)
        for member in cluster["members"]:
            axis.scatter(
                member["normalized_time"],
                demo_index[member["demonstration"]],
                color="#1f77b4",
                s=30,
            )
    axis.set_yticks(range(len(demo_names)), demo_names)
    axis.set_xlabel("normalized demonstration time")
    axis.set_title("Reference-free alignment of candidate boundaries")
    axis.set_xlim(0.0, 1.0)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    demonstrations, manifest = load_dataset(args.data_dir, verify_hashes=True)
    config = SegmentationConfig(
        smoothing_duration_s=args.smoothing_duration,
        minimum_low_speed_duration_s=args.minimum_low_speed_duration,
        maximum_merge_gap_s=args.maximum_merge_gap,
        endpoint_tolerance_s=args.endpoint_tolerance,
        threshold_quantile=args.threshold_quantile,
        alignment_tolerance_normalized=args.alignment_tolerance_normalized,
    )
    result, traces = analyze_segmentation(demonstrations, config)
    result["dataset_sha256"] = manifest["dataset_sha256"]
    result["dataset_dir"] = str(args.data_dir.resolve())
    result["source_git_commit"], result["source_sha256"] = source_identity()
    fingerprint_payload = {
        "method": result["method"],
        "config": result["config"],
        "dataset_sha256": result["dataset_sha256"],
        "source_sha256": result["source_sha256"],
    }
    result["analysis_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis.json"
    velocity_path = output_dir / "velocity_boundaries.png"
    alignment_path = output_dir / "boundary_alignment.png"
    analysis_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_velocity_traces(traces, result["calibrated_thresholds"], velocity_path)
    plot_alignment(result["aligned_boundaries"], alignment_path)

    print(
        json.dumps(
            {
                "dataset_sha256": result["dataset_sha256"],
                "calibrated_thresholds": result["calibrated_thresholds"],
                "consistency": result["consistency"],
                "candidate_boundary_times_s": {
                    trace.demonstration_name: [
                        item["time_s"] for item in trace.summary["candidate_boundaries"]
                    ]
                    for trace in traces
                },
                "analysis": str(analysis_path),
                "visualizations": [str(velocity_path), str(alignment_path)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
