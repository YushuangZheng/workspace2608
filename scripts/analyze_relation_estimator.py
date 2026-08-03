"""Calibrate and replay the phase-independent online relation estimator."""

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

from essay2608.data.dataset import load_dataset
from essay2608.policy.relation import (
    calibrate_relation_estimator,
    replay_relation_estimator,
)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("outputs/single_arm_scientific/relation_calibration_v1"),
)
args = parser.parse_args()


def source_identity() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "source" / "essay2608" / "essay2608" / "policy" / "relation.py",
        root / "source" / "essay2608" / "essay2608" / "data" / "transforms.py",
        root / "source" / "essay2608" / "essay2608" / "data" / "dataset.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, digest.hexdigest()


def plot_replays(replays: list[dict[str, np.ndarray]], names: list[str], path: Path) -> None:
    figure, axes = plt.subplots(len(replays), 1, figsize=(12, 2.1 * len(replays)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, arrays, name in zip(axes, replays, names):
        time = arrays["time"]
        expected = np.isin(arrays["manual_state"], [4, 5, 6])
        axis.fill_between(time, 0.0, 1.0, where=expected, color="#2ca02c", alpha=0.12)
        axis.plot(time, arrays["confidence"], label="EMA confidence", color="#1f77b4")
        axis.plot(time, arrays["connection_score"], label="connection score", color="#ff7f0e", alpha=0.8)
        axis.plot(time, arrays["loss_score"], label="loss score", color="#d62728", alpha=0.8)
        axis.step(time, arrays["connected"].astype(float), label="connected", color="black", where="post")
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("score")
        axis.set_title(name)
        axis.grid(alpha=0.15)
    axes[0].legend(loc="upper right", ncol=4)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle("Frozen-demo replay; green background is evaluation-only states 4–6", y=0.998)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    demonstrations, manifest = load_dataset(args.data_dir, verify_hashes=True)
    config, calibration = calibrate_relation_estimator(demonstrations)
    summaries = []
    replays = []
    for demonstration in demonstrations:
        summary, arrays = replay_relation_estimator(demonstration, config)
        summaries.append(summary)
        replays.append(arrays)

    onset_delays = [item["onset_delay_s"] for item in summaries if item["onset_delay_s"] is not None]
    release_delays = [
        item["release_delay_s"] for item in summaries if item["release_delay_s"] is not None
    ]
    result = {
        "method": "online_relation_estimator_frozen_demo_replay",
        "estimator_phase_independent": True,
        "phase_labels_used_for_calibration_and_evaluation_only": True,
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_dir": str(args.data_dir.resolve()),
        "config": config.as_dict(),
        "calibration": calibration,
        "per_demonstration": summaries,
        "aggregate": {
            "mean_onset_delay_s": float(np.mean(onset_delays)),
            "maximum_onset_delay_s": float(np.max(onset_delays)),
            "mean_release_delay_s": float(np.mean(release_delays)),
            "maximum_release_delay_s": float(np.max(release_delays)),
            "mean_false_positive_fraction": float(
                np.mean([item["false_positive_fraction"] for item in summaries])
            ),
            "mean_false_negative_fraction": float(
                np.mean([item["false_negative_fraction"] for item in summaries])
            ),
        },
    }
    result["source_git_commit"], result["source_sha256"] = source_identity()
    fingerprint_payload = {
        "method": result["method"],
        "dataset_sha256": result["dataset_sha256"],
        "config": result["config"],
        "source_sha256": result["source_sha256"],
    }
    result["analysis_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis.json"
    plot_path = output_dir / "frozen_demo_replay.png"
    analysis_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_replays(replays, [item.path.name for item in demonstrations], plot_path)
    print(
        json.dumps(
            {
                "dataset_sha256": result["dataset_sha256"],
                "config": result["config"],
                "aggregate": result["aggregate"],
                "analysis": str(analysis_path),
                "visualization": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
