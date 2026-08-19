#!/usr/bin/env python3
"""Create a paper-vs-local summary from RACER's official metrics.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/yukun/.cache/racer/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASK_ORDER = (
    "place_cups",
    "place_wine_at_rack_location",
    "sweep_to_dustpan_of_size",
)


def wilson_interval(successes: int, trials: int, z_score: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a Bernoulli proportion."""
    proportion = successes / trials
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument(
        "--paper-reference",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "paper_reference.json",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    metrics_path = args.metrics.resolve()
    output_dir = (args.output_dir or metrics_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reference = json.loads(args.paper_reference.read_text(encoding="utf-8"))
    overall = metrics.get("overall", {})

    rows = []
    for task in TASK_ORDER:
        if task not in overall or task not in metrics:
            raise SystemExit(f"Missing task in metrics: {task}")
        episodes = metrics[task]
        if len(episodes) != 25:
            raise SystemExit(f"Expected 25 episodes for {task}, found {len(episodes)}")
        successes = sum(bool(value.get("success")) for value in episodes.values())
        local_percent = 100.0 * float(overall[task])
        expected_percent = successes * 4.0
        if not math.isclose(local_percent, expected_percent, abs_tol=1e-7):
            raise SystemExit(
                f"Inconsistent rate for {task}: overall={local_percent}, successes={successes}"
            )
        paper = reference["tasks"][task]
        ci_low, ci_high = wilson_interval(successes, 25)
        rows.append(
            {
                "task": task,
                "display_name": paper["display_name"],
                "episodes": 25,
                "successes": successes,
                "local_percent": local_percent,
                "local_wilson95_low_percent": 100.0 * ci_low,
                "local_wilson95_high_percent": 100.0 * ci_high,
                "paper_mean_percent": float(paper["mean"]),
                "paper_std_percent": float(paper["std"]),
                "local_minus_paper_points": local_percent - float(paper["mean"]),
            }
        )

    local_mean = sum(row["local_percent"] for row in rows) / len(rows)
    paper_mean = sum(row["paper_mean_percent"] for row in rows) / len(rows)
    summary = {
        "schema": "racer-three-task-comparison-v1",
        "metrics_path": str(metrics_path),
        "scope": "one released actor checkpoint, 25 fixed episodes per task",
        "local_uncertainty": "two-sided 95% Wilson score interval over 25 Bernoulli episodes",
        "paper_scope": reference["aggregation"],
        "tasks": rows,
        "local_three_task_unweighted_mean_percent": local_mean,
        "paper_three_task_unweighted_mean_percent": paper_mean,
        "warning": reference["comparison_warning"],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# RACER three-task checkpoint comparison",
        "",
        "| Task | Local successes | Local rate [Wilson 95% CI] | Paper mean +/- std | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['display_name']} | {row['successes']}/25 | "
            f"{row['local_percent']:.1f}% "
            f"[{row['local_wilson95_low_percent']:.1f}, "
            f"{row['local_wilson95_high_percent']:.1f}]% | "
            f"{row['paper_mean_percent']:.1f} +/- "
            f"{row['paper_std_percent']:.1f}% | "
            f"{row['local_minus_paper_points']:+.1f} pp |"
        )
    markdown.extend(
        [
            "",
            f"Three-task unweighted mean: local {local_mean:.1f}%, paper {paper_mean:.1f}%.",
            "",
            f"> {reference['comparison_warning']}",
            "",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(markdown), encoding="utf-8")

    labels = [row["display_name"] for row in rows]
    paper_values = [row["paper_mean_percent"] for row in rows]
    paper_errors = [row["paper_std_percent"] for row in rows]
    local_values = [row["local_percent"] for row in rows]
    local_errors = [
        [
            row["local_percent"] - row["local_wilson95_low_percent"]
            for row in rows
        ],
        [
            row["local_wilson95_high_percent"] - row["local_percent"]
            for row in rows
        ],
    ]
    x_positions = list(range(len(rows)))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    paper_bars = axis.bar(
        [value - width / 2 for value in x_positions],
        paper_values,
        width,
        yerr=paper_errors,
        capsize=5,
        label="Paper: 5-seed mean +/- std",
        color="#4C78A8",
    )
    local_bars = axis.bar(
        [value + width / 2 for value in x_positions],
        local_values,
        width,
        yerr=local_errors,
        capsize=5,
        label="Local: one checkpoint, Wilson 95% CI",
        color="#F58518",
    )
    axis.set_ylabel("Success rate (%)")
    axis.set_ylim(0, 110)
    axis.set_xticks(x_positions, labels)
    axis.set_title("RACER released checkpoint: paper vs local fixed episodes")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    axis.bar_label(paper_bars, labels=[f"{value:.1f}" for value in paper_values], padding=4)
    axis.bar_label(local_bars, labels=[f"{value:.1f}" for value in local_values], padding=4)
    figure.savefig(output_dir / "paper_vs_reproduction.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
