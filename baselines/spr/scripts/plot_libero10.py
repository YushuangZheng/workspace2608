#!/data/yukun/miniconda3/envs/dynamac-spr/bin/python
"""Render the validated ten-task SPR LIBERO-Long comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", "/data/yukun/essay2608/baselines/spr/runtime/matplotlib"
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULT_ROOT = Path(
    "/data/yukun/essay2608/baselines/spr/results/libero10_full"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()

    summary_path = args.result_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("verification", {}).get("pass") is not True:
        raise RuntimeError("refusing to plot an unverified aggregate")
    tasks = summary["tasks"]
    if len(tasks) != 10 or any(int(task["episodes"]) != 50 for task in tasks):
        raise RuntimeError("refusing to plot before all 10 x 50 episodes are complete")

    task_rates = [float(task["success_rate_percent"]) for task in tasks]
    aggregate = summary["aggregate"]
    aggregate_rate = float(aggregate["success_rate_percent"])
    ci_low, ci_high = map(float, aggregate["wilson_95_ci_percent"])
    paper_rate = float(summary["paper_reference"]["libero_long_success_rate_percent"])

    labels = [f"Task {task['task_id']}" for task in tasks] + ["Aggregate"]
    y_positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")

    axis.scatter(
        task_rates,
        y_positions[:-1],
        s=62,
        color="#4C78A8",
        edgecolor="#1F4E79",
        linewidth=0.8,
        zorder=3,
    )
    axis.errorbar(
        aggregate_rate,
        y_positions[-1],
        xerr=[[aggregate_rate - ci_low], [ci_high - aggregate_rate]],
        fmt="D",
        markersize=7.5,
        color="#1F4E79",
        ecolor="#1F4E79",
        elinewidth=2.2,
        capsize=5,
        zorder=4,
    )
    axis.axvline(paper_rate, color="#555555", linestyle="--", linewidth=1.6, zorder=1)
    axis.text(
        paper_rate + 0.8,
        -0.72,
        "Paper SPR 82.8%",
        color="#555555",
        fontsize=9.5,
        ha="left",
        va="center",
    )

    for index, rate in enumerate(task_rates):
        axis.text(min(rate + 1.1, 98.3), index, f"{rate:.1f}%", va="center", fontsize=9)
    axis.text(
        min(ci_high + 1.1, 97.0),
        y_positions[-1],
        f"{aggregate_rate:.1f}%  (95% CI {ci_low:.1f}-{ci_high:.1f})",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1F4E79",
    )

    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xticks(np.arange(0, 101, 10))
    axis.set_xlabel("Task success rate (%)")
    axis.grid(axis="x", color="#D9DEE5", linewidth=0.8)
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#7A828A")
    axis.tick_params(colors="#202124")
    axis.set_title(
        "SPR LIBERO-Long success rates by task",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color="#202124",
        pad=31,
    )
    axis.text(
        0,
        1.035,
        "Released-code evaluator; n=50 per task, n=500 aggregate; aggregate error bar is Wilson 95% CI",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#555555",
        ha="left",
    )
    fig.text(
        0.105,
        0.015,
        "Paper comparator: separately trained SPR/Ours 82.8% ten-task aggregate. Ours* 85.4% is excluded.",
        fontsize=8.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.14, right=0.94, top=0.86, bottom=0.10)

    png_path = args.result_root / "spr_libero10_task_rates.png"
    svg_path = args.result_root / "spr_libero10_task_rates.svg"
    fig.savefig(png_path, dpi=200, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
