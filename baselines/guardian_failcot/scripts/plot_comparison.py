#!/usr/bin/env python3
"""Plot the matched Guardian Table-II paper/reproduction comparison."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / "runtime/matplotlib"),
)
import matplotlib.pyplot as plt


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half_width, center + half_width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = [f"{row['benchmark']} · {row['mode']}" for row in rows]
    paper = [100.0 * float(row["paper_accuracy"]) for row in rows]
    reproduced = [100.0 * float(row["reproduced_accuracy"]) for row in rows]
    intervals = [
        tuple(100.0 * value for value in wilson_interval(int(row["correct"]), int(row["n"])))
        for row in rows
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    y_positions = list(range(len(rows)))[::-1]

    for y, paper_value, local_value, interval, row in zip(
        y_positions, paper, reproduced, intervals, rows
    ):
        ax.plot([paper_value, local_value], [y, y], color="#9ca3af", linewidth=1.5, zorder=1)
        ax.plot([interval[0], interval[1]], [y, y], color="#005f73", linewidth=3.0, zorder=2)
        ax.scatter(local_value, y, s=60, color="#005f73", edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(paper_value, y, s=68, marker="D", facecolor="none", edgecolor="#bb3e03", linewidth=1.8, zorder=4)
        ax.text(
            101.0,
            y,
            f"{local_value:.1f}% ({row['correct']}/{row['n']})  Δ {float(row['delta_pp']):+.1f} pp",
            va="center",
            ha="left",
            fontsize=9,
            color="#374151",
        )

    ax.set_yticks(y_positions, labels)
    ax.set_xlim(55, 119)
    ax.set_xticks(range(60, 101, 10))
    ax.set_xlabel("Accuracy (%)")
    fig.suptitle(
        "Guardian/FailCoT — matched Table II OOD reproduction",
        x=0.17,
        y=0.965,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.17,
        0.905,
        "Official thinking checkpoint + official OOD release; unmodified evaluator",
        fontsize=10,
        color="#4b5563",
        ha="left",
    )
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=60, color="#005f73", label="Local reproduction (95% Wilson CI)")
    ax.scatter([], [], s=68, marker="D", facecolor="none", edgecolor="#bb3e03", linewidth=1.8, label="Paper")
    fig.legend(loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.045))
    fig.text(
        0.01,
        0.012,
        "Intervals reflect finite benchmark samples; paper entries are published point estimates.",
        fontsize=8.5,
        color="#6b7280",
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.84, bottom=0.20)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
