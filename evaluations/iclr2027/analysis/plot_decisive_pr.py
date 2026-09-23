"""Render the descriptive main10 precision--recall comparison.

Figure contract
---------------
Core conclusion: the monitors exhibit different event sensitivity--precision
trade-offs; the PR curve is descriptive and does not choose the fixed-FI
operating point.
Results-level question: how much event precision is retained as recall grows on
the identical scored shadow population?
Archetype: single-panel quantitative comparison.
Target/output: ICLR appendix, 89 mm wide, editable SVG/PDF plus 600 dpi TIFF.
Statistics: three supervised-GRU seeds are shown individually; no unreported
averaging or uncertainty transformation is applied.
Reviewer risk: do not visually imply that a PR threshold was selected on test.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "evaluations"
    / "iclr2027"
    / "results"
    / "reviewer_decisive"
    / "equal_fi"
    / "derived"
    / "main10_pr_curve.csv"
)
METRICS = SOURCE.with_name("main10_recall_at_fixed_fi.csv")
OUTPUT = ROOT / "iclr2027" / "figures" / "Figure6"


def _alignment_helper():
    scripts = os.environ.get("NATURE_FIGURE_SKILL_SCRIPTS")
    if not scripts:
        raise RuntimeError("NATURE_FIGURE_SKILL_SCRIPTS must identify the QA scripts")
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _prevalence() -> float:
    rows = _rows(METRICS)
    row = next(
        value
        for value in rows
        if value["method_key"] == "m2"
        and float(value["target_calibration_fi_per_1000_cycles"]) == 1.0
    )
    positives = int(row["physically_triggered_events"])
    scored = int(row["scored_nominal_episodes"]) + int(
        row["scored_perturbed_episodes"]
    )
    return positives / scored


def _curve(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    recall = np.asarray([float(row["recall"]) for row in rows], dtype=np.float64)
    precision = np.asarray(
        [float(row["precision"]) for row in rows], dtype=np.float64
    )
    if not np.all(np.isfinite(recall)) or not np.all(np.isfinite(precision)):
        raise ValueError("PR source contains non-finite values")
    if np.all(np.diff(recall) <= 1e-12):
        recall = recall[::-1]
        precision = precision[::-1]
    elif not np.all(np.diff(recall) >= -1e-12):
        order = np.argsort(recall, kind="stable")
        recall = recall[order]
        precision = precision[order]
    if np.any(np.diff(recall) < -1e-12):
        raise RuntimeError("recall is not monotone after ordering")
    return recall, precision


def render() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows(SOURCE):
        grouped[row["method_key"]].append(row)
    required = {
        "m2",
        "m3",
        "m4_seed1103",
        "m4_seed2207",
        "m4_seed3301",
        "m6",
    }
    if set(grouped) != required:
        raise RuntimeError(f"unexpected PR methods: {sorted(grouped)}")

    width_in = 89.0 / 25.4
    fig, ax = plt.subplots(figsize=(width_in, 2.75), constrained_layout=True)
    styles = {
        "m2": ("Trajectory-Likelihood", "#767676", 1.0, 0.95, "-"),
        "m3": ("FAIL-Detect", "#42949E", 1.0, 0.95, "-"),
        "m6": ("TSF-Monitor", "#B64342", 1.8, 1.0, "-"),
    }
    for key in ("m2", "m3"):
        label, color, linewidth, alpha, linestyle = styles[key]
        recall, precision = _curve(grouped[key])
        ax.step(
            recall,
            precision,
            where="post",
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=label,
        )
    for index, key in enumerate(("m4_seed1103", "m4_seed2207", "m4_seed3301")):
        recall, precision = _curve(grouped[key])
        ax.step(
            recall,
            precision,
            where="post",
            color="#0F4D92",
            linewidth=0.8,
            alpha=0.52,
            label="Sup. GRU (3 seeds)" if index == 0 else None,
        )
    label, color, linewidth, alpha, linestyle = styles["m6"]
    recall, precision = _curve(grouped["m6"])
    ax.step(
        recall,
        precision,
        where="post",
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        label=label,
        zorder=5,
    )
    prevalence = _prevalence()
    ax.axhline(
        prevalence,
        color="#B8B8B8",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=0,
        label=f"Event prevalence ({100 * prevalence:.1f}%)",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Event recall")
    ax.set_ylabel("Event precision")
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="both", color="#E6E6E6", linewidth=0.5, zorder=-10)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
        borderaxespad=0.0,
    )
    fig.canvas.draw()
    require_alignment = _alignment_helper()
    require_alignment(
        fig,
        json_out=str(OUTPUT) + ".alignment.json",
        overlay_svg=str(OUTPUT) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT) + ".svg", bbox_inches="tight")
    fig.savefig(str(OUTPUT) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUTPUT) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(OUTPUT) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
