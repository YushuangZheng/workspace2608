"""Generate additive phase-level attribution from saved rollout traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from essay2608.eval import analyze_phase_trace, compare_paired_methods, summarize_success_metrics
from essay2608.policy.base import PHASE_NAMES


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=Path, default=Path("outputs/single_arm_strict/v2"))
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("outputs/single_arm_scientific/audit_v1/phase_diagnostics"),
)
parser.add_argument("--methods", nargs="+", default=["mask_only", "full_dynamac"])
parser.add_argument("--control_dt", type=float, default=0.02)
args = parser.parse_args()


def load_analysis(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz")) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return analyze_phase_trace(arrays, result, control_dt=args.control_dt)


def plot_phase_attribution(comparison: dict, output_path: Path) -> None:
    path_difference = [
        comparison["phase_mean_differences"][name]["candidate_minus_baseline_path_m"] * 1000.0
        for name in PHASE_NAMES
    ]
    step_difference = [
        comparison["phase_mean_differences"][name]["candidate_minus_baseline_steps"]
        for name in PHASE_NAMES
    ]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ["tab:green" if value < 0.0 else "tab:red" for value in path_difference]
    axes[0].bar(np.arange(len(PHASE_NAMES)), path_difference, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Full - Mask path [mm]")
    axes[1].bar(np.arange(len(PHASE_NAMES)), step_difference, color="tab:blue")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Full - Mask steps")
    axes[1].set_xticks(np.arange(len(PHASE_NAMES)), PHASE_NAMES, rotation=35, ha="right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    trial_dir = args.result_dir.resolve() / "trials"
    all_result_paths = sorted(trial_dir.glob("*.json"))
    all_results = [json.loads(path.read_text(encoding="utf-8")) for path in all_result_paths]
    paths = sorted(
        path
        for method in args.methods
        for path in trial_dir.glob(f"{method}__*__seed_*.json")
    )
    if not paths:
        raise FileNotFoundError(f"No matching trials in {trial_dir}")
    analyses = [load_analysis(path) for path in paths]
    residual = max(abs(item["path_partition_residual_m"]) for item in analyses)
    if residual > 1.0e-6:
        raise RuntimeError(f"Phase partition does not reproduce saved path: {residual}")
    comparison = compare_paired_methods(analyses)
    summary = {
        "source_result_dir": str(args.result_dir.resolve()),
        "control_dt": args.control_dt,
        "maximum_path_partition_residual_m": residual,
        "success_metric_audit": summarize_success_metrics(all_results),
        "comparison": comparison,
        "trials": analyses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_phase_attribution(comparison, args.output_dir / "phase_attribution.png")
    print(json.dumps({key: value for key, value in summary.items() if key != "trials"}, indent=2))
    print(f"Saved phase diagnostics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
