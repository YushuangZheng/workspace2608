"""Audit and plot policy-command discontinuities around frame switches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=Path, default=Path("outputs/single_arm_strict/v2"))
parser.add_argument("--method", default="full_dynamac")
parser.add_argument("--seed", type=int, default=6202)
parser.add_argument("--output_dir", type=Path, default=Path("outputs/single_arm_strict/transition_analysis"))
parser.add_argument("--no_plots", action="store_true")
args = parser.parse_args()


def jumps(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(values, axis=0), axis=-1)


def analyze_trial(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz")) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    raw = arrays["raw_action_position"]
    limited = arrays["policy_action_position"]
    applied = arrays["action"][:, :3]
    ee = arrays["ee_position"]
    raw_jump = jumps(raw)
    limited_jump = jumps(limited)
    applied_jump = jumps(applied)
    ee_speed = jumps(ee) / 0.02
    maximum_index = int(np.argmax(raw_jump))
    entry = {
        "file": path.name,
        "condition": result["condition"],
        "seed": result["seed"],
        "max_raw_action_jump_m": float(raw_jump[maximum_index]),
        "max_raw_action_jump_step": maximum_index + 1,
        "phase_before_after": arrays["phase"][maximum_index : maximum_index + 2].astype(int).tolist(),
        "max_rate_limited_action_jump_m": float(np.max(limited_jump)),
        "max_applied_action_jump_m": float(np.max(applied_jump)),
        "max_ee_speed_m_s": float(np.max(ee_speed)),
        "rate_limited_steps": int(np.sum(arrays["action_rate_limited"])),
        "frame_switch_diagnostics": result["metrics"]["frame_switch_diagnostics"],
    }
    arrays.update(
        raw_jump=raw_jump,
        limited_jump=limited_jump,
        applied_jump=applied_jump,
        ee_speed=ee_speed,
    )
    return entry, arrays


def plot_trial(entry: dict, arrays: dict[str, np.ndarray], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for coordinate, label in enumerate(("x", "y", "z")):
        axes[0].plot(arrays["raw_action_position"][:, coordinate], linestyle="--", alpha=0.55, label=f"raw {label}")
        axes[0].plot(arrays["policy_action_position"][:, coordinate], label=f"limited {label}")
    axes[0].set_ylabel("command position [m]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(np.arange(1, len(arrays["raw_jump"]) + 1), arrays["raw_jump"], label="raw policy")
    axes[1].plot(np.arange(1, len(arrays["limited_jump"]) + 1), arrays["limited_jump"], label="rate-limited policy")
    axes[1].plot(
        np.arange(1, len(arrays["applied_jump"]) + 1),
        arrays["applied_jump"],
        label="after perturbation",
        alpha=0.7,
    )
    axes[1].set_ylabel("command jump [m]")
    axes[1].legend()
    axes[2].plot(np.arange(1, len(arrays["ee_speed"]) + 1), arrays["ee_speed"])
    axes[2].set_ylabel("EE speed [m/s]")
    axes[2].set_xlabel("control step")
    for switch in entry["frame_switch_diagnostics"]:
        for axis in axes:
            axis.axvline(switch["step"], color="black", alpha=0.18, linewidth=0.8)
    figure.suptitle(f"{entry['condition']} / seed {entry['seed']}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    trial_dir = args.result_dir.resolve() / "trials"
    pattern = f"{args.method}__*__seed_{args.seed}.json"
    paths = sorted(trial_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trials match {trial_dir / pattern}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in paths:
        entry, arrays = analyze_trial(path)
        entries.append(entry)
        if not args.no_plots:
            plot_trial(entry, arrays, args.output_dir / f"{path.stem}.png")
    summary = {
        "method": args.method,
        "seed": args.seed,
        "maximum_raw_action_jump_m": max(entry["max_raw_action_jump_m"] for entry in entries),
        "maximum_rate_limited_action_jump_m": max(
            entry["max_rate_limited_action_jump_m"] for entry in entries
        ),
        "maximum_applied_action_jump_m": max(entry["max_applied_action_jump_m"] for entry in entries),
        "maximum_ee_speed_m_s": max(entry["max_ee_speed_m_s"] for entry in entries),
        "trials": entries,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
