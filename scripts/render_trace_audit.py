#!/usr/bin/env python3
"""Render the fixed representative single-arm trace audit set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from essay2608.eval.trace_visual import render_audit_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials_dir",
        type=Path,
        default=Path("outputs/single_arm_scientific/v1/trials"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/recovery_scientific/trace_audit_v1"),
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--maximum_frames", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_audit_set(
        args.trials_dir,
        args.output_dir,
        fps=args.fps,
        maximum_frames=args.maximum_frames,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "trials": len(manifest["trials"]), "consistent": manifest["all_failure_taxonomies_consistent"]}))


if __name__ == "__main__":
    main()
