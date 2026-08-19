#!/usr/bin/env python3
"""Run one bounded FailGen episode through the released public API.

This is a functional smoke test.  It intentionally stores only keyframe PNGs
and never claims to implement the paper's training or evaluation protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from failgen.env_wrapper import FailGenEnvWrapper
from failgen.fail_grasp import GraspFailure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tries", default=1, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_tries < 1:
        raise SystemExit("--max-tries must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    wrapper = FailGenEnvWrapper(
        task_name="basketball_in_hoop",
        headless=True,
        record=False,
        save_data=True,
        save_path=str(args.output),
        save_keyframes_only=True,
    )

    try:
        target = None
        for failure in wrapper.manager._failures:
            enabled = failure.failure_type == GraspFailure.FAILURE_TYPE
            failure.set_enabled(enabled)
            if enabled:
                target = failure

        if target is None:
            raise RuntimeError("basketball_in_hoop has no released grasp failure")
        if str(wrapper.config.data.renderer) != "opengl3":
            raise RuntimeError(
                f"Expected the official opengl3 renderer, got "
                f"{wrapper.config.data.renderer!r}"
            )
        if not target.waypoints_indices:
            raise RuntimeError("released grasp failure has no waypoint index")

        waypoint_index = target.waypoints_indices[0]
        target.change_waypoint_fail_name(f"waypoint{waypoint_index}")
        wrapper.reset()

        demo = None
        success = True
        for _ in range(args.max_tries):
            demo, success = wrapper.get_failure()
            if demo is not None and not success:
                break

        if demo is None or success:
            raise RuntimeError(
                "FailGen did not produce the requested failure within the "
                f"{args.max_tries} allowed attempt(s)"
            )

        wrapper.save_keyframe_data(
            ep_idx=0,
            fail_type=GraspFailure.FAILURE_TYPE,
            wp_idx=waypoint_index,
        )
        artifact_dir = args.output / (
            "basketball_in_hoop_grasp_"
            f"wp{waypoint_index}_episode0"
        )
        artifacts = sorted(artifact_dir.glob("*.png"))
        if not artifacts:
            raise RuntimeError(f"No keyframe PNGs saved under {artifact_dir}")

        print("Saved FailGen smoke episode 1 / 1")
        print(f"Renderer: {wrapper.config.data.renderer}")
        print(f"Waypoint index: {waypoint_index}")
        print(f"Keyframe PNGs: {len(artifacts)}")
        print(f"Output: {artifact_dir}")
        return 0
    finally:
        wrapper.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
