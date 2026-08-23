#!/usr/bin/env python3
"""Generate one bounded episode with the released AHA FailGen wrapper.

The caller is responsible for process isolation, Xvfb, timeouts, and retries.
This worker launches exactly one task environment and calls ``get_failure``
exactly once. It never changes the released FailGen implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from failgen.env_wrapper import FailGenEnvWrapper


# Same order as FAILURES_LIST in the released ex_custom_data_generator.py.
OFFICIAL_FAILURE_ORDER = (
    "grasp",
    "slip",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "translation_x",
    "translation_y",
    "translation_z",
    "no_rotation",
    "wrong_sequence",
    "wrong_object",
)


class FailGenNotProduced(RuntimeError):
    """The single permitted FailGen call did not yield a failed demo."""


class ReleasedConfigurationError(RuntimeError):
    """The released task configuration cannot support the bounded protocol."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--max-tries", default=1, type=int)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_failure(wrapper: FailGenEnvWrapper):
    by_type = {failure.failure_type: failure for failure in wrapper.manager._failures}
    for failure_type in OFFICIAL_FAILURE_ORDER:
        candidate = by_type.get(failure_type)
        if candidate is None:
            continue
        if failure_type == "wrong_object" or candidate.waypoints_indices:
            return candidate
    raise ReleasedConfigurationError(
        "No released failure type with a usable waypoint is configured"
    )


def artifact_path(output: Path, task: str, failure_type: str, waypoint: int) -> Path:
    if waypoint == -1:
        return output / f"{task}_{failure_type}_episode0"
    return output / f"{task}_{failure_type}_wp{waypoint}_episode0"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_tries != 1:
        raise ReleasedConfigurationError("bounded evaluation requires --max-tries 1")

    args.output.mkdir(parents=True, exist_ok=False)
    wrapper = FailGenEnvWrapper(
        task_name=args.task,
        headless=True,
        record=False,
        save_data=True,
        save_path=str(args.output),
        save_keyframes_only=True,
    )

    try:
        target = select_failure(wrapper)
        for failure in wrapper.manager._failures:
            failure.set_enabled(failure is target)

        renderer = str(wrapper.config.data.renderer)
        if renderer != "opengl3":
            raise ReleasedConfigurationError(
                f"Expected released renderer opengl3, got {renderer!r}"
            )

        failure_type = target.failure_type
        waypoint = -1
        if failure_type != "wrong_object":
            waypoint = int(target.waypoints_indices[0])
            target.change_waypoint_fail_name(f"waypoint{waypoint}")

        wrapper.reset()
        demo, success = wrapper.get_failure()
        if demo is None or success:
            raise FailGenNotProduced(
                "The one permitted get_failure call did not produce a failed demo"
            )

        wrapper.save_keyframe_data(
            ep_idx=0,
            fail_type=failure_type,
            wp_idx=waypoint,
        )
        artifact_dir = artifact_path(
            args.output, args.task, failure_type, waypoint
        )
        png_count = len(list(artifact_dir.glob("*.png")))
        if png_count == 0:
            raise RuntimeError(f"No keyframe PNGs saved under {artifact_dir}")

        return {
            "schema_version": 1,
            "status": "success",
            "task": args.task,
            "episode": 0,
            "max_tries": 1,
            "get_failure_calls": 1,
            "failure_type": failure_type,
            "waypoint": waypoint,
            "renderer": renderer,
            "artifact_dir": str(artifact_dir.resolve()),
            "png_count_unverified": png_count,
            "finished_at_utc": utc_now(),
        }
    finally:
        wrapper.shutdown()


def main() -> int:
    args = parse_args()
    started = utc_now()
    try:
        result = run(args)
        result["started_at_utc"] = started
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "task": args.task,
            "episode": 0,
            "max_tries": args.max_tries,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
