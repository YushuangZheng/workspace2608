#!/usr/bin/env python3
"""Fail closed unless one fixed RACER episode completed with valid artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


NATIVE_FAILURE = re.compile(
    r"segmentation fault|sigsegv|invalid pointer|core dumped|swrast_dri",
    re.IGNORECASE,
)
CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--actor-log", type=Path, required=True)
    parser.add_argument("--task", default="place_cups")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    episodes = metrics.get(args.task)
    if not isinstance(episodes, dict) or set(episodes) != {str(args.episode)}:
        raise SystemExit(f"expected exactly {args.task} episode {args.episode} in metrics")
    record = episodes[str(args.episode)]
    if not isinstance(record.get("success"), bool):
        raise SystemExit("episode success field is not boolean")
    if not isinstance(record.get("episode_len"), int) or record["episode_len"] < 1:
        raise SystemExit("episode length is missing or invalid")
    if set(metrics.get("overall", {})) != {args.task, "avg_success_rate"}:
        raise SystemExit("overall metrics do not describe exactly the gated task")

    episode_dir = args.metrics.parent / args.task / str(args.episode)
    statistic = episode_dir / "episode_statistic.json"
    if not statistic.is_file():
        raise SystemExit(f"episode statistic is missing: {statistic}")
    json.loads(statistic.read_text(encoding="utf-8"))
    markers = list(episode_dir.glob("success_step*")) + list(episode_dir.glob("failure_step*"))
    if len(markers) != 1:
        raise SystemExit("expected exactly one success/failure marker")
    for camera in CAMERAS:
        gif = episode_dir / f"{camera}_rgb.gif"
        if not gif.is_file() or gif.stat().st_size == 0:
            raise SystemExit(f"camera GIF is missing or empty: {gif}")

    actor_log = args.actor_log.read_text(encoding="utf-8", errors="replace")
    match = NATIVE_FAILURE.search(actor_log)
    if match:
        raise SystemExit(f"native-renderer failure marker in actor log: {match.group(0)}")

    print(
        json.dumps(
            {
                "ok": True,
                "task": args.task,
                "episode": args.episode,
                "success": record["success"],
                "episode_len": record["episode_len"],
                "camera_gifs": len(CAMERAS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
