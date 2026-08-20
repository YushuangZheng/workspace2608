#!/usr/bin/env python3
"""Fail closed unless one fixed RACER episode completed with valid artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


NATIVE_FAILURE = re.compile(
    r"segmentation fault|sigsegv|invalid pointer|core dumped|swrast_dri",
    re.IGNORECASE,
)
CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
EXPECTED_GIF_SIZE = (256, 346)
EXPECTED_POINT_CLOUD_SHAPE = (3, 512, 512)
MIN_POINT_CLOUD_AXIS_SPAN = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gif(path: Path):
    try:
        with Image.open(path) as image:
            if image.format != "GIF":
                raise ValueError(f"format {image.format!r} is not GIF")
            if image.size != EXPECTED_GIF_SIZE:
                raise ValueError(f"size {image.size} != {EXPECTED_GIF_SIZE}")
            image.verify()

        frame_count = 0
        pixel_min = 255
        pixel_max = 0
        with Image.open(path) as image:
            for frame in ImageSequence.Iterator(image):
                if frame.size != EXPECTED_GIF_SIZE:
                    raise ValueError(
                        f"frame {frame_count} size {frame.size} != {EXPECTED_GIF_SIZE}"
                    )
                rgb = frame.convert("RGB")
                rgb.load()
                array = np.asarray(rgb)
                if array.shape != (EXPECTED_GIF_SIZE[1], EXPECTED_GIF_SIZE[0], 3):
                    raise ValueError(f"decoded frame shape is invalid: {array.shape}")
                pixel_min = min(pixel_min, int(array.min()))
                pixel_max = max(pixel_max, int(array.max()))
                frame_count += 1
        if frame_count < 1:
            raise ValueError("GIF has no decodable frames")
        if pixel_max <= pixel_min:
            raise ValueError("decoded pixels are degenerate")
    except (OSError, SyntaxError, ValueError) as error:
        raise SystemExit(f"invalid camera GIF {path}: {error}") from error
    return {
        "frames": frame_count,
        "size": list(EXPECTED_GIF_SIZE),
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
    }


def validate_point_cloud_evidence(path: Path, task: str, episode: int):
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema") != "racer_gate_point_cloud_evidence_v1":
        raise SystemExit("point-cloud evidence schema is invalid")
    if evidence.get("task") != task or evidence.get("episode") != episode:
        raise SystemExit("point-cloud evidence task/episode does not match the gate")
    cameras = evidence.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(CAMERAS):
        raise SystemExit("point-cloud evidence must contain exactly four cameras")
    npz_name = evidence.get("npz_file")
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise SystemExit("point-cloud NPZ filename is invalid")
    npz_path = path.parent / npz_name
    if not npz_path.is_file() or npz_path.stat().st_size == 0:
        raise SystemExit(f"raw point-cloud NPZ is missing or empty: {npz_path}")
    if evidence.get("npz_sha256") != sha256_file(npz_path):
        raise SystemExit("raw point-cloud NPZ SHA-256 does not match its evidence")

    results = {}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            expected_keys = {f"{camera}_point_cloud" for camera in CAMERAS}
            if set(archive.files) != expected_keys:
                raise ValueError("raw NPZ must contain exactly four point-cloud arrays")
            for camera in CAMERAS:
                key = f"{camera}_point_cloud"
                array = np.asarray(archive[key])
                if array.shape != EXPECTED_POINT_CLOUD_SHAPE:
                    raise ValueError(
                        f"{key} shape {array.shape} != {EXPECTED_POINT_CLOUD_SHAPE}"
                    )
                if array.dtype != np.float32:
                    raise ValueError(f"{key} dtype {array.dtype} != float32")
                if not np.isfinite(array).all():
                    raise ValueError(f"{key} contains non-finite values")
                contiguous = np.ascontiguousarray(array)
                flattened = contiguous.reshape(3, -1)
                axis_span = flattened.max(axis=1) - flattened.min(axis=1)
                if not np.all(axis_span > MIN_POINT_CLOUD_AXIS_SPAN):
                    raise ValueError(
                        f"{key} is degenerate; axis spans={axis_span.tolist()}"
                    )
                array_hash = hashlib.sha256(
                    contiguous.tobytes(order="C")
                ).hexdigest()
                metadata = cameras[camera]
                if metadata.get("key") != key:
                    raise ValueError(f"{camera} evidence key is inconsistent")
                if metadata.get("shape") != list(EXPECTED_POINT_CLOUD_SHAPE):
                    raise ValueError(f"{key} evidence shape is inconsistent")
                if metadata.get("dtype") != contiguous.dtype.str:
                    raise ValueError(f"{key} evidence dtype is inconsistent")
                if metadata.get("finite") is not True:
                    raise ValueError(f"{key} evidence does not attest finite values")
                if metadata.get("finite_count") != int(contiguous.size):
                    raise ValueError(f"{key} evidence finite count is inconsistent")
                if metadata.get("value_count") != int(contiguous.size):
                    raise ValueError(f"{key} evidence value count is inconsistent")
                if metadata.get("nondegenerate") is not True:
                    raise ValueError(f"{key} evidence does not attest nondegeneracy")
                if metadata.get("array_sha256") != array_hash:
                    raise ValueError(f"{key} raw-array SHA-256 is inconsistent")
                claimed_span = np.asarray(metadata.get("axis_span"), dtype=np.float64)
                if claimed_span.shape != (3,) or not np.allclose(
                    claimed_span, axis_span, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(f"{key} evidence axis span is inconsistent")
                results[camera] = {
                    "shape": list(contiguous.shape),
                    "dtype": contiguous.dtype.str,
                    "axis_span": [float(value) for value in axis_span],
                    "array_sha256": array_hash,
                }
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid point-cloud evidence {path}: {error}") from error
    return {"npz_sha256": evidence["npz_sha256"], "cameras": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--actor-log", type=Path, required=True)
    parser.add_argument("--point-cloud-evidence", type=Path, required=True)
    parser.add_argument("--task", default="place_cups")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    episodes = metrics.get(args.task)
    if not isinstance(episodes, dict) or set(episodes) != {str(args.episode)}:
        raise SystemExit(f"expected exactly {args.task} episode {args.episode} in metrics")
    record = episodes[str(args.episode)]
    if record.get("success") is not True:
        raise SystemExit("gated episode must have success=true")
    if not isinstance(record.get("episode_len"), int) or record["episode_len"] < 1:
        raise SystemExit("episode length is missing or invalid")
    if record.get("retry_times") != 0:
        raise SystemExit("gated episode must have retry_times=0")
    overall = metrics.get("overall", {})
    if set(overall) != {args.task, "avg_success_rate"}:
        raise SystemExit("overall metrics do not describe exactly the gated task")
    if overall[args.task] != 1.0 or overall["avg_success_rate"] != 1.0:
        raise SystemExit("overall gate metrics must report success rate 1.0")

    episode_dir = args.metrics.parent / args.task / str(args.episode)
    statistic = episode_dir / "episode_statistic.json"
    if not statistic.is_file():
        raise SystemExit(f"episode statistic is missing: {statistic}")
    json.loads(statistic.read_text(encoding="utf-8"))
    success_markers = list(episode_dir.glob("success_step*"))
    failure_markers = list(episode_dir.glob("failure_step*"))
    expected_marker = episode_dir / f"success_step{record['episode_len']}"
    if success_markers != [expected_marker] or failure_markers:
        raise SystemExit(
            "expected exactly the success marker matching episode_len and no failure marker"
        )
    gif_evidence = {}
    for camera in CAMERAS:
        gif = episode_dir / f"{camera}_rgb.gif"
        if not gif.is_file() or gif.stat().st_size == 0:
            raise SystemExit(f"camera GIF is missing or empty: {gif}")
        gif_evidence[camera] = validate_gif(gif)

    point_cloud_evidence = validate_point_cloud_evidence(
        args.point_cloud_evidence, args.task, args.episode
    )

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
                "camera_gifs": gif_evidence,
                "point_clouds": point_cloud_evidence,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
