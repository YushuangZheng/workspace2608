#!/usr/bin/env python3
"""Record the four raw gate-reset point clouds without changing policy input."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
EXPECTED_SHAPE = (3, 512, 512)
MIN_AXIS_SPAN = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_point_cloud(value, *, key: str):
    array = np.asarray(value)
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(f"{key} shape {array.shape} != {EXPECTED_SHAPE}")
    if array.dtype != np.float32:
        raise ValueError(f"{key} dtype {array.dtype} != float32")
    contiguous = np.ascontiguousarray(array)
    finite = np.isfinite(contiguous)
    if not finite.all():
        raise ValueError(f"{key} contains non-finite values")
    flattened = contiguous.reshape(3, -1)
    axis_min = flattened.min(axis=1)
    axis_max = flattened.max(axis=1)
    axis_span = axis_max - axis_min
    if not np.all(axis_span > MIN_AXIS_SPAN):
        raise ValueError(f"{key} is degenerate; axis spans={axis_span.tolist()}")
    return contiguous, {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "finite": True,
        "finite_count": int(finite.sum()),
        "value_count": int(contiguous.size),
        "axis_min": [float(value) for value in axis_min],
        "axis_max": [float(value) for value in axis_max],
        "axis_span": [float(value) for value in axis_span],
        "nondegenerate": True,
        "array_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def write_point_cloud_evidence(
    obs_dict, *, output: Path, task_name: str, episode_num: int
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output.with_name("gate_point_clouds.npz")
    if output.exists() or npz_path.exists():
        raise FileExistsError(f"gate point-cloud evidence already exists under {output.parent}")

    arrays = {}
    cameras = {}
    for camera in CAMERAS:
        key = f"{camera}_point_cloud"
        if key not in obs_dict:
            raise KeyError(f"reset observation is missing {key}")
        array, metadata = analyze_point_cloud(obs_dict[key], key=key)
        arrays[key] = array
        cameras[camera] = {"key": key, **metadata}

    temporary_npz = npz_path.with_name(f".{npz_path.name}.tmp.{os.getpid()}")
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary_npz, npz_path)

    payload = {
        "schema": "racer_gate_point_cloud_evidence_v1",
        "source": "RLBenchSim.reset obs_dict before policy preprocessing",
        "task": task_name,
        "episode": episode_num,
        "npz_file": npz_path.name,
        "npz_sha256": _sha256(npz_path),
        "cameras": cameras,
    }
    temporary_json = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, output)


def evidence_recording_simulator(base_class):
    """Return a transparent simulator subclass that records only its first reset."""

    class EvidenceRecordingSimulator(base_class):
        def __init__(self, *args, **kwargs):
            if "task_name" in kwargs:
                self._evidence_task_name = kwargs["task_name"]
            elif args:
                self._evidence_task_name = args[0]
            else:
                raise ValueError("task_name is required for gate evidence")
            self._evidence_written = False
            super().__init__(*args, **kwargs)

        def set_new_task(self, task_name: str):
            result = super().set_new_task(task_name)
            self._evidence_task_name = task_name
            return result

        def reset(self, episode_num: int = 0, not_load_image: bool = True):
            result = super().reset(episode_num, not_load_image)
            evidence_value = os.environ.get("RACER_POINT_CLOUD_EVIDENCE")
            if evidence_value and not self._evidence_written:
                obs_dict, _observation = result
                write_point_cloud_evidence(
                    obs_dict,
                    output=Path(evidence_value),
                    task_name=self._evidence_task_name,
                    episode_num=episode_num,
                )
                self._evidence_written = True
            return result

    EvidenceRecordingSimulator.__name__ = f"EvidenceRecording{base_class.__name__}"
    return EvidenceRecordingSimulator
