"""GUI 补采帧读取和完整性校验。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from essay2608.data.robodojo import _load_captured_frames


def test_capture_frames_are_loaded_as_time_series(tmp_path) -> None:
    root = tmp_path / "captured"
    path = root / "push_T" / "episode_0000000.jsonl"
    path.parent.mkdir(parents=True)
    records = [{"type": "metadata", "schema": "essay2608.robodojo.gui_capture.v1"}]
    for index in range(3):
        records.append(
            {
                "type": "step",
                "index": index,
                "frames": {
                    "t": [index, 0, 0, 1, 0, 0, 0],
                    "target_t": [0, 0, 0, 1, 0, 0, 0],
                },
            }
        )
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    frames, metadata = _load_captured_frames(root, "push_T", 0, 3)

    assert metadata["source"] == "gui_capture"
    assert frames is not None
    assert frames["t"].shape == (3, 7)
    assert np.allclose(frames["t"][2, :3], [2, 0, 0])


def test_capture_frames_reject_empty_frame_set(tmp_path) -> None:
    root = tmp_path / "captured"
    path = root / "push_T" / "episode_0000000.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            (
                json.dumps({"type": "metadata"}),
                json.dumps({"type": "step", "index": 0, "frames": {}}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="没有任务真值帧"):
        _load_captured_frames(root, "push_T", 0, 1)


def test_capture_frames_reject_failed_training_audit(tmp_path) -> None:
    root = tmp_path / "captured"
    path = root / "push_T" / "episode_0000000.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            (
                json.dumps({"type": "metadata"}),
                json.dumps(
                    {
                        "type": "step",
                        "index": 0,
                        "frames": {"t": [0, 0, 0, 1, 0, 0, 0]},
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".audit.json").write_text(
        json.dumps({"accepted_for_training": False}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="未通过训练门禁"):
        _load_captured_frames(root, "push_T", 0, 1)
