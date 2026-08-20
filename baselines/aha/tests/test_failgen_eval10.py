from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "failgen_eval10.py"
SPEC = importlib.util.spec_from_file_location("failgen_eval10", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixed_args(**overrides):
    values = {
        "max_tries": 1,
        "max_restarts": 1,
        "total_timeout_seconds": 7200,
        "task_timeout_seconds": 600,
        "attempt_timeout_seconds": 300,
        "display_min": 120,
        "display_max": 199,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fixed_protocol_limits_are_enforced():
    MODULE.validate_args(fixed_args())
    with pytest.raises(ValueError, match="max-tries 1"):
        MODULE.validate_args(fixed_args(max_tries=2))
    with pytest.raises(ValueError, match="max-restarts 1"):
        MODULE.validate_args(fixed_args(max_restarts=2))
    with pytest.raises(ValueError, match="7200"):
        MODULE.validate_args(fixed_args(total_timeout_seconds=7201))


def test_official_eval_order_is_exact(tmp_path):
    body = "\n".join(f'        "{task}"' for task in MODULE.OFFICIAL_EVAL_TASKS)
    script = tmp_path / "eval.sh"
    script.write_text(f"tasks=(\n{body}\n)\n", encoding="utf-8")
    MODULE.verify_official_tasks(script)

    script.write_text(f"tasks=(\n{body}\n        \"extra_task\"\n)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        MODULE.verify_official_tasks(script)


def test_png_integrity_requires_complete_equal_camera_streams(tmp_path):
    attempt = tmp_path / "attempt_1"
    artifacts = attempt / "artifacts" / "episode"
    artifacts.mkdir(parents=True)
    for camera in MODULE.CAMERAS:
        for index in range(2):
            Image.new("RGB", (8, 6), color=(index, 2, 3)).save(
                artifacts / f"{camera}_{index}.png"
            )

    result = MODULE.verify_pngs(artifacts, attempt)
    assert result["status"] == "complete"
    assert result["png_count"] == 6
    assert result["camera_counts"] == {
        "front": 2,
        "overhead": 2,
        "wrist": 2,
    }
    assert all(len(item["sha256"]) == 64 for item in result["files"])

    (artifacts / "wrist_1.png").unlink()
    with pytest.raises(ValueError, match="counts differ"):
        MODULE.verify_pngs(artifacts, attempt)
