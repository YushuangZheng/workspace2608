from __future__ import annotations

import argparse
import importlib.util
import subprocess
import time
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "failgen_eval10.py"
SPEC = importlib.util.spec_from_file_location("failgen_eval10", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GATE_SCRIPT = Path(__file__).parents[1] / "scripts" / "tmux_gate.py"
GATE_SPEC = importlib.util.spec_from_file_location("tmux_gate", GATE_SCRIPT)
assert GATE_SPEC is not None and GATE_SPEC.loader is not None
GATE = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(GATE)


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


def test_tmux_gate_ignores_dead_remain_on_exit_panes(tmp_path):
    (tmp_path / "111").mkdir()
    (tmp_path / "444").mkdir()
    rows = (
        "dynamac_spr_full_20260821\t1\t222",
        "dynamac_guardian_full_20260821\t0\t111",
        "dynamac_spr_stale_pid\t0\t333",
        "unrelated_session\t0\t444",
    )
    assert GATE.live_blocking_sessions(rows, proc_root=tmp_path) == (
        "dynamac_guardian_full_20260821",
    )


def task_args(tmp_path):
    return fixed_args(
        output_root=tmp_path,
        task_timeout_seconds=60,
        attempt_timeout_seconds=30,
    )


@pytest.mark.parametrize(
    "failure_class",
    (
        "failgen_not_produced",
        "released_configuration",
        "image_integrity",
        "timeout",
        "xvfb_or_runner_startup",
        "simulator_or_worker_error",
    ),
)
def test_non_retryable_failure_runs_once(monkeypatch, tmp_path, failure_class):
    calls = []

    def fake_run_worker(args, task, attempt_dir, timeout_seconds):
        calls.append(attempt_dir)
        if failure_class == "image_integrity":
            return {
                "status": "worker_success",
                "failure_class": None,
                "worker": {
                    "artifact_dir": str(attempt_dir / "artifacts"),
                    "failure_type": "grasp",
                    "waypoint": 1,
                    "renderer": "opengl3",
                },
            }
        return {"status": "failed", "failure_class": failure_class}

    monkeypatch.setattr(MODULE, "run_worker", fake_run_worker)
    if failure_class == "image_integrity":
        monkeypatch.setattr(
            MODULE,
            "verify_pngs",
            lambda *unused: (_ for _ in ()).throw(ValueError("bad png")),
        )

    result = MODULE.run_task(
        task_args(tmp_path), "stack_chairs", 1, time.monotonic() + 60
    )
    assert len(calls) == 1
    assert result["attempts_used"] == 1
    assert result["coppelia_restart_count"] == 0
    assert result["failure_class"] == failure_class


@pytest.mark.parametrize("retryable", ("worker_signal", "simulator_crash"))
def test_only_explicit_crashes_get_one_restart(
    monkeypatch, tmp_path, retryable
):
    calls = []

    def fake_run_worker(args, task, attempt_dir, timeout_seconds):
        calls.append(attempt_dir)
        if len(calls) == 1:
            return {"status": "failed", "failure_class": retryable}
        return {
            "status": "worker_success",
            "failure_class": None,
            "worker": {
                "artifact_dir": str(attempt_dir / "artifacts"),
                "failure_type": "grasp",
                "waypoint": 1,
                "renderer": "opengl3",
            },
        }

    monkeypatch.setattr(MODULE, "run_worker", fake_run_worker)
    monkeypatch.setattr(
        MODULE,
        "verify_pngs",
        lambda *unused: {
            "status": "complete",
            "png_count": 3,
            "camera_counts": {"front": 1, "overhead": 1, "wrist": 1},
            "files": [],
        },
    )

    result = MODULE.run_task(
        task_args(tmp_path), "stack_chairs", 1, time.monotonic() + 60
    )
    assert len(calls) == 2
    assert result["status"] == "success"
    assert result["attempts_used"] == 2
    assert result["coppelia_restart_count"] == 1


def test_worker_failure_classification_is_conservative():
    assert MODULE.classify_worker_failure(-11, {}) == "worker_signal"
    assert (
        MODULE.classify_worker_failure(
            1, {"error": "Remote API connection to CoppeliaSim was lost"}
        )
        == "simulator_crash"
    )
    assert (
        MODULE.classify_worker_failure(1, {"error": "ordinary Python error"})
        == "simulator_or_worker_error"
    )


@pytest.mark.parametrize(
    ("renderer", "expected"),
    (
        ("llvmpipe (LLVM 15.0.7, 256 bits)", True),
        ("NVIDIA GeForce RTX 4090/PCIe/SSE2", False),
    ),
)
def test_graphics_probe_requires_llvmpipe(
    monkeypatch, tmp_path, renderer, expected
):
    completed = subprocess.CompletedProcess(
        args=["glxinfo", "-B"],
        returncode=0,
        stdout=f"OpenGL renderer string: {renderer}\n",
        stderr="original stderr\n",
    )
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: completed)
    raw_output = tmp_path / "glxinfo_B.txt"

    result = MODULE.probe_graphics_renderer({}, raw_output, 5)

    assert result["renderer"] == renderer
    assert result["verified_llvmpipe"] is expected
    assert result["stdout"] == completed.stdout
    assert result["stderr"] == completed.stderr
    assert completed.stdout in raw_output.read_text(encoding="utf-8")
    assert completed.stderr in raw_output.read_text(encoding="utf-8")


def test_worker_never_starts_when_llvmpipe_probe_fails(monkeypatch, tmp_path):
    class FinishedXvfb:
        pid = 999

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(
        MODULE,
        "launch_xvfb",
        lambda *unused: (FinishedXvfb(), 120, 12345),
    )
    monkeypatch.setattr(MODULE, "worker_environment", lambda *unused: {})
    monkeypatch.setattr(
        MODULE,
        "probe_graphics_renderer",
        lambda *unused: {
            "return_code": 0,
            "renderer": "NVIDIA GeForce RTX 4090/PCIe/SSE2",
            "verified_llvmpipe": False,
            "stdout": "gpu renderer",
            "stderr": "",
        },
    )

    def forbidden_worker_start(*args, **kwargs):
        raise AssertionError("worker must not start without llvmpipe")

    monkeypatch.setattr(MODULE.subprocess, "Popen", forbidden_worker_start)
    result = MODULE.run_worker(
        argparse.Namespace(), "stack_chairs", tmp_path / "attempt_1", 30
    )
    assert result["status"] == "failed"
    assert result["failure_class"] == "graphics_probe_failed"
    assert result["graphics_probe"]["verified_llvmpipe"] is False


def test_summary_uses_actual_cli_deadline(tmp_path):
    summary = MODULE.write_summary(tmp_path, [], "start", deadline_seconds=123)
    assert summary["deadline_seconds"] == 123
    assert summary["total_timeout_seconds"] == 123
