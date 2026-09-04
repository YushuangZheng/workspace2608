# ruff: noqa: UP045
"""Resume-safe server-B supervisor for live RVT and RACER reproduction gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional, TextIO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait-for",
        type=Path,
        default=Path(
            "/home/ubuntu/workspace/_runs/fail_detect/"
            "square_flow_seed1103_full_20260904/final_reproduction_result.json"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/ubuntu/workspace/_runs/native_systems/live_close_jar_20260904"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/ubuntu/workspace/essay2608"),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--skip-rich-vlm", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(
    command: list[str],
    *,
    cwd: Path,
    log: Path,
    env: dict[str, str],
) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND {json.dumps(command)}\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with {completed.returncode}: {command}")


def _start(
    command: list[str],
    *,
    cwd: Path,
    log: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[bytes], TextIO]:
    stream = log.open("a", encoding="utf-8")
    stream.write(f"COMMAND {json.dumps(command)}\n")
    stream.flush()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    return process, stream


def _wait_http(
    url: str, process: subprocess.Popen[bytes], timeout_seconds: int
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited with code {process.returncode}: {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"service did not become ready: {url}")


def _stop(
    process: Optional[subprocess.Popen[bytes]], stream: Optional[TextIO]
) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    if stream is not None:
        stream.close()


def _simulator_env(base: dict[str, str], conda_lib: Path) -> dict[str, str]:
    coppelia = Path(
        "/home/ubuntu/workspace/essay2608/"
        "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    )
    env = dict(base)
    env.update(
        {
            "COPPELIASIM_ROOT": str(coppelia),
            "QT_PLUGIN_PATH": str(coppelia),
            "QT_QPA_PLATFORM": "xcb",
            "QT_XCB_GL_INTEGRATION": "xcb_glx",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "MESA_GL_VERSION_OVERRIDE": "3.3",
            "LD_LIBRARY_PATH": ":".join(
                [
                    str(coppelia),
                    str(conda_lib),
                    base.get("LD_LIBRARY_PATH", ""),
                ]
            ),
        }
    )
    return env


def _xvfb(command: list[str]) -> list[str]:
    return [
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1280x1024x24 +extension GLX +render",
        *command,
    ]


def _prepare_llava_adapter(target: Path) -> Path:
    source = Path("/home/ubuntu/workspace/_models/racer/llava-lora-rich")
    base = Path("/home/ubuntu/workspace/_models/racer/llama3-llava-next-8b")
    vision = Path(
        "/home/ubuntu/workspace/_models/racer/clip-vit-large-patch14-336"
    )
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        output = target / item.name
        if item.name == "config.json":
            config = json.loads(item.read_text(encoding="utf-8"))
            config["mm_vision_tower"] = str(vision)
            _write_json(output, config)
        elif item.name == "adapter_config.json":
            config = json.loads(item.read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = str(base)
            _write_json(output, config)
        elif not output.exists():
            output.symlink_to(item.resolve())
    return target


def main() -> None:
    args = _parse_args()
    wait_for = args.wait_for.resolve()
    run_dir = args.run_dir.resolve()
    project_root = args.project_root.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    supervisor_log = run_dir / "supervisor.log"

    def status(stage: str, state: str, **extra: Any) -> None:
        payload = {"stage": stage, "state": state, "updated_unix": time.time(), **extra}
        _write_json(status_path, payload)
        with supervisor_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    status("wait_for_fail_detect", "waiting", prerequisite=str(wait_for))
    while not wait_for.is_file():
        time.sleep(args.poll_seconds)
    prerequisite = json.loads(wait_for.read_text(encoding="utf-8"))
    if prerequisite.get("status") != "pass":
        raise RuntimeError(f"FAIL-Detect prerequisite did not pass: {prerequisite}")
    status("wait_for_fail_detect", "complete")

    base_env = dict(os.environ)
    rvt_python = Path("/home/ubuntu/miniforge3/envs/rvt-official/bin/python")
    racer_python = Path("/home/ubuntu/miniforge3/envs/racer-official/bin/python")
    service_python = Path("/home/ubuntu/miniforge3/envs/racer-services/bin/python")
    rvt_result = run_dir / "rvt_close_jar_live.json"
    racer_goal_result = run_dir / "racer_close_jar_task_goal_live.json"
    racer_rich_result = run_dir / "racer_close_jar_rich_live.json"

    language_process: Optional[subprocess.Popen[bytes]] = None
    language_stream: Optional[TextIO] = None
    llava_process: Optional[subprocess.Popen[bytes]] = None
    llava_stream: Optional[TextIO] = None
    try:
        language_env = dict(base_env)
        language_env["CUDA_VISIBLE_DEVICES"] = "1"
        status("t5_service", "starting")
        language_process, language_stream = _start(
            [
                str(service_python),
                "-m",
                "evaluations.iclr2027.native_systems.racer.reproduction.language_server",
            ],
            cwd=project_root,
            log=run_dir / "t5_service.log",
            env=language_env,
        )

        if not rvt_result.is_file():
            status("rvt_live_close_jar", "running")
            rvt_env = _simulator_env(base_env, rvt_python.parent.parent / "lib")
            rvt_env["CUDA_VISIBLE_DEVICES"] = "0"
            _run(
                _xvfb(
                    [
                        str(rvt_python),
                        "-m",
                        "evaluations.iclr2027.native_systems.rvt.reproduction.live_nominal",
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--output",
                        str(rvt_result),
                    ]
                ),
                cwd=project_root,
                log=run_dir / "rvt_live_close_jar.log",
                env=rvt_env,
            )
        status("rvt_live_close_jar", "complete", result=str(rvt_result))

        _wait_http(
            "http://127.0.0.1:8000/health", language_process, timeout_seconds=1200
        )
        status("t5_service", "ready", pid=language_process.pid)

        if not args.skip_rich_vlm and not racer_rich_result.is_file():
            adapter = _prepare_llava_adapter(run_dir / "llava_adapter_local")
            llava_env = dict(base_env)
            llava_env["CUDA_VISIBLE_DEVICES"] = "2"
            status("llava_service", "starting")
            llava_process, llava_stream = _start(
                [
                    str(service_python),
                    "/home/ubuntu/workspace/_external/Open-LLaVA-NeXT/deploy/llava_server.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "21002",
                    "--model-path",
                    str(adapter),
                    "--model-base",
                    "/home/ubuntu/workspace/_models/racer/llama3-llava-next-8b",
                    "--model-name",
                    "llava_llama3_lora",
                    "--device",
                    "cuda",
                    "--limit-model-concurrency",
                    "1",
                ],
                cwd=Path("/home/ubuntu/workspace/_external/Open-LLaVA-NeXT"),
                log=run_dir / "llava_service.log",
                env=llava_env,
            )

        if not racer_goal_result.is_file():
            status("racer_task_goal_live_close_jar", "running")
            racer_env = _simulator_env(
                base_env, racer_python.parent.parent / "lib"
            )
            racer_env["CUDA_VISIBLE_DEVICES"] = "0"
            _run(
                _xvfb(
                    [
                        str(racer_python),
                        "-m",
                        "evaluations.iclr2027.native_systems.racer.reproduction.live_nominal",
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--output",
                        str(racer_goal_result),
                    ]
                ),
                cwd=project_root,
                log=run_dir / "racer_task_goal_live_close_jar.log",
                env=racer_env,
            )
        status(
            "racer_task_goal_live_close_jar",
            "complete",
            result=str(racer_goal_result),
        )

        if not args.skip_rich_vlm and not racer_rich_result.is_file():
            if llava_process is None:
                raise RuntimeError("LLaVA service was not started")
            _wait_http(
                "http://127.0.0.1:21002/test",
                llava_process,
                timeout_seconds=1200,
            )
            status("llava_service", "ready", pid=llava_process.pid)
            status("racer_rich_live_close_jar", "running")
            racer_env = _simulator_env(
                base_env, racer_python.parent.parent / "lib"
            )
            racer_env["CUDA_VISIBLE_DEVICES"] = "0"
            _run(
                _xvfb(
                    [
                        str(racer_python),
                        "-m",
                        "evaluations.iclr2027.native_systems.racer.reproduction.live_nominal",
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--use-vlm",
                        "--output",
                        str(racer_rich_result),
                    ]
                ),
                cwd=project_root,
                log=run_dir / "racer_rich_live_close_jar.log",
                env=racer_env,
            )
            status(
                "racer_rich_live_close_jar",
                "complete",
                result=str(racer_rich_result),
            )
        elif not args.skip_rich_vlm:
            status(
                "racer_rich_live_close_jar",
                "complete",
                result=str(racer_rich_result),
            )
    finally:
        _stop(llava_process, llava_stream)
        _stop(language_process, language_stream)

    results = {
        "rvt": json.loads(rvt_result.read_text(encoding="utf-8")),
        "racer_task_goal": json.loads(
            racer_goal_result.read_text(encoding="utf-8")
        ),
    }
    if not args.skip_rich_vlm:
        results["racer_rich"] = json.loads(
            racer_rich_result.read_text(encoding="utf-8")
        )
    final_path = run_dir / "final_result.json"
    _write_json(
        final_path,
        {
            "status": "pass",
            "scope": "server_b_live_nominal_native_system_reproduction",
            "formal_evaluation": False,
            "formal_evaluation_reason": "requires_server_a_frozen_native6_manifest",
            "results": results,
        },
    )
    status("complete", "pass", result=str(final_path))


if __name__ == "__main__":
    main()
