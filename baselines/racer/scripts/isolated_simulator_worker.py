#!/usr/bin/env python3
"""Run one RACER simulator behind a small authenticated local RPC channel."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any


UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "upstream"


class _SelfTestSimulator:
    """Dependency-free simulator used only by the protocol regression test."""

    def __init__(self, task_name: str, dataset_root: str, **_: Any) -> None:
        self.task_name = task_name
        self.dataset_root = dataset_root
        self.transition = None

    def reset(self, episode_num: int = 0, not_load_image: bool = True):
        observation = {
            "episode": episode_num,
            "not_load_image": not_load_image,
            "task": self.task_name,
        }
        return observation, {"raw": observation.copy()}

    def set_new_task(self, task_name: str) -> None:
        self.task_name = task_name

    def set_new_dataset(self, dataset_root: str) -> None:
        self.dataset_root = dataset_root

    @property
    def task_goal(self) -> str:
        return f"goal:{self.task_name}"

    def step(self, action):
        action[0] = action[0] + 1
        self.transition = {
            "action": action,
            "dataset_root": self.dataset_root,
            "terminal": True,
        }
        return self.transition

    def is_success(self) -> bool:
        return self.transition is not None

    def get_video_frames(self, res: int = 128, return_pil: bool = True):
        return [{"res": res, "return_pil": return_pil}]

    def close(self) -> None:
        return None


def _load_simulator(self_test: bool):
    if self_test:
        return _SelfTestSimulator
    # This import intentionally occurs only in the clean worker process. The
    # policy/PyTorch3D stack remains in the evaluator parent process.
    sys.path.insert(0, str(UPSTREAM_ROOT))
    from racer.evaluation.simulator import RLBenchSim

    return RLBenchSim


def _reply(connection, *, value=None, error: BaseException | None = None) -> None:
    if error is None:
        connection.send({"ok": True, "value": value})
        return
    connection.send(
        {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    )


def serve(socket_path: Path, authkey: bytes, self_test: bool) -> int:
    listener = None
    connection = None
    simulator = None
    exit_status = 0
    try:
        socket_path.unlink(missing_ok=True)
        listener = Listener(str(socket_path), family="AF_UNIX", authkey=authkey)
        connection = listener.accept()
        simulator_class = _load_simulator(self_test)

        while True:
            request = connection.recv()
            operation = request.get("operation")
            arguments = request.get("arguments", {})
            try:
                if operation == "init":
                    if simulator is not None:
                        raise RuntimeError("simulator was already initialized")
                    simulator = simulator_class(**arguments)
                    _reply(connection, value=None)
                elif operation == "close":
                    if simulator is not None:
                        simulator.close()
                        simulator = None
                    _reply(connection, value=None)
                    break
                else:
                    if simulator is None:
                        raise RuntimeError("simulator is not initialized")
                    if operation == "reset":
                        value = simulator.reset(**arguments)
                    elif operation == "set_new_task":
                        value = simulator.set_new_task(**arguments)
                    elif operation == "set_new_dataset":
                        value = simulator.set_new_dataset(**arguments)
                    elif operation == "task_goal":
                        value = simulator.task_goal
                    elif operation == "step":
                        action = arguments["action"]
                        transition = simulator.step(action)
                        value = {"transition": transition, "action": action}
                    elif operation == "is_success":
                        value = simulator.is_success()
                    elif operation == "get_video_frames":
                        value = simulator.get_video_frames(**arguments)
                    else:
                        raise ValueError(f"unsupported operation: {operation!r}")
                    _reply(connection, value=value)
            except BaseException as error:  # Send diagnostics, then fail closed.
                _reply(connection, error=error)
                exit_status = 1
                break
    except EOFError:
        exit_status = 1
    except BaseException:
        traceback.print_exc()
        exit_status = 1
    finally:
        if simulator is not None:
            try:
                simulator.close()
            except BaseException:
                traceback.print_exc()
                exit_status = 1
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        socket_path.unlink(missing_ok=True)
    return exit_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authkey-hex", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if not args.socket.is_absolute() or args.socket.parent != Path("/tmp"):
        raise SystemExit("worker socket must be an absolute path directly under /tmp")
    try:
        authkey = bytes.fromhex(args.authkey_hex)
    except ValueError as error:
        raise SystemExit(f"invalid authentication key: {error}") from error
    if len(authkey) < 16:
        raise SystemExit("authentication key is too short")
    return serve(args.socket, authkey, args.self_test)


if __name__ == "__main__":
    sys.exit(main())
