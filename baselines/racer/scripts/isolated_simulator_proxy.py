#!/usr/bin/env python3
"""Drop-in RLBenchSim proxy whose simulator lives in a clean subprocess."""

from __future__ import annotations

import atexit
import os
import secrets
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any


class IsolatedSimulatorError(RuntimeError):
    pass


class SpawnIsolatedRLBenchSim:
    """Preserve RACER's simulator API while isolating native renderer state."""

    def __init__(
        self,
        task_name: str,
        dataset_root: str,
        episode_length: int = 30,
        record_every_n: int = 5,
        resolution: int = 512,
        record_queue=None,
        never_terminal: bool = False,
        unseen_task: bool = False,
    ) -> None:
        if record_queue is not None:
            raise IsolatedSimulatorError("record_queue is unsupported across the isolation boundary")
        runtime_value = os.environ.get("RACER_ISOLATION_RUNTIME_DIR")
        if not runtime_value:
            raise IsolatedSimulatorError("RACER_ISOLATION_RUNTIME_DIR is required")
        self.runtime_dir = Path(runtime_value).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.socket_path = Path(
            f"/tmp/racer-isolated-{os.getuid()}-{os.getpid()}-{secrets.token_hex(6)}.sock"
        )
        self.authkey = secrets.token_bytes(32)
        self.connection = None
        self.worker_returncode = None
        self.closed = False
        self._timeout = float(os.environ.get("RACER_ISOLATION_RPC_TIMEOUT", "300"))
        if not (self._timeout > 0):
            raise IsolatedSimulatorError("RACER_ISOLATION_RPC_TIMEOUT must be positive")

        worker = Path(__file__).with_name("isolated_simulator_worker.py")
        command = [
            sys.executable,
            "-u",
            str(worker),
            "--socket",
            str(self.socket_path),
            "--authkey-hex",
            self.authkey.hex(),
        ]
        if os.environ.get("RACER_ISOLATION_WORKER_SELF_TEST") == "1":
            command.append("--self-test")
        self.worker_log_path = self.runtime_dir / f"simulator_worker_{os.getpid()}.log"
        self._worker_log = self.worker_log_path.open("ab", buffering=0)
        self.worker = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._worker_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        atexit.register(self._best_effort_cleanup)
        try:
            self._connect()
            self._request(
                "init",
                {
                    "task_name": task_name,
                    "dataset_root": dataset_root,
                    "episode_length": episode_length,
                    "record_every_n": record_every_n,
                    "resolution": resolution,
                    "record_queue": None,
                    "never_terminal": never_terminal,
                    "unseen_task": unseen_task,
                },
            )
        except BaseException:
            self._best_effort_cleanup()
            raise

    def _connect(self) -> None:
        deadline = time.monotonic() + min(self._timeout, 60.0)
        last_error = None
        while time.monotonic() < deadline:
            returncode = self.worker.poll()
            if returncode is not None:
                self.worker_returncode = returncode
                raise IsolatedSimulatorError(
                    f"simulator worker exited during startup with status {returncode}; "
                    f"see {self.worker_log_path}"
                )
            try:
                self.connection = Client(
                    str(self.socket_path), family="AF_UNIX", authkey=self.authkey
                )
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError) as error:
                last_error = error
                time.sleep(0.05)
        raise IsolatedSimulatorError(
            f"simulator worker connection timed out: {last_error}; see {self.worker_log_path}"
        )

    def _request(self, operation: str, arguments: dict[str, Any] | None = None):
        if self.closed or self.connection is None:
            raise IsolatedSimulatorError("simulator proxy is closed")
        self.connection.send({"operation": operation, "arguments": arguments or {}})
        if not self.connection.poll(self._timeout):
            raise IsolatedSimulatorError(
                f"simulator operation {operation!r} timed out; see {self.worker_log_path}"
            )
        try:
            response = self.connection.recv()
        except EOFError as error:
            raise IsolatedSimulatorError(
                f"simulator worker disconnected during {operation!r}; see {self.worker_log_path}"
            ) from error
        if not response.get("ok"):
            raise IsolatedSimulatorError(
                f"simulator operation {operation!r} failed with "
                f"{response.get('error_type')}: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("value")

    def reset(self, episode_num: int = 0, not_load_image: bool = True):
        return self._request(
            "reset", {"episode_num": episode_num, "not_load_image": not_load_image}
        )

    def set_new_task(self, task_name: str) -> None:
        self._request("set_new_task", {"task_name": task_name})

    def set_new_dataset(self, dataset_root: str) -> None:
        self._request("set_new_dataset", {"dataset_root": dataset_root})

    @property
    def task_goal(self):
        return self._request("task_goal")

    def step(self, action):
        result = self._request("step", {"action": action})
        mutated_action = result["action"]
        try:
            action[...] = mutated_action
        except (TypeError, IndexError):
            action[:] = mutated_action
        return result["transition"]

    def is_success(self) -> bool:
        return bool(self._request("is_success"))

    def get_video_frames(self, res: int = 128, return_pil: bool = True):
        return self._request(
            "get_video_frames", {"res": res, "return_pil": return_pil}
        )

    def close(self) -> None:
        if self.closed:
            return
        request_error = None
        try:
            self._request("close")
        except BaseException as error:
            request_error = error
        finally:
            self.closed = True
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        exit_error = None
        try:
            self.worker_returncode = self.worker.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            exit_error = error
            self.worker.terminate()
            try:
                self.worker_returncode = self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker_returncode = self.worker.wait(timeout=5)
        finally:
            self._worker_log.close()
            self.socket_path.unlink(missing_ok=True)
            atexit.unregister(self._best_effort_cleanup)
        if exit_error is not None:
            raise IsolatedSimulatorError(
                f"simulator worker did not exit naturally; see {self.worker_log_path}"
            ) from exit_error
        if request_error is not None:
            raise request_error
        if self.worker_returncode != 0:
            raise IsolatedSimulatorError(
                f"simulator worker exited with status {self.worker_returncode}; "
                f"see {self.worker_log_path}"
            )

    def _best_effort_cleanup(self) -> None:
        if getattr(self, "closed", True):
            return
        self.closed = True
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None
        worker = getattr(self, "worker", None)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        self.worker_returncode = worker.returncode if worker is not None else None
        worker_log = getattr(self, "_worker_log", None)
        if worker_log is not None and not worker_log.closed:
            worker_log.close()
        socket_path = getattr(self, "socket_path", None)
        if socket_path is not None:
            socket_path.unlink(missing_ok=True)
