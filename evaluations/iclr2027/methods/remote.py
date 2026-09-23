"""Run a PyTorch runtime monitor in the frozen policy Python environment."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.interfaces.runtime_monitor import (
    EpisodeContext,
    RuntimeMonitor,
)


def _serializable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(mapping)
    schedule = result.get("_threshold_schedule")
    if schedule is not None:
        if hasattr(schedule, "to_dict"):
            result["_threshold_schedule"] = {
                "kind": "time_varying",
                "payload": schedule.to_dict(),
            }
        elif hasattr(schedule, "value"):
            result["_threshold_schedule"] = {
                "kind": "constant",
                "value": float(schedule.value),
            }
        else:
            raise TypeError("unsupported remote threshold schedule")
    return result


class RemoteRuntimeMonitor(RuntimeMonitor):
    """Strict request/response proxy; evaluator-only fields never cross it."""

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        python = Path(os.environ["DYNAMAC_POLICY_PYTHON"])
        self._process = subprocess.Popen(
            [
                str(python),
                "-m",
                "evaluations.iclr2027.methods.remote_worker",
                json.dumps(_serializable_mapping(mapping), separators=(",", ":")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"monitor worker failed to initialize: {ready}")
        self.checkpoint_hash = ready.get("checkpoint_hash")
        self._output = None

    def _read(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("monitor worker stdout is unavailable")
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"monitor worker exited unexpectedly: {code}")
        value = json.loads(line)
        if value.get("status") == "error":
            raise RuntimeError(value.get("error", "monitor worker error"))
        return value

    def _request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if self._process.stdin is None:
            raise RuntimeError("monitor worker stdin is unavailable")
        self._process.stdin.write(
            json.dumps({"operation": operation, **payload}, separators=(",", ":"))
            + "\n"
        )
        self._process.stdin.flush()
        return self._read()

    def reset(self, episode_context: EpisodeContext) -> None:
        response = self._request("reset", context=asdict(episode_context))
        if response.get("status") != "ok":
            raise RuntimeError("monitor worker rejected reset")
        self._output = None

    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        response = self._request(
            "observe",
            observation=dict(observation),
            action=dict(action),
            policy_state=dict(policy_state),
        )
        self._output = response["output"]

    def score(self) -> Mapping[str, float]:
        if self._output is None:
            raise RuntimeError("no observed cycle")
        return dict(self._output["scores"])

    def alarm(self) -> bool:
        return False if self._output is None else bool(self._output["alarm"])

    @property
    def threshold(self) -> float | None:
        return None if self._output is None else self._output["threshold"]

    @property
    def persistence_count(self) -> int:
        return 0 if self._output is None else int(self._output["persistence_count"])

    @property
    def output_metadata(self) -> Mapping[str, Any]:
        return {} if self._output is None else dict(self._output["metadata"])

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                self._request("close")
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["RemoteRuntimeMonitor"]
