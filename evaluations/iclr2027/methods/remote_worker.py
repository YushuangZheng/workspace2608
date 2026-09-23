"""JSON-lines worker for a method-scoped PyTorch runtime monitor."""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext


@dataclass(frozen=True)
class _ConstantSchedule:
    value: float

    def threshold(self, _index: int) -> float:
        return self.value


def _decode_mapping(raw: str) -> dict[str, Any]:
    mapping = json.loads(raw)
    schedule = mapping.get("_threshold_schedule")
    if isinstance(schedule, Mapping):
        if schedule.get("kind") == "time_varying":
            from evaluations.iclr2027.methods.fail_detect.conformal import (
                TimeVaryingConformalBand,
            )

            mapping["_threshold_schedule"] = TimeVaryingConformalBand.from_dict(
                schedule["payload"]
            )
        elif schedule.get("kind") == "constant":
            mapping["_threshold_schedule"] = _ConstantSchedule(
                float(schedule["value"])
            )
        else:
            raise ValueError("unsupported remote threshold schedule")
    return mapping


def _monitor(mapping: Mapping[str, Any]):
    module_name, attribute = str(mapping["factory"]).split(":", 1)
    constructor = getattr(importlib.import_module(module_name), attribute)
    return constructor.from_mapping(mapping)


def _write(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(value), separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        monitor = _monitor(_decode_mapping(sys.argv[1]))
        _write(
            {
                "status": "ready",
                "checkpoint_hash": getattr(monitor, "checkpoint_hash", None),
            }
        )
        for line in sys.stdin:
            request = json.loads(line)
            operation = request["operation"]
            if operation == "close":
                _write({"status": "ok"})
                return 0
            if operation == "reset":
                monitor.reset(EpisodeContext(**request["context"]))
                _write({"status": "ok"})
                continue
            if operation == "observe":
                monitor.observe(
                    request["observation"],
                    request["action"],
                    request["policy_state"],
                )
                _write({"status": "ok", "output": monitor.cycle_output()})
                continue
            raise ValueError(f"unsupported monitor worker operation: {operation}")
        return 0
    except Exception:
        _write({"status": "error", "error": traceback.format_exc()})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
