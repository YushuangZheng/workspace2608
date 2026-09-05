"""Pure shadow-mode adapter: score causal records without action authority."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from evaluations.iclr2027.interfaces.runtime_monitor import RuntimeMonitor


def shadow_observe(
    monitor: RuntimeMonitor,
    feature: Mapping[str, Any],
) -> dict[str, Any]:
    """Update a monitor and return its diagnostic output only."""

    monitor.observe(
        {
            "arms": deepcopy(feature["arms"]),
            "task_state": deepcopy(feature["task_state"]),
            "observation_timestamp": int(feature["observation_timestamp"]),
            "previous_action_resolution": deepcopy(feature["action_resolution"]),
        },
        {
            "action": deepcopy(feature["action"]),
            "action_timestamp": int(feature["action_timestamp"]),
            "cycle": int(feature["cycle"]),
        },
        deepcopy(feature["policy_state"]),
    )
    scores = {str(name): float(value) for name, value in monitor.score().items()}
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("monitor returned a non-finite score")
    threshold = getattr(monitor, "threshold", None)
    if threshold is not None:
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError("monitor returned a non-finite threshold")
    persistence = getattr(
        monitor,
        "persistence_count",
        getattr(monitor, "_streak", int(bool(monitor.alarm()))),
    )
    metadata = getattr(monitor, "output_metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("monitor output metadata must be a mapping")
    return {
        "cycle": int(feature["cycle"]),
        "scores": scores,
        "alarm": bool(monitor.alarm()),
        "threshold": threshold,
        "persistence_count": int(persistence),
        "metadata": dict(metadata),
    }


def shadow_passthrough_action(
    monitor: RuntimeMonitor,
    feature: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    """Return the byte-equivalent command and separate shadow diagnostics."""

    command = list(feature["action"])
    diagnostic = shadow_observe(monitor, feature)
    if command != list(feature["action"]):
        raise RuntimeError("shadow monitor mutated the action")
    return command, diagnostic


__all__ = ["shadow_observe", "shadow_passthrough_action"]
