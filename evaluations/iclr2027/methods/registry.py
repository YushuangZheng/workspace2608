"""Frozen method identities and runtime factories for server A."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.interfaces.runtime_monitor import RuntimeMonitor

METHOD_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "methods"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    paper_name: str
    policy_type: str
    feature_profile: str | None
    monitor: Mapping[str, Any] | None
    recovery: Mapping[str, Any] | None
    runtime: Mapping[str, Any] | None
    config_path: Path
    config_sha256: str


def load_method_spec(method: str | Path) -> MethodSpec:
    path = Path(method)
    if path.suffix != ".json":
        path = METHOD_CONFIG_ROOT / f"{path}.json"
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "essay2608.iclr2027.method-config.v1":
        raise ValueError(f"unsupported method config: {path}")
    policy_type = str(payload["policy_type"])
    if policy_type not in {"dynamac", "closed_loop_multistream"}:
        raise ValueError(f"unsupported policy type: {policy_type}")
    profile = payload.get("feature_profile")
    if policy_type == "dynamac" and profile is not None:
        raise ValueError("DynaMAC method configs cannot select a closed-loop profile")
    return MethodSpec(
        method_id=str(payload["method_id"]),
        paper_name=str(payload["paper_name"]),
        policy_type=policy_type,
        feature_profile=None if profile is None else str(profile),
        monitor=payload.get("monitor"),
        recovery=payload.get("recovery"),
        runtime=payload.get("runtime"),
        config_path=path,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _monitor_mapping(
    spec: MethodSpec,
    calibration: Mapping[str, Any] | None,
    task_id: str | None,
) -> Mapping[str, Any] | None:
    if spec.monitor is None:
        return None
    value = dict(spec.monitor)
    if calibration is None:
        return value
    if task_id is None:
        raise ValueError("a task id is required with a calibration artifact")
    if calibration.get("schema") != "essay2608.iclr2027.monitor-calibration.v1":
        raise ValueError("unsupported monitor calibration artifact")
    if calibration.get("method_id") != spec.method_id:
        raise ValueError("calibration artifact names a different method")
    identity = calibration.get("method_config_identity", {})
    if identity.get("sha256") != spec.config_sha256:
        raise ValueError("calibration artifact method-config hash mismatch")
    entry = calibration.get("tasks", {}).get(task_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"calibration artifact has no threshold for {task_id}")
    value["threshold"] = float(entry["threshold"])
    value["persistence_cycles"] = int(entry["persistence_cycles"])
    return value


def _external_monitor(mapping: Mapping[str, Any]) -> RuntimeMonitor:
    factory = mapping.get("factory")
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("external monitor requires a module:factory entry")
    module_name, attribute = factory.split(":", 1)
    if not module_name.startswith("evaluations.iclr2027.methods."):
        raise ValueError("external monitor factory must live under the method tree")
    constructor = getattr(importlib.import_module(module_name), attribute)
    monitor = (
        constructor.from_mapping(mapping)
        if hasattr(constructor, "from_mapping")
        else constructor(mapping)
    )
    if not isinstance(monitor, RuntimeMonitor):
        raise TypeError("external monitor factory did not return RuntimeMonitor")
    return monitor


def build_monitor(
    spec: MethodSpec,
    *,
    calibration: Mapping[str, Any] | None = None,
    task_id: str | None = None,
) -> RuntimeMonitor | None:
    mapping = _monitor_mapping(spec, calibration, task_id)
    if mapping is None:
        return None
    kind = mapping.get("kind")
    if kind == "no_progress":
        from .restart.monitor import NoProgressMonitor

        return NoProgressMonitor(
            consecutive_stopped_cycles=int(
                mapping["consecutive_stopped_cycles"]
            ),
            minimum_command_distance_m=float(
                mapping.get("minimum_command_distance_m", 0.005)
            ),
            maximum_realized_motion_m=float(
                mapping.get("maximum_realized_motion_m", 0.001)
            ),
        )
    if kind == "trajectory_likelihood":
        from .trajectory_likelihood.monitor import TrajectoryLikelihoodMonitor

        return TrajectoryLikelihoodMonitor.from_mapping(mapping)
    if kind == "ours_task_state":
        from .ours_monitor.monitor import OursTaskStateMonitor

        return OursTaskStateMonitor()
    if "factory" in mapping:
        return _external_monitor(mapping)
    raise ValueError(f"unsupported runtime monitor: {kind!r}")


__all__ = ["METHOD_CONFIG_ROOT", "MethodSpec", "build_monitor", "load_method_spec"]
