"""Run one Native-6 v3 episode with the frozen M5 policy on server A.

The module is an integration layer only.  It delegates policy execution to
``shared_episode`` and physical intervention to the already frozen Native-6
v3 adapter.  Neither the M5 policy nor the event-grounded fault semantics are
reimplemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from evaluations.iclr2027.interfaces.feature_schema import AUDIT_SCHEMA
from evaluations.iclr2027.native6_v3.result import (
    PROTOCOL_REVISION,
    RESULT_SCHEMA,
    validate_native6_v3_result,
)
from evaluations.iclr2027.native_systems.rvt.e6_v3.physics import PhysicalAdapter
from evaluations.iclr2027.runners import shared_episode
from evaluations.iclr2027.runners.episode_io import load_episode, resolve_cycle_file


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IDENTITY = (
    ROOT / "evaluations" / "iclr2027" / "results" / "native_v3" / "ours"
    / "OURS_SYSTEM_IDENTITY.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe(episode_id: str) -> str:
    return str(episode_id).replace("/", "__")


def _record_path(output_root: Path, episode_id: str) -> Path:
    key = hashlib.sha256(str(episode_id).encode("utf-8")).hexdigest()[:20]
    return Path(output_root) / "records" / f"{key}.json"


def _completed_step_confirms_maintained_contact_loss(
    sample: Mapping[str, Any], target_objects: set[str]
) -> bool:
    """Return whether one completed physics step proves contact loss.

    This is the same frozen physical-effect definition used by the Native-6
    adapter: both gripper fingers are open and none of the relation targets is
    in physical contact.  Evaluating it at the completed-step callback avoids
    a later task callback hiding an effect that already occurred.
    """

    return bool(
        target_objects
        and all(float(value) > 0.9 for value in sample["actual_open_amount"])
        and not target_objects.intersection(sample["contact_objects"])
    )


class OursPhysicalAdapter(PhysicalAdapter):
    """Expose the TaskEnvironment API while retaining the common v3 adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Audit-only pending state.  It does not alter a command, simulator
        # step, policy observation, or recovery decision.
        self._effect_targets_pending: set[str] = set()
        super().__init__(*args, **kwargs)

    def _relation_loss(self, decision: Any) -> None:
        if decision.interaction_source == "maintained_contact":
            self._effect_targets_pending = set(decision.object_names)
        super()._relation_loss(decision)

    def _observe_step(self, record: Any) -> None:
        super()._observe_step(record)
        if self.state.physical_effect_confirmed:
            self._effect_targets_pending.clear()
            return
        if not self._effect_targets_pending:
            return
        sample = self.events[-1]
        if not _completed_step_confirms_maintained_contact_loss(
            sample, self._effect_targets_pending
        ):
            return
        evidence = {
            "source": "maintained_contact",
            "actual_open_amount": sample["actual_open_amount"],
            "contact_objects": sample["contact_objects"],
            "confirmation_source": "completed_simulator_step",
        }
        self.state.confirm_physical_effect(
            sim_step=int(sample["sim_step"]),
            simulation_time_s=float(sample["simulation_time_s"]),
            evidence=evidence,
        )
        self.onset = {
            "sim_step": int(sample["sim_step"]),
            "simulation_time_s": float(sample["simulation_time_s"]),
        }
        self._effect_targets_pending.clear()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend.raw, name)

    def protocol_metadata(self) -> dict[str, Any]:
        value = self.summary()
        event = self.state.trigger_event
        return {
            **value,
            # Backwards-compatible names consumed only by the shared episode
            # recorder.  Native-v3 truth remains the simulator-step summary.
            "triggered": bool(value["injection_triggered"]),
            "physical_effect_observed": bool(value["physical_effect_confirmed"]),
            "target_objects": [] if event is None else list(event.object_names),
            "events": self.events,
            "protocol_revision": PROTOCOL_REVISION,
        }


class NativeV3Auditor:
    """Project simulator-step v3 truth onto the shared cycle audit record."""

    def __init__(self, environment: OursPhysicalAdapter, *_args: Any, **_kwargs: Any) -> None:
        self.environment = environment
        self.violation_onset_cycle = None
        self.violation_end_cycle = None
        self.relation_restored_cycle = None
        self.legal_reentry_cycle = None
        self._last_cycle = 0

    def before_step(self, cycle: int, _observation: Any, _action: Any) -> None:
        self._last_cycle = int(cycle)

    def _update(self, cycle: int) -> dict[str, Any]:
        value = self.environment.summary()
        if value["violation_onset_sim_step"] is not None and self.violation_onset_cycle is None:
            self.violation_onset_cycle = int(cycle)
        if value["violation_end_sim_step"] is not None and self.violation_end_cycle is None:
            self.violation_end_cycle = int(cycle)
        if value["relation_restored_sim_step"] is not None and self.relation_restored_cycle is None:
            self.relation_restored_cycle = int(cycle)
        return value

    def after_step(self, cycle: int, _observation: Any, _injector: Any) -> dict[str, Any]:
        self._last_cycle = int(cycle)
        value = self._update(cycle)
        success, terminate = self.environment._scene.task.success()
        sample = self.environment.sample()
        relation = "linked" if sample["relation_source"] is not None else "external"
        return {
            "schema": AUDIT_SCHEMA,
            "cycle": int(cycle),
            "eligible": bool(value["eligible"]),
            "physically_triggered": bool(value["physically_triggered"]),
            "violation_onset_cycle": self.violation_onset_cycle,
            "violation_end_cycle": self.violation_end_cycle,
            "expected_relation": (
                {"single": "linked"}
                if self.environment.row["fault_family"] in {"missed_interaction", "relation_loss"}
                else None
            ),
            "physical_relation": {"single": relation},
            "relation_restored_cycle": self.relation_restored_cycle,
            "task_boundary_state": None,
            "legal_reentry_cycle": self.legal_reentry_cycle,
            "oracle_recoverable": bool(success or not terminate),
            "task_success": bool(success),
        }

    def summary(self) -> dict[str, Any]:
        value = self._update(self._last_cycle)
        event = self.environment.state.trigger_event
        success, terminate = self.environment._scene.task.success()
        return {
            "schema": AUDIT_SCHEMA,
            "eligible": bool(value["eligible"]),
            "physically_triggered": bool(value["physically_triggered"]),
            "violation_onset_cycle": self.violation_onset_cycle,
            "violation_end_cycle": self.violation_end_cycle,
            "relation_restored_cycle": self.relation_restored_cycle,
            "legal_reentry_cycle": self.legal_reentry_cycle,
            "target_objects": [] if event is None else list(event.object_names),
            "oracle_recoverable": bool(success or not terminate),
        }


def build_native_record(
    row: Mapping[str, Any],
    episode_path: Path,
    identity: Mapping[str, Any],
    *,
    physical_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Convert one persisted shared result into the strict Native-v3 schema."""

    episode = load_episode(episode_path)
    cycle_path = resolve_cycle_file(episode_path, episode)
    physical = dict(physical_summary or {})
    physical.setdefault("eligible", False)
    physical.setdefault("injection_triggered", False)
    physical.setdefault("physical_effect_confirmed", False)
    physical.setdefault("physically_triggered", False)
    physical.setdefault("completed_sim_steps", 0)
    physical.setdefault("final_simulation_time_s", 0.0)
    for prefix in ("eligible", "trigger", "effect", "violation_onset", "violation_end", "relation_restored"):
        physical.setdefault(prefix + "_sim_step", None)
        physical.setdefault(prefix + "_time_s", None)
    native = {
        **{key: row[key] for key in (
            "episode_id", "pair_id", "task", "variation", "seed", "condition",
            "fault_family", "fault_severity", "trigger_rule", "event_ordinal", "horizon",
        )},
        "schema": RESULT_SCHEMA,
        "protocol_revision": PROTOCOL_REVISION,
        "method_id": "ours",
        **{key: identity[key] for key in (
            "config_identity", "checkpoint_identity", "environment_identity",
            "fault_adapter_identity", "fault_config_identity", "audit_protocol_revision",
        )},
        "manifest_identity": str(identity["manifest_identity"]),
        "eligible": bool(physical["eligible"]),
        "injection_triggered": bool(physical["injection_triggered"]),
        "physical_effect_confirmed": bool(physical["physical_effect_confirmed"]),
        "physically_triggered": bool(physical["physically_triggered"]),
        "eligible_sim_step": physical["eligible_sim_step"],
        "trigger_sim_step": physical["trigger_sim_step"],
        "effect_sim_step": physical["effect_sim_step"],
        "eligible_time_s": physical["eligible_time_s"],
        "trigger_time_s": physical["trigger_time_s"],
        "effect_time_s": physical["effect_time_s"],
        "violation_onset_sim_step": physical["violation_onset_sim_step"],
        "violation_onset_time_s": physical["violation_onset_time_s"],
        "violation_end_sim_step": physical["violation_end_sim_step"],
        "violation_end_time_s": physical["violation_end_time_s"],
        "relation_restored_sim_step": physical["relation_restored_sim_step"],
        "relation_restored_time_s": physical["relation_restored_time_s"],
        "completed_sim_steps": int(physical["completed_sim_steps"]),
        "final_simulation_time_s": float(physical["final_simulation_time_s"]),
        "final_success": bool(episode["final_success"]),
        "cycles": int(episode["cycles"]),
        "termination_reason": str(episode["termination_reason"]),
        "infrastructure_error": episode["termination_reason"] == "infrastructure_error",
        "error_detail": episode.get("error"),
        "wall_seconds": float(episode["wall_seconds"]),
        "peak_memory_kib": int(episode["peak_memory_kib"]),
        "cycle_file": str(cycle_path.relative_to(ROOT)),
        "cycle_file_sha256": _sha256(cycle_path),
        "cycle_records": int(episode["cycle_records"]),
    }
    validate_native6_v3_result(native, row)
    return native


def run_one(
    row: Mapping[str, Any],
    output_root: Path,
    *,
    identity_path: Path = DEFAULT_IDENTITY,
    policy_python: Path = shared_episode.DEFAULT_POLICY_PYTHON,
) -> dict[str, Any]:
    identity = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    if identity["manifest_identity"] != _sha256(ROOT / identity["manifest_path"]):
        raise RuntimeError("Native-v3 Ours manifest identity changed")
    adapter_holder: dict[str, Any] = {}

    def factory(task_environment: Any, _task: Any, **_kwargs: Any) -> OursPhysicalAdapter:
        adapter = OursPhysicalAdapter(SimpleNamespace(raw=task_environment), dict(row))
        adapter_holder["adapter"] = adapter
        return adapter

    original_factory = shared_episode.build_fault_environment
    original_auditor = shared_episode.PhysicalEventAuditor
    original_environment = shared_episode._environment

    def environment_factory(task: Any):
        environment, restore = original_environment(task)
        original_shutdown = environment.shutdown

        def shutdown() -> Any:
            adapter = adapter_holder.get("adapter")
            if adapter is not None and not adapter_holder.get("closed"):
                adapter_holder["final_summary"] = adapter.summary()
                adapter.close()
                adapter_holder["closed"] = True
            return original_shutdown()

        environment.shutdown = shutdown
        return environment, restore

    shared_episode.build_fault_environment = factory
    shared_episode.PhysicalEventAuditor = NativeV3Auditor
    shared_episode._environment = environment_factory
    internal_row = dict(row)
    # The shared recorder computes only an obsolete cycle-floor diagnostic
    # from this field.  The patched v3 adapter ignores it and triggers solely
    # on the first eligible physical event.
    internal_row["trigger_stage"] = "early"
    try:
        shared_episode.run_episode(
            internal_row,
            output_root,
            policy_python=policy_python,
            method="m5_full",
        )
        episode_path = Path(output_root) / "episodes" / f"{_safe(row['episode_id'])}.json"
        adapter = adapter_holder.get("adapter")
        native = build_native_record(
            row,
            episode_path,
            identity,
            physical_summary=adapter_holder.get("final_summary"),
        )
        _atomic_json(_record_path(output_root, str(row["episode_id"])), native)
        return native
    finally:
        adapter = adapter_holder.get("adapter")
        if adapter is not None and not adapter_holder.get("closed"):
            adapter.close()
        shared_episode.build_fault_environment = original_factory
        shared_episode.PhysicalEventAuditor = original_auditor
        shared_episode._environment = original_environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--policy-python", type=Path, default=shared_episode.DEFAULT_POLICY_PYTHON)
    # Accepted for compatibility with the common queue; Ours is always M5.
    parser.add_argument("--method", default="m5_full")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    row = shared_episode._load_manifest_row(args.manifest, args.episode_id)
    result = run_one(
        row,
        args.output_root,
        identity_path=args.identity,
        policy_python=args.policy_python,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 2 if result["infrastructure_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
