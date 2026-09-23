"""Prepare, execute, and aggregate Ours under Native-6 v3 Amendment 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.runners.a6_endpoint_compatibility import (
    assert_frozen_m5_post_e4,
)

from evaluations.iclr2027.native6_v3.gate_amendment_5 import (
    build_development_gate_amendment_5,
    reaudit_ours_completed_step_effects,
)
from evaluations.iclr2027.native6_v3.result import validate_native6_v3_result
from evaluations.iclr2027.runners.e6_ours_v3_episode import (
    _atomic_json,
    _record_path,
    build_native_record,
)
from evaluations.iclr2027.runners.episode_io import load_episode
from evaluations.iclr2027.runners.launch import DynamicQueue, Running, _rows, _safe
from integrations.rlbench.rlbench_dynamac.eval.v4_formal_launch import (
    _launch_environment,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
MANIFESTS = {
    "development": EVAL_ROOT / "manifests" / "native6_v3_development.jsonl",
    "formal": EVAL_ROOT / "manifests" / "native6_v3_perturbed.jsonl",
}
OUTPUTS = {
    phase: EVAL_ROOT / "results" / "native_v3" / "ours" / phase
    for phase in MANIFESTS
}
METHOD_CONFIG = EVAL_ROOT / "configs" / "methods" / "m5_full.json"
A5_PLAN = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_RUN_PLAN.json"
M5_POST_E4_FREEZE = EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
WORKERS = 32
TIMEOUT_SECONDS = 900.0
EPISODE_MODULE = "evaluations.iclr2027.runners.e6_ours_v3_episode"

FAULT_ADAPTER_FILES = (
    EVAL_ROOT / "configs" / "shared" / "native6_object_contact_mapping_v3.json",
    EVAL_ROOT / "configs" / "shared" / "native6_physical_protocol_v3.json",
    EVAL_ROOT / "configs" / "shared" / "native6_physical_protocol_v3_amendment_4.json",
    EVAL_ROOT / "configs" / "shared" / "native6_physical_protocol_v3_amendment_5.json",
    EVAL_ROOT / "native6_v3" / "events.py",
    EVAL_ROOT / "native6_v3" / "physical_clock.py",
    EVAL_ROOT / "native6_v3" / "object_mapping.py",
    EVAL_ROOT / "native6_v3" / "open_drawer_missed_interaction.py",
    EVAL_ROOT / "native_systems" / "rvt" / "e6_v3" / "physics.py",
    EVAL_ROOT / "native_systems" / "rvt" / "e6_v3" / "amendment2.py",
    EVAL_ROOT / "native_systems" / "rvt" / "e6_v3" / "mapping" / "candidate_filter.py",
    EVAL_ROOT / "runners" / "e6_ours_v3_episode.py",
    EVAL_ROOT / "native6_v3" / "gate_amendment_5.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(paths: Iterable[Path]) -> tuple[str, list[dict[str, str]]]:
    entries = []
    aggregate = hashlib.sha256()
    for path in sorted({Path(value).resolve() for value in paths}):
        relative = str(path.relative_to(ROOT))
        digest = _sha256(path)
        entries.append({"path": relative, "sha256": digest})
        aggregate.update(f"{relative}\0{digest}\n".encode("utf-8"))
    return aggregate.hexdigest(), entries


def _assert_frozen_m5() -> dict[str, Any]:
    freeze = assert_frozen_m5_post_e4()
    if _sha256(METHOD_CONFIG) != freeze["official_config_alias"]["sha256"]:
        raise RuntimeError("post-E4 frozen M5 method config changed")
    return freeze


def prepare(phase: str) -> dict[str, Any]:
    freeze = _assert_frozen_m5()
    manifest = MANIFESTS[phase]
    expected = 180 if phase == "development" else 600
    rows = _rows(manifest)
    if len(rows) != expected:
        raise RuntimeError(f"Native-v3 {phase} manifest must contain {expected} rows")
    adapter_identity, adapter_entries = _aggregate(FAULT_ADAPTER_FILES)
    protocol_identity, protocol_entries = _aggregate(
        (
            EVAL_ROOT / "configs" / "shared" / "native6_physical_protocol_v3.json",
            EVAL_ROOT / "configs" / "shared" / "native6_physical_protocol_v3_amendment_4.json",
            EVAL_ROOT / "configs" / "shared" / "native6_physical_protocol_v3_amendment_5.json",
        )
    )
    identity = {
        "schema": "essay2608.iclr2027.native6-v3-ours-system-identity.v1",
        "phase": phase,
        "method_id": "ours",
        "config_identity": _sha256(METHOD_CONFIG),
        "checkpoint_identity": freeze["model_tree_identity"]["aggregate_sha256"],
        "environment_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
        "fault_adapter_identity": adapter_identity,
        "fault_config_identity": protocol_identity,
        "audit_protocol_revision": "native6_event_grounded_physics_v3",
        "manifest_path": str(manifest.relative_to(ROOT)),
        "manifest_identity": _sha256(manifest),
        "episodes": expected,
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "one_episode_per_job": True,
        "fault_adapter_files": adapter_entries,
        "fault_protocol_files": protocol_entries,
        "a5_run_plan": str(A5_PLAN.relative_to(ROOT)),
        "a5_run_plan_sha256": _sha256(A5_PLAN),
        "m5_post_e4_freeze": str(M5_POST_E4_FREEZE.relative_to(ROOT)),
        "m5_post_e4_freeze_sha256": _sha256(M5_POST_E4_FREEZE),
    }
    path = OUTPUTS[phase] / "SYSTEM_IDENTITY.json"
    _atomic_json(path, identity)
    return identity


class OursNativeV3Queue(DynamicQueue):
    def __init__(self, *args: Any, identity_path: Path, **kwargs: Any) -> None:
        self.identity_path = Path(identity_path).resolve()
        super().__init__(*args, **kwargs)

    def _launch(self, row: dict[str, Any], lane_index: int) -> None:
        lane = self.lanes[lane_index]
        identifier = _safe(row["episode_id"])
        attempt = self.attempts[row["episode_id"]]
        log_path = self.log_root / f"{identifier}.attempt{attempt}.log"
        stream = log_path.open("w", encoding="utf-8")
        command = [
            str(self.xvfb_run), "--auto-servernum", "--server-num", str(180 + lane_index),
            "--error-file", str(self.log_root / f"{identifier}.xvfb.log"),
            "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
            str(self.sim_python), "-m", EPISODE_MODULE,
            "--manifest", str(self.manifest), "--episode-id", str(row["episode_id"]),
            "--output-root", str(self.output_root), "--policy-python", str(self.policy_python),
            "--identity", str(self.identity_path), "--method", "m5_full",
        ]
        environment = _launch_environment(self.policy_python, lane.gpu)
        lane_tmp = self.tmp_root / f"lane_{lane_index:02d}"
        lane_tmp.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(lane_tmp)

        def affinity() -> None:
            os.setsid()
            os.sched_setaffinity(0, set(lane.logical_cpus))

        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            preexec_fn=affinity,
        )
        self.running[lane_index] = Running(
            process, row, lane_index, stream, time.monotonic()
        )
        self.peak_active = max(self.peak_active, len(self.running))


def aggregate(phase: str) -> list[dict[str, Any]]:
    output = OUTPUTS[phase]
    identity = json.loads((output / "SYSTEM_IDENTITY.json").read_text(encoding="utf-8"))
    rows = _rows(MANIFESTS[phase])
    records = []
    for row in rows:
        record_path = _record_path(output, row["episode_id"])
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            episode_path = output / "episodes" / f"{_safe(row['episode_id'])}.json"
            if not episode_path.is_file():
                raise RuntimeError("missing Native-v3 Ours result: " + row["episode_id"])
            episode = load_episode(episode_path)
            if episode["termination_reason"] != "infrastructure_error":
                raise RuntimeError("non-infrastructure result misses Native-v3 record: " + row["episode_id"])
            record = build_native_record(
                row, episode_path, identity, physical_summary=None
            )
            _atomic_json(record_path, record)
        validate_native6_v3_result(record, row)
        if record["manifest_identity"] != identity["manifest_identity"]:
            raise RuntimeError("Native-v3 record uses another manifest identity")
        records.append(record)
    aggregate_path = output / "episodes.jsonl"
    temporary = aggregate_path.with_name(aggregate_path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    os.replace(temporary, aggregate_path)
    return records


def development_gate_amendment_5(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    corrected, corrections = reaudit_ours_completed_step_effects(
        records,
        episode_root=OUTPUTS["development"] / "episodes",
    )
    audit = {
        "schema": "essay2608.iclr2027.native6-v3-completed-step-reaudit.v1",
        "protocol_amendment": "native6_event_grounded_physics_v3_amendment_5",
        "raw_records_mutated": False,
        "task_success_mutated": False,
        "corrections": corrections,
    }
    _atomic_json(OUTPUTS["development"] / "REAUDIT_AMENDMENT_5.json", audit)
    gate = build_development_gate_amendment_5(corrected, system_id="ours")
    _atomic_json(OUTPUTS["development"] / "GATE_AMENDMENT_5.json", gate)
    return gate


def run(phase: str) -> int:
    prepare(phase)
    manifest = MANIFESTS[phase]
    output = OUTPUTS[phase]
    rows = _rows(manifest)
    queue = OursNativeV3Queue(
        rows,
        manifest=manifest,
        output_root=output,
        workers=WORKERS,
        sim_python=Path(os.environ.get("DYNAMAC_SIM_PYTHON", sys.executable)),
        policy_python=Path(os.environ.get("DYNAMAC_POLICY_PYTHON", sys.executable)),
        xvfb_run=Path("/usr/bin/xvfb-run"),
        success_target=None,
        retry_infrastructure=1,
        episode_timeout_seconds=TIMEOUT_SECONDS,
        method="m5_full",
        calibration_artifact=None,
        policy_diagnostics_dir=None,
        identity_path=output / "SYSTEM_IDENTITY.json",
    )
    status = queue.run()
    records = aggregate(phase)
    if phase == "development":
        gate = development_gate_amendment_5(records)
        if gate["status"] != "PASS":
            return 1
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "aggregate", "gate"))
    parser.add_argument("--phase", choices=("development", "formal"), default="development")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.phase), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "run":
        return run(args.phase)
    records = aggregate(args.phase)
    if args.command == "gate":
        if args.phase != "development":
            raise ValueError("the Amendment 4 gate is development-only")
        gate = development_gate_amendment_5(records)
        print(json.dumps({"status": gate["status"], "episodes": len(records)}))
        return 0 if gate["status"] == "PASS" else 1
    print(json.dumps({"phase": args.phase, "episodes": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
