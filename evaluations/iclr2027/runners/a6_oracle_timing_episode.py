"""Run one E3-C diagnostic episode with independent oracle event timing.

This entry point delegates simulator execution and result accounting to the
frozen shared episode runner.  It replaces only the policy subprocess with the
diagnostic-only oracle timing server and forwards the first independently
audited violation onset.  No fault identity or recovery semantics cross into
the policy process.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.audit.physical_events import (
    PhysicalEventAuditor as FrozenPhysicalEventAuditor,
)
from evaluations.iclr2027.methods.registry import MethodSpec, load_method_spec
from evaluations.iclr2027.runners import shared_episode
from integrations.rlbench.rlbench_tsf.protocol import (
    tsf_gripper_timing_metadata,
)
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    FORMAL_POLICY_CLOCK_SEMANTICS_ID as POLICY_CLOCK_SEMANTICS_ID,
)
from integrations.rlbench.rlbench_dynamac.eval import (
    direct_evaluate,
    unimanual_evaluate,
)


ORACLE_SERVER_MODULE = (
    "integrations.rlbench.rlbench_tsf.oracle_timing_server"
)
ORACLE_EPISODE_SCHEMA = "essay2608.iclr2027.oracle-timing-episode.v1"

_ACTIVE_WORKER: "OraclePolicyProcess | None" = None
_ACTIVE_AUDITOR: "OracleTimingAuditor | None" = None


class OraclePolicyProcess:
    """Minimal evaluator client for the isolated oracle timing server."""

    def __init__(
        self,
        python: Path,
        task: str,
        base_models_dir: Path,
        tsf_models_dir: Path,
        *,
        bimanual: bool,
        feature_profile: str,
        diagnostics_dir: Path | None,
        task_specs_path: Path | None,
        boundary_config: Path | None,
        timeout: float = 120.0,
    ) -> None:
        self.timeout = float(timeout)
        if self.timeout <= 0.0:
            raise ValueError("policy timeout must be positive")
        self.policy_type = "task_state_feedback"
        self._bimanual = bool(bimanual)
        self._observation_payload = (
            direct_evaluate._observation_payload
            if self._bimanual
            else unimanual_evaluate._observation_payload
        )
        command = [
            str(python),
            "-m",
            ORACLE_SERVER_MODULE,
            "serve",
            "--task",
            task,
            "--models-dir",
            str(Path(tsf_models_dir).resolve()),
            "--base-models-dir",
            str(Path(base_models_dir).resolve()),
            "--feature-profile",
            str(feature_profile),
        ]
        if task_specs_path is not None:
            command.extend(["--task-specs", str(Path(task_specs_path).resolve())])
        if diagnostics_dir is not None:
            command.extend(["--diagnostics-dir", str(Path(diagnostics_dir).resolve())])
        if boundary_config is not None:
            command.extend(["--boundary-config", str(Path(boundary_config).resolve())])
        self.process = subprocess.Popen(
            command,
            cwd=str(shared_episode.REPOSITORY_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            response = self.request("ping")
            if (
                not response.get("ready")
                or bool(response.get("bimanual")) != self._bimanual
                or response.get("task") != task
            ):
                raise RuntimeError("oracle policy worker identity mismatch")
            self.policy_steps = int(response.get("policy_steps", 0))
            if self.policy_steps < 1:
                raise RuntimeError("oracle policy worker reported an empty trajectory")
            self.model_identity = response.get("model_identity")
            if not isinstance(self.model_identity, dict):
                raise RuntimeError("oracle policy worker omitted model identity")
            oracle_identity = self.model_identity.get("oracle_timing_diagnostic")
            if not isinstance(oracle_identity, dict):
                raise RuntimeError("oracle timing identity is missing")
            if response.get("policy_clock_semantics_id") != POLICY_CLOCK_SEMANTICS_ID:
                raise RuntimeError("oracle policy clock semantics mismatch")
            if (
                response.get("gripper_timing")
                != tsf_gripper_timing_metadata()
                or response.get("policy_type") != self.policy_type
            ):
                raise RuntimeError("oracle policy gripper/protocol identity mismatch")
        except Exception:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            raise

    def request(self, command: str, observation: Any = None, **fields: Any) -> dict[str, Any]:
        request = {"command": command, **fields}
        if observation is not None:
            request["observation"] = self._observation_payload(observation)
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            self.process.terminate()
            raise TimeoutError("oracle policy worker response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("oracle policy worker exited without a response")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"oracle policy worker error: {response.get('error')}")
        return response

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.request("close")
            self.process.wait(timeout=5)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _oracle_policy_process(
    task: Any,
    policy_python: Path,
    method: MethodSpec,
    *,
    diagnostics_dir: Path | None = None,
) -> OraclePolicyProcess:
    global _ACTIVE_WORKER
    if method.policy_type != "task_state_feedback" or method.feature_profile != "full":
        raise ValueError("oracle timing diagnostic requires the frozen Full method")
    boundary_root = None
    tsf_models = shared_episode.TSF_MODELS
    if method.runtime is not None:
        configured = method.runtime.get("boundary_config_root")
        if configured is not None:
            boundary_root = shared_episode.REPOSITORY_ROOT / str(configured)
        configured = method.runtime.get("tsf_models_root")
        if configured is not None:
            candidate = (shared_episode.REPOSITORY_ROOT / str(configured)).resolve()
            candidate.relative_to(shared_episode.REPOSITORY_ROOT.resolve())
            tsf_models = candidate
    boundary_config = (
        None
        if boundary_root is None
        else Path(boundary_root) / f"{task.task_id}.json"
    )
    worker = OraclePolicyProcess(
        policy_python,
        task.task_id,
        (
            shared_episode.BIMANUAL_MODELS
            if task.spec.bimanual
            else shared_episode.SINGLE_MODELS
        ),
        tsf_models,
        bimanual=task.spec.bimanual,
        feature_profile="full",
        diagnostics_dir=diagnostics_dir,
        task_specs_path=(None if task.spec.bimanual else shared_episode.TASK_SPECS_PATH),
        boundary_config=boundary_config,
    )
    _ACTIVE_WORKER = worker
    return worker


class OracleTimingAuditor(FrozenPhysicalEventAuditor):
    """Forward exactly one audited onset cycle to the diagnostic worker."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global _ACTIVE_AUDITOR
        super().__init__(*args, **kwargs)
        self.oracle_event_response: dict[str, Any] | None = None
        self.oracle_final_status: dict[str, Any] | None = None
        _ACTIVE_AUDITOR = self

    def after_step(
        self,
        cycle: int,
        observation: Any,
        injector: Mapping[str, Any] | None,
    ) -> dict:
        record = super().after_step(cycle, observation, injector)
        if self.violation_onset_cycle is not None and self.oracle_event_response is None:
            if _ACTIVE_WORKER is None:
                raise RuntimeError("oracle auditor has no active policy worker")
            response = _ACTIVE_WORKER.request(
                "oracle_event",
                violation_onset_cycle=int(self.violation_onset_cycle),
            )
            self.oracle_event_response = dict(response["oracle_timing"])
        elif self.oracle_event_response is not None and self.oracle_final_status is None:
            if _ACTIVE_WORKER is None:
                raise RuntimeError("oracle auditor lost its active policy worker")
            response = _ACTIVE_WORKER.request("oracle_status")
            status = dict(response["oracle_timing"])
            if status.get("triggered_tick") is not None or status.get("expired"):
                self.oracle_final_status = status
        return record


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_episode(
    row: Mapping[str, Any],
    output_root: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    global _ACTIVE_WORKER, _ACTIVE_AUDITOR
    method = load_method_spec(kwargs.get("method", "m5_full"))
    if method.method_id != "m5_full":
        raise ValueError("oracle feasibility diagnostic accepts only m5_full")
    original_policy_process = shared_episode._policy_process
    original_auditor = shared_episode.PhysicalEventAuditor
    _ACTIVE_WORKER = None
    _ACTIVE_AUDITOR = None
    shared_episode._policy_process = _oracle_policy_process
    shared_episode.PhysicalEventAuditor = OracleTimingAuditor
    try:
        result = shared_episode.run_episode(row, output_root, **kwargs)
    finally:
        shared_episode._policy_process = original_policy_process
        shared_episode.PhysicalEventAuditor = original_auditor
    auditor = _ACTIVE_AUDITOR
    status = None
    event_response = None
    if auditor is not None:
        event_response = auditor.oracle_event_response
        status = auditor.oracle_final_status or event_response
    oracle = {
        "schema": ORACLE_EPISODE_SCHEMA,
        "diagnostic_only": True,
        "input": {
            "violation_onset_cycle": result.get("audit", {}).get(
                "violation_onset_cycle"
            )
        },
        "fault_identity_visible_to_policy": False,
        "repair_target_visible_to_policy": False,
        "reentry_state_visible_to_policy": False,
        "event_forwarded": event_response is not None,
        "server_status": status,
        "oracle_alarm_cycle": (
            None if status is None else status.get("triggered_tick")
        ),
        "final_success": bool(result.get("final_success")),
    }
    path = (
        Path(output_root)
        / "episodes"
        / (str(row["episode_id"]).replace("/", "__") + ".json")
    )
    # ``shared_episode.run_episode`` returns the in-memory summary, while its
    # EpisodeWriter commits additional storage identity fields (cycle path,
    # record count, and SHA256) to disk.  Extend that committed artifact rather
    # than overwriting it with the smaller in-memory summary.
    persisted = json.loads(path.read_text(encoding="utf-8"))
    if persisted.get("episode_id") != str(row["episode_id"]):
        raise RuntimeError("oracle episode artifact identity mismatch")
    persisted["oracle_timing_diagnostic"] = oracle
    _atomic_json(path, persisted)
    return persisted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, default=shared_episode.DEFAULT_POLICY_PYTHON)
    parser.add_argument("--method", default="m5_full")
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--policy-diagnostics-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    row = shared_episode._load_manifest_row(args.manifest, args.episode_id)
    result = run_episode(
        row,
        args.output_root,
        policy_python=args.policy_python,
        method=args.method,
        calibration_artifact=args.calibration_artifact,
        policy_diagnostics_dir=args.policy_diagnostics_dir,
    )
    return 0 if result["reason"] != "infrastructure_error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
