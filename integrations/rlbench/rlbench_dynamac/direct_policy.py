"""Direct training and JSON-lines policy serving for configured RLBench tasks.

This is intentionally a small execution entry point.  Training reads the five
saved low-dimensional demonstrations through the current adapter, fits the
current :mod:`essay2608.policy` implementation, and writes either one
unimanual checkpoint or one checkpoint per arm. Serving keeps simulator-only
RLBench/PyRep imports in a Python 3.8 process while the policy runs in its
Python 3.10 environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .demo_adapter import (
    DYNAMAC_GRIPPER_TARGET_TIMING,
    DYNAMAC_POSE_TARGET_TIMING,
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from .records import atomic_json, reserve_output
from .runtime import (
    bimanual_action_to_rlbench,
    bimanual_observations_from_rlbench,
    unimanual_action_to_rlbench,
    unimanual_observation_from_rlbench,
)
from .task_specs import get_task_spec, load_task_specs
from .v3_protocol import (
    bimanual_checkpoint_trigger_audit,
    build_v3_trigger_anchor_evidence,
    checkpoint_trigger_audit,
)

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    INTEGRATION_ROOT
    / "data"
    / "dynamac_table_ii_g5_a51b4e_128x128_seed0_20260811"
    / "stage_5_demos"
)
# Release defaults always name the immutable configuration file that is part of
# that release's identity.  ``dynamac_rlbench_local.json`` remains a historical
# compatibility alias and is deliberately not used for V3 authentication.
DEFAULT_CONFIG = INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v3.json"
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3"
TRAINING_MANIFEST_SCHEMA_V2 = "dynamac-direct-training-v2"
TRAINING_MANIFEST_SCHEMA_V3 = "dynamac-direct-training-v3"
AUTHENTICATED_TRAINING_MANIFEST_SCHEMAS = frozenset(
    {TRAINING_MANIFEST_SCHEMA_V2, TRAINING_MANIFEST_SCHEMA_V3}
)
V3_ADAPTER_PROTOCOL = {
    "schema": "rlbench-dynamac-demo-adapter-v3",
    "pose_target_timing": DYNAMAC_POSE_TARGET_TIMING,
    "gripper_action_timing": DYNAMAC_GRIPPER_TARGET_TIMING,
    "action_timing": "obs[t] current state",
    "pose_and_gripper_sample_aligned": True,
}
POLICY_CLOCK_SEMANTICS_ID = (
    "policy-tick-transaction-commit-on-primary-action-success-v1"
)


def _episode_number(path: Path) -> tuple[int, str]:
    name = path.parent.name
    suffix = name.removeprefix("episode")
    return (int(suffix), name) if suffix.isdigit() else (sys.maxsize, name)


def demonstration_paths(data_root: Path, task: str, count: int = 5) -> list[Path]:
    """Return the first ``count`` naturally ordered low-dimensional episodes."""

    if count < 1:
        raise ValueError("demonstration count must be positive")
    episode_root = data_root / task / "all_variations" / "episodes"
    paths = sorted(episode_root.glob("episode*/low_dim_obs.pkl"), key=_episode_number)
    if len(paths) < count:
        raise FileNotFoundError(
            f"{task}: found {len(paths)} low_dim_obs.pkl files under {episode_root}; "
            f"need {count}"
        )
    return paths[:count]


def load_policy_config(path: Path) -> Any:
    from essay2608.policy import DynaMACConfig

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"policy config must be a JSON object: {path}")
    return DynaMACConfig(**payload)


def train_task(
    task: str,
    *,
    data_root: Path,
    models_dir: Path,
    config_path: Path,
    demonstration_count: int = 5,
) -> dict[str, Any]:
    """Fit, validate, and atomically publish one complete task model."""

    from .evaluation_split import validate_training_entry_paths

    validate_training_entry_paths(data_root, models_dir, config_path)

    output = models_dir / task
    with reserve_output(output):
        staging = Path(
            tempfile.mkdtemp(prefix=f".{task}.staging-", dir=str(models_dir))
        )
        try:
            summary = _train_task_into(
                task,
                data_root=data_root,
                output=staging,
                config_path=config_path,
                demonstration_count=demonstration_count,
            )
            _validate_published_model(task, staging, summary)
            os.rename(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return summary


def _train_task_into(
    task: str,
    *,
    data_root: Path,
    output: Path,
    config_path: Path,
    demonstration_count: int,
) -> dict[str, Any]:
    """Fit all artifacts inside an unpublished staging directory."""

    from essay2608.policy import BimanualDynaMAC, DynaMAC

    spec = get_task_spec(task)
    paths = demonstration_paths(data_root, task, demonstration_count)
    episodes = load_low_dim_obs_pickles(paths)
    config = load_policy_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"staging directory is not empty: {output}")
    names = [path.parent.name for path in paths]
    if spec.bimanual:
        converted = make_bimanual_demonstrations(episodes, spec, names=names)
        policy = BimanualDynaMAC(config=config)
        policy.fit(converted.left_demonstrations, converted.right_demonstrations)
        policy.left.save(output / "left.npz")
        policy.right.save(output / "right.npz")
        summary = {
            "task": task,
            "bimanual": True,
            "demonstrations": names,
            "config": asdict(config),
            "adapter": converted.audit,
            "left": {
                "skills": list(policy.left.skill_sequence),
                "durations": [skill.duration for skill in policy.left.skills],
                "config": asdict(policy.left.config),
                "fingerprint": policy.left.fingerprint(),
            },
            "right": {
                "skills": list(policy.right.skill_sequence),
                "durations": [skill.duration for skill in policy.right.skills],
                "config": asdict(policy.right.config),
                "fingerprint": policy.right.fingerprint(),
            },
        }
        checkpoint_audit = bimanual_checkpoint_trigger_audit(policy)
    else:
        converted = make_unimanual_demonstrations(episodes, spec, names=names)
        policy = DynaMAC(config=config).fit(converted.demonstrations)
        policy.save(output / "model.npz")
        summary = {
            "task": task,
            "bimanual": False,
            "demonstrations": names,
            "config": asdict(config),
            "skills": list(policy.skill_sequence),
            "durations": [skill.duration for skill in policy.skills],
            "fingerprint": policy.fingerprint(),
            "adapter": converted.audit,
        }
        checkpoint_audit = checkpoint_trigger_audit(policy)
    summary["manifest_schema"] = TRAINING_MANIFEST_SCHEMA_V3
    summary["checkpoint_trigger_audit"] = checkpoint_audit
    summary["v3_trigger_anchor_evidence"] = build_v3_trigger_anchor_evidence(
        task,
        checkpoint_audit,
        summary,
    )
    atomic_json(output / "training.json", summary)
    return summary


def _adapter_protocol_identity(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Return the release-relevant adapter fields without copying bulky audits."""

    audit = manifest.get("adapter")
    if not isinstance(audit, dict):
        return None
    return {key: audit.get(key) for key in V3_ADAPTER_PROTOCOL}


def _validate_v3_adapter_protocol(manifest: dict[str, Any]) -> None:
    if _adapter_protocol_identity(manifest) != V3_ADAPTER_PROTOCOL:
        raise RuntimeError(
            "staged V3 manifest does not use aligned obs[t] pose/gripper semantics"
        )


def _validate_v3_trigger_protocol(
    task: str,
    manifest: dict[str, Any],
    checkpoint_audit: dict[str, Any],
) -> None:
    if manifest.get("checkpoint_trigger_audit") != checkpoint_audit:
        raise RuntimeError("staged V3 checkpoint trigger audit mismatch")
    expected = build_v3_trigger_anchor_evidence(task, checkpoint_audit, manifest)
    if manifest.get("v3_trigger_anchor_evidence") != expected:
        raise RuntimeError("staged V3 trigger anchor evidence mismatch")


def _validate_published_model(
    task: str,
    output: Path,
    summary: dict[str, Any],
) -> None:
    """Reload staged checkpoints and bind them to their manifest fingerprints."""

    from essay2608.policy import BimanualDynaMAC, DynaMAC, DynaMACConfig

    if summary.get("task") != task:
        raise RuntimeError("staged training manifest has the wrong task identity")
    manifest_schema = summary.get("manifest_schema")
    if manifest_schema not in AUTHENTICATED_TRAINING_MANIFEST_SCHEMAS:
        raise RuntimeError("staged training manifest schema is not authenticated")
    if manifest_schema == TRAINING_MANIFEST_SCHEMA_V3:
        _validate_v3_adapter_protocol(summary)
    if summary.get("bimanual"):
        base_record = summary.get("config")
        if not isinstance(base_record, dict):
            raise RuntimeError("staged bimanual manifest is missing its base config")
        try:
            base_config = DynaMACConfig(**base_record)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("staged bimanual manifest has an invalid base config") from exc
        if base_record != asdict(base_config):
            raise RuntimeError("staged bimanual manifest base config is not canonical")
        expected = BimanualDynaMAC(config=base_config)
        left = DynaMAC.load(output / "left.npz")
        right = DynaMAC.load(output / "right.npz")
        for name, policy, expected_policy in (
            ("left", left, expected.left),
            ("right", right, expected.right),
        ):
            record = summary.get(name)
            if not isinstance(record, dict):
                raise RuntimeError(f"staged manifest is missing {name} policy metadata")
            if record.get("fingerprint") != policy.fingerprint():
                raise RuntimeError(f"staged {name} checkpoint fingerprint mismatch")
            if record.get("config") != asdict(policy.config):
                raise RuntimeError(f"staged {name} checkpoint config mismatch")
            if asdict(policy.config) != asdict(expected_policy.config):
                raise RuntimeError(
                    f"staged {name} checkpoint config is not derived from the base config"
                )
        identity_fields = (
            "model_schema_version",
            "selection_semantics_id",
            "tapas_reference_commit",
        )
        left_identity = left.summary()
        right_identity = right.summary()
        if any(left_identity[field] != right_identity[field] for field in identity_fields):
            raise RuntimeError("staged bimanual checkpoints have mismatched model identity")
        if manifest_schema == TRAINING_MANIFEST_SCHEMA_V3:
            _validate_v3_trigger_protocol(
                task,
                summary,
                bimanual_checkpoint_trigger_audit(
                    BimanualDynaMAC(left=left, right=right)
                ),
            )
    else:
        policy = DynaMAC.load(output / "model.npz")
        if summary.get("fingerprint") != policy.fingerprint():
            raise RuntimeError("staged checkpoint fingerprint mismatch")
        if summary.get("config") != asdict(policy.config):
            raise RuntimeError("staged checkpoint config mismatch")
        if manifest_schema == TRAINING_MANIFEST_SCHEMA_V3:
            _validate_v3_trigger_protocol(
                task,
                summary,
                checkpoint_trigger_audit(policy),
            )


class _WireObservation:
    """Attribute-compatible low-dimensional observation from one JSON request."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.task_low_dim_state = np.asarray(payload["task_low_dim_state"], dtype=np.float64)
        if "gripper_pose" in payload:
            self.gripper_pose = np.asarray(payload["gripper_pose"], dtype=np.float64)
        else:
            self.left = _WireArm(payload["left"])
            self.right = _WireArm(payload["right"])


class _WireArm:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.gripper_pose = np.asarray(payload["gripper_pose"], dtype=np.float64)


class PolicyServer:
    """Stateful current-core policy used by the Python 3.8 simulator process."""

    def __init__(self, task: str, models_dir: Path) -> None:
        from essay2608.policy import BimanualDynaMAC, DynaMAC

        spec = get_task_spec(task)
        self.task = task
        self.bimanual = spec.bimanual
        task_dir = models_dir / task
        manifest_path = task_dir / "training.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"training manifest must be a JSON object: {manifest_path}")
        if manifest.get("task") not in {None, task}:
            raise ValueError("training manifest task does not match the requested policy")
        if manifest.get("bimanual") not in {None, self.bimanual}:
            raise ValueError("training manifest arm count does not match the task")
        manifest_authenticated = (
            manifest.get("manifest_schema")
            in AUTHENTICATED_TRAINING_MANIFEST_SCHEMAS
        )
        if manifest_authenticated:
            _validate_published_model(task, task_dir, manifest)
        self.policy = (
            BimanualDynaMAC(
                left=DynaMAC.load(task_dir / "left.npz"),
                right=DynaMAC.load(task_dir / "right.npz"),
            )
            if self.bimanual
            else DynaMAC.load(task_dir / "model.npz")
        )
        self.model_identity = (
            {
                "model_schema_version": self.policy.left.summary()[
                    "model_schema_version"
                ],
                "selection_semantics_id": self.policy.left.summary()[
                    "selection_semantics_id"
                ],
                "tapas_reference_commit": self.policy.left.summary()[
                    "tapas_reference_commit"
                ],
                "left_config": asdict(self.policy.left.config),
                "right_config": asdict(self.policy.right.config),
                "left_fingerprint": self.policy.left.fingerprint(),
                "right_fingerprint": self.policy.right.fingerprint(),
                "training_manifest_schema": manifest.get("manifest_schema"),
                "manifest_authenticated": manifest_authenticated,
                "training_config": manifest.get("config") if manifest_authenticated else None,
                "training_adapter_protocol": (
                    _adapter_protocol_identity(manifest)
                    if manifest_authenticated
                    else None
                ),
                "checkpoint_trigger_audit_fingerprint": (
                    manifest.get("checkpoint_trigger_audit", {}).get("fingerprint")
                    if manifest_authenticated
                    and isinstance(manifest.get("checkpoint_trigger_audit"), dict)
                    else None
                ),
                "v3_trigger_anchor_evidence": (
                    manifest.get("v3_trigger_anchor_evidence")
                    if manifest_authenticated
                    else None
                ),
            }
            if self.bimanual
            else {
                "model_schema_version": self.policy.summary()["model_schema_version"],
                "selection_semantics_id": self.policy.summary()[
                    "selection_semantics_id"
                ],
                "tapas_reference_commit": self.policy.summary()[
                    "tapas_reference_commit"
                ],
                "config": asdict(self.policy.config),
                "fingerprint": self.policy.fingerprint(),
                "training_manifest_schema": manifest.get("manifest_schema"),
                "manifest_authenticated": manifest_authenticated,
                "training_config": manifest.get("config") if manifest_authenticated else None,
                "training_adapter_protocol": (
                    _adapter_protocol_identity(manifest)
                    if manifest_authenticated
                    else None
                ),
                "checkpoint_trigger_audit_fingerprint": (
                    manifest.get("checkpoint_trigger_audit", {}).get("fingerprint")
                    if manifest_authenticated
                    and isinstance(manifest.get("checkpoint_trigger_audit"), dict)
                    else None
                ),
                "v3_trigger_anchor_evidence": (
                    manifest.get("v3_trigger_anchor_evidence")
                    if manifest_authenticated
                    else None
                ),
            }
        )
        self._pending_transaction: dict[str, Any] | None = None
        self._next_transaction_id = 1

    def _capture_runtime(self) -> Any:
        """Capture all mutable policy state before one tentative prediction."""

        if not self.bimanual:
            return self.policy._capture_runtime_state()
        return {
            "left": self.policy.left._capture_runtime_state(),
            "right": self.policy.right._capture_runtime_state(),
            "last_left_action": deepcopy(self.policy._last_left_action),
            "last_right_action": deepcopy(self.policy._last_right_action),
        }

    def _restore_runtime(self, state: Any) -> None:
        if not self.bimanual:
            self.policy._restore_runtime_state(state)
            return
        self.policy.left._restore_runtime_state(state["left"])
        self.policy.right._restore_runtime_state(state["right"])
        self.policy._last_left_action = deepcopy(state["last_left_action"])
        self.policy._last_right_action = deepcopy(state["last_right_action"])

    def _resolve_transaction(self, request: dict[str, Any], *, commit: bool) -> dict[str, Any]:
        pending = self._pending_transaction
        if pending is None:
            raise RuntimeError("no policy action is awaiting commit or abort")
        transaction_id = request.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy transaction id must be an integer")
        if transaction_id != pending["transaction_id"]:
            raise RuntimeError("policy transaction id does not match the pending action")
        if not commit:
            self._restore_runtime(pending["runtime"])
        self._pending_transaction = None
        return {
            "ok": True,
            "transaction_id": transaction_id,
            "committed": bool(commit),
            "aborted": bool(not commit),
            "complete": self.policy.complete,
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "ping":
            if self.bimanual:
                policy_steps = max(
                    sum(skill.duration for skill in self.policy.left.skills),
                    sum(skill.duration for skill in self.policy.right.skills),
                )
            else:
                policy_steps = sum(skill.duration for skill in self.policy.skills)
            return {
                "ok": True,
                "ready": True,
                "task": self.task,
                "bimanual": self.bimanual,
                "policy_steps": policy_steps,
                "model_identity": self.model_identity,
                "policy_clock_semantics_id": POLICY_CLOCK_SEMANTICS_ID,
            }
        if command == "close":
            if self._pending_transaction is not None:
                self._restore_runtime(self._pending_transaction["runtime"])
                self._pending_transaction = None
            return {"ok": True, "closed": True}
        if command == "commit":
            return self._resolve_transaction(request, commit=True)
        if command == "abort":
            return self._resolve_transaction(request, commit=False)
        if command not in {"reset", "act"}:
            raise ValueError("command must be ping, reset, act, commit, abort, or close")
        observation = _WireObservation(request["observation"])
        if self.bimanual:
            left, right = bimanual_observations_from_rlbench(observation, self.task)
            if command == "reset":
                if self._pending_transaction is not None:
                    self._restore_runtime(self._pending_transaction["runtime"])
                    self._pending_transaction = None
                self.policy.reset(left, right, mode_strategy="map")
                return {"ok": True, "complete": self.policy.complete}
            if self._pending_transaction is not None:
                raise RuntimeError("the previous policy action still awaits commit or abort")
            if self.policy.complete:
                return {"ok": True, "complete": True, "action": None}
            runtime = self._capture_runtime()
            try:
                action = self.policy.act(left, right)
            except Exception:
                self._restore_runtime(runtime)
                raise
            wire_action = bimanual_action_to_rlbench(action).tolist()
        else:
            current = unimanual_observation_from_rlbench(observation, self.task)
            if command == "reset":
                if self._pending_transaction is not None:
                    self._restore_runtime(self._pending_transaction["runtime"])
                    self._pending_transaction = None
                self.policy.reset(current, mode_strategy="map")
                return {"ok": True, "complete": self.policy.complete}
            if self._pending_transaction is not None:
                raise RuntimeError("the previous policy action still awaits commit or abort")
            if self.policy.complete:
                return {"ok": True, "complete": True, "action": None}
            runtime = self._capture_runtime()
            try:
                action = self.policy.act(current)
            except Exception:
                self._restore_runtime(runtime)
                raise
            wire_action = unimanual_action_to_rlbench(action).tolist()
        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        self._pending_transaction = {
            "transaction_id": transaction_id,
            "runtime": runtime,
        }
        return {
            "ok": True,
            "complete": self.policy.complete,
            "complete_after_commit": self.policy.complete,
            "action": wire_action,
            "transaction_id": transaction_id,
        }


def serve(task: str, models_dir: Path) -> int:
    """Serve one policy over newline-delimited JSON on stdin/stdout."""

    server = PolicyServer(task, models_dir)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = server.handle(request)
        except Exception as exc:  # report request errors to the simulator process
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, separators=(",", ":")), flush=True)
        if response.get("closed"):
            return 0
    return 0


def _tasks(values: Sequence[str]) -> list[str]:
    available = load_task_specs()
    if not values or values == ["all"]:
        return [name for name, spec in available.items() if spec.bimanual]
    if values == ["all-unimanual"]:
        return [name for name, spec in available.items() if not spec.bimanual]
    unknown = sorted(set(values).difference(available))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; choose from {sorted(available)}")
    return list(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="fit current DynaMAC from saved demos")
    train.add_argument("--task", action="append", default=[], help="task name or all")
    train.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    train.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--demonstrations", type=int, default=5)

    worker = subparsers.add_parser("serve", help="run JSON-lines policy worker")
    worker.add_argument("--task", required=True)
    worker.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return serve(args.task, args.models_dir)
    for task in _tasks(args.task):
        summary = train_task(
            task,
            data_root=args.data_root,
            models_dir=args.models_dir,
            config_path=args.config,
            demonstration_count=args.demonstrations,
        )
        if summary["bimanual"]:
            detail = (
                f"left={summary['left']['durations']} "
                f"right={summary['right']['durations']}"
            )
        else:
            detail = f"durations={summary['durations']}"
        print(f"{task}: trained {len(summary['demonstrations'])} demonstrations; {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
