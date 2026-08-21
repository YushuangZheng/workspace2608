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
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    DYNAMAC_GRIPPER_TARGET_TIMING,
    DYNAMAC_POSE_TARGET_TIMING,
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.core.gripper_timing import (
    GLOBAL_GRIPPER_TIMING_PROTOCOL_ID,
    apply_global_gripper_timing,
    global_gripper_timing_metadata,
    native_gripper_to_wire,
)
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json, reserve_output
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    bimanual_action_to_rlbench,
    bimanual_observations_from_rlbench,
    unimanual_action_to_rlbench,
    unimanual_observation_from_rlbench,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import TaskSpec, get_task_spec, load_task_specs
from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
    bimanual_checkpoint_trigger_audit,
    build_v3_trigger_anchor_evidence,
    checkpoint_trigger_audit,
)

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
DEFAULT_DATA_ROOT = (
    INTEGRATION_ROOT
    / "data"
    / "training"
    / "main"
)
# The current models and training manifest authenticate this immutable
# configuration by its release path and SHA-256.
DEFAULT_CONFIG = INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v3.json"
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3"
TRAINING_MANIFEST_SCHEMA_V2 = "dynamac-direct-training-v2"
TRAINING_MANIFEST_SCHEMA_V3 = "dynamac-direct-training-v3"
TRAINING_MANIFEST_SCHEMA_V4 = "dynamac-direct-training-v4"
AUTHENTICATED_TRAINING_MANIFEST_SCHEMAS = frozenset(
    {
        TRAINING_MANIFEST_SCHEMA_V2,
        TRAINING_MANIFEST_SCHEMA_V3,
        TRAINING_MANIFEST_SCHEMA_V4,
    }
)
V3_ADAPTER_PROTOCOL = {
    "schema": "rlbench-dynamac-demo-adapter-v3",
    "pose_target_timing": DYNAMAC_POSE_TARGET_TIMING,
    "gripper_action_timing": DYNAMAC_GRIPPER_TARGET_TIMING,
    "action_timing": "obs[t] current state",
    "pose_and_gripper_sample_aligned": True,
}
POLICY_CLOCK_SEMANTICS_ID = (
    "policy-tick-single-primary-request-then-raw-joint-hold-commit-v1"
)


def v4_quaternion_batch_gauge_identity() -> dict[str, str]:
    """Return the training-only quaternion gauge bound into V4 manifests.

    Existing schema-13 checkpoints contain all fitted arrays and keep their
    historical load/inference behavior.  This identity distinguishes newly
    fitted V4 artifacts without changing that read-only checkpoint contract.
    """

    from essay2608.policy import QUATERNION_BATCH_GAUGE_PROTOCOL_ID

    return {
        "protocol_id": QUATERNION_BATCH_GAUGE_PROTOCOL_ID,
        "application": "training_pose_batch_preprocessing",
        "legacy_checkpoint_compatibility": (
            "existing_schema13_arrays_load_without_refit_or_runtime_change"
        ),
    }


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


def task_demonstration_paths(task_data_dir: Path, count: int = 5) -> list[Path]:
    """Return demos when ``task_data_dir`` is already the task directory.

    Historical releases pass a shared data root to :func:`demonstration_paths`.
    A task-scoped release may instead version one task directly, without adding
    a redundant second task-name directory.
    """

    if count < 1:
        raise ValueError("demonstration count must be positive")
    episode_root = task_data_dir / "all_variations" / "episodes"
    paths = sorted(episode_root.glob("episode*/low_dim_obs.pkl"), key=_episode_number)
    if len(paths) < count:
        raise FileNotFoundError(
            f"found {len(paths)} low_dim_obs.pkl files under {episode_root}; "
            f"need {count}"
        )
    return paths[:count]


def _resolve_task_spec(task: str, task_spec: TaskSpec | None) -> TaskSpec:
    spec = get_task_spec(task) if task_spec is None else task_spec
    if spec.task_name != task:
        raise ValueError(
            f"injected task spec names {spec.task_name!r}, expected {task!r}"
        )
    return spec


def _v4_store_policy_spec_identity() -> tuple[TaskSpec, dict[str, Any]]:
    """Load the tracked V4 StoreBottle spec without affecting V3 defaults."""

    from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
        store_bottle_policy_spec_identity,
        store_bottle_semantic_task_spec,
    )

    return store_bottle_semantic_task_spec(), store_bottle_policy_spec_identity()


def _resolve_manifest_task_spec(
    task: str,
    manifest: Mapping[str, Any],
    task_spec: TaskSpec | None,
) -> TaskSpec:
    if manifest.get("manifest_schema") != TRAINING_MANIFEST_SCHEMA_V4:
        return _resolve_task_spec(task, task_spec)
    current, expected_identity = _v4_store_policy_spec_identity()
    if task != current.task_name:
        raise RuntimeError("V4 training manifests are currently StoreBottle-only")
    training_identity = manifest.get("training_identity")
    policy_identity = (
        training_identity.get("policy_spec")
        if isinstance(training_identity, dict)
        else None
    )
    if policy_identity != expected_identity:
        raise RuntimeError("V4 StoreBottle policy spec identity mismatch")
    if task_spec is not None and task_spec != current:
        raise RuntimeError("injected V4 task spec differs from its authenticated manifest")
    return current


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
    task_spec: TaskSpec | None = None,
    task_data_dir: Path | None = None,
    manifest_schema: str = TRAINING_MANIFEST_SCHEMA_V3,
    training_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit, validate, and atomically publish one complete task model."""

    from integrations.rlbench.rlbench_dynamac.eval.evaluation_split import validate_training_entry_paths

    validate_training_entry_paths(data_root, models_dir, config_path)
    if task_data_dir is not None:
        validate_training_entry_paths(task_data_dir)
    if manifest_schema not in AUTHENTICATED_TRAINING_MANIFEST_SCHEMAS:
        raise ValueError(f"unsupported training manifest schema: {manifest_schema}")
    if manifest_schema == TRAINING_MANIFEST_SCHEMA_V4 and training_identity is None:
        raise ValueError("V4 training requires an authenticated training identity")

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
                task_spec=task_spec,
                task_data_dir=task_data_dir,
                manifest_schema=manifest_schema,
                training_identity=training_identity,
            )
            _validate_published_model(
                task,
                staging,
                summary,
                expected_training_identity=training_identity,
            )
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
    task_spec: TaskSpec | None = None,
    task_data_dir: Path | None = None,
    manifest_schema: str = TRAINING_MANIFEST_SCHEMA_V3,
    training_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit all artifacts inside an unpublished staging directory."""

    from essay2608.policy import BimanualDynaMAC, DynaMAC

    spec = _resolve_task_spec(task, task_spec)
    paths = (
        demonstration_paths(data_root, task, demonstration_count)
        if task_data_dir is None
        else task_demonstration_paths(task_data_dir, demonstration_count)
    )
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
    summary["manifest_schema"] = manifest_schema
    summary["checkpoint_trigger_audit"] = checkpoint_audit
    if training_identity is not None:
        summary["training_identity"] = deepcopy(dict(training_identity))
    if manifest_schema == TRAINING_MANIFEST_SCHEMA_V4:
        summary["quaternion_batch_gauge"] = v4_quaternion_batch_gauge_identity()
    if manifest_schema == TRAINING_MANIFEST_SCHEMA_V3:
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
    *,
    expected_training_identity: Mapping[str, Any] | None = None,
) -> None:
    """Reload staged checkpoints and bind them to their manifest fingerprints."""

    from essay2608.policy import BimanualDynaMAC, DynaMAC, DynaMACConfig

    if summary.get("task") != task:
        raise RuntimeError("staged training manifest has the wrong task identity")
    manifest_schema = summary.get("manifest_schema")
    if manifest_schema not in AUTHENTICATED_TRAINING_MANIFEST_SCHEMAS:
        raise RuntimeError("staged training manifest schema is not authenticated")
    if manifest_schema in {
        TRAINING_MANIFEST_SCHEMA_V3,
        TRAINING_MANIFEST_SCHEMA_V4,
    }:
        _validate_v3_adapter_protocol(summary)
    if manifest_schema == TRAINING_MANIFEST_SCHEMA_V4:
        identity = summary.get("training_identity")
        if not isinstance(identity, dict):
            raise RuntimeError("staged V4 manifest is missing its training identity")
        gauge_identity = v4_quaternion_batch_gauge_identity()
        if summary.get("quaternion_batch_gauge") != gauge_identity:
            raise RuntimeError("staged V4 manifest has the wrong quaternion batch gauge")
        if identity.get("quaternion_batch_gauge") != gauge_identity:
            raise RuntimeError("staged V4 training identity has the wrong quaternion batch gauge")
        _resolve_manifest_task_spec(task, summary, None)
        if expected_training_identity is not None and identity != dict(
            expected_training_identity
        ):
            raise RuntimeError("staged V4 training identity mismatch")
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

    def __init__(
        self,
        task: str,
        models_dir: Path,
        *,
        task_spec: TaskSpec | None = None,
        expected_training_identity: Mapping[str, Any] | None = None,
    ) -> None:
        from essay2608.policy import BimanualDynaMAC, DynaMAC

        self.task = task
        task_dir = models_dir / task
        manifest_path = task_dir / "training.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"training manifest must be a JSON object: {manifest_path}")
        if manifest.get("task") not in {None, task}:
            raise ValueError("training manifest task does not match the requested policy")
        spec = _resolve_manifest_task_spec(task, manifest, task_spec)
        self.task_spec = spec
        self.bimanual = spec.bimanual
        if manifest.get("bimanual") not in {None, self.bimanual}:
            raise ValueError("training manifest arm count does not match the task")
        manifest_authenticated = (
            manifest.get("manifest_schema")
            in AUTHENTICATED_TRAINING_MANIFEST_SCHEMAS
        )
        if manifest_authenticated:
            _validate_published_model(
                task,
                task_dir,
                manifest,
                expected_training_identity=expected_training_identity,
            )
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
        if "training_identity" in manifest:
            self.model_identity["training_identity"] = deepcopy(
                manifest["training_identity"]
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

    def _apply_global_gripper_timing(self, wire_action, preview):
        """Apply the same learned-boundary rule to every task and arm."""

        original = np.asarray(wire_action, dtype=np.float64)
        if self.bimanual:
            by_index = {7: preview.right, 16: preview.left}
        else:
            by_index = {7: preview}
        next_commands = {
            index: native_gripper_to_wire(item.gripper)
            for index, item in by_index.items()
        }
        boundaries = {
            index: bool(item.crosses_skill_boundary)
            for index, item in by_index.items()
        }
        emitted = apply_global_gripper_timing(
            original,
            next_wire_gripper_by_index=next_commands,
            crosses_skill_boundary_by_index=boundaries,
        )
        return emitted, {
            "protocol_id": GLOBAL_GRIPPER_TIMING_PROTOCOL_ID,
            "changed_action_indices": np.flatnonzero(emitted != original).tolist(),
            "next_wire_gripper_by_action_index": {
                str(index): value for index, value in next_commands.items()
            },
            "crosses_skill_boundary_by_action_index": {
                str(index): value for index, value in boundaries.items()
            },
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
                "gripper_timing": global_gripper_timing_metadata(),
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
            left, right = bimanual_observations_from_rlbench(
                observation,
                getattr(self, "task_spec", self.task),
            )
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
                gripper_preview = self.policy.preview_next_gripper()
                action = self.policy.act(left, right)
                wire_action, gripper_timing = self._apply_global_gripper_timing(
                    bimanual_action_to_rlbench(action), gripper_preview
                )
            except Exception:
                self._restore_runtime(runtime)
                raise
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
                gripper_preview = self.policy.preview_next_gripper()
                action = self.policy.act(current)
                wire_action, gripper_timing = self._apply_global_gripper_timing(
                    unimanual_action_to_rlbench(action), gripper_preview
                )
            except Exception:
                self._restore_runtime(runtime)
                raise
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
            "action": wire_action.tolist(),
            "transaction_id": transaction_id,
            "gripper_timing": gripper_timing,
        }


def serve(
    task: str,
    models_dir: Path,
    *,
    task_spec: TaskSpec | None = None,
    expected_training_identity: Mapping[str, Any] | None = None,
) -> int:
    """Serve one policy over newline-delimited JSON on stdin/stdout."""

    server = PolicyServer(
        task,
        models_dir,
        task_spec=task_spec,
        expected_training_identity=expected_training_identity,
    )
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
