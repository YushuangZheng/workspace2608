from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluations.iclr2027.calibration.monitor import (
    episode_nonconformity,
    split_conformal_upper_threshold,
)
from evaluations.iclr2027.delivery.verify_b_delivery import verify_b_delivery
from evaluations.iclr2027.methods.registry import build_monitor, load_method_spec
from evaluations.iclr2027.runners.shadow import shadow_passthrough_action


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_persisted_episode_statistic_breaks_on_unavailable_cycle() -> None:
    scores = [0.1, 0.8, 0.9, 0.7, 0.95, 0.85]
    available = [True, True, False, True, True, True]
    assert episode_nonconformity(scores, available, 3) == 0.7


def test_split_conformal_threshold_uses_episode_independent_order_statistic() -> None:
    threshold, rank = split_conformal_upper_threshold(range(50), 0.05)
    assert rank == 49
    assert threshold == 48.0


def test_calibration_identity_is_enforced_by_monitor_factory() -> None:
    spec = load_method_spec("m2_trajectory_likelihood")
    artifact = {
        "schema": "essay2608.iclr2027.monitor-calibration.v1",
        "method_id": spec.method_id,
        "method_config_identity": {"sha256": spec.config_sha256},
        "tasks": {
            "close_jar": {"threshold": 0.5, "persistence_cycles": 5}
        },
    }
    monitor = build_monitor(spec, calibration=artifact, task_id="close_jar")
    assert monitor is not None and monitor.threshold == 0.5
    artifact["method_config_identity"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        build_monitor(spec, calibration=artifact, task_id="close_jar")


def test_shadow_output_has_contract_fields_and_cannot_change_action() -> None:
    class Monitor:
        threshold = 0.2
        persistence_count = 1
        output_metadata = {"source": "test"}

        def observe(self, observation, action, policy_state):
            self.seen = (observation, action, policy_state)

        def score(self):
            return {"score": 0.3}

        def alarm(self):
            return True

    feature = {
        "cycle": 7,
        "arms": {"single": {}},
        "task_state": [],
        "observation_timestamp": 7,
        "action_resolution": {},
        "action": [1.0, 2.0],
        "action_timestamp": 7,
        "policy_state": {},
    }
    action, diagnostic = shadow_passthrough_action(Monitor(), feature)
    assert action == feature["action"]
    assert diagnostic == {
        "cycle": 7,
        "scores": {"score": 0.3},
        "alarm": True,
        "threshold": 0.2,
        "persistence_count": 1,
        "metadata": {"source": "test"},
    }


def test_b_delivery_verifier_accepts_only_scoped_authenticated_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    method = root / "evaluations/iclr2027/methods/fail_detect/adapter.py"
    method.parent.mkdir(parents=True)
    method.write_text("# adapter\n", encoding="utf-8")
    contract = Path("evaluations/iclr2027/configs/shared/b_delivery_contract.json")
    artifact_contract = Path(
        "evaluations/iclr2027/configs/shared/artifact_contract.json"
    )
    manifest = tmp_path / "delivery.json"
    _write_json(
        manifest,
        {
            "schema": "essay2608.iclr2027.b-to-a-delivery.v1",
            "delivery_id": "test",
            "created_utc": "2026-09-05T00:00:00Z",
            "producer": "server_b",
            "base_commit": "0" * 40,
            "files": [
                {
                    "path": "evaluations/iclr2027/methods/fail_detect/adapter.py",
                    "bytes": method.stat().st_size,
                    "sha256": _sha256(method),
                }
            ],
            "checkpoint_manifests": [],
        },
    )
    result = verify_b_delivery(
        manifest,
        contract_path=contract,
        artifact_contract_path=artifact_contract,
        repository_root=root,
    )
    assert result["status"] == "PASS"
    assert result["copy_or_overwrite_performed"] is False

    forbidden = root / "evaluations/iclr2027/runners/changed.py"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("# forbidden\n", encoding="utf-8")
    _write_json(
        manifest,
        {
            "schema": "essay2608.iclr2027.b-to-a-delivery.v1",
            "delivery_id": "test-forbidden",
            "created_utc": "2026-09-05T00:00:00Z",
            "producer": "server_b",
            "base_commit": "0" * 40,
            "files": [
                {
                    "path": "evaluations/iclr2027/runners/changed.py",
                    "bytes": forbidden.stat().st_size,
                    "sha256": _sha256(forbidden),
                }
            ],
            "checkpoint_manifests": [],
        },
    )
    with pytest.raises(ValueError, match="outside B ownership"):
        verify_b_delivery(
            manifest,
            contract_path=contract,
            artifact_contract_path=artifact_contract,
            repository_root=root,
        )


def test_b_delivery_verifier_binds_checkpoint_to_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    checkpoint = (
        root
        / "evaluations/iclr2027/artifacts/checkpoints/m4/main10/budget_200/seed_0/model.pt"
    )
    config = root / "evaluations/iclr2027/configs/methods/m4_failure_supervised.json"
    checkpoint_manifest = checkpoint.with_name("checkpoint_manifest.json")
    checkpoint.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen weights")
    config.write_text("{}\n", encoding="utf-8")
    checkpoint_relative = str(checkpoint.relative_to(root))
    config_relative = str(config.relative_to(root))
    checkpoint_manifest_relative = str(checkpoint_manifest.relative_to(root))
    _write_json(
        checkpoint_manifest,
        {
            "schema": "essay2608.iclr2027.monitor-checkpoint.v1",
            "method_id": "m4_failure_supervised",
            "checkpoint_relative_path": checkpoint_relative,
            "checkpoint_sha256": _sha256(checkpoint),
            "config_relative_path": config_relative,
            "config_sha256": _sha256(config),
            "training_budget": 200,
            "training_seed": 0,
            "held_out_family": None,
            "feature_schema": "essay2608.iclr2027.causal-features.v1",
        },
    )
    files = []
    for path in (checkpoint, config, checkpoint_manifest):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    delivery = tmp_path / "checkpoint_delivery.json"
    _write_json(
        delivery,
        {
            "schema": "essay2608.iclr2027.b-to-a-delivery.v1",
            "delivery_id": "m4-test",
            "created_utc": "2026-09-05T00:00:00Z",
            "producer": "server_b",
            "base_commit": "0" * 40,
            "files": files,
            "checkpoint_manifests": [checkpoint_manifest_relative],
        },
    )
    result = verify_b_delivery(
        delivery,
        contract_path=Path(
            "evaluations/iclr2027/configs/shared/b_delivery_contract.json"
        ),
        artifact_contract_path=Path(
            "evaluations/iclr2027/configs/shared/artifact_contract.json"
        ),
        repository_root=root,
    )
    assert result["checkpoint_manifests_verified"] == 1

    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_b_delivery(
            delivery,
            contract_path=Path(
                "evaluations/iclr2027/configs/shared/b_delivery_contract.json"
            ),
            artifact_contract_path=Path(
                "evaluations/iclr2027/configs/shared/artifact_contract.json"
            ),
            repository_root=root,
        )
