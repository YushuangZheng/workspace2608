from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from essay2608.policy import (
    DynaMAC,
    DynaMACConfig,
    QUATERNION_BATCH_GAUGE_PROTOCOL_ID,
)

from integrations.rlbench.rlbench_dynamac.data import direct_policy
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    PolicyServer,
    TRAINING_MANIFEST_SCHEMA_V4,
    _resolve_manifest_task_spec,
    v4_quaternion_batch_gauge_identity,
)
from integrations.rlbench.rlbench_dynamac.data.store_bottle_release_v4 import (
    INHERITED_MODE,
    RETRAINED_MODE,
    SWEEP_RETRAINED_MODE,
    build_model_release_manifest,
    load_model_release_plan,
    sweep_training_input_identity,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    STORE_BOTTLE_TASK_NAME,
    store_bottle_policy_spec_identity,
)
from integrations.rlbench.rlbench_dynamac.data.store_bottle_v4 import (
    COLLECTION_MANIFEST_SCHEMA,
    DEFAULT_POLICY_CONFIG,
    TRAINING_IDENTITY_SCHEMA,
    _canonical_sha256,
    _policy_spec_identity_from_config,
    build_store_bottle_training_identity,
    collection_dry_run,
    load_store_bottle_collection_manifest,
    load_store_bottle_training_protocol,
    train_store_bottle_v4,
    training_dry_run,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import get_task_spec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_collection(root: Path) -> dict:
    episodes = []
    for episode in range(5):
        directory = root / "all_variations" / "episodes" / f"episode{episode}"
        directory.mkdir(parents=True)
        low_dim = directory / "low_dim_obs.pkl"
        variation = directory / "variation_number.pkl"
        low_dim.write_bytes(f"demo-{episode}".encode())
        variation.write_bytes(b"variation-0")
        episodes.append(
            {
                "episode": episode,
                "seed": 4104000000 + episode,
                "variation": 0,
                "observations": 2,
                "success_verified": True,
                "files": {
                    "low_dim_obs": {
                        "path": f"all_variations/episodes/episode{episode}/low_dim_obs.pkl",
                        "bytes": low_dim.stat().st_size,
                        "sha256": _sha256(low_dim),
                    },
                    "variation_number": {
                        "path": f"all_variations/episodes/episode{episode}/variation_number.pkl",
                        "bytes": variation.stat().st_size,
                        "sha256": _sha256(variation),
                    },
                },
            }
        )
    policy_spec = store_bottle_policy_spec_identity()
    protocol = load_store_bottle_training_protocol()
    manifest = {
        "schema": COLLECTION_MANIFEST_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "semantic_version": policy_spec["semantic_version"],
        "semantic_fingerprint": policy_spec["semantic_fingerprint"],
        "policy_spec": policy_spec,
        "scenario": "static",
        "environment_intervention": "none",
        "demonstrations": 5,
        "base_seed": 4104000000,
        "variation": 0,
        "success_authority": protocol["collection"]["success_authority"],
        "evaluation_artifacts_included": False,
        "episodes": episodes,
    }
    manifest["fingerprint"] = _canonical_sha256(manifest)
    (root / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_protocol_binds_disjoint_g5_seeds_and_one_policy_spec() -> None:
    protocol = load_store_bottle_training_protocol()

    assert protocol["collection"]["demonstrations"] == 5
    assert protocol["collection"]["base_seed"] == 4104000000
    assert protocol["split"]["training_seed_range"] == [4104000000, 4104000004]
    assert protocol["split"]["evaluation_seed_start"] == 2608000000
    assert protocol["training"]["other_tasks_trained"] is False
    assert _policy_spec_identity_from_config(protocol) == (
        store_bottle_policy_spec_identity()
    )
    assert store_bottle_policy_spec_identity()["frame_objects"] == [
        {"frame": "bottle", "scene_object": "bottle"},
        {"frame": "fridge", "scene_object": "fridge_base"},
    ]


def test_collection_and_training_dry_runs_write_nothing(tmp_path: Path) -> None:
    data = tmp_path / "data" / "store_bottle"
    models = tmp_path / "models" / "v4"

    collection = collection_dry_run(data)
    smoke = collection_dry_run(tmp_path / "smoke", smoke=True)
    training = training_dry_run(data, models, DEFAULT_POLICY_CONFIG)

    assert collection["seeds"] == list(range(4104000000, 4104000005))
    assert collection["simulator_launched"] is False
    assert collection["files_written"] is False
    assert smoke["demonstrations"] == 1
    assert smoke["official_training_input"] is False
    assert training["tasks_trained"] == [STORE_BOTTLE_TASK_NAME]
    assert training["other_tasks_trained"] is False
    assert training["models_written"] is False
    assert not data.exists()
    assert not models.exists()


def test_collection_loader_authenticates_all_five_successes_and_hashes(
    tmp_path: Path,
) -> None:
    expected = _write_collection(tmp_path)

    loaded = load_store_bottle_collection_manifest(
        tmp_path / "collection_manifest.json"
    )

    assert loaded == expected
    assert [row["seed"] for row in loaded["episodes"]] == list(
        range(4104000000, 4104000005)
    )
    target = tmp_path / loaded["episodes"][2]["files"]["low_dim_obs"]["path"]
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_store_bottle_collection_manifest(
            tmp_path / "collection_manifest.json"
        )


def test_training_identity_binds_manifest_config_and_v4_frames(tmp_path: Path) -> None:
    collection = _write_collection(tmp_path)

    identity = build_store_bottle_training_identity(tmp_path, DEFAULT_POLICY_CONFIG)

    assert identity["schema"] == TRAINING_IDENTITY_SCHEMA
    assert identity["policy_spec"] == store_bottle_policy_spec_identity()
    assert identity["collection"]["manifest_fingerprint"] == collection["fingerprint"]
    assert identity["collection"]["all_success_verified"] is True
    assert identity["tasks_trained"] == [STORE_BOTTLE_TASK_NAME]
    assert identity["other_tasks_trained"] is False
    assert identity["policy_config"]["sha256"] == _sha256(DEFAULT_POLICY_CONFIG)
    assert identity["quaternion_batch_gauge"] == v4_quaternion_batch_gauge_identity()
    assert (
        identity["quaternion_batch_gauge"]["protocol_id"]
        == QUATERNION_BATCH_GAUGE_PROTOCOL_ID
    )


def test_existing_v3_schema13_checkpoint_stays_read_only_load_compatible() -> None:
    integration_root = DEFAULT_POLICY_CONFIG.parents[2]
    task_root = integration_root / "models" / "v3" / STORE_BOTTLE_TASK_NAME
    if not all((task_root / name).is_file() for name in ("left.npz", "training.json")):
        pytest.skip("optional released V3 checkpoint is not present in this checkout")
    manifest = json.loads((task_root / "training.json").read_text(encoding="utf-8"))

    left = DynaMAC.load(task_root / "left.npz")

    assert left.summary()["model_schema_version"] == 13
    assert left.fingerprint() == manifest["left"]["fingerprint"]


def test_train_entry_injects_only_the_store_v4_spec(monkeypatch, tmp_path: Path) -> None:
    identity = {
        "schema": TRAINING_IDENTITY_SCHEMA,
        "policy_spec": store_bottle_policy_spec_identity(),
    }
    captured = {}

    monkeypatch.setattr(
        (
            "integrations.rlbench.rlbench_dynamac.data.store_bottle_v4."
            "build_store_bottle_training_identity"
        ),
        lambda *_args: identity,
    )

    def fake_train(task, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return {"task": task, "bimanual": True}

    monkeypatch.setattr(direct_policy, "train_task", fake_train)
    result = train_store_bottle_v4(tmp_path / "data", tmp_path / "models", DEFAULT_POLICY_CONFIG)

    assert result["task"] == STORE_BOTTLE_TASK_NAME
    assert captured["demonstration_count"] == 5
    assert captured["task_data_dir"] == tmp_path / "data"
    assert captured["models_dir"] == tmp_path / "models"
    assert captured["manifest_schema"] == TRAINING_MANIFEST_SCHEMA_V4
    assert captured["training_identity"] is identity
    assert captured["task_spec"].frame_names == ("bottle", "fridge")


def test_v4_manifest_selects_corrected_spec_but_v3_remains_legacy() -> None:
    identity = {"policy_spec": store_bottle_policy_spec_identity()}

    selected = _resolve_manifest_task_spec(
        STORE_BOTTLE_TASK_NAME,
        {
            "manifest_schema": TRAINING_MANIFEST_SCHEMA_V4,
            "training_identity": identity,
        },
        None,
    )
    legacy = _resolve_manifest_task_spec(
        STORE_BOTTLE_TASK_NAME,
        {"manifest_schema": "dynamac-direct-training-v3"},
        None,
    )

    assert selected.frame_names == ("bottle", "fridge")
    assert legacy == get_task_spec(STORE_BOTTLE_TASK_NAME)
    assert legacy.frame_names == ("bottle", "fridge_root")
    forged = json.loads(json.dumps(identity))
    forged["policy_spec"]["frame_names"][1] = "fridge_root"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _resolve_manifest_task_spec(
            STORE_BOTTLE_TASK_NAME,
            {
                "manifest_schema": TRAINING_MANIFEST_SCHEMA_V4,
                "training_identity": forged,
            },
            None,
        )


class _FakeArm:
    def __init__(self) -> None:
        self.skills = []
        self.config = DynaMACConfig()

    def summary(self):
        return {
            "model_schema_version": "fake",
            "selection_semantics_id": "fake",
            "tapas_reference_commit": "fake",
        }

    def fingerprint(self):
        return "fake"


class _FakeBimanual:
    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right
        self.complete = False
        self.seen_reset = None

    def reset(self, left, right, mode_strategy):
        self.seen_reset = (left, right, mode_strategy)


class _FakeDynaMAC:
    @staticmethod
    def load(_path):
        return _FakeArm()


def test_policy_server_uses_authenticated_v4_spec_online(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / STORE_BOTTLE_TASK_NAME
    model.mkdir()
    training_identity = {"policy_spec": store_bottle_policy_spec_identity()}
    (model / "training.json").write_text(
        json.dumps(
            {
                "manifest_schema": TRAINING_MANIFEST_SCHEMA_V4,
                "task": STORE_BOTTLE_TASK_NAME,
                "bimanual": True,
                "training_identity": training_identity,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(direct_policy, "_validate_published_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("essay2608.policy.DynaMAC", _FakeDynaMAC)
    monkeypatch.setattr("essay2608.policy.BimanualDynaMAC", _FakeBimanual)
    seen = {}

    def fake_observations(_observation, spec):
        seen["spec"] = spec
        return object(), object()

    monkeypatch.setattr(direct_policy, "bimanual_observations_from_rlbench", fake_observations)
    server = PolicyServer(STORE_BOTTLE_TASK_NAME, tmp_path)
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    response = server.handle(
        {
            "command": "reset",
            "observation": {
                "left": {"gripper_pose": pose},
                "right": {"gripper_pose": pose},
                "task_low_dim_state": pose * 2,
            },
        }
    )

    assert response == {"ok": True, "complete": False}
    assert server.task_spec.frame_names == ("bottle", "fridge")
    assert seen["spec"] == server.task_spec
    assert server.model_identity["training_identity"] == training_identity


def test_release_plan_retrains_store_and_sweep_then_inherits_everything_else(
    tmp_path: Path,
) -> None:
    plan = load_model_release_plan()
    source = tmp_path / "v3"
    target = tmp_path / "v4"
    for entry in plan["entries"]:
        if entry["mode"] != INHERITED_MODE:
            continue
        directory = source / entry["source"]
        directory.mkdir(parents=True)
        for name in entry["required_artifacts"]:
            (directory / name).write_bytes((entry["model_id"] + name).encode())

    manifest = build_model_release_manifest(
        source_models_dir=source,
        target_models_dir=target,
    )

    store = next(row for row in manifest["entries"] if row["mode"] == RETRAINED_MODE)
    sweep = next(
        row for row in manifest["entries"] if row["mode"] == SWEEP_RETRAINED_MODE
    )
    inherited = [row for row in manifest["entries"] if row["mode"] == INHERITED_MODE]
    assert store["model_id"] == STORE_BOTTLE_TASK_NAME
    assert store["status"] == "pending_store_bottle_v4_training"
    assert sweep["model_id"] == "bimanual_sweep_to_dustpan"
    assert sweep["status"] == "pending_sweep_dust_v4_training"
    assert len(inherited) == 7
    assert all(row["status"] == "pending_byte_copy_from_v3" for row in inherited)
    assert manifest["copy_or_training_performed"] is False
    assert manifest["complete"] is False
    assert not target.exists()


def test_release_manifest_accepts_sweep_bound_to_augmented_inputs_without_v3(
    tmp_path: Path,
) -> None:
    plan = load_model_release_plan()
    source = tmp_path / "v3"
    target = tmp_path / "v4"
    for entry in plan["entries"]:
        if entry["mode"] != INHERITED_MODE:
            continue
        directory = source / entry["source"]
        directory.mkdir(parents=True)
        for name in entry["required_artifacts"]:
            (directory / name).write_bytes((entry["model_id"] + name).encode())

    identity = sweep_training_input_identity()
    sweep_dir = target / "bimanual_sweep_to_dustpan"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "left.npz").write_bytes(b"new-left")
    (sweep_dir / "right.npz").write_bytes(b"new-right")
    augmentation = {
        "schema": identity["schema"],
        "manifest_path": identity["manifest_path"],
        "manifest_sha256": identity["manifest_sha256"],
        "data_root": identity["data_root"],
        "episodes": identity["inputs"],
        "status_at_fit": identity["status_at_fit"],
    }
    (sweep_dir / "training.json").write_text(
        json.dumps(
            {
                "manifest_schema": "dynamac-direct-training-v3",
                "task": "bimanual_sweep_to_dustpan",
                "bimanual": True,
                "demonstrations": [f"episode{episode}" for episode in range(5)],
                "training_data_augmentation": augmentation,
            }
        ),
        encoding="utf-8",
    )

    manifest = build_model_release_manifest(
        source_models_dir=source,
        target_models_dir=target,
    )
    sweep = next(
        row for row in manifest["entries"] if row["mode"] == SWEEP_RETRAINED_MODE
    )
    assert sweep["status"] == "retrained_sweep_dust_v4_verified"
    assert "source_inventory" not in sweep
