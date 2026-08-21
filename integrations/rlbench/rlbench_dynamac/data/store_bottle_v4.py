"""Reproducible StoreBottle V4 collection, training, and policy serving.

Collection is the only command that imports RLBench/PyRep.  ``--dry-run`` for
collection and training performs no simulator launch and writes no artifacts.
The official protocol is closed to five successful static demonstrations; it
does not accept evaluation-set or result files as inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
TRAINING_PROTOCOL_PATH = (
    INTEGRATION_ROOT / "configs" / "v4" / "store_bottle_training.json"
)
DEFAULT_DATA_DIR = (
    INTEGRATION_ROOT
    / "data"
    / "training"
    / "main"
    / "bimanual_put_bottle_in_fridge"
)
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v4"
DEFAULT_POLICY_CONFIG = (
    INTEGRATION_ROOT / "configs" / "v4" / "dynamac_store_bottle.json"
)
COLLECTION_MANIFEST_SCHEMA = (
    "rlbench-store-bottle-v4-static-demonstrations-v1"
)
SMOKE_MANIFEST_SCHEMA = "rlbench-store-bottle-v4-static-smoke-v1"
TRAINING_IDENTITY_SCHEMA = "rlbench-store-bottle-training-identity-v4"
COLLECTION_PLAN_SCHEMA = "rlbench-store-bottle-collection-plan-v4"
TRAINING_PLAN_SCHEMA = "rlbench-store-bottle-training-plan-v4"


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: %s" % path)
    return value


def _repository_path(relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError("%s must be a repository-relative path" % label)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("%s must stay inside the repository" % label)
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("%s escapes the repository" % label) from exc
    return resolved


def _reject_evaluation_or_result_path(path, label):
    resolved = Path(path).resolve()
    for forbidden in (
        INTEGRATION_ROOT / "evaluation_sets",
        INTEGRATION_ROOT / "results",
    ):
        try:
            resolved.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise ValueError("%s may not use evaluation_sets or results" % label)


def load_store_bottle_training_protocol(path=TRAINING_PROTOCOL_PATH):
    """Load the closed five-demo StoreBottle V4 protocol."""

    protocol_path = Path(path)
    value = _load_object(protocol_path)
    if set(value) != {
        "schema",
        "protocol_id",
        "release",
        "semantic_config",
        "task",
        "collection",
        "paths",
        "training",
        "split",
    }:
        raise ValueError("StoreBottle V4 training protocol fields are invalid")
    task = value.get("task")
    collection = value.get("collection")
    paths = value.get("paths")
    training = value.get("training")
    split = value.get("split")
    if (
        value.get("schema") != "rlbench-store-bottle-training-protocol-v4"
        or value.get("protocol_id")
        != "store_bottle_static_g5_seed4104000000_v4"
        or value.get("release") != "v4"
        or not isinstance(task, dict)
        or task
        != {
            "task_name": "bimanual_put_bottle_in_fridge",
            "module": "integrations.rlbench.rlbench_dynamac.store_bottle_live_v4",
            "class_name": "BimanualPutBottleInFridgeSemanticV4",
        }
        or not isinstance(collection, dict)
        or collection.get("scenario") != "static"
        or collection.get("environment_intervention") != "none"
        or collection.get("demonstrations") != 5
        or collection.get("base_seed") != 4104000000
        or collection.get("variation") != 0
        or collection.get("max_demo_attempts") != 10
        or collection.get("robot_setup") != "dual_panda"
        or collection.get("static_positions") is not False
        or not isinstance(paths, dict)
        or paths.get("data_directory")
        != (
            "integrations/rlbench/data/training/main/"
            "bimanual_put_bottle_in_fridge"
        )
        or paths.get("policy_config")
        != "integrations/rlbench/configs/v4/dynamac_store_bottle.json"
        or paths.get("models_directory") != "integrations/rlbench/models/v4"
        or paths.get("model_directory")
        != "integrations/rlbench/models/v4/bimanual_put_bottle_in_fridge"
        or not isinstance(training, dict)
        or training.get("manifest_schema") != "dynamac-direct-training-v4"
        or training.get("pose_frames") != ["bottle", "fridge"]
        or training.get("online_pose_frames") != ["bottle", "fridge"]
        or training.get("other_tasks_trained") is not False
        or not isinstance(split, dict)
        or split.get("evaluation_set_id") != "rlbench_eval_v2"
        or split.get("evaluation_seed_start") != 2608000000
        or split.get("training_seed_range") != [4104000000, 4104000004]
        or split.get("training_and_evaluation_seeds_disjoint") is not True
        or split.get("evaluation_artifacts_allowed_as_training_inputs") is not False
    ):
        raise ValueError("StoreBottle V4 training protocol is invalid")
    if collection["base_seed"] + collection["demonstrations"] - 1 >= 2**32 - 1:
        raise ValueError("StoreBottle training seeds exceed NumPy's seed range")
    for label, relative in (
        ("semantic_config", value["semantic_config"]),
        ("data_directory", paths["data_directory"]),
        ("policy_config", paths["policy_config"]),
        ("models_directory", paths["models_directory"]),
        ("model_directory", paths["model_directory"]),
        ("release_plan", paths["release_plan"]),
    ):
        resolved = _repository_path(relative, label)
        if label in {"data_directory", "policy_config", "models_directory", "model_directory"}:
            _reject_evaluation_or_result_path(resolved, label)
    return value


def _semantic_payload(protocol):
    payload = _load_object(
        _repository_path(protocol["semantic_config"], "semantic_config")
    )
    if (
        payload.get("schema") != "rlbench-store-bottle-semantic-scene-v4"
        or payload.get("semantic_version") != "store_bottle_clean_v4"
        or payload.get("task", {}).get("module") != protocol["task"]["module"]
        or payload.get("task", {}).get("class_name")
        != protocol["task"]["class_name"]
        or [item.get("name") for item in payload.get("observation", {}).get("pose_chunks", [])]
        != ["bottle", "fridge"]
        or [
            item.get("scene_object_name")
            for item in payload.get("observation", {}).get("pose_chunks", [])
        ]
        != ["bottle", "fridge_base"]
    ):
        raise ValueError("training protocol and StoreBottle semantics disagree")
    return payload


def _policy_spec_identity_from_config(protocol):
    from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import store_bottle_semantic_fingerprint

    semantic = _semantic_payload(protocol)
    chunks = semantic["observation"]["pose_chunks"]
    return {
        "schema": "rlbench-store-bottle-policy-spec-v4",
        "semantic_schema": semantic["schema"],
        "semantic_version": semantic["semantic_version"],
        "semantic_fingerprint": store_bottle_semantic_fingerprint(semantic),
        "task": semantic["task"]["task_name"],
        "paper_task_name": semantic["task"]["paper_task_name"],
        "module": semantic["task"]["module"],
        "class_name": semantic["task"]["class_name"],
        "bimanual": True,
        "frame_names": [item["name"] for item in chunks],
        "frame_objects": [
            {"frame": item["name"], "scene_object": item["scene_object_name"]}
            for item in chunks
        ],
        "pose_chunks": [
            {
                "name": item["name"],
                "role": item["role"],
                "index": index,
                "source_slice": [7 * index, 7 * (index + 1)],
            }
            for index, item in enumerate(chunks)
        ],
        "expected_low_dim_size": 14,
        "source_expression": semantic["observation"]["source_expression"],
        "source_status": semantic["observation"]["source_status"],
        "candidate_frame_policy": "ALL_GET_LOW_DIM_STATE_FRAMES_IN_SOURCE_ORDER",
        "candidate_frame_policy_source_status": "AUTHOR_EMAIL_EXPLICIT_20260814",
        "segmentation_coordination": "independent",
        "segmentation_coordination_source_status": (
            "AUTHOR_EMAIL_EXPLICIT_STOREBOTTLE_INDEPENDENT_20260814"
        ),
        "segmentation_debug_plots_required": False,
    }


def collection_dry_run(
    data_dir=DEFAULT_DATA_DIR,
    headless=True,
    protocol_path=TRAINING_PROTOCOL_PATH,
    smoke=False,
):
    protocol = load_store_bottle_training_protocol(protocol_path)
    data_dir = Path(data_dir)
    _reject_evaluation_or_result_path(data_dir, "collection output")
    return {
        "schema": COLLECTION_PLAN_SCHEMA,
        "dry_run": True,
        "protocol_id": protocol["protocol_id"] + ("-smoke" if smoke else ""),
        "task": protocol["task"],
        "policy_spec": _policy_spec_identity_from_config(protocol),
        "scenario": "static",
        "environment_intervention": "none",
        "data_directory": str(data_dir),
        "demonstrations": 1 if smoke else 5,
        "seeds": [4104000000] if smoke else list(range(4104000000, 4104000005)),
        "variation": 0,
        "headless": bool(headless),
        "simulator_launched": False,
        "files_written": False,
        "official_training_input": not smoke,
    }


def training_dry_run(
    data_dir=DEFAULT_DATA_DIR,
    models_dir=DEFAULT_MODELS_DIR,
    policy_config=DEFAULT_POLICY_CONFIG,
    protocol_path=TRAINING_PROTOCOL_PATH,
):
    protocol = load_store_bottle_training_protocol(protocol_path)
    for label, path in (
        ("training data", data_dir),
        ("model output", models_dir),
        ("policy config", policy_config),
    ):
        _reject_evaluation_or_result_path(path, label)
    task = protocol["task"]["task_name"]
    return {
        "schema": TRAINING_PLAN_SCHEMA,
        "dry_run": True,
        "protocol_id": protocol["protocol_id"],
        "task": task,
        "tasks_trained": [task],
        "other_tasks_trained": False,
        "policy_spec": _policy_spec_identity_from_config(protocol),
        "data_directory": str(Path(data_dir)),
        "collection_manifest_present": (Path(data_dir) / "collection_manifest.json").is_file(),
        "policy_config": str(Path(policy_config)),
        "policy_config_present": Path(policy_config).is_file(),
        "models_directory": str(Path(models_dir)),
        "model_directory": str(Path(models_dir) / task),
        "manifest_schema": "dynamac-direct-training-v4",
        "demonstrations": 5,
        "models_written": False,
    }


def _episode_file_record(data_dir, path):
    return {
        "path": Path(path).relative_to(data_dir).as_posix(),
        "bytes": Path(path).stat().st_size,
        "sha256": _file_sha256(path),
    }


def _validate_demo(demo):
    if not demo:
        raise RuntimeError("RLBench returned an empty StoreBottle demonstration")
    for observation in demo:
        state = observation.task_low_dim_state
        if isinstance(state, tuple):
            if len(state) != 1:
                raise RuntimeError("StoreBottle task state tuple is malformed")
            state = state[0]
        value = np.asarray(state, dtype=np.float64)
        if value.shape != (14,) or not np.all(np.isfinite(value)):
            raise RuntimeError("StoreBottle V4 observations must be two finite poses")
        for arm in (observation.left, observation.right):
            pose = np.asarray(arm.gripper_pose, dtype=np.float64)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise RuntimeError("StoreBottle demo contains an invalid gripper pose")


def _observation_config():
    from rlbench.observation_config import ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.gripper_open = True
    config.gripper_pose = True
    config.task_low_dim_state = True
    return config


def _collection_action_mode():
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import BimanualJointPosition
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

    return BimanualMoveArmThenGripper(
        BimanualJointPosition(),
        BimanualDiscrete(),
    )


def _store_task_class(protocol):
    module = importlib.import_module(protocol["task"]["module"])
    return getattr(module, protocol["task"]["class_name"])


def collect_store_bottle_v4(
    data_dir=DEFAULT_DATA_DIR,
    headless=True,
    protocol_path=TRAINING_PROTOCOL_PATH,
    _smoke=False,
):
    """Atomically collect the five successful static V4 demonstrations."""

    from rlbench.environment import Environment

    from integrations.rlbench.rlbench_dynamac.core.records import atomic_json, reserve_output

    protocol = load_store_bottle_training_protocol(protocol_path)
    data_dir = Path(data_dir)
    _reject_evaluation_or_result_path(data_dir, "collection output")
    if _smoke and data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
        raise ValueError("smoke collection requires a non-release output directory")
    demonstration_count = 1 if _smoke else 5
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with reserve_output(data_dir):
        staging = Path(
            tempfile.mkdtemp(prefix=".store_bottle_v4.staging-", dir=str(data_dir.parent))
        )
        environment = None
        launched = False
        try:
            environment = Environment(
                action_mode=_collection_action_mode(),
                obs_config=_observation_config(),
                headless=bool(headless),
                static_positions=protocol["collection"]["static_positions"],
                robot_setup=protocol["collection"]["robot_setup"],
            )
            environment.launch()
            launched = True
            task_environment = environment.get_task(_store_task_class(protocol))
            if task_environment.variation_count() != 1:
                raise RuntimeError("StoreBottle V4 protocol expects one variation")
            episodes = []
            for episode in range(demonstration_count):
                seed = protocol["collection"]["base_seed"] + episode
                random.seed(seed)
                np.random.seed(seed)
                task_environment.set_variation(0)
                (demo,) = task_environment.get_demos(
                    amount=1,
                    live_demos=True,
                    max_attempts=protocol["collection"]["max_demo_attempts"],
                )
                _validate_demo(demo)
                target = staging / "all_variations" / "episodes" / ("episode%d" % episode)
                target.mkdir(parents=True, exist_ok=False)
                for observation in demo:
                    perception = getattr(observation, "perception_data", None)
                    if isinstance(perception, dict):
                        perception.clear()
                low_dim = target / "low_dim_obs.pkl"
                variation_number = target / "variation_number.pkl"
                with low_dim.open("wb") as stream:
                    pickle.dump(demo, stream, protocol=pickle.HIGHEST_PROTOCOL)
                with variation_number.open("wb") as stream:
                    pickle.dump(0, stream, protocol=pickle.HIGHEST_PROTOCOL)
                episodes.append(
                    {
                        "episode": episode,
                        "seed": seed,
                        "variation": 0,
                        "observations": len(demo),
                        "success_verified": True,
                        "files": {
                            "low_dim_obs": _episode_file_record(staging, low_dim),
                            "variation_number": _episode_file_record(
                                staging, variation_number
                            ),
                        },
                    }
                )
                print(
                    "StoreBottle V4 demo %d/%d: %d observations"
                    % (episode + 1, demonstration_count, len(demo)),
                    flush=True,
                )
            policy_spec = _policy_spec_identity_from_config(protocol)
            manifest = {
                "schema": SMOKE_MANIFEST_SCHEMA if _smoke else COLLECTION_MANIFEST_SCHEMA,
                "protocol_id": protocol["protocol_id"] + ("-smoke" if _smoke else ""),
                "semantic_version": policy_spec["semantic_version"],
                "semantic_fingerprint": policy_spec["semantic_fingerprint"],
                "policy_spec": policy_spec,
                "scenario": "static",
                "environment_intervention": "none",
                "demonstrations": demonstration_count,
                "base_seed": 4104000000,
                "variation": 0,
                "success_authority": protocol["collection"]["success_authority"],
                "evaluation_artifacts_included": False,
                "episodes": episodes,
            }
            manifest["fingerprint"] = _canonical_sha256(manifest)
            atomic_json(staging / "collection_manifest.json", manifest)
            os.rename(staging, data_dir)
        finally:
            if launched and environment is not None:
                environment.shutdown()
            if staging.exists():
                shutil.rmtree(staging)
    return manifest


def collect_store_bottle_v4_smoke(
    output_dir,
    headless=True,
    protocol_path=TRAINING_PROTOCOL_PATH,
):
    """Collect one non-training demo to validate the live simulator entry."""

    return collect_store_bottle_v4(
        output_dir,
        headless,
        protocol_path,
        _smoke=True,
    )


def load_store_bottle_collection_manifest(path=None, verify_files=True):
    """Authenticate the five-demo manifest and optionally every saved file."""

    manifest_path = (
        DEFAULT_DATA_DIR / "collection_manifest.json" if path is None else Path(path)
    )
    _reject_evaluation_or_result_path(manifest_path, "collection manifest")
    value = _load_object(manifest_path)
    expected_fields = {
        "schema",
        "protocol_id",
        "semantic_version",
        "semantic_fingerprint",
        "policy_spec",
        "scenario",
        "environment_intervention",
        "demonstrations",
        "base_seed",
        "variation",
        "success_authority",
        "evaluation_artifacts_included",
        "episodes",
        "fingerprint",
    }
    unsigned = {key: item for key, item in value.items() if key != "fingerprint"}
    protocol = load_store_bottle_training_protocol()
    from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import store_bottle_policy_spec_identity

    if (
        set(value) != expected_fields
        or value.get("schema") != COLLECTION_MANIFEST_SCHEMA
        or value.get("protocol_id") != protocol["protocol_id"]
        or value.get("policy_spec") != store_bottle_policy_spec_identity()
        or value.get("semantic_version")
        != value["policy_spec"]["semantic_version"]
        or value.get("semantic_fingerprint")
        != value["policy_spec"]["semantic_fingerprint"]
        or value.get("scenario") != "static"
        or value.get("environment_intervention") != "none"
        or value.get("demonstrations") != 5
        or value.get("base_seed") != 4104000000
        or value.get("variation") != 0
        or value.get("success_authority")
        != protocol["collection"]["success_authority"]
        or value.get("evaluation_artifacts_included") is not False
        or value.get("fingerprint") != _canonical_sha256(unsigned)
    ):
        raise ValueError("StoreBottle V4 collection manifest is invalid")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 5:
        raise ValueError("StoreBottle V4 collection must contain five demos")
    data_dir = manifest_path.parent
    for index, episode in enumerate(episodes):
        if (
            not isinstance(episode, dict)
            or set(episode)
            != {
                "episode",
                "seed",
                "variation",
                "observations",
                "success_verified",
                "files",
            }
            or episode.get("episode") != index
            or episode.get("seed") != 4104000000 + index
            or episode.get("variation") != 0
            or not isinstance(episode.get("observations"), int)
            or episode.get("observations") < 2
            or episode.get("success_verified") is not True
            or not isinstance(episode.get("files"), dict)
            or set(episode["files"]) != {"low_dim_obs", "variation_number"}
        ):
            raise ValueError("StoreBottle V4 collection episode is invalid")
        expected_paths = {
            "low_dim_obs": "all_variations/episodes/episode%d/low_dim_obs.pkl" % index,
            "variation_number": (
                "all_variations/episodes/episode%d/variation_number.pkl" % index
            ),
        }
        for role, relative in expected_paths.items():
            record = episode["files"][role]
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or record.get("path") != relative
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] < 1
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in record["sha256"])
            ):
                raise ValueError("StoreBottle V4 episode file record is invalid")
            file_path = data_dir / relative
            if verify_files and (
                not file_path.is_file()
                or file_path.stat().st_size != record["bytes"]
                or _file_sha256(file_path) != record["sha256"]
            ):
                raise ValueError("StoreBottle V4 episode file hash mismatch")
    return value


def build_store_bottle_training_identity(
    data_dir=DEFAULT_DATA_DIR,
    policy_config=DEFAULT_POLICY_CONFIG,
):
    """Bind training to the five demos, policy config, and exact V4 spec."""

    from integrations.rlbench.rlbench_dynamac.data.direct_policy import v4_quaternion_batch_gauge_identity

    data_dir = Path(data_dir)
    policy_config = Path(policy_config)
    for label, path in (("training data", data_dir), ("policy config", policy_config)):
        _reject_evaluation_or_result_path(path, label)
    if policy_config.resolve() != DEFAULT_POLICY_CONFIG.resolve():
        raise ValueError("StoreBottle V4 training requires the pinned V4 policy config")
    manifest_path = data_dir / "collection_manifest.json"
    collection = load_store_bottle_collection_manifest(manifest_path, verify_files=True)
    if not policy_config.is_file():
        raise FileNotFoundError(policy_config)
    identity = {
        "schema": TRAINING_IDENTITY_SCHEMA,
        "policy_spec": collection["policy_spec"],
        "collection": {
            "schema": collection["schema"],
            "manifest_path": "collection_manifest.json",
            "manifest_sha256": _file_sha256(manifest_path),
            "manifest_fingerprint": collection["fingerprint"],
            "demonstrations": 5,
            "seeds": list(range(4104000000, 4104000005)),
            "variation": 0,
            "all_success_verified": True,
        },
        "policy_config": {
            "path": "integrations/rlbench/configs/v4/dynamac_store_bottle.json",
            "sha256": _file_sha256(policy_config),
        },
        "quaternion_batch_gauge": v4_quaternion_batch_gauge_identity(),
        "evaluation_artifacts_included": False,
        "tasks_trained": ["bimanual_put_bottle_in_fridge"],
        "other_tasks_trained": False,
    }
    identity["fingerprint"] = _canonical_sha256(identity)
    return identity


def train_store_bottle_v4(
    data_dir=DEFAULT_DATA_DIR,
    models_dir=DEFAULT_MODELS_DIR,
    policy_config=DEFAULT_POLICY_CONFIG,
):
    """Train exactly the StoreBottle left/right policies and no other task."""

    from integrations.rlbench.rlbench_dynamac.data.direct_policy import TRAINING_MANIFEST_SCHEMA_V4, train_task
    from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import store_bottle_semantic_task_spec

    identity = build_store_bottle_training_identity(data_dir, policy_config)
    return train_task(
        "bimanual_put_bottle_in_fridge",
        data_root=Path(data_dir),
        task_data_dir=Path(data_dir),
        models_dir=Path(models_dir),
        config_path=Path(policy_config),
        demonstration_count=5,
        task_spec=store_bottle_semantic_task_spec(),
        manifest_schema=TRAINING_MANIFEST_SCHEMA_V4,
        training_identity=identity,
    )


def serve_store_bottle_v4(models_dir=DEFAULT_MODELS_DIR):
    """Serve StoreBottle; PolicyServer authenticates and selects the V4 spec."""

    from integrations.rlbench.rlbench_dynamac.data.direct_policy import serve

    return serve("bimanual_put_bottle_in_fridge", Path(models_dir))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect five static successful demos")
    collect.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    collect.add_argument("--dry-run", action="store_true")
    display = collect.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    collect.set_defaults(headless=True)

    smoke = commands.add_parser(
        "smoke-collect",
        help="collect one successful demo outside the release data directory",
    )
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--dry-run", action="store_true")
    smoke_display = smoke.add_mutually_exclusive_group()
    smoke_display.add_argument("--headless", dest="headless", action="store_true")
    smoke_display.add_argument("--no-headless", dest="headless", action="store_false")
    smoke.set_defaults(headless=True)

    train = commands.add_parser("train", help="train only StoreBottle left/right")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    train.add_argument("--config", type=Path, default=DEFAULT_POLICY_CONFIG)
    train.add_argument("--dry-run", action="store_true")

    worker = commands.add_parser("serve", help="serve the authenticated V4 policy")
    worker.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)

    release = commands.add_parser("release-manifest", help="inventory V4 model provenance")
    release.add_argument("--source-models-dir", type=Path)
    release.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    release.add_argument("--output", type=Path)
    release.add_argument("--require-complete", action="store_true")
    release.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        result = (
            collection_dry_run(args.data_dir, args.headless)
            if args.dry_run
            else collect_store_bottle_v4(args.data_dir, args.headless)
        )
    elif args.command == "smoke-collect":
        result = (
            collection_dry_run(args.output_dir, args.headless, smoke=True)
            if args.dry_run
            else collect_store_bottle_v4_smoke(args.output_dir, args.headless)
        )
    elif args.command == "train":
        result = (
            training_dry_run(args.data_dir, args.models_dir, args.config)
            if args.dry_run
            else train_store_bottle_v4(args.data_dir, args.models_dir, args.config)
        )
    elif args.command == "serve":
        return serve_store_bottle_v4(args.models_dir)
    else:
        from integrations.rlbench.rlbench_dynamac.data.store_bottle_release_v4 import (
            DEFAULT_V3_MODELS_DIR,
            build_model_release_manifest,
            write_model_release_manifest,
        )

        source = args.source_models_dir or DEFAULT_V3_MODELS_DIR
        result = build_model_release_manifest(
            source_models_dir=source,
            target_models_dir=args.models_dir,
            require_complete=args.require_complete,
        )
        if args.output is not None:
            write_model_release_manifest(args.output, result)
        elif not args.dry_run:
            raise ValueError("release-manifest requires --dry-run or --output")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTION_MANIFEST_SCHEMA",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_POLICY_CONFIG",
    "TRAINING_IDENTITY_SCHEMA",
    "TRAINING_PROTOCOL_PATH",
    "build_store_bottle_training_identity",
    "collect_store_bottle_v4",
    "collect_store_bottle_v4_smoke",
    "collection_dry_run",
    "load_store_bottle_collection_manifest",
    "load_store_bottle_training_protocol",
    "serve_store_bottle_v4",
    "train_store_bottle_v4",
    "training_dry_run",
]
