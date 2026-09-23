"""Export M3 causal features from the five frozen successful demonstrations.

This is a teacher-forced replay of the frozen DynaMAC action policy.  It uses
the saved physical observations as the state sequence while querying the same
policy server used online for the contemporaneous action and reference-state
metadata.  No calibration rollout, injected fault, audit label, or sealed-test
record is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from essay2608.policy.dynamac import tapas_subsample_rows
from evaluations.iclr2027.interfaces.feature_schema import FeatureRecord
from evaluations.iclr2027.methods.fail_detect.preprocessing import FeatureLayout
from evaluations.iclr2027.runners.shared_episode import _compact_policy_state
from integrations.rlbench.iclr2027.build_assets import (
    DATA_ROOT,
    DYNAMAC_ROOT,
    BIMANUAL_DATA_ROOT,
    DYNAMAC_BACKBONE_ROOT,
    BIMANUAL_SEGMENTATION_CONFIG,
    SEGMENTATION_CONFIG,
)
from integrations.rlbench.iclr2027.task_registry import experiment_task, experiment_task_set
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    PolicyServer,
    demonstration_paths,
)
from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
    load_rlbench_segmentation_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "evaluations/iclr2027/artifacts/training/m3/demo_features"
)
FEATURE_DATA_SCHEMA = "essay2608.iclr2027.m3-demo-features.v1"
DEMONSTRATION_COUNT = 5
TRANSLATION_TOLERANCE_M = 0.001
ROTATION_TOLERANCE_RAD = math.radians(0.1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observations(episode: Any) -> list[Any]:
    values = list(getattr(episode, "_observations", episode))
    if not values:
        raise ValueError("demonstration contains no observations")
    return values


def _flat_task_state(observation: Any) -> np.ndarray:
    value = observation.task_low_dim_state
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if not result.size or not np.isfinite(result).all():
        raise ValueError("demonstration task state must be finite and non-empty")
    return result


def _wire_observation(observation: Any, *, bimanual: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_low_dim_state": _flat_task_state(observation).tolist()
    }
    if not bimanual:
        result.update(
            {
                "gripper_pose": np.asarray(
                    observation.gripper_pose, dtype=np.float64
                ).tolist(),
                "gripper_open": float(observation.gripper_open),
            }
        )
        return result
    for arm in ("left", "right"):
        current = getattr(observation, arm)
        result[arm] = {
            "gripper_pose": np.asarray(
                current.gripper_pose, dtype=np.float64
            ).tolist(),
            "gripper_open": float(current.gripper_open),
        }
    return result


def _feature_arms(observation: Any, *, bimanual: bool) -> dict[str, Any]:
    payload = _wire_observation(observation, bimanual=bimanual)
    if not bimanual:
        return {
            "single": {
                "ee_pose_xyzw": payload["gripper_pose"],
                "gripper_open": payload["gripper_open"],
            }
        }
    return {
        arm: {
            "ee_pose_xyzw": payload[arm]["gripper_pose"],
            "gripper_open": payload[arm]["gripper_open"],
        }
        for arm in ("left", "right")
    }


def _pose_distance(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(current[:3] - target[:3]))
    left = current[3:]
    right = target[3:]
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    rotation = 2.0 * math.acos(float(np.clip(abs(np.dot(left, right)), -1.0, 1.0)))
    return translation, rotation


def _arm_targets(action: np.ndarray, *, bimanual: bool) -> dict[str, np.ndarray]:
    if not bimanual:
        return {"single": action[:7]}
    # The frozen dual-arm wire ABI is right followed by left.
    return {"right": action[:7], "left": action[9:16]}


def _arm_pose(observation: Any, arm: str) -> np.ndarray:
    source = observation if arm == "single" else getattr(observation, arm)
    return np.asarray(source.gripper_pose, dtype=np.float64)


def _previous_resolution(
    previous_observation: Any | None,
    current_observation: Any,
    previous_action: np.ndarray | None,
    *,
    bimanual: bool,
) -> dict[str, Any]:
    if previous_observation is None or previous_action is None:
        return {
            "aggregate": "initial",
            "per_arm": {},
            "primary_action_applied": False,
        }
    statuses: dict[str, str] = {}
    improved = False
    for arm, target in _arm_targets(previous_action, bimanual=bimanual).items():
        current = _arm_pose(current_observation, arm)
        before = _arm_pose(previous_observation, arm)
        translation, rotation = _pose_distance(current, target)
        if (
            translation <= TRANSLATION_TOLERANCE_M
            and rotation <= ROTATION_TOLERANCE_RAD
        ):
            statuses[arm] = "reached"
            improved = True
            continue
        before_translation, before_rotation = _pose_distance(before, target)
        before_score = max(
            before_translation / TRANSLATION_TOLERANCE_M,
            before_rotation / ROTATION_TOLERANCE_RAD,
        )
        current_score = max(
            translation / TRANSLATION_TOLERANCE_M,
            rotation / ROTATION_TOLERANCE_RAD,
        )
        if current_score < before_score - 1.0e-6:
            statuses[arm] = "progressed"
            improved = True
        else:
            statuses[arm] = "stopped"
    aggregate = (
        "reached"
        if all(value == "reached" for value in statuses.values())
        else ("progressed" if improved else "stopped")
    )
    return {
        "aggregate": aggregate,
        "per_arm": statuses,
        "primary_action_applied": True,
    }


def _indices_by_skill(skill: np.ndarray, durations: Sequence[int]) -> np.ndarray:
    sequence = []
    for label in dict.fromkeys(int(value) for value in skill.tolist()):
        sequence.append(label)
    if len(sequence) != len(durations):
        raise ValueError("demonstration and checkpoint skill counts disagree")
    selected = []
    for label, duration in zip(sequence, durations, strict=True):
        source = np.flatnonzero(skill == label)
        if not len(source):
            raise ValueError("empty demonstration skill")
        values = tapas_subsample_rows(source[:, None], int(duration)).reshape(-1)
        selected.extend(int(value) for value in values)
    return np.asarray(selected, dtype=np.int64)


def _selected_indices(
    converted: Any,
    demonstration_index: int,
    manifest: Mapping[str, Any],
    raw_length: int,
    *,
    bimanual: bool,
) -> np.ndarray:
    if not bimanual:
        demo = converted.demonstrations[demonstration_index]
        return _indices_by_skill(demo.skill, manifest["durations"])
    totals = {
        arm: sum(int(value) for value in manifest[arm]["durations"])
        for arm in ("left", "right")
    }
    if len(set(totals.values())) != 1:
        raise ValueError("bimanual checkpoint does not use one shared policy horizon")
    # Independent arm segmentations can place their boundaries at different raw
    # samples.  Online they nevertheless advance on one shared policy clock, so
    # the physically faithful common replay is one global temporal alignment.
    source = np.arange(raw_length, dtype=np.int64)[:, None]
    return tapas_subsample_rows(source, next(iter(totals.values()))).reshape(-1).astype(int)


def _load_task_inputs(task_id: str) -> tuple[Any, list[Any], list[Path], Path, dict]:
    task = experiment_task(task_id)
    data_root = BIMANUAL_DATA_ROOT if task.spec.bimanual else DATA_ROOT
    model_root = DYNAMAC_BACKBONE_ROOT if task.spec.bimanual else DYNAMAC_ROOT
    paths = demonstration_paths(data_root, task_id, DEMONSTRATION_COUNT)
    episodes = load_low_dim_obs_pickles(paths)
    names = [path.parent.name for path in paths]
    segmentation = load_rlbench_segmentation_config(
        BIMANUAL_SEGMENTATION_CONFIG if task.spec.bimanual else SEGMENTATION_CONFIG
    )
    converted = (
        make_bimanual_demonstrations(
            episodes, task.spec, names=names, config=segmentation
        )
        if task.spec.bimanual
        else make_unimanual_demonstrations(
            episodes, task.spec, names=names, config=segmentation
        )
    )
    manifest_path = model_root / task_id / "training.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return task, episodes, paths, model_root, {"path": manifest_path, "value": manifest, "converted": converted}


def export_task(task_id: str, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    task, episodes, paths, model_root, loaded = _load_task_inputs(task_id)
    manifest_path = loaded["path"]
    manifest = loaded["value"]
    converted = loaded["converted"]
    vectors = []
    episode_offsets = []
    layout: FeatureLayout | None = None
    for demonstration_index, (episode, source_path) in enumerate(
        zip(episodes, paths, strict=True)
    ):
        observations = _observations(episode)
        selected = _selected_indices(
            converted,
            demonstration_index,
            manifest,
            len(observations),
            bimanual=task.spec.bimanual,
        )
        server = PolicyServer(
            task_id,
            model_root,
            # The authenticated V4 bottle checkpoint binds its own semantic
            # policy spec; all other Main-10 models use the experiment spec.
            task_spec=(
                None
                if manifest.get("manifest_schema") == "dynamac-direct-training-v4"
                else task.spec
            ),
        )
        first_payload = _wire_observation(
            observations[int(selected[0])], bimanual=task.spec.bimanual
        )
        server.handle({"command": "reset", "observation": first_payload})
        start = len(vectors)
        previous_observation = None
        previous_action = None
        for cycle, raw_index in enumerate(selected.tolist()):
            observation = observations[int(raw_index)]
            payload = _wire_observation(observation, bimanual=task.spec.bimanual)
            response = server.handle({"command": "act", "observation": payload})
            action = response.get("action")
            if action is None:
                raise RuntimeError("DynaMAC completed before its frozen policy horizon")
            action_array = np.asarray(action, dtype=np.float64)
            record = FeatureRecord(
                episode_id=f"normal_demonstrations/{task_id}/{demonstration_index:04d}",
                cycle=cycle,
                observation_timestamp=cycle,
                action_timestamp=cycle,
                arms=_feature_arms(observation, bimanual=task.spec.bimanual),
                task_state=tuple(_flat_task_state(observation).tolist()),
                action=tuple(action_array.tolist()),
                policy_state=_compact_policy_state(response, cycle),
                action_resolution=_previous_resolution(
                    previous_observation,
                    observation,
                    previous_action,
                    bimanual=task.spec.bimanual,
                ),
            ).to_dict()
            current_layout = FeatureLayout.from_record(record)
            if layout is None:
                layout = current_layout
            elif current_layout != layout:
                raise ValueError("causal feature layout changed across demonstrations")
            vectors.append(layout.encode_validated(record))
            server.handle(
                {
                    "command": "commit",
                    "transaction_id": response["transaction_id"],
                    "primary_action_status": "reached",
                }
            )
            previous_observation = observation
            previous_action = action_array
        episode_offsets.append(
            {
                "demonstration": demonstration_index,
                "source": str(source_path.relative_to(REPOSITORY_ROOT)),
                "source_sha256": _sha256(source_path),
                "raw_steps": len(observations),
                "selected_steps": len(selected),
                "start": start,
                "stop": len(vectors),
            }
        )
    if layout is None:
        raise RuntimeError("no demonstration features were exported")
    features = np.stack(vectors).astype(np.float32, copy=False)
    output_dir = Path(output_root) / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npz"
    np.savez_compressed(feature_path, features=features)
    index = {
        "schema": FEATURE_DATA_SCHEMA,
        "task": task_id,
        "split": "normal_demonstrations",
        "demonstration_count": DEMONSTRATION_COUNT,
        "contains_failure_train": False,
        "contains_normal_calibration": False,
        "contains_sealed_test": False,
        "contains_fault_or_audit_labels": False,
        "layout": layout.to_dict(),
        "rows": int(features.shape[0]),
        "feature_shape": list(features.shape),
        "features_relative_path": str(feature_path.relative_to(REPOSITORY_ROOT)),
        "features_sha256": _sha256(feature_path),
        "policy_training_manifest": {
            "path": str(manifest_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(manifest_path),
        },
        "episodes": episode_offsets,
        "alignment": (
            "global_shared_policy_clock_tapas_subsample"
            if task.spec.bimanual
            else "per_skill_tapas_subsample"
        ),
    }
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    main10 = [task.task_id for task in experiment_task_set("main10")]
    tasks = main10 if not args.task or args.task == ["all"] else args.task
    unknown = set(tasks).difference(main10)
    if unknown:
        raise ValueError(f"M3 Main-10 export received unknown tasks: {sorted(unknown)}")
    for task_id in tasks:
        result = export_task(task_id, args.output)
        print(json.dumps({"task": task_id, "rows": result["rows"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "export_task"]
