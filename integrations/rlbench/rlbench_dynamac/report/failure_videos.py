"""Replay and record audited trajectories without changing policy semantics.

Existing evaluation files contain low-dimensional episode records only.  This
tool replays explicitly selected successful or failed episodes with the same
evaluator, model, seed, variation and controller, while enabling RGB
observations solely for video capture.  A replay is published only when its
outcome still matches the selected source row and the loaded model identity
exactly matches the source evaluation.

The simulator side remains Python 3.8 compatible.  Frames are passed directly
to the system ``ffmpeg`` executable, so OpenCV and imageio are not required in
the pinned RLBench environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from integrations.rlbench.rlbench_dynamac.eval import (
    direct_evaluate,
    table_iii_coordination,
    unimanual_evaluate,
)
from integrations.rlbench.rlbench_dynamac.eval.eval_set import _is_sha256, fixed_coordination_sources, fixed_environment_plans
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    FORMAL_POLICY_CLOCK_SEMANTICS_ID,
    bind_staged_source_plan,
    global_ik_controller_metadata,
    initialize_fresh_task_generation,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_eval_v4 import (
    V4_STORE_MODE_MOVED_ENTITIES,
    V4_STORE_MOTION_PROTOCOL_ID,
    V4_STORE_RUNTIME_LOADER_ID,
    V4_STORE_TRIGGER_STEPS,
    load_v4_store_intervention_protocol,
    load_v4_store_motion_plan_batch,
    load_v4_store_motion_source_protocol,
    v4_store_task_identity_components,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import STORE_BOTTLE_TASK_NAME
from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
    load_v3_intervention_protocol,
    load_v3_motion_source_protocol,
)
from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import (
    V4_COORDINATION_PROTOCOL_ID,
    V4_COORDINATION_SMOOTH_POLICY_TICKS,
    V4_COORDINATION_TRIGGER_STEP,
    V4_LIFT_MOTION_PROTOCOL_ID,
    V4_LIFT_RUNTIME_LOADER_ID,
    V4_LIFT_TASK,
    V4_LIFT_TRIGGER_STEP,
    load_v4_coordination_intervention_protocol,
    load_v4_lift_intervention_protocol,
    load_v4_lift_motion_plan_batch,
    load_v4_lift_motion_source_protocol,
    v4_lift_task_identity_components,
)

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3"
DEFAULT_OUTPUT_ROOT = INTEGRATION_ROOT / "results" / "failure_videos" / "v3"
DEFAULT_REPLAY_OVERHEAD_PERSPECTIVE_DEGREES = 70.0
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
REPLAY_OUTPUT_ROOTS = (
    (INTEGRATION_ROOT / "results" / "failure_videos").resolve(),
    (INTEGRATION_ROOT / "results" / "v4" / "replay_video").resolve(),
)
DEFAULT_SOURCE_RESULTS = {
    "bimanual_handover_item": (
        INTEGRATION_ROOT
        / "results"
        / "v3"
        / "table_ii"
        / "bimanual_handover_item_static_seed2608000000_n200_h1000.json"
    ),
    "bimanual_sweep_to_dustpan": (
        INTEGRATION_ROOT
        / "results"
        / "v3"
        / "table_ii"
        / "bimanual_sweep_to_dustpan_static_seed2608000000_n200_h1000.json"
    ),
    "bimanual_lift_tray": (
        INTEGRATION_ROOT
        / "results"
        / "v3"
        / "table_ii"
        / "bimanual_lift_tray_static_seed2608000000_n200_h1000.json"
    ),
    "wipe_desk": (
        INTEGRATION_ROOT
        / "results"
        / "v3"
        / "table_i"
        / "wipe_desk_static_variation0_seed2608000000_n200_h1000.json"
    ),
}
BIMANUAL_TASKS = set(direct_evaluate.TASKS)
SUPPORTED_TASKS = set(unimanual_evaluate.TASKS) | BIMANUAL_TASKS
COORDINATION_SCHEMAS = {
    "dynamac-table-iii-coordination-local-v1",
    "dynamac-table-iii-coordination-local-v2",
    "dynamac-table-iii-coordination-local-v3",
    "dynamac-table-iii-coordination-local-v4",
}
SCENARIOS = {"static", "smooth", "teleport"}
DEFAULT_CAMERAS = ("front", "overhead")
SCHEMA = "dynamac-confirmed-trajectory-videos-v1"
REPLAY_ATTEMPT_PROTOCOL_ID = (
    "same-sealed-source-episode-fresh-replay-bounded-first-match-v1"
)
V4_STORE_EVALUATION_SET_ID = "rlbench_eval_v2"
V4_EVALUATION_SET_ID = "rlbench_eval_v2"
DEFAULT_SCENARIO_STEPS = 10
LEGACY_MAX_PRIMARY_ACTION_ATTEMPTS = 3


def _is_v4_store_source(source, task=None):
    """Identify the one release-aware replay path supported beyond V3."""

    selected_task = task or _source_policy_task(source)
    return (
        isinstance(source, dict)
        and source.get("release") == "v4"
        and selected_task == STORE_BOTTLE_TASK_NAME
    )


def _is_v4_lift_source(source, task=None):
    selected_task = task or _source_policy_task(source)
    return (
        isinstance(source, dict)
        and source.get("release") == "v4"
        and selected_task == V4_LIFT_TASK
    )


def _is_v4_coordination_source(source):
    return (
        isinstance(source, dict)
        and source.get("release") == "v4"
        and source.get("schema") == "dynamac-table-iii-coordination-local-v4"
    )


def _is_v4_source(source, task=None):
    """Return whether ``source`` is any current-release V4 result.

    StoreBottle, LiftTray, and coordination cells retain specialized identity
    checks below.  The other Table I--III cells deliberately reuse the sealed
    V3 task/intervention plans with the V4 controller and model release, so
    they still need V4 model routing and V4 replay metadata even though they
    do not have a task-specific V4 loader.
    """

    if not isinstance(source, dict) or source.get("release") != "v4":
        return False
    return task is None or _source_policy_task(source) == task


def _v4_replay_manifest_metadata(source, task):
    """Return release metadata without mislabelling generic V4 cells."""

    if not _is_v4_source(source, task):
        return {}
    metadata = {"release": "v4"}
    if _is_v4_store_source(source, task):
        metadata["task_identity"] = v4_store_task_identity_components()
    elif _is_v4_lift_source(source, task):
        metadata["task_identity"] = v4_lift_task_identity_components()
    elif _is_v4_coordination_source(source):
        metadata["coordination_protocol_id"] = V4_COORDINATION_PROTOCOL_ID
    return metadata


def _validate_v4_store_source_identity(source, task):
    """Fail closed unless a source is an authenticated formal V4 Store cell."""

    if not _is_v4_store_source(source, task):
        return
    fixed = source.get("fixed_eval_set")
    model = source.get("model_identity")
    training = model.get("training_identity") if isinstance(model, dict) else None
    protocol = source.get("scenario_protocol")
    motion = protocol.get("motion_protocol") if isinstance(protocol, dict) else None
    if (
        source.get("scenario") not in {"static", "teleport"}
        or not isinstance(fixed, dict)
        or fixed.get("evaluation_set_id") != V4_STORE_EVALUATION_SET_ID
        or not isinstance(model, dict)
        or model.get("training_manifest_schema") != "dynamac-direct-training-v4"
        or model.get("manifest_authenticated") is not True
        or not isinstance(training, dict)
        or training.get("schema") != "rlbench-store-bottle-training-identity-v4"
        or training.get("tasks_trained") != [STORE_BOTTLE_TASK_NAME]
        or training.get("other_tasks_trained") is not False
        or not isinstance(protocol, dict)
        or not isinstance(motion, dict)
        or motion.get("protocol_id") != V4_STORE_MOTION_PROTOCOL_ID
    ):
        raise RuntimeError("source is not an authenticated formal V4 StoreBottle cell")


def _validate_v4_lift_source_identity(source, task):
    if not _is_v4_lift_source(source, task):
        return
    fixed = source.get("fixed_eval_set")
    model = source.get("model_identity")
    protocol = source.get("scenario_protocol")
    motion = protocol.get("motion_protocol") if isinstance(protocol, dict) else None
    expected_status = (
        "STATIC_REFERENCE"
        if source.get("scenario") == "static"
        else "V4_LIFT_SKILL0_TICK35_TASK_SCOPED"
    )
    if (
        source.get("scenario") not in {"static", "teleport"}
        or not isinstance(fixed, dict)
        or fixed.get("evaluation_set_id") != V4_EVALUATION_SET_ID
        or not isinstance(model, dict)
        or model.get("training_manifest_schema") != "dynamac-direct-training-v3"
        or model.get("manifest_authenticated") is not True
        or not isinstance(protocol, dict)
        or protocol.get("protocol_valid") is not True
        or protocol.get("status") != expected_status
        or not isinstance(motion, dict)
        or motion.get("protocol_id") != V4_LIFT_MOTION_PROTOCOL_ID
    ):
        raise RuntimeError("source is not an authenticated formal V4 LiftTray cell")


def _validate_v4_coordination_source_identity(source, task):
    if not _is_v4_coordination_source(source):
        return
    fixed = source.get("fixed_eval_set")
    model = source.get("model_identity")
    protocol = source.get("coordination_protocol")
    expected_arm = {
        "coordination_hand_left": "left",
        "coordination_hand_right": "right",
    }.get(source.get("scenario"))
    if (
        task != "bimanual_handover_item"
        or source.get("task") != "bimanual_handover_item_dynamic"
        or expected_arm is None
        or not isinstance(fixed, dict)
        or fixed.get("evaluation_set_id") != V4_EVALUATION_SET_ID
        or not isinstance(model, dict)
        or model.get("training_manifest_schema") != "dynamac-direct-training-v3"
        or model.get("manifest_authenticated") is not True
        or not isinstance(protocol, dict)
        or protocol.get("protocol_valid") is not True
        or protocol.get("protocol_id") != V4_COORDINATION_PROTOCOL_ID
        or protocol.get("perturbed_arm") != expected_arm
        or protocol.get("trigger_policy_step") != V4_COORDINATION_TRIGGER_STEP
    ):
        raise RuntimeError("source is not an authenticated formal V4 coordination cell")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_episode_indices(values):
    if not values:
        raise ValueError("at least one --episode is required")
    result = []
    seen = set()
    for value in values:
        episode = int(value)
        if episode < 0:
            raise ValueError("episode indices must be non-negative")
        if episode in seen:
            raise ValueError(f"duplicate episode index: {episode}")
        seen.add(episode)
        result.append(episode)
    return tuple(result)


def _source_family(payload, task):
    if payload.get("schema") in COORDINATION_SCHEMAS:
        return "coordination"
    if task in BIMANUAL_TASKS:
        return "bimanual"
    if task in unimanual_evaluate.TASKS:
        return "unimanual"
    raise ValueError(f"unsupported replay task: {task!r}")


def _require_current_evaluator_protocol(payload, task):
    """Fail closed instead of replaying an archived run with new semantics."""

    family = _source_family(payload, task)
    expected = (
        table_iii_coordination.EVALUATION_PROTOCOL_ID
        if family == "coordination"
        else (
            unimanual_evaluate.EVALUATION_PROTOCOL_ID
            if family == "unimanual"
            else direct_evaluate.EVALUATION_PROTOCOL_ID
        )
    )
    actual = payload.get("evaluation_protocol_id")
    if actual != expected:
        raise RuntimeError(
            "source evaluation protocol does not match the current replay evaluator: "
            f"source={actual!r}, current={expected!r}. Use a protocol-matched "
            "archived runner or regenerate the evaluation before recording."
        )
    return expected


def _source_policy_task(payload):
    return payload.get("policy_task_alias") or payload.get("task")


def _load_source(path, task, episode_indices, *, expected_success=False):
    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if _source_policy_task(payload) != task:
        raise ValueError(
            f"source policy task {_source_policy_task(payload)!r} does not match requested {task!r}"
        )
    family = _source_family(payload, task)
    scenario = payload.get("scenario")
    if family == "coordination":
        if scenario not in {"coordination_hand_left", "coordination_hand_right"}:
            raise ValueError("coordination source has an unsupported scenario")
        protocol = payload.get("coordination_protocol")
        if not isinstance(protocol, dict) or not protocol.get("protocol_valid"):
            raise ValueError("coordination source has no valid protocol")
    elif scenario not in SCENARIOS:
        raise ValueError(f"source evaluation has an unsupported scenario: {scenario!r}")
    if not isinstance(payload.get("seed"), int) or payload["seed"] < 0:
        raise ValueError("source evaluation has an invalid seed")
    if not isinstance(payload.get("horizon"), int) or payload["horizon"] < 1:
        raise ValueError("source evaluation has an invalid horizon")
    model_identity = payload.get("model_identity")
    if not isinstance(model_identity, dict) or not model_identity.get("manifest_authenticated"):
        raise ValueError("source evaluation is not bound to an authenticated model")
    _validate_v4_store_source_identity(payload, task)
    _validate_v4_lift_source_identity(payload, task)
    _validate_v4_coordination_source_identity(payload, task)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("source evaluation has no episode rows")
    by_episode = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("episode"), int):
            raise ValueError("source evaluation contains an invalid episode row")
        if row["episode"] in by_episode:
            raise ValueError("source evaluation contains duplicate episode rows")
        by_episode[row["episode"]] = row
    selected = {}
    for episode in episode_indices:
        if episode not in by_episode:
            raise ValueError(f"episode {episode} is absent from the source evaluation")
        row = by_episode[episode]
        if bool(row.get("success")) is not bool(expected_success):
            expected = "successful" if expected_success else "failed"
            actual = "successful" if bool(row.get("success")) else "failed"
            raise ValueError(
                f"episode {episode} was {actual} in the source evaluation; "
                f"expected a {expected} row"
            )
        # Current formal evaluators retain legitimate episodes that terminate
        # before a dynamic trigger (and smooth episodes that terminate after
        # only an effective prefix).  Admit those canonical accounting states;
        # older rows without that accounting retain the original requirement
        # that the intervention was effective.
        if (
            _protocol_condition_signature(payload, row) is None
            and not _protocol_effective(payload, row)
        ):
            raise ValueError(
                f"episode {episode} did not exercise the source scenario"
            )
        selected[episode] = row
    return payload, selected, hashlib.sha256(raw).hexdigest(), source_path


def _auxiliary_source_rows(source, episode_indices):
    """Select authenticated source rows without imposing a target outcome."""

    if not episode_indices:
        return {}
    rows = source.get("results")
    if not isinstance(rows, list):
        raise ValueError("source evaluation has no episode rows")
    by_episode = {
        row.get("episode"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("episode"), int)
    }
    selected = {}
    for episode in episode_indices:
        if episode not in by_episode:
            raise ValueError(f"warm-up episode {episode} is absent from the source evaluation")
        selected[episode] = by_episode[episode]
    return selected


def _load_sealed_replay_batch(source, task):
    """Reload and authenticate the canonical plan batch used by a result."""

    fixed = source.get("fixed_eval_set")
    if not isinstance(fixed, dict):
        raise ValueError("V3 failure replay requires fixed-eval-set evidence")
    eval_set_id = fixed.get("evaluation_set_id")
    if not isinstance(eval_set_id, str) or not eval_set_id:
        raise ValueError("source evaluation has an invalid fixed eval-set ID")

    family = _source_family(source, task)
    v4_store = _is_v4_store_source(source, task)
    v4_lift = _is_v4_lift_source(source, task)
    v4_coordination = _is_v4_coordination_source(source)
    if family == "coordination":
        manifest, batch = fixed_coordination_sources(eval_set_id)
        selected_sha256 = batch["sha256"]
        selected_fingerprint = batch["batch_fingerprint"]
    else:
        if v4_store:
            manifest, batch = fixed_environment_plans(
                eval_set_id,
                task,
                runtime_loaders={
                    V4_STORE_RUNTIME_LOADER_ID: load_v4_store_motion_plan_batch
                },
            )
        elif v4_lift:
            manifest, batch = fixed_environment_plans(
                eval_set_id,
                task,
                runtime_loaders={
                    V4_LIFT_RUNTIME_LOADER_ID: load_v4_lift_motion_plan_batch
                },
            )
        else:
            manifest, batch = fixed_environment_plans(eval_set_id, task)
        reference = manifest["payload"]["environment_plan_batches"][task]
        selected_sha256 = reference["sha256"]
        selected_fingerprint = batch["payload"]["batch_fingerprint"]

    if _is_v4_source(source, task):
        # The global manifest/spec hashes are provenance, not admission keys:
        # resealing an unrelated cell must not invalidate this replay source.
        # The cell-selected batch remains a strict identity boundary.
        if (
            eval_set_id != V4_EVALUATION_SET_ID
            or fixed.get("formal_access")
            != "canonical_id_read_only_no_generation"
        ):
            raise RuntimeError("source result has an invalid formal fixed eval identity")
        for field in ("manifest_sha256", "spec_sha256"):
            if not _is_sha256(fixed.get(field)):
                raise RuntimeError(
                    f"source result {field} is not a SHA-256 provenance record"
                )
        for field, expected in (
            ("selected_batch_sha256", selected_sha256),
            ("selected_batch_fingerprint", selected_fingerprint),
        ):
            if not _is_sha256(fixed.get(field)) or fixed.get(field) != expected:
                raise RuntimeError(
                    "source result does not match its current cell-selected "
                    "evaluation batch"
                )
    else:
        expected_fixed = {
            "manifest_sha256": manifest["manifest_sha256"],
            "spec_sha256": manifest["payload"]["spec"]["sha256"],
            "selected_batch_sha256": selected_sha256,
            "selected_batch_fingerprint": selected_fingerprint,
        }
        if any(fixed.get(key) != value for key, value in expected_fixed.items()):
            raise RuntimeError("source result does not match its sealed fixed eval set")

    payload = batch.get("payload")
    plans = batch.get("plans")
    if not isinstance(payload, dict) or not isinstance(plans, list):
        raise RuntimeError("sealed replay batch did not load completely")
    if v4_store and (
        eval_set_id != V4_STORE_EVALUATION_SET_ID
        or payload.get("runtime_loader") != V4_STORE_RUNTIME_LOADER_ID
        or payload.get("task_identity", {}).get("components")
        != v4_store_task_identity_components()
    ):
        raise RuntimeError("sealed V4 StoreBottle replay identity is invalid")
    if v4_lift and (
        eval_set_id != V4_EVALUATION_SET_ID
        or payload.get("runtime_loader") != V4_LIFT_RUNTIME_LOADER_ID
        or payload.get("task_identity", {}).get("components")
        != v4_lift_task_identity_components()
    ):
        raise RuntimeError("sealed V4 LiftTray replay identity is invalid")
    if v4_coordination and eval_set_id != V4_EVALUATION_SET_ID:
        raise RuntimeError("sealed V4 coordination replay identity is invalid")
    if (
        source.get("seed") != payload.get("base_seed")
        or source.get("episodes") != payload.get("episodes")
        or source.get("variation_schedule") != payload.get("variation_schedule")
        or len(plans) != payload.get("episodes")
    ):
        raise RuntimeError("source result identity differs from its sealed plan batch")
    return {
        "family": family,
        "evaluation_set_id": eval_set_id,
        "manifest": manifest,
        "batch": batch,
        "plans": plans,
        "variation_schedule": payload["variation_schedule"],
        "v4_store": v4_store,
        "v4_lift": v4_lift,
        "v4_coordination": v4_coordination,
    }


def _sealed_episode_plan(replay_batch, original, episode):
    """Bind one source row to its exact sealed A/B or coordination-A plan."""

    plans = replay_batch["plans"]
    variations = replay_batch["variation_schedule"]
    if (
        isinstance(episode, bool)
        or not isinstance(episode, int)
        or episode < 0
        or episode >= len(plans)
        or episode >= len(variations)
    ):
        raise ValueError("replay episode lies outside the sealed plan batch")
    plan = plans[episode]
    variation = variations[episode]
    if plan.variation != variation:
        raise RuntimeError("sealed replay plan variation is inconsistent")

    plan_fingerprint = plan.fingerprint()
    if replay_batch["family"] == "coordination":
        binding = original.get("staged_source_binding")
        row_fingerprint = (
            binding.get("plan_fingerprint") if isinstance(binding, dict) else None
        )
        if replay_batch.get("v4_coordination") and (
            not isinstance(binding, dict)
            or binding.get("matched") is not True
            or binding.get("source_reconstruction", {}).get("passed") is not True
        ):
            raise RuntimeError("source row has an invalid V4 coordination binding")
    elif replay_batch.get("v4_store"):
        row_fingerprint = original.get("motion_plan_fingerprint")
        binding = original.get("staged_source_binding")
        if (
            not isinstance(binding, dict)
            or binding.get("plan_fingerprint") != plan_fingerprint
            or binding.get("formal_source_bound") is not True
            or binding.get("task_semantics_matched") is not True
            or binding.get("task_tree_matched") is not True
            or binding.get("low_dim_frames_matched") is not True
        ):
            raise RuntimeError("source row has an invalid V4 StoreBottle binding")
    elif replay_batch.get("v4_lift"):
        row_fingerprint = original.get("motion_plan_fingerprint")
        binding = original.get("staged_source_binding")
        if (
            not isinstance(binding, dict)
            or binding.get("motion_plan_fingerprint") != plan_fingerprint
            or binding.get("formal_source_bound") is not True
            or binding.get("task_semantics_matched") is not True
            or binding.get("task_tree_matched") is not True
            or binding.get("deterministic_source_reconstruction", {}).get("passed")
            is not True
        ):
            raise RuntimeError("source row has an invalid V4 LiftTray binding")
    else:
        row_fingerprint = original.get("motion_plan_fingerprint")
        binding = original.get("staged_source_binding")
        if (
            isinstance(binding, dict)
            and binding.get("motion_plan_fingerprint") != plan_fingerprint
        ):
            raise RuntimeError("source row has inconsistent staged-plan binding")
    if row_fingerprint != plan_fingerprint:
        raise RuntimeError("source row does not match its sealed replay plan")

    fresh = original.get("fresh_task_generation")
    source_seed = plan.validation.get("source_seed")
    if (
        not isinstance(fresh, dict)
        or fresh.get("episode_seed") != source_seed
        or fresh.get("variation") != variation
    ):
        raise RuntimeError("source row has inconsistent formal reset evidence")
    return plan, int(variation), int(source_seed)


def _validate_replay_plan(replay_batch, replay, plan):
    if replay_batch["family"] == "coordination":
        binding = replay.get("staged_source_binding")
        replay_fingerprint = (
            binding.get("plan_fingerprint") if isinstance(binding, dict) else None
        )
        if replay_batch.get("v4_coordination") and (
            not isinstance(binding, dict)
            or binding.get("matched") is not True
            or binding.get("source_reconstruction", {}).get("passed") is not True
        ):
            raise RuntimeError("replay did not retain its V4 coordination binding")
    elif replay_batch.get("v4_store"):
        binding = replay.get("staged_source_binding")
        replay_fingerprint = replay.get("motion_plan_fingerprint")
        if (
            not isinstance(binding, dict)
            or binding.get("plan_fingerprint") != plan.fingerprint()
            or binding.get("formal_source_bound") is not True
            or binding.get("task_semantics_matched") is not True
            or binding.get("task_tree_matched") is not True
            or binding.get("low_dim_frames_matched") is not True
        ):
            raise RuntimeError("replay did not retain its V4 StoreBottle binding")
    elif replay_batch.get("v4_lift"):
        binding = replay.get("staged_source_binding")
        replay_fingerprint = replay.get("motion_plan_fingerprint")
        if (
            not isinstance(binding, dict)
            or binding.get("motion_plan_fingerprint") != plan.fingerprint()
            or binding.get("formal_source_bound") is not True
            or binding.get("task_semantics_matched") is not True
            or binding.get("task_tree_matched") is not True
            or binding.get("deterministic_source_reconstruction", {}).get("passed")
            is not True
        ):
            raise RuntimeError("replay did not retain its V4 LiftTray binding")
    else:
        replay_fingerprint = replay.get("motion_plan_fingerprint")
    fresh = replay.get("fresh_task_generation")
    if (
        replay_fingerprint != plan.fingerprint()
        or not isinstance(fresh, dict)
        or fresh.get("episode_seed") != plan.validation.get("source_seed")
        or fresh.get("variation") != plan.variation
    ):
        raise RuntimeError("replay did not retain its sealed plan/reset binding")


def _rgb_from_observation(observation, camera):
    perception = getattr(observation, "perception_data", None)
    if not isinstance(perception, dict):
        return None
    value = perception.get(f"{camera}_rgb")
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{camera} RGB frame must have shape HxWx3")
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _compose_frame(observation, cameras):
    frames = []
    used = []
    for camera in cameras:
        frame = _rgb_from_observation(observation, camera)
        if frame is not None:
            frames.append(frame)
            used.append(camera)
    if not frames:
        raise ValueError("observation contains no requested RGB camera")
    height = frames[0].shape[0]
    if any(frame.shape[0] != height for frame in frames):
        raise ValueError("camera frames must have equal heights")
    return np.concatenate(frames, axis=1), tuple(used)


class _FFmpegWriter:
    def __init__(self, path, *, width, height, fps, ffmpeg):
        self.path = Path(path)
        self.frames = 0
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame):
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            error = b""
            if self._process.stderr is not None:
                error = self._process.stderr.read()
            raise RuntimeError(
                f"ffmpeg stopped while recording: {error.decode(errors='replace')}"
            ) from exc
        self.frames += 1

    def close(self):
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        error = b""
        if self._process.stderr is not None:
            error = self._process.stderr.read()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {return_code}: {error.decode(errors='replace')}"
            )
        if self.frames < 1 or not self.path.is_file():
            raise RuntimeError("ffmpeg did not produce a video")

    def abort(self):
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()


class ObservationRecorder:
    """Lazy RGB recorder with a deterministic front-only fallback."""

    def __init__(self, path, *, cameras, fps, ffmpeg, writer_class=_FFmpegWriter):
        self.path = Path(path)
        self.requested_cameras = tuple(cameras)
        self.fps = int(fps)
        self.ffmpeg = Path(ffmpeg)
        self.writer_class = writer_class
        self.used_cameras = None
        self.frame_shape = None
        self._writer = None

    @property
    def frames(self):
        return 0 if self._writer is None else self._writer.frames

    def capture(self, observation):
        cameras = self.requested_cameras if self.used_cameras is None else self.used_cameras
        frame, used = _compose_frame(observation, cameras)
        if self.used_cameras is None:
            # Front is the stable common view.  Overhead is included when the
            # scene exposes it; missing optional views do not prevent capture.
            if "front" not in used:
                raise ValueError("front RGB camera is unavailable")
            self.used_cameras = used
            self.frame_shape = tuple(int(value) for value in frame.shape)
            self._writer = self.writer_class(
                self.path,
                width=frame.shape[1],
                height=frame.shape[0],
                fps=self.fps,
                ffmpeg=self.ffmpeg,
            )
        elif tuple(frame.shape) != self.frame_shape:
            raise ValueError("RGB frame shape changed during an episode")
        self._writer.write(frame)

    def close(self):
        if self._writer is None:
            raise RuntimeError("episode produced no RGB frames")
        self._writer.close()

    def abort(self):
        if self._writer is not None:
            self._writer.abort()


class RecordingTaskEnvironment:
    """Transparent TaskEnvironment proxy that observes returned snapshots."""

    def __init__(self, task_environment, recorder):
        self._task_environment = task_environment
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._task_environment, name)

    def reset(self):
        descriptions, observation = self._task_environment.reset()
        self._recorder.capture(observation)
        return descriptions, observation

    def get_observation(self):
        observation = self._task_environment.get_observation()
        self._recorder.capture(observation)
        return observation

    def step(self, action):
        observation, reward, terminate = self._task_environment.step(action)
        self._recorder.capture(observation)
        return observation, reward, terminate


def _protocol_effective(source, replay):
    family = _source_family(source, _source_policy_task(source))
    if family == "coordination":
        if _is_v4_coordination_source(source):
            protocol = source.get("coordination_protocol", {})
            intervention = replay.get("coordination_intervention")
            return bool(
                isinstance(intervention, dict)
                and protocol.get("protocol_id") == V4_COORDINATION_PROTOCOL_ID
                and intervention.get("schema")
                == "dynamac-coordination-policy-clocked-smooth-intervention-v4"
                and intervention.get("protocol_id") == V4_COORDINATION_PROTOCOL_ID
                and intervention.get("perturbed_arm")
                == protocol.get("perturbed_arm")
                and intervention.get("trigger_step")
                == V4_COORDINATION_TRIGGER_STEP
                and replay.get("committed_policy_steps", 0)
                >= (
                    V4_COORDINATION_TRIGGER_STEP
                    + V4_COORDINATION_SMOOTH_POLICY_TICKS
                )
                and replay.get("perturbed_steps")
                == V4_COORDINATION_SMOOTH_POLICY_TICKS
                and intervention.get("smooth_policy_ticks_planned")
                == V4_COORDINATION_SMOOTH_POLICY_TICKS
                and intervention.get("smooth_policy_ticks_elapsed")
                == V4_COORDINATION_SMOOTH_POLICY_TICKS
                and intervention.get("completed") is True
                and intervention.get("terminal_during_smooth_window") is False
                and intervention.get("policy_requests")
                == V4_COORDINATION_SMOOTH_POLICY_TICKS
                and intervention.get("policy_clock_advances")
                == V4_COORDINATION_SMOOTH_POLICY_TICKS
                and intervention.get("persistent_offset") is True
            )
        return int(replay.get("perturbed_steps", 0)) > 0
    if source.get("scenario") == "static":
        return True
    if _is_v4_store_source(source):
        mode = replay.get("store_mode")
        if mode not in V4_STORE_MODE_MOVED_ENTITIES:
            return False
        required = sorted(V4_STORE_MODE_MOVED_ENTITIES[mode])
        if replay.get("store_required_entities") != required:
            return False
        if (
            replay.get("dynamic_condition_exercised") is not True
            or replay.get("intervention_effective") is not True
            or replay.get("intervention_complete") is not True
        ):
            return False
        per_entity = replay.get("store_entity_interventions")
        events = replay.get("scenario_events")
        if not isinstance(per_entity, dict) or not isinstance(events, list):
            return False
        for entity in required:
            status = per_entity.get(entity)
            if (
                not isinstance(status, dict)
                or status.get("required") is not True
                or status.get("trigger_step") != V4_STORE_TRIGGER_STEPS[entity]
                or status.get("eligible") is not True
                or status.get("reached") is not True
                or status.get("applied") is not True
                or status.get("effective") is not True
                or status.get("complete") is not True
            ):
                return False
        if len(events) != len(required):
            return False
        event_entities = []
        for event in events:
            if not isinstance(event, dict):
                return False
            entity = event.get("entity")
            if (
                entity not in required
                or event.get("kind") != "teleport_store_entity"
                or event.get("step") != V4_STORE_TRIGGER_STEPS[entity]
                or event.get("trigger_step") != V4_STORE_TRIGGER_STEPS[entity]
                or event.get("applied") is not True
                or event.get("complete") is not True
                or event.get("protocol_effective") is not True
                or event.get("policy_observation_refreshed") is not True
                or event.get("new_robot_collision_pairs") != []
            ):
                return False
            event_entities.append(entity)
        return sorted(event_entities) == required and len(set(event_entities)) == len(
            required
        )
    if _is_v4_lift_source(source):
        events = replay.get("scenario_events")
        if (
            replay.get("trigger_step") != V4_LIFT_TRIGGER_STEP
            or replay.get("intervention_eligible") is not True
            or replay.get("intervention_reached") is not True
            or replay.get("pre_intervention_terminal") is not False
            or replay.get("dynamic_condition_exercised") is not True
            or replay.get("intervention_effective") is not True
            or replay.get("intervention_complete") is not True
            or replay.get("motion_plan_protocol_id") != V4_LIFT_MOTION_PROTOCOL_ID
            or not isinstance(events, list)
            or len(events) != 1
        ):
            return False
        event = events[0]
        audit = event.get("formal_intervention_state_audit")
        reference = event.get("motion_plan_reference")
        return bool(
            isinstance(event, dict)
            and event.get("kind") == "teleport_task"
            and event.get("step") == V4_LIFT_TRIGGER_STEP
            and event.get("trigger_step") == V4_LIFT_TRIGGER_STEP
            and event.get("applied") is True
            and event.get("protocol_effective") is True
            and event.get("policy_observation_refreshed") is True
            and event.get("commanded_root_pose_reached") is True
            and event.get("goal_root_pose_reached") is True
            and event.get("actual_root_motion") is True
            and event.get("motion_protocol", {}).get("protocol_id")
            == V4_LIFT_MOTION_PROTOCOL_ID
            and isinstance(audit, dict)
            and audit.get("passed") is True
            and audit.get("no_new_robot_external_collision_pairs") is True
            and isinstance(reference, dict)
            and reference.get("motion_plan_fingerprint")
            == replay.get("motion_plan_fingerprint")
        )
    events = (
        replay.get("scenario_events", [])
        if family == "bimanual"
        else replay.get("interventions", [])
    )
    effective = [
        event
        for event in events
        if event.get("applied") and event.get("protocol_effective") is True
    ]
    if source.get("scenario") != "smooth":
        return bool(effective)

    protocol = (
        source.get("scenario_protocol", {}) if family == "bimanual" else source.get("protocol", {})
    )
    expected_calls = protocol.get(
        "smooth_interpolation_calls",
        protocol.get("smooth_motion_calls", 10),
    )
    expected_calls = 10 if expected_calls is None else int(expected_calls)
    return (
        len(events) == expected_calls
        and len(effective) == expected_calls
        and events[-1].get("complete") is True
        and events[-1].get("endpoint_applied") is True
    )


def _protocol_condition_signature(source, row):
    """Project one canonical result row onto its intervention-accounting state.

    ``None`` means that the row predates the current explicit accounting
    fields.  The projection intentionally excludes policy outcome and timing:
    those are checked independently by the replay confirmation path.
    """

    family = _source_family(source, _source_policy_task(source))
    if source.get("scenario") == "static":
        return ("static_reference",)
    if family == "coordination":
        if _is_v4_coordination_source(source):
            if "coordination_intervention" not in row:
                return None
            intervention = row.get("coordination_intervention")
            if intervention is None:
                return ("v4_coordination", "pre_trigger_terminal")
            if not isinstance(intervention, dict):
                return None
            return (
                "v4_coordination",
                "intervention",
                intervention.get("smooth_policy_ticks_elapsed"),
                intervention.get("completed"),
                intervention.get("terminal_during_smooth_window"),
                intervention.get("policy_clock_advances"),
                intervention.get("persistent_offset"),
            )
        if "perturbed_steps" not in row:
            return None
        return ("coordination", bool(int(row.get("perturbed_steps", 0)) > 0))

    accounting_fields = (
        "intervention_eligible",
        "intervention_reached",
        "pre_intervention_terminal",
        "dynamic_condition_exercised",
        "dynamic_condition_unexercised",
        "intervention_effective",
        "intervention_complete",
    )
    if any(field not in row for field in accounting_fields):
        return None
    event_key = "scenario_events" if family == "bimanual" else "interventions"
    events = row.get(event_key)
    if not isinstance(events, list) or any(
        not isinstance(event, dict) for event in events
    ):
        return None
    applied_progress = tuple(
        (
            event.get("kind"),
            event.get("entity"),
            event.get("applied"),
            event.get("protocol_effective"),
            event.get("complete"),
            event.get("endpoint_applied"),
        )
        for event in events
        if event.get("applied") is True
    )
    store_progress = ()
    if _is_v4_store_source(source):
        required = row.get("store_required_entities")
        per_entity = row.get("store_entity_interventions")
        if not isinstance(required, list) or not isinstance(per_entity, dict):
            return None
        store_progress = tuple(
            (
                entity,
                per_entity.get(entity, {}).get("eligible"),
                per_entity.get(entity, {}).get("reached"),
                per_entity.get(entity, {}).get("applied"),
                per_entity.get(entity, {}).get("effective"),
                per_entity.get(entity, {}).get("complete"),
            )
            for entity in required
            if isinstance(entity, str)
        )
        if len(store_progress) != len(required):
            return None
    return (
        "dynamic_environment",
        *(row.get(field) for field in accounting_fields),
        applied_progress,
        store_progress,
    )


def _source_condition_reproduced(source, original, replay):
    """Compare canonical source/replay intervention states.

    Legacy rows without explicit episode accounting keep the historical
    effective-intervention check.  Current formal rows must reproduce the same
    exercised, pre-trigger, or effective-prefix state.
    """

    source_signature = _protocol_condition_signature(source, original)
    if source_signature is None:
        return _protocol_effective(source, replay)
    return _protocol_condition_signature(source, replay) == source_signature


def _validate_replay(
    original,
    replay,
    source_identity,
    loaded_identity,
    source=None,
    *,
    expected_success=False,
):
    if loaded_identity != source_identity:
        raise RuntimeError("loaded model identity differs from the source evaluation")
    if bool(original.get("success")) is not bool(expected_success):
        expected = "success" if expected_success else "failure"
        raise ValueError(f"the selected source row is not a {expected}")
    if bool(replay.get("success")) is not bool(expected_success):
        expected = "successful" if expected_success else "failed"
        actual = "successful" if bool(replay.get("success")) else "failed"
        raise RuntimeError(
            f"episode {original.get('episode')} was {actual} during replay; "
            f"expected it to remain {expected}"
        )
    if replay.get("episode") != original.get("episode"):
        raise RuntimeError("replay episode identity differs from the source row")
    if int(replay.get("invalid_actions", 0)) != int(original.get("invalid_actions", 0)):
        raise RuntimeError(
            f"episode {original.get('episode')} changed its invalid-action count; "
            "no trajectory video will be published"
        )
    if source is not None and not _source_condition_reproduced(
        source,
        original,
        replay,
    ):
        raise RuntimeError(
            f"episode {original.get('episode')} did not reproduce the source "
            "condition; no trajectory video will be published"
        )
    confirmation = {
        (
            "replay_confirmed_success"
            if expected_success
            else "replay_confirmed_failure"
        ): True,
        "expected_outcome": "success" if expected_success else "failure",
        "same_reason": replay.get("reason") == original.get("reason"),
        "same_steps": replay.get("steps") == original.get("steps"),
        "same_invalid_actions": True,
    }
    if source is not None:
        confirmation["source_condition_reproduced"] = True
    return confirmation


def _observation_config(cameras, resolution):
    from rlbench.observation_config import CameraConfig, ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.gripper_open = True
    config.gripper_pose = True
    config.task_low_dim_state = True
    config.camera_configs = {
        camera: CameraConfig(
            rgb=True,
            depth=False,
            point_cloud=False,
            mask=False,
            image_size=tuple(resolution),
        )
        for camera in cameras
    }
    return config


def _configure_replay_cameras(
    environment,
    cameras,
    overhead_perspective_degrees,
    all_perspective_degrees=None,
):
    """Zoom the overhead replay view without changing camera poses or front FOV."""

    requested = tuple(cameras)
    if all_perspective_degrees is not None:
        desired = {camera: float(all_perspective_degrees) for camera in requested}
        mode = "fixed_all_camera_perspective_same_pose_v2"
    else:
        desired = {
            camera: (
                float(overhead_perspective_degrees)
                if camera == "overhead" and overhead_perspective_degrees is not None
                else None
            )
            for camera in requested
        }
        mode = "overhead_zoom_same_pose_front_unchanged_v2"
    for camera, angle in desired.items():
        if angle is not None and not 20.0 <= angle <= 120.0:
            raise ValueError(
                f"replay perspective for {camera} must be between 20 and 120 degrees"
            )
    if not any(angle is not None for angle in desired.values()):
        return {
            "mode": "scene_default",
            "perspective_degrees_by_camera": desired,
            "camera_pose_changed": False,
            "policy_inputs_changed": False,
        }
    scene = getattr(environment, "_scene", None)
    sensors = getattr(scene, "camera_sensors", None)
    if not isinstance(sensors, dict):
        raise RuntimeError("RLBench camera sensors are unavailable after launch")
    rows = {}
    for camera in requested:
        sensor = sensors.get(camera)
        if sensor is None:
            raise RuntimeError(f"requested replay camera is unavailable: {camera}")
        pose_before = [float(value) for value in sensor.get_pose()]
        before = float(sensor.get_perspective_angle())
        angle = desired[camera]
        if angle is not None:
            sensor.set_perspective_angle(angle)
        after = float(sensor.get_perspective_angle())
        pose_after = [float(value) for value in sensor.get_pose()]
        expected = before if angle is None else angle
        if abs(after - expected) > 1.0e-4 or not np.allclose(
            pose_before,
            pose_after,
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise RuntimeError("replay camera configuration did not apply exactly")
        # Prime each explicit sensor before a task model is loaded.  CoppeliaSim
        # otherwise returns a permanently black first render from cam_front for
        # some task models (notably WipeDesk), while later sensors render
        # normally.  This only initializes replay rendering; no captured frame
        # is retained and policy observations are unchanged.
        sensor.handle_explicitly()
        primed_frame = np.asarray(sensor.capture_rgb())
        if primed_frame.ndim != 3 or primed_frame.shape[-1] != 3:
            raise RuntimeError("replay camera pre-task render has an invalid shape")
        rows[camera] = {
            "perspective_configured": angle is not None,
            "perspective_degrees_before": before,
            "perspective_degrees_after": after,
            "pose_before": pose_before,
            "pose_after": pose_after,
            "pre_task_render_primed": True,
        }
    return {
        "mode": mode,
        "perspective_degrees_by_camera": desired,
        "camera_pose_changed": False,
        "policy_inputs_changed": False,
        "pre_task_render_primed": True,
        "cameras": rows,
    }


def _source_protocol_budgets(source):
    v4_store = _is_v4_store_source(source)
    v4_lift = _is_v4_lift_source(source)
    v4_coordination = _is_v4_coordination_source(source)
    intervention = (
        load_v4_store_intervention_protocol()
        if v4_store
        else (
            load_v4_lift_intervention_protocol()
            if v4_lift
            else (
                load_v4_coordination_intervention_protocol()
                if v4_coordination
                else load_v3_intervention_protocol()
            )
        )
    )
    motion = (
        load_v4_store_motion_source_protocol()
        if v4_store
        else (
            load_v4_lift_motion_source_protocol()
            if v4_lift
            else load_v3_motion_source_protocol()
        )
    )
    family = _source_family(source, _source_policy_task(source))
    settling = source.get("final_settling_protocol")
    controller = source.get("controller")
    if not isinstance(controller, dict):
        raise RuntimeError("source result has no controller identity")
    current_controller = global_ik_controller_metadata()
    current_markers = (
        "primary_action_requests_per_policy_tick",
        "policy_clock_semantics_id",
    )
    if any(marker in controller for marker in current_markers):
        if (
            any(
                controller.get(field) != value
                for field, value in current_controller.items()
            )
            or controller.get("worker_clock_handshake_id")
            != FORMAL_POLICY_CLOCK_SEMANTICS_ID
            or "primary_action_retry" in controller
            or controller.get("primary_action_requests_per_policy_tick")
            != DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
        ):
            raise RuntimeError("source result has a noncanonical formal controller")
        primary_action_attempts = DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
    else:
        retry = controller.get("primary_action_retry")
        if (
            not isinstance(retry, dict)
            or retry.get("max_primary_action_attempts_per_policy_tick")
            != LEGACY_MAX_PRIMARY_ACTION_ATTEMPTS
        ):
            raise RuntimeError("source result has noncanonical execution budgets")
        primary_action_attempts = LEGACY_MAX_PRIMARY_ACTION_ATTEMPTS
    expected_settling_steps = intervention.get(
        "final_settling_physics_steps",
        DEFAULT_FINAL_SETTLING_PHYSICS_STEPS if v4_coordination else None,
    )
    if (
        not isinstance(settling, dict)
        or settling.get("maximum_physics_steps")
        != expected_settling_steps
    ):
        raise RuntimeError("source result has noncanonical execution budgets")
    protocol = (
        source.get("scenario_protocol", {})
        if family == "bimanual"
        else source.get("protocol", {}) if family == "unimanual" else {}
    )
    if family == "bimanual" and (
        protocol.get("max_sampling_attempts")
        != motion["goal_sampling_max_attempts"]
        or protocol.get("smooth_interpolation_calls")
        != (
            intervention.get("smooth_steps")
            if source.get("scenario") == "smooth"
            else None
        )
    ):
        raise RuntimeError("source bimanual intervention budgets are invalid")
    if family == "unimanual" and (
        protocol.get("intervention_max_attempts")
        != motion["goal_sampling_max_attempts"]
        or protocol.get("smooth_motion_calls") != intervention["smooth_steps"]
    ):
        raise RuntimeError("source unimanual intervention budgets are invalid")
    return {
        "smooth_steps": (
            DEFAULT_SCENARIO_STEPS
            if v4_lift
            else intervention.get("smooth_steps")
        ),
        "motion_attempts": motion["goal_sampling_max_attempts"],
        "final_settling_steps": expected_settling_steps,
        "primary_action_attempts": primary_action_attempts,
    }


def _authenticated_replay_trigger(source, task, worker, budgets):
    """Re-resolve the recorded trigger from the loaded checkpoint identity."""

    family = _source_family(source, task)
    if _is_v4_store_source(source, task):
        protocol = source.get("scenario_protocol", {})
        args = SimpleNamespace(
            release="v4",
            task=task,
            scenario=source.get("scenario"),
            scenario_steps=budgets["smooth_steps"],
            scenario_max_attempts=budgets["motion_attempts"],
            final_settling_steps=budgets["final_settling_steps"],
            scenario_trigger_step=None,
            scenario_reference_steps=protocol.get("trigger_reference_steps"),
            models_dir=direct_evaluate.V4_MODELS_DIR,
        )
        registry, authentication = direct_evaluate._authenticated_v4_store_triggers(
            args,
            worker,
        )
        trigger = (
            None
            if source.get("scenario") == "static"
            else {
                name: authentication["triggers"][name]["trigger_step"]
                for name in sorted(V4_STORE_TRIGGER_STEPS)
            }
        )
    elif _is_v4_lift_source(source, task):
        protocol = source.get("scenario_protocol", {})
        requested_trigger = (
            None
            if source.get("scenario") == "static"
            else protocol.get("trigger_policy_step")
        )
        args = SimpleNamespace(
            release="v4",
            task=task,
            scenario=source.get("scenario"),
            scenario_steps=budgets["smooth_steps"],
            scenario_max_attempts=budgets["motion_attempts"],
            final_settling_steps=budgets["final_settling_steps"],
            scenario_trigger_step=requested_trigger,
            scenario_reference_steps=protocol.get("trigger_reference_steps"),
        )
        registry, authentication = direct_evaluate._authenticated_v4_lift_trigger(
            args,
            worker,
        )
        trigger = (
            None
            if source.get("scenario") == "static"
            else authentication["trigger_step"]
        )
    elif _is_v4_coordination_source(source):
        protocol = source["coordination_protocol"]
        args = SimpleNamespace(
            arm=protocol["perturbed_arm"],
            trigger_step=protocol.get("trigger_policy_step"),
            max_primary_action_attempts=budgets["primary_action_attempts"],
        )
        registry, authentication, trigger = (
            table_iii_coordination._authenticated_v4_coordination_trigger(
                args,
                worker,
            )
        )
    elif family == "coordination":
        protocol = source["coordination_protocol"]
        args = SimpleNamespace(
            arm=protocol["perturbed_arm"],
            trigger_step=protocol.get("trigger_policy_step"),
            final_settling_steps=budgets["final_settling_steps"],
        )
        registry, authentication, trigger = (
            table_iii_coordination._authenticated_v3_coordination_trigger(
                args,
                worker,
            )
        )
    else:
        protocol = (
            source.get("scenario_protocol", {})
            if family == "bimanual"
            else source.get("protocol", {})
        )
        requested_trigger = (
            None
            if source.get("scenario") == "static"
            else protocol.get("trigger_policy_step")
        )
        if family == "bimanual":
            args = SimpleNamespace(
                task=task,
                scenario_steps=budgets["smooth_steps"],
                scenario_max_attempts=budgets["motion_attempts"],
                final_settling_steps=budgets["final_settling_steps"],
                scenario_trigger_step=requested_trigger,
                scenario_reference_steps=protocol.get("trigger_reference_steps"),
            )
            registry, authentication = (
                direct_evaluate._authenticated_v3_dynamic_trigger(args, worker)
            )
        else:
            args = SimpleNamespace(
                task=task,
                smooth_steps=budgets["smooth_steps"],
                intervention_attempts=budgets["motion_attempts"],
                final_settling_steps=budgets["final_settling_steps"],
                trigger_step=requested_trigger,
            )
            registry, authentication = (
                unimanual_evaluate._authenticated_v3_dynamic_trigger(args, worker)
            )
        trigger = (
            None
            if source.get("scenario") == "static"
            else authentication["trigger_step"]
        )

    if (
        protocol.get("trigger_authentication") != authentication
        or protocol.get("trigger_policy_step") != trigger
        or protocol.get("intervention_registry_schema") != registry["schema"]
        or protocol.get("intervention_registry_fingerprint")
        != registry["fingerprint"]
    ):
        raise RuntimeError(
            "source trigger is not authenticated by the loaded checkpoint"
        )
    return trigger


def _run_selected_episode(
    task_environment,
    worker,
    task,
    source,
    episode,
    *,
    motion_plan,
    trigger_step,
    descriptions,
    observation,
    fresh_task_generation,
    staged_source_binding=None,
    budgets,
):
    family = _source_family(source, task)
    if family == "coordination":
        protocol = source["coordination_protocol"]
        return table_iii_coordination._run_episode(
            task_environment,
            worker,
            episode=episode,
            variation=motion_plan.variation,
            seed=source["seed"],
            horizon=source["horizon"],
            arm=protocol["perturbed_arm"],
            trigger=int(trigger_step),
            max_primary_action_attempts=budgets["primary_action_attempts"],
            final_settling_steps=budgets["final_settling_steps"],
            descriptions=descriptions,
            observation=observation,
            fresh_task_generation=fresh_task_generation,
            staged_source_binding=staged_source_binding,
            release="v4" if _is_v4_coordination_source(source) else "v3",
        )
    if family == "bimanual" and _is_v4_store_source(source, task):
        return direct_evaluate._run_v4_store_episode(
            task_environment,
            worker,
            episode,
            source["horizon"],
            scenario=source["scenario"],
            max_primary_action_attempts=budgets["primary_action_attempts"],
            motion_plan=motion_plan,
            final_settling_steps=budgets["final_settling_steps"],
            descriptions=descriptions,
            observation=observation,
            fresh_task_generation=fresh_task_generation,
        )
    if family == "bimanual":
        protocol = source.get("scenario_protocol", {})
        return direct_evaluate._run_episode(
            task_environment,
            worker,
            episode,
            source["seed"],
            source["horizon"],
            scenario=source["scenario"],
            scenario_trigger_fraction=protocol.get(
                "trigger_fraction_of_nominal_policy_length",
                protocol.get("trigger_fraction", 1.0 / 3.0),
            ),
            scenario_trigger_step=trigger_step,
            scenario_reference_steps=worker.policy_steps,
            scenario_steps=budgets["smooth_steps"],
            scenario_max_attempts=budgets["motion_attempts"],
            max_primary_action_attempts=budgets["primary_action_attempts"],
            motion_plan=motion_plan,
            final_settling_steps=budgets["final_settling_steps"],
            descriptions=descriptions,
            observation=observation,
            fresh_task_generation=fresh_task_generation,
        )
    protocol = source.get("protocol", {})
    run_args = SimpleNamespace(
        seed=source["seed"],
        variation=motion_plan.variation,
        horizon=source["horizon"],
        scenario=source["scenario"],
        trigger_fraction=protocol.get(
            "trigger_fraction_of_nominal_policy_length",
            protocol.get("trigger_fraction_of_fitted_policy", 1.0 / 3.0),
        ),
        trigger_step=trigger_step,
        smooth_steps=budgets["smooth_steps"],
        intervention_attempts=budgets["motion_attempts"],
        max_primary_action_attempts=budgets["primary_action_attempts"],
        final_settling_steps=budgets["final_settling_steps"],
    )
    return unimanual_evaluate._run_episode(
        task_environment,
        worker,
        run_args,
        episode,
        motion_plan=motion_plan,
        descriptions=descriptions,
        observation=observation,
        fresh_task_generation=fresh_task_generation,
    )


def _v4_models_dir(source, task, family):
    """Return the only admitted checkpoint root for a V4 replay."""

    if not _is_v4_source(source, task):
        return None
    suffix = ("table_iii",) if family == "coordination" else ()
    return INTEGRATION_ROOT.joinpath("models", "v4", *suffix)


def _runtime_components(args, source):
    from rlbench.environment import Environment

    family = _source_family(source, args.task)
    v4_models_dir = _v4_models_dir(source, args.task, family)
    if family == "coordination":
        module_name = table_iii_coordination.TASK_MODULE
        class_name = table_iii_coordination.TASK_CLASS
        action_mode = direct_evaluate._make_action_mode()
        environment = Environment(
            action_mode=action_mode,
            obs_config=_observation_config(args.cameras, args.resolution),
            headless=args.headless,
            robot_setup="dual_panda",
        )
        worker_class = direct_evaluate.PolicyProcess
        default_python = table_iii_coordination.DEFAULT_POLICY_PYTHON
        default_models_dir = v4_models_dir or table_iii_coordination.DEFAULT_MODELS_DIR
    elif family == "bimanual":
        v4_store = _is_v4_store_source(source, args.task)
        if not v4_store:
            module_name, class_name = direct_evaluate.TASKS[args.task]
        action_mode = direct_evaluate._make_action_mode()
        environment = Environment(
            action_mode=action_mode,
            obs_config=_observation_config(args.cameras, args.resolution),
            headless=args.headless,
            robot_setup="dual_panda",
        )
        worker_class = direct_evaluate.PolicyProcess
        default_python = direct_evaluate.DEFAULT_POLICY_PYTHON
        default_models_dir = v4_models_dir or direct_evaluate.DEFAULT_MODELS_DIR
    else:
        module_name, class_name = unimanual_evaluate.TASKS[args.task]
        action_mode = unimanual_evaluate._make_action_mode()
        environment = Environment(
            action_mode=action_mode,
            obs_config=_observation_config(args.cameras, args.resolution),
            headless=args.headless,
        )
        worker_class = unimanual_evaluate.PolicyProcess
        default_python = unimanual_evaluate.DEFAULT_POLICY_PYTHON
        default_models_dir = v4_models_dir or unimanual_evaluate.DEFAULT_MODELS_DIR
    if family == "bimanual" and _is_v4_store_source(source, args.task):
        task_class = direct_evaluate.v4_store_task_class()
    else:
        task_class = getattr(importlib.import_module(module_name), class_name)
    policy_python = args.policy_python or default_python
    models_dir = args.models_dir or default_models_dir
    if v4_models_dir is not None and (
        Path(models_dir).resolve() != v4_models_dir.resolve()
    ):
        raise RuntimeError(
            f"V4 replay requires the current release model root: {v4_models_dir}"
        )
    worker = worker_class(
        policy_python,
        args.task,
        models_dir,
        timeout=args.policy_timeout,
    )
    return environment, worker, task_class


def _default_output_key(source, task):
    scenario = source["scenario"]
    return task if scenario == "static" else f"{task}_{scenario}"


def _validate_output_key(value):
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value
    ):
        raise ValueError("--output-key must contain only lowercase letters, digits, '_' or '-'")
    return value


def _validate_replay_output_target(target):
    """Reject repository destinations outside the two replay-only roots."""

    candidate = Path(target)
    if candidate.is_symlink():
        raise RuntimeError("refusing to replace a replay-directory symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return resolved
    inside_replay_root = False
    for root in REPLAY_OUTPUT_ROOTS:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved != root:
            inside_replay_root = True
            break
    if not inside_replay_root:
        raise ValueError(
            "repository replay output must stay below results/failure_videos "
            "or results/v4/replay_video; formal results are never overwrite targets"
        )
    return resolved


def _publish_replay_directory(staging, target, *, overwrite):
    """Atomically install a complete replay directory, restoring on failure."""

    staging = Path(staging)
    target = _validate_replay_output_target(target)
    if not staging.is_dir():
        raise RuntimeError("replay staging directory is unavailable")
    if target.is_symlink():
        raise RuntimeError("refusing to replace a replay-directory symlink")
    if target.exists() and not target.is_dir():
        raise RuntimeError("replay output target exists and is not a directory")
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite trajectory videos: {target}")
    if not target.exists():
        os.replace(str(staging), str(target))
        return {"overwritten": False, "atomic_directory_swap": True}

    backup_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.backup-", dir=str(target.parent))
    )
    backup = backup_root / "previous"
    moved_previous = False
    try:
        os.replace(str(target), str(backup))
        moved_previous = True
        os.replace(str(staging), str(target))
    except Exception:
        if moved_previous and backup.exists() and not target.exists():
            os.replace(str(backup), str(target))
        if backup_root.exists() and not any(backup_root.iterdir()):
            backup_root.rmdir()
        raise
    else:
        shutil.rmtree(backup_root)
    return {"overwritten": True, "atomic_directory_swap": True}


def _validate_max_replay_attempts(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("--max-replay-attempts-per-episode must be an integer")
    if value < 1:
        raise ValueError("--max-replay-attempts-per-episode must be positive")
    return value


def _replay_attempt_protocol(maximum_attempts):
    maximum_attempts = _validate_max_replay_attempts(maximum_attempts)
    return {
        "protocol_id": REPLAY_ATTEMPT_PROTOCOL_ID,
        "max_replay_attempts_per_episode": maximum_attempts,
        "attempt_ordinals_are_one_based": True,
        "fresh_task_generation_per_attempt": True,
        "seed_variation_plan_and_controller_unchanged_between_attempts": True,
        "retryable_completed_result_mismatches": [
            "outcome",
            "source_condition",
            "invalid_actions",
        ],
        "candidate_traversal": "episode_major_in_caller_order",
        "exceptions_fail_closed": True,
        "first_matching_attempt_only": True,
        "one_video_per_source_episode": True,
        "source_evaluation_immutable": True,
        "replays_are_video_confirmation_not_evaluation": True,
    }


def _replay_output_stem(episode, episode_seed, attempt_ordinal):
    return (
        f"episode_{episode:03d}_seed_{episode_seed:03d}_"
        f"attempt_{attempt_ordinal:03d}"
    )


def _assess_replay_attempt(original, replay, source, *, expected_success):
    """Classify only the three completed-result mismatches as retryable."""

    if replay.get("episode") != original.get("episode"):
        raise RuntimeError("replay episode identity differs from the source row")
    if bool(original.get("success")) is not bool(expected_success):
        expected = "success" if expected_success else "failure"
        raise ValueError(f"the selected source row is not a {expected}")

    outcome_matched = bool(replay.get("success")) is bool(expected_success)
    condition_reproduced = _source_condition_reproduced(
        source,
        original,
        replay,
    )
    invalid_actions_matched = int(replay.get("invalid_actions", 0)) == int(
        original.get("invalid_actions", 0)
    )
    confirmed = bool(
        outcome_matched and condition_reproduced and invalid_actions_matched
    )
    if confirmed:
        disposition = "published_first_match"
    elif not outcome_matched:
        disposition = "discarded_outcome_mismatch"
    elif not condition_reproduced:
        disposition = "discarded_source_condition_mismatch"
    else:
        disposition = "discarded_invalid_actions_mismatch"
    return {
        "replay_confirmed_outcome": outcome_matched,
        "source_condition_reproduced": condition_reproduced,
        "same_invalid_actions": invalid_actions_matched,
        "confirmed_for_publication": confirmed,
        "disposition": disposition,
    }


def _run_bounded_replay_attempts(maximum_attempts, runner):
    """Run a source episode until its first match or its fixed bound."""

    maximum_attempts = _validate_max_replay_attempts(maximum_attempts)
    records = []
    for ordinal in range(1, maximum_attempts + 1):
        record = runner(ordinal)
        if not isinstance(record, dict):
            raise TypeError("replay attempt runner must return a dictionary")
        if record.get("replay_attempt_ordinal") != ordinal:
            raise RuntimeError("replay attempt ordinal is inconsistent")
        if not isinstance(record.get("confirmed_for_publication"), bool):
            raise RuntimeError("replay attempt publication decision is missing")
        records.append(record)
        if record["confirmed_for_publication"]:
            return record, records
    return None, records


def record(args):
    episodes = _normalize_episode_indices(args.episode)
    warmup_episodes = tuple(
        _normalize_episode_indices(getattr(args, "warmup_episode", None))
        if getattr(args, "warmup_episode", None)
        else ()
    )
    overlap = sorted(set(episodes) & set(warmup_episodes))
    if overlap:
        raise ValueError(f"warm-up and recorded episodes overlap: {overlap}")
    expected_success = args.expected_outcome == "success"
    expected_outcome = "success" if expected_success else "failure"
    schema = SCHEMA
    maximum_attempts = _validate_max_replay_attempts(
        getattr(args, "max_replay_attempts_per_episode", 1)
    )
    attempt_protocol = _replay_attempt_protocol(maximum_attempts)
    minimum_confirmed = len(episodes) if args.minimum_confirmed is None else args.minimum_confirmed
    if minimum_confirmed < 1 or minimum_confirmed > len(episodes):
        raise ValueError("--minimum-confirmed must be between one and the number of candidates")
    source_path = args.source_result or DEFAULT_SOURCE_RESULTS.get(args.task)
    if source_path is None:
        raise ValueError(f"--source-result is required because {args.task!r} has no default")
    source, originals, source_sha, resolved_source = _load_source(
        source_path,
        args.task,
        episodes,
        expected_success=expected_success,
    )
    warmup_originals = _auxiliary_source_rows(source, warmup_episodes)
    replay_protocol_id = _require_current_evaluator_protocol(source, args.task)
    replay_batch = _load_sealed_replay_batch(source, args.task)
    budgets = _source_protocol_budgets(source)
    output_key = _validate_output_key(args.output_key or _default_output_key(source, args.task))
    target = _validate_replay_output_target(
        Path(args.output_root).resolve() / output_key
    )
    overwrite = bool(getattr(args, "overwrite", True))
    target_existed_before_run = target.exists()
    if target_existed_before_run and not overwrite:
        raise FileExistsError(f"refusing to overwrite trajectory videos: {target}")
    if target.is_symlink() or (target_existed_before_run and not target.is_dir()):
        raise RuntimeError("replay output target must be a real directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_key}.staging-", dir=str(target.parent)))
    environment = worker = None
    launched = False
    manifest_rows = []
    attempts = []
    warmup_ledger = []
    camera_configuration = None
    try:
        environment, worker, task_class = _runtime_components(args, source)
        if worker.model_identity != source["model_identity"]:
            raise RuntimeError("loaded model identity differs from the source evaluation")
        trigger_step = _authenticated_replay_trigger(
            source,
            args.task,
            worker,
            budgets,
        )
        environment.launch()
        launched = True
        camera_configuration = _configure_replay_cameras(
            environment,
            args.cameras,
            getattr(
                args,
                "overhead_perspective_degrees",
                DEFAULT_REPLAY_OVERHEAD_PERSPECTIVE_DEGREES,
            ),
            getattr(args, "camera_perspective_degrees", None),
        )
        for warmup_episode in warmup_episodes:
            warmup_plan, warmup_variation, warmup_reset_seed = (
                _sealed_episode_plan(
                    replay_batch,
                    warmup_originals[warmup_episode],
                    warmup_episode,
                )
            )
            (
                warmup_environment,
                warmup_descriptions,
                warmup_observation,
                warmup_generation,
            ) = initialize_fresh_task_generation(
                environment,
                task_class,
                episode_seed=warmup_reset_seed,
                variation=warmup_variation,
                verify_instance=False,
            )
            warmup_binding = None
            if replay_batch["family"] == "coordination":
                warmup_binding = bind_staged_source_plan(
                    warmup_environment,
                    warmup_plan,
                    descriptions=warmup_descriptions,
                    fresh_task_generation=warmup_generation,
                )
                warmup_observation = warmup_environment.get_observation()
            warmup_replay = _run_selected_episode(
                warmup_environment,
                worker,
                args.task,
                source,
                warmup_episode,
                motion_plan=warmup_plan,
                trigger_step=trigger_step,
                descriptions=warmup_descriptions,
                observation=warmup_observation,
                fresh_task_generation=warmup_generation,
                staged_source_binding=warmup_binding,
                budgets=budgets,
            )
            _validate_replay_plan(replay_batch, warmup_replay, warmup_plan)
            warmup_source = warmup_originals[warmup_episode]
            warmup_ledger.append(
                {
                    "episode": warmup_episode,
                    "episode_seed": source["seed"] + warmup_episode,
                    "formal_reset_seed": warmup_reset_seed,
                    "variation": warmup_variation,
                    "sealed_plan_fingerprint": warmup_plan.fingerprint(),
                    "fresh_generation_index": warmup_generation.get(
                        "generation_index"
                    ),
                    "source_outcome": (
                        "success" if warmup_source.get("success") else "failure"
                    ),
                    "replay_outcome": (
                        "success" if warmup_replay.get("success") else "failure"
                    ),
                    "source_condition_effective": _protocol_effective(
                        source,
                        warmup_source,
                    ),
                    "replay_condition_effective": _protocol_effective(
                        source,
                        warmup_replay,
                    ),
                    "source_invalid_actions": int(
                        warmup_source.get("invalid_actions", 0)
                    ),
                    "replay_invalid_actions": int(
                        warmup_replay.get("invalid_actions", 0)
                    ),
                    "source_reason": warmup_source.get("reason"),
                    "replay_reason": warmup_replay.get("reason"),
                    "video_recorded": False,
                    "purpose": "same_environment_generation_history_prefix",
                }
            )
            print(
                f"{args.task} warm-up episode {warmup_episode}: "
                f"{warmup_ledger[-1]['replay_outcome']}, no video",
                flush=True,
            )
        for episode in episodes:
            episode_seed = source["seed"] + episode
            motion_plan, variation, formal_reset_seed = _sealed_episode_plan(
                replay_batch,
                originals[episode],
                episode,
            )
            plan_fingerprint = motion_plan.fingerprint()

            def run_attempt(attempt_ordinal):
                (
                    task_environment,
                    descriptions,
                    observation,
                    fresh_task_generation,
                ) = initialize_fresh_task_generation(
                    environment,
                    task_class,
                    episode_seed=formal_reset_seed,
                    variation=variation,
                    verify_instance=False,
                )
                stem = _replay_output_stem(
                    episode,
                    episode_seed,
                    attempt_ordinal,
                )
                video_path = staging / f"{stem}.mp4"
                recorder = ObservationRecorder(
                    video_path,
                    cameras=args.cameras,
                    fps=args.fps,
                    ffmpeg=args.ffmpeg,
                )
                proxy = RecordingTaskEnvironment(task_environment, recorder)
                try:
                    recorder.capture(observation)
                    staged_source_binding = None
                    if replay_batch["family"] == "coordination":
                        staged_source_binding = bind_staged_source_plan(
                            proxy,
                            motion_plan,
                            descriptions=descriptions,
                            fresh_task_generation=fresh_task_generation,
                        )
                        observation = proxy.get_observation()
                    replay = _run_selected_episode(
                        proxy,
                        worker,
                        args.task,
                        source,
                        episode,
                        motion_plan=motion_plan,
                        trigger_step=trigger_step,
                        descriptions=descriptions,
                        observation=observation,
                        fresh_task_generation=fresh_task_generation,
                        staged_source_binding=staged_source_binding,
                        budgets=budgets,
                    )
                    _validate_replay_plan(replay_batch, replay, motion_plan)
                    recorder.close()
                    if _is_v4_source(source, args.task) and tuple(
                        recorder.used_cameras
                    ) != tuple(args.cameras):
                        raise RuntimeError(
                            "V4 delivery requires every requested camera"
                        )
                except Exception:
                    recorder.abort()
                    raise

                assessment = _assess_replay_attempt(
                    originals[episode],
                    replay,
                    source,
                    expected_success=expected_success,
                )
                attempt = {
                    "episode": episode,
                    "episode_seed": episode_seed,
                    "formal_reset_seed": formal_reset_seed,
                    "variation": variation,
                    "sealed_plan_fingerprint": plan_fingerprint,
                    "replay_attempt_ordinal": attempt_ordinal,
                    "original_result": originals[episode],
                    "replay_result": replay,
                    "expected_outcome": expected_outcome,
                    **assessment,
                }
                if not attempt["confirmed_for_publication"]:
                    video_path.unlink()
                    reason = {
                        "discarded_outcome_mismatch": (
                            "replay outcome did not match the source row"
                        ),
                        "discarded_source_condition_mismatch": (
                            "source condition was not reproduced"
                        ),
                        "discarded_invalid_actions_mismatch": (
                            "invalid-action count differed from the source row"
                        ),
                    }[attempt["disposition"]]
                    print(
                        f"{args.task} episode {episode} attempt "
                        f"{attempt_ordinal}/{maximum_attempts}: {reason}; "
                        "discarded from the trajectory-video set",
                        flush=True,
                    )
                    return attempt

                confirmation = _validate_replay(
                    originals[episode],
                    replay,
                    source["model_identity"],
                    worker.model_identity,
                    source,
                    expected_success=expected_success,
                )
                v4_store_metadata = (
                    {
                        "store_mode": replay.get("store_mode"),
                        "store_required_entities": replay.get(
                            "store_required_entities"
                        ),
                        "store_entity_interventions": replay.get(
                            "store_entity_interventions"
                        ),
                    }
                    if _is_v4_store_source(source, args.task)
                    else {}
                )
                v4_release_metadata = _v4_replay_manifest_metadata(
                    source,
                    args.task,
                )
                sidecar = {
                    "schema": schema,
                    "task": args.task,
                    "source_task": source.get("task"),
                    "scenario": source["scenario"],
                    "episode": episode,
                    "episode_seed": episode_seed,
                    "formal_reset_seed": formal_reset_seed,
                    "variation": variation,
                    "sealed_plan_fingerprint": plan_fingerprint,
                    "replay_attempt_ordinal": attempt_ordinal,
                    "replay_attempt_disposition": attempt["disposition"],
                    "confirmed_for_publication": True,
                    "replay_attempt_protocol": attempt_protocol,
                    "fixed_eval_set": source["fixed_eval_set"],
                    "source_result": str(resolved_source),
                    "source_result_sha256": source_sha,
                    "evaluation_protocol_id": replay_protocol_id,
                    "source_protocol": (
                        source.get("coordination_protocol")
                        or source.get("scenario_protocol")
                        or source.get("protocol")
                    ),
                    "model_identity": worker.model_identity,
                    "original_result": originals[episode],
                    "replay_result": replay,
                    **confirmation,
                    **v4_release_metadata,
                    **v4_store_metadata,
                    "video": {
                        "file": video_path.name,
                        "sha256": _sha256(video_path),
                        "requested_cameras": list(args.cameras),
                        "used_cameras": list(recorder.used_cameras),
                        "layout": "horizontal_left_to_right",
                        "source_resolution": list(args.resolution),
                        "encoded_resolution": [
                            recorder.frame_shape[1],
                            recorder.frame_shape[0],
                        ],
                        "fps": args.fps,
                        "frames": recorder.frames,
                        "codec": "H.264/yuv420p",
                    },
                    "policy_inputs_unchanged": True,
                    "capture_granularity": "returned_high_level_observations",
                    "camera_configuration": camera_configuration,
                    "warmup_prefix": {
                        "episodes": list(warmup_episodes),
                        "same_environment_and_worker": True,
                        "video_recorded": False,
                        "ledger": warmup_ledger,
                    },
                }
                sidecar_path = staging / f"{stem}.json"
                atomic_json(sidecar_path, sidecar)
                manifest_rows.append(
                    {
                        "episode": episode,
                        "episode_seed": episode_seed,
                        "formal_reset_seed": formal_reset_seed,
                        "variation": variation,
                        "sealed_plan_fingerprint": plan_fingerprint,
                        "replay_attempt_ordinal": attempt_ordinal,
                        "replay_attempt_disposition": attempt["disposition"],
                        "confirmed_for_publication": True,
                        "video": video_path.name,
                        "video_sha256": sidecar["video"]["sha256"],
                        "metadata": sidecar_path.name,
                        "metadata_sha256": _sha256(sidecar_path),
                        "expected_outcome": expected_outcome,
                        "replay_confirmed_outcome": True,
                        **v4_release_metadata,
                        **v4_store_metadata,
                    }
                )
                print(
                    f"{args.task} episode {episode} attempt "
                    f"{attempt_ordinal}/{maximum_attempts}: confirmed "
                    f"{expected_outcome}, wrote {recorder.frames} frames",
                    flush=True,
                )
                return attempt

            _accepted, episode_attempts = _run_bounded_replay_attempts(
                maximum_attempts,
                run_attempt,
            )
            attempts.extend(episode_attempts)
            if len(manifest_rows) == minimum_confirmed:
                break
        if len(manifest_rows) < minimum_confirmed:
            raise RuntimeError(
                f"only {len(manifest_rows)} of {minimum_confirmed} required "
                f"{expected_outcome} replays were confirmed; nothing will be published"
            )
        manifest = {
            "schema": schema,
            "task": args.task,
            "source_task": source.get("task"),
            "scenario": source["scenario"],
            "output_key": output_key,
            "source_result": str(resolved_source),
            "source_result_sha256": source_sha,
            "fixed_eval_set": source["fixed_eval_set"],
            "model_identity": worker.model_identity,
            "evaluation_protocol_id": replay_protocol_id,
            "source_protocol": (
                source.get("coordination_protocol")
                or source.get("scenario_protocol")
                or source.get("protocol")
            ),
            "policy_controller_semantics": "identical_to_source_evaluator",
            "policy_inputs_unchanged": True,
            "expected_outcome": expected_outcome,
            "required_confirmed_trajectories": minimum_confirmed,
            "candidate_episode_order": list(episodes),
            "replay_attempt_protocol": attempt_protocol,
            "attempts": attempts,
            "episodes": manifest_rows,
            "warmup_prefix": {
                "episodes": list(warmup_episodes),
                "same_environment_and_worker": True,
                "video_recorded": False,
                "ledger": warmup_ledger,
            },
            "camera_configuration": camera_configuration,
            "publication": {
                "target_existed_before_run": target_existed_before_run,
                "overwrite_enabled": overwrite,
                "complete_staging_before_target_swap": True,
                "atomic_directory_swap": True,
                "formal_result_overwrite_forbidden": True,
            },
            **_v4_replay_manifest_metadata(source, args.task),
        }
        atomic_json(staging / "manifest.json", manifest)
        _publish_replay_directory(staging, target, overwrite=overwrite)
        print(f"published {target}", flush=True)
        return target
    finally:
        if worker is not None:
            worker.close()
        if launched:
            environment.shutdown()
        if staging.exists():
            shutil.rmtree(staging)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(SUPPORTED_TASKS))
    parser.add_argument(
        "--episode",
        action="append",
        type=int,
        required=True,
        help="Exact source-evaluation episode index; repeat for multiple videos.",
    )
    parser.add_argument(
        "--warmup-episode",
        action="append",
        type=int,
        default=None,
        help=(
            "Execute this sealed source episode without video before recorded "
            "episodes, preserving one Environment/worker generation history."
        ),
    )
    parser.add_argument(
        "--expected-outcome",
        choices=("failure", "success"),
        default="failure",
        help=(
            "Required source and replay outcome. Defaults to failure for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--minimum-confirmed",
        type=int,
        default=None,
        help=(
            "Publish after this many candidates reproduce the requested outcome. "
            "By default, every selected episode must match."
        ),
    )
    parser.add_argument(
        "--max-replay-attempts-per-episode",
        type=int,
        default=1,
        help=(
            "Bounded fresh replays of each fixed source episode. Only completed "
            "outcome, source-condition, or invalid-action mismatches are retried; "
            "all exceptions fail closed."
        ),
    )
    parser.add_argument("--source-result", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-key",
        default=None,
        help=(
            "Destination directory name below --output-root. Dynamic and "
            "coordination replays should use one key per paper-result cell."
        ),
    )
    overwrite = parser.add_mutually_exclusive_group()
    overwrite.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Atomically replace an existing replay-only target after staging.",
    )
    overwrite.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if the exact replay target already exists.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument("--policy-python", type=Path, default=None)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 360),
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=("front", "overhead"),
        default=DEFAULT_CAMERAS,
    )
    parser.add_argument(
        "--camera-perspective-degrees",
        type=float,
        default=None,
        help=(
            "Optional legacy override for every requested RGB camera. By "
            "default the front FOV remains unchanged."
        ),
    )
    parser.add_argument(
        "--overhead-perspective-degrees",
        type=float,
        default=DEFAULT_REPLAY_OVERHEAD_PERSPECTIVE_DEGREES,
        help=(
            "Replay-only overhead-camera FOV. The 70-degree default slightly "
            "zooms the right-hand overhead view without changing the front FOV."
        ),
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fps < 1 or any(value < 1 for value in args.resolution):
        raise ValueError("fps and resolution must be positive")
    if args.policy_timeout <= 0.0:
        raise ValueError("policy timeout must be positive")
    if args.camera_perspective_degrees is not None and not (
        20.0 <= args.camera_perspective_degrees <= 120.0
    ):
        raise ValueError("--camera-perspective-degrees must be between 20 and 120")
    if args.overhead_perspective_degrees is not None and not (
        20.0 <= args.overhead_perspective_degrees <= 120.0
    ):
        raise ValueError("--overhead-perspective-degrees must be between 20 and 120")
    _validate_max_replay_attempts(args.max_replay_attempts_per_episode)
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg executable is unavailable: {args.ffmpeg}")
    return 0 if record(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
