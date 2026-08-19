"""Local Table III coordination diagnostic for the dynamic HandOver task.

The paper does not publish the arm-trajectory perturbation magnitude, axis,
start time, or demonstration manifest.  This runner therefore uses a frozen
V3 diagnostic protocol and marks every result as non-comparable with the paper:

* train from five live ``BimanualHandoverItemDynamic`` demonstrations;
* authenticate the preregistered arm-specific trigger against the checkpoint;
* add a persistent +3 cm world-z offset to one arm's predicted EE targets.

Simulator commands are Python 3.8 compatible.  Policy fitting and serving are
delegated to the existing Python 3.10 implementation.  Data, models, and
results live below separate ``table_iii_coordination`` directories.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import random
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .direct_evaluate import (
    DEFAULT_V4_EVALUATION_VIDEO_ROOT,
    GRIPPER_PROTOCOL,
    PolicyProcess,
    _commit_formal_result_with_optional_v4_videos,
    _finalize_v4_video_cell,
    _make_action_mode,
    _noop_action,
    _prepare_v4_video_cell,
    _run_episode_with_optional_v4_video,
    _v4_video_capture_enabled,
    _v4_video_cell,
    evaluation_protocol_id,
)
from .eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
    fixed_coordination_sources,
    validate_formal_artifact_paths,
)
from .evaluation_videos import LightweightCaptureConfig
from .records import atomic_json, reserve_output
from .runtime import (
    DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    PrimaryActionRetryBudget,
    bind_staged_source_plan,
    final_settling_metadata,
    initialize_fresh_task_generation,
    run_final_settling,
)
from .v3_protocol import (
    bimanual_checkpoint_trigger_audit,
    build_v3_trigger_anchor_evidence,
    load_v3_intervention_protocol,
    resolve_authenticated_v3_trigger,
)
from .v4_dynamic_protocol import (
    V4_COORDINATION_CARTESIAN_SUBSTEPS,
    V4_COORDINATION_PROTOCOL_ID,
    V4_COORDINATION_TRANSLATION_METERS,
    V4_COORDINATION_TRIGGER_STEP,
    load_v4_coordination_intervention_protocol,
    v4_coordination_trigger_authentication,
)

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
POLICY_TASK = "bimanual_handover_item"
TASK_MODULE = "rlbench.bimanual_tasks.bimanual_handover_item_dynamic"
TASK_CLASS = "BimanualHandoverItemDynamic"
DEFAULT_DATA_ROOT = INTEGRATION_ROOT / "data" / "table_iii_coordination" / "g5_seed0"
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3" / "table_iii"
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v3" / "table_iii_coordination"
DEFAULT_POLICY_PYTHON = Path(os.environ.get("DYNAMAC_POLICY_PYTHON", "python3.10"))
DEFAULT_POLICY_CONFIG = (
    INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v3.json"
)
LOCAL_PROTOCOL_CONFIG = INTEGRATION_ROOT / "configs" / "table_iii_coordination_local.json"
PERTURBATION_METERS = (0.0, 0.0, 0.03)
EXPECTED_VARIATION_COUNT = 5
V4_VIDEO_PAPER_TARGET = 0.97
EVALUATION_PROTOCOL_ID = evaluation_protocol_id(
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
)


def _authenticated_v3_coordination_trigger(args, worker):
    """Authenticate the frozen coordination tick(s) before simulator launch."""

    protocol = load_v3_intervention_protocol()
    final_steps = getattr(
        args,
        "final_settling_steps",
        DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    )
    if final_steps != protocol["final_settling_physics_steps"]:
        raise RuntimeError(
            "V3 final-settling budget differs from the frozen intervention protocol"
        )
    requested = getattr(args, "trigger_step", None)
    if args.arm == "none":
        if requested is not None:
            raise RuntimeError("the unperturbed coordination baseline has no trigger")
        authentication = {
            side: resolve_authenticated_v3_trigger(
                worker.model_identity,
                scenario=f"coordination_hand_{side}",
            )
            for side in ("left", "right")
        }
        return protocol, authentication, None

    scenario = f"coordination_hand_{args.arm}"
    authentication = resolve_authenticated_v3_trigger(
        worker.model_identity,
        scenario=scenario,
    )
    trigger = authentication["trigger_step"]
    if authentication["profile"].get("perturbed_arm") != args.arm:
        raise RuntimeError("authenticated V3 coordination arm is inconsistent")
    if requested is not None and requested != trigger:
        raise RuntimeError(
            "command-line coordination trigger differs from V3 preregistration"
        )
    if trigger >= worker.policy_steps:
        raise RuntimeError("authenticated coordination trigger lies outside policy clock")
    return protocol, authentication, trigger


def _authenticated_v4_coordination_trigger(args, worker):
    """Authenticate the fixed one-shot V4 handover perturbation."""

    protocol = load_v4_coordination_intervention_protocol()
    if getattr(args, "max_primary_action_attempts", None) != 3:
        raise RuntimeError("V4 coordination preserves the preregistered 3-attempt budget")
    requested = getattr(args, "trigger_step", None)
    if args.arm == "none":
        if requested is not None:
            raise RuntimeError("the unperturbed V4 coordination baseline has no trigger")
        return protocol, None, None
    authentication = v4_coordination_trigger_authentication(
        arm=args.arm,
        policy_steps=worker.policy_steps,
    )
    trigger = authentication["trigger_step"]
    if requested is not None and requested != trigger:
        raise RuntimeError("V4 coordination trigger must be committed policy tick 235")
    return protocol, authentication, trigger


def _validate_published_model(*args, **kwargs):
    """Import policy-only validation lazily so Python 3.8 can run evaluation."""

    from .direct_policy import _validate_published_model as validate

    return validate(*args, **kwargs)


def _task_class():
    return getattr(importlib.import_module(TASK_MODULE), TASK_CLASS)


def _observation_config(*, capture_front_rgb=False):
    from rlbench.observation_config import ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.gripper_open = True
    config.gripper_pose = True
    config.task_low_dim_state = True
    if capture_front_rgb:
        from rlbench.observation_config import CameraConfig

        config.camera_configs = {
            "front": CameraConfig(
                rgb=True,
                depth=False,
                point_cloud=False,
                mask=False,
            )
        }
    return config


def _collection_action_mode():
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import BimanualJointPosition
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

    return BimanualMoveArmThenGripper(
        BimanualJointPosition(),
        BimanualDiscrete(),
    )


def _episode_dir(data_root, episode):
    return Path(data_root) / POLICY_TASK / "all_variations" / "episodes" / f"episode{episode}"


def collect(args):
    """Collect the five local dynamic-HandOver expert demonstrations."""

    from rlbench.environment import Environment

    manifest_path = Path(args.data_root) / "collection_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing collection: {manifest_path}")
    environment = Environment(
        action_mode=_collection_action_mode(),
        obs_config=_observation_config(),
        headless=args.headless,
        robot_setup="dual_panda",
    )
    records = []
    launched = False
    try:
        environment.launch()
        launched = True
        task_environment = environment.get_task(_task_class())
        variations = task_environment.variation_count()
        for episode in range(args.demonstrations):
            target = _episode_dir(args.data_root, episode)
            if target.exists():
                raise FileExistsError(f"refusing to overwrite existing episode: {target}")
            episode_seed = args.seed + episode
            variation = episode % variations
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            task_environment.set_variation(variation)
            (demo,) = task_environment.get_demos(amount=1, live_demos=True)
            target.mkdir(parents=True, exist_ok=False)
            for observation in demo:
                perception = getattr(observation, "perception_data", None)
                if isinstance(perception, dict):
                    perception.clear()
            with (target / "low_dim_obs.pkl").open("wb") as stream:
                pickle.dump(demo, stream, protocol=pickle.HIGHEST_PROTOCOL)
            with (target / "variation_number.pkl").open("wb") as stream:
                pickle.dump(variation, stream, protocol=pickle.HIGHEST_PROTOCOL)
            record = {
                "episode": episode,
                "seed": episode_seed,
                "variation": variation,
                "observations": len(demo),
                "path": str((target / "low_dim_obs.pkl").resolve()),
            }
            records.append(record)
            print(
                f"dynamic HandOver demo {episode + 1}/{args.demonstrations}: "
                f"{len(demo)} observations",
                flush=True,
            )
    finally:
        if launched:
            environment.shutdown()

    manifest = {
        "schema": "dynamac-table-iii-coordination-demos-v1",
        "task": "bimanual_handover_item_dynamic",
        "task_module": TASK_MODULE,
        "task_class": TASK_CLASS,
        "policy_task_alias": POLICY_TASK,
        "demonstrations": args.demonstrations,
        "base_seed": args.seed,
        "paper_cohort": False,
        "paper_comparable": False,
        "claim_boundary": (
            "Live demonstrations from the public RLBench task; the paper's "
            "demonstration/seed manifest is unpublished."
        ),
        "episodes": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return manifest


def train(args):
    """Fit, authenticate, and atomically publish the coordination policy."""

    from .evaluation_split import validate_training_entry_paths

    validate_training_entry_paths(
        getattr(args, "data_root", DEFAULT_DATA_ROOT),
        args.models_dir,
        getattr(args, "config", DEFAULT_POLICY_CONFIG),
    )

    output = Path(args.models_dir) / POLICY_TASK
    with reserve_output(output):
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{POLICY_TASK}.staging-",
                dir=str(Path(args.models_dir)),
            )
        )
        try:
            summary = _train_into(args, staging)
            _validate_published_model(POLICY_TASK, staging, summary)
            os.rename(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    print(
        f"trained dynamic HandOver: left={summary['left']['durations']} "
        f"right={summary['right']['durations']}",
        flush=True,
    )
    return summary


def _train_into(args, output):
    """Fit all coordination artifacts inside one unpublished staging directory."""

    from essay2608.policy import BimanualDynaMAC

    from .demo_adapter import load_low_dim_obs_pickles, make_bimanual_demonstrations
    from .direct_policy import demonstration_paths, load_policy_config
    from .tapas_segmentation import load_rlbench_segmentation_config
    from .task_specs import get_task_spec

    protocol = json.loads(LOCAL_PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    segment_protocol = protocol["segmentation"]
    base_segmentation = load_rlbench_segmentation_config().for_task(POLICY_TASK)
    segmentation = replace(
        base_segmentation,
        boundary_selection=segment_protocol["boundary_selection"],
        expected_boundary_count=segment_protocol["expected_boundary_count"],
        provenance={
            **{
                key: value
                for key, value in base_segmentation.provenance.items()
                if key != "task_profiles"
            },
            "task_profiles": {},
            "coordination_local_protocol": segment_protocol,
        },
    )
    paths = demonstration_paths(Path(args.data_root), POLICY_TASK, args.demonstrations)
    names = [path.parent.name for path in paths]
    episodes = load_low_dim_obs_pickles(paths)
    output = Path(output)
    if not output.is_dir() or any(output.iterdir()):
        raise RuntimeError(f"coordination staging directory is not empty: {output}")
    debug_plot = output / "segmentation.png"
    converted = make_bimanual_demonstrations(
        episodes,
        get_task_spec(POLICY_TASK),
        names=names,
        config=segmentation,
        debug_plot_path=debug_plot,
    )
    policy_config = load_policy_config(Path(args.config))
    policy = BimanualDynaMAC(config=policy_config)
    policy.fit(converted.left_demonstrations, converted.right_demonstrations)
    policy.left.save(output / "left.npz")
    policy.right.save(output / "right.npz")
    summary = _training_summary(
        policy=policy,
        converted=converted,
        names=names,
        policy_config=policy_config,
        debug_plot=debug_plot,
    )
    atomic_json(output / "training.json", summary)
    return summary


def _training_summary(*, policy, converted, names, policy_config, debug_plot):
    """Build the authenticated manifest shared by staging and tests."""

    summary = {
        "task": POLICY_TASK,
        "bimanual": True,
        "demonstrations": names,
        "config": asdict(policy_config),
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
        "segmentation": converted.segmentation.audit,
        "segmentation_debug_plot": debug_plot.name,
        "local_protocol_config": str(LOCAL_PROTOCOL_CONFIG.resolve()),
        "manifest_schema": "dynamac-direct-training-v3",
        "training_task": "bimanual_handover_item_dynamic",
        "policy_task_alias": POLICY_TASK,
        "paper_cohort": False,
        "paper_comparable": False,
        "claim_boundary": (
            "The dynamic task has the same five object-pose observation schema "
            "as HandOver, but uses locally collected randomized-handover demos."
        ),
    }
    checkpoint_audit = bimanual_checkpoint_trigger_audit(policy)
    summary["checkpoint_trigger_audit"] = checkpoint_audit
    summary["v3_trigger_anchor_evidence"] = build_v3_trigger_anchor_evidence(
        POLICY_TASK,
        checkpoint_audit,
        summary,
    )
    return summary


def _perturb_action(action, arm, enabled):
    result = np.asarray(action, dtype=np.float64).copy()
    if not enabled or arm == "none":
        return result
    start = 9 if arm == "left" else 0
    result[start : start + 3] += np.asarray(PERTURBATION_METERS)
    return result


def _v4_coordination_cartesian_command(
    observation,
    *,
    arm,
    measured_pose,
    fraction,
):
    """Hold everything except one measured EE translated along world +z."""

    if arm not in {"left", "right"}:
        raise ValueError("V4 coordination perturbation arm must be left or right")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("V4 coordination Cartesian fraction must lie in (0, 1]")
    source = np.asarray(measured_pose, dtype=np.float64)
    if source.shape != (7,) or not np.all(np.isfinite(source)):
        raise ValueError("V4 coordination measured EE pose must be finite 7D")
    command = _noop_action(observation)
    start = 9 if arm == "left" else 0
    command[start : start + 3] = source[:3] + (
        float(fraction) * np.asarray(V4_COORDINATION_TRANSLATION_METERS)
    )
    command[start + 3 : start + 7] = source[3:7]
    return command


def _apply_v4_coordination_intervention(
    task_environment,
    observation,
    *,
    arm,
    max_attempts,
    invalid_action_error,
):
    """Execute the one-shot measured-pose push without touching policy time."""

    if max_attempts != 3:
        raise ValueError("V4 coordination Cartesian retry budget must remain three")
    arm_observation = getattr(observation, arm, None)
    measured_pose = np.asarray(
        getattr(arm_observation, "gripper_pose", ()),
        dtype=np.float64,
    )
    if measured_pose.shape != (7,) or not np.all(np.isfinite(measured_pose)):
        raise RuntimeError("V4 coordination measured EE pose is unavailable")
    current_observation = observation
    reward = 0.0
    terminate = False
    invalid_actions = 0
    action_attempts = 0
    substeps_completed = 0
    attempts_by_substep = []
    for substep in range(1, V4_COORDINATION_CARTESIAN_SUBSTEPS + 1):
        fraction = substep / float(V4_COORDINATION_CARTESIAN_SUBSTEPS)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            action_attempts += 1
            command = _v4_coordination_cartesian_command(
                current_observation,
                arm=arm,
                measured_pose=measured_pose,
                fraction=fraction,
            )
            try:
                current_observation, reward, terminate = task_environment.step(
                    command
                )
            except invalid_action_error:
                invalid_actions += 1
                if attempts >= max_attempts:
                    attempts_by_substep.append(attempts)
                    return current_observation, 0.0, False, {
                        "schema": "dynamac-coordination-cartesian-intervention-v4",
                        "protocol_id": V4_COORDINATION_PROTOCOL_ID,
                        "perturbed_arm": arm,
                        "trigger_step": V4_COORDINATION_TRIGGER_STEP,
                        "measured_start_pose": measured_pose.tolist(),
                        "translation_world_m": list(
                            V4_COORDINATION_TRANSLATION_METERS
                        ),
                        "orientation_fixed": True,
                        "cartesian_substeps_planned": (
                            V4_COORDINATION_CARTESIAN_SUBSTEPS
                        ),
                        "cartesian_substeps_completed": substeps_completed,
                        "attempts_by_substep": attempts_by_substep,
                        "action_attempts": action_attempts,
                        "invalid_actions": invalid_actions,
                        "max_attempts_per_substep": max_attempts,
                        "policy_requests": 0,
                        "policy_clock_advances": 0,
                        "persistent_offset": False,
                        "observation_refreshed_after_substeps": False,
                        "completed": False,
                        "retry_exhausted": True,
                    }
                # InvalidAction does not commit a simulator target; refresh the
                # measured hold poses before retrying this same Cartesian point.
                current_observation = task_environment.get_observation()
                continue
            substeps_completed += 1
            attempts_by_substep.append(attempts)
            break
        if reward > 0.0 or terminate:
            break
    completed = substeps_completed == V4_COORDINATION_CARTESIAN_SUBSTEPS
    refreshed = False
    if completed:
        current_observation = task_environment.get_observation()
        refreshed = True
    final_arm_observation = getattr(current_observation, arm, None)
    final_pose = np.asarray(
        getattr(final_arm_observation, "gripper_pose", ()),
        dtype=np.float64,
    )
    return current_observation, reward, terminate, {
        "schema": "dynamac-coordination-cartesian-intervention-v4",
        "protocol_id": V4_COORDINATION_PROTOCOL_ID,
        "perturbed_arm": arm,
        "trigger_step": V4_COORDINATION_TRIGGER_STEP,
        "measured_start_pose": measured_pose.tolist(),
        "measured_final_pose": (
            final_pose.tolist() if final_pose.shape == (7,) else None
        ),
        "translation_world_m": list(V4_COORDINATION_TRANSLATION_METERS),
        "orientation_fixed": True,
        "cartesian_substeps_planned": V4_COORDINATION_CARTESIAN_SUBSTEPS,
        "cartesian_substeps_completed": substeps_completed,
        "attempts_by_substep": attempts_by_substep,
        "action_attempts": action_attempts,
        "invalid_actions": invalid_actions,
        "max_attempts_per_substep": max_attempts,
        "policy_requests": 0,
        "policy_clock_advances": 0,
        "persistent_offset": False,
        "observation_refreshed_after_substeps": refreshed,
        "completed": completed,
        "retry_exhausted": False,
        "terminal_during_intervention": bool(reward > 0.0 or terminate),
    }


def _run_episode(
    task_environment,
    worker,
    *,
    episode,
    variation,
    seed,
    horizon,
    arm,
    trigger,
    max_primary_action_attempts=DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    final_settling_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    descriptions=None,
    observation=None,
    fresh_task_generation=None,
    staged_source_binding=None,
    release="v3",
):
    from rlbench.backend.exceptions import InvalidActionError

    episode_seed = seed + episode
    if observation is None or not isinstance(fresh_task_generation, dict):
        raise RuntimeError("formal episode requires fresh task-generation input")
    if release not in {"v3", "v4"}:
        raise ValueError("coordination release must be v3 or v4")
    worker.request("reset", observation)
    invalid_actions = 0
    retry_budget = PrimaryActionRetryBudget(max_primary_action_attempts)
    perturbed_steps = 0
    perturbed_attempts = 0
    committed_policy_steps = 0
    v4_intervention = None

    def finish(row):
        row.setdefault("variation", variation)
        row.setdefault("committed_policy_steps", committed_policy_steps)
        row.setdefault(
            "final_settling",
            {
                **final_settling_metadata(final_settling_steps),
                "attempted": False,
                "available": True,
                "steps_executed": 0,
                "first_terminal_step": None,
                "stop_reason": "not_entered",
                "success": False,
                "terminate": False,
            },
        )
        row.setdefault("perturbed_steps", perturbed_steps)
        row.setdefault("perturbed_attempts", perturbed_attempts)
        if release == "v4":
            row.setdefault("coordination_intervention", v4_intervention)
        row.setdefault("fresh_task_generation", fresh_task_generation)
        row.setdefault("staged_source_binding", staged_source_binding)
        return row

    for control_step in range(horizon):
        if (
            release == "v4"
            and arm != "none"
            and v4_intervention is None
            and committed_policy_steps == trigger
        ):
            observation, reward, terminate, v4_intervention = (
                _apply_v4_coordination_intervention(
                    task_environment,
                    observation,
                    arm=arm,
                    max_attempts=max_primary_action_attempts,
                    invalid_action_error=InvalidActionError,
                )
            )
            invalid_actions += v4_intervention["invalid_actions"]
            perturbed_steps += v4_intervention["cartesian_substeps_completed"]
            perturbed_attempts += v4_intervention["action_attempts"]
            if v4_intervention["retry_exhausted"]:
                return finish({
                    "episode": episode,
                    "seed": episode_seed,
                    "success": False,
                    "steps": control_step,
                    "control_attempts": control_step,
                    "reason": "coordination_intervention_retry_exhausted",
                    "invalid_actions": invalid_actions,
                })
            if reward > 0.0 or terminate:
                return finish({
                    "episode": episode,
                    "seed": episode_seed,
                    "success": bool(reward > 0.0),
                    "steps": control_step,
                    "control_attempts": control_step,
                    "reason": (
                        "success_during_coordination_intervention"
                        if reward > 0.0
                        else "terminate_during_coordination_intervention"
                    ),
                    "invalid_actions": invalid_actions,
                })
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            settling = run_final_settling(
                task_environment,
                physics_steps=final_settling_steps,
            )
            if settling["success"]:
                reason = "success_after_final_settling"
            elif settling["terminate"]:
                reason = "terminate_during_final_settling"
            elif settling["available"]:
                reason = "policy_complete_after_final_settling"
            else:
                reason = "policy_complete"
            return finish({
                "episode": episode,
                "seed": episode_seed,
                "success": bool(settling["success"]),
                "steps": control_step,
                "control_attempts": control_step,
                "reason": reason,
                "invalid_actions": invalid_actions,
                "final_settling": settling,
            })
        transaction_id = response.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy worker did not return an action transaction id")
        enabled = (
            release == "v3"
            and arm != "none"
            and committed_policy_steps >= trigger
        )
        primary_action_succeeded = False
        retry_exhausted = False
        try:
            command = _perturb_action(action, arm, enabled)
            perturbed_attempts += int(enabled)
            observation, reward, terminate = task_environment.step(command)
            primary_action_succeeded = True
        except InvalidActionError:
            invalid_actions += 1
            retry_exhausted = retry_budget.record_failure()
            worker.request("abort", transaction_id=transaction_id)
            try:
                observation, reward, terminate = task_environment.step(_noop_action(observation))
            except InvalidActionError:
                return finish({
                    "episode": episode,
                    "seed": episode_seed,
                    "success": False,
                    "steps": control_step + 1,
                    "control_attempts": control_step + 1,
                    "reason": "noop_failed",
                    "invalid_actions": invalid_actions,
                })
        except Exception:
            worker.request("abort", transaction_id=transaction_id)
            raise
        if primary_action_succeeded:
            commit = worker.request("commit", transaction_id=transaction_id)
            retry_budget.record_success()
            policy_complete = bool(commit.get("complete"))
            committed_policy_steps += 1
            perturbed_steps += int(enabled)
        else:
            policy_complete = False
        settling = None
        if reward > 0.0:
            reason = "success"
        elif terminate:
            reason = "terminate"
        elif retry_exhausted:
            reason = "primary_action_retry_exhausted"
        elif policy_complete:
            settling = run_final_settling(
                task_environment,
                physics_steps=final_settling_steps,
            )
            if settling["success"]:
                reason = "success_after_final_settling"
            elif settling["terminate"]:
                reason = "terminate_during_final_settling"
            elif settling["available"]:
                reason = "policy_complete_after_final_settling"
            else:
                reason = "policy_complete"
        else:
            continue
        return finish({
            "episode": episode,
            "seed": episode_seed,
            "success": bool(reward > 0.0 or (settling or {}).get("success")),
            "steps": control_step + 1,
            "control_attempts": control_step + 1,
            "reason": reason,
            "invalid_actions": invalid_actions,
            **(
                {"primary_action_attempts": retry_budget.attempts}
                if reason == "primary_action_retry_exhausted"
                else {}
            ),
            **({"final_settling": settling} if settling is not None else {}),
        })
    return finish({
        "episode": episode,
        "seed": episode_seed,
        "success": False,
        "steps": horizon,
        "control_attempts": horizon,
        "reason": "horizon",
        "invalid_actions": invalid_actions,
    })


def evaluate(args):
    """Evaluate one explicit local arm-trajectory perturbation condition."""

    output = Path(args.output)
    validate_formal_artifact_paths(output=output, models_dir=args.models_dir)
    with reserve_output(output):
        return _evaluate_reserved(args, output)


def _evaluate_reserved(args, output):
    """Run one coordination evaluation while its result path is reserved."""

    video_capture_enabled = _v4_video_capture_enabled(args)
    from rlbench.environment import Environment

    if args.eval_set_id is None:
        raise RuntimeError("formal coordination evaluation requires --eval-set-id")
    if args.seed != GLOBAL_EVAL_SEED_START or args.episodes != FIXED_EVAL_EPISODES:
        raise RuntimeError(
            "formal coordination seed/episodes differ from the fixed eval set"
        )
    eval_set, selected_batch = fixed_coordination_sources(args.eval_set_id)
    source_plans = selected_batch["plans"]
    max_primary_action_attempts = getattr(
        args,
        "max_primary_action_attempts",
        DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    )
    video_capture_config = (
        LightweightCaptureConfig() if video_capture_enabled else None
    )
    video_cell_dir = None
    video_cell_key = None
    episode_videos = []
    if video_capture_enabled:
        scenario = (
            f"coordination_hand_{args.arm}"
            if args.arm != "none"
            else "local_baseline"
        )
        video_cell_dir, video_cell_key = _v4_video_cell(
            getattr(
                args,
                "v4_evaluation_video_root",
                DEFAULT_V4_EVALUATION_VIDEO_ROOT,
            ),
            "bimanual_handover_item_dynamic",
            scenario,
        )
        video_cell_dir = _prepare_v4_video_cell(video_cell_dir)
    action_mode = _make_action_mode()
    environment = Environment(
        action_mode=action_mode,
        obs_config=(
            _observation_config(capture_front_rgb=True)
            if video_capture_enabled
            else _observation_config()
        ),
        headless=args.headless,
        robot_setup="dual_panda",
    )
    worker = PolicyProcess(
        args.policy_python,
        POLICY_TASK,
        args.models_dir,
        timeout=args.policy_timeout,
    )
    policy_steps = worker.policy_steps
    launched = False
    episodes = []
    try:
        if getattr(args, "release", "v3") == "v4":
            intervention_registry, trigger_authentication, trigger = (
                _authenticated_v4_coordination_trigger(args, worker)
            )
        else:
            intervention_registry, trigger_authentication, trigger = (
                _authenticated_v3_coordination_trigger(args, worker)
            )
        environment.launch()
        launched = True
        task_class = _task_class()
        variation_count = task_class(
            environment._pyrep,
            environment._robot,
        ).variation_count()
        if variation_count != EXPECTED_VARIATION_COUNT:
            raise RuntimeError(
                "V3 coordination task variation count differs from the frozen "
                f"protocol: expected {EXPECTED_VARIATION_COUNT}, got "
                f"{variation_count}"
            )
        variation_schedule = [
            episode % variation_count for episode in range(args.episodes)
        ]
        for episode in range(args.episodes):
            (
                task_environment,
                descriptions,
                observation,
                fresh_task_generation,
            ) = initialize_fresh_task_generation(
                environment,
                task_class,
                episode_seed=source_plans[episode].validation["source_seed"],
                variation=variation_schedule[episode],
                verify_instance=False,
            )
            def run_formal_episode(formal_task_environment):
                source_binding = bind_staged_source_plan(
                    formal_task_environment,
                    source_plans[episode],
                    descriptions=descriptions,
                    fresh_task_generation=fresh_task_generation,
                )
                bound_observation = formal_task_environment.get_observation()
                episode_kwargs = dict(
                    episode=episode,
                    variation=variation_schedule[episode],
                    seed=args.seed,
                    horizon=args.horizon,
                    arm=args.arm,
                    trigger=trigger,
                    max_primary_action_attempts=max_primary_action_attempts,
                    observation=bound_observation,
                    fresh_task_generation=fresh_task_generation,
                    staged_source_binding=source_binding,
                )
                if hasattr(args, "final_settling_steps"):
                    episode_kwargs["final_settling_steps"] = (
                        args.final_settling_steps
                    )
                if getattr(args, "release", "v3") == "v4":
                    episode_kwargs["release"] = "v4"
                return _run_episode(
                    formal_task_environment,
                    worker,
                    **episode_kwargs,
                )

            record, episode_video = _run_episode_with_optional_v4_video(
                task_environment,
                observation,
                enabled=video_capture_enabled,
                cell_dir=video_cell_dir,
                cell_key=video_cell_key,
                episode=episode,
                episode_seed=args.seed + episode,
                run_episode=run_formal_episode,
                capture_config=video_capture_config,
            )
            episodes.append(record)
            if episode_video is not None:
                episode_videos.append(episode_video)
            successes = sum(int(item["success"]) for item in episodes)
            print(
                f"{args.arm} episode {episode + 1}/{args.episodes}: "
                f"{record['reason']} ({successes}/{len(episodes)})",
                flush=True,
            )
    finally:
        worker.close()
        if launched:
            environment.shutdown()

    successes = sum(int(item["success"]) for item in episodes)
    release = getattr(args, "release", "v3")
    v4_coordination = release == "v4"
    payload = {
        "schema": (
            "dynamac-table-iii-coordination-local-v4"
            if v4_coordination
            else "dynamac-table-iii-coordination-local-v3"
        ),
        **({"release": "v4"} if v4_coordination else {}),
        "task": "bimanual_handover_item_dynamic",
        "policy_task_alias": POLICY_TASK,
        "result_family": "bimanual_coordination",
        "scenario": (f"coordination_hand_{args.arm}" if args.arm != "none" else "local_baseline"),
        "episodes": len(episodes),
        "episodes_requested": args.episodes,
        "episodes_completed": len(episodes),
        "variation_count": variation_count,
        "variation_schedule": variation_schedule,
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "seed": args.seed,
        "horizon": args.horizon,
        "learned_policy_steps": policy_steps,
        "evaluation_protocol_id": evaluation_protocol_id(
            max_primary_action_attempts
        ),
        "controller": {
            "command": "absolute_world_end_effector_pose",
            "primary_ik": "jacobian",
            "fallback_ik": "sampling",
            "sampling_trials": 100,
            "sampling_max_configs": 5,
            "sampling_max_time_ms": 10,
            "sampling_ignore_collisions": not v4_coordination,
            **(
                {
                    "sampling_candidate_selection": "nearest_current_joint_l2",
                    "ik_candidate_validation": (
                        "shape_finite_noncyclic_joint_limits"
                    ),
                }
                if v4_coordination
                else {}
            ),
            "joint_target_max_steps": 200,
            "failed_action": "abort_policy_target_then_current_pose_current_gripper_noop",
            "failed_action_next_tick": "retry_same_policy_tick_from_fresh_observation",
            "primary_action_retry": PrimaryActionRetryBudget(
                max_primary_action_attempts
            ).metadata(),
            "policy_clock_rollback": True,
            "policy_clock_semantics_id": worker.policy_clock_semantics_id,
            "coordination_trigger_clock": "successfully_committed_policy_ticks",
            "formal_episode_initialization": DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
            "final_settling": final_settling_metadata(
                getattr(
                    args,
                    "final_settling_steps",
                    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
                )
            ),
            "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
        "invalid_actions": sum(item["invalid_actions"] for item in episodes),
        "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
        "paper_comparable": False,
        "paper_target": (0.97 if args.arm in {"left", "right"} else None),
        "claim_boundary": (
            "Diagnostic only: the paper does not publish the perturbation axis, "
            "magnitude, start time, duration, demonstration IDs, or evaluation seeds."
        ),
        "coordination_protocol": {
            "source": (
                "V4_MEASURED_EE_ONE_SHOT_CARTESIAN_PROTOCOL"
                if v4_coordination
                else "LOCAL_EXPLICIT_DIAGNOSTIC_20260815"
            ),
            **(
                {"protocol_id": V4_COORDINATION_PROTOCOL_ID}
                if v4_coordination
                else {}
            ),
            "paper_comparable": False,
            "protocol_valid": True,
            "perturbed_arm": args.arm,
            "translation_world_m": list(
                V4_COORDINATION_TRANSLATION_METERS
                if v4_coordination
                else PERTURBATION_METERS
            ),
            "application": (
                "measured current EE to world +z in 10 Cartesian substeps once, then unmodified policy"
                if v4_coordination
                else "persistent offset on every predicted EE target from trigger"
            ),
            **(
                {
                    "cartesian_substeps": (
                        V4_COORDINATION_CARTESIAN_SUBSTEPS
                        if args.arm != "none"
                        else None
                    ),
                    "orientation": (
                        "fixed at measured pre-intervention orientation"
                        if args.arm != "none"
                        else None
                    ),
                    "other_arm_and_grippers": (
                        "hold current measured state each substep"
                        if args.arm != "none"
                        else None
                    ),
                    "persistent_policy_target_offset": False,
                    "policy_clock_advances_during_intervention": False,
                }
                if v4_coordination
                else {}
            ),
            "legacy_one_third_trigger_disabled": True,
            "trigger_reference_domain": (
                "successfully_committed_policy_ticks"
            ),
            "trigger_policy_step": trigger,
            "trigger_authentication": trigger_authentication,
            "intervention_registry_schema": intervention_registry["schema"],
            "intervention_registry_fingerprint": intervention_registry["fingerprint"],
            "randomized_handover_location": (
                "public BimanualHandoverItemDynamic; waypoint world-z offset "
                "sampled uniformly in [-0.02, 0.02] m per episode"
            ),
        },
        "results": episodes,
        "fixed_eval_set": {
            "evaluation_set_id": eval_set["payload"]["evaluation_set_id"],
            "manifest_sha256": eval_set["manifest_sha256"],
            "spec_sha256": eval_set["payload"]["spec"]["sha256"],
            "selected_batch_sha256": selected_batch["sha256"],
            "selected_batch_fingerprint": selected_batch["batch_fingerprint"],
            "formal_access": "canonical_id_read_only_no_generation",
        },
        "fresh_task_generation": {
            "required_per_formal_episode": True,
            "all_episodes_recorded": all(
                isinstance(item.get("fresh_task_generation"), dict)
                for item in episodes
            ),
            "evidence": [item["fresh_task_generation"] for item in episodes],
        },
        "final_settling_protocol": final_settling_metadata(
            getattr(
                args,
                "final_settling_steps",
                DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
            )
        ),
    }
    if video_capture_enabled:
        video_capture_metadata = {
            "release": "v4",
            "cell_key": video_cell_key,
            "cell_dir": str(video_cell_dir),
            "episodes_recorded": len(episode_videos),
            "capture_config": dict(video_capture_config.audit()),
            "paper_success_rate": (
                V4_VIDEO_PAPER_TARGET
                if args.arm in {"left", "right"}
                else None
            ),
        }

        def finalize_videos():
            return _finalize_v4_video_cell(
                video_cell_dir,
                episode_videos,
                cell_key=video_cell_key,
                successes=successes,
                episodes=args.episodes,
                paper_success_rate=(
                    V4_VIDEO_PAPER_TARGET
                    if args.arm in {"left", "right"}
                    else None
                ),
                cell_metadata={
                    "evaluator": "table_iii_coordination",
                    "task": "bimanual_handover_item_dynamic",
                    "arm": args.arm,
                    "formal_result": str(output),
                },
            )
    else:
        video_capture_metadata = None
        finalize_videos = None
    _commit_formal_result_with_optional_v4_videos(
        output,
        payload,
        enabled=video_capture_enabled,
        capture_metadata=video_capture_metadata,
        finalize_videos=finalize_videos,
    )
    print(f"wrote {output}")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collector = subparsers.add_parser("collect")
    collector.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    collector.add_argument("--demonstrations", type=int, default=5)
    collector.add_argument("--seed", type=int, default=0)
    display = collector.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    collector.set_defaults(headless=True)

    trainer = subparsers.add_parser("train")
    trainer.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    trainer.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    trainer.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_POLICY_CONFIG,
    )
    trainer.add_argument("--demonstrations", type=int, default=5)

    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--arm", choices=("left", "right", "none"), required=True)
    evaluator.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    evaluator.add_argument("--episodes", type=int, default=FIXED_EVAL_EPISODES)
    evaluator.add_argument("--seed", type=int, default=GLOBAL_EVAL_SEED_START)
    evaluator.add_argument("--eval-set-id", default=None)
    evaluator.add_argument("--horizon", type=int, default=1000)
    evaluator.add_argument(
        "--trigger-step",
        type=int,
        default=None,
        help="Explicit committed-policy trigger tick from the V3 profile.",
    )
    evaluator.add_argument(
        "--final-settling-steps",
        type=int,
        default=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    )
    evaluator.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    evaluator.add_argument("--policy-timeout", type=float, default=120.0)
    evaluator.add_argument(
        "--max-primary-action-attempts",
        type=int,
        default=DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
        help="Maximum primary InvalidAction attempts for one policy clock tick.",
    )
    evaluator.add_argument(
        "--release",
        choices=("v3", "v4"),
        default="v3",
        help="Evaluation release gate; V4 requires formal episode video capture.",
    )
    evaluator.add_argument(
        "--record-v4-evaluation-videos",
        action="store_true",
        help=(
            "Stream front RGB from the actual formal V4 episodes and retain "
            "the fixed outcome-stratified sample; disabled by default for V3."
        ),
    )
    evaluator.add_argument(
        "--v4-evaluation-video-root",
        type=Path,
        default=DEFAULT_V4_EVALUATION_VIDEO_ROOT,
        help="Root for V4 formal evaluation video cells.",
    )
    evaluator.add_argument("--output", type=Path, required=True)
    display = evaluator.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    evaluator.set_defaults(headless=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        if args.demonstrations < 1 or args.seed < 0:
            raise ValueError("demonstrations must be positive and seed non-negative")
        collect(args)
    elif args.command == "train":
        if args.demonstrations < 1:
            raise ValueError("demonstrations must be positive")
        train(args)
    else:
        if (
            args.episodes < 1
            or args.seed < 0
            or args.horizon < 1
            or args.max_primary_action_attempts < 1
            or args.final_settling_steps < 0
            or (args.trigger_step is not None and args.trigger_step < 0)
        ):
            raise ValueError("episodes/horizon must be positive and seed non-negative")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
