"""Local Table III coordination diagnostic for the dynamic HandOver task.

The paper does not publish the arm-trajectory perturbation magnitude, axis,
start time, or demonstration manifest.  This runner therefore uses a frozen
V3 diagnostic protocol and marks every result as non-comparable with the paper:

* train from five live ``BimanualHandoverItemDynamic`` demonstrations;
* authenticate the preregistered arm-specific trigger against the checkpoint;
* ramp a +3 cm world-z target offset over ten normal policy ticks, then keep it.

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

from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    DEFAULT_V4_EVALUATION_VIDEO_ROOT,
    GRIPPER_PROTOCOL,
    PolicyProcess,
    _commit_formal_result_with_optional_v4_videos,
    _finalize_v4_video_cell,
    _make_action_mode,
    _prepare_v4_video_cell,
    _run_episode_with_optional_v4_video,
    _v4_video_capture_enabled,
    _v4_video_cell,
    evaluation_protocol_id,
)
from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
    fixed_coordination_sources,
    validate_formal_artifact_paths,
)
from integrations.rlbench.rlbench_dynamac.report.evaluation_videos import (
    LightweightCaptureConfig,
)
from integrations.rlbench.rlbench_dynamac.core.gripper_timing import (
    global_gripper_timing_metadata,
)
from integrations.rlbench.rlbench_dynamac.core.records import (
    atomic_json,
    reserve_output,
)
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    FORMAL_POLICY_CLOCK_SEMANTICS_ID,
    commit_joint_hold_after_primary_failure,
    bind_staged_source_plan,
    final_settling_metadata,
    global_ik_controller_metadata,
    initialize_fresh_task_generation,
    run_final_settling,
    step_current_joint_hold_noop,
)
from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
    bimanual_checkpoint_trigger_audit,
    build_v3_trigger_anchor_evidence,
    load_v3_intervention_protocol,
    resolve_authenticated_v3_trigger,
)
from integrations.rlbench.rlbench_dynamac.protocols.v4_dynamic_protocol import (
    V4_COORDINATION_PROTOCOL_ID,
    V4_COORDINATION_SMOOTH_POLICY_TICKS,
    V4_COORDINATION_TRANSLATION_METERS,
    V4_COORDINATION_TRIGGER_STEP,
    load_v4_coordination_intervention_protocol,
    v4_coordination_trigger_authentication,
)

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT

POLICY_TASK = "bimanual_handover_item"
TASK_MODULE = "rlbench.bimanual_tasks.bimanual_handover_item_dynamic"
TASK_CLASS = "BimanualHandoverItemDynamic"
DEFAULT_DATA_ROOT = INTEGRATION_ROOT / "data" / "training" / "coordination"
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v3" / "table_iii"
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v3" / "table_iii_coordination"
DEFAULT_POLICY_PYTHON = Path(os.environ.get("DYNAMAC_POLICY_PYTHON", "python3.10"))
DEFAULT_POLICY_CONFIG = INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v3.json"
LOCAL_PROTOCOL_CONFIG = (
    INTEGRATION_ROOT / "configs" / "table_iii_coordination_local.json"
)
PERTURBATION_METERS = (0.0, 0.0, 0.03)
EXPECTED_VARIATION_COUNT = 5
V4_VIDEO_PAPER_TARGET = 0.97
EVALUATION_PROTOCOL_ID = evaluation_protocol_id(DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS)


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
        raise RuntimeError(
            "authenticated coordination trigger lies outside policy clock"
        )
    return protocol, authentication, trigger


def _authenticated_v4_coordination_trigger(args, worker):
    """Authenticate the fixed policy-clocked V4 handover perturbation."""

    protocol = load_v4_coordination_intervention_protocol()
    if getattr(args, "max_primary_action_attempts", None) != 1:
        raise RuntimeError("formal policy ticks require exactly one primary request")
    requested = getattr(args, "trigger_step", None)
    if args.arm == "none":
        if requested is not None:
            raise RuntimeError(
                "the unperturbed V4 coordination baseline has no trigger"
            )
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

    from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
        _validate_published_model as validate,
    )

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
    return (
        Path(data_root)
        / POLICY_TASK
        / "all_variations"
        / "episodes"
        / f"episode{episode}"
    )


def collect(args):
    """Collect the five local dynamic-HandOver expert demonstrations."""

    from rlbench.environment import Environment

    manifest_path = Path(args.data_root) / "collection_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing collection: {manifest_path}"
        )
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
                raise FileExistsError(
                    f"refusing to overwrite existing episode: {target}"
                )
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
                "path": (
                    (target / "low_dim_obs.pkl").relative_to(args.data_root).as_posix()
                ),
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

    from integrations.rlbench.rlbench_dynamac.eval.evaluation_split import (
        validate_training_entry_paths,
    )

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

    from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
        load_low_dim_obs_pickles,
        make_bimanual_demonstrations,
    )
    from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
        demonstration_paths,
        load_policy_config,
    )
    from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
        load_rlbench_segmentation_config,
    )
    from integrations.rlbench.rlbench_dynamac.core.task_specs import get_task_spec

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


def _offset_action(action, arm, fraction, translation):
    result = np.asarray(action, dtype=np.float64).copy()
    if arm == "none" or float(fraction) == 0.0:
        return result
    if arm not in {"left", "right"}:
        raise ValueError("coordination perturbation arm must be left or right")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("coordination offset fraction must lie in (0, 1]")
    offset = np.asarray(translation, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("coordination translation must be finite 3D")
    start = 9 if arm == "left" else 0
    result[start : start + 3] += float(fraction) * offset
    return result


def _perturb_action(action, arm, enabled):
    """Retain the frozen V3 full-offset behavior."""

    return _offset_action(
        action,
        arm,
        1.0 if enabled else 0.0,
        PERTURBATION_METERS,
    )


def _v4_coordination_fraction(committed_policy_steps, trigger):
    """Return the ramped/persistent offset fraction for one policy tick."""

    if committed_policy_steps < trigger:
        return None
    elapsed = committed_policy_steps - trigger + 1
    return min(
        elapsed / float(V4_COORDINATION_SMOOTH_POLICY_TICKS),
        1.0,
    )


def _new_v4_coordination_intervention(observation, *, arm, trigger):
    """Create the per-episode audit for the policy-clocked smooth window."""

    arm_observation = getattr(observation, arm, None)
    measured_pose = np.asarray(
        getattr(arm_observation, "gripper_pose", ()),
        dtype=np.float64,
    )
    if measured_pose.shape != (7,) or not np.all(np.isfinite(measured_pose)):
        raise RuntimeError("V4 coordination measured EE pose is unavailable")
    return {
        "schema": "dynamac-coordination-policy-clocked-smooth-intervention-v4",
        "protocol_id": V4_COORDINATION_PROTOCOL_ID,
        "perturbed_arm": arm,
        "trigger_step": trigger,
        "measured_start_pose": measured_pose.tolist(),
        "measured_pose_after_smooth_window": None,
        "translation_world_m": list(V4_COORDINATION_TRANSLATION_METERS),
        "orientation_from_policy": True,
        "smooth_policy_ticks_planned": V4_COORDINATION_SMOOTH_POLICY_TICKS,
        "smooth_policy_ticks_elapsed": 0,
        "fractions_requested": [],
        "fractions_applied": [],
        "joint_hold_fractions": [],
        "policy_requests": 0,
        "policy_clock_advances": 0,
        "offset_actions_applied": 0,
        "joint_hold_commits": 0,
        "smooth_window_invalid_actions": 0,
        "persistent_policy_ticks_committed": 0,
        "persistent_offset": True,
        "observation_refresh": "normal_committed_step_return_before_next_request",
        "completed": False,
        "terminal_during_smooth_window": False,
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
    if max_primary_action_attempts != 1:
        raise ValueError("formal coordination requests each policy tick exactly once")
    joint_hold_commits = 0
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
        row.setdefault("primary_failure_joint_hold_commits", joint_hold_commits)
        row.setdefault("policy_clock_semantics_id", FORMAL_POLICY_CLOCK_SEMANTICS_ID)
        if release == "v4":
            row.setdefault("coordination_intervention", v4_intervention)
        row.setdefault("fresh_task_generation", fresh_task_generation)
        row.setdefault("staged_source_binding", staged_source_binding)
        return row

    control_attempts = 0
    while committed_policy_steps < horizon:
        v4_fraction = None
        v4_smooth_window_tick = False
        if release == "v4" and arm != "none" and committed_policy_steps >= trigger:
            if v4_intervention is None:
                if committed_policy_steps != trigger:
                    raise RuntimeError(
                        "V4 coordination smooth window did not start at its trigger"
                    )
                v4_intervention = _new_v4_coordination_intervention(
                    observation,
                    arm=arm,
                    trigger=trigger,
                )
            v4_fraction = _v4_coordination_fraction(
                committed_policy_steps,
                trigger,
            )
            v4_smooth_window_tick = (
                committed_policy_steps < trigger + V4_COORDINATION_SMOOTH_POLICY_TICKS
            )
        response = worker.request("act", observation)
        if v4_smooth_window_tick:
            v4_intervention["policy_requests"] += 1
            v4_intervention["fractions_requested"].append(float(v4_fraction))
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
            return finish(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "success": bool(settling["success"]),
                    "steps": control_attempts,
                    "control_attempts": control_attempts,
                    "reason": reason,
                    "invalid_actions": invalid_actions,
                    "final_settling": settling,
                }
            )
        transaction_id = response.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy worker did not return an action transaction id")
        control_attempts += 1
        enabled = (
            release == "v3" and arm != "none" and committed_policy_steps >= trigger
        )
        v4_enabled = v4_fraction is not None
        primary_action_applied = False
        try:
            if v4_enabled:
                command = _offset_action(
                    action,
                    arm,
                    v4_fraction,
                    V4_COORDINATION_TRANSLATION_METERS,
                )
            else:
                command = _perturb_action(action, arm, enabled)
            perturbed_attempts += int(enabled or v4_enabled)
            observation, reward, terminate = task_environment.step(command)
            primary_action_applied = True
        except InvalidActionError:
            invalid_actions += 1
            if v4_smooth_window_tick:
                v4_intervention["smooth_window_invalid_actions"] += 1
            try:
                observation, reward, terminate, policy_complete = (
                    commit_joint_hold_after_primary_failure(
                        task_environment,
                        worker,
                        transaction_id=transaction_id,
                    )
                )
            except InvalidActionError:
                return finish(
                    {
                        "episode": episode,
                        "seed": episode_seed,
                        "success": False,
                        "steps": control_attempts,
                        "control_attempts": control_attempts,
                        "reason": "joint_hold_failed",
                        "invalid_actions": invalid_actions,
                    }
                )
            joint_hold_commits += 1
            committed_policy_steps += 1
        except Exception:
            worker.request("abort", transaction_id=transaction_id)
            raise
        if primary_action_applied:
            commit = worker.request("commit", transaction_id=transaction_id)
            policy_complete = bool(commit.get("complete"))
            committed_policy_steps += 1
            perturbed_steps += int(enabled)
        if v4_smooth_window_tick:
            v4_intervention["policy_clock_advances"] += 1
            v4_intervention["smooth_policy_ticks_elapsed"] += 1
            perturbed_steps += 1
            if primary_action_applied:
                v4_intervention["offset_actions_applied"] += 1
                v4_intervention["fractions_applied"].append(float(v4_fraction))
            else:
                v4_intervention["joint_hold_commits"] += 1
                v4_intervention["joint_hold_fractions"].append(float(v4_fraction))
            if (
                v4_intervention["smooth_policy_ticks_elapsed"]
                == V4_COORDINATION_SMOOTH_POLICY_TICKS
            ):
                final_arm_observation = getattr(observation, arm, None)
                final_pose = np.asarray(
                    getattr(final_arm_observation, "gripper_pose", ()),
                    dtype=np.float64,
                )
                if final_pose.shape == (7,) and np.all(np.isfinite(final_pose)):
                    v4_intervention["measured_pose_after_smooth_window"] = (
                        final_pose.tolist()
                    )
                v4_intervention["completed"] = True
        elif v4_enabled:
            v4_intervention["persistent_policy_ticks_committed"] += 1
        if v4_smooth_window_tick and (reward > 0.0 or terminate):
            v4_intervention["terminal_during_smooth_window"] = True
        settling = None
        if reward > 0.0:
            reason = "success"
        elif terminate:
            reason = "terminate"
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
        return finish(
            {
                "episode": episode,
                "seed": episode_seed,
                "success": bool(reward > 0.0 or (settling or {}).get("success")),
                "steps": control_attempts,
                "control_attempts": control_attempts,
                "reason": reason,
                "invalid_actions": invalid_actions,
                **({"final_settling": settling} if settling is not None else {}),
            }
        )
    return finish(
        {
            "episode": episode,
            "seed": episode_seed,
            "success": False,
            "steps": control_attempts,
            "control_attempts": control_attempts,
            "reason": "horizon",
            "invalid_actions": invalid_actions,
        }
    )


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
    video_capture_config = LightweightCaptureConfig() if video_capture_enabled else None
    video_cell_dir = None
    video_cell_key = None
    episode_videos = []
    if video_capture_enabled:
        scenario = (
            f"coordination_hand_{args.arm}" if args.arm != "none" else "local_baseline"
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
                    episode_kwargs["final_settling_steps"] = args.final_settling_steps
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
        "scenario": (
            f"coordination_hand_{args.arm}" if args.arm != "none" else "local_baseline"
        ),
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
        "evaluation_protocol_id": evaluation_protocol_id(max_primary_action_attempts),
        "controller": {
            **global_ik_controller_metadata(),
            "worker_clock_handshake_id": worker.policy_clock_semantics_id,
            "worker_gripper_timing_handshake": worker.gripper_timing,
            "coordination_trigger_clock": "successfully_committed_policy_ticks",
            "coordination_intervention_execution": {
                "uses_same_global_action_mode_ik_chain": True,
                "policy_transaction": True,
                "intervention_and_policy_share_action": True,
                "policy_requests": V4_COORDINATION_SMOOTH_POLICY_TICKS,
                "policy_clock_advances": V4_COORDINATION_SMOOTH_POLICY_TICKS,
                "smooth_policy_ticks": V4_COORDINATION_SMOOTH_POLICY_TICKS,
                "max_primary_action_attempts_per_policy_tick": 1,
                "invalid_action_behavior": (
                    "commit_joint_hold_and_advance_same_policy_tick"
                ),
                "raw_joint_hold_commit_on_failure": True,
                "observation_refresh": (
                    "normal_committed_step_return_before_next_request"
                ),
            },
            "formal_episode_initialization": DETERMINISTIC_SOURCE_RESET_PROTOCOL_ID,
            "final_settling": final_settling_metadata(
                getattr(
                    args,
                    "final_settling_steps",
                    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
                )
            ),
            "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
            "gripper_timing": global_gripper_timing_metadata(),
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
        "invalid_actions": sum(item["invalid_actions"] for item in episodes),
        "gripper_protocol": GRIPPER_PROTOCOL.metadata(),
        "gripper_timing": global_gripper_timing_metadata(),
        "paper_comparable": False,
        "paper_target": (0.97 if args.arm in {"left", "right"} else None),
        "claim_boundary": (
            "Diagnostic only: the paper does not publish the perturbation axis, "
            "magnitude, start time, duration, demonstration IDs, or evaluation seeds."
        ),
        "coordination_protocol": {
            "source": (
                "V4_POLICY_CLOCKED_SMOOTH_PERSISTENT_OFFSET_PROTOCOL"
                if v4_coordination
                else "LOCAL_EXPLICIT_DIAGNOSTIC_20260815"
            ),
            **({"protocol_id": V4_COORDINATION_PROTOCOL_ID} if v4_coordination else {}),
            "paper_comparable": False,
            "protocol_valid": True,
            "perturbed_arm": args.arm,
            "translation_world_m": list(
                V4_COORDINATION_TRANSLATION_METERS
                if v4_coordination
                else PERTURBATION_METERS
            ),
            "application": (
                "ramp world +z target offset across 10 normal policy ticks, then keep the full offset"
                if v4_coordination
                else "persistent offset on every predicted EE target from trigger"
            ),
            **(
                {
                    "smooth_policy_ticks": (
                        V4_COORDINATION_SMOOTH_POLICY_TICKS
                        if args.arm != "none"
                        else None
                    ),
                    "orientation": (
                        "unmodified policy target" if args.arm != "none" else None
                    ),
                    "other_arm_and_grippers": (
                        "unmodified policy targets" if args.arm != "none" else None
                    ),
                    "smooth_fractions": "1/10 through 10/10",
                    "persistent_policy_target_offset": True,
                    "policy_clock_advances_during_intervention": True,
                }
                if v4_coordination
                else {}
            ),
            "legacy_one_third_trigger_disabled": True,
            "trigger_reference_domain": ("successfully_committed_policy_ticks"),
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
                isinstance(item.get("fresh_task_generation"), dict) for item in episodes
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
                V4_VIDEO_PAPER_TARGET if args.arm in {"left", "right"} else None
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
                    V4_VIDEO_PAPER_TARGET if args.arm in {"left", "right"} else None
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
