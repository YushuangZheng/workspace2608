"""Replay a retained ICLR 2027 episode and record an outcome-verified video.

This is deliberately separate from the formal rollout launcher.  It reuses the
frozen manifest row, method configuration, policy, and controlled executor, but
enables RGB observations only for post-evaluation visualization.  The video is
published only when the replay preserves the requested source outcome and all
immutable identities.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.runners import shared_episode
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.report.failure_videos import (
    ObservationRecorder,
    RecordingTaskEnvironment,
    _observation_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_environment(task: Any, resolution: tuple[int, int]):
    import rlbench.environment as environment_module

    configuration = _observation_config(("front", "overhead"), resolution)
    if task.spec.bimanual:
        from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
            _controller_config,
            _make_action_mode,
        )

        environment = environment_module.Environment(
            action_mode=_make_action_mode(
                shared_episode.CONTROLLED_EXECUTOR_PROFILE,
                _controller_config(shared_episode.CONTROLLED_EXECUTOR_PROFILE),
            ),
            obs_config=configuration,
            headless=True,
            robot_setup="dual_panda",
        )
        return environment, (lambda: None)

    from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
        _controller_config,
        _make_action_mode,
        _prepare_low_dim_headless_scene,
    )

    environment = environment_module.Environment(
        action_mode=_make_action_mode(
            shared_episode.CONTROLLED_EXECUTOR_PROFILE,
            _controller_config(shared_episode.CONTROLLED_EXECUTOR_PROFILE),
        ),
        obs_config=configuration,
        headless=True,
    )
    restore, _metadata = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=True,
        camera_observations_requested=True,
    )
    return environment, restore


def _canonical_identity_value(value: Any) -> Any:
    """Normalize the one audited feature-profile field rename.

    Retained M5/M6 rows used ``auxiliary_verification_recovery`` before that
    aggregate audit flag was split into two explicit, behavior-equivalent
    booleans.  No model, threshold, or runtime choice changed.  All other
    identity fields remain exact.
    """

    if isinstance(value, list):
        return [_canonical_identity_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result = {
        str(key): _canonical_identity_value(item) for key, item in value.items()
    }
    if "auxiliary_verification_recovery" in result:
        enabled = result.pop("auxiliary_verification_recovery")
        result.setdefault("active_relation_verification", enabled)
        result.setdefault("state_aware_recovery_reentry", enabled)
    return result


def _identity_subset(result: Mapping[str, Any]) -> dict[str, Any]:
    identity = result.get("method_config_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("episode has no method_config_identity")
    return {
        "path": identity.get("path"),
        "sha256": identity.get("sha256"),
        "policy_model": _canonical_identity_value(identity.get("policy_model")),
        "fault_config_sha256": identity.get("fault_config_sha256"),
        "monitor_calibration": identity.get("monitor_calibration"),
    }


def _episode_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable manifest identity reproduced by a replay."""

    return {
        key: result.get(key)
        for key in (
            "episode_id",
            "split",
            "task",
            "task_level",
            "variation",
            "seed",
            "condition",
            "fault_family",
            "fault_severity",
            "trigger_stage",
        )
    }


def _fault_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    protocol = result.get("fault_protocol")
    if not isinstance(protocol, Mapping):
        return {
            "family": None,
            "triggered": False,
            "physical_effect_observed": False,
            "component_families": [],
        }
    components = protocol.get("components")
    component_families = []
    if isinstance(components, list):
        component_families = [
            str(component.get("family"))
            for component in components
            if isinstance(component, Mapping)
        ]
    return {
        "family": protocol.get("family"),
        "triggered": bool(protocol.get("triggered")),
        "physical_effect_observed": bool(
            protocol.get("physical_effect_observed")
        ),
        "component_families": component_families,
    }


def _validate_source_identity(
    source: Mapping[str, Any], row: Mapping[str, Any]
) -> None:
    row_identity = _episode_identity(row)
    source_identity = _episode_identity(source)
    if source_identity != row_identity:
        differing = {
            key: {"manifest": row_identity[key], "source": source_identity[key]}
            for key in row_identity
            if row_identity[key] != source_identity[key]
        }
        raise ValueError(f"source result differs from manifest identity: {differing}")


def _validate_perturbation_reproduction(
    source: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    condition = source.get("condition")
    source_signature = _fault_signature(source)
    replay_signature = _fault_signature(replay)
    source_audit = source.get("audit")
    replay_audit = replay.get("audit")
    source_triggered = bool(
        isinstance(source_audit, Mapping)
        and source_audit.get("physically_triggered")
    )
    replay_triggered = bool(
        isinstance(replay_audit, Mapping)
        and replay_audit.get("physically_triggered")
    )
    if condition == "nominal":
        if source_triggered or replay_triggered:
            raise RuntimeError("nominal replay unexpectedly contains a physical fault")
    elif condition == "perturbed":
        if not source_triggered:
            raise ValueError("selected source perturbation was not physically triggered")
        if not replay_triggered:
            raise RuntimeError("replay did not reproduce the physical perturbation")
        if not (
            replay_signature["triggered"]
            and replay_signature["physical_effect_observed"]
        ):
            raise RuntimeError("replay fault protocol did not produce a physical effect")
        if replay_signature["family"] != source_signature["family"]:
            raise RuntimeError("replay fault family differs from the retained source")
        if (
            replay_signature["component_families"]
            != source_signature["component_families"]
        ):
            raise RuntimeError(
                "replay composed-fault components differ from the retained source"
            )
    else:
        raise ValueError(f"unsupported replay condition: {condition}")
    return {
        "condition": condition,
        "source_physically_triggered": source_triggered,
        "replay_physically_triggered": replay_triggered,
        "source_fault_signature": source_signature,
        "replay_fault_signature": replay_signature,
    }


def record_replay(
    *,
    manifest: Path,
    episode_id: str,
    method: str,
    source_result: Path,
    expected_success: bool,
    output_video: Path,
    output_records: Path,
    policy_python: Path,
    fps: int,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    source = shared_episode._load_json(source_result)
    if source.get("episode_id") != episode_id:
        raise ValueError("source result and requested episode differ")
    if bool(source.get("success")) is not expected_success:
        raise ValueError("source result does not have the requested outcome")
    row = shared_episode._load_manifest_row(manifest, episode_id)
    _validate_source_identity(source, row)
    output_video = Path(output_video).resolve()
    sidecar = output_video.with_suffix(".json")
    if output_video.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite replay: {output_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    staging = output_video.with_name(f".{output_video.stem}.staging.mp4")
    staging.unlink(missing_ok=True)
    recorder = ObservationRecorder(
        staging,
        cameras=("front", "overhead"),
        fps=fps,
        ffmpeg=Path("/usr/bin/ffmpeg"),
    )

    original_environment = shared_episode._environment
    original_initialize = shared_episode.initialize_fresh_task_generation
    original_fault_environment = shared_episode.build_fault_environment

    def initialize_with_capture(*args, **kwargs):
        values = original_initialize(*args, **kwargs)
        recorder.capture(values[2])
        return values

    def fault_environment_with_capture(*args, **kwargs):
        wrapped = original_fault_environment(*args, **kwargs)
        return RecordingTaskEnvironment(wrapped, recorder)

    shared_episode._environment = lambda task: _video_environment(task, resolution)
    shared_episode.initialize_fresh_task_generation = initialize_with_capture
    shared_episode.build_fault_environment = fault_environment_with_capture
    try:
        replay = shared_episode.run_episode(
            row,
            output_records,
            policy_python=policy_python,
            method=method,
        )
    finally:
        shared_episode._environment = original_environment
        shared_episode.initialize_fresh_task_generation = original_initialize
        shared_episode.build_fault_environment = original_fault_environment

    try:
        if replay.get("reason") == "infrastructure_error":
            raise RuntimeError(f"replay infrastructure error: {replay.get('error')}")
        if bool(replay.get("success")) is not expected_success:
            raise RuntimeError("replay outcome differs from the retained source")
        if replay.get("episode_id") != source.get("episode_id"):
            raise RuntimeError("replay episode identity differs from source")
        if replay.get("seed") != source.get("seed") or replay.get("variation") != source.get("variation"):
            raise RuntimeError("replay seed or variation differs from source")
        if replay.get("method_id") != source.get("method_id"):
            raise RuntimeError("replay method differs from source")
        if _episode_identity(replay) != _episode_identity(source):
            raise RuntimeError("replay manifest identity differs from source")
        if _identity_subset(replay) != _identity_subset(source):
            raise RuntimeError("replay method/policy/config identity differs from source")
        perturbation_audit = _validate_perturbation_reproduction(source, replay)
        recorder.close()
        os.replace(staging, output_video)
    except Exception:
        recorder.abort()
        staging.unlink(missing_ok=True)
        raise

    payload = {
        "schema": "essay2608.iclr2027.outcome-verified-replay-video.v1",
        "episode_id": episode_id,
        "task": source["task"],
        "method_id": source["method_id"],
        "expected_outcome": "success" if expected_success else "fail",
        "source_result": str(source_result),
        "source_result_sha256": _sha256(source_result),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "video": str(output_video),
        "video_sha256": _sha256(output_video),
        "frames": int(recorder.frames),
        "cameras": ["front", "overhead"],
        "resolution_per_camera": list(resolution),
        "fps": int(fps),
        "source_reason": source.get("reason"),
        "replay_reason": replay.get("reason"),
        "source_cycles": source.get("cycles"),
        "replay_cycles": replay.get("cycles"),
        "outcome_reproduced": True,
        "identity_reproduced": True,
        "perturbation_reproduced": True,
        "perturbation_audit": perturbation_audit,
        "replay_is_visualization_not_formal_evaluation": True,
    }
    atomic_json(sidecar, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--expected-outcome", choices=("success", "fail"), required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, default=shared_episode.DEFAULT_POLICY_PYTHON)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--resolution", type=int, nargs=2, default=(320, 240))
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    payload = record_replay(
        manifest=args.manifest,
        episode_id=args.episode_id,
        method=args.method,
        source_result=args.source_result,
        expected_success=args.expected_outcome == "success",
        output_video=args.output_video,
        output_records=args.output_records,
        policy_python=args.policy_python,
        fps=args.fps,
        resolution=tuple(args.resolution),
    )
    print(payload["video"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
