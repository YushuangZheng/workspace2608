"""Visual audit utilities for persisted single-arm evaluation traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from essay2608.policy.base import PHASE_NAMES


@dataclass(frozen=True)
class AuditTrial:
    """One preselected representative trace and its audit role."""

    label: str
    stem: str


DEFAULT_AUDIT_TRIALS = (
    AuditTrial("skill_ordinary_success", "skill_dynamac__arm_offset__seed_6300"),
    AuditTrial("skill_ordinary_failure", "skill_dynamac__smooth_object__seed_6301"),
    AuditTrial("legacy_ordinary_success", "full_dynamac__arm_offset__seed_6300"),
    AuditTrial("legacy_ordinary_failure", "full_dynamac__arm_offset__seed_6303"),
    AuditTrial("relation_ordinary_success", "relation_dynamac__arm_offset__seed_6300"),
    AuditTrial("relation_ordinary_failure", "relation_dynamac__arm_offset__seed_6303"),
    AuditTrial("legacy_drop", "full_dynamac__drop_after_grasp__seed_6300"),
    AuditTrial("relation_drop", "relation_dynamac__drop_after_grasp__seed_6300"),
    AuditTrial("legacy_miss", "full_dynamac__close_without_grasp__seed_6300"),
    AuditTrial("relation_miss", "relation_dynamac__close_without_grasp__seed_6300"),
)


REQUIRED_ARRAYS = {
    "ee_position",
    "object_position",
    "target_position",
    "action",
    "phase",
    "connected",
    "perturbation_event",
    "raw_action_position",
    "policy_action_position",
    "relation_state",
    "relation_confidence",
    "gripper_opening_m",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trial(trials_dir: Path, stem: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and validate one JSON/NPZ trace pair."""

    json_path = trials_dir / f"{stem}.json"
    npz_path = trials_dir / f"{stem}.npz"
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"Missing trace pair for {stem} in {trials_dir}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"{stem} is missing arrays: {sorted(missing)}")
        arrays = {name: archive[name].copy() for name in archive.files}
    steps = len(arrays["phase"])
    per_step_arrays = {name: value for name, value in arrays.items() if not name.startswith("terminal_")}
    lengths = {name: len(value) for name, value in per_step_arrays.items()}
    bad_lengths = {name: length for name, length in lengths.items() if length != steps}
    if bad_lengths:
        raise ValueError(f"{stem} has inconsistent trace lengths: {bad_lengths}")
    if int(metadata["metrics"]["steps"]) != steps:
        raise ValueError(f"{stem} JSON steps do not match NPZ steps")
    return metadata, arrays


def reconstruct_active_frames(
    steps: int,
    switch_diagnostics: list[dict[str, Any]],
) -> list[tuple[str, ...] | None]:
    """Reconstruct active frames when at least one persisted switch anchors the sequence.

    The frozen v1 NPZ omitted the per-step frame list. Switch diagnostics still
    preserve exact before/after lists for traces containing a switch. A trace
    without any switch has no persisted anchor, so it remains explicitly unknown.
    """

    frames: list[tuple[str, ...] | None] = [None] * steps
    if not switch_diagnostics:
        return frames
    diagnostics = sorted(switch_diagnostics, key=lambda item: int(item["step"]))
    first_step = int(diagnostics[0]["step"])
    if not 0 < first_step < steps:
        raise ValueError(f"Invalid first frame-switch step: {first_step}")
    current = tuple(str(value) for value in diagnostics[0]["before"])
    for index in range(first_step):
        frames[index] = current
    cursor = first_step
    for diagnostic in diagnostics:
        step = int(diagnostic["step"])
        if step < cursor or not 0 < step < steps:
            raise ValueError(f"Invalid frame-switch step: {step}")
        for index in range(cursor, step):
            frames[index] = current
        current = tuple(str(value) for value in diagnostic["after"])
        cursor = step
    for index in range(cursor, steps):
        frames[index] = current
    return frames


def audit_failure_taxonomy(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Cross-check persisted failure taxonomy against its geometric trace evidence."""

    metrics = metadata["metrics"]
    has_terminal_snapshot = "terminal_object_position" in arrays and "terminal_target_position" in arrays
    final_object = arrays["terminal_object_position"] if has_terminal_snapshot else arrays["object_position"][-1]
    final_target = arrays["terminal_target_position"] if has_terminal_snapshot else arrays["target_position"][-1]
    final_xy = float(np.linalg.norm(final_object[:2] - final_target[:2]))
    recorded_xy = float(metrics["final_xy_error_m"])
    reason = str(metrics["failure_reason"])
    success = bool(metrics["success"])
    terminal_alignment = bool(np.isclose(final_xy, recorded_xy, atol=1.0e-6))
    checks = {"success_reason_agrees": (reason == "success") == success}
    criteria = metrics["success_criteria"]
    if reason == "placement_xy_above_threshold":
        checks["placement_failure_semantics"] = bool(
            not success
            and metrics["policy_complete"]
            and not metrics["environment_done"]
            and metrics["gripper_released"]
            and metrics["object_on_support"]
            and metrics["stable_after_release"]
            and final_xy >= float(criteria["xy_threshold_m"])
        )
    elif reason == "environment_terminated":
        checks["environment_failure_semantics"] = bool(not success and metrics["environment_done"])
    elif reason == "success":
        checks["success_semantics"] = bool(
            metrics["policy_complete"]
            and not metrics["environment_done"]
            and metrics["gripper_released"]
            and metrics["object_on_support"]
            and metrics["stable_after_release"]
            and final_xy < float(criteria["xy_threshold_m"])
        )
    else:
        checks["known_failure_reason"] = reason in {
            "policy_incomplete",
            "not_released",
            "not_on_support",
            "unstable_after_release",
        }
    return {
        "consistent": all(checks.values()),
        "checks": checks,
        "terminal_snapshot_persisted": has_terminal_snapshot,
        "terminal_trace_alignment": terminal_alignment,
        "terminal_trace_gap_m": abs(final_xy - recorded_xy),
        "success": success,
        "failure_reason": reason,
        "trace_final_xy_error_m": final_xy,
        "recorded_final_xy_error_m": recorded_xy,
    }


def _range(values: np.ndarray, minimum_span: float = 0.1) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = max(high - low, minimum_span)
    padding = 0.12 * span
    return low - padding, high + padding


def _world_point(
    xy: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[int, int]:
    left, top, width, height = 55, 115, 700, 545
    x = left + int((float(xy[0]) - x_range[0]) / (x_range[1] - x_range[0]) * width)
    y = top + height - int((float(xy[1]) - y_range[0]) / (y_range[1] - y_range[0]) * height)
    return x, y


def _draw_polyline(
    canvas: np.ndarray,
    values: np.ndarray,
    end: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if end <= 0:
        return
    points = np.asarray([_world_point(point, x_range, y_range) for point in values[: end + 1]], dtype=np.int32)
    cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)


def _put(
    canvas: np.ndarray,
    text: str,
    point: tuple[int, int],
    scale: float = 0.48,
    color: tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_series(
    canvas: np.ndarray,
    values: np.ndarray,
    index: int,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    fixed_range: tuple[float, float] | None = None,
) -> None:
    left, top, width, height = box
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (90, 90, 90), 1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return
    low, high = fixed_range or _range(finite, minimum_span=0.01)
    high = max(high, low + 1.0e-9)
    points = []
    for step in range(index + 1):
        value = float(values[step])
        if not np.isfinite(value):
            continue
        x = left + int(step / max(len(values) - 1, 1) * width)
        y = top + height - int(np.clip((value - low) / (high - low), 0.0, 1.0) * height)
        points.append((x, y))
    if len(points) > 1:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    cursor_x = left + int(index / max(len(values) - 1, 1) * width)
    cv2.line(canvas, (cursor_x, top), (cursor_x, top + height), (180, 180, 180), 1)


def render_frame(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    frames: list[tuple[str, ...] | None],
    index: int,
) -> np.ndarray:
    """Render one top-down trace reconstruction frame."""

    canvas = np.full((720, 1280, 3), (24, 26, 30), dtype=np.uint8)
    method = str(metadata["method"])
    condition = str(metadata["condition"])
    metrics = metadata["metrics"]
    all_xy = np.concatenate(
        (
            arrays["ee_position"][:, :2],
            arrays["object_position"][:, :2],
            arrays["target_position"][:, :2],
            arrays["raw_action_position"][:, :2],
            arrays["policy_action_position"][:, :2],
        ),
        axis=0,
    )
    x_range = _range(all_xy[:, 0], minimum_span=0.25)
    y_range = _range(all_xy[:, 1], minimum_span=0.25)
    cv2.rectangle(canvas, (55, 115), (755, 660), (100, 100, 100), 1)
    _put(canvas, "TRACE RECONSTRUCTION (not camera footage)", (55, 38), 0.72, (110, 210, 255), 2)
    _put(canvas, f"{method} | {condition} | seed {metadata['seed']}", (55, 72), 0.62, (240, 240, 240), 2)
    _put(canvas, "top-down XY (m)", (55, 103), 0.48, (185, 185, 185))

    _draw_polyline(canvas, arrays["ee_position"][:, :2], index, x_range, y_range, (255, 210, 80), 2)
    _draw_polyline(canvas, arrays["object_position"][:, :2], index, x_range, y_range, (50, 150, 255), 2)
    _draw_polyline(canvas, arrays["policy_action_position"][:, :2], index, x_range, y_range, (220, 80, 220), 1)
    ee = _world_point(arrays["ee_position"][index, :2], x_range, y_range)
    obj = _world_point(arrays["object_position"][index, :2], x_range, y_range)
    target = _world_point(arrays["target_position"][index, :2], x_range, y_range)
    policy_target = _world_point(arrays["policy_action_position"][index, :2], x_range, y_range)
    raw_target = _world_point(arrays["raw_action_position"][index, :2], x_range, y_range)
    cv2.circle(canvas, target, 16, (80, 220, 100), 2, cv2.LINE_AA)
    cv2.circle(canvas, obj, 9, (50, 150, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, ee, 8, (255, 210, 80), -1, cv2.LINE_AA)
    cv2.drawMarker(canvas, policy_target, (220, 80, 220), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, raw_target, (150, 150, 150), cv2.MARKER_TILTED_CROSS, 14, 1, cv2.LINE_AA)
    _put(canvas, "EE", (ee[0] + 10, ee[1] - 8), 0.4, (255, 210, 80))
    _put(canvas, "object", (obj[0] + 10, obj[1] + 18), 0.4, (50, 150, 255))
    _put(canvas, "target", (target[0] + 18, target[1]), 0.4, (80, 220, 100))

    phase_index = int(arrays["phase"][index])
    phase = PHASE_NAMES[phase_index] if 0 <= phase_index < len(PHASE_NAMES) else f"unknown_{phase_index}"
    relation = str(arrays["relation_state"][index])
    confidence = float(arrays["relation_confidence"][index])
    active = frames[index]
    active_text = ",".join(active) if active is not None else "UNAVAILABLE in frozen v1"
    xy_error = float(
        np.linalg.norm(arrays["object_position"][index, :2] - arrays["target_position"][index, :2])
    )
    event = bool(arrays["perturbation_event"][index])
    connected = bool(arrays["connected"][index])
    opening = float(arrays["gripper_opening_m"][index])
    lines = [
        f"step: {index}/{len(arrays['phase']) - 1}",
        f"phase: {phase} ({phase_index})",
        f"relation: {relation}",
        f"confidence: {confidence:.3f}",
        f"connected: {connected}",
        f"active frames: {active_text}",
        f"object-target XY: {xy_error:.4f} m",
        f"gripper opening: {opening:.4f} m",
        "recovery: NOT_IMPLEMENTED (baseline)",
        f"perturbation event: {event}",
        f"final: {metrics['failure_reason']}",
    ]
    for offset, line in enumerate(lines):
        color = (80, 120, 255) if (event and "perturbation event" in line) else (225, 225, 225)
        _put(canvas, line, (790, 55 + offset * 27), 0.45, color)

    xy_errors = np.linalg.norm(arrays["object_position"][:, :2] - arrays["target_position"][:, :2], axis=1)
    _put(canvas, "XY error (m)", (790, 382), 0.42, (200, 200, 200))
    _draw_series(canvas, xy_errors, index, (790, 392, 445, 75), (50, 150, 255))
    threshold = float(metrics["success_criteria"]["xy_threshold_m"])
    _put(canvas, f"final threshold: {threshold:.3f} m", (1010, 382), 0.38, (80, 220, 100))
    _put(canvas, "relation confidence", (790, 494), 0.42, (200, 200, 200))
    _draw_series(canvas, arrays["relation_confidence"], index, (790, 504, 445, 65), (220, 170, 40), (0.0, 1.0))
    _put(canvas, "gripper opening (m)", (790, 596), 0.42, (200, 200, 200))
    _draw_series(canvas, arrays["gripper_opening_m"], index, (790, 606, 445, 54), (210, 100, 210))
    _put(canvas, "EE trail", (55, 695), 0.42, (255, 210, 80))
    _put(canvas, "object trail", (160, 695), 0.42, (50, 150, 255))
    _put(canvas, "policy target +", (285, 695), 0.42, (220, 80, 220))
    _put(canvas, "raw target x", (430, 695), 0.42, (150, 150, 150))
    return canvas


def _sample_indices(arrays: dict[str, np.ndarray], metadata: dict[str, Any], maximum_frames: int) -> list[int]:
    steps = len(arrays["phase"])
    stride = max(1, int(np.ceil(steps / maximum_frames)))
    indices = set(range(0, steps, stride))
    indices.add(steps - 1)
    indices.update(int(index) for index in np.flatnonzero(arrays["perturbation_event"]))
    indices.update(int(index) for index in metadata["metrics"].get("relation_state_transition_steps", []))
    indices.update(int(index) for index in metadata["metrics"].get("frame_switch_steps", []))
    return sorted(indices)


def render_trial(
    trials_dir: Path,
    output_dir: Path,
    trial: AuditTrial,
    fps: float = 15.0,
    maximum_frames: int = 180,
) -> dict[str, Any]:
    """Render one trace to MP4, snapshots, and a machine-readable audit row."""

    metadata, arrays = load_trial(trials_dir, trial.stem)
    frames = reconstruct_active_frames(
        len(arrays["phase"]), metadata["metrics"].get("frame_switch_diagnostics", [])
    )
    audit = audit_failure_taxonomy(metadata, arrays)
    if not audit["consistent"]:
        raise ValueError(f"Failure taxonomy disagrees with trace for {trial.stem}: {audit['checks']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{trial.label}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer for {video_path}")
    sample_indices = _sample_indices(arrays, metadata, maximum_frames)
    for index in sample_indices:
        writer.write(render_frame(metadata, arrays, frames, index))
    writer.release()

    event_indices = np.flatnonzero(arrays["perturbation_event"])
    event_index = int(event_indices[0]) if len(event_indices) else len(arrays["phase"]) // 2
    snapshot_indices = [0, event_index, len(arrays["phase"]) - 1]
    snapshots = [render_frame(metadata, arrays, frames, index) for index in snapshot_indices]
    contact_sheet = np.concatenate([cv2.resize(frame, (640, 360)) for frame in snapshots], axis=1)
    contact_path = output_dir / f"{trial.label}_contact.png"
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError(f"Could not write {contact_path}")

    json_path = trials_dir / f"{trial.stem}.json"
    npz_path = trials_dir / f"{trial.stem}.npz"
    return {
        "label": trial.label,
        "source_stem": trial.stem,
        "method": metadata["method"],
        "condition": metadata["condition"],
        "seed": metadata["seed"],
        "experiment_fingerprint": metadata["experiment_fingerprint"],
        "source_json_sha256": sha256_file(json_path),
        "source_npz_sha256": sha256_file(npz_path),
        "video": video_path.name,
        "video_sha256": sha256_file(video_path),
        "contact_sheet": contact_path.name,
        "contact_sheet_sha256": sha256_file(contact_path),
        "rendered_source_steps": sample_indices,
        "active_frames_reconstructable": any(value is not None for value in frames),
        "failure_taxonomy_audit": audit,
    }


def render_audit_set(
    trials_dir: Path,
    output_dir: Path,
    trials: tuple[AuditTrial, ...] = DEFAULT_AUDIT_TRIALS,
    fps: float = 15.0,
    maximum_frames: int = 180,
) -> dict[str, Any]:
    """Render the fixed ten-trial audit set and persist its provenance."""

    if output_dir.resolve() == trials_dir.resolve() or trials_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Audit output must not be written inside the frozen trial directory.")
    rows = [render_trial(trials_dir, output_dir, trial, fps, maximum_frames) for trial in trials]
    manifest = {
        "schema_version": 1,
        "artifact_type": "single_arm_trace_reconstruction_visual_audit",
        "camera_footage": False,
        "source_directory": str(trials_dir),
        "rendering": {
            "fps": fps,
            "maximum_frames_per_trial": maximum_frames,
            "video_codec": "mp4v",
            "resolution": [1280, 720],
            "recovery_overlay": "NOT_IMPLEMENTED",
        },
        "all_failure_taxonomies_consistent": all(row["failure_taxonomy_audit"]["consistent"] for row in rows),
        "trials": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
