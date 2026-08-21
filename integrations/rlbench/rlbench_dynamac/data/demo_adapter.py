"""Offline RLBench-demonstration adapter for the local DynaMAC core.

This module imports neither RLBench nor PyRep at module-import time.  The
current local ``DynaMACDemonstration`` type is loaded lazily only when
constructing demonstrations, so offline data conversion stays simulator-free.
Pose/frame schemas come from the pinned public RLBench fork, while skill labels
come from the independent TAPAS-code-aligned NumPy port plus the author's
2026-08-14 semantic clarification.  Pose and gripper targets use the clarified
current-observation time-state stream; the legacy next-observation helpers
remain public compatibility APIs.

RLBench's ``low_dim_obs.pkl`` is a Python pickle and therefore is not a safe
interchange format.  :func:`load_low_dim_obs_pickle` uses a narrow allowlist,
local inert proxy classes, file/array limits, and recursive result validation.
That materially reduces exposure but is not a cryptographic sandbox; only
files produced by the pinned collection workflow should be accepted.
"""

from __future__ import annotations

import io
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from essay2608.policy.tapas_segmentation import (
    TAPAS_NUMPY_PORT_SOURCE_STATUS,
    BimanualTAPASSegmentation,
    TAPASSegmentation,
    TAPASSegmentationConfig,
    segment_bimanual_trajectories,
    segment_trajectories,
)

from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
    DYNAMAC_CURRENT_STATE_TIMING,
    current_gripper_state,
    load_rlbench_segmentation_config,
    save_bimanual_segmentation_debug_plot,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    RLBENCH_REFERENCE_COMMIT,
    TaskSpec,
    get_task_spec,
    xyzw_to_wxyz,
)

Array = np.ndarray
DEFAULT_MAX_PICKLE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_ARRAY_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_GRAPH_OBJECTS = 2_000_000
PICKLE_LOADER_STATUS = "RESTRICTED_ALLOWLISTED_RLBENCH_PICKLE_LOADER"
ADAPTER_CLAIM_BOUNDARY = (
    "RLBench observations and TAPAS helpers are pinned public-code inputs. Candidate-frame, "
    "time-state pose, boundary-union, and task-coordination semantics follow the author's "
    "2026-08-14 clarification; implementation remains an independent adapter."
)
DYNAMAC_POSE_TARGET_TIMING = "time-state current EE pose from obs[t]"
DYNAMAC_GRIPPER_TARGET_TIMING = "time-state current gripper state from obs[t]"


class UnsafeLowDimPickleError(ValueError):
    """Raised when a pickle requests a class or graph shape outside the allowlist."""


class _PickleRecordProxy:
    """Inert state holder used instead of importing RLBench observation classes."""


class _DemoProxy(_PickleRecordProxy):
    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, index: int) -> Any:
        return self._observations[index]

    def __iter__(self):
        return iter(self._observations)


def _numpy_multiarray() -> Any:
    """Resolve NumPy's pickle ABI namespace across NumPy 1.x and 2.x."""

    namespace = getattr(np, "_core", None)
    if namespace is None:  # NumPy 1.x compatibility path
        namespace = getattr(np, "core")
    return namespace.multiarray


def _numpy_reconstruct() -> Any:
    # ``_reconstruct`` is the representation used by the pinned NumPy/RLBench
    # pickles.  Fetching it without a dynamic import keeps find_class closed.
    return _numpy_multiarray()._reconstruct


def _numpy_frombuffer(
    buffer: bytes | bytearray | memoryview,
    dtype: Any,
    shape: Sequence[int],
    order: str,
) -> Array:
    """Local equivalent of NumPy's protocol-5 pickle helper."""

    parsed_dtype = np.dtype(dtype)
    if parsed_dtype.hasobject:
        raise UnsafeLowDimPickleError("object-dtype NumPy arrays are forbidden")
    try:
        return np.frombuffer(buffer, dtype=parsed_dtype).reshape(tuple(shape), order=order)
    except (TypeError, ValueError) as exc:
        raise UnsafeLowDimPickleError("invalid NumPy from-buffer pickle payload") from exc


_PROXY_GLOBALS = {
    ("rlbench.demo", "Demo"): _DemoProxy,
    ("rlbench.backend.observation", "Observation"): _PickleRecordProxy,
    ("rlbench.backend.observation", "UnimanualObservation"): _PickleRecordProxy,
    ("rlbench.backend.observation", "UnimanualObservationData"): _PickleRecordProxy,
    ("rlbench.backend.observation", "BimanualObservation"): _PickleRecordProxy,
}


class _RestrictedRLBenchUnpickler(pickle.Unpickler):
    """Unpickler with no dynamic imports and a closed global allowlist."""

    def find_class(self, module: str, name: str) -> Any:
        proxy = _PROXY_GLOBALS.get((module, name))
        if proxy is not None:
            return proxy
        allowed_numpy = {
            ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct(),
            ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct(),
            ("numpy.core.multiarray", "scalar"): _numpy_multiarray().scalar,
            ("numpy._core.multiarray", "scalar"): _numpy_multiarray().scalar,
            ("numpy.core.numeric", "_frombuffer"): _numpy_frombuffer,
            ("numpy._core.numeric", "_frombuffer"): _numpy_frombuffer,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        value = allowed_numpy.get((module, name))
        if value is not None:
            return value
        allowed_builtins = {
            ("builtins", "set"): set,
            ("builtins", "frozenset"): frozenset,
            ("__builtin__", "set"): set,
            ("__builtin__", "frozenset"): frozenset,
        }
        value = allowed_builtins.get((module, name))
        if value is not None:
            return value
        raise UnsafeLowDimPickleError(f"low_dim_obs.pkl requested forbidden global {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise UnsafeLowDimPickleError(f"pickle persistent IDs are forbidden: {pid!r}")


def _validate_pickle_graph(
    root: Any,
    *,
    max_array_bytes: int,
    max_graph_objects: int,
) -> None:
    stack = [root]
    seen: set[int] = set()
    total_array_bytes = 0
    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen) > max_graph_objects:
            raise UnsafeLowDimPickleError("pickle object graph exceeds configured limit")
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            continue
        if isinstance(value, np.generic):
            if value.dtype.hasobject:
                raise UnsafeLowDimPickleError("object-typed NumPy scalar is forbidden")
            continue
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise UnsafeLowDimPickleError("object-dtype NumPy arrays are forbidden")
            total_array_bytes += value.nbytes
            if total_array_bytes > max_array_bytes:
                raise UnsafeLowDimPickleError("pickle NumPy payload exceeds configured byte limit")
            continue
        if isinstance(value, Mapping):
            stack.extend(value.keys())
            stack.extend(value.values())
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
            continue
        if isinstance(value, _PickleRecordProxy):
            stack.append(vars(value))
            continue
        raise UnsafeLowDimPickleError(
            f"pickle produced forbidden object type {type(value).__module__}."
            f"{type(value).__qualname__}"
        )


def load_low_dim_obs_pickle(
    path: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_PICKLE_BYTES,
    max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
    max_graph_objects: int = DEFAULT_MAX_GRAPH_OBJECTS,
) -> Any:
    """Load a pinned-RLBench low-dimensional episode with a closed allowlist.

    The loader never imports classes named by the pickle.  RLBench ``Demo`` and
    observation objects become local attribute-compatible proxies; numeric
    arrays are reconstructed through the minimal NumPy globals required by the
    format.  Pickle should still be treated as trusted-project data rather than
    an untrusted network protocol.
    """

    source = Path(path)
    if max_file_bytes < 1 or max_array_bytes < 1 or max_graph_objects < 1:
        raise ValueError("pickle safety limits must be positive")
    stat = source.stat()
    if not source.is_file():
        raise FileNotFoundError(f"low-dimensional pickle is not a file: {source}")
    if stat.st_size > max_file_bytes:
        raise UnsafeLowDimPickleError(f"pickle has {stat.st_size} bytes, limit is {max_file_bytes}")
    payload = source.read_bytes()
    try:
        value = _RestrictedRLBenchUnpickler(
            io.BytesIO(payload), fix_imports=False, encoding="bytes"
        ).load()
    except UnsafeLowDimPickleError:
        raise
    except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError) as exc:
        raise UnsafeLowDimPickleError(f"invalid restricted RLBench pickle: {source}") from exc
    _validate_pickle_graph(
        value,
        max_array_bytes=max_array_bytes,
        max_graph_objects=max_graph_objects,
    )
    return value


def load_low_dim_obs_pickles(paths: Sequence[str | Path]) -> list[Any]:
    """Load an ordered episode collection using the restricted loader."""

    if not paths:
        raise ValueError("at least one low_dim_obs.pkl path is required")
    return [load_low_dim_obs_pickle(path) for path in paths]


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        try:
            return value[name]
        except KeyError as exc:
            raise ValueError(f"observation mapping has no {name!r} field") from exc
    if not hasattr(value, name):
        raise ValueError(f"observation has no {name!r} field")
    return getattr(value, name)


def _episode_observations(episode: Any) -> list[Any]:
    if hasattr(episode, "_observations"):
        observations = list(episode._observations)
    elif isinstance(episode, Sequence) and not isinstance(
        episode, (str, bytes, bytearray, np.ndarray)
    ):
        observations = list(episode)
    else:
        try:
            observations = list(iter(episode))
        except TypeError as exc:
            raise ValueError("episode must be a Demo or observation sequence") from exc
    if not observations:
        raise ValueError("RLBench episode must contain at least one observation")
    return observations


def _scalar_gripper(value: Any, *, label: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"{label} gripper_open must be scalar, got {array.shape}")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} gripper_open must be finite and in [0, 1]")
    return result


def _arm_fields(observation: Any, arm: str | None) -> tuple[Any, Any]:
    if arm is None:
        return _field(observation, "gripper_pose"), _field(observation, "gripper_open")
    try:
        nested = _field(observation, arm)
    except ValueError:
        # A mapping-shaped synthetic fixture may expose flattened arm fields;
        # the pinned observation class itself uses nested ``left``/``right``.
        return (
            _field(observation, f"{arm}_gripper_pose"),
            _field(observation, f"{arm}_gripper_open"),
        )
    return _field(nested, "gripper_pose"), _field(nested, "gripper_open")


@dataclass(frozen=True)
class ArmEpisodeArrays:
    """Validated core-convention arrays for one arm and one episode."""

    ee_pose: Array
    gripper_state: Array
    frames: dict[str, Array]


@dataclass(frozen=True)
class BimanualEpisodeArrays:
    """Sample-aligned arrays for both arms and their shared task frames."""

    left: ArmEpisodeArrays
    right: ArmEpisodeArrays


def extract_unimanual_episode(episode: Any, task: str | TaskSpec) -> ArmEpisodeArrays:
    """Extract one unimanual episode without importing the simulator."""

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if spec.bimanual:
        raise ValueError(f"{spec.task_name} is bimanual")
    observations = _episode_observations(episode)
    ee_pose = np.stack(
        [xyzw_to_wxyz(_arm_fields(observation, None)[0]) for observation in observations]
    )
    gripper = np.asarray(
        [
            _scalar_gripper(_arm_fields(observation, None)[1], label="unimanual")
            for observation in observations
        ],
        dtype=np.float64,
    )
    frame_samples = [
        spec.extract_pose_chunks(_field(observation, "task_low_dim_state"))
        for observation in observations
    ]
    frames = {
        name: np.stack([sample[name] for sample in frame_samples]) for name in spec.frame_names
    }
    return ArmEpisodeArrays(ee_pose=ee_pose, gripper_state=gripper, frames=frames)


def extract_bimanual_episode(episode: Any, task: str | TaskSpec) -> BimanualEpisodeArrays:
    """Extract paired left/right arrays from one bimanual observation stream."""

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if not spec.bimanual:
        raise ValueError(f"{spec.task_name} is unimanual")
    observations = _episode_observations(episode)
    frame_samples = [
        spec.extract_pose_chunks(_field(observation, "task_low_dim_state"))
        for observation in observations
    ]
    frames = {
        name: np.stack([sample[name] for sample in frame_samples]) for name in spec.frame_names
    }

    def arm_arrays(arm: str) -> ArmEpisodeArrays:
        fields = [_arm_fields(observation, arm) for observation in observations]
        ee_pose = np.stack([xyzw_to_wxyz(pose) for pose, _ in fields])
        gripper = np.asarray(
            [_scalar_gripper(value, label=arm) for _, value in fields],
            dtype=np.float64,
        )
        return ArmEpisodeArrays(
            ee_pose=ee_pose,
            gripper_state=gripper,
            frames={name: value.copy() for name, value in frames.items()},
        )

    return BimanualEpisodeArrays(left=arm_arrays("left"), right=arm_arrays("right"))


def _core_demonstration_type() -> Any:
    """Return the current project implementation's demonstration type."""

    from essay2608.policy import DynaMACDemonstration

    return DynaMACDemonstration


def _resolve_demonstration_type(demonstration_type: type[Any] | None) -> type[Any]:
    """Select the data class used for converted demonstrations.

    Callers may still pass the class explicitly for testing or embedding.  The
    default deliberately follows the current project source; obsolete sealed
    schema snapshots are no longer part of the active RLBench path.
    """

    if demonstration_type is None:
        return _core_demonstration_type()
    if not isinstance(demonstration_type, type):
        raise TypeError("demonstration_type must be a class")
    return demonstration_type


def _names_for(spec: TaskSpec, count: int, names: Sequence[str] | None) -> list[str]:
    if names is None:
        return [f"{spec.task_name}_episode_{index:03d}" for index in range(count)]
    result = list(names)
    if len(result) != count or any(not isinstance(name, str) or not name for name in result):
        raise ValueError("names must contain one non-empty string per episode")
    if len(set(result)) != len(result):
        raise ValueError("demonstration names must be unique")
    return result


def _config_or_default(
    config: TAPASSegmentationConfig | Mapping[str, Any] | None,
) -> TAPASSegmentationConfig:
    if config is None:
        return load_rlbench_segmentation_config()
    if isinstance(config, TAPASSegmentationConfig):
        return config
    return TAPASSegmentationConfig.from_mapping(config)


def _validate_segmentation_lengths(
    segmentation: TAPASSegmentation,
    episodes: Sequence[ArmEpisodeArrays],
) -> None:
    expected = tuple(len(episode.ee_pose) for episode in episodes)
    if segmentation.trajectory_lengths != expected:
        raise ValueError(
            f"segmentation lengths {segmentation.trajectory_lengths} do not match {expected}"
        )


@dataclass(frozen=True)
class UnimanualDemonstrationResult:
    """DynaMAC inputs plus the exact segmentation evidence used to build them."""

    demonstrations: list[Any]
    segmentation: TAPASSegmentation
    audit: dict[str, Any]

    def __iter__(self):
        yield self.demonstrations
        yield self.segmentation


@dataclass(frozen=True)
class BimanualDemonstrationResult:
    """Paired per-arm DynaMAC inputs and task-coordinated boundary evidence."""

    left_demonstrations: list[Any]
    right_demonstrations: list[Any]
    segmentation: BimanualTAPASSegmentation
    audit: dict[str, Any]

    @property
    def demonstrations(self) -> tuple[list[Any], list[Any]]:
        return self.left_demonstrations, self.right_demonstrations

    def __iter__(self):
        yield self.left_demonstrations
        yield self.right_demonstrations
        yield self.segmentation


def make_unimanual_demonstrations(
    episodes: Sequence[Any],
    task: str | TaskSpec,
    *,
    names: Sequence[str] | None = None,
    config: TAPASSegmentationConfig | Mapping[str, Any] | None = None,
    segmentation: TAPASSegmentation | None = None,
    signed_gripper: bool = True,
    demonstration_type: type[Any] | None = None,
) -> UnimanualDemonstrationResult:
    """Build a list of core demonstrations and retain its boundary audit."""

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if spec.bimanual:
        raise ValueError(f"{spec.task_name} requires make_bimanual_demonstrations")
    if not episodes:
        raise ValueError("at least one RLBench episode is required")
    arrays = [extract_unimanual_episode(episode, spec) for episode in episodes]
    demo_names = _names_for(spec, len(arrays), names)
    cfg = _config_or_default(config).for_task(spec.task_name)
    if segmentation is None:
        segmentation = segment_trajectories(
            [episode.ee_pose for episode in arrays],
            frame_trajectories=[episode.frames for episode in arrays],
            gripper_states=[episode.gripper_state for episode in arrays],
            config=cfg,
        )
    _validate_segmentation_lengths(segmentation, arrays)

    Demonstration = _resolve_demonstration_type(demonstration_type)
    demonstrations: list[Any] = []
    for name, episode, labels in zip(demo_names, arrays, segmentation.skill_labels, strict=True):
        gripper_action = current_gripper_state(episode.gripper_state, signed=signed_gripper)[
            :, None
        ]
        demonstrations.append(
            Demonstration(
                ee_pose=episode.ee_pose,
                action_pose=episode.ee_pose.copy(),
                gripper=gripper_action,
                frames=episode.frames,
                skill=labels,
                name=name,
            )
        )
    audit = {
        "schema": "rlbench-dynamac-demo-adapter-v3",
        "task": spec.task_name,
        "bimanual": False,
        "demonstration_names": demo_names,
        "trajectory_lengths": [len(episode.ee_pose) for episode in arrays],
        "frame_names": list(spec.frame_names),
        "task_frame_source_status": spec.source_status,
        "candidate_frame_policy": spec.candidate_frame_policy,
        "candidate_frame_policy_source_status": spec.candidate_frame_policy_source_status,
        "rlbench_reference_commit": RLBENCH_REFERENCE_COMMIT,
        "pose_conversion": "RLBench world xyzw -> core world wxyz",
        "pose_target_timing": DYNAMAC_POSE_TARGET_TIMING,
        "gripper_action_timing": DYNAMAC_GRIPPER_TARGET_TIMING,
        "action_timing": DYNAMAC_CURRENT_STATE_TIMING,
        "pose_and_gripper_sample_aligned": True,
        "gripper_encoding": "2 * gripper_open - 1" if signed_gripper else "native [0, 1]",
        "segmentation": segmentation.audit,
        "adapter_claim_boundary": ADAPTER_CLAIM_BOUNDARY,
    }
    return UnimanualDemonstrationResult(demonstrations, segmentation, audit)


def make_bimanual_demonstrations(
    episodes: Sequence[Any],
    task: str | TaskSpec,
    *,
    names: Sequence[str] | None = None,
    config: TAPASSegmentationConfig | Mapping[str, Any] | None = None,
    segmentation: BimanualTAPASSegmentation | None = None,
    signed_gripper: bool = True,
    demonstration_type: type[Any] | None = None,
    debug_plot_path: str | Path | None = None,
) -> BimanualDemonstrationResult:
    """Build paired lists with task-configured left/right TAPAS labels.

    Task-frame trajectories are shared because they come from the same
    observation snapshot.  The opposite EE is deliberately not inserted here:
    essay2608's ``BimanualDynaMAC.fit`` injects it from the paired demonstrations
    and thereby keeps the two observation streams sample-aligned.
    """

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if not spec.bimanual:
        raise ValueError(f"{spec.task_name} requires make_unimanual_demonstrations")
    if not episodes:
        raise ValueError("at least one RLBench episode is required")
    arrays = [extract_bimanual_episode(episode, spec) for episode in episodes]
    demo_names = _names_for(spec, len(arrays), names)
    cfg = _config_or_default(config).for_task(spec.task_name)
    if segmentation is None:
        segmentation = segment_bimanual_trajectories(
            [episode.left.ee_pose for episode in arrays],
            [episode.right.ee_pose for episode in arrays],
            frame_trajectories=[episode.left.frames for episode in arrays],
            left_gripper_states=[episode.left.gripper_state for episode in arrays],
            right_gripper_states=[episode.right.gripper_state for episode in arrays],
            config=cfg,
            coordination=spec.segmentation_coordination,
            coordination_source_status=spec.segmentation_coordination_source_status,
            debug_plots_required=spec.segmentation_debug_plots_required,
        )
    if segmentation.coordination != spec.segmentation_coordination:
        raise ValueError(
            f"{spec.task_name} requires {spec.segmentation_coordination} bimanual "
            f"segmentation, received {segmentation.coordination}"
        )
    _validate_segmentation_lengths(segmentation.left, [episode.left for episode in arrays])
    _validate_segmentation_lengths(segmentation.right, [episode.right for episode in arrays])
    if debug_plot_path is not None:
        save_bimanual_segmentation_debug_plot(
            [episode.left.ee_pose for episode in arrays],
            [episode.right.ee_pose for episode in arrays],
            [episode.left.gripper_state for episode in arrays],
            [episode.right.gripper_state for episode in arrays],
            segmentation,
            debug_plot_path,
            title=f"{spec.paper_task_name} segmentation",
        )

    Demonstration = _resolve_demonstration_type(demonstration_type)
    left_demonstrations: list[Any] = []
    right_demonstrations: list[Any] = []
    for index, (name, episode) in enumerate(zip(demo_names, arrays, strict=True)):
        left_demonstrations.append(
            Demonstration(
                ee_pose=episode.left.ee_pose,
                action_pose=episode.left.ee_pose.copy(),
                gripper=current_gripper_state(episode.left.gripper_state, signed=signed_gripper)[
                    :, None
                ],
                frames=episode.left.frames,
                skill=segmentation.left.labels_for(index),
                name=f"{name}_left",
            )
        )
        right_demonstrations.append(
            Demonstration(
                ee_pose=episode.right.ee_pose,
                action_pose=episode.right.ee_pose.copy(),
                gripper=current_gripper_state(episode.right.gripper_state, signed=signed_gripper)[
                    :, None
                ],
                frames=episode.right.frames,
                skill=segmentation.right.labels_for(index),
                name=f"{name}_right",
            )
        )
    audit = {
        "schema": "rlbench-dynamac-demo-adapter-v3",
        "task": spec.task_name,
        "bimanual": True,
        "demonstration_names": demo_names,
        "trajectory_lengths": [len(episode.left.ee_pose) for episode in arrays],
        "frame_names": list(spec.frame_names),
        "task_frame_source_status": spec.source_status,
        "candidate_frame_policy": spec.candidate_frame_policy,
        "candidate_frame_policy_source_status": spec.candidate_frame_policy_source_status,
        "rlbench_reference_commit": RLBENCH_REFERENCE_COMMIT,
        "pose_conversion": "RLBench world xyzw -> core world wxyz",
        "pose_target_timing": DYNAMAC_POSE_TARGET_TIMING,
        "gripper_action_timing": DYNAMAC_GRIPPER_TARGET_TIMING,
        "action_timing": DYNAMAC_CURRENT_STATE_TIMING,
        "pose_and_gripper_sample_aligned": True,
        "gripper_encoding": "2 * gripper_open - 1" if signed_gripper else "native [0, 1]",
        "segmentation_source_status": TAPAS_NUMPY_PORT_SOURCE_STATUS,
        "bimanual_segmentation_source_status": segmentation.coordination_source_status,
        "task_segmentation_coordination": spec.segmentation_coordination,
        "task_segmentation_coordination_source_status": (
            spec.segmentation_coordination_source_status
        ),
        "segmentation_debug_plots_required": spec.segmentation_debug_plots_required,
        "segmentation": segmentation.audit,
        "opposite_ee_frame_added_by_adapter": False,
        "opposite_ee_frame_injection": "essay2608.BimanualDynaMAC.fit from paired snapshots",
        "adapter_claim_boundary": ADAPTER_CLAIM_BOUNDARY,
    }
    return BimanualDemonstrationResult(
        left_demonstrations,
        right_demonstrations,
        segmentation,
        audit,
    )


adapt_unimanual_demonstrations = make_unimanual_demonstrations
adapt_bimanual_demonstrations = make_bimanual_demonstrations


__all__ = [
    "ADAPTER_CLAIM_BOUNDARY",
    "ArmEpisodeArrays",
    "BimanualDemonstrationResult",
    "BimanualEpisodeArrays",
    "DEFAULT_MAX_ARRAY_BYTES",
    "DEFAULT_MAX_GRAPH_OBJECTS",
    "DEFAULT_MAX_PICKLE_BYTES",
    "DYNAMAC_GRIPPER_TARGET_TIMING",
    "DYNAMAC_POSE_TARGET_TIMING",
    "PICKLE_LOADER_STATUS",
    "UnimanualDemonstrationResult",
    "UnsafeLowDimPickleError",
    "adapt_bimanual_demonstrations",
    "adapt_unimanual_demonstrations",
    "extract_bimanual_episode",
    "extract_unimanual_episode",
    "load_low_dim_obs_pickle",
    "load_low_dim_obs_pickles",
    "make_bimanual_demonstrations",
    "make_unimanual_demonstrations",
]
