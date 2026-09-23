"""Shared, explicit causal input projection for M3 and M4 (no audit inputs).

Only the public FeatureRecord is accepted. Optional stream *names* and Gaussian
parameters are not encoded; stream counts/weight summaries are. Layout comes
from task geometry, never fault labels, family names, or evaluation trajectories.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from evaluations.iclr2027.interfaces.feature_schema import (
    FEATURE_SCHEMA,
    validate_feature_record,
)

ENCODER_SCHEMA = "essay2608.iclr2027.monitor-vector.v1"


def _number(value: Any) -> float:
    if not isinstance(value, (bool, int, float)) or not np.isfinite(value):
        raise ValueError("numeric monitor input must be finite")
    return float(value)


def _optional(mapping: Mapping[str, Any], key: str) -> list[float]:
    value = mapping.get(key)
    return [0.0, 0.0] if value is None else [1.0, _number(value)]


def _physical_response_flags(status: Any, *, global_status: bool) -> list[float]:
    """Encode response semantics without treating healthy progress as a stall.

    The Hybrid shared executor distinguishes a target that was fully reached
    from one that made bounded physical progress.  Both are informative
    successful responses to the previous command; only ``stopped`` (or an
    unknown non-success value) belongs to the blocked-response channel.  The
    legacy encoder collapsed ``progressed`` with ``stopped``, even though the
    legacy training executor had emitted almost exclusively ``reached``.  Keep
    the frozen vector width while restoring the intended physical semantics.
    """

    successful = status in ("reached", "progressed")
    blocked = status is not None and not successful and (
        not global_status or status != "initial"
    )
    if global_status:
        return [
            float(status is None),
            float(status == "initial"),
            float(successful),
            float(blocked),
        ]
    return [float(status is None), float(successful), float(blocked)]


@dataclass(frozen=True)
class FeatureLayout:
    arms: tuple[str, ...]
    task_state_dim: int
    action_dim: int
    schema: str = ENCODER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENCODER_SCHEMA or self.arms not in (
            ("single",),
            ("left", "right"),
        ):
            raise ValueError("unsupported monitor layout")
        if self.task_state_dim <= 0 or self.action_dim != 9 * len(self.arms):
            raise ValueError("invalid frozen task/action dimensions")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FeatureLayout:
        clean = validate_feature_record(record)
        arms = ("single",) if "single" in clean["arms"] else ("left", "right")
        return cls(arms, len(clean["task_state"]), len(clean["action"]))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureLayout:
        return cls(
            tuple(value["arms"]), value["task_state_dim"], value["action_dim"], value["schema"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "arms": list(self.arms)}

    @property
    def input_dim(self) -> int:
        # arm pose/gripper; raw task/action; optional policy_step; each arm:
        # ref state (3 presence/value pairs), stream summaries (6), resolution (3);
        # global primary-applied presence/value and aggregate resolution (4).
        return 23 * len(self.arms) + self.task_state_dim + self.action_dim + 8

    def encode(self, record: Mapping[str, Any]) -> np.ndarray:
        return self.encode_validated(validate_feature_record(record))

    def encode_validated(self, clean: Mapping[str, Any]) -> np.ndarray:
        """For A's canonical loader output only; public inference uses encode()."""
        if (
            set(clean["arms"]) != set(self.arms)
            or len(clean["task_state"]) != self.task_state_dim
            or len(clean["action"]) != self.action_dim
        ):
            raise ValueError("feature dimensions differ from frozen checkpoint layout")
        out = []
        for arm in self.arms:
            out.extend(clean["arms"][arm]["ee_pose_xyzw"])
            out.append(clean["arms"][arm]["gripper_open"])
        out.extend(clean["task_state"])
        # Preserve A's raw action order, including its bimanual right/left order.
        out.extend(clean["action"])
        policy = clean["policy_state"]
        out.extend(_optional(policy, "policy_step"))
        reference = policy.get("reference_state_if_available") or {}
        streams = policy.get("stream_metadata") or {}
        resolution = clean["action_resolution"]
        per_arm = resolution.get("per_arm") or {}
        for arm in self.arms:
            ref = reference if arm == "single" else reference.get(arm, {})
            meta = streams if arm == "single" else streams.get(arm, {})
            for key in ("mode", "skill", "progress"):
                out.extend(_optional(ref, key))
            for key in ("active_streams", "selected_streams"):
                values = meta.get(key)
                if values is not None and not isinstance(values, (list, tuple)):
                    raise ValueError("stream identities must be a list")
                out.append(float(len(values)) if values is not None else 0.0)
            weights = meta.get("poe_weights")
            if weights is not None and not isinstance(weights, Mapping):
                raise ValueError("poe_weights must be a mapping")
            vals = [_number(v) for v in (weights or {}).values()]
            out.extend(
                [
                    float(weights is not None),
                    sum(vals),
                    sum(v * v for v in vals),
                    max(vals, default=0.0),
                ]
            )
            status = per_arm.get(arm)
            out.extend(_physical_response_flags(status, global_status=False))
        out.extend(_optional(resolution, "primary_action_applied"))
        status = resolution.get("aggregate")
        out.extend(_physical_response_flags(status, global_status=True))
        result = np.asarray(out, dtype=np.float32)
        if result.shape != (self.input_dim,) or not np.isfinite(result).all():
            raise ValueError("invalid encoded monitor vector")
        return result


def runtime_record(
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    policy_state: Mapping[str, Any],
    episode_id: str,
) -> dict[str, Any]:
    """ABI: full record or observation fields + {'action', 'action_timestamp'}."""
    value = dict(observation)
    value.setdefault("schema", FEATURE_SCHEMA)
    value.setdefault("episode_id", episode_id)
    if action:
        value["action"] = action["action"]
        value["action_timestamp"] = action["action_timestamp"]
    value["policy_state"] = dict(policy_state)
    clean = validate_feature_record(value)
    if clean["episode_id"] != episode_id:
        raise ValueError("record belongs to a different episode; call reset first")
    return clean
