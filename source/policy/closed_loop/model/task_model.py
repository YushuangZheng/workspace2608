"""Unified offline task model for closed-loop multi-stream execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np

from .boundary_model import (
    BoundaryId,
    BoundaryModel,
    LocalCompletionModel,
    RelationGuardDistribution,
    ReliabilityStatistics,
)
from .relation_events import (
    LinkPendingCandidate,
    LinkRecoveryAnchor,
    RelationEventId,
    RelationStateKey,
    UnlinkEventMetadata,
)
from .scene_factors import FactorDistribution, FactorId, FactorSpace
from .state_index import StateId, StateTopology, build_state_topology

if TYPE_CHECKING:
    from ...dynamac import DynaMAC, DynaMACAction, DynaMACObservation

Array = np.ndarray


@dataclass
class StateNode:
    """All offline information addressed by one shared ``StateId``."""

    state_id: StateId
    topology: StateTopology
    stream_means: dict[str, Array]
    stream_covariances: dict[str, Array]
    demo_relation_scores: dict[str, Array]
    demo_relation_priors: dict[str, Array]
    mode_priors: Array
    selected_frames: tuple[str, ...]
    mode_selected_frames: tuple[tuple[str, ...], ...]
    gripper_commands: Array
    # Equation (6) defines the candidate expert bank used by task-state
    # evidence.  Action relevance is a stricter, offline LODO screen used only
    # for normal PoE execution.  Keeping the two masks separate prevents an
    # action-routing decision from deleting relation/progress evidence.
    action_relevant_frames: tuple[str, ...] | None = None
    mode_action_relevant_frames: tuple[tuple[str, ...], ...] | None = None
    scene_factor_models: dict[FactorId, dict[int, FactorDistribution]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if set(self.stream_means) != set(self.stream_covariances):
            raise ValueError("StateNode 的流均值与协方差键必须一致")
        if set(self.demo_relation_scores) != set(self.demo_relation_priors):
            raise ValueError("关系分数与关系先验必须覆盖同一组参考系")
        modes = len(self.mode_priors)
        mode_priors = np.asarray(self.mode_priors, dtype=np.float64)
        if (
            mode_priors.ndim != 1
            or modes == 0
            or np.any(mode_priors < 0.0)
            or not np.isclose(np.sum(mode_priors), 1.0)
        ):
            raise ValueError("StateNode 模态先验必须是非空归一化向量")
        self.mode_priors = mode_priors.copy()
        if len(self.mode_selected_frames) != modes:
            raise ValueError("StateNode 每个模态必须具有对应的流选择结果")
        unknown_selected = set(self.selected_frames).difference(self.stream_means)
        if unknown_selected:
            raise ValueError(f"StateNode 选择了未建模流：{sorted(unknown_selected)}")
        mode_union: set[str] = set()
        for selected in self.mode_selected_frames:
            unknown_mode = set(selected).difference(self.stream_means)
            if unknown_mode:
                raise ValueError(
                    f"StateNode 模态选择了未建模流：{sorted(unknown_mode)}"
                )
            mode_union.update(selected)
        if set(self.selected_frames) != mode_union:
            raise ValueError("StateNode selected_frames 必须等于逐模态选择结果的并集")
        if self.action_relevant_frames is None:
            self.action_relevant_frames = tuple(self.selected_frames)
        if self.mode_action_relevant_frames is None:
            self.mode_action_relevant_frames = tuple(self.mode_selected_frames)
        if len(self.mode_action_relevant_frames) != modes:
            raise ValueError("StateNode 每个模态必须具有对应的动作相关流结果")
        action_union: set[str] = set()
        for mode, relevant in enumerate(self.mode_action_relevant_frames):
            if not set(relevant).issubset(self.mode_selected_frames[mode]):
                raise ValueError("动作相关流必须是 Equation (6) 模态候选流的子集")
            action_union.update(relevant)
        if set(self.action_relevant_frames) != action_union:
            raise ValueError("action_relevant_frames 必须等于逐模态结果的并集")
        for name, score in self.demo_relation_scores.items():
            values = np.asarray(score, dtype=np.float64)
            if (
                values.shape != (modes,)
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise ValueError(f"参考系 {name} 的关系连接分数必须为有限非负 [M] 向量")
            self.demo_relation_scores[name] = values.copy()
        for name, prior in self.demo_relation_priors.items():
            values = np.asarray(prior, dtype=np.float64)
            if (
                values.shape != (modes, 2)
                or np.any(values < 0.0)
                or not np.allclose(np.sum(values, axis=1), 1.0)
            ):
                raise ValueError(
                    f"参考系 {name} 的关系先验必须是逐模态归一化 [M,2] 数组"
                )
            self.demo_relation_priors[name] = values.copy()
        for name, mean in self.stream_means.items():
            covariance = np.asarray(self.stream_covariances[name], dtype=np.float64)
            values = np.asarray(mean, dtype=np.float64)
            if values.shape != (modes, 7) or covariance.shape != (modes, 6, 6):
                raise ValueError(f"流 {name} 必须保留 [M,7] 均值和 [M,6,6] 协方差")
        gripper = np.asarray(self.gripper_commands, dtype=np.float64)
        if gripper.ndim != 2 or gripper.shape[0] != modes or gripper.shape[1] == 0:
            raise ValueError("StateNode 夹爪命令必须为非空 [M,G] 数组")
        self.gripper_commands = gripper.copy()
        for factor_id, distributions in self.scene_factor_models.items():
            if not distributions or any(
                mode < 0 or mode >= modes for mode in distributions
            ):
                raise ValueError(f"场景因子 {factor_id.token} 的模态索引无效")


@dataclass
class ClosedLoopTaskModel:
    """Additional closed-loop statistics bound to one fitted DynaMAC."""

    base_policy: DynaMAC
    states: dict[StateId, StateNode]
    skill_states: dict[int, tuple[StateId, ...]]
    boundaries: dict[BoundaryId, BoundaryModel]
    link_anchors: dict[RelationEventId, LinkRecoveryAnchor]
    unlink_events: dict[RelationEventId, UnlinkEventMetadata]
    link_pending_events: dict[RelationEventId, LinkPendingCandidate] = field(
        default_factory=dict
    )
    link_origins: dict[RelationStateKey, RelationEventId] = field(default_factory=dict)
    arm_id: str = "single"
    relation_frames: tuple[str, ...] = ()
    builder_config: dict[str, Any] = field(default_factory=dict)
    base_policy_fingerprint: str = ""
    schema_version: int = 5

    def __post_init__(self) -> None:
        if not self.base_policy.fitted:
            raise RuntimeError("ClosedLoopTaskModel 需要已拟合的基础 DynaMAC")
        fingerprint = self.base_policy.fingerprint()
        if not self.base_policy_fingerprint:
            self.base_policy_fingerprint = fingerprint
        elif self.base_policy_fingerprint != fingerprint:
            raise ValueError("闭环任务模型与基础 DynaMAC 指纹不匹配")
        if set(self.states) != {
            state for states in self.skill_states.values() for state in states
        }:
            raise ValueError("skill_states 必须完整且唯一地覆盖所有状态")
        expected = sum(skill.duration for skill in self.base_policy.skills)
        if len(self.states) != expected:
            raise ValueError("闭环任务状态数量与基础 DynaMAC 不一致")
        if any(name.startswith("virtual_skill_") for name in self.relation_frames):
            raise ValueError("虚拟技能帧不能进入关系状态空间")
        if any(event_id.transition != "link" for event_id in self.link_anchors):
            raise ValueError("正式 LINK 锚点必须使用 link 事件标识")
        if any(event_id.transition != "unlink" for event_id in self.unlink_events):
            raise ValueError("UNLINK 元数据必须使用 unlink 事件标识")
        for event_id, candidate in self.link_pending_events.items():
            if (
                event_id != candidate.event_id
                or event_id.transition != "link_pending"
                or candidate.arm_id != self.arm_id
                or candidate.frame_id not in self.relation_frames
                or candidate.candidate_state not in self.states
                or candidate.context_state not in self.states
            ):
                raise ValueError("LINK_PENDING 候选与任务模型不一致")
        unknown_origins = set(self.link_origins.values()).difference(self.link_anchors)
        if unknown_origins:
            raise ValueError("link_origin 引用了不存在的 LINK 锚点")
        for key, event_id in self.link_origins.items():
            if (
                key.arm_id != self.arm_id
                or event_id.arm_id != self.arm_id
                or key.frame_id != event_id.frame_id
                or key.state_id not in self.states
            ):
                raise ValueError("link_origin 的机械臂、参考系或状态与事件不一致")

    def query_state(
        self,
        observation: DynaMACObservation,
        state_id: StateId,
        stream_weights: dict[str, float] | None = None,
        *,
        mode_index: int | None = None,
    ) -> DynaMACAction:
        if state_id not in self.states:
            raise KeyError(f"闭环任务模型不存在状态 {state_id}")
        return self.base_policy.query_state(
            observation,
            state_id,
            stream_weights,
            mode_index=mode_index,
        )

    def state(self, state_id: StateId) -> StateNode:
        try:
            return self.states[state_id]
        except KeyError as exc:
            raise KeyError(f"闭环任务模型不存在状态 {state_id}") from exc

    def active_link_pending_candidate(
        self,
        frame_id: str,
        state_id: StateId,
        mode_by_skill: Mapping[int, int] | None = None,
    ) -> LinkPendingCandidate | None:
        """Resolve the latest still-active relation occurrence at a state.

        ``LINK_PENDING`` identifies an event occurrence, not only its closing
        sample.  Its context therefore remains active after ``candidate_state``
        until a later enabled LINK, LINK_PENDING, or UNLINK for the same
        arm-frame pair replaces it.  This does not create a formal
        ``link_origin`` and does not assert that the relation is linked.
        """

        if state_id not in self.states:
            raise KeyError(f"闭环任务模型不存在状态 {state_id}")
        ordered_states = tuple(
            state
            for skill in sorted(self.skill_states)
            for state in self.skill_states[skill]
        )
        global_index = {state: index for index, state in enumerate(ordered_states)}
        context_index = global_index[state_id]

        def enabled(event_id: RelationEventId) -> bool:
            if event_id.frame_id != frame_id:
                return False
            if mode_by_skill is None:
                return True
            selected_mode = mode_by_skill.get(event_id.skill_index)
            return selected_mode is None or selected_mode == event_id.mode

        transitions: list[tuple[int, int, str, LinkPendingCandidate | None]] = []
        for event_id, candidate in self.link_pending_events.items():
            if enabled(event_id):
                transitions.append(
                    (
                        global_index[candidate.candidate_state],
                        0,
                        event_id.token,
                        candidate,
                    )
                )
        for event_id, anchor in self.link_anchors.items():
            if not enabled(event_id):
                continue
            event_state = (
                anchor.linked_entry_states[0]
                if anchor.linked_entry_states
                else anchor.context_state
            )
            transitions.append((global_index[event_state], 1, event_id.token, None))
        for event_id, event in self.unlink_events.items():
            if enabled(event_id):
                transitions.append(
                    (global_index[event.release_state], 1, event_id.token, None)
                )

        reached = [row for row in transitions if row[0] <= context_index]
        if not reached:
            return None
        return max(reached, key=lambda row: row[:3])[3]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "base_policy_fingerprint": self.base_policy_fingerprint,
            "base_policy_selection_semantics_id": self.base_policy.selection_semantics_id,
            "state_count": len(self.states),
            "skill_state_counts": {
                str(index): len(states) for index, states in self.skill_states.items()
            },
            "boundary_count": len(self.boundaries),
            "link_anchor_count": len(self.link_anchors),
            "link_pending_count": len(self.link_pending_events),
            "unlink_event_count": len(self.unlink_events),
            "link_origin_count": len(self.link_origins),
            "relation_frames": list(self.relation_frames),
            "scene_factor_count": sum(
                sum(
                    len(distributions)
                    for distributions in node.scene_factor_models.values()
                )
                for node in self.states.values()
            ),
            "action_relevance": self.builder_config.get("action_stream_relevance", []),
            "builder_config": self.builder_config,
        }

    @staticmethod
    def _state_meta(state_id: StateId) -> list[int]:
        return list(state_id.as_tuple())

    @staticmethod
    def _distribution_meta(
        distribution: FactorDistribution,
        arrays: dict[str, Array],
        prefix: str,
    ) -> dict[str, Any]:
        mean_key = f"{prefix}__mean"
        covariance_key = f"{prefix}__covariance"
        arrays[mean_key] = distribution.mean
        arrays[covariance_key] = distribution.covariance
        return {
            "mean": mean_key,
            "covariance": covariance_key,
            "sample_count": distribution.sample_count,
            "space": distribution.space,
            "observability": distribution.observability,
            "stable_fraction": distribution.stable_fraction,
            "loo_gain": distribution.loo_gain,
            "loo_accuracy": distribution.loo_accuracy,
            "neighborhood_radius": distribution.neighborhood_radius,
        }

    @staticmethod
    def _distribution_from_meta(
        metadata: dict[str, Any],
        archive: Any,
    ) -> FactorDistribution:
        space_value = str(metadata.get("space", "se3"))
        if space_value not in {"se3", "euclidean"}:
            raise ValueError(f"未知场景因子空间：{space_value}")
        return FactorDistribution(
            mean=archive[metadata["mean"]].copy(),
            covariance=archive[metadata["covariance"]].copy(),
            sample_count=int(metadata["sample_count"]),
            space=cast(FactorSpace, space_value),
            observability=float(metadata["observability"]),
            stable_fraction=float(metadata.get("stable_fraction", 1.0)),
            loo_gain=float(metadata["loo_gain"]),
            loo_accuracy=float(metadata["loo_accuracy"]),
            neighborhood_radius=int(metadata.get("neighborhood_radius", 0)),
        )

    def save(self, path: str | Path) -> None:
        """Save only additional statistics; base stream arrays stay external."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Array] = {}
        state_records = []
        for state_index, (state_id, node) in enumerate(sorted(self.states.items())):
            state_records.append(
                {
                    "state_id": self._state_meta(state_id),
                    "relation_scores": {
                        name: score.tolist()
                        for name, score in sorted(node.demo_relation_scores.items())
                    },
                    "relation_priors": {
                        name: prior.tolist()
                        for name, prior in sorted(node.demo_relation_priors.items())
                    },
                    # The closed-loop sidecar may remove an Eq.6-selected peer
                    # end-effector from execution after bimanual dependency
                    # analysis.  Persist that strict subset instead of silently
                    # reconstructing the frozen base-policy selection on load.
                    "selected_frames": list(node.selected_frames),
                    "mode_selected_frames": [
                        list(selected) for selected in node.mode_selected_frames
                    ],
                    "action_relevant_frames": list(node.action_relevant_frames),
                    "mode_action_relevant_frames": [
                        list(selected) for selected in node.mode_action_relevant_frames
                    ],
                    "scene_factors": [
                        {
                            "factor_id": factor_id.token,
                            "mode_distributions": {
                                str(mode): self._distribution_meta(
                                    distribution,
                                    arrays,
                                    f"state_{state_index}__factor_{factor_index}__mode_{mode}",
                                )
                                for mode, distribution in sorted(distributions.items())
                            },
                        }
                        for factor_index, (factor_id, distributions) in enumerate(
                            sorted(node.scene_factor_models.items())
                        )
                    ],
                }
            )

        boundary_records = []
        for boundary_index, (boundary_id, boundary) in enumerate(
            sorted(self.boundaries.items())
        ):
            local = boundary.local_completion_model
            boundary_records.append(
                {
                    "boundary_id": {
                        "arm_id": boundary_id.arm_id,
                        "source_skill": boundary_id.source_skill,
                        "target_skill": boundary_id.target_skill,
                    },
                    "terminal_window": [
                        self._state_meta(state) for state in boundary.terminal_window
                    ],
                    "local_completion": {
                        "terminal_states": [
                            self._state_meta(state) for state in local.terminal_states
                        ],
                        "goal_distributions": {
                            name: self._distribution_meta(
                                distribution,
                                arrays,
                                f"boundary_{boundary_index}__goal_{goal_index}",
                            )
                            for goal_index, (name, distribution) in enumerate(
                                sorted(local.goal_distributions.items())
                            )
                        },
                        "minimum_goal_log_likelihood": local.minimum_goal_log_likelihood,
                        "own_relation_conditions": {
                            name: {
                                "external": condition.external,
                                "linked": condition.linked,
                                "required_state": condition.required_state,
                            }
                            for name, condition in sorted(
                                local.own_relation_conditions.items()
                            )
                        },
                    },
                    "relation_conditions": {
                        name: {
                            "external": condition.external,
                            "linked": condition.linked,
                            "required_state": condition.required_state,
                        }
                        for name, condition in sorted(
                            boundary.relation_conditions.items()
                        )
                    },
                    "scene_conditions": [
                        {
                            "factor_id": factor_id.token,
                            "minimum_compatibility": boundary.scene_condition_thresholds[
                                factor_id
                            ],
                            "distribution": self._distribution_meta(
                                distribution,
                                arrays,
                                f"boundary_{boundary_index}__scene_{scene_index}",
                            ),
                        }
                        for scene_index, (factor_id, distribution) in enumerate(
                            sorted(boundary.scene_conditions.items())
                        )
                    ],
                    "condition_reliability": {
                        name: {
                            "observed_fraction": value.observed_fraction,
                            "stable_fraction": value.stable_fraction,
                        }
                        for name, value in sorted(
                            boundary.condition_reliability.items()
                        )
                    },
                    "affected_arms": list(boundary.affected_arms),
                    "transaction_group": boundary.transaction_group,
                }
            )

        link_records = []
        for event_index, (event_id, anchor) in enumerate(
            sorted(self.link_anchors.items())
        ):
            prefix = f"link_{event_index}"
            arrays[f"{prefix}__means"] = anchor.local_means
            arrays[f"{prefix}__covariances"] = anchor.local_covariances
            arrays[f"{prefix}__gripper"] = anchor.gripper_commands
            link_records.append(
                {
                    "event_id": event_id.__dict__,
                    "context_state": self._state_meta(anchor.context_state),
                    "local_means": f"{prefix}__means",
                    "local_covariances": f"{prefix}__covariances",
                    "gripper_commands": f"{prefix}__gripper",
                    "linked_entry_states": [
                        self._state_meta(state) for state in anchor.linked_entry_states
                    ],
                    "support_fraction": anchor.support_fraction,
                    "demonstration_indices": list(anchor.demonstration_indices),
                    "event_local_indices": list(anchor.event_local_indices),
                }
            )

        unlink_records = []
        for event_index, (event_id, event) in enumerate(
            sorted(self.unlink_events.items())
        ):
            target_key = None
            if event.local_detachment_target is not None:
                target_key = f"unlink_{event_index}__target"
                arrays[target_key] = event.local_detachment_target
            unlink_records.append(
                {
                    "event_id": event_id.__dict__,
                    "release_state": self._state_meta(event.release_state),
                    "legal_reentry_states": [
                        self._state_meta(state) for state in event.legal_reentry_states
                    ],
                    "local_detachment_target": target_key,
                    "support_fraction": event.support_fraction,
                    "demonstration_indices": list(event.demonstration_indices),
                    "event_local_indices": list(event.event_local_indices),
                }
            )

        link_pending_records = []
        for event_index, (event_id, candidate) in enumerate(
            sorted(self.link_pending_events.items())
        ):
            prefix = f"link_pending_{event_index}"
            arrays[f"{prefix}__means"] = candidate.local_means
            arrays[f"{prefix}__covariances"] = candidate.local_covariances
            arrays[f"{prefix}__gripper"] = candidate.gripper_commands
            link_pending_records.append(
                {
                    "event_id": event_id.__dict__,
                    "candidate_state": self._state_meta(candidate.candidate_state),
                    "context_state": self._state_meta(candidate.context_state),
                    "local_means": f"{prefix}__means",
                    "local_covariances": f"{prefix}__covariances",
                    "gripper_commands": f"{prefix}__gripper",
                    "support_fraction": candidate.support_fraction,
                    "demonstration_indices": list(candidate.demonstration_indices),
                    "event_local_indices": list(candidate.event_local_indices),
                }
            )

        metadata = {
            "schema": "essay2608.closed_loop_task_model.v5",
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "base_policy_fingerprint": self.base_policy_fingerprint,
            "base_policy_selection_semantics_id": self.base_policy.selection_semantics_id,
            "relation_frames": list(self.relation_frames),
            "builder_config": self.builder_config,
            "states": state_records,
            "boundaries": boundary_records,
            "link_anchors": link_records,
            "link_pending_events": link_pending_records,
            "unlink_events": unlink_records,
            "link_origins": [
                {
                    "arm_id": key.arm_id,
                    "frame_id": key.frame_id,
                    "state_id": self._state_meta(key.state_id),
                    "mode": key.mode,
                    "event_id": event_id.__dict__,
                }
                for key, event_id in sorted(self.link_origins.items())
            ],
        }
        with path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                metadata_json=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                ),
                **arrays,
            )

    @classmethod
    def load(cls, path: str | Path, base_policy: DynaMAC) -> ClosedLoopTaskModel:
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            schema = metadata.get("schema")
            if schema not in {
                "essay2608.closed_loop_task_model.v3",
                "essay2608.closed_loop_task_model.v4",
                "essay2608.closed_loop_task_model.v5",
            }:
                raise ValueError("不支持的闭环任务模型 schema")
            if metadata.get("base_policy_fingerprint") != base_policy.fingerprint():
                raise ValueError("闭环任务模型与提供的基础 DynaMAC 指纹不匹配")
            if (
                metadata.get("base_policy_selection_semantics_id")
                != base_policy.selection_semantics_id
            ):
                raise ValueError("闭环任务模型与基础 DynaMAC 选择语义不匹配")
            topology = build_state_topology(base_policy)
            states: dict[StateId, StateNode] = {}
            skill_states: dict[int, list[StateId]] = {
                index: [] for index in range(len(base_policy.skills))
            }
            for record in metadata["states"]:
                state_id = StateId.from_tuple(record["state_id"])
                skill = base_policy.skills[state_id.skill_index]
                stream_means = {
                    name: stream.mean[:, state_id.local_index]
                    for name, stream in skill.streams.items()
                }
                stream_covariances = {
                    name: stream.covariance[:, state_id.local_index]
                    for name, stream in skill.streams.items()
                }
                scene_factors = {
                    FactorId.from_token(item["factor_id"]): {
                        int(mode): cls._distribution_from_meta(value, archive)
                        for mode, value in item["mode_distributions"].items()
                    }
                    for item in record["scene_factors"]
                }
                if schema in {
                    "essay2608.closed_loop_task_model.v4",
                    "essay2608.closed_loop_task_model.v5",
                }:
                    selected_frames = tuple(record["selected_frames"])
                    mode_selected_frames = tuple(
                        tuple(selected) for selected in record["mode_selected_frames"]
                    )
                else:
                    # Read-only compatibility for stage-one schema v3.  Those
                    # sidecars predate peer execution-dependency filtering and
                    # therefore used the frozen DynaMAC selection verbatim.
                    selected_frames = tuple(skill.selected_frames)
                    mode_selected_frames = tuple(
                        tuple(
                            name
                            for name, stream in skill.streams.items()
                            if name in skill.selected_frames
                            and stream.is_selected(mode)
                        )
                        for mode in range(len(skill.mode_priors))
                    )
                if schema == "essay2608.closed_loop_task_model.v5":
                    action_relevant_frames = tuple(record["action_relevant_frames"])
                    mode_action_relevant_frames = tuple(
                        tuple(selected)
                        for selected in record["mode_action_relevant_frames"]
                    )
                else:
                    # Older sidecars predate the separate action-value mask;
                    # reading them preserves their candidate mask verbatim.
                    action_relevant_frames = selected_frames
                    mode_action_relevant_frames = mode_selected_frames
                states[state_id] = StateNode(
                    state_id=state_id,
                    topology=topology[state_id],
                    stream_means=stream_means,
                    stream_covariances=stream_covariances,
                    demo_relation_scores={
                        name: np.asarray(value, dtype=np.float64)
                        for name, value in record["relation_scores"].items()
                    },
                    demo_relation_priors={
                        name: np.asarray(values, dtype=np.float64)
                        for name, values in record["relation_priors"].items()
                    },
                    mode_priors=skill.mode_priors,
                    selected_frames=selected_frames,
                    mode_selected_frames=mode_selected_frames,
                    gripper_commands=skill.gripper[:, state_id.local_index],
                    action_relevant_frames=action_relevant_frames,
                    mode_action_relevant_frames=mode_action_relevant_frames,
                    scene_factor_models=scene_factors,
                )
                skill_states[state_id.skill_index].append(state_id)

            boundaries: dict[BoundaryId, BoundaryModel] = {}
            for record in metadata["boundaries"]:
                boundary_id = BoundaryId(**record["boundary_id"])
                local_meta = record["local_completion"]
                local = LocalCompletionModel(
                    terminal_states=tuple(
                        StateId.from_tuple(value)
                        for value in local_meta["terminal_states"]
                    ),
                    goal_distributions={
                        name: cls._distribution_from_meta(value, archive)
                        for name, value in local_meta["goal_distributions"].items()
                    },
                    minimum_goal_log_likelihood={
                        name: float(value)
                        for name, value in local_meta[
                            "minimum_goal_log_likelihood"
                        ].items()
                    },
                    own_relation_conditions={
                        name: RelationGuardDistribution(**value)
                        for name, value in local_meta.get(
                            "own_relation_conditions", {}
                        ).items()
                    },
                )
                boundaries[boundary_id] = BoundaryModel(
                    boundary_id=boundary_id,
                    source_skill=boundary_id.source_skill,
                    target_skill=boundary_id.target_skill,
                    terminal_window=tuple(
                        StateId.from_tuple(value) for value in record["terminal_window"]
                    ),
                    local_completion_model=local,
                    relation_conditions={
                        name: RelationGuardDistribution(**value)
                        for name, value in record["relation_conditions"].items()
                    },
                    scene_conditions={
                        FactorId.from_token(
                            item["factor_id"]
                        ): cls._distribution_from_meta(item["distribution"], archive)
                        for item in record["scene_conditions"]
                    },
                    scene_condition_thresholds={
                        FactorId.from_token(item["factor_id"]): float(
                            item["minimum_compatibility"]
                        )
                        for item in record["scene_conditions"]
                    },
                    condition_reliability={
                        name: ReliabilityStatistics(**value)
                        for name, value in record["condition_reliability"].items()
                    },
                    affected_arms=tuple(record["affected_arms"]),
                    transaction_group=record["transaction_group"],
                )

            link_anchors: dict[RelationEventId, LinkRecoveryAnchor] = {}
            for record in metadata["link_anchors"]:
                event_id = RelationEventId(**record["event_id"])
                link_anchors[event_id] = LinkRecoveryAnchor(
                    event_id=event_id,
                    arm_id=event_id.arm_id,
                    frame_id=event_id.frame_id,
                    context_state=StateId.from_tuple(record["context_state"]),
                    local_means=archive[record["local_means"]].copy(),
                    local_covariances=archive[record["local_covariances"]].copy(),
                    gripper_commands=archive[record["gripper_commands"]].copy(),
                    linked_entry_states=tuple(
                        StateId.from_tuple(value)
                        for value in record["linked_entry_states"]
                    ),
                    support_fraction=float(record["support_fraction"]),
                    demonstration_indices=tuple(record["demonstration_indices"]),
                    event_local_indices=tuple(record["event_local_indices"]),
                )

            unlink_events: dict[RelationEventId, UnlinkEventMetadata] = {}
            for record in metadata["unlink_events"]:
                event_id = RelationEventId(**record["event_id"])
                target_key = record["local_detachment_target"]
                unlink_events[event_id] = UnlinkEventMetadata(
                    event_id=event_id,
                    arm_id=event_id.arm_id,
                    frame_id=event_id.frame_id,
                    release_state=StateId.from_tuple(record["release_state"]),
                    legal_reentry_states=tuple(
                        StateId.from_tuple(value)
                        for value in record["legal_reentry_states"]
                    ),
                    local_detachment_target=(
                        None if target_key is None else archive[target_key].copy()
                    ),
                    support_fraction=float(record["support_fraction"]),
                    demonstration_indices=tuple(record["demonstration_indices"]),
                    event_local_indices=tuple(record["event_local_indices"]),
                )

            link_pending_events: dict[RelationEventId, LinkPendingCandidate] = {}
            for record in metadata["link_pending_events"]:
                event_id = RelationEventId(**record["event_id"])
                link_pending_events[event_id] = LinkPendingCandidate(
                    event_id=event_id,
                    arm_id=event_id.arm_id,
                    frame_id=event_id.frame_id,
                    candidate_state=StateId.from_tuple(record["candidate_state"]),
                    context_state=StateId.from_tuple(record["context_state"]),
                    local_means=archive[record["local_means"]].copy(),
                    local_covariances=archive[record["local_covariances"]].copy(),
                    gripper_commands=archive[record["gripper_commands"]].copy(),
                    support_fraction=float(record["support_fraction"]),
                    demonstration_indices=tuple(record["demonstration_indices"]),
                    event_local_indices=tuple(record["event_local_indices"]),
                )

            link_origins = {}
            for record in metadata["link_origins"]:
                key = RelationStateKey(
                    arm_id=record["arm_id"],
                    frame_id=record["frame_id"],
                    state_id=StateId.from_tuple(record["state_id"]),
                    mode=int(record["mode"]),
                )
                link_origins[key] = RelationEventId(**record["event_id"])

        return cls(
            base_policy=base_policy,
            states=states,
            skill_states={
                index: tuple(sorted(values)) for index, values in skill_states.items()
            },
            boundaries=boundaries,
            link_anchors=link_anchors,
            unlink_events=unlink_events,
            link_pending_events=link_pending_events,
            link_origins=link_origins,
            arm_id=str(metadata["arm_id"]),
            relation_frames=tuple(metadata["relation_frames"]),
            builder_config=dict(metadata["builder_config"]),
            base_policy_fingerprint=str(metadata["base_policy_fingerprint"]),
            schema_version=int(metadata["schema_version"]),
        )


__all__ = ["ClosedLoopTaskModel", "StateNode"]
