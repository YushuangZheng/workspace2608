"""Causal no-action-authority replay adapter for Ours-Monitor on M0 logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from essay2608.policy.tsf import StateId
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext, RuntimeMonitor
from integrations.rlbench.iclr2027.task_registry import experiment_task
from integrations.rlbench.rlbench_tsf.policy_server import TSFPolicyServer
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.task_specs import xyzw_to_wxyz


ROOT = Path(__file__).resolve().parents[3]
SINGLE_MODELS = INTEGRATION_ROOT / "models" / "iclr2027" / "dynamac"
BIMANUAL_MODELS = INTEGRATION_ROOT / "models" / "v4"
TSF_MODELS = INTEGRATION_ROOT / "models" / "iclr2027" / "tsf"
BOUNDARY_ROOT = (
    ROOT
    / "evaluations"
    / "iclr2027"
    / "artifacts"
    / "calibration"
    / "normal_task_boundaries"
    / "main10"
    / "runtime_configs"
)


class OursCausalShadowMonitor(RuntimeMonitor):
    """Run task-state inference on logged M0 observations without action authority.

    The TSF policy computes the same persisted mismatch signal as M5/M6,
    but its proposed action is discarded.  At the next observation we commit the
    action actually recorded in the M0 trajectory, preserving causal progress and
    relation updates while guaranteeing that the shadow path cannot change physics.
    """

    def __init__(
        self,
        task_id: str,
        *,
        tsf_models: Path = TSF_MODELS,
        boundary_root: Path = BOUNDARY_ROOT,
    ) -> None:
        task = experiment_task(task_id)
        base_models = BIMANUAL_MODELS if task.spec.bimanual else SINGLE_MODELS
        self.server = TSFPolicyServer(
            task_id,
            tsf_models,
            base_models,
            feature_profile="generic_retry",
            # Mirror the formal endpoint launcher: bimanual releases resolve
            # their authenticated task specification from the checkpoint
            # manifest, while the ICLR single-arm extensions use tasks.json.
            task_spec=None if task.spec.bimanual else task.spec,
            boundary_config=boundary_root / f"{task_id}.json",
        )
        self.task_id = task_id
        self.bimanual = task.spec.bimanual
        self.context = None
        self.initialized = False
        self.previous_action: list[float] | None = None
        self.previous_reference_states: dict[str, StateId] | None = None
        self._alarm = False
        self._reason_count = 0
        self._continuous_score = 0.0
        self._metadata: dict[str, Any] = {}
        self._terminal_state: str | None = None

    def reset(self, episode_context: EpisodeContext) -> None:
        if episode_context.task_id != self.task_id:
            raise ValueError("shadow task identity mismatch")
        if episode_context.bimanual != self.bimanual:
            raise ValueError("shadow arm-count identity mismatch")
        if self.server._pending is not None:
            self.server.policy.abort()
            self.server._pending = None
        self.context = episode_context
        self.initialized = False
        self.previous_action = None
        self.previous_reference_states = None
        self._alarm = False
        self._reason_count = 0
        self._continuous_score = 0.0
        self._metadata = {}
        self._terminal_state = None

    def _payload(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        arms = observation["arms"]
        payload: dict[str, Any] = {
            "task_low_dim_state": list(observation["task_state"]),
        }
        if self.bimanual:
            for arm in ("left", "right"):
                payload[arm] = {
                    "gripper_pose": list(arms[arm]["ee_pose_xyzw"]),
                    "gripper_open": float(arms[arm]["gripper_open"]),
                }
        else:
            payload.update(
                {
                    "gripper_pose": list(arms["single"]["ee_pose_xyzw"]),
                    "gripper_open": float(arms["single"]["gripper_open"]),
                }
            )
        return payload

    def _logged_commands(self, action: list[float]) -> dict[str, np.ndarray]:
        value = np.asarray(action, dtype=np.float64)
        if self.bimanual:
            if value.shape != (18,):
                raise ValueError("bimanual shadow action must have 18 values")
            return {
                "right": xyzw_to_wxyz(value[:7]),
                "left": xyzw_to_wxyz(value[9:16]),
            }
        if value.shape != (9,):
            raise ValueError("single-arm shadow action must have 9 values")
        return {"single": xyzw_to_wxyz(value[:7])}

    def _reference_states(
        self, policy_state: Mapping[str, Any]
    ) -> dict[str, StateId]:
        raw = policy_state.get("reference_state_if_available")
        if not isinstance(raw, Mapping):
            raise ValueError("shadow 回放缺少动作 reference_state")
        by_arm = raw if self.bimanual else {"single": raw}
        if set(by_arm) != set(self.server.arms):
            raise ValueError("shadow 动作 reference_state 未覆盖全部机械臂")
        result = {}
        for arm, value in by_arm.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"shadow {arm} reference_state 必须为对象")
            state = StateId(int(value["skill"]), int(value["progress"]))
            mode = int(value["mode"])
            expected_mode = self.server.policy._mode_by_arm_skill[arm][
                state.skill_index
            ]
            if mode != expected_mode:
                raise ValueError(
                    f"shadow {arm} 记录模态 {mode} 与冻结路径 {expected_mode} 不一致"
                )
            result[arm] = state
        return result

    def _commit_previous(self, resolution: Mapping[str, Any]) -> None:
        if self.server._pending is None or self.previous_action is None:
            return
        if self.previous_reference_states is None:
            raise RuntimeError("shadow 缺少上一动作引用状态")
        self.server._pending["commands"] = self._logged_commands(self.previous_action)
        aggregate = str(resolution.get("aggregate", "stopped"))
        per_arm = resolution.get("per_arm")
        if not isinstance(per_arm, Mapping) or set(per_arm) != set(self.server.arms):
            per_arm = {arm: aggregate for arm in self.server.arms}
        applied = bool(resolution.get("primary_action_applied", True))
        self.server._resolve(
            {
                "transaction_id": self.server._pending["transaction_id"],
                "primary_action_status": aggregate,
                "primary_action_statuses": dict(per_arm),
                "primary_action_applied": applied,
            },
            commit=True,
            executed_reference_states=(
                self.previous_reference_states if applied else None
            ),
        )

    def observe(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        policy_state: Mapping[str, Any],
    ) -> None:
        if self.context is None:
            raise RuntimeError("reset must precede observe")
        payload = self._payload(observation)
        if self._terminal_state is not None:
            self._alarm = self._terminal_state == "structured_failure"
            self._reason_count = int(self._alarm)
            self._continuous_score = float(self._alarm)
            self._metadata = {
                "source": "tsf_task_state_on_logged_m0_trajectory",
                "action_authority": False,
                "proposed_action_discarded": True,
                "terminal_state_latched": self._terminal_state,
            }
            self.previous_action = list(action["action"])
            return
        if not self.initialized:
            self.server._reset(payload)
            self.initialized = True
            reference_states = self._reference_states(policy_state)
            source_states = reference_states
        else:
            resolution = observation.get("previous_action_resolution", {})
            if not isinstance(resolution, Mapping):
                raise ValueError("previous action resolution must be a mapping")
            self._commit_previous(resolution)
            if self.previous_reference_states is None:
                raise RuntimeError("shadow 缺少产生当前观测的动作引用状态")
            source_states = self.previous_reference_states
            reference_states = self._reference_states(policy_state)
            if self.server.policy.failed or self.server.policy.complete:
                self._terminal_state = (
                    "structured_failure"
                    if self.server.policy.failed
                    else "policy_complete"
                )
                self._alarm = self.server.policy.failed
                self._reason_count = int(self._alarm)
                self._continuous_score = float(self._alarm)
                self._metadata = {
                    "source": "tsf_task_state_on_logged_m0_trajectory",
                    "action_authority": False,
                    "proposed_action_discarded": True,
                    "terminal_state_latched": self._terminal_state,
                }
                self.previous_action = list(action["action"])
                self.previous_reference_states = reference_states
                return
        self.server.policy.synchronize_observer_references(source_states)
        response = self.server._act(
            payload,
            observer_only=True,
            observer_reference_states=reference_states,
        )
        self.server.policy.synchronize_observer_transition_context(
            source_states,
            reference_states,
        )
        monitor = response.get("policy_state", {}).get("monitor", {})
        reasons = monitor.get("reasons", ()) if isinstance(monitor, Mapping) else ()
        self._reason_count = len(reasons) if isinstance(reasons, (list, tuple)) else 0
        self._alarm = bool(
            monitor.get("alarm", False) if isinstance(monitor, Mapping) else False
        )
        self._continuous_score = float(
            monitor.get("continuous_score", float(self._alarm))
            if isinstance(monitor, Mapping)
            else float(self._alarm)
        )
        self._metadata = {
            "source": "tsf_task_state_on_logged_m0_trajectory",
            "action_authority": False,
            "proposed_action_discarded": True,
        }
        self.previous_action = list(action["action"])
        self.previous_reference_states = reference_states

    def score(self) -> Mapping[str, float]:
        return {
            "task_state_mismatch": self._continuous_score,
            "trigger_reasons": float(self._reason_count),
        }

    def alarm(self) -> bool:
        return self._alarm

    @property
    def threshold(self) -> float:
        return 1.0

    @property
    def persistence_count(self) -> int:
        return int(self._alarm)

    @property
    def output_metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def close(self) -> None:
        if self.server._pending is not None:
            self.server.policy.abort()
            self.server._pending = None


__all__ = ["OursCausalShadowMonitor"]
