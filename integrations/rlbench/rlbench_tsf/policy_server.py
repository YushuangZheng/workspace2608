"""JSON-lines RLBench worker for :class:`TSFMultiStreamPolicy`."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from essay2608.policy import DynaMAC
from essay2608.policy.tsf import (
    BoundaryRuntimeConfig,
    TSFFeatureProfile,
    TSFMultiStreamPolicy,
)
from integrations.rlbench.rlbench_tsf.observation_adapter import (
    TSFObservationAdapter,
    commands_to_rlbench,
)
from integrations.rlbench.rlbench_tsf.protocol import (
    tsf_gripper_timing_metadata,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    POLICY_CLOCK_SEMANTICS_ID,
    PolicyServer as BaselinePolicyServer,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import TaskSpec, load_task_specs


TSF_GRIPPER_TIMING = tsf_gripper_timing_metadata()


class TSFPolicyServer:
    """Keep the simulator protocol outside the reusable policy core."""

    def __init__(
        self,
        task: str,
        models_dir: Path,
        base_models_dir: Path,
        *,
        diagnostics_dir: Path | None = None,
        feature_profile: str = "full",
        task_spec: TaskSpec | None = None,
        boundary_config: Path | None = None,
    ) -> None:
        baseline = BaselinePolicyServer(task, base_models_dir, task_spec=task_spec)
        self.task = task
        self.task_spec = baseline.task_spec
        self.bimanual = baseline.bimanual
        if self.bimanual:
            base_policies = {
                "left": baseline.policy.left,
                "right": baseline.policy.right,
            }
        else:
            base_policies = {"single": baseline.policy}
        runtime_boundary = (
            None
            if boundary_config is None
            else BoundaryRuntimeConfig.from_json(boundary_config)
        )
        self.policy = TSFMultiStreamPolicy.load(
            models_dir / task,
            base_policies=base_policies,
            feature_profile=feature_profile,
            boundary_config=runtime_boundary,
        )
        self.adapter = TSFObservationAdapter(self.task_spec)
        self.arms = self.policy.arms
        self.model_identity = {
            **baseline.model_identity,
            "policy_type": self.policy.name,
            "tsf_bundle": self.policy.summary(),
            "tsf_feature_profile": self.policy.feature_profile.to_dict(),
            "boundary_runtime_config": (
                None
                if boundary_config is None
                else {
                    "path": str(Path(boundary_config).resolve()),
                    "sha256": hashlib.sha256(
                        Path(boundary_config).read_bytes()
                    ).hexdigest(),
                }
            ),
        }
        self.diagnostics_dir = diagnostics_dir
        self._episode_index = -1
        self._tick = 0
        self._previous_ee: dict[str, np.ndarray | None] = {
            arm: None for arm in self.arms
        }
        self._previous_command: dict[str, np.ndarray | None] = {
            arm: None for arm in self.arms
        }
        self._previous_command_covariance: dict[str, np.ndarray | None] = {
            arm: None for arm in self.arms
        }
        self._pending: dict[str, Any] | None = None
        self._next_transaction_id = 1

    def _adapt(self, payload: Mapping[str, Any]):
        return self.adapter.build(
            payload,
            tick=self._tick,
            previous_ee_pose=self._previous_ee,
            previous_command_pose=self._previous_command,
            previous_command_covariance=self._previous_command_covariance,
        )

    def _save_diagnostics(self) -> None:
        if self.diagnostics_dir is None or self._episode_index < 0:
            return
        path = (
            self.diagnostics_dir
            / self.task
            / f"episode_{self._episode_index:04d}.jsonl"
        )
        self.policy.diagnostics.save(path)

    def _reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._pending is not None:
            self.policy.abort()
            self._pending = None
        self._save_diagnostics()
        self._episode_index += 1
        self._tick = 0
        self._previous_ee = {arm: None for arm in self.arms}
        self._previous_command = {arm: None for arm in self.arms}
        self._previous_command_covariance = {arm: None for arm in self.arms}
        batch = self._adapt(payload)
        self.policy.reset(batch.dynamac, mode_strategy="map")
        return {"ok": True, "complete": self.policy.complete}

    def _act(
        self,
        payload: Mapping[str, Any],
        *,
        observer_only: bool = False,
        observer_reference_states: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("上一闭环动作尚未 commit 或 abort")
        batch = self._adapt(payload)
        if observer_only:
            if observer_reference_states is not None:
                self.policy.capture_observer_reference_frames(
                    observer_reference_states,
                    batch.dynamac,
                )
            cycle = self.policy.act(
                batch.dynamac,
                batch.runtime,
                observer_only=True,
            )
        else:
            if observer_reference_states is not None:
                raise ValueError("实时策略不能接收观测回放引用状态")
            cycle = self.policy.act(batch.dynamac, batch.runtime)
        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        self._pending = {
            "transaction_id": transaction_id,
            "pre_action_ee": {
                arm: batch.runtime[arm].ee_pose.copy() for arm in self.arms
            },
            "commands": {arm: cycle.commands[arm].pose.copy() for arm in self.arms},
            "command_covariances": {
                arm: cycle.commands[arm].covariance.copy() for arm in self.arms
            },
            "gripper_authorization": {
                arm: cycle.commands[arm].gripper_authorized for arm in self.arms
            },
        }
        action = commands_to_rlbench(cycle.commands, bimanual=self.bimanual)
        failure_reasons = {
            arm: result.failure_reason
            for arm, result in cycle.arms.items()
            if result.failure_reason is not None
        }
        arm_policy_state = {}
        cycle_diagnostics = getattr(cycle, "diagnostics", {})
        diagnostic_arms = (
            cycle_diagnostics.get("arms", {})
            if isinstance(cycle_diagnostics, Mapping)
            else {}
        )
        for arm in self.arms:
            diagnostic = diagnostic_arms.get(arm, {})
            execution = diagnostic.get("execution") or {}
            controllers = getattr(self.policy, "execution_controllers", {})
            controller = controllers.get(arm) if isinstance(controllers, Mapping) else None
            reference = (
                None if controller is None else controller.cursor.reference_state
            )
            arm_policy_state[arm] = {
                "skill_index": None if reference is None else reference.skill_index,
                "time_index": None if reference is None else reference.local_index,
                "mode": (
                    None
                    if reference is None
                    else self.policy._mode_by_arm_skill[arm][reference.skill_index]
                ),
                "active_frames": list(execution.get("participating_streams", ())),
                "selected_frames": list(execution.get("participating_streams", ())),
                "poe_weights": dict(execution.get("poe_weights", {})),
                "progress_status": diagnostic.get("progress_status"),
                "progress_confidence": diagnostic.get("progress_confidence"),
            }
        raw_trigger = (
            cycle_diagnostics.get("recovery_trigger", {})
            if isinstance(cycle_diagnostics, Mapping)
            else {}
        )
        # Expose a causal, action-free operating-point score in addition to
        # the frozen binary alarm.  Each mismatch counter is normalized by
        # the exact persistence threshold that already governs the policy;
        # no evaluator label or fault metadata enters this calculation.  The
        # native alarm remains authoritative for execution, while shadow
        # analyses may select a stricter multiplier on a disjoint normal-only
        # calibration split when comparing recall at a common false-
        # intervention cost.
        normalized_mismatch_scores = []
        mismatch_counts = {}
        mismatch_thresholds = {}
        for arm, arm_cycle in cycle.arms.items():
            execution = getattr(arm_cycle, "execution", None)
            if execution is None:
                continue
            counts = execution.mismatch.counters
            controllers = getattr(self.policy, "execution_controllers", {})
            controller = (
                controllers.get(arm) if isinstance(controllers, Mapping) else None
            )
            if controller is None:
                continue
            config = controller.mismatch_tracker.config
            by_kind = {
                "no_plausible_state": int(counts.no_plausible_state),
                "relation_mismatch": int(counts.relation_mismatch),
                "persistent_hold": int(counts.persistent_hold),
                "stalled_progress": int(counts.stalled_progress),
            }
            thresholds = {
                "no_plausible_state": int(config.no_plausible_cycles),
                "relation_mismatch": int(config.relation_mismatch_cycles),
                "persistent_hold": int(config.persistent_hold_cycles),
                "stalled_progress": int(config.stalled_progress_cycles),
            }
            mismatch_counts[arm] = by_kind
            mismatch_thresholds[arm] = thresholds
            normalized_mismatch_scores.extend(
                by_kind[kind] / thresholds[kind] for kind in by_kind
            )
        continuous_score = max(normalized_mismatch_scores, default=0.0)
        if bool(raw_trigger.get("triggered", False)):
            # Boundary-relation triggers use their own persistence tracker.
            # They are guaranteed to reach the native operating point even
            # though that tracker is not one of the per-arm mismatch counters.
            continuous_score = max(1.0, continuous_score)
        monitor_state = {
            "alarm": bool(raw_trigger.get("triggered", False)),
            "reasons": list(raw_trigger.get("reasons", ())),
            "intents": list(raw_trigger.get("intents", ())),
            "continuous_score": float(continuous_score),
            "native_threshold": 1.0,
            "mismatch_counts": mismatch_counts,
            "mismatch_thresholds": mismatch_thresholds,
        }
        policy_state = (
            {**arm_policy_state, "monitor": monitor_state}
            if self.bimanual
            else {**arm_policy_state["single"], "monitor": monitor_state}
        )
        evaluator_audit = {}
        for arm in self.arms:
            diagnostic = diagnostic_arms.get(arm, {})
            recovery = diagnostic.get("recovery")
            evaluator_audit[arm] = {
                "mode_before": diagnostic.get("mode_before"),
                "mode_after": diagnostic.get("mode_after"),
                "reentry_committed": bool(
                    isinstance(recovery, Mapping)
                    and recovery.get("reentry") is not None
                ),
            }
        return {
            "ok": True,
            "complete": self.policy.complete,
            "complete_after_commit": self.policy.complete,
            "policy_failed": cycle.lifecycle.value == "failed",
            "failure_reasons": failure_reasons,
            "action": action.tolist(),
            "transaction_id": transaction_id,
            "policy_mode": {arm: cycle.arms[arm].mode_after.value for arm in self.arms},
            # Evaluator-only lifecycle facts.  The formal runner persists this
            # beside the physical audit after monitor inference; it is never
            # included in the causal feature record seen by any monitor.
            "evaluator_audit": {"arms": evaluator_audit},
            "gripper_authorization": {
                arm: cycle.commands[arm].gripper_authorized for arm in self.arms
            },
            "policy_state": policy_state,
        }

    def _resolve(
        self,
        request: Mapping[str, Any],
        *,
        commit: bool,
        executed_reference_states: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._pending is None:
            raise RuntimeError("没有待处理的闭环动作事务")
        allowed_fields = {"command", "transaction_id"}
        if commit:
            allowed_fields.update(
                {
                    "primary_action_status",
                    "primary_action_statuses",
                    "primary_action_applied",
                }
            )
        unknown_fields = set(request).difference(allowed_fields)
        if unknown_fields:
            raise ValueError(f"闭环动作事务包含未知字段：{sorted(unknown_fields)}")
        transaction_id = request.get("transaction_id")
        if transaction_id != self._pending["transaction_id"]:
            raise RuntimeError("闭环动作事务编号不匹配")
        if not commit:
            self.policy.abort()
            action_status = "aborted"
        else:
            explicit_status = request.get("primary_action_status")
            if explicit_status is None:
                raise ValueError("commit 必须显式提供 primary_action_status")
            if not isinstance(explicit_status, str):
                raise TypeError("primary_action_status 必须为字符串")
            action_status = explicit_status
            if action_status not in {"reached", "progressed", "stopped"}:
                raise ValueError("primary_action_status 取值不受支持")
            primary_action_applied = request.get("primary_action_applied", True)
            if not isinstance(primary_action_applied, bool):
                raise TypeError("primary_action_applied 必须为布尔值")
            if not primary_action_applied and action_status != "stopped":
                raise ValueError("未应用主动作时 primary_action_status 必须为 stopped")
            explicit_statuses = request.get("primary_action_statuses")
            if explicit_statuses is None:
                action_statuses = {arm: action_status for arm in self.arms}
            else:
                if not isinstance(explicit_statuses, Mapping):
                    raise TypeError("primary_action_statuses 必须为逐臂映射")
                if set(explicit_statuses) != set(self.arms):
                    raise ValueError("primary_action_statuses 必须覆盖全部机械臂")
                action_statuses = {}
                for arm, status in explicit_statuses.items():
                    if status not in {"reached", "progressed", "stopped"}:
                        raise ValueError(f"primary_action_statuses[{arm}] 取值不受支持")
                    action_statuses[arm] = str(status)
                if action_status == "reached" and any(
                    status != "reached" for status in action_statuses.values()
                ):
                    raise ValueError("整体 reached 要求所有机械臂均 reached")
            if not primary_action_applied and any(
                status != "stopped" for status in action_statuses.values()
            ):
                raise ValueError("未应用主动作时全部机械臂状态必须为 stopped")
            action_response = {
                arm: status in {"reached", "progressed"}
                for arm, status in action_statuses.items()
            }
            action_completed = {
                arm: primary_action_applied and status == "reached"
                for arm, status in action_statuses.items()
            }
            command_applied = {arm: primary_action_applied for arm in self.arms}
            commit_kwargs = {
                "task_command_applied": command_applied,
                "absolute_target_completed": action_completed,
            }
            if executed_reference_states is not None:
                commit_kwargs["executed_reference_states"] = executed_reference_states
            self.policy.commit(**commit_kwargs)
            self._previous_ee = self._pending["pre_action_ee"]
            self._previous_command = (
                self._pending["commands"]
                if primary_action_applied
                else {
                    arm: self._pending["pre_action_ee"][arm].copy() for arm in self.arms
                }
            )
            self._previous_command_covariance = (
                self._pending["command_covariances"]
                if primary_action_applied
                else {arm: None for arm in self.arms}
            )
            self.policy.diagnostics.annotate_last(
                "rlbench_action_resolution",
                {
                    "status": action_status,
                    "status_by_arm": dict(action_statuses),
                    "primary_action_applied": primary_action_applied,
                    "task_command_applied": dict(command_applied),
                    "action_response_observed": dict(action_response),
                    "absolute_target_completed": dict(action_completed),
                    "gripper_authorization": dict(
                        self._pending["gripper_authorization"]
                    ),
                },
            )
            self._tick += 1
        self._pending = None
        return {
            "ok": True,
            "transaction_id": transaction_id,
            "committed": commit,
            "aborted": not commit,
            "primary_action_status": action_status,
            "primary_action_statuses": (None if not commit else dict(action_statuses)),
            "primary_action_applied": (None if not commit else primary_action_applied),
            "complete": self.policy.complete,
        }

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "ping":
            return {
                "ok": True,
                "ready": True,
                "task": self.task,
                "bimanual": self.bimanual,
                "policy_steps": max(
                    len(model.states) for model in self.policy.task_models.values()
                ),
                "model_identity": self.model_identity,
                "policy_clock_semantics_id": POLICY_CLOCK_SEMANTICS_ID,
                "gripper_timing": TSF_GRIPPER_TIMING,
                "policy_type": self.policy.name,
            }
        if command == "close":
            if self._pending is not None:
                self.policy.abort()
                self._pending = None
            self._save_diagnostics()
            return {"ok": True, "closed": True}
        if command == "retry_current_skill":
            if self._pending is not None:
                raise RuntimeError("待提交动作必须先 abort 才能执行 Skill-Retry")
            entries = self.policy.restart_current_skill_reference()
            return {
                "ok": True,
                "retried": True,
                "reference_entries": {
                    arm: {
                        "skill": state.skill_index,
                        "progress": state.local_index,
                    }
                    for arm, state in entries.items()
                },
            }
        if command == "commit":
            return self._resolve(request, commit=True)
        if command == "abort":
            return self._resolve(request, commit=False)
        if command not in {"reset", "act"}:
            raise ValueError(
                "command 必须为 ping/reset/act/commit/abort/"
                "retry_current_skill/close"
            )
        payload = request.get("observation")
        if not isinstance(payload, Mapping):
            raise TypeError("reset/act 必须携带 RLBench 观测对象")
        return self._reset(payload) if command == "reset" else self._act(payload)


def serve(
    task: str,
    models_dir: Path,
    base_models_dir: Path,
    *,
    diagnostics_dir: Path | None = None,
    feature_profile: str = "full",
    task_spec: TaskSpec | None = None,
    boundary_config: Path | None = None,
) -> int:
    server = TSFPolicyServer(
        task,
        models_dir,
        base_models_dir,
        diagnostics_dir=diagnostics_dir,
        feature_profile=feature_profile,
        task_spec=task_spec,
        boundary_config=boundary_config,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = server.handle(request)
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, separators=(",", ":")), flush=True)
        if response.get("closed"):
            return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--task", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--base-models-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--task-specs", type=Path)
    parser.add_argument("--boundary-config", type=Path)
    parser.add_argument(
        "--feature-profile",
        choices=TSFFeatureProfile.names(),
        default="full",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_spec = (
        None
        if args.task_specs is None
        else load_task_specs(args.task_specs)[args.task]
    )
    return serve(
        args.task,
        args.models_dir,
        args.base_models_dir,
        diagnostics_dir=args.diagnostics_dir,
        feature_profile=args.feature_profile,
        task_spec=task_spec,
        boundary_config=args.boundary_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
