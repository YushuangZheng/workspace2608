"""JSON-lines RLBench worker for :class:`ClosedLoopMultiStreamPolicy`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from essay2608.policy import DynaMAC
from essay2608.policy.closed_loop import (
    ClosedLoopFeatureProfile,
    ClosedLoopMultiStreamPolicy,
)
from integrations.rlbench.rlbench_closed_loop.observation_adapter import (
    ClosedLoopObservationAdapter,
    commands_to_rlbench,
)
from integrations.rlbench.rlbench_closed_loop.protocol import (
    closed_loop_gripper_timing_metadata,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    POLICY_CLOCK_SEMANTICS_ID,
    PolicyServer as BaselinePolicyServer,
)


CLOSED_LOOP_GRIPPER_TIMING = closed_loop_gripper_timing_metadata()


class ClosedLoopPolicyServer:
    """Keep the simulator protocol outside the reusable policy core."""

    def __init__(
        self,
        task: str,
        models_dir: Path,
        base_models_dir: Path,
        *,
        diagnostics_dir: Path | None = None,
        feature_profile: str = "full",
    ) -> None:
        baseline = BaselinePolicyServer(task, base_models_dir)
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
        self.policy = ClosedLoopMultiStreamPolicy.load(
            models_dir / task,
            base_policies=base_policies,
            feature_profile=feature_profile,
        )
        self.adapter = ClosedLoopObservationAdapter(self.task_spec)
        self.arms = self.policy.arms
        self.model_identity = {
            **baseline.model_identity,
            "policy_type": self.policy.name,
            "closed_loop_bundle": self.policy.summary(),
            "closed_loop_feature_profile": self.policy.feature_profile.to_dict(),
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

    def _act(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("上一闭环动作尚未 commit 或 abort")
        batch = self._adapt(payload)
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
        return {
            "ok": True,
            "complete": self.policy.complete,
            "complete_after_commit": self.policy.complete,
            "policy_failed": cycle.lifecycle.value == "failed",
            "failure_reasons": failure_reasons,
            "action": action.tolist(),
            "transaction_id": transaction_id,
            "policy_mode": {arm: cycle.arms[arm].mode_after.value for arm in self.arms},
            "gripper_authorization": {
                arm: cycle.commands[arm].gripper_authorized for arm in self.arms
            },
        }

    def _resolve(self, request: Mapping[str, Any], *, commit: bool) -> dict[str, Any]:
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
            self.policy.commit(
                task_command_applied=command_applied,
                absolute_target_completed=action_completed,
            )
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
                "gripper_timing": CLOSED_LOOP_GRIPPER_TIMING,
                "policy_type": self.policy.name,
            }
        if command == "close":
            if self._pending is not None:
                self.policy.abort()
                self._pending = None
            self._save_diagnostics()
            return {"ok": True, "closed": True}
        if command == "commit":
            return self._resolve(request, commit=True)
        if command == "abort":
            return self._resolve(request, commit=False)
        if command not in {"reset", "act"}:
            raise ValueError("command 必须为 ping/reset/act/commit/abort/close")
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
) -> int:
    server = ClosedLoopPolicyServer(
        task,
        models_dir,
        base_models_dir,
        diagnostics_dir=diagnostics_dir,
        feature_profile=feature_profile,
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
    parser.add_argument(
        "--feature-profile",
        choices=ClosedLoopFeatureProfile.names(),
        default="full",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return serve(
        args.task,
        args.models_dir,
        args.base_models_dir,
        diagnostics_dir=args.diagnostics_dir,
        feature_profile=args.feature_profile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
