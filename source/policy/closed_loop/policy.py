"""Phase-six environment-neutral orchestration of stages two through five."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Hashable, Literal, Mapping

import numpy as np

from ..dynamac import DynaMAC, DynaMACAction, DynaMACObservation
from .ablation import ClosedLoopFeatureProfile
from .belief_updater import BeliefUpdater, ClosedLoopBelief
from .bimanual_controller import BoundaryCycleResult, MultiArmBoundaryController
from .config import ClosedLoopPolicyConfig
from .diagnostics import DiagnosticRecorder, json_ready, state_token
from .execution_controller import (
    ClosedLoopExecutionController,
    ExecutionCycleResult,
)
from .frame_roles import RelationVerificationRequest
from .progress_filter import ProgressStatus
from .recovery import (
    ClosedLoopRecoveryManager,
    ExecutionMode,
    RecoveryManagerResult,
    RecoveryPhase,
    RecoverySafetyStatus,
    RecoveryTriggerDecision,
    RecoveryTriggerTracker,
)
from .reentry import ReentryEvaluation
from .relation_filter import RelationDecision
from .relation_verification import (
    AuxiliaryAction,
    SafetyConstraintStatus,
    VerificationPhase,
)
from .runtime_observation import RuntimeObservation
from .serialization import load_policy_bundle, save_policy_bundle
from .state import ArmCommand, ArmCycleResult, PolicyCycleResult, PolicyLifecycle
from .state_index import StateId
from .task_model import ClosedLoopTaskModel

Array = np.ndarray


class ClosedLoopMultiStreamPolicy:
    """One transactional control chain with no RLBench dependency.

    ``act`` prepares and validates a complete tick.  The caller must resolve it
    with ``commit`` after the environment accepts the action, or ``abort`` if
    no control step was consumed.  No second tick may be prepared while one is
    pending.
    """

    name = "closed_loop_multistream"

    def __init__(
        self,
        task_models: Mapping[str, ClosedLoopTaskModel],
        config: ClosedLoopPolicyConfig,
        *,
        feature_profile: ClosedLoopFeatureProfile | str = "full",
    ) -> None:
        if not task_models:
            raise ValueError("闭环顶层策略至少需要一只机械臂")
        self.task_models = dict(task_models)
        if any(model.arm_id != arm for arm, model in self.task_models.items()):
            raise ValueError("闭环顶层策略模型 arm_id 与映射键不一致")
        self.arms = tuple(sorted(self.task_models))
        self.config = config
        self.feature_profile = (
            ClosedLoopFeatureProfile.named(feature_profile)
            if isinstance(feature_profile, str)
            else feature_profile
        )
        if not isinstance(self.feature_profile, ClosedLoopFeatureProfile):
            raise TypeError("feature_profile 必须是名称或 ClosedLoopFeatureProfile")
        self.belief_updaters = {
            arm: BeliefUpdater(model, config.belief)
            for arm, model in self.task_models.items()
        }
        self.execution_controllers = {
            arm: ClosedLoopExecutionController(
                model,
                config.execution,
                config.belief.progress_filter,
                dynamic_frame_roles=self.feature_profile.dynamic_frame_roles,
            )
            for arm, model in self.task_models.items()
        }
        self.recovery_managers = {
            arm: ClosedLoopRecoveryManager(model, config.recovery)
            for arm, model in self.task_models.items()
        }
        self.boundary_controller = MultiArmBoundaryController(
            self.task_models,
            self.execution_controllers,
            config.boundary,
            belief_updaters=self.belief_updaters,
            relation_scene_guards=(self.feature_profile.relation_scene_boundary_guards),
        )
        self.recovery_trigger = RecoveryTriggerTracker(
            self.task_models,
            config.recovery.recovery,
        )
        self.diagnostics = DiagnosticRecorder()
        self._pending_snapshot: dict[str, Any] | None = None
        self._initialized = False
        self._mode_by_arm_skill: dict[str, dict[int, int]] = {}
        self._virtual_frames: dict[str, dict[str, Array]] = {
            arm: {} for arm in self.arms
        }
        self._last_executed_reference: dict[str, StateId | None] = {
            arm: None for arm in self.arms
        }
        self._permitted_boundaries: dict[str, frozenset[Any]] = {
            arm: frozenset() for arm in self.arms
        }
        self._last_cycle: PolicyCycleResult | None = None
        self._lifecycle = PolicyLifecycle.RUNNING
        self._terminal_actions_committed: set[str] = set()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def pending(self) -> bool:
        return self._pending_snapshot is not None

    @property
    def complete(self) -> bool:
        # Match the baseline protocol: completion means every arm's final
        # reference command has been accepted by the environment.  RLBench
        # remains the task-success authority and may use its existing final
        # settling window after this policy-level completion signal.
        return bool(
            self._initialized
            and not self.failed
            and set(self._terminal_actions_committed) == set(self.arms)
            and all(
                self.recovery_managers[arm].mode == ExecutionMode.TASK
                and self.execution_controllers[arm].cursor.reference_state
                == max(self.task_models[arm].states)
                for arm in self.arms
            )
        )

    @property
    def failed(self) -> bool:
        return self._lifecycle == PolicyLifecycle.FAILED

    @property
    def last_cycle(self) -> PolicyCycleResult | None:
        return self._last_cycle

    def _base_runtime(self) -> dict[str, Any]:
        return {
            arm: model.base_policy._capture_runtime_state()
            for arm, model in self.task_models.items()
        }

    def _restore_base_runtime(self, state: Mapping[str, Any]) -> None:
        for arm, value in state.items():
            self.task_models[arm].base_policy._restore_runtime_state(value)

    def reset(
        self,
        observations: Mapping[str, DynaMACObservation],
        *,
        mode_strategy: Literal["map", "sample"] = "map",
    ) -> None:
        if set(observations) != set(self.arms):
            raise ValueError("闭环顶层 reset 必须一次提供全部机械臂观测")
        base_runtime = self._base_runtime()
        try:
            for arm in self.arms:
                self.task_models[arm].base_policy.reset(
                    observations[arm],
                    mode_strategy=mode_strategy,
                )
            self._mode_by_arm_skill = {
                arm: {
                    skill: mode
                    for skill, mode in enumerate(
                        self.task_models[arm].base_policy.selected_mode_path
                    )
                }
                for arm in self.arms
            }
            for arm in self.arms:
                initial = min(self.task_models[arm].states)
                mode = self._mode_by_arm_skill[arm][initial.skill_index]
                node = self.task_models[arm].state(initial)
                initial_relations = {
                    frame: values[mode].copy()
                    for frame, values in node.demo_relation_priors.items()
                }
                self.belief_updaters[arm].reset(
                    initial_progress={initial: 1.0},
                    initial_relations=initial_relations,
                )
                self.execution_controllers[arm].reset(initial)
                self.recovery_managers[arm].reset()
                skill_label = (
                    self.task_models[arm].base_policy.skills[initial.skill_index].label
                )
                self._virtual_frames[arm] = {
                    f"virtual_skill_{skill_label}": observations[arm].ee_pose.copy()
                }
                self._last_executed_reference[arm] = None
                self._permitted_boundaries[arm] = frozenset()
            self.boundary_controller.reset()
            self.recovery_trigger = RecoveryTriggerTracker(
                self.task_models,
                self.config.recovery.recovery,
            )
            self.diagnostics.reset()
            self._pending_snapshot = None
            self._last_cycle = None
            self._lifecycle = PolicyLifecycle.RUNNING
            self._terminal_actions_committed.clear()
            self._initialized = True
        except Exception:
            self._restore_base_runtime(base_runtime)
            raise

    def _snapshot(self) -> dict[str, Any]:
        memo: dict[int, Any] = {id(model): model for model in self.task_models.values()}
        memo.update(
            {
                id(model.base_policy): model.base_policy
                for model in self.task_models.values()
            }
        )
        components = deepcopy(
            (
                self.belief_updaters,
                self.execution_controllers,
                self.recovery_managers,
                self.boundary_controller,
                self.recovery_trigger,
            ),
            memo,
        )
        return {
            "components": components,
            "virtual_frames": deepcopy(self._virtual_frames),
            "last_executed_reference": deepcopy(self._last_executed_reference),
            "permitted_boundaries": deepcopy(self._permitted_boundaries),
            "last_cycle": self._last_cycle,
            "lifecycle": self._lifecycle,
            "terminal_actions_committed": deepcopy(self._terminal_actions_committed),
            "diagnostic_length": len(self.diagnostics.records),
        }

    def _restore(self, snapshot: Mapping[str, Any]) -> None:
        (
            self.belief_updaters,
            self.execution_controllers,
            self.recovery_managers,
            self.boundary_controller,
            self.recovery_trigger,
        ) = snapshot["components"]
        self._virtual_frames = snapshot["virtual_frames"]
        self._last_executed_reference = snapshot["last_executed_reference"]
        self._permitted_boundaries = snapshot["permitted_boundaries"]
        self._last_cycle = snapshot["last_cycle"]
        self._lifecycle = snapshot["lifecycle"]
        self._terminal_actions_committed = snapshot["terminal_actions_committed"]
        self.diagnostics.truncate(int(snapshot["diagnostic_length"]))

    def abort(self) -> None:
        if self._pending_snapshot is None:
            raise RuntimeError("没有待回滚的闭环策略周期")
        snapshot = self._pending_snapshot
        self._pending_snapshot = None
        self._restore(snapshot)

    def commit(
        self,
        *,
        task_command_applied: bool | Mapping[str, bool] = True,
        absolute_target_completed: bool | Mapping[str, bool] | None = None,
        executed_reference_states: Mapping[str, StateId] | None = None,
    ) -> PolicyCycleResult:
        if self._pending_snapshot is None or self._last_cycle is None:
            raise RuntimeError("没有待提交的闭环策略周期")

        def normalize_flags(
            value: bool | Mapping[str, bool],
            *,
            name: str,
        ) -> dict[str, bool]:
            if isinstance(value, (bool, np.bool_)):
                return {arm: bool(value) for arm in self.arms}
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} 必须为布尔值或逐臂布尔映射")
            if set(value) != set(self.arms):
                raise ValueError(f"{name} 必须覆盖全部机械臂")
            normalized = {}
            for arm, flag in value.items():
                if not isinstance(flag, (bool, np.bool_)):
                    raise TypeError(f"{name}[{arm}] 必须为布尔值")
                normalized[arm] = bool(flag)
            return normalized

        applied_by_arm = normalize_flags(
            task_command_applied,
            name="task_command_applied",
        )
        completion_by_arm = normalize_flags(
            (
                task_command_applied
                if absolute_target_completed is None
                else absolute_target_completed
            ),
            name="absolute_target_completed",
        )
        if any(completion_by_arm[arm] and not applied_by_arm[arm] for arm in self.arms):
            raise ValueError("绝对目标完成必须先向对应机械臂施加任务命令")
        if not all(applied_by_arm.values()) and executed_reference_states is not None:
            raise ValueError("存在未响应机械臂时不能提交完整回放动作引用状态")
        if executed_reference_states is not None:
            if set(executed_reference_states) != set(self.arms):
                raise ValueError("回放动作引用必须覆盖全部机械臂")
            for arm, state in executed_reference_states.items():
                if state not in self.task_models[arm].states:
                    raise KeyError(f"{arm} 回放动作引用状态不存在：{state}")
        result = self._last_cycle
        if executed_reference_states is not None:
            # Fixed demonstration playback does not apply the counterfactual
            # policy command.  Its adapter supplies the recorded StateId of
            # the action that actually produced the next saved observation.
            self._last_executed_reference = dict(executed_reference_states)
        else:
            # Keep the tentative task-action anchor whenever the environment
            # actually applied that arm's command.  The executor's physical
            # reached/progressed/stopped classification is deliberately not
            # consumed here: actual motion is already part of the next
            # observation, while exact completion only controls the separate
            # gripper/terminal transaction.
            for arm in self.arms:
                if not applied_by_arm[arm]:
                    self._last_executed_reference[arm] = None
        for arm in self.arms:
            final_state = max(self.task_models[arm].states)
            at_final_task_state = bool(
                self.recovery_managers[arm].mode == ExecutionMode.TASK
                and self.execution_controllers[arm].cursor.reference_state
                == final_state
            )
            if not at_final_task_state:
                self._terminal_actions_committed.discard(arm)
            command = result.commands[arm]
            gripper_command_applied = bool(
                applied_by_arm[arm]
                and (
                    command.gripper_authorized is True
                    or (command.gripper_authorized is None and completion_by_arm[arm])
                )
            )
            if (
                at_final_task_state
                and gripper_command_applied
                and self._last_executed_reference.get(arm) == final_state
            ):
                self._terminal_actions_committed.add(arm)
        self.diagnostics.annotate_last(
            "action_commit",
            {
                "task_command_applied": dict(applied_by_arm),
                "absolute_target_completed": dict(completion_by_arm),
                "gripper_command_applied": {
                    arm: bool(
                        applied_by_arm[arm]
                        and (
                            result.commands[arm].gripper_authorized is True
                            or (
                                result.commands[arm].gripper_authorized is None
                                and completion_by_arm[arm]
                            )
                        )
                    )
                    for arm in self.arms
                },
                "executed_reference_states": (
                    None
                    if executed_reference_states is None
                    else {
                        arm: state_token(state)
                        for arm, state in executed_reference_states.items()
                    }
                ),
            },
        )
        self._pending_snapshot = None
        return result

    def _augment_observations(
        self,
        dynamac: Mapping[str, DynaMACObservation],
        runtime: Mapping[str, RuntimeObservation],
    ) -> tuple[dict[str, DynaMACObservation], dict[str, RuntimeObservation]]:
        augmented_dynamac = {}
        augmented_runtime = {}
        # Boundary scene factors may use either arm end effector as an entity.
        # Materialize the synchronized proprioceptive poses in every runtime
        # view; keep them out of DynaMACObservation so the frozen baseline frame
        # schema and PoE stream selection remain unchanged.
        joint_ee_poses = {
            f"{joint_arm}_ee": runtime[joint_arm].ee_pose.copy()
            for joint_arm in self.arms
        }
        for arm in self.arms:
            for name, pose in joint_ee_poses.items():
                existing = runtime[arm].frame_poses.get(name)
                if existing is not None and not np.allclose(existing, pose):
                    raise ValueError(f"联合快照中的 {name} 位姿不一致")
            joint_runtime = replace(
                runtime[arm],
                frame_poses={**runtime[arm].frame_poses, **joint_ee_poses},
                frame_visibility={
                    **runtime[arm].frame_visibility,
                    **{name: True for name in joint_ee_poses},
                },
                tracking_reliability={
                    **runtime[arm].tracking_reliability,
                    **{name: 1.0 for name in joint_ee_poses},
                },
            )
            reference = self.execution_controllers[arm].cursor.reference_state
            augmented_dynamac[arm], augmented_runtime[arm] = (
                self._ensure_virtual_reference(
                    arm,
                    reference,
                    dynamac[arm],
                    joint_runtime,
                )
            )
        return augmented_dynamac, augmented_runtime

    def _ensure_virtual_reference(
        self,
        arm: str,
        state: StateId,
        dynamac: DynaMACObservation,
        runtime: RuntimeObservation,
    ) -> tuple[DynaMACObservation, RuntimeObservation]:
        """Capture one skill-entry frame before its first same-tick query."""

        label = self.task_models[arm].base_policy.skills[state.skill_index].label
        name = f"virtual_skill_{label}"
        if name not in self._virtual_frames[arm]:
            self._virtual_frames[arm][name] = dynamac.ee_pose.copy()
        frames = {**dynamac.frames, **self._virtual_frames[arm]}
        return (
            DynaMACObservation(dynamac.ee_pose, frames),
            replace(
                runtime,
                frame_poses={
                    **runtime.frame_poses,
                    **self._virtual_frames[arm],
                },
            ),
        )

    def _validate_observations(
        self,
        dynamac: Mapping[str, DynaMACObservation],
        runtime: Mapping[str, RuntimeObservation],
    ) -> int:
        if set(dynamac) != set(self.arms) or set(runtime) != set(self.arms):
            raise ValueError("闭环 act 必须一次提供全部机械臂的两类观测")
        ticks = {runtime[arm].tick for arm in self.arms}
        if len(ticks) != 1:
            raise ValueError("多臂闭环 act 必须使用同一 pre-action tick")
        for arm in self.arms:
            if not np.allclose(dynamac[arm].ee_pose, runtime[arm].ee_pose):
                raise ValueError("DynaMAC 与 RuntimeObservation 末端位姿不一致")
        return ticks.pop()

    def _hold_command(self, observation: RuntimeObservation, source: str) -> ArmCommand:
        return ArmCommand(
            pose=observation.ee_pose,
            covariance=self.config.recovery.recovery.action_covariance,
            gripper=observation.gripper_state,
            source=source,
        )

    @staticmethod
    def _task_command(action: DynaMACAction) -> ArmCommand:
        return ArmCommand(
            pose=action.pose,
            covariance=action.covariance,
            gripper=action.gripper,
            source="task_poe",
        )

    def _frozen_task_command(
        self,
        arm: str,
        belief: ClosedLoopBelief,
        observation: DynaMACObservation,
        runtime: RuntimeObservation,
        *,
        source: str,
    ) -> ArmCommand:
        """Servo one frozen reference without committing normal task progress."""

        action = (
            self.execution_controllers[arm]
            .query_frozen_reference(
                belief,
                observation,
                mode_by_skill=self._mode_by_arm_skill[arm],
            )
            .action
        )
        return (
            self._hold_command(runtime, f"{source}_no_positive_stream_hold")
            if action is None
            else ArmCommand(
                pose=action.pose,
                covariance=action.covariance,
                gripper=action.gripper,
                source=source,
            )
        )

    @staticmethod
    def _auxiliary_command(action: AuxiliaryAction) -> ArmCommand:
        return ArmCommand(
            pose=action.pose,
            covariance=action.covariance,
            gripper=action.gripper_command,
            source=action.source,
        )

    def _task_gripper_authorized(
        self,
        arm: str,
        command: ArmCommand,
        execution: ExecutionCycleResult | None,
        belief: ClosedLoopBelief,
        boundary: BoundaryCycleResult | None,
    ) -> bool | None:
        """Authorize TASK grippers from task belief and boundary semantics.

        Physical executor completion is intentionally absent here.  A skill
        terminal that has a cross-skill successor remains guarded until its
        boundary transaction commits; ordinary and final-task states require
        the progress posterior to agree with the executed reference state.
        """

        if command.source != "task_poe" or execution is None:
            return None
        committed_arms = (
            frozenset()
            if boundary is None or boundary.transaction is None
            else frozenset(request.arm_id for request in boundary.transaction.committed)
        )
        if arm in committed_arms:
            return True
        reference = execution.cursor_after.reference_state
        if (
            belief.progress.status != ProgressStatus.ALIGNED
            or belief.progress.estimated_state != reference
        ):
            return False
        node = self.task_models[arm].states[reference]
        if node.topology.has_cross_skill_successor:
            return False
        return True

    def _current_discrete_action_complete(
        self,
        arm: str,
        observation: RuntimeObservation,
    ) -> bool:
        """Check whether the current StateNode's gripper command is present.

        Continuous state completion remains exclusively beta/reference based.
        This check only prevents the discrete half of an already reached task
        state from being skipped when the controller selects its successor.
        Non-final skill terminals remain owned by the boundary transaction,
        which applies the committed entry state's gripper command.
        """

        reference = self.execution_controllers[arm].cursor.reference_state
        node = self.task_models[arm].state(reference)
        boundary_controller = getattr(self, "boundary_controller", None)
        preparation = (
            None
            if boundary_controller is None
            else boundary_controller.preparations.get(arm)
        )
        if preparation is not None:
            learned_boundary = self.task_models[arm].boundaries[
                preparation.boundary_id
            ]
            if reference in learned_boundary.terminal_window:
                target = np.asarray(
                    preparation.gripper_command, dtype=np.float64
                )
                current = np.asarray(
                    observation.gripper_state, dtype=np.float64
                )
                if target.shape != current.shape:
                    raise ValueError("边界转移准备与运行观测的夹爪维度不一致")
                return bool(
                    np.array_equal(target > 0.5, current > 0.5)
                )
        if node.topology.has_cross_skill_successor:
            return True
        mode = self._mode_by_arm_skill[arm].get(reference.skill_index)
        if mode is None:
            mode = int(
                self.task_models[arm].base_policy.selected_mode_path[
                    reference.skill_index
                ]
            )
        if mode < 0 or mode >= len(node.gripper_commands):
            raise IndexError("当前 StateNode 的夹爪模态索引无效")
        target = np.asarray(node.gripper_commands[mode], dtype=np.float64)
        current = np.asarray(observation.gripper_state, dtype=np.float64)
        if target.shape != current.shape:
            raise ValueError("当前 StateNode 与运行观测的夹爪维度不一致")
        return bool(np.array_equal(target > 0.5, current > 0.5))

    @staticmethod
    def _verification_request(
        arm: str,
        execution: ExecutionCycleResult | None,
        boundary: BoundaryCycleResult | None,
    ) -> RelationVerificationRequest | None:
        requests: list[RelationVerificationRequest] = []
        if execution is not None:
            requests.extend(execution.roles.verification_requests)
        if boundary is not None:
            # A directional guard belongs to the waiting arm's boundary, but
            # its Pending relation can belong to the peer arm that must run
            # VERIFY_LINK.  Collect the shared snapshot's requests first and
            # route them by request.arm_id below.
            for transition in boundary.requests.values():
                requests.extend(transition.verification_requests)
        unique = {
            request.pending_event_id.token: request
            for request in requests
            if request.arm_id == arm
        }
        return unique[min(unique)] if unique else None

    @staticmethod
    def _arm_trigger(
        arm: str,
        trigger: RecoveryTriggerDecision,
    ) -> RecoveryTriggerDecision:
        reasons = tuple(
            reason for reason in trigger.reasons if reason.startswith(f"{arm}:")
        )
        intents = tuple(intent for intent in trigger.intents if intent.arm_id == arm)
        if intents and not reasons:
            reasons = (f"{arm}:relation_intent",)
        return RecoveryTriggerDecision(bool(reasons or intents), reasons, intents)

    def _update_auxiliary(
        self,
        arm: str,
        belief: ClosedLoopBelief,
        dynamac_observation: DynaMACObservation,
        observation: RuntimeObservation,
        *,
        verification_safety: SafetyConstraintStatus,
        recovery_safety: RecoverySafetyStatus,
    ) -> tuple[RecoveryManagerResult, ArmCommand, str | None]:
        manager = self.recovery_managers[arm]
        if manager.mode == ExecutionMode.VERIFY_LINK:
            result = manager.update_verification(
                belief,
                current_pose=observation.ee_pose,
                safety=verification_safety,
            )
            verification = result.verification
            if (
                verification is not None
                and verification.phase == VerificationPhase.COMPLETE
                and verification.decision
                in {RelationDecision.EXTERNAL, RelationDecision.LINKED}
            ):
                if verification.verified_posterior is None:
                    raise RuntimeError("稳定主动验证结果缺少原关系后验")
                assert manager.verification.request is not None
                self.belief_updaters[arm].commit_relation_confirmation(
                    manager.verification.request.frame_id,
                    verification.verified_posterior,
                    verification.decision,
                )
            action = None if verification is None else verification.action
            command = (
                self._hold_command(observation, "verify_link_hold")
                if action is None
                else self._auxiliary_command(action)
            )
            failure = None if verification is None else verification.failure_reason
            return result, command, failure

        if manager.mode != ExecutionMode.RECOVERY:
            raise RuntimeError("辅助动作更新要求 VERIFY_LINK 或 RECOVERY 模式")
        result = manager.update_recovery(
            current_pose=observation.ee_pose,
            current_gripper=observation.gripper_state,
            frame_poses=observation.frame_poses,
            relation_estimates=belief.relation_estimates,
            safety=recovery_safety,
        )
        action = None if result.recovery is None else result.recovery.action
        if action is not None:
            command = self._auxiliary_command(action)
        elif not manager.recovery.has_relation_goals:
            command = self._frozen_task_command(
                arm,
                belief,
                dynamac_observation,
                observation,
                source="recovery_frozen_task_target",
            )
        else:
            # A completed relation repair must not be undone by replaying the
            # pre-fault task target while its legal reentry is being selected.
            command = self._hold_command(observation, "recovery_hold")
        failure = None
        if result.recovery is not None and result.recovery.failure is not None:
            failure = result.recovery.failure.reason
        return result, command, failure

    def _reentry_boundary_arms(self, reentry_arms: set[str]) -> frozenset[str]:
        """Return reentry arms plus every member of their joint transaction."""

        selected = set(reentry_arms)
        groups = set()
        for arm in reentry_arms:
            frozen = self.recovery_managers[arm].frozen_reference
            if frozen is None:
                continue
            boundary = next(
                (
                    value
                    for value in self.task_models[arm].boundaries.values()
                    if value.source_skill == frozen.skill_index
                ),
                None,
            )
            if boundary is not None and boundary.transaction_group is not None:
                groups.add(boundary.transaction_group)
        for arm, model in self.task_models.items():
            if any(
                boundary.transaction_group in groups
                for boundary in model.boundaries.values()
            ):
                selected.add(arm)
        return frozenset(selected)

    @staticmethod
    def _without_cross_skill_decision(
        evaluation: ReentryEvaluation,
    ) -> ReentryEvaluation:
        decision = evaluation.decision
        if decision is None:
            return evaluation
        reasons = dict(evaluation.rejection_reasons)
        reasons[decision.state_id] = ("boundary_transaction_not_committed",)
        return ReentryEvaluation(
            None,
            evaluation.scores,
            reasons,
            alignment_state=None,
        )

    def act(
        self,
        dynamac_observations: Mapping[str, DynaMACObservation],
        runtime_observations: Mapping[str, RuntimeObservation],
        *,
        verification_safety_by_arm: Mapping[str, SafetyConstraintStatus] | None = None,
        recovery_safety_by_arm: Mapping[str, RecoverySafetyStatus] | None = None,
        grasp_events: Mapping[str, Hashable] | None = None,
    ) -> PolicyCycleResult:
        if not self._initialized:
            raise RuntimeError("闭环顶层策略尚未 reset")
        if self._pending_snapshot is not None:
            raise RuntimeError("上一闭环策略周期尚未 commit 或 abort")
        if self.failed:
            raise RuntimeError("闭环顶层策略已进入结构化失败状态")
        if self.complete:
            raise RuntimeError("闭环顶层策略已完成")
        snapshot = self._snapshot()
        try:
            result = self._act_impl(
                dynamac_observations,
                runtime_observations,
                verification_safety_by_arm=verification_safety_by_arm or {},
                recovery_safety_by_arm=recovery_safety_by_arm or {},
                grasp_events=grasp_events or {},
            )
        except Exception:
            self._restore(snapshot)
            raise
        self._pending_snapshot = snapshot
        self._last_cycle = result
        return result

    def _act_impl(
        self,
        dynamac_observations: Mapping[str, DynaMACObservation],
        runtime_observations: Mapping[str, RuntimeObservation],
        *,
        verification_safety_by_arm: Mapping[str, SafetyConstraintStatus],
        recovery_safety_by_arm: Mapping[str, RecoverySafetyStatus],
        grasp_events: Mapping[str, Hashable],
    ) -> PolicyCycleResult:
        tick = self._validate_observations(dynamac_observations, runtime_observations)
        dynamac, runtime = self._augment_observations(
            dynamac_observations, runtime_observations
        )
        modes_before = {arm: self.recovery_managers[arm].mode for arm in self.arms}
        auxiliary_active = any(
            mode != ExecutionMode.TASK for mode in modes_before.values()
        )
        permitted_for_update = dict(self._permitted_boundaries)
        self._permitted_boundaries = {arm: frozenset() for arm in self.arms}
        beliefs: dict[str, ClosedLoopBelief] = {}
        executions: dict[str, ExecutionCycleResult] = {}
        commands: dict[str, ArmCommand] = {}
        for arm in self.arms:
            updater = self.belief_updaters[arm]
            if modes_before[arm] == ExecutionMode.TASK and not auxiliary_active:
                action_executed = self._last_executed_reference[arm] is not None
                beliefs[arm] = updater.update(
                    runtime[arm],
                    executed_reference_state=self._last_executed_reference[arm],
                    action_executed=action_executed,
                    permitted_boundaries=permitted_for_update.get(arm, frozenset()),
                    mode_by_skill=self._mode_by_arm_skill[arm],
                )
                self.recovery_managers[arm].record_task_pose(runtime[arm].ee_pose)
                execution = self.execution_controllers[arm].update(
                    beliefs[arm],
                    dynamac[arm],
                    mode_by_skill=self._mode_by_arm_skill[arm],
                    current_discrete_action_complete=(
                        self._current_discrete_action_complete(arm, runtime[arm])
                    ),
                    action_executed=action_executed,
                )
                executions[arm] = execution
                action = execution.weighted_action.action
                commands[arm] = (
                    self._hold_command(runtime[arm], "task_no_positive_stream_hold")
                    if action is None
                    else self._task_command(action)
                )
            else:
                beliefs[arm] = updater.update_frozen(
                    runtime[arm],
                    mode_by_skill=self._mode_by_arm_skill[arm],
                )
                if modes_before[arm] == ExecutionMode.TASK:
                    commands[arm] = self._frozen_task_command(
                        arm,
                        beliefs[arm],
                        dynamac[arm],
                        runtime[arm],
                        source="auxiliary_peer_frozen_task_target",
                    )

        boundary = None
        # VERIFY_LINK/RECOVERY is a top-level task mode.  While one arm runs an
        # auxiliary action, every normal progress cursor and entry transaction
        # stays frozen; peer arms still servo their committed targets above.
        task_arms = (
            frozenset()
            if auxiliary_active
            else frozenset(
                arm for arm, mode in modes_before.items() if mode == ExecutionMode.TASK
            )
        )
        if task_arms:
            boundary = self.boundary_controller.update(
                beliefs,
                arms=task_arms,
                mode_by_arm_skill=self._mode_by_arm_skill,
            )
            if boundary.transaction is not None:
                self._permitted_boundaries.update(
                    boundary.transaction.permitted_boundaries
                )
                for request in boundary.transaction.committed:
                    arm = request.arm_id
                    # Match the frozen DynaMAC hybrid boundary clock exactly:
                    # this cycle still executes the source skill's terminal
                    # continuous target.  The target skill's virtual frame is
                    # therefore captured by ``_augment_observations`` from the
                    # *next* post-action observation, not from this cycle's
                    # pre-terminal pose.  The entry gripper command is stored
                    # directly in StateNode and does not require an early
                    # target-skill pose query.
                    refreshed = self.execution_controllers[
                        arm
                    ].query_after_boundary_transition(
                        executions[arm],
                        beliefs[arm],
                        dynamac[arm],
                        mode_by_skill=self._mode_by_arm_skill[arm],
                    )
                    executions[arm] = refreshed
                    action = refreshed.weighted_action.action
                    commands[arm] = (
                        self._hold_command(
                            runtime[arm], "boundary_no_positive_stream_hold"
                        )
                        if action is None
                        else self._task_command(action)
                    )

        trigger = (
            self.recovery_trigger.update(
                {arm: result.mismatch for arm, result in executions.items()},
                transition_requests=None if boundary is None else boundary.requests,
                beliefs=beliefs,
                mode_by_arm_skill=self._mode_by_arm_skill,
            )
            if self.feature_profile.auxiliary_verification_recovery
            else RecoveryTriggerDecision(False, (), ())
        )
        recovery_results: dict[str, RecoveryManagerResult] = {}
        failures: dict[str, str] = {}
        for arm in self.arms:
            manager = self.recovery_managers[arm]
            if (
                manager.mode == ExecutionMode.TASK
                and self.feature_profile.auxiliary_verification_recovery
            ):
                verification_request = self._verification_request(
                    arm,
                    executions.get(arm),
                    boundary,
                )
                arm_trigger = self._arm_trigger(arm, trigger)
                grasp_event = (
                    None
                    if verification_request is None
                    else grasp_events.get(
                        arm, verification_request.pending_event_id.token
                    )
                )
                task_state = self.execution_controllers[arm].cursor.reference_state
                if verification_request is not None and manager.can_begin_verification(
                    verification_request,
                    beliefs[arm],
                    task_state=task_state,
                    grasp_event=grasp_event,
                ):
                    manager.begin_verification(
                        verification_request,
                        beliefs[arm],
                        task_state=task_state,
                        grasp_event=grasp_event,
                        current_pose=runtime[arm].ee_pose,
                        current_gripper=runtime[arm].gripper_state,
                    )
                elif arm_trigger.triggered:
                    source = task_state
                    manager.begin_recovery(
                        arm_trigger,
                        source_state=source,
                        mode=self._mode_by_arm_skill[arm][source.skill_index],
                    )

            if manager.mode != ExecutionMode.TASK:
                recovery_result, command, failure = self._update_auxiliary(
                    arm,
                    beliefs[arm],
                    dynamac[arm],
                    runtime[arm],
                    verification_safety=verification_safety_by_arm.get(
                        arm, SafetyConstraintStatus()
                    ),
                    recovery_safety=recovery_safety_by_arm.get(
                        arm, RecoverySafetyStatus()
                    ),
                )
                recovery_results[arm] = recovery_result
                commands[arm] = command
                if failure is not None:
                    failures[arm] = failure

        reentry_arms = {
            arm
            for arm in self.arms
            if self.recovery_managers[arm].mode == ExecutionMode.RECOVERY
            and self.recovery_managers[arm].recovery.phase == RecoveryPhase.REENTRY
        }
        reentry_evaluations: dict[str, ReentryEvaluation] = {}
        mixed_boundary_evaluation = None
        mixed_preview = None
        if reentry_arms and boundary is None:
            selected = self._reentry_boundary_arms(reentry_arms)
            sources = {
                arm: (
                    self.recovery_managers[arm].frozen_reference
                    if arm in reentry_arms
                    else self.execution_controllers[arm].cursor.reference_state
                )
                for arm in selected
            }
            if any(state is None for state in sources.values()):
                raise RuntimeError("恢复重入边界缺少冻结源状态")
            source_states = {
                arm: state for arm, state in sources.items() if state is not None
            }
            mixed_boundary_evaluation = self.boundary_controller.evaluate(
                beliefs,
                arms=selected,
                source_states=source_states,
                mode_by_arm_skill=self._mode_by_arm_skill,
            )
            if mixed_boundary_evaluation.requests:
                mixed_preview = self.boundary_controller.transactions.preview(
                    tuple(mixed_boundary_evaluation.requests.values())
                )

        preview_permissions = (
            {} if mixed_preview is None else mixed_preview.permitted_boundaries
        )
        for arm in reentry_arms:
            permissions = (
                permitted_for_update.get(arm, frozenset())
                | self._permitted_boundaries.get(arm, frozenset())
                | preview_permissions.get(arm, frozenset())
            )
            reentry_evaluations[arm] = self.recovery_managers[arm].select_reentry(
                beliefs[arm],
                permitted_boundaries=permissions,
                mode_by_skill=self._mode_by_arm_skill[arm],
            )

        committed_reentry_arms: set[str] = set()
        if mixed_boundary_evaluation is not None and mixed_preview is not None:
            by_group: dict[str, list[Any]] = {}
            independent = []
            for request in mixed_preview.committed:
                if request.transaction_group is None:
                    independent.append(request)
                else:
                    by_group.setdefault(request.transaction_group, []).append(request)

            eligible = []
            for request in independent:
                if request.arm_id in reentry_arms:
                    decision = reentry_evaluations[request.arm_id].decision
                    if (
                        decision is not None
                        and decision.crossed_boundary == request.boundary_id
                    ):
                        eligible.append(request)
                        committed_reentry_arms.add(request.arm_id)
                elif self.recovery_managers[request.arm_id].mode == ExecutionMode.TASK:
                    eligible.append(request)
            for members in by_group.values():
                group_ready = True
                group_reentry = set()
                for request in members:
                    if request.arm_id in reentry_arms:
                        decision = reentry_evaluations[request.arm_id].decision
                        if (
                            decision is None
                            or decision.crossed_boundary != request.boundary_id
                        ):
                            group_ready = False
                            break
                        group_reentry.add(request.arm_id)
                    elif (
                        self.recovery_managers[request.arm_id].mode
                        != ExecutionMode.TASK
                    ):
                        group_ready = False
                        break
                if group_ready:
                    eligible.extend(members)
                    committed_reentry_arms.update(group_reentry)

            if eligible:
                boundary = self.boundary_controller.commit_requests(
                    mixed_boundary_evaluation,
                    requests=tuple(eligible),
                    externally_committed_arms=frozenset(committed_reentry_arms),
                )
                assert boundary.transaction is not None
                self._permitted_boundaries.update(
                    boundary.transaction.permitted_boundaries
                )
                for request in boundary.transaction.committed:
                    arm = request.arm_id
                    if arm in committed_reentry_arms:
                        continue
                    dynamac[arm], runtime[arm] = self._ensure_virtual_reference(
                        arm,
                        request.target_state,
                        dynamac[arm],
                        runtime[arm],
                    )
                    refreshed = self.execution_controllers[
                        arm
                    ].query_after_boundary_transition(
                        executions[arm],
                        beliefs[arm],
                        dynamac[arm],
                        mode_by_skill=self._mode_by_arm_skill[arm],
                    )
                    executions[arm] = refreshed
                    action = refreshed.weighted_action.action
                    commands[arm] = (
                        self._hold_command(
                            runtime[arm], "boundary_no_positive_stream_hold"
                        )
                        if action is None
                        else self._task_command(action)
                    )
            else:
                boundary = mixed_boundary_evaluation

        for arm, evaluation in reentry_evaluations.items():
            decision = evaluation.decision
            if (
                decision is not None
                and decision.crossed_boundary is not None
                and arm not in committed_reentry_arms
            ):
                evaluation = self._without_cross_skill_decision(evaluation)
            if evaluation.decision is None and evaluation.alignment_state is not None:
                weighted = self.execution_controllers[arm].query_reentry_alignment(
                    evaluation.alignment_state,
                    beliefs[arm],
                    dynamac[arm],
                    mode_by_skill=self._mode_by_arm_skill[arm],
                )
                if weighted.action is not None:
                    action = weighted.action
                    commands[arm] = ArmCommand(
                        pose=action.pose,
                        covariance=(
                            action.covariance
                            + np.eye(6, dtype=np.float64)
                            * self.config.recovery.recovery.covariance_inflation
                        ),
                        gripper=action.gripper,
                        source="recovery_reentry_alignment",
                    )
            result = self.recovery_managers[arm].commit_reentry(
                evaluation,
                belief=beliefs[arm],
                observation=runtime[arm],
                belief_updater=self.belief_updaters[arm],
                execution_controller=self.execution_controllers[arm],
            )
            recovery_results[arm] = result
            if result.recovery is not None and result.recovery.failure is not None:
                failures[arm] = result.recovery.failure.reason

        lifecycle = PolicyLifecycle.FAILED if failures else PolicyLifecycle.RUNNING
        self._lifecycle = lifecycle
        arm_results = {}
        for arm in self.arms:
            mode_after = self.recovery_managers[arm].mode
            preparation = self.boundary_controller.preparations.get(arm)
            preparation_active = (
                preparation is not None and mode_after == ExecutionMode.TASK
            )
            if preparation_active:
                assert preparation is not None
                commands[arm] = replace(
                    commands[arm],
                    gripper=preparation.gripper_command.copy(),
                    gripper_authorized=True,
                )
            commands[arm] = replace(
                commands[arm],
                gripper_authorized=(
                    True
                    if preparation_active
                    else self._task_gripper_authorized(
                        arm,
                        commands[arm],
                        executions.get(arm),
                        beliefs[arm],
                        boundary,
                    )
                ),
            )
            arm_results[arm] = ArmCycleResult(
                arm_id=arm,
                mode_before=modes_before[arm],
                mode_after=mode_after,
                belief=beliefs[arm],
                command=commands[arm],
                execution=executions.get(arm),
                recovery=recovery_results.get(arm),
                failure_reason=failures.get(arm),
            )
            self._last_executed_reference[arm] = (
                executions[arm].progress_anchor_state
                if arm in executions
                and commands[arm].source == "task_poe"
                and mode_after == ExecutionMode.TASK
                else None
            )

        diagnostic = self._diagnostic(
            tick,
            arm_results,
            boundary,
            trigger,
            runtime_observations=runtime,
        )
        diagnostic = self.diagnostics.append(diagnostic)
        return PolicyCycleResult(
            tick=tick,
            arms=arm_results,
            boundary=boundary,
            lifecycle=lifecycle,
            diagnostics=diagnostic,
        )

    def _diagnostic(
        self,
        tick: int,
        arms: Mapping[str, ArmCycleResult],
        boundary: BoundaryCycleResult | None,
        trigger: RecoveryTriggerDecision,
        *,
        runtime_observations: Mapping[str, RuntimeObservation],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tick": tick,
            "lifecycle": self._lifecycle.value,
            "feature_profile": self.feature_profile.to_dict(),
            "recovery_trigger": json_ready(trigger),
            "boundary": json_ready(boundary),
            "arms": {},
        }
        for arm, cycle in arms.items():
            belief = cycle.belief
            execution = cycle.execution
            preparation = self.boundary_controller.preparations.get(arm)
            result["arms"][arm] = {
                "observation": {
                    "ee_pose": runtime_observations[arm].ee_pose,
                    "frame_poses": runtime_observations[arm].frame_poses,
                    "gripper_state": runtime_observations[arm].gripper_state,
                    "frame_visibility": runtime_observations[arm].frame_visibility,
                    "tracking_reliability": runtime_observations[
                        arm
                    ].tracking_reliability,
                },
                "mode_before": cycle.mode_before.value,
                "mode_after": cycle.mode_after.value,
                "nominal_state": state_token(belief.progress.nominal_state),
                "estimated_state": state_token(belief.progress.estimated_state),
                "reference_state": state_token(
                    self.execution_controllers[arm].cursor.reference_state
                ),
                "progress_prior": {
                    state_token(state): value
                    for state, value in belief.progress.prior.items()
                },
                "progress_posterior": {
                    state_token(state): value
                    for state, value in belief.progress.posterior.items()
                },
                "progress_confidence": belief.progress.confidence,
                "progress_status": belief.progress.status.value,
                "command_tracking": {
                    "available": belief.runtime_features.command_tracking_available,
                    "compatibility": (
                        belief.runtime_features.command_tracking_compatibility
                    ),
                    "mahalanobis_squared": (
                        belief.runtime_features.command_tracking_mahalanobis_squared
                    ),
                    "diagnostic_only_for_progress": True,
                    "control_equivalence_threshold": (
                        self.config.execution.minimum_action_equivalence_compatibility
                    ),
                },
                "motion_features": {
                    "actual_ee_motion": belief.runtime_features.actual_ee_motion,
                    "actual_motion_magnitude": (
                        belief.runtime_features.actual_motion_magnitude
                    ),
                    "action_excitation": belief.runtime_features.action_excitation,
                    "frame_world_motion": belief.runtime_features.frame_world_motion,
                    "relative_motion_residuals": (
                        belief.runtime_features.relative_motion_residuals
                    ),
                },
                "relations": {
                    frame: {
                        "demo_prior": estimate.demonstration_prior,
                        "predicted": estimate.predicted,
                        "observation_likelihood": estimate.observation_likelihood,
                        "posterior": estimate.posterior,
                        "informative": estimate.informative,
                        "decision": estimate.decision_state.value,
                        "information_weight": estimate.information_weight,
                        "informative_evidence_direction": (
                            estimate.informative_evidence_direction.value
                        ),
                    }
                    for frame, estimate in belief.relation_estimates.items()
                },
                "candidate_scores": {
                    state_token(state): score
                    for state, score in belief.candidate_scores.items()
                },
                "execution": (
                    None
                    if execution is None
                    else {
                        "decision": execution.decision.value,
                        "reasons": execution.reasons,
                        "control_equivalence": execution.control_equivalence,
                        "roles": execution.roles,
                        "participating_streams": execution.weighted_action.participating_frames,
                        "poe_weights": execution.weighted_action.stream_weights,
                        "query_diagnostics": (
                            None
                            if execution.weighted_action.action is None
                            else execution.weighted_action.action.diagnostics
                        ),
                    }
                ),
                "recovery": cycle.recovery,
                "transition_preparation": preparation,
                "action": cycle.command,
                "failure_reason": cycle.failure_reason,
            }
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arms": list(self.arms),
            "models": {arm: self.task_models[arm].summary() for arm in self.arms},
            "config": self.config.to_dict(),
            "feature_profile": self.feature_profile.to_dict(),
            "initialized": self.initialized,
            "pending": self.pending,
            "failed": self.failed,
            "complete": self.complete,
            "diagnostic_ticks": len(self.diagnostics.records),
        }

    def save(self, directory: str | Path) -> Path:
        if self.pending:
            raise RuntimeError("待提交周期存在时不能保存闭环策略")
        return save_policy_bundle(
            directory,
            task_models=self.task_models,
            config=self.config,
            summary=self.summary(),
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        base_policies: Mapping[str, DynaMAC],
        feature_profile: ClosedLoopFeatureProfile | str = "full",
    ) -> ClosedLoopMultiStreamPolicy:
        models, config, _ = load_policy_bundle(
            directory,
            base_policies=base_policies,
        )
        return cls(models, config, feature_profile=feature_profile)


__all__ = ["ClosedLoopMultiStreamPolicy"]
